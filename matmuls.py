import torch
from torch.autograd.function import InplaceFunction
import custom_mm


def cublas_matmul(a: torch.Tensor, b: torch.Tensor, torch_: bool = False) -> torch.Tensor:
    '''
    Uses cuBLAS kernel to perform matrix multiplication.

    :param a:
    :param b:
    :param torch_: Set to true if data is passed in in col-major (expected row-major)
    :returns: Matrix multiplication output
    '''
    a = a.contiguous()
    b = b.contiguous()

    if len(a.shape) == 1 or len(b.shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b
    elif len(a.shape) == 3 and len(b.shape) == 2:
        assert a.shape[-1] == b.shape[0]
        lda, dim1, dim2 = a.shape
        _a = a.reshape(lda*dim1, dim2)
        if not torch_:
            _c = custom_mm.cublas_mmul(_a, b)
        else:
            _c = custom_mm.cublas_mmul(b.t(), _a.t()).t()
        return _c.reshape(lda, dim1, -1).clone().detach()
    elif len(a.shape) == 2 and len(b.shape) == 3:
        assert a.shape[-1] == b.shape[1]
        ldb, dim1, dim2 = b.shape
        _b = b.reshape(ldb*dim1, dim2)
        if not torch_:
            _c = custom_mm.cublas_mmul(a, _b)
        else:
            _c = custom_mm.cublas_mmul(_b.t(), a.t()).t()
        return _c.reshape(ldb, dim1, -1).clone().detach()
    elif len(a.shape) >= 3 and len(b.shape) >= 3:
        _, a_dim2 = a.shape[-2:]
        b_dim1, _ = b.shape[-2:]
        lda, ldb = a.shape[0], b.shape[0]
        assert lda == ldb
        assert a_dim2 == b_dim1
        if len(a.shape) == 3 and len(b.shape) == 3:
            _c = torch.stack([cublas_matmul(a[i], b[i], torch_)
                              for i in range(lda)])
        else:
            _c = torch.stack([cublas_matmul(a[i], b[i], torch_)
                              for i in range(lda)])
        return _c.clone().detach()
    elif len(a.shape) == 2 and len(b.shape) == 2:
        assert a.shape[-1] == b.shape[0]
        if not torch_:
            return custom_mm.cublas_mmul(a, b)
        else:
            return custom_mm.cublas_mmul(b.t(), a.t()).t()
    else:
        print('Multiplication with matrix dimensions is not implemented in cuBLAS')
        return a @ b


def cusparse_matmul(a: torch.Tensor, b: torch.Tensor, torch_: bool = False) -> torch.Tensor:
    '''
    Uses cuSPARSE kernel to perform matrix multiplication.

    :param a:
    :param b:
    :param torch_: Set to true if data is passed in in col-major (expected row-major)
    :returns: Matrix multiplication output
    '''
    init_type = torch.double
    if (a.dtype == torch.float or b.dtype == torch.float):
        init_type = torch.float
        a = a.to(torch.double)
        b = b.to(torch.double)

    a = a.contiguous()
    b = b.contiguous()

    if len(a.shape) == 1 or len(b.shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuSPARSE')
        return torch.matmul(a, b).to(init_type)
    elif len(a.shape) == 3 and len(b.shape) == 2:
        assert a.shape[-1] == b.shape[0]
        lda, dim1, dim2 = a.shape
        _a = a.reshape(lda*dim1, dim2)
        if not torch_:
            _c = custom_mm.cusparse_mmul(_a, b)
        else:
            _c = custom_mm.cusparse_mmul(b.t(), _a.t()).t()
        return _c.reshape(lda, dim1, -1).clone().detach().to(init_type)
    elif len(a.shape) == 2 and len(b.shape) == 3:
        assert a.shape[-1] == b.shape[1]
        ldb, dim1, dim2 = b.shape
        _b = b.reshape(ldb*dim1, dim2)
        if not torch_:
            _c = custom_mm.cusparse_mmul(a, _b)
        else:
            _c = custom_mm.cusparse_mmul(_b.t(), a.t()).t()
        return _c.reshape(ldb, dim1, -1).clone().detach().to(init_type)
    elif len(a.shape) >= 3 and len(b.shape) >= 3:
        _, a_dim2 = a.shape[-2:]
        b_dim1, _ = b.shape[-2:]
        lda, ldb = a.shape[0], b.shape[0]
        assert lda == ldb
        assert a_dim2 == b_dim1
        if len(a.shape) == 3 and len(b.shape) == 3:
            _c = torch.stack([cusparse_matmul(a[i], b[i], torch_)
                              for i in range(lda)])
        else:
            _c = torch.stack([cusparse_matmul(a[i], b[i], torch_)
                              for i in range(lda)])
        return _c.clone().detach().to(init_type)
    elif len(a.shape) == 2 and len(b.shape) == 2:
        assert a.shape[-1] == b.shape[0]
        if not torch_:
            return custom_mm.cusparse_mmul(a, b).to(init_type)
        else:
            return custom_mm.cusparse_mmul(b.t(), a.t()).t().to(init_type)
    else:
        print('Multiplication with matrix dimensions is not implemented in cuSPARSE')
        return torch.matmul(a, b).to(init_type)


class cublasMM(InplaceFunction):

    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return cublas_matmul(m1, m2).to("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = cublas_matmul(
                grad_output, m2.transpose(-1, -2)).to("cuda" if torch.cuda.is_available() else "cpu")

        if ctx.needs_input_grad[1]:
            grad_m2 = cublas_matmul(
                m1.transpose(-1, -2), grad_output).to("cuda" if torch.cuda.is_available() else "cpu")

        return grad_m1, grad_m2


class cusparseMM(InplaceFunction):

    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return cusparse_matmul(m1, m2).to("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = cusparse_matmul(
                grad_output, m2.transpose(-1, -2)).to("cuda" if torch.cuda.is_available() else "cpu")

        if ctx.needs_input_grad[1]:
            grad_m2 = cusparse_matmul(
                m1.transpose(-1, -2), grad_output).to("cuda" if torch.cuda.is_available() else "cpu")

        return grad_m1, grad_m2

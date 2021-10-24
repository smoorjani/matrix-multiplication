import torch
import time
from torch.autograd.function import InplaceFunction
import custom_mm


def custom_matmul(a: torch.Tensor,
                  b: torch.Tensor,
                  mm_op=custom_mm.cublas_mmul,
                  bmm_op=custom_mm.cublas_bmm) -> torch.Tensor:
    '''
    Uses cuBLAS kernel to perform matrix multiplication.

    :param a:
    :param b:
    :param torch_: Set to true if data is passed in in col-major (expected row-major)
    :returns: Matrix multiplication output
    '''
    a_shape = a.shape
    b_shape = b.shape
    c = None
    t0 = time.time()
    if len(a_shape) == 1 or len(b_shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b
    elif len(a_shape) == 3 and len(b_shape) == 2:
        assert a_shape[-1] == b_shape[0]
        lda, dim1, dim2 = a_shape
        _a = a.reshape(lda * dim1, dim2)
        c = mm_op(_a, b).reshape(lda, dim1, -1)
    elif len(a_shape) == 2 and len(b_shape) == 3:
        assert a_shape[-1] == b_shape[1]
        ldb, dim1, dim2 = b_shape
        _b = b.reshape(ldb * dim1, dim2)
        c = mm_op(a, _b).reshape(ldb, dim1, -1)
    elif len(a_shape) >= 3 and len(b_shape) >= 3:
        ti0 = time.time()
        a_dim1, a_dim2 = a_shape[-2:]
        b_dim1, b_dim2 = b_shape[-2:]
        lda, ldb = a_shape[0], b_shape[0]
        assert lda == ldb
        assert a_dim2 == b_dim1
        if len(a_shape) == 3 and len(b_shape) == 3:
            bmm0 = time.time()
            c = bmm_op(a, b, 3)
            bmm0 = time.time() - bmm0
            print('Our BMM time: ', bmm0)
        elif len(a_shape) == 4 and len(b_shape) == 4:
            bmm0 = time.time()
            # a = a.reshape(-1, a_dim1, a_dim2)
            # b = b.reshape(-1, b_dim1, b_dim2)
            c = bmm_op(a, b, 4)
            # c = c.reshape(lda, -1, a_dim1, b_dim2)
            bmm0 = time.time() - bmm0
            print('Our BMM time: ', bmm0)
        else:
            c = torch.stack([custom_matmul(a[i], b[i], mm_op, bmm_op)
                             for i in range(lda)])
    elif len(a_shape) == 2 and len(b_shape) == 2:
        assert a_shape[-1] == b_shape[0]
        c = mm_op(a, b)
    else:
        print(
            'Multiplication with matrix dimensions is not implemented in cuBLAS'
        )
        return a @ b
    return c


class cublasMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return custom_matmul(
            m1, m2)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2))

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(
                m1.transpose(-1, -2),
                grad_output)

        return grad_m1, grad_m2

class cublasltMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return custom_matmul(
            m1, m2, custom_mm.cublaslt_mmul).to("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), custom_mm.cublaslt_mmul).to("cuda" if torch.cuda.is_available() else "cpu")

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(
                m1.transpose(-1, -2),
                grad_output, custom_mm.cublaslt_mmul).to("cuda" if torch.cuda.is_available() else "cpu")

        return grad_m1, grad_m2

class cusparseMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return custom_matmul(m1, m2, custom_mm.cusparse_mmul).to(
            "cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), custom_mm.cusparse_mmul).to(
                    "cuda" if torch.cuda.is_available() else "cpu")

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(m1.transpose(
                -1, -2), grad_output, custom_mm.cusparse_mmul).to(
                    "cuda" if torch.cuda.is_available() else "cpu")

        return grad_m1, grad_m2

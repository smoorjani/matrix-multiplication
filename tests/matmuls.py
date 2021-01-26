import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

import custom_mm


def cublas_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    Uses cuBLAS kernel to perform matrix multiplication.

    :param a:
    :param b: 
    :returns: Matrix multiplication output
    '''
    assert a.shape[-1] == b.shape[0]
    if len(a.shape) == 1 or len(b.shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b
    # batched matmul (16,768,768) (768,768)
    # (16*768, 768) (768, 768)

    elif len(a.shape) == 3 and len(b.shape) == 2:
        lda, dim1, dim2 = a.shape
        _a = a.view(lda*dim1, dim2)
        _c = custom_mm.cublas_mmul(b.t(), _a.t()).t()
        return _c.view(lda, dim1, dim2)
        # return torch.stack([custom_mm.cublas_mmul(b.t(), a[i].t()).t()
        #                     for i in range(a.shape[0])]).cuda()
    elif len(a.shape) == 2 and len(b.shape) == 3:
        ldb, dim1, dim2 = b.shape
        _b = b.view(ldb*dim1, dim2)
        _c = custom_mm.cublas_mmul(_b.t(), a.t()).t()
        return _c.view(ldb, dim1, dim2)
        # return torch.stack([custom_mm.cublas_mmul(b[i].t(), a.t()).t()
        #                     for i in range(b.shape[0])]).cuda()
    elif len(a.shape) == 2 and len(b.shape) == 2:
        return custom_mm.cublas_mmul(b.t(), a.t()).t()
    else:
        print('Multiplication with matrix dimensions is not implemented in cuBLAS')
        return a @ b


def cusparse_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    Uses cuSPARSE kernel to perform matrix multiplication.

    :param a:
    :param b: 
    :returns: Matrix multiplication output
    '''
    if len(a.shape) == 1 or len(b.shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b
    # batched matmul
    elif len(a.shape) == 3 and len(b.shape) == 2:
        return torch.stack([custom_mm.cusparse_mmul(b.t(), a[i].t()).t()
                            for i in range(a.shape[0])]).cuda()
    elif len(a.shape) == 2 and len(b.shape) == 3:
        return torch.stack([custom_mm.cusparse_mmul(b[i].t(), a.t()).t()
                            for i in range(b.shape[0])]).cuda()
    elif len(a.shape) == 2 and len(b.shape) == 2:
        return custom_mm.cusparse_mmul(b.t(), a.t()).t()
    else:
        print('Multiplication with matrix dimensions is not implemented in cuBLAS')
        return a @ b


class cublasMM(InplaceFunction):

    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return cublas_matmul(m1, m2)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = cublas_matmul(grad_output, m2.t())
            # m2 = m2.t().t()
            # grad_m1 = custom_mm.cublas_mmul(m2, grad_output.t()).t()

        if ctx.needs_input_grad[1]:
            grad_m2 = cublas_matmul(m1.t(), grad_output)
            # grad_m2 = custom_mm.cublas_mmul(grad_output.t(), m1).t()

        return grad_m1, grad_m2


class cusparseMM(InplaceFunction):

    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return custom_mm.cusparse_mmul(m2.t(), m1.t()).t()

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = cusparse_matmul(grad_output, m2.t())
            # m2 = m2.t().t()
            # grad_m1 = custom_mm.cusparse_mmul(m2, grad_output.t()).t()

        if ctx.needs_input_grad[1]:
            grad_m2 = cusparse_matmul(m1.t(), grad_output)
            # grad_m2 = custom_mm.cusparse_mmul(grad_output.t(), m1).t()

        return grad_m1, grad_m2

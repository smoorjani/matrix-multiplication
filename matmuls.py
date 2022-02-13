'''
This file handles the python backend of the matrix multiplication.
All tensor checks and logic should either be placed here or in `custom_mm.cpp`

Run `python setup.py install` to build this file.
'''

import torch
from torch.autograd.function import InplaceFunction
import custom_mm


def custom_matmul(a: torch.Tensor,
                  b: torch.Tensor,
                  mm_op=custom_mm.cublas_mmul,
                  bmm_op=custom_mm.cublas_bmm,
                  transa=False,
                  transb=False) -> torch.Tensor:
    '''
    Uses ``mm_op`` or ``bmm_op`` kernel to perform matrix multiplication.

    :param a:
    :param b:
    :param mm_op: kernel to perform basic matrix multiplication
    :param bmm_op: kernel to perform batched matrix multiplication
    :param transa: transpose A
    :param transb: transpose B
    :returns: Matrix multiplication output
    '''
    a_shape = a.shape
    b_shape = b.shape

    # create tensor C to store results in
    c_rows = a_shape[-2] if not transa else a_shape[-1]
    c_cols = b_shape[-1] if not transb else b_shape[-2]
    c = torch.zeros(
        tuple(list(a_shape[:-2]) + [c_rows, c_cols]), device=torch.device('cuda'))

    if len(a_shape) == 1 or len(b_shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b

    if len(a_shape) == 3 and len(b_shape) == 2:
        # flatten A into a 2d tensor
        lda, dim1, dim2 = a_shape
        _a = a.reshape(lda * dim1, dim2)
        c = mm_op(_a, b, c, transa, transb).reshape(lda, dim1, -1)
    elif len(a_shape) == 2 and len(b_shape) == 3:
        # flatten B into a 2d tensor
        ldb, dim1, dim2 = b_shape
        _b = b.reshape(ldb * dim1, dim2)
        c = mm_op(a, _b, c, transa, transb).reshape(ldb, dim1, -1)
    elif len(a_shape) >= 3 and len(b_shape) >= 3:
        lda, ldb = a_shape[0], b_shape[0]
        assert lda == ldb
        if len(a_shape) == 3 and len(b_shape) == 3:
            c = bmm_op(a, b, c, 3, transa, transb)
        elif len(a_shape) == 4 and len(b_shape) == 4:
            c = bmm_op(a, b, c, 4, transa, transb)
        else:
            # if tensor is 5d or larger, use a for loop to calculate
            c = torch.stack([custom_matmul(a[i], b[i], mm_op, bmm_op)
                             for i in range(lda)])
    elif len(a_shape) == 2 and len(b_shape) == 2:
        print('matmul python call.')
        c = mm_op(a, b, c, transa, transb)
    else:
        print(
            'Multiplication with matrix dimensions is not implemented in cuBLAS'
        )
        return a @ b
    return c

'''
Matrix multiplication classes

Ensure the forward and backward passes are defined for torch.autograd
To add another, just change the mm/bmm operation
'''

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


class cublasTransaMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        ctx.save_for_backward(m1, m2)
        return custom_matmul(
            m1, m2, transa=True)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), transa=True)

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(
                m1.transpose(-1, -2),
                grad_output, transa=True)

        return grad_m1, grad_m2


class cublasTransbMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        ctx.save_for_backward(m1, m2)
        return custom_matmul(
            m1, m2, transb=True)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), transb=True)

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(
                m1.transpose(-1, -2),
                grad_output, transb=True)

        return grad_m1, grad_m2


class cublasTransabMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        ctx.save_for_backward(m1, m2)
        return custom_matmul(
            m1, m2, transa=True, transb=True)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), transa=True, transb=True)

        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(
                m1.transpose(-1, -2),
                grad_output, transa=True, transb=True)

        return grad_m1, grad_m2

class cusparseMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        ctx.save_for_backward(m1, m2)
        return custom_matmul(m1, m2, custom_mm.cusparse_mmul)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = custom_matmul(grad_output, m2.transpose(
                -1, -2), custom_mm.cusparse_mmul)
        if ctx.needs_input_grad[1]:
            grad_m2 = custom_matmul(m1.transpose(
                -1, -2), grad_output, custom_mm.cusparse_mmul)

        return grad_m1, grad_m2


def naive_matmul(a: torch.Tensor,
                 b: torch.Tensor,
                 mm_op=custom_mm.naive_bmm) -> torch.Tensor:
    '''
    Uses our naive kernel to perform matrix multiplication.

    :param a:
    :param b:
    :param mm_op: kernel to perform basic matrix multiplication
    :returns: Matrix multiplication output
    '''
    a_shape = a.shape
    b_shape = b.shape

    c_rows = a_shape[-2]
    c_cols = b_shape[-1]
    c = torch.zeros(
        tuple(list(a_shape[:-2]) + [c_rows, c_cols]), device=torch.device('cuda'))

    if len(a_shape) == 1 or len(b_shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b

    if len(a_shape) == 3 and len(b_shape) == 2:
        # flatten A into a 2d tensor
        lda, dim1, dim2 = a_shape
        _a = a.reshape(lda * dim1, dim2)
        c = mm_op(_a, b, c, 2).reshape(lda, dim1, -1)
    elif len(a_shape) == 2 and len(b_shape) == 3:
        # flatten B into a 2d tensor
        ldb, dim1, dim2 = b_shape
        _b = b.reshape(ldb * dim1, dim2)
        c = mm_op(a, _b, c, 2).reshape(ldb, dim1, -1)
    elif len(a_shape) >= 3 and len(b_shape) >= 3:
        lda, ldb = a_shape[0], b_shape[0]
        assert lda == ldb
        c = torch.stack([naive_matmul(a[i], b[i], mm_op)
                         for i in range(lda)])
    elif len(a_shape) == 2 and len(b_shape) == 2:
        c = mm_op(a, b, c, 2)
    else:
        print(
            'Multiplication with matrix dimensions is not implemented in cuBLAS'
        )
        return a @ b
    return c

class naiveMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return naive_matmul(m1, m2)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = naive_matmul(grad_output, m2.transpose(
                -1, -2))

        if ctx.needs_input_grad[1]:
            grad_m2 = naive_matmul(
                m1.transpose(-1, -2),
                grad_output)

        return grad_m1, grad_m2


class naiveSpMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return naive_matmul(m1, m2,
                            mm_op=custom_mm.naive_spmm)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = naive_matmul(grad_output, m2.transpose(
                -1, -2),
                mm_op=custom_mm.naive_spmm)

        if ctx.needs_input_grad[1]:
            grad_m2 = naive_matmul(
                m1.transpose(-1, -2),
                grad_output,
                mm_op=custom_mm.naive_spmm)

        return grad_m1, grad_m2

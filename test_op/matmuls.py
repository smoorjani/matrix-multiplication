import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

import custom_mm


def cublas_matmul(a, b):
    if len(a.shape) >= 3 and len(b.shape) >= 2:
        return torch.stack([custom_mm.cublas_mmul(b.t(), a[i].t()).t()
                            for i in range(a.shape[0])]).cuda()

    return custom_mm.cublas_mmul(b.t(), a.t()).t()


def cusparse_matmul(a, b):
    if len(a.shape) >= 3 and len(b.shape) >= 2:
        return torch.stack([custom_mm.cusparse_mmul(b.t(), a[i].t()).t()
                            for i in range(a.shape[0])]).cuda()

    return custom_mm.cusparse_mmul(b.t(), a.t()).t()


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

import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

import custom_mm

class cublasMM(InplaceFunction):

    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return custom_mm.cublas_mmul(m2.t(), m1.t()).t()

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            # m2 = m2.t().t()
            grad_m1 = custom_mm.cublas_mmul(m2, grad_output.t()).t()
        
        if ctx.needs_input_grad[1]:
            grad_m2 = custom_mm.cublas_mmul(grad_output.t(), m1).t()
        
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
            # m2 = m2.t().t()
            grad_m1 = custom_mm.cusparse_mmul(m2, grad_output.t()).t()
        
        if ctx.needs_input_grad[1]:
            grad_m2 = custom_mm.cusparse_mmul(grad_output.t(), m1).t()
        
        return grad_m1, grad_m2

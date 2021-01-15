import math
import torch
import torch.nn as nn
import custom_mm

from torch.autograd.function import InplaceFunction

'''
https://cs231n.github.io/optimization-2/#mat
https://gist.github.com/anonymous/49c10bc17ac4a97307d52c07d01a2870
'''

class cublasMM(InplaceFunction):

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

class cublasLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(cublasLinear, self).__init__()
        torch.manual_seed(0)
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, inp):
        x, y = inp.shape
        if y != self.in_features:
            print('Invalid dimensions')
            return 0
        t = cublasMM.apply(inp, self.weight.t())
        output = t.clone()

        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

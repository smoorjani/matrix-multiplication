import math
import torch
import torch.nn as nn
import cublas_mm

class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__()
        
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
        #output = cublas_mm.mmul(inp, self.weight.t())
        output = inp @ self.weight.t()

        if self.bias is not None:
            output += self.bias
        ret = output
        return ret
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

import torch
import custom_mm

a = torch.rand(4,4)
b = torch.rand(4,4)
print(a, b)

c = custom_mm.mmul(a,b)
print('output:', c)
print('expected:', torch.matmul(a,b))


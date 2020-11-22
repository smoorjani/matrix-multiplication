import torch
import cublas_mm

a = torch.rand(4,4)
b = torch.rand(4,4)
print(a, b)

c = cublas_mm.mmul(a,b)
print('output:', c)
print('expected:', torch.matmul(a,b))


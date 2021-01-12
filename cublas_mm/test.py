import torch
import custom_mm

a = torch.rand(3,4)
b = torch.rand(4,5)
print(a, b)

print('expected:', torch.matmul(a,b))

#print('output:', custom_mm.cublas_mmul(b,a))
print('output:', custom_mm.cublas_mmul(a,b))

import torch
import custom_mm

a = torch.rand(4,4, dtype=torch.double)
b = torch.rand(4,4, dtype=torch.double)
print('a:', a)
print('b:', b)

print('expected:', torch.matmul(a,b))

print('output:', custom_mm.cusparse_mmul(a,b))
#print('output:', custom_mm.cublas_mmul(a,b))

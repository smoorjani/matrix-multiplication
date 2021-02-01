import torch
import custom_mm

custom_mm.init_cublas()
a = torch.rand(4, 4)
b = torch.rand(4, 4)
print('a:', a)
print('b:', b)

print('expected:', torch.matmul(a, b))
print('output:', custom_mm.cublas_mmul(a, b))
custom_mm.destroy_cublas()
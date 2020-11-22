import torch
import cublas_mm

a = torch.rand(4,4)
b = torch.rand(4,4)
print(a, '\n', b)

c = cublas_mm.mmul(a,b)
print('\n', c, '\n', torch.matmul(a,b))
#print('\n', torch.matmul(torch.transpose(a,0,1), torch.transpose(b,0,1)))
print(torch.matmul(b,a))


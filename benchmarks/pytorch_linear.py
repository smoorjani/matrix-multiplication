'''
This test file exists for when we have changed the torch.nn.Linear module to use our kernel
'''
import torch
import torch.nn as nn
import matmuls

layer = nn.Linear(768,768).cuda()
x = torch.rand(128, 768)

output = layer(x)
print(output)

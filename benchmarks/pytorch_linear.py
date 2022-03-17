'''
This test file exists for when we have changed the torch.nn.Linear module to use our kernel
'''
import torch
import torch.nn as nn
import matmuls
import random

random.seed(0)
torch.manual_seed(0)

layer = nn.Linear(768,768).cuda()
x = torch.rand(16, 16, 768).cuda()

output = layer(x)
print(output)


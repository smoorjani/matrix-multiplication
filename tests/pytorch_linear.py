'''
This test file exists for when we have changed the torch.nn.Linear module to use our kernel
'''

import torch
import torch.nn as nn
import random

torch.manual_seed(0)
random.seed(0)

layer = nn.Linear(768,768)
# x = torch.rand(128, 768)
x = torch.rand(16, 16, 768)

output = layer(x)
print(output)

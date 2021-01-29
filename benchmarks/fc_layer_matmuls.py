import torch
from scipy.sparse import random
import numpy as np
import time
import torch_blocksparse
from custom_mm import (
    cublas_mmul,
    cusparse_mmul,
    init_cublas,
    destroy_cublas,
    init_cusparse,
    destroy_cusparse
)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..tests.cublas_fc_layer import cublasLinear
from ..tests.cusparse_fc_layer import cusparseLinear

import torchvision.transforms
from torchvision import datasets, transforms

# Seed to reproduce results
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

init_cublas()
init_cusparse()

# Hyperparameters
batch_size = 32
learning_rate = 0.01
epochs = 1
log_interval = 10

layer_size = 1024


class regNet(nn.Module):
    # Regular MM operation
    def __init__(self):
        super(regNet, self).__init__()
        self.fc1 = nn.Linear(layer_size*layer_size, 10)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x)


class cubNet(nn.Module):
    # Cublas MM operation
    def __init__(self):
        super(cubNet, self).__init__()
        self.fc1 = cublasLinear(layer_size*layer_size, 10)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x)


class cuspNet(nn.Module):
    # Cusparse MM operation
    def __init__(self):
        super(cuspNet, self).__init__()
        self.fc1 = cusparseLinear(layer_size*layer_size, 10)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x)


reg_net = regNet()
cub_net = cubNet()
cusp_net = cuspNet()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/files/', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(
                           (0.1307,), (0.3081,)
                       )
                   ])),
    batch_size=batch_size, shuffle=True)

for net in [reg_net, cub_net, cusp_net]:
    # Load in MNIST data

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    train_losses = []

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())

        print('Epoch took {} with training loss of {}'.format(
            time.time() - epoch_t0, np.average(train_losses)))


destroy_cublas()
destroy_cusparse()

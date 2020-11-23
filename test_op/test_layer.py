'''
Code derived from
https://github.com/adventuresinML/adventures-in-ml-code/blob/master/pytorch_nn.py

https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py

'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torchvision import datasets, transforms

from fc_layer import regLinear, cublasLinear

# Seed to reproduce results
np.random.seed(0)
torch.manual_seed(0)

# Hyperparameters
batch_size=1
learning_rate=0.01
epochs=1
log_interval=10

# Regular MM operation 
class regNet(nn.Module):
    def __init__(self):
        super(regNet, self).__init__()
        self.fc1 = regLinear(28*28,10)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x)

# Cublas MM operation
class cubNet(nn.Module):
    def __init__(self):
        super(cubNet, self).__init__()
        self.fc1 = cublasLinear(28*28,10)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x)

reg_net = regNet()
cub_net = cubNet()

for net in [reg_net, cub_net]:
    # Load in MNIST data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    # main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 10:
                # break because of extended execution time
                break
            
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28*28)

            optimizer.zero_grad()
            net_out = net(data)

            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss))

    for param in net.parameters():
        print(param)


    

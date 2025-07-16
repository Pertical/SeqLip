import os
import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = (1,1,28,28)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# see:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.maxpool = nn.MaxPool2d(2) 

        self.fc1 = nn.Linear(16*4*4,120) # note cifar10 is 16*5*5
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def net():
    # load the checkpoint file
    net = LeNet()
    ckpt_file = 'models/custom_net_ckpt_99.ckpt'
    net.load_state_dict(torch.load(ckpt_file)['net'])
    
    # create an ordered list of the network's functions
    relu = torch.nn.ReLU(inplace=False)
    flatten = nn.Flatten()
    net.functions = [net.conv1,
                     relu,
                     net.maxpool,
                     net.conv2,
                     relu,
                     net.maxpool,
                     flatten,
                     net.fc1,
                     relu,
                     net.fc2,
                     relu,
                     net.fc3]

    return net

# the mnist_5 network, but intended to be run through the functions intended
# for custom networks
import torch.nn as nn
from models import mnist_5

input_size = (1,1,28,28)

def net():

    # get the original network
    net = mnist_5.mnist_5()

    # get a list of the network's functions
    relu = nn.ReLU(inplace=False)
    flatten = nn.Flatten()
    net.functions = [net.conv1,
                     relu,
                     net.conv2,
                     relu,
                     net.conv3,
                     relu,
                     net.conv4,
                     relu,
                     net.conv5,
                     flatten]
    return net

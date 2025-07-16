""" Starting point of the script is a saving of all singular values and vectors
in custom_save/

We perform the 100-optimization implemented in optim_nn_pca_greedy
"""
import os
import math
import torch
import torch.nn as nn
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

#from experiments.bruteforce_optim import optim_nn_pca_greedy
from seqlip import optim_nn_pca_greedy

#import models.mnist_5_custom as custom_network # verification
import models.custom_network as custom_network

# network
save_dir = 'custom_save' # folder to save results in
net = custom_network.net()
input_size = custom_network.input_size
#net = net.cuda()

n_sv = 200 # number of singular values to use in the k_generic_power_method in the max_eigenvalue function

for p in net.parameters():
    p.requires_grad = False

compute_module_input_sizes(net, input_size)

# indices of convolutions and linear layers
convs = []
lins = []
for i, function in enumerate(net.functions):
    if isinstance(function, nn.Conv2d):
        convs.append(i)
    elif isinstance(function, nn.Linear):
        lins.append(i)

'''
# Indices of convolutions and linear layers
convs = [0, 3, 6, 8, 10]
lins = [1, 4, 6]
'''

lip_spectral = 1
lip = 1

##############
# Convolutions
##############
for i in range(len(convs) - 1):
    print('Dealing with convolution {}'.format(i))
    U = torch.load(os.path.join(save_dir, 'feat-left-sing-Conv2d-{}'.format(convs[i])))
    U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
    su = torch.load(os.path.join(save_dir, 'feat-singular-Conv2d-{}'.format(convs[i])))
    su = su[:n_sv]

    V = torch.load(os.path.join(save_dir, 'feat-right-sing-Conv2d-{}'.format(convs[i+1])))
    V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
    sv = torch.load(os.path.join(save_dir, 'feat-singular-Conv2d-{}'.format(convs[i+1])))
    sv = sv[:n_sv]
    print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
    print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

    U, V = U.cpu(), V.cpu()


    if i == 0:
        sigmau = torch.diag(torch.Tensor(su))
    else:
        sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))
    if i == len(convs) - 2:
        sigmav = torch.diag(torch.Tensor(sv))
    else:
        sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))
    expected = sigmau[0,0] * sigmav[0,0]
    print('Expected: {}'.format(expected))
    lip_spectral *= float(expected)

    try:
        curr, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau)
        print('Approximation: {}'.format(curr))
        lip *= float(curr)
    except:
        print('Probably something went wrong...')
        lip *= float(expected)


#########
# Linears
#########
for i in range(len(lins) - 1):
    print('Dealing with linear layer {}'.format(i))
    U = torch.load(os.path.join(save_dir, 'feat-left-sing-Linear-{}'.format(lins[i])))
    U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
    su = torch.load(os.path.join(save_dir, 'feat-singular-Linear-{}'.format(lins[i])))
    su = su[:n_sv]

    V = torch.load(os.path.join(save_dir, 'feat-right-sing-Linear-{}'.format(lins[i+1])))
    V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
    sv = torch.load(os.path.join(save_dir, 'feat-singular-Linear-{}'.format(lins[i+1])))
    sv = sv[:n_sv]
    print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
    print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

    U, V = U.cpu(), V.cpu()

    sigmau = torch.diag(torch.Tensor(su))
    sigmav = torch.diag(torch.Tensor(sv))
    if i == 0:
        sigmau = torch.diag(torch.Tensor(su))
    else:
        sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))
    if i == len(lins) - 2:
        sigmav = torch.diag(torch.Tensor(sv))
    else:
        sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))
    expected = sigmau[0,0] * sigmav[0,0]
    print('Expected: {}'.format(expected))
    lip_spectral *= float(expected)

    try:
        curr, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau)
        print('Approximation: {}'.format(curr))
        lip *= float(curr)
    except:
        print('Probably something went wrong...')
        lip *= float(expected)

print('Lipschitz spectral: {}'.format(lip_spectral))
print('Lipschitz approximation: {}'.format(lip))

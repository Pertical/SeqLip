""" Starting point of the script is a saving of all singular values and vectors
in alex_save/

We perform the 100-optimization implemented in optim_nn_pca_greedy
"""
import math
import torch
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

#from experiments.bruteforce_optim import optim_nn_pca_greedy
from seqlip import optim_nn_pca_greedy

alex = torchvision.models.alexnet(pretrained=True)
alex = alex.cuda()

for p in alex.parameters():
    p.requires_grad = False

compute_module_input_sizes(alex, [1, 3, 224, 224])

#n_sv = 1
n_sv = 200

U = torch.load('alex_save/feat-left-sing-Conv2d-8')

# Indices of convolutions and linear layers
convs = [0, 3, 6, 8, 10]
lins = [1, 4, 6]
layers = ['Conv2d-0', 'Conv2d-3', 'Conv2d-6', 'Conv2d-8', 'Conv2d-10', 
          'Linear-1', 'Linear-4', 'Linear-6']

lip_spectral = 1
lip = 1


##########################
# Convolutions and Linears
##########################
for i in range(len(layers) - 1):
    print('Dealing with ', layers[i])
    U = torch.load('alex_save/feat-left-sing-' + layers[i])
    U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
    su = torch.load('alex_save/feat-singular-' + layers[i])
    su = su[:n_sv]

    V = torch.load('alex_save/feat-right-sing-' + layers[i+1])
    V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
    sv = torch.load('alex_save/feat-singular-' + layers[i+1])
    sv = sv[:n_sv]
    print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
    print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

    U, V = U.cpu(), V.cpu()

    # equivalent to "if i == 0" in the original code 
    if (layers[i] == 'Conv2d-0') or (layers[i] == 'Linear-1'): 
        sigmau = torch.diag(torch.Tensor(su))
    else:
        sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))

    # equivalent to "if i == len(convs)-2" or "if i == len(lins) -2" in the
    # original code 
    if (layers[i] == 'Conv2d-8') or (layers[i] == 'Linear-4'): 
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

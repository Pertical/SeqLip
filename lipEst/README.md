*(in this fork, I try to correct errors in the original repo)*

# Lipschitz Estimation

Repository for the paper [Lipschitz Regularity of deep neural networks:
analysis and efficient estimation](https://arxiv.org/abs/1805.10965)

Basic Python dependencies needed: PyTorch >= 0.3


### Code organisation

* `lipschitz_approximations.py`: many estimators
* `lipschitz_utils.py`: toolbox for the different estimators
* `seqlip.py`: SeqLip and GreedySeqLip
* `training.py`: general scheme for train/test
* `utils.py`: utility functions

### How to run the examples

**run the MNIST example:**

```bash
$ python -m models.mnist_5            # train the network
$ python -m experiments.mnist_get_sv  # save the singular values
$ python -m experiments.mnist         # run the example
```

**run the AlexNet example:**
```bash
$ python -m experiments.alex_get_sv   # save the singular values
$ python -m experiments.alex          # run the example
```

**run a custom network:**

Define your network in the file `experiments/custom_network.py`. This file should have a funtion `net()` which returns the (trained) network object. This object should have a `functions` attribute which is an ordered list of all the functions in the network. 

```bash
$ python -m experiments.custom_get_sv # save the singular values
$ python -m experiments.custom        # run the example
```

Note that I verified the custom network code was working correctly by running it on the MNIST network.

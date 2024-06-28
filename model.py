import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.


def Model(input_dims, output_dims):
    '''this shouldn't need to be a class. this model should be very simple.
    especially for cartpole, like, 2 dense, an output, and no convolutions 
    should be more than enough.'''

    '''PyTorch defines the size of the matrix multiplication operation between layers
        (why there's 2 dimensions) rather than defining the size of each individual layer
        like TensorFlow does.'''
    model = nn.Sequential(
        nn.Linear(input_dims, 20),
        nn.ReLU(),
        nn.Linear(20, 18),
        nn.ReLU(),
        nn.Linear(18, output_dims),
    )

    return model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.


def Model():
    '''this shouldn't need to be a class. this model should be very simple.
    especially for cartpole, like, 2 dense, an output, and no convolutions 
    should be more than enough.'''

    model = nn.Sequential(
        nn.Linear(4, 18),
        nn.ReLU(),
        nn.Linear(18, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    return
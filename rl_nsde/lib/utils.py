import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.)


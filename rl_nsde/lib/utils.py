import torch
import numpy as np
from typing import List


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.)

def toggle(parameters: List[torch.nn.Parameter], to: bool):
    for p in parameters:
        p.requires_grad_(to)


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()



def sample_x0(batch_size, d, device):
    x0 = -2 + 4*torch.rand(batch_size, d, device=device)
    return x0

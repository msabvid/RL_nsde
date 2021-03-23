import torch
import torch.nn as nn
from typing import Tuple, List
import tqdm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from rl_nsde.control.pontryagin import LQR_solved
from rl_nsde.lib.config import CoefsLQR
from rl_nsde.lib.utils import sample_x0, to_numpy




def evaluate(ts: torch.Tensor,
        d,
        coefs_lqr,
        control_problem,
        device,
        base_dir):

    control_solved = LQR_solved(d=d, **vars(coefs_lqr)) 
    
    #x0 = sample_x0(batch_size=10000, d=d, device=device)
    batch_size = 10000
    x0 = torch.ones(batch_size, d, device=device)
    with torch.no_grad():
        x, brownian_increments, actions, rewards = control_problem.sdeint(ts=ts, x0=x0)
        x_solved, _, actions_solved, rewards_solved = control_solved.sdeint(ts=ts, x0=x0, brownian_increments=brownian_increments)
    
    increments = (ts[1:] - ts[:-1]).reshape(1,-1,1).repeat(batch_size,1,1)
    value = (rewards[:,:-1,:]*increments).sum(1) + rewards[:,-1,:]
    value = to_numpy(value)
    value_solved = (rewards_solved[:,:-1,:]*increments).sum(1) + rewards_solved[:,-1,:]
    value_solved = to_numpy(value_solved)

    plt.hist(value, label="DL", alpha=0.5, bins=100)
    plt.hist(value_solved, label="analytical solution", alpha=0.5, bins=100)
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'hist_values.pdf'))
    






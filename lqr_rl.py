import torch
import torch.nn as nn
from typing import Tuple, List
import tqdm
import argparse
import os
import numpy as np
from dataclasses import dataclass

from control.pontryagin import Controlled_NSDE, LQR, RL_NSDE
from lib.functions import Drift_linear, Diffusion_constant, QuadraticRunningCost, QuadraticFinalCost



class CoefsLQR:
    """Coefficients that we use in the LQR model
    """
    def __init__(self, sigma, d, device):
        self.L = torch.zeros(d, d).to(device)
        self.M = torch.eye(d).to(device)
        self.C = torch.zeros(d, d).to(device)
        self.D = torch.zeros(d, d).to(device)
        self.F = torch.zeros(d, d).to(device)
        self.R = torch.eye(d).to(device)
        self.sigma = 1



def sample_x0(batch_size, d, device):
    x0 = -2 + 4*torch.rand(batch_size, d, device=device)
    return x0


def train(T: int,
        n_steps: int,
        d: int,
        ffn_hidden: List[int],
        max_updates: int,
        batch_size: int,
        base_dir: str,
        device: str,
        sigma: float,
        bsde_it: int,
        policy_it: int):
    
    # create model
    coefs_lqr = CoefsLQR(sigma=sigma, d=d, device=device)

    fbsde = FBSDE(d=d, 
            ffn_hidden=ffn_hidden, 
            drift=drift_lqr, 
            diffusion=diffusion_lqr, 
            running_cost=running_cost,
            final_cost=final_cost)
    fbsde.to(device)
    ts = torch.linspace(0, T, n_steps+1, device=device)

    # optimizers
    optimizer_policy = torch.optim.RMSprop(fbsde.alpha.parameters(), lr=0.001)
    parameters_bsde = list(fbsde.Y.parameters())+list(fbsde.Z.parameters())
    optimizer_bsde = torch.optim.RMSprop(parameters_bsde, lr=0.001)

    # Train
    pbar = tqdm.tqdm(total=max_updates)
    count_updates = 0
    while(True):
        pbar.write("Solving BSDE...")
        # solve bsde
        for it in range(bsde_it):
            optimizer_bsde.zero_grad()
            x0 = sample_x0(batch_size=batch_size, d=d, device=device)
            loss = fbsde.bsdeint(ts, x0)
            loss.backward()
            optimizer_bsde.step()
            count_updates += 1
        pbar.write("loss bsde={:.4f}".format(loss.item()))
        pbar.write("Improving policy...")
        # improve policy
        for it in range(policy_it):
            optimizer_policy.zero_grad()
            x0 = sample_x0(batch_size=batch_size, d=d, device=device)
            loss = fbsde.loss_policy(ts, x0)
            loss.backward()
            optimizer_policy.step()
            count_updates += 1
        pbar.write("loss policy={:.4f}".format(loss.item()))
        pbar.update(bsde_it + policy_it)
        if count_updates > max_updates:
            break
    result = {"state":fbsde.state_dict()}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # general aguments for code to work
    parser.add_argument("--base_dir", default="./numerical_results",type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--use_cuda", action='store_true', default=True)
    parser.add_argument("--seed", default=0, type=int)
    # arguments for network architecture and for training
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--d", default=2, type=int)
    parser.add_argument("--hidden_dims", default=[20,20], nargs="+", type=int)
    parser.add_argument("--max_updates", type=int, default=500)
    parser.add_argument('--ffn_hidden', default=[20,20,20], nargs="+", type=int, help="hidden sizes of ffn networks approximations")
    # arguments for LQR problem set up
    parser.add_argument("--T", default=5, type=int, help="horizon time of control problem")
    parser.add_argument("--n_steps", default=100, type=int, help="equally distributed steps where ODE is evaluated")
    parser.add_argument("--sigma", default=1, type=float, help="constant diffusion forward process")
    # training BSDE and policy
    parser.add_argument("--bsde_it", default=50, type=int, help="bsde training iterations")
    parser.add_argument("--policy_it", default=10, type=int, help="policy training iterations")

    parser.add_argument("--visualize", action="store_true", default=False)
        
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    results_path = args.base_dir
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if args.visualize:
        visualize(T=args.T,
                n_steps=args.n_steps,
                d=args.d,
                ffn_hidden=args.ffn_hidden,
                base_dir=results_path,
                sigma=args.sigma)
    else:
        train(T=args.T,
                n_steps=args.n_steps,
                d=args.d,
                ffn_hidden=args.ffn_hidden,
                max_updates=args.max_updates,
                batch_size=args.batch_size,
                base_dir=results_path,
                device=device,
                sigma=args.sigma,
                bsde_it=args.bsde_it,
                policy_it=args.policy_it)

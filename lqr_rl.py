import torch
import torch.nn as nn
from typing import Tuple, List
import tqdm
import argparse
import os
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from rl_nsde.control.pontryagin import Controlled_NSDE, LQR, RL_NSDE
from rl_nsde.lib.config import CoefsLQR
from rl_nsde.lib.utils import toggle
from rl_nsde.control.lqr import riccati_ode, optimal_policy




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
        policy_it: int,
        nsde_it: int):
    
    # create model
    coefs_lqr = CoefsLQR(sigma=sigma, d=d, device=device)
    # we define the nsde
    nsde = RL_NSDE(d=d, ffn_hidden=ffn_hidden, **vars(coefs_lqr))
    nsde.to(device)
    # we define the LQR
    lqr = LQR(d=d, ffn_hidden=ffn_hidden, **vars(coefs_lqr))
    lqr.to(device)
    toggle(list(lqr.Y.parameters())+list(lqr.Z.parameters()), to=False)
    # time discretisation
    ts = torch.linspace(0, T, n_steps+1, device=device)

    # optimizers
    optimizer_policy = torch.optim.RMSprop(nsde.alpha.parameters(), lr=0.001)
    parameters_bsde = list(nsde.Y.parameters())+list(nsde.Z.parameters())
    optimizer_bsde = torch.optim.RMSprop(parameters_bsde, lr=0.001)
    parameters_nsde = list(nsde.drift.parameters()) + list(nsde.diffusion.parameters())
    optimizer_nsde = torch.optim.RMSprop(parameters_nsde, lr=1e-3)

    # Train
    pbar = tqdm.tqdm(total=max_updates)
    count_updates = 0
    loss_nsde_tracker, loss_bsde_tracker, loss_alpha_tracker = [], [], []
    while(True):
        
        # improve nsde
        pbar.write("Improving nsde ...")
        lqr.alpha.hard_update(nsde.alpha)
        toggle(nsde.alpha.parameters(), to=False)
        toggle(parameters_bsde, to=False)
        toggle(parameters_nsde, to=True)
        for it in range(nsde_it):
            optimizer_nsde.zero_grad()
            x0 = sample_x0(batch_size=batch_size, d=d, device=device)
            cost_lqr = lqr.get_cost_episode(ts=ts, x0=x0)
            cost_nsde = nsde.get_cost_episode(ts=ts, x0=x0)
            loss = (cost_lqr.mean()-cost_nsde.mean())**2
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters_nsde, 1e4)
            optimizer_nsde.step()
            count_updates += 1
            pbar.write("loss nsde={:.4f}".format(loss.item()))
            loss_nsde_tracker.append(loss.detach().item())
        pbar.update(nsde_it)
        
        #for k in range(5):
        # solve bsde
        pbar.write("Solving BSDE...")
        toggle(nsde.alpha.parameters(), to=False)
        toggle(parameters_bsde, to=True)
        toggle(parameters_nsde, to=False)
        for it in range(bsde_it):
            optimizer_bsde.zero_grad()
            x0 = sample_x0(batch_size=batch_size, d=d, device=device)
            loss = nsde.fbsdeint(ts, x0)
            loss.backward()
            optimizer_bsde.step()
            count_updates += 1
            pbar.write("loss bsde={:.4f}".format(loss.item()))
            loss_bsde_tracker.append(loss.detach().item())
        pbar.update(bsde_it)
        # improve policy
        pbar.write("Improving policy...")
        # we use the Augmented Hamiltonian
        nsde.policy_old.hard_update(nsde.alpha)
        toggle(nsde.policy_old.parameters(), to=False)
        toggle(nsde.alpha.parameters(), to=True)
        toggle(parameters_bsde, to=False)
        toggle(parameters_nsde, to=False)
        for it in range(policy_it):
            optimizer_policy.zero_grad()
            x0 = sample_x0(batch_size=batch_size, d=d, device=device)
            #loss = nsde.loss_policy(ts, x0)
            loss = nsde.loss_policy_augmented_Hamiltonian(ts, x0, rho=0.1)
            loss.backward()
            optimizer_policy.step()
            count_updates += 1
            loss_alpha_tracker.append(loss.detach().item())
            pbar.write("loss policy={:.4f}".format(loss.item()))
        pbar.update(policy_it)

        if count_updates > max_updates:
            break
    result = {"state":nsde.state_dict()}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

    # plots Losses
    plt.plot(np.array(loss_nsde_tracker))
    plt.title("Loss nsde")
    plt.savefig(os.path.join(base_dir,"loss_nsde.pdf"))
    plt.close()
    plt.plot(np.array(loss_bsde_tracker))
    plt.title("Loss bsde")
    plt.savefig(os.path.join(base_dir,"loss_bsde.pdf"))
    plt.close()
    plt.plot(np.array(loss_alpha_tracker))
    plt.title("Loss alpha")
    plt.savefig(os.path.join(base_dir,"loss_alpha.pdf"))
    plt.close()

    # plots drift and diffusion
    with torch.no_grad(): 
        x0 = sample_x0(batch_size=batch_size, d=d, device=device)
        x,brownian_increments, actions, reards = nsde.sdeint(ts, x0)
        model_drift = nsde.drift(x[:,:-1,:], actions[:,:-1,:]) # (batch_size, L, d)
        lqr_drift = torch.cat([lqr.drift(x[:,idx,:], actions[:,idx,:]).unsqueeze(1) for idx in range(actions.shape[1]-1)],1)
        
        model_diffusion = nsde.diffusion(x[:,:-1,:]) # (batch_size, L, d)
        lqr_diffusion = torch.cat([lqr.diffusion(x[:,idx,:]).unsqueeze(1) for idx in range(actions.shape[1]-1)],1)

    
    S = riccati_ode(ts=ts, **vars(coefs_lqr))
    optimal_pol = torch.cat([optimal_policy(coefs_lqr.D, coefs_lqr.M, S, x[:,idx,:]) for idx in range(actions.shape[1]-1)],1) 
        



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
    parser.add_argument("--max_updates", type=int, default=500)
    parser.add_argument('--ffn_hidden', default=[20,20,20], nargs="+", type=int, help="hidden sizes of ffn networks approximations")
    # arguments for LQR problem set up
    parser.add_argument("--T", default=5, type=int, help="horizon time of control problem")
    parser.add_argument("--n_steps", default=100, type=int, help="equally distributed steps where ODE is evaluated")
    parser.add_argument("--sigma", default=1, type=float, help="constant diffusion forward process")
    # training BSDE and policy
    parser.add_argument("--bsde_it", default=20, type=int, help="bsde training iterations")
    parser.add_argument("--policy_it", default=10, type=int, help="policy training iterations")
    parser.add_argument("--nsde_it", default=10, type=int, help="policy training iterations")

    parser.add_argument("--visualize", action="store_true", default=False)
        
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    results_path = os.path.join(args.base_dir, 'lqr_rl')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
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
            policy_it=args.policy_it,
            nsde_it=args.nsde_it)

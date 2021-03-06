import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod
import copy
from functools import partial

from ..lib.networks import FFN, Linear
from .functions import Hamiltonian, Drift_linear, Diffusion_constant, QuadraticRunningCost, QuadraticFinalCost
from ..lib.utils import to_numpy
from .lqr import riccati_ode, optimal_policy

class Controlled_NSDE(nn.Module):
    """
    Base class for a controlled NSDE
    """

    def __init__(self, d, ffn_hidden, **kwargs):
        super().__init__()
        
        self.d = d # dimension of the problem
        self.ffn_hidden = ffn_hidden

        self.drift = self._drift(**kwargs)#Drift_linear(L=lqr_config['L'], M=lqr_config['M'])
        self.diffusion = self._diffusion(**kwargs)#Diffusion_constant(sigma=lqr_config['sigma'])
        
        self.running_cost = self._running_cost(**kwargs)#QuadraticRunningCost(C=lqr_config['C'], D=lqr_config['D'], F=lqr_config['F'])
        self.final_cost = self._final_cost(**kwargs)#QuadraticFinalCost(R=lqr_config['R'])
        
        self.alpha = FFN(sizes = [d+1] + ffn_hidden + [d]) # +1 is for time
        self.policy_old = copy.deepcopy(self.alpha) # we will use this if we use the augmented Hamiltonian
        
        self.Y = FFN(sizes = [d+1] + ffn_hidden + [d]) # Adjoint state. +1 is for time
        self.Z = FFN(sizes = [d+1] + ffn_hidden + [d*d]) # Diffusion of adjoint BSDE. It takes values in R^{d\times d}
        self.H = Hamiltonian(drift = self.drift, diffusion=self.diffusion, running_cost=self.running_cost)
    
    @abstractmethod
    def _drift(self, **kwargs):
        ...
    
    @abstractmethod
    def _diffusion(self, **kwargs):
        ...

    @abstractmethod
    def _running_cost(self, **kwargs):
        ...

    @abstractmethod
    def _final_cost(self, **kwargs):
        ...
    
    def sdeint(self, ts, x0, brownian_increments = None):
        """
        Euler scheme to solve the SDE. Equivalent to playing an episode
        Parameters
        ----------
        ts: torch.Tensor
            timegrid. Vector of length L
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, L, d)
        brownian_increments:
            If not None, the brownian increments are given. They will be used to compare different policies on the same paths
        
        Returns
        -------
        x: torch.Tensor
            Sample of sde. Tensor of shape (batch_size, L, d)
        brownian_increments: torch.Tensor of shape (batch_size, L, d)
        actions: torch.Tensor
            Actions teaken. Tensor of shape (batch_size, L-1, d)
        rewards: torch.Tensor
            Rewards obtained. Tensor of shape (batch_size, L, 1). Last reward at index L is terminal cost
            
        Note
        ----
        I am assuming uncorrelated Brownian motion
        """
        batch_size = x0.shape[0]
        device = x0.device
        if brownian_increments == None:
            brownian_increments = torch.randn(batch_size, len(ts)-1, self.d, device=device) * (ts[1:] - ts[:-1]).reshape(1,-1,1).sqrt()
        #brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        x = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        x[:,0,:], x_old = x0, x0
        actions = torch.zeros_like(brownian_increments) # (batch_size, L, d)
        rewards = torch.zeros(batch_size, len(ts), 1, device=device)
        for idx, t in enumerate(ts[:-1]):
            h = ts[idx+1]-ts[idx]
            current_t = torch.ones(batch_size,1, device=device)*t
            a = self.alpha(current_t,x_old)
            # store action and reward
            actions[:,idx,:] = a
            rewards[:,idx,:] = self.running_cost(x_old, a)  
            # step of Euler scheme of controlled sde
            dW = brownian_increments[:,idx,:]
            x_new = x_old + self.drift(x_old, a)*h + self.diffusion(x_old)*dW
            x[:,idx+1,:] = x_new
            x_old = x_new

        # final reward
        rewards[:,-1,:] = self.final_cost(x_old)
        return x, brownian_increments, actions, rewards
    
    
    def fbsdeint(self, ts: torch.Tensor, x0: torch.Tensor): 
        """
        Forward backward sde. 
        Local errors at each timestep in timegrid ts to solve the BSDE
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        
        """
        # we go forward the sde
        with torch.no_grad():
            x, brownian_increments, _, _ = self.sdeint(ts, x0)
        
        device=x.device
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        
        Y = self.Y(tx) # (batch_size, L, d)
        Z = self.Z(tx).view(batch_size, len(ts), self.d, self.d) # (batch_size, L, d, d)
        loss_fn = nn.MSELoss()
        loss = 0
        # we get the local errors of the bsde to solve it, i.e. find parametrisation of the processes Y and Z
        final_value = self.final_cost.dx(x[:,-1,:], create_graph=False, retain_graph=False) # final value of the bsde, (batch_size, d)
        for idx,t in enumerate(ts):
            if t==ts[-1]:
                pred = Y[:,idx,:]
                target = final_value
            else:
                h = ts[idx+1] - ts[idx]
                y = Y[:,idx,:]
                with torch.no_grad():
                    current_t = t*torch.ones(batch_size, 1, device=device)
                    a = self.alpha(current_t, x[:,idx,:])
                z = Z[:,idx,:]
                stoch_int = torch.bmm(Z[:,idx,...], brownian_increments[:,idx,:].unsqueeze(2)).squeeze(2) # (batch_size, d)
                dHdx = self.H.dx(x=x[:,idx,:],
                        a=a,
                        y=y,
                        z=z,
                        create_graph=True,
                        retain_graph=True)
                pred = y - dHdx*h + stoch_int # euler timestep
                target = Y[:,idx+1,:].detach()
            loss += loss_fn(pred, target)
        return loss

    
    def loss_policy(self, ts: torch.Tensor, x0: torch.Tensor):
        """ 
        We want to find
        - argmin_{a} H(t,X,Y,Z,a) for all t, along the controlled sde
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        rho: float
            Parameter of Augmented Hamiltonian
        """
        with torch.no_grad():
            x, brownian_increments, _, _ = self.sdeint(ts, x0)
        batch_size = x.shape[0]
        device=x.device
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        with torch.no_grad():
            Y = self.Y(tx) # (batch_size, L, d)
            Z = self.Z(tx).view(batch_size, len(ts), self.d, self.d) # (batch_size, L, d, d)
        
        loss = 0
        for idx, t in enumerate(ts[:-1]):
            current_t = t*torch.ones(batch_size, 1, device=device)
            y = Y[:,idx,:]
            a = self.alpha(current_t, x[:,idx,:])
            z = Z[:,idx,:]
            H = self.H(x=x[:,idx,:],a=a,y=y,z=z)
            loss += H * (ts[idx+1]-ts[idx])
        return loss.mean()
    
    def loss_policy_augmented_Hamiltonian(self, ts: torch.Tensor, x0: torch.Tensor, rho: float):
        """ 
        We want to find
        - argmin_{a} H(t,X,Y,Z,a) for all t, along the controlled sde
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        rho: float
            Parameter of Augmented Hamiltonian
        policy_old: FFN
            Copy of the old policy parametrisation, needed in the Augmented Hamiltonian
        """
        with torch.no_grad():
            x, brownian_increments, _, _ = self.sdeint(ts, x0)
        batch_size = x.shape[0]
        device=x.device
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        with torch.no_grad():
            Y = self.Y(tx) # (batch_size, L, d)
            Z = self.Z(tx).view(batch_size, len(ts), self.d, self.d) # (batch_size, L, d, d)
        
        loss = 0
        for idx, t in enumerate(ts[:-1]):
            current_t = t*torch.ones(batch_size, 1, device=device)
            y = Y[:,idx,:]
            a = self.alpha(current_t, x[:,idx,:])
            a_old = self.policy_old(current_t, x[:,idx,:])
            z = Z[:,idx,:]
            H = self.H(x=x[:,idx,:],a=a,y=y,z=z) # (batch_size, 1)
            augmented_H = H + 0.5 * rho *((self.drift(x[:,idx,:], a) - self.drift(x[:,idx,:], a_old))**2).sum(1, keepdim=True)
            augmented_H += 0.5 * rho * ((self.H.dx(x=x[:,idx,:],a=a_old, y=y, z=z, create_graph=False, retain_graph=False) - self.H.dx(x=x[:,idx,:],a=a,y=y,z=z, create_graph=True, retain_graph=True))**2).sum(1,keepdim=True)
            loss += augmented_H * (ts[idx+1]-ts[idx])
        return loss.mean()
    
    
    def get_cost_episode(self, ts: torch.Tensor, x0: torch.Tensor):
    
        _,_,_, rewards = self.sdeint(ts, x0)
        batch_size = x0.shape[0]
        h = (ts[1:] - ts[:-1]).reshape(1,-1,1).repeat(batch_size, 1, 1) # (batch_size, L-1, 1)
        running_cost = (rewards[:,:-1,:] * h).sum(1) # Riemann sum of integral of running cost. (batch_size, 1)
        cost = running_cost + rewards[:,-1,:] # We sum terminal cost
        return cost # (batch_size, 1)

            
            
class LQR(Controlled_NSDE):

    def __init__(self, d: int, ffn_hidden: List[int], **kwargs):
        """
        Initialisation of the LQR problem that we want to solve.
        
        Paramters
        ---------
        d: int
            dim of the process
        ffn_hidden: List[int]
            hidden sizes of the Feedforward networks that parametrise the policy, Y, Z
        kwargs: Dict with config of LQR
        """
        super().__init__(d=d, ffn_hidden=ffn_hidden, **kwargs)
    
    def _drift(self, **kwargs):
        return Drift_linear(L=kwargs['L'], M=kwargs['M'])
    
    def _diffusion(self, **kwargs):
        return Diffusion_constant(sigma=kwargs['sigma']) 

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])


class LQR_solved:

    def __init__(self, d: int, **kwargs):
        """
        Initialisation of the LQR problem that we want to solve.
        
        Paramters
        ---------
        d: int
            dim of the process
        ffn_hidden: List[int]
            hidden sizes of the Feedforward networks that parametrise the policy, Y, Z
        kwargs: Dict with config of LQR
        """
        self.M = kwargs['M']
        self.L = kwargs['L']
        self.C = kwargs['C']
        self.D = kwargs['D']
        self.F = kwargs['F']
        self.R = kwargs['R']
        self.d = d
    
        self.drift = self._drift(**kwargs)#Drift_linear(L=lqr_config['L'], M=lqr_config['M'])
        self.diffusion = self._diffusion(**kwargs)#Diffusion_constant(sigma=lqr_config['sigma'])
        
        self.running_cost = self._running_cost(**kwargs)#QuadraticRunningCost(C=lqr_config['C'], D=lqr_config['D'], F=lqr_config['F'])
        self.final_cost = self._final_cost(**kwargs)#QuadraticFinalCost(R=lqr_config['R'])
    
    def _drift(self, **kwargs):
        return Drift_linear(L=kwargs['L'], M=kwargs['M'])
    
    def _diffusion(self, **kwargs):
        return Diffusion_constant(sigma=kwargs['sigma']) 

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])
    
    def sdeint(self, ts, x0, brownian_increments = None):
        """
        Euler scheme to solve the SDE with the optimal policy given by the analytical solution. Equivalent to playing an episode
        Parameters
        ----------
        ts: torch.Tensor
            timegrid. Vector of length L
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian_increments:
            If not None, the brownian increments are given. They will be used to compare different policies on the same paths
        
        Returns
        -------
        x: torch.Tensor
            Sample of sde. Tensor of shape (batch_size, L, d)
        brownian_increments: torch.Tensor of shape (batch_size, L, d)
        actions: torch.Tensor
            Actions teaken. Tensor of shape (batch_size, L-1, d)
        rewards: torch.Tensor
            Rewards obtained. Tensor of shape (batch_size, L, 1). Last reward at index L is terminal cost
            
        Note
        ----
        I am assuming uncorrelated Brownian motion
        """
        batch_size = x0.shape[0]
        device = x0.device
        if brownian_increments == None:
            brownian_increments = torch.randn(batch_size, len(ts)-1, self.d, device=device) * (ts[1:] - ts[:-1]).reshape(1,-1,1).sqrt()
        x = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        x[:,0,:], x_old = x0, x0
        actions = torch.zeros_like(brownian_increments) # (batch_size, L, d)
        rewards = torch.zeros(batch_size, len(ts), 1, device=device)
        S = riccati_ode(L=self.L, M=self.M, C=self.C, D=self.D, R=self.R, ts=ts)

        for idx, t in enumerate(ts[:-1]):
            h = ts[idx+1]-ts[idx]
            a = optimal_policy(x=x_old, D=self.D, M=self.M, S=S[idx])
            # store action and reward
            actions[:,idx,:] = a
            rewards[:,idx,:] = self.running_cost(x_old, a)  
            # step of Euler scheme of controlled sde
            dW = brownian_increments[:,idx,:]
            x_new = x_old + self.drift(x_old, a)*h + self.diffusion(x_old)*dW
            x[:,idx+1,:] = x_new
            x_old = x_new
        # final reward
        rewards[:,-1,:] = self.final_cost(x_old)
        return x, brownian_increments, actions, rewards
    

class RL_IRL_NSDE(Controlled_NSDE):    
    
    """
    We want to learn everything from the data:
        - The drift and the diffusion of the nsde
        - The optimal policy
        - The running cost and the final cost
    """

    def __init__(self, d, ffn_hidden, **kwargs):
        super().__init__(d=d, ffn_hidden=ffn_hidden, **kwargs)

    def _drift(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d])
    
    def _diffusion(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d]) # I consider the diffusion to be diagonal

    def _running_cost(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [1])

    def _final_cost(self, **kwargs):
        return FFN(sizes=[self.d] + self.ffn_hidden + [1])


class RL_NSDE(Controlled_NSDE):    
    
    """
    Model-based RL. We learn the following from the data:
        - The drift and the diffusion of the snde
        - The optimal policy
    The running cost and the final cost are given.

    lqr_config: Dict with config of LQR
    """
    
    def __init__(self, d, ffn_hidden, **kwargs):
        super().__init__(d=d, ffn_hidden=ffn_hidden, **kwargs)

    def _drift(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d])
    
    def _diffusion(self, **kwargs):
        return FFN(sizes=[self.d] + self.ffn_hidden + [self.d], output_activation=nn.Softplus) # I consider the diffusion to be diagonal

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])


class RL_Linear_NSDE(Controlled_NSDE):    
    
    """
    Model-based RL. We learn the following from the data:
        - The drift and the diffusion of the nsde
            - We impose the drift to be linear on (x,a), and the diffusion to be constant
        - The optimal policy
    The running cost and the final cost are given.

    lqr_config: Dict with config of LQR
    """
    
    def __init__(self, d, ffn_hidden, **kwargs):
        super().__init__(d=d, ffn_hidden=ffn_hidden, **kwargs)

    def _drift(self, **kwargs):
        return Linear(self.d+self.d, self.d)
        #return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d])
    
    def _diffusion(self, **kwargs):
        self.sigma = nn.Parameter(torch.tensor(1.))
        return Diffusion_constant(self.sigma)
        #return FFN(sizes=[self.d] + self.ffn_hidden + [self.d], output_activation=nn.Softplus) # I consider the diffusion to be diagonal

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])

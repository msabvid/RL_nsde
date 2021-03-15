import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod

from ..lib.networks import FFN
from .functions import Hamiltonian, Drift_linear, Diffusion_constant, QuadraticRuningCost, QuadraticFinalCost



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
        
        self.alpha = FFN(sizes = [d] + ffn_hidden + [d]) # 
        self.Y = FFN(sizes = [d+1] + ffn_hidden + [d]) # Adjoint state. +1 is for time
        self.Z = FFN(sizes = [d+1] + ffn_hidden + [d*d]) # Diffusion of adjoint BSDE. It takes values in R^{d\times d}
        self.H = Hamiltonian(drift = self.drift, diffusion=self.diffusion, running_cost=self.running_cost)
    
    @abstractmethod
    def _drift(self, **kwargs):
        ...
    
    @abstractmethod
    def _diffusion(self, **kwargs):
        ...

    @abtsractmethod
    def _running_cost(self, **kwargs):
        ...

    @abstractmethod
    def _final_cost(self, **kwargs):
        ...
    
    def sdeint(self, ts, x0):
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
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        x = torch.zeros(batch_size, len(ts), self.d, device=device) # (batch_size, L, d)
        x[:,0,:], x_old = x0, x0
        actions = torch.zeros_like(brownian_increments) # (batch_size, L, d)
        rewards = torch.zeros(batch_size, len(ts), 1)
        for idx, t in enumerate(ts[:-1]):
            h = ts[idx+1]-ts[idx]
            a = self.alpha(x)
            # store action and reward
            actions[:,idx,:] = a
            rewards[:,idx,:] = self.running_cost(x_old, a)  
            # step of Euler scheme of controlled sde
            dW = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            brownian_increments[:,idx,:] = dW #torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
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
                    a = self.alpha(x)
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
        for idx, t in enumerate(ts):
            current_t = t*torch.ones(batch_size, 1, device=device)
            y = Y[:,idx,:]
            a = self.alpha(x)
            z = Z[:,idx,:]
            H = self.H(x=x[:,idx,:],a=a,y=y,z=z)
            loss += H
        return loss.mean()

            
            
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
        super(Controlled_NSDE, self).__init__()
    
    def _drift(self, **kwargs):
        return Drift_linear(L=kwargs['L'], M=kwargs['M'])
    
    def _diffusion(self, **kwargs):
        return Diffusion_constant(sigma=kwargs['sigma']) 

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])


class RL_IRL_NSDE(Controlled_NSDE):    
    
        """
        We want to learn everything from the data:
            - The drift and the diffusion of the nsde
            - The optimal policy
            - The running cost and the final cost
        
        Paramters
        ---------
        d: int
            dim of the process
        ffn_hidden: List[int]
            hidden sizes of the Feedforward networks that parametrise the policy, Y, Z
        lqr_config: Dict with config of LQR
        """

    def __init__(self, d, ffn_hidden, **kwargs)
        super(Controlled_NSDE, self).__init__()

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

        Paramters
        ---------
        d: int
            dim of the process
        ffn_hidden: List[int]
            hidden sizes of the Feedforward networks that parametrise the policy, Y, Z
        lqr_config: Dict with config of LQR
        """
    
    def __init__(self, d, ffn_hidden, **kwargs)
        super(Controlled_NSDE, self).__init__()

    def _drift(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d])
    
    def _diffusion(self, **kwargs):
        return FFN(sizes=[self.d+self.d] + self.ffn_hidden + [self.d]) # I consider the diffusion to be diagonal

    def _running_cost(self, **kwargs):
        return QuadraticRunningCost(C=kwargs['C'], D=kwargs['D'], F=kwargs['F']) 

    def _final_cost(self, **kwargs):
        return QuadraticFinalCost(R=kwargs['R'])

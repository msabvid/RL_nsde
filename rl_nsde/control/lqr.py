import numpy as np
from scipy.integrate import odeint
from functools import reduce
import torch
from abc import abstractmethod, ABC

from ..lib.utils import to_numpy


def riccati_ode(L,M,C,D,R,ts, **kwargs):
    """
    Solve the Riccati ODE
    S'(t) = S(t)*M*D^{-1}*M^T*S(t) - C, S(T) = R 

    """
    device = L.device
    L = to_numpy(L)
    M = to_numpy(M)
    C = to_numpy(C)
    D = to_numpy(D)
    R = to_numpy(R)
    ts = to_numpy(ts)
    
    d = R.shape[0]
    
    def func(y,t):
        y = y.reshape(d,d)
        f = reduce(lambda x,y: np.matmul(x,y), [y,M,np.linalg.inv(D),M.T,y]) - C
        return -1 * f.flatten() # We multiply by -1 because we are reversing time
    
    sol = odeint(func, y0=R.flatten(), t=ts)
    sol = np.flip(sol, axis=0) # we reverse time back to its original form
    return torch.tensor(sol.reshape(-1,d,d).copy(), device=device).float()




def optimal_policy(D,M,S,x):
    """
    alpha^*(t,x) = -D^{-1}*M^T*S * x

    Parameters
    ----------
    D: torch.Tensor of shape (d,d) given in the LQR problem
    M: torch.Tensor of shape (d,d) given in the LQR problem
    S: torch.Tensor of shape (d,d) solution of the Ricatti equation
    x: torch.Tensor of shape (batch_size, d)
    """

    mat = reduce(lambda x,y: torch.matmul(x,y), [torch.inverse(D),M.T,S]) # (d,d)
    return torch.matmul(-mat, x.unsqueeze(2)).squeeze(2) # (batch_size, d)

    



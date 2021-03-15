import torch


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

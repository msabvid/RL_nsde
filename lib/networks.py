import torch
import torch.nn as nn
from collections import namedtuple
from typing import Tuple

class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, *args):
        x = torch.cat(args, -1)
        return self.net(x)
    
    def dx(self, *args, create_graph=False, retain_graph=False):
        x = torch.cat(args, -1).requires_grad_(True)
        g = self.forward(x)
        dgdx = torch.autograd(g, x, grad_outputs=torch.ones_like(g), only_inputs=True, create_graph=create_graph, retain_graph=retain_graph)[0]
        return dgdx

    def hard_update(self, new):
        for old_p, new_p in zip(self.parameters(), new.parameters()):
            old_p.data.copy_(new_p.data)






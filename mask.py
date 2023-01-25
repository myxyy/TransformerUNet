import torch
import torch.nn as nn

class SelfMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        i = dim
        self.mask = torch.full((i,i), float('-inf')).triu()
    def forward(self, x):
        return x + self.mask

class CrossEncodeMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        i = dim
        self.mask = nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).tril(-1).view(1,1,i,i)).view(i*2,i).transpose(1,0)
    def forward(self, x):
        return x + self.mask

class CrossDecodeMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        i = dim
        self.mask = nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).triu().view(1,1,i,i)).view(i*2,i)
    def forward(self, x):
        return x + self.mask

import torch
import torch.nn as nn

class SelfMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask = nn.Parameter(-1/torch.full((dim,dim), 1.).tril()+1, requires_grad=False)
    def forward(self, x):
        return x + self.mask

class CrossEncodeMask(nn.Module):
    def __init__(self, dim_kv, dim_q):
        super().__init__()
        dim_min = min(dim_kv, dim_q)
        self.mask = nn.Parameter((-1/(1-nn.Upsample((dim_kv, dim_q))(torch.full((dim_min,dim_min),1.).tril().view(1,1,dim_min,dim_min)).view(dim_kv, dim_q).roll(1,0)).index_add(0,torch.tensor([0]),torch.full((1,dim_q),1.))+1).transpose(1,0), requires_grad=False)
    def forward(self, x):
        return x + self.mask

class CrossDecodeMask(nn.Module):
    def __init__(self, dim_kv, dim_q):
        super().__init__()
        dim_min = min(dim_kv, dim_q)
        self.mask = nn.Parameter(-1/nn.Upsample((dim_q, dim_kv))(torch.full((dim_min,dim_min),1.).tril().view(1,1,dim_min,dim_min)).view(dim_q, dim_kv)+1, requires_grad=False)
    def forward(self, x):
        return x + self.mask

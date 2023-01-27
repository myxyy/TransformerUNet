import torch
import torch.nn as nn

class SelfMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask = torch.full((dim,dim), float('-inf')).triu(1).cuda()
    def forward(self, x):
        return x + self.mask

class CrossEncodeMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.mask = nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).tril(-1).view(1,1,i,i)).view(i*2,i).transpose(1,0).cuda()
        self.mask = nn.Upsample((dim*2,dim))(torch.full((dim,dim), float('-inf')).tril(-1).view(1,1,dim,dim)).view(dim*2,dim).roll(-1,0).index_add(0,torch.tensor([2*dim-1]),torch.full((1,dim),float('-inf'))).transpose(1,0).cuda()
    def forward(self, x):
        return x + self.mask

class CrossDecodeMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.mask = nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).triu().view(1,1,i,i)).view(i*2,i).cuda()
        self.mask = nn.Upsample((dim*2,dim))(torch.full((dim,dim), float('-inf')).triu(1).view(1,1,dim,dim)).view(dim*2,dim).cuda()
    def forward(self, x):
        return x + self.mask

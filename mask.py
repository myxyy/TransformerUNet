import torch
import torch.nn as nn

class CrossEncodeMask(nn.Module):
    def __init__(self, dim_kv, dim_q, span=None):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(dim_kv), torch.arange(dim_q), indexing='ij')
        mask = torch.full((dim_kv, dim_q), -float('inf'))
        is_zero = (row * dim_q) <= (col * dim_kv)
        if span is not None:
            is_span = row > (col * dim_kv / dim_q).to(int) - span
            is_zero = torch.logical_and(is_zero, is_span)
        zero_ind = torch.where(is_zero)
        mask[row[zero_ind],col[zero_ind]] = 0
        mask = mask.transpose(1,0)
        self.mask = nn.Parameter(mask, requires_grad=False)
        #self.mask = nn.Parameter((-1/(1-nn.Upsample((dim_kv, dim_q))(torch.full((dim_min,dim_min),1.).tril().view(1,1,dim_min,dim_min)).view(dim_kv, dim_q).roll(1,0)).index_add(0,torch.tensor([0]),torch.full((1,dim_q),1.))+1).transpose(1,0), requires_grad=False)
    def forward(self, x):
        return x + self.mask

class SelfMask(CrossEncodeMask):
    def __init__(self, dim, span=None):
        super().__init__(dim, dim, span)

class CrossDecodeMask(nn.Module):
    def __init__(self, dim_kv, dim_q, span=None):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(dim_q), torch.arange(dim_kv), indexing='ij')
        mask = torch.full((dim_q, dim_kv), -float('inf'))
        is_zero = (row * dim_kv) >= (col * dim_q)
        if span is not None:
            is_span = row < (col * dim_q / dim_kv).to(int) + span
            is_zero = torch.logical_and(is_zero, is_span)
        zero_ind = torch.where(is_zero)
        mask[row[zero_ind],col[zero_ind]] = 0
        mask[row[zero_ind],col[zero_ind]] = 0
        self.mask = nn.Parameter(mask, requires_grad=False)
        #self.mask = nn.Parameter(-1/nn.Upsample((dim_q, dim_kv))(torch.full((dim_min,dim_min),1.).tril().view(1,1,dim_min,dim_min)).view(dim_q, dim_kv)+1, requires_grad=False)
    def forward(self, x):
        return x + self.mask

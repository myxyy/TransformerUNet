import copy
import torch
import torch.nn as nn
import math

class SparseMHAEncoder(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_QK, dim_V, head_num, dim_out, span, stride, length_q, length_kv):
        super().__init__()
        self.head_num = head_num
        self.dim_QK = dim_QK
        self.dim_V = dim_V
        self.qQ = nn.Linear(dim_q, head_num * dim_QK, bias=False)
        self.kK = nn.Linear(dim_k, head_num * dim_QK, bias=False)
        self.vV = nn.Linear(dim_v, head_num * dim_V, bias=False)
        self.linear_out = nn.Linear(head_num * dim_V, dim_out, bias=False)
        self.span = span
        self.stride = stride
        self.length_q = length_q
        self.length_kv = length_kv

        row, col = torch.meshgrid(torch.arange(span), torch.arange(length_q), indexing='ij')
        kvi = row - (span-1) + col*stride
        is_valid_kvi = torch.logical_and(0 <= kvi, kvi < length_kv)
        ivkvi_row = row[torch.where(~is_valid_kvi)]
        ivkvi_col = col[torch.where(~is_valid_kvi)]
        table_QK_mask = torch.zeros(span, length_q)
        table_QK_mask[ivkvi_row, ivkvi_col] = -float('inf')
        self.table_QK_mask = nn.Parameter(table_QK_mask, requires_grad=False)

        conv_K_weight = torch.zeros(span, 1, span)
        conv_K_weight[torch.arange(span),:,torch.arange(span)] = 1
        self.conv_K = nn.Conv1d(1, span, span, stride)
        self.conv_K.weight.data = conv_K_weight
        self.conv_K.weight.requires_grad = False

        self.conv_V = self.conv_K

    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v):
        batch = q.shape[0]

        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vV(v) # (batch, length_kv, head_num * dim_V)
        
        bh = batch * self.head_num
        K = K.view(batch, self.length_kv, self.head_num, self.dim_QK).permute(0,2,3,1).reshape(bh*self.dim_QK, 1, self.length_kv)
        V = V.view(batch, self.length_kv, self.head_num, self.dim_V).permute(0,2,3,1).reshape(bh*self.dim_V, 1, self.length_kv)

        K = torch.nn.functional.pad(K,(self.span-1,0))
        table_K = self.conv_K(K) # (bh*dim_QK, span, ?)
        #table_K = table_K[:,:,:self.length_q]
        table_K = table_K.reshape(bh, self.dim_QK, self.span, self.length_q, 1).permute(0,2,3,1,4) # (bh, span, length_q, dim_QK, 1)
        table_Q = Q.expand(self.span,batch,self.length_q,self.head_num*self.dim_QK)
        table_Q = table_Q.reshape(self.span,batch,self.length_q,self.head_num,1,self.dim_QK)
        table_Q = table_Q.permute(1,3,0,2,4,5).reshape(bh,self.span,self.length_q,1,self.dim_QK) # (bh, span, length_q, 1, dim_QK)
        table_QK = torch.matmul(table_Q, table_K)/(self.dim_QK ** 0.5)
        table_QK = table_QK.reshape(bh, self.span, self.length_q) # (bh, span, length_q)
        table_QK = table_QK + self.table_QK_mask
        table_QK = table_QK.softmax(1)
        V = torch.nn.functional.pad(V,(self.span-1,0))
        table_V = self.conv_V(V).reshape(bh, self.dim_V, self.span, self.length_q).permute(0,2,3,1) # (bh, span, length_q, dim_V)
        table_QKV = table_QK.unsqueeze(3) * table_V # (bh, span, length_q, dim_V)
        out = table_QKV.sum(1)
        out = out.reshape(batch, self.head_num, self.length_q, self.dim_V)
        out = out.transpose(2,1)
        out = out.reshape(batch, self.length_q, self.head_num*self.dim_V)
        out = self.linear_out(out)

        return out

class SparseSelfTransformerBlock(nn.Module):
    def __init__(self, dim_q, head_num, dropout, span, stride, length_q, length_kv):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_self = SparseMHAEncoder(dim_q, dim_q, dim_q, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride, length_q, length_kv)
        self.mlp1 = nn.Linear(dim_q, dim_q*2)
        self.mlp2 = nn.Linear(dim_q*2, dim_q)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # key_value is assumed to be layer-normalized
    def forward(self, query):

        x = self.layer_norm(query)
        x = self.mmha_self(x, x, x)
        x = self.dropout(x)
        query = x + query

        x = self.layer_norm(query)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        query = x + query

        return query

class SparseSelfTransformer(nn.Module):
    def __init__(self, dim_q, head_num, depth, dropout, span, stride, length_q, length_kv):
        super().__init__()
        encoder_block = SparseSelfTransformerBlock(dim_q, head_num, dropout, span, stride, length_q, length_kv)
        self.encoder_block_list = nn.ModuleList([copy.deepcopy(encoder_block) for i in range(depth)])

    def forward(self, query):
        for encoder_block in self.encoder_block_list:
            query = encoder_block(query)
        return query

class SparseCrossTransformerEncoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, dropout, span, stride, length_q, length_kv):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_cross = SparseMHAEncoder(dim_q, dim_kv, dim_kv, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride, length_q, length_kv)
        self.mlp1 = nn.Linear(dim_q, dim_q*2)
        self.mlp2 = nn.Linear(dim_q*2, dim_q)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # key_value is assumed to be layer-normalized
    def forward(self, query, key_value):

        x = self.layer_norm(query)
        x = self.mmha_cross(x, key_value, key_value)
        x = self.dropout(x)
        query = x + query

        x = self.layer_norm(query)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        query = x + query

        return query

class SparseCrossTransformerEncoder(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, depth, dropout, span, stride, length_q, length_kv):
        super().__init__()
        decoder_block = SparseCrossTransformerEncoderBlock(dim_q, dim_kv, head_num, dropout, span, stride, length_q, length_kv)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value_normed)
        return query

class SparseMHADecoder(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_QK, dim_V, head_num, dim_out, span, stride, length_q, length_kv):
        super().__init__()
        self.head_num = head_num
        self.dim_QK = dim_QK
        self.dim_V = dim_V
        self.qQ = nn.Linear(dim_q, head_num * dim_QK, bias=False)
        self.kK = nn.Linear(dim_k, head_num * dim_QK, bias=False)
        self.vV = nn.Linear(dim_v, head_num * dim_V, bias=False)
        self.linear_out = nn.Linear(head_num * dim_V, dim_out, bias=False)

        self.length_q = length_q
        self.length_kv = length_kv

        self.span = span
        self.stride = stride

        width = math.ceil(span/stride)
        self.width = width
        conv_K_weight = torch.zeros(width, 1, width)
        conv_K_weight[torch.arange(width),:,torch.arange(width)] = 1
        self.conv_K = nn.Conv1d(1, width, width)
        self.conv_K.weight.data = conv_K_weight
        self.conv_K.weight.requires_grad = False

        self.conv_V = self.conv_K

        mask_ind_row, mask_ind_col = torch.meshgrid(torch.arange(length_q), torch.arange(width), indexing='ij')
        original_col = mask_ind_col - (width - 1) + mask_ind_row // stride
        is_valid_col = torch.logical_and(0 <= original_col, original_col < length_kv)
        is_valid_row = mask_ind_row % stride <= (span - 1) - (width - mask_ind_col - 1) * stride
        is_valid = torch.logical_and(is_valid_col, is_valid_row)
        mask_trans_ind = torch.where(is_valid)
        table_QK_mask = torch.full((length_q, width), -float('inf'))
        table_QK_mask[mask_ind_row[mask_trans_ind],mask_ind_col[mask_trans_ind]] = 0
        self.table_QK_mask = nn.Parameter(table_QK_mask, requires_grad=False)

    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v):
        batch = q.shape[0]
        bh = batch * self.head_num

        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vV(v) # (batch, length_kv, head_num * dim_V)

        K = K.view(batch, self.length_kv, self.head_num, self.dim_QK).permute(0,2,3,1).reshape(bh * self.dim_QK, 1, self.length_kv)
        K = torch.nn.functional.pad(K,(self.width-1,0))
        table_K = self.conv_K(K) # (bh * dim_QK, width, length_kv)
        table_K = table_K.expand(self.stride, bh * self.dim_QK, self.width, self.length_kv).permute(1,2,3,0).reshape(bh * self.dim_QK, self.width, self.length_kv * self.stride)
        if self.length_kv * self.stride < self.length_q:
            table_K = table_K[:,:,:,0:self.length_q]
        table_K = table_K.reshape(bh, self.dim_QK, self.width, self.length_q, 1).permute(0,3,2,1,4) # (bh, length_q, width, dim_QK, 1)
        table_Q = Q.expand(self.width, batch, self.length_q, self.head_num * self.dim_QK)
        table_Q = table_Q.reshape(self.width, batch, self.length_q, self.head_num, self.dim_QK)
        table_Q = table_Q.permute(1,3,2,0,4).reshape(bh, self.length_q, self.width, 1, self.dim_QK) # (bh, length_q, width, 1, dim_QK)
        table_QK = torch.matmul(table_Q, table_K).reshape(bh, self.length_q, self.width) # (bh, length_q, width)
        table_QK = table_QK + self.table_QK_mask
        table_QK = table_QK.softmax(2)
        V = V.view(batch, self.length_kv, self.head_num, self.dim_V).permute(0,2,3,1).reshape(bh * self.dim_V, 1, self.length_kv)
        V = torch.nn.functional.pad(V,(self.width-1,0))
        table_V = self.conv_V(V) # (bh * dim_V, width, length_kv)
        table_V = table_V.expand(self.stride, bh * self.dim_V, self.width, self.length_kv).permute(1,2,3,0).reshape(bh * self.dim_V, self.width, self.length_kv * self.stride)
        if self.length_kv * self.stride < self.length_q:
            table_V = table_V[:,:,:,0:self.length_q]
        table_V = table_V.reshape(bh, self.dim_V, self.width, self.length_q).permute(0,3,2,1) # (bh, length_q, width, dim_V, 1)
        table_QKV = table_QK.unsqueeze(3) * table_V # (bh, length_q, width, dim_V)
        out = table_QKV.sum(2)
        out = out.reshape(batch, self.head_num, self.length_q, self.dim_V)
        out = out.transpose(2,1)
        out = out.reshape(batch, self.length_q, self.head_num*self.dim_V)
        out = self.linear_out(out)

        return out

class SparseCrossTransformerDecoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, dropout, span, stride, length_q, length_kv):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_cross = SparseMHADecoder(dim_q, dim_kv, dim_kv, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride, length_q, length_kv)
        self.mlp1 = nn.Linear(dim_q, dim_q*2)
        self.mlp2 = nn.Linear(dim_q*2, dim_q)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # key_value is assumed to be layer-normalized
    def forward(self, query, key_value):

        x = self.layer_norm(query)
        x = self.mmha_cross(x, key_value, key_value)
        x = self.dropout(x)
        query = x + query

        x = self.layer_norm(query)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        query = x + query

        return query

class SparseCrossTransformerDecoder(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, depth, dropout, span, stride, length_q, length_kv):
        super().__init__()
        decoder_block = SparseCrossTransformerDecoderBlock(dim_q, dim_kv, head_num, dropout, span, stride, length_q, length_kv)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value_normed)
        return query


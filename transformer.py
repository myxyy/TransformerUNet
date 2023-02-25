import copy
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_QK, dim_V, head_num, dim_out):
        super().__init__()
        self.head_num = head_num
        self.dim_QK = dim_QK
        self.dim_V = dim_V
        self.qQ = nn.Linear(dim_q, head_num * dim_QK, bias=False)
        self.kK = nn.Linear(dim_k, head_num * dim_QK, bias=False)
        self.vV = nn.Linear(dim_v, head_num * dim_V, bias=False)
        self.linear_out = nn.Linear(head_num * dim_V, dim_out, bias=False)
        self.softmax = nn.Softmax(dim=3)

    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v, mask=None):
        batch = q.shape[0]
        length_q = q.shape[1]
        length_kv = k.shape[1]

        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vV(v) # (batch, length_kv, head_num * dim_V)
        
        Q = Q.view(batch, length_q, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_q, dim_QK)
        K = K.view(batch, length_kv, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_kv, dim_QK)
        V = V.view(batch, length_kv, self.head_num, self.dim_V).permute(0,2,1,3) # (batch, head_num, length_kv, dim_V)

        # print(f"Q.shape:{Q.shape}")
        # print(f"K.shape:{K.shape}")
        # print(f"V.shape:{V.shape}")
        QK = torch.matmul(Q, K.transpose(3, 2)) / (self.dim_QK ** 0.5) # (batch, head_num, length_q, length_kv)
        if mask is not None:
            QK = mask(QK)

        softmax_QK = self.softmax(QK)
        QKV = torch.matmul(softmax_QK, V) # (batch, head_num, length_q, dim_V)
        QKV = QKV.permute(0,2,1,3).reshape(batch, length_q, self.head_num * self.dim_V)
        out = self.linear_out(QKV)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_self = MultiHeadAttention(dim_q, dim_q, dim_q, dim_q//head_num, dim_q//head_num, head_num, dim_q)
        self.mmha_cross = MultiHeadAttention(dim_q, dim_kv, dim_kv, dim_q//head_num, dim_q//head_num, head_num, dim_q)
        self.mlp1 = nn.Linear(dim_q, dim_q*2)
        self.mlp2 = nn.Linear(dim_q*2, dim_q)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # key_value is assumed to be layer-normalized
    def forward(self, query, key_value, mask_cross):

        x = self.layer_norm(query)
        x = self.mmha_cross(x, key_value, key_value, mask_cross)
        x = self.dropout(x)
        query = x + query

        x = self.layer_norm(query)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        query = x + query

        return query

class TransformerDecoder(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, depth, dropout):
        super().__init__()
        decoder_block = DecoderBlock(dim_q, dim_kv, head_num, dropout)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value, mask_cross):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value_normed, mask_cross)
        return query

class EncoderBlock(nn.Module):
    def __init__(self, dim_q, head_num, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_self = MultiHeadAttention(dim_q, dim_q, dim_q, dim_q//head_num, dim_q//head_num, head_num, dim_q)
        self.mlp1 = nn.Linear(dim_q, dim_q*2)
        self.mlp2 = nn.Linear(dim_q*2, dim_q)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    # key_value is assumed to be layer-normalized
    def forward(self, query, mask_self):

        x = self.layer_norm(query)
        x = self.mmha_self(x, x, x, mask_self)
        x = self.dropout(x)
        query = x + query

        x = self.layer_norm(query)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        query = x + query

        return query

class TransformerEncoder(nn.Module):
    def __init__(self, dim_q, head_num, depth, dropout):
        super().__init__()
        encoder_block = EncoderBlock(dim_q, head_num, dropout)
        self.encoder_block_list = nn.ModuleList([copy.deepcopy(encoder_block) for i in range(depth)])

    def forward(self, query, mask_self):
        for encoder_block in self.encoder_block_list:
            query = encoder_block(query, mask_self)
        return query

class SparseMHAEncoder(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_QK, dim_V, head_num, dim_out, span, stride):
        super().__init__()
        self.head_num = head_num
        self.dim_QK = dim_QK
        self.dim_V = dim_V
        self.qQ = nn.Linear(dim_q, head_num * dim_QK, bias=False)
        self.kK = nn.Linear(dim_k, head_num * dim_QK, bias=False)
        self.vV = nn.Linear(dim_v, head_num * dim_V, bias=False)
        self.linear_out = nn.Linear(head_num * dim_V, dim_out, bias=False)
        self.softmax = nn.Softmax(dim=3)
        self.span = span
        self.stride = stride

    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v):
        batch = q.shape[0]
        length_q = q.shape[1]
        length_kv = k.shape[1]

        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vV(v) # (batch, length_kv, head_num * dim_V)
        
        Q = Q.view(batch, length_q, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_q, dim_QK)
        K = K.view(batch, length_kv, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_kv, dim_QK)
        V = V.view(batch, length_kv, self.head_num, self.dim_V).permute(0,2,1,3) # (batch, head_num, length_kv, dim_V)

        # print(f"Q.shape:{Q.shape}")
        # print(f"K.shape:{K.shape}")
        # print(f"V.shape:{V.shape}")
        row, column = torch.meshgrid(torch.arange(self.span), torch.arange(length_q))
        kvi = row - (self.span-1) + torch.arange(0,self.stride*length_q,self.stride) # key-value index
        is_valid_kv = torch.logical_and(kvi >= 0, kvi < length_kv)
        vkvi_row, vkvi_column = torch.where(is_valid_kv) # valid key-value index
        ivkvi_row, ivkvi_column = torch.where(~is_valid_kv) # invalid key-value index
        QK_table = torch.zeros(batch, self.head_num, self.span, length_q).cuda() # (batch, head_num, span, length_q)
        QK_table[:,:,vkvi_row,vkvi_column] = torch.matmul(Q[:,:,column[vkvi_row,vkvi_column],:].unsqueeze(3), K[:,:,kvi[vkvi_row,vkvi_column],:].unsqueeze(4)).reshape(batch,self.head_num,len(vkvi_row))/(self.dim_QK ** 0.5)
        QK_table[:,:,ivkvi_row,ivkvi_column] = torch.tensor(-float('inf'))
        QKs_table = QK_table.softmax(3) # (batch, head_num, span, length_q)
        QKs_table_V = QKs_table.unsqueeze(4).repeat(1,1,1,1,self.dim_V) # (batch, head_num, span, length_q, dim_V)
        QKs_table_V[:,:,vkvi_row, vkvi_column,:] *= V[:,:,kvi[vkvi_row, vkvi_column],:]
        QKV_table = QKs_table_V
        QKV = QKV_table.sum(2).permute(0,2,1,3).reshape(batch, length_q, self.head_num * self.dim_V)
        out = self.linear_out(QKV)
        return out

class SparseSelfTransformerBlock(nn.Module):
    def __init__(self, dim_q, head_num, dropout, span, stride):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_self = SparseMHAEncoder(dim_q, dim_q, dim_q, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride)
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
    def __init__(self, dim_q, head_num, depth, dropout, span, stride):
        super().__init__()
        encoder_block = SparseSelfTransformerBlock(dim_q, head_num, dropout, span, stride)
        self.encoder_block_list = nn.ModuleList([copy.deepcopy(encoder_block) for i in range(depth)])

    def forward(self, query):
        for encoder_block in self.encoder_block_list:
            query = encoder_block(query)
        return query


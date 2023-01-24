import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_QK, dim_V, head_num, dim_out)
        super().__init__()
        self.head_num = head_num
        self.dim_QK = dim_QK
        self.dim_V = dim_V
        self.qQ = nn.Linear(dim_q, head_num * dim_QK, bias=False)
        self.kK = nn.Linear(dim_k, head_num * dim_QK, bias=False)
        self.vV = nn.Linear(dim_v, dim_V, bias=False)
        self.linear_out = nn.Linear(head_num * dim_V, dim_out)
        self.softmax = nn.Softmax(dim=3)

    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v, mask=None):
        batch = q.shape[0]
        length_q = q.shape[1]
        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vv(v) # (batch, length_kv, head_num * dim_V)
        
        Q = torch.stack(Q.tensor_split(self.head_num, dim=2), dim=1) # (batch, head_num, length_q, dim_QK)
        K = torch.stack(K.tensor_split(self.head_num, dim=2), dim=1) # (batch, head_num, length_kv, dim_QK)
        V = torch.stack(V.tensor_split(self.head_num, dim=2), dim=1) # (batch, head_num, length_kv, dim_V)

        QK = torch.matmul(Q, K.transpose(3, 2)) / (self.dim_QK ** 0.5) # (batch, head_num, length_q, length_kv)
        if mask is not None:
            QK = QK + mask

        softmax_QK = self.softmax(QK)
        QKV = torch.matmul(softmax_QK, V) # (batch, head_num, length_q, dim_V)
        QKV = QKV.permute(0,2,1,3).reshape(batch, length_q, self.head_num * dim_V)
        out = self.linear_out(QKV)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_self = MultiHeadAttention(dim_q, dim_q, dim_q, dim_q, dim_q, head_num, dim_q)
        self.mmha_cross = MultiHeadAttention(dim_q, dim_kv, dim_kv, dim_q, dim_q, head_num, dim_q)
        self.mlp1 = nn.Linear(dim_q, dim_q)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(dim_q, dim_q)

    # key_value is assumed to be layer-normalized
    def forward(self, query, key_value, mask_self, mask_cross):
        query = self.layer_norm(query)
        x = self.mmha_self(query, query, query, mask_self)
        query = x + query
        query = self.layer_norm(query)
        x = self.mmha_cross(query, key_value, key_value, mask_cross)
        query = x + query
        query = self.layer_norm(query)
        x = self.mlp1(query)
        x = self.act(x)
        x = self.mlp2(x)
        query = x + query
        return query

class TransformerDecoder(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, depth):
        super().__init__()
        decoder_block = DecoderBlock(dim_q, dim_kv, head_num)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value, mask_self, mask_cross):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value, mask_self, mask_cross)
        return query
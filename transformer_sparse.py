import copy
import torch
import torch.nn as nn

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
        row, column = torch.meshgrid(torch.arange(self.span), torch.arange(length_q), indexing='ij')
        kvi = row - (self.span-1) + torch.arange(0,self.stride*length_q,self.stride) # key-value index
        is_valid_kvi = torch.logical_and(kvi >= 0, kvi < length_kv)
        vkvi_row, vkvi_column = torch.where(is_valid_kvi) # valid key-value index
        QK_table = torch.full((batch, self.head_num, self.span, length_q), -float('inf')).cuda() # (batch, head_num, span, length_q)
        QK_table[:,:,vkvi_row,vkvi_column] = torch.matmul(Q[:,:,column[vkvi_row,vkvi_column],:].unsqueeze(3), K[:,:,kvi[vkvi_row,vkvi_column],:].unsqueeze(4)).reshape(batch,self.head_num,len(vkvi_row))/(self.dim_QK ** 0.5)
        QKs_table = QK_table.softmax(2) # (batch, head_num, span, length_q)
        QKs_table_V = QKs_table.unsqueeze(4).repeat(1,1,1,1,self.dim_V) # (batch, head_num, span, length_q, dim_V)
        QKs_table_V[:,:,vkvi_row, vkvi_column,:] *= V[:,:,kvi[vkvi_row, vkvi_column],:]
        QKV_table = QKs_table_V
        QKV = QKV_table.sum(2).permute(0,2,1,3).reshape(batch, length_q, self.head_num * self.dim_V)
        out = self.linear_out(QKV)
        #print(f'length_q:{length_q}, length_kv:{length_kv}, len(vkvi_row):{len(vkvi_row)}, len(ivkvi_row):{len(ivkvi_row)}')
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

class SparseCrossTransformerEncoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, dropout, span, stride):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_cross = SparseMHAEncoder(dim_q, dim_kv, dim_kv, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride)
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
    def __init__(self, dim_q, dim_kv, head_num, depth, dropout, span, stride):
        super().__init__()
        decoder_block = SparseCrossTransformerEncoderBlock(dim_q, dim_kv, head_num, dropout, span, stride)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value_normed)
        return query

class SparseMHADecoder(nn.Module):
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
        row, column = torch.meshgrid(torch.arange(self.span // self.stride), torch.arange(length_q), indexing='ij')
        qi_column = torch.ceil((column - (self.span-1))/self.stride).to(int) + row
        qi_row = -self.stride * qi_column + column
        is_valid_qi = torch.logical_and(torch.logical_and(qi_row >= 0, qi_row < self.span),torch.logical_and(qi_column >= 0, qi_column < length_kv))
        vqi_row, vqi_column = torch.where(is_valid_qi)
        QK_table = torch.full((batch, self.head_num, self.span // self.stride, length_q), -float('inf')).cuda()
        QK_table[:,:,vqi_row,vqi_column] = torch.matmul(Q[:,:,column[vqi_row,vqi_column],:].unsqueeze(3),K[:,:,qi_column[vqi_row,vqi_column],:].unsqueeze(4)).reshape(batch,self.head_num,len(vqi_row))/(self.dim_QK ** 0.5)
        QKs_table = QK_table.softmax(2) # (batch, head_num, span//stride, length_q)
        QKs_table_V = QKs_table.unsqueeze(4).repeat(1,1,1,1,self.dim_V) # (batch, head_num, span//stride, length_q, dim_V)
        QKs_table_V[:,:,vqi_row, vqi_column,:] *= V[:,:,qi_column[vqi_row, vqi_column],:]
        QKV_table = QKs_table_V
        QKV = QKV_table.sum(2).permute(0,2,1,3).reshape(batch, length_q, self.head_num * self.dim_V)
        out = self.linear_out(QKV)
        #print(f'length_q:{length_q}, length_kv:{length_kv}, len(vkvi_row):{len(vkvi_row)}, len(ivkvi_row):{len(ivkvi_row)}')
        return out

class SparseCrossTransformerDecoderBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, head_num, dropout, span, stride):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_q)
        self.mmha_cross = SparseMHADecoder(dim_q, dim_kv, dim_kv, dim_q//head_num, dim_q//head_num, head_num, dim_q, span, stride)
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
    def __init__(self, dim_q, dim_kv, head_num, depth, dropout, span, stride):
        super().__init__()
        decoder_block = SparseCrossTransformerDecoderBlock(dim_q, dim_kv, head_num, dropout, span, stride)
        self.decoder_block_list = nn.ModuleList([copy.deepcopy(decoder_block) for i in range(depth)])
        self.layer_norm = nn.LayerNorm(dim_kv)

    def forward(self, query, key_value):
        key_value_normed = self.layer_norm(key_value)
        for decoder_block in self.decoder_block_list:
            query = decoder_block(query, key_value_normed)
        return query

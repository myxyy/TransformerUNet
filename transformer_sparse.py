import copy
import torch
import torch.nn as nn

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

        row, column = torch.meshgrid(torch.arange(span), torch.arange(length_q), indexing='ij')
        kvi = row - (span-1) + torch.arange(0,stride*length_q,stride) # key-value index
        is_valid_kvi = torch.logical_and(kvi >= 0, kvi < length_kv)
        ivkvi_row, ivkvi_column = torch.where(~is_valid_kvi) # invalid key-value index

        conv_K_weight_ind_key, conv_K_weight_ind_span = torch.meshgrid(torch.arange(dim_QK), torch.arange(span), indexing='ij')
        conv_K_weight = torch.zeros(dim_QK, span, dim_QK, span)
        conv_K_weight[conv_K_weight_ind_key, conv_K_weight_ind_span, conv_K_weight_ind_key, conv_K_weight_ind_span] = 1
        conv_K_weight = conv_K_weight.reshape(dim_QK*span, dim_QK, span)
        self.conv_K = nn.Conv1d(dim_QK, dim_QK*span, span, stride, padding=span-1)
        self.conv_K.weight.data = conv_K_weight
        self.conv_K.weight.requires_grad = False
        self.table_QK_mask = torch.zeros(span, length_q).cuda()
        self.table_QK_mask[ivkvi_row, ivkvi_column] = -float('inf')
        conv_V_weight_ind_key, conv_V_weight_ind_span = torch.meshgrid(torch.arange(dim_QK), torch.arange(span), indexing='ij')
        conv_V_weight = torch.zeros(dim_V, span, dim_V, span)
        conv_V_weight[conv_V_weight_ind_key, conv_V_weight_ind_span, conv_V_weight_ind_key, conv_V_weight_ind_span] = 1
        conv_V_weight = conv_V_weight.reshape(dim_V*span, dim_V, span)
        self.conv_V = nn.Conv1d(dim_V, dim_V*span, span, stride, padding=span-1)
        self.conv_V.weight.data = conv_V_weight
        self.conv_V.weight.requires_grad = False
 
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
        Q = Q.view(batch, self.length_q, self.head_num, self.dim_QK).permute(0,2,3,1).reshape(bh, self.dim_QK, self.length_q)
        K = K.view(batch, self.length_kv, self.head_num, self.dim_QK).permute(0,2,3,1).reshape(bh, self.dim_QK, self.length_kv)
        V = V.view(batch, self.length_kv, self.head_num, self.dim_V).permute(0,2,3,1).reshape(bh, self.dim_V, self.length_kv)

        table_K = self.conv_K(K) # (bh, dim_QK*span, ?)
        table_K = table_K[:,:,0:self.length_q]
        table_K = table_K.reshape(bh, self.dim_QK, self.span, self.length_q).permute(0,2,3,1) # (bh, span, length_q, dim_QK)
        Q = Q.expand(self.span,bh,self.dim_QK,self.length_q) # (span, bh, dim_QK, length_q)
        Q = Q.permute(1,0,3,2).unsqueeze(3)
        table_K = table_K.unsqueeze(4)
        table_QK = torch.matmul(Q, table_K)/(self.dim_QK ** 0.5)
        table_QK = table_QK.reshape(bh, self.span, self.length_q)
        table_QK = table_QK + self.table_QK_mask
        table_QK = table_QK.softmax(1)
        table_V = self.conv_V(V)[:,:,0:self.length_q].reshape(bh, self.dim_V, self.span, self.length_q).permute(0,2,3,1) # (bh, span, length_q, dim_V)
        table_QKV = table_QK.unsqueeze(3) * table_V # (bh, span, length_q, dim_V)
        out = self.linear_out(table_QKV.sum(1).reshape(batch, self.head_num, self.length_q, self.dim_V).permute(0,2,1,3).reshape(batch, self.length_q, self.head_num*self.dim_V))

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

        self.mask = torch.full((length_q, length_kv), -float('inf')).cuda()
        mask_ind_row, mask_ind_col = torch.meshgrid(torch.arange(length_q), torch.arange(length_kv), indexing='ij')
        mask_ind_span = mask_ind_row - mask_ind_col*stride
        mask_trans_ind = torch.logical_and(0 <= mask_ind_span, mask_ind_span < span)
        self.mask[mask_ind_row[mask_trans_ind],mask_ind_col[mask_trans_ind]] = 0


    # (batch, length_q, dim_q)
    # (batch, length_kv, dim_k)
    # (batch, length_kv, dim_v)
    # -> (batch, length_q, dim_out)
    def forward(self, q, k, v):
        batch = q.shape[0]

        Q = self.qQ(q) # (batch, length_q, head_num * dim_QK)
        K = self.kK(k) # (batch, length_kv, head_num * dim_QK)
        V = self.vV(v) # (batch, length_kv, head_num * dim_V)
        
        Q = Q.view(batch, self.length_q, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_q, dim_QK)
        K = K.view(batch, self.length_kv, self.head_num, self.dim_QK).permute(0,2,1,3) # (batch, head_num, length_kv, dim_QK)
        V = V.view(batch, self.length_kv, self.head_num, self.dim_V).permute(0,2,1,3) # (batch, head_num, length_kv, dim_V)

        QK = torch.matmul(Q, K.transpose(3,2))
        QK = QK + self.mask
        QK_softmax = QK.softmax(2)
        QKV = torch.matmul(QK_softmax, V)
        QKV = QKV.permute(0,2,1,3).reshape(batch, self.length_q, self.head_num * self.dim_V)
        out = self.linear_out(QKV)
        #print(f'length_q:{length_q}, length_kv:{length_kv}, len(vkvi_row):{len(vkvi_row)}, len(ivkvi_row):{len(ivkvi_row)}')
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


from transformer import TransformerDecoder, TransformerEncoder
from positional_encoding import PositionalEncoding
from mask import SelfMask, CrossEncodeMask, CrossDecodeMask

import torch
import torch.nn as nn

class TransformerUNetSequence(nn.Module):
    class ResMLP(nn.Module):
        def __init__(self, dim, dropout):
            super().__init__()
            self.layer_norm = nn.LayerNorm(dim)
            self.mlp1 = nn.Linear(dim, dim*2)
            self.act = nn.GELU()
            self.mlp2 = nn.Linear(dim*2, dim)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            res = x
            x = self.layer_norm(x)
            x = self.mlp1(x)
            x = self.act(x)
            x = self.mlp2(x)
            x = self.dropout(x)
            return x + res
    
    def __init__(self, length_log_2: int, depth_unet: int, depth_transformer: int, dim: int, dim_scale: float, head_num: int, dropout: int):
        super().__init__()
        if length_log_2 < depth_unet:
            raise ValueError("TUNError")
        self.length_log_2 = length_log_2
        self.depth_unet = depth_unet
        self.encoder_list = nn.ModuleList([TransformerDecoder(dim*(int)(dim_scale**(i+1)), dim*(int)(dim_scale**i), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.decoder_list = nn.ModuleList([TransformerDecoder(dim*(int)(dim_scale**i), dim*(int)(dim_scale**(i+1)), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.self_encoder_pre_list = nn.ModuleList([TransformerEncoder(dim*(int)(dim_scale**i), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.self_encoder_middle_list = nn.ModuleList([TransformerEncoder(dim*(int)(dim_scale**i), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.self_encoder_post_list = nn.ModuleList([TransformerEncoder(dim*(int)(dim_scale**i), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.self_mask_list = nn.ModuleList([SelfMask(2**(length_log_2-i)) for i in range(depth_unet)])
        self.encoder_cross_mask_list = nn.ModuleList([CrossEncodeMask(2**(length_log_2-i),2**(length_log_2-(i+1))) for i in range(depth_unet)])
        self.decoder_cross_mask_list = nn.ModuleList([CrossDecodeMask(2**(length_log_2-(i+1)),2**(length_log_2-i)) for i in range(depth_unet)])
        self.positional_encoding_init = PositionalEncoding(2 ** length_log_2, dim)
        self.positional_encoding_list = nn.ModuleList([PositionalEncoding(2**(length_log_2-i-1), dim*(int)(dim_scale**(i+1))) for i in range(depth_unet)])
        dim_last = dim*(int)(dim_scale**depth_unet)
        self.ff_last_stacked = nn.Sequential(*[self.ResMLP(dim_last, dropout) for i in range(depth_transformer*3)])


    # (batch, 2**length_log_2, dim) -> (batch, 2**length_log_2, dim)
    def unet_rec(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        if depth < self.depth_unet:
            batch = x.shape[0]
            encoder = self.encoder_list[depth]
            decoder = self.decoder_list[depth]
            self_encoder_pre = self.self_encoder_pre_list[depth]
            self_encoder_middle = self.self_encoder_middle_list[depth]
            self_encoder_post = self.self_encoder_post_list[depth]
            self_mask = self.self_mask_list[depth]
            encoder_cross_mask = self.encoder_cross_mask_list[depth]
            decoder_cross_mask = self.decoder_cross_mask_list[depth]
            positional_encoding = self.positional_encoding_list[depth]().repeat(batch, 1, 1)

            x = self_encoder_pre(x, self_mask)
            y = encoder(positional_encoding, x, encoder_cross_mask)
            y = self.unet_rec(y, depth + 1)
            x = self_encoder_middle(x, self_mask)
            x = decoder(x, y, decoder_cross_mask)
            x = self_encoder_post(x, self_mask)
        else:
            x = self.ff_last_stacked(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_rec(self.positional_encoding_init(x), 0)
       
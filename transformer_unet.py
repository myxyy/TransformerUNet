from transformer import TransformerDecoder, TransformerEncoder
from positional_encoding import PositionalEncoding
from mask import SelfMask, CrossEncodeMask, CrossDecodeMask

import torch
import torch.nn as nn
import math

class TransformerUNetSequence(nn.Module):
    def __init__(self, length: int, downsample_rate: float, depth_unet: int, depth_transformer: int, dim: int, dim_scale: float, head_num: int, dropout: int, enable_pre=True, enable_middle=True, enable_post=True):
        super().__init__()
        self.enable_pre = enable_pre
        self.enable_middle = enable_middle
        self.enable_post = enable_post
        depth_unet = min(depth_unet, -(int)(math.log(length, downsample_rate)))
        self.depth_unet = depth_unet
        self.dim = dim
        self.dim_scale = dim_scale
        self.encoder_list = nn.ModuleList([TransformerDecoder(self.level_i_dim(i+1), self.level_i_dim(i), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        self.decoder_list = nn.ModuleList([TransformerDecoder(self.level_i_dim(i), self.level_i_dim(i+1), head_num, depth_transformer, dropout) for i in range(depth_unet)])
        if enable_pre:
            self.self_encoder_pre_list = nn.ModuleList([TransformerEncoder(self.level_i_dim(i), head_num, depth_transformer, dropout) for i in range(depth_unet+1)])
        if enable_middle:
            self.self_encoder_middle_list = nn.ModuleList([TransformerEncoder(self.level_i_dim(i), head_num, depth_transformer, dropout) for i in range(depth_unet+1)])
        if enable_post:
            self.self_encoder_post_list = nn.ModuleList([TransformerEncoder(self.level_i_dim(i), head_num, depth_transformer, dropout) for i in range(depth_unet+1)])
        self.self_mask_list = nn.ModuleList([SelfMask((int)(length*(downsample_rate**i))) for i in range(depth_unet+1)])
        self.encoder_cross_mask_list = nn.ModuleList([CrossEncodeMask((int)(length*(downsample_rate**i)),(int)(length*(downsample_rate**(i+1)))) for i in range(depth_unet)])
        self.decoder_cross_mask_list = nn.ModuleList([CrossDecodeMask((int)(length*(downsample_rate**(i+1))),(int)(length*(downsample_rate**i))) for i in range(depth_unet)])
        self.positional_encoding_list = nn.ModuleList([PositionalEncoding((int)(length*(downsample_rate**i)), self.level_i_dim(i)) for i in range(depth_unet+1)])

    def level_i_dim(self, i):
        return (int)(math.ceil(self.dim*(self.dim_scale**i)/2)*2)

    # (batch, length, dim) -> (batch, length, dim)
    def unet_rec(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        self_mask = self.self_mask_list[depth]
        if depth < self.depth_unet:
            batch = x.shape[0]
            encoder = self.encoder_list[depth]
            decoder = self.decoder_list[depth]
            encoder_cross_mask = self.encoder_cross_mask_list[depth]
            decoder_cross_mask = self.decoder_cross_mask_list[depth]
            positional_encoding = self.positional_encoding_list[depth+1]().repeat(batch, 1, 1)

        if self.enable_pre:
            self_encoder_pre = self.self_encoder_pre_list[depth]
            x = self_encoder_pre(x, self_mask)

        if depth < self.depth_unet:
            y = encoder(positional_encoding, x, encoder_cross_mask)
            y = self.unet_rec(y, depth + 1)

        if self.enable_middle:
            self_encoder_middle = self.self_encoder_middle_list[depth]
            x = self_encoder_middle(x, self_mask)

        if depth < self.depth_unet:
            x = decoder(x, y, decoder_cross_mask)

        if self.enable_post:
            self_encoder_post = self.self_encoder_post_list[depth]
            x = self_encoder_post(x, self_mask)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_rec(self.positional_encoding_list[0](x), 0)
       
from transformer import TransformerDecoder
from positional_encoding import PositionalEncoding

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class TransformerUNetSequence(nn.Module):
    def __init__(self, length_log_2: int, depth_unet: int, depth_transformer: int, dim: int, head_num: int):
        super().__init__()
        if length_log_2 < depth_unet:
            raise ValueError("TUNError")
        self.length_log_2 = length_log_2
        self.depth_unet = depth_unet
        self.encoder_list = nn.ModuleList([TransformerDecoder(dim*(2**(i+1)), dim*(2**i), head_num, depth_transformer) for i in range(depth_unet)])
        self.decoder_list = nn.ModuleList([TransformerDecoder(dim*(2**i), dim*(2**(i+1)), head_num, depth_transformer) for i in range(depth_unet)])
        self.self_mask_list = [torch.full((i,i), float('-inf')).triu()  for i in (2**(length_log_2-torch.arange(depth_unet+1))).tolist()]
        self.encoder_cross_mask_list = [nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).tril(-1).view(1,1,i,i)).view(i*2,i).transpose(1,0) for i in (2**(length_log_2-torch.arange(depth_unet)-1)).tolist()]
        self.decoder_cross_mask_list = [nn.Upsample((i*2,i))(torch.full((i,i), float('-inf')).triu().view(1,1,i,i)).view(i*2,i) for i in (2**(length_log_2-torch.arange(depth_unet)-1)).tolist()]
        self.positional_encoding_init = PositionalEncoding(2 ** length_log_2, dim)
        self.positional_encoding_list = [PositionalEncoding(2**(length_log_2-i-1), dim*(2**(i+1))) for i in range(depth_unet)]
        self.apply(self._init_weights)

    # (batch, 2**length_log_2, dim) -> (batch, 2**length_log_2, dim)
    def unet_rec(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        if depth >= self.depth_unet:
            return x
        encoder = self.encoder_list[depth]
        decoder = self.decoder_list[depth]
        encoder_self_mask = self.self_mask_list[depth+1]
        decoder_self_mask = self.self_mask_list[depth]
        encoder_cross_mask = self.encoder_cross_mask_list[depth]
        decoder_cross_mask = self.decoder_cross_mask_list[depth]
        positional_encoding = self.positional_encoding_list[depth](torch.zeros(x.shape[0], x.shape[1]//2, x.shape[2]*2))
        y = encoder(positional_encoding, x, encoder_self_mask, encoder_cross_mask)
        y = self.unet_rec(y, depth + 1)
        x = decoder(x, y, decoder_self_mask, decoder_cross_mask)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_rec(self.positional_encoding_init(x), 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


        
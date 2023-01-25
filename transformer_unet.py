from transformer import TransformerDecoder
from positional_encoding import PositionalEncoding
from mask import SelfMask, CrossEncodeMask, CrossDecodeMask

import torch
import torch.nn as nn

class TransformerUNetSequence(nn.Module):
    def __init__(self, length_log_2: int, depth_unet: int, depth_transformer: int, dim: int, head_num: int):
        super().__init__()
        if length_log_2 < depth_unet:
            raise ValueError("TUNError")
        self.length_log_2 = length_log_2
        self.depth_unet = depth_unet
        self.encoder_list = nn.ModuleList([TransformerDecoder(dim*(2**(i+1)), dim*(2**i), head_num, depth_transformer) for i in range(depth_unet)])
        self.decoder_list = nn.ModuleList([TransformerDecoder(dim*(2**i), dim*(2**(i+1)), head_num, depth_transformer) for i in range(depth_unet)])
        self.self_mask_list = nn.ModuleList([SelfMask(i) for i in (2**(length_log_2-torch.arange(depth_unet+1))).tolist()])
        self.encoder_cross_mask_list = nn.ModuleList([CrossEncodeMask(i) for i in (2**(length_log_2-torch.arange(depth_unet)-1)).tolist()])
        self.decoder_cross_mask_list = nn.ModuleList([CrossDecodeMask(i) for i in (2**(length_log_2-torch.arange(depth_unet)-1)).tolist()])
        self.positional_encoding_init = PositionalEncoding(2 ** length_log_2, dim)
        self.positional_encoding_list = nn.ModuleList([PositionalEncoding(2**(length_log_2-i-1), dim*(2**(i+1))) for i in range(depth_unet)])

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
        positional_encoding = self.positional_encoding_list[depth](torch.zeros(x.shape[0], x.shape[1]//2, x.shape[2]*2).cuda())

        y = encoder(positional_encoding, x, encoder_self_mask, encoder_cross_mask)
        y = self.unet_rec(y, depth + 1)
        x = decoder(x, y, decoder_self_mask, decoder_cross_mask)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet_rec(self.positional_encoding_init(x), 0)
       
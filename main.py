from transformer_unet import TransformerUNetSequence
from transformer_unet import SparseTransformerUNetSequence

import pytorch_lightning as pl
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
import torch
from torchmetrics import MeanMetric
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

class GPTUNet(pl.LightningModule):
    logger: TensorBoardLogger
    def __init__(self, length, downsample_rate, depth_unet, depth_transformer=1, dim_scale=1, head_num=8, dropout=0.1, vocab_size=256, dim=512, enable_pre=True, enable_middle=True, enable_post=True):
        super().__init__()
        self.length = length
        self.vocab_size = vocab_size
        self.transformer_u_net = TransformerUNetSequence(length, downsample_rate, depth_unet, depth_transformer, dim, dim_scale, head_num, dropout, enable_pre=enable_pre, enable_middle=enable_middle, enable_post=enable_post)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        data, next = batch
        x = nn.functional.one_hot(data, self.vocab_size).float()
        #x_next = nn.functional.one_hot(next, 256).float()
        x_next = next
        x_hat = self.token_out(self.transformer_u_net(self.token_in(x)))
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1))
        self.log("train_loss", loss, on_epoch=False)
        return loss

    def forward(self, x):
        x = nn.functional.one_hot(x, self.vocab_size).float()
        x_hat = self.token_out(self.transformer_u_net(self.token_in(x)))
        x_hat = x_hat.softmax(2)
        return x_hat

    def training_epoch_end(self, outputs):
        torch.save(self.state_dict(), 'weight.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

class SparseGPTUNet(pl.LightningModule):
    logger: TensorBoardLogger
    def __init__(self, length, downsample_rate, depth_unet, depth_transformer=1, dim_scale=1, head_num=8, dropout=0.1, vocab_size=256, dim=512, enable_pre=True, enable_middle=True, enable_post=True):
        super().__init__()
        self.length = length
        self.vocab_size = vocab_size
        self.transformer_u_net = SparseTransformerUNetSequence(length, downsample_rate, depth_unet, depth_transformer, dim, dim_scale, head_num, dropout, enable_pre=enable_pre, enable_middle=enable_middle, enable_post=enable_post)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        data, next = batch
        x = nn.functional.one_hot(data, self.vocab_size).float()
        #x_next = nn.functional.one_hot(next, 256).float()
        x_next = next
        x_hat = self.token_out(self.transformer_u_net(self.token_in(x)))
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1))
        self.log("train_loss", loss, on_epoch=False)
        return loss

    def forward(self, x):
        x = nn.functional.one_hot(x, self.vocab_size).float()
        x_hat = self.token_out(self.transformer_u_net(self.token_in(x)))
        x_hat = x_hat.softmax(2)
        return x_hat

    def training_epoch_end(self, outputs):
        torch.save(self.state_dict(), 'weight.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

model = SparseGPTUNet(
    length=1024,
    downsample_rate=0.5,
    depth_unet=10,
    depth_transformer=2,
    dim_scale=1.2,
    dim=256,
    dropout=0.2,
    enable_pre=False,
    enable_middle=True,
    enable_post=False
)
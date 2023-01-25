from transformer_unet import TransformerUNetSequence

import pytorch_lightning as pl
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
import torch
from torchmetrics import MeanMetric
import torch.nn as nn
from text_loader import TextDataset
from pytorch_lightning.loggers import TensorBoardLogger

class TransformerUNetGPT(pl.LightningModule):
    logger: TensorBoardLogger
    def __init__(self, length_log_2=12, depth_unet=4, depth_transformer=1, head_num=8):
        super().__init__()
        self.transformer_u_net = TransformerUNetSequence(length_log_2, depth_unet, depth_transformer, 256, head_num)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self._init_weights)
        self.train_loss_epoch = MeanMetric()
        self.validate_loss_epoch = MeanMetric()
        self.train_loss_epoch.reset()
        self.validate_loss_epoch.reset()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        x = nn.functional.one_hot(batch, 256)
        x_hat = self.transformer_u_net(x)
        loss = self.criterion(x_hat, x)
        self.train_loss_epoch.update(loss)
        return loss

    def training_epoch_end(self, outputs):
        self.train_loss_epoch.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

if __name__ == '__main__':
    transforms = transforms.Compose([])
    dataset = TextDataset('natsume.txt', 2**12, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = TransformerUNetGPT()
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=500)
    trainer.fit(model, dataloader)



 
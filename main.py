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
    def __init__(self, length_log_2, depth_unet=3, depth_transformer=1, dim_scale=1, head_num=8):
        super().__init__()
        self.transformer_u_net = TransformerUNetSequence(length_log_2, depth_unet, depth_transformer, 256, dim_scale, head_num)
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
        data, next = batch
        x = nn.functional.one_hot(data, 256).float()
        #x_next = nn.functional.one_hot(next, 256).float()
        x_next = next
        x_hat = self.transformer_u_net(x)
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,256), x_next.view(-1))
        self.train_loss_epoch.update(loss)
        return loss

    def training_epoch_end(self, outputs):
        self.train_loss_epoch.reset()
        torch.save(self.state_dict(), 'weight.pth')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

if __name__ == '__main__':
    transforms = transforms.Compose([])
    length_log_2 = 10
    dataset = TextDataset('natsume.txt', 2**length_log_2, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    model = TransformerUNetGPT(length_log_2=length_log_2, depth_unet=length_log_2-1, depth_transformer=1)
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=500)
    trainer.fit(model, dataloader)



 
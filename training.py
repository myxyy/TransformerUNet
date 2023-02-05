from main import GPTUNet, model
import torchvision.transforms as transforms
from text_loader import TextDataset
import torch
import pytorch_lightning as pl
import os

if __name__ == '__main__':
    transforms = transforms.Compose([])
    length_log_2 = 8
    dataset = TextDataset('data.txt', 2**length_log_2, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    #model = GPTUNet(length_log_2=length_log_2, depth_unet=6, depth_transformer=6, dim_scale=1.1)
    if os.path.isfile('weight.pth'):
        model.load_state_dict(torch.load('weight.pth'))
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=500)
    trainer.fit(model, dataloader)
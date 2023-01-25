import torch
import torch.nn as nn
import numpy
from torch.utils.data import Dataset
from typing import Tuple

class TextDataset(Dataset):
    def __init__(self, path: str, size: int, transforms) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms
        self.text = numpy.array([i for i in open(path, 'r', encoding='utf-8').read().encode(encoding='utf-8')])

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.text[index:index+self.size]
        data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.text) - self.size + 1
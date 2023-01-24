import torch
from torch.utils.data import Dataset
from typing import Tuple

class TextDataset(Dataset):
    def __init__(self, path: str, size: int, transforms) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms
        self.text = open(path, 'r', encoding='utf-8').read().encode(encoding='utf-8')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.text[index * self.size:(index+1) * self.size]
        next = self.text[index * self.size + 1:(index+1) * self.size + 1]
        return data, next

    def __len__(self) -> int:
        return (len(self.text) - 1) // self.size
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
        return self.text[index:self.size]

    def __len__(self) -> int:
        return len(self.text) - self.size + 1
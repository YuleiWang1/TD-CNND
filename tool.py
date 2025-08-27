import torch
from torch import nn
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class MyDataset(Dataset):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        return x1, x2

class Pixel(Dataset):
    def __init__(self, x1):
        self.x1 = x1

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        return x1

    def __len__(self):
        return len(self.x1)


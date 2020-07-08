# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:19:21 2020

@author: USER
"""
from torch.utils.data import Dataset
class ImageDataset(Dataset):
    def __init__(self, X):
        self.data = X
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)


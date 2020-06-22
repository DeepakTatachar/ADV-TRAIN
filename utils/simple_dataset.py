"""
@author: Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class SimpleOnehot(Dataset):
    def __init__(self, train=True, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.transform = transform
        self.train = train

    def __len__(self):
        if(self.train):
            return 200000
        else:
            return 10000

    def __getitem__(self, idx):
        image = torch.zeros((1,32,32))
        index_x = np.random.randint(0, 32)
        index_y = np.random.randint(0, 32)
        image[0][index_x][index_y] = 1

        if(index_x < 16 and index_y < 16):
            label = 0
        elif(index_x > 16 and index_y < 16):
            label = 1
        elif(index_x < 16 and index_y > 16):
            label = 2
        else:
            label = 3

        return image, label
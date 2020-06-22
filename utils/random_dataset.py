"""
@author: Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class RandomDataset(Dataset):
    def __init__(self, train=True, transform=None, num_classes=10):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.transform = transform
        self.train = train
        self.num_classes = num_classes

        # Generate a sequence of seeds which are initalized before the call to random
        # To make sure the sequence is the same
        self.seed_seq = np.array(range(self.__len__()))

        if(self.train):
            np.random.seed(0)
        else:
            np.random.seed(50)

        np.random.shuffle(self.seed_seq)

    def __len__(self):
        if(self.train):
            return 200000
        else:
            return 10000

    def __getitem__(self, idx):
        seed = self.seed_seq[idx]
        np.random.seed(seed)
        img = np.random.rand(32*32)
        img = img.reshape(1, 32, 32)
        label = np.random.randint(low=0, high=self.num_classes)

        # Convert to torch Tensors
        image = torch.Tensor(img)
        return image, label
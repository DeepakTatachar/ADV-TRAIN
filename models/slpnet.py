"""
@author: Deepak Ravikumar Tatachar
@copyright: Nanoelectronics Research Laboratory
"""

import torch.nn as nn
from utils.custom_Conv2D import Conv2D

class SLPConvNet(nn.Module):
    def __init__(self, num_classes=4, return_act = False, down=None, right=None):
        self.num_classes=num_classes
        self.return_act = return_act
        super(SLPConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2)
        self.hidden = nn.Linear(3072, 512)
        self.classifier = nn.Linear(512, self.num_classes)
        self.down = down
        self.right =right
        if down==None or right==None:
            self.custom_conv = True
        else:
            self.custom_conv = False
        
    def forward(self, x):
        if self.custom_conv:
            x = nn.functional.relu(self.conv(x))                
        else:
            x = nn.functional.relu( Conv2D(x, weight=self.conv.weight, bias=self.conv.bias, stride=1, padding=2, down=down, right=right) )
        x = x.flatten(1)
        x = nn.functional.relu(self.hidden(x))
        x = self.classifier(x)
        return x

class SLPNet(nn.Module):
    def __init__(self, num_classes=4, return_act = False):
        self.num_classes=num_classes
        self.return_act = return_act
        super(SLPNet, self).__init__()
        self.inp = nn.Linear(1024, 3072)
        self.hidden = nn.Linear(3072, 512)
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = nn.functional.relu(self.inp(x))
        x = nn.functional.relu(self.hidden(x))
        x = self.classifier(x)
        return x

"""
@author: Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""

import torch.nn as nn
from advtrain.utils.quant_dorefa import *

class LeNet5(nn.Module):

    def __init__(self,input_size=1, num_classes=10, return_act = False):
        self.num_classes=num_classes
        self.return_act = return_act

        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 6, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features = 6)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 16)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.linear3 = nn.Linear(5*5*16 ,120)
        self.batchnorm3 = nn.BatchNorm1d(num_features = 120)
        self.linear4 = nn.Linear(120, 84)
        self.batchnorm4 = nn.BatchNorm1d(num_features = 84)
        self.classifier = nn.Linear(84,self.num_classes)
        

    def forward(self, x):
        x = self.avg_pool1(nn.functional.relu(self.batchnorm1(self.conv1(x))))
        x = self.avg_pool2(nn.functional.relu(self.batchnorm2(self.conv2(x))))
        x = x.flatten(1)
        if self.return_act == True:
            return x
        x = nn.functional.relu(self.batchnorm3(self.linear3(x)))
        x = nn.functional.relu(self.batchnorm4(self.linear4(x)))
        x = self.classifier(x)
        return x

class LeNet5_Dorefa(nn.Module):

    def __init__(self,input_size=1, num_classes=10, abit=32, wbit=32):
        self.abit = abit
        self.wbit = wbit 
        self.num_classes=num_classes
        super(LeNet5_Dorefa, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=self.wbit)
        Linear = linear_Q_fn(w_bit=self.wbit)
        self.feature_quant=activation_quantize_fn(a_bit=self.abit)
        

        self.conv1 = Conv2d(input_size, 6, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features = 6)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 16)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.linear3 = Linear(5*5*16 ,120)
        self.batchnorm3 = nn.BatchNorm1d(num_features = 120)
        self.linear4 = Linear(120, 84)
        self.batchnorm4 = nn.BatchNorm1d(num_features = 84)
        self.classifier = Linear(84,self.num_classes)
        

    def forward(self, x):
        x = self.feature_quant(self.avg_pool1(nn.functional.relu(self.batchnorm1(self.conv1(x)))))
        x = self.feature_quant(self.avg_pool2(nn.functional.relu(self.batchnorm2(self.conv2(x)))))
        x = x.flatten(1)
        x = self.feature_quant(nn.functional.relu(self.batchnorm3(self.linear3(x))))
        x = self.feature_quant(nn.functional.relu(self.batchnorm4(self.linear4(x))))
        x = self.classifier(x)
        return x
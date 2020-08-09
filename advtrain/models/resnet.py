"""
@author: Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from advtrain.utils.quant_dorefa import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, blk_stride=[1,2,2,2], num_classes=10, in_channels=3, num_features=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.type = 'single'
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=blk_stride[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=blk_stride[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=blk_stride[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=blk_stride[3])
        if num_features !=0:
            self.linear = nn.Linear(num_features, num_classes)
        elif(num_classes < 200):
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(2048 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
      
        out = self.linear(out)
        return out

def ResNet_(cfg='18', num_classes=10, in_channels=3):
    if cfg=='':
        cfg='18'
    num_blk= {
        '18' : [2,2,2,2],
        '34' : [3,4,6,3],
        '50' : [3,4,6,3],
        '101' : [3,4,23,3],
        '152' : [3,8,36,3],
    }
    blk_type = {
        '18'    : BasicBlock,
        '34'    : BasicBlock,
        '50'    : Bottleneck,
        '101'   : Bottleneck,
        '152'   : Bottleneck,
    }
    stride = {
        '18' : [1,2,2,2],
        '34' : [1,2,2,2],
        '50' : [1,2,2,2],
        '101' : [1,2,2,2],
        '152' : [1,2,2,2],
    }
    num_features = {
        '18' : 0,
        '34' : 0,
        '50' : 0,
        '101' : 0,
        '152' : 0,
    }
    return ResNet(blk_type[cfg], num_blk[cfg], num_classes=num_classes, in_channels=in_channels, blk_stride= stride[cfg], num_features= num_features[cfg]  )


#################################################################################################################################################################################

class BasicBlock_Dorefa(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1,w_bit=32,a_bit=32):
        self.wbit=w_bit
        self.abit=a_bit
        super(BasicBlock_Dorefa, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=self.wbit)
        Linear = linear_Q_fn(w_bit=self.wbit)
        self.feature_quant=activation_quantize_fn(a_bit=self.abit)
        
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.feature_quant(x)
        out = self.feature_quant(F.relu(self.bn1(self.conv1(out))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_Dorefa(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,w_bit=32,a_bit=32):
        self.wbit=w_bit
        self.abit=a_bit
        super(Bottleneck_Dorefa, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=self.wbit)
        Linear = linear_Q_fn(w_bit=self.wbit)
        self.feature_quant=activation_quantize_fn(a_bit=self.abit)
        
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.feature_quant(x)
        out = self.feature_quant(F.relu(self.bn1(self.conv1(out))))
        out = self.feature_quant(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_Dorefa(nn.Module):
    def __init__(self, block, num_blocks, blk_stride=[1,2,2,2],num_classes=10,in_channels=3,w_bit=32, a_bit=32,q_out=False,q_in=False, num_features=0) :
        self.wbit=w_bit
        self.abit=a_bit
        super(ResNet_Dorefa, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=self.wbit)
        Linear = linear_Q_fn(w_bit=self.wbit)
        Conv2dfp = conv2d_Q_fn(w_bit=32)
        Linearfp = linear_Q_fn(w_bit=32)
        self.feature_quant=activation_quantize_fn(a_bit=self.abit)
        
        self.in_planes = 64
        self.type = 'single'
        if q_in:
            self.conv1 = Conv2dfp(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=blk_stride[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=blk_stride[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=blk_stride[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=blk_stride[3])
        if num_features !=0:
            if q_out:
                self.linear = Linear(num_features, num_classes)
            else:
                 self.linear = Linearfp(num_features, num_classes) 
        elif(num_classes < 200):
            if q_out:
                self.linear = Linear(512 * block.expansion, num_classes)
            else:
                 self.linear = Linearfp(512 * block.expansion, num_classes)   
        else:
            if q_out:
                self.linear = Linear(2048 * block.expansion, num_classes)
            else:
                self.linear = Linearfp(2048 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,self.wbit,self.abit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.feature_quant(F.avg_pool2d(out, 4))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet_Dorefa_(cfg='18', num_classes=10, in_channels=3,wbit=32,abit=32,q_out=False,q_in=False):
    num_blk= {
        '18' : [2,2,2,2],
        '34' : [3,4,6,3],
        '50' : [3,4,6,3],
        '101' : [3,4,23,3],
        '152' : [3,8,36,3],
    }
    blk_type = {
        '18'    : BasicBlock,
        '34'    : BasicBlock,
        '50'    : Bottleneck,
        '101'   : Bottleneck,
        '152'   : Bottleneck,
    }
    stride = {
        '18' : [1,2,2,2],
        '34' : [1,2,2,2],
        '50' : [1,2,2,2],
        '101' : [1,2,2,2],
        '152' : [1,2,2,2],
    }
    num_features = {
        '18' : 0,
        '34' : 0,
        '50' : 0,
        '101' : 0,
        '152' : 0,
    }
    
    return ResNet_Dorefa(blk_type[cfg], num_blk[cfg], num_classes=num_classes, in_channels=in_channels, blk_stride= stride[cfg],w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in, num_features= num_features[cfg])

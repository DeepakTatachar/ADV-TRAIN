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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.type = 'single'
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if(num_classes < 200):
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


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

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
    def __init__(self, block, num_blocks, num_classes=10,w_bit=32, a_bit=32,q_out=False,q_in=False):
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
            self.conv1 = Conv2dfp(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if(num_classes < 200):
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


def ResNet18_Dorefa(num_classes=10,wbit=32,abit=32,q_out=False,q_in=False):
    return ResNet_Dorefa(BasicBlock_Dorefa, [2, 2, 2, 2], num_classes=num_classes,w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in)


def ResNet34_Dorefa(num_classes=10,wbit=32,abit=32,q_out=False,q_in=False):
    return ResNet_Dorefa(BasicBlock_Dorefa, [3, 4, 6, 3], num_classes=num_classes,w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in)


def ResNet50_Dorefa(num_classes=10,wbit=32,abit=32,q_out=False,q_in=False):
    return ResNet_Dorefa(Bottleneck_Dorefa, [3, 4, 6, 3], num_classes=num_classes,w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in)


def ResNet101_Dorefa(num_classes=10,wbit=32,abit=32,q_out=False,q_in=False):
    return ResNet_Dorefa(Bottleneck_Dorefa, [3, 4, 23, 3], num_classes=num_classes,w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in)


def ResNet152_Dorefa(num_classes=10,wbit=32,abit=32,q_out=False,q_in=False):
    return ResNet_Dorefa(Bottleneck_Dorefa, [3, 8, 36, 3], num_classes=num_classes,w_bit=wbit,a_bit=abit,q_out=q_out,q_in=q_in)

'''
Implementation of FE in ACDS-STR based on ABINET.

Copyright 2023 xuefanfu
'''

import math
import torch.nn as nn
import torch.nn.functional as F
import torch

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers,is_freeze):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.smooth = [nn.Sequential(
                nn.Conv2d(320, 320,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(320),
            ), nn.Sequential(
                nn.Conv2d(352, 352,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(352),
            ),nn.Identity()]
        self.smooth = nn.Sequential(*self.smooth)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1)
        if is_freeze:
            self.freeze(self)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def freeze(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        fpn =[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fpn.append(x)# 32
        x = self.layer1(x)
        x = self.layer2(x)# 64
        fpn.append(x)
        x = self.layer3(x) # 128
        x = self.layer4(x) # 256
        fpn.append(x)
        x = self.layer5(x)
        fpn.reverse()
        for i, (x_fpn,smooth) in enumerate(zip(fpn,self.smooth)):
            if i==2:
                break
            if i!= 0:
                x_fpn = x_fpn_temp
            if i!=2:
                x_fpn_temp=torch.cat((F.interpolate(x_fpn, scale_factor=2, mode='nearest'),fpn[i+1]),dim=1)
            x_fpn_temp=smooth(x_fpn_temp)
        return x,x_fpn_temp


def resnet45(is_freeze):
    return ResNet(BasicBlock, [3, 4, 6, 6, 3],is_freeze)

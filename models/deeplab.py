# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 15:10:50 2020
@author: Diego Wanderley, Filipa Rocha, Henrique Carvalho
@python: 3.6
@description: Proposed DeepLabv3 Network and auxiliary classes.
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP #, DeepLabV3


def get_backbone(name, pretrained=True):
    """ Loading backbone, defining names for skip-connections and encoder output. """

    # TODO: More backbones

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnext50':
        backbone = models.resnext50_32x4d(pretrained=pretrained)
    elif name == 'resnext101':
        backbone = models.resnext101_32x8d(pretrained=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet18') or name.startswith('resnet34'):
        feature_sizes = [   backbone.conv1.out_channels,
                            backbone.layer1[-1].conv2.out_channels,
                            backbone.layer2[-1].conv2.out_channels,
                            backbone.layer3[-1].conv2.out_channels,
                            backbone.layer4[-1].conv2.out_channels  ]
        feature_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    elif name.startswith('resnet') or name.startswith('resnext'):
        feature_sizes = [   backbone.conv1.out_channels,
                            backbone.layer1[-1].conv3.out_channels,
                            backbone.layer2[-1].conv3.out_channels,
                            backbone.layer3[-1].conv3.out_channels,
                            backbone.layer4[-1].conv3.out_channels ]
        feature_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_sizes, feature_names


class ResNetBackbone(nn.Module):
    '''
    ResNet backbone for Feature Pyramid Net.
    '''
    def __init__(self, backbone_model='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        # load backbone
        backbone, features, _ = get_backbone(backbone_model, pretrained=pretrained)
        # backbone input conv
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        #self.relu = self.backbone.relu
        self.maxpool = backbone.maxpool
        # Sequence 1
        self.layer1 = backbone.layer1
        # Sequence 2
        self.layer2 = backbone.layer2
        # Sequence 3
        self.layer3 = backbone.layer3
        # Sequence 4
        self.layer4 = backbone.layer4
        # Output features
        self.features = features
        # Load weigths
        if pretrained:
            self.conv1.load_state_dict(backbone.conv1.state_dict())
            self.bn1.load_state_dict(backbone.bn1.state_dict())
            self.layer1.load_state_dict(backbone.layer1.state_dict())
            self.layer2.load_state_dict(backbone.layer2.state_dict())
            self.layer3.load_state_dict(backbone.layer3.state_dict())
            self.layer4.load_state_dict(backbone.layer4.state_dict())

    def forward(self, x):
        # backbone input conv
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        # Bottleneck sequence
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # output features
        return x1, x2, x3, x4


class DeepLabv3Plus(nn.Module):
    def __init__(self, n_classes=256, layer_base=1, backbone_name='resnet50', pretrained=False):
        super(DeepLabv3Plus, self).__init__()

        # Parameters
        self.layer_base = layer_base
        # Backbone
        self.backbone = ResNetBackbone(backbone_model=backbone_name, pretrained=pretrained)
        featues = self.backbone.features
        self.low_features = featues[layer_base]
        self.high_features = featues[-1]
        # Neck
        self.aspp = ASPP(self.high_features, [12, 24, 36])
        # adopt [1x1, 48] for channel reduction.
        self.conv1 = nn.Sequential()
        self.conv1.add_module('0', nn.Conv2d(self.low_features, 48, 1, bias=False) ) #nn.Conv2d(256, 48, 1, bias=False))
        self.conv1.add_module('1', nn.BatchNorm2d(48))
        self.conv1.add_module('2', nn.ReLU())
        self.conv1.add_module('3', nn.Dropout(0.5))
        # Final conv block
        self.conv2 = nn.Sequential()
        self.conv2.add_module('0', nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2.add_module('1', nn.BatchNorm2d(256))
        self.conv2.add_module('2', nn.ReLU())
        self.conv2.add_module('3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2.add_module('4', nn.BatchNorm2d(256))
        self.conv2.add_module('5', nn.ReLU())
        self.conv2.add_module('6', nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        x_low =  features[self.layer_base-1]
        x_high = features[-1]
        size = ( x_low.shape[-2], x_low.shape[-1] )
        # ASPP
        x1 = self.aspp(x_high)
        # Upsampling
        x2 = F.interpolate( x1, mode='bilinear', align_corners=True, size=size )
        # Low Level
        y = self.conv1(x_low)
        # Combination
        z = torch.cat((x2, y), dim=1)
        z = self.conv2(z)
        # z = F.interpolate(z, size=x.shape[2:], mode='bilinear', align_corners=True)
        z = self.dropout(z)

        out = {
            '0': z,
            '1': y
        }
        return out


if __name__ == "__main__":

    image = torch.randn(4, 3, 512, 512)
    model = DeepLabv3Plus(backbone_name='resnet50', pretrained=True)

    output = model(image)
    print(output.shape)
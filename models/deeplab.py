# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 15:10:50 2020
@author: Diego Wanderley, Filipa Rocha, Henrique Carvalho
@python: 3.6
@description: Proposed DeepLabv3 Network and auxiliary classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP #, DeepLabV3

try: 
    from models.backbones import get_backbone
except:
    from backbones import get_backbone


class ResNetBackbone(nn.Module):
    '''
    ResNet backbone for Feature Pyramid Net.
    '''
    def __init__(self, backbone_model='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        # load backbone
        backbone, out_channels = get_backbone(backbone_model, pretrained=pretrained)
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
        self.sizes = out_channels
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
        return x2, x3, x4


class DeepLabv3(nn.Module):
    def __init__(self, n_classes=256, backbone_name='resnet101', pretrained=False):
        super(DeepLabv3, self).__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained
        # Backbone
        self.backbone, num_features = get_backbone(backbone_name, pretrained=True)
        # Head
        self.classifier = DeepLabHead(num_features, n_classes)
        # model
        #self.deeplab = DeepLabV3(backbone, head)

    def forward(self, x):
        feat = self.backbone(x)
        x_out = self.classifier(feat)
        #x_out = self.deeplab(x)
        
        return x_out


class DeepLabv3Plus(nn.Module):
    def __init__(self, n_classes=256, backbone_name='resnet101', pretrained=False):
        super(DeepLabv3Plus, self).__init__()

        # Backbone
        backbone, num_features = ResNetBackbone(backbone_name=backbone_name, pretrained=pretrained)
        
        # Neck
        self.aspp = ASPP(3, [12, 24, 36])
        self.global_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x1 = self.aspp(x)
        x2 = self.global_pool(x)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x3 = torch.cat((x1, x2), dim=1)

        return x3


if __name__ == "__main__":

    image = torch.randn(4, 3, 512, 512)
    model = DeepLabv3Plus(backbone_name='resnet50', pretrained=True)

    output = model(image)
    print(output.shape)
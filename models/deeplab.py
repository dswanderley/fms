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
from torchvision.models.segmentation.deeplabv3 import DeepLabHead#, DeepLabV3

try: 
    from models.backbones import get_backbone
except:
    from backbones import get_backbone



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


if __name__ == "__main__":

    image = torch.randn(4, 3, 512, 512)
    model = DeepLabv3(backbone_name='resnet50', pretrained=True)

    output = model(image)
    print(output.shape)
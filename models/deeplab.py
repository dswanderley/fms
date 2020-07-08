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
        self.backbone = ResNetBackbone(backbone_model=backbone_name, pretrained=pretrained)
        featues = self.backbone.features
        # Neck
        self.aspp = ASPP(featues[4], [12, 24, 36])
        self.global_pool = nn.Sequential( nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(featues[4], 256, 1, stride=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU() )

        self.conv1 = nn.Sequential( nn.Conv2d(512, 256, 1, bias=False),#nn.Conv2d(1280, 256, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU() )

        self.relu = nn.ReLU()

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Sequential( nn.Conv2d(featues[1], 48, 1, bias=False), #nn.Conv2d(256, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU() )

        # Final conv block
        self.conv3 = nn.Sequential()
        self.conv3.add_module('0', nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv3.add_module('1', nn.BatchNorm2d(256))
        self.conv3.add_module('2', nn.ReLU())
        self.conv3.add_module('3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv3.add_module('4', nn.BatchNorm2d(256))
        self.conv3.add_module('5', nn.ReLU())
        self.conv3.add_module('6', nn.Conv2d(256, n_classes, kernel_size=1, stride=1))


    def forward(self, x):
        # Backbone
        c2, _, _, c5 = self.backbone(x)
        size = ( int(math.ceil(x.shape[-2]/4)), int(math.ceil(x.shape[-1]/4)) )
        # ASPP
        x1 = self.aspp(c5)
        x2 = self.global_pool(c5)
        x2 = F.upsample(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((x1, x2), dim=1)
        # Upsampling
        x4 = self.conv1(x3)
        x4 = F.interpolate( x4, mode='bilinear', align_corners=True, size=size )
        # Low Level
        y = self.conv2(c2)
        # Combination
        z = torch.cat((x4, y), dim=1)
        z = self.conv3(z)
        #z = F.interpolate(z, size=x.shape[2:], mode='bilinear', align_corners=True)

        return z


if __name__ == "__main__":

    image = torch.randn(4, 3, 512, 512)
    model = DeepLabv3Plus(backbone_name='resnet50', pretrained=True)

    output = model(image)
    print(output.shape)
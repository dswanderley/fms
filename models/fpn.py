# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:55:10 2020
@author: Diego Wanderley
@python: 3.6
@description: Feature Pyramid Network class and auxiliary classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as K


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
        backbone, feature_sizes, _ = get_backbone(backbone_model, pretrained=pretrained)
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
        self.fpn_sizes = feature_sizes[-3:]
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


class PyramidFeatures(nn.Module):
    '''
    Features Pyramid Network
    '''
    def __init__(self, num_features=256, backbone_name='resnet50', pretrained=False):
        super(PyramidFeatures, self).__init__()
        # parametes
        self.num_features = num_features
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        # Bottom-up pathway
        self.backbone = ResNetBackbone(backbone_model=backbone_name, pretrained=pretrained)
        # Lateral convolution pathway
        self.latlayer1 = nn.Conv2d(self.backbone.fpn_sizes[2], num_features, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.backbone.fpn_sizes[1], num_features, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.backbone.fpn_sizes[0], num_features, kernel_size=1, stride=1, padding=0)
        # Top-down pathway
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Top layer convs
        self.toplayer1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
       
    def forward(self, x):
        # Bottom-up pathway
        c3, c4, c5 = self.backbone(x)
        self.upsample1 = nn.Upsample(size=c4.shape[-2:], mode='nearest')
        self.upsample2 = nn.Upsample(size=c3.shape[-2:], mode='nearest')
        # Top-down pathway
        p5 = self.latlayer1(c5)
        p4 = self.latlayer2(c4)
        p3 = self.latlayer3(c3)
        p4 = self.upsample1(p5) + p4
        p3 = self.upsample2(p4) + p3
        # Top layers output
        p5 = self.toplayer1(p5)
        p4 = self.toplayer2(p4)
        p3 = self.toplayer3(p3)
        # Output from lower to higher level (larger to smaller spatial size)
        return p3, p4, p5


class GroupedPyramidFeatures(nn.Module):
    '''
    Grouped Features Pyramid Network
    '''
    def __init__(self, out_features=256, num_features=256, backbone_name='resnet50', pretrained=False):
        super(GroupedPyramidFeatures, self).__init__()

        # parametes
        self.num_features = num_features
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        # Network
        self.fpn = PyramidFeatures(num_features = num_features, backbone_name = backbone_name, pretrained = pretrained)
        # Output
        self.conv = nn.Conv2d(num_features * 3, out_features, kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):

        p3, p4, p5 = self.fpn(x)
        
        out_size = p3.shape[-2:]

        p5_up = K.upsample(p5, size=out_size, mode='nearest')
        p4_up = K.upsample(p4, size=out_size, mode='nearest')

        out  = torch.cat((p3, p4_up, p5_up), 1)
        
        out = self.conv(out) 

        return out



if __name__ == "__main__":
    from torch.autograd import Variable

    net = GroupedPyramidFeatures(backbone_name='resnet50',  pretrained=True)
    preds = net( Variable( torch.randn(2,3,840,600) ) )

    for p in preds:
        print(p.shape)

    print('')
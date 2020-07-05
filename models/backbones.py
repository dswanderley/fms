
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 16:25:01 2020
@author: Diego Wanderley
@python: 3.7
@description: Load backbones from torchvision models.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name, pretrained=True):
    """ 
    Returns the backone of a CNN and its number of output channels.

        Parameters:
            name (string):      CNN name
            pretrained (bool):  Pretrained weights condition

        Returns:
            backbone (Sequential):  The CNN backone warped as a torch.nn.Sequential object
            out_channels (int):     The depth of the last layer
    """
    
    # Mobilenet
    if name == 'mobilenet':
        backbone = models.mobilenet_v2(pretrained=pretrained).features
        out_channels = 1280
    # ResNet
    elif name == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 512
    elif name == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 512
    elif name == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 2048
    elif name == 'resnet101':
        resnet = models.resnet101(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 2048
    # ResNeXt
    elif name == 'resnext50':
        resnext = models.resnext50_32x4d(pretrained=pretrained)
        modules = list(resnext.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 2048
    elif name == 'resnext101':
        resnext = models.resnext101_32x8d(pretrained=pretrained)
        modules = list(resnext.children())[:-1]
        backbone = nn.Sequential(*modules)
        out_channels = 2048
    # Error
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, out_channels




if __name__ == "__main__":
    from torch.autograd import Variable

    my_backbone, channels = get_backbone('resnext101', pretrained=True)

    print('')
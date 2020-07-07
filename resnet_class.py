import utils
import torch
import torch.nn as nn
import torchvision.models as models


class resnet18_classifier(nn.Module):

    def __init__(self, backbone, pretrained=False):
        super(resnet18_classifier, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)

        self.backbone.fc = nn.Linear(512, 2)

        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.backbone.forward(x)

        out = self.sm(x)

        return out



import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import math
import time
import numpy as np

darknet_cfg = {
    19 : [[32,3],'M',[64,3],'M',[128,3],[64,1],[128,3],'M',[256,3],[128,1],[256,3],'M',[512,3],[256,1],[512,3],[256,1],[512,3],'M',[1024,3],[512,1],[1024,3],[512,1],[1024,3]]
}

class darknet(nn.Module):
    def __init__(self, depth=19, init_weights=True, cfg=None):
        super(darknet, self).__init__()
        if cfg is None:
            cfg = darknet_cfg[depth]

        self.feature = self.make_layers(cfg, True)
        num_classes = 30
        self.classifier = nn.Sequential(nn.Conv2d(cfg[-1][0], out_channels=num_classes, kernel_size=1, stride=1, padding=1, bias=False), nn.AvgPool2d(7))
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, out_channels=l[0], kernel_size=l[1], stride=1, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(l[0]), nn.LeakyReLU(0.1)]
                else:
                    layers += [conv2d, nn.LeakyReLU(0.1)]
                in_channels = l[0]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        y = x.view(x.size(0), -1)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


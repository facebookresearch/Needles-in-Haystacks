# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# adapted from https://github.com/wielandbrendel/bag-of-local-features-models
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.optim as optim
import numpy as np
import torchvision


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm=lambda x: nn.InstanceNorm2d(x, affine=True),
                 kernel_size=1, dropout=False, compensate=False):
        super(Bottleneck, self).__init__()
        mid_planes = planes
        if compensate:
            mid_planes = int(2.5 * planes)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)

        self.conv2 = nn.Conv2d(planes, mid_planes, kernel_size=kernel_size,
                               stride=stride,
                               padding=(kernel_size - 1) // 2, # used to be 0
                               bias=False)  # changed padding from (kernel_size - 1) // 2
        self.bn2 = norm(mid_planes)

        self.drop = nn.Dropout2d(p=0.2) if dropout else lambda x: x
        self.conv3 = nn.Conv2d(mid_planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNetEncoder(nn.Module):
    norms = {
        'in_aff': lambda x: nn.InstanceNorm2d(x, affine=True),
        'in': nn.InstanceNorm2d,
        'bn': nn.BatchNorm2d
    }

    def __init__(self, block, layers, strides=[1, 2, 2, 2], wide_factor=1,
                 kernel3=[0, 0, 0, 0], dropout=False, inp_channels=3,
                 compensate=False, norm='in_aff'):
        self.planes = int(64 * wide_factor)
        self.inplanes = int(64 * wide_factor)
        self.compensate = compensate
        self.dropout = dropout
        self.norm = norm
        super(BagNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, self.planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.planes, self.planes, kernel_size=3,
                               stride=1, padding=0, bias=False)
        self.bn1 = self.norms[self.norm](self.planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.planes, layers[0],
                                       stride=strides[0], kernel3=kernel3[0],
                                       prefix='layer1')
        self.layer2 = self._make_layer(block, self.planes * 2, layers[1],
                                       stride=strides[1], kernel3=kernel3[1],
                                       prefix='layer2')
        self.layer3 = self._make_layer(block, self.planes * 4, layers[2],
                                       stride=strides[2], kernel3=kernel3[2],
                                       prefix='layer3')
        self.layer4 = self._make_layer(block, self.planes * 8, layers[3],
                                       stride=strides[3], kernel3=kernel3[3],
                                       prefix='layer4')
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.InstanceNorm2d) and self.norm == 'in_aff':
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0,
                    prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norms[self.norm](planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel, dropout=self.dropout,
                            norm=self.norms[self.norm],
                            compensate=(self.compensate and kernel == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel,
                                norm=self.norms[self.norm],
                                compensate=(self.compensate and kernel == 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Network(nn.Module):
    def encode(self, inputs):
        encoded = self.cnn(inputs)

        return encoded

    def mean_forward(self, encoded):
        encoded = encoded.mean(2)

        return encoded, None

    def max_forward(self, encoded):
        encoded, _ = encoded.max(2)

        return encoded, None

    def logsumexp_forward(self, encoded):
        encoded = encoded.logsumexp(2)

        return encoded, None

    def soft_mil_att_forward(self, encoded):
        summarized = self.sum_mlp(encoded).view(encoded.shape[0], -1)

        att = F.softmax(summarized, 1)

        encoded = (encoded * att.unsqueeze(1)).sum(2)

        return encoded, att

    def __init__(self, k=0, att_type='mean_forward', dropout=False,
                 width=0, num_classes=2, inp_channels=1, norm='in_aff',
                 kernel3=[1, 1, 1, 1], compensate=False):
        super(Network, self).__init__()

        self.att_type = att_type
        self.k = k

        block = Bottleneck

        wide_factor = 2 ** width

        encoding_size = int(512 * block.expansion * wide_factor)

        self.cnn = BagNetEncoder(
            block, [3, 4, 6, 3], strides=[2, 2, 2, 1], kernel3=kernel3,
            dropout=dropout, inp_channels=inp_channels, compensate=compensate,
            norm=norm, wide_factor=wide_factor)

        self.encoding_size = encoding_size

        self.mlp = nn.Linear(encoding_size, num_classes)

        if 'mil_att' in self.att_type:
            self.sum_mlp = nn.Sequential(nn.Conv1d(encoding_size, 128, kernel_size=1), nn.Tanh(), nn.Conv1d(128, 1, kernel_size=1))

    def forward(self, inputs, local_predictions=False):
        """
        We assume inputs is [batch, C, H, W] for a single image
        """
        encoded = self.encode(inputs)

        if local_predictions:
            localised = self.mlp(encoded.permute(0, 2, 3, 1))

        num_images, c, h, w = encoded.shape
        encoded = encoded.view(num_images, c, -1)

        f = getattr(self, self.att_type, None)
        if f is None:
            raise Exception('att type not fround')
        encoded, att = f(encoded)

        out = self.mlp(encoded)
        if local_predictions:
            return out, att, localised
        return out, att

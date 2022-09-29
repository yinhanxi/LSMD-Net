#!/usr/bin/env python
import os
import numpy as np
from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Submodule import SubModule
from .util_conv import BasicConv, Conv2x, BasicBlock, conv3x3, conv1x1
from .util_conv import BasicConv2d, BasicTransposeConv2d

import timm

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False), nn.BatchNorm2d(out_planes))

class PSMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(PSMBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeatUp(SubModule):
    def __init__(self, typ, chns):
        super().__init__()
        self.typ = typ

        if not self.typ == 'psm':
            self.deconv32_16 = Conv2x(chns[4], chns[3], deconv=True, concat=True)
            self.deconv16_8 = Conv2x(chns[3]*2, chns[2], deconv=True, concat=True)
            self.deconv8_4 = Conv2x(chns[2]*2, chns[1], deconv=True, concat=True)
            self.conv4 = BasicConv(chns[1]*2, chns[1]*2, kernel_size=3, stride=1, padding=1)

            self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        if self.typ == 'psm':
            return featL, featR

        if featR is not None:
            y4, y8, y16, y32 = featR
            x16 = self.deconv32_16(x32, x16)
            y16 = self.deconv32_16(y32, y16)

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

            return [x4, x8, x16, x32], [y4, y8, y16, y32]
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(x4)
            return [x4, x8, x16, x32]


class Feature(SubModule):
    def __init__(self, typ, chns, layers, from_scratch=True):
        super().__init__()
        self.typ = typ
        self.from_scratch = from_scratch

        if self.typ == 'custom_basic':

            self.conv_stem = nn.Sequential(
                BasicConv(3, chns[0], kernel_size=3, stride=2, padding=1),
                BasicConv(chns[0], chns[0], kernel_size=3, stride=1, padding=1),
                )
            self.block1a = nn.Sequential(
                BasicConv(chns[0], chns[1], kernel_size=3, stride=2, padding=1),
                BasicConv(chns[1], chns[1], kernel_size=3, stride=1, padding=1),
                )
            self.block2a = nn.Sequential(
                BasicConv(chns[1], chns[2], kernel_size=3, stride=2, padding=1),
                BasicConv(chns[2], chns[2], kernel_size=3, stride=1, padding=1),
                )
            self.block3a = nn.Sequential(
                BasicConv(chns[2], chns[3], kernel_size=3, stride=2, padding=1),
                BasicConv(chns[3], chns[3], kernel_size=3, stride=1, padding=1),
                )
            self.block4a = nn.Sequential(
                BasicConv(chns[3], chns[4], kernel_size=3, stride=2, padding=1),
                BasicConv(chns[4], chns[4], kernel_size=3, stride=1, padding=1),
                )

            self.deconv4a = Conv2x(chns[4], chns[3], deconv=True, keep_concat=False)
            self.deconv3a = Conv2x(chns[3], chns[2], deconv=True, keep_concat=False)
            self.deconv2a = Conv2x(chns[2], chns[1], deconv=True, keep_concat=False)
            self.deconv1a = Conv2x(chns[1], chns[0], deconv=True, keep_concat=False)

            self.conv1b = Conv2x(chns[0], chns[1], keep_concat=False)
            self.conv2b = Conv2x(chns[1], chns[2], keep_concat=False)
            self.conv3b = Conv2x(chns[2], chns[3], keep_concat=False)
            self.conv4b = Conv2x(chns[3], chns[4], keep_concat=False)

            self.weight_init()

        elif self.typ == 'custom_res':

            self.chns = chns
            block_n = [2, 2, 2, 2]

            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, chns[0], 3, 1, 1),
                nn.BatchNorm2d(chns[0]), nn.ReLU6())

            inp = chns[0]
            self.conv = nn.ModuleList()
            for i in range(len(chns)-1):
                conv: List[nn.ModuleList] = []
                for n in range(block_n[i]):
                    if n == 0:
                        stride = 2
                        downsample = nn.Sequential(
                            conv1x1(inp, chns[i+1], stride),
                            nn.BatchNorm2d(chns[i+1]),
                        )
                    else:
                        stride, downsample = 1, None
                    conv.append(BasicBlock(inp, chns[i+1], stride, downsample=downsample))
                    inp = chns[i+1]
                self.conv.append(nn.Sequential(*conv))

            self.weight_init()

        elif self.typ == 'psm':
            self.inplanes = 32
            self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 32, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 32, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True))

            self.layer1 = self._psm_make_layer(PSMBasicBlock, 32, 3, 1,1,1)
            self.layer2 = self._psm_make_layer(PSMBasicBlock, 64, 16, 2,1,1)
            self.layer3 = self._psm_make_layer(PSMBasicBlock, 128, 3, 1,1,1)
            self.layer4 = self._psm_make_layer(PSMBasicBlock, 128, 3, 1,1,2)

            self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 64, kernel_size=1, padding=0, stride = 1, bias=False))

        else:
            model = timm.create_model(self.typ, pretrained=not self.from_scratch, features_only=True)

            if 'resnet' in self.typ:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.act1 = model.act1
                self.maxpool = model.maxpool

                self.layer1 = model.layer1
                self.layer2 = model.layer2
                self.layer3 = model.layer3
                self.layer4 = model.layer4
            else:
                self.conv_stem = model.conv_stem
                self.bn1 = model.bn1
                self.act1 = model.act1

                self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
                self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
                self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
                self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
                self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        if self.typ == 'custom_basic':
            x2 = self.conv_stem(x)
            x4 = self.block1a(x2)
            x8 = self.block2a(x4)
            x16 = self.block3a(x8)
            x32 = self.block4a(x16)

            x16 = self.deconv4a(x32, x16)
            x8 = self.deconv3a(x16, x8)
            x4 = self.deconv2a(x8, x4)
            x2 = self.deconv1a(x4, x2)

            x4 = self.conv1b(x2, x4)
            x8 = self.conv2b(x4, x8)
            x16 = self.conv3b(x8, x16)
            x32 = self.conv4b(x16, x32)

            return x2, [x4, x8, x16, x32]

        elif self.typ == 'custom_res':
            x = self.conv_stem(x)

            outs = []
            x_ = x
            for i in range(len(self.chns)-1):
                x_ = self.conv[i](x_)
                outs.append(x_)

            return x, outs

        elif self.typ == 'psm':
            output2     = self.firstconv(x)
            output2     = self.layer1(output2)
            output_raw  = self.layer2(output2)
            output      = self.layer3(output_raw)
            output_skip = self.layer4(output)


            output_branch1 = self.branch1(output_skip)
            output_branch1 = F.upsample(
                output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch2 = self.branch2(output_skip)
            output_branch2 = F.upsample(
                output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch3 = self.branch3(output_skip)
            output_branch3 = F.upsample(
                output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch4 = self.branch4(output_skip)
            output_branch4 = F.upsample(
                output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
            output_feature = self.lastconv(output_feature)

            xout = [output_feature, output_feature, output_feature, output_feature]
            return output2, xout

        else:
            if 'resnet' in self.typ:
                x2 = self.act1(self.bn1(self.conv1(x)))
                x4 = self.layer1(self.maxpool(x2))
                x8 = self.layer2(x4)
                x16 = self.layer3(x8)
                x32 = self.layer4(x16)
            else:
                x = self.act1(self.bn1(self.conv_stem(x)))
                x2 = self.block0(x)
                x4 = self.block1(x2)

                # return x4,x4,x4,x4
                x8 = self.block2(x4)
                x16 = self.block3(x8)
                x32 = self.block4(x16)

            x_out = [x4, x8, x16, x32]

            return x2, x_out

    def _psm_make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)
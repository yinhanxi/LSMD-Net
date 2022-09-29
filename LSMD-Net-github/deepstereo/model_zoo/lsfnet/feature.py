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


class Stereo_FeatUp(SubModule):
    def __init__(self, typ, chns):
        super().__init__()
        self.typ = typ

        self.deconv32_16 = Conv2x(chns[4], chns[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chns[3]*2, chns[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chns[2]*2, chns[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chns[1]*2, chns[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
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

class Mono_Feat(SubModule):
    def __init__(self, typ, chns,out_chn):
        super().__init__()
        self.typ = typ

        self.conv16 = BasicConv(chns[3],out_chn, kernel_size=3, stride=1, padding=1)
        self.conv8 = BasicConv(chns[2],out_chn, kernel_size=3, stride=1, padding=1)
        self.conv4 = BasicConv(chns[1],out_chn, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(chns[0],out_chn, kernel_size=3, stride=1, padding=1)
        #self.conv4 = BasicConv(chns[1]*2, chns[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL):
        x2, x4, x8, x16 = featL

        x16 = self.conv16(x16)

        x8 = self.conv8(x8)

        x4 = self.conv4(x4)

        x2 = self.conv2(x2)

        return [x2, x4, x8, x16]


class Feature(SubModule):
    def __init__(self, typ, chns, layers, pretrained=True):
        super().__init__()
        self.typ = typ
        self.pretrained = pretrained

        model = timm.create_model(self.typ, pretrained=self.pretrained, features_only=True)

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

            return x2,x_out

class SparseConv(nn.Module):
	# Convolution layer for sparse data
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, bias=True):
		super(SparseConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
		self.if_bias = bias
		if self.if_bias:
			self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)

		nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
		self.pool.require_grad = False

	def forward(self, input):
		x, m = input
		mc = m.expand_as(x)
		x = x * mc
		x = self.conv(x)

		weights = torch.ones_like(self.conv.weight)
		mc = F.conv2d(mc, weights, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
		mc = torch.clamp(mc, min=1e-5)
		mc = 1. / mc
		x = x * mc
		if self.if_bias:
			x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
		m = self.pool(m)

		return x, m

class SparseConvBlock(nn.Module):

	def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, dilation=1, bias=True):
		super(SparseConvBlock, self).__init__()
		self.sparse_conv = SparseConv(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, input):
		x, m = input
		x, m = self.sparse_conv((x, m))
		assert (m.size(1)==1)
		x = self.relu(x)
		return x, m

class SparseConvNet(nn.Module):

	def __init__(self, in_channel=1, out_channel=16, kernels=[11,7,5,3,3], mid_channel=16):
		super(SparseConvNet, self).__init__()
		channel = in_channel
		convs = []
		for i in range(len(kernels)):
			if  i%2:
				stride = 2
			else:
			    stride = 1	
			assert (kernels[i]%2==1)
			convs += [SparseConvBlock(channel, mid_channel, kernels[i], stride=stride, padding=(kernels[i]-1)//2)]
			channel = mid_channel
		self.sparse_convs = nn.Sequential(*convs)
		self.mask_conv = nn.Conv2d(mid_channel+1, out_channel, 1)

	def forward(self, x):
		m = (x>0).detach().float()
		x, m = self.sparse_convs((x,m))
		x = torch.cat((x,m), dim=1)
		x = self.mask_conv(x)
		#print(x.shape)
		return x

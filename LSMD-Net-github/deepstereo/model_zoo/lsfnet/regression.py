#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spixel_utils import spixel
from .Submodule import SubModule
from .util_conv import BasicConv

class Regression(SubModule):
    def __init__(self,
                 max_disp=192,
                 top_k=2):
        super().__init__()
        self.max_disp = max_disp
        self.top_k = top_k
        self.ind_init = False

    def forward(self, cost, spg):
        b, _, h, w = spg.shape
        corr, disp = self.topkpool(cost, self.top_k)
        corr = F.softmax(corr, 2)

        disp_4 = torch.sum(corr*disp, 2, keepdim=True)
        disp_4 = disp_4.reshape(b, 1, disp_4.shape[-2], disp_4.shape[-1])

        disp_1 = (spixel.upfeat(disp_4, spg, 4, 4))
        # disp_1 = (spixel.upfeatHW(disp_4, spg, h, w))

        disp_1 = disp_1 * 4
        disp_4 = disp_4

        return [disp_1, disp_4]

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(2, True)
            pool_ind_ = ind[:, :, :k]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_-1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_+1, (self.max_disp // 4)*pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(2, True)
            pool_ind = ind[:, :, :k]
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        return cv, disp


class MLP(SubModule):
    def __init__(self, filter_channels,no_residual=False):
        super(MLP, self).__init__()

        self.filters = []
        self.no_residual = no_residual
        #self.activation =  nn.LeakyReLU()

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv2d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv2d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv2d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = f(y)
            else:
                y = f(
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            #if i != len(self.filters) - 1:
                #y = self.activation(y)
        return y

class GCERegression(SubModule):
    def __init__(self,max_disp,x2_channel,x1_channel):
        super(GCERegression,self).__init__()
        self.max_disp = max_disp
        self.upconv_2x = BasicConv(self.max_disp//4,x2_channel,is_3d=False,deconv=True,kernel_size=4,stride=2,padding=1)
        self.upconv_1x = BasicConv(x2_channel,x1_channel,is_3d=False,deconv=True,kernel_size=4,stride=2,padding=1)
        self.conv = BasicConv(x1_channel,1,is_3d=False,deconv=False,kernel_size=3,stride=1,padding=1,bn=False,relu=False)
        self.weight_init()

    def forward(self,cost_4x,feat):

        feat_2x,feat_1x = feat

        cost_2x = self.upconv_2x(cost_4x)
        #cost_2x = torch.sigmoid(feat_2x)*cost_2x

        cost_1x = self.upconv_1x(cost_2x)
        cost_1x = torch.sigmoid(feat_1x)*cost_1x

        cost_1x = self.conv(cost_1x)

        #return cost_1x[:,0,:,:],cost_1x[:,1,:,:]
        return cost_1x

class Uncertainty(SubModule):
    def __init__(self, filter_channels,no_residual=False):
        super(Uncertainty, self).__init__()

        self.filters = []
        self.no_residual = no_residual
        #self.activation =  nn.LeakyReLU()
        self.len = len(filter_channels)-1

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv2d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    3,1,1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv2d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            3,1,1))
                else:
                    self.filters.append(nn.Conv2d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        3,1,1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = f(y)
            else:
                if i==0:
                    y=f(y)
                elif i==self.len-1 or i==self.len-2:
                    y=F.interpolate(y,scale_factor=2, mode='bilinear')
                    tmpy=F.interpolate(tmpy,scale_factor=2, mode='bilinear')
                    y=f(torch.cat([y, tmpy],1))
                else:
                    y=f(torch.cat([y, tmpy], 1))
            #if i != len(self.filters) - 1:
            #    y = self.activation(y)
        return y
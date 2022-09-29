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

        disp_1= (spixel.upfeat(disp_4, spg, 4, 4))
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


class Regression2(SubModule):
    def __init__(self, disp_pos, disp_neg, slant):
        super(Regression2, self).__init__()
        self.disp_pos = disp_pos
        self.disp_neg = disp_neg
        self.slant = slant

        self.ind_init = False
        self.grid_initialized = False

    def forward(self, cost, disp_4b, spg=None, training=False):
        b, _, _, h, w = cost.shape

        corr, disp = self.topkpool(cost, 2)
        corr_ = F.softmax(corr[:, :1], 2)

        disp_4 = torch.sum(corr_*disp, 2, keepdim=True)
        disp_4 = disp_4.reshape(b, h, w, 1).permute(0, 3, 1, 2)
        disp_4 = disp_4 + disp_4b

        if spg is None:
            return disp_4

        if self.slant:
            slant = torch.sum(corr_*corr[:, 1:], 2)
            if not self.grid_initialized:
                self.init_grid()
            disp_1 = (spixel.upfeat_slant(disp_4, spg, slant, 4, 4))
        else:
            disp_1 = (spixel.upfeat(disp_4, spg, 4, 4))

        if training:
            return [disp_1, disp_4]
        else:
            return [disp_1]

    def topkpool(self, cost, k):
        if not self.ind_init:
            self.ind_ = torch.arange(
                -self.disp_neg, self.disp_pos + 1,
                device=cost.device).reshape(1, 1, -1, 1, 1)
            self.ind = self.ind_.repeat(
                1, 1, 1, cost.shape[-2], cost.shape[-1])
            self.ind_init = True
        if self.ind.shape[-2:] != cost.shape[-2:]:
            self.ind = self.ind_.repeat(
                1, 1, 1, cost.shape[-2], cost.shape[-1])

        pool_ind = cost[:, :1].argsort(2, True)[:, :, :k]
        if self.slant:
            pool_ind_ = pool_ind.repeat(1, 3, 1, 1, 1)
            cv = torch.gather(cost, 2, pool_ind_)
        else:
            cv = torch.gather(cost, 2, pool_ind)

        disp = torch.gather(self.ind.repeat(
            cv.shape[0], 1, 1, 1, 1), 2, pool_ind)

        return cv, disp

    def init_grid(self, ):
        grid_x = torch.arange(0, 4, dtype=torch.float) - 1.5
        grid_y = torch.arange(0, 4, dtype=torch.float) - 1.5

class MLP(SubModule):
    def __init__(self, filter_channels,no_residual=False):
        super(MLP, self).__init__()

        self.filters = []
        self.no_residual = no_residual
        self.activation =  nn.LeakyReLU()

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
            if i != len(self.filters) - 1:
                y = self.activation(y)
        return y
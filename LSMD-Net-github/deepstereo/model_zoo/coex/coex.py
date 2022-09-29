from timeit import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...blocks import upsample_disp
from .feature import Feature, FeatUp
from .utils import AttentionCostVolume
from .aggregation import Aggregation
from .regression import Regression,MLP
from .util_conv import BasicConv, Conv2x
from ...blocks import upsample_disp
from .spixel_utils import spixel
from einops import repeat

ARCH_CONFIGS = {
    'mobilenetv3_large_100': {
        'channels': [16,24,40,112,160],
        'layers': [1,2,3,5,6],
    },
    'mobilenetv2_120d': {
        'channels': [24,32,40,112,192],
        'layers': [1,2,3,5,6],
    },
    'mobilenetv2_100': {
        'channels': [16,24,32,96,160],
        'layers': [1,2,3,5,6],
    },
    'mnasnet_100': {
        'channels': [16,24,40,96,192],
        'layers': [1,2,3,5,6],
    },
    'efficientnet_b0': {
        'channels': [16,24,40,112,192],
        'layers': [1,2,3,5,6],
    },
    'efficientnet_b3a': {
        'channels': [24,32,48,136,232],
        'layers': [1,2,3,5,6],
    },
    'mixnet_xl': {
        'channels': [40,48,64,192,320],
        'layers': [1,2,3,5,6],
    },
    'dla34': {
        'channels': [32,64,128,256,512],
        'layers': [1,2,3,5,6],
    },
}

class CoEx(nn.Module):
    def __init__(self, max_disp, 
                 typ='mobilenetv2_100', 
                 spixel_branch_channels=[32,48],
                 from_scratch=True,
                 corr_volume=True,
                 matching_weighted=False,
                 matching_head=1,
                 gce=True,
                 agg_disp_strides = 2,
                 agg_chns = [16, 32, 48],
                 agg_blocks_num = [2, 2, 2],
                 reg_top_k = 2,
                 loss_type = 'unimodal', # unimodal,L1
    ):
        super().__init__()
        self.typ = typ
        cfg = ARCH_CONFIGS[typ]
        chns = cfg['channels']
        layers = cfg['layers']

        self.max_disp = max_disp
        self.loss_type = loss_type
        # set up the feature extraction first
        self.feature = Feature(typ, chns, layers, from_scratch=from_scratch)
        self.up = FeatUp(typ, chns)

        self.corr_volume = corr_volume
        if self.corr_volume:
            self.cost_volume = AttentionCostVolume(
                max_disp,
                chns[1]*2 + spixel_branch_channels[1],
                chns[1]*2,
                1,
                weighted=matching_weighted)
        else:
            self.cost_conv = BasicConv(
                chns[1]*2 + spixel_branch_channels[1],
                chns[1]*2,
                kernel_size=3,
                padding=1,
                stride=1)
            self.cost_desc = nn.Conv2d(
                chns[1]*2,
                chns[1],
                kernel_size=1,
                padding=0,
                stride=1)
            matching_head = chns[1]*2
        
        div = [2,2,2,2,1]
        self.cost_agg = Aggregation(
            chns,
            max_disp=max_disp,
            matching_head=matching_head,
            gce=gce,
            disp_strides=agg_disp_strides,
            channels=agg_chns,
            blocks_num=agg_blocks_num,
            spixel_branch_channels=spixel_branch_channels)

        self.regression = Regression(
            max_disp=max_disp,
            top_k=reg_top_k)

        self.stem_2 = nn.Sequential(
            BasicConv(3, spixel_branch_channels[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spixel_branch_channels[0], spixel_branch_channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spixel_branch_channels[0]), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(spixel_branch_channels[0], spixel_branch_channels[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spixel_branch_channels[1], spixel_branch_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(spixel_branch_channels[1]), nn.ReLU()
            )

        self.spx = nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1)
        self.spx_2 = Conv2x(chns[1], 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(chns[1]*2 + spixel_branch_channels[1], chns[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(chns[1], chns[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chns[1]), nn.ReLU()
            )

        if loss_type == 'unimodal':
            self.mlp = MLP(filter_channels=[max_disp//4,512,256,128,64,1])
            self.activation = nn.Sigmoid()

    def forward(self, x_l, x_r,train=False):
        assert x_l.shape == x_r.shape

        # # Matching comp
        _,x = self.feature(x_l)
        _,y = self.feature(x_r)

        x, y = self.up(x, y)

        stem_2x = self.stem_2(x_l)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(x_r)
        stem_4y = self.stem_4(stem_2y)

        x[0] = torch.cat((x[0], stem_4x), 1)
        y[0] = torch.cat((y[0], stem_4y), 1)

        # Cost volume processing

        if self.corr_volume:
            cost = (self.cost_volume(x[0], y[0]))[:, :, :-1]
        else:
            refimg_fea = self.cost_conv(x[0])
            targetimg_fea = self.cost_conv(y[0])
            refimg_fea = self.cost_desc(refimg_fea)
            targetimg_fea = self.cost_desc(targetimg_fea)

            cost = torch.FloatTensor(
                refimg_fea.size()[0],
                refimg_fea.size()[1]*2,
                self.D,
                refimg_fea.size()[2],
                refimg_fea.size()[3]).zero_()

            for i in range(self.D):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                    cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
            cost = cost.contiguous()

        cost = self.cost_agg(x, cost)

        # spixel guide comp
        spx_4x = self.spx_4(x[0])
        spx_2x = self.spx_2(spx_4x,stem_2x)
        spx_pred = self.spx(spx_2x)
        spx_pred = F.softmax(spx_pred,1)
        pred = self.regression(cost, spx_pred)

        if self.loss_type == 'unimodal':
            cost = torch.squeeze(cost,1)
            #b,d,h,w = cost.shape
            #disp_range = repeat(torch.arange(0,d).type_as(cost),'d -> b d h w',b=b,h=h,w=w)
            #mlp_output = self.mlp(torch.cat((cost,disp_range),1))
            mlp_output = self.mlp(cost)
            #mlp_output = self.activation(mlp_output)
            #pred = (spixel.upfeat(mlp_output[:,:1,:,:],spx_pred,4,4))*self.max_disp
            #prob = (spixel.upfeat(mlp_output[:,1:2,:,:],spx_pred,4,4))
            #pred_mid = F.interpolate(mlp_output[:,:1,:,:],scale_factor=4,mode='bilinear',align_corners=True)*self.max_disp
            #prob_mid = F.interpolate(mlp_output[:,1:2,:,:],scale_factor=4,mode='bilinear',align_corners=True)
            var = F.interpolate(mlp_output,scale_factor=4,mode='bilinear',align_corners=True)
            #prob = torch.clamp(self.activation(var),1e-2,1.0)
            #prob = self.activation(var)
            prob=var.exp()
            return {'disp':pred,'prob':prob}
        elif self.loss_type == 'L1':
            pred = self.regression(cost, spx_pred)
            #mask = (disp_input>0).float()
            #pred[0] = pred[0]*(1-mask)+disp_input
            return {'disp':pred}
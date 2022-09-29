from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...blocks import upsample_disp
from .feature import Feature, Stereo_FeatUp,Mono_Feat,SparseConvNet
from .utils import AttentionCostVolume
from .aggregation import Aggregation
from .regression import Regression,MLP,GCERegression,Uncertainty
from .util_conv import BasicConv, Conv2x
from ...blocks import upsample_disp
from .spixel_utils import spixel

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

'''class LSFNet(nn.Module):
    def __init__(self, max_disp, typ='mobilenetv2_100', spixel_branch_channels=[32,48],
                 pretrained=True,
                 corr_volume=True,
                 matching_weighted=False,
                 matching_head=1,
                 gce=True,
                 agg_disp_strides = 2,
                 agg_chns = [16, 32, 48],
                 agg_blocks_num = [2, 2, 2],
                 reg_top_k = 2,
    ):
        super().__init__()
        self.typ = typ
        cfg = ARCH_CONFIGS[typ]
        chns = cfg['channels']
        layers = cfg['layers']

        self.max_disp = max_disp

        # set up the feature extraction first
        self.feature = Feature(typ, chns, layers, pretrained=pretrained)
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

        # monocular
        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        disp_channel = 1

        self.depth_encoder1 = DepthEncoder(disp_channel, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(disp_channel+1, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(disp_channel+1, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.stereo_uncertainty = nn.Sequential(
            BasicConv(max_disp//4,max_disp//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(max_disp//4,max_disp//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(max_disp//4), nn.ReLU()
            )
        self.monocular_uncertainty = nn.Sequential(
            BasicConv(ddcd_layers,ddcd_layers, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ddcd_layers,ddcd_layers, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ddcd_layers), nn.ReLU()
            )
        self.uncertainty = nn.Sequential(
            BasicConv(max_disp//4+ddcd_layers,max_disp//4+ddcd_layers, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(max_disp//4+ddcd_layers,max_disp//4+ddcd_layers, 3, 1, 1, bias=False),
            nn.BatchNorm2d(max_disp//4+ddcd_layers), nn.ReLU()
            )
        self.monocular_mlp = MLP(filter_channels=[ddcd_layers,512,256,128,64,1])
        self.stereo_mlp = MLP(filter_channels=[max_disp//4,512,256,128,64,1])
        self.mlp = MLP(filter_channels=[max_disp//4+ddcd_layers,512,256,128,64,1])
        self.pi_activation = nn.Sigmoid()
        self.sigma_activation = nn.ELU()

    def forward(self,x_l,x_r,disp_l,train=False):
        assert x_l.shape == x_r.shape

        # # Matching comp
        x1,x2, x = self.feature(x_l)
        _,_, y = self.feature(x_r)

        print(1111111111111111111)
        print(x1.shape,x2.shape,x[0].shape,x[1].shape,x[2].shape,x[3].shape)

        x, y = self.up(x,y)

        stem_2x = self.stem_2(x_l)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(x_r)
        stem_4y = self.stem_4(stem_2y)

        x[0] = torch.cat((x[0],stem_4x), 1)
        y[0] = torch.cat((y[0],stem_4y), 1)


        # Cost volume processing

        if self.corr_volume:
            cost = (self.cost_volume(x[0],y[0]))[:, :, :-1]
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
        spx_pred = F.softmax(spx_pred, 1)
        pred = self.regression(cost, spx_pred)
        stereo_disp_pred = pred[0]
        stereo_disp_mid = pred[1:]
        
        C = (disp_l>0).float()

        enc_c = self.rgb_encoder(x_l)
        print(2222222222222222)
        print(enc_c[0].shape,enc_c[1].shape,enc_c[2].shape,enc_c[3].shape,enc_c[4].shape)
        raise

       ## for the 1/4 res
        disp_14 = 0.25*F.avg_pool2d(disp_l, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(disp_14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        disp_12 = 0.5*F.avg_pool2d(disp_l, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = 2*F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((disp_12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = 2*F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((disp_l,predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = 4*F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        mono_disp_pred=output_d11
        mono_disp_mid = [output_d12,output_d14]

        
        cost_feat = torch.squeeze(cost,1)
        stereo_prob = self.stereo_mlp(cost_feat)
        stereo_prob = F.interpolate(stereo_prob,scale_factor=4,mode='bilinear',align_corners=True)
        stereo_prob = self.sigma_activation(stereo_prob)+1+1e-15
        #stereo_prob = torch.clamp(stereo_prob,0.1)

        monocular_feat = torch.cat((enc_d11[2],enc_c[2]),1)
        monocular_prob = self.monocular_mlp(monocular_feat)
        monocular_prob = F.interpolate(monocular_prob,scale_factor=4,mode='bilinear',align_corners=True)
        monocular_prob = self.sigma_activation(monocular_prob)+1+1e-15
        #monocular_prob = torch.clamp(monocular_prob,0.1)

        cost_feat = self.stereo_uncertainty(cost_feat)
        monocular_feat = self.monocular_uncertainty(monocular_feat)
        uncertainty_feat = torch.cat((cost_feat,monocular_feat),1)
        uncertainty_feat = self.uncertainty(uncertainty_feat)
        pi = self.mlp(uncertainty_feat)
        pi = F.interpolate(pi,scale_factor=4,mode='bilinear',align_corners=True)
        pi = self.pi_activation(pi)


        return {'disp_monocular':mono_disp_pred,
                'prob_monocular':monocular_prob,
                'disp_stereo':stereo_disp_pred,
                'prob_stereo':stereo_prob,
                'pi':pi,
                'disp_stereo_mid':stereo_disp_mid,
                'disp_monocular_mid':mono_disp_mid,
                'disp':stereo_disp_pred*(pi>=0.5)+mono_disp_pred*(pi<0.5),
                'disp1':stereo_disp_pred*pi+mono_disp_pred*(1-pi)
        }

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + scale*F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + scale*F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + scale*F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  )

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers), 
                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), 
                                  #nn.BatchNorm2d(layers),
                                  )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers), 
                                  )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), 
                                  #nn.BatchNorm2d(layers),
                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):
        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d

def disp_depth(x,fb):
    C = (x>0).float()
    x = x+(1-C)
    x = fb/(x)
    x = x*C
    return x'''

class LSFNet(nn.Module):
    def __init__(self, max_disp, typ='mobilenetv2_100', spixel_branch_channels=[32,48],
                 pretrained=True,
                 corr_volume=True,
                 matching_weighted=False,
                 matching_head=1,
                 gce=True,
                 agg_disp_strides = 2,
                 agg_chns = [16, 32, 48],
                 agg_blocks_num = [2, 2, 2],
                 reg_top_k = 2,
    ):
        super().__init__()
        self.typ = typ
        cfg = ARCH_CONFIGS[typ]
        chns = cfg['channels']
        layers = cfg['layers']

        self.max_disp = max_disp

        # set up the feature extraction first
        self.feature = Feature(typ, chns, layers, pretrained=pretrained)
        self.stereo_up = Stereo_FeatUp(typ, chns)

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

        # monocular
        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        #self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)
        self.mono_feat = Mono_Feat(typ,[16,24,32,96],cenc_layers)

        disp_channel = 1

        self.depth_encoder1 = DepthEncoder(disp_channel, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(disp_channel+1, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(disp_channel+1, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.stereo_uncertainty = nn.Sequential(
            BasicConv(max_disp//4,max_disp//4, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(max_disp//4,max_disp//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(max_disp//4), nn.ReLU()
            )
        self.monocular_uncertainty = nn.Sequential(
            BasicConv(ddcd_layers,ddcd_layers, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ddcd_layers,ddcd_layers, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ddcd_layers), nn.ReLU()
            )
        self.uncertainty = nn.Sequential(
            BasicConv(max_disp//4+ddcd_layers,max_disp//4+ddcd_layers, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(max_disp//4+ddcd_layers,max_disp//4+ddcd_layers, 3, 1, 1, bias=False),
            nn.BatchNorm2d(max_disp//4+ddcd_layers), nn.ReLU()
            )
        self.monocular_mlp = MLP(filter_channels=[ddcd_layers,512,256,128,64,1])
        self.stereo_mlp = MLP(filter_channels=[max_disp//4,512,256,128,64,1])
        self.mlp = MLP(filter_channels=[max_disp//4+ddcd_layers,512,256,128,64,1])
        self.pi_activation = nn.Sigmoid()
        self.sigma_activation = nn.ELU()

    def forward(self,x_l,x_r,disp_l,train=False):
        assert x_l.shape == x_r.shape

        # # Matching comp
        x2,x = self.feature(x_l)
        x2,y = self.feature(x_r)

        x_stereo, y_stereo = self.stereo_up(x,y)

        stem_2x = self.stem_2(x_l)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(x_r)
        stem_4y = self.stem_4(stem_2y)

        x_stereo[0] = torch.cat((x_stereo[0],stem_4x), 1)
        y_stereo[0] = torch.cat((y_stereo[0],stem_4y), 1)


        # Cost volume processing

        if self.corr_volume:
            cost = (self.cost_volume(x_stereo[0],y_stereo[0]))[:, :, :-1]


        cost = self.cost_agg(x_stereo, cost)

        # spixel guide comp
        spx_4x = self.spx_4(x_stereo[0])
        spx_2x = self.spx_2(spx_4x,stem_2x)
        spx_pred = self.spx(spx_2x)
        spx_pred = F.softmax(spx_pred, 1)
        pred = self.regression(cost, spx_pred)
        stereo_disp_pred = pred[0]
        stereo_disp_mid = pred[1:]

        #########################################
        
        C = (disp_l>0).float()

        x_mono = [x2]+x[:3]
        enc_c = self.mono_feat(x_mono)

       ## for the 1/4 res
        disp_14 = 0.25*F.avg_pool2d(disp_l, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(disp_14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[1:4])

        ## for the 1/2 res
        disp_12 = 0.5*F.avg_pool2d(disp_l, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = 2*F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((disp_12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[0:3])

        ## for the 1/1 res
        predict_d11 = 2*F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((disp_l,predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:2])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = 4*F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        mono_disp_pred=output_d11
        mono_disp_mid = [output_d12,output_d14]

        
        cost_feat = torch.squeeze(cost,1)
        stereo_prob = self.stereo_mlp(cost_feat)
        stereo_prob = F.interpolate(stereo_prob,scale_factor=4,mode='bilinear',align_corners=True)
        stereo_prob = self.sigma_activation(stereo_prob)+1+1e-15

        monocular_feat = torch.cat((enc_d11[2],enc_c[1]),1)
        monocular_prob = self.monocular_mlp(monocular_feat)
        monocular_prob = F.interpolate(monocular_prob,scale_factor=4,mode='bilinear',align_corners=True)
        monocular_prob = self.sigma_activation(monocular_prob)+1+1e-15

        cost_feat = self.stereo_uncertainty(cost_feat)
        monocular_feat = self.monocular_uncertainty(monocular_feat)
        uncertainty_feat = torch.cat((cost_feat,monocular_feat),1)
        uncertainty_feat = self.uncertainty(uncertainty_feat)
        pi = self.mlp(uncertainty_feat)
        pi = F.interpolate(pi,scale_factor=4,mode='bilinear',align_corners=True)
        pi = self.pi_activation(pi)

        return {'disp_monocular':mono_disp_pred,
                'prob_monocular':monocular_prob,
                'disp_stereo':stereo_disp_pred,
                'prob_stereo':stereo_prob,
                'pi':pi,
                'disp_stereo_mid':stereo_disp_mid,
                'disp_monocular_mid':mono_disp_mid,
                'disp':stereo_disp_pred*(pi>=0.5)+mono_disp_pred*(pi<0.5),
        }

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers),
                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + scale*F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + scale*F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + scale*F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2

class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  #nn.BatchNorm2d(layers // 2),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):
        if len(pre_cx)==3:
            x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
            x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
            x0 = pre_dx[0] + pre_cx[0]
        else:
            x2 = pre_dx[2] + pre_cx[1]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
            x1 = pre_dx[1] + pre_cx[0]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
            x0 = pre_dx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d

def disp_depth(x,fb):
    C = (x>0).float()
    x = x+(1-C)
    x = fb/(x)
    x = x*C
    return x
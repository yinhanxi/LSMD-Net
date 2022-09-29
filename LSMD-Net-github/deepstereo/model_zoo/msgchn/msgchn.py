import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .feature import RGBEncoder,DepthEncoder,DepthDecoder
from .regression import MLP,upfeat,BasicConv,Conv2x

class MSGCHN(nn.Module):
    def __init__(self,maxdisp):
        super(MSGCHN, self).__init__()

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

        self.uncertainty = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(ddcd_layers, ddcd_layers, 3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(ddcd_layers, ddcd_layers, 3, stride=1, padding=1),
                                  )
        self.mlp = MLP(filter_channels=[ddcd_layers,512,256,128,64,1])
        self.activation = nn.Sigmoid()


    def forward(self, x_l,disp_l,train=True):
        C = (disp_l>0).float()
        enc_c = self.rgb_encoder(x_l)

       ## for the 1/4 res
        disp_d14 = 0.25*F.avg_pool2d(disp_l, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(disp_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        disp_d12 = 0.5*F.avg_pool2d(disp_l, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = 2*F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((disp_d12,predict_d12), 1)
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

        feature = torch.cat((enc_d11[2],enc_c[2]),1)
        #feature = self.uncertainty(feature)
        var = self.mlp(feature)
        var = F.interpolate(var,scale_factor=4,mode='bilinear',align_corners=True)
        prob = self.activation(var)

        disp_preds=[]
        disp_preds.append(output_d11)
        disp_preds.append(output_d12)
        disp_preds.append(output_d14)

        return {'disp':disp_preds,'prob':prob}

'''class Monocular(nn.Module):
    def __init__(self,maxdisp):
        super(Monocular, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.maxdisp = maxdisp

    def forward(self,input_rgb,input_d):
        C = (input_d > 0).float()
        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = 0.25*F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = 0.5*F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)*2
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)*2
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = dcd_d12[3] + predict_d12
        output_d14 = dcd_d14[3]
        return output_d11,output_d12,output_d14'''


'''class MSGCHN(nn.Module):
    def __init__(self,max_disp):
        super(MSGCHN, self).__init__()

        self.max_disp = max_disp

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        disp_channel = 1

        self.depth_encoder1 = DepthEncoder(disp_channel, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(disp_channel+ddcd_layers//2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(disp_channel+ddcd_layers//2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.spx = nn.ConvTranspose2d(2*cenc_layers, 9, kernel_size=4, stride=2, padding=1)
        self.spx_2 = Conv2x(cenc_layers,cenc_layers, True)
        self.spx_4 = nn.Sequential(
            BasicConv(cenc_layers,cenc_layers,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(cenc_layers,cenc_layers,3,1,1,bias=False),
            nn.BatchNorm2d(cenc_layers), nn.ReLU()
            )

        self.mlp = MLP(filter_channels=[denc_layers,1024,512,256,128,2])
        self.activation = nn.Sigmoid()


    def forward(self, x_l,disp_l,train=True):
        C = (disp_l>0).float()
        enc_c = self.rgb_encoder(x_l)

       ## for the 1/4 res
        disp_d14 = 0.25*F.avg_pool2d(disp_l, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(disp_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        disp_d12 = 0.5*F.avg_pool2d(disp_l, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = 2*F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((disp_d12,predict_d12), 1)
        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = 2*F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((disp_l,predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        #spx_4x = self.spx_4(enc_c[2])
        #spx_2x = self.spx_2(spx_4x,enc_c[1])
        #spx = self.spx(spx_2x)
        mlp_pred = self.mlp(enc_d11[2])
        #mlp_pred = self.activation(mlp_pred)
        #pred = (upfeat(mlp_pred[:,:1,:,:],spx,4,4))*self.max_disp
        #prob = (upfeat(mlp_pred[:,1:2,:,:],spx,4,4))
        pred = 4*F.interpolate(mlp_pred[:,:1,:,:], scale_factor=4, mode='bilinear', align_corners=True)
        prob = 4*F.interpolate(mlp_pred[:,1:2,:,:], scale_factor=4, mode='bilinear', align_corners=True)
        #output_d11 = dcd_d11[3] + predict_d11
        #output_d12 = predict_d11
        #output_d14 = 4*F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        #disp_preds=[]
        #disp_preds.append(output_d11)
        #disp_preds.append(output_d12)
        #disp_preds.append(output_d14)

        return {'disp':pred,'prob':prob}'''
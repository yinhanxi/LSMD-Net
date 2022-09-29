import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

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
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)*scale

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)*scale

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)*scale

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

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
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
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
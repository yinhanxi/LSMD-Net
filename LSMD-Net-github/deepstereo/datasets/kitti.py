import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import pylab as plt
import cv2
import torch.nn.functional as F

from .io_ import pfm_imread

class KITTIFusionDataset(Dataset):
    def __init__(self, data_root, lst, pipeline=None):
        self.data_root = data_root
        if isinstance(lst, str):
            if not osp.exists(lst):
                lst = osp.join(data_root, lst)
                assert osp.exists(lst)
            lst = json.load(open(lst))['data']

        self.lst = lst
        self.pipeline = pipeline
        self.mode = 'lr'
        self.FIXED_SHAPE = (256, 1216)
        #self.FIXED_SHAPE = (352, 1216)

    def __len__(self):
        return len(self.lst)

    def load_img(self, p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p).convert('RGB'))

    def load_depth(self, p):
        depth = Image.open(p)
        depth = np.array(depth).astype(np.float32) / 256.0
        return depth

    def depth2disp(self,depth):
        baseline = 0.54
        width_to_focal = dict()
        width_to_focal[1242] = 721.5377
        width_to_focal[1241] = 718.856
        width_to_focal[1224] = 707.0493
        width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
        width_to_focal[1238] = 718.3351

        focal_length = width_to_focal[depth.shape[1]]
        invalid_mask = depth <= 0
        disp = baseline * focal_length / (depth + 1E-8)
        disp[invalid_mask] = 0
        return disp,baseline * focal_length
    
    def crop(self,img):
        h = img.shape[0]
        return img[h-self.FIXED_SHAPE[0]:,:self.FIXED_SHAPE[1]]

    def __getitem__(self, idx):
        filenames = self.lst[idx]
        img_ref = self.load_img(filenames['img_ref'])
        img_tgt = self.load_img(filenames['img_tgt'])
        if 'depth_ref' in filenames:
            depth_ref = self.load_depth(filenames['depth_ref'])
            depth_tgt = self.load_depth(filenames['depth_tgt'])
            disp_ref,bf = self.depth2disp(depth_ref)
            disp_tgt,_ = self.depth2disp(depth_tgt)
        elif 'disp_ref' in filenames:
            disp_ref = self.load_depth(filenames['disp_ref'])
            disp_tgt = self.load_depth(filenames['disp_tgt'])
            depth_ref,bf = self.depth2disp(disp_ref)
            depth_tgt,_ = self.depth2disp(disp_tgt)
        sdepth_ref = self.load_depth(filenames['sdepth_ref'])
        sdepth_tgt = self.load_depth(filenames['sdepth_tgt'])
        disp_input_ref,_ = self.depth2disp(sdepth_ref)
        disp_input_tgt,_ = self.depth2disp(sdepth_tgt)

        img_ref = self.crop(img_ref)
        img_tgt = self.crop(img_tgt)
        disp_ref = self.crop(disp_ref)
        disp_tgt = self.crop(disp_tgt)
        disp_input_ref = self.crop(disp_input_ref)
        disp_input_tgt = self.crop(disp_input_tgt)
        depth_ref = self.crop(depth_ref)
        sdepth_ref = self.crop(sdepth_ref)
        depth_tgt = self.crop(depth_tgt)
        sdepth_tgt = self.crop(sdepth_tgt)


        d = {
            'mode': self.mode,
            'filenames': filenames,
            'origin': {
                'img_ref': img_ref,
                'img_tgt': img_tgt,
                'disp_ref': disp_ref,
                'disp_tgt': disp_tgt,
            },
            'img_ref': img_ref,
            'img_tgt': img_tgt,
            'disp_ref': disp_ref,
            'disp_tgt': disp_tgt,
            'disp_input_ref':disp_input_ref,
            'disp_input_tgt':disp_input_tgt,
            'bf':bf,
            'depth_ref':depth_ref,
            'depth_input_ref':sdepth_ref,
            'depth_tgt':depth_tgt,
            'depth_input_tgt':sdepth_tgt,
        }

        if self.pipeline:
            d = self.pipeline(d)

        return d

class KITTIStereoDataset(Dataset):
    def __init__(self, data_root, lst, pipeline=None):
        self.data_root = data_root
        if isinstance(lst, str):
            if not osp.exists(lst):
                lst = osp.join(data_root, lst)
                assert osp.exists(lst)
            lst = json.load(open(lst))['data']

        self.lst = lst
        self.pipeline = pipeline
        self.mode = 'lr'
        self.FIXED_SHAPE = (256, 1216)

    def __len__(self):
        return len(self.lst)

    def load_img(self, p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p).convert('RGB'))

    def load_depth(self, p):
        depth = Image.open(p)
        depth = np.array(depth).astype(np.float32) / 256.0
        return depth

    def depth2disp(self,depth):
        baseline = 0.54
        width_to_focal = dict()
        width_to_focal[1242] = 721.5377
        width_to_focal[1241] = 718.856
        width_to_focal[1224] = 707.0493
        width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
        width_to_focal[1238] = 718.3351

        focal_length = width_to_focal[depth.shape[1]]
        invalid_mask = depth <= 0
        disp = baseline * focal_length / (depth + 1E-8)
        disp[invalid_mask] = 0
        return disp,baseline * focal_length
    
    def crop(self,img):
        h = img.shape[0]
        return img[h-self.FIXED_SHAPE[0]:,:self.FIXED_SHAPE[1]]

    def __getitem__(self, idx):
        filenames = self.lst[idx]
        img_ref = self.load_img(filenames['img_ref'])
        img_tgt = self.load_img(filenames['img_tgt'])
        disp_ref = self.load_depth(filenames['disp_ref'])
        disp_tgt = self.load_depth(filenames['disp_tgt'])
        sdepth_ref = self.load_depth(filenames['sdepth_ref'])
        sdepth_tgt = self.load_depth(filenames['sdepth_tgt'])
        disp_input_ref,bf = self.depth2disp(sdepth_ref)
        disp_input_tgt,_ = self.depth2disp(sdepth_tgt)
        depth_ref,_ = self.depth2disp(disp_ref)
        depth_tgt,_ = self.depth2disp(disp_tgt)

        img_ref = self.crop(img_ref)
        img_tgt = self.crop(img_tgt)
        disp_ref = self.crop(disp_ref)
        disp_tgt = self.crop(disp_tgt)
        disp_input_ref = self.crop(disp_input_ref)
        disp_input_tgt = self.crop(disp_input_tgt)
        depth_ref = self.crop(depth_ref)
        sdepth_ref = self.crop(sdepth_ref)
        depth_tgt = self.crop(depth_tgt)
        sdepth_tgt = self.crop(sdepth_tgt)

        d = {
            'mode': self.mode,
            'filenames': filenames,
            'origin': {
                'img_ref': img_ref,
                'img_tgt': img_tgt,
                'disp_ref': disp_ref,
                'disp_tgt': disp_tgt,
            },
            'img_ref': img_ref,
            'img_tgt': img_tgt,
            'disp_ref': disp_ref,
            'disp_tgt': disp_tgt,
            'disp_input_ref':disp_input_ref,
            'disp_input_tgt':disp_input_tgt,
            'bf':bf,
            'depth_ref':depth_ref,
            'depth_input_ref':sdepth_ref,
            'depth_tgt':depth_tgt,
            'depth_input_tgt':sdepth_tgt,
        }

        if self.pipeline:
            d = self.pipeline(d)

        return d
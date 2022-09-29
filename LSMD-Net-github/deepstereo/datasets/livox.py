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

class LivoxStereoDataset(Dataset):
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
        #self.FIXED_SHAPE = (924, 1024)
        self.FIXED_SHAPE = (1024, 1024)

    def __len__(self):
        return len(self.lst)

    def load_img(self, p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p).convert('RGB').transpose(Image.ROTATE_270))

    def load_disp(self, p):
        #p = osp.join(self.data_root, p)
        #p=p.replace('disp','depth')
        data = np.load(p)
        data = cv2.transpose(cv2.flip(data,0))
        data = np.ascontiguousarray(data, dtype=np.float32)

        return data
    
    def crop(self,img):
        #w = img.shape[0]
        return img[:self.FIXED_SHAPE[0],:self.FIXED_SHAPE[1]]

    def __getitem__(self, idx):
        filenames = self.lst[idx]
        img_ref = self.load_img(filenames['img_ref'])
        img_tgt = self.load_img(filenames['img_tgt'])
        disp_ref = self.load_disp(filenames['disp_ref_6s'])
        disp_tgt = self.load_disp(filenames['disp_tgt_6s'])
        disp_input_ref = self.load_disp(filenames['disp_ref_300ms'])
        disp_input_tgt = self.load_disp(filenames['disp_tgt_300ms'])
        depth_ref = self.load_disp(filenames['depth_ref_6s'])/1000
        depth_tgt = self.load_disp(filenames['depth_tgt_6s'])/1000
        depth_input_ref = self.load_disp(filenames['depth_ref_300ms'])/1000
        depth_input_tgt = self.load_disp(filenames['depth_tgt_300ms'])/1000

        if 'outdoor' in filenames['img_ref']:
            bf = 223253.57128808307/1000
        elif 'indoor' in filenames['img_ref']:
            bf = 218749.33978274805/1000
        else:
            raise

        img_ref = self.crop(img_ref)
        img_tgt = self.crop(img_tgt)
        disp_ref = self.crop(disp_ref)
        disp_tgt = self.crop(disp_tgt)
        disp_input_ref = self.crop(disp_input_ref)
        disp_input_tgt = self.crop(disp_input_tgt)
        depth_ref = self.crop(depth_ref)
        depth_tgt = self.crop(depth_tgt)
        depth_input_ref = self.crop(depth_input_ref)
        depth_input_tgt = self.crop(depth_input_tgt)


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
            'depth_ref': depth_ref,
            'depth_tgt': depth_tgt,
            'depth_input_ref':depth_input_ref,
            'depth_input_tgt':depth_input_tgt,
            'bf':bf
        }

        if self.pipeline:
            d = self.pipeline(d)

        return d

class LivoxMonocularDataset(Dataset):
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

    def __len__(self):
        return len(self.lst)

    def load_img(self, p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p).convert('RGB'))

    def load_disp(self, p):
        #p = osp.join(self.data_root, p)
        #p=p.replace('disp','depth')
        data = np.load(p)
        #data = cv2.transpose(cv2.flip(data,0))
        data = np.ascontiguousarray(data, dtype=np.float32)

        return data

    def __getitem__(self, idx):
        filenames = self.lst[idx]
        img_ref = self.load_img(filenames['img_ref'])
        depth_ref = self.load_disp(filenames['depth_ref_6s'])
        depth_input = self.load_disp(filenames['depth_ref_300ms'])

        d = {
            'mode': self.mode,
            'filenames': filenames,
            'origin': {
                'img': img_ref,
                'depth': depth_ref,
            },
            'img': img_ref,
            'depth': depth_ref,
            'depth_input':depth_input,
        }

        if self.pipeline:
            d = self.pipeline(d)

        return d
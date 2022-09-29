import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import pylab as plt
import cv2

from .io_ import pfm_imread

class DrivingStereoDataset(Dataset):
    def __init__(self, data_root, lst, pipeline=None):
        self.data_root = data_root
        if isinstance(lst, str):
            if not osp.exists(lst):
                lst = osp.join(data_root, lst)
                assert osp.exists(lst)
            lst = json.load(open(lst))['data']

        self.lst = lst[:1000]
        self.pipeline = pipeline
        self.mode = 'lr'

    def __len__(self):
        return len(self.lst)

    def load_img(self, p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p).convert('RGB'))

    def load_disp(self, p):
        #p = osp.join(self.data_root, p)
        #data = np.array(Image.open(p))
        data = cv2.imread(p)[:,:,0]
        data = np.ascontiguousarray(data, dtype=np.float32)

        return data

    def __getitem__(self, idx):
        filenames = self.lst[idx]
        img_ref = self.load_img(filenames['img_ref'])[:400,:879,:]
        img_tgt = self.load_img(filenames['img_tgt'])[:400,:879,:]
        disp_ref = self.load_disp(filenames['disp_ref'])[:400,:879]

        d = {
            'mode': self.mode,
            'filenames': filenames,
            'origin': {
                'img_ref': img_ref,
                'img_tgt': img_tgt,
                'disp_ref': disp_ref,
            },
            'img_ref': img_ref,
            'img_tgt': img_tgt,
            'disp_ref': disp_ref,
        }

        if self.pipeline:
            d = self.pipeline(d)

        return d
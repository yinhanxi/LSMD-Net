import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from scipy.interpolate import griddata

from .io_ import pfm_imread

class SceneFlowStereoDataset(Dataset):
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
        data, scale = pfm_imread(p)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def load_data(self, p):
        #p = osp.join(self.data_root, p)
        return np.load(p)

    def load_mask(self,p):
        #p = osp.join(self.data_root, p)
        return np.array(Image.open(p))[:,:,0]

    def load_txt(self,camera,img):
        with open(camera,'r') as f:
            lines = f.readlines()
        num = int(img[-8:-5])
        assert lines[(num-1)*4].strip('\n')==f'Frame {num}'
        l = float(lines[(num-1)*4+1].strip('\n').split()[8])
        r = float(lines[(num-1)*4+2].strip('\n').split()[8])
        b = abs(l-r)
        f = 1050.0
        return [f*b]
        



    def __getitem__(self, idx):
        fnames = self.lst[idx]
        img_ref = self.load_img(fnames['img_ref'])
        img_tgt = self.load_img(fnames['img_tgt'])
        disp_ref = self.load_disp(fnames['disp_ref'])*self.load_mask(fnames['lidar_mask_train'])
        disp_tgt = self.load_disp(fnames['disp_tgt'])
        #lidar_mask_train = self.load_mask(fnames['lidar_mask_train'])
        disp_input = self.load_mask(fnames['lidar_mask_input'])*disp_ref
        #fb = self.load_txt(fnames['camera_data'],fnames['img_ref'])
        

        d = {
            'mode': self.mode,
            'filenames': fnames,
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
            'disp_input':disp_input,
            'idx':idx,
        }

        for key in ['slanted_plane_ref', 'slanted_plane_tgt']:
            if key in fnames:
                arr = np.dstack([self.load_data(i) for i in fnames[key]])
                d['origin'][key] = arr
                d[key] = arr

        if self.pipeline:
            d = self.pipeline(d)

        return d

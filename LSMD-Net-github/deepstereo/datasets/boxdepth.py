import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from scipy.interpolate import griddata

from .io_ import pfm_imread

class BoxDepthMonocularDataset(Dataset):
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

    def load_depth(self, p):
        #p = osp.join(self.data_root, p)
        data = np.load(p)
        return data

    def __getitem__(self, idx):
        fnames = self.lst[idx]
        img = self.load_img(fnames['img'])
        depth = self.load_depth(fnames['depth'])
        
        d = {
            'mode': self.mode,
            'filenames': fnames,
            'origin': {
                'img': img,
                'depth': depth,
            },
            'img': img,
            'depth': depth,
        }
        if self.pipeline:
            d = self.pipeline(d)

        return d

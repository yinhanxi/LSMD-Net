#!/usr/bin/env python

# for opencv imshow
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import os
import os.path as osp
import sys
cur_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, '..'))

from torch.utils.data import DataLoader
import random
import numpy as np
import numpy.linalg as npl
import cv2
from skimage.io import imread
from deepstereo.base import import_data_module

from tqdm import tqdm
from pprint import pprint
import pylab as plt
import click

def get_sp_normal(coeff):
    nz = np.zeros_like(coeff[...,1])
    normals = np.dstack((coeff[...,1], coeff[...,2], nz))
    den = npl.norm(normals, axis=-1, keepdims=True)
    normals /= den
    return normals

@click.command()
@click.option('--data_root', default='')
@click.option('--tp', '-t', default='train')
@click.option('--ignore_only', is_flag=True)
@click.option('--no_show', is_flag=True)
@click.option('--verbose', '-v', count=True)
@click.option('--proc_f', default='')
@click.option('-seed', default=7)
@click.argument('data_module')
def main(data_root, tp, ignore_only, no_show, verbose, proc_f, seed, data_module):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

    if not data_root:
        data_root = osp.join(cur_dir, '../data')

    data_module = import_data_module(data_module, data_root, batch_size=1, to_tensor=False)
    data_module.setup()
    loader = data_module.get_dataloader(tp)

    proc_fn = None
    if proc_f:
        m, key = import_module_and_key(proc_f)
        proc_fn = m.PROCESSORS.get(key, 'default')

    for item in tqdm(loader):
        pprint (item['filenames'])
        if proc_fn:
            item = proc_fn(item)

        is_lr = item['mode'][0] == 'lr'


        if is_lr:
            fig, axs = plt.subplots(2,3, figsize=(16, 12))

            col = 0

            img_ref = item['img_ref'][0]
            axs[0, col].title.set_text("img_ref")
            axs[0, col].imshow(img_ref)
            axs[0, col].xaxis.set_ticks([])
            axs[0, col].yaxis.set_ticks([])

            img_tgt = item['img_tgt'][0]
            axs[1, col].title.set_text("img_tgt")
            axs[1, col].imshow(img_tgt)
            axs[1, col].xaxis.set_ticks([])
            axs[1, col].yaxis.set_ticks([])

            col = 1

            depth_ref = item['disp_ref'][0]
            axs[0, col].title.set_text("disp_ref")
            axs[0, col].imshow(depth_ref)
            axs[0, col].xaxis.set_ticks([])
            axs[0, col].yaxis.set_ticks([])

            depth_input_ref = item['disp_input_ref'][0]
            axs[1, col].title.set_text("disp_input_ref")
            axs[1, col].imshow(depth_input_ref)
            axs[1, col].xaxis.set_ticks([])
            axs[1, col].yaxis.set_ticks([])

            col = 2

            disp_ref = item['depth_ref'][0]
            axs[0, col].title.set_text("depth_ref")
            axs[0, col].imshow(disp_ref)
            axs[0, col].xaxis.set_ticks([])
            axs[0, col].yaxis.set_ticks([])

            disp_input = item['depth_input_ref'][0]
            axs[1, col].title.set_text("depth_input_ref")
            axs[1, col].imshow(disp_input)
            axs[1, col].xaxis.set_ticks([])
            axs[1, col].yaxis.set_ticks([])

            '''if 'disp_tgt' in item:
                disp_tgt = item['disp_tgt'][0]
                axs[1, col].title.set_text("disp_tgt")
                axs[1, col].imshow(disp_tgt)

                if 'lidar_disp' in item:
                    lidar_disp = item['lidar_disp'][0]
                    axs[2, 0].title.set_text("lidar_disp")
                    axs[2, 0].imshow(lidar_disp)
                    lidar_mask = item['lidar_mask'][0]
                    axs[2, col].title.set_text("lidar_mask")
                    axs[2, col].imshow(lidar_mask)'''
            plt.show()


if __name__ == '__main__':
    main()
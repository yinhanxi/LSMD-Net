#!/usr/bin/env python
from operator import imod
import os
import os.path as osp
import sys
import albumentations as A
import scipy.io as io
import skimage
import skimage.io

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from deepstereo.base import import_model
from deepstereo.lightning import find_latest_checkpoint
from deepstereo.datasets.sceneflow import SceneFlowStereoDataset
from deepstereo.datasets.drivingstereo import DrivingStereoDataset
from deepstereo.datasets.livox import LivoxStereoDataset,LivoxMonocularDataset
from deepstereo.datasets.boxdepth import BoxDepthMonocularDataset
from deepstereo.datasets.kitti import KITTIStereoDataset,KITTIFusionDataset
from deepstereo.datasets.transform import make_pipeline_stereo_test,make_pipeline_monocular_test
from samples.data_module import make_pipeline_stereo_train
from pprint import pprint
from loguru import logger
import pylab as plt
import json
import click
from deepstereo.metric import calc_endpoint_error,calc_D1_metric,calc_D1_metric_with_thres,mae,imae,rmse,irmse,MAE,iMAE,iRMSE,RMSE,err_1px,err_2px,err_3px,differential_entropy
from deepstereo.loss import warp_disp,calc_losses_disp_unsup,calc_loss_disp_photometric
import time
import cv2
import torch.nn.functional as F
from kornia.losses import ssim_loss, inverse_depth_smoothness_loss
from einops import repeat

@click.command()
@click.option('--gpu', default=3)
@click.option('--batch_size', default=1)
@click.option('--worker_nr', default=4)
@click.option('--dataset', '--ds', default='')
@click.option('--crop_w', default=768)
@click.option('--crop_h', default=384)
@click.option('--debug', is_flag=True)
@click.option('--shuffle', is_flag=True)
@click.option('--no_disp_tgt', is_flag=False)
@click.argument('model')
@click.argument('checkpoint')
@click.argument('data_list')
def main(gpu, batch_size, worker_nr, dataset, crop_w, crop_h,
         debug, shuffle, no_disp_tgt, model, checkpoint, data_list):

    if osp.isdir(checkpoint) and osp.exists(osp.join(checkpoint, 'lightning_logs')):
        checkpoint = find_latest_checkpoint(osp.join(checkpoint, 'lightning_logs'))

    print ("loading checkpoint", checkpoint)
    cfg = import_model(model)
    if batch_size > 0:
        cfg.batch_size = batch_size

    model = cfg.model_class(**cfg.model_kws)
    weights = torch.load(checkpoint, map_location=f'cuda:{gpu}')['state_dict']
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(weights)
    model.load_state_dict(weights)
    model.to(gpu)
    model.eval()


    d = json.load(open(data_list))
    dataset = d['dataset']
    lst = d['data']

    if dataset == 'stereo_livox':
        ds = LivoxStereoDataset(
            osp.dirname(data_list),
            lst,
            pipeline=make_pipeline_stereo_test((1024,1024),with_disp_tgt=False,with_lidar_disp=True,with_lidar_depth=True),
        )
    elif dataset == 'monocular_livox':
        ds = LivoxMonocularDataset(
            osp.dirname(data_list),
            lst,
            pipeline=make_pipeline_monocular_test((896,896),with_lidar_depth=True),
        )
    elif dataset == 'stereo_kitti':
        ds = KITTIStereoDataset(
            osp.dirname(data_list),
            lst,
            pipeline=make_pipeline_stereo_test((1216,256),with_disp_tgt=False,with_lidar_disp=True,with_lidar_depth=True),
        )
    elif dataset == 'fusion_kitti':
        ds = KITTIFusionDataset(
            osp.dirname(data_list),
            lst,
            pipeline=make_pipeline_stereo_test((1024,256),with_disp_tgt=False,with_lidar_disp=True,with_lidar_depth=True),
        )
    else:
        raise

    dl = DataLoader(ds, shuffle=shuffle)


    sum_epe = 0
    sum_epe_mid = 0
    sum_epe_stereo = 0
    sum_D1 = 0
    sum_1px=0
    sum_2px=0
    sum_3px=0
    sum_imae = 0
    sum_mae = 0
    sum_irmse = 0
    sum_rmse = 0
    sum_time = 0
    num = 0
    
    crop = A.CenterCrop(256,1024)
    for bi, batch in enumerate(dl):
        if True:
            for k in ['img_ref', 'img_tgt','disp_ref','disp_input_ref','disp_input_tgt','depth_ref','depth_input_ref','bf']:
                batch[k] = batch[k].to(gpu)
            with torch.no_grad():
                start = time.time()
                #pred_dic = model(batch['img_ref'],batch['img_tgt'], batch['disp_input_ref'][:, None, ...],batch['disp_input_tgt'][:, None, ...],bi)
                pred_dic = model(batch['img_ref'],batch['img_tgt'], batch['disp_input_ref'][:, None, ...],bi)
                #pred_dic = model(batch['img_ref'],batch['img_tgt'], batch['depth_input_ref'][:, None, ...],batch['bf'],bi)
                #pred_dic = model(batch['img_ref'],batch['img_tgt'],bi)
                #pred_dic = model(batch['img_ref'], batch['depth_input_ref'][:, None, ...],bi)
                sum_time = sum_time+time.time()-start

            
            v_img_ref = crop(image=batch['origin']['img_ref'][0])['image'].cpu().numpy()
            v_img_tgt = crop(image=batch['origin']['img_tgt'][0])['image'].cpu().numpy()
            depth_gt = batch['depth_ref'][:, None, ...].cpu()
            disp_gt = batch['disp_ref'][:, None, ...].cpu()
            depth_input = batch['depth_input_ref'][:, None, ...].cpu()
            disp_input = batch['disp_input_ref'][:, None, ...].cpu()


            #pred_disp = pred_dic['disp'][0].cpu()
            pred_disp = pred_dic['disp'].cpu()
            pred_prob = pred_dic['pi'].cpu()
            pred_depth = disp2depth(pred_disp,batch['bf'])


            mask = (disp_gt>0)*(disp_gt<192)*(depth_gt>1e-3)*(pred_depth>1e-3)
            sum_epe += calc_endpoint_error(pred_disp,disp_gt,mask)

            pred_disp = pred_disp.numpy()
            disp_gt = disp_gt.numpy()
            pred_depth = pred_depth.numpy()
            depth_gt = depth_gt.numpy()
            valid_mask = (disp_gt>0)*(disp_gt<192)*(depth_gt>1e-3)*(pred_depth>1e-3)

            pred_disp = pred_disp[valid_mask]
            disp_gt = disp_gt[valid_mask]

            sum_1px += err_1px(pred_disp,disp_gt)
            sum_2px += err_2px(pred_disp,disp_gt)
            sum_3px += err_3px(pred_disp,disp_gt)

            pred_depth = pred_depth[valid_mask]
            depth_gt = depth_gt[valid_mask]

            sum_imae += imae(pred_depth,depth_gt)
            sum_mae += mae(pred_depth,depth_gt)
            sum_irmse += irmse(pred_depth,depth_gt)
            sum_rmse += rmse(pred_depth,depth_gt)

        num+=1
        if num%100==0:
            print(num)
    print(len(dl))
    print(sum_time/num)
    print("avg-epe:",sum_epe/len(dl))
    print("avg-1px:",sum_1px/len(dl))
    print("avg-2px:",sum_2px/len(dl))
    print("avg-3px:",sum_3px/len(dl))
    print("imae:",sum_imae/len(dl))
    print("mae:",sum_mae/len(dl))
    print("irmse:",sum_irmse/len(dl))
    print("rmse:",sum_rmse/len(dl))
    print("time-per-picture:",sum_time/len(dl))

def disp2depth(disp,bf):
        _,c,h,w=disp.shape
        bf = repeat(bf.type_as(disp), 'b -> b c h w',c=c,h=h,w=w)
        invalid_mask = disp <= 0
        depth = bf/ (disp + 1E-8)
        depth[invalid_mask] = 0
        return depth

if __name__ == '__main__':
    main()

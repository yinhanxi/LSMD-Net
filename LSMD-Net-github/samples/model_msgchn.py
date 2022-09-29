import os
import os.path as osp
import sys
import random
from deepstereo.model_zoo.lsfnet.util_conv import InvertedResidual3d

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))
sys.path.insert(0, cur_dir)

from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from deepstereo.model_zoo.msgchn import MSGCHN

from deepstereo.loss import calc_losses_disp_sup, calc_losses_disp_unsup,unimodal_loss
from deepstereo.metric import calc_endpoint_error,iMAE,MAE,RMSE,iRMSE
from deepstereo.visualize import TrainerVisdom
from data_module import DATA_MODULES
from visdom import Visdom
from einops import repeat

def inverse_depth(depth):
    invalid_mask = depth <= 0
    inv_depth = 700/(depth + 1E-8)
    inv_depth[invalid_mask] = 0
    return inv_depth

def disp2depth(disp,bf):
        _,c,h,w=disp.shape
        bf = repeat(bf.type_as(disp), 'b -> b c h w',c=c,h=h,w=w)
        invalid_mask = disp <= 0
        depth = bf/ (disp + 1E-8)
        depth[invalid_mask] = 0
        return depth

class MSGCHNModel(pl.LightningModule):
    def __init__(self, max_disp, optim_kws={}, viz_train=False):
        super().__init__()

        self.name = 'msgchn'

        self.net = MSGCHN(max_disp)

        self.optim_kws = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'weight_decay': 2e-4,
            **optim_kws,
        }
        self.max_disp = max_disp
        self.viz_train = viz_train

    def setup(self, stage):
        tp = stage if self.viz_train else 'none'
        self.viz = TrainerVisdom(tp, env=self.name)

    def forward(self,img_l,lidar_depth_l,batch_idx=0):
        #inv_depth_input = inverse_depth(lidar_depth_l)
        preds = self.net(img_l,lidar_depth_l)
        return preds

    def training_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        depth_gt = batch['depth_ref'][:, None, ...]
        #depth_input = batch['depth_input_ref'][:, None, ...]
        disp_gt = batch['disp_ref'][:, None, ...]
        disp_input = batch['disp_input_ref'][:, None, ...]

        #mask = (disp_gt<self.max_disp)*(disp_gt>0)
        #inv_depth_input = inverse_depth(depth_input)
        #inv_depth_gt = inverse_depth(depth_gt)

        pred_dic = self.net(img_l,disp_input,train=True)

        mask = disp_gt>0
        if self.current_epoch<6:
            loss=calc_losses_disp_sup(pred_dic['disp'][1:],disp_gt,mask,weights=[1.0,1.0])\
                + unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        elif self.current_epoch<11:
            loss=calc_losses_disp_sup(pred_dic['disp'][1:],disp_gt,mask,weights=[0.3,0.15])\
                + unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        else:
            loss=unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        #loss = unimodal_loss(pred_dic['disp'],pred_dic['prob'],disp_gt,mask)
        #loss=calc_losses_disp_sup(pred_dic['disp'],disp_gt,mask,weights=[1.0,0.3,0.15])
        #pred_depth = inverse_depth(pred_dic['disp'][0])
        pred_depth = disp2depth(pred_dic['disp'][0],batch['bf'])

        #epe = calc_endpoint_error(pred_dic['disp'][0].detach(),disp_gt,mask)
        imae = iMAE(pred_depth,depth_gt).detach()
        mae = MAE(pred_depth,depth_gt).detach()
        rmse = RMSE(pred_depth,depth_gt).detach()
        irmse = iRMSE(pred_depth,depth_gt).detach()

        return {
            'loss': loss,
            'iMAE': imae,
            'MAE': mae,
            'RMSE': rmse,
            'iRMSE': irmse,
        }

    def validation_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        depth_gt = batch['depth_ref'][:, None, ...]
        #depth_input = batch['depth_input_ref'][:, None, ...]
        disp_gt = batch['disp_ref'][:, None, ...]
        disp_input = batch['disp_input_ref'][:, None, ...]

        #mask = (disp_gt<self.max_disp)*(disp_gt>0)
        #inv_depth_input = inverse_depth(depth_input)
        #inv_depth_gt = inverse_depth(depth_gt)

        pred_dic = self.net(img_l,disp_input,train=True)

        mask = disp_gt>0
        if self.current_epoch<6:
            loss=calc_losses_disp_sup(pred_dic['disp'][1:],disp_gt,mask,weights=[1.0,1.0])\
                + unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        elif self.current_epoch<11:
            loss=calc_losses_disp_sup(pred_dic['disp'][1:],disp_gt,mask,weights=[0.3,0.15])\
                + unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        else:
            loss=unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask,dist='laplacian')
        #loss = unimodal_loss(pred_dic['disp'],pred_dic['prob'],disp_gt,mask)
        #loss=calc_losses_disp_sup(pred_dic['disp'],disp_gt,mask,weights=[1.0,0.3,0.15])
        #pred_depth = inverse_depth(pred_dic['disp'][0])
        pred_depth = disp2depth(pred_dic['disp'][0],batch['bf'])

        #epe = calc_endpoint_error(pred_dic['disp'][0].detach(),disp_gt,mask)
        imae = iMAE(pred_depth,depth_gt).detach()
        mae = MAE(pred_depth,depth_gt).detach()
        rmse = RMSE(pred_depth,depth_gt).detach()
        irmse = iRMSE(pred_depth,depth_gt).detach()

        return {
            'loss': loss,
            'iMAE': imae,
            'MAE': mae,
            'RMSE': rmse,
            'iRMSE': irmse,
        }

    def run_epoch_end(self, stage, outputs):
        loss = torch.stack([i['loss'] for i in outputs]).mean()
        imae = torch.stack([i['iMAE'] for i in outputs]).mean()
        mae = torch.stack([i['MAE'] for i in outputs]).mean()
        rmse = torch.stack([i['RMSE'] for i in outputs]).mean()
        irmse = torch.stack([i['iRMSE'] for i in outputs]).mean()

        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_MAE', mae, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_RMSE', rmse, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_iMAE', imae, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_iRMSE', irmse, prog_bar=True, logger=True, sync_dist=True)

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                f'{stage}/avg_loss_on_epoch', loss,
                global_step=self.current_epoch)

    def training_epoch_end(self, outputs):
        return self.run_epoch_end('train', outputs)

    def validation_epoch_end(self, outputs):
        return self.run_epoch_end('val', outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optim_kws)

MODELS = edict({
    'msgchn_192': {
        'model_class': MSGCHNModel,
        'model_kws': {
            'max_disp': 192,#ADAM 100 SceneFlow 192
            'optim_kws': {
                # 'lr': 1e-3,
                'lr': 1e-3,
            },
            'viz_train': False,
        },
        'batch_size': 12,
        'data_modules': {
            'stereo_sceneflow_driving': DATA_MODULES['stereo_sceneflow_driving'],
            'stereo_drivingstereo': DATA_MODULES['stereo_drivingstereo'],
            'stereo_livox': DATA_MODULES['stereo_livox'],
            'fusion_kitti': DATA_MODULES['fusion_kitti'],
        },
        'trainer_kws': {
            'max_epochs': 30,
            'callbacks': [
                EarlyStopping('val_loss', mode='min', patience=30),
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='epoch_{epoch:02d}-EPE_{epe:.2f}',
                    verbose=True,
                    auto_insert_metric_name=False,
                ),
            ],
            # 'profiler': 'advanced',
        },
    },
})

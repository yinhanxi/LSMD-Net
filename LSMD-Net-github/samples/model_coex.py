import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))
sys.path.insert(0, cur_dir)

from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from deepstereo.model_zoo.coex import CoEx
import pylab as plt

from deepstereo.loss import calc_losses_disp_sup, calc_losses_disp_unsup,unimodal_loss,bimodal_loss
from deepstereo.metric import calc_endpoint_error,calc_D1_metric_with_thres
from deepstereo.visualize import TrainerVisdom
from data_module import DATA_MODULES
from visdom import Visdom
from einops import repeat

class CoExModel(pl.LightningModule):
    def __init__(self, max_disp, optim_kws={}, viz_train=False):
        super().__init__()

        self.name = 'coex'

        self.loss_type = 'L1'

        self.net = CoEx(max_disp=max_disp,loss_type=self.loss_type)

        self.optim_kws = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'weight_decay': 1e-4,
            **optim_kws,
        }
        self.max_disp = max_disp
        self.viz_train = viz_train

    def setup(self, stage):
        tp = stage if self.viz_train else 'none'
        self.viz = TrainerVisdom(tp, env=self.name)

    def forward(self,img_l,img_r,batch_idx=0):
        preds = self.net(img_l,img_r)
        return preds

    def training_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        img_r = batch['img_tgt']
        disp_gt = batch['disp_ref'][:, None, ...]
        #lidar_mask_train = batch['lidar_mask_train'][:, None, ...].bool()
        #disp_input = batch['disp_input'][:, None, ...].bool()


        mask = (disp_gt<self.max_disp)*(disp_gt>0)
        pred_dic = self.net(img_l, img_r,train=True)

        if self.loss_type == 'L1':
            loss = calc_losses_disp_sup(pred_dic['disp'],disp_gt,mask,weights=[1.0,0.3])
            epe = calc_endpoint_error(pred_dic['disp'][0].detach(), disp_gt, mask)
        elif self.loss_type == 'unimodal':
            loss = unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask)\
                +calc_losses_disp_sup([pred_dic['disp'][1]],disp_gt,mask,weights=[0.3])
            epe = calc_endpoint_error(pred_dic['disp'][0].detach(), disp_gt, mask)

        px_1 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,1).detach()
        px_2 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,2).detach()
        px_3 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,3).detach()
        return {
            'loss': loss,
            'EPE': epe,
            "3px":px_3
        }

    def validation_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        img_r = batch['img_tgt']
        disp_gt = batch['disp_ref'][:, None, ...]
        #lidar_mask_train = batch['lidar_mask_train'][:, None, ...].bool()
        #lidar_mask_input = batch['lidar_mask_input'][:, None, ...].bool()


        mask = (disp_gt<self.max_disp)*(disp_gt>0)
        pred_dic = self.net(img_l, img_r,train=True)

        if self.loss_type == 'L1':
            loss = calc_losses_disp_sup(pred_dic['disp'],disp_gt,mask,weights=[1.0,0.3])
            epe = calc_endpoint_error(pred_dic['disp'][0].detach(), disp_gt, mask)
        elif self.loss_type == 'unimodal':
            loss = unimodal_loss(pred_dic['disp'][0],pred_dic['prob'],disp_gt,mask)\
                +calc_losses_disp_sup([pred_dic['disp'][1]],disp_gt,mask,weights=[0.3])
            epe = calc_endpoint_error(pred_dic['disp'][0].detach(), disp_gt, mask)
        
        px_1 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,1).detach()
        px_2 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,2).detach()
        px_3 = calc_D1_metric_with_thres(pred_dic['disp'][0], disp_gt, mask,3).detach()

        return {
            'loss': loss,
            'EPE': epe,
            "3px":px_3
        }

    def run_epoch_end(self, stage, outputs):
        loss = torch.stack([i['loss'] for i in outputs]).mean()
        epe = torch.stack([i['EPE'] for i in outputs]).mean()
        px_3 = torch.stack([i['3px'] for i in outputs]).mean()

        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_epe', epe, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{stage}_3px', px_3, prog_bar=True, logger=True, sync_dist=True)

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
    'coex_192': {
        'model_class': CoExModel,
        'model_kws': {
            'max_disp': 192,#ADAM 100 SceneFlow 192
            'optim_kws': {
                # 'lr': 1e-3,
                'lr': 1e-3,
            },
            'viz_train': False,
        },
        'batch_size': 6,
        'data_modules': {
            'stereo_sceneflow_driving': DATA_MODULES['stereo_sceneflow_driving'],
            'stereo_drivingstereo': DATA_MODULES['stereo_drivingstereo'],
            'stereo_livox': DATA_MODULES['stereo_livox'],
            'fusion_kitti': DATA_MODULES['fusion_kitti'],
        },
        'trainer_kws': {
            'max_epochs': 200,
            'callbacks': [
                EarlyStopping('val_epe', mode='min', patience=200),
                ModelCheckpoint(
                    monitor='val_epe',
                    filename='epoch_{epoch:02d}-EPE_{epe:.2f}',
                    verbose=True,
                    auto_insert_metric_name=False,
                ),
            ],
            # 'profiler': 'advanced',
        },
    },
})

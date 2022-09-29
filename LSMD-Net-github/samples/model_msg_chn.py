import os
import os.path as osp
import sys
import random

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))
sys.path.insert(0, cur_dir)

from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from deepstereo.model_zoo.msg_chn import MSG_CHN

from deepstereo.loss import calc_losses_disp_sup, calc_losses_disp_unsup,mseloss,unimodal_loss
from deepstereo.metric import calc_endpoint_error,iMAE,MAE,RMSE,iRMSE
from deepstereo.visualize import TrainerVisdom
from data_module import DATA_MODULES
from visdom import Visdom
from einops import repeat

def depth2disp(depth,bf):
        _,c,h,w=depth.shape
        bf = repeat(bf.type_as(depth), 'b -> b c h w',c=c,h=h,w=w)
        invalid_mask = depth <= 0
        disp = bf/ (depth + 1E-8)
        disp[invalid_mask] = 0
        return disp

class MSG_CHNModel(pl.LightningModule):
    def __init__(self, max_disp, optim_kws={}, viz_train=False):
        super().__init__()

        self.name = 'msg_chn'

        self.net = MSG_CHN()

        self.optim_kws = {
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
        preds = self.net(img_l,lidar_depth_l)
        return preds

    def training_step(self, batch, batch_idx):
        img = batch['img_ref']
        depth_gt = batch['depth_ref'][:, None, ...]
        depth_input = batch['depth_input_ref'][:, None, ...]

        '''batch_size,_,h,w = img.shape


        #mask = (depth_gt>0)
        random_num = torch.rand(batch_size,2)
        random_h = 500
        random_w = 400
        random_y = random_num[:,0]*(h-random_h)
        random_x = random_num[:,1]*(w-random_w)
        mask = torch.ones(batch_size,1,h,w).to(img.device)
        for i in range(batch_size):
            mask[i,0,int(random_y[i]):int(random_y[i]+random_h),int(random_x[i]):int(random_x[i]+random_w)]=0'''

        pred_dic = self.net(img,depth_input,train=True)

        loss1 = mseloss(pred_dic[0],depth_gt)
        #loss1 = unimodal_loss(pred_dic[0],pred_val,depth_gt,mask, dist="gaussian")
        #loss = calc_losses_disp_sup([pred_disp_0,pred_disp_1,pred_disp_2],disp_gt,mask,weights=[1.0,0.3,0.15])
        #loss = calc_losses_disp_sup(pred_dic,depth_gt,mask,weights=[1.0,0.3,0.15])


        if self.current_epoch<60:
            loss2 = mseloss(pred_dic[1],depth_gt)
            loss3 = mseloss(pred_dic[2],depth_gt)
            #loss2 = unimodal_loss(pred_dic[1],pred_val,depth_gt,mask, dist="gaussian")
            #loss3 = unimodal_loss(pred_dic[2],pred_val,depth_gt,mask, dist="gaussian")
            loss = loss1+loss2+loss3
        elif self.current_epoch<60:
            loss2 = mseloss(pred_dic[1],depth_gt)
            loss3 = mseloss(pred_dic[2],depth_gt)
            #loss2 = unimodal_loss(pred_dic[1],pred_val,depth_gt,mask, dist="gaussian")
            #loss3 = unimodal_loss(pred_dic[2],pred_val,depth_gt,mask, dist="gaussian")
            loss = loss1+0.1*loss2+0.1*loss3
        else:
            loss = loss1

        imae = iMAE(pred_dic[0],depth_gt).detach()
        mae = MAE(pred_dic[0],depth_gt).detach()
        rmse = RMSE(pred_dic[0],depth_gt).detach()
        irmse = iRMSE(pred_dic[0],depth_gt).detach()

        return {
            'loss': loss,
            'iMAE': imae,
            'MAE': mae,
            'RMSE': rmse,
            'iRMSE': irmse,
        }

    def validation_step(self, batch, batch_idx):
        img = batch['img_ref']
        depth_gt = batch['depth_ref'][:, None, ...]
        depth_input = batch['depth_input_ref'][:, None, ...]

        '''batch_size,_,h,w = img.shape


        #mask = (depth_gt>0)
        random_num = torch.rand(batch_size,2)
        random_h = 500
        random_w = 400
        random_y = random_num[:,0]*(h-random_h)
        random_x = random_num[:,1]*(w-random_w)
        mask = torch.ones(batch_size,1,h,w).to(img.device)
        for i in range(batch_size):
            mask[i,0,int(random_y[i]):int(random_y[i]+random_h),int(random_x[i]):int(random_x[i]+random_w)]=0'''

        pred_dic = self.net(img,depth_input,train=True)

        loss1 = mseloss(pred_dic[0],depth_gt)
        #loss1 = unimodal_loss(pred_dic[0],pred_val,depth_gt,mask, dist="gaussian")
        #loss = calc_losses_disp_sup([pred_disp_0,pred_disp_1,pred_disp_2],disp_gt,mask,weights=[1.0,0.3,0.15])
        #loss = calc_losses_disp_sup(pred_dic,depth_gt,mask,weights=[1.0,0.3,0.15])


        if self.current_epoch<60:
            loss2 = mseloss(pred_dic[1],depth_gt)
            loss3 = mseloss(pred_dic[2],depth_gt)
            #loss2 = unimodal_loss(pred_dic[1],pred_val,depth_gt,mask, dist="gaussian")
            #loss3 = unimodal_loss(pred_dic[2],pred_val,depth_gt,mask, dist="gaussian")
            loss = loss1+loss2+loss3
        elif self.current_epoch<60:
            loss2 = mseloss(pred_dic[1],depth_gt)
            loss3 = mseloss(pred_dic[2],depth_gt)
            #loss2 = unimodal_loss(pred_dic[1],pred_val,depth_gt,mask, dist="gaussian")
            #loss3 = unimodal_loss(pred_dic[2],pred_val,depth_gt,mask, dist="gaussian")
            loss = loss1+0.1*loss2+0.1*loss3
        else:
            loss = loss1

        imae = iMAE(pred_dic[0],depth_gt).detach()
        mae = MAE(pred_dic[0],depth_gt).detach()
        rmse = RMSE(pred_dic[0],depth_gt).detach()
        irmse = iRMSE(pred_dic[0],depth_gt).detach()

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
    'msg_chn_192': {
        'model_class': MSG_CHNModel,
        'model_kws': {
            'max_disp': 192,#ADAM 100 SceneFlow 192
            'optim_kws': {
                # 'lr': 1e-3,
                'lr': 1e-3,
            },
            'viz_train': False,
        },
        'batch_size': 18,
        'data_modules': {
            'monocular_livox': DATA_MODULES['monocular_livox'],
            'fusion_kitti': DATA_MODULES['fusion_kitti'],
            'stereo_livox': DATA_MODULES['stereo_livox'],
        },
        'trainer_kws': {
            'max_epochs': 200,
            'callbacks': [
                EarlyStopping('val_loss', mode='min', patience=200),
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='epoch_{epoch:02d}-loss_{epe:.2f}',
                    verbose=True,
                    auto_insert_metric_name=False,
                ),
            ],
            # 'profiler': 'advanced',
        },
    },
})

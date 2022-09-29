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
from deepstereo.model_zoo.lsfnet import LSFNet
import pylab as plt

from deepstereo.loss import calc_losses_disp_sup, calc_losses_disp_unsup,unimodal_loss,bimodal_loss,negative_log_prob_loss
from deepstereo.metric import calc_endpoint_error,calc_thres_metric,calc_D1_metric_with_thres,iMAE,MAE,RMSE,iRMSE
from deepstereo.visualize import TrainerVisdom
from deepstereo.blocks import disp2depth
from data_module import DATA_MODULES
from visdom import Visdom
from einops import repeat
import random

def disp2depth(disp,bf):
    _,c,h,w=disp.shape
    bf = repeat(bf.type_as(disp), 'b -> b c h w',c=c,h=h,w=w)
    invalid_mask = disp <= 0
    depth = bf/ (disp + 1E-8)
    depth[invalid_mask] = 0
    return depth


class LSFNetModel(pl.LightningModule):
    def __init__(self, max_disp, optim_kws={}, viz_train=False):
        super().__init__()

        self.name = 'lsfnet'

        self.net = LSFNet(max_disp=max_disp)

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

    def forward(self,img_l,img_r,disp_l, batch_idx=0):
        preds = self.net(img_l,img_r,disp_l)
        return preds

    def training_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        img_r = batch['img_tgt']
        disp_gt = batch['disp_ref'][:, None, ...]
        disp_input = batch['disp_input_ref'][:, None, ...]
        depth_gt = batch['depth_ref'][:, None, ...]

        mask = (disp_gt<self.max_disp)*(disp_gt>0)

        '''batch_size,_,h,w = img_l.shape
        mask_input = torch.ones_like(mask)
        win_h = int(h/2)
        win_w = int(w/1.5)
        random_num = torch.rand(2)
        random_h = int(random_num[0]*(h-win_h))
        random_w = int(random_num[1]*(w-win_w))
        mask_input[:,0,random_h:random_h+win_h,random_w:random_w+win_w]=0

        pred_dic = self.net(img_l,img_r,disp_input*mask_input,train=True)'''
        pred_dic = self.net(img_l,img_r,disp_input,train=True)

        loss_stereo_sup = unimodal_loss(pred_dic['disp_stereo']/self.max_disp,pred_dic['prob_stereo'],disp_gt/self.max_disp,mask,dist="laplacian")
        loss_stereo_sup_mid = calc_losses_disp_sup(pred_dic['disp_stereo_mid'],disp_gt,mask,weights=[1.0])
        loss_mono_sup = unimodal_loss(pred_dic['disp_monocular']/self.max_disp,pred_dic['prob_monocular'],disp_gt/self.max_disp,mask,dist="laplacian")
        loss_mono_sup_mid = calc_losses_disp_sup(pred_dic['disp_monocular_mid'],disp_gt,mask,weights=[1.0,0.5])
        loss_sup = negative_log_prob_loss(pred_dic['disp_stereo']/self.max_disp,pred_dic['disp_monocular']/self.max_disp,\
            pred_dic['prob_stereo'],pred_dic['prob_monocular'],pred_dic['pi'],(1-pred_dic['pi']),disp_gt/self.max_disp,mask,dist="laplacian")
        
        '''if self.current_epoch<3:
            loss = 0.2*(loss_mono_sup+4*loss_mono_sup_mid)+0.6*(loss_stereo_sup+4*loss_stereo_sup_mid)+0.2*loss_sup
        elif self.current_epoch<5:
            loss = 0.6*(loss_mono_sup+4*loss_mono_sup_mid)+0.2*(loss_stereo_sup+4*loss_stereo_sup_mid)+0.2*loss_sup
        elif self.current_epoch<10:
            loss = 0.1*loss_stereo_sup+0.1*loss_mono_sup+0.8*loss_sup'''
        #if self.current_epoch<100:
        #    loss = 0.25*loss_stereo_sup+loss_stereo_sup_mid+0.125*loss_mono_sup+loss_mono_sup_mid+loss_sup
        #else:
        loss = 0.6*loss_sup+0.2*loss_stereo_sup+0.2*loss_mono_sup

        epe = calc_endpoint_error(pred_dic['disp'], disp_gt, mask).detach()
        px_3 = calc_D1_metric_with_thres(pred_dic['disp'], disp_gt, mask,3).detach()

        pred_depth = disp2depth(pred_dic['disp'],batch['bf'])
        if len(pred_depth[mask])==0:
            '''print(batch['filenames'])
            ax1=plt.subplot(221)
            ax1.imshow(255*img_l[0].permute(1, 2, 0).cpu().numpy())
            ax2=plt.subplot(222)
            ax2.imshow(depth_gt[0][0].cpu().numpy())
            ax3=plt.subplot(223)
            ax3.imshow(pred_depth[0][0].detach().cpu().numpy())
            plt.show()'''
            imae=torch.tensor(0).to(pred_depth.device)
            irmse=torch.tensor(0).to(pred_depth.device)
            mae=torch.tensor(0).to(pred_depth.device)
            rmse=torch.tensor(0).to(pred_depth.device)
        else:
            imae = iMAE(pred_depth*mask,depth_gt*mask).detach()
            irmse = iRMSE(pred_depth*mask,depth_gt*mask).detach()
            mae = MAE(pred_depth*mask,depth_gt*mask).detach()
            rmse = RMSE(pred_depth*mask,depth_gt*mask).detach()


        '''return {
            'loss': loss,
            'EPE': epe,
            "3px":px_3,
            'MAE':mae,
            'RMSE':rmse,
            "iMAE":imae,
            "iRMSE":irmse,
        }'''
        return {
            'loss': loss,
            'EPE': epe.detach(),
            "3px":px_3.detach(),
            'MAE':mae.detach(),
            'RMSE':rmse.detach(),
            "iMAE":imae.detach(),
            "iRMSE":irmse.detach(),
        }

        

    def validation_step(self, batch, batch_idx):
        img_l = batch['img_ref']
        img_r = batch['img_tgt']
        disp_gt = batch['disp_ref'][:, None, ...]
        disp_input = batch['disp_input_ref'][:, None, ...]
        depth_gt = batch['depth_ref'][:, None, ...]

        mask = (disp_gt<self.max_disp)*(disp_gt>0)

        '''batch_size,_,h,w = img_l.shape
        mask_input = torch.ones_like(mask)
        win_h = int(h/2)
        win_w = int(w/1.5)
        random_num = torch.rand(2)
        random_h = int(random_num[0]*(h-win_h))
        random_w = int(random_num[1]*(w-win_w))
        mask_input[:,0,random_h:random_h+win_h,random_w:random_w+win_w]=0

        pred_dic = self.net(img_l,img_r,disp_input*mask_input,train=True)'''
        pred_dic = self.net(img_l,img_r,disp_input,train=True)

        loss_stereo_sup = unimodal_loss(pred_dic['disp_stereo']/self.max_disp,pred_dic['prob_stereo'],disp_gt/self.max_disp,mask,dist="laplacian")
        loss_stereo_sup_mid = calc_losses_disp_sup(pred_dic['disp_stereo_mid'],disp_gt,mask,weights=[1.0])
        loss_mono_sup = unimodal_loss(pred_dic['disp_monocular']/self.max_disp,pred_dic['prob_monocular'],disp_gt/self.max_disp,mask,dist="laplacian")
        loss_mono_sup_mid = calc_losses_disp_sup(pred_dic['disp_monocular_mid'],disp_gt,mask,weights=[1.0,0.5])
        loss_sup = negative_log_prob_loss(pred_dic['disp_stereo']/self.max_disp,pred_dic['disp_monocular']/self.max_disp,\
            pred_dic['prob_stereo'],pred_dic['prob_monocular'],pred_dic['pi'],(1-pred_dic['pi']),disp_gt/self.max_disp,mask,dist="laplacian")
        
        '''if self.current_epoch<3:
            loss = 0.2*(loss_mono_sup+4*loss_mono_sup_mid)+0.6*(loss_stereo_sup+4*loss_stereo_sup_mid)+0.2*loss_sup
        elif self.current_epoch<5:
            loss = 0.6*(loss_mono_sup+4*loss_mono_sup_mid)+0.2*(loss_stereo_sup+4*loss_stereo_sup_mid)+0.2*loss_sup
        elif self.current_epoch<10:
            loss = 0.1*loss_stereo_sup+0.1*loss_mono_sup+0.8*loss_sup'''
        #if self.current_epoch<100:
        #    loss = 0.25*loss_stereo_sup+loss_stereo_sup_mid+0.125*loss_mono_sup+loss_mono_sup_mid+0.5*loss_sup
        #else:
        loss = 0.6*loss_sup+0.2*loss_stereo_sup+0.2*loss_mono_sup


        epe = calc_endpoint_error(pred_dic['disp'], disp_gt, mask).detach()
        px_3 = calc_D1_metric_with_thres(pred_dic['disp'], disp_gt, mask,3).detach()

        pred_depth = disp2depth(pred_dic['disp'],batch['bf'])
        if len(pred_depth[mask])==0:
            '''print(batch['filenames'])
            ax1=plt.subplot(221)
            ax1.imshow(255*img_l[0].permute(1, 2, 0).cpu().numpy())
            ax2=plt.subplot(222)
            ax2.imshow(depth_gt[0][0].cpu().numpy())
            ax3=plt.subplot(223)
            ax3.imshow(pred_depth[0][0].detach().cpu().numpy())
            plt.show()'''
            imae=torch.tensor(0).to(pred_depth.device)
            irmse=torch.tensor(0).to(pred_depth.device)
            mae=torch.tensor(0).to(pred_depth.device)
            rmse=torch.tensor(0).to(pred_depth.device)
        else:
            imae = iMAE(pred_depth*mask,depth_gt*mask).detach()
            irmse = iRMSE(pred_depth*mask,depth_gt*mask).detach()
            mae = MAE(pred_depth*mask,depth_gt*mask).detach()
            rmse = RMSE(pred_depth*mask,depth_gt*mask).detach()


        '''return {
            'loss': loss,
            'EPE': epe,
            "3px":px_3,
            'MAE':mae,
            'RMSE':rmse,
            "iMAE":imae,
            "iRMSE":irmse,
        }'''
        return {
            'loss': loss,
            'EPE': epe.detach(),
            "3px":px_3.detach(),
            'MAE':mae.detach(),
            'RMSE':rmse.detach(),
            "iMAE":imae.detach(),
            "iRMSE":irmse.detach(),
            #"5":loss_branches.detach(),
            #'6':loss_stereo_sup.detach(),
            #'7':loss_stereo_sup_mid.detach(),
            #'8':loss_mono_sup.detach(),
            #'9':loss_mono_sup_mid.detach(),
            #'10':loss_sup.detach(),
        }


    def run_epoch_end(self, stage, outputs):
        loss = torch.stack([i['loss'] for i in outputs]).mean()
        epe = torch.stack([i['EPE'] for i in outputs]).mean()
        px3 = torch.stack([i['3px'] for i in outputs]).mean()
        imae = torch.stack([i['iMAE'] for i in outputs]).mean()
        irmse = torch.stack([i['iRMSE'] for i in outputs]).mean()
        mae = torch.stack([i['MAE'] for i in outputs]).mean()
        rmse = torch.stack([i['RMSE'] for i in outputs]).mean()
        #loss5 = torch.stack([i['5'] for i in outputs]).mean()
        #loss6 = torch.stack([i['6'] for i in outputs]).mean()
        #loss7 = torch.stack([i['7'] for i in outputs]).mean()
        #loss8 = torch.stack([i['8'] for i in outputs]).mean()
        #loss9 = torch.stack([i['9'] for i in outputs]).mean()
        #loss10 = torch.stack([i['10'] for i in outputs]).mean()

        if stage=='val':
            #self.log(f'{stage}_loss', loss, prog_bar=True, logger=True, sync_dist=True)
            self.log('3px', px3, prog_bar=True, logger=True, sync_dist=True)
            self.log('epe', epe, prog_bar=True, logger=True, sync_dist=True)
            self.log('mae', mae, prog_bar=True, logger=True, sync_dist=True)
            self.log('rmse', rmse, prog_bar=True, logger=True, sync_dist=True)
            self.log('imae', imae, prog_bar=True, logger=True, sync_dist=True)
            self.log('irmse', irmse, prog_bar=True, logger=True, sync_dist=True)
            #self.log('5',loss5,prog_bar=True, logger=True, sync_dist=True)
            #self.log('6',loss6,prog_bar=True, logger=True, sync_dist=True)
            #self.log('7',loss7,prog_bar=True, logger=True, sync_dist=True)
            #self.log('8',loss8,prog_bar=True, logger=True, sync_dist=True)
            #self.log('9',loss9,prog_bar=True, logger=True, sync_dist=True)
            #self.log('*',loss10,prog_bar=True, logger=True, sync_dist=True)
            

        '''if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(  
                f'{stage}/avg_loss_on_epoch', loss,
                global_step=self.current_epoch)'''

    def training_epoch_end(self, outputs):
        return self.run_epoch_end('train', outputs)

    def validation_epoch_end(self, outputs):
        return self.run_epoch_end('val', outputs)

    def configure_optimizers(self):
        optim_kws = self.optim_kws
        #if self.current_epoch>7:
        #    optim_kws['lr']=1e-4
        return torch.optim.Adam(self.parameters(), **optim_kws)

MODELS = edict({
    'lsfnet_192': {
        'model_class': LSFNetModel,
        'model_kws': {
            'max_disp': 192,#ADAM 100 SceneFlow 192
            'optim_kws': {
                # 'lr': 1e-3,
                'lr': 1e-3,
            },
            'viz_train': False,
        },
        'batch_size': 4 ,
        'data_modules': {
            'stereo_sceneflow_driving': DATA_MODULES['stereo_sceneflow_driving'],
            'stereo_drivingstereo': DATA_MODULES['stereo_drivingstereo'],
            'stereo_livox': DATA_MODULES['stereo_livox'],
            'fusion_kitti': DATA_MODULES['fusion_kitti'],
        },
        'trainer_kws': {
            'max_epochs': 10,
            'callbacks': [
                EarlyStopping('epe', mode='min', patience=10),
                ModelCheckpoint(
                    monitor='epe',
                    filename='epoch_{epoch:02d}',
                    verbose=True,
                    auto_insert_metric_name=False,
                ),
            ],
            # 'profiler': 'advanced',
        },
    },
})

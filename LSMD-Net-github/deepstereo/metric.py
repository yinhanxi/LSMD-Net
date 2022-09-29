import torch
import numpy as np
import math
import torch.nn.functional as F
from .blocks import upsample_disp

def calc_endpoint_error(pred, gt, mask):
    mask = mask.bool()
    if pred.shape[-1] < gt.shape[-1]:
            scale = gt.shape[-1] // pred.shape[-1]
            pred = upsample_disp(pred, scale)
    epe = F.l1_loss(pred[mask], gt[mask], reduction='mean')
    return epe

def calc_D1_metric(D_est, D_gt, mask):
    #thres=3px
    mask = mask.bool()
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def calc_D1_metric_with_thres(D_est, D_gt, mask, thres):
    mask = mask.bool()
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > thres) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def calc_thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    mask = mask.bool()
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

def root_mean_sq_err(src, tgt):
    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    return np.sqrt(np.mean((1000*((1.0 / tgt) - (1.0 / src))) ** 2))

def inv_mean_abs_err(src, tgt):
    return np.mean(1000*np.abs((1.0 / tgt) - (1.0 / src)))

def iMAE(outputs,target): # m->1/km
        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs == 0] = -1
        target[target == 0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        val_pixels = (target > 0).float()*(outputs>0).float()
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt)

def MAE(outputs, target): # m->mm
        val_pixels = ((target > 0).float()*(outputs>0).float())
        err = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt) * 1000

def RMSE(outputs, target):
        val_pixels = ((target > 0).float()*(outputs>0).float())
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        #return torch.sqrt(torch.mean(loss / cnt))
        return torch.mean(torch.sqrt(loss / cnt))  * 1000

def iRMSE(outputs, target):
        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs==0] = -1
        target[target==0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        val_pixels = (target > 0).float()*(outputs>0).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        #return torch.sqrt(torch.mean(loss / cnt))
        return torch.mean(torch.sqrt(loss / cnt))

'''def mae(pred, target):
    """ Mean Average Error (MAE) """
    val_pixels = (target > 0).astype(float)*(pred > 0).astype(float)
    return np.absolute(pred*val_pixels - target*val_pixels).sum()/val_pixels.sum()


def imae(pred, target):
    """ inverse Mean Average Error in 1/km (iMAE) """
    pred = pred/1000
    target = target/1000
    pred[pred == 0] = -1
    target[target == 0] = -1
    pred = 1. / pred
    target = 1. / target
    pred[pred == -1] = 0
    target[target == -1] = 0
    val_pixels = (target > 0).astype(float)*(pred > 0).astype(float)
    return np.absolute(pred*val_pixels-target*val_pixels).sum()/val_pixels.sum()


def rmse(pred, target):
    """ Root Mean Square Error (RMSE) """
    val_pixels = (target > 0).astype(float)*(pred > 0).astype(float)
    return math.sqrt(np.power((pred*val_pixels - target*val_pixels), 2).sum()/val_pixels.sum())


def irmse(pred, target):
    """ inverse Root Mean Square Error in 1/km (iRMSE) """
    pred = pred / 1000.
    target = target / 1000.
    pred[pred==0] = -1
    target[target==0] = -1
    pred = 1. / pred
    target = 1. / target
    pred[pred == -1] = 0
    target[target == -1] = 0
    val_pixels =(target > 0).astype(float)*(pred > 0).astype(float)
    return math.sqrt(np.power((pred*val_pixels-target*val_pixels), 2).sum()/val_pixels.sum())'''

def mae(pred, target):
    """ Mean Average Error (MAE) """
    return np.absolute(pred - target).mean()


def imae(pred, target):
    """ inverse Mean Average Error in 1/km (iMAE) """
    return np.absolute(1000*(1./pred-1./target)).mean()


def rmse(pred, target):
    """ Root Mean Square Error (RMSE) """
    return math.sqrt(np.power((pred - target), 2).mean())


def irmse(pred, target):
    """ inverse Root Mean Square Error in 1/km (iRMSE) """
    return math.sqrt(np.power(1000*(1./pred - 1./target), 2).mean())

def err_1px(pred, target):
    """ 1-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 1) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])


def err_2px(pred, target):
    """ 2-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 2) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])


def err_3px(pred, target):
    """ 3-pix error; used in stereo depth """
    abs_err = np.absolute(pred - target)
    correct = (abs_err < 3) | (abs_err < (target * 0.05))
    return 1 - (float(correct.sum()) / target.shape[0])

def laplacian(x, mu, b):
    return 0.5 * np.exp(-(np.abs(mu-x)/b))/b

def differential_entropy(mu0, mu1, sigma0, sigma1, pi0, pi1, n=2000, a=-1., b=2.):
    eps = 1e-6
    f = lambda x: pi0 * laplacian(x, mu0, sigma0) + pi1 * laplacian(x, mu1, sigma1)
    for k in range(1, n):
        x = a + k*(b-a)/n
        fx = f(x)
        mask = fx<eps
        ent_i = fx * np.log(fx)
        ent_i[mask] = 0
        sum = ent_i if k==1 else sum + ent_i
    return -(b - a)*(f(a)/2 + f(b)/2 + sum)/n
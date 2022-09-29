import torch
import torch.nn.functional as F
from .blocks import upsample_disp
from kornia.losses import ssim_loss, inverse_depth_smoothness_loss
import math

loss_fns = {
    'smooth_l1': F.smooth_l1_loss,
    'l1': F.l1_loss,
    'mse':F.mse_loss
}

def calc_losses_disp_sup(preds, gt, mask, weights=None, loss_fn='smooth_l1'):
    if weights is None:
        weights = [1] * len(preds)
    assert len(weights) == len(preds)
    loss_fn = loss_fns[loss_fn]
    mask = mask.bool()
    losses = []

    for pred, w in zip(preds, weights):
        if pred.shape[-1] < gt.shape[-1]:
            scale = gt.shape[-1] // pred.shape[-1]
            pred = upsample_disp(pred, scale)

        loss = loss_fn(pred[mask], gt[mask], reduction='mean')
        losses.append(w * loss)

    return sum(losses)

def warp_disp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0,1,w).repeat(b,h,1).type_as(img)
    y_base = torch.linspace(0,1,h).repeat(b,w,1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base+x_shifts,y_base),dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field-1, mode='bilinear', padding_mode='zeros', align_corners=False)

    return output

'''def calc_loss_disp_photometric(pred_l, img_l, img_r, disp_l,disp_r,valid_mask=None, win_size=11, alpha=0.85, reduction='mean'):
    b, _, h, w = img_l.shape
    image_warped = warp_disp(img_r,-pred_l)
    LRC_mask = (torch.abs(warp_disp(disp_r,-disp_l)-disp_l)/disp_l.abs())<0.05

    valid_mask = torch.ones(b, 1, h, w).to(img_l.device) if valid_mask is None else valid_mask
    warped_mask = (image_warped>0)

    src = image_warped * valid_mask*warped_mask*LRC_mask
    dst = img_l * valid_mask*warped_mask*LRC_mask

    loss = alpha * ssim_loss(src, dst, win_size, reduction=reduction) + (1 - alpha) * F.l1_loss(src, dst, reduction=reduction)
    return loss'''

def calc_loss_disp_photometric(pred_l, img_l, img_r,valid_mask=None, win_size=11, alpha=0.85, reduction='mean'):
    b, _, h, w = img_l.shape
    image_warped = warp_disp(img_r,-pred_l)

    valid_mask = torch.ones(b, 1, h, w).to(img_l.device) if valid_mask is None else valid_mask
    warped_mask = (image_warped>0)

    src = image_warped * valid_mask*warped_mask
    dst = img_l * valid_mask*warped_mask

    loss = alpha * ssim_loss(src, dst, win_size, reduction=reduction) + (1 - alpha) * F.l1_loss(src, dst, reduction=reduction)
    return loss

'''def calc_losses_disp_unsup(preds, img_l, img_r, disp_l,disp_r,valid_mask_l, weights):
    losses_P, losses_S = [], []
    for pred, w in zip(preds, weights):
        if pred.shape[-1] < img_l.shape[-1]:
            scale = img_l.shape[-1] // pred.shape[-1]
            pred = upsample_disp(pred, scale)

        loss_P = calc_loss_disp_photometric(pred, img_l, img_r,disp_l,disp_r, valid_mask_l)
        loss_S = inverse_depth_smoothness_loss(pred,img_l)
        losses_P.append(w * loss_P)
        losses_S.append(w * loss_S)

    return torch.stack(losses_P).sum(), torch.stack(losses_S).sum()'''

def calc_losses_disp_unsup(preds, img_l, img_r, valid_mask_l, weights):
    losses_P, losses_S = [], []
    for pred, w in zip(preds, weights):
        if pred.shape[-1] < img_l.shape[-1]:
            scale = img_l.shape[-1] // pred.shape[-1]
            pred = upsample_disp(pred, scale)

        loss_P = calc_loss_disp_photometric(pred, img_l, img_r,valid_mask_l)
        loss_S = inverse_depth_smoothness_loss(pred,img_l)
        losses_P.append(w * loss_P)
        losses_S.append(w * loss_S)

    return torch.stack(losses_P).sum(), torch.stack(losses_S).sum()

def SILogLoss(input, target, mask=None):
    if input.shape[-2:]!=target.shape[-2:]:
        input = torch.nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

    if mask is not None:
        input = input[mask]
        target = target[mask]
    g = torch.log(input) - torch.log(target)
    # n, c, h, w = g.shape
    # norm = 1/(h*w)
    # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

    Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
    return 10 * torch.sqrt(Dg)

def unimodal_loss(mu, sigma, labels,mask,dist="laplacian"):
    sigma = sigma[mask]
    labels = labels[mask]
    mu = mu[mask]
    if dist == "laplacian":
        loss = torch.abs(mu-labels)/sigma+torch.log(sigma)
    elif dist == "gaussian":
        loss = ((mu-labels)**2)/(2*(sigma**2))+torch.log(sigma)
    loss = loss.sum()/loss.shape[0]
    return loss

def bimodal_loss(mu0, mu1, sigma0, sigma1, w0, w1, labels,mask,dist="laplacian"):
    mu0 = mu0[mask]
    mu1 = mu1[mask]
    sigma0 = sigma0[mask]
    sigma1 = sigma1[mask]
    w0 = w0[mask]
    w1 = w1[mask]
    labels = labels[mask]
    loss = - torch.log(w0 * distribution(mu0, sigma0, labels, dist) + \
                       w1 * distribution(mu1, sigma1, labels, dist))

    return loss.sum()/loss.shape[0]

def gaussian(mu, sigma, labels):
    return torch.exp(-0.5*(mu-labels)** 2/ sigma** 2)/sigma

def laplacian(mu, b, labels):
    return 0.5 * torch.exp(-(torch.abs(mu-labels)/b))/b

def distribution(mu, sigma, labels, dist="gaussian"):
    return gaussian(mu, sigma, labels) if dist=="gaussian" else \
           laplacian(mu, sigma, labels)

def mseloss(outputs, target):
        val_pixels = torch.ne(target, 0).float()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(loss ** 2) / torch.sum(val_pixels)

def negative_log_prob_loss(mu0, mu1, sigma0, sigma1, w0, w1, labels,mask,dist="laplacian"):
    mu0 = mu0[mask]
    mu1 = mu1[mask]
    sigma0 = sigma0[mask]
    sigma1 = sigma1[mask]
    w0 = w0[mask]+1e-15
    w1 = w1[mask]+1e-15
    labels = labels[mask]
    dist0 = torch.log(w0)+log_distribution(mu0,sigma0,labels,dist)
    dist1 = torch.log(w1)+log_distribution(mu1,sigma1,labels,dist)
    #loss = -torch.log(torch.exp(torch.log(w0)+log_distribution(mu0,sigma0,labels,dist))\
    #    +torch.exp(torch.log(w1)+log_distribution(mu1,sigma1,labels,dist)))
    dist_max,_ = torch.cat((dist0[:,None],dist1[:,None]),1).max(1)
    loss = -dist_max-torch.log(torch.exp(dist0-dist_max)+torch.exp(dist1-dist_max))
    return loss.mean()

LOG2PI = math.log(2 * math.pi)

def log_distribution(mu,sigma,labels,dist):
    if dist == 'gaussian':
        return -torch.log(sigma)-0.5*LOG2PI-0.5*torch.pow((labels-mu)/sigma,2)
    elif dist == 'laplacian':
        return -torch.log(sigma)-math.log(2)-(torch.abs(labels-mu))/sigma
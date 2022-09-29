import torch.nn.functional as F

def upsample_disp(disp, scale, mode='bilinear'):
    return F.interpolate(disp * scale, scale_factor=scale, mode=mode, align_corners=True)

def disp2depth(disp,width):
    baseline = 0.54
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
    width_to_focal[1238] = 718.3351

    focal_length = width_to_focal[width]
    invalid_mask = disp <= 0
    depth = baseline * focal_length / (disp + 1E-8)
    depth[invalid_mask] = 0
    depth = depth.clamp(max=100.0) # NOTE: clamp to maximum depth as 100 for KITTI
    return depth

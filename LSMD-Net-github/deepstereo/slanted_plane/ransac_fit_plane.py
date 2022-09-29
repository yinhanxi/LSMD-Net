import os
import os.path as osp

cur_dir = osp.dirname(osp.abspath(__file__))
import numpy as np
import random
import math
import cupy as cp

module = cp.RawModule(
    code=open(osp.join(cur_dir, 'ransac_fit_plane_kernel.cu')).read())

ransac_fit_plane_kernel = module.get_function('ransac_fit_plane_kernel')

def ransac_fit_plane(src, mode='full', disp_th=1,
                     reg_win=9, ransac_sample_nr=30, ransac_iters=10, size_divisor=1, kernel_blocks=32):

    data_img = np.ascontiguousarray(src, dtype='f4')
    if size_divisor > 1:
        rest = data_img.shape[0] % size_divisor
        if rest:
            data_img = np.pad(data_img, pad_width=((0, size_divisor - rest), (0, 0)), mode='zero')

        rest = data_img.shape[1] % size_divisor
        if rest:
            data_img = np.pad(data_img, pad_width=((0, 0), (0, size_divisor - rest)), mode='zero')

    disp_th = float(disp_th)
    height = data_img.shape[0]
    width = data_img.shape[1]

    reg_win = reg_win
    reg_win_rad = (reg_win - 1) // 2

    win_size = reg_win ** 2

    valid_width = width - reg_win_rad * 2
    valid_height = height - reg_win_rad * 2
    valid_pixels = valid_width * valid_height

    # gather pixels for a regression window
    data = []
    for i in range(-reg_win_rad, reg_win_rad + 1):
        start_i = i + reg_win_rad
        stop_i = data_img.shape[0] + i - reg_win_rad

        for j in range(-reg_win_rad, reg_win_rad + 1):
            start_j = j + reg_win_rad
            stop_j = data_img.shape[1] + j - reg_win_rad
            data.append(data_img[start_i:stop_i, start_j:stop_j])

    data = np.array(data)
    data = data.transpose(1, 2, 0)

    samples = np.arange(-reg_win_rad, reg_win_rad + 1, dtype=np.float32)

    ord_v = np.tile(samples.reshape(reg_win, 1), (1, reg_win))
    ord_u = np.tile(samples.reshape(1, reg_win), (reg_win, 1))

    sample_list = []
    for i in range(ransac_iters):
        sequence = np.arange(win_size, dtype=int).tolist()
        sample_list.append(random.sample(sequence, ransac_sample_nr))
    sample_list = np.array(sample_list)

    coeff_gpu = cp.empty(shape=(valid_height, valid_width, 3), dtype='f4')

    grid_nr = int(math.ceil(valid_pixels / kernel_blocks))

    ransac_fit_plane_kernel((grid_nr,), (kernel_blocks,), (
        cp.asarray(ord_v),
        cp.asarray(ord_u),
        cp.asarray(data),
        coeff_gpu,
        valid_pixels,
        win_size,
        cp.asarray(sample_list),
        ransac_iters,
        ransac_sample_nr,
        disp_th,
    ))

    coeff = coeff_gpu.get()
    coeff_img = np.pad(coeff, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad), (0, 0)), mode='edge')
    return coeff_img

import os
import os.path as osp
from glob import glob

def find_latest_checkpoint(d):
    latest_ver = sorted([(int(i.split('_')[-1]), i) for i in os.listdir(d)])[-1][-1]

    #cb_dir = osp.join(d, latest_ver, 'checkpoints')
    cb_dir = osp.join(d,'version_21', 'checkpoints')

    cb_fs = sorted(glob(osp.join(cb_dir, '*.ckpt')), reverse=True)
    for cb_f in cb_fs:
        if '.tmp_end' not in cb_f:
            return cb_f

    raise

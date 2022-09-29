#!/usr/bin/env python
import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from deepstereo.base import import_model
from pprint import pprint
from loguru import logger

import click

@click.command()
@click.option('--data_root', default='')
@click.option('--work_dir', '-w', default='')
@click.option('--resume_from', default='')
@click.option('--load_from', default='')
@click.option('--gpus', default='0')
@click.option('--seed', default=5)
@click.option('--deterministic', is_flag=True)
@click.option('--batch_size', default=0)
@click.option('--worker_nr', default=4)
@click.option('--dataset', '--ds', default='')
@click.option('--debug', is_flag=True)
@click.option('--checkpoint', default=None)
@click.argument('model')
def main(data_root, work_dir, resume_from, load_from,
         gpus,  seed, deterministic,
         batch_size, worker_nr, dataset, debug, checkpoint,model):

    cfg = import_model(model, dataset)
    if batch_size > 0:
        cfg.batch_size = batch_size

    if not data_root:
        data_root = osp.join(cur_dir, "..", "data")

    torch.backends.cudnn.benchmark = cfg.get('no_cudnn_benchmark', True)
    if work_dir:
        cfg.work_dir = work_dir
    elif 'work_dir' not in cfg:
        cfg.work_dir = f'./work_dirs/{cfg.net_type}-{cfg.ds}'

    os.makedirs(cfg.work_dir, exist_ok=True)

    if resume_from:
        cfg.resume_from = resume_from
    elif load_from:
        cfg.load_from = load_from

    model = cfg.model_class(**cfg.model_kws)
    #for param_tensor in model.state_dict():
    #    print(param_tensor)

    if checkpoint is not None:
        print('load ckpt:',checkpoint)
        weights = torch.load(checkpoint,map_location=torch.device('cpu'))['state_dict']
        model_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #torch.cuda.empty_cache()


    dm = cfg.dm_class(data_root=data_root, batch_size=cfg.batch_size, data_loader_kws={
        'num_workers': worker_nr,
        # 'persistent_workers': True,
    }, **cfg.dm_kws)

    if deterministic:
        cfg.trainer_kws['deterministic'] = True
        if seed:
            pl.seed_everything(seed, workers=True)

    gpus = list(set(int(i) for i in gpus.split(',')))
    if len(gpus) > 1:
        cfg.trainer_kws.update({
            'accelerator': 'ddp_spawn',
        })

    trainer = pl.Trainer(
        default_root_dir=cfg.work_dir,
        gpus=gpus,
        **cfg.trainer_kws
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()

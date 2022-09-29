import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '../'))

from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from deepstereo.datasets.sceneflow import SceneFlowStereoDataset
from deepstereo.datasets.drivingstereo import DrivingStereoDataset
from deepstereo.datasets.livox import LivoxStereoDataset, LivoxMonocularDataset
from deepstereo.datasets.boxdepth import BoxDepthMonocularDataset
from deepstereo.datasets.kitti import KITTIFusionDataset,KITTIStereoDataset

import albumentations as A
from deepstereo.datasets.transform import StereoAugmentor, make_pipeline_stereo_test, make_pipeline_monocular_test,make_pipeline_stereo_val,img_transform,MonocularAugmentor

def make_pipeline_stereo_train(crop_size, to_tensor=True, **kws):
    post_transform = img_transform if to_tensor else None
    return StereoAugmentor(
        color_sym_aug=[
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        ], color_asym_aug=[
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0, hue=0),
        ], geo_aug=[
            A.RandomCrop(crop_size[1], crop_size[0]),
        ], post_transform=post_transform, **kws)

def make_pipeline_monocular_train(crop_size, to_tensor=True, **kws):
    post_transform = img_transform if to_tensor else None
    return MonocularAugmentor(
        color_sym_aug=[
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        ], color_asym_aug=[
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0, hue=0),
        ], 
        geo_aug=[
            A.RandomCrop(crop_size[1], crop_size[0]),
        ], post_transform=post_transform, **kws)

make_pipeline_stereo_dic = {
    'train': make_pipeline_stereo_train,
    'val': make_pipeline_stereo_val,
    'test': make_pipeline_stereo_test,
}

make_pipeline_monocular_dic = {
    'train': make_pipeline_monocular_train,
    'val': make_pipeline_monocular_test,
    'test': make_pipeline_monocular_test,
}

class SceneFlowStereoDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (512, 256)
            test_crop_size = (928, 512)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return SceneFlowStereoDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_stereo_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor, with_disp_tgt=True, with_slanted_plane=True,with_lidar_disp=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)

class DrivingStereoDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (512,256)
            test_crop_size = (928,512)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return DrivingStereoDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_stereo_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor,with_disp_tgt=False)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)

class LivoxStereoDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (512,256)
            test_crop_size = (896,896)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return LivoxStereoDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline,
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_stereo_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor,with_disp_tgt=False,with_lidar_disp=True,with_lidar_depth=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)

class LivoxMonocularDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (800,704)
            test_crop_size = (1024,1024)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return LivoxMonocularDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_monocular_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor,with_lidar_depth=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)

class BoxDepthMonocularDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (608,512)
            test_crop_size =  (608,512)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return BoxDepthMonocularDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_monocular_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)

class KITTIFusionDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (512,256)
            test_crop_size = (1216,256)  ##(1216,352)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return KITTIFusionDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline,
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_stereo_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor,with_disp_tgt=False,with_lidar_disp=True,with_lidar_depth=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)


class KITTIStereoDataModule(LightningDataModule):
    def __init__(self, data_root, folder, subset, batch_size, pipeline_kws={}, data_loader_kws={}, to_tensor=True):
        super().__init__()
        if not pipeline_kws:
            train_crop_size = (512,256)
            test_crop_size = (1216,256)  ##(1216,352)
            pipeline_kws = {
                'train': {
                    'crop_size': train_crop_size,
                },
                'val': {
                    'crop_size': train_crop_size,
                },
                'test_cfg': {
                    'crop_size': test_crop_size,
                },
            }

        self.data_root = osp.join(data_root, folder)
        self.subset = subset
        if isinstance(batch_size, int):
            batch_size = {
                'train': batch_size,
                'val': batch_size,
                'test': batch_size,
            }
        self.bs_train = batch_size['train']
        self.bs_val = batch_size['val']
        self.bs_test = batch_size['test']
        self.pipeline_kws = pipeline_kws
        self.to_tensor = to_tensor
        self.data_loader_kws = data_loader_kws

    def get_dataset(self, tp, pipeline):
        return KITTIStereoDataset(
            data_root=self.data_root,
            lst=f'{self.subset}_{tp}.json',
            pipeline=pipeline,
        )

    def get_pipeline(self, tp):
        p_kws = self.pipeline_kws.get(tp, {})
        return make_pipeline_stereo_dic[tp](p_kws['crop_size'], to_tensor=self.to_tensor,with_disp_tgt=False,with_lidar_disp=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.get_dataset('train', self.get_pipeline('train'))
            self.ds_val = self.get_dataset('val', self.get_pipeline('val'))

        elif stage == 'test' or stage is None:
            p_test_kws = self.pipeline_kws.get('test', {})
            self.ds_test = self.get_dataset('test', self.get_pipeline('test'))

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.bs_train, shuffle=True, **self.data_loader_kws)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.bs_val, **self.data_loader_kws)

    def test_dataloader(self):
        return DataLoader(self.ds_test, self.bs_test, **self.data_loader_kws)

    def get_dataloader(self, tp, **kws):
        return getattr(self, f'{tp}_dataloader')(**kws)


DATA_MODULES = edict({
    'stereo_sceneflow_driving': {
        'dm_class': SceneFlowStereoDataModule,
        'dm_kws': {
            'folder': 'sceneflow',
            'subset': 'stereo_sceneflow_driving',
        }
    },
    'stereo_drivingstereo': {
        'dm_class': DrivingStereoDataModule,
        'dm_kws': {
            'folder': 'drivingstereo',
            'subset': 'stereo_drivingstereo',
        }
    },
    'stereo_livox': {
        'dm_class': LivoxStereoDataModule,
        'dm_kws': {
            'folder': 'livox',
            'subset': 'stereo_livox',
        }
    },
    'monocular_livox': {
        'dm_class': LivoxMonocularDataModule,
        'dm_kws': {
            'folder': 'livox',
            'subset': 'monocular_livox',
        }
    },
    'monocular_boxdepth': {
        'dm_class': BoxDepthMonocularDataModule,
        'dm_kws': {
            'folder': 'boxdepth',
            'subset': 'monocular_boxdepth',
        }
    },
    'fusion_kitti': {
        'dm_class': KITTIFusionDataModule,
        'dm_kws': {
            'folder': 'kitti',
            'subset': 'fusion_kitti',
        }
    },
    'stereo_kitti': {
        'dm_class': KITTIStereoDataModule,
        'dm_kws': {
            'folder': 'kitti',
            'subset': 'stereo_kitti',
        }
    }
})

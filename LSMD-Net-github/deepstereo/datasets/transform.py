import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def get_transform(t):
    if isinstance(t, list):
        return A.Compose(t)
    return t

class StereoAugmentor:
    def __init__(self, color_sym_aug=None, color_asym_aug=None, geo_aug=None, post_transform=None, with_disp_tgt=True, with_slanted_plane=True,with_lidar_disp=False,with_lidar_depth=False):
        self.color_sym_aug = None
        if color_sym_aug:
            self.color_sym_aug = A.Compose(
                color_sym_aug,
                additional_targets={
                    'img_tgt': 'image',
                })

        self.color_asym_aug = get_transform(color_asym_aug)
        self.with_disp_tgt = with_disp_tgt
        self.with_slanted_plane = with_slanted_plane
        self.with_lidar_disp = with_lidar_disp
        self.with_lidar_depth = with_lidar_depth

        if isinstance(geo_aug, list):
            adds = {
                'img_tgt': 'image',
                'disp_ref': 'image',
            }

            if with_lidar_disp:
                adds.update({
                    'disp_input_ref': 'image',
                    'disp_input_tgt': 'image',
                })

            if with_lidar_depth:
                adds.update({
                    'depth_ref':'image',
                    'depth_input_ref': 'image',
                    'depth_input_tgt': 'image',
                })

            if with_slanted_plane:
                adds.update({
                    'slanted_plane_ref': 'image',
                })

            if with_disp_tgt:
                adds['disp_tgt'] = 'image'
                if with_slanted_plane:
                    adds.update({
                        'slanted_plane_tgt': 'image'
                    })

            geo_aug = A.Compose(
                geo_aug,
                additional_targets=adds)

        self.geo_aug = geo_aug
        self.post_transform = get_transform(post_transform)

    def __call__(self, x):
        img_ref = x['img_ref']
        img_tgt = x['img_tgt']
        disp_ref = x['disp_ref']

        geo_aug_kws = {
            'img_tgt': img_tgt,
            'disp_ref': disp_ref,
        }

        with_lidar_disp = self.with_lidar_disp and  'disp_input_ref' in x
        with_lidar_depth = self.with_lidar_depth and  'depth_input_ref' in x
        with_disp_tgt = self.with_disp_tgt and 'disp_tgt' in x
        with_slanted_plane_ref = self.with_slanted_plane and 'slanted_plane_ref' in x
        with_slanted_plane_tgt = self.with_disp_tgt and self.with_slanted_plane and 'slanted_plane_tgt' in x

        if with_lidar_disp:
            geo_aug_kws['disp_input_ref'] = x['disp_input_ref']
            geo_aug_kws['disp_input_tgt'] = x['disp_input_tgt']
        
        if with_lidar_depth:
            geo_aug_kws['depth_ref'] = x['depth_ref']
            geo_aug_kws['depth_input_ref'] = x['depth_input_ref']
            geo_aug_kws['depth_input_tgt'] = x['depth_input_tgt']
        
        if with_disp_tgt:
            geo_aug_kws['disp_tgt'] = x['disp_tgt']

        if with_slanted_plane_ref:
            geo_aug_kws['slanted_plane_ref'] = x['slanted_plane_ref']

        if with_slanted_plane_tgt:
            geo_aug_kws['slanted_plane_tgt'] = x['slanted_plane_tgt']

        o = x.copy()  

        if self.geo_aug:
            t = self.geo_aug(image=img_ref, **geo_aug_kws)

            o.update({
                'img_ref': t['image'],
                'img_tgt': t['img_tgt'],
                'disp_ref': t['disp_ref'],
            })

            if with_lidar_disp:
                o['disp_input_ref'] = t['disp_input_ref']
                o['disp_input_tgt'] = t['disp_input_tgt']
            
            if with_lidar_depth:
                o['depth_ref'] = t['depth_ref']
                o['depth_input_ref'] = t['depth_input_ref']
                o['depth_input_tgt'] = t['depth_input_tgt']

            if with_disp_tgt:
                o['disp_tgt'] = t['disp_tgt']

            if with_slanted_plane_ref:
                o['slanted_plane_ref'] = t['slanted_plane_ref']

            if with_slanted_plane_tgt:
                o['slanted_plane_tgt'] = t['slanted_plane_tgt']

        if self.color_sym_aug:
            t = self.color_sym_aug(image=o['img_ref'], img_tgt=o['img_tgt'])

            o.update({
                'img_ref': t['image'],
                'img_tgt': t['img_tgt'],
            })

        if self.color_asym_aug:
            o.update({
                'img_ref': self.color_asym_aug(image=o['img_ref'])['image'],
                'img_tgt': self.color_asym_aug(image=o['img_tgt'])['image'],
            })

        if self.post_transform:
            o.update({
                'img_ref': self.post_transform(image=o['img_ref'])['image'],
                'img_tgt': self.post_transform(image=o['img_tgt'])['image'],
            })

            '''if with_lidar_disp:
                o.update({
                'disp_input':torch.FloatTensor(o['disp_input']),
            })'''
        return o

class MonocularAugmentor:
    def __init__(self, color_sym_aug=None,color_asym_aug=None,geo_aug=None,post_transform=None,with_lidar_depth=False):
        self.color_sym_aug = None
        if color_sym_aug:
            self.color_sym_aug = A.Compose(
                color_sym_aug)

        self.color_asym_aug = get_transform(color_asym_aug)

        if isinstance(geo_aug, list):
            adds = {
                'img': 'image',
                'depth': 'image',
            }

            if with_lidar_depth:
                adds.update({
                    'depth_input': 'image',
                })

            geo_aug = A.Compose(
                geo_aug,
                additional_targets=adds)

        self.geo_aug = geo_aug
        self.post_transform = get_transform(post_transform)
        self.with_lidar_depth = with_lidar_depth

    def __call__(self, x):

        with_lidar_depth = self.with_lidar_depth and  'depth_input' in x

        img = x['img']
        depth = x['depth']

        geo_aug_kws = {
            'depth': depth
        }

        if with_lidar_depth:
            geo_aug_kws['depth_input'] = x['depth_input']

        o = x.copy()

        if self.geo_aug:
            t = self.geo_aug(image=img, **geo_aug_kws)

            o.update({
                'img': t['image'],
                'depth': t['depth'],
            })

            if with_lidar_depth:
                o['depth_input'] = t['depth_input']

        if self.color_sym_aug:
            t = self.color_sym_aug(image=o['img'])

            o.update({
                'img': t['image'],
            })

        if self.color_asym_aug:
            o.update({
                'img': self.color_asym_aug(image=o['img'])['image'],
            })

        if self.post_transform:
            o.update({
                'img': self.post_transform(image=o['img'])['image'],
                #'depth': torch.FloatTensor(o['depth']),
            })

        return o

img_transform = [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #A.Normalize(mean=(.0, 0, 0), std=(1., 1., 1.)),
    #A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
]

def make_pipeline_stereo_test(crop_size, to_tensor=True, **kws):
    post_transform = img_transform if to_tensor else None
    return StereoAugmentor(geo_aug=[
       A.CenterCrop(crop_size[1], crop_size[0])
    ], post_transform=post_transform, **kws)

def make_pipeline_stereo_val(crop_size, to_tensor=True, **kws):
    post_transform = img_transform if to_tensor else None
    return StereoAugmentor(geo_aug=[
       A.RandomCrop(crop_size[1], crop_size[0])
    ], post_transform=post_transform, **kws)

def make_pipeline_monocular_test(crop_size, to_tensor=True, **kws):
    post_transform = img_transform if to_tensor else None
    return MonocularAugmentor(geo_aug=[
        A.CenterCrop(crop_size[1], crop_size[0])
    ], post_transform=post_transform, **kws)

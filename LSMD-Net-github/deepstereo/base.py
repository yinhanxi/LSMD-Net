import sys
import os.path as osp
import importlib
from easydict import EasyDict as edict

def file_name(fname):
    return osp.splitext(osp.basename(fname))[0]

def import_module_from_file(fname):
    sys.path.insert(0, osp.abspath(osp.dirname(fname)))

    module_name = file_name(fname)
    # spec = importlib.util.spec_from_file_location(module_name, fname)
    # m = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(m)
    m = importlib.import_module(module_name)
    return m

def import_module_and_key(fname, key=None):
    parts = fname.split(':')
    module_f = parts[0]
    if not key and len(parts) <= 1:
        key = 'default'

    if not key:
        key = parts[-1]

    m = import_module_from_file(module_f)
    return m, key

def import_model(model_f, ds=None):
    m, net_type = import_module_and_key(model_f)
    if hasattr(m, 'MODELS'):
        models = m.MODELS
    else:
        assert hasattr(m, 'models')
        models = m.models

    cfg = models[net_type]
    if ds:
        cfg.update(cfg.data_modules[ds])

    cfg.net_type = net_type
    cfg.ds = ds
    return cfg

def import_data_module(filename, data_root, batch_size, ds=None, **kws):
    mod_ds, ds_tp  = import_module_and_key(filename, ds)
    ds_cfg = mod_ds.DATA_MODULES[ds_tp]
    cls = ds_cfg['dm_class']
    kws = {**ds_cfg['dm_kws'], **kws}
    return cls(data_root=data_root, batch_size=batch_size, **kws)

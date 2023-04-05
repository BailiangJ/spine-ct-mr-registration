import torch.nn as nn
from mmcv.utils import Registry

POOLING_LAYERS = Registry('pooling layer')

POOLING_LAYERS.register_module('MaxPool2d', module=nn.MaxPool2d)
POOLING_LAYERS.register_module('MaxPool3d', module=nn.MaxPool3d)
POOLING_LAYERS.register_module('AvgPool2d', module=nn.AvgPool2d)
POOLING_LAYERS.register_module('AvgPool3d', module=nn.AvgPool3d)


def build_pooling_layer(cfg: dict, *args, **kwargs) -> nn.Module:
    """Build pooling layer.

    Args:
        cfg (None or dict): The pooling layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a pooling layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding pooling layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding pooling layer.

    Returns:
        nn.Module: Created pooling layer.
    """
    if cfg is None:
        cfg_ = dict(type='AvgPool2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in POOLING_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        pooling_layer = POOLING_LAYERS.get(layer_type)

    layer = pooling_layer(*args, **kwargs, **cfg_)

    return layer

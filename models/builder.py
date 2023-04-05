from typing import Optional, Sequence, Union

import torch.nn as nn
from mmengine import MODELS as MMENGINE_MODELS
from mmengine import Registry, build_from_cfg
from torch.nn import Module

MODELS = Registry('models', parent=MMENGINE_MODELS)
LOSSES = MODELS
ENCODERS = MODELS
DECODERS = MODELS
FLOW_ESTIMATORS = MODELS
BACKBONES = MODELS


def build(cfg: Union[Sequence[dict], dict],
          registry: Registry,
          default_args: Optional[dict] = None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg: dict) -> Module:
    """Build loss function.

    Args:
        cfg (dict): Config for loss function.
    Returns:
        Module: Loss function.
    """
    return build(cfg, LOSSES)


def build_encoder(cfg: dict) -> Module:
    """Build encoder for flow estimator.

    Args:
        cfg (dict): Config for encoder.
    Returns:
        Module: Encoder module.
    """
    return build(cfg, ENCODERS)


def build_decoder(cfg: dict) -> Module:
    """Build decoder for flow estimator.

    Args:
        cfg (dict): Config for decoder.
    Returns:
        Module: Decoder module.
    """
    return build(cfg, DECODERS)


def build_flow_estimator(cfg: dict) -> Module:
    """Build flow estimator.

    Args:
        cfg (dict): Config for optical flow estimator.
    Returns:
        Module: Flow estimator.
    """
    return build(cfg, FLOW_ESTIMATORS)


def build_backbone(cfg: dict) -> Module:
    """Build backbone.

    Args:
        cfg (dict): Config for optical flow estimator.
    Returns:
        Module: Backbone.
    """
    return build(cfg, BACKBONES)

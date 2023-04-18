from typing import Optional, Sequence, Union

import torch.nn as nn
from mmengine import MODELS as MMENGINE_MODELS
from mmengine import Config, ConfigDict, Registry, build_from_cfg
from torch.nn import Module

MODELS = Registry('models', parent=MMENGINE_MODELS)
LOSSES = MODELS
ENCODERS = MODELS
DECODERS = MODELS
FLOW_ESTIMATORS = MODELS
BACKBONES = MODELS
METRICS = MODELS
REGISTRATION_HEAD = MODELS

CFG = Union[dict, Config, ConfigDict]


def build(cfg: Union[Sequence[CFG], CFG],
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
    cfg_ = cfg.copy()
    cfg_.pop('weight', None)
    return build(cfg_, LOSSES)


def build_metrics(cfg: dict) -> Module:
    """Build metric function.

    Args:
        cfg (dict): Config for encoder.
    Returns:
        Module: Metric function.
    """
    return build(cfg, METRICS)


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


def build_registration_head(cfg: dict) -> Module:
    """Build registration head.

    Args:
        cfg (dict): Config for registration head.
    Returns:
        Module: Registration head.
    """
    return build(cfg, REGISTRATION_HEAD)

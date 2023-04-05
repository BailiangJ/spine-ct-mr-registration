from typing import Optional, Sequence, Union

import torch.nn as nn
from mmcv.utils import Registry
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import CONV_LAYERS

CONV_LAYERS.register_module('ConvTranspose2d', module=nn.ConvTranspose2d)
CONV_LAYERS.register_module('ConvTranspose3d', module=nn.ConvTranspose3d)

UPSAMPLE_LAYERS = Registry('upsample layer')
UPSAMPLE_LAYERS.register_module('Upsample', module=nn.Upsample)


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(BaseModule):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
        scale_factor (int): Scaling factor of upsampling. Default: 2.
        conv_cfg (dict): Config dict for convolution layer.
            Default: dict(type='ConvTranspose2d').
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 scale_factor: int = 2,
                 conv_cfg: dict = dict(type='ConvTranspose2d'),
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[Union[dict, list]] = None,
                 ) -> None:
        super(DeconvModule, self).__init__(init_cfg)

        assert (kernel_size - scale_factor >= 0) and \
               (kernel_size - scale_factor) % 2 == 0, \
            f'kernel_size should be greater than or equal to scale_factor ' \
            f'and (kernel_size - scale_factor) should be even numbers, ' \
            f'while the kernel size is {kernel_size} and scale_factor is ' \
            f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.deconv_upsamping = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.deconv_upsamping(x)
        return out


@UPSAMPLE_LAYERS.register_module()
class InterpConv(BaseModule):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(type='Upsample',
                scale_factor=2, mode='bilinear', align_corners=False).
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 upsample_cfg: dict = dict(type="Upsample",
                                           scale_factor=2, mode='bilinear', align_corners=False),
                 conv_first: bool = False,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[Union[dict, list]] = None,
                 ) -> None:
        super(InterpConv, self).__init__(init_cfg)

        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        upsample = build_upsample_layer(upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.interp_upsample(x)
        return out


def build_upsample_layer(cfg, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        upsample = UPSAMPLE_LAYERS.get(layer_type)

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer

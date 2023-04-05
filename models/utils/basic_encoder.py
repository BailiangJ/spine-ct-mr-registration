from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .pooling import build_pooling_layer
from ..builder import ENCODERS


class BasicConvBlock(BaseModule):
    """Basic convolution block

    This module consists of several plain convolution layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolution layers. Defaults: 3.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolution layer to downsample the input feature
            map. Options are 1 or 2. Defaults: 1.
        dilation (int): Whether use dialted convolution to expand the
            receptive field. Set dilation rate of each convolution layer and
            the dilation rate of the first convolution layer is always 1.
            Default: 1.
        kernel_size (int): Kernel size of each feature level. Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: None.
        pool_cfg (dict, optional): Config dict for pooling layer after
            conv block. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Union[int, Sequence[int]],
                 num_convs: int = 1,
                 stride: int = 1,
                 dilation: int = 1,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 pool_cfg: Optional[dict] = None,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super(BasicConvBlock, self).__init__(init_cfg)
        if pool_cfg and stride == 2:
            print("using both pooling and stride conv for downsampling.")

        convs = []
        in_channels = in_channels
        for i in range(num_convs):
            k = kernel_size[i] if isinstance(kernel_size, (tuple, list)) else kernel_size
            out_ch = out_channels[i] if isinstance(out_channels, (tuple, list)) else out_channels

            # first pooling then conv, level 0 doesn't do pooling in encoder
            # better for the skip connection
            if pool_cfg:
                convs.append(build_pooling_layer(pool_cfg))

            convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=k // 2 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channels = out_ch

        self.layers = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out

@ENCODERS.register_module()
class BasicEncoder(BaseModule):
    """A basic pyramid feature extraction sub-network for UNet.

    Args:
        in_channels (int): Number of input channels.
        pyramid_levels (Sequence[str]): The list of feature pyramid that are
            the keys for output dict.
        num_convs (Sequence[int]): Numbers of conv layers for each
            pyramid level.
        out_channels (Sequence[int]): List of numbers of output
            channels of each pyramid level.
        strides (Sequence[int]): List of strides of each pyramid level.
            If it use pooling layer for downsampling, they are all 1.
            Default: (1,).
        dilations (Sequence[int]): List of dilation of each pyramid level.
            Default: (1,).
        kernel_size (Sequence[int], int): Kernel size of each feature
            level. Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict): Config dict for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.2).
        pool_cfg (dict, optional): Config dict for pooling layer.
            Default: dict(type='MaxPool2d', kernel_size=2).
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 pyramid_levels: Sequence[str],
                 num_convs: Sequence[int],
                 out_channels: Sequence[int],
                 strides: Sequence[int] = (1,),
                 dilations: Sequence[int] = (1,),
                 kernel_size: Union[Sequence[int], int] = 3,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.2),
                 pool_cfg: Optional[dict] = dict(type='MaxPool2d', kernel_size=2),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super(BasicEncoder).__init__(init_cfg)

        assert len(out_channels) == len(num_convs) == len(strides) == len(
            dilations) == len(pyramid_levels)
        if pool_cfg and any((s == 2 for s in strides)):
            print("using both pooling and stride conv for downsampling.")
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.strides = strides
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pool_cfg = pool_cfg

        self.encoder = nn.ModuleList()
        pool_cfg_ = None
        for i in range(len(out_channels)):
            if isinstance(self.kernel_size, (list, tuple)) and len(
                    self.kernel_size) == len(out_channels):
                kernel_size_ = self.kernel_size[i]
            elif isinstance(self.kernel_size, int):
                kernel_size_ = self.kernel_size
            else:
                TypeError('kernel_size must be list, tuple or int, '
                          f'but got {type(kernel_size)}')

            if i != 0 and self.pool_cfg:
                pool_cfg_ = self.pool_cfg

            self.encoder.append(
                self._make_layer(
                    in_channels,
                    out_channels[i],
                    num_convs[i],
                    strides[i],
                    dilations[i],
                    kernel_size=kernel_size_,
                    pool_cfg=pool_cfg_))
            in_channels = out_channels[i][-1] if isinstance(
                out_channels[i], (tuple, list)) else out_channels[i]

    def _make_layer(self,
                    in_channels: int,
                    out_channels: int,
                    num_convs: int,
                    stride: int,
                    dilation: int,
                    kernel_size: int = 3,
                    pool_cfg: Optional[dict] = None) -> nn.Module:
        return BasicConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            pool_cfg=pool_cfg,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x: torch.Tensor) -> list:
        """Forward function for BasicEncoder.
        Args:
            x (Tensor): The input data.
        Returns:
            dict: The feature pyramid extracted from input data.
        """
        skips = [x]
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
        return skips

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .upsample import UPSAMPLE_LAYERS, build_upsample_layer
from .basic_encoder import BasicConvBlock
from ..builder import DECODERS


class UpConvBlock(BaseModule):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
            high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        kernel_size (int): Kernel size of the convolutional layer. Default: 3.
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='Upsample',mode='bilinear',scale_factor=2).
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
                 conv_block: nn.Sequential,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: Union[int, Sequence[int]],
                 num_convs: int = 2,
                 stride: int = 1,
                 dilation: int = 1,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 upsample_cfg: dict = dict(type='Upsample', mode='bilinear', scale_factor=2),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 init_cfg: Optional[Union[dict, list]] = None,
                 ) -> None:
        super(UpConvBlock, self).__init__(init_cfg)

        self.conv_block = conv_block(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.upsample = build_upsample_layer(
            cfg=upsample_cfg
        )

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out

@DECODERS.register_module()
class BasicDecoder(BaseModule):
    """A basic pyramid feature extraction sub-network for UNet.

    Args:
        in_channels (int): Number of input channels.
        skip_channels (Sequence[int]): List of numbers of skip channels for each
            pyramid level.
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
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='Upsample',mode='bilinear',scale_factor=2).
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Default: None.
        act_cfg (dict): Config dict for each activation layer in ConvModule.
            Default: dict(type='LeakyReLU', negative_slope=0.2).
        init_cfg (dict, list, optional): Config for module initialization.
    """

    def __init__(self,
                 in_channels: int,
                 skip_channels: Sequence[int],
                 pyramid_levels: Sequence[str],
                 num_convs: Sequence[int],
                 out_channels: Sequence[int],
                 strides: Sequence[int] = (1,),
                 dilations: Sequence[int] = (1,),
                 kernel_size: Union[Sequence[int], int] = 3,
                 upsample_cfg: dict = dict(type='Upsample', mode='bilinear', scale_factor=2),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.2),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super(BasicDecoder).__init__(init_cfg)

        assert len(skip_channels) == len(out_channels) == len(num_convs) == len(strides) \
               == len(dilations) == len(pyramid_levels)

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.pyramid_levels = pyramid_levels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.strides = strides
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.upsample_cfg = upsample_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.decoder = nn.ModuleList()
        for i in range(len(out_channels)):
            if isinstance(self.kernel_size, (list, tuple)) and len(
                    self.kernel_size) == len(out_channels):
                kernel_size_ = self.kernel_size[i]
            elif isinstance(self.kernel_size, int):
                kernel_size_ = self.kernel_size
            else:
                TypeError('kernel_size must be list, tuple or int, '
                          f'but got {type(kernel_size)}')

            self.decoder.append(
                self._make_layer(
                    in_channels,
                    skip_channels[i],
                    out_channels[i],
                    num_convs[i],
                    strides[i],
                    dilations[i],
                    kernel_size=kernel_size_,
                )
            )
            in_channels = out_channels[i][-1] if isinstance(
                out_channels[i], (tuple, list)) else out_channels[i]

    def _make_layer(self,
                    in_channels: int,
                    skip_channels: int,
                    out_channels: int,
                    num_convs: int,
                    stride: int,
                    dilation: int,
                    kernel_size: int = 3, ) -> nn.Module:
        return UpConvBlock(
            conv_block=BasicConvBlock,
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_size,
            upsample_cfg=self.upsample_cfg,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        """Forward function for BasicDecoder"""
        x = skips.pop()
        for i, dec in enumerate(self.decoder):
            x = dec(skips.pop(), x)
        return x

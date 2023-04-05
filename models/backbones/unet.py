import warnings
from typing import Optional

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES, build_decoder, build_encoder


@BACKBONES.register_module()
class UNet(BaseModule):
    """UNet backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.
    Args:
        enc_cfg (dict): Config dict for encoder.
        dec_cfg (dict): Config dict for decoder.
        remain_cfg (dict): Config dict for remaining convolutions.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 enc_cfg: dict,
                 dec_cfg: dict,
                 remain_cfg: dict,
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = None):
        super(UNet, self).__init__(init_cfg)

        self.norm_eval = norm_eval
        self.encoder = build_encoder(enc_cfg)
        self.decoder = build_decoder(dec_cfg)
        self.remain = build_encoder(remain_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        out = self.decoder(skips)
        outs = self.remain(out)
        return outs.pop()

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

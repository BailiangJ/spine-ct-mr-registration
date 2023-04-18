from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import FLOW_ESTIMATORS, build_backbone, build_encoder
from ..utils import BasicConvBlock


@FLOW_ESTIMATORS.register_module()
class VoxelMorph(BaseModule):
    """VoxelMorhp model.

    Args:
        unet_cfg (dict): Config dict for UNet.
        flow_conv_cfg (dict): Config dict for flow convolution.
        init_cfg (dict, optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                 unet_cfg: dict,
                 flow_conv_cfg: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(VoxelMorph, self).__init__(init_cfg)
        self.unet = build_backbone(unet_cfg)
        self.flow_conv = BasicConvBlock(**flow_conv_cfg)

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """

        Args:
            source (torch.Tensor): source image of shape [batch_size, channels, ...].
            target (torch.Tensor): target image of shape [batch_size, channels, ...].

        Returns:
            flow (torch.Tensor): output flow field of shape [batch_size, spatial_dims, ...]
        """

        # concatenate source and target images
        x = torch.cat([source, target], dim=1)
        x = self.unet(x)

        # generate flow field
        flow = self.flow_conv(x)
        return flow

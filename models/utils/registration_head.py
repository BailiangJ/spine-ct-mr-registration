from typing import Sequence, Union, Optional

import torch
import torch.nn as nn
from ..builder import REGISTRATION_HEAD
from .resize_flow import ResizeFlow
from .integrate import VecIntegrate
from .warp import Warp


@REGISTRATION_HEAD.register_module()
class RegistrationHead(nn.Module):
    """Registration head for voxelmorph.

    Args:
        image_size (Sequence[int]): size of input image.
        int_steps (int): number of steps in scaling and squaring integration. (Default: 0, which means no integration.)
        resize_scale (int): scale factor of flow field. (The output flow field might be half resolution.
            Implicit requirement, image size should be divisible by resize scale factor.)
        resize_first (bool): whether to resize before integration, only matters when int_steps>0. (Default: False)
        bidir (bool): whether to run bidirectional registration, only matters when int_steps>0. (Default: False)
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self,
                 image_size: Sequence[int],
                 int_steps: int = 0,
                 resize_scale: int = 2,
                 resize_first: bool = False,
                 bidir: bool = False,
                 interp_mode: str = "bilinear") -> None:
        super().__init__()
        if int_steps > 0:
            flow_size = [s / resize_scale for s in image_size]
            self.integrate = VecIntegrate(flow_size, int_steps, interp_mode)
        else:
            self.integrate = None

        self.resize_flow = ResizeFlow(resize_scale, ndim=len(image_size))
        self.resize_first = resize_first
        self.bidir = bidir if int_steps > 0 else False
        self.warp = Warp(image_size, interp_mode)

    def forward(self,
                vec_flow: torch.Tensor,
                source: torch.Tensor,
                target: torch.Tensor,
                source_oh: Optional[torch.Tensor] = None,
                target_oh: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        """
        Args:
            vec_flow (torch.Tensor): flow field predicted by network.
            source (torch.Tensor): source image, tensor of shape [BCHWD].
            target (torch.Tensor): target image, tensor of shape [BCHWD].
            source_oh (torch.Tensor|None): one-hot segmentation label of source image,
                tensor of shape [BCHWD]. (Default: None)
            target_oh (torch.Tensor|None): one-hot segmentation label of target image,
                tensor of shape [BCHWD]. (Default: None)

        """
        if self.resize_first:
            vec_flow = self.resize_flow(vec_flow)

        # integrate by scaling and squaring to generate diffeomorphic flow
        fwd_flow = self.integrate(vec_flow) if self.integrate else vec_flow
        bck_flow = self.integrate(-vec_flow) if self.bidir else None

        if not self.resize_first:
            fwd_flow = self.resize_flow(fwd_flow)
            bck_flow = self.resize_flow(bck_flow) if bck_flow is not None else None

        # warp image with displacement field
        y_source = self.warp(source, fwd_flow)
        y_target = self.warp(target, bck_flow) if self.bidir else None

        # warp one-hot label with displacement field
        y_source_oh = self.warp(source_oh, fwd_flow) if source_oh is not None else None
        y_target_oh = self.warp(target_oh, bck_flow) if (self.bidir and target_oh is not None) else None

        return fwd_flow, bck_flow, y_source, y_target, y_source_oh, y_target_oh

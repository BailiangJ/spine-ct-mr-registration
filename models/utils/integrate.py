from typing import Sequence

import torch
from torch import nn
from .warp import Warp

class VecIntegrate(nn.Module):
    """
    Integration of Stationary Velocity Field (SVF) using Scaling and Squaring.

    Args:
        image_size (Sequence[int]): size of input image.
        num_steps (int): number of integration steps.
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """
    def __init__(self,image_size:Sequence[int], num_steps:int, interp_mode:str="bilinear")->None:
        super().__init__()

        assert num_steps>=0, f'num_steps should be >= 0, found:{num_steps}'
        self.num_steps = num_steps
        self.warp_layer = Warp(image_size=image_size, interp_mode=interp_mode)

    def forward(self, vec:torch.Tensor)->torch.Tensor:
        ddf: torch.Tensor = vec / (2.0**self.num_steps)
        for _ in range(self.num_steps):
            ddf = ddf + self.warp_layer(image=ddf, flow=ddf)
        return ddf
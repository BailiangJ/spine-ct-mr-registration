from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ResizeFlow(nn.Module):
    """
    Resize and rescale a flow field.

    Args:
        scale_factor (float): scaling factor of resizing.
        ndim (int): number of dimensions.
    """

    def __init__(self, scale_factor: float, ndim: int):
        super().__init__()
        self.scale_factor = scale_factor
        if ndim == 2:
            self.interp_mode = 'bilinear'
        elif ndim == 3:
            self.interp_mode = 'trilinear'
        else:
            raise KeyError(f'Unsupported ndim for ResizeFlow:{ndim}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.scale_factor, recompute_scale_factor=True,
                              mode=self.interp_mode)
            x = self.scale_factor * x

        elif self.scale_factor > 1:
            # multiply first to save memory
            x = self.scale_factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.scale_factor, recompute_scale_factor=True,
                              mode=self.interp_mode)

        return x

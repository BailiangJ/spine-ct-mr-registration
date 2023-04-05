from typing import Sequence, List

import torch
from torch import nn
import torch.nn.functional as F


class Warp(nn.Module):
    """
    Warp an image with given flow / dense displacement field (DDF).

    Args:
        image_size (Sequence[int]): size of input image.
        interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
    """

    def __init__(self, image_size: Sequence[int], interp_mode: str = 'bilinear') -> None:
        super().__init__()

        self.ndim = len(image_size)
        self.image_size = image_size
        self.interp_mode = interp_mode

        # create reference grid
        grid = self.get_reference_grid(image_size)
        grid = grid.unsqueeze(0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    @staticmethod
    def get_reference_grid(image_size: Sequence[int]) -> torch.Tensor:
        """
        Generate unnormalized reference coordinate grid.
        Args:
            image_size (Sequence[int]): size of input image

        Returns:
            grid: torch.FloatTensor

        """
        mesh_points = [torch.arange(0, dim, dtype=torch.float) for dim in image_size]
        grid = torch.stack(torch.meshgrid(*mesh_points),
                           dim=0)  # (spatial_dims, ...)
        return grid

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image with flow.
        Args:
            image (torch.Tensor): input image of shape [batch_size, channels, ...]
            flow (torch.Tensor): flow field of shape [batch_size, spatial_dims, ...]

        Returns:
            torch.Tensor: Warped image.
        """
        assert self.image_size == image.shape[2:] == flow.shape[2:]

        # deformation
        sample_grid = self.grid + flow

        # normalize
        # F.grid_sample takes normalized grid with range of [-1,1]
        for i, dim in enumerate(self.image_size):
            sample_grid[..., i] = sample_grid[..., i] * 2 / (dim - 1) - 1

        index_ordering: List[int] = list(range(self.ndim - 1, -1, -1))
        # F.grid_sample takes grid in a reverse order
        sample_grid = sample_grid[..., index_ordering]  # x,y,z -> z,y,x

        return F.grid_sample(image, sample_grid, align_corners=True, mode=self.interp_mode)

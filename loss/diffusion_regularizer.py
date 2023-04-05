import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Union


def spatial_gradient(x: torch.Tensor, dim: int, mode: str = 'forward') -> torch.Tensor:
    """
        Calculate gradients on single dimension of a tensor using central finite difference.
        It moves the tensor along the dimension to calculate the approximate gradient
        dx[i] = (x[i+1] - x[i-1]) / 2.
        or forward/backward finite difference
        dx[i] = x[i+1] - x[i]

        Adopted from:
            Project-MONAI (https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/deform.py)
        Args:
            x: the shape should be BCH(WD).
            dim: dimension to calculate gradient along.
            mode: flag deciding whether to use central or forward finite difference,
                    ['forward','central']
        Returns:
            gradient_dx: the shape should be
    """

    slice_all = slice(None)
    slicing_s, slicing_e = [slice_all] * x.ndim, [slice_all] * x.ndim
    if mode == 'central':
        slicing_s[dim] = slice(2, None)
        slicing_e[dim] = slice(None, -2)
        return (x[slicing_s] - x[slicing_e]) / 2.0
    elif mode == 'forward':
        slicing_s[dim] = slice(1, None)
        slicing_e[dim] = slice(None, -1)
        return x[slicing_s] - x[slicing_e]
    else:
        raise ValueError(f'Unsupported finite difference method: {mode}, available options are ["forward", "central"].')


class GradientDiffusionLoss(_Loss):
    """
    Calculate the diffusion regularizer (smoothness regularizer) on the spatial gradients of displacement/velocity field.

    """

    def __init__(self, penalty: Union[int, str] = 'l1', loss_mult: Union[float, None] = None) -> None:
        """
        Args:
            penalty: flag decide l1/l2 norm of diffusion to compute
            loss_mult: loss multiplier depending on the downsize of displacement/velocity field, loss_mult = int_downsize
        """
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, _, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BCH(WD)
        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(f"Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}")
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 4:
                raise ValueError(f"All spatial dimensions must be > 4, got spatial dimensions {pred.shape[2:]}")
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f"Number of vector components, {pred.shape[1]}, does not match number of spatial dimensions, {pred.ndim - 2}"
            )

        #TODO: forward mode and central mode cause different result, the reason is still unknown

        # Using forward mode to be consistent with voxelmorph paper
        first_order_gradient = [spatial_gradient(pred, dim, mode='forward') for dim in range(2, pred.ndim)]

        loss = torch.tensor(0).float().to(pred.device)
        for dim, g in enumerate(first_order_gradient):
            if self.penalty == 'l1':
                loss += torch.mean(torch.abs(first_order_gradient[dim]))
            elif self.penalty == 'l2':
                loss += torch.mean(first_order_gradient[dim] ** 2)
            else:
                raise ValueError(f'Unsupported norm: {self.penalty}, available options are ["l1","l2"].')

        if self.loss_mult is not None:
            loss *= self.loss_mult
        return loss / float(pred.ndim - 2)

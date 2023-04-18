from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..builder import LOSSES
from .kernels import (gradient_kernel_1d, gradient_kernel_2d,
                      gradient_kernel_3d, spatial_filter_nd)


def _grad_param(ndim, method, axis):
    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.tensor(kernel, dtype=torch.float32))


# TODO: extend to flexible batch_size and 2D
@LOSSES.register_module('oc_pc')
class OrthonormalPropernessCondition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_x_kernel = _grad_param(3, method='default', axis=0)
        self.grad_y_kernel = _grad_param(3, method='default', axis=1)
        self.grad_z_kernel = _grad_param(3, method='default', axis=2)
        self.grad_x_kernel.requires_grad = False
        self.grad_y_kernel.requires_grad = False
        self.grad_z_kernel.requires_grad = False

    def first_order_derivative(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disp: the shape should be BCHWD, with B=1, C=3
        """
        # (3,1,1,H,W,D)
        gradx = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_x_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_x_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_x_kernel, mode='constant')
        ],
            dim=0)
        grady = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_y_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_y_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_y_kernel, mode='constant')
        ],
            dim=0)
        gradz = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_z_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_z_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_z_kernel, mode='constant')
        ],
            dim=0)

        # (3,3,1,H,W,D)
        grad_disp = torch.cat([gradx, grady, gradz], dim=1)
        # [dphi_x/dx, dphi_x/dy, dphi_x/dz]
        # [dphi_y/dx, dphi_y/dy, dphi_y/dz]
        # [dphi_z/dx, dphi_z/dy, dphi_z/dz]
        grad_deform = grad_disp + torch.eye(3, 3).view(3, 3, 1, 1, 1,
                                                       1).to(disp)

        return grad_deform

    def forward(self,
                y_source_oh: torch.Tensor,
                source_oh: None,
                disp_field: torch.Tensor,
                neg_flow: None) -> Sequence[torch.Tensor]:
        """
        Compute the orthonormality condition of displacement field
        Args:
            y_source_oh (torch.tensor): (hard) one-hot format, the shape should be BNHW[D]
            disp_field (torch.tensor): the shape should be BCHWD, with B=1, C=3

        Returns:
            E_oc (torch.tensor): the orthonormality condition energy
            E_pc (torch.tensor): the properness condition energy
        """
        # exclude background
        y_source_oh = y_source_oh[:, 1:]
        y_source_oh = y_source_oh.squeeze().sum(dim=0)
        # neg_y_source_oh = 1 - y_source_oh

        grad_deform = self.first_order_derivative(disp_field)

        # compute the Jacobian determinant
        pc = grad_deform[0, 0, ...] * (
                grad_deform[1, 1, ...] * grad_deform[2, 2, ...] - grad_deform[1, 2, ...] * grad_deform[2, 1, ...]) - \
             grad_deform[1, 0, ...] * (
                     grad_deform[0, 1, ...] * grad_deform[2, 2, ...] - grad_deform[0, 2, ...] * grad_deform[
                 2, 1, ...]) + \
             grad_deform[2, 0, ...] * (
                     grad_deform[0, 1, ...] * grad_deform[1, 2, ...] - grad_deform[0, 2, ...] * grad_deform[
                 1, 1, ...])

        pc = pc.squeeze()
        E_pc = torch.sum((pc - 1) ** 2 * y_source_oh) / y_source_oh.sum()

        # compute the inner product of the Jacobian
        # which should be identity accroding to orthonormality
        oc = torch.einsum('kilhwd,kjlhwd->ijlhwd', grad_deform, grad_deform)

        # (9,1,H,W,D)
        oc = oc.flatten(start_dim=0, end_dim=1) - torch.eye(3, 3).view(
            9, 1, 1, 1, 1).to(disp_field)
        # (1,H,W,D)
        E_oc = torch.sum(oc ** 2, dim=0)
        E_oc = E_oc.squeeze()

        E_oc = torch.sum(E_oc * y_source_oh) / y_source_oh.sum()

        return E_oc, E_pc

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from torch import nn
from torch.nn.parameter import Parameter


def get_reference_grid(image_size: Sequence[int]) -> torch.Tensor:
    """
    Generate a unnormalized coordinate grid
    Args:
        image_size: shape of input image, e.g. (64,128,128)
    """
    mesh_points = [torch.arange(0, dim) for dim in image_size]
    grid = torch.stack(torch.meshgrid(*mesh_points),
                       dim=0).to(dtype=torch.float)  # (spatial_dims, ...)
    return grid


def solve_SVD(fixed_pnts: torch.Tensor, moving_pnts: torch.Tensor,
              fixed_cm: torch.Tensor, moving_cm: torch.Tensor):
    """
    Solve rigid motion using least suqare method with SVD
    Args:
        fixed_pnts: torch.Tensor, points in fixed image
        moving_pnts: torch.Tensor, corresponding points in moving image
        fixed_cm: torch.Tensor, center mass of fixed image
        moving_cm: torch.Tensor, center mass of moving image

    Returns:
        R: torch.Tensor, rotation matrix
        t: torch.Tensor, translation vector
    """
    # demean the point clouds
    fixed_cm_rw = fixed_pnts - fixed_cm
    moving_cm_rw = moving_pnts - moving_cm

    # solve rotation and translation with SVD
    H = fixed_cm_rw @ moving_cm_rw.T
    U, S, Vt = torch.linalg.svd(H, full_matrices=True)

    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ fixed_cm + moving_cm
    return R, t


def sample_correspondence(label_map: torch.Tensor,
                          flow: torch.Tensor) -> torch.Tensor:
    """
    Sample correspondence between fixed and moving images
    Args:
        label_map: (soft) one-hot label mask of fixed image, tensor of shape BNHWD, with B=1
        flow: dense displacement field mapping from fixed image to moving image

    Returns:
        two corresponding point clouds
        src_pnts_list:
        dst_pnts_list:
    """
    num_ch = label_map.shape[1]
    src_pnts_list = []
    des_pnts_list = []
    for ch in range(num_ch):
        # sample a set of points in (soft) one-hot label of fixed image
        valid_points = (label_map[0, ch] >= 0.5).nonzero()

        valid_len = valid_points.shape[0]
        indices = torch.randint(valid_len, [self.num_samples]).to(self._device)

        src_pnts = torch.index_select(valid_points.float(), 0, indices)

        sample_grid = src_pnts.detach().clone()

        sample_flow = sample_displacement_flow(sample_grid, flow,
                                               self._image_size)
        sample_flow = sample_flow.squeeze()

        # (3, num_samples)
        src_pnts = src_pnts.transpose(0, 1)
        des_pnts = src_pnts + sample_flow

        src_pnts_list.append(src_pnts)
        des_pnts_list.append(des_pnts)

    return src_pnts_list, des_pnts_list


def sample_displacement_flow(sample_grid: torch.Tensor, flow: torch.Tensor,
                             image_size: Sequence[int]) -> torch.Tensor:
    """Sample 3D displacement flow at certain locations.

    TODO: adapt it to be compatiable with 2D images

    reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    Args:
        sample_grid: torch.Tensor, shape (num_samples, 3), coordinate of sampling locations
        flow: torch.Tensor, shape 3,HWD, dense displacement field
        image_size: shape of input image

    Returns:
        sample_flow: torch.Tensor, sampled displacement vectors
    """
    _dim = len(image_size)

    # normalize
    # F.grid_sample takes normalized grid with range at [-1,1]
    for i, dim in enumerate(image_size):
        sample_grid[..., i] = sample_grid[..., i] * 2 / (dim - 1) - 1

    index_ordering: List[int] = list(range(_dim - 1, -1, -1))
    # F.grid_sample takes grid in a different order
    sample_grid = sample_grid[..., index_ordering]  # x,y,z -> z,y,x
    # reshape to (1,1,1,num_samples,3) for grid_sample
    sample_grid = sample_grid[None, None, None, ...]

    sample_flow = F.grid_sample(flow,
                                sample_grid,
                                mode='nearest',
                                padding_mode='zeros',
                                align_corners=True)
    return sample_flow


def get_mass_center(label_map: torch.Tensor, grid: torch.Tensor,
                    dim: int) -> torch.Tensor:
    """
    Get the mass center of one-hot mask
    Args:
        label_map: tensor of shape BNHWD, with B=1, one-hot label mask
        grid: tensor of shape 3HWD, reference grid of image
        dim: int, number of dimensions, 3 in our usage
    Returns:
        center_mass: tensor of shape (3,N)

    """
    label_map = label_map.squeeze(0)
    intensity_sum = torch.sum(label_map, dim=list(range(1, dim + 1)))
    # center_mass_i shape N
    center_mass_x = torch.sum(label_map * grid[0, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass_y = torch.sum(label_map * grid[1, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass_z = torch.sum(label_map * grid[2, ...],
                              dim=list(range(1, dim + 1))) / intensity_sum
    center_mass = torch.stack([center_mass_x, center_mass_y, center_mass_z],
                              dim=0)
    return center_mass


class RigidTransformation(nn.Module):
    """Rigid centered transformation for 3D."""
    def __init__(self,
                 moving_image,
                 opt_cm=False,
                 num_samples=256,
                 dtype=torch.float32,
                 device='cpu'):
        """
        Args:
            moving_image: tensor of shape BNHWD, with B=1
        """
        super().__init__()
        self.moving_image = moving_image
        moving_image = moving_image.squeeze(0)
        self._num_ch = moving_image.shape[0]
        self._image_size = moving_image.shape[1:]
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        self.opt_cm = opt_cm
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        self.register_buffer('grid', grid)

        self.center_mass_x, self.center_mass_y, self.center_mass_z = get_mass_center(
            moving_image, self.grid, self._dim)

        self.phi_x = Parameter(torch.tensor([0.0] * self._num_ch))
        self.phi_y = Parameter(torch.tensor([0.0] * self._num_ch))
        self.phi_z = Parameter(torch.tensor([0.0] * self._num_ch))

        self.t_x = Parameter(torch.tensor([0.0] * self._num_ch))
        self.t_y = Parameter(torch.tensor([0.0] * self._num_ch))
        self.t_z = Parameter(torch.tensor([0.0] * self._num_ch))

        self.num_samples = num_samples

    def init_translation(self, fixed_image):
        fixed_image = fixed_image.squeeze(0)
        assert fixed_image.shape[1:] == self._image_size

        fixed_image_center_mass_x, fixed_image_center_mass_y, fixed_image_center_mass_z = get_mass_center(
            fixed_image, self.grid, self._dim)

        self.t_x = Parameter(self.center_mass_x - fixed_image_center_mass_x)
        self.t_y = Parameter(self.center_mass_y - fixed_image_center_mass_y)
        self.t_z = Parameter(self.center_mass_z - fixed_image_center_mass_z)
        # print(f"tx, ty, tz: {self.t_x, self.t_y, self.t_z}")

    def init_transform(self, fixed_image, flow):
        """
        Initialize rotation and translation from
            Least Square Method solving Rigid motion from correspondences
            https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        Args:
            fixed_image: tensor of shape BNHWD, with B=1
            flow: dense displacement field mapping from fixed_image to moving_image

        Returns:

        """
        fixed_pnts_list, moving_pnts_list = sample_correspondence(
            fixed_image, flow)

        fixed_image = fixed_image.squeeze(0)
        assert fixed_image.shape[1:] == self._image_size
        fixed_cm_list = get_mass_center(fixed_image, self.grid, self._dim)

        moving_cm_list = torch.stack(
            [self.center_mass_x, self.center_mass_y, self.center_mass_z],
            dim=0)

        phi_x_list = []
        phi_y_list = []
        phi_z_list = []
        t_x_list = []
        t_y_list = []
        t_z_list = []

        for ch in range(self._num_ch):
            fixed_pnts = fixed_pnts_list[ch]
            moving_pnts = moving_pnts_list[ch]

            fixed_cm = fixed_cm_list[:, [ch]]
            moving_cm = moving_cm_list[:, [ch]]

            R, t = solve_SVD(fixed_pnts, moving_pnts, fixed_cm, moving_cm)
            t_x, t_y, t_z = t.squeeze()

            rotation = Rot.from_matrix(R.cpu())
            phi_x, phi_y, phi_z = rotation.as_euler('xyz')

            phi_x_list.append(phi_x)
            phi_y_list.append(phi_y)
            phi_z_list.append(phi_z)

            t_x_list.append(t_x)
            t_y_list.append(t_y)
            t_z_list.append(t_z)

        self.phi_x = Parameter(torch.tensor(phi_x_list))
        self.phi_y = Parameter(torch.tensor(phi_y_list))
        self.phi_z = Parameter(torch.tensor(phi_z_list))

        self.t_x = Parameter(torch.tensor(t_x_list))
        self.t_y = Parameter(torch.tensor(t_y_list))
        self.t_z = Parameter(torch.tensor(t_z_list))

    def _compute_transformation_3d(self):
        self.trans_matrix_pos = torch.diag(torch.ones(self._dim + 1)).repeat(
            self._num_ch, 1, 1).to(dtype=self._dtype, device=self._device)
        rotation_matrix = torch.zeros(self._dim + 1, self._dim + 1)
        rotation_matrix[-1, -1] = 1
        self.rotation_matrix = rotation_matrix.repeat(self._num_ch, 1, 1).to(
            dtype=self._dtype, device=self._device)

        self.trans_matrix_pos[:, 0, 3] = self.t_x
        self.trans_matrix_pos[:, 1, 3] = self.t_y
        self.trans_matrix_pos[:, 2, 3] = self.t_z

        R_x = torch.diag(torch.ones(self._dim + 1)).repeat(
            self._num_ch, 1, 1).to(dtype=self._dtype, device=self._device)
        R_x[:, 1, 1] = torch.cos(self.phi_x)
        R_x[:, 1, 2] = -torch.sin(self.phi_x)
        R_x[:, 2, 1] = torch.sin(self.phi_x)
        R_x[:, 2, 2] = torch.cos(self.phi_x)

        R_y = torch.diag(torch.ones(self._dim + 1)).repeat(
            self._num_ch, 1, 1).to(dtype=self._dtype, device=self._device)
        R_y[:, 0, 0] = torch.cos(self.phi_y)
        R_y[:, 0, 2] = torch.sin(self.phi_y)
        R_y[:, 2, 0] = -torch.sin(self.phi_y)
        R_y[:, 2, 2] = torch.cos(self.phi_y)

        R_z = torch.diag(torch.ones(self._dim + 1)).repeat(
            self._num_ch, 1, 1).to(dtype=self._dtype, device=self._device)
        R_z[:, 0, 0] = torch.cos(self.phi_z)
        R_z[:, 0, 1] = -torch.sin(self.phi_z)
        R_z[:, 1, 0] = torch.sin(self.phi_z)
        R_z[:, 1, 1] = torch.cos(self.phi_z)

        self.rotation_matrix = torch.einsum(
            'bij, bjk->bik', torch.einsum('bij, bjk->bik', R_z, R_y), R_x)

    def _compute_transformation_matrix(self):
        transformation_matrix = torch.einsum(
            'bij, bjk->bik', self.trans_matrix_pos,
            self.rotation_matrix)[:, 0:self._dim, :]
        return transformation_matrix

    def _compute_dense_flow(self, transformation_matrix, return_orig=False):
        # （N, 3, HWD)
        flow = torch.einsum('qijk,bpq->bpijk', self.grid,
                            transformation_matrix.reshape(-1, 3, 4))
        if not return_orig:
            # normalize flow values to [-1, 1] for grid_sample
            for i in range(self._dim):
                flow[:, i, ...] = 2 * (flow[:, i, ...] /
                                       (self._image_size[i] - 1) - 0.5)

            # [X, Y, Z, [x,y,z]]
            flow = flow.permute([0] + list(range(2, 2 + self._dim)) + [1])
            index_ordering: List[int] = list(range(self._dim - 1, -1, -1))
            flow = flow[..., index_ordering]  # x,y,z -> z,y,x
        else:
            flow = flow - self.grid[None, :3, ...]
        return flow

    @property
    def transformation_matrix(self):
        self._compute_transformation_3d()
        return self._compute_transformation_matrix()

    @property
    def dense_flow(self):
        self._compute_transformation_3d()
        transformation_matrix = self._compute_transformation_matrix()
        return self._compute_dense_flow(transformation_matrix,
                                        return_orig=False)

    @property
    def orig_flow(self):
        self._compute_transformation_3d()
        transformation_matrix = self._compute_transformation_matrix()
        return self._compute_dense_flow(transformation_matrix,
                                        return_orig=True)


def get_closest_rigid(source_oh: torch.Tensor,
                      target_oh: torch.Tensor,
                      disp_field: torch.Tensor,
                      lr: float = 1e-2,
                      num_iteration: int = 50,
                      dtype=torch.float32):
    """
    Rigid registration of vertebrae labels
    Args:
        source_oh: （hard） one-hot format, the shape should be BNHW[D], the background is excluded
        target_oh: (soft) one-hot format, the shape should be BNHW[D], the background is excluded
        disp_field: dense displacement field mapping from target label to source label, with shape B3HW[D]
        lr: learning rate of Adam optimizer
        num_iteration: number of iterations in the affine registration
    Return:
        resample_source: (soft) one-hot format, the shape should be BNHW[D]
    """
    # print(f"Before registration, dice:{compute_meandice(source_oh, target_oh, include_background=True)}")
    try:
        device = torch.device('cuda', source_oh.get_device())
    except RuntimeError:
        device = 'cpu'

    rigid_transform = RigidTransformation(source_oh,
                                          opt_cm=False,
                                          dtype=dtype,
                                          device=device)
    rigid_transform.init_transform(target_oh, disp_field)

    # BNHWD, B = 1
    source_oh = source_oh.squeeze(0).unsqueeze(1)
    target_oh = target_oh.squeeze(0).unsqueeze(1)

    optimizer = torch.optim.Adam(rigid_transform.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.5)

    for i in range(num_iteration):
        optimizer.zero_grad()
        flow = rigid_transform.dense_flow
        resample_source = F.grid_sample(source_oh,
                                        grid=flow,
                                        mode='bilinear',
                                        align_corners=True)

        loss = torch.sum((target_oh - resample_source)**2)
        # print(f"iter {i}, loss: {loss}")

        loss.backward()
        optimizer.step()
        scheduler.step()

    # print(f"after reg: {loss}")

    flow = rigid_transform.dense_flow
    # since we are using bilinear resampling in the model
    # we also use bilinear resampling here
    resample_source = F.grid_sample(source_oh,
                                    grid=flow,
                                    mode='bilinear',
                                    align_corners=True)
    # print(f"After registration, dice:{compute_meandice(resample_source, target_oh, include_background=True)}")
    # BNHWD, B = 1
    resample_source = resample_source.squeeze(1).unsqueeze(0)

    for param in rigid_transform.parameters():
        param.requires_grad = False

    return resample_source.detach(), rigid_transform.orig_flow.detach()

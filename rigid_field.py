import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, Tuple, List
from voxelmorph.torch.layers import ResizeTransform
from rigid_utils import get_reference_grid, solve_SVD, sample_displacement_flow, get_mass_center


class RigidFieldLoss(nn.Module):
    """
    Compute rigid motion between input one-hot mask and target one-hot mask using SVD
    """

    def __init__(self,
                 image_size: Union[List[int], Tuple[int, ...]] = (64, 128, 128),
                 num_samples: int = 16,
                 downsize: int = 2,
                 inv: bool = False,
                 include_background: bool = False,
                 dtype=torch.float32, device='cpu'):
        """

        Args:
            image_size: shape of input image
            num_samples: int, number of sampling correspondences
            downsize: int, downsampling factor for computer field MSE loss
            inv: bool, flag deciding the direction of sampling correspondences
        """
        super().__init__()
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        # (4,HWD)
        self.register_buffer("grid", grid)
        self.num_samples = num_samples
        self.inv = inv
        self.include_background = include_background
        self.resize = ResizeTransform(downsize, self._dim)

    def sample_correspondence(self, label_map: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Sample correspondence between fixed and moving images
        Args:
            label_map: one-hot label mask, tensor of shape BNHWD, with B=1
                    if inv:
                        input = moving image
                    else:
                        input = fixed image
            flow: dense displacement field mapping from
                    if inv:
                        moving image to fixed image
                    else:
                        fixed image to moving image

        Returns:
            fixed_pnts_list:
            moving_pnts_list:
        """
        num_ch = label_map.shape[1]
        src_pnts_list = []
        des_pnts_list = []
        for ch in range(num_ch):
            valid_points = (label_map[0, ch] == 1).nonzero()
            valid_len = valid_points.shape[0]
            indices = torch.randint(valid_len, [self.num_samples]).to(self._device)

            src_pnts = torch.index_select(valid_points.float(), 0, indices)

            sample_grid = src_pnts.detach().clone()

            sample_flow = sample_displacement_flow(sample_grid, flow, self._image_size)
            sample_flow = sample_flow.squeeze()

            # (3, num_samples)
            src_pnts = src_pnts.transpose(0, 1)
            des_pnts = src_pnts + sample_flow

            src_pnts_list.append(src_pnts)
            des_pnts_list.append(des_pnts)

        return src_pnts_list, des_pnts_list

    def lsq_rigid_motion(self, y_source_pnts_list: List[torch.Tensor], source_pnts_list: List[torch.Tensor],
                         y_source_cm_list: torch.Tensor, source_cm_list: torch.Tensor) -> torch.Tensor:
        """
        Least Square Method solving Rigid motion from correspondences
        https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        Args:
            y_source_pnts_list: list of length N, tensor of shape (3, num_samples)
            source_pnts_list: list of length N, tensor of shape (3, num_samples)
            y_source_cm_list: center mass of y_source_oh, tensor of shape (3, N)
            source_cm_list: center mass of source_oh, tensor of shape (3, N)

        Returns:
            transform_matrix: tensor of shape (N, 4, 4)
        """
        num_ch = len(y_source_pnts_list)
        trans_matrix_list = []
        for ch in range(num_ch):
            # points in fixed image
            y_source_pnts = y_source_pnts_list[ch]
            # corresponding points in target image
            source_pnts = source_pnts_list[ch]

            y_source_cm = y_source_cm_list[:, [ch]]
            source_cm = source_cm_list[:, [ch]]

            R, t = solve_SVD(y_source_pnts, source_pnts, y_source_cm, source_cm)

            trans_matrix_pos = torch.diag(torch.ones(4)).to(dtype=self._dtype, device=self._device)
            # trans_matrix_cm = torch.diag(torch.ones(4)).to(dtype=self._dtype, device=self._device)
            # trans_matrix_cm_rw = torch.diag(torch.ones(4)).to(dtype=self._dtype, device=self._device)
            trans_matrix_rot = torch.diag(torch.ones(4)).to(dtype=self._dtype, device=self._device)

            trans_matrix_pos[:3, [3]] = t
            # trans_matrix_cm[:3, [3]] = -y_source_cm
            # trans_matrix_cm_rw[:3, [3]] = y_source_cm
            trans_matrix_rot[:3, :3] = R

            # trans_matrix = trans_matrix_pos @ trans_matrix_cm @ trans_matrix_rot @ trans_matrix_cm_rw
            trans_matrix = trans_matrix_pos @ trans_matrix_rot
            trans_matrix_list.append(trans_matrix)

        # (N, 4, 4)
        transform_matrices = torch.stack(trans_matrix_list, dim=0)
        return transform_matrices[:, :3, :]

    def forward(self, y_source_oh: torch.Tensor, source_oh: torch.Tensor,
                flow: torch.Tensor,
                neg_flow: torch.Tensor) -> torch.Tensor:
        """

        Args:
            y_source_oh: one-hot mask, tensor of shape BNHWD, with B=1
            source_oh: one-hot mask, tensor of shape BNHWD, with B=1
            flow: tensor of shape B3HWD, with B=1, dense displacement field mapping from y_source_oh to source_oh
            neg_flow: tensor of shape B3HWD, with B=1, dense displacement field mapping from source_oh to y_source_oh

        Returns:

        """
        with torch.no_grad():
            if not self.include_background:
                y_source_oh = y_source_oh[:, 1:]
                source_oh = source_oh[:, 1:]

            # BNHWD
            # exclude low volume mask
            valid_ch = torch.logical_and(y_source_oh.sum(dim=(0, 2, 3, 4)) > 100, source_oh.sum(dim=(0, 2, 3, 4)) > 100)
            y_source_oh = y_source_oh[:, valid_ch, ...]
            source_oh = source_oh[:, valid_ch, ...]

            y_source_cm_list = get_mass_center(y_source_oh, self.grid, self._dim)
            source_cm_list = get_mass_center(source_oh, self.grid, self._dim)

            if self.inv:
                source_pnts_list, y_source_pnts_list = self.sample_correspondence(source_oh, neg_flow)
            else:
                y_source_pnts_list, source_pnts_list = self.sample_correspondence(y_source_oh, flow)

            transform_matrices = self.lsq_rigid_motion(y_source_pnts_list, source_pnts_list,
                                                       y_source_cm_list, source_cm_list)

            # (N1HWD)
            y_source_oh = y_source_oh.squeeze(0).unsqueeze(1)
            # source_oh = source_oh.squeeze(0).unsqueeze(1)

            # (N3HWD), N=self._num_ch, [[x,y,z], X,Y,Z]
            rigid_flow = torch.einsum("qijk,bpq->bpijk", self.grid, transform_matrices.reshape(-1, 3, 4))
            rigid_flow = rigid_flow - self.grid[None, :3, ...]
            # (1,3,HWD)
            # select displacement flow inside label areas
            rigid_flow = torch.sum(rigid_flow * y_source_oh, dim=0, keepdim=True)
            # rigid_flow = self.resize(rigid_flow)

        # (1,3,HWD)
        flow = torch.sum(y_source_oh, dim=0, keepdim=True) * flow
        # flow = self.resize(flow)
        loss = torch.mean(torch.linalg.norm(rigid_flow - flow, dim=1))
        return loss

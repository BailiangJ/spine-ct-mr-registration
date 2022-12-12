import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Union
from monai.utils.enums import LossReduction


def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise squared euclidean distance of input coordinates.
    Args:
        x: input coordinates, input shape should be (1, dim, #input points)
    Returns:
        dist: pairwise distance matrix, (#input points, #input points)
    """
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

class MINDSSCLoss(_Loss):
    def __init__(self,
                 radius: int = 2,
                 dilation: int = 2,
                 penalty: str = 'l2',
                 reduction: Union[LossReduction, str] = LossReduction.MEAN):
        super().__init__(reduction=LossReduction(reduction).value)
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels()

    def build_kernels(self):
        # define start and end locations for self-similarity pattern
        # (6, 3)
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        # (6, 6), diagonal elements are zero
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        # select neighbours with inter - Euclidean distance = sqrt(2) and discard duplicates
        # 12 edges in total
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        # self-similarity context: 12 elements
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        # create a kernel that selects the start node of the 12 edges
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        # create a kernel that selects the end node of the 12 edges
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad3d(self.dilation)
        rpad2 = ReplicationPad3d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: the shape should be B[NDHW].
        """
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2((F.conv3d(self.rpad1(img), mshift1, dilation=self.dilation) -
                                       F.conv3d(self.rpad1(img), mshift2, dilation=self.dilation)) ** 2),
                           self.kernel_size, stride=1)

        # MIND equation
        # subtract the channel-wise minimum and divide by the clamped channel-wise mean to normalize MIND-SSC so that
        # the maximum of exp(-mind) equals 1
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.penalty == 'l1':
            mind_loss = torch.abs(self.mind(input) - self.mind(target))
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            mind_loss = torch.square(self.mind(input) - self.mind(target))

        if self.reduction == LossReduction.MEAN.value:
            mind_loss = torch.mean(mind_loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            mind_loss = torch.sum(mind_loss)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return mind_loss
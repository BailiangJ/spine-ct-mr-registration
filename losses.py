import warnings
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from monai.utils.enums import LossReduction
from typing import Union, Tuple, Optional, Callable, List
from torch.nn import ReplicationPad3d
import torch.nn.functional as F


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added to the intersection and union components of the inter-over-union calculation to smooth results
    respectively, these values should be small. The `include_background` class attribute can be set to False for
    an instance of DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be
    background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
            self,
            include_background: bool = False,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: （soft） one-hot format, the shape should be BNHW[D]
            target: (soft) one-hot format, the shape should be BNHW[D]

        Raises:
            AssertionError: When source and target (after one hot transform if setted)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        # input, target = inout_onehot(input, target)
        if source.shape != target.shape:
            raise AssertionError(f"input shape{source.shape} doesn't not equal target shape {target.shape}.")
        n_pred_ch = source.shape[1]

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                source = source[:, 1:]

        if target.shape != source.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({source.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(source.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * source, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            source = torch.pow(source, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(source, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class GlobalMutualInformationLoss(_Loss):
    """
    Differentiable global mutual information loss via Parzen windowing method.

    Reference:
        https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1
    """

    def __init__(
            self,
            num_bins: int = 23,
            sigma_ratio: float = 0.5,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-7,
            smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super(GlobalMutualInformationLoss, self).__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError(f"num_bins must > 0, got {num_bins}")
        bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(self, pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: the shape should be B[NDHW].
        """
        pred = torch.clamp(pred, 0, 1)
        pred = pred.reshape(pred.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(pred) * (pred - self.bin_centers.to(pred)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        wa, pa = self.parzen_windowing(pred)  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        wb, pb = self.parzen_windowing(target)  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        pab = torch.bmm(wa.permute(0, 2, 1), wb).div(wa.shape[1])  # (batch, num_bins, num_bins)

        papb = torch.bmm(pa.permute(0, 2, 1), pb)  # (batch, num_bins, num_bins)
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr), dim=(1, 2)
        )  # (batch)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return mi.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(mi).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


def pdist_squared(x):
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
    def __init__(self, radius: int = 2, dilation: int = 2,
                 penalty: str = 'l2',
                 # split_gpu: bool = False,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN):
        super().__init__(reduction=LossReduction(reduction).value)
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels()

    def build_kernels(self):
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        # self-similarity context: 12 elements
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad3d(self.dilation)
        rpad2 = ReplicationPad3d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2((F.conv3d(self.rpad1(img), mshift1, dilation=self.dilation) -
                                       F.conv3d(self.rpad1(img), mshift2, dilation=self.dilation)) ** 2),
                           self.kernel_size, stride=1)

        # MIND equation
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

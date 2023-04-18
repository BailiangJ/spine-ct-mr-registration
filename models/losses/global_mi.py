from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.utils.enums import LossReduction
from monai.utils.module import look_up_option
from torch.nn.modules.loss import _Loss

from ..builder import LOSSES


@LOSSES.register_module('global_mi')
class GlobalMutualInformationLoss(_Loss):
    """Differentiable global mutual information loss via Parzen windowing method.

    Directly from Project-MONAI: https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/image_dissimilarity.py

    Reference:
        https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1
    """
    def __init__(
        self,
        kernel_type: str = 'gaussian',
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``}
                ``"gaussian"``: adapted from DeepReg
                Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1.
                ``"b-spline"``: based on the method of Mattes et al [1,2] and adapted from ITK
                References:
                  [1] "Nonrigid multimodality image registration"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.
                  [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"
                      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank
                      IEEE Transactions in Medical Imaging. Vol.22, No.1,
                      January 2003. pp.120-128.
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
        super().__init__(reduction=LossReduction(reduction).value)
        if num_bins <= 0:
            raise ValueError('num_bins must > 0, got {num_bins}')
        bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.kernel_type = look_up_option(kernel_type,
                                          ['gaussian', 'b-spline'])
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        if self.kernel_type == 'gaussian':
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(self, pred: torch.Tensor,
                         target: torch.Tensor) -> Sequence[torch.Tensor]:
        if self.kernel_type == 'gaussian':
            pred_weight, pred_probability = self.parzen_windowing_gaussian(
                pred)
            target_weight, target_probability = self.parzen_windowing_gaussian(
                target)
        elif self.kernel_type == 'b-spline':
            # a third order BSpline kernel is used for the pred image intensity PDF.
            pred_weight, pred_probability = self.parzen_windowing_b_spline(
                pred, order=3)
            # a zero order (box car) BSpline kernel is used for the target image intensity PDF.
            target_weight, target_probability = self.parzen_windowing_b_spline(
                target, order=0)
        else:
            raise ValueError
        return pred_weight, pred_probability, target_weight, target_probability

    def parzen_windowing_b_spline(self, img: torch.Tensor,
                                  order: int) -> Sequence[torch.Tensor]:
        """
        Parzen windowing with b-spline kernel (adapted from ITK)
        Args:
            img: the shape should be B[NDHW].
            order: int.
        """

        # Compute binsize for the histograms.
        #
        # The binsize for the image intensities needs to be adjusted so that
        # we can avoid dealing with boundary conditions using the cubic
        # spline as the Parzen window.  We do this by increasing the size
        # of the bins so that the joint histogram becomes "padded" at the
        # borders. Because we are changing the binsize,
        # we also need to shift the minimum by the padded amount in order to
        # avoid minimum values filling in our padded region.
        #
        # Note that there can still be non-zero bin values in the padded region,
        # it's just that these bins will never be a central bin for the Parzen
        # window.
        _max, _min = torch.max(img), torch.min(img)
        padding = 2
        bin_size = (_max - _min) / (self.num_bins - 2 * padding)
        norm_min = torch.div(_min, bin_size) - padding

        # assign bin/window index to each voxel
        window_term = torch.div(img, bin_size) - norm_min  # B[NDHW]
        # make sure the extreme values are in valid (non-padded) bins
        window_term = torch.clamp(window_term, padding,
                                  self.num_bins - padding - 1)  # B[NDHW]
        window_term = window_term.reshape(window_term.shape[0], -1,
                                          1)  # (batch, num_sample, 1)
        bins = torch.arange(self.num_bins, device=window_term.device).reshape(
            1, 1, -1)  # (1, 1, num_bins)
        sample_bin_matrix = torch.abs(
            bins - window_term)  # (batch, num_sample, num_bins)

        # b-spleen kernel
        # (4 - 6 * abs ** 2 + 3 * abs ** 3) / 6 when 0 <= abs < 1
        # (2 - abs) ** 3 / 6 when 1 <= abs < 2
        weight = torch.zeros_like(
            sample_bin_matrix,
            dtype=torch.float)  # (batch, num_sample, num_bins)
        if order == 0:
            weight = weight + (sample_bin_matrix <
                               0.5) + (sample_bin_matrix == 0.5) * 0.5
        elif order == 3:
            weight = (
                weight +
                (4 - 6 * sample_bin_matrix**2 + 3 * sample_bin_matrix**3) *
                (sample_bin_matrix < 1) / 6)
            weight = weight + (2 - sample_bin_matrix)**3 * (
                sample_bin_matrix >= 1) * (sample_bin_matrix < 2) / 6
        else:
            raise ValueError(
                f'Do not support b-spline {order}-order parzen windowing')

        weight = weight / torch.sum(
            weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bins)
        probability = torch.mean(weight, dim=-2,
                                 keepdim=True)  # (batch, 1, num_bins)
        return weight, probability

    def parzen_windowing_gaussian(self,
                                  img: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Parzen windowing with gaussian kernel (adapted from DeepReg implementation)
        Note: the input is expected to range between 0 and 1
        Args:
            img: the shape should be B[NDHW].
        """
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(-self.preterm.to(img) *
                           (img - self.bin_centers.to(img))**
                           2)  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(
            weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2,
                                 keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(
                f'ground truth has differing shape ({target.shape}) from pred ({pred.shape})'
            )
        wa, pa, wb, pb = self.parzen_windowing(
            pred, target)  # (batch, num_sample, num_bin), (batch, 1, num_bin)

        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(
            wa.shape[1])  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1),
                         pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(pab *
                       torch.log((pab + self.smooth_nr) /
                                 (papb + self.smooth_dr) + self.smooth_dr),
                       dim=(1, 2))  # (batch)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return mi.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(
                mi).neg()  # average over the batch and channel ndims
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )

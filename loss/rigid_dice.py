from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.utils.enums import LossReduction
from torch.nn.modules.loss import _Loss

from utils import RigidTransformation, get_closest_rigid


class RigidDiceLoss(_Loss):
    """Compute the dice loss between the prediction and closest rigidly transformed
    label."""
    def __init__(self,
                 image_size: Sequence[int] = (64, 128, 128),
                 downsize: int = 2,
                 include_background: bool = False,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN):
        """
        Args:
            dice_weight: float, the weight of dice loss
            mse_weight: float, the wieght of mse loss
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        # the first channel won't be background, inputs are one-hot format
        self.dice_loss_func = DiceLoss(include_background=True,
                                       to_onehot_y=False,
                                       reduction=reduction)

    def forward(
        self,
        source_mask: torch.Tensor,
        y_source_mask: torch.Tensor,
        disp_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_mask: （hard） one-hot format, the shape should be BNHW[D]
            y_source_mask: (soft) one-hot format, the shape should be BNHW[D]
            disp_field: displacement field, tensor of shape （13HWD)

        Return:
            loss: dice_weight * dice_loss + mse_weight * mse_loss
        """
        if not self.include_background:
            source_mask = source_mask[:, 1:]
            y_source_mask = y_source_mask[:, 1:]

        # BNHWD
        # exclude low volume mask
        valid_ch = torch.logical_and(
            y_source_mask.sum(dim=(0, 2, 3, 4)) > 100,
            source_mask.sum(dim=(0, 2, 3, 4)) > 100)
        y_source_mask = y_source_mask[:, valid_ch, ...]
        source_mask = source_mask[:, valid_ch, ...]

        # rigid_y_source_mask is soft one-hot
        rigid_y_source_mask, rigid_flow = get_closest_rigid(
            source_mask.detach(), y_source_mask.detach(), disp_field.detach())
        dice_loss = self.dice_loss_func(y_source_mask, rigid_y_source_mask)
        return dice_loss

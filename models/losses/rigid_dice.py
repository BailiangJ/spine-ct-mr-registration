from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.utils.enums import LossReduction
from torch.nn.modules.loss import _Loss

from ..builder import LOSSES
from .rigid_utils import RigidTransformation, get_closest_rigid


@LOSSES.register_module('rigid_dice')
class RigidDiceLoss(_Loss):
    """Compute the dice loss between the prediction and closest rigidly transformed
    label.

    Args:
            include_background (bool): whether to include the background channel in Dice loss computation.
            reduction (LossReduction, str): {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
    """

    # TODO: add device to __init__
    def __init__(
            self,
            include_background: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        # the first channel won't be background, inputs are one-hot format
        self.dice_loss_func = DiceLoss(include_background=True,
                                       to_onehot_y=False,
                                       reduction=self.reduction)

    def forward(self,
                y_source_oh: torch.Tensor,
                source_oh: torch.Tensor,
                flow: torch.Tensor,
                neg_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            source_oh (torch.tensor): （hard） one-hot format, the shape should be BNHW[D], B=1
            y_source_oh (torch.tensor): (soft) one-hot format, the shape should be BNHW[D], B=1
            flow (torch.tensor): displacement field, tensor of shape （13HWD)

        Return:
            dice_loss (torch.tensor): the dice loss between the prediction and closest rigidly transformed label.
        """
        if not self.include_background:
            source_oh = source_oh[:, 1:]
            y_source_oh = y_source_oh[:, 1:]

        # BNHWD
        # exclude low volume mask
        valid_ch = torch.logical_and(
            y_source_oh.sum(dim=(0, 2, 3, 4)) > 100,
            source_oh.sum(dim=(0, 2, 3, 4)) > 100)
        y_source_oh = y_source_oh[:, valid_ch, ...]
        source_oh = source_oh[:, valid_ch, ...]

        # rigid_y_source_mask is soft one-hot
        rigid_y_source_mask, rigid_flow = get_closest_rigid(
            source_oh.detach(), y_source_oh.detach(), flow.detach())
        dice_loss = self.dice_loss_func(y_source_oh, rigid_y_source_mask)
        return dice_loss

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(include_background={self.include_background},'
                     f'reduction={self.reduction})')
        return repr_str

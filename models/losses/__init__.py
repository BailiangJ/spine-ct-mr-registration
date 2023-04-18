from .diffusion_regularizer import GradientDiffusionLoss
from .global_mi import GlobalMutualInformationLoss
from .kernels import *
from .mind import MINDSSCLoss
from .oc_pc_loss import OrthonormalPropernessCondition
from .rigid_dice import RigidDiceLoss
from .rigid_field import RigidFieldLoss
from .rigid_utils import *
from monai.losses import LocalNormalizedCrossCorrelationLoss
from ..builder import LOSSES

LOSSES.register_module('lncc', module=LocalNormalizedCrossCorrelationLoss)

__all__ = [
    'GradientDiffusionLoss', 'GlobalMutualInformationLoss', 'MINDSSCLoss',
    'OrthonormalPropernessCondition', 'RigidDiceLoss', 'RigidFieldLoss'
]

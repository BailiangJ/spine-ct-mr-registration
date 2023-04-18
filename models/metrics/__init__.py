from .sdlogjac import SDlogDetJac
from ..builder import METRICS
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric

METRICS.register_module('dice', module=DiceMetric)
METRICS.register_module('haus_dist', module=HausdorffDistanceMetric)
METRICS.register_module('surf_dist', module=SurfaceDistanceMetric)

__all__=['SDlogDetJac']
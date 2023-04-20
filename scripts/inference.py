import os
import sys

import numpy as np
import torch

sys.path.append('../')
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)

from datasets import load_test_data
from models import (RigidDiceLoss, SDlogDetJac, build_flow_estimator,
                    build_registration_head)

from .utils import set_seed


def inference(config_file: str):
    # load configuration
    cfg = Config.fromfile(config_file)

    # load pretrained model
    model = build_flow_estimator(cfg.vxm_cfg)
    model.init_weights()
    model.to(cfg.device)
    model.eval()

    # load data
    test_dataset = load_test_data(**cfg.testset_cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # metric functions
    compute_jacdet = SDlogDetJac()
    compute_dice = DiceMetric(include_background=False, reduction='mean')
    compute_haus95_dist = HausdorffDistanceMetric(percentile=95,
                                                  include_background=False)

    # rigid losses
    compute_rigid_dice = RigidDiceLoss(include_background=False,
                                       reduction='mean')

    for i, data in enumerate(test_loader):
        target = data['mr'].to(cfg.device).float()
        target_oh = data['mr_oh'].to(cfg.device).float()
        source = data['ct'].to(cfg.device).float()
        source_oh = data['ct_oh'].to(cfg.device).float()

        image_size = list(target.shape[-3:])

        # one-hot label with shape [BNHWD] B=1
        mask = torch.logical_and(
            target_oh.sum(dim=(2, 3, 4)) > 500,
            source_oh.sum(dim=(2, 3, 4)) > 500)
        target_oh = target_oh[mask].unsqueeze(0)
        source_oh = source_oh[mask].unsqueeze(0)

        # registration head
        # if the image size are consistent in test set, it can be moved out of the for loop
        cfg.registration_cfg.image_size = image_size
        register = build_registration_head(cfg.registration_cfg)
        register.to(cfg.device)

        # forward run
        with torch.no_grad():
            flow = model(source, target)
            fwd_flow, bck_flow, y_source, y_target, y_source_oh, y_target_oh = register(
                flow, source, target, source_oh, target_oh)

            sdlog_jacdet, non_pos_jacdet = compute_jacdet(
                fwd_flow.detach().cpu())
            fwd_dice = compute_dice(y_source_oh, target_oh)
            fwd_haus95_dist = compute_haus95_dist(y_source_oh, target_oh)
            with torch.enable_grad():
                rigid_dice = compute_rigid_dice(y_source_oh, source_oh,
                                                fwd_flow, bck_flow)

            print(f'data {i} - '
                  f'sdlog_jacdet:{sdlog_jacdet:.2f},'
                  f'non_pos_jacdet:{non_pos_jacdet * 100:.2f},'
                  f'fwd_dice:{fwd_dice.mean():.2f},'
                  f'fwd_haus95_dist:{fwd_haus95_dist.mean():.2f},'
                  f'rigid_dice:{rigid_dice.detach().cpu():.2f}')


if __name__ == '__main__':
    import pathlib

    import configargparse

    p = configargparse.ArgParser()
    p.add_argument('--config-file',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of configure file')
    args = p.parse_args()
    set_seed(2023)
    inference(args.config_file)

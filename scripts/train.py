import os
import sys

import numpy as np
import torch
import wandb

sys.path.append('../')
from epoch_run import run_epoch
from mmengine import Config
from monai.data import DataLoader
from utils import set_seed, worker_init_fn

from datasets import load_train_val_data
from models import (build_flow_estimator, build_loss, build_metrics,
                    build_registration_head)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(config_file: str):
    # load configuration
    cfg = Config.fromfile(config_file)

    wandb.init(project=cfg.project,
               name=cfg.name,
               group=cfg.group,
               config=dict(cfg))
    config = wandb.config

    # define output directory
    model_dir = os.path.join(config.out_path, config.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # initialize model or load pretrained model
    model = build_flow_estimator(cfg.vxm_cfg)
    model.init_weights()
    model.to(cfg.device)
    wandb.watch(model, log='gradients', log_freq=100, log_graph=True)

    # build registration head module
    register = build_registration_head(cfg.registration_cfg)
    register.to(cfg.device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          gamma=cfg.decay_rate)

    # loss functions
    loss_funcs = dict()
    loss_funcs.update({'sim': build_loss(cfg.sim_loss_cfg)})
    loss_funcs.update({'reg': build_loss(cfg.reg_loss_cfg)})
    for loss_cfg in cfg.rigid_losses_cfgs:
        loss_funcs.update({loss_cfg.type: build_loss(loss_cfg)})

    # loss weights
    loss_weights = dict()
    loss_weights.update({'sim': cfg.sim_loss_cfg.weight})
    loss_weights.update({'reg': cfg.reg_loss_cfg.weight})
    for loss_cfg in cfg.rigid_losses_cfgs:
        loss_weights.update({loss_cfg.type: loss_cfg.weight})

    # metric functions
    metric_func = build_metrics(cfg.metric_cfg)

    # load data
    train_dataset = load_train_val_data(**cfg.trainset_cfg)
    val_dataset = load_train_val_data(**cfg.valset_cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            worker_init_fn=worker_init_fn)

    best_dice = 1
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        if epoch % cfg.val_interval == 0 or epoch == cfg.start_epoch:
            phase = 'val'
            model.eval()
            with torch.no_grad():
                val_dice = run_epoch(cfg, model, register, val_loader,
                                     loss_funcs, loss_weights, optimizer,
                                     metric_func, phase)

        phase = 'train'
        model.train()
        run_epoch(cfg, model, register, train_loader, loss_funcs, loss_weights,
                  optimizer, metric_func, phase)

        lr_scheduler.step()

        if epoch % cfg.save_interval == 0 and epoch != 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '%04d.pth' % epoch))

    torch.save(model.state_dict(),
               os.path.join(model_dir, '%04d.pth' % cfg.max_epochs))


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
    train(args.config_file)

from typing import Dict, Union, Callable
from mmengine import Config, ConfigDict
from torch.nn import Module
from torch import nn
from torch.optim import Optimizer
from monai.data import DataLoader
import numpy as np
import torch
import wandb
from collections.abc import Sequence

CFG = Union[dict, Config, ConfigDict]


def run_epoch(cfg: CFG,
              model: Module,
              register: Module,
              dataloader: DataLoader,
              loss_funcs: Dict,
              loss_weights: Dict,
              optimizer: Optimizer,
              metric_func: Callable,
              phase: str,
              )->torch.Tensor:
    logging_dict = dict()
    for data in dataloader:
        for target, target_oh, source, source_oh in zip(
                data['mr'].to(cfg.device).float(),
                data['mr_oh'].to(cfg.device).float(),
                data['ct'].to(cfg.device).float(),
                data['ct_oh'].to(cfg.device).float()):

            # resume the batch size dimension
            target = target.unsqueeze(0)
            source = source.unsqueeze(0)

            # ignore low-volume vertebra
            # one-hot label with shape [NHWD]
            mask = torch.logical_and(target_oh.sum(dim=(1, 2, 3)) > 500, source_oh.sum(dim=(1, 2, 3)) > 500)
            target_oh = target_oh[mask].unsqueeze(0)
            source_oh = source_oh[mask].unsqueeze(0)
            if source_oh.shape[1] == 1:
                print("skip data with only background label.")
                continue

            flow = model(source, target)
            fwd_flow, bck_flow, y_source, y_target, y_source_oh, y_target_oh = register(flow, source, target, source_oh,
                                                                                        target_oh)

            total_loss = 0.

            # compute image similarity
            fwd_sim = loss_funcs['sim'](y_source, target)
            bck_sim = loss_funcs['sim'](y_target, source) if y_target is not None else None
            w_sim = loss_weights['sim']
            #
            total_loss += w_sim * (fwd_sim + bck_sim) if bck_sim is not None else w_sim * fwd_sim
            #
            logging_dict.update({'fwd_sim': fwd_sim.detach().cpu()})
            if bck_sim is not None:
                logging_dict.update({'bck_sim': bck_sim.detach().cpu()})

            # compute gradient diffusion regularizer on flow
            reg = loss_funcs['reg'](flow)
            w_reg = loss_weights['reg']
            #
            total_loss += w_reg * reg
            #
            logging_dict.update({'reg': reg.detach().cpu()})

            with torch.enable_grad():
                # computing rigid dice requires gradient computation during validation run
                # compute rigidity losses
                rigid_loss_keys = list(loss_funcs.keys())
                rigid_loss_keys.remove('sim')
                rigid_loss_keys.remove('reg')
                for key in rigid_loss_keys:
                    weight = loss_weights[key]
                    if weight is None or weight == [None] * 2:
                        continue

                    rigid_loss = loss_funcs[key](y_source_oh, source_oh, fwd_flow, bck_flow)

                    if isinstance(weight, Sequence):
                        for l, w in zip(rigid_loss, weight):
                            total_loss += w * l
                        #
                        logging_dict.update({'oc': rigid_loss[0].detach().cpu(),
                                             'pc': rigid_loss[1].detach().cpu()})
                    else:
                        total_loss += weight * rigid_loss
                        #
                        logging_dict.update({key:rigid_loss.detach().cpu()})

            logging_dict.update({'total_loss': total_loss.detach().cpu()})

            if phase == 'train':
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            with torch.no_grad():
                # dice metric
                fwd_dice = metric_func(y_source_oh.detach(), target_oh.detach())
                bck_dice = metric_func(y_target_oh.detach(), source_oh.detach()) if y_target_oh is not None else None

                #
                logging_dict.update({'fwd_dice': fwd_dice.mean().cpu()})
                if bck_dice is not None:
                    logging_dict.update({'bck_dice': bck_dice.mean().cpu()})

            wandb.log({phase:logging_dict})

    # return average dice score metric to save the best model
    # return

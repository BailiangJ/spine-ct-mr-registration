import glob
import os
from typing import Sequence, Union

import numpy as np
import torch
from mmengine import Config, ConfigDict
from monai.data import CacheDataset
from monai.transforms import (AddChanneld, CastToTyped, Compose, LoadImaged,
                              Orientationd, RandAffined, EnsureChannelFirstd,
                              RandCropByLabelClassesd, ResizeWithPadOrCropd,
                              ScaleIntensityd, ScaleIntensityRanged, Spacingd,
                              ToTensord)

from .transforms import CropForegroundd, MergeMaskd, OneHotd


def load_train_val_data(data_dir: str,
                        image_size: Sequence[int],
                        aug_cfg: Union[Config, ConfigDict, dict],
                        num_classes: int = 26,
                        num_samples: int = 2,
                        *args,
                        **kwargs):
    """
    Construct train / validation dataset.
    Args:
        data_dir (str): root directory of data.
        image_size (sequence[int]): size of image in dataset.
        aug_cfg (Config|ConfigDict|dict): config dict of random affine augmentation.
        num_classes (int): number of total classes in spine segmentation label.
        num_samples (int): number of cropped samples per data.

    Returns:
        dataset (CacheDataset)

    """

    data_files = sorted(
        glob.glob(os.path.join(data_dir, '**/ct.nii.gz'), recursive=False))
    dirs_list = [os.path.dirname(ct_file) for ct_file in data_files]
    data_dicts = [
        {
            'mr': os.path.join(dir, 'mr_t1.nii.gz'),
            'ct': os.path.join(dir, 'ct.nii.gz'),
            'ct_mask': os.path.join(dir, 'ct_mask.nii.gz'),
            'mr_mask': os.path.join(dir, 'mr_mask.nii.gz')  # optional
        } for dir in dirs_list
    ]

    data_transforms = Compose([
        LoadImaged(keys=['mr', 'mr_mask', 'ct', 'ct_mask']),
        AddChanneld(keys=['mr', 'mr_mask', 'ct', 'ct_mask']),
        Spacingd(keys=['mr', 'ct'], pixdim=(1, 1, 1), mode='bilinear'),
        Spacingd(keys=['mr_mask', 'ct_mask'], pixdim=(1, 1, 1),
                 mode='nearest'),
        Orientationd(keys=['mr', 'mr_mask', 'ct', 'ct_mask'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['ct'],
                             a_min=-1000,
                             a_max=2000,
                             b_min=0.0,
                             b_max=1.0,
                             clip=True),
        ScaleIntensityd(keys=['mr'], minv=0.0, maxv=1.0),
        RandAffined(keys=['ct', 'ct_mask'],
                    mode=['bilinear', 'nearest'],
                    prob=aug_cfg.prob,
                    translate_range=aug_cfg.translate_range,
                    rotate_range=aug_cfg.rotate_range,
                    padding_mode=aug_cfg.padding_mode),
        CropForegroundd(keys=['mr', 'mr_mask', 'ct', 'ct_mask'],
                        source_key=['mr_mask', 'ct_mask'],
                        k_divisible=16),  # cropped size is divisible by 16
        MergeMaskd(keys=['mr_mask', 'ct_mask']),
        OneHotd(keys=['mr_mask', 'ct_mask']),
        RandCropByLabelClassesd(
            keys=['mr', 'mr_mask', 'mr_oh', 'ct', 'ct_mask', 'ct_oh'],
            label_key='merge_mask',
            num_classes=num_classes,
            spatial_size=image_size,
            num_samples=num_samples,
            allow_smaller=True),
        ResizeWithPadOrCropd(
            keys=['mr', 'mr_mask', 'mr_oh', 'ct', 'ct_mask', 'ct_oh'],
            spatial_size=image_size,
            mode='constant'),
        ToTensord(keys=['mr', 'mr_mask', 'mr_oh', 'ct', 'ct_mask', 'ct_oh'],
                  track_meta=False),
        CastToTyped(keys=['mr', 'ct'], dtype=torch.float32),
        CastToTyped(keys=['mr_mask', 'ct_mask'], dtype=torch.int64),
        CastToTyped(keys=['mr_oh', 'ct_oh'], dtype=torch.int64)
    ])

    return CacheDataset(data=data_dicts,
                        transform=data_transforms,
                        *args,
                        **kwargs)


def load_test_data(data_dir: str, *args, **kwargs):
    """
    Construct test dataset.
    Args:
        data_dir (str): root directory of data.

    Returns:

    """

    data_files = sorted(
        glob.glob(os.path.join(data_dir, '**/ct.nii.gz'), recursive=False))
    dirs_list = [os.path.dirname(ct_file) for ct_file in data_files]
    data_dicts = [
        {
            'mr': os.path.join(dir, 'mr_t1.nii.gz'),
            'ct': os.path.join(dir, 'ct.nii.gz'),
            'ct_mask': os.path.join(dir, 'ct_mask.nii.gz'),
            'mr_mask': os.path.join(dir, 'mr_mask.nii.gz')  # optional
        } for dir in dirs_list
    ]

    data_transforms = Compose([
        LoadImaged(keys=['mr', 'mr_mask', 'ct', 'ct_mask']),
        EnsureChannelFirstd(keys=['mr', 'mr_mask', 'ct', 'ct_mask'], channel_dim='no_channel'),
        Spacingd(keys=['mr', 'ct'], pixdim=(1, 1, 1), mode='bilinear'),
        Spacingd(keys=['mr_mask', 'ct_mask'], pixdim=(1, 1, 1),
                 mode='nearest'),
        Orientationd(keys=['mr', 'mr_mask', 'ct', 'ct_mask'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['ct'],
                             a_min=-1000,
                             a_max=2000,
                             b_min=0.0,
                             b_max=1.0,
                             clip=True),
        ScaleIntensityd(keys=['mr'], minv=0.0, maxv=1.0),
        CropForegroundd(keys=['mr', 'mr_mask', 'ct', 'ct_mask'],
                        source_key=['mr_mask', 'ct_mask'],
                        k_divisible=16),
        OneHotd(keys=['mr_mask', 'ct_mask']),
        ToTensord(keys=['mr', 'mr_mask', 'mr_oh', 'ct', 'ct_mask', 'ct_oh'],
                  track_meta=False),
        CastToTyped(keys=['mr', 'ct'], dtype=torch.float32),
        CastToTyped(keys=['mr_mask', 'ct_mask'], dtype=torch.int64),
        CastToTyped(keys=['mr_oh', 'ct_oh'], dtype=torch.int64)
    ])

    return CacheDataset(data=data_dicts,
                        transform=data_transforms,
                        *args,
                        **kwargs)

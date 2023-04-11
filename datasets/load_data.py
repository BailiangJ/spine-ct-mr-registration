import glob
import os
import numpy as np
import torch
from typing import Sequence
from monai.data import CacheDataset


def load_train_val_data(data_dir: str, image_size: Sequence[int], affine_aug_prob: float = 0.0, *args, **kwargs):
    """
    Construct train / validation dataset.
    Args:
        data_dir (str): root directory of data.
        image_size (sequence[int]): size of image in dataset.
        affine_aug_prob (float): probability of affine augmentation.

    Returns:
        dataset (CacheDataset)

    """

    data_files = sorted(glob.glob(os.path.join(data_dir, "**/ct.nii.gz"), recursive=False))
    dirs_list = [os.path.dirname(ct_file) for ct_file in data_files]
    data_dicts = [{"mr": os.path.join(dir, "mr_t1.nii.gz"),
                   "ct": os.path.join(dir, "ct.nii.gz"),
                   "ct_mask": os.path.join(dir, "ct_mask"),
                   "mr_mask": os.path.join(dir, "mr_mask")  # optional
                   } for dir in dirs_list]

    data_transforms = Compose([
        LoadImaged(keys=["mr", "mr_mask", "ct", "ct_mask"]),
        AddChanneld(keys=["mr", "mr_mask", "ct", "ct_mask"]),
        Spacingd(keys=["mr", "ct"], pixdim=(1, 1, 1), mode="bilinear"),
        Spacingd(keys=["mr_mask", "ct_mask"], pixdim=(1, 1, 1), mode="nearest"),
        Orientationd(keys=["mr", "mr_mask", "ct", "ct_mask"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityd(keys=["mr"], minv=0.0, maxv=1.0),
        RandAffined(
            keys=["ct", "ct_mask"],
            mode=["bilinear", "nearest"],
            prob=affine_aug_prob,
            translate_range=(3, 3, 3),
            rotate_range=(np.pi / 60, np.pi / 60, np.pi / 60),
            padding_mode="zeros"
        ),
        CropForegroundd(keys=["mr", "mr_mask", "ct", "ct_mask"], source_key=["mr_mask", "ct_mask"], k_divisible=16),
        MergeMaskd(keys=["mr_mask", "ct_mask"]),
        RandCropByLabelClassesd(keys=["mr", "mr_mask", "ct", "ct_mask"], label_key="merge_mask", num_classes=26,
                                spatial_size=image_size, num_samples=2, mode="constant"),
        ResizeWithPadOrCropd(keys=["mr", "mr_mask", "ct", "ct_mask"], spatial_size=image_size, mode="constant"),
        ToTensord(keys=["mr", "mr_mask", "ct", "ct_mask"], track_meta=False),
        CastToTyped(keys=["mr", "ct"], dtype=torch.float32),
        CastToTyped(keys=["mr_mask", "ct_mask"], dtype=torch.int64)
    ])

    return CacheDataset(data=data_dicts, transform=data_transforms, *args, **kwargs)


def load_test_data(data_dir: str, *args, **kwargs):
    """
    Construct test dataset.
    Args:
        data_dir (str): root directory of data.

    Returns:

    """

    data_files = sorted(glob.glob(os.path.join(data_dir, "**/ct.nii.gz"), recursive=False))
    dirs_list = [os.path.dirname(ct_file) for ct_file in data_files]
    data_dicts = [{"mr": os.path.join(dir, "mr_t1.nii.gz"),
                   "ct": os.path.join(dir, "ct.nii.gz"),
                   "ct_mask": os.path.join(dir, "ct_mask"),
                   "mr_mask": os.path.join(dir, "mr_mask")  # optional
                   } for dir in dirs_list]

    data_transforms = Compose([
        LoadImaged(keys=["mr", "mr_mask", "ct", "ct_mask"]),
        AddChanneld(keys=["mr", "mr_mask", "ct", "ct_mask"]),
        Spacingd(keys=["mr", "ct"], pixdim=(1, 1, 1), mode="bilinear"),
        Spacingd(keys=["mr_mask", "ct_mask"], pixdim=(1, 1, 1), mode="nearest"),
        Orientationd(keys=["mr", "mr_mask", "ct", "ct_mask"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityd(keys=["mr"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["mr", "mr_mask", "ct", "ct_mask"], source_key=["mr_mask", "ct_mask"], k_divisible=16),
        ToTensord(keys=["mr", "mr_mask", "ct", "ct_mask"], track_meta=False),
        CastToTyped(keys=["mr", "ct"], dtype=torch.float32),
        CastToTyped(keys=["mr_mask", "ct_mask"], dtype=torch.int64)
    ])

    return CacheDataset(data=data_dicts, transforms=data_transforms, *args, **kwargs)

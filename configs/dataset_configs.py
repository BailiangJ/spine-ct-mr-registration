import os
import glob
import torch
import numpy as np
from datasets import MergeMaskd, CropForegroundd
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, Spacingd, Orientationd, ResizeWithPadOrCropd, Resized, ScaleIntensityRanged,
    ToTensord, CastToTyped, ScaleIntensityd, RandSpatialCropSamplesd, RandAffined, RandCropByLabelClassesd
)

affine_aug_prob = 0.0
image_size = (80, 128, 128)

data_root = "/SpineCTMR"
data_files = sorted(glob.glob(os.path.join(data_root, "**/ct.nii.gz"), recursive=False))
dirs_list = [os.path.dirname(ct_file) for ct_file in data_files]
data_dicts = [{"mr": os.path.join(dir, "mr_t1.nii.gz"),
               "ct": os.path.join(dir, "ct.nii.gz"),
               "ct_mask": os.path.join(dir, "ct_mask"),
               "mr_mask": os.path.join(dir, "mr_mask")  # optional
               } for dir in dirs_list]

train_transforms = Compose([
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

test_transforms = Compose([
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
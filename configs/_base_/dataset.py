import numpy as np

image_size = [64, 128, 64]
trainset_cfg = dict(
    data_dir='/SpineCTMR/train/**',
    image_size=image_size,
    aug_cfg=dict(
        prob=0.0,
        translate_range=[3] * 3,
        rotate_range=[np.pi / 60] * 3,
        padding_mode='zeros'
    ),
    num_classes=26,
    num_samples=2,
    # check CacheDataset in MONAI
    cache_rate=0.1,
    num_workers=4,
)

valset_cfg = dict(
    data_dir='/SpineCTMR/val/**',
    image_size=image_size,
    aug_cfg=dict(
        prob=0.0,
        translate_range=[3] * 3,
        rotate_range=[np.pi / 60] * 3,
        padding_mode='zeros'
    ),
    num_classes=26,
    num_samples=2,
    # check CacheDataset in MONAI
    cache_rate=0.1,
    num_workers=4
)

testset_cfg = dict(
    data_dir='/SpineCTMR/test/**',
    cache_rate=0.1,
    num_workers=4
)

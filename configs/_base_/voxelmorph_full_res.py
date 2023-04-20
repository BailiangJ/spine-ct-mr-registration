_base_ = ['./voxelmorph_half_res.py']

decoder_cfg = dict(
    type='BasicDecoder',
    skip_channels=[64, 64, 32, 32],
    pyramid_levels=['3', '2', '1', '0'],
    out_channels=[64, 64, 64, 64],
    num_convs=[1] * 4,
    strides=[1] * 4,
    dilations=[1] * 4,
)
remain_cfg = dict(
    type='BasicEncoder',
    pyramid_levels=['0', '0'],
)
vxm_cfg = dict(type='VoxelMorph',
               unet_cfg=dict(type='UNet',
                             decoder_cfg=decoder_cfg,
                             remain_cfg=remain_cfg))

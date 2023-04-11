encoder_cfg = dict(
    type='BasicEncoder',
    in_channels=2,
    out_channels=[32, 32, 64, 64, 64],
    pyramid_levels=['0', '1', '2', '3', '4'],
    num_convs=[1] * 5,
    strides=[1] * 5,
    dilations=[1] * 5,
    kernel_size=3,
    conv_cfg=dict(type='Conv3d'),
    norm_cfg=None,
    act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
    pool_cfg=dict(type='MaxPool3d', kernel_size=2),
    init_cfg=None
)
decoder_cfg = dict(
    type='BasicDecoder',
    in_channels=64,
    skip_channels=[64, 64, 32],
    pyramid_levels=['3', '2', '1'],
    out_channels=[64, 64, 64],
    num_convs=[1] * 3,
    strides=[1] * 3,
    dilations=[1] * 3,
    kernel_size=3,
    upsample_cfg=dict(type='Upsample', mode='nearest', scale_factor=2),
    conv_cfg=dict(type='Conv3d'),
    norm_cfg=None,
    act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
    init_cfg=None
)
remain_cfg = dict(
    type='BasicEncoder',
    in_channels=64,
    out_channels=[32, 32],
    pyramid_levels=['3', '3'],
    num_convs=[1] * 2,
    strides=[1] * 2,
    dilations=[1] * 2,
    kernel_size=3,
    conv_cfg=dict(type='Conv3d'),
    norm_cfg=None,
    act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
    pool_cfg=None,
    init_cfg=None
)
flow_conv_cfg = dict(
    in_channels=32,
    out_channels=3,
    num_convs=1,
    stride=1,
    dilation=1,
    kernel_size=3,
    conv_cfg=dict(type='Conv3d'),
    norm_cfg=None,
    act_cfg=None,
    pool_cfg=None,
    init_cfg=dict(
        type='Normal',
        layer='Conv3d',
        mean=0,
        std=1e-5,
        bias=0, )
)
vxm_cfg = dict(
    type='VoxelMorph',
    unet_cfg=dict(
        type='UNet',
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        remain_cfg=remain_cfg,
        norm_eval=False,
        init_cfg=None
    ),
    flow_conv_cfg=flow_conv_cfg,
    init_cfg=None
)
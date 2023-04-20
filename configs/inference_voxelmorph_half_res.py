_base_ = ['./_base_/voxelmorph_half_res.py', './_base_/dataset.py']

device = 'cuda:0'
image_size = [64, 128, 64]

# model_path
load_model = '/saved_models/0100.pth'
vxm_cfg = dict(
    init_cfg=dict(type='Pretrained', checkpoint=load_model, _delete_=True))

# registration head
registration_cfg = dict(type='RegistrationHead',
                        image_size=image_size,
                        int_steps=7,
                        resize_scale=2,
                        resize_first=False,
                        bidir=True,
                        interp_mode='bilinear')

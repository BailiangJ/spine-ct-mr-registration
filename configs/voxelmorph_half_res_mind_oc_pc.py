_base_ = ['./_base_/voxelmorph_half_res.py', './_base_/dataset.py']

device = 'cuda:0'
# wandb
project = ''
group = ''
name = ''

# output directory
out_path = ''
model_dir = 'saved_models'

# dataset
image_size = [64, 128, 64]
trainset_cfg = dict(image_size=image_size)
valset_cfg = dict(image_size=image_size)

# if load pretrained
use_last_ckpt = False
load_model = 'weights.pth'
if use_last_ckpt:
    vxm_cfg = dict(init_cfg=
                   dict(type='Pretrained', checkpoint=load_model, _delete_=True))

# optimizer and lr_scheduler
lr = 1e-4
decay_rate = 0.999
start_epoch = 0
max_epochs = 400
val_interval = 20
save_interval = 100

# losses
# dissimilarity loss
sim_loss_cfg = dict(type='mind', radius=2, dilation=2, penalty='l2', weight=1.0)
# regularization loss
reg_loss_cfg = dict(type='diffusion', penalty='l2', loss_mult=2, weight=0.01)
# if weight = None, we don't compute it. It is suggested to delete the configdict directly.
# if weight = 0.0, we don't train with it but log it
rigid_losses_cfgs = [
    dict(type='oc_pc', weight=[0.005, 0.01]),
    # the lr in rigid_utils.get_closest_rigid is also suggested to take a look and tune
    # it is recommended to first train with sim+reg to get a pretrained model then
    # finetune with the rigid_dice/rigid_field loss
    dict(type='rigid_dice', include_background=False, reduction='mean', weight=0.1),
    dict(type='rigid_field', image_size=image_size, num_samples=256, inv=False,
         include_background=False, device=device, weight=0.01)
]

# registration head
registration_cfg = dict(
    type='RegistrationHead',
    image_size=image_size,
    int_steps=7,
    resize_scale=2,
    resize_first=False,
    bidir=True,
    interp_mode='bilinear'
)

# metrics
metric_cfg = dict(type='dice', include_background=False, reduction='mean')

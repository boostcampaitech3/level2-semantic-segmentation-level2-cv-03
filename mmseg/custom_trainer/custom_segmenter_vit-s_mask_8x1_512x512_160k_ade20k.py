_base_ = [
    '/opt/ml/input/mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask.py',
    '/opt/ml/input/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
    '/opt/ml/input/mmsegmentation/_myexperiment/trash_dataset.py', ## fixed
    '/opt/ml/input/mmsegmentation/configs/_base_/default_runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'  # noqa

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=11,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        # loss_decode=dict(
        #     _delete_=True,
        #     type='LovaszLoss', loss_name='loss_Lovasz',loss_weight=1.0, per_image = True)
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        # ]
    ),
    test_cfg=dict(_delete_=True, mode='whole')
)
albu_train_transforms = [
    dict(type='RandomResizedCrop',height= 512, width = 512, scale=(0.5, 1), ratio=(0.5, 2), p=1.0),
    dict(type='GridDropout',p=0.5),
]
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', coco_json_path='/opt/ml/input/data/train.json'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.25, 4.0)), ####### (384, 512), (512, 384), (448, 512), (512, 448), 
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False,
    ),
    dict(type='PhotoMetricDistortion'),
    # dict(type='RandomCutOut', prob=0.5, n_holes = 50, cutout_shape = (10, 10),  seg_fill_in=255),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),######
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512), #####################
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,###############
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)

# lr_config = dict(
#     _delete_=True,
#     policy='Cyclic',
#     by_epoch = False,
#     target_ratio=(1.0,0.005),
#     step_ratio_up=0.2,
#     cyclic_times=160,  ###############################
#     warmup = 'exp',
#     warmup_iters = 300,
#     warmup_ratio = 0.001,
#     warmup_by_epoch = False,
# )
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=16,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
workflow = [('train', 1)]  
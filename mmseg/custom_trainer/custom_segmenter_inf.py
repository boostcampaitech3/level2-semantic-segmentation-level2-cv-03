_base_ = [
    '/opt/ml/input/mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask.py',
    '/opt/ml/input/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
    '/opt/ml/input/mmsegmentation/_myexperiment/trash_dataset.py', ## fixed
    '/opt/ml/input/mmsegmentation/configs/_base_/default_runtime.py'
]

checkpoint = '/opt/ml/working/segmenter_vit-s_aug1234111_8/iter_3000.pth'  # noqa
# checkpoint = '/opt/ml/input/segmentation/mmsegmentation/_myexperiment/pretrain/vit_small_p16_384.pth'

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
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(
        _delete_=True,
        mode='whole')
)

optimizer = dict(lr=0.01, weight_decay=0.0)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', coco_json_path='/opt/ml/input/data/train.json'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=16,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
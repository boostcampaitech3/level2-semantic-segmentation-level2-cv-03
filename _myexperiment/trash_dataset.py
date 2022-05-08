# dataset settings
dataset_type = 'TrashDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CustomLoadAnnotations', coco_json_path='/opt/ml/input/data/train.json'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        coco_json_path='/opt/ml/input/data/train.json',
        is_valid=False,
        img_dir='/opt/ml/input/data',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        coco_json_path='/opt/ml/input/data/val.json',
        is_valid=True,
        img_dir='/opt/ml/input/data',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        coco_json_path='/opt/ml/input/data/test.json',
        is_valid=True,
        img_dir='/opt/ml/input/data',
        pipeline=test_pipeline)
)
_base_ = [
    '/opt/ml/input/mmsegmentation/configs/_base_/models/fcn_r50-d8.py', ## model
    '/opt/ml/input/mmsegmentation/configs/_base_/schedules/schedule_20k.py', ## scheduler
    '/opt/ml/input/mmsegmentation/_myexperiment/trash_dataset.py', ## dataset => 고정
    '/opt/ml/input/mmsegmentation/configs/_base_/default_runtime.py' ## runtime -> 수정필요
]

model = dict(
    decode_head=dict(
        num_classes=11,
    ),
    auxiliary_head=dict(
        num_classes=11,
    )
)
model = dict(
    decode_head=dict(loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    auxiliary_head=dict(loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce',loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    )


optimizer = dict(
    _delete_=True,
    type='Adam', 
    lr=3e-4
)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
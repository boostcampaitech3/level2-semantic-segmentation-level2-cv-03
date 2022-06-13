_base_ = [
    '/opt/ml/input/mmsegmentation/configs/_base_/models/fcn_hr18.py', ## model
    '/opt/ml/input/mmsegmentation/configs/_base_/schedules/schedule_160k.py',
    '/opt/ml/input/mmsegmentation/_myexperiment/trash_dataset.py',
    '/opt/ml/input/mmsegmentation/configs/_base_/default_runtime.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

model = dict(
    decode_head=dict(
        num_classes=11
    )
)
model = dict(
    decode_head=dict(loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
)

# optimizer = dict(
#     _delete_=True,
#     type='Adam', 
#     lr=3e-4
# )
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)
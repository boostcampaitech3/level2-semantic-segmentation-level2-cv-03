checkpoint_config = dict(interval=80000, max_keep_ckpts=3)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        
        # Wandb Logger 
        
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='hyunjin-segmentation',
                entity='cv-3-bitcoin',
                name='knet_upernet_swinl_AdamW_dropout_iter80000'
            ))
        
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# optimizer
optimizer = dict(
  type='AdamW', 
  lr=0.01, 
  momentum=0.9, 
  weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='Cyclic',
      base_lr=0.00005,
      cycle_momentum=False,
      gamma= 0.5,
      max_lr=0.000001,
      mode="exp_range",
      step_size_up=4)
      
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
# checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from pycocotools.coco import COCO
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2

import my_pipeline

#### Custom ####

# config file 들고오기
cfg = mmcv.Config.fromfile('/opt/ml/input/mmsegmentation/_myexperiment/custom_segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py')

# checkpoint path
checkpoint_path = '/opt/ml/working/segmenter_vit-b_mask_8x1_512x512_160k_ade20k.py_2/iter_60000.pth'

#저장할 csv 파일명 ***.csv로 적어주세요
csv_file_name = 'submission1.csv'

#### Custom ####

if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

cfg.model.pretrained = None
cfg.data.test.test_mode = True
# cfg.model.test_cfg=dict(mode='whole')

rank, _ = get_dist_info()

# build dataset & dataloader
dataset = build_dataset(cfg.data.test)
# The default loader config
loader_cfg = dict(
    # cfg.gpus will be ignored if distributed
    num_gpus=1,
    dist=False,
    shuffle=False)
# The overall dataloader settings
loader_cfg.update({
    k: v
    for k, v in cfg.data.items() if k not in [
        'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        'test_dataloader'
    ]
})
test_loader_cfg = {
    **loader_cfg,
    'samples_per_gpu': 1,
    'shuffle': False,  # Not shuffle by default
    **cfg.data.get('test_dataloader', {})
}
# build the dataloader
data_loader = build_dataloader(dataset, **test_loader_cfg)


cfg.model.train_cfg = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = dataset.CLASSES
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = dataset.PALETTE
# model.CLASSES = dataset.CLASSES
torch.cuda.empty_cache()

model = MMDataParallel(model, device_ids=[0])
output = single_gpu_test(
            model,
            data_loader)
# output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산



# submission 양식에 맞게 output 후처리
prediction_strings = []
file_names = []
coco = COCO(cfg.data.test.coco_json_path)
img_ids = coco.getImgIds()

submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
count = 0
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
for i, out in tqdm(enumerate(output)):

    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    print(out.shape)
    out = cv2.resize(out, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    submission = submission.append({"image_id" : image_info['file_name'], "PredictionString" : ' '.join(str(e.flatten())[1:-1] for e in out)},
                                   ignore_index=True)
  
submission.to_csv(csv_file_name, index=False)


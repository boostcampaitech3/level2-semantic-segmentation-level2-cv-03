import torch
import os
import os.path as osp
import glob
from pathlib import Path
import re


import mmcv

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import set_random_seed, train_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm

import my_pipeline

import wandb

def increment_path(model_dir, exp_name, exist_ok=False): ### Custom : 실험명 자동 네이밍 기능
    """ Automatically increment path, i.e. trained_models/exp --> trained_models/exp0, trained_models/exp1 etc.
    Args:
        exist_ok (bool): whether increment path (increment if False).
    """
    path = osp.join(model_dir, exp_name)
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(exp_name)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s_(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{exp_name}_{n}"


def check_envrionment():
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
    print('torch.cuda.get_device_name(0): ',torch.cuda.get_device_name(0))
    print('torch.cuda.device_count(): ',torch.cuda.device_count())

def exp_settings(cfg, seed, exp_name):

    cfg.seed = seed # seed 지정
    cfg.gpu_ids = range(1)
    set_random_seed(seed, deterministic=False)
    os.environ['PYTHONHASHSEED'] = str(seed)

    job_folder = f'./working/{exp_name}' # Folder to store model logs and weight files

    cfg.work_dir = job_folder
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir)) # Create work_dir
    print("Job folder:", job_folder)  

    ## config 파일 저장
    cfg.dump(osp.join(cfg.work_dir, exp_name))

    wandb.login()

    return cfg

def custom(): ## Custom settings
    wnb_project_name = 'cv-3-bitcoin'

    ## 변경 : 실험 환경 세팅 ##
    wnb_username = 'sunhyuk-segmentation' #
    seed = 21
    model_name = 'fcn_hr48_512x512_160k_ade20k'
    config_file_path = '/opt/ml/input/mmsegmentation/_myexperiment/my_hrnet_config.py' 
    exp_name = increment_path('./working', model_name)

    cfg = mmcv.Config.fromfile(config_file_path) 

    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='WandbLoggerHook',
            interval=50,
            init_kwargs=dict(project=wnb_username,entity=wnb_project_name, name=exp_name)
            )
        ]
    )
    cfg.data.samples_per_gpu = 32

    return cfg, seed, exp_name

if __name__ == '__main__':

    check_envrionment()
    params = custom()
    cfg = exp_settings(*params)

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model)
    model.init_weights()

    # config.model 에서 norm_cfg = dict(type='SyncBN', requires_grad=True) 라면 아래의 방법이 필요
    # 혹은 dict(type='BN', requires_grad=True)로 변환하여 아래 코드 주석 처리
    if cfg.norm_cfg.type == 'SyncBN':
        model = revert_sync_batchnorm(model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    print(model.CLASSES)

    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
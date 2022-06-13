import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
import random
import yaml
from utils import DictAsMember
from trainer import Trainer

def seed_everything(random_seed):
    # seed 고정
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DictAsMember(config)
    seed_everything(config.seed)

    trainer = Trainer(config, **config.trainer)
    trainer.trainer_train(config.train)
    
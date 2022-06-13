import warnings
warnings.filterwarnings("ignore")

import argparse
from importlib import import_module

from utils import *

from trainer import Inferencer
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DictAsMember(config)

    inferencer = Inferencer(**config.inferencer)

    inferencer.inferencer_inference(config.inference)
    
    print("Inference completed!")
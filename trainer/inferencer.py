import os
from importlib import import_module
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.transform import *
from dataset import CustomDataLoader
from utils import * # wnadb 관련 함수
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm
import ttach as tta

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            # inference (512 x 512)
            try:
                outs = model(torch.stack(imgs).to(device))['out']
            except TypeError:
                outs = model(torch.stack(imgs).to(device))

            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


class Inferencer:
    def __init__(self, model_path, test_path, save_dir):
        self.num_classes = 11
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.save_dir = save_dir # 결과 output 저장 위치
        makedirs(self.save_dir)
        
        self.dataset_path = "../data"
        self.test_path = self.dataset_path + test_path

        self.model_path = model_path
        self.model_name = '_'.join(self.model_path.split('/')[-2:])[:-3] + '.csv'
    
    def inferencer_inference(self, config):
        transform_module = getattr(import_module("dataset"), config.augmentation.name)
        test_transform = transform_module('test')

        if config.model.name.endswith('smp'):
            if config.model.preprocess == True:
                print('Preprocessing is undergoing!')
                preprocessing_fn = smp.encoders.get_preprocessing_fn(config.model.args.encoder_name, config.model.args.encoder_weights)
                preprocessing = get_preprocessing(preprocessing_fn)
            else:
                preprocessing = None    
        else:
            preprocessing = None
        
        test_dataset = CustomDataLoader(data_dir=self.test_path, mode='test', transform=test_transform, preprocessing=preprocessing)

        test_loader = DataLoader(dataset=test_dataset, 
                                collate_fn=collate_fn,
                                **config.test_data_loader)
        ## model
        model_module = getattr(import_module("model"), config.model.name)
        model = model_module(**config.model.args)
        # print(torch.load(self.model_path, map_location=self.device).keys())
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        ## tta transform
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90([0, 90, 180, 270]),
                tta.Rotate90([0, 90, 270]),
                tta.Resize([(256,256),(768,768)], (512,512)),
                # tta.Scale([1.5, 0.5]),
            ]
        )

        if config.augmentation.tta:
            print('----TTA is applied!!----')
            model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='max')

        model.to(self.device)
        model.eval()

        # sample_submisson.csv 열기
        submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

        # test set에 대한 prediction
        file_names, preds = test(model, test_loader, self.device)

        # PredictionString 대입
        for file_name, string in zip(file_names, preds):
            submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                        ignore_index=True)

        # submission.csv로 저장
        if config.augmentation.tta:
            csv_name = 'tta_' + self.model_name
            submission.to_csv(os.path.join(self.save_dir, csv_name), index=False)
        else:
            submission.to_csv(os.path.join(self.save_dir, self.model_name), index=False)





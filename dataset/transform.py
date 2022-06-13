import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VariousAugment:
    def __init__(self, flag='train'):
        self.flag = flag
        if self.flag == 'train':
            self.transform = A.Compose([
                A.RandomScale(scale_limit=[0.9, 1.5], p=0.7),
                A.HorizontalFlip(p=0.5),
                A.augmentations.transforms.GaussNoise(var_limit=(10.0, 20.0), p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256,width=256), # mask 있는 곳 위주로 crop
                A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=1.0),
                ToTensorV2()
            ])
        
        elif self.flag == 'val':
            self.transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ])
        
        elif self.flag == 'test':
            self.transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ])
        
    def __call__(self, image, mask=None):
        if self.flag == 'test':
            return self.transform(image=image)
        return self.transform(image=image, mask=mask)


class BasicTransform:
    def __init__(self, flag='train'):
        self.flag = flag
        if self.flag == 'train':
            self.transform = A.Compose([
                ToTensorV2()
            ])
        
        elif self.flag == 'val':
            self.transform = A.Compose([
                ToTensorV2()
            ])
        
        elif self.flag == 'test':
            self.transform = A.Compose([
                ToTensorV2()
            ])
    
    def __call__(self, image, mask=None):
        if self.flag == 'test':
            return self.transform(image=image)
        return self.transform(image=image, mask=mask)


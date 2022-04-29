import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BasicTransform:
    def __init__(self, val_flag=False, preprocessing_fn=None):
        if not val_flag: # augmentation while training
            self.transform = A.Compose([
                                A.Lambda(image=preprocessing_fn),
                                ToTensorV2()
                            ])
        else: # augmetation while validating
            self.transform = A.Compose([
                                A.Lambda(image=preprocessing_fn),
                                ToTensorV2()
                            ])
    
    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


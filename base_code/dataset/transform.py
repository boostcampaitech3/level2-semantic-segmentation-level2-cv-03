import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BasicTransform:
    def __init__(self, val_flag=False):
        if not val_flag: # augmentation while training
            self.transform = A.Compose([
                                A.HorizontalFlip(),
                                # A.normalize(mean, std),
                                A.GridDropout(holes_number_x=3, holes_number_y=3, p=1, random_offset=True),
                                # A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256,width=256),
                                ToTensorV2(),
                                
                            ])
        else: # augmetation while validating
            self.transform = A.Compose([
                                ToTensorV2()
                            ])
    
    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


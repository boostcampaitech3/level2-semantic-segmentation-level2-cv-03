import torch.nn as nn
import torchvision
from torchvision import models

class FCN_Resnet_50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    def forward(self, images):
        return self.model(images)

class Deeplabv3_resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, 11, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, 11, kernel_size=(1, 1), stride=(1,1))
    def forward(self, images):
        return self.model(images)

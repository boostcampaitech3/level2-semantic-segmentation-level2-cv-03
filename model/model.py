import torch.nn as nn
import torchvision
from torchvision import models

import segmentation_models_pytorch as smp



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


# ENCODER = 'tu-xception65'
# ENCODER_WEIGHTS = 'imagenet'
# # ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

# # create segmentation model with pretrained encoder
# model = smp.DeepLabV3Plus(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS,
#     in_channels=3,
#     classes=11,
# )

# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


def build_model(config):
    decoder = getattr(smp, config.decoder)
    model = decoder(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=11,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.encoder, config.encoder_weights)
    return model, preprocessing_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

_criterion_entrypoints = {
    'cross_entropy': "CrossEntropy",
    'focal':"FocalLoss",
    'diceloss':'DiceCELoss'
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]

class DiceCELoss(nn.Module):
    '''
    CrossEntropy_loss + Dice_loss에 서로 다른 가중치를 주어 사용하는 loss   
    semantic segmentation에서 이미지의 pixel 위치를 맞추는 것도 중요하지만 class를 
    틀린다면 결과값은 0이 될 것이다. 따라서 class 분류에 좋은 성능을 보이는 CE_loss와 
    Pixel 위치를 맞추는 Dice Loss를 혼합하여 가중치를 적용. 
    Kaggle에서도 자주 사용되며 1위 solution으로 0.75*CE_Loss+0.25*Dice_loss 를 사용한
    방법이 효과적이었다.
    '''

    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + 1e-7)).mean()
        dice_loss = (1 - dice_loss)

        ce = F.cross_entropy(inputs, targets, reduction='mean')
        dice_bce = ce * 0.75 + dice_loss * 0.25
        return dice_bce

CrossEntropy = nn.CrossEntropyLoss

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
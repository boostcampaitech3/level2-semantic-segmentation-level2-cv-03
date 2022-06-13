import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

_criterion_entrypoints = {
    'cross_entropy': "CrossEntropy",
    'DiceCELoss' : 'DiceCELoss',
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


CrossEntropy = nn.CrossEntropyLoss

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
        dice_loss = (1 - dice_loss)

        ce = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        dice_bce = ce * 0.7 + dice_loss * 0.3
        return dice_bce
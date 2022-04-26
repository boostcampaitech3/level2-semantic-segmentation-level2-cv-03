import torch
import torch.nn as nn
import torch.optim as optim

_criterion_entrypoints = {
    'cross_entropy': "CrossEntropy",
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


CrossEntropy = nn.CrossEntropyLoss
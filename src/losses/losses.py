import torch
from .focal_loss import FocalLoss

KEY2LOSSES = {
    "bce" : torch.nn.BCEWithLogitsLoss,
    'ce': torch.nn.CrossEntropyLoss,
    'focal': FocalLoss,
}
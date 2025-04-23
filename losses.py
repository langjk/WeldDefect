# losses.py
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import FocalLoss
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.82, gamma=2):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma) 

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return dice_loss + focal_loss

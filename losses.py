# losses.py
import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.losses import TverskyLoss

class DiceTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.tversky(inputs, targets)

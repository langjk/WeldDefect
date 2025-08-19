# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.losses import TverskyLoss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in dense prediction - AutoCast Safe"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用logits版本以兼容autocast
        # 假设inputs是logits（未经sigmoid的原始输出）
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 使用BCE with logits（autocast安全）
        BCE_logits = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率用于focal权重
        pt = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, pt, 1 - pt)
        
        # 计算focal loss权重
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * BCE_logits

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.tversky(inputs, targets)

class ImprovedLoss(nn.Module):
    """Improved loss function combining Dice, Tversky, and Focal losses"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.tversky = TverskyLoss(sigmoid=True, alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    def forward(self, inputs, targets):
        # 所有损失函数都期望logits输入（未经sigmoid激活）
        # MONAI的DiceLoss和TverskyLoss设置了sigmoid=True，会内部处理
        # FocalLoss也已修改为处理logits输入
        dice_loss = self.dice(inputs, targets)
        tversky_loss = self.tversky(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        # 加权组合 - 强调精确率（减少假阳性）
        return dice_loss + tversky_loss + 0.5 * focal_loss

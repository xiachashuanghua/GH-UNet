import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceIoULoss', 'LovaszHingeLoss',]


class BCEDiceIoULoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=1.5, iou_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(self, input, target):
        # Binary Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(input, target)

        # Dice Loss
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input_flat = input.contiguous().view(num, -1)
        target_flat = target.contiguous().view(num, -1)
        intersection = (input_flat * target_flat)
        dice = (2. * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice.sum() / num

        # IoU Loss
        union = input_flat.sum(1) + target_flat.sum(1) - intersection.sum(1)
        iou = (intersection.sum(1) + smooth) / (union + smooth)
        iou_loss = 1 - iou.sum() / num

        # Combined Loss
        combined_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss) + (self.iou_weight * iou_loss)
        return combined_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss



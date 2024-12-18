import torch
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
from sklearn.metrics import roc_auc_score

import numpy as np
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output >= 0.5
    target_ = target >= 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output >= 0.5
    target_ = target >= 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    if target_.sum() > 0 and output_.sum()>0:
        hd_ = hd(output_, target_)
        hd95_ = hd95(output_, target_)
    else:
        hd_=0
        hd95_=0
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    TP = np.logical_and(output_ == 1, target_ == 1).sum()
    TN = np.logical_and(output_ == 0, target_ == 0).sum()
    FP = np.logical_and(output_ == 1, target_ == 0).sum()
    FN = np.logical_and(output_ == 0, target_ == 1).sum()

    # Accuracy
    acc_ = (TP + TN) / (TP + TN + FP + FN)
    mae_ = np.mean(np.logical_xor(output_, target_))
    output_flat = output_.flatten()
    target_flat = target_.flatten()

    # 计算AUC
    try:
        auc_ = roc_auc_score(target_flat, output_flat)
    except ValueError:
        auc_ = 0  # 如果无法计算 AUC（例如，所有标签都是同一个类），返回 0



    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_,acc_,mae_,auc_









import torch
import torch.nn.functional as F
from torch import nn



import torch

def soft_dice_loss(pred, targets):
    num = targets.size(0)
    smooth = 1e-6
    probs = torch.sigmoid(pred)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)

    intersection = (m1 * m2)

    dice = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    dice_loss = 1 - (dice.sum() / num)
    return dice_loss

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2):

    # alpha: 平衡因子    gamma: 调制因子

    be_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # 计算二元交叉熵损失
    pt = torch.exp(-be_loss)  # 计算 e^(-ce_loss)，用于调制难易样本的权重
    focal_loss = alpha * (1 - pt)**gamma * be_loss  # 计算 Focal Loss
    return focal_loss.mean()  # 返回平均的 Focal Loss

def BCE_Focal( predict, target):
    gamma = 2
    alpha = 0.25
    pt = torch.sigmoid(predict)  # sigmoide获取概率
    loss = - alpha * (1 - pt) ** gamma * target * torch.log(pt) - (1 - alpha) * pt ** gamma * (1 - target) * torch.log(1 - pt)
    loss = torch.mean(loss)
    return loss

def bce_dice_loss(pred, targets):


    criterionBCE = torch.nn.BCEWithLogitsLoss()

    loss = criterionBCE(pred, targets)

    loss += soft_dice_loss(pred, targets)
    return loss

def gmm_bce_dice_loss(pred, targets):

    # loss = torch.mean(- (targets * torch.log((torch.sigmoid(pred)))))

    num = targets.size(0)
    smooth = 1e-6
    probs = torch.sigmoid(pred)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)

    intersection = (m1 * m2)

    dice = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    loss = 1 - (dice.sum() / num)

    return loss


def wight_bce_loss(pred, targets):

    criterionBCE = torch.nn.BCELoss(weight=torch.tensor(0.3)).cuda()

    loss = criterionBCE(pred, targets)

    return loss

def bce_logits_loss(pred, targets):

    criterionBCE = torch.nn.BCEWithLogitsLoss()

    loss = criterionBCE(pred, targets)

    return loss


def KL_loss(pred, gt):

    pred = pred + 1e-8
    gt = gt + 1e-8
    loss = F.kl_div(torch.log(pred), gt)

    return loss





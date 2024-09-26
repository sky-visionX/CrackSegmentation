import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mmseg.models.sam.clip_text import get_data
from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


# 换decoder版本


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


def soft_dice_loss(pred, targets):
    num = targets.size(0)
    smooth = 1
    probs = torch.sigmoid(pred)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)

    intersection = (m1 * m2)

    dice = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    dice_loss = 1 - dice.sum() / num
    #print(dice_loss)
    return dice_loss

def soft_dice_loss_2(pred, targets):
    num = targets.size(0)
    smooth = 1e-10
    probs = torch.sigmoid(pred)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)

    intersection = (m1 * m2)

    # Filter out only the pixels where target is 1 and prediction is 0
    relevant_pixels = (m2 == 1) & (m1 < 0.5)

    if relevant_pixels.sum() == 0:
        return torch.tensor(0.0)  # Return 0 loss if no relevant pixels found

    dice_numerator = 2. * intersection[relevant_pixels].sum() + smooth
    dice_denominator = m1[relevant_pixels].sum() + m2[relevant_pixels].sum() + smooth

    dice = dice_numerator / dice_denominator
    dice_loss = 1 - dice
    # print(dice_loss)
    return dice_loss


def weighted_bce_loss(pred, targets, weight_factor=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(pred, targets, reduction='none')

    weighted_bce_loss = bce_loss + weight_factor * (targets * (1 - torch.sigmoid(pred)))

    return weighted_bce_loss.mean()

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2):

    # alpha: 平衡因子    gamma: 调制因子

    # logits = torch.sigmoid()   不需要经过这个，bce中已经实现了

    be_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # 计算二元交叉熵损失
    pt = torch.exp(-be_loss)  # 计算 e^(-ce_loss)，用于调制难易样本的权重
    focal_loss = alpha * (1 - pt)**gamma * be_loss  # 计算 Focal Loss

    return focal_loss.mean()  # 返回平均的 Focal Loss

def gradient_harmonized_loss(logits, targets, bins=30, alpha=0.75, beta=1.0):
    # 将 logits 转换为概率值，使用 sigmoid 函数（假设 logits 不是概率值）
    logits = torch.sigmoid(logits)

    # 计算预测概率和预测概率与真实标签的差异
    prob = torch.sigmoid(logits)  # 将 logits 转换为概率值
    diff = torch.abs(prob - targets)  # 计算预测概率与真实标签的差异

    # 初始化一个空的张量以存储权重
    weights = torch.zeros_like(diff)

    # 定义用于划分差异的区间边界并计算区间索引
    edges = torch.arange(0, 1 + 1e-6, 1.0 / bins)
    edges = edges.cuda()
    inds = torch.bucketize(diff, edges)
    total = diff.numel()  # 元素总数

    # 遍历区间以计算权重
    for i in range(bins):
        mask = inds == i  # 创建当前区间的布尔掩码
        num_in_bin = mask.sum()  # 计算当前区间中的元素数量
        if num_in_bin > 0:
            grad_in_bin = num_in_bin / total  # 计算当前区间内的梯度
            weights[mask] = 1.0 / (grad_in_bin * bins)  # 基于梯度分配权重

    # 使用 alpha 和 beta 参数调整权重，并将其从计算图中分离
    weights = (weights * (alpha - beta) + beta).detach()

    # 使用加权二元交叉熵计算 GHM 损失
    ghm_loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction='mean')

    return ghm_loss


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_classes=1
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False



        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask):

        self.input = input.to(self.device)

        get_data(self.input)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = 1

        # Embed prompts

        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks


    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)

        # Predict masks
        # low_res_masks, iou_predictions = self.mask_decoder(
        #     image_embeddings=self.features,
        #     image_pe=self.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=False,
        # )

        masks = self.mask_decoder(
            self.features
        )

        # Upscale the masks to the original image resolution
        # masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self, eopch):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)

        # self.loss_G = binary_focal_loss(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += soft_dice_loss(self.pred_mask, self.gt_mask)
            # if eopch > 30:
            #     weight_factor = 40
            #     # bce_loss = F.binary_cross_entropy_with_logits(self.pred_mask, self.gt_mask, reduction='none')
            #     r = weight_factor * (self.gt_mask * (1 - torch.sigmoid(self.pred_mask)))
            #     r = r.mean()
            #     self.loss_G += r
            #     # print(self.loss_G)
            # else:
            #     self.loss_G += soft_dice_loss(self.pred_mask, self.gt_mask)

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        # print(epoch)
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G(epoch)  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

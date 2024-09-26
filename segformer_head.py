import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F


class SegformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim):
        super(SegformerHead, self).__init__()
        self.linear_c4 = nn.Linear(in_channels[3], embed_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embed_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embed_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embed_dim)
        self.linear_fuse = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c1.shape

        _c4 = self.linear_c4(c4.flatten(2).transpose(1, 2)).permute(0, 2, 1).view(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3.flatten(2).transpose(1, 2)).permute(0, 2, 1).view(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2.flatten(2).transpose(1, 2)).permute(0, 2, 1).view(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1.flatten(2).transpose(1, 2)).permute(0, 2, 1).view(n, -1, c1.shape[2], c1.shape[3])

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        fused = self.linear_fuse(_c)
        fused = self.dropout(fused)
        logits = self.linear_pred(fused)
        logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)

        return logits

if __name__ == '__main__':
    # 输入特征图列表
    out = [torch.rand(1, 32, 256, 256), torch.rand(1, 64, 128, 128), torch.rand(1, 128, 64, 64),
           torch.rand(1, 256, 32, 32)]

    # 初始化SegformerHead
    in_channels = [32, 64, 128, 256]  # 与输入特征图通道数匹配
    embed_dim = 256
    num_classes = 1
    head = SegformerHead(in_channels, num_classes, embed_dim)

    # 前向传播
    output = head(out)
    print(output.shape)  # 应该是torch.Size([1, 1, 1024, 1024])






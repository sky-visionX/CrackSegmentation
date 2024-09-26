import torch
from torch import nn

class MaskDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(MaskDecoder, self).__init__()

        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.cls_seg = nn.Conv2d(256, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        x = self.UP_stage_4(x)
        x = self.cls_seg(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn(1, 256, 64, 64)

    dercoder = MaskDecoder(num_classes=1)

    out = dercoder(inputs)

    print(out.shape)
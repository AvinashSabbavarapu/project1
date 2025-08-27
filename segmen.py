import torch
import torch.nn as nn
import torchvision.models as models

class Segmentor(nn.Module):
    def __init__(self, num_classes=91):  # COCO has 80+ categories
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, H/32, W/32]

        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        features = self.encoder(x)
        masks = self.seg_head(features)
        return masks


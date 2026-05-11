# -*- coding: utf-8 -*-
"""VT-CNN2 风格轻量模型。

原始 VT-CNN2 常用于 RadioML AMR。本实现更轻，适合 CPU 运行。
"""
import torch
from torch import nn

class VTCNN2Lite(nn.Module):
    def __init__(self, num_classes=11, input_channels=2, feature_dim=128, dropout=0.5):
        super().__init__()
        # 将输入 [B,2,128] 视为 [B,1,2,128] 做 2D 卷积
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=(2, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.flatten_dim = 64 * 1 * 128
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        x = x.unsqueeze(1)  # [B,1,2,128]
        h = self.features(x)
        h = h.flatten(1)
        h = self.fc1(h)
        feat = self.fc2(h)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

# -*- coding: utf-8 -*-
"""轻量 1D ResNet。CPU 上可作为扩展模型。"""
import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))

class ResNet1DLite(nn.Module):
    def __init__(self, num_classes=11, input_channels=2, feature_dim=128, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(BasicBlock1D(64), BasicBlock1D(64), BasicBlock1D(64))
        self.fc = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        h = self.blocks(self.stem(x))
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        feat = self.fc(h)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

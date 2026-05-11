# -*- coding: utf-8 -*-
"""轻量 1D CNN。CPU 友好。"""
import torch
from torch import nn
import torch.nn.functional as F

class CNN1DLite(nn.Module):
    def __init__(self, num_classes=11, input_channels=2, feature_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        h = self.conv(x)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        feat = self.fc(h)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

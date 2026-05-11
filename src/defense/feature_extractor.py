# -*- coding: utf-8 -*-
"""特征提取模块。"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.io import ensure_dir


def extract_features(model, dataset, device, batch_size=512, save_prefix=None):
    """提取模型倒数第二层特征。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    feats, labels, snrs, indices, is_poison = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            logits, feat = model(x, return_features=True)
            feats.append(feat.cpu().numpy())
            labels.append(batch['y'].numpy())
            snrs.append(batch['snr'].numpy())
            indices.append(batch['index'].numpy())
            is_poison.append(batch['is_poison'].numpy())
    out = {
        'features': np.concatenate(feats, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'snr': np.concatenate(snrs, axis=0),
        'indices': np.concatenate(indices, axis=0),
        'is_poison': np.concatenate(is_poison, axis=0),
    }
    if save_prefix is not None:
        save_prefix = Path(save_prefix)
        ensure_dir(save_prefix.parent)
        np.savez_compressed(save_prefix, **out)
    return out

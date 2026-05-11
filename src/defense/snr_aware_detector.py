# -*- coding: utf-8 -*-
"""SNR-aware 特征异常检测。"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import ensure_dir


def assign_snr_bin(snr, snr_bins):
    """将一个 SNR 值分配到 low/mid/high 等 bin。"""
    for name, bounds in snr_bins.items():
        lo, hi = bounds
        if lo <= snr <= hi:
            return name
    return 'other'


def snr_aware_feature_scores(features, labels, snrs, indices, is_poison=None, snr_bins=None, percentile=95, save_path=None):
    """按类别和 SNR bin 计算 diagonal Mahalanobis 异常分数。"""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    snrs = np.asarray(snrs)
    indices = np.asarray(indices)
    if is_poison is None:
        is_poison = np.zeros(len(labels), dtype=np.int64)
    if snr_bins is None:
        snr_bins = {'low': [-20, -8], 'mid': [-6, 6], 'high': [8, 18]}

    bins = np.array([assign_snr_bin(int(s), snr_bins) for s in snrs])
    scores = np.zeros(len(labels), dtype=np.float32)
    suspicious = np.zeros(len(labels), dtype=np.int64)

    for label in np.unique(labels):
        for b in np.unique(bins):
            mask = (labels == label) & (bins == b)
            idx = np.where(mask)[0]
            if len(idx) < 5:
                continue
            f = features[idx]
            mean = f.mean(axis=0, keepdims=True)
            var = f.var(axis=0, keepdims=True) + 1e-6
            # diagonal Mahalanobis，避免 CPU 上完整协方差求逆过慢或奇异
            dist = np.mean((f - mean) ** 2 / var, axis=1)
            scores[idx] = dist
            th = np.percentile(dist, percentile)
            suspicious[idx] = (dist >= th).astype(np.int64)

    df = pd.DataFrame({
        'index': indices,
        'label': labels,
        'snr': snrs,
        'snr_bin': bins,
        'score_feature': scores,
        'is_suspicious_feature': suspicious,
        'is_poison': is_poison,
    })
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return df

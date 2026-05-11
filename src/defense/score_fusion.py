# -*- coding: utf-8 -*-
"""检测分数融合。"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.metrics import binary_detection_metrics
from src.utils.io import ensure_dir, save_json


def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def fuse_scores(feature_df, dual_df=None, alpha=0.6, beta=0.4, percentile=95, save_path=None, metric_path=None):
    """融合 SNR-aware feature score 和 dual-domain score。"""
    df = feature_df.copy()
    df['score_feature_norm'] = _minmax(df['score_feature'].values)

    if dual_df is not None and len(dual_df) > 0:
        dual_small = dual_df[['index', 'score_dual']].copy()
        dual_small['score_dual_norm'] = _minmax(dual_small['score_dual'].values)
        df = df.merge(dual_small[['index', 'score_dual_norm']], on='index', how='left')
        df['score_dual_norm'] = df['score_dual_norm'].fillna(0.0)
    else:
        df['score_dual_norm'] = 0.0

    df['score_final'] = alpha * df['score_feature_norm'] + beta * df['score_dual_norm']
    th = np.percentile(df['score_final'].values, percentile)
    df['is_suspicious_final'] = (df['score_final'] >= th).astype(int)

    metrics = None
    if 'is_poison' in df.columns:
        metrics = binary_detection_metrics(df['is_suspicious_final'].values, df['is_poison'].values)
        metrics['threshold'] = float(th)

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    if metric_path is not None and metrics is not None:
        save_json(metrics, metric_path)
    return df, metrics

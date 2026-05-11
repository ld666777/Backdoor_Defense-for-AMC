# -*- coding: utf-8 -*-
"""可疑样本过滤。"""
import numpy as np
from src.data.dataset import make_subset_dataset


def filter_suspicious_samples(dataset, fused_df):
    """删除最终标记为可疑的训练样本。"""
    # dataset.indices 是原始样本索引，fused_df.index 字段也是原始索引
    suspicious_indices = set(fused_df.loc[fused_df['is_suspicious_final'] == 1, 'index'].astype(int).tolist())
    keep_mask = np.array([int(idx) not in suspicious_indices for idx in dataset.indices], dtype=bool)
    return make_subset_dataset(dataset, keep_mask), keep_mask

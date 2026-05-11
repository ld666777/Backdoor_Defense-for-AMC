# -*- coding: utf-8 -*-
"""数据划分工具。"""
import numpy as np


def stratified_split_by_label_snr(y, snr, train_ratio=0.6, val_ratio=0.2, seed=2026):
    """按 label + SNR 分层划分，避免某些 SNR 在测试集中缺失。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    snr = np.asarray(snr)
    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(y):
        for s in np.unique(snr):
            idx = np.where((y == label) & (snr == s))[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

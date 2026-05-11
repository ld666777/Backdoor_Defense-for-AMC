# -*- coding: utf-8 -*-
"""数据预处理。"""
from pathlib import Path
import numpy as np
from .rml2016_loader import load_rml2016a
from .split import stratified_split_by_label_snr
from src.utils.io import ensure_dir, save_json, save_torch


def normalize_iq_per_sample(x, eps=1e-8):
    """逐样本能量归一化。

    x shape: [N, 2, 128]
    """
    power = np.mean(x[:, 0, :] ** 2 + x[:, 1, :] ** 2, axis=1, keepdims=True)
    scale = np.sqrt(power + eps).reshape(-1, 1, 1)
    return x / scale


def filter_data(x, y, snr, mod_names, selected_snr='all', selected_mods='all'):
    """按 SNR 和调制类型筛选数据。"""
    mask = np.ones(len(y), dtype=bool)
    if selected_snr != 'all' and selected_snr is not None:
        mask &= np.isin(snr, np.asarray(selected_snr, dtype=np.int64))
    if selected_mods != 'all' and selected_mods is not None:
        mod_to_label = {m: i for i, m in enumerate(mod_names)}
        labels = [mod_to_label[m] for m in selected_mods]
        mask &= np.isin(y, labels)
    return x[mask], y[mask], snr[mask]


def prepare_rml_data(cfg, logger=None):
    """读取、筛选、归一化并划分数据。"""
    if logger:
        logger.info('正在读取 RML2016.10a 数据集')
    raw = load_rml2016a(cfg.paths.raw_path)
    x, y, snr = raw['x'], raw['y'], raw['snr']

    x, y, snr = filter_data(
        x, y, snr,
        raw['mod_names'],
        selected_snr=cfg.data.selected_snr,
        selected_mods=cfg.data.selected_mods,
    )

    if cfg.data.get('sample_limit', None):
        limit = int(cfg.data.sample_limit)
        x, y, snr = x[:limit], y[:limit], snr[:limit]

    if bool(cfg.data.normalize):
        if logger:
            logger.info('正在进行逐样本 IQ 能量归一化')
        x = normalize_iq_per_sample(x)

    train_idx, val_idx, test_idx = stratified_split_by_label_snr(
        y, snr,
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.project.seed),
    )

    processed_dir = ensure_dir(cfg.paths.processed_dir)
    save_torch({'x': x[train_idx], 'y': y[train_idx], 'snr': snr[train_idx]}, processed_dir / 'train.pt')
    save_torch({'x': x[val_idx], 'y': y[val_idx], 'snr': snr[val_idx]}, processed_dir / 'val.pt')
    save_torch({'x': x[test_idx], 'y': y[test_idx], 'snr': snr[test_idx]}, processed_dir / 'test.pt')

    meta = {
        'mod_names': raw['mod_names'],
        'snr_values': sorted([int(v) for v in np.unique(snr)]),
        'label_to_mod': {str(k): v for k, v in raw['label_to_mod'].items()},
        'mod_to_label': raw['mod_to_label'],
        'num_train': int(len(train_idx)),
        'num_val': int(len(val_idx)),
        'num_test': int(len(test_idx)),
    }
    save_json(meta, processed_dir / 'meta.json')

    if logger:
        logger.info(f'数据预处理完成：train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')
    return meta

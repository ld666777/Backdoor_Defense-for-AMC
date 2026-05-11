# -*- coding: utf-8 -*-
"""训练集投毒。"""
from pathlib import Path
import numpy as np
from src.utils.io import load_torch, save_torch, ensure_dir, load_json
from .attack_factory import apply_trigger


def make_poisoned_train_set(cfg, logger=None):
    """生成后门训练集。

    逻辑：从非目标类样本中随机选择 poison_rate 比例，加入触发器并改成目标标签。
    """
    train_path = Path(cfg.paths.processed_dir) / 'train.pt'
    meta_path = Path(cfg.paths.processed_dir) / 'meta.json'
    data = load_torch(train_path)
    meta = load_json(meta_path)

    mod_to_label = meta['mod_to_label']
    target_class = cfg.attack.target_class
    if target_class not in mod_to_label:
        raise ValueError(f'目标类别 {target_class} 不在数据集中，可选：{list(mod_to_label.keys())}')
    target_label = int(mod_to_label[target_class])

    x = np.asarray(data['x'], dtype=np.float32).copy()
    y = np.asarray(data['y'], dtype=np.int64).copy()
    snr = np.asarray(data['snr'], dtype=np.int64).copy()

    rng = np.random.default_rng(int(cfg.project.seed))
    candidate = np.where(y != target_label)[0]
    num_poison = max(1, int(len(candidate) * float(cfg.attack.poison_rate)))
    poison_idx = rng.choice(candidate, size=num_poison, replace=False)

    x[poison_idx] = apply_trigger(x[poison_idx], cfg)
    y[poison_idx] = target_label
    is_poison = np.zeros(len(y), dtype=np.int64)
    is_poison[poison_idx] = 1

    poison_dir = ensure_dir(cfg.paths.poisoned_dir)
    out_name = f"train_poison_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{target_class}.pt"
    out_path = poison_dir / out_name
    save_torch({'x': x, 'y': y, 'snr': snr, 'is_poison': is_poison, 'poison_indices': poison_idx}, out_path)

    if logger:
        logger.info(f'投毒训练集已保存：{out_path}')
        logger.info(f'投毒样本数：{num_poison}, 目标类别：{target_class}({target_label})')
    return out_path

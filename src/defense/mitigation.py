# -*- coding: utf-8 -*-
"""后门模型修复。"""
from pathlib import Path
import torch
from src.utils.io import load_torch, ensure_dir
from src.data.dataset import RMLDataset
from src.models.model_factory import build_model
from src.train.repair_trainer import RepairTrainer
from src.defense.sample_filter import filter_suspicious_samples


def _backdoor_ckpt_path(cfg):
    return Path(cfg.paths.checkpoint_dir) / 'backdoor' / f'{cfg.model.name}_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{cfg.attack.target_class}.pt'


def repair_backdoor_model(cfg, device, fused_df, logger=None):
    """基于检测结果过滤训练集，然后对后门模型进行小学习率修复。"""
    poison_path = Path(cfg.paths.poisoned_dir) / f"train_poison_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{cfg.attack.target_class}.pt"
    poison_data = load_torch(poison_path)
    val_data = load_torch(Path(cfg.paths.processed_dir) / 'val.pt')
    train_ds = RMLDataset(poison_data, is_poison=poison_data.get('is_poison', None))
    val_ds = RMLDataset(val_data)

    filtered_ds, keep_mask = filter_suspicious_samples(train_ds, fused_df)
    if logger:
        logger.info(f'修复训练：原训练样本 {len(train_ds)}，过滤后 {len(filtered_ds)}')

    model = build_model(cfg).to(device)
    ckpt = torch.load(_backdoor_ckpt_path(cfg), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    save_path = ensure_dir(Path(cfg.paths.checkpoint_dir) / 'repaired') / f'{cfg.model.name}_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{cfg.attack.target_class}_repaired.pt'
    log_csv_path = Path(cfg.paths.result_dir) / 'csv' / 'repair_train_log.csv'

    trainer = RepairTrainer(model, device, logger)
    return trainer.fit(filtered_ds, val_ds, cfg, save_path, log_csv_path)

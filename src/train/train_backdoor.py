# -*- coding: utf-8 -*-
"""训练后门 AMR 模型。"""
from pathlib import Path
from src.utils.io import load_torch, ensure_dir
from src.data.dataset import RMLDataset
from src.models.model_factory import build_model
from .trainer import Trainer


def _poison_train_path(cfg):
    target = cfg.attack.target_class
    return Path(cfg.paths.poisoned_dir) / f"train_poison_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{target}.pt"


def train_backdoor_model(cfg, device, logger=None):
    poison_path = _poison_train_path(cfg)
    poison_data = load_torch(poison_path)
    val_data = load_torch(Path(cfg.paths.processed_dir) / 'val.pt')
    train_ds = RMLDataset(poison_data, is_poison=poison_data.get('is_poison', None))
    val_ds = RMLDataset(val_data)

    model = build_model(cfg).to(device)
    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoint_dir) / 'backdoor')
    save_path = ckpt_dir / f'{cfg.model.name}_{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{cfg.attack.target_class}.pt'
    log_csv_path = Path(cfg.paths.result_dir) / 'csv' / 'backdoor_train_log.csv'

    if logger:
        logger.info('开始训练后门模型，验证集保持干净')
    trainer = Trainer(model, device, logger)
    return trainer.fit(train_ds, val_ds, cfg, save_path, log_csv_path)

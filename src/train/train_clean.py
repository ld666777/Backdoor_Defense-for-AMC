# -*- coding: utf-8 -*-
"""训练干净 AMR 模型。"""
from pathlib import Path
from src.utils.io import load_torch, ensure_dir
from src.data.dataset import RMLDataset
from src.models.model_factory import build_model
from .trainer import Trainer


def train_clean_model(cfg, device, logger=None):
    train_data = load_torch(Path(cfg.paths.processed_dir) / 'train.pt')
    val_data = load_torch(Path(cfg.paths.processed_dir) / 'val.pt')
    train_ds = RMLDataset(train_data)
    val_ds = RMLDataset(val_data)

    model = build_model(cfg).to(device)
    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoint_dir) / 'clean')
    save_path = ckpt_dir / f'{cfg.model.name}_clean.pt'
    log_csv_path = Path(cfg.paths.result_dir) / 'csv' / 'clean_train_log.csv'

    if logger:
        logger.info('开始训练干净模型')
    trainer = Trainer(model, device, logger)
    return trainer.fit(train_ds, val_ds, cfg, save_path, log_csv_path)

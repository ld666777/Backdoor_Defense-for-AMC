# -*- coding: utf-8 -*-
"""带 RF 一致性正则的修复训练器。"""
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.attacks.trigger_phase import add_phase_trigger
from src.attacks.trigger_time import add_time_trigger
from src.train.trainer import run_one_epoch
from src.utils.io import ensure_dir, save_csv


def rf_consistency_loss(model, batch, logits, weight=0.2):
    """RF 一致性损失。

    对样本做轻微相位旋转和微小时间扰动，要求预测分布接近。
    """
    x = batch['x']
    with torch.no_grad():
        p0 = F.softmax(logits, dim=1)
    x_aug = add_phase_trigger(x, length=32, angle=0.08, position='middle')
    x_aug = add_time_trigger(x_aug, length=4, amplitude=0.01, position='tail')
    logits_aug = model(x_aug)
    p1 = F.softmax(logits_aug, dim=1)
    return weight * F.mse_loss(p1, p0)


class RepairTrainer:
    def __init__(self, model, device, logger=None):
        self.model = model
        self.device = device
        self.logger = logger

    def fit(self, train_dataset, val_dataset, cfg, save_path, log_csv_path=None):
        batch_size = int(cfg.repair.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.repair.lr), weight_decay=float(cfg.train.weight_decay))
        save_path = Path(save_path)
        ensure_dir(save_path.parent)

        best_acc, best_epoch = -1.0, -1
        bad_count = 0
        patience = int(cfg.repair.get('early_stop_patience', 4))
        rows = []

        def extra(model, batch, logits):
            if bool(cfg.repair.get('use_consistency_loss', True)):
                return rf_consistency_loss(model, batch, logits, weight=float(cfg.repair.lambda_consistency))
            return 0.0

        for epoch in range(1, int(cfg.repair.epochs) + 1):
            tr_loss, tr_acc = run_one_epoch(self.model, train_loader, optimizer, self.device, train=True, extra_loss_fn=extra)
            va_loss, va_acc = run_one_epoch(self.model, val_loader, optimizer, self.device, train=False)
            rows.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})
            if self.logger:
                self.logger.info(f'修复 Epoch {epoch:03d}: train_acc={tr_acc:.4f}, val_acc={va_acc:.4f}')
            if va_acc > best_acc:
                best_acc = va_acc
                best_epoch = epoch
                bad_count = 0
                torch.save({'model_state_dict': self.model.state_dict(), 'best_acc': best_acc, 'best_epoch': best_epoch}, save_path)
            else:
                bad_count += 1
                if bad_count >= patience:
                    if self.logger:
                        self.logger.info(f'修复训练早停：best_epoch={best_epoch}, best_val_acc={best_acc:.4f}')
                    break
            if log_csv_path:
                save_csv(rows, log_csv_path)
        return {'best_acc': best_acc, 'best_epoch': best_epoch, 'save_path': str(save_path)}

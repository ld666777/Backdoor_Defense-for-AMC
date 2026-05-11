# -*- coding: utf-8 -*-
"""通用训练器。"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.utils.io import ensure_dir, save_csv


def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_one_epoch(model, loader, optimizer, device, train=True, extra_loss_fn=None):
    """训练或验证一个 epoch。"""
    model.train(train)
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total_num = 0.0, 0, 0

    for batch in loader:
        batch = _to_device(batch, device)
        x, y = batch['x'], batch['y']
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        if extra_loss_fn is not None and train:
            loss = loss + extra_loss_fn(model, batch, logits)
        if train:
            loss.backward()
            optimizer.step()
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((pred == y).sum().item())
        total_num += int(y.numel())
    return total_loss / max(total_num, 1), total_correct / max(total_num, 1)


class Trainer:
    """普通分类模型训练器。"""
    def __init__(self, model, device, logger=None):
        self.model = model
        self.device = device
        self.logger = logger

    def fit(self, train_dataset, val_dataset, cfg, save_path, log_csv_path=None, extra_loss_fn=None):
        batch_size = int(cfg.train.batch_size)
        num_workers = int(cfg.train.get('num_workers', 0))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )

        best_acc = -1.0
        best_epoch = -1
        patience = int(cfg.train.get('early_stop_patience', 5))
        bad_count = 0
        rows = []
        save_path = Path(save_path)
        ensure_dir(save_path.parent)

        for epoch in range(1, int(cfg.train.epochs) + 1):
            tr_loss, tr_acc = run_one_epoch(self.model, train_loader, optimizer, self.device, train=True, extra_loss_fn=extra_loss_fn)
            va_loss, va_acc = run_one_epoch(self.model, val_loader, optimizer, self.device, train=False)
            rows.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})
            if self.logger:
                self.logger.info(f'Epoch {epoch:03d}: train_acc={tr_acc:.4f}, val_acc={va_acc:.4f}, val_loss={va_loss:.4f}')

            if va_acc > best_acc:
                best_acc = va_acc
                best_epoch = epoch
                bad_count = 0
                torch.save({'model_state_dict': self.model.state_dict(), 'best_acc': best_acc, 'best_epoch': best_epoch}, save_path)
            else:
                bad_count += 1
                if bad_count >= patience:
                    if self.logger:
                        self.logger.info(f'早停触发：best_epoch={best_epoch}, best_val_acc={best_acc:.4f}')
                    break

            if log_csv_path:
                save_csv(rows, log_csv_path)

        return {'best_acc': best_acc, 'best_epoch': best_epoch, 'save_path': str(save_path)}

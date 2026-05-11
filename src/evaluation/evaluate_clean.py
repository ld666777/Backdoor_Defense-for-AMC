# -*- coding: utf-8 -*-
"""干净测试精度评估。"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils.io import ensure_dir


def evaluate_clean(model, dataset, device, batch_size=512, save_path=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    rows = []
    total_correct, total_num = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct = (pred == y).cpu().numpy().astype(int)
            total_correct += int(correct.sum())
            total_num += int(len(correct))
            for i in range(len(correct)):
                rows.append({'index': int(batch['index'][i]), 'label': int(batch['y'][i]), 'snr': int(batch['snr'][i]), 'pred': int(pred[i].cpu()), 'correct': int(correct[i])})
    df = pd.DataFrame(rows)
    acc = total_correct / max(total_num, 1)
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return acc, df


def summarize_acc_by_snr(df, save_path=None):
    out = df.groupby('snr')['correct'].mean().reset_index().rename(columns={'correct': 'acc'})
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        out.to_csv(save_path, index=False, encoding='utf-8-sig')
    return out

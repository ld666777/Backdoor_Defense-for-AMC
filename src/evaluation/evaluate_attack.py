# -*- coding: utf-8 -*-
"""攻击成功率 ASR 评估。"""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.attacks.attack_factory import apply_trigger
from src.utils.io import ensure_dir, load_json


def evaluate_asr(model, dataset, cfg, device, batch_size=512, save_path=None):
    """对测试集加入触发器，统计预测为目标类的比例。"""
    meta = load_json(Path(cfg.paths.processed_dir) / 'meta.json')
    target_label = int(meta['mod_to_label'][cfg.attack.target_class])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    rows = []
    hit, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            # ASR 通常排除原本就是目标类的样本
            mask = batch['y'] != target_label
            if int(mask.sum()) == 0:
                continue
            x = batch['x'][mask].to(device)
            y = batch['y'][mask]
            snr = batch['snr'][mask]
            idx = batch['index'][mask]
            x_trig = apply_trigger(x, cfg)
            logits = model(x_trig)
            pred = logits.argmax(dim=1).cpu()
            success = (pred == target_label).numpy().astype(int)
            hit += int(success.sum())
            total += int(len(success))
            for i in range(len(success)):
                rows.append({'index': int(idx[i]), 'label': int(y[i]), 'snr': int(snr[i]), 'pred': int(pred[i]), 'target_label': target_label, 'success': int(success[i])})
    df = pd.DataFrame(rows)
    asr = hit / max(total, 1)
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return asr, df


def summarize_asr_by_snr(df, save_path=None):
    out = df.groupby('snr')['success'].mean().reset_index().rename(columns={'success': 'asr'})
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        out.to_csv(save_path, index=False, encoding='utf-8-sig')
    return out

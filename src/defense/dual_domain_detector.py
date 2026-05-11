# -*- coding: utf-8 -*-
"""Dual-domain 一致性检测。"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.attacks.trigger_phase import add_phase_trigger
from src.attacks.trigger_time import add_time_trigger
from src.attacks.trigger_freq import add_freq_trigger
from src.utils.io import ensure_dir


def _entropy(p):
    return -(p * torch.log(p + 1e-12)).sum(dim=1)


def _make_augments(x, k):
    """生成调制保持增强样本。增强幅度较小，不应改变真实调制类别。"""
    outs = []
    for i in range(k):
        if i % 4 == 0:
            outs.append(add_phase_trigger(x, length=32, angle=0.05 + 0.01 * i, position='middle'))
        elif i % 4 == 1:
            outs.append(add_time_trigger(x, length=4, amplitude=0.005 + 0.002 * i, position='tail'))
        elif i % 4 == 2:
            outs.append(add_freq_trigger(x, amplitude=0.005, freq_bin=3 + i))
        else:
            outs.append(x + torch.randn_like(x) * 0.005)
    return outs


def dual_domain_scores(model, dataset, device, batch_size=256, transform_num=8, sample_limit=None, save_path=None):
    """计算 dual-domain 分数。

    分数越大，表示预测稳定性/特征稳定性越异常。
    """
    model.eval()
    if sample_limit is not None and int(sample_limit) < len(dataset):
        subset_idx = np.arange(int(sample_limit))
        eval_dataset = Subset(dataset, subset_idx)
    else:
        eval_dataset = dataset

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            logits0, feat0 = model(x, return_features=True)
            p0 = F.softmax(logits0, dim=1)
            conf0 = p0.max(dim=1).values
            ent0 = _entropy(p0)

            conf_list, ent_list, feat_dist_list = [], [], []
            for x_aug in _make_augments(x, int(transform_num)):
                logits_aug, feat_aug = model(x_aug, return_features=True)
                p_aug = F.softmax(logits_aug, dim=1)
                conf_list.append(p_aug.max(dim=1).values)
                ent_list.append(_entropy(p_aug))
                feat_dist_list.append(torch.norm(feat_aug - feat0, dim=1))

            conf_aug = torch.stack(conf_list, dim=0)
            ent_aug = torch.stack(ent_list, dim=0)
            feat_dist = torch.stack(feat_dist_list, dim=0)

            # 触发器样本常表现为对目标类别过度稳定或特征扰动异常
            confidence_stability = 1.0 - conf_aug.std(dim=0)
            entropy_drop = torch.clamp(ent0 - ent_aug.mean(dim=0), min=0.0)
            feature_shift = feat_dist.mean(dim=0)
            score = confidence_stability + entropy_drop + feature_shift

            for i in range(x.shape[0]):
                rows.append({
                    'index': int(batch['index'][i]),
                    'label': int(batch['y'][i]),
                    'snr': int(batch['snr'][i]),
                    'score_dual': float(score[i].cpu()),
                    'confidence_stability': float(confidence_stability[i].cpu()),
                    'entropy_drop': float(entropy_drop[i].cpu()),
                    'feature_shift': float(feature_shift[i].cpu()),
                    'is_poison': int(batch['is_poison'][i]),
                })
    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return df

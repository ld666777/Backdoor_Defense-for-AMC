# -*- coding: utf-8 -*-
"""统一评估入口。"""
from pathlib import Path
import torch
import pandas as pd
from src.utils.io import load_torch, load_json, save_json, ensure_dir
from src.data.dataset import RMLDataset
from src.models.model_factory import build_model
from .evaluate_clean import evaluate_clean, summarize_acc_by_snr
from .evaluate_attack import evaluate_asr, summarize_asr_by_snr
from .confusion import plot_confusion_matrix


def _load_model_from_ckpt(cfg, device, ckpt_path):
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def evaluate_checkpoint(cfg, device, ckpt_path, tag, logger=None):
    test_data = load_torch(Path(cfg.paths.processed_dir) / 'test.pt')
    test_ds = RMLDataset(test_data)
    model = _load_model_from_ckpt(cfg, device, ckpt_path)
    result_dir = Path(cfg.paths.result_dir)
    csv_dir = ensure_dir(result_dir / 'csv')
    fig_dir = ensure_dir(result_dir / 'figures')

    acc, clean_df = evaluate_clean(model, test_ds, device, batch_size=int(cfg.evaluation.batch_size), save_path=csv_dir / f'{tag}_clean_predictions.csv')
    acc_snr = summarize_acc_by_snr(clean_df, csv_dir / f'{tag}_acc_by_snr.csv')

    asr, asr_df = evaluate_asr(model, test_ds, cfg, device, batch_size=int(cfg.evaluation.batch_size), save_path=csv_dir / f'{tag}_trigger_predictions.csv')
    asr_snr = summarize_asr_by_snr(asr_df, csv_dir / f'{tag}_asr_by_snr.csv')

    meta = load_json(Path(cfg.paths.processed_dir) / 'meta.json')
    if bool(cfg.evaluation.get('calculate_confusion', True)):
        plot_confusion_matrix(clean_df['label'].values, clean_df['pred'].values, meta['mod_names'], fig_dir / f'{tag}_confusion.pdf')

    summary = {'tag': tag, 'clean_acc': float(acc), 'asr': float(asr), 'ckpt_path': str(ckpt_path)}
    save_json(summary, result_dir / 'reports' / f'{tag}_summary.json')
    if logger:
        logger.info(f'评估完成：{tag}, clean_acc={acc:.4f}, ASR={asr:.4f}')
    return summary

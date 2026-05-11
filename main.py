# -*- coding: utf-8 -*-
"""AMR 后门防御实验主入口。"""
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.utils.config import load_config, update_config
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.device import setup_device
from src.utils.io import ensure_dir, load_torch
from src.data.preprocess import prepare_rml_data
from src.data.dataset import RMLDataset
from src.attacks.poison_dataset import make_poisoned_train_set
from src.train.train_clean import train_clean_model
from src.train.train_backdoor import train_backdoor_model
from src.models.model_factory import build_model
from src.defense.feature_extractor import extract_features
from src.defense.snr_aware_detector import snr_aware_feature_scores
from src.defense.dual_domain_detector import dual_domain_scores
from src.defense.score_fusion import fuse_scores
from src.defense.mitigation import repair_backdoor_model
from src.evaluation.evaluator import evaluate_checkpoint
from src.visualization.plot_acc_snr import plot_acc_vs_snr
from src.visualization.plot_asr_snr import plot_asr_vs_snr
from src.visualization.export_latex_tables import export_main_table


def parse_args():
    parser = argparse.ArgumentParser(description='AMR 后门攻击防御 CPU 实验')
    parser.add_argument('--config', type=str, default='configs/quick_cpu.yaml')
    parser.add_argument('--stage', type=str, default='all', choices=['prepare_data', 'train_clean', 'make_poison', 'train_backdoor', 'detect', 'repair', 'evaluate', 'plot', 'all'])
    parser.add_argument('--attack_type', type=str, default=None)
    parser.add_argument('--poison_rate', type=float, default=None)
    parser.add_argument('--target_class', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def init_dirs(cfg):
    for p in [cfg.paths.processed_dir, cfg.paths.poisoned_dir, cfg.paths.checkpoint_dir, cfg.paths.result_dir]:
        ensure_dir(p)
    for sub in ['logs', 'csv', 'figures', 'latex_tables', 'reports']:
        ensure_dir(Path(cfg.paths.result_dir) / sub)


def poison_name(cfg):
    return f"{cfg.attack.type}_pr{int(float(cfg.attack.poison_rate)*1000):03d}_{cfg.attack.target_class}"


def backdoor_ckpt(cfg):
    return Path(cfg.paths.checkpoint_dir) / 'backdoor' / f'{cfg.model.name}_{poison_name(cfg)}.pt'


def repaired_ckpt(cfg):
    return Path(cfg.paths.checkpoint_dir) / 'repaired' / f'{cfg.model.name}_{poison_name(cfg)}_repaired.pt'


def clean_ckpt(cfg):
    return Path(cfg.paths.checkpoint_dir) / 'clean' / f'{cfg.model.name}_clean.pt'


def run_detect(cfg, device, logger):
    """执行后门样本检测，并返回融合分数 DataFrame。"""
    poison_path = Path(cfg.paths.poisoned_dir) / f'train_poison_{poison_name(cfg)}.pt'
    poison_data = load_torch(poison_path)
    train_ds = RMLDataset(poison_data, is_poison=poison_data.get('is_poison', None))

    model = build_model(cfg).to(device)
    ckpt = torch.load(backdoor_ckpt(cfg), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    logger.info('开始提取后门模型训练集特征')
    feat_pack = extract_features(
        model, train_ds, device,
        batch_size=int(cfg.evaluation.batch_size),
        save_prefix=Path(cfg.paths.result_dir) / 'csv' / 'features_train_backdoor.npz',
    )

    logger.info('开始执行 SNR-aware 特征异常检测')
    feature_df = snr_aware_feature_scores(
        feat_pack['features'], feat_pack['labels'], feat_pack['snr'], feat_pack['indices'],
        is_poison=feat_pack['is_poison'],
        snr_bins=cfg.defense.snr_bins,
        percentile=float(cfg.defense.anomaly_percentile),
        save_path=Path(cfg.paths.result_dir) / 'csv' / 'snr_aware_scores.csv',
    )

    dual_df = None
    if bool(cfg.defense.get('dual_domain_enable', True)):
        logger.info('开始执行 dual-domain 一致性检测')
        dual_df = dual_domain_scores(
            model, train_ds, device,
            batch_size=int(cfg.train.batch_size),
            transform_num=int(cfg.defense.transform_num),
            sample_limit=cfg.defense.get('sample_limit', None),
            save_path=Path(cfg.paths.result_dir) / 'csv' / 'dual_domain_scores.csv',
        )

    logger.info('开始融合检测分数')
    fused_df, metrics = fuse_scores(
        feature_df, dual_df,
        alpha=float(cfg.defense.fusion_alpha),
        beta=float(cfg.defense.fusion_beta),
        percentile=float(cfg.defense.suspicious_percentile),
        save_path=Path(cfg.paths.result_dir) / 'csv' / 'fused_detection_scores.csv',
        metric_path=Path(cfg.paths.result_dir) / 'reports' / 'detection_metrics.json',
    )
    if metrics:
        logger.info(f"检测指标：precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    return fused_df


def run_repair(cfg, device, logger):
    fused_path = Path(cfg.paths.result_dir) / 'csv' / 'fused_detection_scores.csv'
    if not fused_path.exists():
        fused_df = run_detect(cfg, device, logger)
    else:
        fused_df = pd.read_csv(fused_path)
    return repair_backdoor_model(cfg, device, fused_df, logger)


def run_evaluate(cfg, device, logger):
    summaries = []
    if clean_ckpt(cfg).exists():
        summaries.append(evaluate_checkpoint(cfg, device, clean_ckpt(cfg), 'clean_model', logger))
    if backdoor_ckpt(cfg).exists():
        summaries.append(evaluate_checkpoint(cfg, device, backdoor_ckpt(cfg), 'backdoor_model', logger))
    if repaired_ckpt(cfg).exists():
        summaries.append(evaluate_checkpoint(cfg, device, repaired_ckpt(cfg), 'repaired_model', logger))
    return summaries


def run_plot(cfg, logger):
    csv_dir = Path(cfg.paths.result_dir) / 'csv'
    fig_dir = Path(cfg.paths.result_dir) / 'figures'
    acc_paths, acc_labels = [], []
    asr_paths, asr_labels = [], []
    for tag in ['clean_model', 'backdoor_model', 'repaired_model']:
        p = csv_dir / f'{tag}_acc_by_snr.csv'
        if p.exists():
            acc_paths.append(p); acc_labels.append(tag)
        q = csv_dir / f'{tag}_asr_by_snr.csv'
        if q.exists():
            asr_paths.append(q); asr_labels.append(tag)
    if acc_paths:
        plot_acc_vs_snr(acc_paths, acc_labels, fig_dir / 'acc_vs_snr.pdf')
        logger.info('已生成 ACC vs SNR 图')
    if asr_paths:
        plot_asr_vs_snr(asr_paths, asr_labels, fig_dir / 'asr_vs_snr.pdf')
        logger.info('已生成 ASR vs SNR 图')
    table_path = export_main_table(cfg.paths.result_dir)
    if table_path:
        logger.info(f'已生成 LaTeX 主表格：{table_path}')


def main():
    args = parse_args()
    cfg = update_config(load_config(args.config), args)
    set_seed(int(cfg.project.seed))
    init_dirs(cfg)
    logger = get_logger('amr', Path(cfg.paths.result_dir) / 'logs' / f'{args.stage}_{poison_name(cfg)}.log')
    device = setup_device(cfg, logger)

    stage = args.stage
    if stage in ['prepare_data', 'all']:
        prepare_rml_data(cfg, logger)
    if stage in ['train_clean', 'all']:
        train_clean_model(cfg, device, logger)
    if stage in ['make_poison', 'all']:
        make_poisoned_train_set(cfg, logger)
    if stage in ['train_backdoor', 'all']:
        train_backdoor_model(cfg, device, logger)
    if stage in ['detect', 'all']:
        run_detect(cfg, device, logger)
    if stage in ['repair', 'all']:
        run_repair(cfg, device, logger)
    if stage in ['evaluate', 'all']:
        run_evaluate(cfg, device, logger)
    if stage in ['plot', 'all']:
        run_plot(cfg, logger)

    logger.info('当前阶段执行完成')


if __name__ == '__main__':
    main()

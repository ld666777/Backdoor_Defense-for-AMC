# -*- coding: utf-8 -*-
"""论文批量实验脚本（CPU 版）。

用途：
1. 自动生成攻击扫描、多模型对比、消融实验的单次实验配置；
2. 逐个调用 main.py 完成投毒、训练、检测、修复、评估与绘图；
3. 每个 run 使用独立结果目录，避免 CSV/图片互相覆盖；
4. 汇总所有 run 的指标，导出论文 LaTeX 表格。

推荐运行：
    python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all

先查看任务，不实际执行：
    python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all --dry_run
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# 允许脚本从项目根目录外启动时仍能 import src。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper.aggregate_results import collect_run_summary, collect_snr_curves, export_paper_tables, write_markdown_report


def load_yaml(path):
    """读取 YAML 文件。"""
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    """保存 YAML 文件，确保中文不乱码。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(obj, path):
    """保存 JSON 文件，确保中文不乱码。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def deep_update(base, patch):
    """递归更新字典，用于将 paper.overrides 写入 base config。"""
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def pr_code(poison_rate):
    """将投毒比例转为稳定的文件名片段，例如 0.03 -> pr003。"""
    return f'pr{int(float(poison_rate) * 1000):03d}'


def safe_name(text):
    """将实验名转为安全目录名。"""
    return str(text).replace('/', '_').replace(' ', '_').replace('%', 'pct').replace('.', 'p').replace('\\', '_')


def run_id_of(group, model, attack_type, poison_rate, target_class, variant='Full'):
    """统一生成 run_id，方便断点续跑和结果定位。"""
    return '_'.join([
        safe_name(group),
        safe_name(model),
        safe_name(attack_type),
        pr_code(poison_rate),
        safe_name(target_class),
        safe_name(variant),
    ])


def make_base_single_config(batch_cfg):
    """从 base_config 加载单次实验配置，并写入批量实验通用路径和覆盖项。"""
    base_path = PROJECT_ROOT / batch_cfg['base_config']
    cfg = load_yaml(base_path)
    paper = batch_cfg['paper']

    cfg['paths']['processed_dir'] = paper.get('processed_dir', cfg['paths'].get('processed_dir', 'data/processed'))
    cfg['paths']['checkpoint_dir'] = paper.get('checkpoint_dir', cfg['paths'].get('checkpoint_dir', 'checkpoints'))

    if paper.get('sample_limit', None) is not None:
        cfg['data']['sample_limit'] = paper['sample_limit']

    deep_update(cfg, paper.get('overrides', {}))
    return cfg


def apply_run_overrides(cfg, run):
    """把某个 run 的模型、攻击、防御、路径信息写入单次实验配置。"""
    cfg = copy.deepcopy(cfg)
    cfg['model']['name'] = run['model']
    cfg['attack']['type'] = run['attack_type']
    cfg['attack']['poison_rate'] = float(run['poison_rate'])
    cfg['attack']['target_class'] = run['target_class']

    # 每个 run 的结果目录独立，避免图表和 CSV 覆盖。
    cfg['paths']['result_dir'] = str(run['result_dir'])

    # 投毒数据按 run 隔离，便于排查不同攻击设置。
    cfg['paths']['poisoned_dir'] = str(run['poisoned_dir'])

    # checkpoint 默认按模型共享，避免重复训练 clean/backdoor；消融实验 repair checkpoint 可能覆盖，但结果已保存在各自 run 目录。
    cfg['paths']['checkpoint_dir'] = str(run['checkpoint_dir'])

    # 防御消融开关。
    if not run.get('snr_aware', True):
        # 不做 SNR-aware：仍按类别分组，但把所有 SNR 放进同一个 all bin。
        cfg['defense']['snr_bins'] = {'all': [-999, 999]}
    if not run.get('dual_domain', True):
        cfg['defense']['dual_domain_enable'] = False
    else:
        cfg['defense']['dual_domain_enable'] = True
    if 'fusion_alpha' in run:
        cfg['defense']['fusion_alpha'] = float(run['fusion_alpha'])
    if 'fusion_beta' in run:
        cfg['defense']['fusion_beta'] = float(run['fusion_beta'])
    if not run.get('consistency_repair', True):
        cfg['repair']['use_consistency_loss'] = False
    else:
        cfg['repair']['use_consistency_loss'] = True

    return cfg


def build_runs(batch_cfg, mode):
    """根据 mode 构造所有待运行的实验列表。"""
    paper = batch_cfg['paper']
    root_dir = PROJECT_ROOT / paper.get('root_dir', 'results/paper_experiments')
    poisoned_root = PROJECT_ROOT / paper.get('poisoned_dir', 'data/poisoned/paper_experiments')
    checkpoint_root = PROJECT_ROOT / paper.get('checkpoint_dir', 'checkpoints/paper_experiments')

    runs = []

    def add_run(group, model, attack_type, poison_rate, target_class, variant='Full', **kwargs):
        rid = run_id_of(group, model, attack_type, poison_rate, target_class, variant)
        run_dir = root_dir / 'runs' / rid
        run = {
            'run_id': rid,
            'experiment_group': group,
            'model': model,
            'attack_type': attack_type,
            'poison_rate': float(poison_rate),
            'target_class': target_class,
            'defense_variant': variant,
            'result_dir': run_dir,
            'poisoned_dir': poisoned_root / rid,
            'checkpoint_dir': checkpoint_root / model,
            **kwargs,
        }
        runs.append(run)

    if mode in ['sweep', 'all']:
        sw = batch_cfg.get('sweep', {})
        for model in sw.get('models', []):
            for attack_type in sw.get('attack_types', []):
                for poison_rate in sw.get('poison_rates', []):
                    for target_class in sw.get('target_classes', []):
                        add_run('sweep', model, attack_type, poison_rate, target_class, 'Full')

    if mode in ['model', 'model_comparison', 'all'] and batch_cfg.get('model_comparison', {}).get('enable', True):
        mc = batch_cfg.get('model_comparison', {})
        for model in mc.get('models', []):
            add_run('model_comparison', model, mc.get('attack_type', 'time_add'), mc.get('poison_rate', 0.03), mc.get('target_class', 'QPSK'), 'Full')

    if mode in ['ablation', 'all'] and batch_cfg.get('ablation', {}).get('enable', True):
        ab = batch_cfg.get('ablation', {})
        for variant in ab.get('variants', []):
            add_run(
                'ablation',
                ab.get('model', paper.get('primary_model', 'vtcnn2_lite')),
                ab.get('attack_type', paper.get('primary_attack', 'time_add')),
                ab.get('poison_rate', paper.get('primary_poison_rate', 0.03)),
                ab.get('target_class', paper.get('primary_target', 'QPSK')),
                variant.get('name', 'variant'),
                snr_aware=variant.get('snr_aware', True),
                dual_domain=variant.get('dual_domain', True),
                consistency_repair=variant.get('consistency_repair', True),
                fusion_alpha=variant.get('fusion_alpha', 0.6),
                fusion_beta=variant.get('fusion_beta', 0.4),
                variant_description=variant.get('description', ''),
            )

    # 去重：model_comparison 可能和 sweep 中的 primary run 重合，但结果目录不同。这里保留，因为论文表格分组不同。
    return runs


def checkpoint_paths(run):
    """给定 run，返回 clean/backdoor/repaired checkpoint 路径。"""
    ckpt_dir = Path(run['checkpoint_dir'])
    model = run['model']
    poison = f"{run['attack_type']}_{pr_code(run['poison_rate'])}_{run['target_class']}"
    return {
        'clean': ckpt_dir / 'clean' / f'{model}_clean.pt',
        'backdoor': ckpt_dir / 'backdoor' / f'{model}_{poison}.pt',
        'repaired': ckpt_dir / 'repaired' / f'{model}_{poison}_repaired.pt',
    }


def poison_path(run):
    """给定 run，返回投毒训练集路径。"""
    poison = f"{run['attack_type']}_{pr_code(run['poison_rate'])}_{run['target_class']}"
    return Path(run['poisoned_dir']) / f'train_poison_{poison}.pt'


def should_skip_stage(stage, run, skip_existing):
    """根据已有文件判断某个 stage 是否可以跳过。"""
    if not skip_existing:
        return False
    paths = checkpoint_paths(run)
    if stage == 'make_poison':
        return poison_path(run).exists()
    if stage == 'train_backdoor':
        return paths['backdoor'].exists()
    if stage == 'detect':
        return (Path(run['result_dir']) / 'csv' / 'fused_detection_scores.csv').exists()
    if stage == 'repair':
        # 消融实验的 repaired checkpoint 可能同名覆盖，因此用 run 目录下的 repair log 判断。
        return (Path(run['result_dir']) / 'csv' / 'repair_train_log.csv').exists()
    if stage == 'evaluate':
        return (Path(run['result_dir']) / 'reports' / 'repaired_model_summary.json').exists()
    if stage == 'plot':
        return (Path(run['result_dir']) / 'latex_tables' / 'table_main_acc_asr.tex').exists()
    return False


def call_main(config_path, stage, extra_env=None):
    """调用 main.py 的某个 stage。"""
    cmd = [sys.executable, 'main.py', '--config', str(config_path), '--stage', stage]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print('[运行]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, env=env)


def ensure_data_and_clean_models(batch_cfg, base_single_cfg, runs, dry_run=False, force_prepare=False):
    """准备数据，并为每个模型训练一次 clean model。"""
    paper = batch_cfg['paper']
    generated_dir = PROJECT_ROOT / paper.get('generated_config_dir', 'results/paper_experiments/generated_configs')
    skip_existing = bool(paper.get('skip_existing', True))

    # 准备数据：只做一次。所有论文实验共享同一个 processed_paper。
    data_cfg = copy.deepcopy(base_single_cfg)
    data_cfg['paths']['result_dir'] = str(PROJECT_ROOT / paper.get('root_dir', 'results/paper_experiments') / 'prepare_data')
    data_cfg['paths']['poisoned_dir'] = str(PROJECT_ROOT / paper.get('poisoned_dir', 'data/poisoned/paper_experiments'))
    data_cfg['paths']['checkpoint_dir'] = str(PROJECT_ROOT / paper.get('checkpoint_dir', 'checkpoints/paper_experiments') / 'prepare')
    data_config_path = generated_dir / 'prepare_data.yaml'
    save_yaml(data_cfg, data_config_path)

    train_pt = PROJECT_ROOT / data_cfg['paths']['processed_dir'] / 'train.pt'
    if force_prepare or not train_pt.exists() or not skip_existing:
        if dry_run:
            print(f'[预览] 将准备数据：{data_config_path}')
        else:
            call_main(data_config_path, 'prepare_data')
    else:
        print(f'[跳过] 已存在处理后数据：{train_pt}')

    if not paper.get('train_clean_if_missing', True):
        return

    # 每个模型训练一次 clean model，后续所有攻击共享。
    models = sorted({r['model'] for r in runs})
    for model in models:
        model_cfg = copy.deepcopy(base_single_cfg)
        model_cfg['model']['name'] = model
        model_cfg['paths']['result_dir'] = str(PROJECT_ROOT / paper.get('root_dir', 'results/paper_experiments') / 'clean_models' / model)
        model_cfg['paths']['poisoned_dir'] = str(PROJECT_ROOT / paper.get('poisoned_dir', 'data/poisoned/paper_experiments') / '_clean_dummy')
        model_cfg['paths']['checkpoint_dir'] = str(PROJECT_ROOT / paper.get('checkpoint_dir', 'checkpoints/paper_experiments') / model)
        config_path = generated_dir / f'clean_{model}.yaml'
        save_yaml(model_cfg, config_path)
        clean_ckpt = PROJECT_ROOT / model_cfg['paths']['checkpoint_dir'] / 'clean' / f'{model}_clean.pt'
        if clean_ckpt.exists() and skip_existing:
            print(f'[跳过] 已存在干净模型：{clean_ckpt}')
            continue
        if dry_run:
            print(f'[预览] 将训练干净模型：model={model}, config={config_path}')
        else:
            call_main(config_path, 'train_clean')
            # 训练完顺手评估 clean model，方便后续总表记录 clean baseline。
            call_main(config_path, 'evaluate')


def run_one_experiment(batch_cfg, base_single_cfg, run, dry_run=False, only_stages=None):
    """执行一个单次实验 run。"""
    paper = batch_cfg['paper']
    generated_dir = PROJECT_ROOT / paper.get('generated_config_dir', 'results/paper_experiments/generated_configs')
    stages = only_stages or paper.get('stages', ['make_poison', 'train_backdoor', 'detect', 'repair', 'evaluate', 'plot'])
    skip_existing = bool(paper.get('skip_existing', True))

    cfg = apply_run_overrides(base_single_cfg, run)
    config_path = generated_dir / f"{run['run_id']}.yaml"
    save_yaml(cfg, config_path)

    # 写 run_meta，后续汇总表会读取它。
    meta = {
        'run_id': run['run_id'],
        'experiment_group': run['experiment_group'],
        'defense_variant': run.get('defense_variant', 'Full'),
        'variant_description': run.get('variant_description', ''),
        'model': run['model'],
        'attack_type': run['attack_type'],
        'poison_rate': float(run['poison_rate']),
        'target_class': run['target_class'],
        'snr_aware': run.get('snr_aware', True),
        'dual_domain': run.get('dual_domain', True),
        'consistency_repair': run.get('consistency_repair', True),
        'config_path': str(config_path),
    }
    save_json(meta, Path(run['result_dir']) / 'run_meta.json')

    print('\n' + '=' * 90)
    print(f"[实验] {run['run_id']}")
    print(f"[设置] group={run['experiment_group']}, model={run['model']}, attack={run['attack_type']}, pr={run['poison_rate']}, target={run['target_class']}, variant={run.get('defense_variant')}")
    print('=' * 90)

    if dry_run:
        print(f'[预览] 配置文件：{config_path}')
        print(f'[预览] 结果目录：{run["result_dir"]}')
        print(f'[预览] 将执行 stages：{stages}')
        return

    for stage in stages:
        if should_skip_stage(stage, run, skip_existing):
            print(f'[跳过] {stage} 已有结果')
            continue
        call_main(config_path, stage)


def aggregate(batch_cfg):
    """汇总结果并导出论文表格。"""
    paper = batch_cfg['paper']
    root_dir = PROJECT_ROOT / paper.get('root_dir', 'results/paper_experiments')
    print('[汇总] 正在收集 run summary ...')
    df = collect_run_summary(root_dir)
    print(f'[汇总] 已收集 {len(df)} 个 run')
    collect_snr_curves(root_dir)
    out_dir = export_paper_tables(
        root_dir,
        primary_model=paper.get('primary_model', 'vtcnn2_lite'),
        primary_target=paper.get('primary_target', 'QPSK'),
        primary_attack=paper.get('primary_attack', 'time_add'),
    )
    report_path = write_markdown_report(root_dir)
    print(f'[汇总] LaTeX 表格目录：{out_dir}')
    print(f'[汇总] Markdown 报告：{report_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='论文批量实验 CPU 脚本')
    parser.add_argument('--config', type=str, default='configs/paper_experiments_cpu.yaml', help='批量实验配置文件')
    parser.add_argument('--mode', type=str, default='all', choices=['sweep', 'model', 'model_comparison', 'ablation', 'all', 'aggregate'], help='运行哪一组论文实验')
    parser.add_argument('--dry_run', action='store_true', help='只打印任务，不实际执行训练')
    parser.add_argument('--force_prepare', action='store_true', help='强制重新预处理数据')
    parser.add_argument('--stages', type=str, default=None, help='只运行指定 stages，用逗号分隔，例如 detect,repair,evaluate,plot')
    return parser.parse_args()


def main():
    args = parse_args()
    batch_cfg = load_yaml(PROJECT_ROOT / args.config)

    if args.mode == 'aggregate':
        aggregate(batch_cfg)
        return

    base_single_cfg = make_base_single_config(batch_cfg)
    runs = build_runs(batch_cfg, args.mode)
    print(f'[信息] 共生成 {len(runs)} 个实验任务')
    if len(runs) == 0:
        print('[警告] 没有待运行任务，请检查配置。')
        return

    ensure_data_and_clean_models(batch_cfg, base_single_cfg, runs, dry_run=args.dry_run, force_prepare=args.force_prepare)

    only_stages = None
    if args.stages:
        only_stages = [s.strip() for s in args.stages.split(',') if s.strip()]

    for run in runs:
        run_one_experiment(batch_cfg, base_single_cfg, run, dry_run=args.dry_run, only_stages=only_stages)

    if not args.dry_run:
        aggregate(batch_cfg)
    else:
        print('[预览] dry_run 模式不会执行汇总。')


if __name__ == '__main__':
    main()

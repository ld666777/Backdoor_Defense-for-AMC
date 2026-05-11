# -*- coding: utf-8 -*-
"""论文批量实验结果汇总与 LaTeX 表格导出。

本模块只读取已经生成的 run 目录，不会重新训练模型。
所有输出文件均使用 UTF-8 编码，便于直接复制到论文 LaTeX 工程中。
"""
from pathlib import Path
import json
import math
import pandas as pd


def _read_json(path):
    """安全读取 JSON，文件不存在时返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _safe_float(x):
    """将 None/NaN 安全转换为 float 或 NaN。"""
    try:
        if x is None:
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def _fmt4(x):
    """LaTeX 表格中统一保留 4 位小数。"""
    try:
        if pd.isna(x):
            return '--'
        return f'{float(x):.4f}'
    except Exception:
        return '--'


def _fmt_pct(x):
    """将比例格式化为百分比。"""
    try:
        if pd.isna(x):
            return '--'
        return f'{float(x) * 100:.1f}\\%'
    except Exception:
        return '--'


def collect_run_summary(root_dir):
    """扫描 paper_experiments/runs 下的所有实验结果，生成总 summary。

    Parameters
    ----------
    root_dir: str or Path
        论文批量实验根目录，例如 results/paper_experiments。

    Returns
    -------
    pandas.DataFrame
        每一行对应一个单次实验 run。
    """
    root_dir = Path(root_dir)
    runs_dir = root_dir / 'runs'
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _read_json(run_dir / 'run_meta.json') or {}
        reports_dir = run_dir / 'reports'
        clean = _read_json(reports_dir / 'clean_model_summary.json') or {}
        backdoor = _read_json(reports_dir / 'backdoor_model_summary.json') or {}
        repaired = _read_json(reports_dir / 'repaired_model_summary.json') or {}
        detect = _read_json(reports_dir / 'detection_metrics.json') or {}

        b_asr = _safe_float(backdoor.get('asr'))
        r_asr = _safe_float(repaired.get('asr'))
        b_acc = _safe_float(backdoor.get('clean_acc'))
        r_acc = _safe_float(repaired.get('clean_acc'))

        row = {
            'run_id': run_dir.name,
            'defense_variant': meta.get('defense_variant', 'Full'),
            'experiment_group': meta.get('experiment_group', 'unknown'),
            'model': meta.get('model'),
            'attack_type': meta.get('attack_type'),
            'poison_rate': _safe_float(meta.get('poison_rate')),
            'target_class': meta.get('target_class'),
            'clean_model_acc': _safe_float(clean.get('clean_acc')),
            'backdoor_clean_acc': b_acc,
            'backdoor_asr': b_asr,
            'repaired_clean_acc': r_acc,
            'repaired_asr': r_asr,
            'asr_reduction': b_asr - r_asr if not (pd.isna(b_asr) or pd.isna(r_asr)) else math.nan,
            'relative_asr_reduction': (b_asr - r_asr) / (b_asr + 1e-12) if not (pd.isna(b_asr) or pd.isna(r_asr)) else math.nan,
            'repair_acc_drop': b_acc - r_acc if not (pd.isna(b_acc) or pd.isna(r_acc)) else math.nan,
            'detect_precision': _safe_float(detect.get('precision')),
            'detect_recall': _safe_float(detect.get('recall')),
            'detect_f1': _safe_float(detect.get('f1')),
            'detect_fpr': _safe_float(detect.get('false_positive_rate')),
            'run_dir': str(run_dir),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = root_dir / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df) > 0:
        df.to_csv(out_dir / 'all_runs_summary.csv', index=False, encoding='utf-8-sig')
    return df


def collect_snr_curves(root_dir):
    """汇总所有 run 的 ACC/ASR by SNR 曲线，便于论文统一画图。"""
    root_dir = Path(root_dir)
    runs_dir = root_dir / 'runs'
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = _read_json(run_dir / 'run_meta.json') or {}
        csv_dir = run_dir / 'csv'
        for tag in ['clean_model', 'backdoor_model', 'repaired_model']:
            acc_path = csv_dir / f'{tag}_acc_by_snr.csv'
            if acc_path.exists():
                acc_df = pd.read_csv(acc_path)
                for _, r in acc_df.iterrows():
                    rows.append({
                        **meta,
                        'run_id': run_dir.name,
                        'model_tag': tag,
                        'metric': 'acc',
                        'snr': r.get('snr'),
                        'value': r.get('acc'),
                    })
            asr_path = csv_dir / f'{tag}_asr_by_snr.csv'
            if asr_path.exists():
                asr_df = pd.read_csv(asr_path)
                for _, r in asr_df.iterrows():
                    rows.append({
                        **meta,
                        'run_id': run_dir.name,
                        'model_tag': tag,
                        'metric': 'asr',
                        'snr': r.get('snr'),
                        'value': r.get('asr'),
                    })

    df = pd.DataFrame(rows)
    out_dir = root_dir / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df) > 0:
        df.to_csv(out_dir / 'all_snr_curves.csv', index=False, encoding='utf-8-sig')
    return df


def _write_table(df, out_path, columns, rename=None, column_format=None):
    """将 DataFrame 写成 IEEE 风格 LaTeX 表格片段。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        out_path.write_text('% 暂无可用实验结果。请先运行 paper_experiments_cpu.py。\n', encoding='utf-8')
        return out_path
    table = df[columns].copy()
    if rename:
        table = table.rename(columns=rename)
    latex = table.to_latex(index=False, escape=False, column_format=column_format)
    out_path.write_text(latex, encoding='utf-8')
    return out_path


def export_paper_tables(root_dir, primary_model='vtcnn2_lite', primary_target='QPSK', primary_attack='time_add'):
    """根据总 summary 导出论文所需表格。

    输出：
    - table_main_acc_asr.tex
    - table_poison_rate.tex
    - table_ablation.tex
    - table_detection.tex
    - table_model_comparison.tex
    """
    root_dir = Path(root_dir)
    summary_path = root_dir / 'summary' / 'all_runs_summary.csv'
    if not summary_path.exists():
        df = collect_run_summary(root_dir)
    else:
        df = pd.read_csv(summary_path)
    out_dir = root_dir / 'latex_tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        for name in ['table_main_acc_asr.tex', 'table_poison_rate.tex', 'table_ablation.tex', 'table_detection.tex', 'table_model_comparison.tex']:
            (out_dir / name).write_text('% 暂无可用实验结果。\n', encoding='utf-8')
        return out_dir

    # 主表：Full 方法在不同攻击、投毒比例、目标类别下的防御效果。
    main = df[(df['defense_variant'] == 'Full') & (df['experiment_group'].isin(['sweep', 'model_comparison']))].copy()
    main = main.sort_values(['model', 'attack_type', 'target_class', 'poison_rate'])
    for c in ['clean_model_acc', 'backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'relative_asr_reduction']:
        main[c] = main[c].map(_fmt4)
    _write_table(
        main,
        out_dir / 'table_main_acc_asr.tex',
        ['model', 'attack_type', 'poison_rate', 'target_class', 'backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'relative_asr_reduction'],
        rename={
            'model': 'Backbone', 'attack_type': 'Attack', 'poison_rate': 'PR', 'target_class': 'Target',
            'backdoor_clean_acc': 'ACC$_b$', 'backdoor_asr': 'ASR$_b$',
            'repaired_clean_acc': 'ACC$_r$', 'repaired_asr': 'ASR$_r$',
            'relative_asr_reduction': 'ASR Red.'
        },
        column_format='lllcccccc'
    )

    # 投毒比例表：主模型、主攻击、主目标。
    pr = df[(df['defense_variant'] == 'Full') & (df['model'] == primary_model) & (df['attack_type'] == primary_attack) & (df['target_class'] == primary_target)].copy()
    pr = pr.sort_values('poison_rate')
    for c in ['backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'detect_f1']:
        pr[c] = pr[c].map(_fmt4)
    _write_table(
        pr,
        out_dir / 'table_poison_rate.tex',
        ['poison_rate', 'backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'detect_f1'],
        rename={
            'poison_rate': 'Poisoning Rate', 'backdoor_clean_acc': 'ACC$_b$', 'backdoor_asr': 'ASR$_b$',
            'repaired_clean_acc': 'ACC$_r$', 'repaired_asr': 'ASR$_r$', 'detect_f1': 'Detection F1'
        },
        column_format='lccccc'
    )

    # 消融表。
    ab = df[df['experiment_group'] == 'ablation'].copy()
    ab = ab.sort_values('defense_variant')
    for c in ['repaired_clean_acc', 'repaired_asr', 'detect_precision', 'detect_recall', 'detect_f1', 'repair_acc_drop']:
        ab[c] = ab[c].map(_fmt4)
    _write_table(
        ab,
        out_dir / 'table_ablation.tex',
        ['defense_variant', 'repaired_clean_acc', 'repaired_asr', 'detect_precision', 'detect_recall', 'detect_f1', 'repair_acc_drop'],
        rename={
            'defense_variant': 'Variant', 'repaired_clean_acc': 'ACC$_r$', 'repaired_asr': 'ASR$_r$',
            'detect_precision': 'Prec.', 'detect_recall': 'Rec.', 'detect_f1': 'F1', 'repair_acc_drop': '$\\Delta$ACC'
        },
        column_format='lcccccc'
    )

    # 检测表：Full 方法在各攻击下的检测性能。
    det = df[(df['defense_variant'] == 'Full') & (df['experiment_group'].isin(['sweep', 'model_comparison']))].copy()
    det = det.sort_values(['model', 'attack_type', 'poison_rate', 'target_class'])
    for c in ['detect_precision', 'detect_recall', 'detect_f1', 'detect_fpr']:
        det[c] = det[c].map(_fmt4)
    _write_table(
        det,
        out_dir / 'table_detection.tex',
        ['model', 'attack_type', 'poison_rate', 'target_class', 'detect_precision', 'detect_recall', 'detect_f1', 'detect_fpr'],
        rename={
            'model': 'Backbone', 'attack_type': 'Attack', 'poison_rate': 'PR', 'target_class': 'Target',
            'detect_precision': 'Prec.', 'detect_recall': 'Rec.', 'detect_f1': 'F1', 'detect_fpr': 'FPR'
        },
        column_format='llllcccc'
    )

    # 多模型表。
    mc = df[df['experiment_group'] == 'model_comparison'].copy()
    mc = mc.sort_values('model')
    for c in ['backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'detect_f1']:
        mc[c] = mc[c].map(_fmt4)
    _write_table(
        mc,
        out_dir / 'table_model_comparison.tex',
        ['model', 'backdoor_clean_acc', 'backdoor_asr', 'repaired_clean_acc', 'repaired_asr', 'detect_f1'],
        rename={
            'model': 'Backbone', 'backdoor_clean_acc': 'ACC$_b$', 'backdoor_asr': 'ASR$_b$',
            'repaired_clean_acc': 'ACC$_r$', 'repaired_asr': 'ASR$_r$', 'detect_f1': 'Detection F1'
        },
        column_format='lccccc'
    )
    return out_dir


def write_markdown_report(root_dir):
    """生成一个便于人工检查的 Markdown 总报告。"""
    root_dir = Path(root_dir)
    df = collect_run_summary(root_dir)
    report_dir = root_dir / 'summary'
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / 'paper_experiment_report.md'
    if df is None or len(df) == 0:
        path.write_text('# Paper Experiment Report\n\n暂无结果。\n', encoding='utf-8')
        return path

    lines = []
    lines.append('# Paper Experiment Report')
    lines.append('')
    lines.append(f'- Total runs: {len(df)}')
    lines.append(f'- Mean repaired ACC: {_fmt4(df["repaired_clean_acc"].mean())}')
    lines.append(f'- Mean repaired ASR: {_fmt4(df["repaired_asr"].mean())}')
    lines.append(f'- Mean detection F1: {_fmt4(df["detect_f1"].mean())}')
    lines.append('')
    def _markdown_table(xdf, cols):
        # 不依赖 tabulate，避免额外安装包。
        if xdf is None or len(xdf) == 0:
            return '暂无数据。'
        header = '| ' + ' | '.join(cols) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
        body = []
        for _, row in xdf[cols].iterrows():
            body.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
        return '\n'.join([header, sep] + body)

    lines.append('## Best ASR Reduction Runs')
    top = df.sort_values('relative_asr_reduction', ascending=False).head(10)
    lines.append(_markdown_table(top, ['run_id', 'model', 'attack_type', 'poison_rate', 'target_class', 'defense_variant', 'relative_asr_reduction']))
    lines.append('')
    lines.append('## Worst Repaired ASR Runs')
    worst = df.sort_values('repaired_asr', ascending=False).head(10)
    lines.append(_markdown_table(worst, ['run_id', 'model', 'attack_type', 'poison_rate', 'target_class', 'defense_variant', 'repaired_asr']))
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

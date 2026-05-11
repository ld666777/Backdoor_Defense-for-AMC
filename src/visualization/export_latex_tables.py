# -*- coding: utf-8 -*-
"""导出 LaTeX 表格。"""
from pathlib import Path
import json
import pandas as pd
from src.utils.io import ensure_dir


def export_main_table(result_dir):
    """根据 summary JSON 生成主结果表格。"""
    result_dir = Path(result_dir)
    reports = result_dir / 'reports'
    rows = []
    for p in reports.glob('*_summary.json'):
        with p.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        rows.append(obj)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    table = df[['tag', 'clean_acc', 'asr']].copy()
    table['clean_acc'] = table['clean_acc'].map(lambda x: f'{x:.4f}')
    table['asr'] = table['asr'].map(lambda x: f'{x:.4f}')
    latex = table.to_latex(index=False, escape=False, column_format='lcc')
    out = ensure_dir(result_dir / 'latex_tables') / 'table_main_acc_asr.tex'
    out.write_text(latex, encoding='utf-8')
    return out

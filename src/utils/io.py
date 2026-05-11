# -*- coding: utf-8 -*-
"""文件读写工具。"""
from pathlib import Path
import json
import torch
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def save_torch(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, path)


def load_torch(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)


def save_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')

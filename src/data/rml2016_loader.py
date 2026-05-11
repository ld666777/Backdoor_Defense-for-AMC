# -*- coding: utf-8 -*-
"""RML2016.10a 数据读取。"""
from pathlib import Path
import pickle
import numpy as np


def _decode_if_bytes(x):
    """兼容 Python2 pickle 中的 bytes 字符串。"""
    if isinstance(x, bytes):
        return x.decode('utf-8')
    return x


def load_rml2016a(raw_path):
    """读取 RML2016.10a_dict.pkl。

    返回：
        dict: x [N,2,128], y [N], snr [N], mod_names 等。
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f'找不到数据集文件：{raw_path}')

    # 注意：RML2016.10a 的 pickle 可能由 Python 2 生成，因此需要 encoding='latin1'
    with raw_path.open('rb') as f:
        data = pickle.load(f, encoding='latin1')

    keys = list(data.keys())
    mods = sorted({_decode_if_bytes(k[0]) for k in keys})
    snrs = sorted({int(k[1]) for k in keys})
    mod_to_label = {m: i for i, m in enumerate(mods)}
    label_to_mod = {i: m for m, i in mod_to_label.items()}

    x_list, y_list, snr_list = [], [], []
    for mod in mods:
        for snr in snrs:
            key = (mod, snr)
            if key not in data:
                # 某些 pickle 的 key 中 mod 可能是 bytes
                key = (mod.encode('utf-8'), snr)
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[1] != 2:
                raise ValueError(f'数据 shape 异常：key={key}, shape={arr.shape}')
            x_list.append(arr)
            y_list.append(np.full(arr.shape[0], mod_to_label[mod], dtype=np.int64))
            snr_list.append(np.full(arr.shape[0], int(snr), dtype=np.int64))

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    snr = np.concatenate(snr_list, axis=0)

    return {
        'x': x,
        'y': y,
        'snr': snr,
        'mod_names': mods,
        'snr_values': snrs,
        'label_to_mod': label_to_mod,
        'mod_to_label': mod_to_label,
    }

# -*- coding: utf-8 -*-
"""相位域触发器。"""
import numpy as np
import torch


def add_phase_trigger(x, length=32, angle=0.35, position='tail'):
    """对局部 IQ 片段施加相位旋转。"""
    is_torch = torch.is_tensor(x)
    y = x.clone() if is_torch else np.array(x, copy=True)
    n = y.shape[-1]
    if position == 'tail':
        start = n - length
    elif position == 'head':
        start = 0
    else:
        start = max(0, (n - length) // 2)
    c, s = np.cos(angle), np.sin(angle)
    if is_torch:
        c = torch.tensor(c, dtype=y.dtype, device=y.device)
        s = torch.tensor(s, dtype=y.dtype, device=y.device)
    i = y[..., 0, start:start+length].clone() if is_torch else y[..., 0, start:start+length].copy()
    q = y[..., 1, start:start+length].clone() if is_torch else y[..., 1, start:start+length].copy()
    y[..., 0, start:start+length] = i * c - q * s
    y[..., 1, start:start+length] = i * s + q * c
    return y

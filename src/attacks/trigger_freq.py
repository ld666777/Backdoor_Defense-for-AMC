# -*- coding: utf-8 -*-
"""频域/窄带正弦触发器。"""
import numpy as np
import torch


def add_freq_trigger(x, amplitude=0.05, freq_bin=7):
    """加入一个低幅度复正弦扰动，模拟频域触发器。"""
    is_torch = torch.is_tensor(x)
    y = x.clone() if is_torch else np.array(x, copy=True)
    n = y.shape[-1]
    t = np.arange(n)
    pi = amplitude * np.cos(2 * np.pi * freq_bin * t / n)
    pq = amplitude * np.sin(2 * np.pi * freq_bin * t / n)
    if is_torch:
        pi = torch.tensor(pi, dtype=y.dtype, device=y.device)
        pq = torch.tensor(pq, dtype=y.dtype, device=y.device)
    y[..., 0, :] += pi
    y[..., 1, :] += pq
    return y

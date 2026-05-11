# -*- coding: utf-8 -*-
"""时间域加性触发器。"""
import numpy as np
import torch


def _get_pos(n, length, position):
    if position == 'tail':
        return n - length
    if position == 'head':
        return 0
    if position == 'middle':
        return max(0, (n - length) // 2)
    return int(position)


def add_time_trigger(x, length=8, amplitude=0.08, position='tail'):
    """对 IQ 样本加入局部正弦触发器。

    支持 numpy 和 torch，输入 shape 可以是 [2,N] 或 [B,2,N]。
    """
    is_torch = torch.is_tensor(x)
    y = x.clone() if is_torch else np.array(x, copy=True)
    n = y.shape[-1]
    start = _get_pos(n, length, position)
    t = np.arange(length)
    pattern_i = amplitude * np.sin(2 * np.pi * t / max(length, 1))
    pattern_q = amplitude * np.cos(2 * np.pi * t / max(length, 1))
    if is_torch:
        pi = torch.tensor(pattern_i, dtype=y.dtype, device=y.device)
        pq = torch.tensor(pattern_q, dtype=y.dtype, device=y.device)
        y[..., 0, start:start+length] += pi
        y[..., 1, start:start+length] += pq
    else:
        y[..., 0, start:start+length] += pattern_i
        y[..., 1, start:start+length] += pattern_q
    return y

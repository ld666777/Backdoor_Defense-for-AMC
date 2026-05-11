# -*- coding: utf-8 -*-
"""设备选择工具。"""
import torch


def setup_device(cfg, logger=None):
    """根据配置选择设备。本项目默认 CPU。"""
    num_threads = int(cfg.device.get('num_threads', 4))
    torch.set_num_threads(num_threads)

    use_gpu = bool(cfg.device.get('use_gpu', False))
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        msg = '当前设备：CUDA GPU'
    else:
        device = torch.device('cpu')
        msg = f'当前设备：CPU，torch 线程数：{num_threads}'

    if logger:
        logger.info(msg)
    return device

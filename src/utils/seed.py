# -*- coding: utf-8 -*-
"""随机种子设置。"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 2026):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # CPU 模式下 deterministic 影响较小，但保留以减少随机性
    torch.use_deterministic_algorithms(False)

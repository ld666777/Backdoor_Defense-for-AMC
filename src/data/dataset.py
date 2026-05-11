# -*- coding: utf-8 -*-
"""PyTorch Dataset 封装。"""
import numpy as np
import torch
from torch.utils.data import Dataset

class RMLDataset(Dataset):
    """RML2016.10a 数据集封装。"""
    def __init__(self, data_dict, is_poison=None, indices=None):
        self.x = np.asarray(data_dict['x'], dtype=np.float32)
        self.y = np.asarray(data_dict['y'], dtype=np.int64)
        self.snr = np.asarray(data_dict['snr'], dtype=np.int64)
        if is_poison is None:
            self.is_poison = np.zeros(len(self.y), dtype=np.int64)
        else:
            self.is_poison = np.asarray(is_poison, dtype=np.int64)
        if indices is None:
            self.indices = np.arange(len(self.y), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x': torch.from_numpy(self.x[idx]),
            'y': torch.tensor(self.y[idx], dtype=torch.long),
            'snr': torch.tensor(self.snr[idx], dtype=torch.long),
            'index': torch.tensor(self.indices[idx], dtype=torch.long),
            'is_poison': torch.tensor(self.is_poison[idx], dtype=torch.long),
        }


def make_subset_dataset(dataset, keep_mask):
    """根据 bool mask 构造子数据集。"""
    keep_mask = np.asarray(keep_mask).astype(bool)
    data = {
        'x': dataset.x[keep_mask],
        'y': dataset.y[keep_mask],
        'snr': dataset.snr[keep_mask],
    }
    return RMLDataset(data, is_poison=dataset.is_poison[keep_mask], indices=dataset.indices[keep_mask])

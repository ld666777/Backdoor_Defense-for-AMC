# -*- coding: utf-8 -*-
"""模型工厂。"""
from .cnn1d_lite import CNN1DLite
from .vtcnn2_lite import VTCNN2Lite
from .resnet1d_lite import ResNet1DLite


def build_model(cfg):
    name = cfg.model.name.lower()
    kwargs = dict(
        num_classes=int(cfg.model.num_classes),
        input_channels=int(cfg.model.input_channels),
        feature_dim=int(cfg.model.feature_dim),
        dropout=float(cfg.model.dropout),
    )
    if name == 'cnn1d_lite':
        return CNN1DLite(**kwargs)
    if name == 'vtcnn2_lite':
        return VTCNN2Lite(**kwargs)
    if name == 'resnet1d_lite':
        return ResNet1DLite(**kwargs)
    raise ValueError(f'未知模型名称：{cfg.model.name}')

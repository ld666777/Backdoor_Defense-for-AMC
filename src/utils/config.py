# -*- coding: utf-8 -*-
"""配置读取工具。

本文件将 YAML 配置读取为支持点号访问的对象，例如 cfg.train.epochs。
"""
from pathlib import Path
import yaml

class AttrDict(dict):
    """支持 cfg.xxx 访问方式的字典。"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj


def load_config(path):
    """读取 YAML 配置文件。"""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return _to_attr_dict(cfg)


def update_config(cfg, args):
    """用命令行参数覆盖配置。只覆盖非空字段。"""
    if getattr(args, 'model', None):
        cfg.model.name = args.model
    if getattr(args, 'attack_type', None):
        cfg.attack.type = args.attack_type
    if getattr(args, 'poison_rate', None) is not None:
        cfg.attack.poison_rate = float(args.poison_rate)
    if getattr(args, 'target_class', None):
        cfg.attack.target_class = args.target_class
    if getattr(args, 'seed', None) is not None:
        cfg.project.seed = int(args.seed)
    return cfg

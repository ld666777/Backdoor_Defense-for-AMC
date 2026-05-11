# -*- coding: utf-8 -*-
"""攻击触发器工厂。"""
from .trigger_time import add_time_trigger
from .trigger_phase import add_phase_trigger
from .trigger_freq import add_freq_trigger


def apply_trigger(x, cfg):
    """根据配置向样本加入触发器。"""
    attack_type = cfg.attack.type
    if attack_type == 'time_add':
        return add_time_trigger(
            x,
            length=int(cfg.attack.trigger_length),
            amplitude=float(cfg.attack.trigger_amplitude),
            position=cfg.attack.trigger_position,
        )
    if attack_type == 'phase':
        return add_phase_trigger(
            x,
            length=int(cfg.attack.get('trigger_length', 32)),
            angle=float(cfg.attack.get('phase_angle', 0.35)),
            position=cfg.attack.trigger_position,
        )
    if attack_type == 'freq':
        return add_freq_trigger(
            x,
            amplitude=float(cfg.attack.trigger_amplitude),
            freq_bin=int(cfg.attack.get('freq_bin', 7)),
        )
    raise ValueError(f'未知攻击类型：{attack_type}')

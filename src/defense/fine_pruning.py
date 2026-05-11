# -*- coding: utf-8 -*-
"""Fine-pruning 占位模块。

说明：为了保证 CPU 版代码稳定，默认主流程不启用通道剪枝。
后续如果论文需要对比 Fine-Pruning，可以在这里扩展通道激活统计和剪枝逻辑。
"""


def fine_pruning_placeholder(model, prune_ratio=0.1):
    """当前不实际剪枝，仅返回原模型。"""
    return model

# -*- coding: utf-8 -*-
"""混淆矩阵绘图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils.io import ensure_dir


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Normalized Confusion Matrix')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path)
    plt.close(fig)

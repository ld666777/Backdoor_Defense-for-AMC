# -*- coding: utf-8 -*-
"""t-SNE 可视化。

注意：t-SNE 在 CPU 上较慢，默认不在 main.py 中执行。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils.io import ensure_dir


def plot_tsne(features, labels, save_path, max_points=3000):
    features = np.asarray(features)
    labels = np.asarray(labels)
    if len(features) > max_points:
        idx = np.random.default_rng(2026).choice(len(features), size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
    emb = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=6)
    fig.colorbar(sc, ax=ax)
    ax.set_title('t-SNE Feature Visualization')
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path)
    plt.close(fig)

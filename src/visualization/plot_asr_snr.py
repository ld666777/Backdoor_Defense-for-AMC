# -*- coding: utf-8 -*-
"""绘制 ASR vs SNR。"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.io import ensure_dir


def plot_asr_vs_snr(csv_paths, labels, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, lab in zip(csv_paths, labels):
        df = pd.read_csv(path)
        ax.plot(df['snr'], df['asr'], marker='o', label=lab)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Attack Success Rate')
    ax.set_title('ASR vs. SNR')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix('.png'), dpi=300)
    plt.close(fig)

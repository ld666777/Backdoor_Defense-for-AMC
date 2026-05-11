# -*- coding: utf-8 -*-
"""常用指标计算。"""
import numpy as np


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(axis=1)
    labels = np.asarray(labels)
    return float((preds == labels).mean()) if len(labels) else 0.0


def classification_accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean()) if len(labels) else 0.0


def binary_detection_metrics(pred, truth):
    """计算二分类检测指标。pred/truth 为 0/1。"""
    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth).astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fpr),
    }


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

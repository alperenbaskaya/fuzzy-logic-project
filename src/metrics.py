"""Performans ölçütleri.

Pozitif sınıf = non-self (anomali, etiket 1).
  DR  (Detection Rate) = TP / (TP + FN)   -- recall ile aynı
  FAR (False Alarm Rate) = FP / (FP + TN)
"""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, tn, fp, fn


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = confusion(y_true, y_pred)
    dr = tp / (tp + fn) if tp + fn else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * prec * dr / (prec + dr) if prec + dr else 0.0
    return {"DR": 100 * dr, "FAR": 100 * far, "Accuracy": 100 * acc,
            "Precision": 100 * prec, "F1": 100 * f1}

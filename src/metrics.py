from __future__ import annotations

import numpy as np

NAMES = ("Negative", "Neutral", "Positive")


def classification_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(truth, prediction):
        cm[a, b] += 1
    support = cm.sum(axis=1)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "by_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(support[i])}
                     for i, name in enumerate(NAMES)},
        "confusion_matrix": cm.tolist(),
    }


def regression_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = truth - prediction
    if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
    else:
        pearson = 0.0
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson": pearson}


def all_metrics(truth_cls, pred_cls, truth_reg, pred_reg):
    return {**classification_metrics(truth_cls, pred_cls),
            **regression_metrics(truth_reg, pred_reg)}

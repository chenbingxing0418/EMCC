"""Evaluation metrics used by training and streaming inference."""

from __future__ import annotations

import numpy as np


def metric_bundle(labels: np.ndarray, preds: np.ndarray, label_names: list[str]):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    class_count = len(label_names)
    per_class = []
    precision_values = []
    recall_values = []
    f1_values = []
    support_values = []
    correct = int((labels == preds).sum())
    total = int(labels.size)

    for class_id, class_name in enumerate(label_names):
        true_mask = labels == class_id
        pred_mask = preds == class_id
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        support = int(true_mask.sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        support_values.append(support)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    supports = np.asarray(support_values, dtype=np.float64)
    support_total = float(supports.sum())
    if support_total > 0:
        weights = supports / support_total
        weighted_precision = float(np.sum(np.asarray(precision_values) * weights))
        weighted_recall = float(np.sum(np.asarray(recall_values) * weights))
        weighted_f1 = float(np.sum(np.asarray(f1_values) * weights))
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    return {
        "accuracy": correct / max(total, 1),
        "macro_precision": float(np.mean(precision_values)) if class_count else 0.0,
        "macro_recall": float(np.mean(recall_values)) if class_count else 0.0,
        "macro_f1": float(np.mean(f1_values)) if class_count else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }

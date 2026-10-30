"""Small tabular report builders."""

from __future__ import annotations

import numpy as np

from progressive.early_exit import probability_margin


def confusion_matrix_rows(labels: np.ndarray, preds: np.ndarray, label_names: list[str]) -> list[dict]:
    rows = []
    for true_idx, true_name in enumerate(label_names):
        row = {"true_class_id": true_idx, "true_class": true_name}
        mask = labels == true_idx
        for pred_idx, pred_name in enumerate(label_names):
            row[f"pred_{pred_name}"] = int((preds[mask] == pred_idx).sum())
        rows.append(row)
    return rows


def per_class_rows(metrics: dict) -> list[dict]:
    return list(metrics.get("per_class", []))


def sample_prediction_rows(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    reliability: np.ndarray,
    exit_score: np.ndarray,
    observed: np.ndarray,
    early_preds: np.ndarray,
    exit_steps: np.ndarray,
    label_names: list[str],
    prefix_names: list[str],
    prefix_budgets: list[int],
) -> list[dict]:
    margins = probability_margin(probabilities)
    rows = []
    for idx in range(len(labels)):
        step = int(exit_steps[idx])
        pred_id = int(early_preds[idx])
        label_id = int(labels[idx])
        rows.append(
            {
                "sample_id": int(sample_ids[idx]),
                "true_id": label_id,
                "true_label": label_names[label_id],
                "early_pred_id": pred_id,
                "early_pred": label_names[pred_id],
                "correct": int(pred_id == label_id),
                "exit_step": step + 1,
                "exit_prefix": prefix_names[step],
                "exit_prefix_budget": int(prefix_budgets[step]),
                "exit_observed_packets": float(observed[idx, step]),
                "exit_reliability": float(reliability[idx, step]),
                "exit_score": float(exit_score[idx, step]),
                "exit_confidence": float(probabilities[idx, step, pred_id]),
                "exit_margin": float(margins[idx, step]),
            }
        )
    return rows


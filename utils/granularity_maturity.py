"""JSD-based evidence maturity priors.

The prior is computed once from the training split. It is intentionally used as
an evidence-maturity prior instead of a hard classifier: structure still decides
whether packet/burst/flow evidence exists, while JSD measures whether the
distribution at a prefix is already stable and class-separable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STAGE_NAMES = ("packet", "burst", "flow")


@dataclass
class GranularityMaturityPrior:
    budgets: list[int]
    prior: np.ndarray
    rows: list[dict[str, float | int | str]]


def build_granularity_maturity_prior(train_dataset, args) -> GranularityMaturityPrior:
    budgets = [int(value) for value in train_dataset.prefix_packet_budgets]
    labels = train_dataset.labels_numpy()
    indices = _stratified_indices(
        labels,
        max_samples=int(args.granularity_maturity_prior_max_samples),
        seed=int(args.seed) + 2028,
    )
    labels = labels[indices]
    classes = np.unique(labels)
    bins = max(4, int(args.granularity_maturity_prior_bins))

    features_by_stage = {
        stage_name: [
            _stage_features(stage_name, dataset, indices)
            for dataset in train_dataset.datasets
        ]
        for stage_name in STAGE_NAMES
    }

    prior = np.ones((len(budgets), len(STAGE_NAMES)), dtype=np.float32)
    rows: list[dict[str, float | int | str]] = []
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        stage_features = features_by_stage[stage_name]
        final_features = stage_features[-1]
        raw_rows = []
        for step, (budget, features) in enumerate(zip(budgets, stage_features)):
            metrics = _stage_jsd_metrics(
                features=features,
                final_features=final_features,
                labels=labels,
                classes=classes,
                bins=bins,
            )
            raw_rows.append(
                {
                    "prefix_index": step + 1,
                    "prefix_budget": int(budget),
                    "stage": stage_name,
                    **metrics,
                }
            )

        max_separability = max(float(row["separability_jsd"]) for row in raw_rows)
        max_separability = max(max_separability, 1e-6)
        stage_scores = []
        for row in raw_rows:
            stability = float(row["stability_score"])
            separability_norm = float(row["separability_jsd"]) / max_separability
            score = float(np.sqrt(max(stability, 0.0) * max(separability_norm, 0.0)))
            stage_scores.append(score)
            row["separability_norm"] = separability_norm
            row["granularity_maturity_raw"] = score

        monotonic_scores = np.maximum.accumulate(np.asarray(stage_scores, dtype=np.float32))
        for row, score in zip(raw_rows, monotonic_scores):
            row["granularity_maturity"] = float(np.clip(score, 0.0, 1.0))
            prior[int(row["prefix_index"]) - 1, stage_idx] = row["granularity_maturity"]
            rows.append(row)

    return GranularityMaturityPrior(budgets=budgets, prior=prior, rows=rows)


def write_granularity_maturity_prior_report(report: GranularityMaturityPrior, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "granularity_maturity_prior.npy", report.prior.astype(np.float32))
    fieldnames = [
        "prefix_index",
        "prefix_budget",
        "stage",
        "feature_count",
        "stability_jsd",
        "stability_score",
        "separability_jsd",
        "separability_norm",
        "granularity_maturity_raw",
        "granularity_maturity",
    ]
    with (output_dir / "granularity_maturity_prior.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report.rows)


def _stratified_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if max_samples <= 0 or labels.size <= max_samples:
        return np.arange(labels.size, dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, max_samples // max(len(classes), 1))
    selected = []
    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        take = min(per_class, class_indices.size)
        if take:
            selected.append(rng.choice(class_indices, size=take, replace=False))

    selected_indices = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if selected_indices.size < max_samples:
        missing = max_samples - selected_indices.size
        mask = np.ones(labels.size, dtype=bool)
        mask[selected_indices] = False
        remaining = np.flatnonzero(mask)
        if remaining.size:
            extra = rng.choice(remaining, size=min(missing, remaining.size), replace=False)
            selected_indices = np.concatenate([selected_indices, extra])
    selected_indices = np.unique(selected_indices)
    rng.shuffle(selected_indices)
    return selected_indices.astype(np.int64)


def _stage_features(stage_name: str, dataset, indices: np.ndarray) -> np.ndarray:
    if stage_name == "packet":
        return _packet_features(dataset, indices)
    if stage_name == "burst":
        return _burst_features(dataset, indices)
    if stage_name == "flow":
        return _flow_features(dataset, indices)
    raise ValueError(f"unknown stage: {stage_name}")


def _packet_features(dataset, indices: np.ndarray) -> np.ndarray:
    byte_x = np.asarray(dataset.byte_x[indices, :, : dataset.byte_head], dtype=np.float32)
    packet_x = np.asarray(dataset.packet_x[indices], dtype=np.float32)
    observed = np.asarray(dataset.observed_packets[indices], dtype=np.float32).reshape(-1, 1)
    packet_mask = np.abs(packet_x).sum(axis=-1) > 0
    packet_stats = _masked_mean_std(packet_x, packet_mask)

    byte_mask = packet_mask[..., None]
    valid_packet_count = byte_mask.sum(axis=(1, 2), keepdims=False).reshape(-1, 1)
    valid_byte_count = np.maximum(valid_packet_count * float(byte_x.shape[-1]), 1.0)
    byte_sum = (byte_x * byte_mask).sum(axis=(1, 2), keepdims=False).reshape(-1, 1)
    byte_mean = byte_sum / valid_byte_count
    byte_centered = (byte_x - byte_mean.reshape(-1, 1, 1)) * byte_mask
    byte_std = np.sqrt((byte_centered ** 2).sum(axis=(1, 2), keepdims=False).reshape(-1, 1) / valid_byte_count)
    byte_nonzero = ((byte_x > 0) & byte_mask).sum(axis=(1, 2), keepdims=False).reshape(-1, 1) / valid_byte_count

    if packet_x.shape[-1] > 2:
        valid_pairs = packet_mask[:, :-1] & packet_mask[:, 1:]
        direction_switch = (
            np.abs(packet_x[:, 1:, 2] - packet_x[:, :-1, 2]) > 0.5
        ) & valid_pairs
        switch_count = direction_switch.sum(axis=1, keepdims=True).astype(np.float32)
    else:
        switch_count = np.zeros((packet_x.shape[0], 1), dtype=np.float32)

    observed_norm = observed / max(float(packet_x.shape[1]), 1.0)
    return _clean_features(
        np.concatenate([packet_stats, byte_mean, byte_std, byte_nonzero, switch_count, observed_norm], axis=1)
    )


def _burst_features(dataset, indices: np.ndarray) -> np.ndarray:
    burst_x = np.asarray(dataset.burst_x[indices], dtype=np.float32)
    packet_x = np.asarray(dataset.packet_x[indices], dtype=np.float32)
    observed = np.asarray(dataset.observed_packets[indices], dtype=np.float32).reshape(-1, 1)
    burst_mask = np.abs(burst_x).sum(axis=-1) > 0
    burst_stats = _masked_mean_std(burst_x, burst_mask)
    burst_count = burst_mask.sum(axis=1, keepdims=True).astype(np.float32)

    packet_mask = np.abs(packet_x).sum(axis=-1) > 0
    if packet_x.shape[-1] > 2:
        valid_pairs = packet_mask[:, :-1] & packet_mask[:, 1:]
        direction_switch = (
            np.abs(packet_x[:, 1:, 2] - packet_x[:, :-1, 2]) > 0.5
        ) & valid_pairs
        switch_count = direction_switch.sum(axis=1, keepdims=True).astype(np.float32)
    else:
        switch_count = np.zeros((packet_x.shape[0], 1), dtype=np.float32)

    observed_norm = observed / max(float(packet_x.shape[1]), 1.0)
    burst_count_norm = burst_count / max(float(burst_x.shape[1]), 1.0)
    switch_norm = switch_count / max(float(packet_x.shape[1] - 1), 1.0)
    return _clean_features(
        np.concatenate([burst_stats, burst_count_norm, switch_norm, observed_norm], axis=1)
    )


def _flow_features(dataset, indices: np.ndarray) -> np.ndarray:
    flow_x = np.asarray(dataset.flow_x[indices], dtype=np.float32)
    observed = np.asarray(dataset.observed_packets[indices], dtype=np.float32).reshape(-1, 1)
    observed_norm = observed / 20.0
    return _clean_features(np.concatenate([flow_x, observed_norm], axis=1))


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    denom = np.maximum(mask_f.sum(axis=1, keepdims=True), 1.0)
    mean = (values * mask_f[..., None]).sum(axis=1) / denom
    centered = (values - mean[:, None, :]) * mask_f[..., None]
    std = np.sqrt((centered ** 2).sum(axis=1) / denom)
    return np.concatenate([mean, std], axis=1)


def _clean_features(features: np.ndarray) -> np.ndarray:
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _stage_jsd_metrics(
    features: np.ndarray,
    final_features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    bins: int,
) -> dict[str, float | int]:
    feature_count = int(features.shape[1])
    stability_values = []
    separability_values = []
    for feature_idx in range(feature_count):
        edges = _hist_edges([features[:, feature_idx], final_features[:, feature_idx]], bins)
        if edges is None:
            continue
        current_probs = _class_histograms(features[:, feature_idx], labels, classes, edges)
        final_probs = _class_histograms(final_features[:, feature_idx], labels, classes, edges)
        stability_values.extend(
            _jsd(current_probs[class_idx], final_probs[class_idx])
            for class_idx in range(len(classes))
        )
        separability_values.extend(_pairwise_jsd(current_probs))

    stability_jsd = float(np.mean(stability_values)) if stability_values else 0.0
    separability_jsd = float(np.mean(separability_values)) if separability_values else 0.0
    return {
        "feature_count": feature_count,
        "stability_jsd": stability_jsd,
        "stability_score": float(np.clip(1.0 - stability_jsd, 0.0, 1.0)),
        "separability_jsd": float(np.clip(separability_jsd, 0.0, 1.0)),
    }


def _hist_edges(values: list[np.ndarray], bins: int) -> np.ndarray | None:
    merged = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return None
    low = float(np.min(merged))
    high = float(np.max(merged))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-8:
        return None
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(merged, quantiles))
    if edges.size < 3:
        edges = np.linspace(low, high, bins + 1)
    edges = np.asarray(edges, dtype=np.float32)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def _class_histograms(values: np.ndarray, labels: np.ndarray, classes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    histograms = []
    for class_id in classes:
        class_values = values[labels == class_id]
        hist, _ = np.histogram(class_values, bins=edges)
        hist = hist.astype(np.float64) + 1e-8
        histograms.append(hist / hist.sum())
    return np.stack(histograms, axis=0)


def _pairwise_jsd(probabilities: np.ndarray) -> list[float]:
    values = []
    for left in range(probabilities.shape[0]):
        for right in range(left + 1, probabilities.shape[0]):
            values.append(_jsd(probabilities[left], probabilities[right]))
    return values


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    middle = 0.5 * (left + right)
    return float(0.5 * _kl(left, middle) + 0.5 * _kl(right, middle))


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    mask = left > 0
    return float(np.sum(left[mask] * np.log2(left[mask] / np.maximum(right[mask], 1e-12))))


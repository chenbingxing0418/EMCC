"""Write progressive evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from progressive.early_exit import rule_from_args


def write_progressive_outputs(output_dir: Path, split: str, result: dict, args, prefix_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(output_dir, split, result, args, prefix_names)
    _write_prefix_metrics(output_dir, result)
    _write_rows(output_dir / "progressive_early_exit_per_class.csv", result.get("early_exit_per_class", []))
    _write_rows(output_dir / "progressive_early_exit_confusion_matrix.csv", result.get("early_exit_confusion_matrix", []))
    _write_rows(output_dir / "progressive_early_exit_predictions.csv", result.get("sample_predictions", []))


def _write_summary(output_dir: Path, split: str, result: dict, args, prefix_names: list[str]) -> None:
    summary_result = {key: value for key, value in result.items() if key != "sample_predictions"}
    summary = {
        "split": split,
        "prefixes": list(prefix_names),
        "early_exit_rule": _rule_summary(args),
        **summary_result,
    }
    (output_dir / "progressive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _rule_summary(args) -> dict:
    rule = rule_from_args(args)
    return {
        "threshold": rule.threshold,
        "min_step": rule.min_step,
        "use_symmetric_kl_stability": rule.use_symmetric_kl,
        "use_history_prediction_stability": rule.use_history_jsd,
        "kl_epsilon": rule.kl_epsilon,
        "jsd_prev_epsilon": rule.jsd_prev_epsilon,
        "jsd_avg_epsilon": rule.jsd_avg_epsilon,
        "history_window": rule.history_window,
        "online_test_eval": bool(args.use_online_test_eval),
        "force_exit_at_flow_end": bool(getattr(args, "force_exit_at_flow_end", True)),
    }


def _write_prefix_metrics(output_dir: Path, result: dict) -> None:
    rows = []
    for prefix, metrics in result["prefix_metrics"].items():
        row = {"prefix": prefix}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)) or value is None:
                row[key] = value
        rows.append(row)
    _write_rows(output_dir / "progressive_prefix_metrics.csv", rows)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


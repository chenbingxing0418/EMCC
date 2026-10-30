"""Path helpers for progressive prefix datasets."""

from __future__ import annotations

from pathlib import Path

from data.progressive_dataset import prefix_packet_budget


def resolve_progressive_prefix_dirs(args) -> list[Path]:
    if args.joint_prefix_dirs:
        return normalize_prefix_dirs(args.joint_prefix_dirs)

    min_prefix = int(args.progressive_prefix_min)
    max_prefix = int(args.progressive_prefix_max)
    if min_prefix < 1 or max_prefix < min_prefix:
        raise ValueError("--progressive-prefix-min/max must define a positive increasing range")

    data_dir = Path(args.data_dir)
    parent = data_dir.parent if data_dir.name.startswith("prefix_") else data_dir
    candidates = [parent / f"prefix_{value}" for value in range(min_prefix, max_prefix + 1)]
    missing = [path.name for path in candidates if not path.exists()]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise ValueError(f"missing progressive prefix directories: {preview}{suffix}")
    return normalize_prefix_dirs(candidates)


def normalize_prefix_dirs(prefix_dirs) -> list[Path]:
    dirs = [Path(path) for path in prefix_dirs]
    if not dirs:
        raise ValueError("progressive training requires at least one prefix directory")
    budgets = [prefix_packet_budget(path.name) for path in dirs]
    if any(value <= 0 for value in budgets):
        joined = ", ".join(path.name for path in dirs)
        raise ValueError(f"prefix directories must be named like prefix_K, got: {joined}")
    ordered = sorted(zip(budgets, dirs), key=lambda item: item[0])
    ordered_budgets = [budget for budget, _ in ordered]
    if len(set(ordered_budgets)) != len(ordered_budgets):
        raise ValueError(f"duplicate prefix budgets are not allowed: {ordered_budgets}")
    if any(left >= right for left, right in zip(ordered_budgets, ordered_budgets[1:])):
        raise ValueError(f"prefix budgets must be strictly increasing: {ordered_budgets}")
    return [path for _, path in ordered]


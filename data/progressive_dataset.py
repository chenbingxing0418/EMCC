"""Dataset helpers for progressive multi-prefix early classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import GraphTrafficDataset


def prefix_packet_budget(prefix_name: str) -> int:
    """Extract the nominal packet budget from names such as prefix_3."""
    tail = prefix_name.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


class ProgressivePrefixDataset(Dataset):
    """Return aligned prefix observations for the same flow/window sample.

    The underlying prefix directories must share the same split labels and sample
    order. Each item stacks the five prefix views along dimension 0:
    byte_head: [T, packets, byte_head], packet: [T, packets, packet_dim], etc.
    """

    def __init__(self, prefix_dirs: list[Path] | tuple[Path, ...], split: str, byte_head: int):
        self.prefix_dirs = [Path(path) for path in prefix_dirs]
        if not self.prefix_dirs:
            raise ValueError("ProgressivePrefixDataset requires at least one prefix directory")
        self.split = split
        self.prefix_names = [path.name for path in self.prefix_dirs]
        self.prefix_packet_budgets = [prefix_packet_budget(name) for name in self.prefix_names]
        if any(value <= 0 for value in self.prefix_packet_budgets):
            raise ValueError(f"{split}: prefix directories must be named like prefix_K")
        if any(
            left >= right
            for left, right in zip(self.prefix_packet_budgets, self.prefix_packet_budgets[1:])
        ):
            raise ValueError(
                f"{split}: prefix directories must be ordered by increasing packet budget, "
                f"got {self.prefix_names}"
            )
        self.datasets = [
            GraphTrafficDataset(data_dir, split, byte_head)
            for data_dir in self.prefix_dirs
        ]

        reference = self.datasets[0].labels_numpy()
        for data_dir, dataset in zip(self.prefix_dirs[1:], self.datasets[1:]):
            labels = dataset.labels_numpy()
            if labels.shape != reference.shape or not np.array_equal(labels, reference):
                raise ValueError(
                    f"{data_dir}: {split} labels/sample order do not match {self.prefix_dirs[0]}"
                )

        observed = np.stack(
            [
                np.asarray(dataset.observed_packets, dtype=np.int64)
                for dataset in self.datasets
            ],
            axis=1,
        )
        if observed.shape[1] > 1 and (np.diff(observed, axis=1) < 0).any():
            raise ValueError(
                f"{split}: observed packet counts must be non-decreasing across prefixes; "
                "check that prefix directories are aligned to the same samples"
            )
        for budget, counts in zip(self.prefix_packet_budgets, observed.T):
            if (counts > budget).any():
                raise ValueError(f"{split}: observed packet count exceeds prefix budget {budget}")

        # Expose the first prefix feature metadata so existing model builders can
        # infer dimensions without knowing about the progressive wrapper.
        first = self.datasets[0]
        self.byte_x = first.byte_x
        self.packet_x = first.packet_x
        self.flow_x = first.flow_x
        self.burst_x = first.burst_x
        self.byte_head = first.byte_head

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int):
        items = [dataset[idx] for dataset in self.datasets]
        label = items[0]["label"]
        return {
            "byte_head": torch.stack([item["byte_head"] for item in items], dim=0),
            "byte_header_payload": torch.stack([item["byte_header_payload"] for item in items], dim=0),
            "packet": torch.stack([item["packet"] for item in items], dim=0),
            "flow": torch.stack([item["flow"] for item in items], dim=0),
            "burst": torch.stack([item["burst"] for item in items], dim=0),
            "observed_packet_count": torch.stack(
                [item["observed_packet_count"] for item in items],
                dim=0,
            ),
            "label": label,
            "sample_id": torch.tensor(int(idx), dtype=torch.long),
        }

    def labels_numpy(self) -> np.ndarray:
        return self.datasets[0].labels_numpy()

    def describe(self) -> str:
        pieces = [
            f"{name}: {dataset.describe()}"
            for name, dataset in zip(self.prefix_names, self.datasets)
        ]
        return " | ".join(pieces)


def build_progressive_splits(prefix_dirs, byte_head: int):
    prefix_dirs = [Path(path) for path in prefix_dirs]
    train_dataset = ProgressivePrefixDataset(prefix_dirs, "train", byte_head)
    val_dataset = ProgressivePrefixDataset(prefix_dirs, "val", byte_head)
    test_dataset = ProgressivePrefixDataset(prefix_dirs, "test", byte_head)
    return train_dataset, val_dataset, test_dataset

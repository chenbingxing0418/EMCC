"""Dataset helpers used by the CUT_ISCXVPN training script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class GraphTrafficDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, byte_head: int):
        self.byte_x = np.load(data_dir / f"{split}_byte.npy", mmap_mode="r")
        segment_path = data_dir / f"{split}_byte_header_payload.npy"
        self.byte_header_payload = (
            np.load(segment_path, mmap_mode="r")
            if segment_path.exists()
            else None
        )
        self.packet_x = np.load(data_dir / f"{split}_packet.npy", mmap_mode="r")
        self.flow_x = np.load(data_dir / f"{split}_flow.npy", mmap_mode="r")
        self.burst_x = np.load(data_dir / f"{split}_burst.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.observed_packets = np.load(data_dir / f"{split}_observed_packets.npy", mmap_mode="r")
        self.byte_head = min(byte_head, int(self.byte_x.shape[-1]))
        expected = len(self.y)
        lengths = {
            "byte": len(self.byte_x),
            "byte_header_payload": len(self.byte_header_payload) if self.byte_header_payload is not None else expected,
            "packet": len(self.packet_x),
            "flow": len(self.flow_x),
            "burst": len(self.burst_x),
            "observed_packets": len(self.observed_packets),
            "labels": expected,
        }
        bad_lengths = {name: length for name, length in lengths.items() if length != expected}
        if bad_lengths:
            raise ValueError(f"{data_dir}/{split}: feature arrays are not aligned: {bad_lengths}")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        if self.byte_header_payload is None:
            byte_header_payload = np.zeros(
                (int(self.byte_x.shape[1]), self.byte_head),
                dtype=np.int64,
            )
        else:
            byte_header_payload = np.array(
                self.byte_header_payload[idx, :, : self.byte_head],
                copy=True,
            )
        packet = np.array(self.packet_x[idx], copy=True)
        if self.byte_header_payload is not None and packet.shape[-1] >= 10:
            valid_packet = np.abs(packet).sum(axis=-1) > 0
            # Keep packet statistics compatible with the 0617 data used by the
            # earlier EMCC runs. Header/payload structure is provided through
            # byte_header_payload; changing packet columns here made the
            # comparison conflate two preprocessing changes.
            packet[:, 3] = packet[:, 0]
            packet[:, 9] = valid_packet.astype(np.float32)
        return {
            "byte_head": torch.from_numpy(np.array(self.byte_x[idx, :, : self.byte_head], copy=True)).float(),
            "byte_header_payload": torch.from_numpy(byte_header_payload).long(),
            "packet": torch.from_numpy(packet).float(),
            "flow": torch.from_numpy(np.array(self.flow_x[idx], copy=True)).float(),
            "burst": torch.from_numpy(np.array(self.burst_x[idx], copy=True)).float(),
            "observed_packet_count": torch.tensor(int(self.observed_packets[idx]), dtype=torch.long),
            "label": torch.tensor(int(self.y[idx]), dtype=torch.long),
        }

    def labels_numpy(self) -> np.ndarray:
        return np.asarray(self.y, dtype=np.int64)

    def describe(self) -> str:
        return (
            f"byte_head={(len(self), int(self.byte_x.shape[1]), self.byte_head)}, "
            f"byte_header_payload={'yes' if self.byte_header_payload is not None else 'no'}, "
            f"packet={tuple(int(v) for v in self.packet_x.shape)}, "
            f"flow={tuple(int(v) for v in self.flow_x.shape)}, "
            f"burst={tuple(int(v) for v in self.burst_x.shape)}, "
            f"observed_packets=[{int(self.observed_packets.min())}, {int(self.observed_packets.max())}], "
            f"labels={tuple(int(v) for v in self.y.shape)}"
        )


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["sample_id"] = torch.tensor(int(idx), dtype=torch.long)
        return item

    def labels_numpy(self) -> np.ndarray:
        return collect_labels(self.dataset)

    def describe(self) -> str:
        return self.dataset.describe()


def load_label_map(data_dir: Path) -> tuple[dict[str, int], list[str]]:
    with (data_dir / "label_map.json").open("r", encoding="utf-8") as fp:
        label_map = json.load(fp)
    label_map = {name: int(idx) for name, idx in label_map.items()}
    label_names = [str(idx) for idx in range(len(label_map))]
    for name, idx in label_map.items():
        label_names[idx] = name
    return label_map, label_names


def collect_labels(dataset) -> np.ndarray:
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([collect_labels(subset) for subset in dataset.datasets])
    return dataset.labels_numpy()


def build_trainval_and_test(data_dir, byte_head: int):
    train_dataset = GraphTrafficDataset(data_dir, "train", byte_head)
    val_dataset = GraphTrafficDataset(data_dir, "val", byte_head)
    trainval_dataset = IndexedDataset(ConcatDataset([train_dataset, val_dataset]))
    train_labels = collect_labels(trainval_dataset.dataset)
    test_dataset = IndexedDataset(GraphTrafficDataset(data_dir, "test", byte_head))
    return train_dataset, val_dataset, trainval_dataset, train_labels, test_dataset


def build_joint_trainval_and_tests(data_dirs, byte_head: int):
    """Build one training set from aligned prefix variants and one test set per prefix."""
    data_dirs = [Path(data_dir) for data_dir in data_dirs]
    if not data_dirs:
        raise ValueError("joint prefix training requires at least one data directory")

    train_datasets = [GraphTrafficDataset(data_dir, "train", byte_head) for data_dir in data_dirs]
    val_datasets = [GraphTrafficDataset(data_dir, "val", byte_head) for data_dir in data_dirs]
    raw_test_datasets = [GraphTrafficDataset(data_dir, "test", byte_head) for data_dir in data_dirs]

    for split, datasets in (
        ("train", train_datasets),
        ("val", val_datasets),
        ("test", raw_test_datasets),
    ):
        reference = datasets[0].labels_numpy()
        for data_dir, dataset in zip(data_dirs[1:], datasets[1:]):
            labels = dataset.labels_numpy()
            if labels.shape != reference.shape or not np.array_equal(labels, reference):
                raise ValueError(
                    f"{data_dir}: {split} labels/sample order do not match {data_dirs[0]}"
                )

    joint_parts = []
    for train_dataset, val_dataset in zip(train_datasets, val_datasets):
        joint_parts.extend((train_dataset, val_dataset))
    trainval_dataset = IndexedDataset(ConcatDataset(joint_parts))
    train_labels = collect_labels(trainval_dataset.dataset)
    test_datasets = {
        data_dir.name: IndexedDataset(dataset)
        for data_dir, dataset in zip(data_dirs, raw_test_datasets)
    }
    return train_datasets, val_datasets, trainval_dataset, train_labels, test_datasets

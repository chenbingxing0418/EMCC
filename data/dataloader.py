"""DataLoader helpers for EMCC experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


GRANULARITY_VIEWS = ("packet", "burst", "flow", "fused")


def make_generator(seed: int, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, sampler=None, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )


def make_balanced_sampler(
    labels: np.ndarray,
    num_classes: int,
    alpha: float,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_weights = (counts.sum() / np.maximum(counts, 1.0)) ** alpha
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return (
        batch["byte_head"].to(device, non_blocking=True),
        batch["byte_header_payload"].to(device, non_blocking=True),
        batch["packet"].to(device, non_blocking=True),
        batch["flow"].to(device, non_blocking=True),
        batch["burst"].to(device, non_blocking=True),
        batch["observed_packet_count"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def make_loaders(args, trainval_dataset, test_dataset, train_labels, num_classes):
    sampler = None
    if args.sampler_alpha > 0:
        sampler = make_balanced_sampler(
            train_labels,
            num_classes,
            args.sampler_alpha,
            generator=make_generator(args.seed, offset=1001),
        )
    train_loader = build_loader(
        trainval_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        sampler=sampler,
        generator=make_generator(args.seed, offset=2002),
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed, offset=3003),
    )
    return train_loader, test_loader


def make_test_loader(args, test_dataset, offset: int = 3003):
    return build_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed, offset=offset),
    )

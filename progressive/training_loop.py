"""Training loop helpers."""

from __future__ import annotations

import torch
from tqdm import tqdm

from data.dataloader import build_loader, make_balanced_sampler, make_generator
from progressive.forward import forward_progressive
from progressive.losses import LOSS_KEYS, progressive_loss
from utils.training import autocast_context, sample_training_step_indices


def make_progressive_loaders(args, train_dataset, val_dataset, test_dataset, train_labels, num_classes):
    sampler = None
    if args.sampler_alpha > 0:
        sampler = make_balanced_sampler(
            train_labels,
            num_classes,
            args.sampler_alpha,
            generator=make_generator(args.seed, offset=1101),
        )
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        sampler=sampler,
        generator=make_generator(args.seed, offset=2102),
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed, offset=3103),
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        generator=make_generator(args.seed, offset=4104),
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args, prefix_budgets):
    model.train()
    totals = {key: 0.0 for key in LOSS_KEYS}
    final_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="train", leave=False, disable=args.disable_tqdm):
        optimizer.zero_grad(set_to_none=True)
        step_indices = sample_training_step_indices(batch["byte_head"].size(1), args)
        selected_budgets = [prefix_budgets[idx] for idx in step_indices]

        with autocast_context(args, device):
            outputs = forward_progressive(
                model,
                batch,
                device,
                prefix_budgets=prefix_budgets,
                step_indices=step_indices,
            )
            loss, stats = progressive_loss(outputs, model, criterion, args, selected_budgets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(outputs["labels"].size(0))
        total_samples += batch_size
        final_correct += int(outputs["logits"][:, -1].argmax(dim=-1).eq(outputs["labels"]).sum().item())
        for key, value in stats.items():
            totals[key] += float(value.item()) * batch_size

    denom = max(total_samples, 1)
    stats = {key: value / denom for key, value in totals.items()}
    stats["final_acc"] = final_correct / denom
    return stats


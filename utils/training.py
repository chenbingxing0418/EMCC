"""Runtime helpers shared by progressive training loops."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def move_progressive_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return (
        batch["byte_head"].to(device, non_blocking=True),
        batch["byte_header_payload"].to(device, non_blocking=True),
        batch["packet"].to(device, non_blocking=True),
        batch["flow"].to(device, non_blocking=True),
        batch["burst"].to(device, non_blocking=True),
        batch["observed_packet_count"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def amp_enabled(args: SimpleNamespace, device: torch.device) -> bool:
    return bool(args.use_amp and device.type == "cuda")


def autocast_context(args: SimpleNamespace, device: torch.device):
    if not amp_enabled(args, device):
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(args: SimpleNamespace, device: torch.device):
    enabled = amp_enabled(args, device)
    return torch.amp.GradScaler("cuda", enabled=enabled)


def sample_training_step_indices(steps: int, args: SimpleNamespace) -> list[int]:
    count = int(args.progressive_train_prefix_count)
    if count <= 0 or count >= steps:
        return list(range(steps))
    if count == 1:
        return [steps - 1]

    selected = {0, steps - 1}
    needed = max(count - len(selected), 0)
    middle_count = max(steps - 2, 0)
    for slot in range(min(needed, middle_count)):
        start = 1 + (slot * middle_count) // needed
        end = 1 + ((slot + 1) * middle_count) // needed
        if end <= start:
            end = start + 1
        choice = start + int(torch.randint(end - start, (1,)).item())
        selected.add(choice)
    return sorted(selected)

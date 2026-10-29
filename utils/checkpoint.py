"""Checkpoint helpers for progressive training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


def serializable_args(args: SimpleNamespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (tuple, list)):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def model_state_cpu(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def progressive_checkpoint_paths(args: SimpleNamespace) -> tuple[Path, Path]:
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    return checkpoint_dir / "last_checkpoint.pt", checkpoint_dir / "best_checkpoint.pt"


def save_best_model_artifact(
    args: SimpleNamespace,
    label_map: dict[str, int],
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_score: float,
) -> None:
    atomic_torch_save(
        {
            "model_state_dict": best_state,
            "label_map": label_map,
            "args": serializable_args(args),
            "protocol": "clean progressive EMCC-JSD; val-selected early exit",
            "best_epoch": best_epoch,
            "best_val_early_exit_macro_f1": best_score,
        },
        Path(args.model_path),
    )


def save_progressive_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    args: SimpleNamespace,
    label_map: dict[str, int],
    epoch: int,
    best_epoch: int,
    best_score: float,
    best_state: dict[str, torch.Tensor] | None,
    best_val: dict | None,
    prefix_names: list[str],
    prefix_budgets: list[int],
) -> None:
    atomic_torch_save(
        {
            "model_state_dict": model_state_cpu(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "label_map": label_map,
            "args": serializable_args(args),
            "protocol": "clean progressive EMCC-JSD; val-selected early exit",
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_early_exit_macro_f1": float(best_score),
            "best_state_dict": best_state,
            "best_val": best_val,
            "prefix_names": list(prefix_names),
            "prefix_packet_budgets": list(prefix_budgets),
        },
        path,
    )


def load_progressive_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    device: torch.device,
    label_map: dict[str, int],
) -> tuple[int, float, int, dict[str, torch.Tensor] | None, dict | None]:
    checkpoint = torch_load_checkpoint(path, device)
    checkpoint_label_map = checkpoint.get("label_map")
    if checkpoint_label_map is not None and checkpoint_label_map != label_map:
        raise ValueError(f"{path}: checkpoint label_map does not match current data")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1

    best_score = float(checkpoint.get("best_val_early_exit_macro_f1", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_state = checkpoint.get("best_state_dict")
    best_val = checkpoint.get("best_val")
    print(
        f"resume={path} start_epoch={start_epoch} "
        f"best_epoch={best_epoch} best_val_exit_macro_f1={best_score:.4f}"
    )
    return start_epoch, best_score, best_epoch, best_state, best_val


def load_eval_state(path: Path, model, device: torch.device, label_map: dict[str, int]) -> dict:
    checkpoint = torch_load_checkpoint(path, device)
    checkpoint_label_map = checkpoint.get("label_map")
    if checkpoint_label_map is not None and checkpoint_label_map != label_map:
        raise ValueError(f"{path}: checkpoint label_map does not match current data")
    state = checkpoint["model_state_dict"]
    state_source = "model_state_dict"
    model.load_state_dict(state)
    print(f"eval_checkpoint={path} state={state_source}")
    return checkpoint

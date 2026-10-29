"""Forward utilities for progressive prefix evaluation."""

from __future__ import annotations

import numpy as np
import torch

from utils.training import move_progressive_batch


def normalized_entropy_confidence(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return (1.0 - entropy / np.log(max(logits.size(-1), 2))).clamp(0.0, 1.0)


def set_model_granularity_maturity_prior(model, prefix_budgets: list[int], prior: np.ndarray) -> None:
    for module in model.modules():
        if hasattr(module, "set_granularity_maturity_prior"):
            module.set_granularity_maturity_prior(prefix_budgets, prior)


def forward_progressive(
    model,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    prefix_budgets: list[int] | tuple[int, ...],
    step_indices: list[int] | None = None,
) -> dict[str, torch.Tensor | list[dict]]:
    byte_x, byte_header_payload, packet_x, flow_x, burst_x, observed, labels = move_progressive_batch(batch, device)
    total_steps = int(byte_x.size(1))
    selected_steps = list(range(total_steps)) if step_indices is None else list(step_indices)
    if not selected_steps:
        raise ValueError("step_indices must contain at least one prefix step")

    logits_by_step = []
    reliability_by_step = []
    route_by_step = []
    maturity_by_step = []
    safe_stop_by_step = []
    available_by_step = []
    stage_logits_by_step = []
    granularity_maturity_by_step = []
    prediction_stage_by_step = []
    aux_by_step = []

    for step in selected_steps:
        logits, aux = model(
            byte_x[:, step],
            packet_x[:, step],
            flow_x[:, step],
            burst_x[:, step],
            observed[:, step],
            int(prefix_budgets[step]),
            byte_header_payload=byte_header_payload[:, step],
        )
        logits_by_step.append(logits)
        reliability_by_step.append(aux.get("exit_score", normalized_entropy_confidence(logits)))
        route_by_step.append(aux.get("fusion_gates"))
        maturity_by_step.append(aux.get("prefix_calibrated_availability"))
        safe_stop_by_step.append(aux.get("stage_safe_stop_scores"))
        available_by_step.append(aux.get("stage_availability_mask"))
        stage_logits_by_step.append(aux.get("stage_logits"))
        granularity_maturity_by_step.append(aux.get("granularity_maturity_prior"))
        prediction_stage_by_step.append(aux.get("prediction_stage"))
        aux_by_step.append(aux)

    return {
        "logits": torch.stack(logits_by_step, dim=1),
        "reliability": torch.stack(reliability_by_step, dim=1),
        "route_weights": _stack_optional(route_by_step, dim=1),
        "maturity": _stack_optional(maturity_by_step, dim=1),
        "stage_safe_stop": _stack_optional(safe_stop_by_step, dim=1),
        "stage_available": _stack_optional(available_by_step, dim=1),
        "stage_logits": _stack_optional(stage_logits_by_step, dim=1),
        "granularity_maturity": _stack_optional(granularity_maturity_by_step, dim=1),
        "prediction_stage": _stack_optional(prediction_stage_by_step, dim=1),
        "aux_by_step": aux_by_step,
        "observed": observed[:, selected_steps],
        "labels": labels,
        "step_indices": selected_steps,
    }


def _stack_optional(values: list[torch.Tensor | None], dim: int) -> torch.Tensor | None:
    if not values or any(value is None for value in values):
        return None
    return torch.stack(values, dim=dim)


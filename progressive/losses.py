"""Progressive sequence losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from loss.objective import compute_total_loss


LOSS_KEYS = ("loss", "objective", "ce", "margin", "consistency", "monotonic", "risk", "latency")


def progressive_loss(
    outputs: dict[str, torch.Tensor],
    model,
    criterion,
    args,
    prefix_budgets: list[int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["logits"]
    labels = outputs["labels"]
    reliability = outputs["reliability"]
    aux_by_step = outputs["aux_by_step"]

    objective, ce, margin = _emcc_objective(logits, aux_by_step, labels, model, criterion, args)
    consistency = _distill_to_last_prefix(logits, args.progressive_distill_temperature)
    monotonic = _monotonic_reliability_penalty(reliability)
    risk = _safe_stop_risk(logits, reliability, labels, outputs, args, prefix_budgets)
    latency = _latency_penalty(logits, reliability, labels, args, prefix_budgets)

    total = (
        objective
        + float(args.progressive_consistency_weight) * consistency
        + float(args.progressive_monotonic_weight) * monotonic
        + float(args.progressive_risk_weight) * risk
        + float(args.progressive_latency_weight) * latency
    )
    stats = {
        "loss": total.detach(),
        "objective": objective.detach(),
        "ce": ce.detach(),
        "margin": margin.detach(),
        "consistency": consistency.detach(),
        "monotonic": monotonic.detach(),
        "risk": risk.detach(),
        "latency": latency.detach(),
    }
    return total, stats


def _emcc_objective(logits, aux_by_step, labels, model, criterion, args):
    objectives = []
    ce_terms = []
    margin_terms = []
    for step, aux in enumerate(aux_by_step):
        objective, stats = compute_total_loss(
            model=model,
            logits=logits[:, step],
            aux=aux,
            labels=labels,
            criterion=criterion,
            args=args,
        )
        objectives.append(objective)
        ce_terms.append(stats["cls_loss"])
        margin_terms.append(stats["margin_loss"])
    return torch.stack(objectives).mean(), torch.stack(ce_terms).mean(), torch.stack(margin_terms).mean()


def _distill_to_last_prefix(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if logits.size(1) <= 1:
        return logits.new_zeros(())
    temperature = max(float(temperature), 1e-4)
    teacher = torch.softmax(logits[:, -1].detach() / temperature, dim=-1)
    student = F.log_softmax(logits[:, :-1] / temperature, dim=-1)
    teacher = teacher.unsqueeze(1).expand_as(student)
    return F.kl_div(
        student.reshape(-1, student.size(-1)),
        teacher.reshape(-1, teacher.size(-1)),
        reduction="batchmean",
    ) * (temperature**2)


def _monotonic_reliability_penalty(reliability: torch.Tensor) -> torch.Tensor:
    if reliability.size(1) <= 1:
        return reliability.new_zeros(())
    return F.relu(reliability[:, :-1] - reliability[:, 1:]).mean()


def _safe_stop_risk(
    logits: torch.Tensor,
    reliability: torch.Tensor,
    labels: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    args,
    prefix_budgets: list[int],
) -> torch.Tensor:
    if float(args.progressive_risk_weight) <= 0:
        return logits.new_zeros(())

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correctness = predictions.eq(labels.unsqueeze(1)).float()
        future_stable = torch.ones_like(correctness)
        for step in range(logits.size(1) - 2, -1, -1):
            future_stable[:, step] = (
                predictions[:, step].eq(predictions[:, step + 1]) & future_stable[:, step + 1].bool()
            ).float()
        target = correctness * future_stable

    weights = _early_prefix_weights(logits, prefix_budgets, args.progressive_risk_early_weight)
    reliability_logits = torch.logit(reliability.float().clamp(1e-4, 1.0 - 1e-4))
    sample_loss = F.binary_cross_entropy_with_logits(reliability_logits, target, reduction="none")
    main_loss = (sample_loss * weights.unsqueeze(0)).mean()
    stage_loss = _stage_safe_stop_risk(outputs, labels, weights)
    return main_loss if stage_loss is None else 0.5 * main_loss + 0.5 * stage_loss


def _stage_safe_stop_risk(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    prefix_weights: torch.Tensor,
) -> torch.Tensor | None:
    stage_safe_stop = outputs.get("stage_safe_stop")
    stage_logits = outputs.get("stage_logits")
    stage_available = outputs.get("stage_available")
    if stage_safe_stop is None or stage_logits is None or stage_available is None:
        return None

    with torch.no_grad():
        stage_preds = stage_logits.argmax(dim=-1)
        stage_correct = stage_preds.eq(labels.view(-1, 1, 1))
        future_stable = torch.ones_like(stage_correct, dtype=torch.bool)
        for step in range(stage_logits.size(1) - 2, -1, -1):
            future_stable[:, step] = stage_preds[:, step].eq(stage_preds[:, step + 1]) & future_stable[:, step + 1]
        target = (stage_correct & future_stable & stage_available.bool()).to(torch.float32)

    score_logits = torch.logit(stage_safe_stop.float().clamp(1e-4, 1.0 - 1e-4))
    loss = F.binary_cross_entropy_with_logits(score_logits, target.to(score_logits.dtype), reduction="none")
    mask = stage_available.float()
    weights = prefix_weights.view(1, -1, 1)
    return (loss * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)


def _latency_penalty(
    logits: torch.Tensor,
    reliability: torch.Tensor,
    labels: torch.Tensor,
    args,
    prefix_budgets: list[int],
) -> torch.Tensor:
    if float(args.progressive_latency_weight) <= 0:
        return logits.new_zeros(())
    if int(args.current_epoch) < int(args.progressive_latency_start_epoch):
        return logits.new_zeros(())

    with torch.no_grad():
        correctness = logits.argmax(dim=-1).eq(labels.unsqueeze(1)).float()
    early_bonus = 1.0 - _budget_norm(logits, prefix_budgets)
    numerator = ((1.0 - reliability.float()) * correctness * early_bonus.unsqueeze(0)).sum()
    denominator = (correctness * early_bonus.unsqueeze(0)).sum().clamp_min(1.0)
    return numerator / denominator


def _early_prefix_weights(logits: torch.Tensor, prefix_budgets: list[int], early_weight: float) -> torch.Tensor:
    budget_norm = _budget_norm(logits, prefix_budgets)
    return 1.0 + float(early_weight) * (1.0 - budget_norm)


def _budget_norm(logits: torch.Tensor, prefix_budgets: list[int]) -> torch.Tensor:
    steps = int(logits.size(1))
    budgets = torch.as_tensor(prefix_budgets, dtype=torch.float32, device=logits.device)
    if budgets.numel() != steps or not torch.all(budgets > 0):
        return torch.linspace(1.0 / max(steps, 1), 1.0, steps, dtype=logits.dtype, device=logits.device)
    return budgets / budgets.max().clamp_min(1.0)


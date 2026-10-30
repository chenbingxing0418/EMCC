"""Training objective for EMCC.

The model produces one cascaded prediction plus packet, burst, and flow stage
predictions. The objective keeps the optimization aligned with the method:
each stage learns class evidence only after that evidence is mature, and the
safe-stop score is calibrated separately from maturity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


STAT_NAMES = (
    "loss",
    "cls_loss",
    "stage_loss",
    "calibration_loss",
    "margin_loss",
    "packet_stage_loss",
    "burst_stage_loss",
    "flow_stage_loss",
    "packet_maturity",
    "burst_maturity",
    "flow_maturity",
    "packet_exit_score",
    "burst_exit_score",
    "flow_exit_score",
)


def zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_zeros(())


def _stage_cross_entropy(
    stage_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    stage_count = stage_logits.size(1)
    losses = [
        F.cross_entropy(stage_logits[:, stage_idx], labels, reduction="none")
        for stage_idx in range(stage_count)
    ]
    return torch.stack(losses, dim=1)


def _per_sample_classification_loss(
    criterion,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    name = criterion.__class__.__name__
    if name == "CrossEntropyLoss":
        return F.cross_entropy(
            logits,
            labels,
            weight=criterion.weight,
            label_smoothing=float(criterion.label_smoothing),
            reduction="none",
        )
    if name == "BalancedSoftmaxLoss":
        adjusted = logits + criterion.log_counts.to(logits.device)
        return F.cross_entropy(
            adjusted,
            labels,
            label_smoothing=float(criterion.label_smoothing),
            reduction="none",
        )
    if name == "LogitAdjustedCrossEntropyLoss":
        adjusted = logits + float(criterion.tau) * criterion.log_prior.to(logits.device)
        return F.cross_entropy(
            adjusted,
            labels,
            label_smoothing=float(criterion.label_smoothing),
            reduction="none",
        )
    if name == "FocalLoss":
        weight = criterion.weight.to(logits.device) if criterion.weight is not None else None
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=float(criterion.label_smoothing),
            reduction="none",
        )
        pt = torch.exp(-ce).clamp(max=1.0)
        return ((1.0 - pt) ** float(criterion.gamma)) * ce
    return None


def _stage_loss_weights(logits: torch.Tensor, args) -> torch.Tensor:
    return logits.new_tensor(
        [
            float(args.emcc_packet_loss_weight),
            float(args.emcc_burst_loss_weight),
            float(args.emcc_flow_loss_weight),
        ]
    )


def true_vs_nearest_negative_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if margin <= 0:
        return zero_like(logits)
    class_count = logits.size(-1)
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.masked_fill(
        F.one_hot(labels, num_classes=class_count).bool(),
        torch.finfo(logits.dtype).min,
    )
    nearest_negative = masked_logits.max(dim=1).values
    loss = F.relu(float(margin) + nearest_negative - true_logits)
    if sample_weight is not None:
        weight = sample_weight.to(loss.dtype).clamp_min(0.0)
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss.mean()


def stage_margin_loss(
    stage_logits: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    args,
    stage_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    margin = float(args.emcc_logit_margin)
    if margin <= 0:
        return zero_like(logits)
    losses = []
    for stage_idx in range(stage_logits.size(1)):
        sample_weight = None if stage_mask is None else stage_mask[:, stage_idx]
        losses.append(
            true_vs_nearest_negative_margin(
                stage_logits[:, stage_idx],
                labels,
                margin=margin,
                sample_weight=sample_weight,
            )
        )
    weights = _stage_loss_weights(logits, args)
    return (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1e-6)


def stage_evidence_loss(
    stage_logits: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    criterion,
    args,
    stage_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_losses = []
    for stage_idx in range(stage_logits.size(1)):
        if stage_mask is None:
            stage_losses.append(criterion(stage_logits[:, stage_idx], labels))
            continue
        weights = stage_mask[:, stage_idx].to(stage_logits.dtype).clamp(0.0, 1.0)
        active = weights.gt(1e-6)
        if active.any():
            per_sample = _per_sample_classification_loss(criterion, stage_logits[:, stage_idx], labels)
            if per_sample is None:
                stage_losses.append(criterion(stage_logits[active, stage_idx], labels[active]))
            else:
                stage_losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1e-6))
        else:
            stage_losses.append(zero_like(logits))
    per_stage_loss = torch.stack(stage_losses)
    weights = _stage_loss_weights(logits, args)
    stage_loss = (per_stage_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    per_sample_ce = _stage_cross_entropy(stage_logits, labels).detach()
    if stage_mask is None:
        per_stage_ce = per_sample_ce.mean(dim=0)
    else:
        mask = stage_mask.to(per_sample_ce.dtype)
        per_stage_ce = (per_sample_ce * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
    return stage_loss, per_stage_ce


def maturity_confidence_calibration(
    aux: dict,
    labels: torch.Tensor,
    logits: torch.Tensor,
    args,
) -> torch.Tensor:
    stage_logits = aux.get("stage_logits")
    safe_stop_logits = aux.get("stage_safe_stop_logits")
    stage_mask = aux.get("stage_availability_mask")
    if stage_logits is None or safe_stop_logits is None or stage_mask is None:
        return zero_like(logits)

    score_logits = safe_stop_logits.float()
    with torch.no_grad():
        stage_correct = stage_logits.argmax(dim=-1).eq(labels.unsqueeze(1)).to(score_logits.dtype)
        active = stage_mask.to(score_logits.dtype).clamp(0.0, 1.0)
        target = stage_correct.to(score_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(score_logits, target, reduction="none")
    stage_weights = _stage_loss_weights(logits, args).unsqueeze(0).to(loss.dtype)
    weighted_mask = active * stage_weights
    return (loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)


def compute_total_loss(
    model,
    logits: torch.Tensor,
    aux: dict,
    labels: torch.Tensor,
    criterion,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del model
    cls_loss = criterion(logits, labels)

    stage_logits = aux.get("stage_logits")
    if stage_logits is None:
        stage_loss = zero_like(logits)
        stage_margin = zero_like(logits)
        per_stage_ce = logits.new_zeros(3)
    else:
        stage_mask = aux.get("stage_supervision_mask", aux.get("stage_availability_mask"))
        stage_loss, per_stage_ce = stage_evidence_loss(
            stage_logits,
            labels,
            logits,
            criterion,
            args,
            stage_mask=stage_mask,
        )
        stage_margin = stage_margin_loss(stage_logits, labels, logits, args, stage_mask=stage_mask)

    calibration_loss = maturity_confidence_calibration(aux, labels, logits, args)
    main_margin = true_vs_nearest_negative_margin(
        logits,
        labels,
        margin=float(args.emcc_logit_margin),
    )
    margin_loss = 0.5 * main_margin + 0.5 * stage_margin
    total = (
        cls_loss
        + float(args.emcc_stage_loss_weight) * stage_loss
        + float(args.emcc_calibration_loss_weight) * calibration_loss
        + float(args.emcc_margin_loss_weight) * margin_loss
    )

    maturity = aux.get("prefix_calibrated_availability")
    if maturity is None:
        maturity_mean = logits.new_ones(3)
    else:
        maturity_mean = maturity.mean(dim=0)
    exit_scores = aux.get("stage_exit_scores")
    if exit_scores is None:
        exit_score_mean = logits.new_zeros(3)
    else:
        exit_score_mean = exit_scores.mean(dim=0)

    stats = {
        "loss": total,
        "cls_loss": cls_loss,
        "stage_loss": stage_loss,
        "calibration_loss": calibration_loss,
        "margin_loss": margin_loss,
        "packet_stage_loss": per_stage_ce[0],
        "burst_stage_loss": per_stage_ce[1],
        "flow_stage_loss": per_stage_ce[2],
        "packet_maturity": maturity_mean[0],
        "burst_maturity": maturity_mean[1],
        "flow_maturity": maturity_mean[2],
        "packet_exit_score": exit_score_mean[0],
        "burst_exit_score": exit_score_mean[1],
        "flow_exit_score": exit_score_mean[2],
    }
    return total, stats


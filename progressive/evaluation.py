"""Validation and causal online evaluation for progressive EMCC."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from progressive.early_exit import (
    can_exit_at_step,
    choose_early_exits,
    early_exit_scores,
    flow_has_ended,
    history_jsd,
    jensen_shannon,
    rule_from_args,
    selected_granularity_maturity,
    symmetric_kl,
    update_history,
)
from progressive.forward import forward_progressive, normalized_entropy_confidence
from progressive.losses import progressive_loss
from progressive.report_rows import confusion_matrix_rows, per_class_rows, sample_prediction_rows
from utils.evaluation import metric_bundle
from utils.training import autocast_context


@torch.no_grad()
def evaluate_offline(
    model,
    loader,
    criterion,
    device: torch.device,
    args,
    label_names: list[str],
    prefix_names: list[str],
    prefix_budgets: list[int],
    include_diagnostics: bool = False,
) -> dict:
    model.eval()
    rule = rule_from_args(args)
    losses = []
    logits_all = []
    reliability_all = []
    observed_all = []
    labels_all = []
    sample_ids_all = []
    route_all = []
    maturity_all = []
    stage_safe_stop_all = []
    stage_available_all = []
    stage_logits_all = []
    granularity_maturity_all = []
    prediction_stage_all = []

    for batch in tqdm(loader, desc="eval", leave=False, disable=args.disable_tqdm):
        with autocast_context(args, device):
            outputs = forward_progressive(model, batch, device, prefix_budgets=prefix_budgets)
        loss, _ = progressive_loss(outputs, model, criterion, args, prefix_budgets)
        batch_size = int(outputs["labels"].size(0))
        losses.append(float(loss.item()) * batch_size)

        logits_all.append(outputs["logits"].detach().cpu())
        reliability_all.append(outputs["reliability"].detach().cpu())
        observed_all.append(outputs["observed"].detach().cpu())
        labels_all.append(outputs["labels"].detach().cpu())
        sample_ids_all.append(batch["sample_id"].detach().cpu())
        _append_optional(route_all, outputs["route_weights"])
        _append_optional(maturity_all, outputs["maturity"])
        _append_optional(stage_safe_stop_all, outputs["stage_safe_stop"])
        _append_optional(stage_available_all, outputs["stage_available"])
        _append_optional(stage_logits_all, outputs["stage_logits"])
        _append_optional(granularity_maturity_all, outputs["granularity_maturity"])
        _append_optional(prediction_stage_all, outputs["prediction_stage"])

    logits = torch.cat(logits_all, dim=0)
    probabilities = torch.softmax(logits, dim=-1).numpy()
    reliability = torch.cat(reliability_all, dim=0).numpy()
    observed = torch.cat(observed_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    sample_ids = torch.cat(sample_ids_all, dim=0).numpy()
    preds = probabilities.argmax(axis=-1)
    granularity_maturity = _cat_optional(granularity_maturity_all)
    prediction_stage = _cat_optional(prediction_stage_all)
    exit_score = early_exit_scores(
        probabilities,
        reliability,
        rule,
        granularity_maturity=granularity_maturity,
        prediction_stage=prediction_stage,
    )
    early_preds, exit_steps, exit_prediction_stability_jsd, exit_history_prediction_stability_jsd = choose_early_exits(
        probabilities,
        exit_score,
        rule,
        granularity_maturity=granularity_maturity,
        prediction_stage=prediction_stage,
        observed=observed,
        prefix_budgets=prefix_budgets,
        force_exit_at_flow_end=bool(getattr(args, "force_exit_at_flow_end", True)),
    )

    result = {
        "loss": sum(losses) / max(len(labels), 1),
        "prefix_metrics": _prefix_metrics(
            labels=labels,
            preds=preds,
            probabilities=probabilities,
            reliability=reliability,
            exit_score=exit_score,
            observed=observed,
            label_names=label_names,
            prefix_names=prefix_names,
            maturity=_cat_optional(maturity_all),
            granularity_maturity=granularity_maturity,
            prediction_stage=prediction_stage,
            stage_safe_stop=_cat_optional(stage_safe_stop_all),
            stage_available=_cat_optional(stage_available_all),
            stage_logits=_cat_optional(stage_logits_all),
            loss=sum(losses) / max(len(labels), 1),
        ),
        "early_exit": _early_exit_metrics(
            labels=labels,
            preds=early_preds,
            exit_steps=exit_steps,
            observed=observed,
            label_names=label_names,
            prefix_names=prefix_names,
            prefix_budgets=prefix_budgets,
            exit_prediction_stability_jsd=exit_prediction_stability_jsd,
            exit_history_prediction_stability_jsd=exit_history_prediction_stability_jsd,
            full_prefix_calls=len(labels) * len(prefix_names),
            forward_prefix_calls=int((exit_steps + 1).sum()),
        ),
        "stability": _stability_metrics(preds, reliability, prefix_names),
    }
    route = _cat_optional(route_all)
    if route is not None:
        _add_route_metrics(result, route, prefix_names)
    if include_diagnostics:
        _add_diagnostics(
            result,
            labels,
            early_preds,
            sample_ids,
            probabilities,
            reliability,
            exit_score,
            observed,
            exit_steps,
            label_names,
            prefix_names,
            prefix_budgets,
        )
    if bool(getattr(args, "save_eval_arrays", False)):
        maturity = _cat_optional(maturity_all)
        stage_safe_stop = _cat_optional(stage_safe_stop_all)
        stage_available = _cat_optional(stage_available_all)
        stage_logits = _cat_optional(stage_logits_all)
        result["_eval_arrays"] = {
            "probabilities": probabilities.astype(np.float32),
            "reliability": reliability.astype(np.float32),
            "observed": observed.astype(np.float32),
            "labels": labels.astype(np.int64),
            "sample_ids": sample_ids.astype(np.int64),
            "prediction_stage": (
                np.empty((0,), dtype=np.int64)
                if prediction_stage is None
                else prediction_stage.astype(np.int64)
            ),
            "maturity": (
                np.empty((0,), dtype=np.float32)
                if maturity is None
                else maturity.astype(np.float32)
            ),
            "stage_safe_stop": (
                np.empty((0,), dtype=np.float32)
                if stage_safe_stop is None
                else stage_safe_stop.astype(np.float32)
            ),
            "stage_available": (
                np.empty((0,), dtype=np.bool_)
                if stage_available is None
                else stage_available.astype(np.bool_)
            ),
            "stage_logits": (
                np.empty((0,), dtype=np.float32)
                if stage_logits is None
                else stage_logits.astype(np.float32)
            ),
            "granularity_maturity": (
                np.empty((0,), dtype=np.float32)
                if granularity_maturity is None
                else granularity_maturity.astype(np.float32)
            ),
        }
    return result


@torch.no_grad()
def evaluate_online(
    model,
    loader,
    device: torch.device,
    args,
    label_names: list[str],
    prefix_names: list[str],
    prefix_budgets: list[int],
    include_diagnostics: bool = False,
) -> dict:
    model.eval()
    rule = rule_from_args(args)
    prefix_count = len(prefix_names)
    class_count = len(label_names)
    labels_all = []
    preds_all = []
    exit_steps_all = []
    observed_all = []
    sample_ids_all = []
    exit_prediction_stability_jsd_all = []
    exit_history_prediction_stability_jsd_all = []
    forward_counts = np.zeros(prefix_count, dtype=np.int64)
    exit_counts = np.zeros(prefix_count, dtype=np.int64)
    policy_exit_counts = np.zeros(prefix_count, dtype=np.int64)
    flow_end_exit_counts = np.zeros(prefix_count, dtype=np.int64)
    horizon_exit_counts = np.zeros(prefix_count, dtype=np.int64)

    for batch in tqdm(loader, desc="online_eval", leave=False, disable=args.disable_tqdm):
        labels_np = batch["label"].detach().cpu().numpy().astype(np.int64)
        observed_np = batch["observed_packet_count"].detach().cpu().numpy()
        sample_ids_np = batch["sample_id"].detach().cpu().numpy().astype(np.int64)
        batch_size = int(labels_np.shape[0])
        active = np.ones(batch_size, dtype=bool)
        selected_preds = np.zeros(batch_size, dtype=np.int64)
        exit_steps = np.full(batch_size, prefix_count - 1, dtype=np.int64)
        exit_observed = observed_np[:, -1].astype(np.float32)
        exit_prediction_stability_jsd = np.zeros(batch_size, dtype=np.float32)
        exit_history_prediction_stability_jsd = np.zeros(batch_size, dtype=np.float32)
        previous_probs = [None for _ in range(batch_size)]
        previous_preds = [None for _ in range(batch_size)]
        histories = [[] for _ in range(batch_size)]

        for step in range(prefix_count):
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break
            forward_counts[step] += int(active_idx.size)
            logits, aux = _forward_active_prefix(model, batch, active_idx, step, prefix_budgets, device, args)
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            reliability = aux.get("exit_score", normalized_entropy_confidence(logits)).detach().float().cpu().numpy()
            prediction_stage = _tensor_to_numpy(aux.get("prediction_stage"), dtype=np.int64)
            granularity_maturity = _tensor_to_numpy(aux.get("granularity_maturity_prior"), dtype=np.float32)
            score = early_exit_scores(
                probabilities[:, None, :],
                reliability[:, None],
                rule,
                granularity_maturity=None if granularity_maturity is None else granularity_maturity[:, None, :],
                prediction_stage=None if prediction_stage is None else prediction_stage[:, None],
            )[:, 0]

            for local_idx, row_idx in enumerate(active_idx):
                probs = probabilities[local_idx]
                pred = int(probs.argmax())
                if granularity_maturity is None:
                    selected_prior = None
                elif prediction_stage is None:
                    selected_prior = float(granularity_maturity[local_idx].max())
                else:
                    selected_prior = float(
                        granularity_maturity[local_idx, int(np.clip(prediction_stage[local_idx], 0, 2))]
                    )
                force_final = step == prefix_count - 1
                force_flow_end = (
                    bool(getattr(args, "force_exit_at_flow_end", True))
                    and flow_has_ended(observed_np[row_idx], step, prefix_budgets[-1])
                )
                should_exit = can_exit_at_step(
                    probs=probs,
                    pred=pred,
                    score=float(score[local_idx]),
                    previous_probs=previous_probs[row_idx],
                    previous_pred=previous_preds[row_idx],
                    history=histories[row_idx],
                    step=step,
                    rule=rule,
                    jsd_prior=selected_prior,
                )
                if should_exit or force_flow_end or force_final:
                    selected_preds[row_idx] = pred
                    exit_steps[row_idx] = step
                    exit_observed[row_idx] = float(observed_np[row_idx, step])
                    exit_prediction_stability_jsd[row_idx] = 0.0 if previous_probs[row_idx] is None else jensen_shannon(probs, previous_probs[row_idx])
                    if rule.use_history_jsd:
                        exit_history_prediction_stability_jsd[row_idx] = history_jsd(probs, histories[row_idx])
                    active[row_idx] = False
                    exit_counts[step] += 1
                    if force_flow_end:
                        flow_end_exit_counts[step] += 1
                    elif force_final:
                        horizon_exit_counts[step] += 1
                    else:
                        policy_exit_counts[step] += 1
                else:
                    if rule.use_history_jsd:
                        histories[row_idx] = update_history(histories[row_idx], probs, rule.history_window)
                    previous_probs[row_idx] = probs.copy()
                    previous_preds[row_idx] = pred

        labels_all.append(labels_np)
        preds_all.append(selected_preds)
        exit_steps_all.append(exit_steps)
        observed_all.append(exit_observed)
        sample_ids_all.append(sample_ids_np)
        exit_prediction_stability_jsd_all.append(exit_prediction_stability_jsd)
        exit_history_prediction_stability_jsd_all.append(exit_history_prediction_stability_jsd)

    labels = np.concatenate(labels_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    exit_steps = np.concatenate(exit_steps_all, axis=0)
    exit_observed = np.concatenate(observed_all, axis=0)
    sample_ids = np.concatenate(sample_ids_all, axis=0)
    exit_prediction_stability_jsd = np.concatenate(exit_prediction_stability_jsd_all, axis=0)
    exit_history_prediction_stability_jsd = np.concatenate(exit_history_prediction_stability_jsd_all, axis=0)
    full_calls = int(labels.size * prefix_count)
    forward_calls = int(forward_counts.sum())
    result = {
        "loss": None,
        "prefix_metrics": {
            prefix_names[idx]: {
                "online_forward_count": int(forward_counts[idx]),
                "online_exit_count": int(exit_counts[idx]),
                "online_forward_ratio": float(forward_counts[idx] / max(labels.size, 1)),
                "online_exit_ratio": float(exit_counts[idx] / max(labels.size, 1)),
                "online_policy_exit_count": int(policy_exit_counts[idx]),
                "online_flow_end_exit_count": int(flow_end_exit_counts[idx]),
                "online_horizon_exit_count": int(horizon_exit_counts[idx]),
            }
            for idx in range(prefix_count)
        },
        "early_exit": _early_exit_metrics_from_selected(
            labels=labels,
            preds=preds,
            exit_steps=exit_steps,
            exit_observed=exit_observed,
            label_names=label_names,
            prefix_names=prefix_names,
            prefix_budgets=prefix_budgets,
            exit_prediction_stability_jsd=exit_prediction_stability_jsd,
            exit_history_prediction_stability_jsd=exit_history_prediction_stability_jsd,
            full_prefix_calls=full_calls,
            forward_prefix_calls=forward_calls,
        ),
        "stability": {
            "mean_exit_prediction_stability_jsd": float(exit_prediction_stability_jsd.mean()),
            "mean_exit_history_prediction_stability_jsd": float(exit_history_prediction_stability_jsd.mean()),
            "use_symmetric_kl_stability": bool(rule.use_symmetric_kl),
            "use_history_prediction_stability": bool(rule.use_history_jsd),
            "prediction_stability_window": int(rule.history_window),
            "prediction_stability_jsd_epsilon": float(rule.jsd_prev_epsilon),
            "history_prediction_stability_jsd_epsilon": float(rule.jsd_avg_epsilon),
        },
        "online_eval": {
            "enabled": True,
            "causal": True,
            "sample_count": int(labels.size),
            "sample_ids_first": [int(value) for value in sample_ids[:10]],
        },
    }
    result["early_exit"]["exit_reason_counts"] = {
        "policy": int(policy_exit_counts.sum()),
        "flow_end": int(flow_end_exit_counts.sum()),
        "horizon": int(horizon_exit_counts.sum()),
    }
    result["online_eval"]["force_exit_at_flow_end"] = bool(
        getattr(args, "force_exit_at_flow_end", True)
    )
    if include_diagnostics:
        metrics = metric_bundle(labels, preds, label_names)
        result["early_exit_confusion_matrix"] = confusion_matrix_rows(labels, preds, label_names)
        result["early_exit_per_class"] = per_class_rows(metrics)
    return result


def _forward_active_prefix(model, batch, active_idx, step: int, prefix_budgets, device, args):
    del args
    index = torch.as_tensor(active_idx, dtype=torch.long)
    byte_x = batch["byte_head"][index, step].to(device, non_blocking=True)
    byte_header_payload = batch["byte_header_payload"][index, step].to(device, non_blocking=True)
    packet_x = batch["packet"][index, step].to(device, non_blocking=True)
    flow_x = batch["flow"][index, step].to(device, non_blocking=True)
    burst_x = batch["burst"][index, step].to(device, non_blocking=True)
    observed = batch["observed_packet_count"][index, step].to(device, non_blocking=True)
    return model(
        byte_x,
        packet_x,
        flow_x,
        burst_x,
        observed,
        int(prefix_budgets[step]),
        byte_header_payload=byte_header_payload,
    )


def _prefix_metrics(
    labels,
    preds,
    probabilities,
    reliability,
    exit_score,
    observed,
    label_names,
    prefix_names,
    maturity,
    granularity_maturity,
    prediction_stage,
    stage_safe_stop,
    stage_available,
    stage_logits,
    loss,
) -> dict:
    del probabilities
    selected_jsd = selected_granularity_maturity(granularity_maturity, prediction_stage)
    stage_names = ("packet", "burst", "flow")
    stage_probs = None if stage_logits is None else torch.softmax(torch.from_numpy(stage_logits), dim=-1).numpy()
    stage_preds = None if stage_probs is None else stage_probs.argmax(axis=-1)
    rows = {}
    for step, prefix_name in enumerate(prefix_names):
        metrics = metric_bundle(labels, preds[:, step], label_names)
        row = {
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "mean_reliability": float(reliability[:, step].mean()),
            "mean_exit_score": float(exit_score[:, step].mean()),
            "mean_observed_packets": float(observed[:, step].mean()),
        }
        if maturity is not None:
            for stage_idx, stage_name in enumerate(stage_names):
                row[f"{stage_name}_maturity"] = float(maturity[:, step, stage_idx].mean())
        if granularity_maturity is not None:
            for stage_idx, stage_name in enumerate(stage_names):
                row[f"{stage_name}_granularity_maturity"] = float(granularity_maturity[:, step, stage_idx].mean())
        if selected_jsd is not None:
            row["selected_granularity_maturity"] = float(selected_jsd[:, step].mean())
        if stage_safe_stop is not None:
            for stage_idx, stage_name in enumerate(stage_names):
                row[f"{stage_name}_safe_stop"] = float(stage_safe_stop[:, step, stage_idx].mean())
        if stage_available is not None:
            available = stage_available.astype(bool)
            relative_cost = 1.0 + available[:, step, 1].astype(np.float32) + available[:, step, 2].astype(np.float32)
            row["mean_relative_flops"] = float(relative_cost.mean())
            row["mean_saved_relative_flops"] = float(3.0 - relative_cost.mean())
            for stage_idx, stage_name in enumerate(stage_names):
                row[f"{stage_name}_available_ratio"] = float(available[:, step, stage_idx].mean())
        if stage_preds is not None:
            stage_f1 = {}
            for stage_idx, stage_name in enumerate(stage_names):
                stage_metric = metric_bundle(labels, stage_preds[:, step, stage_idx], label_names)
                stage_f1[stage_name] = stage_metric["macro_f1"]
                row[f"{stage_name}_stage_f1"] = stage_metric["macro_f1"]
            row["burst_gain_over_packet"] = stage_f1["burst"] - stage_f1["packet"]
            row["flow_gain_over_burst"] = stage_f1["flow"] - stage_f1["burst"]
        rows[prefix_name] = row
    return rows


def _early_exit_metrics(
    labels,
    preds,
    exit_steps,
    observed,
    label_names,
    prefix_names,
    prefix_budgets,
    exit_prediction_stability_jsd,
    exit_history_prediction_stability_jsd,
    full_prefix_calls,
    forward_prefix_calls,
) -> dict:
    selected_observed = observed[np.arange(len(labels)), exit_steps]
    return _early_exit_metrics_from_selected(
        labels,
        preds,
        exit_steps,
        selected_observed,
        label_names,
        prefix_names,
        prefix_budgets,
        exit_prediction_stability_jsd,
        exit_history_prediction_stability_jsd,
        full_prefix_calls,
        forward_prefix_calls,
    )


def _early_exit_metrics_from_selected(
    labels,
    preds,
    exit_steps,
    exit_observed,
    label_names,
    prefix_names,
    prefix_budgets,
    exit_prediction_stability_jsd,
    exit_history_prediction_stability_jsd,
    full_prefix_calls,
    forward_prefix_calls,
) -> dict:
    metrics = metric_bundle(labels, preds, label_names)
    selected_budgets = np.asarray(prefix_budgets, dtype=np.float32)[exit_steps]
    full_calls = max(int(full_prefix_calls), 1)
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "mean_exit_step": float(exit_steps.mean() + 1.0),
        "mean_exit_prefix_budget": float(selected_budgets.mean()),
        "mean_exit_observed_packets": float(exit_observed.mean()),
        "exit_counts": {prefix_names[step]: int((exit_steps == step).sum()) for step in range(len(prefix_names))},
        "mean_exit_prediction_stability_jsd": float(exit_prediction_stability_jsd.mean()),
        "mean_exit_history_prediction_stability_jsd": float(exit_history_prediction_stability_jsd.mean()),
        "forward_prefix_calls": int(forward_prefix_calls),
        "full_prefix_calls": int(full_prefix_calls),
        "mean_forward_calls_per_sample": float(forward_prefix_calls / max(len(labels), 1)),
        "saved_forward_fraction": float(1.0 - forward_prefix_calls / full_calls),
    }


def _stability_metrics(preds, reliability, prefix_names) -> dict:
    flips = (preds[:, 1:] != preds[:, :-1]).sum(axis=1) if preds.shape[1] > 1 else np.zeros(len(preds))
    monotonic = reliability[:, :-1] > (reliability[:, 1:] + 1e-6) if reliability.shape[1] > 1 else np.zeros((len(preds), 0), dtype=bool)
    return {
        "mean_prediction_flips": float(flips.mean()),
        "no_flip_rate": float((flips == 0).mean()),
        "monotonic_violation_rate": float(monotonic.mean()) if monotonic.size else 0.0,
        "final_agreement": {
            prefix_names[step]: float((preds[:, step] == preds[:, -1]).mean())
            for step in range(len(prefix_names) - 1)
        },
    }


def _add_route_metrics(result: dict, route: np.ndarray, prefix_names: list[str]) -> None:
    route_names = ["packet", "burst", "flow"]
    if route.shape[-1] > 3:
        route_names.append("continue")
    result["mean_route_weights"] = {
        prefix_names[step]: {
            route_name: float(route[:, step, route_idx].mean())
            for route_idx, route_name in enumerate(route_names)
        }
        for step in range(len(prefix_names))
    }
    for step, prefix_name in enumerate(prefix_names):
        row = result["prefix_metrics"][prefix_name]
        for route_idx, route_name in enumerate(route_names):
            row[f"{route_name}_route_ratio"] = float(route[:, step, route_idx].mean())


def _add_diagnostics(
    result,
    labels,
    early_preds,
    sample_ids,
    probabilities,
    reliability,
    exit_score,
    observed,
    exit_steps,
    label_names,
    prefix_names,
    prefix_budgets,
) -> None:
    metrics = metric_bundle(labels, early_preds, label_names)
    result["early_exit_confusion_matrix"] = confusion_matrix_rows(labels, early_preds, label_names)
    result["early_exit_per_class"] = per_class_rows(metrics)
    result["sample_predictions"] = sample_prediction_rows(
        sample_ids=sample_ids,
        labels=labels,
        probabilities=probabilities,
        reliability=reliability,
        exit_score=exit_score,
        observed=observed,
        early_preds=early_preds,
        exit_steps=exit_steps,
        label_names=label_names,
        prefix_names=prefix_names,
        prefix_budgets=prefix_budgets,
    )


def _append_optional(target: list[torch.Tensor], value: torch.Tensor | None) -> None:
    if value is not None:
        target.append(value.detach().cpu())


def _cat_optional(values: list[torch.Tensor]) -> np.ndarray | None:
    if not values:
        return None
    return torch.cat(values, dim=0).numpy()


def _tensor_to_numpy(value, dtype):
    if value is None:
        return None
    return value.detach().cpu().numpy().astype(dtype)


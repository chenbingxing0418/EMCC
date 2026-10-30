"""Simple early-exit rules shared by validation and causal online testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EarlyExitRule:
    threshold: float
    min_step: int
    use_symmetric_kl: bool
    use_history_jsd: bool
    kl_epsilon: float
    jsd_prev_epsilon: float
    jsd_avg_epsilon: float
    jsd_prior_gamma: float
    history_window: int


def rule_from_args(args) -> EarlyExitRule:
    return EarlyExitRule(
        threshold=float(args.early_exit_threshold),
        min_step=max(int(args.early_exit_min_step), 1),
        use_symmetric_kl=bool(args.use_symmetric_kl_stability),
        use_history_jsd=bool(args.use_history_prediction_stability),
        kl_epsilon=max(float(args.prediction_stability_kl_epsilon), 0.0),
        jsd_prev_epsilon=max(float(args.prediction_stability_jsd_epsilon), 0.0),
        jsd_avg_epsilon=max(float(args.history_prediction_stability_jsd_epsilon), 0.0),
        jsd_prior_gamma=max(float(getattr(args, "prediction_stability_prior_gamma", 0.0)), 0.0),
        history_window=max(int(args.prediction_stability_window), 1),
    )


def probability_margin(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape[-1] < 2:
        return np.ones(probabilities.shape[:-1], dtype=np.float32)
    top2 = np.partition(probabilities, -2, axis=-1)[..., -2:]
    return (top2[..., 1] - top2[..., 0]).astype(np.float32)


def symmetric_kl(left: np.ndarray, right: np.ndarray) -> float:
    left = _as_distribution(left)
    right = _as_distribution(right)
    return float(0.5 * np.sum(left * (np.log(left) - np.log(right))) + 0.5 * np.sum(right * (np.log(right) - np.log(left))))


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = _as_distribution(left)
    right = _as_distribution(right)
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * (np.log(left) - np.log(midpoint))) + 0.5 * np.sum(right * (np.log(right) - np.log(midpoint))))


def selected_granularity_maturity(granularity_maturity: np.ndarray | None, prediction_stage: np.ndarray | None) -> np.ndarray | None:
    if granularity_maturity is None:
        return None
    granularity_maturity = np.asarray(granularity_maturity, dtype=np.float32)
    if granularity_maturity.ndim != 3 or granularity_maturity.shape[-1] != 3:
        return None
    if prediction_stage is None:
        return granularity_maturity.max(axis=-1)
    prediction_stage = np.asarray(prediction_stage, dtype=np.int64)
    if prediction_stage.shape != granularity_maturity.shape[:2]:
        return granularity_maturity.max(axis=-1)
    rows = np.arange(granularity_maturity.shape[0])[:, None]
    steps = np.arange(granularity_maturity.shape[1])[None, :]
    return granularity_maturity[rows, steps, np.clip(prediction_stage, 0, 2)]


def early_exit_scores(
    probabilities: np.ndarray,
    reliability: np.ndarray,
    rule: EarlyExitRule,
    granularity_maturity: np.ndarray | None = None,
    prediction_stage: np.ndarray | None = None,
) -> np.ndarray:
    del granularity_maturity, prediction_stage
    confidence = np.asarray(probabilities, dtype=np.float32).max(axis=-1)
    score = 0.7 * np.asarray(reliability, dtype=np.float32) + 0.3 * confidence
    return np.clip(score, 0.0, 1.0)


def choose_early_exits(
    probabilities: np.ndarray,
    exit_score: np.ndarray,
    rule: EarlyExitRule,
    granularity_maturity: np.ndarray | None = None,
    prediction_stage: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    prefix_budgets: list[int] | tuple[int, ...] | None = None,
    force_exit_at_flow_end: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    exit_score = np.asarray(exit_score, dtype=np.float32)
    preds = probabilities.argmax(axis=-1)
    sample_count, step_count = preds.shape
    selected_preds = preds[:, -1].copy()
    exit_steps = np.full(sample_count, step_count - 1, dtype=np.int64)
    exit_prediction_stability_jsd = np.zeros(sample_count, dtype=np.float32)
    exit_history_prediction_stability_jsd = np.zeros(sample_count, dtype=np.float32)
    selected_prior = selected_granularity_maturity(granularity_maturity, prediction_stage)

    for sample_idx in range(sample_count):
        history: list[np.ndarray] = []
        for step in range(step_count):
            probs = probabilities[sample_idx, step]
            policy_exit = can_exit_at_step(
                probs=probs,
                pred=int(preds[sample_idx, step]),
                score=float(exit_score[sample_idx, step]),
                previous_probs=probabilities[sample_idx, step - 1] if step > 0 else None,
                previous_pred=int(preds[sample_idx, step - 1]) if step > 0 else None,
                history=history,
                step=step,
                rule=rule,
                jsd_prior=(
                    None
                    if selected_prior is None
                    else float(selected_prior[sample_idx, step])
                ),
            )
            flow_end = (
                force_exit_at_flow_end
                and observed is not None
                and prefix_budgets is not None
                and flow_has_ended(observed[sample_idx], step, prefix_budgets[-1])
            )
            if policy_exit or flow_end:
                selected_preds[sample_idx] = int(preds[sample_idx, step])
                exit_steps[sample_idx] = step
                exit_prediction_stability_jsd[sample_idx] = (
                    0.0
                    if step == 0
                    else jensen_shannon(probs, probabilities[sample_idx, step - 1])
                )
                if rule.use_history_jsd:
                    exit_history_prediction_stability_jsd[sample_idx] = history_jsd(probs, history)
                break
            if rule.use_history_jsd:
                history = update_history(history, probs, rule.history_window)
    return selected_preds, exit_steps, exit_prediction_stability_jsd, exit_history_prediction_stability_jsd


def flow_has_ended(
    observed_by_step: np.ndarray,
    step: int,
    final_prefix_budget: int,
) -> bool:
    """Return whether a short flow has reached its last real packet."""
    observed_by_step = np.asarray(observed_by_step)
    if observed_by_step.ndim != 1 or observed_by_step.size == 0:
        raise ValueError("observed_by_step must be a non-empty one-dimensional array")
    if step < 0 or step >= observed_by_step.size:
        raise IndexError(f"step {step} is outside observed sequence length {observed_by_step.size}")
    full_observed = int(observed_by_step[-1])
    return (
        full_observed < int(final_prefix_budget)
        and int(observed_by_step[step]) >= full_observed
    )


def can_exit_at_step(
    probs: np.ndarray,
    pred: int,
    score: float,
    previous_probs: np.ndarray | None,
    previous_pred: int | None,
    history: list[np.ndarray],
    step: int,
    rule: EarlyExitRule,
    jsd_prior: float | None = None,
) -> bool:
    if step + 1 < rule.min_step:
        return False
    if score < rule.threshold:
        return False
    if previous_probs is None or previous_pred is None:
        return False
    if pred != previous_pred:
        return False
    if rule.use_symmetric_kl and symmetric_kl(probs, previous_probs) > rule.kl_epsilon:
        return False
    jsd_epsilon = rule.jsd_prev_epsilon
    if rule.jsd_prior_gamma > 0.0 and jsd_prior is not None:
        maturity = float(np.clip(jsd_prior, 1.0e-4, 1.0))
        jsd_epsilon *= maturity**rule.jsd_prior_gamma
    if jensen_shannon(probs, previous_probs) > jsd_epsilon:
        return False
    if rule.use_history_jsd and history and history_jsd(probs, history) > rule.jsd_avg_epsilon:
        return False
    return True


def update_history(history: list[np.ndarray], probs: np.ndarray, window: int) -> list[np.ndarray]:
    next_history = [*history, np.asarray(probs, dtype=np.float32).copy()]
    return next_history[-window:]


def history_jsd(probs: np.ndarray, history: list[np.ndarray]) -> float:
    if not history:
        return 0.0
    average = np.mean(np.stack(history, axis=0), axis=0)
    return jensen_shannon(probs, average)


def _as_distribution(values: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0)
    return result / max(float(result.sum()), 1e-8)


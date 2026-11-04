"""Configuration for the EMCC experiment."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.paths import DEFAULT_DATA_DIR, MY_MODEL_DIR


MODEL_NAME = "EMCC-ISCXVPN2016"
ARCHITECTURE = "evidence_maturity_cascaded_classifier"
DEFAULT_RESULTS_DIR = MY_MODEL_DIR / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "iscxvpn2016"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "best_model.pt"


@dataclass
class EMCCConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    joint_prefix_dirs: tuple[Path, ...] = ()
    model_path: Path = DEFAULT_MODEL_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    byte_head: int = 64
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7
    deterministic: bool = True
    use_amp: bool = True
    disable_tqdm: bool = bool(os.environ.get("PYCHARM_HOSTED"))
    log_every: int = 1
    current_epoch: int = 0

    hidden_dim: int = 128
    num_heads: int = 4
    gnn_layers: int = 1
    dropout: float = 0.25
    byte_encoder_type: str = "tcn"
    byte_transformer_layers: int = 2
    byte_tcn_layers: int = 4
    byte_tcn_kernel_size: int = 3
    byte_mask_mode: str = "raw"
    classifier_type: str = "prototype"
    prototypes_per_class: int = 1
    prototype_temperature: float = 0.20
    use_packet_gnn: bool = True
    use_burst_gnn: bool = True
    use_burst_cross_attention: bool = True
    use_flow_cross_attention: bool = True
    use_packet_granularity: bool = True
    use_burst_granularity: bool = True
    use_flow_granularity: bool = True

    emcc_availability_temperature: float = 1.5
    emcc_packet_exit_threshold: float = 0.80
    emcc_burst_exit_threshold: float = 0.85
    emcc_stage_loss_weight: float = 0.50
    emcc_calibration_loss_weight: float = 0.10
    emcc_packet_loss_weight: float = 1.00
    emcc_burst_loss_weight: float = 1.00
    emcc_flow_loss_weight: float = 1.00
    emcc_margin_loss_weight: float = 0.00
    emcc_logit_margin: float = 0.20
    prefix_burst_center: float = 4.0
    # The strict-flow experiment defaults to a moderately relaxed threshold.
    # The one-click runner also evaluates centers 4, 6, 8, and 10.
    prefix_flow_center: float = 6.0

    use_granularity_maturity_prior: bool = True
    granularity_maturity_prior_bins: int = 16
    granularity_maturity_prior_max_samples: int = 4096
    granularity_maturity_prior_ema_alpha: float = 0.60
    granularity_maturity_prior_floor: float = 0.70
    granularity_maturity_prior_blend: float = 0.50
    burst_evidence_availability_threshold: float = 0.30
    flow_evidence_availability_threshold: float = 0.40

    loss_type: str = "ce"
    label_smoothing: float = 0.02
    logit_adjust_tau: float = 1.0
    focal_gamma: float = 2.0
    sampler_alpha: float = 0.25

    progressive_consistency_weight: float = 0.20
    progressive_monotonic_weight: float = 0.05
    progressive_risk_weight: float = 0.05
    progressive_risk_early_weight: float = 1.0
    progressive_latency_weight: float = 0.02
    progressive_latency_start_epoch: int = 5
    progressive_distill_temperature: float = 2.0
    progressive_prefix_min: int = 1
    progressive_prefix_max: int = 20
    progressive_train_prefix_count: int = 5
    progressive_eval_test: bool = True
    progressive_checkpoint_every: int = 1

    early_exit_threshold: float = 0.70
    prediction_stability_kl_epsilon: float = 0.20
    early_exit_min_step: int = 3
    use_online_test_eval: bool = True
    force_exit_at_flow_end: bool = True
    use_symmetric_kl_stability: bool = False
    use_history_prediction_stability: bool = False
    prediction_stability_window: int = 3
    prediction_stability_jsd_epsilon: float = 0.020
    history_prediction_stability_jsd_epsilon: float = 0.030
    prediction_stability_prior_gamma: float = 0.50

    resume: Path | None = None
    eval_only: bool = False
    eval_test_only: bool = False
    eval_checkpoint: Path | None = None

    def to_args(self) -> SimpleNamespace:
        data = asdict(self)
        data["use_tabular_skip"] = False
        return SimpleNamespace(**data)


def parse_config() -> EMCCConfig:
    cfg = EMCCConfig()
    parser = argparse.ArgumentParser(description="EMCC progressive classifier")

    _add_paths(parser, cfg)
    _add_runtime(parser, cfg)
    _add_backbone(parser, cfg)
    _add_emcc(parser, cfg)
    _add_loss(parser, cfg)
    _add_progressive(parser, cfg)
    _add_early_exit(parser, cfg)
    _add_checkpoints(parser, cfg)
    _add_booleans(parser, cfg)

    parsed = parser.parse_args()
    for key, value in vars(parsed).items():
        setattr(cfg, key, value)
    return cfg


def _add_paths(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    parser.add_argument("--joint-prefix-dirs", type=Path, nargs="+", default=cfg.joint_prefix_dirs)
    parser.add_argument("--model-path", type=Path, default=cfg.model_path)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)


def _add_runtime(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--byte-head", type=int, default=cfg.byte_head)
    parser.add_argument("--log-every", type=int, default=cfg.log_every)


def _add_backbone(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    parser.add_argument("--num-heads", type=int, default=cfg.num_heads)
    parser.add_argument("--gnn-layers", type=int, default=cfg.gnn_layers)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--byte-encoder-type", choices=("tcn", "transformer"), default=cfg.byte_encoder_type)
    parser.add_argument("--byte-transformer-layers", type=int, default=cfg.byte_transformer_layers)
    parser.add_argument("--byte-tcn-layers", type=int, default=cfg.byte_tcn_layers)
    parser.add_argument("--byte-tcn-kernel-size", type=int, default=cfg.byte_tcn_kernel_size)
    parser.add_argument("--byte-mask-mode", choices=("raw", "mask_ip_port", "payload_only"), default=cfg.byte_mask_mode)
    parser.add_argument("--classifier-type", choices=("prototype", "linear"), default=cfg.classifier_type)
    parser.add_argument("--prototypes-per-class", type=int, default=cfg.prototypes_per_class)
    parser.add_argument("--prototype-temperature", type=float, default=cfg.prototype_temperature)


def _add_emcc(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--emcc-availability-temperature", "--emcc-maturity-temperature", dest="emcc_availability_temperature", type=float, default=cfg.emcc_availability_temperature)
    parser.add_argument("--emcc-packet-exit-threshold", type=float, default=cfg.emcc_packet_exit_threshold)
    parser.add_argument("--emcc-burst-exit-threshold", type=float, default=cfg.emcc_burst_exit_threshold)
    parser.add_argument("--emcc-stage-loss-weight", type=float, default=cfg.emcc_stage_loss_weight)
    parser.add_argument("--emcc-calibration-loss-weight", type=float, default=cfg.emcc_calibration_loss_weight)
    parser.add_argument("--emcc-packet-loss-weight", type=float, default=cfg.emcc_packet_loss_weight)
    parser.add_argument("--emcc-burst-loss-weight", type=float, default=cfg.emcc_burst_loss_weight)
    parser.add_argument("--emcc-flow-loss-weight", type=float, default=cfg.emcc_flow_loss_weight)
    parser.add_argument("--emcc-margin-loss-weight", type=float, default=cfg.emcc_margin_loss_weight)
    parser.add_argument("--emcc-logit-margin", type=float, default=cfg.emcc_logit_margin)
    parser.add_argument("--prefix-burst-center", type=float, default=cfg.prefix_burst_center)
    parser.add_argument("--prefix-flow-center", type=float, default=cfg.prefix_flow_center)
    parser.add_argument("--granularity-maturity-prior-bins", "--jsd-maturity-bins", dest="granularity_maturity_prior_bins", type=int, default=cfg.granularity_maturity_prior_bins)
    parser.add_argument("--granularity-maturity-prior-max-samples", "--jsd-maturity-max-samples", dest="granularity_maturity_prior_max_samples", type=int, default=cfg.granularity_maturity_prior_max_samples)
    parser.add_argument("--granularity-maturity-prior-ema-alpha", "--jsd-maturity-ema-alpha", dest="granularity_maturity_prior_ema_alpha", type=float, default=cfg.granularity_maturity_prior_ema_alpha)
    parser.add_argument("--granularity-maturity-prior-floor", "--jsd-maturity-floor", dest="granularity_maturity_prior_floor", type=float, default=cfg.granularity_maturity_prior_floor)
    parser.add_argument("--granularity-maturity-prior-blend", "--jsd-structure-blend", dest="granularity_maturity_prior_blend", type=float, default=cfg.granularity_maturity_prior_blend)
    parser.add_argument("--burst-evidence-availability-threshold", "--structural-maturity-burst-threshold", dest="burst_evidence_availability_threshold", type=float, default=cfg.burst_evidence_availability_threshold)
    parser.add_argument("--flow-evidence-availability-threshold", "--structural-maturity-flow-threshold", dest="flow_evidence_availability_threshold", type=float, default=cfg.flow_evidence_availability_threshold)


def _add_loss(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--loss-type", choices=("ce", "balanced_softmax", "logit_adjusted_ce", "focal"), default=cfg.loss_type)
    parser.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    parser.add_argument("--logit-adjust-tau", type=float, default=cfg.logit_adjust_tau)
    parser.add_argument("--focal-gamma", type=float, default=cfg.focal_gamma)
    parser.add_argument("--sampler-alpha", type=float, default=cfg.sampler_alpha)


def _add_progressive(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--progressive-consistency-weight", type=float, default=cfg.progressive_consistency_weight)
    parser.add_argument("--progressive-monotonic-weight", type=float, default=cfg.progressive_monotonic_weight)
    parser.add_argument("--progressive-risk-weight", type=float, default=cfg.progressive_risk_weight)
    parser.add_argument("--progressive-risk-early-weight", type=float, default=cfg.progressive_risk_early_weight)
    parser.add_argument("--progressive-latency-weight", type=float, default=cfg.progressive_latency_weight)
    parser.add_argument("--progressive-latency-start-epoch", type=int, default=cfg.progressive_latency_start_epoch)
    parser.add_argument("--progressive-distill-temperature", type=float, default=cfg.progressive_distill_temperature)
    parser.add_argument("--progressive-train-prefix-count", type=int, default=cfg.progressive_train_prefix_count)
    parser.add_argument("--progressive-prefix-min", type=int, default=cfg.progressive_prefix_min)
    parser.add_argument("--progressive-prefix-max", type=int, default=cfg.progressive_prefix_max)
    parser.add_argument("--progressive-checkpoint-every", type=int, default=cfg.progressive_checkpoint_every)


def _add_early_exit(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--early-exit-threshold", type=float, default=cfg.early_exit_threshold)
    parser.add_argument("--prediction-stability-kl-epsilon", "--early-exit-kl-epsilon", dest="prediction_stability_kl_epsilon", type=float, default=cfg.prediction_stability_kl_epsilon)
    parser.add_argument("--early-exit-min-step", type=int, default=cfg.early_exit_min_step)
    parser.add_argument("--prediction-stability-window", "--online-stability-window", dest="prediction_stability_window", type=int, default=cfg.prediction_stability_window)
    parser.add_argument("--prediction-stability-jsd-epsilon", "--online-jsd-prev-epsilon", dest="prediction_stability_jsd_epsilon", type=float, default=cfg.prediction_stability_jsd_epsilon)
    parser.add_argument("--history-prediction-stability-jsd-epsilon", "--online-jsd-avg-epsilon", dest="history_prediction_stability_jsd_epsilon", type=float, default=cfg.history_prediction_stability_jsd_epsilon)
    parser.add_argument("--prediction-stability-prior-gamma", "--online-jsd-prior-gamma", dest="prediction_stability_prior_gamma", type=float, default=cfg.prediction_stability_prior_gamma)


def _add_checkpoints(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    parser.add_argument("--resume", type=Path, default=cfg.resume)
    parser.add_argument("--eval-checkpoint", type=Path, default=cfg.eval_checkpoint)


def _add_booleans(parser: argparse.ArgumentParser, cfg: EMCCConfig) -> None:
    _add_bool_pair(parser, "use_amp", cfg.use_amp)
    _add_bool_pair(parser, "deterministic", cfg.deterministic)
    _add_bool_pair(parser, "disable_tqdm", cfg.disable_tqdm)
    _add_bool_pair(parser, "progressive_eval_test", cfg.progressive_eval_test)
    _add_bool_pair(parser, "eval_only", cfg.eval_only)
    _add_bool_pair(parser, "eval_test_only", cfg.eval_test_only)
    _add_bool_pair(parser, "use_online_test_eval", cfg.use_online_test_eval)
    _add_bool_pair(parser, "force_exit_at_flow_end", cfg.force_exit_at_flow_end)
    _add_bool_pair(parser, "use_symmetric_kl_stability", cfg.use_symmetric_kl_stability)
    _add_bool_pair(parser, "use_history_prediction_stability", cfg.use_history_prediction_stability)
    _add_deprecated_bool_pair(parser, "use_history_prediction_stability", "use-history-jsd-stability")
    _add_bool_pair(parser, "use_packet_gnn", cfg.use_packet_gnn)
    _add_bool_pair(parser, "use_burst_gnn", cfg.use_burst_gnn)
    _add_bool_pair(parser, "use_burst_cross_attention", cfg.use_burst_cross_attention)
    _add_bool_pair(parser, "use_flow_cross_attention", cfg.use_flow_cross_attention)
    _add_bool_pair(parser, "use_packet_granularity", cfg.use_packet_granularity)
    _add_bool_pair(parser, "use_burst_granularity", cfg.use_burst_granularity)
    _add_bool_pair(parser, "use_flow_granularity", cfg.use_flow_granularity)
    _add_bool_pair(parser, "use_granularity_maturity_prior", cfg.use_granularity_maturity_prior)
    _add_deprecated_bool_pair(parser, "use_granularity_maturity_prior", "use-jsd-maturity-prior")


def _add_deprecated_bool_pair(parser: argparse.ArgumentParser, name: str, flag: str) -> None:
    parser.add_argument(f"--{flag}", dest=name, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(f"--no-{flag}", dest=name, action="store_false", help=argparse.SUPPRESS)


def _add_bool_pair(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    flag = name.replace("_", "-")
    parser.add_argument(f"--{flag}", dest=name, action="store_true")
    parser.add_argument(f"--no-{flag}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})






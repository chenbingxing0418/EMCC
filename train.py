"""Final progressive EMCC training entrypoint with causal online JSD stability."""

from __future__ import annotations

import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from config.emcc_config import parse_config
from data.dataset import load_label_map
from data.progressive_dataset import build_progressive_splits
from loss.classification import make_criterion
from models.model_factory import build_base_model
from progressive.evaluation import evaluate_offline, evaluate_online
from progressive.forward import set_model_granularity_maturity_prior
from progressive.outputs import write_progressive_outputs
from progressive.paths import resolve_progressive_prefix_dirs
from progressive.training_loop import make_progressive_loaders, train_one_epoch
from utils.checkpoint import (
    load_eval_state,
    load_progressive_checkpoint,
    model_state_cpu,
    progressive_checkpoint_paths,
    save_best_model_artifact,
    save_progressive_checkpoint,
)
from utils.seed import set_seed
from utils.granularity_maturity import build_granularity_maturity_prior, write_granularity_maturity_prior_report
from utils.training import amp_enabled, format_duration, make_grad_scaler


def main() -> None:
    run_start = time.perf_counter()
    wall_start = datetime.now()
    cfg = parse_config()
    args = cfg.to_args()
    prefix_dirs = resolve_progressive_prefix_dirs(args)
    args.joint_prefix_dirs = tuple(prefix_dirs)

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    label_map, label_names = load_and_check_label_map(prefix_dirs)
    train_dataset, val_dataset, test_dataset = build_progressive_splits(prefix_dirs, args.byte_head)
    train_labels = train_dataset.labels_numpy()
    train_loader, val_loader, test_loader = make_progressive_loaders(
        args,
        train_dataset,
        val_dataset,
        test_dataset,
        train_labels,
        len(label_names),
    )

    model = build_base_model(args, train_dataset, train_labels, len(label_names), device)
    if not args.eval_only:
        prepare_jsd_prior(model, train_dataset, args)
    criterion = make_criterion(args, train_labels, len(label_names), class_weights=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path, best_checkpoint_path = progressive_checkpoint_paths(args)

    print_run_summary(args, wall_start, prefix_dirs, train_dataset, val_dataset, test_dataset, label_map, model, device)

    if args.eval_only:
        run_eval_only(model, val_loader, test_loader, criterion, device, args, label_map, label_names, train_dataset)
        print_elapsed(run_start)
        return

    best_state, best_epoch, best_score, best_val, start_epoch = load_resume_state(
        args,
        model,
        optimizer,
        scheduler,
        device,
        label_map,
    )
    checkpoint_every = max(int(args.progressive_checkpoint_every), 1)

    for epoch in range(start_epoch, args.epochs + 1):
        args.current_epoch = epoch
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args,
            train_dataset.prefix_packet_budgets,
        )
        scheduler.step()
        val_result = evaluate_offline(
            model,
            val_loader,
            criterion,
            device,
            args,
            label_names,
            train_dataset.prefix_names,
            train_dataset.prefix_packet_budgets,
        )
        score = float(val_result["early_exit"]["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = deepcopy(val_result)
            best_state = model_state_cpu(model)
            save_best_model_artifact(args, label_map, best_state, best_epoch, best_score)
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, args, label_map, epoch, best_epoch, best_score, best_state, best_val, train_dataset)

        if epoch % checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(last_checkpoint_path, model, optimizer, scheduler, args, label_map, epoch, best_epoch, best_score, best_state, best_val, train_dataset)

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print_epoch(epoch, args.epochs, train_stats, val_result, train_dataset.prefix_names[-1])

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    save_best_model_artifact(args, label_map, best_state, best_epoch, best_score)
    print(f"best_epoch={best_epoch} best_val_exit_macro_f1={best_score:.4f}")
    print(f"model_saved={args.model_path}")
    run_final_evaluation(model, val_loader, test_loader, criterion, device, args, label_names, train_dataset)
    print_elapsed(run_start)


def load_and_check_label_map(prefix_dirs: list[Path]) -> tuple[dict[str, int], list[str]]:
    label_map, label_names = load_label_map(prefix_dirs[0])
    for data_dir in prefix_dirs[1:]:
        candidate_map, candidate_names = load_label_map(data_dir)
        if candidate_map != label_map or candidate_names != label_names:
            raise ValueError(f"{data_dir}: label_map.json does not match {prefix_dirs[0]}")
    return label_map, label_names


def load_resume_state(args, model, optimizer, scheduler, device, label_map):
    if not args.resume:
        return None, 0, -1.0, None, 1
    start_epoch, best_score, best_epoch, best_state, best_val = load_progressive_checkpoint(
        Path(args.resume),
        model,
        optimizer,
        scheduler,
        device,
        label_map,
    )
    return best_state, best_epoch, best_score, best_val, start_epoch


def save_checkpoint(path, model, optimizer, scheduler, args, label_map, epoch, best_epoch, best_score, best_state, best_val, train_dataset) -> None:
    save_progressive_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        args,
        label_map,
        epoch,
        best_epoch,
        best_score,
        best_state,
        best_val,
        train_dataset.prefix_names,
        train_dataset.prefix_packet_budgets,
    )


def run_eval_only(model, val_loader, test_loader, criterion, device, args, label_map, label_names, train_dataset) -> None:
    checkpoint = Path(args.eval_checkpoint or args.resume or args.model_path)
    load_eval_state(checkpoint, model, device, label_map)
    run_final_evaluation(model, val_loader, test_loader, criterion, device, args, label_names, train_dataset)


def prepare_jsd_prior(model, train_dataset, args) -> None:
    if not args.use_granularity_maturity_prior:
        return
    report = build_granularity_maturity_prior(train_dataset, args)
    set_model_granularity_maturity_prior(model, train_dataset.prefix_packet_budgets, report.prior)
    write_granularity_maturity_prior_report(report, args.output_dir)


def run_final_evaluation(model, val_loader, test_loader, criterion, device, args, label_names, train_dataset) -> None:
    if not bool(getattr(args, "eval_test_only", False)):
        val_result = evaluate_offline(
            model,
            val_loader,
            criterion,
            device,
            args,
            label_names,
            train_dataset.prefix_names,
            train_dataset.prefix_packet_budgets,
            include_diagnostics=True,
        )
        val_output_dir = args.output_dir / "val"
        write_progressive_outputs(val_output_dir, "val", val_result, args, train_dataset.prefix_names)
        print(
            f"val_exit_acc={val_result['early_exit']['accuracy']:.4f} "
            f"val_exit_macro_f1={val_result['early_exit']['macro_f1']:.4f} "
            f"val_{train_dataset.prefix_names[-1]}_f1="
            f"{val_result['prefix_metrics'][train_dataset.prefix_names[-1]]['macro_f1']:.4f} "
            f"val_outputs={val_output_dir}"
        )
    else:
        print("validation evaluation skipped by --eval-test-only.")

    if not args.progressive_eval_test:
        print("test evaluation skipped by --no-progressive-eval-test.")
        return

    if args.use_online_test_eval:
        test_result = evaluate_online(
            model,
            test_loader,
            device,
            args,
            label_names,
            train_dataset.prefix_names,
            train_dataset.prefix_packet_budgets,
            include_diagnostics=True,
        )
        test_output_dir = args.output_dir / "test_online"
    else:
        test_result = evaluate_offline(
            model,
            test_loader,
            criterion,
            device,
            args,
            label_names,
            train_dataset.prefix_names,
            train_dataset.prefix_packet_budgets,
            include_diagnostics=True,
        )
        test_output_dir = args.output_dir / "test"
    write_progressive_outputs(test_output_dir, "test", test_result, args, train_dataset.prefix_names)
    print(
        f"test_online={int(args.use_online_test_eval)} "
        f"causal={int(test_result.get('online_eval', {}).get('causal', False))} "
        f"test_exit_acc={test_result['early_exit']['accuracy']:.4f} "
        f"test_exit_macro_f1={test_result['early_exit']['macro_f1']:.4f} "
        f"mean_exit_observed_packets="
        f"{test_result['early_exit']['mean_exit_observed_packets']:.2f} "
        f"exit_step={test_result['early_exit']['mean_exit_step']:.2f} "
        f"test_outputs={test_output_dir}"
    )


def print_run_summary(args, wall_start, prefix_dirs, train_dataset, val_dataset, test_dataset, label_map, model, device) -> None:
    print("EMCC progressive training on ISCXVPN2016")
    print(f"start_time={wall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"prefix_dirs={', '.join(str(path) for path in prefix_dirs)}")
    print(f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
    print(f"runtime: device={device}, batch_size={args.batch_size}, amp={int(amp_enabled(args, device))}")
    print(f"prefixes={train_dataset.prefix_names}")
    print(f"label_map={label_map}")
    print(
        f"loss: objective + {args.progressive_consistency_weight}*distill "
        f"+ {args.progressive_monotonic_weight}*monotonic "
        f"+ {args.progressive_risk_weight}*risk "
        f"+ {args.progressive_latency_weight}*latency"
    )
    print(
        f"early_exit: threshold={args.early_exit_threshold:.3f}, min_step={args.early_exit_min_step}, "
        f"jsd_prev={args.prediction_stability_jsd_epsilon:.4f}, "
        f"history_jsd={int(args.use_history_prediction_stability)}, "
        f"symmetric_kl={int(args.use_symmetric_kl_stability)}, "
        f"force_flow_end={int(getattr(args, 'force_exit_at_flow_end', True))}, "
        f"online_test={int(args.use_online_test_eval)}"
    )
    print(
        f"flow_availability: strict_cascade=1, "
        f"burst_center={args.prefix_burst_center:.1f}, "
        f"flow_center={args.prefix_flow_center:.1f}"
    )
    print(
        "jsd_calibrated_availability: "
        f"enabled={int(args.use_granularity_maturity_prior)}, "
        f"ema_alpha={args.granularity_maturity_prior_ema_alpha:.2f}, "
        f"floor={args.granularity_maturity_prior_floor:.2f}, "
        f"structure_blend={args.granularity_maturity_prior_blend:.2f}, "
        f"burst_threshold={args.burst_evidence_availability_threshold:.2f}, "
        f"flow_threshold={args.flow_evidence_availability_threshold:.2f}, "
        f"online_prior_gamma={args.prediction_stability_prior_gamma:.2f}"
    )
    print("stage_selection: highest_available_cascade=1")
    print(f"model_params={sum(parameter.numel() for parameter in model.parameters()):,}")


def print_epoch(epoch: int, epochs: int, train_stats: dict, val_result: dict, last_prefix: str) -> None:
    print(
        f"Epoch {epoch:02d}/{epochs} "
        f"train_loss={train_stats['loss']:.4f} obj={train_stats['objective']:.4f} "
        f"ce={train_stats['ce']:.4f} risk={train_stats['risk']:.4f} "
        f"latency={train_stats['latency']:.4f} train_final_acc={train_stats['final_acc']:.4f} "
        f"val_exit_acc={val_result['early_exit']['accuracy']:.4f} "
        f"val_exit_f1={val_result['early_exit']['macro_f1']:.4f} "
        f"val_{last_prefix}_f1={val_result['prefix_metrics'][last_prefix]['macro_f1']:.4f} "
        f"mean_exit_observed_packets="
        f"{val_result['early_exit']['mean_exit_observed_packets']:.2f} "
        f"exit_step={val_result['early_exit']['mean_exit_step']:.2f} "
    )


def print_elapsed(run_start: float) -> None:
    elapsed = time.perf_counter() - run_start
    print(f"elapsed={format_duration(elapsed)} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()


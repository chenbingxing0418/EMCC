"""Main classification loss factory."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, labels: np.ndarray, num_classes: int, label_smoothing: float):
        super().__init__()
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        self.register_buffer("log_counts", torch.log(torch.from_numpy(np.maximum(counts, 1.0))))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.to(logits.device), labels, label_smoothing=self.label_smoothing)


class LogitAdjustedCrossEntropyLoss(nn.Module):
    def __init__(self, labels: np.ndarray, num_classes: int, tau: float, label_smoothing: float):
        super().__init__()
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        prior = counts / counts.sum()
        self.register_buffer("log_prior", torch.log(torch.from_numpy(np.maximum(prior, 1e-12))))
        self.tau = tau
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior.to(logits.device)
        return F.cross_entropy(adjusted, labels, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight: torch.Tensor | None, label_smoothing: float):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone())
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce).clamp(max=1.0)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_criterion(args, labels: np.ndarray, num_classes: int, class_weights: torch.Tensor | None) -> nn.Module:
    if args.loss_type == "balanced_softmax":
        return BalancedSoftmaxLoss(labels, num_classes, args.label_smoothing)
    if args.loss_type == "logit_adjusted_ce":
        return LogitAdjustedCrossEntropyLoss(labels, num_classes, args.logit_adjust_tau, args.label_smoothing)
    if args.loss_type == "focal":
        return FocalLoss(args.focal_gamma, class_weights, args.label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

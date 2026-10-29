"""Burst discovery, burst graph relations, and burst reliability evidence."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.model_constants import BURST_NODES, PACKET_NODES


def build_burst_relation_masks(burst_mask: torch.Tensor) -> torch.Tensor:
    """Create self, adjacent, and dense burst relations."""

    device = burst_mask.device
    bsz = burst_mask.size(0)
    rel = torch.zeros(bsz, 3, BURST_NODES, BURST_NODES, dtype=torch.bool, device=device)
    valid_pair = burst_mask.unsqueeze(1) & burst_mask.unsqueeze(2)

    eye = torch.eye(BURST_NODES, dtype=torch.bool, device=device).unsqueeze(0)
    rel[:, 0] = eye & valid_pair

    for i in range(BURST_NODES - 1):
        rel[:, 1, i, i + 1] = True
        rel[:, 1, i + 1, i] = True
    rel[:, 1] &= valid_pair
    rel[:, 2] = valid_pair
    return rel


def packet_to_burst_mean(
    packet_h: torch.Tensor,
    packet_mask: torch.Tensor,
    burst_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback deterministic packet-to-burst assignment from preprocessing features."""

    device = packet_h.device
    bsz = packet_h.size(0)
    burst_mask = burst_x.abs().sum(dim=-1) > 0
    packet_idx = torch.arange(PACKET_NODES, device=device)
    burst_counts = torch.round(burst_x[:, :, 1].clamp_min(0.0) * PACKET_NODES).long()
    burst_counts = burst_counts.masked_fill(~burst_mask, 0)
    starts = torch.cumsum(
        torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=device), burst_counts[:, :-1]], dim=1),
        dim=1,
    )
    ends = starts + burst_counts
    belongs = (packet_idx.view(1, 1, -1) >= starts.unsqueeze(-1)) & (
        packet_idx.view(1, 1, -1) < ends.unsqueeze(-1)
    )
    belongs = belongs & packet_mask.unsqueeze(1) & burst_mask.unsqueeze(-1)
    weights = belongs.to(packet_h.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    burst_packet_h = torch.einsum("bkp,bph->bkh", weights, packet_h) / denom
    packet_burst_edges = belongs.sum(dim=(-1, -2)).to(packet_h.dtype) * 2.0
    return burst_packet_h, burst_mask, packet_burst_edges


class LearnableBurstDiscovery(nn.Module):
    """Learn soft packet-to-burst boundaries from packet embeddings and statistics."""

    def __init__(
        self,
        packet_dim: int,
        burst_dim: int,
        hidden_dim: int,
        dropout: float,
        boundary_temperature: float,
        assignment_temperature: float,
        structure_prior_scale: float,
    ):
        super().__init__()
        self.boundary_temperature = max(float(boundary_temperature), 1e-3)
        self.assignment_temperature = max(float(assignment_temperature), 1e-3)
        self.structure_prior_scale = float(structure_prior_scale)
        self.semantic_boundary = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        structure_hidden = max(hidden_dim // 2, 16)
        self.structure_boundary = nn.Sequential(
            nn.Linear(packet_dim * 3, structure_hidden),
            nn.LayerNorm(structure_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(structure_hidden, 1),
        )
        self.dynamic_burst_feature = nn.Sequential(
            nn.Linear(packet_dim, burst_dim),
            nn.LayerNorm(burst_dim),
            nn.SiLU(),
        )
        self.register_buffer("slot_centers", torch.arange(BURST_NODES, dtype=torch.float32))
        nn.init.zeros_(self.semantic_boundary[-1].weight)
        nn.init.zeros_(self.semantic_boundary[-1].bias)
        nn.init.zeros_(self.structure_boundary[-1].weight)
        nn.init.zeros_(self.structure_boundary[-1].bias)

    def forward(
        self,
        packet_h: torch.Tensor,
        packet_x: torch.Tensor,
        packet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_h, right_h = packet_h[:, :-1], packet_h[:, 1:]
        learned_logits = self.semantic_boundary(
            torch.cat([left_h, right_h, (right_h - left_h).abs()], dim=-1)
        ).squeeze(-1)

        left_x, right_x = packet_x[:, :-1], packet_x[:, 1:]
        learned_logits = learned_logits + self.structure_boundary(
            torch.cat([left_x, right_x, (right_x - left_x).abs()], dim=-1)
        ).squeeze(-1)

        boundary_valid = packet_mask[:, :-1] & packet_mask[:, 1:]
        direction_change = (right_x[..., 2] - left_x[..., 2]).abs().gt(0.5)
        boundary_prior = direction_change.to(packet_h.dtype)
        prior_probability = torch.where(
            direction_change,
            learned_logits.new_full(learned_logits.shape, 0.80),
            learned_logits.new_full(learned_logits.shape, 0.08),
        )
        boundary_logits = (
            learned_logits + self.structure_prior_scale * torch.logit(prior_probability)
        ) / self.boundary_temperature
        boundary_prob = torch.sigmoid(boundary_logits) * boundary_valid.to(packet_h.dtype)

        soft_segment_position = torch.cat(
            [packet_h.new_zeros(packet_h.size(0), 1), torch.cumsum(boundary_prob, dim=1)],
            dim=1,
        ).clamp(max=float(BURST_NODES - 1))
        distance = (soft_segment_position.unsqueeze(1) - self.slot_centers.view(1, -1, 1)).abs()
        assignment = torch.softmax(-distance / self.assignment_temperature, dim=1)
        assignment = assignment * packet_mask.unsqueeze(1).to(packet_h.dtype)
        burst_mass = assignment.sum(dim=-1)
        burst_mask = burst_mass.gt(0.10)

        assignment = assignment * burst_mask.unsqueeze(-1).to(assignment.dtype)
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-6)
        assignment = assignment * packet_mask.unsqueeze(1).to(assignment.dtype)
        burst_mass = assignment.sum(dim=-1)

        normalized_assignment = assignment / burst_mass.unsqueeze(-1).clamp_min(1e-6)
        burst_packet_h = torch.einsum("bkp,bph->bkh", normalized_assignment, packet_h)
        burst_packet_x = torch.einsum("bkp,bpd->bkd", normalized_assignment, packet_x)
        dynamic_burst_x = self.dynamic_burst_feature(burst_packet_x)
        dynamic_burst_x = dynamic_burst_x * burst_mask.unsqueeze(-1).to(dynamic_burst_x.dtype)
        packet_burst_edges = packet_mask.sum(dim=1).to(packet_h.dtype) * 2.0
        return (
            burst_packet_h,
            dynamic_burst_x,
            burst_mask,
            packet_burst_edges,
            assignment,
            boundary_prob,
            boundary_prior,
        )


def boundary_availability_reliability(
    boundary_prob: torch.Tensor | None,
    boundary_valid: torch.Tensor,
    assignment: torch.Tensor | None,
) -> torch.Tensor | None:
    """Estimate burst-branch structure reliability from boundary and routing certainty."""

    if boundary_prob is None or assignment is None:
        return None

    boundary_certainty = (boundary_prob - 0.5).abs() * 2.0
    boundary_certainty = (
        boundary_certainty * boundary_valid.to(boundary_certainty.dtype)
    ).sum(dim=-1) / boundary_valid.sum(dim=-1).clamp_min(1).to(boundary_certainty.dtype)

    packet_valid = assignment.sum(dim=1).gt(0)
    route_prob = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-6)
    route_entropy = -(route_prob.clamp_min(1e-8) * route_prob.clamp_min(1e-8).log()).sum(dim=1)
    route_certainty = 1.0 - route_entropy / math.log(max(assignment.size(1), 2))
    route_certainty = (
        route_certainty * packet_valid.to(route_certainty.dtype)
    ).sum(dim=-1) / packet_valid.sum(dim=-1).clamp_min(1).to(route_certainty.dtype)
    return 0.5 * (boundary_certainty + route_certainty)


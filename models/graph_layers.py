"""Reusable graph layers used by packet, burst, and flow branches."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiRelationGraphAttentionLayer(nn.Module):
    """Graph attention over several relation masks.

    Inputs use shape [batch, nodes, hidden].  relation_masks has shape
    [batch, relations, nodes, nodes] and controls which edges each relation sees.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_relations: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.out_proj = nn.Linear(hidden_dim * num_relations, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _project(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        bsz, nodes, _ = x.shape
        return layer(x).reshape(bsz, nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, relation_masks: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        relation_outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for rel_idx in range(len(self.q_proj)):
            q = self._project(self.q_proj[rel_idx], x)
            k = self._project(self.k_proj[rel_idx], x)
            v = self._project(self.v_proj[rel_idx], x)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = relation_masks[:, rel_idx].unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e4)
            attn = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            relation_outputs.append(torch.matmul(self.dropout(attn), v).transpose(1, 2).reshape(x.size()))
        h = self.norm1(x + self.dropout(self.out_proj(torch.cat(relation_outputs, dim=-1))))
        h = self.norm2(h + self.ffn(h))
        return h * node_mask.unsqueeze(-1).to(h.dtype)


class GraphAttentionPool(nn.Module):
    """Attention pooling that ignores padded graph nodes."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1).masked_fill(~node_mask, -1e4)
        weight = torch.softmax(logits, dim=1) * node_mask.to(x.dtype)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (weight.unsqueeze(-1) * x).sum(dim=1), weight

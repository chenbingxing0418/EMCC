
"""Evidence-Maturity Cascaded Classification model.

EMCC is a staged early-classification backbone. It does not run packet, burst,
and flow evidence as competing parallel branches. Instead, each later stage is
enabled only after its evidence is mature enough, then upgrades the earlier
representation:

packet evidence -> burst upgrade -> flow upgrade -> safe stop or continue.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.burst_branch import build_burst_relation_masks
from models.graph_layers import GraphAttentionPool, MultiRelationGraphAttentionLayer
from models.model_constants import BURST_NODES, PACKET_NODES
from models.packet_branch import PacketNodeBuilder, build_packet_relation_masks


class LinearClassifier(nn.Module):
    """Standard linear evidence classifier."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class PrototypeClassifier(nn.Module):
    """Cosine prototype classifier with one or more evidence centers per class."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        prototypes_per_class: int,
        temperature: float,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.prototypes_per_class = max(1, int(prototypes_per_class))
        self.temperature = max(float(temperature), 1e-3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(
            torch.randn(self.num_classes, self.prototypes_per_class, hidden_dim) * 0.02
        )
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        evidence = torch.nn.functional.normalize(self.dropout(self.norm(evidence)), dim=-1)
        prototypes = torch.nn.functional.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bh,cph->bcp", evidence, prototypes) / self.temperature
        if self.prototypes_per_class == 1:
            logits = similarity.squeeze(-1)
        else:
            logits = torch.logsumexp(similarity, dim=-1)
        return logits + self.class_bias


def build_classifier(
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    classifier_type: str,
    prototypes_per_class: int,
    prototype_temperature: float,
) -> nn.Module:
    if str(classifier_type).lower() == "linear":
        return LinearClassifier(hidden_dim, num_classes, dropout)
    if str(classifier_type).lower() == "prototype":
        return PrototypeClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            prototypes_per_class=prototypes_per_class,
            temperature=prototype_temperature,
        )
    raise ValueError(f"unsupported classifier type: {classifier_type}")


class PacketEvidenceStage(nn.Module):
    """First stage: local packet evidence available from the earliest packets."""

    def __init__(
        self,
        byte_head: int,
        packet_dim: int,
        hidden_dim: int,
        num_heads: int,
        gnn_layers: int,
        dropout: float,
        byte_encoder_type: str,
        byte_transformer_layers: int,
        byte_tcn_layers: int,
        byte_tcn_kernel_size: int,
        byte_mask_mode: str,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.use_gnn = bool(use_gnn)
        self.node_builder = PacketNodeBuilder(
            byte_head=byte_head,
            packet_dim=packet_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
            byte_encoder_type=byte_encoder_type,
            byte_transformer_layers=byte_transformer_layers,
            byte_tcn_layers=byte_tcn_layers,
            byte_tcn_kernel_size=byte_tcn_kernel_size,
            byte_mask_mode=byte_mask_mode,
        )
        self.gnn = nn.ModuleList(
            [MultiRelationGraphAttentionLayer(hidden_dim, num_heads, 3, dropout) for _ in range(max(1, gnn_layers))]
            if self.use_gnn
            else []
        )
        self.pool = GraphAttentionPool(hidden_dim)

    def forward(
        self,
        byte_x: torch.Tensor,
        packet_x: torch.Tensor,
        observed_packet_count: torch.Tensor,
        byte_header_payload: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        nodes, mask = self.node_builder(byte_x, packet_x, byte_header_payload)
        relation_masks = build_packet_relation_masks(mask)
        for layer in self.gnn:
            nodes = layer(nodes, relation_masks, mask)
        representation, attention = self.pool(nodes, mask)
        maturity = observed_packet_count.to(nodes.dtype).ge(1.0).to(nodes.dtype)
        return {
            "nodes": nodes,
            "mask": mask,
            "relation_masks": relation_masks,
            "repr": representation,
            "attention": attention,
            "maturity": maturity,
        }


class BurstEvidenceStage(nn.Module):
    """Second stage: interaction evidence after packet direction structure matures."""

    def __init__(
        self,
        packet_dim: int,
        burst_dim: int,
        hidden_dim: int,
        num_heads: int,
        gnn_layers: int,
        dropout: float,
        burst_center: float,
        availability_temperature: float,
        use_gnn: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_gnn = bool(use_gnn)
        self.use_cross_attention = bool(use_cross_attention)
        self.burst_center = float(burst_center)
        self.availability_temperature = max(float(availability_temperature), 1e-3)
        self.boundary_score = nn.Sequential(
            nn.Linear(hidden_dim * 2 + packet_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.burst_proj = nn.Sequential(
            nn.Linear(hidden_dim + packet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.raw_burst_proj = nn.Sequential(
            nn.Linear(burst_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.evidence_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.position_embed = nn.Embedding(BURST_NODES, hidden_dim)
        if self.use_cross_attention:
            self.burst_query_norm = nn.LayerNorm(hidden_dim)
            self.packet_memory_norm = nn.LayerNorm(hidden_dim)
            self.packet_to_burst_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.packet_upgrade_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.upgrade_dropout = nn.Dropout(dropout)
        self.gnn = nn.ModuleList(
            [MultiRelationGraphAttentionLayer(hidden_dim, num_heads, 3, dropout) for _ in range(max(1, gnn_layers))]
            if self.use_gnn
            else []
        )
        self.pool = GraphAttentionPool(hidden_dim)
        self.register_buffer("slot_centers", torch.arange(BURST_NODES, dtype=torch.float32))

    def forward(
        self,
        packet_nodes: torch.Tensor,
        packet_x: torch.Tensor,
        burst_x: torch.Tensor,
        packet_mask: torch.Tensor,
        observed_packet_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assignment, boundary_prob, boundary_valid, direction_switch = self._packet_to_burst_assignment(
            packet_nodes,
            packet_x,
            packet_mask,
        )
        burst_mass = assignment.sum(dim=-1)
        burst_mask = burst_mass.gt(0.05)
        normalized_assignment = assignment / burst_mass.unsqueeze(-1).clamp_min(1e-6)

        burst_packet_repr = torch.einsum("bkp,bph->bkh", normalized_assignment, packet_nodes)
        burst_packet_feat = torch.einsum("bkp,bpd->bkd", normalized_assignment, packet_x)
        packet_burst_nodes = self.burst_proj(torch.cat([burst_packet_repr, burst_packet_feat], dim=-1))

        raw_burst_x = self._align_raw_burst_features(burst_x)
        raw_burst_mask = raw_burst_x.abs().sum(dim=-1).gt(0)
        raw_burst_nodes = self.raw_burst_proj(raw_burst_x)

        burst_mask = burst_mask | raw_burst_mask
        gate = torch.sigmoid(self.evidence_gate(torch.cat([packet_burst_nodes, raw_burst_nodes], dim=-1)))
        burst_nodes = gate * packet_burst_nodes + (1.0 - gate) * raw_burst_nodes
        positions = torch.arange(BURST_NODES, dtype=torch.long, device=burst_nodes.device)
        burst_nodes = burst_nodes + self.position_embed(positions).unsqueeze(0)

        if self.use_cross_attention:
            packet_context, packet_to_burst_attention = self.packet_to_burst_attention(
                query=self.burst_query_norm(burst_nodes),
                key=self.packet_memory_norm(packet_nodes),
                value=self.packet_memory_norm(packet_nodes),
                key_padding_mask=~packet_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            upgrade_gate = torch.sigmoid(
                self.packet_upgrade_gate(torch.cat([burst_nodes, packet_context], dim=-1))
            )
            burst_nodes = burst_nodes + upgrade_gate * self.upgrade_dropout(packet_context)
        else:
            packet_to_burst_attention = assignment
            upgrade_gate = burst_nodes.new_zeros(burst_nodes.size(0), burst_nodes.size(1), 1)
        burst_nodes = burst_nodes * burst_mask.unsqueeze(-1).to(burst_nodes.dtype)

        if self.use_gnn:
            relation_masks = build_burst_relation_masks(burst_mask)
            for layer in self.gnn:
                burst_nodes = layer(burst_nodes, relation_masks, burst_mask)
        representation, attention = self.pool(burst_nodes, burst_mask)

        direction_switch_count = direction_switch.sum(dim=-1).to(packet_nodes.dtype)
        valid_burst_count = burst_mask.sum(dim=-1).to(packet_nodes.dtype)
        count_maturity = torch.sigmoid(
            (observed_packet_count.to(packet_nodes.dtype) - self.burst_center) / self.availability_temperature
        )
        switch_maturity = torch.sigmoid((direction_switch_count - 1.0) / self.availability_temperature)
        burst_maturity = torch.sigmoid((valid_burst_count - 1.0) / self.availability_temperature)
        maturity = (count_maturity * switch_maturity * burst_maturity).clamp(0.0, 1.0)

        return {
            "nodes": burst_nodes,
            "mask": burst_mask,
            "repr": representation,
            "attention": attention,
            "maturity": maturity,
            "assignment": assignment,
            "boundary_prob": boundary_prob,
            "boundary_valid": boundary_valid,
            "direction_switch_count": direction_switch_count,
            "raw_burst_mask": raw_burst_mask,
            "burst_evidence_gate": gate.squeeze(-1),
            "packet_to_burst_attention": packet_to_burst_attention,
            "packet_upgrade_gate": upgrade_gate.squeeze(-1),
        }

    def _align_raw_burst_features(self, burst_x: torch.Tensor) -> torch.Tensor:
        if burst_x.size(1) == BURST_NODES:
            return burst_x
        if burst_x.size(1) > BURST_NODES:
            return burst_x[:, :BURST_NODES]
        pad_shape = (burst_x.size(0), BURST_NODES - burst_x.size(1), burst_x.size(2))
        padding = burst_x.new_zeros(pad_shape)
        return torch.cat([burst_x, padding], dim=1)

    def _packet_to_burst_assignment(
        self,
        packet_nodes: torch.Tensor,
        packet_x: torch.Tensor,
        packet_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_h, right_h = packet_nodes[:, :-1], packet_nodes[:, 1:]
        left_x, right_x = packet_x[:, :-1], packet_x[:, 1:]
        boundary_valid = packet_mask[:, :-1] & packet_mask[:, 1:]

        direction_switch = (right_x[..., 2] - left_x[..., 2]).abs().gt(0.5) & boundary_valid
        boundary_logits = self.boundary_score(torch.cat([left_h, right_h, left_x, right_x], dim=-1)).squeeze(-1)
        prior = torch.where(
            direction_switch,
            boundary_logits.new_full(boundary_logits.shape, 0.80),
            boundary_logits.new_full(boundary_logits.shape, 0.08),
        )
        boundary_prob = torch.sigmoid(boundary_logits + torch.logit(prior))
        boundary_prob = boundary_prob * boundary_valid.to(boundary_prob.dtype)

        segment_position = torch.cat(
            [packet_nodes.new_zeros(packet_nodes.size(0), 1), torch.cumsum(boundary_prob, dim=1)],
            dim=1,
        ).clamp(max=float(BURST_NODES - 1))
        distance = (segment_position.unsqueeze(1) - self.slot_centers.view(1, -1, 1)).abs()
        assignment = torch.softmax(-distance, dim=1) * packet_mask.unsqueeze(1).to(packet_nodes.dtype)
        assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-6)
        assignment = assignment * packet_mask.unsqueeze(1).to(packet_nodes.dtype)
        return assignment, boundary_prob, boundary_valid, direction_switch


class FlowEvidenceStage(nn.Module):
    """Final stage: consolidate mature packet, burst, and global flow context."""

    def __init__(
        self,
        flow_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        flow_center: float,
        availability_temperature: float,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = bool(use_cross_attention)
        self.flow_center = float(flow_center)
        self.availability_temperature = max(float(availability_temperature), 1e-3)
        self.flow_proj = nn.Sequential(
            nn.Linear(flow_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if self.use_cross_attention:
            self.flow_query_norm = nn.LayerNorm(hidden_dim)
            self.burst_memory_norm = nn.LayerNorm(hidden_dim)
            self.burst_to_flow_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.flow_upgrade_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.upgrade_dropout = nn.Dropout(dropout)

    def forward(
        self,
        burst_nodes: torch.Tensor,
        burst_mask: torch.Tensor,
        flow_x: torch.Tensor,
        observed_packet_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flow_context = self.flow_proj(flow_x)
        if self.use_cross_attention:
            safe_burst_mask = self._safe_memory_mask(burst_mask)
            burst_context, burst_to_flow_attention = self.burst_to_flow_attention(
                query=self.flow_query_norm(flow_context).unsqueeze(1),
                key=self.burst_memory_norm(burst_nodes),
                value=self.burst_memory_norm(burst_nodes),
                key_padding_mask=~safe_burst_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            burst_context = burst_context.squeeze(1)
            upgrade_gate = torch.sigmoid(
                self.flow_upgrade_gate(torch.cat([flow_context, burst_context], dim=-1))
            )
            flow_context = flow_context + upgrade_gate * self.upgrade_dropout(burst_context)
            burst_attention = burst_to_flow_attention.squeeze(1)
            gate = upgrade_gate.squeeze(-1)
        else:
            burst_attention = burst_nodes.new_zeros(burst_nodes.size(0), burst_nodes.size(1))
            gate = flow_context.new_zeros(flow_context.size(0))

        maturity = torch.sigmoid(
            (observed_packet_count.to(flow_x.dtype) - self.flow_center) / self.availability_temperature
        )
        return {
            "repr": flow_context,
            "maturity": maturity.clamp(0.0, 1.0),
            "burst_to_flow_attention": burst_attention,
            "flow_upgrade_gate": gate,
        }

    @staticmethod
    def _safe_memory_mask(mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        return safe_mask


class EMCCBackbone(nn.Module):
    """Evidence-maturity cascaded early classifier."""

    def __init__(
        self,
        num_classes: int,
        byte_head: int,
        packet_dim: int,
        burst_dim: int,
        flow_dim: int,
        hidden_dim: int,
        num_heads: int,
        gnn_layers: int,
        dropout: float,
        burst_center: float,
        flow_center: float,
        availability_temperature: float,
        packet_exit_threshold: float,
        burst_exit_threshold: float,
        byte_encoder_type: str,
        byte_transformer_layers: int,
        byte_tcn_layers: int,
        byte_tcn_kernel_size: int,
        byte_mask_mode: str,
        classifier_type: str,
        prototypes_per_class: int,
        prototype_temperature: float,
        use_packet_gnn: bool = True,
        use_burst_gnn: bool = True,
        use_burst_cross_attention: bool = True,
        use_flow_cross_attention: bool = True,
        use_packet_granularity: bool = True,
        use_burst_granularity: bool = True,
        use_flow_granularity: bool = True,
        use_granularity_maturity_prior: bool = True,
        granularity_maturity_prior_ema_alpha: float = 0.60,
        granularity_maturity_prior_floor: float = 0.90,
        granularity_maturity_prior_blend: float = 0.20,
        burst_evidence_availability_threshold: float = 0.30,
        flow_evidence_availability_threshold: float = 0.30,
    ):
        super().__init__()
        self.use_packet_encoder = True
        self.use_burst_encoder = True
        self.use_flow_encoder = True
        self.classifier_fused_only = True
        self.use_tabular_skip = False
        self.hidden_dim = int(hidden_dim)
        self.packet_exit_threshold = float(packet_exit_threshold)
        self.burst_exit_threshold = float(burst_exit_threshold)
        self.flow_exit_threshold = float(burst_exit_threshold)
        self.use_packet_granularity = bool(use_packet_granularity)
        self.use_burst_granularity = bool(use_burst_granularity)
        self.use_flow_granularity = bool(use_flow_granularity)
        self.use_granularity_maturity_prior = bool(use_granularity_maturity_prior)
        self.granularity_maturity_prior_ema_alpha = min(max(float(granularity_maturity_prior_ema_alpha), 0.0), 1.0)
        self.granularity_maturity_prior_floor = min(max(float(granularity_maturity_prior_floor), 0.0), 1.0)
        self.granularity_maturity_prior_blend = min(max(float(granularity_maturity_prior_blend), 0.0), 1.0)
        self.burst_evidence_availability_threshold = min(max(float(burst_evidence_availability_threshold), 0.0), 1.0)
        self.flow_evidence_availability_threshold = min(max(float(flow_evidence_availability_threshold), 0.0), 1.0)
        if not any(
            (
                self.use_packet_granularity,
                self.use_burst_granularity,
                self.use_flow_granularity,
            )
        ):
            raise ValueError("at least one evidence granularity must remain enabled")
        self.packet_stage = PacketEvidenceStage(
            byte_head,
            packet_dim,
            hidden_dim,
            num_heads,
            gnn_layers,
            dropout,
            byte_encoder_type=byte_encoder_type,
            byte_transformer_layers=byte_transformer_layers,
            byte_tcn_layers=byte_tcn_layers,
            byte_tcn_kernel_size=byte_tcn_kernel_size,
            byte_mask_mode=byte_mask_mode,
            use_gnn=use_packet_gnn,
        )
        self.burst_stage = BurstEvidenceStage(
            packet_dim=packet_dim,
            burst_dim=burst_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            gnn_layers=gnn_layers,
            dropout=dropout,
            burst_center=burst_center,
            availability_temperature=availability_temperature,
            use_gnn=use_burst_gnn,
            use_cross_attention=use_burst_cross_attention,
        )
        self.flow_stage = FlowEvidenceStage(
            flow_dim=flow_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            flow_center=flow_center,
            availability_temperature=availability_temperature,
            use_cross_attention=use_flow_cross_attention,
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.burst_delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.burst_upgrade_alpha = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.flow_delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.flow_upgrade_alpha = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.safe_stop_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )
        self.classifier = build_classifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            classifier_type=classifier_type,
            prototypes_per_class=prototypes_per_class,
            prototype_temperature=prototype_temperature,
        )
        self.register_buffer("granularity_maturity_prior", torch.ones(PACKET_NODES + 1, 3))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_key = prefix + "jsd_maturity_prior"
        new_key = prefix + "granularity_maturity_prior"
        if old_key in state_dict:
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
            state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def node_builder(self):
        return self.packet_stage.node_builder

    def set_granularity_maturity_prior(self, prefix_budgets: list[int] | tuple[int, ...], prior) -> None:
        """Load a smoothed training-set JSD maturity prior into the model."""
        prior_tensor = torch.as_tensor(prior, dtype=self.granularity_maturity_prior.dtype)
        if prior_tensor.ndim != 2 or prior_tensor.size(-1) != 3:
            raise ValueError(f"JSD maturity prior must have shape [prefixes, 3], got {tuple(prior_tensor.shape)}")

        prior_tensor = self._smooth_jsd_prior(prior_tensor.clamp(0.0, 1.0))
        full_prior = torch.ones_like(self.granularity_maturity_prior)
        for row_idx, budget in enumerate(prefix_budgets):
            if row_idx >= prior_tensor.size(0):
                break
            budget_idx = int(max(0, min(PACKET_NODES, int(budget))))
            full_prior[budget_idx] = prior_tensor[row_idx]
        self.granularity_maturity_prior.copy_(full_prior.to(self.granularity_maturity_prior.device))

    def _smooth_jsd_prior(self, prior: torch.Tensor) -> torch.Tensor:
        if prior.size(0) <= 1:
            return prior
        alpha = self.granularity_maturity_prior_ema_alpha
        smoothed = prior.clone()
        for idx in range(1, prior.size(0)):
            smoothed[idx] = alpha * smoothed[idx - 1] + (1.0 - alpha) * prior[idx]
        return torch.cummax(smoothed, dim=0).values

    def forward(
        self,
        byte_x: torch.Tensor,
        packet_x: torch.Tensor,
        flow_x: torch.Tensor,
        burst_x: torch.Tensor,
        observed_packet_count: torch.Tensor | None = None,
        prefix_budget: int | torch.Tensor | None = None,
        byte_header_payload: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 杩欎釜 forward 鍙鐞嗏€滃綋鍓嶄紶鍏ョ殑涓€涓?prefix鈥濈殑涓€鎵规牱鏈€?        # 璁粌銆侀獙璇併€佹祴璇曢兘浼氳皟鐢ㄥ畠锛涙槸鍚﹂亶鍘?prefix_1 鍒?prefix_20锛岀敱澶栧眰 train_progressive.py 鎺у埗銆?        # 鍥犳杩欓噷鏈韩涓嶄細鍋风湅鏈潵 prefix锛屼篃涓嶄細鍦ㄦ祴璇曢樁娈典富鍔ㄤ娇鐢ㄦ渶缁?prefix 鐨勯娴嬪垎甯冦€?
        if observed_packet_count is None:
            # 濡傛灉澶栧眰娌℃湁鏄惧紡浼犲叆褰撳墠 prefix 鐪熷疄鍙鐨勫寘鏁帮紝灏辨牴鎹?packet_x 鐨勯潪闆跺寘缁熻銆?            # packet_x 涓ˉ闆剁殑浣嶇疆涓嶇畻宸茶娴嬪寘锛涜繖鏍峰悗闈㈢殑鎴愮啛搴﹀垽鏂彧鍩轰簬褰撳墠宸茬粡鍒拌揪鐨勫寘銆?
            observed_packet_count = (packet_x.abs().sum(dim=-1) > 0).sum(dim=1)
        # 缁熶竴鎴?[batch] 褰㈢姸锛屽苟闄愬埗鍦?1 鍒?PACKET_NODES 涔嬮棿锛岄伩鍏嶆煡琛ㄦ垨鍥炬瀯寤烘椂鍑虹幇瓒婄晫銆?
        observed_packet_count = observed_packet_count.reshape(-1).clamp(min=1, max=PACKET_NODES)
        jsd_prior = self._lookup_jsd_prior(observed_packet_count, prefix_budget)
        jsd_gate = self._jsd_gate(jsd_prior)

        # JSD prior 鏄寜褰撳墠 prefix 闀垮害鏌ュ埌鐨勨€滅ǔ瀹氭€?鎴愮啛搴﹀厛楠屸€濄€?        # 瀹冩潵鑷缁冮樁娈电粺璁″嚭鏉ョ殑 prior 琛紱鍦ㄦ祴璇曟椂鍙牴鎹綋鍓?prefix_budget 鎴栧綋鍓嶅彲瑙佸寘鏁版煡琛ㄣ€?        # jsd_gate 鎶?prior 杞垚涓€涓俯鍜岄棬鎺э紝鐢ㄦ潵闄嶄綆涓嶇ǔ瀹氶樁娈电殑鎴愮啛搴︽潈閲嶃€?        # 绗竴灞傝瘉鎹細Byte/Packet 绾с€?        # packet_stage 鍐呴儴浼氭妸 byte_x 缂栫爜鎴愬寘鑺傜偣锛屽啀鐢?packet_x 鍜?packet 鍥剧粨鏋勬彁鍙栧眬閮ㄥ寘绾ц瘉鎹€?
        packet = self.packet_stage(
            byte_x,
            packet_x,
            observed_packet_count,
            byte_header_payload=byte_header_payload,
        )
        # packet_available 琛ㄧず褰撳墠 prefix 涓嬪寘绾ц瘉鎹槸鍚﹀凡缁忚揪鍒版渶浣庣粨鏋勬垚鐔熷害銆?        # 鍚庣画 stage_available 鍜?early-exit 鍒嗘暟閮戒細鐢ㄥ埌杩欎釜甯冨皵鎺╃爜銆?
        packet_available = packet["maturity"].ge(0.5)
        packet_stage_enabled = torch.full_like(
            packet_available,
            self.use_packet_granularity,
            dtype=torch.bool,
        )
        packet_stage_available = packet_available & packet_stage_enabled

        # 绗簩灞傝瘉鎹細Burst 绾с€?        # 鍙湁褰撳綋鍓?prefix 宸茬粡鍏峰瓒冲鍖呮暟銆佹柟鍚戝彉鍖?绐佸彂杈圭晫绛夋潯浠舵椂锛屾墠璁や负 burst 璇佹嵁鍙敤銆?
        burst_available = self._burst_available(packet_x, burst_x, packet["mask"], observed_packet_count)
        if not self.use_burst_granularity:
            burst_available = burst_available.new_zeros(burst_available.shape)
            burst = self._empty_burst(packet["nodes"], packet_x, burst_x)
        elif burst_available.any():
            # burst_stage 浣跨敤 packet 鑺傜偣銆佸寘缁熻鍜?burst 鐗瑰緛鏋勯€犵獊鍙戠骇琛ㄧず銆?            # 杩欓噷浠嶇劧鏄湪褰撳墠 prefix 鍐呭仛璁＄畻锛屼笉浼氳鍙栨湭鏉?prefix銆?
            burst = self.burst_stage(packet["nodes"], packet_x, burst_x, packet["mask"], observed_packet_count)
            burst_calibrated_availability = self._apply_jsd_structure_gate(
                burst["maturity"],
                jsd_gate[:, 1],
            )
            burst_available = burst_available & burst_calibrated_availability.ge(
                self.burst_evidence_availability_threshold
            )
            burst_weight = burst_available.to(packet["nodes"].dtype)
            # 瀵?burst 涓嶅彲鐢ㄧ殑鏍锋湰鍋氱‖鎺╃爜锛氭垚鐔熷害銆佽妭鐐广€乵ask銆佽〃绀洪兘缃负涓嶅彲璐＄尞銆?            # 杩欐牱 batch 鍐呮湁浜涙牱鏈彲鐢?burst銆佹湁浜涗笉鍙敤 burst 鏃讹紝涓嶄細浜掔浉姹℃煋銆?            burst["maturity"] = burst_calibrated_availability * burst_weight
            burst["nodes"] = burst["nodes"] * burst_weight.view(-1, 1, 1)
            burst["mask"] = burst["mask"] & burst_available.view(-1, 1)
            burst["repr"] = burst["repr"] * burst_weight.view(-1, 1)
        else:
            # 濡傛灉鏁存壒鏍锋湰閮芥病鏈夊彲鐢?burst锛屽氨杩斿洖涓€涓┖ burst 缁撴瀯锛屼繚鎸佸紶閲忓舰鐘朵竴鑷淬€?            # 杩欐牱鍚庨潰鐨勭骇鑱斿拰 aux 杈撳嚭涓嶇敤鍐欓澶栧垎鏀€?
            burst = self._empty_burst(packet["nodes"], packet_x, burst_x)

        # 绗笁灞傝瘉鎹細Flow 绾с€?        # flow_available 瑕佹眰褰撳墠 prefix 宸叉湁瓒冲鍖呮暟锛屽苟涓?flow_x 涓‘瀹炴湁闈為浂娴佺骇缁熻鐗瑰緛銆?
        flow_available = self._flow_available(
            flow_x,
            observed_packet_count,
            burst_available,
        )
        flow_calibrated_availability = None
        if not self.use_flow_granularity:
            flow_available = flow_available.new_zeros(flow_available.shape)
            flow = self._empty_flow(packet["nodes"], burst_x)
        elif flow_available.any():
            # flow_stage 鐢?burst 鑺傜偣鍜?flow_x 鐢熸垚娴佺骇琛ㄧず銆?            # 瀹冭〃杈剧殑鏄洿鍏ㄥ眬銆佹洿鏅氭垚鐔熺殑娴佺粺璁¤瘉鎹€?
            flow = self.flow_stage(
                burst["nodes"],
                burst["mask"],
                flow_x,
                observed_packet_count,
            )
            flow_calibrated_availability = self._apply_jsd_structure_gate(
                flow["maturity"],
                jsd_gate[:, 2],
            )
            flow_available = flow_available & flow_calibrated_availability.ge(
                self.flow_evidence_availability_threshold
            )
        else:
            # 濡傛灉褰撳墠 batch 鍏ㄩ儴娌℃湁 flow 璇佹嵁锛屽垯鏋勯€犵┖ flow 杈撳嚭锛岄伩鍏嶅悗缁垎绫昏矾寰勪腑鏂€?
            flow = self._empty_flow(packet["nodes"], burst_x)

        # 涓嬮潰鏄€滄畫宸紡璇佹嵁绾ц仈鈥濓紝涓嶆槸绠€鍗曟妸涓変釜鍚戦噺鎷艰捣鏉ャ€?        # packet_repr 鏄渶鏃╁彲鐢ㄧ殑鍩虹璇佹嵁銆?
        if flow_calibrated_availability is not None:
            flow["maturity"] = flow_calibrated_availability * flow_available.to(packet["nodes"].dtype)

        packet_repr = self.evidence_norm(packet["repr"])

        # raw_burst_repr 鏄?burst_stage 鐩存帴浜х敓鐨勭獊鍙戣瘉鎹€?        # burst_delta 瀛︿範鈥渂urst 鐩告瘮 packet 杩樿兘琛ュ厖浠€涔堜俊鎭€濄€?
        raw_burst_repr = self.evidence_norm(burst["repr"])
        burst_delta = self.burst_delta(torch.cat([packet_repr, raw_burst_repr], dim=-1))
        # burst_alpha 鏄牱鏈骇闂ㄦ帶锛屾帶鍒?burst_delta 浠ュ澶ф瘮渚嬪姞鍒?packet_repr 涓娿€?        # 濡傛灉 burst_available=False锛屽垯 alpha 琚箻鎴?0锛宐urst_repr 浼氶€€鍖栦负 packet_repr銆?
        burst_alpha = torch.sigmoid(self.burst_upgrade_alpha(torch.cat([packet_repr, raw_burst_repr], dim=-1))).squeeze(-1)
        burst_alpha = burst_alpha * burst_available.to(burst_alpha.dtype)
        burst_repr = packet_repr + burst_alpha.unsqueeze(-1) * burst_delta

        # raw_flow_repr 鏄?flow_stage 鐩存帴浜х敓鐨勬祦绾ц瘉鎹€?        # flow_delta 瀛︿範鈥渇low 鐩告瘮 burst 绱Н璇佹嵁杩樿兘琛ュ厖浠€涔堝叏灞€淇℃伅鈥濄€?
        raw_flow_repr = self.evidence_norm(flow["repr"])
        flow_delta = self.flow_delta(torch.cat([burst_repr, raw_flow_repr], dim=-1))
        # flow_alpha 鎺у埗 flow_delta 鐨勫姞鍏ユ瘮渚嬨€?        # 濡傛灉 flow_available=False锛屽垯 flow_repr 浼氶€€鍖栦负 burst_repr锛屼笉浼氬己琛屼娇鐢ㄤ笉瀛樺湪鐨?flow 璇佹嵁銆?
        flow_alpha = torch.sigmoid(self.flow_upgrade_alpha(torch.cat([burst_repr, raw_flow_repr], dim=-1))).squeeze(-1)
        flow_alpha = flow_alpha * flow_available.to(flow_alpha.dtype)
        flow_repr = burst_repr + flow_alpha.unsqueeze(-1) * flow_delta

        # stage_repr 淇濆瓨涓変釜鍊欓€夐樁娈电殑璇佹嵁琛ㄧず锛?        # 0=packet锛?=packet+burst 娈嬪樊澧炲己锛?=packet+burst+flow 娈嬪樊澧炲己銆?
        stage_repr = torch.stack(
            [
                packet_repr,
                burst_repr,
                flow_repr,
            ],
            dim=1,
        )

        #  stage_logits:
        #    packet 闃舵瀵规瘡涓被鍒殑鍘熷鍒嗘暟
        #   burst 闃舵瀵规瘡涓被鍒殑鍘熷鍒嗘暟
        #    flow 闃舵瀵规瘡涓被鍒殑鍘熷鍒嗘暟

        #packet_probs = [0.10, 0.70, 0.05, 0.10, 0.03,0.02]
        #burst_probs = [0.05, 0.82, 0.03, 0.06, 0.02,0.02]
        #flow_probs = [0.03, 0.88, 0.02, 0.04, 0.02,0.01]

        # 鍚屼竴涓?prototype classifier 鍒嗗埆浣滅敤鍦ㄤ笁涓樁娈佃〃绀轰笂锛屽緱鍒颁笁涓€欓€夊垎绫荤粨鏋溿€?        # 杩欓噷涓嶆槸鍏堥€夐樁娈靛啀鍒嗙被锛岃€屾槸鍏堣姣忎釜闃舵閮戒骇鐢熷彲鐩戠潱銆佸彲姣旇緝鐨勯娴嬨€?
        stage_logits = torch.stack([self.classifier(stage_repr[:, idx]) for idx in range(3)], dim=1)
        #stage_probs 鏄笁涓樁娈靛悇鑷殑鍒嗙被姒傜巼
        stage_probs = torch.softmax(stage_logits, dim=-1)

        #packet_confidence = 0.70
        #burst_confidence = 0.82
        #flow_confidence = 0.88

        # stage_confidence 鏄瘡涓樁娈靛垎绫绘鐜囩殑鏈€澶у€硷紝鐢ㄤ簬璇婃柇鍜岄儴鍒嗚缁冩崯澶便€?        # 灏辨槸鍙栨瘡涓樁娈垫渶楂樼殑閭ｄ釜绫诲埆姒傜巼锛屼綔涓鸿闃舵鐨勫垎绫荤疆淇″害
        stage_confidence = stage_probs.max(dim=-1).values

        # calibrated_availability 鏄ā鍨嬬粨鏋勫唴閮ㄤ及璁＄殑涓夐樁娈垫垚鐔熷害銆?        # stage_calibrated_availability 鍐嶄箻 JSD gate锛岃鈥滈娴嬬ǔ瀹氭€р€濆弬涓庢垚鐔熷害鍒ゆ柇銆?        # Final rule: no second JSD multiplication here; Burst/Flow were calibrated above.
        granularity_mask = packet["maturity"].new_tensor(
            [
                float(self.use_packet_granularity),
                float(self.use_burst_granularity),
                float(self.use_flow_granularity),
            ]
        ).view(1, 3)
        calibrated_availability = (
            torch.stack([packet["maturity"], burst["maturity"], flow["maturity"]], dim=-1)
            * granularity_mask
        )
        stage_calibrated_availability = calibrated_availability

        # stage_available 鏄‖鍙敤鎬ф帺鐮侊細璇佹嵁涓嶅瓨鍦ㄧ殑闃舵涓嶈兘琚€変綔鏈夋晥鏃╁仠闃舵銆?
        stage_available = torch.stack([packet_stage_available, burst_available, flow_available], dim=-1)

        # safe_stop_head 杈撳叆鐨勬槸涓変釜闃舵鐨勮〃绀猴紝涓烘瘡涓樁娈佃緭鍑轰竴涓€滃綋鍓嶉樁娈垫槸鍚﹁冻澶熷畨鍏ㄥ彲鍋溾€濈殑鍒嗘暟銆?        #  杈撳嚭鐨勬槸姣忎釜闃舵涓€涓師濮嬪垎鏁?logit銆傜劧鍚庣粡杩噑igmoid 鍙樻垚 0 鍒?1 涔嬮棿鐨勫垎鏁帮細
        # stage_safe_stop 瓒婃帴杩?1锛氭ā鍨嬭涓鸿繖涓樁娈垫洿閫傚悎瀹夊叏鏃╁仠
        # 0锛氭ā鍨嬭涓鸿繖涓樁娈典笉閫傚悎鏃╁仠

        # 瀹冧笉鏄垎绫诲櫒锛岃€屾槸鏃╁仠鍒ゆ柇涓殑瀹夊叏鍋滄鍒嗘敮銆?        #鐢卞綋鍓嶉樁娈靛帇缂╁悗鐨勮瘉鎹姸鎬佸緱鍒板嚭涓€涓爣閲忚〃绀鸿繖涓樁娈靛綋鍓嶉€備笉閫傚悎鏃╁仠锛屽洜涓哄彲浠ュ緱鍒版槸鍚﹀儚涓€涓€滃凡缁忔垚鐔熴€佸彲浠ュ仠鈥濈殑鐘舵€?
        stage_safe_stop_logits = self.safe_stop_head(stage_repr.reshape(-1, stage_repr.size(-1))).view(
            stage_repr.size(0),
            3,
        )
        stage_safe_stop = torch.sigmoid(stage_safe_stop_logits)

        # JSD prior 閫氳繃 exit_gate 璋冩暣 early-exit 鍒嗘暟锛?        # JSD 璁や负瓒婄ǔ瀹氾紝褰撳墠闃舵瓒婂鏄撹幏寰楄緝楂?exit score銆?
        exit_scores = (
            stage_safe_stop#锛氳繖涓樁娈佃嚜宸辫涓鸿兘涓嶈兘瀹夊叏鍋?            * stage_available.to(stage_safe_stop.dtype)#杩欎釜闃舵鑷繁璁や负鑳戒笉鑳藉畨鍏ㄥ仠
        ).clamp(0.0, 1.0)

        # _select_stages selects the prediction stage and diagnostic route.
        # The cross-prefix online early-exit rule is implemented outside the model.
        # prediction_stage selects logits for this prefix. route_weights only
        # records this prefix's internal stop/continue decision for diagnostics.
        prediction_stage, route_weights, selected_exit_score = self._select_stages(
            exit_scores,
            stage_available,
        )
        batch_index = torch.arange(stage_logits.size(0), device=stage_logits.device)
        # logits 鏄繖涓?forward 杩斿洖缁欏灞傝缁?璇勪及鐨勨€滃綋鍓?prefix 涓婚娴嬧€濄€?        # 澶栧眰 progressive 閫昏緫浼氭妸涓嶅悓 prefix 鐨?logits 缁勬垚鏃堕棿搴忓垪锛屽啀鍋氬湪绾挎棭鍋滆瘎浼般€?
        logits = stage_logits[batch_index, prediction_stage]

        # aux 淇濆瓨璁粌鎹熷け銆佽瘖鏂€佸彲瑙嗗寲鍜屽湪绾挎棭鍋滄墍闇€鐨勪腑闂撮噺銆?        # 杩欎簺瀛楁澶у涓嶄細鐩存帴浣滀负鏈€缁堝垎绫荤粨鏋滐紝浣嗕細鐢ㄤ簬鐩戠潱鍚勯樁娈点€佽绠楁垚鐔熷害銆佽褰?attention 绛夈€?
        aux = {
            # 琚€変腑鐨勪富棰勬祴闃舵琛ㄧず銆?
            "fused_repr": stage_repr[batch_index, prediction_stage],
            # 涓変釜闃舵鐨勮瘉鎹〃绀猴紝鏂逛究 loss銆佸垎鏋愬拰鍙鍖栥€?
            "packet_repr": stage_repr[:, 0],
            "burst_repr": stage_repr[:, 1],
            "flow_repr": stage_repr[:, 2],
            # 涓変釜闃舵鍚勮嚜鐨勫垎绫?logits銆?
            "stage_logits": stage_logits,
            "shared_grain_logits": stage_logits,
            "stage_confidence": stage_confidence,
            # 鎴愮啛搴︾浉鍏宠緭鍑猴細缁撴瀯鎴愮啛搴︿箻 JSD gate 鍚庡緱鍒?prefix_calibrated_availability銆?
            "prefix_calibrated_availability": stage_calibrated_availability,
            "stage_availability_mask": stage_available,
            # stage_supervision_mask 鎺у埗璁粌鏃跺摢浜涢樁娈靛簲褰撹鐩戠潱锛屼互鍙?JSD prior 瀵圭洃鐫ｅ己搴︾殑璋冭妭銆?
            "stage_supervision_mask": stage_available.to(stage_logits.dtype),
            # 鏃╁仠鐩稿叧鍒嗘暟锛歴afe_stop 鏄綉缁滃ご杈撳嚭锛宔xit_scores 鏄彔鍔?availability 鍜?JSD gate 鍚庣殑鏈€缁堟棭鍋滃垎鏁般€?
            "stage_safe_stop_logits": stage_safe_stop_logits,
            "stage_safe_stop_scores": stage_safe_stop,
            "stage_exit_scores": exit_scores,
            "exit_score": selected_exit_score,
            "prediction_stage": prediction_stage,
            "fusion_gates": route_weights,
            # 涓嬮潰杩欎簺鏉冮噸涓昏鐢ㄤ簬鍏煎鏃ц缁?鍒嗘瀽浠ｇ爜锛岃〃绀哄綋鍓嶉樁娈甸€夋嫨鎴栧彲闈犳€у垎甯冦€?
            "predicted_reliability_weights": torch.softmax(exit_scores, dim=-1),
            "task_fusion_weights": torch.nn.functional.one_hot(prediction_stage, num_classes=3).to(stage_logits.dtype),
            "availability_reliability": stage_calibrated_availability,
            "calibrated_availability": calibrated_availability,
            "granularity_maturity_prior": jsd_prior,
            "granularity_maturity_gate": jsd_gate,
            # attention銆佽竟鐣屻€乤ssignment 绛変腑闂撮噺锛岀敤浜庤В閲?packet/burst/flow 鍐呴儴璇佹嵁鏉ユ簮銆?
            "packet_attention": packet["attention"],
            "burst_attention": burst["attention"],
            "packet_burst_assignment": burst["assignment"],
            "learned_boundary_prob": burst["boundary_prob"],
            "boundary_valid": burst["boundary_valid"],
            "boundary_prior": burst["boundary_prob"].detach(),
            "raw_burst_mask": burst["raw_burst_mask"],
            "burst_evidence_gate": burst["burst_evidence_gate"],
            "packet_to_burst_attention": burst["packet_to_burst_attention"],
            "packet_upgrade_gate": burst["packet_upgrade_gate"],
            "burst_residual_alpha": burst_alpha,
            "burst_to_flow_attention": flow["burst_to_flow_attention"],
            "flow_upgrade_gate": flow["flow_upgrade_gate"],
            "flow_residual_alpha": flow_alpha,
            # 褰撳墠 prefix 瀹為檯鍙鍖呮暟锛屽悗缁湪绾胯瘎浼板拰璇婃柇浼氱敤鍒般€?
            "observed_packet_count": observed_packet_count,
            # 涓嬮潰瀛楁鏄负浜嗗吋瀹规洿鏃╃増鏈殑澶氱矑搴﹀彲瑙嗗寲鎺ュ彛锛屾湰妯″瀷褰撳墠娌℃湁浣跨敤杩欎簺 assignment銆?
            "byte_field_assignment": None,
            "byte_field_mask": None,
            "packet_role_assignment": None,
            "packet_role_mask": None,
            "flow_state_assignment": None,
            "flow_state_mask": None,
        }
        return logits, aux

    def _lookup_jsd_prior(
        self,
        observed_packet_count: torch.Tensor,
        prefix_budget: int | torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_granularity_maturity_prior:
            return torch.ones(observed_packet_count.size(0), 3, dtype=torch.float32, device=observed_packet_count.device)
        if prefix_budget is None:
            budget_idx = observed_packet_count.long()
        elif isinstance(prefix_budget, torch.Tensor):
            budget_idx = prefix_budget.to(observed_packet_count.device).long().reshape(-1)
            if budget_idx.numel() == 1:
                budget_idx = budget_idx.expand_as(observed_packet_count.long())
        else:
            budget_idx = torch.full_like(observed_packet_count.long(), int(prefix_budget))
        budget_idx = budget_idx.clamp(0, PACKET_NODES)
        return self.granularity_maturity_prior.to(observed_packet_count.device)[budget_idx].to(torch.float32)

    def _jsd_gate(self, jsd_prior: torch.Tensor) -> torch.Tensor:
        if not self.use_granularity_maturity_prior:
            return torch.ones_like(jsd_prior)
        return self.granularity_maturity_prior_floor + (1.0 - self.granularity_maturity_prior_floor) * jsd_prior.clamp(0.0, 1.0)

    def _apply_jsd_structure_gate(self, maturity: torch.Tensor, jsd_gate: torch.Tensor) -> torch.Tensor:
        if not self.use_granularity_maturity_prior or self.granularity_maturity_prior_blend <= 0.0:
            return maturity.clamp(0.0, 1.0)
        gate = 1.0 - self.granularity_maturity_prior_blend + self.granularity_maturity_prior_blend * jsd_gate.to(maturity.dtype)
        return (maturity * gate).clamp(0.0, 1.0)

    def _select_stages(
        self,
        exit_scores: torch.Tensor,
        stage_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the prediction stage and diagnostic stop/continue route."""
        # 杩欎釜鍑芥暟鍙仛鈥滃崟涓?prefix 鍐呴儴鈥濈殑闃舵閫夋嫨鍜屽畨鍏ㄥ仠姝㈠垽鏂€?        # 杈撳叆褰㈢姸锛?        #   exit_scores:     [batch, 3]锛屼笁涓樁娈电殑 early-exit 鍒嗘暟锛?=packet锛?=burst锛?=flow銆?        #   stage_available: [batch, 3]锛屼笁涓樁娈靛綋鍓嶆槸鍚︾湡鐨勫彲鐢ㄣ€?        # 杈撳嚭鍚箟锛?        #   prediction_stage: 褰撳墠 prefix 鐢ㄥ摢涓樁娈电殑鍒嗙被 logits 鍋氫富棰勬祴銆?        #   route_weights:    one-hot stop/continue route for logging and diagnostics.
        #   selected_score:   琚?prediction_stage 閫変腑闃舵瀵瑰簲鐨?early-exit 鍒嗘暟銆?        #
        # The local route is derived from prediction_stage; it does not choose
        # another classifier stage.
        thresholds = exit_scores.new_tensor(
            [self.packet_exit_threshold, self.burst_exit_threshold, self.flow_exit_threshold]
        ).view(1, 3)

        # 榛樿涓婚娴嬩娇鐢?packet 闃舵銆?        # packet 鏄渶鏃╁彲鐢ㄧ殑灞€閮ㄨ瘉鎹紝鍥犳浣滀负榛樿棰勬祴闃舵銆?
        fallback_stage = 0
        if not self.use_packet_granularity:
            fallback_stage = 1 if self.use_burst_granularity else 2
        prediction_stage = torch.full(
            (exit_scores.size(0),),
            fallback_stage,
            dtype=torch.long,
            device=exit_scores.device,
        )

        # 濡傛灉 burst 璇佹嵁鍙敤锛屽氨鎶婁富棰勬祴闃舵浠?packet 鍗囩骇鍒?burst銆?        # 杩欓噷涓嶆槸鐪?burst 鐨?confidence 鎴?maturity 璋佹洿澶э紝鑰屾槸閬靛惊 evidence cascade锛?        # 褰撳墠 prefix 宸茬粡褰㈡垚 burst 璇佹嵁鏃讹紝浣跨敤鏇撮珮绮掑害鐨勬畫宸骇鑱旇〃绀恒€?
        if self.use_burst_granularity:
            prediction_stage = torch.where(
                stage_available[:, 1],
                torch.ones_like(prediction_stage),
                prediction_stage,
            )

        # 濡傛灉 flow 璇佹嵁鍙敤锛屽氨缁х画鎶婁富棰勬祴闃舵鍗囩骇鍒?flow銆?        # 鎵€浠?CUT_JSD_ISCXVPN_7_2 鐨勪富棰勬祴瑙勫垯鏄€滄渶楂樺彲鐢ㄧ矑搴︿紭鍏堚€濓細
        # packet -> burst -> flow銆?        # 杩欏拰 JSD_ISCXVPN_7_6 閲屽皾璇曡繃鐨?maturity-routed prediction 涓嶅悓锛?        # 杩欓噷涓嶄細鐢?maturity * confidence 鍘绘姠涓婚娴嬫潈銆?
        if self.use_flow_granularity:
            prediction_stage = torch.where(
                stage_available[:, 2],
                prediction_stage.new_full(prediction_stage.shape, 2),
                prediction_stage,
            )

        # row_idx 鐢ㄦ潵閫愭牱鏈粠 [batch, 3] 涓彇鍑?prediction_stage 瀵瑰簲鐨勫垎鏁般€?
        row_idx = torch.arange(exit_scores.size(0), device=exit_scores.device)

        #  鍏堢湅褰撳墠鏍锋湰 prediction_stage 鏄摢涓樁娈碉紱
        #  鍐嶅彇杩欎釜闃舵瀵瑰簲鐨?exit_score锛?        #  鍐嶅彇杩欎釜闃舵瀵瑰簲鐨?threshold锛?        #  濡傛灉 exit_score >= threshold锛屽氨鍏佽褰撳墠prefix 鏃╁仠



        # selected_score 鏄€滃綋鍓嶄富棰勬祴闃舵鈥濈殑 early-exit 鍒嗘暟銆?        # 涓句緥锛氭煇涓牱鏈?prediction_stage=1锛屽氨鍙彇 exit_scores[:, 1] 鏉ュ垽鏂?burst 褰撳墠鑳戒笉鑳藉畨鍏ㄥ仠銆?
        selected_score = exit_scores[row_idx, prediction_stage]

        # 姣忎釜闃舵鍙互鏈夎嚜宸辩殑鍋滄闃堝€硷細
        # packet_exit_threshold銆乥urst_exit_threshold銆乫low_exit_threshold銆?        # selected_threshold 鍜?selected_score 瀵归綈锛岄兘鏄綋鍓嶄富棰勬祴闃舵瀵瑰簲鐨勫€笺€?
        selected_threshold = thresholds.expand_as(exit_scores)[row_idx, prediction_stage]

        # current_safe 琛ㄧず鈥滃綋鍓嶄富棰勬祴闃舵鏄惁瓒冲瀹夊叏锛屽彲浠ュ湪杩欎釜 prefix 鍋滄鈥濄€?        # 瀹冨彧鍥炵瓟鑳戒笉鑳藉仠锛屼笉鏀瑰彉褰撳墠 prefix 鐢ㄥ摢涓樁娈靛垎绫汇€?        #姣旇緝
        current_safe = selected_score.ge(selected_threshold)

        # 3 鏄汉涓哄畾涔夌殑 continue 鏍囪锛屼笉浠ｈ〃绗?4 涓瘉鎹樁娈点€?        # 濡傛灉 current_safe=False锛岃鏄庡綋鍓?prefix 鏈夐娴嬶紝浣嗕笉鑳芥棭鍋滐紝瑕佺户缁湅涓嬩竴涓?prefix銆?
        continue_stage = torch.full((exit_scores.size(0),), 3, dtype=torch.long, device=exit_scores.device)

        # Internal route: 0/1/2 means the selected stage is locally safe,
        # 3 means this prefix should continue.
        exit_stage = torch.where(current_safe, prediction_stage, continue_stage)

        # route_weights 鏄?[batch, 4]锛?        #   [:,0] packet 鍋?        #   [:,1] burst 鍋?        #   [:,2] flow 鍋?        #   [:,3] continue
        # 瀹冩洿澶氭槸璇婃柇閲忥紝涓嶆槸閲嶆柊铻嶅悎涓変釜鍒嗙被缁撴灉鐨勬潈閲嶃€?
        route_weights = torch.nn.functional.one_hot(exit_stage, num_classes=4).to(exit_scores.dtype)
        return prediction_stage, route_weights, selected_score

    def _burst_available(
        self,
        packet_x: torch.Tensor,
        burst_x: torch.Tensor,
        packet_mask: torch.Tensor,
        observed_packet_count: torch.Tensor,
    ) -> torch.Tensor:
        observed_ready = observed_packet_count.to(packet_x.dtype).ge(float(self.burst_stage.burst_center))
        valid_pairs = packet_mask[:, :-1] & packet_mask[:, 1:]
        if packet_x.size(-1) > 2:
            direction_switch = (packet_x[:, 1:, 2] - packet_x[:, :-1, 2]).abs().gt(0.5) & valid_pairs
        else:
            direction_switch = valid_pairs.new_zeros(valid_pairs.shape)
        switch_ready = direction_switch.sum(dim=-1).ge(1)
        raw_burst_count = self._align_burst_for_mask(burst_x).abs().sum(dim=-1).gt(0).sum(dim=-1)
        burst_ready = raw_burst_count.ge(2) | packet_mask.sum(dim=-1).ge(2)
        return observed_ready & switch_ready & burst_ready

    def _flow_available(
        self,
        flow_x: torch.Tensor,
        observed_packet_count: torch.Tensor,
        burst_available: torch.Tensor,
    ) -> torch.Tensor:
        """Enable Flow only after Burst exists and enough causal packets arrived."""
        observed_ready = observed_packet_count.to(flow_x.dtype).ge(float(self.flow_stage.flow_center))
        flow_present = flow_x.abs().sum(dim=-1).gt(0)
        return burst_available.bool() & observed_ready & flow_present

    @staticmethod
    def _align_burst_for_mask(burst_x: torch.Tensor) -> torch.Tensor:
        if burst_x.size(1) == BURST_NODES:
            return burst_x
        if burst_x.size(1) > BURST_NODES:
            return burst_x[:, :BURST_NODES]
        pad_shape = (burst_x.size(0), BURST_NODES - burst_x.size(1), burst_x.size(2))
        return torch.cat([burst_x, burst_x.new_zeros(pad_shape)], dim=1)

    def _empty_burst(
        self,
        packet_nodes: torch.Tensor,
        packet_x: torch.Tensor,
        burst_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = packet_nodes.size(0)
        hidden_dim = packet_nodes.size(-1)
        burst_nodes = packet_nodes.new_zeros(batch_size, BURST_NODES, hidden_dim)
        burst_mask = torch.zeros(batch_size, BURST_NODES, dtype=torch.bool, device=packet_nodes.device)
        boundary_shape = (batch_size, max(packet_x.size(1) - 1, 1))
        return {
            "nodes": burst_nodes,
            "mask": burst_mask,
            "repr": packet_nodes.new_zeros(batch_size, hidden_dim),
            "attention": packet_nodes.new_zeros(batch_size, BURST_NODES),
            "maturity": packet_nodes.new_zeros(batch_size),
            "assignment": packet_nodes.new_zeros(batch_size, BURST_NODES, packet_x.size(1)),
            "boundary_prob": packet_nodes.new_zeros(boundary_shape),
            "boundary_valid": torch.zeros(boundary_shape, dtype=torch.bool, device=packet_nodes.device),
            "direction_switch_count": packet_nodes.new_zeros(batch_size),
            "raw_burst_mask": torch.zeros(batch_size, self._align_burst_for_mask(burst_x).size(1), dtype=torch.bool, device=packet_nodes.device),
            "burst_evidence_gate": packet_nodes.new_zeros(batch_size, BURST_NODES),
            "packet_to_burst_attention": packet_nodes.new_zeros(batch_size, BURST_NODES, packet_x.size(1)),
            "packet_upgrade_gate": packet_nodes.new_zeros(batch_size, BURST_NODES),
        }

    def _empty_flow(self, packet_nodes: torch.Tensor, burst_x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = packet_nodes.size(0)
        hidden_dim = packet_nodes.size(-1)
        burst_width = self._align_burst_for_mask(burst_x).size(1)
        return {
            "repr": packet_nodes.new_zeros(batch_size, hidden_dim),
            "maturity": packet_nodes.new_zeros(batch_size),
            "burst_to_flow_attention": packet_nodes.new_zeros(batch_size, burst_width),
            "flow_upgrade_gate": packet_nodes.new_zeros(batch_size),
        }


class EvidenceMaturityCascadedClassifier(nn.Module):
    """Compatibility wrapper used by build_base_model()."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = EMCCBackbone(**kwargs)

    @property
    def node_builder(self):
        return self.backbone.node_builder

    def set_granularity_maturity_prior(self, prefix_budgets, prior) -> None:
        self.backbone.set_granularity_maturity_prior(prefix_budgets, prior)

    def forward(
        self,
        byte_x,
        packet_x,
        flow_x,
        burst_x,
        observed_packet_count=None,
        prefix_budget=None,
        byte_header_payload=None,
    ):
        return self.backbone(
            byte_x,
            packet_x,
            flow_x,
            burst_x,
            observed_packet_count,
            prefix_budget,
            byte_header_payload=byte_header_payload,
        )



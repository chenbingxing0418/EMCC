"""Packet-level feature encoding and relation construction."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.model_constants import PACKET_NODES


def byte_field_ids(byte_head: int) -> torch.Tensor:
    """Map fixed IP-packet byte positions to coarse protocol fields."""

    ids = torch.full((int(byte_head),), 14, dtype=torch.long)
    ranges = (
        (0, 1, 1),    # IP version/IHL
        (1, 2, 2),    # IP TOS
        (2, 4, 3),    # IP total length
        (4, 6, 4),    # IP identification
        (6, 8, 5),    # IP flags/fragment offset
        (8, 9, 6),    # TTL
        (9, 10, 7),   # protocol
        (10, 12, 8),  # checksum
        (12, 20, 9),  # source/destination IP, usually masked by preprocessing
        (20, 24, 10), # TCP/UDP ports
        (24, 28, 11), # TCP sequence or UDP length/checksum area
        (28, 32, 12), # TCP acknowledgement or UDP-adjacent area
        (32, 40, 13), # TCP flags/window/checksum/urgent area
    )
    for start, end, value in ranges:
        if start >= byte_head:
            continue
        ids[start:min(end, byte_head)] = value
    return ids


def byte_region_ids(byte_head: int) -> torch.Tensor:
    """Map byte positions to IP, transport, or payload regions."""

    ids = torch.full((int(byte_head),), 3, dtype=torch.long)
    ids[:min(20, byte_head)] = 1
    if byte_head > 20:
        ids[20:min(40, byte_head)] = 2
    return ids


class PacketByteEncoder(nn.Module):
    """Structure-aware Transformer byte encoder for ablation.

    The input bytes come from bytes(pkt[IP]), so fixed byte positions have
    protocol meaning.  This encoder keeps one compact byte evidence path:
    value, position, coarse field, and coarse region embeddings are processed
    by a shallow Transformer and summarized by a CLS token.
    """

    def __init__(
        self,
        byte_head: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        transformer_layers: int,
        mask_mode: str,
    ):
        super().__init__()
        self.byte_head = int(byte_head)
        self.mask_mode = str(mask_mode)
        self.value_embed = nn.Embedding(257, hidden_dim, padding_idx=0)
        self.position_embed = nn.Embedding(self.byte_head, hidden_dim)
        self.field_embed = nn.Embedding(15, hidden_dim)
        self.region_embed = nn.Embedding(4, hidden_dim)
        self.header_payload_embed = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.header_payload_scale = nn.Parameter(torch.tensor(0.0))
        self.value_stem = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, int(num_heads)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(transformer_layers)),
            enable_nested_tensor=False,
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.register_buffer("field_ids", byte_field_ids(self.byte_head), persistent=False)
        self.register_buffer("region_ids", byte_region_ids(self.byte_head), persistent=False)

    def forward(self, byte_x: torch.Tensor, byte_header_payload: torch.Tensor | None = None) -> torch.Tensor:
        bsz, pkt_num, byte_dim = byte_x.shape
        if byte_dim > self.byte_head:
            raise ValueError(f"byte_x dim {byte_dim} exceeds configured byte_head {self.byte_head}")

        flat = self.apply_mask(byte_x).reshape(bsz * pkt_num, byte_dim)
        flat_segments = self.prepare_header_payload(byte_header_payload, byte_x).reshape(bsz * pkt_num, byte_dim)
        packet_mask = flat.abs().sum(dim=-1).gt(0)

        token_ids = (flat * 255.0).round().clamp(0, 255).long() + 1
        token_ids = token_ids.masked_fill(~packet_mask.unsqueeze(-1), 0)
        segment_ids = flat_segments.clamp(0, 2).long()
        segment_ids = segment_ids.masked_fill(~packet_mask.unsqueeze(-1), 0)

        pos = torch.arange(byte_dim, dtype=torch.long, device=byte_x.device)
        field_ids = self.field_ids[:byte_dim].to(byte_x.device)
        region_ids = self.region_ids[:byte_dim].to(byte_x.device)
        byte_h = (
            self.value_embed(token_ids)
            + self.position_embed(pos).unsqueeze(0)
            + self.field_embed(field_ids).unsqueeze(0)
            + self.region_embed(region_ids).unsqueeze(0)
            + self.header_payload_scale.to(byte_x.dtype) * self.header_payload_embed(segment_ids)
            + self.value_stem(flat.unsqueeze(-1))
        )

        cls = self.cls_token.expand(byte_h.size(0), -1, -1)
        h = torch.cat([cls, byte_h], dim=1)
        byte_valid = packet_mask.unsqueeze(-1).expand(-1, byte_dim)
        key_padding_mask = torch.cat(
            [
                torch.zeros(byte_h.size(0), 1, dtype=torch.bool, device=byte_x.device),
                ~byte_valid,
            ],
            dim=1,
        )
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.out(h[:, 0]).reshape(bsz, pkt_num, -1)

    def apply_mask(self, byte_x: torch.Tensor) -> torch.Tensor:
        mode = self.mask_mode.lower()
        if mode == "raw":
            return byte_x
        masked = byte_x.clone()
        if mode == "mask_ip_port":
            masked[..., 12:min(20, masked.size(-1))] = 0
            masked[..., 20:min(24, masked.size(-1))] = 0
            return masked
        if mode == "payload_only":
            masked[..., :min(40, masked.size(-1))] = 0
            return masked
        raise ValueError(f"unsupported byte mask mode: {self.mask_mode}")

    def prepare_header_payload(
        self,
        byte_header_payload: torch.Tensor | None,
        byte_x: torch.Tensor,
    ) -> torch.Tensor:
        if byte_header_payload is None:
            return torch.zeros_like(byte_x, dtype=torch.long)
        return byte_header_payload[..., : byte_x.size(-1)].to(device=byte_x.device, dtype=torch.long)


class ByteTCNBlock(nn.Module):
    """Depthwise dilated temporal convolution over ordered byte positions."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_dim,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(residual + h)


class PacketByteTCNEncoder(nn.Module):
    """Structure-aware TCN byte encoder for compact local byte evidence."""

    def __init__(
        self,
        byte_head: int,
        hidden_dim: int,
        dropout: float,
        tcn_layers: int,
        kernel_size: int,
        mask_mode: str,
    ):
        super().__init__()
        self.byte_head = int(byte_head)
        self.mask_mode = str(mask_mode)
        self.value_embed = nn.Embedding(257, hidden_dim, padding_idx=0)
        self.position_embed = nn.Embedding(self.byte_head, hidden_dim)
        self.field_embed = nn.Embedding(15, hidden_dim)
        self.region_embed = nn.Embedding(4, hidden_dim)
        self.header_payload_embed = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.header_payload_scale = nn.Parameter(torch.tensor(0.0))
        self.value_stem = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        dilations = [2 ** (idx % 4) for idx in range(max(1, int(tcn_layers)))]
        self.tcn = nn.ModuleList(
            [
                ByteTCNBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=max(3, int(kernel_size)) | 1,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.pool_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.register_buffer("field_ids", byte_field_ids(self.byte_head), persistent=False)
        self.register_buffer("region_ids", byte_region_ids(self.byte_head), persistent=False)

    def forward(self, byte_x: torch.Tensor, byte_header_payload: torch.Tensor | None = None) -> torch.Tensor:
        bsz, pkt_num, byte_dim = byte_x.shape
        if byte_dim > self.byte_head:
            raise ValueError(f"byte_x dim {byte_dim} exceeds configured byte_head {self.byte_head}")

        flat = self.apply_mask(byte_x).reshape(bsz * pkt_num, byte_dim)
        flat_segments = self.prepare_header_payload(byte_header_payload, byte_x).reshape(bsz * pkt_num, byte_dim)
        packet_mask = flat.abs().sum(dim=-1).gt(0)

        token_ids = (flat * 255.0).round().clamp(0, 255).long() + 1
        token_ids = token_ids.masked_fill(~packet_mask.unsqueeze(-1), 0)
        segment_ids = flat_segments.clamp(0, 2).long()
        segment_ids = segment_ids.masked_fill(~packet_mask.unsqueeze(-1), 0)

        pos = torch.arange(byte_dim, dtype=torch.long, device=byte_x.device)
        field_ids = self.field_ids[:byte_dim].to(byte_x.device)
        region_ids = self.region_ids[:byte_dim].to(byte_x.device)
        h = (
            self.value_embed(token_ids)
            + self.position_embed(pos).unsqueeze(0)
            + self.field_embed(field_ids).unsqueeze(0)
            + self.region_embed(region_ids).unsqueeze(0)
            + self.header_payload_scale.to(byte_x.dtype) * self.header_payload_embed(segment_ids)
            + self.value_stem(flat.unsqueeze(-1))
        )

        byte_valid = packet_mask.unsqueeze(-1).expand(-1, byte_dim)
        h = h * byte_valid.unsqueeze(-1).to(h.dtype)
        for block in self.tcn:
            h = block(h)
            h = h * byte_valid.unsqueeze(-1).to(h.dtype)

        safe_valid = byte_valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any():
            safe_valid[empty, 0] = True
        scores = self.pool_score(h).squeeze(-1).masked_fill(~safe_valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * byte_valid.to(h.dtype)
        pooled = torch.einsum("bl,blh->bh", weights, h)
        pooled = pooled * packet_mask.unsqueeze(-1).to(pooled.dtype)
        return self.out(pooled).reshape(bsz, pkt_num, -1)

    def apply_mask(self, byte_x: torch.Tensor) -> torch.Tensor:
        mode = self.mask_mode.lower()
        if mode == "raw":
            return byte_x
        masked = byte_x.clone()
        if mode == "mask_ip_port":
            masked[..., 12:min(20, masked.size(-1))] = 0
            masked[..., 20:min(24, masked.size(-1))] = 0
            return masked
        if mode == "payload_only":
            masked[..., :min(40, masked.size(-1))] = 0
            return masked
        raise ValueError(f"unsupported byte mask mode: {self.mask_mode}")

    def prepare_header_payload(
        self,
        byte_header_payload: torch.Tensor | None,
        byte_x: torch.Tensor,
    ) -> torch.Tensor:
        if byte_header_payload is None:
            return torch.zeros_like(byte_x, dtype=torch.long)
        return byte_header_payload[..., : byte_x.size(-1)].to(device=byte_x.device, dtype=torch.long)


class PacketNodeBuilder(nn.Module):
    """Build packet graph nodes from raw bytes and packet statistics."""

    def __init__(
        self,
        byte_head: int,
        packet_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        byte_encoder_type: str,
        byte_transformer_layers: int,
        byte_tcn_layers: int,
        byte_tcn_kernel_size: int,
        byte_mask_mode: str,
    ):
        super().__init__()
        encoder_type = str(byte_encoder_type).lower()
        if encoder_type == "transformer":
            self.byte_encoder = PacketByteEncoder(
                byte_head=byte_head,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_heads=num_heads,
                transformer_layers=byte_transformer_layers,
                mask_mode=byte_mask_mode,
            )
        elif encoder_type == "tcn":
            self.byte_encoder = PacketByteTCNEncoder(
                byte_head=byte_head,
                hidden_dim=hidden_dim,
                dropout=dropout,
                tcn_layers=byte_tcn_layers,
                kernel_size=byte_tcn_kernel_size,
                mask_mode=byte_mask_mode,
            )
        else:
            raise ValueError(f"unsupported byte encoder type: {byte_encoder_type}")
        self.packet_proj = nn.Sequential(
            nn.Linear(hidden_dim + packet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.position_embed = nn.Embedding(PACKET_NODES, hidden_dim)

    def forward(
        self,
        byte_x: torch.Tensor,
        packet_x: torch.Tensor,
        byte_header_payload: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        byte_repr = self.byte_encoder(byte_x, byte_header_payload)
        packet_nodes = self.packet_proj(torch.cat([byte_repr, packet_x], dim=-1))
        pos = torch.arange(PACKET_NODES, dtype=torch.long, device=packet_nodes.device)
        packet_nodes = packet_nodes + self.position_embed(pos).unsqueeze(0)
        packet_mask = packet_x.abs().sum(dim=-1) > 0
        return packet_nodes * packet_mask.unsqueeze(-1).to(packet_nodes.dtype), packet_mask


def build_packet_relation_masks(packet_mask: torch.Tensor) -> torch.Tensor:
    """Create self, adjacent, and skip-edge packet relations."""

    device = packet_mask.device
    bsz = packet_mask.size(0)
    rel = torch.zeros(bsz, 3, PACKET_NODES, PACKET_NODES, dtype=torch.bool, device=device)
    valid_pair = packet_mask.unsqueeze(1) & packet_mask.unsqueeze(2)

    eye = torch.eye(PACKET_NODES, dtype=torch.bool, device=device).unsqueeze(0)
    rel[:, 0] = eye & valid_pair

    for i in range(PACKET_NODES - 1):
        rel[:, 1, i, i + 1] = True
        rel[:, 1, i + 1, i] = True
    rel[:, 1] &= valid_pair

    for step in (2, 4):
        for i in range(PACKET_NODES - step):
            rel[:, 2, i, i + step] = True
            rel[:, 2, i + step, i] = True
    rel[:, 2] &= valid_pair
    return rel

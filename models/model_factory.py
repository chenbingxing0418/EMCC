"""Build the EMCC model."""

from __future__ import annotations

import torch

from models.emcc import EvidenceMaturityCascadedClassifier


def build_base_model(args, train_dataset, train_labels, num_classes: int, device: torch.device):
    del train_labels
    model = EvidenceMaturityCascadedClassifier(
        num_classes=num_classes,
        byte_head=train_dataset.byte_head,
        packet_dim=int(train_dataset.packet_x.shape[-1]),
        burst_dim=int(train_dataset.burst_x.shape[-1]),
        flow_dim=int(train_dataset.flow_x.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        byte_encoder_type=args.byte_encoder_type,
        byte_transformer_layers=args.byte_transformer_layers,
        byte_tcn_layers=args.byte_tcn_layers,
        byte_tcn_kernel_size=args.byte_tcn_kernel_size,
        byte_mask_mode=args.byte_mask_mode,
        classifier_type=args.classifier_type,
        prototypes_per_class=args.prototypes_per_class,
        prototype_temperature=args.prototype_temperature,
        use_packet_gnn=args.use_packet_gnn,
        use_burst_gnn=args.use_burst_gnn,
        use_burst_cross_attention=args.use_burst_cross_attention,
        use_flow_cross_attention=args.use_flow_cross_attention,
        use_packet_granularity=args.use_packet_granularity,
        use_burst_granularity=args.use_burst_granularity,
        use_flow_granularity=args.use_flow_granularity,
        burst_center=args.prefix_burst_center,
        flow_center=args.prefix_flow_center,
        availability_temperature=args.emcc_availability_temperature,
        packet_exit_threshold=args.emcc_packet_exit_threshold,
        burst_exit_threshold=args.emcc_burst_exit_threshold,
        use_granularity_maturity_prior=args.use_granularity_maturity_prior,
        granularity_maturity_prior_ema_alpha=args.granularity_maturity_prior_ema_alpha,
        granularity_maturity_prior_floor=args.granularity_maturity_prior_floor,
        granularity_maturity_prior_blend=args.granularity_maturity_prior_blend,
        burst_evidence_availability_threshold=args.burst_evidence_availability_threshold,
        flow_evidence_availability_threshold=args.flow_evidence_availability_threshold,
    ).to(device)
    return model


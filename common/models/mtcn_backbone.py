"""
多模态时序卷积网络(Enhanced with Hierarchical Transformer)

基于AdoDAS 2026技术指南增强的backbone:
- 保留原有TCN架构作为备选
- 新增层级化Transformer编码器（MBT瓶颈融合）
- 支持注意力统计池化（均值+标准差）
- 支持灵活切换架构模式
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_encoder import (
    HierarchicalEncoderConfig,
    HierarchicalCrossModalEncoder,
    SessionAttentionAggregator,
    TemporalDownsampler,
    AttentiveStatisticsPooling,
)


@dataclass
class BackboneConfig:
    audio_group_dims: dict[str, int] = field(default_factory=dict)
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict)
    video_group_dims: dict[str, int] = field(default_factory=dict)

    d_adapter: int = 64
    d_model: int = 512
    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    asp_alpha: float = 0.5
    asp_beta: float = 0.5
    dropout: float = 0.1

    n_sessions: int = 4
    d_session: int = 16
    d_shared: int = 512

    use_hierarchical_encoder: bool = True
    n_unimodal_layers: int = 4
    n_fusion_layers: int = 2
    n_bottleneck_tokens: int = 4
    n_perceiver_queries: int = 32
    temporal_stride: int = 4


class GroupAdapter(nn.Module):
    """特征组适配器"""
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.proj(self.norm(x))))


class ModalityFusion(nn.Module):
    """模态融合模块"""
    def __init__(self, n_groups: int, d_adapter: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_groups * d_adapter, d_model)

    def forward(self, groups: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(groups, dim=-1))


class DilatedResidualBlock(nn.Module):
    """扩张残差块（TCN组件）"""
    def __init__(
        self, d_model: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        T = x.size(1)

        h = self.norm1(x)
        h = h.transpose(1, 2)
        h = self.conv1(h)[:, :, :T]
        h = F.gelu(h)
        h = self.drop(h)

        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)[:, :, :T]
        h = self.drop(h)
        h = h.transpose(1, 2)

        out = h + residual
        out = out * mask.unsqueeze(-1).float()
        return out


class TCN(nn.Module):
    """时序卷积网络"""
    def __init__(
        self, d_model: int, n_layers: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class ASP(nn.Module):
    """Attentive Statistics Pooling with VAD and quality control signals."""
    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor | None = None,
        qc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) bool
        vad  : (B, T) float
        qc   : (B, T) float
        Returns: (B, 2*D)
        """
        e = self.attn(x).squeeze(-1)
        e = e + self.alpha * vad + self.beta * qc

        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        w = w.masked_fill(~mask, 0.0)

        w_unsq = w.unsqueeze(-1)
        mean = (w_unsq * x).sum(dim=1)

        diff = x - mean.unsqueeze(1)
        var = (w_unsq * diff ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-8))

        return torch.cat([mean, std], dim=-1)


class HierarchicalMTCNBackbone(nn.Module):
    """
    层级化MTCN Backbone

    当 use_hierarchical_encoder=True 时使用层级化Transformer架构:
    - 时序降采样
    - 单模态Transformer编码
    - MBT瓶颈融合
    - Perceiver压缩
    - 注意力统计池化
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_hierarchical = cfg.use_hierarchical_encoder

        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, cfg.d_adapter),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })

        self.audio_group_names = sorted(cfg.audio_group_dims.keys())
        self.audio_pooled_group_names = sorted(cfg.audio_pooled_group_dims.keys())
        self.video_group_names = sorted(cfg.video_group_dims.keys())

        if self.use_hierarchical:
            hier_cfg = HierarchicalEncoderConfig(
                d_model=cfg.d_model,
                n_heads=8,
                dim_feedforward=cfg.d_model * 4,
                n_unimodal_layers=cfg.n_unimodal_layers,
                n_fusion_layers=cfg.n_fusion_layers,
                n_bottleneck_tokens=cfg.n_bottleneck_tokens,
                n_perceiver_queries=cfg.n_perceiver_queries,
                dropout=cfg.dropout,
                temporal_stride=cfg.temporal_stride,
            )
            self.hierarchical_encoder = HierarchicalCrossModalEncoder(hier_cfg)
            pooled_d = cfg.d_model * 2
        else:
            self.audio_fusion = ModalityFusion(
                len(self.audio_group_names), cfg.d_adapter, cfg.d_model
            )
            self.video_fusion = ModalityFusion(
                len(self.video_group_names), cfg.d_adapter, cfg.d_model
            )
            self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
            self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
            self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
            self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
            pooled_d = cfg.d_model * 2

        fusion_in = pooled_d * 2
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session

        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio_adapted = [
            self.audio_adapters[n](batch["audio_groups"][n])
            for n in self.audio_group_names
        ]
        video_adapted = [
            self.video_adapters[n](batch["video_groups"][n])
            for n in self.video_group_names
        ]

        mask_a = batch["mask_audio"]
        mask_v = batch["mask_video"]
        vad = batch.get("vad_signal")
        qc = batch.get("qc_quality")

        if self.use_hierarchical:
            z_a, z_v = self._forward_hierarchical(
                audio_adapted, video_adapted, mask_a, mask_v, vad, qc
            )
        else:
            z_a, z_v = self._forward_tcn(
                audio_adapted, video_adapted, mask_a, mask_v, vad, qc
            )

        parts = [z_a, z_v]
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        parts.append(self.session_embed(batch["session_idx"]))

        z = torch.cat(parts, dim=-1)
        return self.fusion_mlp(z)

    def _forward_hierarchical(
        self,
        audio_adapted: list[torch.Tensor],
        video_adapted: list[torch.Tensor],
        mask_a: torch.Tensor,
        mask_v: torch.Tensor,
        vad: torch.Tensor | None,
        qc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """层级化Transformer前向"""
        audio = self.audio_fusion(audio_adapted)
        video = self.video_fusion(video_adapted)

        audio = audio * mask_a.unsqueeze(-1).float()
        video = video * mask_v.unsqueeze(-1).float()

        hier_out = self.hierarchical_encoder(
            audio, video, mask_a, mask_v, vad, qc
        )

        z_a = hier_out["pooled"][:, :self.cfg.d_model]
        z_v = hier_out["pooled"][:, self.cfg.d_model:]

        return z_a, z_v

    def _forward_tcn(
        self,
        audio_adapted: list[torch.Tensor],
        video_adapted: list[torch.Tensor],
        mask_a: torch.Tensor,
        mask_v: torch.Tensor,
        vad: torch.Tensor | None,
        qc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TCN前向（备用）"""
        a = self.audio_fusion(audio_adapted)
        v = self.video_fusion(video_adapted)

        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        a = self.audio_tcn(a, mask_a)
        v = self.video_tcn(v, mask_v)

        z_a = self.audio_asp(a, mask_a, vad, qc)
        z_v = self.video_asp(v, mask_v, vad, qc)

        return z_a, z_v


MTCNBackbone = HierarchicalMTCNBackbone
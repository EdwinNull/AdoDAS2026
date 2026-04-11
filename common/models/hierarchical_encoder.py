"""
层级化跨模态Transformer编码器

实现基于AdoDAS 2026技术指南的先进架构:
- TemporalDownsampler: 时序降采样（步长4卷积）
- BottleneckTransformer (MBT): 多模态瓶颈Transformer融合
- PerceiverCompressor: Perceiver风格跨模态压缩
- AttentiveStatisticsPooling: 注意力统计池化（均值+标准差）
- SessionAttentionAggregator: 会话级注意力加权聚合
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class HierarchicalEncoderConfig:
    """层级化编码器配置"""
    d_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 2048
    n_unimodal_layers: int = 4
    n_fusion_layers: int = 2
    n_bottleneck_tokens: int = 4
    n_perceiver_queries: int = 32
    dropout: float = 0.1
    temporal_stride: int = 4


class TemporalDownsampler(nn.Module):
    """
    时序降采样模块

    使用步长卷积将长序列（如3000帧WavLM特征）降采样到 manageable 长度。
    步长4的卷积将3000帧降至约750帧。
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(
            out_dim, out_dim,
            kernel_size=5, stride=stride, padding=2
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: (B, T, D) 输入序列
            mask: (B, T) 布尔掩码
        Returns:
            x: (B, T//stride, D) 降采样后序列
            mask: (B, T//stride) 降采样后掩码
        """
        x = self.proj(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)

        if mask is not None:
            mask = mask[:, :: self.conv.stride[0]]

        return x, mask


class AttentiveStatisticsPooling(nn.Module):
    """
    注意力统计池化

    生成加权均值和加权标准差的注意力池化方法。
    同时捕捉中心趋势和变异性，对检测抑郁症典型的面部表情扁平化和韵律范围缩减至关重要。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor | None = None,
        qc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 输入序列
            mask: (B, T) 布尔掩码
            vad: (B, T) 语音活动检测信号（可选）
            qc: (B, T) 质量控制信号（可选）
        Returns:
            (B, 2*D) 拼接的均值和标准差
        """
        e = self.attn(x).squeeze(-1)

        if vad is not None:
            e = e + 0.5 * vad
        if qc is not None:
            e = e + 0.5 * qc

        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        w = w.masked_fill(~mask, 0.0)

        w_unsq = w.unsqueeze(-1)
        mu = (w_unsq * x).sum(dim=1)

        diff = x - mu.unsqueeze(1)
        var = (w_unsq * diff ** 2).sum(dim=1)
        sigma = torch.sqrt(var.clamp(min=1e-8))

        return torch.cat([mu, sigma], dim=-1)


class BottleneckCrossAttention(nn.Module):
    """
    瓶颈交叉注意力模块（MBT核心组件）

    Nagrani et al. (NeurIPS 2021) 提出的多模态瓶颈Transformer设计。
    使用4个可学习瓶颈Token作为中介进行跨模态信息流。
    实现50%的FLOP减少，同时在精度上持平甚至超越完整交叉注意力。
    """

    def __init__(self, d_model: int, n_heads: int, num_bottlenecks: int = 4):
        super().__init__()
        self.num_bottlenecks = num_bottlenecks

        self.bottleneck = nn.Parameter(torch.randn(1, num_bottlenecks, d_model))
        nn.init.normal_(self.bottleneck, std=0.02)

        self.cross_attn_audio_to_bottleneck = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.cross_attn_video_to_bottleneck = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.cross_attn_from_bottleneck = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: (B, T_a, D) 音频序列
            video: (B, T_v, D) 视频序列
            audio_mask: (B, T_a) 音频掩码
            video_mask: (B, T_v) 视频掩码
        Returns:
            audio_out: (B, T_a, D) 融合后的音频序列
            video_out: (B, T_v, D) 融合后的视频序列
        """
        B = audio.size(0)
        bottleneck = self.bottleneck.expand(B, -1, -1)

        bottleneck_mask = torch.zeros(
            B, self.num_bottlenecks, dtype=torch.bool, device=audio.device
        )

        audio_mask_flat = ~audio_mask
        video_mask_flat = ~video_mask

        attn_out, _ = self.cross_attn_audio_to_bottleneck(
            query=bottleneck,
            key=audio,
            value=audio,
            key_padding_mask=audio_mask_flat,
        )
        bottleneck = self.norm1(bottleneck + attn_out)

        attn_out, _ = self.cross_attn_video_to_bottleneck(
            query=bottleneck,
            key=video,
            value=video,
            key_padding_mask=video_mask_flat,
        )
        bottleneck = self.norm2(bottleneck + attn_out)

        bottleneck = self.norm3(bottleneck + self.ffn(bottleneck))

        audio_out, _ = self.cross_attn_from_bottleneck(
            query=audio,
            key=bottleneck,
            value=bottleneck,
        )
        audio_out = audio + audio_out

        video_out, _ = self.cross_attn_from_bottleneck(
            query=video,
            key=bottleneck,
            value=bottleneck,
        )
        video_out = video + video_out

        return audio_out, video_out


class PerceiverCompressor(nn.Module):
    """
    Perceiver风格跨模态压缩器

    使用32个可学习查询Token对所有模态序列进行交叉关注，
    生成压缩后的固定长度表征（32 x d_model），不受输入长度限制。
    交叉注意力复杂度: O(32 x T) ≈ 24000操作
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_queries: int = 32):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio: (B, T_a, D) 音频序列
            video: (B, T_v, D) 视频序列
            audio_mask: (B, T_a) 音频掩码
            video_mask: (B, T_v) 视频掩码
        Returns:
            (B, n_queries, D) 压缩后的表征
        """
        B = audio.size(0)
        queries = self.queries.expand(B, -1, -1)

        combined = torch.cat([audio, video], dim=1)
        combined_mask = torch.cat([audio_mask, video_mask], dim=1)

        key_padding_mask = ~combined_mask

        attn_out, _ = self.cross_attn(
            query=queries,
            key=combined,
            value=combined,
            key_padding_mask=key_padding_mask,
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))

        return queries


class HierarchicalCrossModalEncoder(nn.Module):
    """
    层级化跨模态Transformer编码器

    完整架构:
    1. 特征投影 → 共享维度
    2. 时序降采样（音频/视频独立）
    3. 单模态Transformer编码（4层）
    4. MBT瓶颈融合（2层，4个瓶颈Token）
    5. Perceiver压缩（32个查询）
    6. 注意力统计池化
    """

    def __init__(self, config: HierarchicalEncoderConfig):
        super().__init__()
        self.cfg = config

        self.audio_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_unimodal = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_unimodal_layers
        )
        self.video_unimodal = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_unimodal_layers
        )

        self.bottleneck_fusion = nn.ModuleList([
            BottleneckCrossAttention(
                config.d_model,
                config.n_heads,
                config.n_bottleneck_tokens,
            )
            for _ in range(config.n_fusion_layers)
        ])

        self.perceiver = PerceiverCompressor(
            config.d_model,
            config.n_heads,
            config.n_perceiver_queries,
        )

        self.pooling = AttentiveStatisticsPooling(config.d_model)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
        vad: torch.Tensor | None = None,
        qc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            audio: (B, T_a, D) 音频序列
            video: (B, T_v, D) 视频序列
            audio_mask: (B, T_a) 音频掩码
            video_mask: (B, T_v) 视频掩码
            vad: (B, T) 语音活动检测信号
            qc: (B, T) 质量控制信号
        Returns:
            dict包含:
                - pooled: (B, 2*D) 池化表征
                - perceiver_out: (B, n_queries, D) Perceiver输出
        """
        audio = self.audio_proj(audio)
        video = self.video_proj(video)

        audio = self.audio_unimodal(audio, src_key_padding_mask=~audio_mask)
        video = self.video_unimodal(video, src_key_padding_mask=~video_mask)

        for fusion_layer in self.bottleneck_fusion:
            audio, video = fusion_layer(audio, video, audio_mask, video_mask)

        perceiver_out = self.perceiver(audio, video, audio_mask, video_mask)

        combined = torch.cat([audio, video], dim=1)
        combined_mask = torch.cat([audio_mask, video_mask], dim=1)

        if vad is not None:
            combined_vad = torch.cat([vad, vad], dim=1)
        else:
            combined_vad = None

        if qc is not None:
            combined_qc = torch.cat([qc, qc], dim=1)
        else:
            combined_qc = None

        pooled = self.pooling(combined, combined_mask, combined_vad, combined_qc)

        return {
            "pooled": pooled,
            "perceiver_out": perceiver_out,
            "audio_seq": audio,
            "video_seq": video,
        }


class SessionAttentionAggregator(nn.Module):
    """
    会话级注意力加权聚合器

    用于将单个参与者的4个会话表征聚合成单一的人物级表征。
    注意力加权聚合优于简单均值池化，
    因为B03（最悲伤的记忆）可能比B01（描述昨天）带有更多抑郁症的诊断信号。
    """

    def __init__(self, d_model: int, n_sessions: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_sessions = n_sessions

        self.session_type_embed = nn.Embedding(n_sessions, d_model)

        self.query = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        session_reprs: torch.Tensor,
        session_valid: torch.Tensor,
        session_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            session_reprs: (B, n_sessions, D) 会话表征
            session_valid: (B, n_sessions) 布尔掩码
            session_idx: (B, n_sessions) 会话类型索引（可选）
        Returns:
            (B, D) 人物级表征
        """
        if session_idx is not None:
            type_emb = self.session_type_embed(session_idx)
            session_reprs = session_reprs + type_emb

        scores = self.query(session_reprs).squeeze(-1)
        scores = scores.masked_fill(~session_valid, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~session_valid, 0.0)

        pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)
        return self.proj(pooled)

    def get_attention_weights(
        self,
        session_reprs: torch.Tensor,
        session_valid: torch.Tensor,
        session_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回注意力权重用于分析"""
        if session_idx is not None:
            type_emb = self.session_type_embed(session_idx)
            session_reprs = session_reprs + type_emb

        scores = self.query(session_reprs).squeeze(-1)
        scores = scores.masked_fill(~session_valid, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~session_valid, 0.0)
        return weights

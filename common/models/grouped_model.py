"""
Grouped Model with Enhanced Session Aggregation

基于AdoDAS 2026技术指南改进的分组模型:
- 会话注意力加权聚合
- 会话类型条件化嵌入
- A2→A1桥接支持
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import MTCNBackbone, BackboneConfig


class ParticipantAggregator(nn.Module):
    """
    参与者聚合器

    将单个参与者的4个会话表征聚合成单一的人物级表征。
    支持三种聚合方法:
    - mean: 简单均值池化
    - mlp: MLP投影的均值池化
    - attention: 注意力加权聚合（推荐）
    """

    def __init__(self, d_in: int, d_out: int, method: str = "attention", dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.d_in = d_in
        self.d_out = d_out

        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_out, d_out),
            )
        elif method == "attention":
            self.query = nn.Linear(d_in, 1)
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_out, d_out),
            )
        elif method == "mean":
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

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
        mask = session_valid.float().unsqueeze(-1)
        masked_reprs = session_reprs * mask

        if self.method == "mean":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.proj(pooled)

        elif self.method == "mlp":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.mlp(pooled)

        elif self.method == "attention":
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
    ) -> torch.Tensor:
        """返回注意力权重用于分析"""
        if self.method != "attention":
            return None

        scores = self.query(session_reprs).squeeze(-1)
        scores = scores.masked_fill(~session_valid, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~session_valid, 0.0)
        return weights


class SessionTypeClassifier(nn.Module):
    """会话类型分类器"""
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GroupedModel(nn.Module):
    """
    分组模型

    用于处理参与者级别的预测。
    将4个会话的特征聚合成单一的人物级表征。
    """

    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "attention",
        dropout: float = 0.2,
        use_session_conditioning: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_session_conditioning = use_session_conditioning

        self.aggregator = ParticipantAggregator(
            d_in=d_shared,
            d_out=d_shared,
            method=aggregator_method,
            dropout=dropout,
        )

        self.session_type_head = SessionTypeClassifier(d_in=d_shared)

        if use_session_conditioning:
            n_sessions = backbone.cfg.n_sessions if hasattr(backbone, 'cfg') else 4
            self.session_type_embed = nn.Embedding(n_sessions, d_shared)

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            flat_batch: 展平后的批次数据
            n_participants: 参与者数量
            session_valid: (B, 4) 会话有效性掩码
        Returns:
            dict包含:
                - session_reprs: (n_flat, D) 会话表征
                - participant_repr: (B, D) 人物表征
                - session_type_logits: (n_flat, 4) 会话类型logits
        """
        session_reprs = self.backbone(flat_batch)

        B = n_participants
        session_grid = session_reprs.view(B, 4, -1)

        session_idx = flat_batch.get("session_idx")
        if self.use_session_conditioning and session_idx is not None:
            session_idx_grid = session_idx.view(B, 4)
        else:
            session_idx_grid = None

        participant_repr = self.aggregator(session_grid, session_valid, session_idx_grid)

        session_type_logits = self.session_type_head(session_reprs)

        return {
            "session_reprs": session_reprs,
            "participant_repr": participant_repr,
            "session_type_logits": session_type_logits,
        }


class CORALHead(nn.Module):
    """
    CORAL (Consistent Rank Logits) 序数回归头

    用于A2赛道的21个DASS-21条目预测。
    """

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds

        self.score_fc = nn.Linear(d_in, n_items)

        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        nn.init.constant_(self.raw_thresholds, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_fc(x)

        spacings = F.softplus(self.raw_thresholds)
        thresholds = torch.cumsum(spacings, dim=-1)

        logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0)
        return logits

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)

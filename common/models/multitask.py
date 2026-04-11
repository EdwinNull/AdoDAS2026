"""
统一多任务学习头

基于AdoDAS 2026技术指南实现A1+A2统一多任务学习:
- A2→A1桥接: 从21个DASS-21条目预测推导抑郁/焦虑/压力子量表
- 联合训练相互正则化
- 不确定性加权自动平衡各任务损失
- 子量表分组参数共享
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import UncertaintyWeightedLoss, SubscaleConsistencyLoss


class A2A1BridgeHead(nn.Module):
    """
    A2→A1桥接头

    从21个DASS-21条目预测推导抑郁(D)、焦虑(A)、压力(S)的二分类筛查标签。
    这是本比赛最重要的战略洞察之一。

    DASS-21评分规则:
    - 抑郁 = 2·(d03+d05+d10+d13+d16+d17+d21)
    - 焦虑 = 2·(d02+d04+d07+d09+d15+d19+d20)
    - 压力 = 2·(d01+d06+d08+d11+d12+d14+d18)

    严重程度临界值(青少年标准):
    - 正常: 0-4/0-3/0-7
    - 轻度: 5-6/4-5/8-9
    - 中度: 7-10/6-7/10-12
    - 重度: 11+/8+/13+
    """

    DEPRESSION_ITEMS = [2, 4, 9, 12, 15, 16, 20]
    ANXIETY_ITEMS = [1, 3, 6, 8, 14, 18, 19]
    STRESS_ITEMS = [0, 5, 7, 10, 11, 13, 17]

    SEVERITY_THRESHOLDS = {
        "depression": {"mild": 5, "moderate": 7, "severe": 11},
        "anxiety": {"mild": 4, "moderate": 6, "severe": 8},
        "stress": {"mild": 8, "moderate": 10, "severe": 13},
    }

    def __init__(self, threshold_mode: str = "moderate"):
        super().__init__()
        self.threshold_mode = threshold_mode

        self.dep_thresholds = self.SEVERITY_THRESHOLDS["depression"]
        self.anx_thresholds = self.SEVERITY_THRESHOLDS["anxiety"]
        self.str_thresholds = self.SEVERITY_THRESHOLDS["stress"]

    def forward(
        self,
        item_preds: torch.Tensor,
        item_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            item_preds: (B, 21) 预测的条目值 [0, 3]
            item_logits: (B, 21, 3) 可选，序数logits用于软版本
        Returns:
            (B, 3) D/A/S二分类概率
        """
        dep_idx = torch.tensor(
            self.DEPRESSION_ITEMS, device=item_preds.device, dtype=torch.long
        )
        anx_idx = torch.tensor(
            self.ANXIETY_ITEMS, device=item_preds.device, dtype=torch.long
        )
        stress_idx = torch.tensor(
            self.STRESS_ITEMS, device=item_preds.device, dtype=torch.long
        )

        dep_score = item_preds[:, dep_idx].sum(dim=1) * 2
        anx_score = item_preds[:, anx_idx].sum(dim=1) * 2
        stress_score = item_preds[:, stress_idx].sum(dim=1) * 2

        if self.threshold_mode == "mild":
            dep_binary = (dep_score >= self.dep_thresholds["mild"]).float()
            anx_binary = (anx_score >= self.anx_thresholds["mild"]).float()
            stress_binary = (stress_score >= self.str_thresholds["mild"]).float()
        elif self.threshold_mode == "severe":
            dep_binary = (dep_score >= self.dep_thresholds["severe"]).float()
            anx_binary = (anx_score >= self.anx_thresholds["severe"]).float()
            stress_binary = (stress_score >= self.str_thresholds["severe"]).float()
        else:
            dep_binary = (dep_score >= self.dep_thresholds["moderate"]).float()
            anx_binary = (anx_score >= self.anx_thresholds["moderate"]).float()
            stress_binary = (stress_score >= self.str_thresholds["moderate"]).float()

        return torch.stack([dep_binary, anx_binary, stress_binary], dim=1)

    def get_subscale_scores(self, item_preds: torch.Tensor) -> torch.Tensor:
        """返回原始子量表分数用于分析"""
        dep_idx = torch.tensor(
            self.DEPRESSION_ITEMS, device=item_preds.device, dtype=torch.long
        )
        anx_idx = torch.tensor(
            self.ANXIETY_ITEMS, device=item_preds.device, dtype=torch.long
        )
        stress_idx = torch.tensor(
            self.STRESS_ITEMS, device=item_preds.device, dtype=torch.long
        )

        dep_score = item_preds[:, dep_idx].sum(dim=1) * 2
        anx_score = item_preds[:, anx_idx].sum(dim=1) * 2
        stress_score = item_preds[:, stress_idx].sum(dim=1) * 2

        return torch.stack([dep_score, anx_score, stress_score], dim=1)


class MultitaskA1A2Head(nn.Module):
    """
    统一多任务头

    同时处理:
    1. A1赛道: 抑郁/焦虑/压力二分类
    2. A2赛道: 21个DASS-21条目序数预测
    3. A2→A1桥接: 从条目预测推导二分类

    特点:
    - 不确定性加权自动平衡A1/A2任务
    - 子量表分组共享表征
    - A2→A1桥接提供相互正则化
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 256,
        n_items: int = 21,
        n_classes: int = 4,
        use_subscale_grouping: bool = True,
        use_bridge: bool = True,
        bridge_loss_weight: float = 0.2,
        consistency_loss_weight: float = 0.1,
        bias_init: list[float] | None = None,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_classes - 1
        self.use_subscale_grouping = use_subscale_grouping
        self.use_bridge = use_bridge
        self.bridge_loss_weight = bridge_loss_weight

        self.shared_repr = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        if use_subscale_grouping:
            self.dep_encoder = nn.Linear(d_in, d_hidden // 2)
            self.anx_encoder = nn.Linear(d_in, d_hidden // 2)
            self.stress_encoder = nn.Linear(d_in, d_hidden // 2)

            shared_in = d_hidden + d_hidden // 2 * 3
        else:
            shared_in = d_hidden

        self.shared_proj = nn.Sequential(
            nn.Linear(shared_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.a1_head = nn.Linear(d_hidden, 3)
        if bias_init is not None:
            with torch.no_grad():
                self.a1_head.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

        self.a2_item_heads = nn.ModuleList([
            nn.Linear(d_hidden, self.n_thresholds)
            for _ in range(n_items)
        ])

        if use_subscale_grouping:
            self.subscale_proj = nn.Sequential(
                nn.Linear(d_hidden // 2 * 3, d_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.subscale_head = nn.Linear(d_hidden, 3)

        self.bridge = A2A1BridgeHead()

        self.uncertainty_weighting = UncertaintyWeightedLoss(num_tasks=3)

        self.consistency_loss = SubscaleConsistencyLoss(
            scale=consistency_loss_weight
        )

    def forward_a1(self, x: torch.Tensor) -> torch.Tensor:
        """A1二分类logits"""
        shared = self.shared_repr(x)
        if self.use_subscale_grouping:
            dep = F.gelu(self.dep_encoder(x))
            anx = F.gelu(self.anx_encoder(x))
            stress = F.gelu(self.stress_encoder(x))
            subscale_features = torch.cat([dep, anx, stress], dim=1)
            subscale_repr = self.subscale_proj(subscale_features)
            shared = torch.cat([shared, subscale_repr], dim=1)
        shared = self.shared_proj(shared)
        return self.a1_head(shared)

    def forward_a2(self, x: torch.Tensor) -> torch.Tensor:
        """A2序数预测logits: (B, n_items, n_thresholds)"""
        shared = self.shared_repr(x)
        if self.use_subscale_grouping:
            dep = F.gelu(self.dep_encoder(x))
            anx = F.gelu(self.anx_encoder(x))
            stress = F.gelu(self.stress_encoder(x))
            subscale_features = torch.cat([dep, anx, stress], dim=1)
            subscale_repr = self.subscale_proj(subscale_features)
            shared = torch.cat([shared, subscale_repr], dim=1)
        shared = self.shared_proj(shared)

        item_logits = torch.stack([
            head(shared) for head in self.a2_item_heads
        ], dim=1)
        return item_logits

    def forward_a1_from_a2(self, a2_logits: torch.Tensor) -> torch.Tensor:
        """通过A2→A1桥接从序数预测获取二分类"""
        preds = self.predict_a2(a2_logits)
        return self.bridge(preds)

    def predict_a2(self, logits: torch.Tensor) -> torch.Tensor:
        """从序数logits预测整数标签"""
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)

    def forward(
        self,
        x: torch.Tensor,
        targets_a1: torch.Tensor | None = None,
        targets_a2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        统一前向传播

        Args:
            x: (B, D) 输入表征
            targets_a1: (B, 3) A1目标（可选）
            targets_a2: (B, 21) A2目标（可选）
        Returns:
            dict包含所有logits和损失（如果提供了目标）
        """
        a1_logits = self.forward_a1(x)
        a2_logits = self.forward_a2(x)

        result = {
            "a1_logits": a1_logits,
            "a2_logits": a2_logits,
        }

        if targets_a1 is not None or targets_a2 is not None:
            losses = {}

            if targets_a1 is not None:
                losses["a1"] = F.binary_cross_entropy_with_logits(
                    a1_logits, targets_a1
                )

            if targets_a2 is not None:
                ordinal_targets = self.build_ordinal_targets(
                    targets_a2, self.n_thresholds
                )
                losses["a2"] = F.binary_cross_entropy_with_logits(
                    a2_logits, ordinal_targets
                )

            if self.use_bridge and targets_a1 is not None and targets_a2 is not None:
                a2_preds = self.predict_a2(a2_logits).float()
                bridge_preds = self.bridge(a2_preds)
                losses["bridge"] = F.binary_cross_entropy_with_logits(
                    bridge_preds, targets_a1
                )

            if self.use_subscale_grouping and targets_a2 is not None:
                a2_preds = self.predict_a2(a2_logits).float()
                losses["consistency"] = self.consistency_loss(
                    a2_preds, a1_logits
                )

            if len(losses) > 0:
                result["losses"] = losses

        return result

    @staticmethod
    def build_ordinal_targets(
        labels: torch.Tensor,
        n_thresholds: int = 3,
    ) -> torch.Tensor:
        """将整数标签转换为序数二值目标"""
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets

    def get_attention_weights(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """返回各任务的注意力权重用于分析"""
        weights = {}
        if hasattr(self, "uncertainty_weighting"):
            weights["task"] = self.uncertainty_weighting.get_weights()
        return weights


class A1A2JointTrainer:
    """
    A1+A2联合训练器

    封装多任务学习的训练逻辑，包括:
    - 不确定性加权损失计算
    - A2→A1桥接损失
    - 子量表一致性损失
    """

    def __init__(
        self,
        multitask_head: MultitaskA1A2Head,
        device: torch.device,
        loss_weights: dict[str, float] | None = None,
    ):
        self.head = multitask_head
        self.device = device
        self.loss_weights = loss_weights or {
            "a1": 1.0,
            "a2": 1.0,
            "bridge": 0.2,
            "consistency": 0.1,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        targets_a1: torch.Tensor,
        targets_a2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        计算加权总损失

        Args:
            x: (B, D) 输入表征
            targets_a1: (B, 3) A1目标
            targets_a2: (B, 21) A2目标
        Returns:
            (total_loss, loss_dict)
        """
        result = self.head(x, targets_a1, targets_a2)
        losses = result["losses"]

        loss_dict = {}
        weighted_losses = []

        for name, loss in losses.items():
            weight = self.loss_weights.get(name, 1.0)
            weighted_losses.append(loss * weight)
            loss_dict[name] = loss.item() * weight

        total_loss = sum(weighted_losses)
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def predict(
        self,
        x: torch.Tensor,
        mode: str = "a1_a2",
    ) -> dict[str, torch.Tensor]:
        """
        预测

        Args:
            x: (B, D) 输入表征
            mode: 预测模式
                - "a1_only": 仅A1
                - "a2_only": 仅A2
                - "a1_a2": A1和A2
                - "bridge_only": 仅A2→A1桥接
                - "ensemble": A1 + 桥接集成
        Returns:
            预测结果字典
        """
        a1_logits = self.head.forward_a1(x)
        a2_logits = self.head.forward_a2(x)

        result = {
            "a1_probs": torch.sigmoid(a1_logits),
            "a2_preds": self.head.predict_a2(a2_logits),
        }

        if mode in ("bridge_only", "ensemble"):
            bridge_probs = self.head.forward_a1_from_a2(a2_logits)
            result["bridge_probs"] = torch.sigmoid(bridge_probs)

        if mode == "ensemble":
            a1_probs = result["a1_probs"]
            bridge_probs = result["bridge_probs"]
            result["ensemble_probs"] = (a1_probs + bridge_probs) / 2

        return result

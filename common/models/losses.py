"""
先进损失函数模块

基于AdoDAS 2026技术指南实现的优化损失函数:
- AsymmetricLoss: A1非对称损失，处理类别不平衡
- SoftF1Loss: 软F1损失，直接优化竞赛指标
- DiffQWKLoss: 可微QWK损失，用于A2序数预测
- UncertaintyWeightedLoss: 不确定性加权MTL
- CORNLoss: 条件序数回归损失
- CombinedA1Loss: ASL+软F1组合损失
- CombinedA2Loss: CORN+QWK组合损失
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AsymmetricLoss(nn.Module):
    """
    非对称损失 (Asymmetric Loss)

    对正负样本应用不同的聚焦参数，加上概率偏移机制。
    配置: γ+=0, γ-=2, clip=0.05
    clip参数在处理自评DASS-21筛查中的标签噪声时特别有价值。
    """

    def __init__(
        self,
        gamma_neg: float = 2,
        gamma_pos: float = 0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) 未缩放的logits
            targets: (B, C) 二进制标签（0或1）
        Returns:
            损失标量
        """
        xs_pos = torch.sigmoid(logits)
        xs_neg = (1 - xs_pos + self.clip).clamp(max=1)

        loss = targets * torch.log(xs_pos.clamp(min=1e-8)) + \
               (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + (1 - xs_pos) * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * ((1 - pt).detach() ** gamma)

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftF1Loss(nn.Module):
    """
    软F1损失

    使用TP/FP/FN的可微近似值直接优化F1指标。
    soft_f1 = 2·Σ(p·y) / (2·Σ(p·y) + Σ(p·(1-y)) + Σ((1-p)·y))
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) 未缩放的logits
            targets: (B, C) 二进制标签
        Returns:
            损失标量
        """
        probs = torch.sigmoid(logits)

        tp = (probs * targets).sum(dim=0)
        fp = (probs * (1 - targets)).sum(dim=0)
        fn = ((1 - probs) * targets).sum(dim=0)

        numerator = 2 * tp
        denominator = 2 * tp + fp + fn + 1e-8

        f1_per_class = numerator / denominator
        f1 = f1_per_class.mean()

        return 1 - f1


class DiffQWKLoss(nn.Module):
    """
    可微二次加权Kappa损失

    构建软混淆矩阵计算QWK的可微近似。
    由于可能存在退化的零列解风险，建议与CORN损失联合使用。
    """

    def __init__(self, num_classes: int = 4, weight: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) 未缩放的logits
            targets: (B,) 整数标签 [0, num_classes-1]
        Returns:
            损失标量（1 - QWK）
        """
        probs = torch.softmax(logits, dim=-1)
        N = self.num_classes

        w = torch.zeros(N, N, device=logits.device)
        for i in range(N):
            for j in range(N):
                w[i, j] = (i - j) ** 2 / ((N - 1) ** 2 + 1e-8)

        targets_one_hot = F.one_hot(targets.long(), N).float()

        O = torch.matmul(targets_one_hot.t(), probs)
        hist_true = targets_one_hot.sum(dim=0)
        hist_pred = probs.sum(dim=0)
        E = torch.outer(hist_true, hist_pred) / (targets.size(0) + 1e-8)

        num = (w * O).sum()
        den = (w * E).sum()
        qwk = 1.0 - num / (den + 1e-8)

        return (1 - qwk) * self.weight


class CORNLoss(nn.Module):
    """
    条件序数回归损失 (Conditional Ordinal Regression)

    使用链式法则 P(Y>k | Y≥k) 将每个4类序数问题分解为3个条件二分类任务。
    每个DASS-21条目都有自己的CORN头（3个输出神经元），共享编码器主干。
    优于CORAL因为它消除了可能阻碍性能的权重共享约束。

    DASS-21条目分组:
    - 抑郁: d03, d05, d10, d13, d16, d17, d21
    - 焦虑: d02, d04, d07, d09, d15, d19, d20
    - 压力: d01, d06, d08, d11, d12, d14, d18
    """

    def __init__(
        self,
        n_items: int = 21,
        n_classes: int = 4,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        self.label_smoothing = label_smoothing

    @staticmethod
    def build_ordinal_targets(
        labels: torch.Tensor,
        n_thresholds: int = 3,
    ) -> torch.Tensor:
        """
        将整数标签转换为序数二值目标

        labels: (B, I) 整数标签 [0, 3]
        returns: (B, I, n_thresholds) 二值目标
        """
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, n_items, n_thresholds) 每个条目的序数logits
            targets: (B, n_items) 整数标签 [0, 3]
        Returns:
            损失标量
        """
        ordinal_targets = self.build_ordinal_targets(
            targets, self.n_thresholds
        )

        if self.label_smoothing > 0:
            ordinal_targets = (
                ordinal_targets * (1 - self.label_smoothing) +
                0.5 * self.label_smoothing
            )

        loss = F.binary_cross_entropy_with_logits(
            logits, ordinal_targets, reduction="none"
        )

        return loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权损失 (Uncertainty Weighting)

    Kendall et al. (CVPR 2018) 提出的方法。
    学习同方差不确定性参数（每个任务一个），自动平衡损失幅度。
    L_total = Σ_i [0.5 · exp(-s_i) · L_i + 0.5 · s_i]
    其中 s_i = log(σ²_i) 是可学习的对数方差参数。

    仅增加3个可学习参数，零计算开销，
    自动在训练期间提高较难任务的权重。
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        nn.init.zeros_(self.log_vars)

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: 任务损失列表，长度为num_tasks
        Returns:
            加权后的总损失
        """
        total = sum(
            0.5 * torch.exp(-s) * L + 0.5 * s
            for s, L in zip(self.log_vars, losses)
        )
        return total

    def get_weights(self) -> torch.Tensor:
        """返回每个任务的自动学习权重"""
        return torch.exp(-self.log_vars)


class CombinedA1Loss(nn.Module):
    """
    A1组合损失

    0.7 * ASL + 0.3 * SoftF1
    软F1替代指标直接优化竞赛指标。
    """

    def __init__(
        self,
        gamma_neg: float = 2,
        gamma_pos: float = 0,
        clip: float = 0.05,
        soft_f1_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg, gamma_pos, clip)
        self.soft_f1 = SoftF1Loss()
        self.soft_f1_weight = soft_f1_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 3) D/A/S的logits
            targets: (B, 3) 二进制标签
        Returns:
            加权组合损失
        """
        if self.label_smoothing > 0:
            targets = targets.float() * (1 - self.label_smoothing) + \
                     0.5 * self.label_smoothing

        asl_loss = self.asl(logits, targets)
        soft_f1_loss = self.soft_f1(logits, targets)

        return 0.7 * asl_loss + self.soft_f1_weight * soft_f1_loss


class CombinedA2Loss(nn.Module):
    """
    A2组合损失

    0.7 * CORN + 0.3 * differentiable_QWK
    验证集上进行逐条目阈值优化作为后处理。
    """

    def __init__(
        self,
        n_items: int = 21,
        n_classes: int = 4,
        qwk_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.corn = CORNLoss(n_items, n_classes, label_smoothing)
        self.qwk_loss = DiffQWKLoss(n_classes, qwk_weight)
        self.n_items = n_items

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, n_items, n_thresholds) 序数logits
            targets: (B, n_items) 整数标签
        Returns:
            加权组合损失
        """
        corn_loss = self.corn(logits, targets)

        total_qwk_loss = 0
        for i in range(self.n_items):
            item_logits = logits[:, i, :]
            item_targets = targets[:, i]
            total_qwk_loss = total_qwk_loss + self.qwk_loss(item_logits, item_targets)

        return 0.7 * corn_loss + 0.3 * (total_qwk_loss / self.n_items)


class SubscaleConsistencyLoss(nn.Module):
    """
    子量表一致性损失

    强制预测条目和整体子量表严重程度之间的一致性。
    DASS-21条目映射到子量表:
    - 抑郁 = 2·(d03+d05+d10+d13+d16+d17+d21)
    - 焦虑 = 2·(d02+d04+d07+d09+d15+d19+d20)
    - 压力 = 2·(d01+d06+d08+d11+d12+d14+d18)
    """

    DEPRESSION_ITEMS = [2, 4, 9, 12, 15, 16, 20]
    ANXIETY_ITEMS = [1, 3, 6, 8, 14, 18, 19]
    STRESS_ITEMS = [0, 5, 7, 10, 11, 13, 17]

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        item_preds: torch.Tensor,
        subscale_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_preds: (B, 21) 预测的条目值 [0, 3]
            subscale_logits: (B, 3) D/A/S的logits
        Returns:
            一致性损失
        """
        deps = torch.tensor(self.DEPRESSION_ITEMS, device=item_preds.device)
        anxs = torch.tensor(self.ANXIETY_ITEMS, device=item_preds.device)
        stresses = torch.tensor(self.STRESS_ITEMS, device=item_preds.device)

        dep_sum = item_preds[:, deps].sum(dim=1) * 2
        anx_sum = item_preds[:, anxs].sum(dim=1) * 2
        stress_sum = item_preds[:, stresses].sum(dim=1) * 2

        predicted_subscales = torch.stack([dep_sum, anx_sum, stress_sum], dim=1)

        predicted_probs = torch.sigmoid(subscale_logits) * 42

        loss = F.mse_loss(predicted_probs, predicted_subscales.float())

        return loss * self.scale

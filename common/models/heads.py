"""
任务预测头（Enhanced with Advanced Losses）

基于AdoDAS 2026技术指南优化:
- A1: ASL+软F1组合损失
- A2: CORN+QWK组合损失
- CORN序数回归实现
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):
    """
    A1二分类头

    输出抑郁(D)、焦虑(A)、压力(S)的二分类logits。
    支持使用组合损失(ASL + 软F1)。
    """

    def __init__(
        self,
        d_in: int,
        bias_init: list[float] | None = None,
        use_combined_loss: bool = True,
        gamma_neg: float = 2,
        gamma_pos: float = 0,
        clip: float = 0.05,
        soft_f1_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_combined_loss = use_combined_loss
        self.soft_f1_weight = soft_f1_weight
        self.label_smoothing = label_smoothing

        self.fc = nn.Linear(d_in, 3)
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

        if use_combined_loss:
            self.asl_gamma_neg = gamma_neg
            self.asl_gamma_pos = gamma_pos
            self.asl_clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


class A2OrdinalHead(nn.Module):
    """
    A2序数预测头

    预测21个DASS-21条目的序数值(0-3)。
    使用CORAL(Consistent Rank Logits)方法实现。
    支持多种解码方式和组合损失。
    """

    def __init__(
        self,
        d_in: int,
        n_items: int = 21,
        n_thresholds: int = 3,
        use_corn_loss: bool = True,
        use_qwk_aux: bool = True,
        qwk_weight: float = 0.3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        self.use_corn_loss = use_corn_loss
        self.use_qwk_aux = use_qwk_aux
        self.qwk_weight = qwk_weight
        self.label_smoothing = label_smoothing

        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        """argmax解码"""
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        """单调性约束的argmax解码"""
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
        """期望值解码（推荐）"""
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(
        labels: torch.Tensor,
        n_thresholds: int = 3,
    ) -> torch.Tensor:
        """将整数标签转换为序数二值目标"""
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets


def a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_combined: bool = True,
    gamma_neg: float = 2,
    gamma_pos: float = 0,
    clip: float = 0.05,
    soft_f1_weight: float = 0.3,
) -> torch.Tensor:
    """
    A1损失函数

    Args:
        logits: (B, 3) D/A/S的logits
        targets: (B, 3) 二进制标签
        pos_weight: 正样本权重
        label_smoothing: 标签平滑
        use_combined: 是否使用ASL+软F1组合
        gamma_neg: ASL负样本聚焦参数
        gamma_pos: ASL正样本聚焦参数
        clip: ASL概率偏移参数
        soft_f1_weight: 软F1损失权重
    Returns:
        损失标量
    """
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if not use_combined:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    xs_pos = torch.sigmoid(logits)
    xs_neg = (1 - xs_pos + clip).clamp(max=1)

    asl_loss = targets * torch.log(xs_pos.clamp(min=1e-8)) + \
               (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

    if gamma_neg > 0 or gamma_pos > 0:
        pt = xs_pos * targets + (1 - xs_pos) * (1 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1 - targets)
        asl_loss = asl_loss * ((1 - pt).detach() ** gamma)

    asl_loss = -asl_loss

    tp = (xs_pos * targets).sum(dim=0)
    fp = (xs_pos * (1 - targets)).sum(dim=0)
    fn = ((1 - xs_pos) * targets).sum(dim=0)
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn + 1e-8
    f1_per_class = numerator / denominator
    f1_loss = 1 - f1_per_class.mean()

    combined = 0.7 * asl_loss.mean() + soft_f1_weight * f1_loss
    return combined


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_corn: bool = True,
    use_qwk: bool = True,
    qwk_weight: float = 0.3,
) -> torch.Tensor:
    """
    A2序数回归损失函数

    Args:
        logits: (B, n_items, n_thresholds) 序数logits
        labels: (B, n_items) 整数标签 [0, 3]
        pos_weight: 正样本权重
        label_smoothing: 标签平滑
        use_corn: 是否使用CORN损失
        use_qwk: 是否使用QWK辅助损失
        qwk_weight: QWK损失权重
    Returns:
        损失标量
    """
    B, n_items, n_thresholds = logits.shape

    thresholds = torch.arange(1, n_thresholds + 1, device=logits.device).float()
    targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()

    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    corn = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")

    if not use_qwk:
        return corn

    probs = torch.softmax(logits, dim=-1)
    N = n_thresholds + 1

    w = torch.zeros(N, N, device=logits.device)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / ((N - 1) ** 2 + 1e-8)

    qwk_total = 0
    for i in range(n_items):
        item_logits = logits[:, i, :]
        item_probs = probs[:, i, :]

        item_labels = labels[:, i].long()
        labels_one_hot = F.one_hot(item_labels, N).float()

        O = torch.matmul(labels_one_hot.t(), item_probs)
        hist_true = labels_one_hot.sum(dim=0)
        hist_pred = item_probs.sum(dim=0)
        E = torch.outer(hist_true, hist_pred) / (B + 1e-8)

        num = (w * O).sum()
        den = (w * E).sum()
        qwk = 1.0 - num / (den + 1e-8)
        qwk_total = qwk_total + (1 - qwk)

    qwk_loss = qwk_total / n_items

    return 0.7 * corn + qwk_weight * qwk_loss


class SubscaleAwareA2Head(nn.Module):
    """
    子量表感知的A2预测头

    按抑郁/焦虑/压力分组预测条目，共享子量表级表征。
    """

    DEPRESSION_ITEMS = [2, 4, 9, 12, 15, 16, 20]
    ANXIETY_ITEMS = [1, 3, 6, 8, 14, 18, 19]
    STRESS_ITEMS = [0, 5, 7, 10, 11, 13, 17]

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        n_thresholds: int = 3,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_thresholds = n_thresholds
        self.label_smoothing = label_smoothing

        self.subscale_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.dep_encoder = nn.Linear(d_in, d_hidden // 2)
        self.anx_encoder = nn.Linear(d_in, d_hidden // 2)
        self.stress_encoder = nn.Linear(d_in, d_hidden // 2)

        d_item = d_hidden + d_hidden // 2 * 3
        self.item_heads = nn.ModuleList([
            nn.Linear(d_item, n_thresholds)
            for _ in range(21)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (B, 21, n_thresholds) 的logits"""
        shared = self.subscale_proj(x)

        dep = F.gelu(self.dep_encoder(x))
        anx = F.gelu(self.anx_encoder(x))
        stress = F.gelu(self.stress_encoder(x))
        subscale_features = torch.cat([dep, anx, stress], dim=1)

        item_features = torch.cat([shared.expand(21, -1, -1).transpose(0, 1),
                                    subscale_features.unsqueeze(1).expand(-1, 21, -1)], dim=-1)

        logits = torch.stack([
            self.item_heads[i](item_features[:, i])
            for i in range(21)
        ], dim=1)

        return logits

    def predict_int_monotonic(self, logits: torch.Tensor) -> torch.Tensor:
        """单调性约束解码"""
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

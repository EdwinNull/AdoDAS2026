"""
本模块定义了两个任务预测头和对应的损失函数。

架构定位：
    MTCNBackbone → (B, d_shared)
                        ├── A1Head        → (B, 3)      抑郁/焦虑/压力 logit
                        └── A2OrdinalHead → (B, 21, 3)  21项 × 3个累积阈值 logit

任务说明：
    A1：多标签二元分类，每个指标独立预测是否超过临床阈值（BCE损失）
    A2：序数回归，将整数等级（0-3）转化为3个累积二元问题（序数BCE损失）
        序数回归的优势：利用等级间的顺序关系，比普通4分类更稳健
        例：真实标签=2 → 目标向量=[1,1,0]（>=1通过, >=2通过, >=3不通过）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):
    """
    A1任务预测头 - 三分类二元分类

    用于预测三个心理健康指标: 抑郁(D)、焦虑(A)、压力(S)
    每个指标独立预测，输出一个概率值

    参数:
        d_in: 输入特征维度
        bias_init: 初始偏置值列表 [bias_D, bias_A, bias_S]
                   建议使用 log(p/(1-p)) 初始化，其中p是训练集正样本率
                   作用：让模型初始输出与数据分布一致，加速收敛

    示例:
        # 假设训练集中 D=15%, A=20%, S=18% 正样本
        bias_init = [log(0.15/0.85), log(0.20/0.80), log(0.18/0.82)]
        head = A1Head(d_in=256, bias_init=bias_init)
    """

    def __init__(self, d_in: int, bias_init: list[float] | None = None) -> None:
        super().__init__()
        # 单层线性：共享表示 → 3个独立logit（每个心理健康指标一个）
        self.fc = nn.Linear(d_in, 3)

        # 用先验正样本率初始化偏置：sigmoid(bias) = p_positive
        # 避免训练初期模型输出 0.5 而真实分布是 0.15，减少早期无效迭代
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 (B, d_in)
        返回:
            logits: 未经过sigmoid的预测值 (B, 3)，对应 [logit_D, logit_A, logit_S]
        注意：返回logit而非概率，因为 binary_cross_entropy_with_logits 内部
              用数值稳定方式合并 sigmoid+BCE，避免极端值精度损失
        """
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        """推理时将logits转换为概率值 (B, 3)，每个值在[0,1]"""
        return torch.sigmoid(logits)


class A2OrdinalHead(nn.Module):
    """
    A2任务预测头 - 序数回归

    预测21个心理评估项目的分数（0/1/2/3四个等级）。

    序数回归原理：
        将"预测整数k"转化为"预测k个累积二元问题"
        分数k → 目标向量 = [score>=1, score>=2, score>=3]
            k=0 → [0, 0, 0]
            k=1 → [1, 0, 0]
            k=2 → [1, 1, 0]
            k=3 → [1, 1, 1]
        三个概率满足单调性：p1 >= p2 >= p3

    参数:
        d_in:         输入特征维度
        n_items:      评估项目数（默认21）
        n_thresholds: 阈值数 = 等级数-1（默认3，对应0-3共4个等级）
    """

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        # 输出 21×3=63 个logit，reshape后每项3个累积阈值logit
        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # (B, 63) → (B, 21, 3)：第i项的第j个阈值logit
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        """
        简单解码：各阈值独立判断后求和
        缺陷：可能产生非单调结果（如[0,1,0]，跳过阈值2却超过阈值3）
        """
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        """
        单调解码（推荐）：强制累积概率满足 p1 >= p2 >= p3

        原理（累积链接模型）：
            p1 = sigmoid(logit_1)
            p2 = min(sigmoid(logit_2), p1)   # 强制单调
            p3 = min(sigmoid(logit_3), p2)

            P(score=0) = 1 - p1
            P(score=1) = p1 - p2
            P(score=2) = p2 - p3
            P(score=3) = p3
            # 四个概率之和=1，满足概率分布归一性

        返回概率最大的等级（argmax）
        """
        s = torch.sigmoid(logits)

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)   # 强制 p2 <= p1
        p3 = torch.min(s[..., 2], p2)   # 强制 p3 <= p2

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        """
        期望解码：计算等级期望值后取整

        E[score] = 0×P0 + 1×P1 + 2×P2 + 3×P3 = p1 + p2 + p3
        （三个累积概率之和等于期望值，是序数回归的数学性质）
        """
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3   # E[score] = p1 + p2 + p3
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
        """
        将整数标签转为序数目标向量（向量化实现）

        例：label=2, thresholds=[1,2,3]
            2>=1 → 1, 2>=2 → 1, 2>=3 → 0  →  [1,1,0]

        参数:
            labels: (B, n_items) 整数标签
        返回:
            targets: (B, n_items, n_thresholds) 二元目标，供BCE损失使用
        """
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        # labels.unsqueeze(-1): (B,I,1)，thresholds.view(1,1,-1): (1,1,T)，广播比较
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets


def a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    A1任务损失：带标签平滑的二元交叉熵

    label_smoothing：将硬标签 {0,1} 软化，防止模型过度自信
        例：smoothing=0.1 → 1→0.95, 0→0.05
    pos_weight：正样本损失权重，缓解类别不平衡
        例：负:正=5:1 → pos_weight=5，使正样本梯度×5
    """
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    A2任务损失：序数回归损失

    本质：对 21×3=63 个独立二元问题求 BCE 的平均
    Step 1：整数标签 (B,21) → 序数目标向量 (B,21,3)
    Step 2：可选标签平滑
    Step 3：BCE with logits（数值稳定版）
    """
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

"""
不确定性加权多任务学习 (Uncertainty Weighting for Multi-Task Learning)

参考: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018

核心思想：
    每个任务的损失权重不是固定的，而是通过可学习的"不确定性"参数自动调整。

    传统 MTL：L_total = w1×L1 + w2×L2 + w3×L3  (w1,w2,w3 需要手动调参)
    不确定性加权：L_total = (1/2σ1²)×L1 + log(σ1) + (1/2σ2²)×L2 + log(σ2) + ...

    σ 是可学习参数（任务的"不确定性"）：
        - σ 大 → 任务不确定性高 → 降低该任务的权重
        - σ 小 → 任务确定性高 → 提高该任务的权重
        - log(σ) 项防止 σ 趋向无穷大（正则化）

优势：
    1. 自动平衡多任务，无需手动调参
    2. 训练过程中动态调整权重
    3. 理论基础：贝叶斯深度学习中的同方差不确定性（homoscedastic uncertainty）
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权多任务损失

    参数:
        n_tasks: 任务数量
        init_log_var: 初始 log(σ²) 值（默认0，即 σ²=1）
    """
    def __init__(self, n_tasks: int, init_log_var: float = 0.0):
        super().__init__()
        self.n_tasks = n_tasks
        # 可学习参数：log(σ²)，用 log 保证 σ² > 0
        self.log_vars = nn.Parameter(torch.full((n_tasks,), init_log_var, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            losses: 各任务的损失列表 [L1, L2, ..., Ln]

        返回:
            total_loss: 加权后的总损失
            weights: 各任务的有效权重（用于日志记录）
        """
        if len(losses) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} losses, got {len(losses)}")

        total_loss = 0.0
        weights = {}

        for i, loss in enumerate(losses):
            # 不确定性加权公式：(1 / 2σ²) × L + log(σ)
            # = (1 / 2exp(log_var)) × L + 0.5 × log_var
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            weighted_loss = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            total_loss = total_loss + weighted_loss

            # 记录有效权重（用于监控）
            weights[f"task_{i}_weight"] = precision.item()
            weights[f"task_{i}_sigma"] = torch.exp(0.5 * self.log_vars[i]).item()

        return total_loss, weights


class MultiTaskHead(nn.Module):
    """
    多任务预测头集合

    包含：
        - 主任务头（A1 或 A2）
        - 会话级任务头
        - 辅助任务头（情绪维度、情感分类、AU预测）
    """
    def __init__(
        self,
        d_in: int,
        task_type: str,  # "a1" or "a2"
        enable_emotion_dims: bool = True,
        enable_emotion_cls: bool = True,
        enable_au_pred: bool = True,
    ):
        super().__init__()
        self.task_type = task_type
        self.enable_emotion_dims = enable_emotion_dims
        self.enable_emotion_cls = enable_emotion_cls
        self.enable_au_pred = enable_au_pred

        # 辅助任务头
        if enable_emotion_dims:
            # 情绪维度预测：valence（愉悦度）和 arousal（激活度）
            # 输出范围 [-1, 1]，使用 tanh 激活
            self.emotion_dim_head = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),  # [valence, arousal]
            )

        if enable_emotion_cls:
            # 基础情感分类：4类（快乐、悲伤、愤怒、中性）
            self.emotion_cls_head = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 4),
            )

        if enable_au_pred:
            # 面部动作单元（AU）预测：12个关键AU的多标签分类
            # AU: 1,2,4,5,6,7,9,12,15,17,20,25
            self.au_head = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 12),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        参数:
            x: (B, d_in) 输入特征

        返回:
            outputs: 各辅助任务的输出字典
        """
        outputs = {}

        if self.enable_emotion_dims:
            outputs["emotion_dims"] = self.emotion_dim_head(x)  # (B, 2)

        if self.enable_emotion_cls:
            outputs["emotion_cls"] = self.emotion_cls_head(x)  # (B, 4)

        if self.enable_au_pred:
            outputs["au_logits"] = self.au_head(x)  # (B, 12)

        return outputs


def compute_auxiliary_losses(
    aux_outputs: dict[str, torch.Tensor],
    aux_targets: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """
    计算辅助任务损失

    参数:
        aux_outputs: 辅助任务预测输出
        aux_targets: 辅助任务真实标签（可能为 None，表示该 batch 无标签）

    返回:
        losses: 各辅助任务的损失字典
    """
    losses = {}

    if aux_targets is None:
        # 无辅助标签时，返回零损失
        for key in aux_outputs:
            losses[key] = aux_outputs[key].new_zeros(())
        return losses

    # 情绪维度损失（MSE）
    if "emotion_dims" in aux_outputs and "emotion_dims" in aux_targets:
        pred = torch.tanh(aux_outputs["emotion_dims"])  # 限制到 [-1, 1]
        target = aux_targets["emotion_dims"]
        mask = ~torch.isnan(target).any(dim=-1)  # 过滤无效标签
        if mask.any():
            losses["emotion_dims"] = nn.functional.mse_loss(
                pred[mask], target[mask]
            )
        else:
            losses["emotion_dims"] = pred.new_zeros(())

    # 情感分类损失（CE）
    if "emotion_cls" in aux_outputs and "emotion_cls" in aux_targets:
        logits = aux_outputs["emotion_cls"]
        target = aux_targets["emotion_cls"].long()
        mask = target >= 0  # -1 表示无标签
        if mask.any():
            losses["emotion_cls"] = nn.functional.cross_entropy(
                logits[mask], target[mask]
            )
        else:
            losses["emotion_cls"] = logits.new_zeros(())

    # AU 预测损失（BCE）
    if "au_logits" in aux_outputs and "au_labels" in aux_targets:
        logits = aux_outputs["au_logits"]
        target = aux_targets["au_labels"].float()
        mask = ~torch.isnan(target).any(dim=-1)
        if mask.any():
            losses["au_pred"] = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], target[mask]
            )
        else:
            losses["au_pred"] = logits.new_zeros(())

    return losses

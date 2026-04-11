"""
数据增强工具

基于AdoDAS 2026技术指南实现的数据增强方法:
- Mixup: 嵌入级Mixup (α=0.2)
- TemporalDropout: 片段级时序Dropout
- FeatureMask: 随机掩码模态特征
- VADWeightedNoise: VAD加权噪声
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixupAugmentation:
    """
    嵌入级Mixup增强

    在嵌入空间中对两个样本进行线性插值，适用于抑郁症检测的多模态特征。
    α=0.2 的beta分布提供了适度的增强，避免过度平滑导致失去诊断信息。
    """

    def __init__(self, alpha: float = 0.2, enabled: bool = True):
        self.alpha = alpha
        self.enabled = enabled

    def __call__(
        self,
        features: dict[str, torch.Tensor],
        labels_a1: torch.Tensor | None = None,
        labels_a2: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict | None]:
        """
        Args:
            features: 特征字典，如 {"audio": tensor, "video": tensor, ...}
            labels_a1: (B, 3) A1标签
            labels_a2: (B, 21) A2标签
        Returns:
            mixed_features: 混合后的特征字典
            mixed_labels: 混合后的标签字典
        """
        if not self.enabled or self.alpha <= 0:
            return features, self._make_labels_dict(labels_a1, labels_a2)

        batch_size = next(iter(features.values())).shape[0]
        if batch_size < 2:
            return features, self._make_labels_dict(labels_a1, labels_a2)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        lam = max(lam, 1 - lam)

        indices = torch.randperm(batch_size, device=batch_size.device)

        mixed_features = {}
        for name, feat in features.items():
            if isinstance(feat, torch.Tensor) and feat.dtype in (torch.float32, torch.float16, torch.bfloat16):
                mixed_features[name] = lam * feat + (1 - lam) * feat[indices]
            else:
                mixed_features[name] = feat

        mixed_labels = self._make_labels_dict(labels_a1, labels_a2)
        if mixed_labels is not None:
            for key in mixed_labels:
                if isinstance(mixed_labels[key], torch.Tensor):
                    mixed_labels[key] = lam * mixed_labels[key] + (1 - lam) * mixed_labels[key][indices]

        return mixed_features, mixed_labels

    def _make_labels_dict(
        self,
        labels_a1: torch.Tensor | None,
        labels_a2: torch.Tensor | None,
    ) -> dict | None:
        if labels_a1 is None and labels_a2 is None:
            return None
        labels = {}
        if labels_a1 is not None:
            labels["a1"] = labels_a1
        if labels_a2 is not None:
            labels["a2"] = labels_a2
        return labels


class TemporalDropout(nn.Module):
    """
    片段级时序Dropout

    随机丢弃整个时间片段，增强模型对不完整会话的鲁棒性。
    与会话丢弃(session_dropout)结合使用效果更好。
    """

    def __init__(
        self,
        drop_prob: float = 0.1,
        min片段长度: int = 10,
        enabled: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.min片段_length = min片段长度
        self.enabled = enabled

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入序列
            mask: (B, T) 有效位置掩码
        Returns:
            (x, updated_mask) dropout后的序列和更新后的掩码
        """
        if not self.enabled or self.drop_prob <= 0 or not self.training:
            return x, mask

        B, T, D = x.shape
        new_mask = mask.clone()

        for b in range(B):
            valid_len = mask[b].sum().item()
            if valid_len < self.min片段_length * 2:
                continue

            n_segments = max(1, int(valid_len / self.min片段_length))
            segment_len = valid_len // n_segments

            for seg_idx in range(n_segments):
                if torch.rand(1).item() < self.drop_prob:
                    start = seg_idx * segment_len
                    end = start + segment_len
                    valid_indices = torch.nonzero(mask[b], as_tuple=True)[0]
                    if len(valid_indices) > end:
                        drop_indices = valid_indices[start:end]
                        x[b, drop_indices] = 0
                        new_mask[b, drop_indices] = False

        return x, new_mask


class FeatureMaskAugmentation:
    """
    随机特征掩码增强

    随机将特征维度置零，增强模型对缺失模态/特征的鲁棒性。
    实现10%的随机特征掩码。
    """

    def __init__(
        self,
        mask_prob: float = 0.1,
        modality_specific: bool = True,
        enabled: bool = True,
    ):
        self.mask_prob = mask_prob
        self.modality_specific = modality_specific
        self.enabled = enabled

    def __call__(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: 特征字典，如 {"audio/mel_mfcc": tensor, "video/vision_ssl_embed": tensor, ...}
        Returns:
            masked_features: 掩码后的特征字典
        """
        if not self.enabled or self.mask_prob <= 0:
            return features

        masked_features = {}

        for name, feat in features.items():
            if not isinstance(feat, torch.Tensor) or feat.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                masked_features[name] = feat
                continue

            B, T, D = feat.shape
            mask = torch.rand(B, T, 1, device=feat.device) > self.mask_prob
            masked_features[name] = feat * mask.float()

        return masked_features


class VADWeightedAugmentation:
    """
    VAD加权数据增强

    根据语音活动检测(VAD)信号对特征进行加权增强，
    突出语音活跃区域的特征，弱化沉默区域。
    """

    def __init__(
        self,
        vad_weight_high: float = 1.2,
        vad_weight_low: float = 0.8,
        vad_threshold: float = 0.5,
        enabled: bool = True,
    ):
        self.vad_weight_high = vad_weight_high
        self.vad_weight_low = vad_weight_low
        self.vad_threshold = vad_threshold
        self.enabled = enabled

    def __call__(
        self,
        features: dict[str, torch.Tensor],
        vad: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: 音频特征字典
            vad: (B, T) VAD信号
        Returns:
            enhanced_features: 增强后的特征字典
        """
        if not self.enabled or vad is None:
            return features

        vad_expanded = vad.unsqueeze(-1)

        high_mask = (vad_expanded >= self.vad_threshold).float()
        low_mask = 1 - high_mask

        weights = high_mask * self.vad_weight_high + low_mask * self.vad_weight_low

        enhanced_features = {}
        for name, feat in features.items():
            if "audio" in name.lower() and isinstance(feat, torch.Tensor):
                enhanced_features[name] = feat * weights
            else:
                enhanced_features[name] = feat

        return enhanced_features


class SpecAugment(nn.Module):
    """
    SpecAugment风格的时频掩码

    在时间和频率维度上应用掩码，源自语音识别领域的成功实践。
    适用于频谱类特征（如mel/mfcc）。
    """

    def __init__(
        self,
        time_mask_param: int = 20,
        freq_mask_param: int = 8,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        enabled: bool = True,
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 频谱特征
        Returns:
            masked_x: 掩码后的特征
        """
        if not self.enabled or not self.training:
            return x

        B, T, D = x.shape
        x = x.clone()

        for b in range(B):
            for _ in range(self.n_time_masks):
                t = min(self.time_mask_param, T - 1)
                if t <= 0:
                    continue
                t0 = torch.randint(0, T - t, (1,)).item()
                x[b, t0:t0 + t, :] = 0

            for _ in range(self.n_freq_masks):
                d = min(self.freq_mask_param, D - 1)
                if d <= 0:
                    continue
                d0 = torch.randint(0, D - d, (1,)).item()
                x[b, :, d0:d0 + d] = 0

        return x


class ComposeAugmentations:
    """
    组合多个数据增强方法
    """

    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(
        self,
        features: dict[str, torch.Tensor],
        labels_a1: torch.Tensor | None = None,
        labels_a2: torch.Tensor | None = None,
        vad: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            features: 特征字典
            labels_a1: A1标签
            labels_a2: A2标签
            vad: VAD信号
            mask: 掩码
        Returns:
            增强后的数据和标签
        """
        result = {"features": features, "labels": {}}

        if labels_a1 is not None:
            result["labels"]["a1"] = labels_a1
        if labels_a2 is not None:
            result["labels"]["a2"] = labels_a2

        current_features = features
        current_labels_a1 = labels_a1
        current_labels_a2 = labels_a2

        for aug in self.augmentations:
            if isinstance(aug, MixupAugmentation):
                current_features, mixed_labels = aug(
                    current_features, current_labels_a1, current_labels_a2
                )
                if mixed_labels is not None:
                    current_labels_a1 = mixed_labels.get("a1")
                    current_labels_a2 = mixed_labels.get("a2")
            elif isinstance(aug, FeatureMaskAugmentation):
                current_features = aug(current_features)
            elif isinstance(aug, VADWeightedAugmentation):
                current_features = aug(current_features, vad)
            elif isinstance(aug, SpecAugment):
                for name in current_features:
                    if "mel" in name.lower() or "mfcc" in name.lower():
                        current_features[name] = aug(current_features[name])
            elif isinstance(aug, TemporalDropout) and mask is not None:
                for name in current_features:
                    if isinstance(current_features[name], torch.Tensor):
                        feat, new_mask = aug(current_features[name], mask)
                        current_features[name] = feat
                        mask = new_mask

        result["features"] = current_features
        result["labels"]["a1"] = current_labels_a1
        result["labels"]["a2"] = current_labels_a2

        return result


def create_augmentation_pipeline(
    mixup_alpha: float = 0.2,
    feature_mask_prob: float = 0.1,
    temporal_dropout_prob: float = 0.1,
    use_specaugment: bool = True,
    use_vad_weighting: bool = False,
) -> ComposeAugmentations:
    """
    创建数据增强流水线

    Args:
        mixup_alpha: Mixup的alpha参数
        feature_mask_prob: 特征掩码概率
        temporal_dropout_prob: 时序dropout概率
        use_specaugment: 是否使用SpecAugment
        use_vad_weighting: 是否使用VAD加权
    Returns:
        组合后的增强流水线
    """
    augmentations = []

    if mixup_alpha > 0:
        augmentations.append(MixupAugmentation(alpha=mixup_alpha))

    if feature_mask_prob > 0:
        augmentations.append(FeatureMaskAugmentation(mask_prob=feature_mask_prob))

    if temporal_dropout_prob > 0:
        augmentations.append(TemporalDropout(drop_prob=temporal_dropout_prob))

    if use_specaugment:
        augmentations.append(SpecAugment())

    if use_vad_weighting:
        augmentations.append(VADWeightedAugmentation())

    return ComposeAugmentations(augmentations)

# A2 配置对比：default vs full

## 共享部分

两套配置使用完全相同的模型架构、特征选择、损失函数和数据增强。

### 特征选择

| 模态 | 特征组 | SSL 模型 |
|------|--------|----------|
| 音频 | mel_mfcc, vad, egemaps, ssl_embed | chinese-hubert-large |
| 视频 | headpose_geom, face_behavior, qc_stats, vad_agg, body_pose, global_motion, vision_ssl_embed | vit-mae-base |

掩码策略 `and_core`：音频必须同时有 mel_mfcc + ssl_embed，视频必须同时有 vision_ssl_embed + qc_stats。

### 模型架构

```
GroupAdapter (feat→d_adapter) → ModalityFusion (concat→d_model)
  → 6层 TCN (kernel=3, dilated causal)
  → ASP 池化 (VAD/QC 偏置注意力)
  → Fusion MLP (会话级表示)
  → ParticipantAggregator (MLP 聚合 4 会话)
  → CORAL head (21 条目 × 3 阈值)
```

| 参数 | 值 |
|------|-----|
| d_adapter | 64 |
| d_model | 256 |
| tcn_layers | 6 |
| tcn_kernel_size | 3 |
| dropout | 0.2 |
| d_shared | 256 |
| aggregator | mlp |
| aux_embed_dim | 8 |

### 损失函数

| 组件 | 说明 |
|------|------|
| 序数 BCE | 标准序数回归二元交叉熵 (pos_weight) |
| CORN 条件序数损失 | 条件概率链 P(Y≥k\|Y≥k-1)，保证单调性 |
| 可微 QWK 辅助损失 | soft confusion matrix → QWK 近似，weight=0.3 |
| label_smoothing | 0.05 |

### 训练超参数

| 参数 | 值 |
|------|-----|
| batch_size | 64 |
| lr | 0.001 |
| weight_decay | 0.01 |
| warmup_epochs | 3 |
| epochs | 20 |
| patience | 6 |
| early_stop_metric | primary (QWK) |
| early_stop_min_delta | 0.005 |
| amp | true |
| grad_clip | 1.0 |
| seed | 325799 |

### 数据增强与正则化

| 参数 | 值 |
|------|-----|
| label_smoothing | 0.05 |
| feature_noise_std | 0.01 |
| session_drop_prob | 0.1 |

### LUPI

| 参数 | 值 |
|------|-----|
| aux_lupi.enabled | false（通过 CLI --lupi 独立开启） |

---

## full 独有改动

full 在 default 之上有 **5 个增量**：

### 1. 多任务学习 + 不确定性加权

**这是最核心的区别。** default 只有主任务一个优化目标。full 将训练变为 4 任务联合优化：

| 任务 | 损失 | 说明 |
|------|------|------|
| task 0 | 参与者级主任务 | A2 序数回归 (CORN + QWK) |
| task 1 | 会话级主任务 | 同上，在会话层面 |
| task 2 | 会话类型分类 | 4 类 CE (A01/B01/B02/B03) |
| task 3 | emotion_dims | valence/arousal MSE，弱正则化 |

4 个损失通过 **Kendall 不确定性加权** (`use_uncertainty_weighting: true`) 自动平衡：

```
L_total = Σ [exp(-s_i) * L_i + s_i]
```

其中 `s_i = log(σ²)` 是可学习参数。σ 大 → 不确定性高 → 权重低；σ 小 → 确定性高 → 权重高。

**Per-task log_var 钳制** (S1.2)，防止训练末期的权重失衡：

| 任务 | 钳制 | 效果 |
|------|------|------|
| task 0 (主) | max=0.0 | precision ≥ 1.0，主任务权重不退化 |
| task 1-3 (辅) | min=-0.5 | precision ≤ 1.65，辅助权重不膨胀 |

### 2. 辅助损失权重调整

| 配置 | default | full (UW 关闭时) |
|------|---------|-------------------|
| session_loss_weight | 0.3 | 0.25 |
| session_type_loss_weight | 0.1 | 0.05 |
| emotion_dims_weight | — | 0.05 |

UW 开启时权重由可学习参数自动调节。若关闭 UW，full 用更低的固定权重，减少辅助任务干扰。

### 3. GPU 显存预占

```
gpu_prealloc_gb: 27
```

启动时分配 27GB VRAM 后释放，CUDA 缓存保留内存池，防止训练中途被其他进程抢占 OOM。default 无此机制。

### 4. 跨模态注意力（可用但默认关闭）

```
use_cross_modal: false
cm_n_heads: 1
```

通过 `--extra "--use_cross_modal 1"` 开启。TCN 输出后、ASP 池化前插入双向 A↔V cross-attention + learnable gate。default 无从配置。

### 5. 训练后自动推理

```
run_inference_after_train: true
```

训练结束后自动加载 best checkpoint，对 val 和 test_hidden 生成 submission CSV。default 无。

### 其他细微差异

| 配置 | default | full |
|------|---------|------|
| preload_workers | 16 | 4 |

---

## 一句话总结

> **default = 纯主任务单目标训练。full = 4 任务联合训练，不确定性加权自动平衡，per-task 钳制防退化，辅助任务做弱正则化。**
> 两者用相同的模型、特征和损失函数。full 的增量集中在"怎么训"（多任务平衡）而非"用什么训"（架构/特征）。

## 消融实验结果
### default
2026-05-28 11:00:09 [INFO] Training complete. Best QWK=0.4389, time=28m 41s
2026-05-28 11:00:09 [INFO] Loading best checkpoint for submission generation ...
2026-05-28 11:00:09 [INFO] Submission level: participant
2026-05-28 11:00:09 [INFO] Decode method: auto
2026-05-28 11:00:09 [INFO] Calibrating and selecting A2 decode strategy on val_select ...
2026-05-28 11:00:21 [INFO]   A2 decode comparison on val_select:
2026-05-28 11:00:21 [INFO]     argmax                 QWK=0.4389 MAE=0.3896 | 0=72.2% 1=18.4% 2=7.9% 3=1.5%
2026-05-28 11:00:21 [INFO]     monotonic              QWK=0.3273 MAE=0.4576 | 0=90.1% 1=0.0% 2=0.0% 3=9.9%
2026-05-28 11:00:21 [INFO]     expectation            QWK=0.2246 MAE=0.6852 | 0=17.9% 1=72.4% 2=9.6% 3=0.1%
2026-05-28 11:00:21 [INFO]     calibrated_argmax      QWK=0.4629 MAE=0.3846 | 0=71.8% 1=17.7% 2=8.8% 3=1.6%
2026-05-28 11:00:21 [INFO]     calibrated_monotonic   QWK=0.4057 MAE=0.5275 | 0=83.5% 1=0.0% 2=0.0% 3=16.5%
2026-05-28 11:00:21 [INFO]     calibrated_expectation QWK=0.4148 MAE=0.3415 | 0=71.3% 1=27.0% 2=1.6% 3=0.0%
2026-05-28 11:00:21 [INFO]   Selected A2 strategy (val_select): calibrated_argmax (decode=argmax, QWK=0.4629, MAE=0.3846)
2026-05-28 11:00:28 [INFO]   [Honest] val_holdout: raw QWK=0.4786, calibrated QWK=0.4777, MAE=0.3893
2026-05-28 11:00:28 [INFO]   [Honest] val_holdout distribution: 0=70.2% 1=18.3% 2=10.4% 3=1.2%
2026-05-28 11:00:28 [INFO] Skipping submission generation after training; use infer.py for release inference.
2026-05-28 11:00:28 [INFO] Run complete: a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_102315
2026-05-28 11:00:28 [INFO] Output dir: output/runs/a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_102315

### default+lupi
2026-05-28 11:31:06 [INFO] Training complete. Best QWK=0.4254, time=21m 03s
2026-05-28 11:31:06 [INFO] Loading best checkpoint for submission generation ...
2026-05-28 11:31:07 [INFO] Submission level: participant
2026-05-28 11:31:07 [INFO] Decode method: auto
2026-05-28 11:31:07 [INFO] Calibrating and selecting A2 decode strategy on val_select ...
2026-05-28 11:31:17 [INFO]   A2 decode comparison on val_select:
2026-05-28 11:31:17 [INFO]     argmax                 QWK=0.4254 MAE=0.3970 | 0=67.4% 1=24.0% 2=7.8% 3=0.8%
2026-05-28 11:31:17 [INFO]     monotonic              QWK=0.2814 MAE=0.4557 | 0=91.1% 1=0.0% 2=0.0% 3=8.9%
2026-05-28 11:31:17 [INFO]     expectation            QWK=0.2272 MAE=0.6627 | 0=20.5% 1=70.7% 2=8.8% 3=0.0%
2026-05-28 11:31:17 [INFO]     calibrated_argmax      QWK=0.4510 MAE=0.3771 | 0=70.8% 1=20.8% 2=8.1% 3=0.3%
2026-05-28 11:31:17 [INFO]     calibrated_monotonic   QWK=0.3927 MAE=0.5477 | 0=82.8% 1=0.0% 2=0.0% 3=17.2%
2026-05-28 11:31:17 [INFO]     calibrated_expectation QWK=0.4005 MAE=0.3377 | 0=72.2% 1=27.0% 2=0.7% 3=0.0%
2026-05-28 11:31:17 [INFO]   Selected A2 strategy (val_select): calibrated_argmax (decode=argmax, QWK=0.4510, MAE=0.3771)
2026-05-28 11:31:24 [INFO]   [Honest] val_holdout: raw QWK=0.4659, calibrated QWK=0.4654, MAE=0.3695
2026-05-28 11:31:24 [INFO]   [Honest] val_holdout distribution: 0=70.4% 1=21.1% 2=8.2% 3=0.2%
2026-05-28 11:31:24 [INFO] Skipping submission generation after training; use infer.py for release inference.
2026-05-28 11:31:24 [INFO] Run complete: a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_110130
2026-05-28 11:31:24 [INFO] Output dir: output/runs/a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_110130

### full
2026-05-28 12:07:23 [INFO] Training complete. Best QWK=0.4362, time=26m 35s
2026-05-28 12:07:23 [INFO] Loading best checkpoint for submission generation ...
2026-05-28 12:07:23 [INFO] Submission level: participant
2026-05-28 12:07:23 [INFO] Decode method: auto
2026-05-28 12:07:23 [INFO] Calibrating and selecting A2 decode strategy on val_select ...
2026-05-28 12:07:35 [INFO]   A2 decode comparison on val_select:
2026-05-28 12:07:35 [INFO]     argmax                 QWK=0.4362 MAE=0.3965 | 0=69.1% 1=20.8% 2=9.0% 3=1.1%
2026-05-28 12:07:35 [INFO]     monotonic              QWK=0.3070 MAE=0.4673 | 0=89.4% 1=0.0% 2=0.0% 3=10.6%
2026-05-28 12:07:35 [INFO]     expectation            QWK=0.2388 MAE=0.6724 | 0=18.6% 1=71.0% 2=10.4% 3=0.0%
2026-05-28 12:07:35 [INFO]     calibrated_argmax      QWK=0.4574 MAE=0.3810 | 0=71.2% 1=18.9% 2=9.1% 3=0.9%
2026-05-28 12:07:35 [INFO]     calibrated_monotonic   QWK=0.4018 MAE=0.5261 | 0=83.8% 1=0.0% 2=0.0% 3=16.2%
2026-05-28 12:07:35 [INFO]     calibrated_expectation QWK=0.4020 MAE=0.3383 | 0=72.0% 1=27.2% 2=0.8% 3=0.0%
2026-05-28 12:07:35 [INFO]   Selected A2 strategy (val_select): calibrated_argmax (decode=argmax, QWK=0.4574, MAE=0.3810)
2026-05-28 12:07:41 [INFO]   [Honest] val_holdout: raw QWK=0.4767, calibrated QWK=0.4851, MAE=0.3705
2026-05-28 12:07:41 [INFO]   [Honest] val_holdout distribution: 0=69.2% 1=20.8% 2=9.4% 3=0.6%
2026-05-28 12:07:41 [INFO] Skipping submission generation after training; use infer.py for release inference.
2026-05-28 12:07:41 [INFO] Run complete: a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_113220
2026-05-28 12:07:41 [INFO] Output dir: output/runs/a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_113220

### full+lupi
2026-05-28 12:35:18 [INFO] Training complete. Best QWK=0.4342, time=19m 00s
2026-05-28 12:35:18 [INFO] Loading best checkpoint for submission generation ...
2026-05-28 12:35:18 [INFO] Submission level: participant
2026-05-28 12:35:18 [INFO] Decode method: auto
2026-05-28 12:35:18 [INFO] Calibrating and selecting A2 decode strategy on val_select ...
2026-05-28 12:35:28 [INFO]   A2 decode comparison on val_select:
2026-05-28 12:35:28 [INFO]     argmax                 QWK=0.4342 MAE=0.3880 | 0=71.4% 1=19.4% 2=8.1% 3=1.1%
2026-05-28 12:35:28 [INFO]     monotonic              QWK=0.2938 MAE=0.4648 | 0=90.4% 1=0.0% 2=0.0% 3=9.6%
2026-05-28 12:35:28 [INFO]     expectation            QWK=0.2857 MAE=0.5996 | 0=28.4% 1=62.1% 2=9.5% 3=0.0%
2026-05-28 12:35:28 [INFO]     calibrated_argmax      QWK=0.4494 MAE=0.3995 | 0=69.4% 1=20.0% 2=9.3% 3=1.3%
2026-05-28 12:35:28 [INFO]     calibrated_monotonic   QWK=0.3917 MAE=0.5451 | 0=82.7% 1=0.0% 2=0.0% 3=17.3%
2026-05-28 12:35:28 [INFO]     calibrated_expectation QWK=0.4103 MAE=0.3546 | 0=67.0% 1=30.7% 2=2.3% 3=0.0%
2026-05-28 12:35:28 [INFO]   Selected A2 strategy (val_select): calibrated_argmax (decode=argmax, QWK=0.4494, MAE=0.3995)
2026-05-28 12:35:35 [INFO]   [Honest] val_holdout: raw QWK=0.4613, calibrated QWK=0.4624, MAE=0.3993
2026-05-28 12:35:35 [INFO]   [Honest] val_holdout distribution: 0=66.5% 1=22.5% 2=9.9% 3=1.1%
2026-05-28 12:35:35 [INFO] Skipping submission generation after training; use infer.py for release inference.
2026-05-28 12:35:35 [INFO] Run complete: a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_120841
2026-05-28 12:35:35 [INFO] Output dir: output/runs/a2__grouped__coral__a-base-mel_mfcc+vad+egemaps__a-ssl-chinese-hubert-large__v-base-headpose+facebeh+qc+vadagg+bodypose+globmot__v-ssl-vit-mae-base__mask-andcore__pw_pwthr_autodecode_thrcalib__seed325799__20260528_120841

### 实际赛事方评测结果
- default: 0.0826
- default+lupi: 0.1219
- full: 0.1230
- full+lupi: 0.1431
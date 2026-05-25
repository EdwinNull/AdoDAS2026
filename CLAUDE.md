# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADODAS2026 Baseline - Official baseline implementation for the ADODAS grand challenge (ACMMM 2026). A multimodal deep learning system for predicting depression, anxiety, and stress from audio-visual behavioral data.

**Tasks:**
- Track A1: Binary classification for Depression/Anxiety/Stress (3 labels)
- Track A2: Ordinal regression for 21 DASS-21 items (scores 0-3)

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f envs/adodas.yaml
conda activate adodas
```

## Quick Start

### 启动脚本（推荐）

```bash
# 查看帮助
./run_train.sh --help

# A2 基线训练
./run_train.sh --task a2 --preset default

# A2 MTL + LUPI Phase1+2 同时启用
./run_train.sh --task a2 --preset default --lupi p1+p2

# A1 快速调试（小模型, 2 epochs, 验证 pipeline）
./run_train.sh --task a1 --preset debug

# 自定义超参
./run_train.sh --task a2 --preset default --extra "--batch_size 32 --lr 0.0005"
```

预设说明：
| preset | 配置 | 用途 |
|--------|------|------|
| `default` | `tasks/{task}/default.yaml` | 基线训练 |
| `phase1` | `tasks/{task}/phase1_optimization.yaml` | MTL + 辅助任务 + 增强损失 |
| `debug` | default + 覆盖参数 | 100人, 流式加载, 2 epochs 秒级启动 |

`--lupi` 模式：
| 值 | 效果 |
|----|------|
| `p1` | 启用 Phase 1: 辅助属性预测头 |
| `p2` | 启用 Phase 2: 样本一致性加权 |
| `p1+p2` | 同时启用 Phase 1 + 2 |

### 直接使用 train.py

```bash
# A2 训练
python train.py --task a2 --config tasks/a2/default.yaml

# A2 + LUPI Phase1+2 (CLI 覆盖)
python train.py --task a2 --config tasks/a2/default.yaml \
    --aux_lupi_enabled 1 --aux_lupi_phase1 1 --aux_lupi_phase2 1

# A1 训练
python train.py --task a1 --config tasks/a1/default.yaml
```

### 推理

```bash
# 测试集推理
python infer.py --task a2 --checkpoint <path_to_best.pt> --split test_hidden

# 指定输出
python infer.py --task a2 --checkpoint <path_to_best.pt> --output predictions.csv
```

### HDF5 打包（加速数据加载）

```bash
# 全量一键打包（推荐）
./scripts/pack_all.sh

# 或按 split 分别打包
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_packed.h5

# Debug 子集打包（仅 100 人）
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_debug.h5 \
  --max-participants 100
```

打包完成后在 YAML 中设置 `use_hdf5: true` 即可使用。可通过 `ADODAS_DATA_ROOT` 和 `ADODAS_HDF5_DIR` 环境变量指定数据位置。

### 测试

```bash
python test_mtl_integration.py
python test_phase1_optimization.py
```

## Architecture Overview

### Data Flow: Participant → Sessions → Features → Model → Predictions

1. **Participant-level grouping**: Each participant has 4 sessions (A01, B01, B02, B03)
2. **Session-level features**: Audio (mel_mfcc, VAD, eGeMaps, SSL embeddings) + Video (headpose, face behavior, vision SSL)
3. **Flat batch processing**: All sessions flattened into one batch, then reshaped for aggregation
4. **Hierarchical modeling**: Session-level backbone → Participant-level aggregation → Task heads

### Key Modules

**common/data/**
- `dataset.py`: Base feature loading, alignment to 100ms grid
- `grouped_dataset.py`: Participant-level dataset, groups 4 sessions per sample
- `hdf5_dataset.py`: Fast HDF5-backed dataset for packed features
- `feature_io.py`: Low-level feature file I/O (npz, parquet)

**common/models/**
- `mtcn_backbone.py`: Multimodal Temporal Convolutional Network
  - GroupAdapter: Aligns heterogeneous features to d_adapter
  - ModalityFusion: Fuses multiple feature groups within modality
  - TCN: Dilated causal convolution for temporal modeling (感受野指数增长)
  - ASP: Attentive Statistics Pooling (weighted mean + std, considers VAD/QC)
- `grouped_model.py`: Participant-level model
  - ParticipantAggregator: Aggregates 4 sessions → 1 participant repr (mean/mlp/attention)
  - SessionTypeClassifier: Auxiliary task for session type prediction
  - CORALHead: Ordinal regression head with learnable thresholds
- `heads.py`: Task-specific prediction heads
  - A1Head, A2OrdinalHead: Task heads
  - AuxAttributeHeads: LUPI Phase 1 — predicts 5 aux attributes from participant_repr
  - Losses: ASL, Soft-F1, CORN, QWK, aux_attribute_loss
- `aux_encoder.py`: Encodes 5 auxiliary attributes as input features (embeddings)
- `mtl_uncertainty.py`: Uncertainty-weighted multi-task learning
- `phase1_integration.py`: Optimized model wrapper (MTL + auxiliary tasks + aux_logits pass-through)

**common/runner.py**: Main training/validation loop, checkpoint management, submission generation

**common/utils/**
- `metrics.py`: F1, AUROC, QWK, MAE evaluation
- `ckpt.py`: Checkpoint save/load
- `seed.py`: Reproducibility utilities

### Configuration System

YAML configs in `tasks/{a1,a2}/`:
- `feature_selection`: Which audio/video features to use, SSL model tags
- `mask_policy`: How to handle missing modalities (or/and_core/require_k)
- Model hyperparameters: d_adapter, d_model, tcn_layers, dropout, etc.
- Training: batch_size, lr, epochs, warmup, grad_clip, patience
- Auxiliary attributes: `use_aux_attrs`, `aux_embed_dim`
- Loss functions: `use_combined_loss` (ASL+Soft-F1 for A1), `use_corn_loss`, `use_qwk_aux` (for A2)
- Aggregation: `aggregator` (mean/mlp/attention), `session_loss_weight`, `session_type_loss_weight`
- LUPI: `aux_lupi` block — auxiliary attribute supervision + sample reweighting

### LUPI 配置 (`aux_lupi`)

```yaml
aux_lupi:
  enabled: true              # 总开关, false 时完全回退到 baseline
  phase1_mtl:                # Phase 1: 辅助属性多任务监督
    enabled: true            #   从 participant_repr 预测 5 个辅助属性
    hidden: 64
    weights:                 #   各类别损失权重 (总应为主任务 1/3 ~ 1/2)
      aux_family: 0.05
      aux_only_child: 0.05
      aux_favoritism: 0.05
      aux_academic: 0.15
      aux_emotional: 0.20
  phase2_reweight:           # Phase 2: 样本一致性加权
    enabled: false           #   基于 aux_emotional vs DASS 标签一致性
    method: emotional_consistency
    weight_low: 0.7          #   冲突样本降权 (可能是错标)
    weight_high: 1.2         #   一致样本加权
```

CLI 覆盖参数：
- `--aux_lupi_enabled 1` — 总开关
- `--aux_lupi_phase1 1` — Phase 1
- `--aux_lupi_phase2 1` — Phase 2

Phase 1 和 Phase 2 可独立启用，不互相依赖。所有 LUPI 改动遵循 LUPI 范式：训练时使用辅助属性作为监督信号，推理时不依赖辅助属性。

### Auxiliary Attributes (5 categorical features)

1. 家庭结构 (Family structure): 6 classes (1-6)
2. 独生子女 (Only child): 2 classes (0-1)
3. 父母偏爱 (Parental favoritism): 3 classes (1-3)
4. 成绩变动 (Academic performance change): 3 classes (1-3)
5. 情绪变动 (Emotional state change): 3 classes (1-3)

Encoded via `AuxiliaryAttributeEncoder` with embedding layers, concatenated to participant representation.

### Session Types (4 types)

- A01: Standardized reading passage (北风和太阳)
- B01: Describe yesterday
- B02: Happiest memory from past week
- B03: Saddest memory from past week

## Output Structure

```
<output_dir>/runs/<run_name>/
├── logs/              # Training logs
├── checkpoints/       # best.pt, last.pt
├── calibration/       # Calibration curves (if enabled)
└── submissions/       # CSV predictions for submission
```

## Development Notes

### Data Paths

数据根目录: `/data1/AdoDas`

```
/data1/AdoDas/
├── Train/train.csv              # 训练集 manifest
├── Train/train/train/           # 训练集特征 (SCH_xxx/CLS_xxx/P_xxx/...)
├── Val/val.csv                  # 验证集 manifest
├── Val/val/val/                 # 验证集特征
├── Test/test/test_hidden/       # 测试集特征 (无 manifest, 无标签)
└── output/                      # 训练输出
    └── runs/<run_name>/
        ├── logs/ checkpoints/ calibration/ submissions/
```

逻辑 split 名 → 实际子路径映射 (`SPLIT_DATA_PATH` in `common/data/dataset.py`):
- `train` → `Train/train/train`
- `val` → `Val/val/val`
- `test_hidden` → `Test/test/test_hidden`

### Feature Directory Structure
```
<data_root>/<SPLIT_DATA_PATH[split]>/<anon_school>/<anon_class>/<anon_pid>/
├── audio/
│   ├── mel_mfcc/<session>/sequence.npz
│   ├── vad/<session>/sequence.npz
│   ├── ssl_embed/<audio_ssl_model_tag>/<session>/sequence.npz
│   └── egemaps/<session>/pooled.parquet
└── video/
    ├── headpose_geom/<session>/sequence.npz
    ├── face_behavior/<session>/sequence.npz
    ├── qc_stats/<session>/sequence.npz
    ├── vad_agg/<session>/sequence.npz
    └── vision_ssl_embed/<video_ssl_model_tag>/<session>/sequence.npz
```

### Key Design Patterns

1. **Flat batch processing**: All participant sessions flattened into one batch for efficient GPU utilization, then reshaped for aggregation
2. **Mask-aware operations**: Handle missing sessions/modalities via boolean masks (session_valid, modality_mask)
3. **Hierarchical loss**: Primary task + session-level auxiliary + session-type classification
4. **Temporal alignment**: All features aligned to 100ms grid via `align_to_grid()`
5. **Dilated TCN**: Exponential dilation (1,2,4,8,...) for large receptive field without parameter explosion

### Common Pitfalls

- **Session dropout**: Only applied during training (`session_drop_prob`), not validation/inference
- **Label format**: A1 uses binary labels, A2 uses ordinal (0-3). Check task type before loss computation.
- **Submission level**: Can be "session" or "participant". Participant-level averages 4 session predictions.
- **Decode method for A2**: "auto" selects best on validation (argmax/expectation/monotonic)
- **LUPI disabled by default**: `aux_lupi.enabled: false` in default.yaml. Must explicitly enable via YAML or CLI `--aux_lupi_enabled 1`.
- **LUPI Phase 1 requires aux_attrs in batch**: `aux_favoritism` has ~35% structural missing (only-child → no favoritism). Loss masking handles this via `valid_mask = targets >= 0`.
- **Phase 2 non-MTL vs MTL**: Phase 2 adds a weighted BCE term (coefficient 0.3) — it does NOT replace the main loss. Enhanced losses (ASL, CORN, QWK) are preserved.
- **Checkpoint compatibility**: `infer.py` loads with `strict=False` to tolerate missing/extra aux_heads keys.

### LUPI Implementation (Learning Using Privileged Information)

训练时辅助属性（5 个类别特征）可用，测试时不可用。两个阶段：

**Phase 1 — 多任务辅助监督** (`common/models/heads.py: AuxAttributeHeads`)
- 从 participant_repr (纯音视频表示) 预测 5 个辅助属性
- 损失：加权 CrossEntropy，自动跳过缺失值（-1 mask）
- 设计原理：迫使 backbone 学习编码辅助属性相关的潜变量

**Phase 2 — 样本一致性加权** (`common/runner.py: _compute_aux_consistency_weight`)
- 利用 `aux_emotional` (情绪变动) 与 DASS 标签的一致性识别可能错标样本
- DASS 阳性 + 情绪变差 → 高权重; DASS 阳性 + 情绪变好 → 低权重
- 在非 MTL 模式下替换主损失为加权 per-sample BCE; MTL 模式下作为附加项

**关键约束**:
1. 推理时不依赖辅助属性 (strict=False loading, aux_logits=None 时跳过)
2. `aux_lupi.enabled: false` 完全回退到 baseline 行为
3. Phase 1/2 独立可切换，不互相依赖

详见 `docs/AUX_LUPI_PLAN.md`。

### Phase 1 Optimizations (feature/auxiliary-attributes branch)

- Uncertainty-weighted multi-task learning (automatic task balancing)
- Auxiliary tasks: emotion dimensions, emotion classification, AU prediction
- Class-balanced loss functions: ASL + Soft-F1 for A1, CORN + QWK-aux for A2
- Auxiliary attribute encoding (5 demographic/behavioral features)

See `docs/phase1/` for detailed documentation.

## Documentation

Extensive documentation in `docs/`:
- `architecture.md`: Detailed system architecture diagrams
- `data/`: Dataset and feature loading internals
- `models/`: Model architecture deep dives
- `optimize/`: Optimization strategies (data quality, augmentation, loss functions, etc.)
- `phase1/`: Phase 1 optimization guides and summaries

## 工作原则

以第一性原理，从原始需求和问题本质出发，不从惯例或模板出发：
1. 不要假设我清楚自己需要什么，动机或目标不清晰时，停下来讨论
2. 目标清晰但路径不是最短时，直接告知我并给出建议
3. 遇到问题追根因，不打补丁。每个决策要能回答为什么
4. 输出说重点，砍掉一切不改变决策的信息
5. 给出专业、严谨、简洁不失精准的回答，不要加上"如果你愿意"、"你要做的是"这些词语

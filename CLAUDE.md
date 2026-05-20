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

## Core Commands

### Training
```bash
# Train A1 (binary classification)
python train.py --task a1 --config tasks/a1/default.yaml

# Train A2 (ordinal regression)
python train.py --task a2 --config tasks/a2/default.yaml

# Train with phase1 optimizations (MTL + auxiliary tasks)
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml
```

### Inference
```bash
# Run inference on test set
python infer.py --task a1 --checkpoint <path_to_best.pt> --split test_hidden

# Specify custom config and output
python infer.py --task a2 --checkpoint <path_to_best.pt> --config <config.yaml> --output predictions.csv
```

### Feature Packing (HDF5 acceleration)
```bash
# Pack training features into HDF5 for faster loading
python scripts/pack_features.py \
  --manifest ../manifests/train.csv \
  --feature-root ../train \
  --split train \
  --output ../train_packed.h5
```

### Testing
```bash
# Run integration tests
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
- `heads.py`: Task-specific prediction heads (A1Head, A2OrdinalHead)
- `aux_encoder.py`: Encodes 5 auxiliary attributes (family structure, only child, etc.)
- `mtl_uncertainty.py`: Uncertainty-weighted multi-task learning
- `phase1_integration.py`: Optimized model with MTL + auxiliary tasks

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

### Feature Directory Structure
```
<feature_root>/<split>/<anon_school>/<anon_class>/<anon_pid>/
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
- **Feature root vs split**: Dataset auto-detects split-specific directories (e.g., `../train` → `../val` for split="val")

### Phase 1 Optimizations (feature/auxiliary-attributes branch)

Recent additions for improved performance:
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

# ADODAS2026 Baseline

Official baseline implementation for the ADODAS grand challenge (ACMMM 2026).

## Tasks

- **Track A1**: three binary labels for Depression / Anxiety / Stress.
- **Track A2**: 21 ordinal item predictions with scores in `{0, 1, 2, 3}`.

## Environment

```bash
conda env create -f envs/adodas.yaml
conda activate adodas
```

Data root defaults to `/data1/AdoDas`. Override via `ADODAS_DATA_ROOT`.

## Quick Start

### Training

```bash
# A2 baseline (default preset)
./run_train.sh --task a2 --preset default

# A2 full MTL (uncertainty weighting + auxiliary tasks)
./run_train.sh --task a2 --preset full

# A1
./run_train.sh --task a1 --preset default

# with LUPI
./run_train.sh --task a2 --preset default --lupi both

# wait for GPU ≥28GB free then auto-launch
./run_train.sh --task a2 --preset full --gpu-wait

# Stage 1 ablation chain (4 experiments, sequential)
python scripts/run_ablation.py --stop-on-error
```

| preset | config | description |
|--------|--------|-------------|
| `default` | `tasks/a2/default.yaml` | single-task training, CORN+QWK loss |
| `full` | `tasks/a2/mtl_full.yaml` | 4-task MTL + uncertainty weighting + per-task log_var clamping |
| `debug` | default + overrides | 100 participants, 2 epochs for pipeline validation |

### Inference

```bash
# pure argmax (default, no calibration — recommended for leaderboard)
python scripts/run_predict_a2.py \
  --run-dir output/runs/<run_name> \
  --output output/pred.csv

# calibrated argmax (opt-in)
python scripts/run_predict_a2.py \
  --run-dir output/runs/<run_name> \
  --output output/pred.csv --calibrate
```

### GPU monitoring

```bash
# wait for GPU free ≥28GB, then run command
python scripts/gpu_monitor_train.py \
  --free-gb 28 --idle-duration 30 \
  --command "./run_train.sh --task a2 --preset default"
```

## Config Presets

Both `default` and `full` share the same model architecture, features, and loss functions. The difference is in training strategy:

| | default | full |
|---|---|---|
| training objective | main task only | 4-task MTL (main + session + session_type + emotion_dims) |
| task balancing | fixed weights | Kendall uncertainty weighting + per-task clamping |
| GPU preallocation | none | 27GB |
| cross-modal attention | n/a | available (default off) |

See `docs/default_vs_full_config.md` for full details.

## Project Structure

```
AdoDAS2026/
├── train.py                     # training entry point
├── run_train.sh                 # unified launcher (presets, LUPI, GPU-wait)
├── common/                      # core library
│   ├── data/                    # dataset, feature I/O, HDF5
│   ├── models/                  # backbone, heads, MTL uncertainty
│   ├── utils/                   # metrics, checkpoint, seed
│   └── runner.py                # training / validation loop
├── scripts/                     # utilities
│   ├── run_predict_a2.py        # A2 inference (submission generation)
│   ├── run_ablation.py          # Stage 1 ablation chain
│   ├── gpu_monitor_train.py     # GPU free-memory monitor
│   ├── create_val_split.py      # S0 val_select / val_holdout split
│   ├── pack_features.py         # HDF5 packing
│   └── pack_all.sh
├── tasks/                       # YAML configs
│   ├── a1/{default,mtl_full}.yaml
│   └── a2/{default,mtl_full}.yaml
├── tests/                       # unit tests
├── docs/                        # documentation
├── splits/                      # val_split_v1.json (Stage 0)
└── envs/                        # conda environment
```

## Output Structure

```
output/runs/<run_name>/
├── logs/
├── checkpoints/    # best.pt, last.pt
├── calibration/    # a2_threshold_offsets_grouped.json
└── submissions/
```

## Session Types

| session | content |
|---------|---------|
| A01 | standardized reading passage ("北风和太阳") |
| B01 | describe yesterday |
| B02 | happiest memory from past week |
| B03 | saddest memory from past week |

## Auxiliary Attributes (LUPI)

5 categorical features available in training CSV, encoded via embeddings, used when `--lupi` is enabled:

1. Family structure (6 classes)
2. Only child status (2 classes)
3. Parental favoritism (3 classes)
4. Academic performance change (3 classes)
5. Emotional state change (3 classes)

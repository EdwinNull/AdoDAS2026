# Stage 0 + Stage 1 Implementation Plan

## Context

根据 `ADODAS2026_unified_implementation_plan.md` 文档要求，对 AdoDAS2026 项目实施 Stage 0（诚实评估地基）和 Stage 1（训练流程稳定化）。当前无 GPU 可训练，仅做代码修改，验证延后。

核心诊断：当前 val 集 600 人同时承担选 checkpoint、选解码策略、拟合校准阈值三种角色，已被污染；训练在 epoch 7-16 已到达 QWK 峰值，后续 24 个 epoch 做负功；不确定性加权使主任务权重被压到 0.7、辅助任务权重膨胀到 5.0。

---

## Gap Analysis: 文档要求 vs 当前代码

### Stage 0 — 诚实评估地基

| 子项 | 文档要求 | 当前状态 | 需改动 |
|------|---------|---------|--------|
| S0.1 val 二次切分 | 600人→400(val_select)+200(val_holdout)，participant级分层抽样 | 无切分，整份 val 集一用到底 | **新建切分脚本 + 修改 data loading** |
| S0.2 校准数据源切换 | 阈值校准在 val_select 拟合，val_holdout 上评估 | 校准在整份 val 上做 (`runner.py:1633-1744`) | **修改校准逻辑** |
| S0.3 日志规范 | 每 epoch 同时输出 val_select_qwk 和 val_holdout_qwk | 只输出单一 val QWK (`runner.py:1552-1556`) | **修改日志输出** |

### Stage 1 — 训练流程稳定化

| 子项 | 文档要求 | 当前状态 | 需改动 |
|------|---------|---------|--------|
| S1.1 早停+epochs+LR同步 | epochs=16-20, patience=5-8, warmup=3, cosine T_max 同步到 epochs-warmup | default.yaml: epochs=25, patience=6, warmup=3 (OK); mtl_full.yaml: epochs=20(OK), patience=5 | **default.yaml epochs 25→20；mtl_full.yaml patience 5→6** |
| S1.2 不确定性加权约束 | 主任务权重≥0.85，辅助权重≤2.0。方案A: per-task sigma 下限钳制 | `uw_log_var_clamp=1.0` 对所有任务统一钳制，主任务权重最低可到 0.37 | **mtl_uncertainty.py 增加 per-task clamping** |
| S1.3 辅助权重下调 | session_loss: 0.2-0.3, session_type_loss: 0.05-0.1, emotion_dims: 0.05-0.1 | default: session_loss=0.5, session_type=0.15; mtl_full: session_loss=0.25, session_type=0.08, emotion_dims=0.05 | **default.yaml 下调两个权重；mtl_full.yaml 微调 session_type** |
| S1.4 pos_weight 上限 | 从 10.0→5.0 | 已是 5.0 (`runner.py:377`) | **无需改动** ✓ |
| S1.5 QWK aux 损失核查 | 确认 qwk_weight 乘到了 loss 上，显式打印 aux_qwk_loss | 已实现并连接 (`heads.py:424-481`)，但 loss_components 未在非 MTL 模式传递 | **在 train_one_epoch_grouped 中传递 loss_components，增加日志** |

---

## Implementation Steps

### Step 1: 创建 val 切分脚本 `scripts/create_val_split.py`

- 读取 `/data1/AdoDas/Val/val.csv`
- 按 `anon_pid` 分组（participant level）
- 按 DASS-21 总分 + 抑郁/焦虑/压力 subscores 分层抽样
- 400/200 分割，固定 seed=42
- 输出 `splits/val_split_v1.json`

### Step 2: 修改 `common/data/grouped_dataset.py`

- 新增 `valid_pids` 构造参数，使 `GroupedParticipantDataset` 支持 PID 过滤
- 同样修改 `HDF5GroupedDataset` 支持 PID 过滤

### Step 3: 修改 `common/runner.py` — val_select/val_holdout 双验证

- 加载 `splits/val_split_v1.json`
- 创建 val_select 和 val_holdout 两个 DataLoader
- val_select QWK → 用于 early stopping、best checkpoint 判断
- val_holdout QWK → 仅观察，不参与任何决策
- `collect_val_logits` 改用 val_select DataLoader，增加 val_holdout 上的最终 honest 评估
- run_meta.json 增加 `val_split_version` 和 `val_split_hash` 字段

### Step 4: 修改 `common/runner.py` — 日志规范

epoch 日志格式：
```
Epoch | LR | Train Loss | Val Loss | Q_sel | Q_hout | MAE_sel | MAE_hout | Time
```

### Step 5: 修改 `tasks/a2/default.yaml` — S1.1 + S1.3

- epochs: 25 → 20
- session_loss_weight: 0.5 → 0.3
- session_type_loss_weight: 0.15 → 0.1

### Step 6: 修改 `tasks/a2/mtl_full.yaml` — S1.1 + S1.3

- patience: 5 → 6
- session_type_loss_weight: 0.08 → 0.05
- 新增 `uw_task_log_var_bounds` 配置

### Step 7: 修改 `common/models/mtl_uncertainty.py` — S1.2 per-task 约束

- `UncertaintyWeightedLoss.__init__` 新增 `task_log_var_bounds` 参数
- 新增 `_clamp_log_var()` 方法实现 per-task 钳制
- 对主任务 (task 0): `log_var` 上限钳制在 0.0（保证 precision ≥ 1.0）
- 对辅助任务 (task 1-3): `log_var` 下限钳制在 -0.5（保证 precision ≤ 1.65）

### Step 8: 修改 `common/runner.py` — S1.5 QWK aux loss 日志

- 在 `train_one_epoch_grouped` 中传入 `loss_components` dict 到 `a2_ordinal_loss()`
- 在 detailed losses 日志中输出 `avg_qwk_aux_loss`

### Step 9: 同步 `tasks/a1/default.yaml` 和 `tasks/a1/mtl_full.yaml`

对 A1 副线做对应的 S1.1 调整（epochs、patience、辅助权重下调）。

### Post-Implementation

- runner.py 训后自动推理硬禁用 (`if False:`)
- a2/mtl_full.yaml 移除 `run_inference_after_train`
- README.md 重写
- 项目结构整理

---

## Files Modified (Summary)

| 文件 | 改动类型 | Stage |
|------|---------|-------|
| `scripts/create_val_split.py` | **新建** | S0 |
| `splits/val_split_v1.json` | **新建**（由脚本生成） | S0 |
| `common/data/grouped_dataset.py` | 修改 — 增加 PID 过滤 | S0 |
| `common/data/hdf5_dataset.py` | 修改 — 增加 PID 过滤 | S0 |
| `common/runner.py` | 修改 — 双验证、校准切换、日志更新、QWK aux 日志 | S0, S1 |
| `tasks/a2/default.yaml` | 修改 — epochs, 辅助权重 | S1 |
| `tasks/a2/mtl_full.yaml` | 修改 — patience, session_type_weight, per-task clamp 配置 | S1 |
| `tasks/a1/default.yaml` | 修改 — epochs, 辅助权重 | S1 |
| `tasks/a1/mtl_full.yaml` | 修改 — epochs, patience | S1 |
| `common/models/mtl_uncertainty.py` | 修改 — per-task log_var clamping | S1 |
| `common/models/phase1_integration.py` | 修改 — 传递 per-task clamp 配置 | S1 |

---

## Verification Plan

由于当前无 GPU，验证分为两层：

1. **语法/导入验证**（可立即执行）：
   ```bash
   python -c "from common.runner import main"  # 验证导入无报错
   python scripts/create_val_split.py           # 生成切分文件，检查 JSON 格式
   python -m pytest test_mtl_integration.py -x  # 运行已有单元测试
   ```

2. **训练验证**（需 GPU，延后执行）：
   ```bash
   ./run_train.sh --task a2 --preset default   # 验证 S1 改动后训练不报错
   ```
   验收标准（文档 §2.4, §3.8）：
   - val_holdout QWK 与 val_select QWK gap 在 0.01-0.04
   - 训练在 epoch 12-18 自然停止
   - 主任务权重全程 ≥ 0.85
   - 预测分布末态各类比例接近 GT (0/1/2/3 ≈ 70/22/5/2)

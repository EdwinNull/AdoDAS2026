# ADODAS2026 改进实施统一规划文档（Unified Plan v1.0）

> **适用对象**:实施改进的 agent / 工程师
> **目标项目**:AdoDAS2026 Track A2 baseline(主线)、Track A1 baseline(副线)
> **依据**:三份前置文档的合并整理
> - 训练流程改进指导(`a2_training_modification_guide.md`)
> - 推理与验证策略改进指导(`a2_inference_and_validation_guide.md`)
> - 优化路线指导(`ADODAS2026_optimization_roadmap.md`)
> **本文档不包含具体代码**:所有改动的具体落点、变量名、文件路径由实施方结合项目代码结构自行确定。

---

## 0. 必读:本文档的整体逻辑

### 0.1 三份前置文档的关系

| 前置文档 | 解决的问题层 | 在本文档中的位置 |
|---|---|---|
| 推理与验证策略 | "我们怎么判断改进有效" — 评估方法论 | **Stage 0 + 贯穿全文的验证纪律** |
| 训练流程改进 | "现有训练流程在做负功" — 训练稳定性 | **Stage 1** |
| 优化路线图 | "MTL/损失/池化/架构有改进空间" — 算法升级 | **Stage 2 — Stage 5** |

### 0.2 核心诊断结论(必须先理解)

实施方在动手前必须接受以下事实,否则会做错误的优先级判断:

1. **当前不是经典过拟合问题**:训练 main_loss 仅下降 27%,模型并未记住训练集。单纯加 dropout / label smoothing / 缩小模型规模不是首选方案。
2. **当前 val 集已被反复污染**:同一份 600 人 val 集承担了"选 best epoch + 选解码策略 + 拟合校准阈值"三种角色,日志显示的 `calibrated_argmax QWK=0.4681` 在 test 上预估只能复现到 0.42–0.44。
3. **QWK 峰值出现在 epoch 7–16**:之后约 24 个 epoch 的训练让 QWK 单调退化,40 epoch 中约 60% 在做负功。
4. **退化的根因有三层**:
   - 不确定性加权失控(主任务权重被压到 0.7,辅助任务权重抬到 5.0)
   - 预测分布扁平化(类 3 几乎消失)
   - MTL 辅助任务大部分是主标签的 deterministic 再参数化,没有新信号
5. **改进的真实增益必须在干净的 hold-out 上验证**:在污染的 val 上看到的提升数字大概率包含过拟合噪声。

### 0.3 实施总原则

- **诚实先于聪明**:任何算法改进之前先有干净的评估手段(Stage 0)。
- **稳定先于复杂**:任何架构升级之前先让现有流程不做负功(Stage 1)。
- **可切换、可回滚、可单变量消融**:每一项改动都必须能独立开关、能与前一版本对照。
- **统计显著性优先于单 seed 数字**:单 seed 提升 < 0.005 视为噪声,3 seed std > 0.5 × mean 增益视为不显著。
- **不要混在一起改**:每个 Stage 内部允许小范围合并(明确说明的地方),跨 Stage 严禁合并。

---

## 1. 版本与命名规范(必读,后文所有运行都遵守)

为彻底解决"改进版本之间互相混淆"的问题,本文档定义统一的版本命名空间。**所有 run、checkpoint、配置文件、日志、submission 都必须遵守**。

### 1.1 Stage 编号体系

| Stage | 含义 | 关键交付 |
|---|---|---|
| **S0** | 诚实评估地基(Honest Evaluation Foundation) | val_select/val_holdout 切分文件 + 校准源切换 |
| **S1** | 训练流程稳定化(Training Stabilization) | 早停/学习率/权重约束修复 |
| **S2** | 损失体系与 MTL 重构(Loss & MTL Reconstruction) | 损失插件 + 冗余辅助任务清理 + 真实辅助任务接入 |
| **S3** | 推理鲁棒性(Inference Robustness) | 概率先验偏置 + Top-K ensemble + SWA |
| **S4** | 架构升级(Architecture Upgrade) | CM-ASP + 混合时序 + MulT + 序数损失增强 |
| **S5** | 最终冲刺(Final Push) | K-fold CV + 探索性 backbone(Conformer/Mamba) |

**Stage 之间是强依赖**:S0 必须先于一切,S1 必须先于 S2+,S2 必须先于 S3 的 ensemble 部分。

### 1.2 Run 命名规范

每一次正式 run 必须按以下规范命名(用于目录、checkpoint、`run_meta.json` 中的 `run_name`、submission 文件名):

```
{track}_{stage}_{tag}_seed{n}
```

- `track` ∈ `{a1, a2}`
- `stage` ∈ `{S0, S1, S2, S3, S4, S5}`
- `tag`:对该 Stage 内变体的描述,**只能从下方注册表中取**,严禁自由发挥
- `seed`:整数 seed,固定使用 `{42, 123, 2026}` 三个 seed 作为标准 seed 池

**例**:
- `a2_S0_baseline_seed42`:S0 完成后,在新评估地基上跑的 baseline 复现
- `a2_S1_train-fix_seed42`:S1 完成后的训练流程稳定版本
- `a2_S2_loss-corn-cb_seed42`:S2 中切换到 CORN+ClassBalanced 损失的变体
- `a2_S2_mtl-cleanup_seed42`:S2 中删冗余辅助任务的变体
- `a2_S2_best_seed42`:S2 阶段最终选定的组合(多个子项合并的最优)
- `a2_S3_topk3_seed42`:S3 中 K=3 的 top-K ensemble 变体
- `a2_S5_kfold5_fold3_seed42`:S5 K-fold 中第 3 个 fold,seed=42

### 1.3 Tag 注册表(本文档唯一权威)

| Stage | 允许的 tag | 说明 |
|---|---|---|
| S0 | `baseline` | 旧 val 上原配置复现(参照基准) |
| S0 | `honest-baseline` | 完成 val 切分后旧配置的诚实复现 |
| S1 | `train-fix` | 完成早停/lr/uncertainty 约束/pos_weight 等修复后 |
| S1 | `train-fix-ablation-{X}` | 单变量消融,X ∈ {earlystop, uncert, auxw, posw, qwkaux} |
| S2 | `mtl-cleanup` | 仅删除冗余辅助任务(stage 1-MTL-A) |
| S2 | `mtl-real-{src}` | 接入真实辅助任务,src ∈ {au, ser, va, all} |
| S2 | `mtl-uw` | 启用 Kendall uncertainty weighting(在 cleanup 之上) |
| S2 | `loss-{name}` | 损失替换,name ∈ {corn, corn-cb, asl, ldam, focal} |
| S2 | `best` | S2 阶段确定的最优组合(必须在产出报告中说明组成) |
| S3 | `prior-bias` | A2 概率先验偏置 |
| S3 | `topk{K}` | Top-K checkpoint ensemble |
| S3 | `swa` | Stochastic Weight Averaging |
| S3 | `swa+topk{K}` | SWA 与 Top-K 叠加 |
| S3 | `best` | S3 阶段确定的最优组合 |
| S4 | `pool-cmasp` | Cross-Modal ASP 池化 |
| S4 | `temporal-{name}` | 时序编码器,name ∈ {tcn-tfm, tcn-lstm, msstcn} |
| S4 | `fusion-mult-{struct}` | MulT 融合,struct ∈ {a, b}(对应优化文档 2.2.2 的结构 A/B) |
| S4 | `loss-softqwk` | soft-QWK 辅助损失 |
| S4 | `loss-unimodal` | 单峰正则 |
| S4 | `best` | S4 阶段确定的最优组合 |
| S5 | `kfold{K}_fold{i}` | K-fold 的第 i 个 fold |
| S5 | `kfold{K}_ensemble` | K-fold 全部 fold 的集成结果 |
| S5 | `backbone-{name}` | name ∈ {conformer, mamba, mpcf} |
| S5 | `final` | 提交用的最终模型/集成配置 |

**新增 tag 必须先回写本文档**,严禁在 run 中私自创造未注册 tag。

### 1.4 引用前一版本的命名

每个 Stage 的"起点"必须明确指向上一 Stage 的"best"。例如:

- S2 所有实验的起点 = `a2_S1_train-fix_seed{42,123,2026}` 的平均
- S2 的 `best` 应在 `run_meta.json` 中显式记录其相对 `a2_S1_train-fix` 的 delta
- S3 所有实验的起点 = `a2_S2_best_seed{42,123,2026}` 的平均
- 依此类推

这样从任何一个 run 都能沿着 `parent_run_name` 字段追溯回 baseline。

---

## 2. Stage 0:诚实评估地基

> **核心目标**:让"选 checkpoint + 选策略 + 校准阈值"和"评估泛化性能"使用完全不相交的两份数据。在此之前任何算法改进的"提升幅度"都是噪声混合体。

### 2.1 子项 S0.1:val 集二次切分

**修改要点**:
- 在当前 **600 人 val 集内部**做二次切分,**不动 train 集**。
- 划分粒度:**participant level**(同一被试不跨边界)。
- 划分比例:**400 / 200**(val_select / val_holdout),可酌情 450/150。
- 划分必须**分层抽样**:按 DASS-21 总分(至少按抑郁/焦虑/压力三个 subscore)分层。
- 划分必须**确定性**:固定 seed,落盘保存为 `splits/val_split_v1.json`(participant ID 列表)。
- 切分文件**版本号 v1 写死**,所有后续 run 共用,严禁动态切分。

**用途约束**:
- **val_select**(400 人):承担一切"选择"行为 — 早停判据、best checkpoint 选定、解码策略对比、阈值校准的偏移量拟合。
- **val_holdout**(200 人):**严禁参与任何选择**。仅用于:
  - 每个 epoch 同步报告 QWK(只观察,不进入早停或 checkpoint 保存逻辑)
  - 训练结束后报告"最终诚实 QWK"
  - 对比不同 run 改进幅度的**唯一可信指标**

### 2.2 子项 S0.2:阈值校准数据源切换

**修改要点**:
- 阈值校准在 **val_select** 上拟合 offset。
- 校准后的策略+阈值组合,在 **val_holdout** 上评估。
- val_holdout 上的 `calibrated_argmax` 比 `raw argmax` 的提升幅度才是 calibration 的真实增益。

### 2.3 日志规范变更

完成 S0 后,所有训练日志的每个 epoch 必须同时输出:
- `val_select_qwk`(用于决策)
- `val_holdout_qwk`(仅观察)

训练结束 best-checkpoint 报告:
- val_select 上的 raw 与 calibrated QWK
- val_holdout 上的 raw 与 calibrated QWK

**所有后续 Stage 的指标对比一律以 `val_holdout_qwk_calibrated` 为准。**

### 2.4 S0 验收标准

| 检查项 | 期望值 |
|---|---|
| val_holdout QWK 与 val_select QWK 的 gap | **应当 0.01–0.04(val_holdout 更低)**;<0.005 或 val_holdout > val_select 都说明切分有问题 |
| val_holdout 200 人中类 3 样本数 | ≥ 5;否则分层切分参数有问题,重切 |
| 各 run 是否同一份切分 | 通过比对 `val_split_v1.json` 的 hash |

### 2.5 S0 关键产出

- [ ] `splits/val_split_v1.json`(participant ID + 切分逻辑说明 + seed)
- [ ] `a2_S0_baseline_seed42` 三 seed 训练日志,记录新指标的对照
- [ ] `a2_S0_honest-baseline_seed{42,123,2026}` 的 val_holdout 上 raw / calibrated QWK 数字 — **这是后续所有改进的 baseline**

### 2.6 S0 心态校准

完成 S0 后看到的 QWK 数字会比之前**低 0.02–0.04**。这不是改动让模型变差,而是消除了"虚高"。实施方应抵制"看到数字降了赶紧回滚"的本能反应。

**诚实的低数字 > 不诚实的高数字。** 此后所有改进的 +0.005 都是**真实的 +0.005**。

---

## 3. Stage 1:训练流程稳定化

> **核心目标**:让训练在 QWK 峰值附近自然停止,阻止主任务被辅助任务边缘化,缓解末期预测分布扁平化。

### 3.1 子项概览

| 子项 | tag | 风险 | 预期 val_holdout 增益 |
|---|---|---|---|
| S1.1 早停 + 训练长度 + LR 调度同步 | `train-fix-ablation-earlystop` | 极低 | 几乎为 0,但训练时间砍 60% |
| S1.2 不确定性加权约束 | `train-fix-ablation-uncert` | 中 | +0.01 到 +0.03 |
| S1.3 辅助任务权重整体下调 | `train-fix-ablation-auxw` | 低 | +0.005 到 +0.02 |
| S1.4 pos_weight 上限收紧 | `train-fix-ablation-posw` | 低 | +0.005 到 +0.015 |
| S1.5 核查 QWK 辅助损失生效性 | `train-fix-ablation-qwkaux` | 低 | +0.01 到 +0.02(若原本未生效) |

### 3.2 子项 S1.1:早停 + 训练长度 + LR 调度同步

**修改要点**:
- 早停监控指标从 `val_loss` 改为 **`val_select QWK`**(注意是 val_select,不是 val_holdout)。
- 早停方向:QWK 越大越好,mode 相应调整。
- 早停 patience:从 99 降到 **5–8**。
- 总 epoch 数:从 40 降到 **16–20**。
- warmup 维持 3 epoch,**cosine 衰减总周期同步**到新的总 epoch 数(不能只改 epochs 不改 schedule)。
- 确认 best checkpoint 按 QWK 保存,核对配置与代码逻辑一致(当前 `early_stop_metric=val_loss` 与日志中按 QWK 保存的行为矛盾)。

**验证标准**:
- 训练时间从 ~1h 降到 ~25–30min。
- 训练在 epoch 12–18 自然停止。
- best QWK 应不低于 S0 baseline。

**回滚条件**:
- 早停在 epoch 8 之前触发 → patience 调到 8–10。
- 连续 3 次 run 早停在 epoch 16+ 且 QWK 仍上升 → epochs 上限放回 25。

### 3.3 子项 S1.2:不确定性加权约束

**问题背景**:detailed losses 显示 task 权重严重失衡:

| Epoch | task_0_w(主) | task_1_w(主) | task_2_w(辅) | task_3_w(辅) |
|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | 0.74 | 0.71 | 1.97 | 1.68 |
| 40 | 0.71 | 0.67 | 4.97 | 3.95 |

**修改方案(三选一,按风险从低到高)**:

- **方案 A(推荐先试)**:保留 uncertainty weighting,给每个 sigma 加上**下限钳制**,钳制范围根据训练前几个 epoch 的 sigma 自然分布确定。
- **方案 B**:对主任务 sigma 做 detach 或单独施加正则项,防止主任务 sigma 被推高。
- **方案 C(激进)**:**完全关闭** uncertainty weighting,改用固定权重(主任务 ≥ 50% 占比)。

**验证标准**:
- 训练全程主任务权重不应低于 0.85。
- 训练全程任何辅助任务权重不应超过 2.0。

### 3.4 子项 S1.3:辅助任务权重整体下调

**修改要点**:

| 配置项 | 当前值 | 建议方向 |
|---|---|---|
| `session_loss_weight` | 0.5 | 0.2–0.3 |
| `session_type_loss_weight` | 0.15 | 0.05–0.1 |
| `emotion_dims_weight` | 0.2 | 0.05–0.1 |
| `emotion_cls_weight` | 0.15 | 0.05 或关闭 |
| `au_pred_weight` | 0.1 | 维持关闭 |
| `aux_lupi.phase1` 各 aux 权重 | 0.05–0.2 | 整体砍半 |

**做法**:单调下调而非替换。先把所有辅助权重 ×0.5,看 QWK 反应;有提升再 ×0.5,有下降则回退。

**关键提醒**:S1.3 与 S1.2 应配合考虑,**不要单独做 S1.3 就期望大幅提升**。

### 3.5 子项 S1.4:pos_weight 上限收紧

**修改要点**:
- pos_weight 上限从 10.0 降到 **5.0**(首选)或 **6.0**(保守)。
- 不改变 pos_weight 计算公式(`sqrt(n_neg/n_pos)`),只改 clamp 上限。
- **必须在 S1.2 生效后才评估 S1.4**,否则归因混乱。

**验证标准**:
- 末期预测分布中类 3 比例不低于 1%(GT 是 2.5%)。
- 末期预测分布形态接近 GT(0/1/2/3 ≈ 70/22/5/2,±5% 浮动)。

### 3.6 子项 S1.5:QWK 辅助损失生效性核查

**排查要点**:
1. 检查 `use_qwk_aux: 1` 与 `qwk_weight: 0.3` 是否在训练代码中被读取并连接到实际的可微分 QWK 损失。
2. 检查 `qwk_weight` 是否乘到了真实在反向传播中的 loss 张量上。
3. 若已生效,确认它是否被 uncertainty weighting 也"管理"了 — 如果是,权重可能已被压到接近 0。
4. 在 detailed losses 日志中**显式打印 `aux_qwk_loss` 分量**。

**修改要点(若发现未生效)**:
- 若 `use_qwk_aux` 是无效配置:实现一个可微分 QWK 近似(基于 soft confusion matrix),加入主任务 loss。
- 若已生效但被 uncertainty weighting 压制:让 QWK aux loss 不进入 uncertainty weighting,使用固定权重 0.2–0.5。

### 3.7 S1 实施顺序与消融实验设计

**严禁**一次合并所有 S1.x 改动,否则失败时无法归因。**强制按以下顺序**:

```
a2_S0_honest-baseline (起点)
        │
        ▼
a2_S1_train-fix-ablation-earlystop  ← 单独验证 S1.1
        │  (val_holdout QWK 不退化、训练时间砍 60%)
        ▼
a2_S1_train-fix-ablation-uncert     ← 在 earlystop 之上加 S1.2
        │  (val_holdout QWK +0.01 以上保留,否则回滚)
        ▼
a2_S1_train-fix-ablation-auxw       ← 在上之上加 S1.3
        │  (val_holdout QWK +0.005 以上保留,否则回滚)
        ▼
a2_S1_train-fix-ablation-posw       ← 在上之上加 S1.4
        │
        ▼
a2_S1_train-fix-ablation-qwkaux     ← 在上之上加 S1.5(若需要)
        │
        ▼
a2_S1_train-fix (S1 阶段 best)       ← 3 seed 跑,作为 S2 起点
```

每一步都需要 ≥ 3 seed 跑(seed = {42, 123, 2026}),报告 mean ± std,前后对比仅看 **val_holdout calibrated QWK**。

### 3.8 S1 红线(任何一项触发立即回滚到上一节点)

- 训练在 epoch 5 之前早停
- val_holdout best QWK 比上一节点下降 > 0.01
- 预测分布末态类 0 比例 > 90% 或类 1+2+3 之和 < 10%
- 训练 loss 出现 NaN 或持续上升 > 3 个 epoch

### 3.9 S1 不要做的修改

以下方向看似合理,但**基于日志分析不推荐先做**:
- 加大 dropout / label smoothing / feature noise(train loss 没崩到 0)
- 降低模型容量 / 减少 TCN 层数(瓶颈是任务平衡,不是表达能力)
- 切换 SSL backbone(引入混杂变量,应在流程稳定后做)
- 直接增大学习率(当前 1e-3 已经触发 epoch 16 后的退化)
- 修改 CORAL 头结构(CORAL 阈值漂移是症状不是病因)

---

## 4. Stage 2:损失体系与 MTL 重构

> **核心目标**:删除拖后腿的冗余辅助任务,引入真实增量信号,替换更适合任务结构的损失函数。

### 4.1 重要前置:必须先做 MTL 清理

**根因分析**:当前 valence/arousal、emotion_cls 由 DASS-21 主标签 deterministic 推导(等价于把主标签换形式喂网络),AU head 标签恒为零(死分支)。在此基础上启用 Uncertainty Weighting 等于把注意力分散到没有新信息的副本任务上,**这是 S1 中观察到的"task 权重失衡"的深层原因之一**。

**因此 S2 必须按以下子序列推进**,不可跳步。

### 4.2 子项 S2.1:MTL 清理(Stage 1-MTL-A,必做)

**修改要点**:

| 任务 | 处理 | tag |
|---|---|---|
| `emotion_cls`(DASS 阈值推导) | **删除** | 计入 `mtl-cleanup` |
| `valence/arousal`(DASS 线性推导) | **降级为正则化损失**(权重 ≤ 0.05),或同样删除 | 计入 `mtl-cleanup` |
| `AU 预测`(占位零) | **暂时禁用 head** | 计入 `mtl-cleanup` |

**对应 run**:`a2_S2_mtl-cleanup_seed{42,123,2026}`

**验证标准**:
- val_holdout QWK 应**不下降**,理想情况下小幅上升(+0.003 ~ +0.01)。
- 若仅做 cleanup 就提升 QWK → 是有价值的负面发现(说明原 MTL 在拖后腿),记录到 run_meta。
- 若 cleanup 后 QWK 下降 > 0.005,说明删的某个任务实际有价值,需要逐一定位回滚。

### 4.3 子项 S2.2:损失函数重构(B2 / B3)

**A2 主线推荐**:`CORAL` → `CORN + Class-Balanced`(β=0.999 起调)
- **CORN** 相较 CORAL 不共享 score 投影,对 21 个 item 内部异构(题目难度不同)更友好。
- **Class-Balanced** 按 effective sample number 加权,适合 21 × 4 长尾分布(按 item × bin 联合统计)。

**A1 副线推荐**:`BCE+pos_weight` → `ASL`(Asymmetric Loss)
- ASL 为多标签不平衡量身设计,对硬负例做不对称 focal-style 抑制,比 BCE+pos_weight+硬截断更柔和。

**消融执行顺序**:

| 子步骤 | run name | 对比对象 | 决策 |
|---|---|---|---|
| S2.2.a | `a2_S2_loss-corn_seed*` | `a2_S2_mtl-cleanup` | QWK +0.003 保留;退化 > 0.005 回滚 |
| S2.2.b | `a2_S2_loss-corn-cb_seed*` | `a2_S2_loss-corn` | 同上 |
| S2.2.c(可选)| `a2_S2_loss-corn-cb-ldam_seed*` | `a2_S2_loss-corn-cb` | LDAM margin 项做长尾补强 |
| S2.2.d(A1)| `a1_S2_loss-asl_seed*` | `a1_S2_mtl-cleanup` | A1 单独评估 |

**候选保留**(实现为可切换):Focal Loss、LDAM Loss、Class-Balanced、原 CORAL / 原 BCE(作为回退)。

### 4.4 子项 S2.3:接入真实辅助任务(Stage 1-MTL-B,推荐做)

**候选方案**(agent 评估资源后选择 ≥ 1 个接入):

| 任务 | 数据来源(伪标签) | 接入方式 |
|---|---|---|
| 面部 AU | OpenFace 2.x 或 Py-Feat 离线预提取,parquet 落盘 | 12d AU 强度向量回归监督 |
| 语音情感(SER) | `emotion2vec` / `funASR-emotion` 软标签 | 4–7 类离散 + 软标签分类 |
| 维度情感(VA) | wav2vec2-MSP-Podcast、ABAW 系预训练 | **替换**当前 deterministic 推导的 VA |

**tag**:`mtl-real-au` / `mtl-real-ser` / `mtl-real-va` / `mtl-real-all`

**验证标准**:每个真实辅助任务接入后单独消融,val_holdout QWK +0.003 才保留。

### 4.5 子项 S2.4:启用 Uncertainty Weighting(Stage 1-MTL-C)

**前置条件**:S2.1 完成 + S2.3 至少一个真实任务接入。

**修改要点**:
- 每个任务一个可学习 `log_var` 参数
- 总损失:`Σ_i [exp(-s_i) * L_i + s_i]`
- **必须保留 S1.2 的约束机制**(sigma 钳制或主任务保护),否则会复现 S1 之前的失衡问题

**对应 run**:`a2_S2_mtl-uw_seed*`

**验证标准**:
- MTL 改造前后必须分别报告主任务指标
- 整体 MTL 改造(cleanup + real aux + UW)必须带来 ≥ **+0.005** 的 val_holdout QWK 增益,否则只保留 cleanup 部分

### 4.6 S2 实施顺序与产出

```
a2_S1_train-fix (起点)
        │
        ▼
a2_S2_mtl-cleanup           ← 删冗余,这一步必须先做
        │
        ▼
a2_S2_loss-corn             ← 损失替换 step1
        │
        ▼
a2_S2_loss-corn-cb          ← 损失替换 step2
        │
        ▼
a2_S2_mtl-real-{src}        ← 选择一个真实辅助任务接入
        │
        ▼
a2_S2_mtl-uw                ← 启用 uncertainty weighting
        │
        ▼
a2_S2_best                  ← S2 阶段 best(3 seed,作为 S3 起点)
```

### 4.7 S2 总验收

- 整体 S2 改进必须让 val_holdout calibrated QWK 在 S1 之上 **+0.01 以上**才算阶段通过。
- 若仅个别子项有效:在 run_meta 中明确记录 best 是哪几项的组合。

---

## 5. Stage 3:推理鲁棒性

> **核心目标**:通过推理端的 ensemble / 校准 / 先验注入,平滑单 checkpoint 的随机性,获得稳定的提升。

### 5.1 子项 S3.1:A2 概率先验偏置

**设计**:在 A2 推理时(**不影响训练**),对每题项的预测概率做先验校正。

**执行流程**:
```
模型 logits → softmax/累积概率 → [先验偏置(新增)] → 解码(argmax/monotonic/expectation)→ offset 校准
```

**参数**:
- 先用 train 集统计每题项类别先验 `prior_k`(k ∈ {0,1,2,3})
- 推理前对类别 k 概率乘以 `weight_k`:
  - 中间类(k=1, 2):乘 `α_mid` ∈ [1.0, 1.5],步长 0.05
  - 极端类(k=0, 3):乘 `α_ext` ∈ [0.7, 1.0],步长 0.05
- 在 **val_select** 上网格搜索,**val_holdout** 上评估真实增益

**对应 run**:`a2_S3_prior-bias_seed*`

**验证标准**:
- val_holdout QWK 单独提升 ≥ +0.005 才保留
- 允许 MAE 小幅退化(< 0.02),但需明确记录
- 混淆矩阵:0↔3 误判频次下降、1↔2 召回上升

### 5.2 子项 S3.2:Top-K Checkpoint Logit Ensemble

**核心规则**:
- 训练阶段维护按 val_select QWK 排序的 top-K 队列,推荐 **K = 3**(保守可用 K=5)。
- 推理阶段:**在 CORAL logit 层面做算术平均**(在 sigmoid 之前的 raw logits),严禁在预测类别上投票。
- 与阈值校准的交互:先对 K 个 checkpoint 在 val_select 上做 logit 平均,**再在平均后的 logits 上拟合阈值偏移**(把集成视为一个整体模型)。

**对应 run**:`a2_S3_topk3_seed*`(以及 `a2_S3_topk5_seed*` 作消融对比)

**验证标准**:
- val_holdout 上,K=3 集成 QWK 应至少等于单 best checkpoint,典型提升 +0.005 ~ +0.015。
- 若集成低于单 best,检查 top-K 的 val_select QWK 分布:若 top-3 间差距 > 0.02,缩到 K=2 重试。

**注意**:
- BF16/FP16 训练下,logits 在低精度做平均要先 cast 回 FP32。
- 不同 checkpoint 不能直接做**权重平均**(除非来自接近的优化轨迹)。logit 平均没有此问题。

### 5.3 子项 S3.3:Stochastic Weight Averaging (SWA)

**与 S3.2 的区别**:
- **S3.2(logit ensemble)**:保留 K 个独立模型,推理时分别 forward 再平均,简单但推理成本 ×K。
- **S3.3(SWA)**:训练时维护权重的滑动平均,推理时只用一组平均权重,**推理成本与原来相同**。

**修改要点**:
- SWA 启动时机:epoch 7-10 之后(进入平台/退化区)。
- 平均策略:**等权平均**(简单且实证有效),可选过滤(只在 val_select QWK 高于平均阈值的 epoch 加入)。
- **BatchNorm 重统计**:必须核查模型是否含 BN(项目用 TCN + Attention Pooling,理论上是 LayerNorm,但需确认)。若含 BN,SWA 推理前必须在训练集上重统计 running mean/var。
- Checkpoint 保存:训练结束时保存原 best + SWA averaged,在 val_select 上分别评估,选优。

**对应 run**:`a2_S3_swa_seed*`、`a2_S3_swa+topk3_seed*`(叠加)

**验证标准**:
- val_holdout 上 SWA QWK 不低于 best single checkpoint。
- 若 SWA 低于 single best 超过 0.005:检查是否启动太早、是否未做 BN 重统计、是否学习率衰减期权重已经很集中。

**与 S3.2 的取舍**:
- 资源紧张:**优先选 S3.2**(更简单,与训练流程解耦)。
- 资源充足:S3.2 + S3.3 叠加(把 SWA 作为 top-K 中的一员)。

### 5.4 子项 S3.4(可选):Cross-Modal ASP

**说明**:此项原属优化文档 1.4(Phase 1),但其涉及池化层结构修改,本质属于架构改动而非推理改进。**在本统一文档中归入 S4(架构升级)**。详见 §6.1。

### 5.5 S3 实施顺序

```
a2_S2_best (起点)
        │
        ├── a2_S3_prior-bias           ← 独立改进,可与下方并行
        │
        ├── a2_S3_topk3                ← 独立改进
        │
        ├── a2_S3_swa                  ← 独立改进
        │
        ▼
a2_S3_best                              ← 把上述有效项合并(典型为 prior-bias + topk3)
                                          作为 S4 起点
```

### 5.6 S3 红线

- val_holdout QWK 比上一步下降 > 0.01 → 立即回滚
- 集成后 QWK 反而比单模型低超过 0.005 → 检查 ensemble 实现
- 推理时间增长不成比例(K=3 应大致是 3× 推理时间,变成 5× 说明实现有问题)

---

## 6. Stage 4:架构升级

> **核心目标**:在前 3 个 Stage 提供的稳定地基上,做模型架构本身的升级。

### 6.1 子项 S4.1:Cross-Modal ASP

**设计要点**:
- 音频 ASP 的 attention score query 由 **video 全局向量**参与(video TCN 输出的 mean pooling 或 [CLS] 向量)
- 视频 ASP 反向同理
- **保留原 VAD/QC 偏置项**(不破坏现有显式先验注入)
- 跨模态 query 通过轻量 linear gate 控制
- **缺模态 fallback**:当某 session 某模态全 mask=0 时,cross-modal query 必须退化为本模态自打分(等价回原 ASP)
- 加 dropout 在 cross-modal gate 上,避免过度依赖单边

**对应 run**:`a2_S4_pool-cmasp_seed*`

**验证标准**:val_holdout QWK Δ ≥ +0.005 保留。

**风险与回滚**:缺模态 session 上 NaN → 自打分 fallback 强制启用。

### 6.2 子项 S4.2:混合时序编码器

**候选方案**:

| 方案 | 形态 | 优点 | 风险 | tag |
|---|---|---|---|---|
| A(推荐起点)| TCN(3-4 层)+ Transformer(2-3 层)串联 | 局部+全局互补 | 长序列 attention 显存压力 | `temporal-tcn-tfm` |
| B | TCN + LSTM/GRU 并联 | 训练稳定 | 长序列循环退化 | `temporal-tcn-lstm` |
| C | MS-S-TCN(多尺度共享 TCN) | 改动最小 | 增益上限低 | `temporal-msstcn` |

**推荐顺序**:先 A 与 C 的 ablation,A 不显著优于 C 时保留 C(更安全)。

**工程约束**:
- 沿用现 `valid_mask`,attention mask 正确广播到 padding 位置
- 序列长度截断/分块策略需在 dataset 层或 collate 层明确
- 与现 `feature_noise`、`session_drop` 兼容

**验证标准**:
- 单独消融 val_holdout QWK Δ ≥ +0.008 保留
- 训练耗时增量 < 2× baseline,否则改用方案 C

### 6.3 子项 S4.3:MulT 跨模态融合

**前置条件**:S4.2 已完成。

**设计要点**:
- 用 MulT 替换/增强现 Fusion MLP
- 至少实现 A→V、V→A 两个 cross-modal attention 分支
- 与 S4.1(CM-ASP)形成"深-浅互补":CM-ASP 在池化前轻量 gating,MulT 在表示层深层互注
- 输出与现 `session_repr` 接口一致(不破坏 `ParticipantAggregator`)

**两种结构同时实验**(二选一保留):
- 结构 A:时序编码 → MulT 跨模态 → CM-ASP 池化(`fusion-mult-a`)
- 结构 B:时序编码 → CM-ASP 池化 → MulT 处理池化后表示(`fusion-mult-b`,更轻)

**验证标准**:val_holdout QWK Δ ≥ +0.005 保留。

### 6.4 子项 S4.4:序数损失增强

**子项 S4.4.a:可微 QWK 辅助损失**(与 S1.5 衔接,但为更完整实现):
- 实现 differentiable QWK(soft-QWK,基于 expected confusion matrix)
- 作为辅助损失与主损失加权:`loss = L_main + λ_qwk * L_softqwk`(λ 从 0.1 起调)
- **必须 warmup**:前若干 epoch 仅用 L_main,待主任务稳定后再注入
- 监控梯度范数,必要时 clip

**对应 run**:`a2_S4_loss-softqwk_seed*`

**子项 S4.4.b:单峰分布正则**:
- 约束输出概率关于真值 y_i 单峰(p(k) 随 |k-y_i| 单调下降)
- 实现可选:二项分布参数化 或 软单峰惩罚(对违反单峰的相邻 bin 概率差加 hinge)
- 与 CORN 的单调性约束**互补**(CORN 约束 P(y≥k),单峰约束 P(y=k))

**对应 run**:`a2_S4_loss-unimodal_seed*`

**验证标准**:
- soft-QWK:val_holdout QWK +0.003 起步
- 单峰正则:val_holdout QWK +0.003 起步且 MAE 不退化
- 不达标不保留

### 6.5 S4 实施顺序

```
a2_S3_best (起点)
        │
        ├── a2_S4_pool-cmasp            ← 池化升级
        │
        ├── a2_S4_temporal-tcn-tfm      ← 时序升级
        │       │
        │       └── a2_S4_temporal-msstcn ← 对照(更安全)
        │
        ├── a2_S4_fusion-mult-a         ← 融合升级(在时序之上)
        │
        ├── a2_S4_loss-softqwk          ← 序数损失增强
        │
        └── a2_S4_loss-unimodal         ← 序数损失增强
        │
        ▼
a2_S4_best                              ← S4 综合最优(必须显式记录组成)
```

### 6.6 S4 总验收

- 累计验收门槛:val_holdout QWK ≥ S3 最佳 + 0.015。
- 任一子项若 3 seed std > 0.5 × mean 增益,视为不显著,不保留。

---

## 7. Stage 5:最终冲刺

> **核心目标**:用 K-fold 与探索性 backbone 榨取最后的性能上限。该阶段允许并行试验、允许部分子项不进入最终提交。

### 7.1 子项 S5.1:K-fold 交叉验证

**前置条件**:S0-S4 全部完成,有稳定的"最终配置"。

**修改要点**:
- 在 **train 集 4200 人内部**做 **5-fold** 切分。
- 原 600 人 val 集:**作为全局 hold-out**(不参与训练,不参与 fold 切分,作为最终性能封测)。
- 切分粒度:participant level,按 DASS-21 总分分层。
- 每个 fold 独立训练,每个 fold 内部按 train_fold / val_fold 做正常训练与早停。
- 每个 fold 独立得出 best checkpoint。

**集成推理**:
- 5 份 logits 平均,再走解码与校准
- 校准的阈值偏移:每个 fold 单独拟合后对偏移量做平均(推荐),或在 5 模型平均 logits 上做整体校准

**对应 run**:`a2_S5_kfold5_fold{0-4}_seed42`(每个 fold 一个 run)、`a2_S5_kfold5_ensemble_seed42`(最终集成)

**评估**:5-fold OOF (out-of-fold) 预测拼起来 → train 集上的诚实预测,这是最可靠的本地 QWK 估计指标。

**验证标准**:
- 5 个 fold 的单独 best QWK 在彼此 ±0.02 范围内,差距过大说明 fold 切分有偏
- OOF 拼接 QWK 接近 5 fold 单独 QWK 均值(差距 < 0.01)
- 5-fold ensemble 推理 QWK 比单 fold best 高 +0.005 ~ +0.020

**资源预算**:训练 ×5,配合 S1.1 的 16 epoch 后单次 ~25 min,5 fold 共 ~2 小时。

### 7.2 子项 S5.2:SOTA Backbone 探索

**Conformer(音频侧首选)**:
- 用 Conformer block 替换音频侧时序编码器(或与 S4.2 混合时序串联)
- Conformer = MHSA + Conv 模块,对韵律/局部声学模式天然友好
- 参数规模注意:Conformer-small(约 10-20M)可能超过现 backbone 总和
  - 重新调 LR / weight decay
  - 启用更强正则(增大 feature_noise、SpecAugment-style 频域增强)
  - 重新评估早停策略

**对应 run**:`a2_S5_backbone-conformer_seed*`

**Mamba-VA / SSM(视频或聚合层备选)**:
- 状态空间模型对超长序列推理友好
- 建议先在视频侧实验
- 与 Conformer 互补:Conformer 在音频,Mamba 在视频

**对应 run**:`a2_S5_backbone-mamba_seed*`

**验证标准**:
- 与 S4 最佳模型对比,val_holdout QWK +0.01 才保留为最终单模型候选
- 不达标则只作为 ensemble 候选

### 7.3 子项 S5.3:MPCF 融合(可选)

- Multimodal Progressive Co-Fusion:在 MulT 之上引入逐层 modality-specific + modality-shared 通道
- 替换或叠加在 MulT 上(agent 评估结构合理性后决定)
- **对应 run**:`a2_S5_backbone-mpcf_seed*`
- 验收:val_holdout QWK +0.01 才保留

### 7.4 子项 S5.4:梯度冲突治理 PCGrad / DB-MTL(可选)

**启用条件**:S2 已完成且确实启用了 ≥ 2 个真实辅助任务。

**设计**:
- **PCGrad**:每个任务独立 backward 得到梯度,对冲突梯度做投影
- **DB-MTL**:动态平衡任务梯度幅度
- 与 Uncertainty Weighting 不冲突(UW 调权重,PCGrad 调方向,可叠加)
- **代价**:每 step 多次 backward,吞吐下降约 2-3×

**验证标准**:不要求单独 QWK 提升,但要求多任务整体指标稳定性提升(多 seed 方差下降)。

### 7.5 S5 实施与最终模型选择

```
a2_S4_best (起点)
        │
        ├─ a2_S5_kfold5_ensemble       ← 必做
        │
        ├─ a2_S5_backbone-conformer    ← 探索(允许失败)
        │
        ├─ a2_S5_backbone-mamba        ← 探索
        │
        └─ a2_S5_backbone-mpcf         ← 探索
        │
        ▼
a2_S5_final                            ← 最终提交模型
                                         可能是:
                                         (a) 单一 K-fold ensemble
                                         (b) K-fold ensemble + 探索性 backbone 的二次 ensemble
                                         (c) 由 3-5 个多样模型组成的 ensemble
```

### 7.6 S5 关键决策事项(需用户/项目所有人确认)

这些不阻塞 Stage 0-S4 启动,但执行到对应子项前需明确:
1. AU/SER/VA 真实伪标签的具体外部模型选型
2. Stage 5 GPU 预算上限(影响 Conformer/MulT 规模与 PCGrad 是否启用)
3. 是否允许在 S4/S5 引入音频/视频 SSL 模型的微调(当前 baseline 全部冻结)
4. 最终提交是单模型还是 ensemble,若 ensemble 是否限制模型数

---

## 8. 跨阶段开发与工程规范

### 8.1 可切换性(强约束,每个 PR 都必须保持)

- 所有新增模块必须通过 `tasks/*/default.yaml` flag 启停,默认值**保留 baseline 行为**。
- yaml flag 命名层次化:`loss.a2.type: corn`、`model.pooling: cm_asp`、`model.temporal.encoder: tcn_transformer`。
- **同一 run_id 下不允许中途切换关键 flag**。

### 8.2 实验追踪规范

每个 run 的 `run_meta.json` 必须包含以下字段(在原有基础上扩展):

```jsonc
{
  "run_name": "a2_S2_loss-corn-cb_seed42",
  "stage": "S2",
  "tag": "loss-corn-cb",
  "track": "a2",
  "seed": 42,
  "parent_run_name": "a2_S2_loss-corn_seed42",   // 这一 run 的直接起点
  "baseline_run_name": "a2_S0_honest-baseline_seed42",   // 整条链路的根
  "val_split_version": "v1",                      // 必须等于 splits/val_split_v1.json 的 hash
  "loss_type": "corn_cb",
  "mtl_tasks": ["a2_main", "session_type"],       // 实际启用的任务清单
  "mtl_weighting": "uncertainty",                  // 或 "fixed" / "none"
  "pooling": "asp",                                // baseline ASP / cm_asp
  "temporal_encoder": "tcn",                       // tcn / tcn_transformer / msstcn / ...
  "fusion": "mlp",                                 // mlp / mult_a / mult_b / mpcf
  "decoding": "argmax",                            // raw 决策路径
  "calibration_chain": ["prior_bias", "offset_grid"],   // 推理校准链路
  "metrics": {
    "val_select_qwk_raw": 0.42,
    "val_select_qwk_calibrated": 0.45,
    "val_holdout_qwk_raw": 0.41,                  // 唯一可信指标
    "val_holdout_qwk_calibrated": 0.43,
    "val_holdout_mae": 0.65,
    "train_main_loss_final": 1.32,
    "epochs_trained": 14,
    "epochs_best": 12
  },
  "delta_vs_parent": {
    "val_holdout_qwk_calibrated": 0.008           // 与 parent_run 的差值
  }
}
```

### 8.3 评估纪律(强约束)

- **每个关键实验 ≥ 3 seed**,seed 固定使用 `{42, 123, 2026}`,报告 mean ± std。
- 主指标:A2 mean QWK / A1 macro F1,均看 **val_holdout calibrated**。
- 辅助指标:A2 mean MAE、per-item QWK 方差、混淆矩阵。
- **某子项 3 seed std > 0.5 × mean 增益 → 视为不显著,不保留**。
- 单 seed 提升 < 0.005 不算有效改动。
- 改进 mean 减去 baseline mean 应 > 2 × std,才算统计有效。

### 8.4 校准链路兼容性(强约束)

- 现有 A1 bias 网格、A2 解码 + offset 搜索必须仍可运行。
- S3.1 的"概率先验偏置"在解码之前应用。
- 校准产物文件路径与字段保持向后兼容(新字段以可选形式追加)。
- 校准文件命名:`calibration/{track}_{stage}_{tag}_seed{n}.json`,与 run 名严格对应。

### 8.5 缺失模态/缺失 session 鲁棒性(强约束)

- 所有新增模块(CM-ASP、MulT、Cross-Attention、MPCF 等)在缺失模态时必须 **graceful fallback,不能产生 NaN/Inf**。
- 单元测试覆盖:单模态 session、单 session participant、全模态缺失边界。
- 缺失模态 fallback 行为必须显式记录到 `run_meta.json` 中(避免静默 fallback 影响指标可解释性)。

### 8.6 复现性

- 每个 Stage 末尾打包:`{stage}_best_seed{n}` 的最佳 checkpoint + `config_used.yaml` + 校准 JSON + `run_meta.json` + 关键日志。
- `infer.py` 路径不变,可直接加载任一阶段交付物。
- `splits/val_split_v1.json` 全程不变。若必须升级到 v2,所有运行从 S0 重新开始。

### 8.7 日志输出规范

每个 epoch 日志必须包含(最少):
- `epoch, lr, main_loss, val_select_qwk, val_holdout_qwk`(后两者均含 raw 与 calibrated)
- 各 task loss 分量及当前权重(若启用 MTL)
- A2 末期预测分布(0/1/2/3 比例)
- top-K checkpoint 队列状态(若启用 S3.2)

---

## 9. 风险清单与回滚预案

| 风险 | 阶段 | 信号 | 回滚预案 |
|---|---|---|---|
| val 切分把分布偏移引入 val_select | S0 | val_holdout QWK > val_select QWK,gap > 0.02 | 重新切分,核查分层逻辑 |
| 早停过晚或过早 | S1 | epoch 5 之前停 / epoch 16+ 仍上升 | 调整 patience(5↔10)和 epochs 上限(16↔25) |
| Uncertainty 约束过紧导致主任务收敛慢 | S1 | main_loss 下降明显变慢 | 放宽 sigma 钳制下限,或换方案 B |
| MTL cleanup 反而退化 | S2 | val_holdout QWK 下降 > 0.005 | 逐一回滚被删任务,定位哪个有价值 |
| CORN+CB 在小样本下不稳定 | S2 | 训练 loss 震荡 / NaN | 退回 CORAL,只保留 ClassBalanced |
| 真实辅助任务无收益 | S2 | QWK Δ < 0.003 | 仅保留 mtl-cleanup,不启用 UW |
| CM-ASP 缺模态 session 上 NaN | S4 | val loss 飙升 | 自打分 fallback 强制启用 |
| 概率先验偏置 MAE 显著上升 | S3 | MAE +0.05 以上 | 仅保留中间值 boost,去掉极端值 dampen |
| Top-K 集成反而下降 | S3 | val_holdout 集成 < single best | 检查 top-K 间 QWK 方差,缩到 K=2 |
| SWA 偏低 | S3 | SWA < single best 超过 0.005 | 延后启动 epoch / 添加 BN 重统计 |
| 混合时序在小样本上过拟合 | S4 | train-val gap 拉大 | 退回方案 C(MS-S-TCN)或加大正则 |
| 可微 QWK 训练发散 | S4 | grad norm 爆炸 | 延长 warmup、降 λ 或暂时禁用 |
| Conformer 训练超出预算 | S5 | OOM / 单 epoch > 2 小时 | 退回 Conformer-tiny 或仅音频侧用 |
| PCGrad 训练耗时不可接受 | S5 | step 时间 > 3× | 仅最后 finetune 阶段启用 |

---

## 10. 整体实施时间线建议(参考)

假设单次完整训练 ~25 分钟(S1 后),3 seed 一组需 ~75 分钟:

| 阶段 | 关键 run 数(粗估) | 累计墙钟时间 |
|---|---|---|
| S0 | 3 runs(honest-baseline × 3 seed) | 4 小时(含旧 40-epoch 复现) |
| S1 | 5 子项 × 3 seed = 15 runs | 6–8 小时 |
| S2 | 6 子项 × 3 seed = 18 runs(部分可共用) | 8–10 小时 |
| S3 | 3 子项 × 3 seed + 集成评估 = ~12 runs | 5–6 小时 |
| S4 | 5 子项 × 3 seed = 15 runs | 8–10 小时 |
| S5 | K-fold 5 + 探索性 3 个 × 3 seed = ~14 runs | 12–15 小时 |
| **合计** | **~75 runs** | **~45–55 小时** |

该时间线不含等待、bug 修复、回滚的额外开销。实际项目中应留 ≥ 2× 缓冲。

---

## 11. 最终交付物清单

实施方在完成全部阶段后应交付:

- [ ] `splits/val_split_v1.json`(participant ID + 切分逻辑 + seed)
- [ ] 每个 Stage 的 `{stage}_best` checkpoint(3 seed),含 `config_used.yaml`、校准 JSON、`run_meta.json`
- [ ] 每个 Stage 的对照报告(markdown 表格),包含:
  - 该 Stage 内每个子项的 val_holdout calibrated QWK(mean ± std,3 seed)
  - 与上一 Stage best 的 delta
  - 关键超参与配置
  - 训练时间、推理时间
  - 预测分布健康度(末期 0/1/2/3 比例)
- [ ] 总体进展图:`baseline → S0 → S1 → S2 → S3 → S4 → S5` 的 val_holdout QWK 折线
- [ ] 比赛提交的 submission CSV 与 `a2_S5_final` 的完整配置说明
- [ ] 失败/回滚记录文件(`rollback_log.md`),记录所有被回滚的子项与原因

---

## 12. 元规则:文档版本与冲突处理

- 本文档为 **v1.0**,后续修改必须更新版本号并在尾部追加 changelog。
- 若实施过程中发现某子项无法按预期实施,**先回写本文档**(附阻塞原因)再决定方案变更。
- 若与前置三份源文档(训练改进 / 推理验证 / 优化路线)有冲突,以本文档为准;前置文档作为背景参考。
- 若与项目实际代码结构冲突,以项目实际为准,并在交付报告中注明偏离点与理由。
- 严禁实施方在未回写文档的情况下私自跳过验收门槛、引入未注册 tag、或合并跨 Stage 改动。

---

**文档版本**:v1.0
**整理依据**:`a2_training_modification_guide.md` + `a2_inference_and_validation_guide.md` + `ADODAS2026_optimization_roadmap.md`
**变更纪律**:本文档为执行规约,所有偏离需在交付物中显式记录。

# A2 推理与验证策略改进指导文档

> **适用对象**：实施改进的 agent / 工程师
> **目标项目**：AdoDAS2026 Track A2 baseline（`grouped` 训练流程）
> **依据**：2026-05-26 两次完整 40-epoch 训练日志、A2 解码策略对照与校准结果
> **本文档不包含具体代码**：所有改动的具体落点、变量名、文件路径由实施方结合项目代码结构自行确定。
> **与训练流程改进文档的关系**：本文档是 `a2_training_modification_guide.md` 的姊妹文档。前者关注训练本身（如何训出更好的模型），本文档关注模型选择与评估方法论（如何诚实评估并稳定地拿出推理结果）。**两份文档应配合使用**，本文档的 P0 应在训练改进文档的 P0 之后立即实施，否则训练改动的真实收益无法被诚实评估。

---

## 0. 必读：问题诊断

实施方在动手前必须理解当前推理流程存在的核心问题：

**同一份 val 集（600 人）被反复套用了三次选择：**

1. **第一次套用**：在每个 epoch 计算 val QWK，挑出 QWK 最高的 epoch 作为 best checkpoint。
2. **第二次套用**：在 best checkpoint 之上，对比 argmax / monotonic / expectation 三种解码策略（再加上各自的校准版本），挑出 val 上 QWK 最高的策略。
3. **第三次套用**：阈值校准（threshold offset calibration）的偏移量本身也是在同一份 val 集上拟合出来的。

每一次"选择"都会向 val 集的具体噪声模式过拟合。最终日志里 `calibrated_argmax QWK=0.4681` 这个数字反映的是"在这个特定 600 人 val 集上能榨出多少"，**不是模型对未知 test 数据的真实泛化能力**。

**预估的 val-to-test gap：**

| 偏差来源 | 估计幅度 |
|---|---|
| 挑最佳 epoch（40 个 epoch 中取 max） | -0.015 ~ -0.030 |
| 挑最佳解码策略 + 阈值校准 | -0.010 ~ -0.020 |
| **合计预期 test QWK 衰减** | **-0.025 ~ -0.050** |

即当前 Run 1 显示的 0.4681 在 test 上**更可能落在 0.42–0.44 区间**。这个 gap 是结构性的，不修就会一直存在。

本文档的修改项就是逐步消除这些偏差源。

---

## 1. 修改项总览

| 优先级 | 修改项 | 风险 | 预期效果 | 是否阻塞下一项 |
|---|---|---|---|---|
| P0 | 分裂 val 集为 val_select + val_holdout | 极低 | 获得诚实的泛化估计 | 是 |
| P1 | 阈值校准的数据源切换 | 低 | 减少 calibration overfit | 否 |
| P2 | Top-K checkpoint 推理时集成 | 低 | Test QWK 提升 0.005–0.015 | 否 |
| P3 | Stochastic Weight Averaging | 中 | 与 P2 互为替代/互补 | 否 |
| P4 | K-fold 交叉验证 | 高（5× 训练成本） | 最稳健，提升上限最高 | 否 |

**实施顺序原则**：P0 必须先做并独立验证，否则后续任何改动的"提升幅度"都不可信。P1 可与 P0 同步完成（成本极低）。P2、P3 是 inference 阶段的鲁棒性改进，可以独立做。P4 是最重的兜底方案，资源允许时再上。

---

## 2. P0：分裂 val 集

### 目标
让"选择 checkpoint / 选择策略 / 校准阈值"和"评估最终性能"使用**完全不相交**的两份数据，恢复对模型真实性能的诚实估计。

### 修改要点

**(a) 划分方式**

- 在**当前 600 人的 val 集内部**做二次切分（不要动 train 集）。
- 划分粒度必须是 **participant level**，不是 session level —— 同一被试的不同 session 不能跨划分边界，否则等同于数据泄漏。
- 划分比例建议 **400 / 200**（val_select / val_holdout），可酌情调整为 450/150。
- 必须**分层抽样**：按 DASS-21 总分（或至少按抑郁/焦虑/压力三个 subscore）做分层。原因：类 3 的样本本来就稀有（GT 占 2.5%），若按完全随机切，200 人的 val_holdout 里可能根本没有几个类 3 样本，QWK 估计就极不稳定。
- 划分必须**确定性**（固定 seed），并把划分文件落盘保存，所有后续 run 都用同一份切分。否则每次跑分都会因为切分不同而抖动。

**(b) 用途约束**

- **val_select**：所有"选择"行为发生的场所。包括：
  - 训练时每个 epoch 的早停判据
  - Best checkpoint 的选定
  - 解码策略对比（argmax / monotonic / expectation）
  - 阈值校准的偏移量拟合
- **val_holdout**：**严禁参与任何选择过程**，仅用于：
  - 每个 epoch 同步报告一份 QWK（仅用于事后观察，不进入早停或 checkpoint 保存逻辑）
  - 训练结束后报告"最终诚实 QWK"
  - 对比不同 run 改进幅度的唯一可信指标

**(c) 日志规范**

训练每个 epoch 应同时输出：
- val_select QWK（用于决策）
- val_holdout QWK（仅观察）

最终输出 best-checkpoint 时应同时输出：
- val_select 上的 raw 与 calibrated QWK
- val_holdout 上的 raw 与 calibrated QWK

### 验证标准

- val_holdout QWK 应当**显著低于** val_select QWK，差距约在 0.01–0.04 范围（这正是被本文档 P0 暴露出来的偏差）。
- 若两者几乎相等（差 < 0.005），说明切分有问题（可能切了相关样本）或样本量不足以体现噪声差异。
- val_holdout 的 QWK 在多个 seed 间的标准差应大致与样本量平方根成反比，可以作为切分合理性的旁证。

### 回滚条件

- 若 val_holdout 的 200 人中类 3 样本数 < 5，分层切分参数有问题，重新切。
- 若划分后训练集本身不变，但 best QWK（在 val_select 上）显著低于改动前同配置的成绩超过 0.03，说明切分把分布偏移引入到了 val_select，需要重新切分（不应该出现这种情况，但要兜底）。

---

## 3. P1：阈值校准数据源切换

### 目标
即使做了 P0，calibration 仍然会在 val_select 上拟合阈值偏移量，存在轻度过拟合。需要进一步降低 calibration 的偏差。

### 背景证据

当前日志显示 calibration 给 Run 1 带来 +0.029 提升、Run 2 带来 +0.018 提升。本文档第 0 节的分析估计其中**真实提升只有 +0.005 ~ +0.010**，剩余是过拟合到 val 噪声。

### 修改要点（两条可选路径）

**路径 A（与 P0 配套使用，推荐）**：

- 阈值校准在 **val_select** 上做（拟合 offset）。
- 校准后的策略+阈值组合，在 **val_holdout** 上评估。
- val_holdout 上的 calibrated_argmax 比 raw argmax 的提升幅度，才是 calibration 的真实增益。

**路径 B（如果暂时不做 P0）**：

- 阈值校准在 **训练集** 的预测分布上做，而不是 val。
- 流程：训练结束后，把模型切到 eval 模式（关闭 dropout、label smoothing、feature noise、session drop 等所有训练时随机性），在 train 集上跑一遍 forward，得到 train 预测分布。用 train 上的 logits + GT 来拟合阈值偏移。
- 注意：这条路径仍有一定偏差（train 上模型已经拟合得很好，不完全代表泛化分布），但比直接在 val 上拟合干净。

### 路径选择建议

- 如果 P0 已经完成 → 用路径 A，干净且简单。
- 如果出于实施成本暂时不做 P0 → 用路径 B 作为权宜之计，但应在 P0 完成后切回路径 A。

### 验证标准

- 路径 A：val_holdout 上 calibrated 比 raw 的提升应在 +0.005 到 +0.015 范围内。如果超过 +0.02，说明 val_holdout 与 val_select 切分有泄漏。如果 < 0，说明阈值校准方法本身有问题。
- 路径 B：train 上拟合出的阈值偏移幅度应远小于在 val 上拟合时的偏移幅度（前者更"温和"），这是健康的信号。

---

## 4. P2：Top-K Checkpoint 推理时集成

### 目标
用集成的方式平滑单 epoch 峰值的随机性。代价低（只增加推理时间）、收益稳定。

### 背景
当前流程只保存"val 上 QWK 最高的那一个 epoch"。但 QWK 在峰值附近通常有抖动，单个 epoch 的峰值可能是"运气好"（恰好那个 epoch 的随机性给了 val 一个好分数），不一定有最强泛化。

### 修改要点

**(a) 训练阶段：保存 top-K 个 checkpoint**

- 在训练过程中维护一个按 val_select QWK 排序的 top-K 队列。
- 推荐 **K = 3**（最常用且效益好），保守可用 K=5。
- 每次出现新 best 时更新队列，淘汰队列中 QWK 最低的；其他 epoch 不保存。
- 这 K 个 checkpoint 不要求是连续 epoch —— 它们可能分散在训练曲线的不同位置（特别是 Run 2 数据里 epoch 7 和 epoch 16 的双峰情况，恰好对应不同的"局部最优"）。

**(b) 推理阶段：logit 平均**

- 加载这 K 个 checkpoint，分别在 test 数据上做 forward 推理。
- 在 **CORAL logit 层面**做平均（即在 sigmoid 之前的 raw logits 上做算术平均），**不要**在解码后的预测类别上做投票，那会丢失大量信息。
- 平均后的 logits 走原本相同的解码与校准流程。

**(c) 与阈值校准的交互**

- 每个 checkpoint 单独跑一次校准会过拟合 val。
- 推荐做法：先对 K 个 checkpoint 在 val_select 上做 logit 平均，**再在平均后的 logits 上拟合阈值偏移**。也就是把"集成"看作一个整体模型，对它做一次校准。

### 验证标准

- 在 val_holdout 上，K=3 集成的 QWK 应当**至少等于、通常高于**单个 best checkpoint 的 QWK。提升幅度典型在 +0.005 到 +0.015 之间。
- 若集成反而比单 best 低，说明 top-K 中混入了泛化能力差的 checkpoint。检查 top-K 的 val_select QWK 分布：如果 top-3 之间差距 > 0.02，可能 K 太大引入了噪声 checkpoint，缩到 K=2 重试。

### 风险与注意事项

- K 越大，推理时间越长（线性增加），但收益边际递减，K > 5 通常不值。
- 不同 checkpoint 不能直接做**权重平均**（除非它们来自接近的优化轨迹，否则权重空间相加无意义）。logit 平均没有这个问题。
- 如果使用 BF16/FP16 训练，logits 在低精度下做平均要先 cast 回 FP32 再求平均，避免精度损失累积。

---

## 5. P3：Stochastic Weight Averaging (SWA)

### 目标
和 P2 同样是降低单 epoch 峰值的随机性，但走的是另一条技术路线：在**权重空间**做平均，得到更平滑的局部最优。

### 与 P2 的区别

- **P2 (logit ensemble)**：保留 K 个独立模型，inference 时分别 forward 再平均预测。简单、即插即用、可与任何模型组合。
- **P3 (SWA)**：训练时维护一个权重的滑动平均，inference 时只用这一组平均权重。**只有一个最终模型**，inference 成本与原来相同。

### 修改要点

**(a) SWA 启动时机**

- 在训练曲线进入"平台/退化"区之后启动。看你的数据，大约是从 **epoch 7-10 之后**。
- 启动前的权重不参与平均（这些权重还在快速变化，平均它们会引入劣势权重）。

**(b) 平均策略**

- 每个 epoch 结束后，把当前模型权重以 `1/n` 的比例融入到 SWA running average。
- 推荐先做**等权平均**（简单且实证有效），不需要复杂的指数衰减。
- 也可以只在被 P0 的 val_select QWK 高于某个阈值（如平均阈值）的 epoch 加入 SWA average，过滤掉退化期的劣质权重。

**(c) BatchNorm 重统计**

- SWA 的常见陷阱：模型若包含 BatchNorm 层，平均后的 running mean / running var 是"陈旧"的，需要在训练集上跑一遍 forward-only 重新统计。
- 你的模型用的是 TCN + Attention Pooling，**需要核查**是否含 BN。如果只有 LayerNorm / GroupNorm，则不需要重统计。
- 若有 BN，必须实现 SWA 推理前的 BN 重统计步骤。

**(d) Checkpoint 保存**

- 训练结束时保存：原 best checkpoint + SWA averaged checkpoint。
- 在 val_select 上分别评估两个 checkpoint，选优作为最终推理模型。

### 验证标准

- 在 val_holdout 上，SWA checkpoint 的 QWK 应不低于 best single checkpoint。
- 训练曲线如果显示 QWK 在多个 epoch 间持续上下抖动（你的数据典型情况），SWA 收益通常更明显。
- 若 SWA QWK 比 best single 低超过 0.005，可能是：(1) SWA 启动太早，(2) 含 BN 但未重统计，(3) 学习率衰减期权重已经很集中，平均收益小。

### 与 P2 的取舍

- 如果只能做一个：**优先选 P2**（更简单，与训练流程解耦，可与任何架构搭配）。
- 如果资源充足：P2 与 P3 可以叠加 —— 把 SWA averaged checkpoint 作为 top-K 中的一员加入 ensemble，通常能再有小幅提升。

---

## 6. P4：K-fold 交叉验证

### 目标
最稳健的评估与最终集成方案。同时解决三个问题：消除单一切分带来的方差、获得 fold 级别的模型集成、把训练数据利用更充分。

### 修改要点

**(a) 切分**

- 在 **train 集 4200 人内部**做 5-fold 切分。原先的 600 人 val 集可以：
  - 全部并入 train 作为额外训练数据
  - 或保留作为最终的全局 hold-out
- 切分粒度：participant level，分层（按 DASS-21 总分）。

**(b) 训练**

- 每个 fold 训练一个独立模型，使用本文档 P0 之外的所有训练改进（即配合训练改进文档的最终配置）。
- 每个 fold 内部仍然按 train_fold / val_fold（fold 内 4 份训练 + 1 份验证）做正常训练与早停。
- 每个 fold 独立得出 best checkpoint。

**(c) 集成推理**

- Test 阶段：每个 fold 的模型分别 forward，5 份 logits 做平均，再走解码与校准。
- 校准的阈值偏移可以：
  - 每个 fold 单独拟合，最后对偏移量做平均（更稳）
  - 或在 5 个模型的平均 logits 上做一次整体校准（更直接）
- 推荐先试前者。

**(d) 评估**

- 5-fold 的 out-of-fold (OOF) 预测拼起来可以得到 train 集上的诚实预测，这是最可靠的本地 QWK 估计指标。
- OOF QWK 比当前 val QWK 更接近 test QWK，差距通常在 ±0.01–0.02 内。

### 验证标准

- 5 个 fold 的单独 best QWK 应在彼此 ±0.02 范围内。差距过大说明某个 fold 切分有偏。
- OOF 拼接 QWK 应接近 5 个 fold 单独 QWK 的均值，差距 < 0.01。
- 5-fold ensemble 推理 QWK 应比单 fold best 高 0.005–0.020。

### 资源与时间预算

- 训练时间 ×5。配合训练改进文档 P0 的 16 epoch 设置后，单次训练 ~25 分钟，5 fold 共 ~2 小时。
- Inference 时间 ×5，但只在最终提交时跑一次，不是瓶颈。

### 注意事项

- K-fold 与本文档 P0（val 切分）逻辑上有重叠：K-fold 已经隐式实现了"分裂 val 集"的目的。如果做了 K-fold，原 600 人 val 集应该作为额外的全局 hold-out（不参与训练，不参与 fold 切分），作为最终性能的"封测"。
- 与 P2 / P3 完全兼容。最终方案可以是 "K-fold × top-K checkpoint × SWA" 的三重集成，但这会推高推理时间到 K × top_K × 1 = 15 倍。在比赛截止前做最后冲刺时可以考虑，常规迭代不必。

---

## 7. 总体实施与验证流程

### 推荐节奏

```
当前状态（训练改进文档已完成 P0）
   │
   ▼
[本文档 P0] 分裂 val 集为 val_select + val_holdout
   │  从此所有 QWK 数字看 val_holdout 那一栏
   │  原显示的 0.4681 在 val_holdout 上应在 0.42-0.45 区间
   │
   ▼
[本文档 P1] 阈值校准数据源切换
   │  Calibration 提升幅度应从 +0.029 缩到 +0.005~+0.015
   │  Calibration 不再"虚高"
   │
   ▼
此时回到训练改进文档继续 P1-P4
   所有训练改动的"是否有效"判定标准换成 val_holdout QWK
   │
   ▼
训练改进收敛后
   │
   ▼
[本文档 P2] Top-K checkpoint logit ensemble
   │  保留 top-3 checkpoint，inference 时集成
   │  预期 val_holdout QWK +0.005~+0.015
   │
   ▼
[本文档 P3] SWA（可选，与 P2 互为对照或叠加）
   │
   ▼
[本文档 P4] K-fold 交叉验证（最后冲刺）
   │  全量训练成本提升 5×，最终冲分用
   │
   ▼
最终提交
```

### 每一步的产出要求

每实施一项后，agent 应输出：

1. 修改的配置/代码位置（相对路径表达）。
2. 同一套训练配置在改动前后的对照：
   - val_select QWK（raw 与 calibrated）
   - val_holdout QWK（raw 与 calibrated）
   - 训练时间
   - 推理时间（如改动涉及 inference）
3. 与上一步的 delta 分析。

### 红线（任何一项触发立即回滚）

- val_holdout QWK 比上一步下降超过 0.01（噪声范围以外的退化）。
- val_holdout 与 val_select 的 gap 出现反常（如 val_holdout > val_select 超过 0.02，说明切分有泄漏）。
- 集成（P2/P3/P4）后的 QWK 反而比单模型低超过 0.005。
- 推理时间增长不成比例（如 P2 K=3 应大致是 3× 推理时间，若变成 5× 说明实现有问题）。

---

## 8. 不要做的修改

以下方向在当前问题诊断下属于错误优先级，应避免在本文档目标范围内做：

1. **不要在 calibration 时把多种解码策略再堆叠**（如先 expectation 校准、再 argmax 二次校准）。这会进一步加剧 val 过拟合。
2. **不要试图在 val_holdout 上"偷看一下表现就回去调"**。一旦把 val_holdout 用于任何决策（包括"看一眼然后决定下一步做什么"），它就被污染了，失去 hold-out 价值。
3. **不要在没做 P0 的情况下提前做 P4**。K-fold 在脏 val 切分下做会浪费 5× 的训练算力，因为评估指标本身就不诚实。
4. **不要在 P2/P3 之前做 P4**。先用便宜的方法（P2/P3）确认集成方向有收益，再投入 5× 训练。
5. **不要把训练集预测拿来做集成**（不同于 P1 路径 B 中用 train 预测做阈值校准）。模型在 train 上预测已经接近完美，集成 train 预测不会反映泛化。

---

## 9. 交付物清单

实施方在完成全部修改后应交付：

- [ ] val_select / val_holdout 的划分文件（participant ID 列表，附划分逻辑说明与 seed）。
- [ ] 修改后的配置与代码 diff，每项变更标注对应本文档的 P 编号。
- [ ] 每一阶段的对照实验日志，至少包含 val_holdout QWK 的诚实对照。
- [ ] 最终 inference pipeline 的说明（是 single best / top-K ensemble / SWA / K-fold ensemble 中的哪种组合，校准如何做）。
- [ ] 在 val_holdout 上的最终 raw / calibrated QWK 数字，作为对 test 性能的诚实预期。
- [ ] 比赛提交的 submission CSV 与生成它的具体 checkpoint / ensemble 配置。

---

## 10. 心态校准提醒

最重要的一点：**在做完 P0 之后，看到的 QWK 数字会比当前低**。这不是改动让模型变差了，而是消除了过去的"虚高"。

- 当前日志显示的 calibrated 0.4681 是 val 上的"乐观估计"。
- 做完 P0 后 val_holdout 上看到 0.43–0.44 是正常的、诚实的、可信的。
- 此时做训练改进（参见训练改进文档 P1-P4）每带来 +0.005 的 val_holdout 提升，都是**真实的 +0.005**，在 test 上大概率能复现。

实施方应抵制"看到数字降了赶紧把改动回滚"的本能反应。诚实的低数字 > 不诚实的高数字。

---

*文档结束。如遇与本文档冲突的项目约束，以项目实际为准，并在交付报告中注明偏离点与理由。*

# TAAC-2026-SIGKDD Top 5% 0.8316

Timeline:5.17-5.24
主要基于 https://github.com/nhdzTVlxb/TAAC-2026-Tencent-KDD 的方案，重新做了时间特征并删除了冗余模块。

```markdown
# 推荐系统 / CTR 预测模型迭代复盘

> **阶段总结**：本阶段模型迭代从“增加表达能力”逐步转向“控制冗余、降低偏差、提升线上泛化”。主要工作包括结构精简、时间特征风险排查、缺失值与冷启动建模、特殊 dense 字段建模、loss 函数对比以及训练稳定性检查。最终判断：**当前任务的主要矛盾不是模型不够复杂，而是部分模块重复表达同类信号，导致线下收益难以稳定迁移到线上。**

---

## 📚 目录

1. [项目背景与迭代目标](#1-项目背景与迭代目标)
2. [初始模型框架](#2-初始模型框架)
3. [近一个月主要改进方向](#3-近一个月主要改进方向)
   - 3.1 模型结构精简
   - 3.2 时间特征建模修正
   - 3.3 缺失值与冷启动建模
   - 3.4 特征工程与字段建模改进
   - 3.5 Loss 函数改进
   - 3.6 训练策略与稳定性
4. [主要实验结论](#4-主要实验结论)
5. [当前模型迭代的核心判断](#5-当前模型迭代的核心判断)
6. [推荐保留的最终方向](#6-推荐保留的最终方向)
7. [后续工作](#7-后续工作)
8. [总结](#8-总结)

---

## 1. 项目背景与迭代目标

本项目面向推荐系统 / 广告点击率预测任务，目标是在保证训练稳定性和线上泛化能力的前提下，持续提升模型 AUC 表现。近一个月的工作主要围绕模型结构、特征工程、时间建模、loss 设计和训练策略展开。

整体迭代路线经历了一个明显转变：早期更关注通过增加模块提升表达能力；中后期则更加重视结构冗余、时间特征泄漏、缺失值处理风险以及线上线下不一致问题。

| 核心问题 | 具体关注点 |
| --- | --- |
| 模型结构是否冗余 | CrossNet、SE-Net、NS self-attention、NS output fusion 是否重复强化静态侧信号 |
| 时间特征是否过拟合 | time gap、temporal bias、calendar time、cyclical timestamp encoding 是否产生线下虚高 |
| 冷启动与缺失值如何建模 | 缺失值填补、missing indicator、cold/warm gate、history dropout 等方案的稳定性 |
| Loss 是否贴合 AUC | BCE、BCE + pairwise、BCE + PSL-ReLU 的收益和副作用 |
| 训练策略是否稳定 | batch size、seed、sparse embedding reinit、学习率变化对收敛和泛化的影响 |

## 2. 初始模型框架

当前模型主干可以概括为：

- 稀疏特征 embedding
- dense/scalar 特征投影
- user/item NS tokens 建模
- 序列侧 Transformer Encoder
- target-aware attention / item-conditioned query
- CrossNet / SE-Net / NS self-attention 等特征交互模块
- temporal bias / time gap / calendar time 等时间建模模块
- 最终 MLP 输出层

早期版本中，模型同时开启了较多增强组件，表达能力较强，但也带来了明显问题：线下 AUC 容易上涨，线上收益不稳定，说明模型可能在验证集上学习到了局部偏差或时间分布偏差。

| 模块 | 主要作用 | 潜在风险 |
| --- | --- | --- |
| target-aware attention | 让候选 item 与用户历史行为进行目标感知交互 | 如果候选侧信号过强，可能放大热门 item 偏差 |
| NS self-attention | 建模非序列 tokens 内部交互 | 与 CrossNet / SE-Net 有重复表达风险 |
| NS output fusion | 将最后一层 NS tokens 聚合后接回输出层 | 重复注入静态侧信息，线上不稳定 |
| temporal bias | 在 attention score 中加入时间修正项 | 若设为单调衰减，容易过度强调近因假设 |
| time gap | 显式加入行为间隔 embedding | 可能污染 token embedding，与 temporal bias 信息重叠 |
| CrossNet | 建模显式高阶交叉特征 | 层数过多时容易强化验证集偏差 |
| SE-Net | 对特征通道进行动态重标定 | 可能强化训练集中的特征选择偏见 |
| BCE + pairwise / PSL | 增强排序约束，贴近 AUC 目标 | 权重过大时会干扰概率校准 |

## 3. 近一个月主要改进方向

### 3.1 模型结构精简

结构精简是本阶段最核心的方向之一。模型的主要问题不是表达能力不足，而是部分模块重复表达同一类信号，尤其是非序列静态侧特征被 CrossNet、SE-Net、NS self-attention 和 output fusion 同时强化。

#### 3.1.1 CrossNet 调整

```python
num_cross_layers: 2 → 1
```

实验现象表明，单独将 CrossNet 从 2 层降为 1 层，线上效果不一定更好，甚至可能下降。CrossNet 本身并不是最应该直接删除的模块，它确实提供了有效的显式交互能力，但需要避免与 output fusion、SE-Net 等共同造成过拟合。

#### 3.1.2 SE-Net 分析与删减尝试

SE-Net 的作用是对特征通道进行动态加权。风险在于容易强化训练集中的特征偏见。我们曾尝试去掉 SE-Net，但结果显示单纯去掉 SE-Net 不一定稳定提升。阶段性结论：SE-Net 可以作为可控模块保留，但不宜继续增强。相比直接删除 SE-Net，更值得优先处理的是 output fusion、时间特征冗余和 loss 权重。

#### 3.1.3 NS Output Fusion 调整

NS output fusion 的逻辑是将最后一层交互后的 NS tokens 聚合，再拼接或融合回最终输出层。问题在于 NS tokens 已经在主干中参与交互，再次 fusion 容易重复注入同一类静态信息。观察到关闭 output fusion 后，线上结果曾出现提升。阶段性结论：NS output fusion 不是主干必需项，更倾向于关闭或弱化。

### 3.2 时间特征建模修正

时间建模是近一个月最重要的排查方向之一。原模型中存在多类时间信息，包括行为顺序、time gap、time bucket、temporal bias、calendar time 和 timestamp cyclical encoding。这些信息之间存在明显重叠，且部分绝对时间特征可能造成线下验证集虚高。

| 时间特征 | 含义 | 当前判断 |
| --- | --- | --- |
| position / order | 行为在序列中的顺序 | 应保留，是较稳健的序列信息 |
| time gap | 行为之间的时间间隔 | 有冗余和污染 embedding 风险，倾向关闭 |
| time bucket | 时间间隔分桶 | 可保留，但需防止与 time gap 重复 |
| temporal bias | attention score 中的时间修正项 | 比 time gap 更合理，但不应只做单调衰减 |
| calendar time | 小时、星期几等绝对时间 | 可能导致线上泛化不稳定，谨慎使用 |
| cyclical timestamp encoding | hour/weekday 的周期编码 | 需确认是否真正关闭，避免暗含时间泄漏 |

#### 3.2.1 Time Gap 冗余问题

time gap 直接加到序列 token embedding 上。但模型中已经存在 sequence order、time bucket、temporal bias、target-aware attention 等机制，因此 time gap 容易变成重复信息，甚至引入噪声。观察到关闭 time gap 后，模型效果有提升迹象。

#### 3.2.2 Calendar Time 与线上泛化风险

calendar time 包括 hour、weekday、day-level pattern 和 timestamp cyclical encoding。这类特征在线下验证集上可能有效，但在线上测试集中时间分布发生变化时，可能造成泛化下降。重要风险：即使 `run.sh` 中设置了 `no_calendar_time`，模型内部仍可能通过 `use_cyclical=True` 对 timestamp 做周期编码并加到 NS tokens 上。需要确认是否被真正关闭。

#### 3.2.3 Temporal Bias 的作用与改进方向

temporal bias 通常作用在 attention score 上，形式为：

```
attention_score = QK^T / sqrt(d) + time_bias
```

其意义是让模型在注意力分配时考虑行为发生时间。原始 temporal bias 往往隐含“越近的行为越重要”的假设，但某些长期兴趣也可能非常重要。

改进方向：
- 不只使用单调递减时间衰减
- 引入可学习的 time bucket bias
- 让不同 attention head 学习不同时间偏好
- 对短期兴趣和长期兴趣分别建模
- 保留相对时间，不强化绝对 calendar time

### 3.3 缺失值与冷启动建模

冷启动和缺失值是本项目中的核心难点之一。

#### 3.3.1 缺失值填补

最直接的做法是对 dense/scalar 特征进行填补（均值填补或默认值填补）。但观察到加入缺失值填补后 AUC 有下降。原因：填补值不是真实观测；模型会把填充值误认为真实特征；高缺失字段会引入大量伪信号。阶段性结论：**缺失值不应简单填补**，更合理的方式是让模型知道“这里缺失”。

#### 3.3.2 Missing Indicator

后续尝试是在 dense/scalar 特征中加入 0/1 缺失指示器：

```python
missing_indicator = 1 if value is missing else 0
```

这种方法可以区分真实 0 和缺失 0，对冷启动样本更友好。但实验中也发现缺失指示器虽然线下可能提分，线上不一定稳定。应优先加在稳定且有语义的 dense/scalar 特征上，并做分桶评估。

#### 3.3.3 RLC / RCL-style 关键历史行为筛选

思路：由 user_context 与 item_context 生成 conversion-aware query，再用该 query 对历史行为 token 打分，选择 top-k 个最相关行为，并加权汇总回 query tokens。

```
user_context + item_context → conversion-aware query → top-k history behaviors
```

该模块适合 target-aware sequence modeling，尤其适合冷启动或弱历史样本。但 top-k 选择可能不稳定。阶段性结论：是有潜力的方向，但不适合在比赛最后阶段继续大幅改动。

### 3.4 特征工程与字段建模改进

重点分析了 `user_dense_feats_62/63/64/65/66` 与 `user_dense_feats_89/90/91`。结论：不同 dense 字段的语义不同，不应全部混在同一个 dense projection 中。

#### 3.4.1 user_dense_feats_62-66

这组字段更像 category_id 与 value 的配对特征，因此不能简单当成普通 dense 特征处理。更合理的方式是保留 category-value pairing。

```python
value_scaled = sign(x) * log1p(abs(x) / 10)
```

该处理可以压缩长尾数值、保留正负方向，并降低异常值影响。

#### 3.4.2 user_dense_feats_89-91

这组字段更像“几乎固定模板的 int 序列 + 标准化 scalar”，不适合与 62-66 完全相同地处理。后续建模中，对 89-91 的处理更偏向于保留其序列位置含义、区分真实 0 和 padding 0、不做与 62-66 相同的 log-scale，并在 weighted 对齐分支中进行单独建模。

#### 3.4.3 Weighted 对齐分支

后续提出的更细特征建模版本包括：
- 普通 user_dense_token 只保留较稳定字段
- 62-66 / 89-91 进入 weighted alignment branch
- 89-91 中通过 +1 区分真实 0 和 padding 0
- 62-66 只在模型中做一次 log-scale
- 89-91 不做 log-scale

**核心思想**：不同字段不应全部进入同一个 dense projection，而应根据字段结构分别建模。

### 3.5 Loss 函数改进

| Loss 方案 | 目的 | 阶段性结论 |
| --- | --- | --- |
| BCE | 稳定概率预测与 logloss 校准 | 最稳定，应作为主 loss |
| BCE + pairwise | 增加正负样本排序约束，贴近 AUC | 合理，但 pairwise_lambda 不宜过大 |
| BCE + PSL-ReLU | 强化 margin 与排序能力 | 理论有价值，但当前 0.05 权重偏强，线下表现不佳 |

#### 3.5.1 BCE Baseline

最稳定的基础 loss 是 Binary Cross Entropy。稳定、易收敛、对 CTR 任务天然适配。后续所有 loss 尝试都应该以 BCE 为主，不建议完全替换 BCE。

#### 3.5.2 BCE + Pairwise Ranking Loss

为提升 AUC，引入 pairwise ranking 思路：BCE 负责概率校准，pairwise loss 负责正负样本排序。对 batch size 敏感，batch 越大排序信号越稳定。早期配置中使用 `pairwise_lambda = 0.05`，后续判断可能偏大，更推荐 `0.01 - 0.03`。

#### 3.5.3 BCE + PSL-ReLU

后期参考 PSL 思路，配置如下：

```bash
--loss_type bce_psl
--psl_tau 0.2
--psl_lambda 0.05
--psl_score_scale 1.0
```

实验结果显示线下 AUC 下降，logloss 高于原模板。判断 0.05 权重偏强。若继续尝试，应将 `psl_lambda` 降到 0.01 - 0.02。

### 3.6 训练策略与稳定性

#### 3.6.1 Batch Size

更大的 batch size 可以带来更稳定排序信号和更平滑梯度，但过大也可能导致泛化下降、sharp minimum、显存压力增加。需要同步关注学习率、warmup steps、weight decay、pairwise_lambda 和梯度稳定性。

#### 3.6.2 Seed

默认 seed 为 42。seed 不应被当作主要提分手段，可用于多 seed 融合（rank average 或 probability average），但不适合作为单模型调参主线。

#### 3.6.3 Sparse Embedding Reinit

检查了 sparse embedding reinit 相关逻辑：

```python
reinit_sparse_after_epoch = 1
reinit_cardinality_threshold = 0
```

如果改动后第一个 epoch 之后 AUC 雪崩，需要优先排查 embedding reinit / sparse optimizer。最终倾向恢复原始逻辑，避免引入额外不稳定因素。

## 4. 主要实验结论

| 改动方向 | 观察结果 | 结论 |
| --- | --- | --- |
| 缺失值填补 | 不稳定，可能降低 AUC | 不建议作为主线 |
| 缺失值指示器 | 理论合理，但线上不一定稳定 | 谨慎小范围加入 |
| 关闭 time_gap | 有收益迹象 | 推荐保留该方向 |
| 关闭 output fusion | 有线上提升迹象 | 较值得保留的减法 |
| CrossNet 2 层降 1 层 | 单独改动不一定提升 | 需谨慎，不宜盲删 |
| 去掉 SE-Net | 不一定稳定提升 | 不建议盲目删除 |
| BCE + pairwise | 合理，但 lambda 不宜过大 | 可作为小权重辅助项 |
| BCE + PSL-ReLU | 当前 0.05 权重偏强 | 若继续尝试需降权重 |
| calendar time | 有线下虚高和线上掉分风险 | 谨慎使用 |
| temporal bias | 比 time_gap 更合理 | 避免单调近因假设 |
| 89/90/91 特殊建模 | 有必要 | 不宜简单删除某一维 |
| 62-66 pairing | 应保留配对结构 | 不应混入普通 dense |

## 5. 当前模型迭代的核心判断

经过近一个月实验，我们认为当前模型的主要矛盾不是表达能力不够，而是模型中存在过多可能重复表达同一类信号的模块。

> **核心判断：减少冗余信号 > 继续堆叠复杂模块**

- NS tokens 已经建模静态特征，但 output fusion 再次注入；
- CrossNet、SE-Net、NS self-attention 都在强化非序列侧交互；
- time_gap、time_bucket、temporal bias、calendar time 同时存在；
- 缺失值填补和缺失指示器可能同时影响 dense 表征；
- pairwise / PSL loss 可能干扰 BCE 的概率校准。

因此，后期更有效的策略不是继续堆模块，而是**减少冗余信号、保留稳定交互、降低时间泄漏风险、控制 loss 辅助项权重，并提升线上泛化能力**。

## 6. 推荐保留的最终方向

1. 保留 RankMixer / Transformer / target-aware attention 主干；
2. 保留必要的 NS tokens；
3. 关闭或弱化 NS output fusion；
4. 关闭 time_gap；
5. 谨慎使用 calendar time；
6. 保留 temporal bias，但避免过强单调衰减；
7. CrossNet 不盲目删除，优先控制层数和低秩维度；
8. SE-Net 不继续增强；
9. loss 以 BCE 为主；
10. 若使用 pairwise 或 PSL，只使用小权重；
11. 对 62-66、89-91 这类特殊字段单独建模；
12. 避免大规模缺失值填补；
13. 通过分桶评估观察 cold/warm 样本表现；
14. 最终可以考虑多 seed / 多模型 rank average 融合。

## 7. 后续工作

### 7.1 稳定性验证

- new user / old user 分桶
- new item / old item 分桶
- 高缺失样本 / 低缺失样本分桶
- 长历史 / 短历史用户分桶
- 不同时间段样本分桶

目标：判断模型提升来自真实泛化，还是来自某类样本偏差。

### 7.2 轻量模型融合

建议保留多个差异化模型：稳定 BCE baseline、BCE + small pairwise、关闭 output fusion 的减法模型，以及特殊 dense 字段增强模型。最终采用 rank average。相比单模型继续调参，融合更可能带来稳定线上收益。

### 7.3 冷启动专项优化

后续冷启动方向不建议继续简单填补缺失值，而应转向：
- cold/warm gate
- content feature tower
- ID dropout / history dropout
- missing-aware dense token
- candidate-history 显式匹配
- OOF 统计先验旁路

## 8. 总结

近一个月的模型迭代可以总结为：我们从增强模型表达能力出发，逐步发现当前任务的关键问题不是模型不够复杂，而是模型中存在较多冗余模块和不稳定信号。后续优化重点应从“加模块”转向“做减法、控偏差、稳泛化”。

最终更可靠的路线是保留主干序列建模能力，弱化高风险 fusion 和绝对时间特征，以 BCE 为基础 loss，并用小权重排序损失或模型融合提升 AUC。

> **线下 AUC 提升不等于线上收益。越到比赛后期，越要重视结构简化、时间特征风险、loss 权重控制和泛化稳定性。**

--- 

*整理日期：2026 年 5 月 24 日*
```

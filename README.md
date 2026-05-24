
# TAAC-2026-SIGKDD Top 5% 0.8316
<img width="1520" height="304" alt="c5056db36fec57c17bb0e01e89150b52" src="https://github.com/user-attachments/assets/1293210e-c426-4c43-9d52-899411de64b5" />

ori version是带缺失值填充和RCL的版本，最高得分0.8288，best version是基于 https://github.com/nhdzTVlxb/TAAC-2026-Tencent-KDD 改的，最高得分0.831*

# 推荐系统 / CVR 预测模型迭代复盘

> **阶段总结**：本阶段模型迭代从“增加表达能力”逐步转向“控制冗余、降低偏差、提升线上泛化”。主要工作包括结构精简、时间特征风险排查、缺失值与冷启动建模、特殊 dense 字段建模、loss 函数对比以及训练稳定性检查。最终判断：**当前任务的主要矛盾不是模型不够复杂，而是部分模块重复表达同类信号，导致线下收益难以稳定迁移到线上。**


## Timeline:4.25-5.17

主要基于baseline对数据以及模型进行更改，auc最高到达0.8288

**主要尝试方向：**

**1. 时间特征工程**

在ns token上直接加上固定不可学习的sin/cos循环时间编码（weekday和hour），测试集auc到达0.823

**2. pair特征处理**

在rankmxier tokenizer内部对pair特征的dense部分进行log1p后与int emb拼接并过一层Linear，auc到达0.826

**3. 解决冷启动与缺失值问题**

最主要的尝试是引入 UserSimilarityNullImputer，同时在user_dense加入缺失值指示器（0/1编码），线上提升了1.5k。

它的目标是对用户侧缺失值做补全。

核心思路是：根据 batch 内用户特征计算用户相似度
        ↓
找到相似用户
        ↓
用相似用户的信息补当前用户缺失字段

**4. RCL-style 关键历史行为筛选**

思路：由 user_context 与 item_context 生成 conversion-aware query，再用该 query 对历史行为 token 打分，选择 top-k 个最相关行为，并加权汇总回 query tokens。

```
user_context + item_context → conversion-aware query → top-k history behaviors
```

该模块适合 target-aware sequence modeling，尤其适合冷启动或弱历史样本。但 top-k 选择可能不稳定,在基准模型上有用。

RCL的思路与DIN也比较相近。

**4. 模型架构的一些尝试**

HyFormer 模块改进过程

在 HyFormer 主干结构的改进过程中，我们主要围绕注意力机制、Deep Interest Network 以及 query generation 三个方向进行了探索。整体来看，这一阶段的目标是增强模型对用户历史行为、候选 item 以及非序列特征之间关系的建模能力，但实验结果也表明，部分理论上合理的结构在当前数据和特征工程组合下并未稳定带来收益。

1. TokenFormer Gate 模块的尝试

首先，我们参考了腾讯 TokenFormer 相关工作中的门控思想，在模型中的注意力机制上加入了类似的 gate 结构。该模块的核心思路是利用 query 对 attention output 进行调制，使得注意力输出不再是直接进入后续网络，而是经过一个由 query 控制的动态门控后再参与特征融合。

具体来说，原始注意力机制主要通过 query、key、value 计算 attention score，并得到加权后的 value 表示。而加入 TokenFormer-style gate 后，query 不仅用于计算注意力权重，也进一步参与控制 attention output 的保留程度。这样做的直觉是：不同样本、不同候选 item 下，模型应该动态判断当前 attention 输出是否可靠，以及应该以多大强度注入后续表示。

从实验结果来看，该版本的线下收益并不明显，没有带来稳定的 AUC 提升。但是结合线上测试表现，我们认为该模块对泛化能力并没有明显破坏，甚至表现出一定的稳定性。这说明 query-gated attention 的方向是相对安全的，它可能不会直接大幅提升线下指标，但在控制 attention 输出、降低过拟合风险方面具有一定价值。

因此，这一部分实验的结论是：TokenFormer-style gate 并不是当前阶段最强的提分模块，但它验证了“通过 query 控制 attention 输出强度”这一思路在泛化层面具有一定可行性。

2. Deep Interest Network 模块的尝试

第二个重点方向是 Deep Interest Network，也就是 DIN 模块。我们当时关注到许多推荐系统经验中都提到，DIN 在 CTR 任务中往往能带来明显收益，因此我们进一步查找相关论文，并尝试将其引入到原始模型中。

DIN 的核心思想是：用户兴趣并不是固定不变的，而应该针对不同候选 item 动态生成。换句话说，在预测用户是否会对某个 item 感兴趣时，并不是所有历史行为都同等重要，模型应该根据当前 candidate item 动态计算历史行为序列中每个行为的权重。

用更直观的例子来说，如果当前要预测用户是否对“泳衣”感兴趣，那么用户历史中的“泳镜”“游泳帽”等行为可能应获得更高权重，而与游泳场景无关的历史行为则应被赋予较低权重。DIN 正是通过 target-aware attention 的方式，为不同候选 item 生成不同的用户兴趣表征。

在原始模型上，我们围绕 DIN 尝试了多种改造方式，包括：

将原始 Transformer block 替换为 DIN 模块；
将 cross attention 替换为 DIN 模块；
将 DIN 作为旁路分支作用于最终 output；
将 DIN 表征提前作用于模型输入；
将 DIN 与原有 sequence representation 进行融合。

但这些版本在当时的实验中基本都没有取得正向收益，部分版本甚至出现了接近 0.01 的 AUC 下降。由此可见，DIN 虽然在理论上非常适合推荐系统中的 target-aware interest modeling，但直接替换原模型中的关键结构并不一定有效，甚至可能破坏原有 HyFormer 主干中已经形成的序列建模能力。

后续在开源版本中，我们再次观察到 DIN 模块的使用。回顾之前的实验可以发现，我们早期已经实现过类似思路，但当时由于和其他模块共同作用，未能体现其有效性。后来在后续版本中做消融实验时发现，DIN 模块在当前模型中是有贡献的，去掉后大约会带来千二左右的 AUC 下降。

这说明 DIN 模块本身并不是无效的。早期实验失败，更可能是因为 DIN 与其他特征工程或结构模块之间发生了冲突。例如，缺失值填补、时间特征注入、RLC-style 历史筛选等模块都可能改变用户历史行为或候选 item 表示。当这些模块同时存在时，DIN 的 target-aware attention 可能会被伪特征、错误时间信号或噪声行为误导，导致原本应该提升匹配能力的模块反而产生负收益。

因此，对 DIN 的最终判断是：DIN 是一个有效但敏感的模块。它需要建立在相对干净、稳定的输入表示之上。如果与不稳定的缺失值填补、过强的时间特征或硬选择历史行为模块混合使用，可能出现模块间互相干扰，最终导致“1 + 1 < 0”的效果。

3. Query Generation 的改进尝试

第三个方向是 query generation 的改进。在原始模型中，query 的生成依赖于 sequence mean pooling。我们最开始认为这种方式过于粗糙，因为简单地对用户历史序列做平均池化，可能会损失大量序列内部信息。

从直觉上看，mean pooling 存在几个明显问题：

它忽略了历史行为的重要性差异；
它无法区分近期行为和长期兴趣；
它会把噪声行为和关键行为同等平均；
它对候选 item 缺乏动态适配能力；
它可能削弱用户兴趣的多峰结构。

因此，我们尝试使用多种方式替代或增强 mean pooling，例如引入注意力池化、target-aware pooling、更加复杂的 query generator，以及利用序列最后行为、加权历史表示等方式生成 query。

但实验结果并不理想。多个替代方案上线下验证后，AUC 都出现下降，并没有找到一个稳定优于 mean pooling 的方法。

这说明，虽然 mean pooling 在结构上看起来简单，但它在当前模型中可能承担了一个非常稳定的全局兴趣摘要作用。相比更复杂的 query generation 方法，mean pooling 的优势在于：

参数少，不容易过拟合；
表征平滑，抗噪能力较强；
不依赖额外注意力结构；
与原始 HyFormer 主干配合较稳定；
不会过度放大某些异常历史行为。

因此，我们最终没有贸然替换 mean pooling，而是保留其作为 query generation 的基础部分。后续如果继续改进 query generation，更合理的方向可能不是完全替换 mean pooling，而是在其基础上增加轻量残差或门控增强，让模型在保留稳定全局兴趣表示的同时，适度引入 target-aware 或 recency-aware 信息。

阶段性总结

总体来看，HyFormer 模块上的改进经历了从激进替换到谨慎增强的过程。

TokenFormer-style gate 的实验说明，通过 query 调制 attention output 是一个相对安全的方向，虽然线下收益不明显，但对泛化能力没有明显破坏。DIN 模块的实验则表明，target-aware interest modeling 本身是有价值的，但它对输入质量和周围模块非常敏感。如果与缺失值填补、时间特征或其他历史筛选模块发生冲突，可能导致明显掉点。Query generation 的实验进一步说明，原始 mean pooling 虽然简单，但在当前模型中具有较强的稳定性，复杂替代方案并不一定更优。

因此，这一阶段的重要经验是：在 HyFormer 这类已经具备较强序列建模能力的主干上，不能简单地将某个推荐系统模块整体替换进去。更稳妥的做法是采用轻量化、残差化、门控化的方式进行增强，避免破坏原主干已经学习到的稳定表示。


## Timeline:5.17-5.24

主要基于 https://github.com/nhdzTVlxb/TAAC-2026-Tencent-KDD 的方案，重新做了时间特征并删除了冗余模块。
---

## 📚 目录

1. [项目背景与迭代目标](#1-项目背景与迭代目标)
2. [初始模型框架](#2-初始模型框架)
3. [主要改进方向](#3-主要改进方向)
   - 3.1 模型结构精简
   - 3.2 时间特征建模修正
   - 3.3 缺失值与冷启动建模
   - 3.4 特征工程与字段建模改进
   - 3.5 Loss 函数改进
   - 3.6 训练策略与稳定性
4. [主要实验结论](#4-主要实验结论)
5. [当前模型迭代的核心判断](#5-当前模型迭代的核心判断)
6. [推荐保留的最终方向](#6-推荐保留的最终方向)
7. [总结](#8-总结)

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


## 3. 基于源神模版的主要改进方向

### 3.1 模型结构精简

结构精简是本阶段最核心的方向之一。模型的主要问题不是表达能力不足，而是部分模块重复表达同一类信号，尤其是非序列静态侧特征被 CrossNet、SE-Net、NS self-attention 和 output fusion 同时强化。

##### 3.1.1 CrossNet 调整

```python
num_cross_layers: 2 → 0 num_cross_layers: 2 → 3
```

实验现象表明，单独将 CrossNet 从 2 层降为 1 层，从2层升到3层，线上效果不一定更好，甚至可能下降。CrossNet 本身并不是最应该直接删除的模块，它确实提供了有效的显式交互能力，但需要避免与 output fusion、SE-Net 等共同造成过拟合。

#### 3.1.2 SE-Net 分析与删减尝试

SE-Net 的作用是对特征通道进行动态加权。风险在于容易强化训练集中的特征偏见。我们曾尝试去掉 SE-Net，但结果显示单纯去掉 SE-Net 线上auc结果下降。阶段性结论：SE-Net 可以作为可控模块保留，相比直接删除 SE-Net，更值得优先处理的是 output fusion、时间特征冗余。

#### 3.1.3 NS Output Fusion 调整

NS output fusion 的逻辑是将最后一层交互后的 NS tokens 聚合，再拼接或融合回最终输出层。问题在于 NS tokens 已经在主干中参与交互，再次 fusion 容易重复注入同一类静态信息。我们尝试将output fusion直接关掉，观察到关闭 output fusion 后，测试结果出现提升（上升了千一）。阶段性结论：NS output fusion 不是主干必需项，利用ns output fusion 会导致过拟合，需要删去。

### 3.2 时间特征建模修正

时间建模是近一个月最重要的排查方向之一。原模型中存在多类时间信息，包括行为顺序、time gap、time bucket、temporal bias、calendar time 和 timestamp cyclical encoding。这些信息之间存在明显重叠，且部分绝对时间特征可能造成线下验证集虚高。

| 时间特征 | 含义 | 当前判断 |
| --- | --- | --- |
| position / order | 行为在序列中的顺序 | 应保留，是较稳健的序列信息 |
| time gap | 行为之间的时间间隔 | 有冗余和污染 embedding 风险，倾向关闭 |
| time bucket | 时间间隔分桶 | 可保留，但需防止与 time gap 重复 |
| temporal bias | attention score 中的时间修正项 | 比 time gap 更合理，但不应只做单调衰减 |
| calendar time | 小时、星期几等绝对时间 | 可能导致线上泛化不稳定，谨慎使用 |

#### 3.2.1 Time Gap 冗余问题

time gap 直接加到序列 token embedding 上。但模型中已经存在 sequence order、time bucket、temporal bias、target-aware attention 等机制，因此 time gap 容易变成重复信息，甚至引入噪声。观察到关闭 time gap 后，模型效果有提升迹象。

#### 3.2.2 Calendar Time 与线上泛化风险

calendar time 包括 hour、weekday、day-level pattern 和 timestamp cyclical encoding。这类特征在线下验证集上可能有效，但在线上测试集中时间分布发生变化时，可能造成泛化下降。重要风险：即使 `run.sh` 中设置了 `no_calendar_time`，模型内部仍可能通过 `use_cyclical=True` 对 timestamp 做周期编码并加到 NS tokens 上。需要确认是否被真正关闭。

#### 3.2.3 Temporal Bias 的作用与改进方向

temporal bias 通常作用在 attention score 上，形式为：

```
attention_score = QK^T / sqrt(d) + time_bias
```

其意义是让模型在注意力分配时考虑行为发生时间。原始 temporal bias 往往隐含“越近的行为越重要”的假设，但某些长期兴趣也可能非常重要，在消融实验中，去掉了temporal bias模块之后，auc出现了下降，可以初步判断temporal bias对于最终的auc具有正向贡献。

也许可以尝试的改进方向：
- 不只使用单调递减时间衰减
- 对短期兴趣和长期兴趣分别建模

#### 3.2.4 NS侧时间特征
目前为止我们的所有尝试都表明ns侧和seq侧时间特征存在冲突，两边同时加时间特征会导致auc严重下降。
最终我们选择关闭除时间分桶外的所有序列时间特征，改在为ns token侧注入hour和weekday的固定循环时间编码，我们尝试过可学习时间编码，但是效果明显变差。

### 3.3 缺失值与冷启动建模

原模型的缺失值的填补方法在这一般失效，经多轮验证其与RLC与后面的DIN冲突，思考其原因可能是：缺失值的填补放大了噪声，强行补全缺失值反而会破坏原始缺失模式，引入不稳定的伪信号，并降低线上泛化能力。后续版本不再采用强缺失值填补，而是转向保留缺失事实、减少伪信号污染，并对特殊字段进行单独建模。

### 3.4 特征工程与字段建模改进

重点分析了 `user_dense_feats_62/63/64/65/66` 与 `user_dense_feats_89/90/91`。结论：不同 dense 字段的语义不同，不应全部混在同一个 dense projection 中。

#### 3.4.1 user_dense_feats_62-66

这组字段更像 category_id 与 value 的配对特征，因此不能简单当成普通 dense 特征处理。更合理的方式是保留 category-value pairing。

```python
value_scaled = sign(x) * log1p(abs(x) / 10)
```

该处理可以压缩长尾数值、保留正负方向，并降低异常值影响。

#### 3.4.2 user_dense_feats_89-91

删去


### 3.5 Loss 函数改进

试了以下多种方法，未取得收益

| Loss 方案 | 目的 | 
| --- | --- | --- |
| BCE | 稳定概率预测与 logloss 校准 |
| BCE + pairwise | 增加正负样本排序约束，贴近 AUC |
| BCE + PSL-ReLU | 强化 margin 与排序能力 |

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

改为512，线上auc下降。

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

| 改动方向 | 实验结果 | 结论 |
| --- | --- | --- |
| 缺失值填补 | 负效果 | 删去 |
| 缺失值指示器 | 负效果 | 关闭 |
| 关闭 time_gap | 有收益迹象 | 保留 |
| 关闭 output fusion | 有提升迹象 | 保留 |
| CrossNet 2 层降 1 层 | 单独改动不一定提升 | 未测试 |
| 去掉 SE-Net | 负效果 | 保留 |
| BCE + pairwise | 合理，但 lambda 不宜过大 | 可作为小权重辅助项 |
| BCE + PSL-ReLU | 当前 0.05 权重偏强 | 若继续尝试需降权重 |
| calendar time | 有线下虚高和线上掉分风险 | 谨慎使用 |
| temporal bias | 比 time_gap 更合理 | 避免单调近因假设 |
| 89/90/91 特殊建模 | 待验证 | 不宜简单删除某一维 |
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

## 7. 总结

近一个月的模型迭代可以总结为：我们从增强模型表达能力出发，逐步发现当前任务的关键问题不是模型不够复杂，而是模型中存在较多冗余模块和不稳定信号。后续优化重点应从“加模块”转向“做减法、控偏差、稳泛化”。

最终更可靠的路线是保留主干序列建模能力，弱化高风险 fusion 和绝对时间特征，以 BCE 为基础 loss，并用模型融合提升 AUC。

> **线下 AUC 提升不等于线上收益。越到比赛后期，越要重视结构简化、时间特征风险、loss 权重控制和泛化稳定性。**

--- 

*整理日期：2026 年 5 月 24 日*

# MTS-FSQ 首轮训练结果复核与下一轮推进计划

日期：2026-09-18。基于E00–E07训练产物、现有checkpoint、原始验证协议和本次独立重算。

## 1. 结论与路线选择

**本轮完成了有价值的先导实验，但还没有验收通过“可用的参考风格迁移”。建议保留 frozen NEF + local logit operator 主线；暂停新增 CTMC 实验与 reference 长训练，先统一评估、修正内容标签混淆、补齐实际编辑质量，再用受控实验诊断 reference descriptor 失效。**

这不是训练白做了：数据到模型的学习链路可运行，context infill显著优于边际基线，style-ID有小而一致的似然信号，CTMC和reference分支的弱点已暴露。但“程序完成”“条件似然改善”“可见风格迁移成功”需要分别报告。

| 分支 | 当前证据支持 | 不能据此声称 |
|---|---|---|
| base transport | 有context的masked completion可学习，值得作诊断上游 | 自由生成可用、动作物理质量已达标 |
| style-ID logit | 固定验证集上存在较小style条件信号 | 可见style fidelity、content保持和local编辑均已成功 |
| constant/no-reference | 不看style也能修补base的大部分似然误差 | 修补收益就是style learning |
| birth-death CTMC | 数值运行稳定；此预算下似然不及logit | CTMC普遍无效，或ordinal几何已被否证 |
| reference logit | 已训练模型几乎不随reference变化 | 训练前后一定发生“坍缩”，或reference模型天然不可行 |

E07建议改成：`base=context_infill_promising`、`style_input=weak_likelihood_signal`、`ctmc=current_budget_no_advantage`、`reference=reference_invariant`、`motion_style_quality=not_evaluated`。保留原E07作为历史，不原地擦除。

## 2. 这次复核做了什么

- 读取E03–E06 summary/history/checkpoint、213行operator协议、320行base协议、独立verification脚本与JSON。
- 在CPU、eval模式下，用已有四个operator best checkpoint重算共同213行：保留正确action条件，统一mask、style map和NLL聚合。没有反向传播或更新参数。
- 另重算遗漏action的诊断分支，复现旧verification数字，确定差异来源。
- 逐tensor比较四个operator中冻结transport state：**完全相同**，不是上游权重混用造成的排序。
- 检查reference descriptor与batch内参考交换敏感性；该交换只作敏感性诊断，不冒充严格匹配wrong-style benchmark。
- 核对真实train及验证协议的action×style列联表，查看tokenizer contact sheet与近期bind/mirror修正说明。

新增可复核证据：

```text
outputs/mts_training_analysis_20260918/rescore.py
outputs/mts_training_analysis_20260918/corrected_rescore.json
```

脚本可重放CPU重算；产物保存每行sample_id、clip、kind、style、监督token数、各模型NLL sum、分组聚合及checkpoint SHA。本次没有启动新训练，也没有修改学习实现。

## 3. 主要结果：哪些可以保留

### 3.1 Base：补全明显进步，全遮盖仍没有超过更强数据基线

2,000步训练已完成，best在1,800步。320行固定val协议：

| 模型/基线 | 按mask-kind权重汇总NLL，nat/token |
|---|---:|
| step0 | 2.37173 |
| per-coordinate unigram | 1.98570 |
| best | 1.52342 |
| last | 1.53307 |

这些值支持模型学会了利用上下文。最佳与最后仅差0.00965；单次回升不能判断已收敛或严重过拟合。

但在 **full_generation** 上：

| 方法 | NLL |
|---|---:|
| per-coordinate unigram | 2.04230 |
| **action-conditioned unigram** | **2.01571** |
| transport best | 2.03021 |

因此“full-generation提升0.012”只针对较弱的无action边际基线；与已有的action-conditioned基线比较，模型反而差 **0.01450**。下一轮不能仅用总体平均分数盖过full-mask短板。

现有报告中的8个case与root速度/粗糙度proxy不足以验收生成质量。尤其 `e03_verification.py::motion_proxies` 直接读取normalized decoder feature计算速度比例，并把一阶root velocity差/速度称为jerk_ratio；它不是米制速度或三阶位置jerk。相关“采样粗3.1–3.3倍”应标为特征空间proxy，不能作为物理结论。

**推荐判断：** base允许用于受控context editing诊断；full generation仍待改进。不是必须立即重训base，但后续style实验要分开报告full-mask与context组。

### 3.2 Operator：正确action条件下的统一重算

四臂各1,000步，同seed3407、同base、同213行val。以下统一使用原协议的mask-kind权重；下表不混用micro/macro：

| 方法 | 正确条件NLL | 错style-ID NLL | 相对base改善 |
|---|---:|---:|---:|
| 冻结base | 1.566107 | — | — |
| constant/no-reference logit | 1.543817 | — | 0.022289 |
| **style-ID logit** | **1.538102** | 1.550529 | **0.028004** |
| reference logit | 1.545247 | — | 0.020859 |
| style-ID CTMC | 1.556606 | 1.561002 | 0.009501 |

style-ID对constant的净改善为 **0.005715 nat/token**；对错误ID为 **0.012427**。净style收益占它相对base总改善的约 **20.4%**；约79.6%也能由constant获得。这是似然改善分解，不是严格因果贡献比例。

三个style和五种mask上的style-ID vs constant方向仍均为正，但effect很小。逐style按监督token加权的净差：injured leg约0.00223、injured torso约0.01583、neutral约0.00307。213行中157行正确ID优于平均错误ID。

**可保留结论：有弱style条件信号，主要收益仍是base误差修补。** 无独立评估和动画时不将其升格为完整motion style transfer。

### 3.3 CTMC：保留数值稳定结论，暂缓“ordinal无优势”的推断

correct-ID协议NLL比logit差约 **0.018504**；本预算下没有似然优势。已有训练history的tail/mass/rate记录支持数值未失稳，这不是当前主要失败原因。

但旧ordinal分析有两个问题：

1. `base_mode±1` 经clamp后在边界重复计入同一bin；one-hot位于level0时，所谓adjacent fraction甚至可以得到2。
2. 从base argmax计算styled marginal的距离，不是CTMC的实际 `p0(i)K(i,j)` transition displacement，也不是两个任意marginal间唯一确定的质量搬移。

因此当前结论应为 **“该birth-death参数化在此预算下未取得NLL优势”**。暂不追加CTMC预算是合理资源决策，但不能据此宣称FSQ ordinal geometry已被证伪。

若未来重新比较：共用1D Wasserstein距离 `Σ|CDF_p−CDF_q|`、TV、matched style fidelity下的decoded代价；仅对拥有明确kernel的方法另报 `Σ p0(i)K(i,j)|i−j|`。不要把两种量混成同一指标。

### 3.4 Reference：失效可复现，但具体成因还未证明

本次在213条reference输入（191个不同reference clip）复测：

```text
descriptor mean norm       = 0.539569
mean deviation norm        = 6.43e-6
relative spread            = 1.19e-5
batch reference swap TV    ≈ 2.51e-8
```

这直接支持 **“当前已训练encoder接近reference-invariant”**。并不依赖旧NLL脚本是否漏传action。

但没有该encoder初始化/中间checkpoint的同协议响应，不能区分“训练导致坍缩”与“初始化/架构从一开始就对token不敏感，训练未修复”。下一步先测init→trained各层响应、梯度和输入依赖，再决定加辅助loss；不能只凭最终小方差立即断言机制。

reference臂可学习参数约578万，style-ID约14.4万。相同step不是相同训练FLOPs，也不证明复杂模型已充分收敛；但当前弱响应已经足以否定“本checkpoint可用来做reference风格控制”。

## 4. 报告中必须更正的实验口径

### 4.1 E04–E06 verification漏传了action

三份输出目录内verification脚本均读取：

```python
checkpoint["metadata"]["model_config"].get("content_vocabulary")
```

operator bundle的词表实际在 `model_config["transport"]["content_vocabulary"]`，可直接用 `model.transport.content_vocabulary`。旧脚本得到None，builder未填content_condition。

本次重算复现了旧E04的无action micro NLL 1.713063和constant 1.720332，确认错误来源。训练与训练器val使用了正确条件，**不因此废弃checkpoint或重训**；应修复只读评估并重出报告。

### 4.2 NLL三种平均被混用，改善符号也写反

- 协议objective：每kind内token加权，再按固定kind权重汇总。
- micro：所有监督token合并。
- macro-row：各row NLL直接平均。

旧E05 overall实际是macro-row，却被文档标为token加权；表中logit base引用另一处micro结果，产生“同一base变成两个数”的假象。本次相同condition下四个base完全一致。

`correct_minus_wrong`部分字段实际计算的是 `wrong−correct`。统一改成 `nll_improvement_vs_wrong`，正数表示改善，避免负号误读。

### 4.3 E04没有完成原先的motion验收

verification判据仅覆盖completion、NLL gap、kind/style方向和finite，没有实现计划中content/physics“不全面恶化”和动画“有方向style变化”的门槛。文档自己也承认physics/FK尚未运行。

改为：`execution=complete`、`likelihood_signal=weak_positive`、`motion_style_acceptance=pending`。E03的Go也收窄为可用于补全诊断，不是full-generation质量Go。

### 4.4 Val反复筛查不能当最终泛化证据

本轮213行用于选best、筛CTMC、调strength、判reference。它是开发集，不是未触碰的最终test。strength=1.5最优只是这组开发数据的观察；不得不加标注地把它当test超参。

E06真实wrong-reference对照只保留143行，跳过70行；这143行全部来自 `Basic Locomotion Styles`，只覆盖两个injured风格，不支持“三个style均通过”的设计要求。

此外E06 JSON中存在NaN的TV字段：写键和汇总读键不一致。nearest-centroid leave-one-row-out没有剔除同reference clip/take的其他row，也不能作为独立、无泄漏的style evaluator。恢复原始数值时应同时修严格JSON、样本计数和分组单位。

## 5. 最重要的数据问题：content标签自身混入style

当前operator val的213行分布，经原始protocol逐行核对：

| content/action标签 | style | 行数 |
|---|---|---:|
| Basic Locomotion Styles | injured leg | 74 |
| Basic Locomotion Styles | injured torso | 69 |
| Basic Locomotion Neutral | neutral | 38 |
| Baseline | neutral | 25 |
| Advanced Locomotion | neutral | 6 |
| Complex Actions | neutral | 1 |

在这组val里，`Basic Locomotion Styles`几乎直接告诉模型“非neutral”；其余四类全部neutral。训练分布也强相关，例如 `Basic Locomotion Styles` 中injured leg3510、injured torso3238、neutral586；`Basic Locomotion Neutral`中neutral26336。

**影响：** condition不是真正独立于style的纯content，base/constant已经能利用这条捷径。same-style/different-content标签也未必意味着细粒度动作语义确实不同。reference缺少贡献可能与此有关，但仍是待验证解释，不能单因果归因。

下一轮应建立明确的`content_schema`：第一轮最小ablation把 `Basic Locomotion Neutral/Styles` 合并为 `Basic Locomotion`，保留原标签和映射来源；不改tokenizer或token字节。该合并只去掉明显的标签命名混淆，不保证同类内部的走/跑/转身、速度和接触阶段已匹配。再用clip/action元数据和少量人工核对构建真正的内容匹配case。

更换content map会改变transport条件语义，必须新训练run、新map/hash，不能给旧checkpoint重编号后静默加载。

## 6. Tokenizer与物理测量的约束

已查看9段tokenizer contact sheet：日常片段接近source；jumping jack、盘腿和躺地动作的差异可见。它们是tokenizer自重建，不是operator生成，不能用来验收style效果。

近期发现的bind skeleton与mirror问题应优先统一到所有度量入口。`evaluate_nef_physics.py`仍直接读取checkpoint `stats['ref_pos']`；旧米制FK/contact/foot-height结果不能因为NLL不变就自动保留物理结论。新bind修正可能改变FK/contact门槛和模型排序，应按修正前/后版本区分。

Kabsch对齐能说明误差可被整体刚体变换解释一部分，但 **不能证明错误必然来自global/root stream**。逐帧最佳刚体拟合可能吸收多关节协同误差；`raw_mean−aligned_mean`也不是正交的因果误差分解。

在决定重训tokenizer前，用真实feature做反事实重建：仅替换global-owned features、仅替换Hips/root rotation通道、仅替换局部关节流，分别看FK改善。所有索引从layout取，不手写误归属。镜像和非镜像均保留真实身份，不能把镜像case换成另一clip的图片后当同一case得分。

## 7. 下一轮执行层计划（N00–N06）

面向zcode + deepseek-v4.1-flash。单agent串行推进，一次一个任务；保留所有已有工件与dirty工作区。新的真实训练预算单独批准，历史E阶段授权不自动扩展到本轮。

### N00：统一结果汇总，先重评现有模型

**可立即做，无训练。** 文件：`scripts/evaluate_mts_operator.py`、`metrics.py/eval_protocol.py`和现有测试；把输出目录里的临时verification逻辑收敛到一个正式只读summary入口，避免再维护E04/E05/E06三套评分。

实施：

1. 从实际loaded transport取action vocabulary；conditioner存在却condition=None时默认拒绝评估。显式`omit_action_ablation`才允许缺条件，并改变protocol标签。
2. 每行保存nll_sum/count、sample/take/actor/content/style、model SHA、mask、strength、reference身份。
3. 同时输出micro、macro-row、protocol-weighted三种值，字段带聚合名；表格禁止跨口径拼接。
4. 实际比较四个transport state及输入base distributions；发现不等立即中止“同base”比较。
5. 修NaN、差值符号、hardcoded same_seed=True等；seed/预算从checkpoint读取并验证。
6. 固化本次rescore反例：正确action评分应复现checkpoint val objective（float32容差，例如1e-5）；缺action必须改变实验标签或报错。
7. 输出新 `screening_decision_v2.json`，保留旧文件，标明effect/physics/test哪些仍未测。

**验收：** 213行统一重算与本报告吻合；三种聚合有独立手算fixture；无NaN；每个“pass”可追溯到实际执行的检查。

### N01：统一FK、镜像与物理真值

**无训练，最多先做8个确定性窗口。** 文件：`anim/features.py`、`nef_probe.py`、physics/locality/transport/operator evaluator；只处理测量一致性，不改网络。

实施：

- 建一个共同FK context入口，显式使用bind asset、单位、joint order、mirror transform；保存这些字段和asset SHA。
- 用至少一个非镜像及其真实镜像对，从BVH/direct FK获得独立真值；验证raw features重建与真值位置/朝向一致，不能只比较两个共享错误helper的实现。
- 所有decoder输出先反归一化，再算root速度/位置/foot contact；不使用normalized channel当米制量。
- 加足够decoder history和edit后尾；同时报告warmup与测量区间。旧64帧首帧冷启动的结果单独标记。
- 对躺地、盘腿、正常走三类执行global/local通道替换反事实，决定是否真是root流主导。
- 重评小样本source→reconstruction上限和base→edited变化，明确各自比较对象。

**验收：** raw/mirror几何oracle通过；旧ref_pos路径不再被主评估使用；有physical_metric_version；能区分坐标重建错误、tokenizer重建误差、算子新增误差。

### N02：建立内容/风格可识别的开发benchmark

**先做数据与协议，不训练。** 文件：`pairs.py/windows.py/eval_protocol.py`、新内容映射配置和少量测试。

实施：

1. 保存raw_action与canonical_content，首个映射仅合并Locomotion Neutral/Styles；未知标签不猜，映射版本写入checkpoint。
2. 输出train/val/test的content×style×actor/take覆盖表、majority基线、仅用action预测style的基线。
3. 选择同时存在三个style、且动作/速度范围可比较的case；不足则明确限制为两个injured style，不借缺失格子称三风格benchmark。
4. 建两个子任务：同style跨content参考一致性；固定source content、换target style的真实editing。NLL目标重建与source-to-other-style编辑分开。
5. 保留现有213行作为legacy开发集；新balanced dev集按style/action/take抽样，锁定manifest与独立eval seed。最终test留到配方确定后只评一次。
6. 同一take、镜像、相邻crop合为统计group。reference候选与query以group隔离，禁止leave-one-row-out留下同clip复制体。

**验收：** protocol具有可用真实正负例；所有排除有原因；content条件不再显式包含neutral/styles开关；能够说明每一项泛化轴实际测试了什么。

### N03：先用已有模型补真实编辑证据

**无训练。** 在legacy内容映射下评旧checkpoint；新内容map不得塞给旧模型。N02的新benchmark可以保留raw行动ID供这一步，但结果注明legacy conditioning。

实施：

- 先固定12–24个case，覆盖三个style、context/whole-body、left-arm/leg、radius0/1和selected-span；无法找到的格子写unavailable，不造reference。
- source/base/constant/style-ID各生成steps=1与4，λ=0/0.5/1/1.5，共享每case/draw/step的CRN，至少2个draw。
- 输出N01修正后的off-target FK、边界jerk、接触翻转/脚滑、root轨迹、速度/幅度、token变化和support内变化。
- 生成固定相机并排动画及失败case，source/ref/输出身份明确。全body变化和局部变化分开验收，不能仅看静帧。
- 用原始motion而非operator token-NLL建立小型独立style评价基线：先做固定运动学统计+线性分类器，训练仅train、按take/actor分割，报告balanced accuracy和混淆。若real motion本身也无法区分style，先修标签/任务，不训练更复杂evaluator掩盖。
- classifier只是辅助，style是否肉眼可辨、动作内容是否保留需要与动画共同判断；ref encoder不得同时充当最终评估器。

**验收：** 可以给出“有无可见style响应、是否以明显质量代价换来”的证据。若style-ID仍只有微小NLL收益而无可见响应，N04先走缩小任务/数据路线，不能直接扩矩阵。

### N04：最小复现实验，区分模型问题与标签捷径

**需新训练预算。** 第一轮建议seed3407/3408，固定同一个dev/test manifest与eval seed；不能因训练seed不同重采验证集。

顺序与预算建议：

| 实验 | 建议首轮预算 | 目的 |
|---|---:|---|
| canonical-content base | 20步profile→2,000步pilot，seed3407 | 去掉显式content/style标签混淆；保持tokenizer/tokenstore不变 |
| style-ID logit + constant | 每臂1,000步，seed3407 | 在新content map、同一base下验证style净效应 |
| 第二训练seed | 同两臂各1,000步，seed3408；固定同一个base | 分离operator训练随机性；不声称覆盖base多seed |

每对两臂使用实际相同train pair与mask/strength序列；不要仅设置同一个torch seed，因为不同encoder初始化会消耗不同RNG。保存序列digest或固定pair manifest。

保持D、depth、loss、mask、steps不变，只更换content map；map变更是新实验，不能给旧transport重排condition embedding后续用。

判据：paired style净收益、correct-vs-wrong响应、N03物理/动画效果至少方向一致；以take/actor group bootstrap给条件于该checkpoint的区间，第二seed单独报告。区间不包含模型训练随机性的全部不确定性。

若两seed中收益不稳定或contentmap修正后消失：不扩到三seed主实验，回到数据定义/任务收缩。若一致且可见，再批准扩大到3k步或第三seed；一次只变预算或seed，不同时改架构和loss。

### N05：reference失效机制诊断与两阶段修复

N00–N03可先完成；reference训练必须等诊断明确、数据可辨识后再批准。

**N05a 无训练诊断：**

1. 同一真实reference池测fresh initialization与trained checkpoint的embedding、temporal、graph、pool、descriptor逐层方差/范数、时间打乱与reference交换响应。
2. 用单batch计算梯度（不step、不写checkpoint），检查encoder各层梯度是否到达、是否finite；检查frozen/eval状态与optimizer参数是否确实包含encoder。
3. 比较token输入响应与sinusoidal/stream embedding量级；将position encoding置零作为诊断，不直接改主模型结论。
4. 排查sqrt(variance)在零附近、pooling和normalization的数值行为；有明确反例再改代码。
5. 判定是“初始就不敏感”“端到端学成常量”“数据无法分风格”还是“梯度/实现错误”。

**N05b 受控训练（首轮建议≤1,000步pretrain + ≤1,000步adapter）：**

- 首先用最简单的seen-style分类辅助任务训练reference encoder，均衡三类，按不同content/take做验证；保持global descriptor，不添加region-specific编码器。
- 输出层分类准确并不够，同时要求same-style跨content descriptor稳定、不同style有margin、真实reference交换响应明显高于数值噪声。
- 若该诊断任务可学：冻结encoder，重新训练一个logit operator，与constant和style-ID比较；先不端到端同时改encoder/operator。
- 冻结分支恢复reference响应后，再单独ablate低LR encoder fine-tuning和一个辅助CE项；不要首轮同时加入contrastive、adversarial、cycle、time-warp。
- 如果监督三类encoder也无法泛化，优先修数据/输入，不加大depth/width。

此方案只用于seen-style工程与因果诊断；固定三类分类器不是unseen-style解决方案。reference最终若只会分类三个ID，要明确收缩任务，不把reference接口存在当作zero-shot style。

### N06：决定是否扩大base或重启CTMC

只在前述验证通过后做。当前2,000步base可以继续作为固定诊断上游；full generation质量差时再单独考虑更长预算/更丰富trajectory或phase条件。

- 加预算前分别看full-mask和context验证曲线，并与action-conditioned unigram比较，不只看五类平均。
- 先保留architecture与contentmap，仅增加训练预算；如果有趋势再考虑更密的mask schedule或明确时间条件，逐一ablate。
- CTMC只有在matched style fidelity下出现接触/边界/稳定强度控制的候选收益时再启动shuffled/full-kernel。不以旧ordinal fraction作为继续/停止的唯一依据。
- tokenizer重训仅在N01真实反事实证明它是主瓶颈后考虑；一旦tokenizer权重改变，tokenstore/transport/operator均需重新绑定和重建相应训练，不能混用旧字母索引语义。

## 8. 优先级、成本与停止规则

推荐顺序：**N00 → N01/N02 → N03 → N04；N05a可在不训练的诊断阶段完成，N05b需数据与冻结分支条件满足；N06最后。** 单执行agent仍按顺序串行，避免同时改公共接口。

本轮已有checkpoint重评比重新训练更先。最重要的投资是可信的editing benchmark，而非继续用更低NLL追逐“风格成功”。训练耗时虽然本轮只有分钟级，仍需报wall cap、步骤、设备和输出目录；不把历史授权当新预算。

阶段状态统一：`implementation_complete`、`measurement_valid`、`likelihood_signal`、`motion_quality`、`style_fidelity`、`generalization`分别标记；未测就写not_evaluated，不让“7/7通过”覆盖未实现的要求。

发现非finite、hash/condition/mask不一致、未知label、协议行集变化时停止相关run；研究指标无优势时不自动延长预算。保留失败case与旧报告，以新版本附上更正说明。

## 9. 可复制给执行agent的启动指令

```text
请执行 docs/MTS_FSQ_First_Round_Results_and_Next_Plan_zh.md 的 N00–N03 和 N05a。
保留现有 checkpoint、报告、dirty工作区；不重训，不覆盖旧 verification JSON。
先读本次 corrected_rescore.json，修复 E04–E06 漏传 action 的评分路径，统一三种NLL聚合。
把临时verification收敛到正式只读评估入口，并把缺action、聚合混用、ordinal边界重复计数、
NaN与不完整验收写成回归反例。

接着验证bind/mirror/FK真值、构建content×style覆盖表和真实editing benchmark，
用已有四臂生成固定并排动画、物理与style响应证据。未知content、不可构造的对照格子
明确报告，不借label或复制clip造证据。

reference只做初始化/训练后逐层方差与单batch无更新梯度诊断；先判机制，再提训练修改。
所有长评估设固定样本上限，所有新的优化步骤需另报预算。

最终交付 screening_decision_v2.json、完整score/eval manifest、回归测试结果、
失败案例与N04/N05b精确训练申请。不要把NLL改善写成已验证的可见风格迁移。
```

## 10. 本次交付边界

本次完成结果分析与CPU只读checkpoint重评分，并新增本文件和审计脚本/JSON。没有更改原始训练结果、既有决策文件或训练实现，没有启动新优化步骤。四臂主排序经正确条件复核后保留，但其研究结论范围应按本报告收窄。

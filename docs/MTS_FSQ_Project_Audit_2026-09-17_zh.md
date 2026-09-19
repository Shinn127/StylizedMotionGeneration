# MTS-FSQ 项目审核与下一阶段实施计划

审计日期：2026-09-17；代码基线：`e9aaceb`。本次只新增审计材料，没有修改训练实现或启动训练。

## 1. 结论与决策

**建议继续研究，但先暂停当前 P4/P5 批量训练与论文出图，用一轮“正确性修复 → 最小科研闭环”替换直接恢复 26-cell 矩阵的路线。**

项目有值得保留的资产：NEF ownership、canonical token contract、packed 数据链、actor-holdout 数据产物、物理目标等预算对照、可复用的运动学和可视化基础。但核心 style operator 的训练目标、生成入口、表征输入和评估语义存在实质错误。当前进度不能概括为“方法已验证，只差补齐实验”。

| 问题 | 审核判断 |
|---|---|
| 工程推进是否真实？ | 是。全量数据、tokenizer、token store、transport/operator checkpoints 均有本地产物；不是只有文档和接口。 |
| 是否达到 plan 的科研验收？ | 未达到。Phase 0 部分支持，Phase 1 标签/有效配对需修订，Phase 2 只有训练指标，Phase 3 未正确完成，Phase 4/5 不能验收。 |
| 结果是否符合预期？ | 表征局部性和 physical loss 的方向部分符合；风格迁移、跨内容复用、CTMC 优势没有可靠证据。 |
| 创新性是否达到 SIGGRAPH 预期？ | 当前实现与证据不足以支持。候选贡献仍有研究价值，不能据此保证发表或声称新颖性已成立。 |
| 是否严重“屎山”？ | 核心表征/数据模块尚有边界；MTS 实验编排和评估层已出现严重的语义技术债。应定点修复，不宜全仓推倒重写。 |

审阅范围：plan 与阶段文档；NEF/layout/representation；MTS 全部核心模块及四个 train/eval/generate 入口；v3/v4 数据接入、split/pair/checkpoint 路径；本地 JSON/CSV 和六个 operator checkpoint 元数据。renderer、资产转换和历史 representation 仅做架构定位，未逐行证明其正确性；未观看完整生成视频、复跑全量数据校验或独立重做训练。

## 2. 可以保留的结果，以及它们实际说明什么

### 2.1 NEF 表征和几何：有正面信号，结论要收窄

`outputs/nef_eval_1h/nef_geometry/probe_geometry.json`：128 个 64 帧窗口，40/40 coordinate 的 feature 相邻/远距比值小于 1，中位数 **0.5418**，平均方向一致性 **0.9446**。

这支持继续尝试 ordinal prior，但不证明“FSQ level 具有语义轴”，更不证明 CTMC 优于其他算子。探针里的 far offset 是逐帧随机抽取，相邻 offset 是整段统一 ±1；二者的时间频谱和有效帧集合不同，尤其不能把较大的 far jerk 全归因于几何距离。后续应在相同有效位置、相同时间 support 上做单脉冲/固定 offset 对照，按 step size 分层报告 feature、FK 和速度。

时间探针在 64 帧窗口的第 32 帧编辑，观测截止于第 63 帧，**并未实际观测完后续 33 帧的整个理论影响范围**。RF 的结构计算仍可成立；完整实测需加长窗口，并分别报告直接 feature spill 和 root 积分后的 world-space spill。global/root 编辑可造成后续轨迹偏移，不能把所有 world-space 影响都限为 decoder RF。

局部性已有明确证据，但 part-FSQ 同样做到区域外 FK 变化 0。因此 exact locality 本身不是 NEF 独有贡献。NEF 10% token support 与 part 的 12.5–17.5% support，也不等于“实现了同等强度的风格变化”；需要在相同目标区域运动幅度/风格得分下比较泄漏和物理代价。

### 2.2 Physical objective：观察到有效但有代价的折中

以下数字直接来自 `outputs/stage3_eval/physics_p0/physics.csv`，同一组 128 个 test 窗口：

| 模型 | FK mean（cm） | root drift（cm） | foot slide（m/s） | contact recall |
|---|---:|---:|---:|---:|
| v1 | 8.216 | 0.928 | 0.3298 | 0.3794 |
| 等追加步数 recon+delta control | 8.154 | 0.929 | 0.3284 | 0.3897 |
| v1.1 physical | 8.710 | 2.053 | 0.2765 | 0.5048 |

当前单 seed 对照支持“物理目标改善接触/脚滑，同时牺牲部分 FK/root”，不宜写成全面提升或已建立普遍因果结论。v1.1 相对等预算 control 的脚滑约下降 15.8%，FK 约增加 6.8%，root drift 约为 2.2 倍。

推荐保留两条 checkpoint；在固定验证集上寻找 recon/root 与 contact 的 Pareto 折中。先检查各损失的量纲和梯度贡献，再用少量预算试权重，不应只继续加大 foot loss。物理评估同时报告接触 precision/recall、移动速度、步幅和动态范围，防止“站着不动”取得低脚滑。

### 2.3 数据与训练产物：工程完成不等于阶段验收

- actor-holdout 审计产物报告 train 468 / test 52 演员、交集 0；142,220 个 clip。此次核对的是产物，没有重扫 60 GB feature 字节。
- holdout 专用 NEF checkpoint 的 `val_full.loss=0.153487`、`recon=0.091238`、`delta=0.020750`，与阶段文档相符。
- transport 20,000 step、best val CE 1.42738 是实际记录。它还缺 per-mask 固定验证、完整生成质量和现有 generator 的公平对照，不能据此验收 Phase 2。
- 六个 operator `best.pt` 实际均为 reference encoder，dim 256、best step 6,000，没有 shuffled `level_order`。旧“sandbox/shuffled”目录名不能代表实验身份。
- birth-death 训练 summary 还记录了 **mass_error≈0.679**、平均 uniformization terms=2。当前数学测试通过不追溯保证历史训练数值有效；需绑定代码版本并重验旧 checkpoint 的行为。该历史异常不能被归一化后的合法概率掩盖。

**产物处置：** feature/tokenizer/token store 保留并补充绑定审计；旧 transport 保留作缺陷基线；六个 operator 全部降级为 legacy/debug，主实验重新训练。不能只补训阶段文档列出的四个。

## 3. 阻断研究结论的实现问题

下面区分“最小反例实测”和“代码路径确认”。优先级 P0 表示继续实验前必须修复，P1 表示正式对照前必须完成。

### P0-1 概率被当作 logits，优化目标不是 NLL

位置：`mts_operator/model.py:227` 调用 `masked_cross_entropy(result.probabilities, ...)`；`contract.py:261` 无条件 `log_softmax`。`metrics.py` 对 base/styled probability 也有相同错误。

实测：正确类别概率 0.99 时，真正 NLL 是 **0.0100503**，当前函数输出 **1.38036**。九分类下，即便概率 one-hot，该错误目标仍有约 1.372 的正下界。它仍可能提供一些学习信号，但梯度、校准和模型排序不再等价于计划中的概率似然。

实施：保留 logits CE 给 transport；增加明确的 `masked_nll_from_probs` 或 log-prob NLL 给 operator，禁止用数值范围猜输入类型；统一返回有效 token 数。添加与手算概率、PyTorch NLL 的值/梯度对照，覆盖 padding 和空监督。所有旧 operator 需重训。

### P0-2 val NLL 被 mask 比例压低，best checkpoint 选择失真

位置：`training.py:372–380`：先把已归一化 loss 乘 `supervision_fraction`，最后除 batch 数，而不是监督 token 数总和。`train_mts_operator.py:486` 每次验证只有一个随机 batch，mask 类型也会变化。

实测：真实 batch loss=2、监督比例 0.05 时，记录的 val NLL=**0.1**。这解释了“train≈2、val≈0.09”为什么不能按常规解释为泛化很好。它不能直接与 transport CE 比较。

实施：累加 NLL sum / valid supervised count；固定验证 pairs、window、mask 和 seed；按 mask kind 分组。增加“拆成多个 batch / 添加 padding 后指标不变”的测试。修复后重新选 best；旧 best 无可靠比较意义。

### P0-3 生成入口关闭了风格注入

位置：`generate_mts_operator.py:199`、`evaluate_mts_operator.py:329` 创建 `OperatorBatch` 时没有 `visible_mask`；`model.py:57` 默认全部 visible，故 supervision/edit mask 全 false。

实测：该默认路径 operator active support 总和为 **0**。生成函数仍会从 base 分布抽样，因此“motion 发生变化”不代表 reference 生效。physics sweep 的 λ=0/1 也没有在测真实风格变化。

实施：把观察 mask 与编辑 support 明确定义；局部 inpainting 最小实现用 `visible=~support`，采样写回仅限 `support & ~visible & valid`。但隐藏整片目标后，必须另供真实 content 条件才能主张内容保持。增加真实 CLI smoke：同一输入与随机数，换 reference/λ 能改变 support 内分布，区域外及 visible anchor 不变。

还需区分：**λ=0 回到 base distribution 不等于 λ=0 复原 source tokens**。若产品承诺“strength 0 原动作”，需要 source-anchored 分布或显式 identity 分支，并在训练/评估中对应定义；不要靠锁区域外来替代这个保证。

### P0-4 stream 内 coordinate 身份丢失，输出 head 又限制了表达能力

位置：`embeddings.py:65–76`，先 `E(level)+E(coordinate)` 再求和/均值。所有 coordinate visible 时，在一个 stream 内交换两个 level 不改变这个和（浮点误差除外）。被 mask 的 coordinate 甚至连身份 embedding 都被替换掉。

实测：交换同一 stream 的两个 coordinate 后 embedding 差约 **1.86e-9**，transport logits 差约 **1.19e-7**。这是架构信息损失，不是训练不够。

`transport.py:272` 还把同一 stream 的一组 9-class logits 广播到所有 coordinate，仅加静态 coordinate bias；因此 coordinate 间的 logits 差不随上下文变化。reference encoder 复用同一 embedding，也受影响。

实施：13-stream ownership 保留；stream 内使用有固定 coordinate 顺序的 level embedding 拼接后投影，或 coordinate×level embedding 非线性聚合；mask token 保留 coordinate identity。输出用每 stream 的 `D→K_stream×9` head（按 family 共享也可）。增加 coordinate-swap 可区分测试、不同 coordinate 随 context 独立变化测试。架构版本升级，transport/reference encoder 重训；tokenizer 与 token store 不必因此重建。

### P0-5 默认双向 temporal Transformer 没有时间位置编码

位置：`transport.py:171`、`style_encoder.py:74` 及各自 forward。没有 absolute/relative position 或 temporal convolution；graph 在每帧作用，无法补回时序信息。

实测：无逐帧 content 条件的全 mask 输入，所有帧的 logits **完全一致**；默认双向 style encoder 对随机时间重排的 descriptor 差仅 **1.39e-6**。这意味着当前网络不能可靠地区分相同帧集合的顺序/节奏；一轮 full generation 接近逐位置边际采样。

实施：先采用明确的 sinusoidal/relative temporal position，再以短时卷积作可选对照，避免一次引入多种复杂结构。为 transport 和 reference encoder 都加入“时间打乱应改变时序描述”的测试。full-mask completion 要检查速度谱、contact timing、序列多样性，不能只看 token accuracy。

### P0-6 评估器的对照组与指标命名不成立

位置：`scripts/evaluate_mts_operator.py:275,279,365` 与 `mts_operator/metrics.py`。

- wrong style 是 `reference.roll(1, dims=0)`：同一片段平移一帧不是不同风格；在当前无位置编码+mean/std pooling 的模型里尤其接近不变。
- random reference 是独立均匀 token，属于明显 OOD；可以作为破坏性 sanity check，不能替代真实随机 motion reference。
- content id 是 `index % 4`，不是动作标签；当前无 conditioner 时会被忽略，启用 conditioner 后变成错误条件。
- style retrieval 只把原配 reference 的索引算正确；batch 中其他同 style reference 被误判。先对 batch accuracy `round` 再平均还会把真实命中率压成 0/1。
- style-ID checkpoint 在评估/生成中没有提供 `style_ids`，会报错；应独立定义其评估协议，不能给它 reference retrieval。
- `boundary_jerk_max` 实际是归一化 feature 的一阶差变化，且扫描整个区间；不是物理 jerk（三阶位置差 / dt³）或专门的边界度量。
- physics summary 把 strength 0/1 混合聚合，区域标签又在局部 hard mask 生效前加入部分全身指标；指标的测量条件不一致。

实施：构造带真实 style/content/actor/group 标签的固定 eval manifest；负例为不同 style、尽量匹配 content/速度/演员条件的真实片段；retrieval 用多正例与按 style 均衡候选；累计 hits/count。先建完整 batch 条件再调用全部指标。physics 按 λ/区域/协议分组，保存逐 clip 明细、单位和 metric version。加入独立 style/content evaluator 和盲测动画，模型自身 NLL 只作辅助诊断。

### P0-7 数据语义错误和有效训练分布未被审计

1. **100STYLE 的 BR/BW/FR/FW/SR/SW/ID/TR 是动作类型，不是 performer。** `pairs.py:601` 附近的 `_PERFORMER_CODE`、`split_style_name`、`labels_from_name` 把后缀当演员，并用 clip identity 替代语义 content。[官方数据页](https://www.ianxmason.com/100style/) 明确给出命名，并说明所有风格由同一演员采集。旧 100STYLE performer 分析不能使用。
2. SEED 的 8 类中 5 类只有一个 content。当前 same-style pairing 强制 different-content，故这 5 类无法产生合法 pair；实际能参与的 style 至多是 neutral、injured leg、injured torso 三类，还会受 split/window 有效性进一步限制。不能仅报告“8 个 style”。
3. sampler 默认按 data split 取记录；summary 却报告另一套 `style_split.train_styles/test_unseen_styles` 长度。旧日志的 train=4/unseen=2 不代表实际 operator 训练/测试词表。
4. `held_out_styles` 只在 reference 的 `_accepts` 检查，target 未过滤；**same_content/different_style 模式**可训练 held-out target。已用最小 fixture 复现 `('b','a')`，其中 b 被列为 held-out。默认 same_style 不会因这一条发生同样泄漏，不能泛化指控当前 SEED 主配置已泄漏。
5. `build_pair_audit` 在有 overlapping-unseen actors 时于 `performer_axis` 赋值前读取它；最小 fixture 复现 `UnboundLocalError`。

实施：dataset-specific 标签表优先，不靠大写后缀猜身份；把 style、action、actor、take、mirror family 分成独立字段。所有采样模式统一过滤 target/ref；记录实际接受/拒绝的 pair 数、每 style/content/actor 曝光次数与窗口。先构造 style×content 覆盖矩阵再决定数据：SEED 用于 broad motion 和 unseen performer；100STYLE 用于 style/content 组合，明确不能承担多演员泛化。

### P1-1 工件绑定和 shuffled 对照无法可信复现

- `contract.py:284` 明确排除 tokenizer 权重，只校验 alphabet/layout。相同 40×9 布局不保证两个 independently trained tokenizer 的 token 含义相同。
- packed token manifest 已有 `checkpoint_sha256` 和 normalization hash，但 MTS 入口没有把这些与传入 tokenizer 强制交叉校验。
- eval/generate 对 operator 直接 `torch.load/load_state_dict`，没有复用 operator metadata 校验。
- train 会保存 `level_order`，但 eval/generate 的构造键白名单不包含它，且它不是 state_dict buffer；真正 shuffled 模型即便训好，加载后也会恢复默认邻接。
- MTS payload 没有完整 resolved config、数据 split hash、实际 style-ID map、代码版本；单纯上游文件路径不能作为身份。

实施：分开 `alphabet_hash` 与 `tokenizer_state_hash`；复用已有 SHA256/schema/norm/split 元数据，建一份轻量 run manifest。统一 `load_operator_bundle`，从 checkpoint 重建所有结构参数并检查数据；style-ID map 和 level_order 必须序列化且 round-trip 一致。更换同架构不同权重 tokenizer、替换 normalization 或丢失映射必须 fail-fast。

### P1-2 coupled sampling 不稳定，content regularizer 无有效梯度

- `sampling.py:72` 用 `id(device)` 做 cache key；`tensor.device` 的 Python 对象身份不稳定。实测同一均匀分布两次 `crn.sample` 产生不同 token，缓存变成两项。`paired_comparison` 在内部显式共用同一 uniforms 的路径不受这一点同等影响，但 `support_locality` 两次 `generate_edit` 的配对会受影响。
- `model.py:235` 的 content loss 来自冻结 transport 的 base probability；在默认 freeze 模式下对 operator 没有梯度。测试只验证数值增加，没有验证约束生效。

实施：使用 `(shape, device.type, device.index, sample_id)` 作为稳定 key，或显式传共享 uniforms；不同 sample 独立、相同 sample 跨 λ/方法共享。content term 改为作用于 styled 输出的明确约束；若暂时没有可信可微 content 表征，删除“regularizer”承诺、仅保留诊断，不用常数项伪装。

## 4. 创新性定位与研究改进方案

### 4.1 需要更新的相关工作边界

| 已有工作 | 对本项目的约束 |
|---|---|
| [Motion Puzzle](https://arxiv.org/abs/2202.05274) | 已研究 body-part motion style transfer 和不同部位风格组合；“局部风格迁移”本身不能作为新贡献。 |
| [MoST，CVPR 2024](https://boeun-kim.github.io/page-MoST/) | 已针对不同 action contents 之间的风格迁移与解耦；跨内容任务名称本身不够。 |
| [Decoupling Contact，2024](https://arxiv.org/abs/2409.05387) | 已显式分离/控制 trajectory、contact timing、style；“考虑接触”需要更明确的区别。 |
| [STyMo，SIGGRAPH 2026](https://joseluisponton.com/stymo-project-page/) | 已包含少样本快速学习、静态/动态风格和运行时区域/强度控制，需纳入当前文献与实际 baseline 选择。 |
| [FSQ](https://arxiv.org/abs/2309.15505)、[Discrete Flow Matching](https://arxiv.org/abs/2407.15595) | 离散字母表、离散概率路径是已有基础；本项目的 frozen-generator style operator 与这些训练范式并不相同，但不能仅以使用 FSQ/CTMC 主张创新。 |

以上是针对性文献核查，不是穷尽式新颖性检索，也没有复现外部论文性能。

### 4.2 推荐的主命题

把主线限定为：**在冻结、具有空间所有权的离散动作表征上，用同一全局 reference descriptor 实现跨内容的局部概率编辑，并在匹配风格效果时改善内容保持与物理质量的折中。**

三个需要分别建立的证据：

1. **descriptor 真在编码风格。** 固定 source/content，换正确/错误风格 reference；同 style 跨 content reference 应产生接近的风格，而不是复制动作或演员。
2. **结构化 support 带来实质收益。** 与 part/flat 的公平容量对照，匹配 in-target 风格效果，比较 off-target FK、脚滑、边界和内容。
3. **ordinal kernel 有额外收益。** CTMC 对比 logit、FiLM/AdaLN、full kernel、shuffled adjacency；同一 descriptor/transport/参数预算。至少在样本效率、未见组合、稳定强度控制之一出现稳健收益，否则将 CTMC 降为实现选项。

必须补充的 baseline 是 **不看 reference 的同容量 adapter**。否则 operator 可能只是在补偿弱 transport，NLL 下降被误读为 style learning。再加 constant-style、真实错误 reference、正确 reference；用这四组先判断是否值得扩大训练。

### 4.3 模型调整的最短路径

- 首先修复 coordinate-aware embedding/head 和 temporal position；不改 tokenizer 的 stream ownership。
- 给 `content_condition` 接真实 action 与 trajectory/phase 等已有可获得条件。当前 full mask + 无 conditioner 的模型没有目标内容信息，逻辑上不可能保证指定 content；“不输入 style label”也不等于 learned content/style disentanglement。
- 将随机高 mask/全 mask 与真实局部编辑混合；记录每种 mask 的实际遮盖率。当前 config 有 30% full generation，但“另 30% 高遮盖率”没有清晰验收。
- reference encoder 先用同 style / 不同 content 的一致性或对比目标作为一个单独 ablation；先平衡 style 与 content，再考虑更复杂解耦。避免 encoder 学成 neutral majority 或演员分类器。
- 对需要改变节奏、contact timing 的风格明确给出能力边界。只做同帧 level transport 并不直接提供 time warp；若时序位置修复后仍失败，再单独研究 phase/time-warp 分支，不预先堆叠新模块。

可以探索的高价值扩展是 **在预算约束下的几何感知概率编辑**：利用 decoder 单 coordinate 扰动代价，对 CTMC rate 加非负、identity-preserving 的位移/速度代价约束，比较相同 style fidelity 时的 FK/contact 代价。它是待验证假设，不应现在宣称新方法优势；必须先有正确的基础算子与 shuffled 对照。

对 disjoint multi-style 也要严格定义：当前把多个 region 合成一个 mask、只提供一个 reference，不等于不同区域使用不同 style。可在一次冻结 base/context 上构造各区域 `Q_s`，不相交时按 support 组装；重叠时明确加和/优先级规则。固定 Q 的半群性质不能自动推广到每轮重算 context 的完整生成算法。

## 5. 代码债务与具体治理

### 5.1 不应推倒的部分

NEF 的 layout 唯一所有权、stream 独立 temporal folding、统一 representation adapter、packed store 的逻辑 clip/物理 shard 分离、normalization 独立版本化有清楚用途。v3/v4 共存本身不是问题；不要为了“统一”重写所有旧实验或合并所有 representation。

主要债务集中在：CLI 中重复模型重建；“接受了配置键”却未保证执行语义；shape 相同但 logits/probs 混用；训练、推理、metrics 各自推导 mask；阶段文档比真实验收乐观。其危险性高于单文件长度。

### 5.2 最小重构包

| 改动 | 涉及文件 | 明确验收 |
|---|---|---|
| 分离 logits CE / probability NLL，返回 count | `contract.py/model.py/metrics.py/training.py` | 手算值、梯度、batch partition、padding 一致 |
| 统一 edit batch / sampling 语义 | `model.py/sampling.py`、eval/generate | observed token 不变；真实 reference 生效；CRN 成对一致 |
| 统一 checkpoint 重建与身份验证 | `checkpoint.py`、四个入口 | shuffled/style-ID round-trip，错 tokenizer/norm/split 拒绝 |
| 把 pair/window batch 构造移出训练脚本复用 | `windows.py/pairs.py`、train/eval | 无 skipped-pair style-ID 错位；eval 固定 pair manifest |
| 保存 resolved config 和 run manifest | 两个 train 脚本，少量公共函数 | CLI override 实际生效；未实现字段报错或明确废弃 |
| 精简状态文档 | 三份阶段文档+README | 每个完成标记链接可核对 artifact；历史 snapshot 标明失效范围 |

不建通用 experiment framework，不引入新 registry，不拆几十个微模块。大型 `runner.py`、`preprocess.py` 暂不做纯清理式重构；只有触及具体职责时再抽取函数。

### 5.3 还应顺手修复的可靠性问题

- transport `--checkpoint` 只恢复权重，未恢复 optimizer/global_step/RNG，却打印 resumed；区分 resume 与 warm-start。
- `val_every_steps` 被配置接受但 fit 未执行；AMP 只用 GradScaler 而没有 autocast。当前 fp32 结果不因此无效，但配置承诺应与行为一致。
- transport accuracy 分母未排除 invalid frames；短 clip/padding 生效时会偏低。
- `PairedBatchSource` 跳过无窗口 pair 后，style IDs 仍从原始 pairs 生成，会长度或顺序不匹配。
- holdout 配置仍写 `required_data_schema_version: 3`，实际 store v4；目前入口未强制检查。config 的 `reference_frames`、sampling/loader 若未实现，应减少或明确校验，不保留误导性开关。
- stage status 依赖 `/tmp/run_sandbox.sh` 等临时脚本；移为仓库内的小型 manifest+runner，保存完整命令与退出码。不要把“目录存在/JSON 可写”当作实验成功。
- `docs/seed_actor_holdout.md` 的旧续训建议不能产生干净 unseen actor：从已见过这些演员的旧 tokenizer fine-tune 到新 split，不会消除训练曝光。当前独立 holdout tokenizer 可保留，但该示例需撤回。

性能优化排在科学正确性之后：先测 loader wait、CPU/GPU 同步、CTMC 时间和峰值显存。当前热路径有多处 `bool/float(tensor)` 同步、逐 clip 在线编码、逐候选 retrieval 前向。优先复用 token store、批量 reference scoring、减少日志同步；CTMC 可用三对角递推替代 dense 9×9 运算，但需保留 matrix_exp 值/梯度 oracle 后才优化。

## 6. 分阶段实施计划与停止条件

时间是单人开发/分析工作量估计，不是已批准的 GPU 预算；长训练应另定 step、seed 和预算。

| 阶段 | 预计工作量 | 实施内容 | 必须交付/通过的门槛 |
|---|---|---|---|
| A：可信测量 | 2–4 工作日 | 修 NLL、聚合、generation mask、CRN、eval 负例/检索/单位、shuffled restore、标签解析和审计异常 | 最小数学反例转回归测试；真实 reference/style-ID CLI 都跑通；固定 eval manifest；旧结果失效清单 |
| B：有效内容模型 | 3–5 工作日 | coordinate-aware embedding/head、temporal position、真实 content condition、checkpoint binding | 1–8 clip 过拟合；交换 coordinate/时间顺序能被识别；full/infill/local 逐项可视化；无 style 基线有可接受 motion |
| C：最小科研否证 | 3–5 工作日+核定训练 | 先 3 个有效 style、≥2 个 content；style-ID logit vs CTMC；加入不看 style 的 adapter，再上 reference | balanced style 指标胜过无 reference；同 content 换 style 可见且可测；内容/物理无不可接受退化 |
| D：数据与公平对照 | 1–2 周+核定训练 | 修正 100STYLE 标签；freeze style×content split；NEF/part 主对照，补 FiLM/full/shuffled；flat 作辅助 | 每个测试组合在 operator train 中严格缺席；相同解码/采样/有效训练 token 预算；记录全参数与耗时 |
| E：主结果 | 1–2 周+核定训练/用户研究 | 对筛出的最终设置跑 ≥3 seeds；长序列、multi-style、接触失败集；独立 evaluator 与盲测 | 按 take/actor 聚类 bootstrap 置信区间；匹配 style fidelity 的 Pareto 图；带失败案例的动画与复现包 |

**A/B 完成前不重启 26-cell 矩阵。** 恢复四个旧 sandbox 作业只能修复目录身份，不能修复 objective 和 architecture。

建议提前登记而非看结果后选择的判断规则：

1. 正确 reference 相比真实错误 reference/无 reference，在 balanced style metric 上提升的配对 95% CI 应排除 0；同时满足预先约定的 content/physics 容差。若两轮小试仍无信号，先改数据/条件/encoder，停止扩大 CTMC 实验。
2. CTMC 若不优于同预算 logit/FiLM，或 shuffled adjacency 表现不劣，则撤回 ordinal CTMC 主贡献；保留简单算子完成 NEF 局部编辑研究。
3. NEF 若在匹配编辑效果后不优于 part，不能继续把 ownership 的“零泄漏”当主贡献；转向 edge/coordination/contact 的可测优势，或收缩论文定位。
4. 若数据不具备同 style 跨 content 的覆盖，不应用更复杂正则补造证据；改数据或限定任务范围。

## 7. 额外容易遗漏的科研风险

- **严格局部 vs 物理可行性：** 固定 root/global contact 再大幅改腿，可能保住非目标特征却破坏接触。分开报告 strict support 和允许少量 root/contact adjustment 的 relaxed support；修正区域、幅度都必须透明，不能把后处理泄漏藏掉。
- **zero-shot 的上游曝光：** 明确区分 operator-unseen style 与 tokenizer/transport 也完全 unseen。plan 允许 broad motion 预训练，但论文必须披露；actor-unseen 则必须检查所有学习组件和 normalization 的训练集合。
- **统计单位：** 重叠窗口、原始/镜像、同 take 的多 clip 不能当独立样本。保存 clip/take/actor id，按组估计置信区间，避免用 8,192 帧伪装 8,192 次独立实验。
- **多种指标不能混为一个好坏：** frozen decoder 重建误差是可达质量上限之一；报告 source GT→reconstruction 与 reconstruction→edited 两层误差。style gain、content retention、diversity、contact 均需独立衡量。
- **长时与边界：** 64 帧结果不支撑任意长度/实时控制。验证 chunk overlap、decoder warmup、root 累积漂移、5–30 秒序列稳定性；双向 transport 不能直接宣称 causal realtime。
- **不确定区域和未见风格：** strength 单调改变 TV 不代表语义风格单调；检查饱和、反转、OOD reference，并保留 no-op/reject 的显式失败策略。
- **测试质量：** 当前 synthetic reference 测试存在 reference=target 的捷径；“loss 降了”“loss 加项变大”“shape 正确”不足。优先增加最小反例、输入反事实、真实入口串联，而非为了数量扩展测试。

## 8. 本次验证记录

在 `/home/shinn/miniforge3/envs/mcc/bin/python` 下运行：

```text
pytest -q tests/test_mts_operators.py tests/test_mts_metrics.py
          tests/test_mts_end_to_end.py tests/test_mts_pairs.py
          tests/test_mts_transport_training.py tests/test_nef_token_contract.py
58 passed, 1 warning, 96.58 s
```

另用 CPU 最小反例确认概率/logits 混用、val 聚合错误、生成 support=0、coordinate 交换不敏感、时间顺序不敏感、CRN 不稳定、held-out target 在非 same_style 模式泄漏、审计重叠演员分支报错。主要数值保存在 `outputs/audit_20260917/minimal_probes.json`；反例脚本同目录保存。

这些测试说明 **现有测试通过与科研实现正确之间仍有明显缺口**。本报告给出的是已经定位的缺陷、现有证据的可信范围，以及后续实施与否证门槛；没有将尚未执行的修复、训练或论文性能写成完成结果。

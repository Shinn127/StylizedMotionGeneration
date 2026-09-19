# MTS-FSQ 代码修订执行计划

版本：v1.0；日期：2026-09-17；审计基线：`e9aaceb`。

适用执行方式：zcode + deepseek-v4.1-flash，单 agent 按任务串行执行。本文不依赖某个模型的特殊能力或工具；重点是限制单任务范围、固定接口与测试标准、允许中断续做。

依据：[项目审核与实施计划](MTS_FSQ_Project_Audit_2026-09-17_zh.md)。本文把审核意见转换为代码任务，不重复论文背景。

**交付目标：先获得“目标函数正确、条件真实、采样和评估可信”的 MTS revision 2，再交付可开展小规模科研验证的入口。完成代码任务不等于风格迁移成功，也不等于论文创新已成立。**

## 0. 给执行 agent 的总指令

### 0.1 任务边界

本计划要求修改并验证代码；执行时保留用户已有改动。不要运行昂贵训练、全数据重建、全矩阵、删除/覆盖旧 checkpoint、清理 outputs。CPU 单元测试、小型 synthetic fixture 和本计划限定的短 smoke 可以执行；正式 GPU 训练另需用户明确预算。

- 保留 NEF 的 40×9 alphabet、13-stream 顺序、feature ownership、tokenizer encode/decode 与 representation id。
- 不重写 `runner.py`、`preprocess.py`，不改 renderer，不批量整理历史模型。
- 不新增通用 registry、plugin、配置框架、训练平台或深层类继承。
- 优先修改已有函数。允许新增的核心小模块只有下文明确提出的 `temporal.py` 和 `eval_protocol.py`；需要其他文件时先在进度记录写明必要性。
- 不删除失败测试，不用放宽容差、替换真实负例或减少验证样本来制造通过。
- 不把 zero-initialized operator 的“没有 reference response”误判为训练失败；行为测试用明确设置的非零权重或短训练 fixture。
- 若源码已变化，先按函数名重新定位。审计行号只是线索，不按旧行号机械编辑。
- 本文件中的新接口/CLI 是**待实现契约**，不是当前代码已支持的命令。

### 0.2 执行循环

每次只处理一个任务 ID。一个任务涉及超过约 6 个实现文件时，按其子步骤拆成连续工作单元，不同时重写整条链路。

1. 读本任务和依赖任务完成记录；检查 `git status --short`。
2. 只读任务列出的文件及直接依赖，定位最小改动。
3. 对 bug 先补有独立 oracle 的失败测试；记录失败原因，再改代码。
4. 运行定向测试；修本次引入的问题；检查 diff。
5. 更新进度记录：实际修改、命令/退出码、测试数量、工件和未完成项。
6. 定向验收通过后可继续下一个依赖已满足的任务，不需要逐步请示。需要昂贵训练或新研究决策时只停相关分支。

测试不能只调用两个共享同一实现的 wrapper 相互比较。值、梯度、采样、标签和 checkpoint 都要有可独立判断的反例。

### 0.3 进度文件

由 R00 创建 `docs/MTS_FSQ_Code_Revision_Progress_zh.md`，使用以下结构：

```text
代码起点：<git HEAD + 工作区已有改动>
当前任务：Rxx
任务状态：pending / in_progress / passed / blocked
实际修改：<文件与目的>
验证命令：<精确命令>
结果：<退出码、passed/skipped/failed、证据路径>
待办或限制：<未验证内容，不能写成完成>
下个任务：Ryy
```

只有实际跑过验证才标 `passed`。缺数据/设备时明确写 `blocked` 或部分完成；不要自动用 synthetic 成绩替代真实数据成绩。不要求自动 git commit；工作单元保持可独立审阅即可。

## 1. 全局接口与版本决策：各任务必须遵守

### 1.1 tensor、mask 与 loss

```text
tokens           int64 [B,T,40], 每个有效 token 为 0..8
visible_mask     bool  [B,T,40]，True 表示模型可观察
hard_mask        bool  [T,40] 或 [B,T,40]，True 表示允许编辑
valid_mask       bool  [B,T]，True 表示非 padding
effective_edit   hard_mask & ~visible_mask & valid_mask[...,None]
logits           float [B,T,40,9]，未归一化
probabilities    float [B,T,40,9]，非负且每行和为 1
```

`hard_mask=None` 表示整个 token 域，`valid_mask=None` 表示全帧有效；loss/generate 必须显式提供 `visible_mask`，不再把遗漏 mask 静默解释为全 visible。

空 edit 的 generation 返回 source tokens；空监督训练 batch 跳过 optimizer.step 并计数，验证的空监督子集报告 `null + count=0`，不能报告“loss=0 表现很好”。基础 loss helper 对空 mask 返回保持计算图的零，供调用方安全处理。

### 1.2 strength 与 baseline

- logit/CTMC：λ=0 返回 base probability；这**不保证** source tokens 原样返回。
- arbitrary kernel 的默认对照仍保留“支持区域内 λ=0 为 uniform kernel”的无 identity 设计；所有支持区域外位置必须返回 base probability。
- `identity_mix=True` 的 arbitrary kernel 只允许 `0≤λ≤1`；超范围报错，不做隐式 clamp，不允许负混合权重。
- 所有指标明确区分 `source`、`base_transport`、`styled`。比较 style effect 时以真正 `base_transport` 为基准，不能把 arbitrary kernel 的 λ=0 当作 base。
- 首版不额外加入“λ=0 必须 source identity”的特殊分支，避免训练/推理目标不一致。

### 1.3 revision 与旧工件

- 新 MTS checkpoint schema 升为 2，architecture revision 为 2，metrics schema 为 2；它们是不同字段。
- tokenizer/store 的版本不因 MTS 改动而变化，不修改旧 token 文件。
- 新 train/eval/generate 入口默认拒绝旧 MTS schema，错误说明“旧结果仅用于审计，需重训”。不在此轮维护两套 MTS 架构兼容实现。
- 旧工件保留原地，新增清单记录 invalidation reason；不能自动补齐 hash 后把旧实验升级为可信实验。
- 新输出一律进入 `outputs/mts_revision2/...`；测试使用 pytest `tmp_path`。

### 1.4 两层完成标准

**代码门槛 G-code：** R00–R12 的必要路径通过、三种算子及两种 encoder 均可正确保存/加载/评估，已知数学与数据反例全部被覆盖。

**实验准备门槛 G-ready：** 再完成 R13–R14，真实小样本数据入口通过、预算和有效 style/content 覆盖已输出。此时只能说“具备重新实验条件”。

## 2. 顺序与依赖

按表中顺序执行，避免接口在多个方向同时变动。任务内可拆小批，但不要跨过依赖。

| ID | 工作单元 | 前置 | 主要产出 |
|---|---|---|---|
| R00 | 基线与进度记录 | 无 | baseline、失效工件清单 |
| R01 | CE/NLL、有效计数与聚合 | R00 | 正确 loss/accuracy/validation |
| R02 | mask、算子 support 与数值边界 | R01 | mask 唯一语义、三族算子守约 |
| R03 | CRN、inverse CDF 与单步编辑 | R02 | 成对随机数、锁定正确 |
| R04 | dataset 标签、holdout 与配对 | R00 | 真实标签、有效 pair 审计 |
| R05 | 共用窗口 batch 与真实 action 条件 | R04 | 数据到 model 的条件链 |
| R06 | coordinate-aware embedding/head | R02 | revision 2 架构输入输出 |
| R07 | temporal position | R06 | 时间顺序可表达 |
| R08 | checkpoint bundle 与身份绑定 | R03,R05,R07 | 可验证 round-trip |
| R09 | 固定验证、配置执行与 warm-start | R01,R05,R08 | 稳定选 best、配置不静默失效 |
| R10 | 固定评估协议与 reference 检索 | R03,R04,R05,R08,R09 | eval manifest、多正例指标 |
| R11 | FK/contact/jerk 与分组输出 | R10 | 单位清楚、物理评估真实生效 |
| R12 | 一致的多步 masked generation | R03,R07,R08 | 不重采 observed token 的迭代生成 |
| R13 | 几何与时间影响探针修订 | R00 | 公平 geometry、完整尾部观测 |
| R14 | 真实入口 smoke、实验预检与文档 | R09–R13 | G-ready、后续训练申请输入 |

R04 和 R13 逻辑独立，但为执行 agent 降低认知负担，仍按此表串行做。未经明确要求不要派生其他 agent。

## 3. 逐任务实施说明

### R00：建立可恢复的起点

**文件：** 新建进度记录；其余只读。

1. 确认当前仓库、HEAD、已有 diff。本计划生成时审计报告尚未跟踪，不能删除。
2. Python 优先使用 `/home/shinn/miniforge3/envs/mcc/bin/python`；不存在时定位环境并记录，不安装/升级依赖来绕过失败。
3. 运行已知小测试集，保留完整输出：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_contract.py tests/test_mts_operators.py tests/test_mts_sampling.py \
  tests/test_mts_pairs.py tests/test_mts_metrics.py tests/test_mts_transport_training.py \
  tests/test_mts_end_to_end.py tests/test_nef_token_contract.py
```

4. 阅读 `outputs/audit_20260917/reproduce_probes.py` 与 JSON；如果 outputs 不存在，从审核文档重建必要反例，不能把 ignored 输出当唯一规范。
5. 记录旧六个 operator 失效原因；不移动、删除、覆写它们。临时产物不要提交为源码。

**验收：** 有可复查 baseline 与任务清单；失败若是既有问题明确分类。此任务不做模型修复。

### R01：把 logits CE 与 probability NLL 分开

**文件：** `mts_operator/contract.py`、`model.py`、`training.py`、`metrics.py`；测试沿用 `test_mts_contract.py`、`test_mts_transport_training.py`、`test_mts_end_to_end.py`。

**固定接口决策：**

```python
masked_cross_entropy(logits, targets, *, valid_mask=None,
                     coordinate_mask=None, reduction="mean") -> Tensor
masked_nll_from_probs(probabilities, targets, *, valid_mask=None,
                      coordinate_mask=None, reduction="mean") -> Tensor
```

`reduction` 仅支持 `mean/sum`；保留原 CE 默认返回标量，不批量破坏调用方。有效 mask 和计数用同一个小 helper 生成，计数为 bool mask 的元素数而非平均比例。

**实现步骤：**

1. CE 只接 logits；NLL 计算 `-log(p_target)`，建议 float32 中 `clamp_min(1e-12)`。概率非 finite/负值/质量误差在算子输出或测试边界检查，不在每个调用重复大范围同步。
2. padding/未监督位置先筛选再 gather，避免 invalid target sentinel 触发 gather 越界；对有效 target 仍检查范围。空集合返回与输入连接的零。
3. operator loss 和 metrics 的 probability 路径全部迁移；transport CE 保持 logits 路径。搜索所有 `masked_cross_entropy` 调用确认无遗漏。
4. model metrics 提供 `nll_sum`、`supervised_tokens`，用于验证累计；指标 detached，优化用的 loss 保留梯度。
5. trainer evaluate 按 `sum(nll_sum)/sum(supervised_tokens)` 汇总；transport accuracy 的分子、分母使用同一有效监督 mask。训练日志也累计相同 token 计数，避免仅平均 batch means。
6. `content_weight` 默认 0。冻结 base 的旧 content term 删除，非零值暂时显式报错：该正则尚未实现。不要此时擅自加 KL、cycle 或另一个 encoder。

**独立验收：**

- 9 类中 p_target=.99，NLL 约 .01005033585；uniform NLL 约 log(9)。
- 随机 logits：CE(logits) 与 NLL(softmax(logits)) 在数值与 logits 梯度上匹配。
- 同一数据拼接/拆 batch、添加无效 padding 后总 NLL 和 accuracy 不变。
- 恒定 NLL=2、监督比例=.05，最终 NLL 仍为 2；不能变成 .1。
- 全空监督不写 best、不 step optimizer，报告计数；任一非空样本不会被它稀释。
- 移除旧测试中“content_weight 使标量变大即通过”的断言，改测非零配置被拒绝。

**完成条件：** 定向测试通过，进度记录明确旧 NLL 不可比较。不得在此任务重新训练旧 checkpoint。

### R02：统一 mask，并补齐任意 kernel 的区域约束

**文件：** `model.py`、`operators.py`、`masking.py`；测试 `test_mts_contract.py/test_mts_operators.py/test_mts_end_to_end.py`。

**补充核验：** 编写本计划时已确认默认 arbitrary kernel 的 hard mask 全零仍改变 base probability，因为零 kernel logits 会变 uniform。这是审核报告之外新增的修复项。

**实现步骤：**

1. `OperatorBatch` 增加或复用一个 `effective_edit_mask(spec)`，按 §1.1 处理 broadcast、valid、device；`supervision_mask` 使用同一实现。`visible_mask=None` 在 loss/forward 编辑路径报带字段名的错误。
2. CLI 稍后传明确 mask，禁止通过把所有可见 token 改为 hidden 来“修掉异常”。局部编辑默认 `visible=~support`，另允许 support 内显式 anchor。
3. operator 中把 bool `eligible` 与 float `strength` 分开。输出最后以 `where(eligible[...,None], transformed, base)` 保证区域外 identity；不能用 `strength>0` 代替 eligible，否则会错误改变 arbitrary kernel λ=0 的对照定义。
4. arbitrary 的 `identity_mix=True` 校验 λ∈[0,1]；默认模式保留其不同于 CTMC 的 identity 特性并正确描述。`kernel_offdiagonal_mass` 改为真正非对角元素求和，不能把每个对角值从整行减去。
5. `OperatorOutput` 检查 finite：NaN 不得因 `<0`/质量误差比较均为 False 而漏过。
6. CTMC 保持 row generator / row-vector p 的约定。只有先验证 current implementation 是否失败，才改 uniformization；达到 max_terms 仍超 tail tolerance 时明确失败，不能靠 renormalize 隐藏严重截断。
7. CTMC 对截断前 mass error/nonnegativity 设置合理数值检查：float64 oracle 更严，float32 使用例如 1e-5 质量容差。把理论 Poisson tail 与浮点质量误差分别记录。
8. 明确 `level_order` 为“沿链访问的 level 列表”：边应连接 `order[i]→order[i+1]`；必要时通过逆置换修正 `_level_generator`，用具体排列测试，不靠随机排列只测“不同”。

**验收：**

- 每族 hard mask=0、visible=True、valid=False 的位置均逐位保持 base。
- arbitrary 默认 λ=0：eligible 内 uniform，外部 base；identity_mix 模式 λ=0 全部 base，λ>1 报错。
- CTMC 随机非均匀 rates、边界 level、零 rate、不同 batch strength 对比 `p @ matrix_exp(Q)`；比较值和梯度。
- max_terms 故意设小触发显式失败；NaN input/output 被拒绝。
- 固定 Q 测半群，不能对每轮重算 Q 的 generation 宣称相同性质。

### R03：正确的随机数与单步 token 编辑

**文件：** `sampling.py`、`model.py`、`scripts/generate_mts_operator.py`；测试 `test_mts_sampling.py/test_mts_end_to_end.py`。

**实现步骤：**

1. CRN 使用稳定实验 key，建议 `(sample_id, step_id, shape)`；uniforms 在 CPU 由指定 seed 生成/缓存，需要时搬到目标 device。不要使用 `id(device)`，不要以缓存创建次序隐式决定不同条件的种子。
2. 同一 sample/step 跨 λ、算子条件显式共享 uniforms；不同 sample_id 独立。不要求独立随机 token 一定不同，验收独立 uniforms 更可靠。
3. CPU generator + CUDA probs 的路径保留；有 CUDA 才运行设备测试，缺设备记录 skip。
4. 修 inverse CDF 的边界：按右侧搜索 `CDF>u` 取首个 level，zero-mass bins 不能在 u=0 被选中；u 的规范为 [0,1)，外部 u=1 可显式 clamp 到小于 1 的最大可表示数，u<0/u>1 报错。
5. `generate_edit` 只写回 effective_edit；其余位置复制 source（包含 visible anchor、padding）。空 edit 直接返回 source，不运行无意义采样。
6. generation CLI 显式构造 visible/support，每个 sample 使用新 sample_id；输出保存 mask、sample seed 和 source/base/styled 区别。
7. 当前 `--no-locked-edit` 与 hard support 保证冲突：revision 2 第一版对该组合报错；全身生成通过 whole_body support 表达，不静默越界采样。

**验收：**

- one-hot 在 level 7，u=0/0.5/接近1 都只能取 7。修改现有把 u=0→level0 视为正确的测试。
- 相同 probability 两次 CRN.sample，逐位同 token，cache 项数稳定。
- `--samples 4` 的 uniform key 四个不同；重复执行同一命令可重现。
- 人工 nonzero style head 下换 style/λ 改变 support 内 probability；锁定区域、visible anchor 与 padding 全不变。
- 更换 hidden source token 数值但 mask 不变，不应改变 base 输入看到的信息；避免目标 token 偷看。

### R04：修标签语义、holdout 和 pair 曝光记录

**文件：** `pairs.py`、`scripts/audit_style_pairs.py`、必要时 `windows.py`；测试 `test_mts_pairs.py`。只有确认 packed token 缺身份字段时才触及 `data/packed_token.py`，不重写 token 字节。

**实现步骤：**

1. 为 100STYLE 明确后缀集合 `BR,BW,FR,FW,ID,SR,SW,TR1,TR2,TR3`。它们是 action；style 是剩余名称；所有片段的 performer 使用数据集级已知单演员标记，例如 `100style_actor_0`，并记录来源。TR 可保留子类型，也应有统一 transition action family。
2. 只在明确 100STYLE 来源上使用该解析，不能把其他 dataset 的类似后缀套用。未知 actor 保持 unknown，不用 source group 冒充演员。
3. 显式 metadata 优先于文件名；但若 metadata 来自旧错误解析，必须按 source/provenance 识别并拒绝沿用。保留原始 label 供审计。
4. 提前构建 performer_axis，再生成依赖它的 warnings。unknown actors 时不能声称 overlap=0 或 zero-shot supported。
5. train 阶段统一过滤 target 和 reference 的 held-out style，覆盖所有 pairing mode 及显式 `targets=` 参数；target 的 split 也必须校验。
6. 分离三种集合：configured vocabulary、eligible vocabulary、actually sampled vocabulary。summary 输出真实集合和计数，不再用 style_split 的长度代替。
7. 配对先排除无有效窗口的 clip。保存 rejection reasons（same take、same content、heldout、window unavailable 等）和每 style/action/actor 的有效 pair 数。
8. 增加简单的 `target_sampling: clip_uniform|style_uniform`；主 style sandbox 配置用 style_uniform，再在该 style 合法候选中抽 content/clip。不要通过复制稀有 clip 伪造更多独立样本。
9. 不强行给 SEED 单 content style 配不同 clip 并称 different content；审计明确排除原因。reference 与 target 不得为同 take 的相邻 crop/镜像。

**验收：**

- `100style/Flapping_FW`→style Flapping、action FW、同一演员；不生成“演员 FW”。
- 三种 mode 均无 held-out target/ref；跨 split 显式 target 被拒绝。
- 有演员重叠和无演员元数据的两个 fixture 都能输出合理审计。
- 单 content style 不能出 same-style/different-content pair；实际 style 数与日志一致。
- style_uniform 在固定小数据的确定性抽样上覆盖 eligible styles，且不改变 split isolation。

### R05：共用 batch 来源，并接入真实 action condition

**文件：** `windows.py`、两个 train 脚本、`transport.py`、必要时 `data/loader.py` 的小范围 metadata 开关；测试 `test_mts_transport_training.py/test_mts_pairs.py`。

**首版决策：** 实现 `content.kind=none|action_id`，不在本轮默认接 trajectory 或 phase。action_id 只能约束动作类别，不能宣称保持 source 的精确轨迹/节奏。

**实现步骤：**

1. 将 `PairedBatchSource` 从训练脚本搬到 `windows.py` 或同目录现有合适位置，不复制一份；CLI 只负责解析和调用。
2. 使用 `accepted_pairs` 构建 tokens/style_ids/metadata；跳过窗口失败后所有字段必须仍一一对应。target/reference 长度可不同，分别构造 valid_mask。
3. source 输出 mapping，至少包含 `tokens/valid_mask/content_condition/sample_metadata`；paired source 同理。保留 v3 indices 与 v4 tokens 的入口差异在 reader 内。
4. action vocabulary 从训练集合构建并排序固化。无标签或未知 action 不可默认为 0、行号或 style id：action_id 主实验直接报错；unseen-action 任务留待新条件表征，不用未训练 embedding 冒充泛化。
5. transport 与 operator 使用同一 action map；operator 的映射从 frozen transport 继承，不能自行按当前 eval split 重建。条件来自 target/source action，不来自 reference。
6. `ContentConditioner` 已有 integer embedding，复用它；配置 `content_classes=len(map)` 并在 checkpoint 保存 map。
7. TokenSource 不再丢掉 valid_mask/metadata。packed loader 必要时打开已有 return_metadata，不建立第二个庞大 DataLoader 框架。
8. feature 在线编码路径与预编码 tokens 对齐：读取足够 encoder history，按 `history:history+T` 取 token，使用 tokenizer 自带 feature stats 的 model space；禁止直接取前 T 帧、或使用新 store stats 偷换 tokenizer 坐标系。
9. 默认只选足够长的窗口，短 clip 标记 excluded；若保留 padding 路径必须携带 valid mask。不能把复制 padding 的 frame 全标有效。

**验收：**

- 同一 clip/window 的在线编码与 token store 读取一致（明确 encoder history、边界 padding 约定）。
- fixture 中跳过中间一个 pair 后 style/action IDs 仍对齐。
- 同一个 action 字符串在 train/eval/新进程中得到同一 id；映射变更报错。
- full mask 下更换真实 action，经人工设置 conditioner 或小任务训练后输出不同；更换 reference 不得改 action id。
- content.kind=none 的输出注明 unconditional，不冒称指定内容生成。

### R06：修复 coordinate 信息损失与输出 head

**文件：** `embeddings.py`、`transport.py`、`style_encoder.py`（构造参数透传）、`layout_adapter.py`（仅必要的只读索引）；测试 `test_mts_transport.py/test_mts_end_to_end.py`。

**固定设计：** 每个 stream 内按 layout coordinate 顺序拼接 embedding，再线性投影到 D；每个 stream 独立 `D→K_stream×9` 输出 head。先不用 family tying，避免左右局部顺序歧义；新增参数量必须记录。

```text
level embedding [B,T,40,E]
  -> hidden token 用 learned mask embedding 替换 level 项
  -> 加 coordinate identity（visible 与 hidden 都保留）
  -> 每 stream 按固定顺序 flatten K_stream*E
  -> per-stream Linear(K_stream*E,D) + stream embedding
  -> [B,T,13,D]

stream hidden -> per-stream Linear(D,K_stream*9)
              -> 按 canonical slices 拼成 [B,T,40,9]
```

`E` 增加明确配置 `token_embed_dim`，首版默认 16；D 继续由 transport/style encoder dim 决定。所有 slice 从 layout adapter 获取，不手写 40-coordinate 分组。

**验收：**

- 构造确定性权重，使同一 stream 的两个 coordinate 取值交换产生不同 embedding；测试不能只依赖随机阈值。
- hidden token 的真实 level 变化不改变输入；改变哪一个 coordinate 被 mask 可区分。
- 两种 hidden context 下，同一 stream 两个 coordinate 的 logits 差可以变化，不再只由静态 bias 决定。
- graph_depth=0 下，一个 stream 输入不影响其他 stream 的 output；保持 ownership 测试。
- 序列化 config 含 E 和 architecture revision；旧权重 shape 不匹配时不使用 `strict=False` 静默加载。

### R07：加入明确的时间位置编码

**文件：** 新增 `mts_operator/temporal.py`；修改 `transport.py/style_encoder.py`；测试 `test_mts_transport.py/test_mts_end_to_end.py`。

**固定设计：** 一个共享的 sinusoidal position 函数/小模块，输入 length、D、device、dtype，输出 `[1,T,1,D]`。分别在 transport/reference encoder 的 temporal attention 前加入。首版不引入 RoPE、可学习长表或 temporal convolution。

**实现要求：**

- config 保存 `position_encoding: sinusoidal`，支持训练外更长 T，不设置 64 帧硬上限。
- encoding 只取决于明确的窗口 position，不从 token 值或 batch 顺序推断时间。
- valid padding 继续通过 key-padding mask 与 pooling 排除；绝对位置不能使 padding 进入描述符统计。
- causal 模式保留未来屏蔽；不要因增加位置编码破坏既有 causality。

**验收：**

- 不同 t 的位置向量不同，长序列可构造，无 device/dtype mismatch。
- 固定非退化权重 fixture 上 full mask 各帧输出不再因结构被迫相等。
- reference token 时间反转/重排在确定性 fixture 上改变 descriptor；全相同帧是合法例外，不写普遍“任意重排必变”断言。
- 追加 masked padding 后有效帧输出/pooling 不变（容差按 float32 设置）；causal 前缀不受未来 token 改动影响。

### R08：统一 checkpoint bundle 和工件绑定

**文件：** `checkpoint.py/contract.py`、四个 MTS CLI；测试新增 `tests/test_mts_checkpoint.py`。先公共 API，后逐个迁移 CLI，避免一次改六个入口造成循环导入。

**固定设计：** operator 当前 state_dict 已包含 transport 权重。revision 2 从自身 model_config + state_dict 重建全部 MTS 模型；推理不依赖外部 transport 文件仍在原路径。外部 transport SHA 作为训练 provenance 保存；tokenizer 仍是单独必需依赖。

建议公共入口：

```python
load_operator_bundle(path, *, adapter, tokenizer_identity,
                     device="cpu") -> (checkpoint, model)
validate_store_binding(store, *, tokenizer_identity,
                       expected_data_identity) -> None
```

**metadata 最少字段：**

```text
schema_version=2, architecture_revision=2, metrics_version=2
model_config：transport、encoder kind/config、operator name/config
alphabet_hash / layout_hash / representation metadata
tokenizer_checkpoint_sha256（复用现有 sha256_file）
feature_schema_hash, normalization_hash, split_manifest_hash
action_to_id, style_to_id（style-ID 模型必需）
resolved_config, seed, code_commit, working_tree_dirty
upstream_transport_sha256, training_protocol_id
实际训练允许/排除的 style 和 action 集合
```

**实现步骤：**

1. 区分结构 alphabet identity 与权重 identity；首版使用现有 checkpoint 文件 SHA256，不另发明 tensor hash。即便只是重新封装文件导致 SHA 变化也严格拒绝，并给出原因。
2. 交叉验证 token store 的 checkpoint/schema/norm/split 与 MTS metadata。feature 入口复用已有 normalization/model-space 转换；跨 split 正式评估必须有显式协议，不悄悄豁免。
3. 构造器参数来自完整 versioned config，不再由 eval/generate 各维护白名单。`style_dim` 由 encoder output_dim 显式给定，并验证匹配。
4. `level_order` 全量记录（包括 identity 顺序），加载结果保持完全相同。operator/encoder 类型取结构配置，不再靠 metrics 字段猜。
5. 保存 checkpoint 与 manifest 用临时文件+同目录原子 replace；失败不能留下半个 best.pt。
6. token store 缺 actor 标签时，通过其 source identity 对齐 feature/catalog metadata；不能因没有 performer 列静默声称 actor-disjoint。若需新增 token store metadata，只写新 sidecar/新产物，原文件不原地改写。
7. frozen transport 保持 eval；若支持 freeze_style_encoder，也覆盖 model.train() 后冻结子模块仍为 eval 的行为。

**验收：**

- 三种 operator × reference/style-ID round-trip：同输入 probability 在 CPU eval 下 allclose，shuffled Q 及 level_order 一致。
- 将外部 transport 原路径暂时不可访问（fixture 内）仍能加载 operator bundle。
- 错 tokenizer（同结构不同权重）、错 normalization/split/action map 均明确报错。
- style-ID 映射由 checkpoint 恢复，新 eval 数据排列不影响 ID。
- schema1 默认拒绝；不可用 `strict=False`、metadata defaults 或路径替换绕过。

### R09：固定验证、清理配置假开关与恢复语义

**文件：** `training.py`、两个 train 脚本、MTS revision2 configs；新增 `eval_protocol.py` 的验证窗口清单部分。测试 `test_mts_transport_training.py` 与 `test_mts_cli.py`。

此任务分三次改动：R09a 固定验证，R09b 配置，R09c warm-start。每次先运行自己的定向测试。

**R09a：固定验证**

- 预先记录 validation window/pair、起始帧、mask_kind、mask seed、条件、split identity。一个 row 有固定 sample_id。
- train RNG 与 val RNG 独立；验证不推进训练 mask/pair RNG。
- 每个 mask kind 有固定验证子集；全局 best 依据固定各类权重的 validation objective，按 R01 有效 token 数正确归一化。记录每类 count，空类不能偷偷重新分配权重。
- 不再每 epoch 只抽一个随机 batch。实现 `evaluation.validation_batches_per_kind`，smoke 可设 1，正式配置至少 4；预算和验证种类写入工件。
- best 只接受 finite、有监督计数的验证结果；全空验证直接报错，不退化为 train loss 选 best。

**R09b：配置真执行**

- 统一顺序：YAML→CLI overrides→数据/模型推导→校验→保存 resolved config→构建。
- 新建 `data/configs/mts_revision2_transport.yaml`、`mts_revision2_style.yaml`，不要覆盖旧历史配方。使用 schema4 时正确填 4；数据路径必须显式校验。
- style 配方首轮固定期望 mixture：full_generation=.30、random_coordinate=.30（coordinate_ratio=.80）、stream=.15、temporal_span=.10、spatiotemporal_block=.15。记录实际每类 batch 数和有效隐藏比例；短 smoke 不强求统计频率恰好30%，正式运行检查实际曝光。transport 配方可以保留自己的 mixture，但所有方法对照使用同一冻结 transport。
- 未实现的字段删出新模板；用户传入仍无行为的非默认字段时报错，不能打印 ignored 后继续主实验。
- `val_every_steps` 本轮只支持 0（epoch-end），非零报错，避免再维护第二套 checkpoint 节奏。
- MTS revision2 首轮仅支持 fp32，precision=amp 明确报“本轮未支持”；不再用仅 GradScaler 冒称 AMP。AMP 优化列入后续任务。
- `reference_frames` 真正控制 reference window 长度；如果暂不支持不同长度就拒绝与 frames 不同的值，不静默忽略。
- dry-run 输出模型类型/参数量、style/action map、数据身份、mask mixture、预算与 output；不训练、不写 best。

**R09c：warm-start 而非伪 resume**

- 新训练入口明确 `--warm-start` 只载同 revision 权重；重置 optimizer/global_step/best/RNG，并记录来源。
- 原 `--checkpoint` 如果代表伪 resume，给迁移错误提示，不再打印 resumed。真正 exact resume 暂不实现，不伪造恢复保证。
- 已保存 optimizer 不代表已恢复，文档说明。若后续用户要求 resume，再独立实现 RNG/sampler cursor/optimizer/scaler 状态，不在此任务扩张。

**验收：** 同 checkpoint 重复验证值一致；改变 train seed 不影响固定 val；改变 CLI dim/batch/kind/level_order 确实改变 resolved config/构建结果；无效参数报错；warm-start step 从零开始且不继承旧 best。

### R10：重建 reference/style-ID 的评估协议

**文件：** `eval_protocol.py`、`metrics.py`、`scripts/evaluate_mts_operator.py`、`scripts/generate_mts_operator.py`（style-ID 输入）；测试 `test_mts_metrics.py/test_mts_cli.py`。

**固定 eval manifest：** 保存 JSON metadata + JSONL rows，或一个小 JSON 文件；首版不引入数据库。至少含：

```text
protocol_version, dataset/split/tokenizer identity, vocabulary hash
sample_id, target clip/take/actor/style/action, target start/length
correct reference clip/start/length/labels
wrong-style reference clip/start/length/labels
random-real-reference clip/start/length/labels
candidate references、positive indices、mask seed/kind
region/radius/frame_range、condition、sample seed
```

**实现步骤：**

1. manifest 生成一次、后续只读；eval 使用 `--eval-manifest`。新建 manifest 功能可作为 evaluate 脚本的 `--build-manifest-only` 分支，不再增设通用调度框架。build-only 只需要数据身份与协议参数，不应要求已有训练好的 operator，以免形成“先训练才能固定验证集”的循环依赖。
2. wrong reference 必须 style 不同；优先匹配 action，有条件时匹配 actor。若该行缺合法负例，标记 unavailable/reason，不用 roll 或均匀 token 代替。
3. random reference 从真实合格片段抽取并记录标签；可恰好同 style，不能强行命名 wrong。均匀 token 只保留在明确 `ood_sanity` 附加项。
4. 排除同 take、重叠 crop、镜像对泄漏；用 R04 数据身份，不只检查 clip_id 不同。
5. 多正例 retrieval：候选集中所有同 style 合格 reference 都算正例；命中按 target 累计 hits/count，不能对 batch accuracy round。tie 按预先规定规则（例如固定候选顺序首个最小值）处理；报告相应 chance 和候选分布。
6. base NLL、correct/wrong/random NLL、TV 均使用同一完整 batch（hard mask/visible/valid/content）计算，先设置条件再测量。
7. style-ID 模型提供 `--style-label`，通过 checkpoint style_to_id 转换；评估以正确/错误 ID 对照，reference retrieval 标 `not_applicable`。未知 ID 标签必须报错，不能评未训练 embedding 后称 unseen style。
8. `--held-out-styles` 不作为用户自述即可成立的证据：与 checkpoint 实际训练词表和 manifest 对照，seen/unseen 分组分别报告。data split=test 不自动等于 unseen style。
9. 输出每 sample 原始 row，包含 counts、条件、seed 和 reason。未计算项用 null/reason，不填 0。JSON 禁止 NaN，写出前验证 finite。
10. `content_preservation` 中 NLL proxy 改名为 target_token_nll/likelihood_delta；保留诊断，但不能叫独立 content recognition score。

**验收：**

- toy batch 有两个同 style reference，任一个被排第一都命中。
- 2/8 命中输出 .25；不同 batch partition 汇总不变。
- wrong reference 标签确实不同、随机真实 reference 来自数据。
- 支持同 style 多 content、同 content 多 style，并且没有 split/take 泄漏。
- style-ID 与 reference CLI 均可从保存的 checkpoint 完整执行；候选缺失时报告 unavailable，不崩溃、不假造。
- 不要求未训练模型正确 reference NLL 必胜过 wrong；协议正确与训练效果分开验收。

### R11：重写误名物理指标并保留测量条件

**文件：** `metrics.py`、`scripts/evaluate_mts_operator.py`、`scripts/plot_mts_figures.py`；测试 `test_mts_metrics.py/test_mts_cli.py`。

**实现步骤：**

1. physics 不再重建缺 visible mask 的 batch；从当前完整 batch 只替换 strength，保持其他字段。
2. 输出三种比较：source reconstruction→base sample、base sample→styled sample、source reconstruction→styled sample；base/styled 使用同 sample/step uniforms。原始 GT 可读时再单独报告 GT→reconstruction。
3. tokenizer 解码带足够历史，评估区间与 decoder warmup 分开。首版不使用无历史窗口的开头伪装稳态质量；可丢弃明确 warmup 帧，但记录有效评估范围。
4. jerk 使用 FK world positions：`diff³(position)/dt³`，单位 m/s³；分别测 edit start/stop 附近固定邻域（例如各 3 帧），并另报整个区域内 jerk。窗口不够时 unavailable。
5. 原一阶 feature 指标如需保留，改名 `feature_delta_change_max`；旧 `boundary_jerk_max` 不保留同名不同单位，metrics_version=2。
6. off-target FK 按 layout 的 owned/descendant influence set 区分，不把 ancestor 编辑引起的合法后代变化误标泄漏。支持 radius0/radius1；root/global 的 world-space 影响单独注明。
7. foot slide/contact 复用既有 FK/反归一化函数，报告单位、接触阈值、gate 来源。接触数量为 0 时返回 null+count，不用除1得到漂亮0。
8. 记录源/输出平均速度、foot contact prevalence 等退化诊断；不得只按低脚滑排序。
9. 汇总 key 至少包含 protocol/checkpoint/operator/encoder/split/seen_axis/regions/radius/frame_range/strength/comparison。不能把 λ=0 与 λ=1 混合成一个平均值。
10. plot 只消费相同 metrics_version、同协议的结果；输入不兼容时指出文件和原因，不偷偷拼接旧表。

**验收：** 匀速轨迹 jerk=0，三次多项式轨迹 jerk 与解析值一致；相同 motion 比较为0；边界外异常不污染边界指标；不同 strengths 分组保留；无接触帧有明确 unavailable。

### R12：修正 iterative generation，而非反复全区重采样

**文件：** `sampling.py`、`transport.py`、`model.py`、generate/evaluate CLI；测试 `test_mts_sampling.py/test_mts_transport.py/test_mts_end_to_end.py`。

**固定算法：** 先实现 monotonic masked filling，最简单的确定性位置顺序，随后才考虑 confidence schedule。不要一开始引入多种策略。

```text
editable = 初始 effective_edit
remaining = editable
for step in 0..N-1:
    使用当前 tokens/visible，计算 base 或 styled probabilities
    draw 用 sample_id + step_id 的显式 uniforms
    按每样本固定 coordinate/frame 顺序，从 remaining 中选择本轮份额
    只提交 selected，设 selected 为 visible
    remaining 去掉 selected
最终剩余必须为空；初始 visible、support外、padding 不改
```

- 每轮份额使用 `ceil(remaining_count / remaining_steps)`，每样本独立；edit 为空的样本直接跳过。
- transport/operator 共用“选哪些位置/如何提交”的小函数，避免两套 schedule。
- `steps=1` 等价单步编辑。`steps>1` 不再重复 resample 已 observed 的位置。
- 所有比较记录 steps/schedule/temperature；以同设置比较方法，不把多步带来的质量变化归给 CTMC。
- eval NLL 是固定 teacher-forced mask 协议；generated motion metrics 是 iterative 协议，分别命名。

**验收：** 每个 editable 位置只提交一次；hidden tokens 未暴露前不会影响 logits；N=1/2/8 各自确定性可复现；超出 edited count 的 steps 不报错；source anchors 始终不变。此处不要求复杂生成已经具有高质量。

### R13：修 geometry 与 temporal influence 的测量公平性

**文件：** `nef_probe.py`、`scripts/probe_nef_geometry.py`、`scripts/evaluate_nef_locality.py`（必要的协议字段）；测试 `test_nef_probe.py`。

**实现步骤：**

1. 增加 protocol version=2。保留历史 JSON，但新结果不可直接覆盖同路径。
2. 相邻与远距比较用同一 seed、source、coordinate、time support、有效位置交集；分别测单帧 pulse 和固定 signed offset span。
3. offset amplitude 分层：±1 与 ±d（例如4），只有两者都合法的位置入比较；不要一侧边界裁剪一侧随机跳，再用不同分母求比值。
4. 随机逐帧 far 扰动可作为独立 stress test，命名清楚，不能用其 jerk 支持 ordinal geometry。
5. `frame + decoder_influence_frames < length` 才称完整 temporal probe；例如选择128帧、edit at32。读 layout/model 实际 RF，不写死33。
6. 有效区分 token→feature 直接时间影响、FK descendant 影响和 root 积分后 world-space tail。检查影响范围时用数值容差，并报告实际 tail/窗口截断状态。
7. radius0/radius1 均输出 support 内实际 edit magnitude 与 support 外影响，后续才能匹配编辑强度。

**验收：** toy decoder 的已知 RF 被完整覆盖；截断窗口明确标 truncated；固定幅度/随机幅度两类不能混汇总；边界 illegal 的位置在两臂一致排除。

### R14：集成 smoke、实验预检与状态收口

**文件：** 新增 `tests/test_mts_cli.py`（前述任务逐步补齐）、两个 revision2 configs、README 和三份阶段文档的状态说明；允许新增一个小脚本 `scripts/preflight_mts_revision2.py`。

**R14a：真正的 CLI synthetic smoke**

- 使用临时 miniature store、真实 tokenizer fixture 与真实 MTS 序列化。复用现有测试的骨架/feature-store fixture，不仅 mock `main()`。
- 每种 encoder/operator 至少覆盖构建、≤2训练步、保存、加载、一次 eval/generate；架构 dim=16或32、depth=1、T=8或16、B=1或2。
- style/reference 行来自不同 fixture clip/content，不能 reference=target 作弊。
- 设置 subprocess timeout=120秒、OMP/MKL线程=1；CUDA 可选 guarded smoke，绝不触发真实8000步配方。
- CLI override、未知参数、错误hash、空validation、无效style、shuffled adjacency、输出非finite都要有失败路径测试。

**R14b：只读真实预检**

- 读取指定 store/checkpoint metadata，列出实际可配对 style×action、actor/group overlap、有效窗口数。
- 校验 tokenizer/token store、action map、planned heldouts、mask分布、fresh revision2 output目录。
- unseen performer 必须核查 tokenizer、transport、operator、normalization 的训练曝光；operator-unseen style 与所有上游也未见 style 分开标注。缺上游训练记录时报告 exposure_unknown，不替用户宣称 zero-shot。
- 生成实际 train/eval 命令文本和预算表：steps、B、T、有效token预算估计、seed、模型参数量；不执行命令。
- 旧MTS checkpoint传入正式新实验应被拒绝；holdout tokenizer/token store通过绑定后可复用。
- 真实读取 smoke 上限8个窗口、每条生成1个样本；无已训练 revision2 模型时只验数据/构建，明确“生成效果未验证”。不能用随机模型证明style成功。

**R14c：文档与配置收口**

- 阶段状态文档增加 audit/revision2 状态和旧数字失效范围，撤回“直接恢复旧26-cell即可”的建议。
- 修旧100STYLE performer解释、actor holdout旧权重续训示例和临时 `/tmp` 恢复脚本依赖。
- 保留历史报告，不抹去失败；当前主入口链接新config/protocol与进度记录。
- 扫描 README/doc 中声称支持但未实现的能力：unseen style、multi-style、content loss、AMP、resume 必须准确。

**最终验收：** G-code 与 G-ready 分开报告；列全量相关测试命令/结果、skip原因、剩余研究任务、待用户批准的训练预算。不写“创新性已验证”或“完整方法完成”。

## 4. 建议测试组织与命令

不必为每个微改动建新测试文件。复用已有文件，新增仅 `test_mts_checkpoint.py`、`test_mts_cli.py`，必要时把共享 fixture 放到 `tests/conftest.py`，不要在测试中依赖另一个测试的运行顺序。

每项先跑涉及的单文件。接口改动传播时运行以下相关集合（shell glob 仅测试文件）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_*.py tests/test_nef_token_contract.py tests/test_nef_probe.py \
  tests/test_nef_fsq.py tests/test_packed_downstream.py
```

关键检查表：

| 领域 | 必须防回归的反例 | 禁止的假通过 |
|---|---|---|
| loss | p=.99、uniform、值/梯度、padding、batch partition | 只断言 loss finite/下降 |
| mask | eligible/visible/valid 三者独立；全部算子区域外 identity | 只检最终 token 被 where 锁住 |
| sampling | u=0不落零质量bin；同sample共享/异sample独立 | 只比较shape、所有样本重复也通过 |
| architecture | coordinate swap、context-dependent head、时间顺序 | 仅统计参数数量 |
| labels | 100STYLE action不当actor；三mode heldout | 用不同clip_id代替different content |
| checkpoint | 同结构异权重拒绝、shuffled round-trip | strict=False、缺metadata填默认值 |
| eval | 多正例、2/8=.25、真实负例、固定manifest | roll一帧充当wrong-style |
| physics | 多项式jerk、单位、边界邻域、zero-contact | 一阶normalized feature差叫jerk |
| CLI | 参数→resolved config→模型→artifact | parser接受参数即宣称支持 |

测试失败说明测试正确捕捉问题时，应修实现。若既有断言本身错误（例如u=0取zero-mass level），在进度记录写清数学原因再更新断言。

## 5. 明确留给后续研究/扩展的任务

这些项不阻塞当前代码修订交付，但阻塞相应论文主张。不要执行到一半顺手开始大模型训练。

| 后续 ID | 何时做 | 最小实现/验收方案 |
|---|---|---|
| E01 trajectory condition | action-only模型无法保住source轨迹时 | 复用 packed_trajectory 的 `(values,valid)`；按源clip身份+clip-local offset连接，不能复用不同store的物理offset；target条件、train-only stats；有效mask单独传，尾部不跨clip；对照action-only |
| E02 no-reference baseline | 首轮科研小试必做 | constant global descriptor /同容量无reference adapter；参数量与训练token预算同时报告；用于区分修补transport与学习style |
| E03 FiLM与表征对照 | logit/style-ID有信号后 | frozen同一base上的FiLM adapter；再接part layout，避免硬把NEFLayout用于part；不在核心修复期重写通用layout |
| E04 disjoint multi-style | 单style局部编辑通过后 | 明确每个region自己的reference；固定一次base/context分别构造rate/field；首版重叠mask报错；分别测region style与off-target |
| E05 independent evaluator | 主结果前 | 独立训练/测试split的style/action evaluator，balanced指标与take/actor级bootstrap；与模型自评NLL分开，预训练预算另批 |
| E06 short/long sequence | 64帧质量通过后 | 检查decoder history、chunk重叠、root积分，5–30秒动画；不以单窗口通过声称长期稳定 |
| E07 性能 | profiling确认瓶颈后 | 批量reference评分、缓存窗口、减少CPU/GPU同步；CTMC三对角优化前保留matrix_exp oracle；AMP单独实现并验gradient/finite |
| E08 exact resume | 用户确有断点续训需求时 | 保存并恢复optimizer/step/best/RNG/sampler游标；对比连续N步与分段N步参数/数据顺序；不只恢复state_dict |
| E09 geometry-aware rates | CTMC与shuffled对照建立后 | 在相同style fidelity下比较解码几何/接触预算；属于新假设，不作为基础修复的必做项 |

首轮实验建议按 **base transport → style-ID logit → no-reference对照 → style-ID CTMC → reference encoder** 顺序，而非所有方法同时铺开。若固定小任务仍无法区分正确/错误style，先停在数据/架构诊断，不扩大算力。

## 6. 可直接粘贴给执行 agent 的启动指令

```text
请执行 docs/MTS_FSQ_Agent_Code_Revision_Plan_zh.md。

先读第0–2节和R00，检查当前代码与用户已有改动。创建或恢复
docs/MTS_FSQ_Code_Revision_Progress_zh.md，按R00→R14顺序串行推进。
一次只处理一个任务；按任务补独立反例测试、实现、运行定向验证、检查diff、记录结果。
依赖满足且测试通过后继续下一任务，不必逐步请求确认。

保留NEF tokenizer的40×9/13-stream contract，不改renderer，不做无关重构。
不覆盖或删除旧checkpoint/store。不启动昂贵训练、全数据重建或26-cell矩阵。
允许本计划规定的CPU单元测试与短synthetic CLI smoke。

严格执行计划中loss、mask、strength、schema、action condition、checkpoint绑定的决策。
不要用strict=False、伪造content id、roll reference、降低测试标准或缺指标填0来绕过失败。
如果发现与计划不一致的新问题，先定位并记录证据；在既定范围内修复，研究选择留到后续。

最终交付：代码diff、逐任务状态、实际测试命令与结果、工件路径、未完成项，
分别说明G-code和G-ready是否通过。代码通过不等于风格效果或论文创新性通过。
```

恢复工作时可粘贴：

```text
阅读 docs/MTS_FSQ_Code_Revision_Progress_zh.md 和
docs/MTS_FSQ_Agent_Code_Revision_Plan_zh.md。
先核对记录与当前git diff、测试证据；从第一个未完成且依赖已满足的任务继续。
不重复已经通过且未受新改动影响的工作，不根据旧目录名推断实验完成。
```

## 7. 本文件自身的交付状态

本文件是执行规范。生成时只完成了代码路径复核和一个 arbitrary-kernel 零support的短反例检查；没有执行R00–R14、没有修改训练实现、没有启动训练。后续的完成状态必须以执行进度记录和实际测试证据为准。

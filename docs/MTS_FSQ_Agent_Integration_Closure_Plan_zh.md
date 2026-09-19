# MTS-FSQ 下一阶段执行计划：集成补缺、生成闭环与实验准入

日期：2026-09-18；版本：v1.0。

面向执行层：zcode + deepseek-v4.1-flash，单 agent 串行执行。

输入文档：

- [上一轮代码修订计划](MTS_FSQ_Agent_Code_Revision_Plan_zh.md)
- [执行进度报告](MTS_FSQ_Code_Revision_Progress_zh.md)
- [原始项目审核](MTS_FSQ_Project_Audit_2026-09-17_zh.md)

**当前决策：保留已有 R01–R11 的有效实现，先修复实际入口的集成缺口，再完成 R12–R14。暂不进入大规模训练或新增研究模型。**

本文件补充并细化上一轮计划，不要求从 R00 重做。涉及接口决策时以本文件为准；未涉及的 NEF contract、loss、CTMC 数学、旧工件保留规则继续遵守上一轮计划。

## 1. 本轮复核结论：下一任务不是直接开始 R12

进度报告正确地把 G-code/G-ready 标为未达成，但“R00–R11 已完成”的表述过于乐观。已存在可复用的 loss 修复、coordinate-aware embedding/head、时间编码、CRN、标签解析、checkpoint bundle 和评估结构；主要缺口是这些函数没有在真实入口中完整连接。

本轮重新运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_cli.py tests/test_mts_checkpoint.py \
  tests/test_mts_eval_protocol.py tests/test_mts_metrics.py
```

结果：**41 passed，2.52 秒**。以下反例仍成立，说明 helper 测试通过不等于 CLI 跑通。本轮没有复跑报告所称的全部 478 项，也没有验证训练效果。

| 编号 | 复核事实 | 证据级别 | 下一任务 |
|---|---|---|---|
| F01 | 两份 revision2 YAML 都有 `span_frames`，当前 `MaskConfig.from_mapping` 拒绝该字段 | 最小执行：均报 `Unknown masking options ['span_frames']` | C01 |
| F02 | operator main 在 `sampler = StylePairSampler(...)` 前调用 `sampler.eligible_targets(...)` | 代码路径确认；该路径会读未赋值局部变量 | C01 |
| F03 | manifest-only 的 parser 仍 required 两个 checkpoint 参数，分支前还加载 tokenizer | parser 实测：仅数据参数退出码2 | C01 |
| F04 | evaluator 的 `build_row_batch` 新建不带 visible mask/条件的 batch | 实测：`OperatorBatch.visible_mask is required` | C05 |
| F05 | tokenizer SHA 已写入 payload，但 loader 只验证 representation metadata；store binding helper 没被四个入口调用 | 实测：篡改 tokenizer SHA 后 bundle 仍可加载；调用点搜索确认 | C02 |
| F06 | transport CLI 仍有 `--checkpoint`/“resumed transport”、动态 val、val为空回退train loss、只写store路径 | 代码路径确认 | C04 |
| F07 | validation/eval batch 只用第一行 seed 生成整个 batch mask；style-ID validation 缺 style_ids | 代码路径确认 | C03 |
| F08 | ValidationProtocol 部分类缺失或 NLL 为 NaN 仍返回 usable=True | 实测：缺一类 objective=1、usable=True；NaN objective、usable=True | C03 |
| F09 | eval 把 style label ID 填进 content_condition；generate 没有 style-label 分支 | 代码路径确认 | C05 |
| F10 | hard mask 只加入 locality 子路径；physics 丢条件/区域，仍混合聚合；R11 的三种对比、warmup等尚未接入 | 代码路径确认；不只是 plot 尚未实现 | C05,C07 |
| F11 | strength/locality helper 仍把 operator λ=0 当 base，对 arbitrary kernel 不成立；generate base 是 argmax、styled 是采样 | 代码路径确认 | C06 |
| F12 | 新 config 的 tokenizer 是旧非holdout `1h`，token store 却为 `ah_tokens`；operator指向旧schema transport | 配置核对；实际hash由C02/C10预检确认 | C01,C02 |

补充：报告中“旧 NEF checkpoint、所有 stage3_eval 都因 MTS loss 修订而失效”应纠正。MTS loss 修复不自动使 tokenizer 的重建/物理结果失效；旧产物应按**实际实现和协议**逐项分类。

## 2. 执行规则与交付边界

### 2.1 保留当前工作区

当前 HEAD 仍为 `e9aaceb`，大量 R01–R11 修改尚未提交。**不能 `git reset/clean/stash`、切换检出或从旧 HEAD 覆盖文件来“建立干净基线”。** 将当前文件内容视为起点；执行前记录 diff 和 untracked 文件清单。

本次只修 MTS 主链、已有 probe 和所需文档。保留 NEF 40×9/13-stream tokenizer、已有 packed data 和历史 checkpoints，不重训 tokenizer、不重建全量 token store、不改 renderer、不做无关重构。

不新建训练平台、通用配置框架或大范围 registry。优先复用 `checkpoint.py`、`windows.py`、`eval_protocol.py` 和现有测试文件。允许一个小型 `preflight_mts_revision2.py`，不另建复杂 orchestration 系统。

### 2.2 每个任务的工作方式

每次只执行一个 Cxx；若修改超过约6个实现文件，按文内 a/b 子步骤分开完成。先补独立失败用例，再修实现；定向验证通过后继续下一个任务，无需逐步询问许可。

验收须同时写明：实现文件、实际命令、退出码、passed/skipped/failed、证据路径。不得用“代码里有这个 helper”“parser 接受该字段”“测试字符串不含 resumed”替代实际入口验证。

进度追加到原 `MTS_FSQ_Code_Revision_Progress_zh.md` 的新章节“Integration Closure”，不要删除历史记录。每项使用：

```text
任务：Cxx
状态：pending / in_progress / passed / blocked
原计划映射：Rxx
修复事实：<文件、触发条件、现在行为>
反例：<修改前失败，修改后通过；不是宽松替代测试>
验证：<命令、exit code、数量、输出位置>
仍未验证：<真实训练效果 / CUDA等>
下一任务：Cyy
```

### 2.3 资源与输出

- CPU unit tests、临时 tiny store、每个模型≤2步的 CLI smoke 可直接执行；限制线程数1，单 subprocess timeout 120秒。
- 真实数据只读预检最多8个窗口；不默认运行学习或全量评估。
- 新产物使用 `outputs/mts_revision2_closure/<run_id>/`；不同命令不能覆盖同一实验目录。测试写 `tmp_path`。
- GPU训练、上千步过拟合、正式sweep、evaluator训练均需另给预算。此计划默认只输出命令和预算建议，不自动启动。
- 旧 schema1 拒绝规则保持；当前 schema2 缺必需身份字段也要拒绝，不能补默认值假装可信。
- architecture revision 2 保留。checkpoint字段补齐不改变 tensor架构；评估产物增加明确的 `protocol_id=mts_r2_closure_v1` 与 manifest content hash，避免混用此次补缺前的 v2 文件。

### 2.4 新验收门槛

| 门槛 | 要求 | 不代表什么 |
|---|---|---|
| G-entry | C01–C05完成，真实tiny CLI、数据绑定与固定协议可执行 | 不代表生成质量好 |
| G-code | 再完成C06–C09、所有相关反例通过 | 不代表跨内容style已有效 |
| G-ready | C10–C11完成，真实小数据预检与训练申请包齐全 | 不代表正式训练获授权或论文结论成立 |

## 3. 任务顺序与依赖

| ID | 内容 | 依赖 | 对应原任务 |
|---|---|---|---|
| C00 | 记录起点与建立入口反例 | 无 | R00/R14 |
| C01 | 修配置、构建顺序与CLI前置条件 | C00 | R05/R09/R10 |
| C02 | 真正执行权重/数据/词表绑定 | C01 | R08 |
| C03 | 逐样本固定协议、style-ID validation与完整objective | C02 | R09/R10 |
| C04 | transport训练入口与数据读取补齐 | C03 | R05/R09 |
| C05 | eval/generate完整条件与style-ID接入 | C03,C04 | R10/R11 |
| C06 | 正确base对照及多步monotonic filling | C05 | R03/R12 |
| C07 | 接通物理评估、逐样本输出与plot版本检查 | C06 | R11 |
| C08 | geometry/temporal probe协议修订 | C00，顺序仍放此处 | R13 |
| C09 | 真CLI集成矩阵与回归收口 | C01–C08 | R14 |
| C10 | 真实数据只读预检、有效pair与曝光审计 | C09 | R14 |
| C11 | 状态文档、实验配置与申请包 | C10 | R14/后续实验 |

本轮先完成 C00–C11；trajectory、FiLM、多风格组合、exact resume、AMP优化不混入补缺任务。

## 4. 逐任务实施

### C00：当前文件基线与失败的真实入口测试

**文件：** 进度报告；`tests/test_mts_cli.py`、必要的公共fixture。

1. 保存当前 `git status --short`、`git diff --stat`，记录 untracked 文件，不清理工作区。
2. 运行本文件§1的41项测试基线；记录新结果，不复制旧数量。
3. 给两份revision2 YAML加“实际调用 MaskConfig.from_mapping”的测试，不能只断言YAML中有某字段。
4. manifest-only 用 subprocess 调实际脚本：只有临时store与output，不提供任何模型checkpoint。
5. 构建tiny feature/tokenizer/transport fixture，进入 operator `main()`，不是只测 `resolve_operator_config`。先捕获实际构建错误，后续C01修正。
6. checkpoint测试增加“同结构不同文件SHA”的场景；不能仅改变receptive_field冒充权重绑定测试。

**验收：** 反例失败原因指向F01–F05；没有误触发生产数据/大训练。未修完的测试记录pending，不把它们永久xfail后宣称完成。

### C01：配置能用，初始化顺序正确

**文件：** 两个revision2 YAML、`train_mts_operator.py`、`evaluate_mts_operator.py`；必要的 `train_mts_transport.py` parser小改。拆C01a配置/C01b入口。

**C01a配置：**

- 去掉不支持的 `span_frames`，改用已有 `span_ratio`。若目标是64帧中16帧，填0.25并在配置注释说明；不得添加另一个无效别名掩盖。
- tokenizer主配置指向 `outputs/nef_fsq_soma_packed_40x9_ah/best.pt`，随后C02以真实SHA确认匹配 `seed_soma_pruned_v4_ah_tokens`。
- operator配置指向计划中的新revision2 transport输出，不再引用旧 `outputs/mts_transport/seed_ah/best.pt`。尚未训练时应清晰报缺依赖，不回退旧模型。
- 两个配置的content主路径明确为 `action_id`；允许独立 unconditional smoke 配方，但不能拿none主张指定content。
- 长训练预算仍不自动执行。所有路径、frames、loader metadata、mask比例在dry-run中输出。

**C01b入口：**

1. operator按顺序执行：解析配置→身份预检→读取records→构建sampler→获取eligible集合→确定词表/映射→构建batch source→构建trainer。消除 `sampler` 先用后定义。
2. operator继承transport action map，不要求operator子集词表与broad transport词表完全相等；只要求每个使用action都可在冻结map中查到。不能重新编号。
3. style-ID map从**允许训练且能形成合法pair**的style构建，排除heldout/test-only/无训练pair的style；验证 `num_styles` 与map匹配，不用全store词表填未训练embedding。
4. manifest-only的checkpoint参数改为条件必需；在任何模型load之前分支。只从store/catalog身份和窗口元数据构建，不编码motion、不加载tokenizer。缺必要身份时明确指出，不能编造。
5. 自动构造manifest时 `args.support=None` 不得执行 `list(None)`；空support的语义统一为whole-body，不混用空列表表示“编辑为空”。

**验收：** 两份配置通过真实解析/构建；manifest-only无checkpoint退出0；无合法pair/未知action给明确错误；operator tiny main 能走到trainer构建，不出现UnboundLocalError。

### C02：从“写下SHA”变成“加载必校验SHA”

**文件：** `checkpoint.py`及四个CLI；测试 `test_mts_checkpoint.py/test_mts_cli.py`。先公共校验，再逐入口接入。

**固定接口决策：** 保留representation metadata参数，新增必须的 `tokenizer_checkpoint` 路径或明确 `tokenizer_checkpoint_sha256` 参数，二者取一。不要往representation metadata随便塞一个不会进入fingerprint的字段。

1. bundle/transport load将传入文件SHA与payload `tokenizer_checkpoint_sha256` 比较；缺任一必需值时报错误。不只是“如果两边恰好有值才比”。
2. token store验证：manifest `checkpoint_sha256` == 实际tokenizer SHA == MTS记录；并验证layout/schema/norm/split身份。train/eval/generate/warm-start均接同一helper。
3. feature store不应被强制要求拥有tokenizer `representation_id` 或checkpoint hash；分两类要求：tokenstore需tokenizer身份，featurestore需feature/skeleton/norm/split与model-space转换一致。不要把过严的无关字段校验变成新阻塞。
4. train transport provenance写真实store hashes，不只是路径；operator和transport的上游数据一致性在训练开始前验证。
5. warm-start必须检查词表顺序、style map、tokenizer身份与协议，不只shape。相同参数shape但不同ID含义应拒绝。
6. 记录operator实际允许训练集合与heldouts；外部 `--held-out-styles` 不能自行覆盖训练事实。缺曝光记录标unknown。
7. 实验code identity：当前dirty状态只写HEAD+dirty不足以区分两轮代码。增加涉及源码/config文件的稳定digest清单（包括未跟踪模块），不要求用户先commit，不收集无关/敏感文件。
8. 推理继续从operator内部transport state重建，不恢复外部路径依赖。提供外部transport参数时，缺上游hash不得静默放行。

**验收：** 同结构两份tokenizer文件不同SHA被拒；错误store的norm/split被拒；四个CLI均在前向前失败；真实匹配绑定通过；feature/tokenstore两条合法路径都通过；缺metadata的旧schema2得到清晰错误而非默认放行。

### C03：协议以sample为单位，不以batch为单位

**文件：** `eval_protocol.py`、`windows.py`、operator验证接线；测试 `test_mts_eval_protocol.py/test_mts_cli.py`。

**核心不变量：** 同一manifest行的tokens/mask/condition/候选和随机数，在batch_size、行分组、设备改变后仍代表同一个实验。

1. 每行用自己的seed调用mask generator，B=1；再stack。不得用首行seed一次生成整个batch，也不得根据新batch size重新派生seed。
2. manifest保存完整mask config（ratio、block等），不能只存kind后用默认参数；保存frames、窗口起点和source身份。`sample_id`稳定且唯一。
3. ValidationBatchBuilder接style map/encoder kind；style-ID batch按sample.style生成style_ids，reference模式才要求reference。valid_mask从WindowSample读取。
4. build-only与模型评估使用同一行schema；读取manifest后对identity/protocol/hash做验证。CLI与manifest冲突时明确报错，不悄悄覆盖frames/region/mask。
5. `ValidationProtocol.objective`：正权重kind全部存在、supervised_count>0且每项finite才 `usable=True`；否则objective=null，记录missing/invalid kinds。权重必须finite非负、至少一个正值，并规范化或强制sum=1（本轮选择构建时normalize一次并固化）。
6. 不能让缺失某类降低objective从而更容易选best；缺正权重类必须使本次best比较无效。零权重类可作为诊断缺失，但显式列出。
7. 拒绝无效protocol写入NaN/Infinity；空训练集不允许 `train_records or records` 将test混入词表。
8. `_row`按字段语义切batch：`hard_mask[T,K]`不得因T恰等于B而被当成batch切片，metadata list也需同步。优先用`dataclasses.replace`复用完整batch。

**验收：** 同manifest按B=1/2/4构建，逐sample mask完全相等；显式shuffle行后按ID还原相等；有style-ID的验证完整前向；缺一类/NaN/Inf/零计数都不能更新best；action/style ID无混淆。

### C04：补齐transport训练入口，消除双套数据行为

**文件：** `train_mts_transport.py`、`training.py`、`windows.py`、`eval_protocol.py`；必要的operator callback小改。

当前脚本还有一个本地 `TokenSource`，不是已修订的共用window reader。不要只修共用reader后就宣布transport在线路径已修好。

**C04a 数据与配置：**

- loader adapter输出固定mapping：tokens、valid_mask、content_condition、sample_metadata，不能在unconditional时退化为tensor并丢mask。
- feature在线编码使用正确history/model-space，截取 `history:history+T`；与匹配tokenstore同窗口做数值比对。复用已有reader/normalization helper，允许保留薄batch adapter，不复制encoder语义。
- `freeze(clips=N)`精确截断N个窗口与所有关联字段，避免overfit8实际冻结128个。
- transport也验证fp32、epoch-end validation、未知参数；添加真正`--dry-run`，先解析配置，再身份和构建预检，不step optimizer。
- CLI override后的training段写回resolved config。导出的运行配置必须就是实际执行配置。

**C04b 验证与恢复：**

1. transport固定val窗口和逐行mask，但**不强制same-style pair**，base模型不应被operator数据配对限制缩小验证集。复用协议机制，target-only record允许reference=None。
2. 正式validation只执行一次：当前operator fit验证后on_epoch_end再次evaluate，需改为一次返回结构化report供日志/checkpoint/best共用。
3. last/best payload记录真正用于选best的 `val_objective/per_kind/counts/protocol_hash`。不要summary一个值、checkpoint metrics另一个值。
4. validation缺失/空/非finite：明确失败或best不更新，绝不回退train loss。synthetic overfit模式若只看train，单独写 `overfit_last.pt`，标明非validation-best。
5. transport迁移为`--warm-start`，reset optimizer/step/best；原`--checkpoint`给迁移提示。不实现exact resume，也不保留“resumed”误导文案。
6. dry-run/validation audit的RNG不推进train采样流；freeze模块的train/eval状态由模型/训练入口统一保证。

**验收：** 真正transport subprocess跑两步、保存并加载；两次fixed val一致；参数CLI override进入artifact；warm-start fresh state可观测；空val不能产生best；CPU tiny在线/缓存token路径一致。

### C05：完整batch进入所有eval/generate子路径

**文件：** evaluate/generate脚本、`metrics.py`、`eval_protocol.py`。不再建立脚本专用的不完整batch重建函数。

1. 用一个共用row→OperatorBatch路径，字段齐全：target/ref、target/ref valid、visible、hard mask、anchor、action condition、style_ids、strength、metadata。
2. retrieval只替换reference，保留原target row的一切条件。删除或重写当前 `build_row_batch`；禁止只传target_tokens就重新造batch。
3. style-ID evaluation不进入reference retrieval循环，然后才事后标N/A。直接走正确/错误ID对照；generate支持 `--style-label`，reference模式要求style-clip，style-ID模式不强制style-clip。
4. content_condition永远来自target/source的action；style_ids永远来自style map。更换style label不能改action。所有tensor统一搬到device。
5. evaluator默认从checkpoint读取content kind；显式CLI指定不匹配才报错。不要求用户每次补一个容易错的 `--content-kind`。
6. regions/radius/frame_range来自manifest，构建batch时即生效。likelihood、reference sensitivity、strength、locality、physics共用同条件，不能某项偷偷评whole-body。
7. 单行缺wrong/random/correct时按metric处理：缺wrong不影响该行base/correct；缺correct则该reference指标unavailable，不中断整个run；候选或某一行缺失不能导致整个batch的其他行指标消失。
8. 正常eval路径透传dataset标签来源；不得build-only用修正100STYLE标签，正式eval却遗漏dataset参数。
9. 逐sample row保存实际数值与计数，不只保存描述字段。NLL累计sum/count，retrieval累计hits/count；其他均值明确样本平均还是帧加权，不能对batch mean再无权平均。

**验收：** reference/style-ID的真正CLI都跑完；B=1与B=4逐行NLL、mask、labels一致；相同source换style只改变style字段；local/temporal设置作用到每个指标；缺某个候选只影响对应metric；没有异常后写“成功空表”。

### C06：真实base对照与R12多步生成

**文件：** `sampling.py/model.py/transport.py/metrics.py`、generate/eval接线。拆为C06a base、C06b filling、C06c CLI。

**C06a base：**

- `strength_response`使用`result.base_probabilities`，不使用operator λ=0 的probabilities。arbitrary default λ=0为uniform的设计保留。
- base/styled在同一单步条件下都用同一uniforms采样；另保留base_argmax仅作为明确命名的诊断。不把argmax→sample差异叫style effect。
- locality同时报告分布区域外差异和最终token锁定；仅最终where锁住不证明operator分布遵守mask。
- 统计掩码区分hard support与effective_edit，区域内可见anchors不应稀释active edit response；valid=False不入任何分母。

**C06b monotonic filling：**

```text
remaining = 初始 hard_mask & ~visible & valid
每个step:
  用当前tokens和visible计算base/styled probabilities
  对每个sample，从remaining按(time,coordinate)固定顺序选择
    ceil(remaining_count / remaining_steps) 个位置
  仅选中位置写入drawn，标为visible，从remaining移除
完成时 remaining=empty；初始visible/anchor/support外/padding逐位不变
```

- 复用已有CRN，key包含stable sample_id、draw_id、step_id、[T,K]；不要让整个batch shape/B进入每行随机数身份。不同draw独立，同draw跨条件共享。
- base和styled各自递归更新自己的tokens；step相同、uniforms相同，context允许随各自先前采样分叉。这个测量叫algorithm-level effect，不是固定Q半群比较。
- teacher-forced单步NLL与iterative motion metrics分开标记，不能混成一个loss。
- 共用选择/提交helper，transport和operator不各维护不同schedule。steps=1应等价已修单步，steps大于editable数合法。
- 观察mask与写入mask分开：support外锁定不自动意味着原先hidden值可以被看到；使用caller明确visible，不能偷偷以`~effective_edit`替代。CLI局部编辑可显式设visible=~support。
- 首轮temperature固定1；若保留参数则明确同一规则作用于base/styled，并测试λ=0的identity语义，不为各方法偷用不同温度。

**C06c CLI：** 增加`--steps`，记录schedule、sample/draw/step seeds，保存source/base_sample/styled与masks。`--samples`表示独立draw数，不能共享同一key得到重复样本。

**验收：** 提交trace显示每个editable位置恰一次；hidden token数值改动不提前泄漏；B分组变化结果相同；step=1一致；固定输入/seed/模型重复结果相同；arbitrary λ=0与真正base能被测试区分。

### C07：R11从函数实现到真正的物理测量与出图

**文件：** `metrics.py`、evaluate脚本、`plot_mts_figures.py`；必要的window decode helper放`windows.py`。

1. 从同一生成协议获得source reconstruction、base sample、styled sample。分别输出source→base、base→styled、source→styled三种对比，使用不同 `comparison` 字段。
2. 解码带source对应的左历史prefix，再裁掉warmup；prefix/frame offset/真实边界处理记录在row。若起点无完整历史，明确left-padding约定或排除暖启动帧，不能无声把窗口第0帧当稳态。
3. `physics_metrics`继承当前完整batch的action、style、mask、valid，不能创建丢字段的styled_batch。
4. off-target FK使用layout定义的ownership与descendant influence；radius0/1分别测；root world-space累积影响单独报告。不能仅报告全身平均FK叫局部泄漏。
5. 已有解析jerk测试和m/s³公式保留；补start/stop边界邻域、短序列unavailable、常速度零、局部异常不污染另一个边界的测试。
6. 接触gate来源、dt、单位、有效接触数随row保存；无接触返回null+reason。加平均速度/运动幅度等退化诊断，不以脚静止的低slide为质量提升。
7. 每行数值至少关联sample_id、draw_id、checkpoint hash、protocol hash、steps、strength、region/radius/frame span、comparison、actor/take。
8. summary按上述实验条件分组，绝不能跨strength/region/comparison平均。数值不够只报null+count。
9. 输出写入前校验finite；CSV与JSON实际row数一致。metadata中的batches不替代有效sample count。
10. plot读取时检查metrics_version、protocol_id/hash、units、采样设置。允许同protocol下不同checkpoint的受控比较；拒绝旧结果/不同protocol直接合成均值，错误必须给出文件名与冲突字段。

**验收：** λ0/1输出各自group；三种comparison齐全；identity fixture的base→styled为0；物理CLI有非空有效row；旧schema图表输入明确拒绝；同协议不同模型可绘图。

### C08：完成R13，不能只调整float32容差

**文件：** `nef_probe.py`、probe/locality脚本、`test_nef_probe.py`。

- 按上一计划R13实现single pulse与fixed signed span，near/far共享相同source/coordinate/time support与合法位置交集；按±1、±d分层。
- 原逐帧随机far perturbation仅作独立stress test，不能用它的jerk解释ordinal geometry。
- 根据实际decoder RF选择完整后尾窗口，例如128帧编辑第32帧；若边界不足，`temporal_probe_complete=false`，不宣称影响止于窗口末尾。
- 报token→feature影响与root积分后的world-space tail，二者不能共用相同RF上界。
- 对报告中的旧 `abs=0.0` FK失败先测量确定数值底噪，结构应为精确0的feature层仍用严格检查，world-space float32 FK才用明确量纲/容差，不能全局放宽所有断言。
- 新probe artifact记录protocol revision、支持/排除计数、RF、frame range、单位和随机种子；不覆写历史geometry结果。

**验收：** 已知toy decoder RF完整测出；truncated case正确标记；合法mask一致；随机stress与geometry统计分开；旧容差修复有解释且独立正确性测试保留。

### C09：真正的CLI矩阵，补上此前漏测的集成层

**文件：** `tests/test_mts_cli.py`、`test_mts_checkpoint.py`、必要的公共fixture。优先复用，不再为每个任务建一个大测试包。

必须通过 subprocess 或真实 `main(argv)` 测实际训练/加载/评估/生成，不能仅import helper、搜索脚本文本或mock整条模型加载链。

| Case | 数据与模型 | 必须执行 |
|---|---|---|
| A | tiny feature store、真实tokenizer fixture | build-manifest-only，无MTS checkpoint |
| B | 匹配packed token fixture、action condition | transport 2步→固定val→保存→重新加载 |
| C1–C6 | 3 operator × reference/style-ID | operator 2步→val→save/load→eval→generate |
| D | 一个reference分支，feature在线路径 | 与tokenstore相同窗口对齐；history/valid不丢 |
| E | shuffled CTMC | 保存加载后level_order/Q/概率一致 |
| F | 错hash/空val/非法style/未知配置 | 非零退出且错误指向原因，无best/伪成功metrics |

小模型D=16或32、depth1、T=8或16、B=1或2；参考与目标是不同clip/content。词表故意让action/style ID数值不同，防止错误字段混用碰巧通过。

至少一个fixture的窗口起点非0、padding非全valid、hard mask有anchor；至少一次以不同batch_size重放同manifest。多方法共用同一固定manifest，不让每个分支重新随机造“对照”。

GPU guard仅覆盖CPU generator与CUDA tensor的搬运和style_id device；无CUDA写skip，不能把CPU全绿等同CUDA已验证。

相关集合运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_*.py tests/test_nef_token_contract.py tests/test_nef_probe.py \
  tests/test_nef_fsq.py tests/test_packed_downstream.py
```

完成后可运行一次全量测试；必须记录精确命令（包括deselect）。旧100STYLE外部数据布局问题可单独标明，不能叫“全量只失败1个”同时隐藏另一个被排除的测试。

**验收：** 6组矩阵都有工件而非空summary；G-entry/G-code状态由实际结果决定；任何分支因缺字段失败必须修复再标通过。

### C10：真实数据预检与实验有效性审计

**文件：** 允许新增 `scripts/preflight_mts_revision2.py`；复用checkpoint/pairs/window/protocol功能，不再复制校验逻辑。

预检分两级，解决“先有新transport才能知道数据能不能用”的循环：

- `data`级：只需要tokenizer、feature/tokenstore、目标配置。检查SHA、norm、schema、split与标签，读取最多8个确定性窗口。
- `experiment`级：另需新revision2 transport/operator（如该阶段需要）；检查上游实际绑定、action/style映射、协议、输出目录与训练预算。

输出 `preflight.json`，每项有status/reason/evidence，不仅返回一个True：

1. tokenizer/store是否可复用，错误路径是否已纠正；不得因为文件都存在就判定匹配。
2. train/val/test按actor/take/mirror family隔离，真实actor字段缺失时exposure_unknown；tokenstore未携actor字段时通过source身份对齐metadata，而不是凭行号拼接不同store。
3. style×action可配对矩阵、每style有效targets/references、窗口与拒绝原因；词表存在但无合法训练pair不能算训练覆盖。
4. operator action集合可以是transport vocabulary子集，但未知action必须列出来；动作与style标签不能互当。
5. 已知SEED单content style不能承担same-style/different-content训练；不要悄悄放宽成different clip。
6. heldout style的operator曝光与上游tokenizer/transport曝光分别报告；actor-unseen需所有学习组件和normalization均尊重划分。
7. evaluation manifest中same take/overlap crop/镜像泄漏检查，合法负例不足单独计数。
8. artifact hashes、协议hash、code digest、step/B/T/seed、可学习参数量、mask期望比例都可导出。

预检不能修改全量store，不自动生成新token数据。没有训练好的revision2模型时，只完成data级并明确experiment级待训练，不伪造G-ready的模型性能部分。

**验收：** 真实holdout tokenizer/tokenstore的匹配组合通过data级；旧1h tokenizer+ah tokenstore失败；读窗口上限受测；预检不会创建best或启动训练。

### C11：修正进度叙述并交付下一轮实验申请包

**文件：** 进度报告、README、`mts_operator_stage_status.md`；必要的历史landing/actor holdout勘误；一个小型实验manifest（JSON或YAML）。

1. 将R05/R08/R09/R10/R11历史“passed”保留为当时记录，顶部补充此次复核与Cxx收口证据，不抹除历史问题。
2. 更正“R11除plot均完成”、transport resume已消除、固定协议已batch无关、SHA已绑定等不符合当前历史实现的描述。
3. 对旧工件按类别处理：旧MTS算子/transport不可直接作为新主模型；NEF tokenizer和既有representation physics结果仍按原协议保留，不因MTS的NLL变化笼统作废。
4. 历史配置文件在上一轮已被修改，不能继续声称“历史配方未动”。列出实际差异，必要时新增带明确标签的历史快照文档；不直接checkout旧文件覆盖用户改动。
5. 新实验manifest逐项描述family/encoder/data/protocol/hash/seed/步数/output，不依赖 `/tmp/*.sh`，只生成命令不默认执行。
6. 为正式训练准备参数量和吞吐测量方法；不要沿用旧“3分钟一模型”的估计，架构和有效验证量已变化。

**交付状态模板：**

```text
G-entry：通过/未通过，依据...
G-code：通过/未通过，依据...
G-ready：通过/未通过；data级与experiment级分别说明...
修改文件与主要行为：...
定向/全量/CUDA测试：实际命令和结果...
旧结果仍可使用的范围：...
真实风格效果：未验证/具体证据...
待批准训练：stage、seed、steps、B、T、预计资源测量方法...
```

## 5. G-ready之后的研究小试顺序（不自动执行）

不是直接恢复26-cell矩阵。首先验证最便宜、最可能否证方法的对照：

| Stage | 输入条件 | 工作 | 退出条件 |
|---|---|---|---|
| P0 小数据base | 已核验的1–8 clip、真实action条件 | 短过拟合transport；检查infill/full-generation、速度/contact、保存动画 | 能学到数据且无输入泄漏；不能只看loss下降 |
| P1 算子信号 | 同一个冻结base、相同pair/valid协议 | style-ID logit vs constant-descriptor/no-reference adapter | style条件比同预算无style修补更有效 |
| P2 kernel对照 | P1有信号 | CTMC vs logit vs full kernel vs shuffled，固定同样encoder与预算 | ordinal结构在效果/物理/样本效率至少一项有支持 |
| P3 reference | P2或简单logit可用 | 正确/真实错误/随机真实reference，balanced样本 | 换reference改变style，content与质量可接受 |
| P4主实验 | P3通过 | FiLM、part/NEF、组合split、≥3 seeds、独立evaluator | 置信区间与匹配风格强度的公平对照 |

P1的constant descriptor需要一个明确的最小实现任务：保留同一operator宽度与训练token预算，descriptor与reference/ID独立；记录实际参数差，不能复制零向量但仍偷偷使用reference encoder输出。它是研究对照，不是本轮入口补缺的必做项。

action ID只代表动作类别，不代表完整source轨迹、接触节奏或unseen action泛化。若P0/P1主要失败在内容保持，再执行上一计划E01 trajectory条件；不要先同时加入trajectory、time warp、contrastive loss，让失败无法归因。

用户批准训练前，需要提供具体seed、steps、B、T、输出目录、保存频率、验证预算和观察指标。初始执行预算用短profile估吞吐后制定，不编造总GPU小时。

停止规则保留：无style adapter同样有效则先查base补偿；CTMC不优则收缩贡献；数据无跨content证据则改数据/任务，不靠复杂loss补造证据。

## 6. 给执行agent的可复制提示词

```text
请执行 docs/MTS_FSQ_Agent_Integration_Closure_Plan_zh.md。

这不是从上一轮R00重做。先读本文件§1–3，以及
docs/MTS_FSQ_Code_Revision_Progress_zh.md的当前状态。
现有R01–R11改动大多未提交，必须保留；禁止git reset/clean/stash或覆盖旧工件。

按C00→C11串行执行，一次一个任务。优先复用已有loss、embedding、temporal、CRN和
checkpoint代码，重点补实际CLI接线、数据身份验证和固定协议的逐样本一致性。
先写独立失败用例，再修实现，定向测试通过后继续；不得仅测helper/脚本文本就标CLI通过。

用原进度报告追加Integration Closure章节，记录精确命令、退出码、测试数量、工件与限制。
允许CPU单元测试、tiny fixture及每模型≤2训练步的CLI smoke；不启动真实长训练、
全矩阵、全store重建或删除旧checkpoint。研究扩展留到§5，当前只输出训练申请包。

完成时分别报告G-entry、G-code、G-ready；模型未训练就明确性能未验证。
遇到新缺陷先记录证据，在本任务范围内修复，不降低测试标准或用默认值绕过hash/词表错误。
```

续接提示词：

```text
继续执行 MTS_FSQ_Agent_Integration_Closure_Plan_zh.md。
核对进度报告Integration Closure章节与当前diff，从首个未完成且依赖满足的Cxx继续。
不要重做已经通过且未受改动影响的任务；不要根据旧passed标签推断真实CLI完成。
```

## 7. 本文件交付边界

本次完成进度报告与关键代码路径复核、41项定向测试和少量独立反例，新增本计划。没有修改实现、重写原进度报告、运行GPU训练或执行C00–C11。上述任务完成情况必须由后续执行产生真实证据。

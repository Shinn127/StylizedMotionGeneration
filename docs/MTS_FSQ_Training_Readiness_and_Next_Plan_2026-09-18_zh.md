# MTS-FSQ 开训审核与下一步执行计划

日期：2026-09-18。代码起点：HEAD `e9aaceb` 加当前工作区全部 revision2 修改；这些修改大部分尚未提交。

依据：[进度报告](MTS_FSQ_Code_Revision_Progress_zh.md)、[集成收口计划](MTS_FSQ_Agent_Integration_Closure_Plan_zh.md)、[已有实验申请](mts_revision2_experiment_request_zh.md)。本文面向执行 agent，补充当前开训前缺口，不要求重做 R00–R14/C00–C11。

## 1. 结论：接近可以开训，但当前主配置还不能直接启动正式实验

**可以继续只读预检和 tiny 工程测试；不建议按现有 experiment manifest 直接跑正式 transport/operator。先完成 T00–T04，再开始有预算的真实数据短跑。**

这一轮有实质进展：真正的 CLI 矩阵、6个算子/encoder组合、逐行协议、多步生成、数值/标签/数据输入空间修复已经落地。不能再把项目描述为只有 helper 或脚手架。当前剩余风险主要集中在主配方、验证划分、transport 工件绑定和实验准入，而不是要重做核心架构。

但“G-code通过”目前最多说明覆盖到的工程路径能运行，不能解释为科研验证路径已经正确。最重要的新发现是 **transport 用训练集窗口生成所谓 validation protocol 并据此选 best**。

| 当前活动 | 审核意见 |
|---|---|
| unit/tiny CLI 测试、数据身份检查 | 可继续 |
| 按当前配方直接训练40 epochs | 暂缓 |
| 依赖旧experiment `ok=true`自动放行 | 不可 |
| 完成T00–T04后做20步profile/2步真实链路 | 可申请有限预算开始 |
| profile后做base模型先导训练 | 需通过本文逐级门槛 |
| 三算子×多seed完整矩阵、论文性能结论 | 目前不具备条件 |

没有新revision2 transport，所以operator暂缺上游是正常依赖，不是算法失败；预检必须区分“准备训练base”与“准备训练operator”，不能要求base训练前就有base checkpoint。

## 2. 本次独立核验范围和证据

重新运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_cli_matrix.py tests/test_mts_preflight.py \
  tests/test_mts_eval_protocol.py tests/test_mts_operators.py
```

**73 passed、1 skipped、1 warning，26.97秒。** 本次没有复跑全部542项；进度报告的全套历史结果与 `outputs/mts_revision2_closure/final_full_suite.txt` 一致：542 passed/1 skipped/1 failed。该失败是100STYLE外部布局相关，非SEED开训主阻塞；当前被跳过的设备验证不能算作已通过。

真实数据只读dry-run：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/train_mts_transport.py \
  --config data/configs/mts_revision2_transport.yaml \
  --dry-run --device cpu --dim 32 --batch-size 2 --val-rows 2 \
  --output outputs/training_readiness_audit_20260918/transport_dryrun
```

退出0，没有训练；输出明确 `content_condition.kind=none`。落盘 `validation_protocol.json` 两行的 split 都是 **train**，clip分别为66466、13839。这不是推测。

另运行相同dry-run加 `--overfit-clips 2`：在打印冻结窗口数时触发 `AttributeError: 'dict' object has no attribute 'shape'`，没有开始训练。输出目录为 `outputs/training_readiness_audit_20260918/overfit_dryrun`。

## 3. 正式开训前必须修复的问题

### B1：transport验证来自train，best不是held-out validation best

`scripts/train_mts_transport.py` 的 `window_source` 使用 `windows_by_clip(store, "train", ...)`，`build_target_only_samples(..., split="train")`。这会把固定训练集监控误标成泛化验证。

同一代码还把 `batches_per_kind=1` 写死；默认只测 full_generation。虽然接受 evaluation 段，`validation_batches_per_kind` 没有真正控制协议规模。

**影响：** loss可以下降、best可以写出，但不能据此判断过拟合或选择最佳泛化checkpoint。现有tiny矩阵只断言protocol存在和finite，没检查行的实际split。

### B2：主链content条件不一致

`mts_revision2_transport.yaml` 当前 `data.content.kind=none`，operator主配置为 `action_id`，实验申请又声称两者都使用action条件。

**影响：** 按主配置训练完transport之后，operator的action-conditioned路径会因上游无动作conditioner而拒绝；不能用smoke的unconditional配方作为解决办法。

### B3：transport checkpoint权重绑定未闭合，预检成功分支还会报错

`load_operator_bundle` 已有tokenizer文件绑定，但 `load_mts_checkpoint` 仍仅接受representation metadata，未接收tokenizer文件/SHA。operator读取上游transport和transport warm-start仍走这个入口。

而 `preflight_mts_revision2.py` 在真的存在transport时，给 `load_mts_checkpoint` 传 `tokenizer_checkpoint=...`。本次用函数签名绑定实测：**unexpected keyword argument 'tokenizer_checkpoint'**。

**影响：** 同结构异权重的上游transport没有全链强制拒绝；之前只测试了“还没有transport→blocked”，没有证明“有正确transport→pass”。

### B4：预检在blocked时仍输出ok=true，且形成循环准入条件

现有experiment预检工件中transport/protocol/exposure/checkpoint_bindings均blocked，但 `ok=true`。代码只把fail计入失败，blocked时仍正常退出。`check_protocol(None, ...)` 又始终检查不存在的协议对象。

**影响：** 自动执行agent可能把exit0/ok=true误当可开训；若一律要求现有operator checkpoint，又会变成“训练operator之前必须已有operator”。

### B5：实验清单不是可直接执行的受控预算

- `steps: 40 epochs x steps_per_epoch` 不是整数预算；两份配置 `max_steps/steps_per_epoch` 都为空。
- 用birth-death配置加 `--operator logit_field/arbitrary_kernel`，会保留CTMC专用的max_rate/max_terms/tolerance，当前入口明确拒绝。已按实际接受键核对。
- 路径有 `<run_id>` 未解析，输出根目录与下游依赖路径不统一。
- 原申请把 `history.seconds` 当完整epoch耗时。实际该计时在validation前结束，summary又只留最后一个epoch；不能据此直接推算含验证和保存的总成本。

### B6：过拟合调试入口和记录仍有尾巴

- transport overfit打印代码仍把mapping当Tensor，dry-run已复现崩溃。
- 冻结batch列表按有限长度迭代，设置steps_per_epoch并不保证能消费那么多优化步；必须检查effective steps。
- transport和operator都在fit中evaluate，callback又做一次protocol evaluate，注释“一次验证”与行为不同。
- operator best按protocol objective选择，但payload仍主要记录普通 `val_nll`，缺与选择一致的objective/per-kind/hash；transport的protocol hash只取 `describe()`，未覆盖完整窗口内容，不能证明相同实验。

这些是局部补缺，不构成再做一轮大架构重构的理由。

## 4. 执行任务 T00–T04：通过后才放行真实学习预算

执行规范：一次一个任务；保留当前所有未提交改动；不 reset/clean/stash、不覆盖旧checkpoint；新增测试先失败再通过。每任务把命令/退出码/证据追加到原进度报告的“Training Readiness”章节。

### T00：修正验证划分和主配方（P0）

**文件：** `train_mts_transport.py`、两份revision2主配置、`test_mts_cli_matrix.py`。

1. 正式window_source取val，protocol行写val；按store实际clip split再次断言，而不是只信字符串。train monitor若保留必须独立命名，不得选best。
2. 实际使用 `validation_batches_per_kind` 和 `validation_kinds`，主配置显式列出5类。固定validation样本总数不随训练batch size变化；推荐配置独立 `validation_rows_per_kind`，旧字段做明确迁移，不能接受后忽略。
3. 首轮正式每类64行，共320行；profile每类2行，共10行，二者protocol_id不同，不比较其best数值。采样都只在val集合内。
4. transport主配方改action_id；operator从同一冻结action map继承，必须子集可用，不重编号。unconditional仅保留在明确命名的debug配置。
5. tiny fixture train/val clip编号和值故意不同；测试逐行验证store split_id，保证train∩val和take-group隔离。空val必须失败，不能回退train。

**验收：** 真实数据dry-run输出action词表非空、所有protocol行来自val、kind/row count与配置一致；错split fixture失败。此任务不启动训练。

### T01：修正transport绑定与按阶段预检（P0）

**文件：** `checkpoint.py`、train transport/operator、`preflight_mts_revision2.py`、checkpoint/preflight测试。先helper后逐入口。

1. 给 `load_mts_checkpoint` 接入实际tokenizer文件或SHA参数，调用现有 `require_tokenizer_checkpoint`。正式入口必需校验；tiny fixture也创建真实小文件绑定，不能靠空metadata跳过。
2. 所有调用点统一迁移，包括operator上游、warm-start和preflight；保持bundle已有绑定不回退。添加同结构不同权重反例，不仅比较receptive_field。
3. transport provenance记录完整store/schema/norm/split/layout身份，不能只写store路径。
4. preflight增加显式阶段：`data / transport_train / operator_train / evaluate`，可保留旧level但必须明确映射。使用阶段所需依赖表：

| 阶段 | 必需依赖 | 不应要求 |
|---|---|---|
| data | tokenizer、store、标签、有限窗口 | MTS checkpoint |
| transport_train | data通过、resolved config、真实val协议、确定预算/新输出路径 | 已训练transport/operator |
| operator_train | data通过、已绑定transport/action map、pairs、val协议、预算 | 待训练operator checkpoint |
| evaluate | 相应MTS checkpoint、data绑定、固定eval manifest | 待执行训练 |

5. 汇总 `ready = all(required checks pass)`；required blocked返回ready=false、非零退出；可选项用not_applicable而不是制造blocked。保留diagnostic ok字段也必须与ready区分。
6. protocol传实际文件/对象，不再无条件 `check_protocol(None)`。dry-run先生成无学习的协议，再预检，消除“先训才能验”的循环。
7. 覆盖正确transport的preflight成功分支、错误SHA失败分支、缺依赖blocked分支；不能只测不存在的文件。

**验收：** 正确tiny transport→operator_train预检退出0；同结构错SHA退出非0；transport_train无需自身checkpoint即可ready；任何required blocked不能输出ready=true。

### T02：修调试预算、验证成本与checkpoint证据（P1）

**文件：** 两个train入口、`training.py`、protocol序列化及相关测试。

1. overfit窗口数从 `item['tokens'].shape[0]` 取；固定窗口与mask/condition是否冻结必须落盘。规定重复读取冻结batch到明确max_steps，不因epoch数据短提前结束。
2. overfit只写 `overfit_last.pt` 和训练诊断，不写validation-best。包括operator的过拟合分支，不能回退train loss后叫best。
3. 一次epoch仅运行一次验证，结果结构同时供日志、history、payload和best使用。添加验证调用计数测试。
4. 两个checkpoint都保存相同的 `val_objective/per_kind/counts/full_protocol_hash`；hash包含canonical完整row内容、mask config、窗口/condition和数据身份，改变一行start必须变hash。
5. 增加 `train_seconds/validation_seconds/checkpoint_seconds/total_seconds`；history保存全epoch序列或JSONL，不只最后一项。有效优化step数和supervised token数同时统计。
6. 所有正式run要求显式正整数max_steps，限定epochs×实际steps覆盖目标；短跑不得悄悄少于预算。独立wall-time上限超时后记录interrupted，不宣称训练完成。
7. 输出目录已存在checkpoint/summary时默认拒绝覆盖；warm-start是新run，不是exact resume，两个step计数/学习率语义必须清楚。

**验收：** dry-run+overfit不崩溃；指定5步实际5步；验证一次；summary/payload objective一致；修改完整protocol影响hash；无val的调试run不产best。

### T03：可执行实验配置，移除无效命令（P1）

**文件：** 实验manifest、必要的独立小YAML、实验申请文档；保持实现无通用配置框架。

1. 分别生成transport profile/pilot、style-ID logit、constant/no-reference logit、style-ID CTMC、reference logit配方。每份只含该算子合法字段；CTMC shuffled/full kernel等后加，不用一份CTMC配置硬切全部算子。
2. run_id提前解析成具体新目录；manifest里的实际transport checkpoint与命令一致。预检记录解析后的完整命令/config SHA，禁止残留 `<run_id>` 或含糊steps字符串。
3. 最小no-reference对照：同一个logit operator、同宽constant learned descriptor，完全不读reference或style label；禁止用真实style-ID再称constant。训练参数量差如实报告。
4. 所有方法共用同一个base checkpoint SHA、相同pairs/validation manifest、mask与采样协议；种子从3407开始，先单seed筛查，不直接三seed铺开。
5. 尚未实现no-reference分支时明确task pending，先完成工程profile；不得以“reference乱序”替代“模型完全不看reference”的训练对照。

**验收：** 每份实际CLI dry-run通过，算子特定字段不报错，命令路径可解析，budget为整数且不互相覆盖。新增no-reference分支有独立“不依赖style输入”测试。

### T04：开训门槛复核和源码冻结（Gate）

1. 复跑本文件73项相关集合并增加T00–T03回归；再跑一次真正tiny action-conditioned transport→style-ID logit→eval/generate，包括preflight正路径。
2. 真实数据只读最多8窗口：确认holdout tokenizer/store绑定、val协议split、20-action map中的实际训练词表、三类有效style覆盖及reference条件。
3. 数据的same-style/different-content证据只对eligible style计数，不能把SEED八类全部写为已训练style。
4. 保存当前源码digest与patch、resolved configs、manifest、环境版本；可以由用户选择commit，但“不提交”不是阻塞，只要工件准确绑定dirty源码。
5. 进度报告增加开训门槛：`train_val_disjoint`、`content_chain_consistent`、`transport_sha_bound`、`preflight_required_pass`、`budget_explicit`、`tiny_real_chain_pass`。六项全pass才进入下一节。

**验收：** 交付 `training_gate.json`（每项status/evidence/failed原因），所有required项通过；没有训练效果声明。

## 5. 门槛通过后的分阶段训练计划

以下是建议的优化步预算，不是本次已授权或已执行的训练；GPU时长应在S0实测。超出预算或失败时停止相关stage，不能自动扩大sweep。

### S0：真实数据profile与最短成功链路

- transport：seed3407、fp32、T=64、初始B=8、最多20步，epochs=1、steps_per_epoch=20、max_steps=20。
- 先保留计划中的D256/depth8用于测真实架构吞吐；OOM时只降B，记录新配方。不默默缩模型后套用原预算结论。
- profile val每kind2行；它用于健全性，不用于模型优劣判断。
- 用该临时transport完成style-ID logit **2步**、正确/错误ID评估、steps=1/4生成。此模型只验链路，不用于后续科研对照。
- 记录初始化、稳定训练step耗时、验证耗时、保存耗时和峰值显存。前几步warmup与后续步骤分别记录；不能拿最后epoch.seconds代表总成本。

**通过条件：** loss/grad/probability finite，三方SHA和action map一致，global_step正确，真实val读取无误，所有文件可回读，日志没有静默skipped条件。动画可退化，但必须明确它是20步随机附近模型，不据此否定方法。

### S1：可诊断的小数据拟合

- 1–8个固定clip/window，seed3407，B≤8，最多200步；固定可观察上下文的infill任务，记录样本和mask。
- 看CE/accuracy相对初始是否明确改善，以及token→motion链路是否合理。若完全不动，先查梯度/输入/条件，不直接加训练时长。
- **不要要求全mask+相同action的多个不同动作片段全部精确重建。** 这时条件相同、目标多模态，低到零的逐token损失不是合理验收。全mask另测分布与采样质量。
- 先用单clip或可区分的context证明优化链路，再扩到8clip；overfit结果不作泛化benchmark。

**通过条件：** 可识别条件的fixture/真实infill明显可学；未观察token没有泄漏；解码没有数值崩溃。必要时固定mask分层定位，而非一次更换架构、数据和loss。

### S2：base transport先导训练

- 从新初始化开始，seed3407、fp32、T64，B根据S0实测选择8/16/32；所有后续方法共享该base。
- **首段2,000步**，每200步固定val，预算为10个epoch×200步；如果epoch实际供给不足，T02应提前拒绝或精确循环。
- 正式val每kind64行，共320行，冻结不变；保存last与val-best。至少在step0、200、1000、2000记录可比较指标/生成样本。
- per-kind NLL/accuracy、实际隐藏比例、supervised tokens、token entropy、root/脚步统计、固定动作条件下8组动画。
- 增加便宜数据基线：从train统计的per-coordinate（可按action分层）unigram NLL；infill另有copy-visible/邻近帧基线时注明适用条件。不能只与log(9)比较，因为token边际不均衡。

**继续条件：** 固定val相对数据基线和初始模型改善、至少full/infill两类行为可解释、无运动静止或高频抖动的明显全面退化。若有正面趋势但未收敛，再单独批准扩展到10k/20k步；不预先承诺2000步足够。

exact resume当前未实现。若追加预算，必须明确是新run warm-start（optimizer重置）还是先实现exact resume；不能把两者拼成同一连续训练曲线。

### S3：先证明style条件真的有用

- 冻结S2选出的同一个base；使用真实eligible styles，SEED预期主要是neutral/injured leg/injured torso，实际以最新pair审计为准。
- 先跑 **style-ID logit vs constant/no-reference logit**，每臂上限1,000步，seed3407，B建议16（以显存实测为准），每200步验证。
- 两臂同pair列表、相同mask schedule/有效token预算、相同action map；保证对照差别主要是style输入。
- 固定source/action，切换正确/错误style-ID；记录目标token似然差、分布响应、动作幅度、off-target FK/contact。NLL仅辅助，不能代替独立style识别。
- 选16个固定可解释case做动画并排：source/base/style-A/style-B，记录失败例而不只展示最好结果。

**继续条件：** style-ID臂在balanced per-style结果中有可重复信号，且不是两臂同样改善base。若无信号，先分析base弱、neutral支配、style-content混淆，不急着上CTMC/reference encoder。

### S4：CTMC与reference分开验证

通过S3后：

1. style-ID CTMC与logit同预算比较；若CTMC有候选优势，再加入shuffled adjacency/full kernel。匹配参数量与训练量，报告实际差异。
2. reference logit先做正确/真实错误/随机真实reference；保持content、source、mask和采样随机数不变。
3. reference明确有效后再跑reference CTMC，不同时把encoder问题和kernel问题混在一起。
4. 任何strength曲线以真正base为基准；arbitrary λ0不是base，不能重新引入旧指标错误。

**退出/收缩：** CTMC不优则保留logit主线；shuffled不劣则不主张ordinal先验优势；reference不敏感则回到配对/encoder，暂缓大矩阵。

### S5：正式主实验（本轮不启动）

再规划≥3 seeds、跨style-content组合holdout、part/NEF同效果强度对照、独立style/content evaluator、长序列与接触失败集。统计按take/actor聚类，不能把镜像/重叠窗口当独立样本。

action_id只能支持已知动作类别内的组合/内容保持检验，不能直接宣称unseen action泛化。100STYLE主实验前须修好本机布局并核对官方action语义；SEED实验无需被该外部布局问题全局阻塞。

## 6. 执行层启动提示词

```text
请执行 docs/MTS_FSQ_Training_Readiness_and_Next_Plan_2026-09-18_zh.md。
先只做T00–T04，保留所有现有revision2工作区改动，不重做C00–C11，不reset/clean/stash。
最高优先级：transport val不得取train；transport/operator必须使用同一真实action条件；
transport checkpoint必须校验tokenizer SHA；预检required blocked不能返回ready=true。

一次一个任务，补真实入口/协议语义反例后修复，运行定向测试并记录精确结果。
不要以protocol文件存在、loss finite、helper通过替代split隔离/绑定/预算验收。
修复overfit入口与明确step计数，统一checkpoint的val objective和完整protocol hash，
为各算子生成合法独立配方，不再用CTMC配置直接切logit/full kernel。

交付training_gate.json、实际配置/命令、进度更新和建议GPU预算。
未经新的明确训练授权，不启动S0–S5真实学习；允许既有tiny测试与只读dry-run。
所有研究效果保持“未验证”，直至相应实验产生可信证据。
```

## 7. 本次实际改动与未完成项

本次只新增本审核/计划文档，并运行定向测试、真实数据dry-run和少量接口反例。没有修复上述实现、没有启动真实GPU训练、没有删除或覆盖旧工件。是否正式开训，应以T04的新门槛结果为准，而不是历史完成标签。

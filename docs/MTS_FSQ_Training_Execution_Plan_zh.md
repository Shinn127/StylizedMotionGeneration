# MTS-FSQ 分阶段训练执行计划

版本：v1.0；日期：2026-09-18。

适用执行层：zcode + deepseek-v4.1-flash，单 agent 串行执行。

依据：

- [最新代码进度](MTS_FSQ_Code_Revision_Progress_zh.md)
- [训练规划准入复核](MTS_FSQ_Training_Planning_Review_2026-09-18_zh.md)
- [开训门槛与前序计划](MTS_FSQ_Training_Readiness_and_Next_Plan_2026-09-18_zh.md)
- [实验申请说明](mts_revision2_experiment_request_zh.md)
- `outputs/training_readiness_audit_20260918/training_gate.json`

## 1. 目标、授权边界与最终交付

当前代码已达到“可以设计并准备受控训练”的门槛。本计划将训练拆为：

```text
E00 冻结实际运行身份
  → E01 20-step profile
  → E02 小数据可学习性诊断
  → E03 2,000-step base transport pilot
  → E04 style-ID logit vs constant/no-reference
  → E05 style-ID CTMC
  → E06 reference logit
  → E07 是否进入正式主实验的决策
```

执行 agent 必须先完成 E00。**本文件本身不授权昂贵训练。** E01–E06 中任何真实 GPU 优化步骤，都只有在用户明确批准该阶段预算后才能运行。未获批准时，执行 agent 只能完成 dry-run、preflight、只读检查、配置生成和命令准备，然后停在对应的 `awaiting_budget` 状态。

允许直接执行：CPU/短时测试、dry-run、preflight、最多8个真实窗口的只读检查、不会更新参数的step-0评估。E01虽然只有20步，仍属于真实GPU训练，需明确批准。

不得自动执行：2,000步pilot、1,000步operator、追加10k/20k、三seed、全矩阵、evaluator训练、全量数据重建。不得删除或覆盖旧checkpoint/store。

最终交付不是一句“训练完成”，而是：

- 每个stage独立的resolved config、源码/config/data/protocol hash；
- preflight与训练命令、退出码、实际optimizer steps、wall time、显存；
- `history.jsonl`、`train_summary.json`、last/best checkpoint及其身份；
- 固定逐样本验证与生成工件；
- stage判定：`pass / fail / inconclusive / interrupted / awaiting_budget`；
- 哪个研究命题得到初步支持、被否证或尚不可判断。

## 2. 执行总规则

### 2.1 工作区与身份

当前大量revision2改动仍在dirty worktree中。不要执行 `git reset/clean/stash/checkout`，不要从HEAD覆盖工作区。训练工件绑定实际源码digest，不要求为了训练强行提交；如果用户另行要求commit，再单独处理。

每个真实run必须使用全新输出目录。默认输出存在任何run artifact时停止；不得用 `--allow-existing-output` 绕过。dry-run可以先在同一目录写协议，但开训前确认不存在 `best.pt/last.pt/train_summary.json/history.jsonl/overfit_last.pt`。

正式输出命名规则：

```text
outputs/mts_revision2/<stage>_s<seed>_<YYYYMMDD_HHMM>/
```

当前配方已有固定目录。执行前应复制成一次性run配方或用明确 `--output` 覆盖到含时间戳的新目录，并在run manifest记录最终路径。不能反复写 `transport_profile_seed3407` 或 `transport_pilot_seed3407`。

### 2.2 固定事实

- tokenizer：`outputs/nef_fsq_soma_packed_40x9_ah/best.pt`；必须与token store记录的SHA一致。
- token store：`data/processed/seed_soma_pruned_v4_ah_tokens`。
- T=64，fp32，seed=3407；首轮不启用AMP。
- transport与operator使用同一个冻结action map；训练词表为真实train split可见的17类。
- transport正式validation是seen-action val subset：14,230个满窗val clip中8,550个eligible，5,680个未知action clip被排除并记录。本轮不能声称完整SEED val或unseen-action结果。
- operator可训练的same-style/different-content style目前只有：`neutral / injured leg / injured torso`。不得报告8-style训练或unseen-style。
- operator各arm必须共享同一个base checkpoint SHA、pair/validation protocol完整hash、mask、action map和有效训练token预算。

### 2.3 禁止的快捷方式

- 不用profile的10行protocol与pilot的320行protocol比较best数值。
- 不把20步profile checkpoint用作正式operator上游。
- 不把warm-start称为resume；optimizer、step、best和RNG均重置。
- 不用random/shuffled reference冒充no-reference。no-reference必须是`ConstantStyleEncoder`。
- 不用operator自己的NLL当独立style/content指标；NLL是诊断。
- 不把低foot slide单独解释为运动质量提升；同时检查速度、幅度和接触率。
- 不因loss下降就跳过固定动画、负例和数据基线。

### 2.4 进度文件

在 [代码进度报告](MTS_FSQ_Code_Revision_Progress_zh.md) 顶部追加“Training Execution”章节，不删除历史记录。每个stage使用：

```text
Stage：Exx
状态：pending / preflight_pass / awaiting_budget / running / pass / fail /
      inconclusive / interrupted
授权：用户原话或 awaiting_budget
run_id / output：...
输入身份：tokenizer SHA、store hash、source/config/protocol hash、上游checkpoint SHA
预算：seed、device、B、T、epochs、steps_per_epoch、max_steps、wall cap
命令：dry-run / preflight / train / evaluate / generate
结果：exit、steps、计时、显存、关键train/val指标
判据：逐条pass/fail及证据
剩余限制：...
下一步：...
```

不要只粘贴console最后一行；原始stdout/stderr保存到run目录的 `logs/`。

## 3. E00：冻结本轮训练身份和基线

**目的：** 确保真正执行的文件就是审核通过的文件，并建立后续比较的公共身份。

### E00.1 定向回归

运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
  tests/test_mts_cli_matrix.py tests/test_mts_preflight.py \
  tests/test_mts_checkpoint.py tests/test_mts_eval_protocol.py
```

预期：本计划生成时为94 passed / 1 skipped。数量因后续测试增加可变化，必须exit 0；skip写出测试名和原因。

### E00.2 重新生成身份清单

不要直接沿用T04历史combined digest。本次核验发现冻结后有三份通用配置发生变化。对**拟用配方**重新计算：

```text
mts_revision2_transport_profile.yaml
mts_revision2_transport_pilot.yaml
mts_revision2_style_id_logit.yaml
mts_revision2_noref_logit.yaml
mts_revision2_style_id_ctmc.yaml
mts_revision2_reference_logit.yaml
mts_revision2_experiment_manifest.yaml
```

复用 `source_digest()` / `file_sha256()`，输出 `run_identity.json`：

```json
{
  "code_commit": "...",
  "working_tree_dirty": true,
  "source_digest": "...",
  "config_sha256": {"...": "..."},
  "tokenizer_sha256": "...",
  "token_store_checkpoint_sha256": "...",
  "feature_schema_hash": "...",
  "normalization_hash": "...",
  "split_manifest_hash": "...",
  "environment": {"python": "...", "torch": "...", "cuda": "...", "gpu": "..."}
}
```

### E00.3 公共数据基线

在train split上只读统计并保存 `token_baselines.json`：

- global/per-coordinate unigram NLL；
- 若成本可接受，再按17个action分层统计per-coordinate unigram NLL；
- 每coordinate entropy、level使用率；
- train/val eligible clip/action/style counts。

只从train统计分布，在val上评估NLL。统计脚本必须流式读token store，不把60GB feature store载入内存。若新增脚本，建议 `scripts/evaluate_mts_token_baselines.py`，只负责统计与评估，不构建新模型。

**E00通过条件：** 定向测试exit 0；实际配方/源码/data身份齐全；tokenizer/store SHA相等；公共baseline文件finite；没有覆盖旧输出。否则停止。

## 4. E01 / S0：20步真实profile

**配方：** `data/configs/mts_revision2_transport_profile.yaml`。D256/depth8、B8、T64、seed3407、20steps；validation为5类×2行。

### E01.1 dry-run和预检

为本次run选择新目录 `<PROFILE_OUT>`，例如：

```text
outputs/mts_revision2/transport_profile_s3407_20260918_HHMM
```

执行agent将以下占位符替换为实际绝对或仓库相对路径，并把解析值写入stage记录：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/train_mts_transport.py \
  --config data/configs/mts_revision2_transport_profile.yaml \
  --output <PROFILE_OUT> --dry-run --device auto

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/preflight_mts_revision2.py \
  --config data/configs/mts_revision2_transport_profile.yaml \
  --stage transport_train \
  --validation-protocol <PROFILE_OUT>/validation_protocol.json \
  --max-windows 2 --device auto \
  --output <PROFILE_OUT>/preflight
```

要求preflight `ready=true`、exit0、protocol split仅val、10行、完整hash与dry-run一致。dry-run后若目录只有protocol/dry-run辅助文件，可继续；若已有run artifact则改新目录，不用allow flag。

### E01.2 预算批准点

到这里暂停，向用户提交具体申请：20 optimizer steps，B8，T64，fp32，目标GPU，独立wall cap建议15分钟，输出目录与preflight路径。得到明确授权后才执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/train_mts_transport.py \
  --config data/configs/mts_revision2_transport_profile.yaml \
  --output <PROFILE_OUT> --device auto --max-wall-seconds 900
```

如果train入口因dry-run辅助文件认为目录冲突，先确认命中的具体run artifacts；不得删除。创建新目录，重新dry-run/preflight/train。

### E01.3 自动检查

训练退出后检查：

- exit0；`completed=true`、`optimizer_steps=planned_steps=20`、`steps_shortfall=0`；
- `interrupted=false`；loss、grad相关诊断、val objective/per-kind均finite；
- `best.pt/last.pt/train_summary.json/history.jsonl/validation_protocol.json`存在且可严格回读；
- checkpoint tokenizer SHA、action map、protocol hash与run identity一致；
- history记录 `train_seconds/validation_seconds/checkpoint_seconds/total_seconds`；
- GPU峰值显存。若代码未自动记录峰值，执行命令外围只读采样 `nvidia-smi`，记录方法和误差，不修改训练算法。

计算稳定训练step吞吐时排除初始化与首次CUDA编译影响；如果history只有一个20步epoch，另从step日志估计后10步中位数。总pilot成本按：

```text
10 × (200步训练时间 + 320行validation时间 + checkpoint时间)
```

估计，不能简单用20步总时长×100。

### E01.4 决策

- pass：无OOM/nonfinite/短步，身份和协议一致，可准备E02/E03。
- fail：OOM先只降B到4，再重做新run身份/profile；不能先降D/depth。
- fail：nonfinite、数据/condition/hash错误则停止并诊断，不以更小LR盲目重跑。
- interrupted：保留工件为profile interrupted，不当完成；用新目录重跑。

profile checkpoint不得作为正式operator上游。

## 5. E02 / S1：小数据可学习性诊断

**目的：** 在投入2,000步前确认训练主路径确实可优化。它不是benchmark。

### E02.1 创建独立诊断配方

从profile配方派生新文件 `data/configs/mts_revision2_transport_overfit.yaml`，不要临时修改profile/pilot：

```yaml
evaluation:
  protocol_id: mts-transport-r2-overfit-monitor-v1
  validation_kinds: [random_coordinate, stream, temporal_span, spatiotemporal_block]
  validation_rows_per_kind: 2
training:
  epochs: 4
  steps_per_epoch: 50
  max_steps: 200
  log_every_steps: 10
  output_dir: outputs/mts_revision2/<new-overfit-run>
loader:
  batch_size: 8
```

运行时加 `--overfit-clips 8`。该模式只写monitor/overfit_last，不产validation-best。执行前为配方补测试：预算明确、output唯一、content=action_id、架构与profile一致。

### E02.2 预算与命令

这是最多200步真实GPU训练，需单独授权。dry-run/preflight后执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/train_mts_transport.py \
  --config data/configs/mts_revision2_transport_overfit.yaml \
  --overfit-clips 8 --device auto --max-wall-seconds <approved_cap>
```

### E02.3 判读

必须保存8个固定window的clip/start/action/mask seeds。至少包含：

- 有visible context的random/stream/span/block任务；
- 不以full_generation逐token拟合为主门槛。相同action下多个不同完整motion是多模态目标，不能要求NLL趋零；
- step0/20/50/100/200的同一mask NLL/accuracy；
- decode固定sample，核对hidden target未进入input。

建议通过判据：至少两种有context mask的NLL相对step0和train unigram baseline明显下降、accuracy上升，且趋势不是只由一个window贡献；所有数值finite。这里不硬编码百分比，在工件中报告实际曲线、每window结果和失败例。

如果8clip不动，再做单clip/固定mask≤100步诊断；单clip仍不动才视为优化链故障。不得直接进入pilot碰运气。

## 6. E03 / S2：2,000步base transport pilot

**配方：** `mts_revision2_transport_pilot.yaml`；B16为暂定值。只有E01显存/吞吐支持时保留B16；OOM风险高则派生B8配方并重新dry-run/preflight/hash。修改B会改变有效token/step，报告时同时给optimizer steps与supervised tokens，不与B16假装同预算。

### E03.1 启动前

1. 选择全新 `<PILOT_OUT>`，重算配方hash。
2. dry-run生成320行val协议（5类×64）；确认所有行来自val，完整hash固定。
3. `preflight --stage transport_train` ready=true。
4. 保存step-0模型在固定协议上的per-kind NLL/accuracy、train unigram baseline、固定8–16个生成/重建case。若缺base evaluator，先新增轻量 `scripts/evaluate_mts_transport.py`：读取transport checkpoint、固定protocol和tokenizer，输出逐样本NLL/accuracy、argmax/sample tokens和decoded motion；复用现有bundle/CRN/window/FK helper，不另写数据读取。
5. 获得2,000步GPU预算批准及wall cap。

### E03.2 训练

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python \
  scripts/train_mts_transport.py \
  --config data/configs/mts_revision2_transport_pilot.yaml \
  --output <PILOT_OUT> --device auto --max-wall-seconds <approved_cap>
```

不要从profile warm-start；pilot从新初始化开始。当前无exact resume。若运行被中断，不能用warm-start续成同一2,000步曲线；应选择：完整新跑，或先经用户批准实现exact resume。

### E03.3 监控与结束检查

每epoch（200步）自动固定validation一次。不要并行读取正在写的checkpoint做外部评估。训练期间只观察日志：

- train token-weighted NLL/accuracy、skipped steps、supervised tokens；
- per-kind val objective/count，validation hash不变；
- CTMC诊断尚不适用；
- GPU显存、step time、data wait如已有；
- nonfinite、steps shortfall、wall cap。

结束后用best与last分别跑固定transport evaluator，报告：

- 五种mask的NLL/accuracy及相对step0/unigram改善；
- full generation/infill分开；
- token entropy/diversity，不能只看argmax accuracy；
- source reconstruction→base sample的FK/root/contact/速度/幅度；
- 固定case动画，包含失败案例。

### E03.4 pilot Go/No-Go

Go需要同时满足：

1. run completed，身份与协议一致；
2. 固定val相对step0和train unigram至少在关键mask中有明确改善；
3. train/val曲线无明显持续发散；
4. decoded motion没有全面静止、随机高频抖动或root崩溃；
5. action condition反事实：固定可见context，换action id能影响输出，但不要求它单独确定完整motion；
6. best确实优于last或二者差异有可解释证据。

若不满足，状态为fail/inconclusive，不启动operator。若趋势正面但未收敛，可提出10k/20k扩展申请，附实测吞吐、曲线和新增预算。由于无exact resume，申请同时必须说明是实现resume还是从新初始化完整跑长预算。

## 7. E04 / S3：style输入是否有用

只有E03 Go才开始。首轮比较：

```text
style-ID logit（实验臂）
constant/no-reference logit（关键控制臂）
```

### E04.1 公共协议

1. 将两份配方中的transport checkpoint更新为同一个 `<PILOT_OUT>/best.pt`，并记录SHA。使用派生run配方，不原地改历史模板。
2. 两臂用同一train pair manifest、同一validation完整行集/hash、同一action map、mask schedule、B16、T64、seed3407、1000 optimizer steps。
3. 当前operator脚本会分别构建协议。训练前必须导出并比较完整fingerprint；不同则停止。更稳妥的实现是允许 `--validation-protocol` 只读复用同一个协议文件；如果尚无该入口，先实现并测试，不能只比较protocol_id字符串。
4. style-ID map必须恰为三个eligible style，排序和checkpoint记录一致。constant臂style map为空，但训练target/pair行集完全相同。
5. 先build一次固定test eval manifest，筛选seen-action、合法same-style/different-content与真实wrong-style reference，保存hash；style-ID评估使用ID正误，constant评估reference retrieval为N/A。

### E04.2 训练预算

两个arm各1000步，属于两个独立昂贵操作，需分别或一次明确授权。不要并发运行导致显存/吞吐不可比。建议先style-ID，再constant；每个完成后严格检查再启动下一臂。

命令以配方manifest为准，均显式传同一tokenizer与pilot checkpoint，并用新输出目录。每臂开跑前：dry-run→`operator_train` preflight ready=true→训练。

### E04.3 评估设计

在同一个固定manifest、相同CRN上比较：

- target token NLL base/styled/delta，按style平衡；
- 正确style-ID vs错误style-ID的配对差；
- strength 0/.5/1/1.5/2分布响应；
- whole-body与固定left-arm/radius1，steps=1和4分开；
- source→base、base→styled、source→styled的FK/contact/jerk/速度；
- 每style至少固定若干case，报告全部而非挑最好样本。

constant臂不能做“正确style”检索；它回答的是无style输入时，同容量operator仅靠target/context能改善多少。

### E04.4 S3继续条件

只有style-ID相对constant显示一致的条件效应，才进入CTMC/reference：

- 正确ID优于错误ID，并在三个style上分别报告；
- style-ID对比constant的收益不是只来自neutral或单一mask；
- content/physics代价没有不可接受的全面恶化；
- 视觉case能看到随ID变化的有方向响应，而不是随机变化。

不要在单seed screening阶段做显著性宣称。若无信号，优先诊断style imbalance、pair evidence和base上限，暂停CTMC/reference。

## 8. E05 / S4A：style-ID CTMC筛查

前置：E04 pass。使用 `mts_revision2_style_id_ctmc.yaml` 派生配方，同一base/协议/预算/seed。

检查：uniformization tail、mass_error、min probability、rate范围、有效support比例每epochfinite；发生截断/tolerance错误立即停，不调大max_terms掩盖。

与style-ID logit比较：

- 参数量和实际trainable参数差；
- 相同有效token预算下的per-style/mask NLL与物理代价；
- strength响应是否平滑；
- ordinal邻近编辑的transition distance。

CTMC只有在至少一项明确候选优势且其他质量不崩时，才值得新增shuffled adjacency和arbitrary/full kernel配方。shuffled不劣则ordinal主张不成立；CTMC不优则回到logit主线。

## 9. E06 / S4B：reference encoder筛查

前置：E04 pass；不依赖CTMC pass。先用reference logit隔离encoder问题，不直接reference CTMC。

训练前固定正确/真实错误/随机真实reference，并匹配action优先、排除same take/mirror/crop泄漏。训练与验证采用同一个base、action map和mask协议。

评估：

- correct reference NLL vs wrong/random；
- multi-positive style retrieval（按style平衡，报告chance）；
- 换reference时TV/change，固定source/CRN；
- reference actor/take/content分层，防止记身份或动作；
- 与style-ID上界和constant下界对比。

继续条件：correct reference在三个style均有一致信号，且不是reference动作复制；否则停止reference CTMC，回到encoder/pair设计。

## 10. E07：主实验决策

E04–E06之后写 `screening_decision.json`：

```json
{
  "base_transport": "go|no_go|inconclusive",
  "style_input": "supported|unsupported|inconclusive",
  "ctmc_vs_logit": "candidate_advantage|no_advantage|inconclusive",
  "reference_encoder": "supported|unsupported|inconclusive",
  "next_claim": "...",
  "evidence": ["artifact paths and hashes"],
  "failures": ["..."],
  "requested_next_budget": null
}
```

分支：

- style输入无用：不跑三seed/大矩阵；修数据或收缩任务。
- style-ID有效、CTMC无优势：logit+NEF局部编辑为主线，CTMC降为ablation。
- style-ID与CTMC有效、reference无效：先改reference encoder/配对，不宣称reference style transfer。
- reference有效：再规划三seed、held-out style、part/NEF公平对照、独立evaluator、长序列与失败集。

unseen performer需要检查所有学习组件及normalization曝光；当前actor-holdout store支持这条轴。unseen style仍未建立，后续需显式held-out style并保证有效跨content证据。seen-action subset结果不能推广到未知action。

## 11. 自动停止与人工决策

任何阶段出现以下情况，执行agent应停止当前run的后续依赖，保留工件并报告，不自动改参重跑：

- hash/action/style/protocol身份不匹配；
- required preflight非pass；
- NaN/Inf、CTMC概率或质量错误；
- OOM（只提出降B方案）；
- steps shortfall、wall cap、非零退出；
- 输出目录已有run artifact；
- validation行集或hash在方法间不同；
- 训练结果无法胜过数据/constant基线却准备扩大预算。

允许执行agent自主修复的范围：日志解析、路径拼接、summary字段、不会改变科研设定的明显代码bug，并补回归测试。任何会改变模型、loss、数据集合、mask比例、条件输入、学习率、预算或评估协议的修改，都必须作为新实验版本报告，不能在同一run中偷偷调整。

## 12. 可直接交给执行agent的提示词

```text
请执行 docs/MTS_FSQ_Training_Execution_Plan_zh.md。

先执行E00，只做测试、身份冻结和train-only token baseline；保留当前dirty worktree，
禁止git reset/clean/stash、覆盖旧checkpoint或复用已有run目录。

随后准备E01的dry-run和transport_train preflight。真实GPU优化尚未授权：
在启动20-step profile前停下，提交精确预算、设备、wall cap、输出目录和preflight证据，
等待用户明确批准。用户批准哪个stage，就只执行该stage，不自动继续后续昂贵训练。

每个stage必须使用全新run目录，保存resolved config、源码/config/data/protocol hash、
stdout/stderr、history、summary与checkpoint身份。一次一个run，结束后按文档门槛判定；
fail/inconclusive时不要靠增加steps或改超参自行续跑。

首轮顺序固定：20步profile → ≤200步小数据诊断 → 2000步base pilot →
style-ID logit vs constant/no-reference → style-ID CTMC → reference logit。
profile checkpoint不得作为正式operator上游；operator必须共享同一个pilot best SHA和完整协议hash。

研究效果在相应实验完成前一律写“未验证”。最终分别报告工程完成、模型质量、风格条件、
CTMC优势和reference敏感性，不能用一个NLL覆盖所有结论。
```

## 13. 本文件状态

本文件是训练runbook。生成时完成了关键测试复核与真实只读profile预检；没有启动任何新的真实优化步骤，也没有修改模型实现或现有训练配方。E01及之后需按阶段取得用户预算授权。

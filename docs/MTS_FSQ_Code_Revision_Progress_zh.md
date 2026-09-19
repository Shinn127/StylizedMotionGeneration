# MTS-FSQ 代码修订进度记录

执行规范：[MTS_FSQ_Agent_Code_Revision_Plan_zh.md](MTS_FSQ_Agent_Code_Revision_Plan_zh.md)
依据：[MTS_FSQ_Project_Audit_2026-09-17_zh.md](MTS_FSQ_Project_Audit_2026-09-17_zh.md)

## Training Execution（E00–E01，2026-09-18）

依据：[MTS_FSQ_Training_Execution_Plan_zh.md](MTS_FSQ_Training_Execution_Plan_zh.md)。
本轮只执行 E00（测试/身份冻结/train-only token baseline）并准备 E01 的 dry-run 与 preflight；
未启动任何真实 GPU 优化步骤，未改动模型/loss/数据/协议，未覆盖旧 checkpoint 或 run 目录，
未 reset/clean/stash。工程完成、模型质量、风格条件、CTMC 优势、reference 敏感性分别报告，
在相应实验完成前一律为“未验证”。

| 阶段 | 状态 | 工件 |
|---|---|---|
| **E00 身份与基线** | pass | `outputs/mts_training_execution_20260918/{E00/run_identity.json,E00/token_baselines.json,E00/token_baselines_profile.json,stages/E00.json}` |
| **E01 20-step profile** | pass | `outputs/mts_revision2/transport_profile_s3407_20260918_1635/**`；`stages/E01.json` |
| **E02 ≤200 步诊断** | pass | `outputs/mts_revision2/transport_overfit_s3407_20260918_1713/{dry_run.json,preflight/preflight.json}`；`E02/{run_identity.json,monitor_baseline/**}`；`stages/E02.json` |
| **E03 2,000 步 pilot** | pass（Go） | `outputs/mts_revision2/transport_pilot_s3407_20260918_1753/{dry_run.json,validation_protocol.json,preflight/preflight.json}`；`E03/{run_identity.json,token_baselines_pilot.json,tool_check_profile_eval.json}`；`stages/E03.json` |
| **E04 风格两臂** | pass（supported） | `outputs/mts_revision2/operator_{styleid_logit,noref_logit}_s3407_20260918_1814/{dry_run.json,validation_protocol.json,preflight/preflight.json}`；`E04/run_identity.json`；`stages/E04.json` |
| **E05 style-ID CTMC** | pass（no_advantage） | `outputs/mts_revision2/operator_styleid_ctmc_s3407_20260918_1854/**`；`E05/{e05_verification.json,e05_verification.py}`；`stages/E05.json` |
| **E06 reference logit** | pass（unsupported） | `outputs/mts_revision2/operator_reference_logit_s3407_20260918_1906/**`；`E06/{e06_verification.json,e06_verification.py}`；`stages/E06.json` |
| **E07 主实验决策** | 已生成 | `outputs/mts_training_execution_20260918/screening_decision.json` |

### E00：冻结运行身份与公共基线 — 已完成（pass）

**E00.1 定向回归。** 命令：
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -rs tests/test_mts_cli_matrix.py tests/test_mts_preflight.py tests/test_mts_checkpoint.py tests/test_mts_eval_protocol.py`
→ **exit 0，95 passed，0 skipped（34.86 秒）**，日志 `outputs/mts_training_execution_20260918/E00/logs/e00_targeted_tests.log`。
这四个文件含 5 处条件性 skip（真实工件缺失/无 CUDA 时跳过），本次一条都未触发。
复核文档记录的“94 passed、1 skipped”与本次“95 passed、0 skipped”不是同一口径：
本次多出的一条是此后新增的测试，之前被跳过的条件本次满足，故没有 skip 名可记录。

**E00.2 身份清单。** 新增 `scripts/mts_run_identity.py`（复用 `source_digest()`/`file_sha256()`/
`require_token_store_binding()`），对 7 份拟用配方重算，输出
`outputs/mts_training_execution_20260918/E00/run_identity.json`：

- `code_commit=e9aaceb…`，`working_tree_dirty=true`；
- `source_digest`：**52 个文件**，combined `b22e78c8d4c1c48f…`（含本轮新增的两个脚本；T04 的 50 文件摘要已不再使用）；
  baseline 脚本最后一次修 bug 后整份 identity 已重算，并逐一回查磁盘上每个文件摘要仍与记录一致（0 处不一致）；
- 逐配方 SHA：profile `19939c5e7c417e12…`、pilot `40c636156d085b67…`、style_id_logit `3768175f04436640…`、
  noref_logit `9a4acdb396c8a396…`、style_id_ctmc `969dc2a643a0e86d…`、reference_logit `f19e1e95fb426c17…`、
  experiment_manifest `043791385918b679…`；
- tokenizer 文件 SHA `e4507ff2e2a545e2…` **等于** store 记录的 `checkpoint_sha256`（绑定成立），
  store `feature_schema_hash=f7d840c5431b02f2…`、`normalization_hash=a7d2fb6ef13d6af6…`、
  `split_manifest_hash=8a870d73c20b9865…`，并记录 19 个 shard 的 SHA；
- 环境：python 3.13.15 / torch 2.13.0+cu132 / numpy 2.5.2 / CUDA RTX 5070 Ti，`auto→cuda`。

顺序说明：identity 在 E00 新脚本写完后生成；随后又修了 baseline 脚本协议路径的一处 argmax 索引错误，
于是用最终脚本重跑两份 baseline 并**重算 identity**，当前记录（52 文件、combined `b22e78c8…`）对应磁盘现状。
两份 baseline 在最终修订下重跑数值逐位相同（token_baselines.json 的数值在 argmax 修复前后也一致，
因为该行只在协议打分路径上）；这不是“文件存在”替代验收，而是数值层面的确定性检查。

**E00.3 公共 token baseline。** 新增 `scripts/evaluate_mts_token_baselines.py`（只读流式统计，不建模型），
输出 `E00/token_baselines.json` 与针对 profile 协议行集的 `E00/token_baselines_profile.json`：

- 统计只用 train split：113,883 clip / 49,419,961 帧全量计入，0 个 (coordinate,level) 空单元，
  smoothing=1.0 明确写入工件；每 coordinate 熵 2.196–3.065 bit。
- val：full-window clip 14,230，eligible（action 在冻结词表内）8,550，排除 5,680（`Martial Arts/Other/Sports`），
  与 T04 审计一致；**但 eligible 只覆盖 17 类词表中的 6 类 action**（Advanced Locomotion / Baseline /
  Basic Locomotion Neutral / Basic Locomotion Styles / Complex Actions / Stunts），
  其余 11 类在 val 无样本，因此不能报告“17 类 action 的 val 结果”。
- val unigram NLL（nat/token，越接近 0 越好）：global 2.1644、per-coordinate 1.9634、
  action-conditioned 1.9345（frame-weighted）；clip-weighted 1.9937/1.9636；level accuracy 0.2491。
- 该脚本同时用 `--protocol` 在 10 行 profile 协议上逐 kind 打分，得到“数据基线”：加权 unigram NLL
  **2.0122**（hidden token 9,633），并**独立重算协议完整指纹** = `c2ba221d8b894930…`，与 dry-run 报告一致。
- 正确性：`score_clip` 与逐 token 双重循环的朴素实现在一个真实 val clip 上逐位一致（<1e-12）；
  全部数值 finite 校验通过。过程中发现并修复了脚本自身三个错误（pooled 块未填充、
  level 分布未归一导致“负 NLL”、clip 权重除以 token 数导致量纲错误）以及协议打分路径的一处
  argmax 索引错误；修完后用最终脚本重跑两份 baseline，数值与修复前逐位相同（确定），
  再重算 identity。以上都只改新脚本，未改动任何库代码。

**E00 通过条件**：定向测试 exit 0 ✓；配方/源码/data 身份齐全 ✓；tokenizer 与 store SHA 相等 ✓；
baseline finite ✓；未覆盖旧输出 ✓。

### E01 / S0：20-step profile — 已执行（pass）

- 新 run 目录：`outputs/mts_revision2/transport_profile_s3407_20260918_1635`（此前不存在，未复用固定目录）。
- dry-run（exit 0）：写入 10 行、5 kind × 2 行、`splits=['val']` 的冻结协议，
  `protocol_id=mts-transport-revision2-profile-v1`，hash `c2ba221d8b8949302c19fc27c94eabb0e60382fce169ff6690ad391e737d0830`，
  trainable params 7,641,328，`existing_artifacts=[]`；日志 `logs/dry_run.stdout.log`。
- preflight（exit 0）：`ready=true`、`ok=true`、`required_not_passed=[]`、`failed=[]`、`blocked=[]`；
  required 检查 `tokenizer/store_identity/split_isolation/label_tables/pair_matrix/windows/mask_distribution/validation_protocol/transport_recipe/budget` 全 pass，
  只读窗口 2（cap 2）；报告 `preflight/preflight.json`。
- 预算申请（**尚未执行**）：20 optimizer steps（1 epoch × 20），B=8，T=64，fp32，seed 3407，
  device auto→cuda，wall cap 900 秒，输出到上述 run 目录；
  命令 `python scripts/train_mts_transport.py --config data/configs/mts_revision2_transport_profile.yaml --output <run> --device auto --max-wall-seconds 900`。
- 用户批准前不得运行该命令；pilot 成本待 profile 的 `history.jsonl` 计时按
  `10 × (200 步训练 + 320 行 validation + checkpoint)` 估算，不能用 20 步总时长 ×100。

**E01.2 执行结果（用户批准后运行，exit 0，wall 3.91 秒）**

- 预算与完成度：`optimizer_steps=planned_steps=20`、`steps_shortfall=0`、`completed=true`、
  `interrupted=false`；`supervised_tokens_total=101,589`。
- 指标：train loss 2.1035 / acc 0.1918（该 epoch 内 20 步的平均与总量口径）；val objective **2.0436**
  （per-kind：full_generation 2.2412、random_coordinate 2.2617、spatiotemporal_block 1.7529、
  stream 2.0703、temporal_span 1.8918；9,633 个隐藏 token，`missing=[] invalid=[]`），全部 finite。
- 分项计时：train 0.9218 s、validation 0.0308 s、checkpoint 0.0974 s、total 1.0500 s；进程 wall 3.91 s，
  max RSS 2.05 GB。
- 显存（外部只读 `nvidia-smi` 采样，0.5 s 周期，8 个样本，方法/误差写入工件）：
  device `memory.used` 953 → 峰值 **2,832 MiB** / 总计 16,303 MiB，本训练进程峰值 **1,874 MiB**，
  device 利用率峰值 82%；采样间隔内可能出现未观测峰值，进程在最后一次采样后即退出。
- 身份一致性（`e01_verification.json`）：best.pt 用 `load_mts_checkpoint(..., require_tokenizer=True)`
  严格回读成功（7,641,328 参数）；`tokenizer_checkpoint_sha256` == tokenizer 文件 SHA == E00 identity；
  checkpoint/history/summary 三处 protocol hash 相同且等于 `c2ba221d…`；
  `source_digest` == E00 identity 的 `b22e78c8…`；action map 恰为 train split 的 17 类；
  store 绑定（split 表 SHA、feature_schema/normalization/split_manifest、clip 数）全部一致。
- `best.pt` 与 `last.pt` 语义相同（仅一个 epoch）：所有张量与字段逐位相等，文件 SHA 不同只是因为
  `torch.save` 把目标文件名写进归档成员名（`.best.pt/data.pkl` vs `.last.pt/data.pkl`）。
- **与数据基线的关系（如实记录，不是效果声明）**：同一 10 行 profile 协议上，数据 unigram 基线
  加权 NLL = 2.0122，profile 20 步后 val objective = 2.0436，即 **20 步的模型尚未超过数据基线**。
  这正是 profile 只做流水线验证、不能当质量证据的原因，也说明后续 E02/E03 的必要性。
- 已知缺口：训练器裁剪梯度但不记录梯度范数，“梯度 finite”无法从本次工件证明（只能证 loss/val/参数 finite）；
  step 日志没有时间戳，单步耗时只有“epoch 平均 46.09 ms/step（含 CUDA 首次编译）”这一保守口径。
- 判据（E01.4）：无 OOM、无 nonfinite、步数准确、身份与协议一致、checkpoint 回读通过 → **pass**；
  profile checkpoint 不进入 operator 上游。
- pilot 成本估计（待 E03 用实测替换）：`10 × (200 步 + 320 行 validation + checkpoint)`；
  按 profile 速率 B8 ≈ 103 s，按 B16 每步成本 2 倍 ≈ 195 s（含 warmup 的上界估计）。

### E02 / S1：小数据可学习性诊断 — 已执行（pass）

- 派生配方：`data/configs/mts_revision2_transport_overfit.yaml`（SHA `22d86178f42f1183…`），
  与 profile 的 `tokenizer/data/transport/masking/sampling` 逐块相等（有测试断言），
  `content.kind=action_id`、B=8、T=64、seed 3407、`max_steps=200`。
  **对 plan 的唯一偏离**：plan 的示例块写 `epochs=4, steps_per_epoch=50`，本配方用 `10 × 20`——
  同样 200 步总预算，但 monitor 落在 20/40/…/200，才能给出 plan 的 E02.3 要求的
  “step 20/50/100/200 同一 mask”曲线（4×50 只有 50/100/150/200）。要跑字面 4×50 只需
  `--epochs 4 --steps-per-epoch 50`，配方里的 budget 字段不变。
- 新增 3 个测试（`tests/test_mts_cli_matrix.py`）：配方与 profile 架构一致、overfit 入口冻结窗口且无验证集、
  monitor 每窗口固定 mask 且留下可复现的 manifest；定向集合 **98 passed**（35.7 s），`pytest -k mts` **257 passed**。
- dry-run（exit 0，目录此前不存在）：`overfit_clips=8`、`overfit_windows=8`、`planned_steps=200`、
  `validation.rows=0`（该模式不建验证协议，也不写 `validation_protocol.json`）、`existing_artifacts=[]`，
  trainable params 7,641,328 与 profile 相同；报告落盘 `dry_run.json`（本轮起 dry-run 会写文件，不再只打印）。
- monitor manifest（plan 明确要求保存 clip/start/action/mask seeds）：8 个窗口 = 4 类有 context 的 mask 各 2 个，
  逐窗口记录 `clip_id/target_start/action/mask_kind/mask_seed/hidden_tokens/hidden_fraction`；
  **实现缺口已补**：此前 overfit monitor 每个 epoch 重新采样 mask（曲线混入“换了 mask”这个非学习因素），
  现在每个窗口的 mask 由该窗口自己的 seed 固定（`fix_monitor_masks`），训练仍按 profile 的 mixture 采样。
  hidden token 合计 4,543；其中 `spatiotemporal_block` 每窗口只遮 64 token（2.5%），
  所以判据证据主要来自 `random_coordinate/stream/temporal_span`（1,471/1,664/1,280）。
- preflight：`--stage data`（data stage 不要求 validation protocol）→ **ready=true、ok=true、
  required_not_passed=[]、failed=[]、blocked=[]**，只读窗口 2；报告 `preflight/preflight.json`。
  `transport_train` stage 要求的“冻结验证协议文件”对没有验证集的诊断 run 不适用（dry-run 显示 rows=0），
  这里**没有削弱门禁**：没有伪造协议文件、没有把它指向别处的协议。
- 同一批窗口、同一 mask 的数据基线（只读，CPU）：`E02/monitor_baseline/{monitor_protocol.json,token_baselines_monitor.json}`，
  加权 unigram NLL **2.2293**（random_coordinate 2.1083 / stream 2.2565 / temporal_span 2.4044 /
  spatiotemporal_block 2.1482；action-conditioned 更低）；基线逐窗口复现了 monitor 的 mask
  （hidden token 数与 manifest 完全一致），因此“模型 vs 数据基线”是同条件比较。
- 身份：本轮源码 digest 变为 `efe495d870292acc…`（53 文件），相对 E01 只多出 overfit 配方、
  只改了 `scripts/train_mts_transport.py`（dry-run 落盘 + monitor 固定 mask），**都不在训练数学路径上**；
  E01 的 checkpoint 仍绑定它自己的 `b22e78c8…`。tokenizer/store SHA 不变。
- 预算申请（**尚未执行**）：≤200 optimizer steps（10 × 20），B=8，T=64，fp32，seed 3407，
  `--overfit-clips 8`，device auto→cuda，wall cap 900 秒，输出到上述 run 目录；
  只写 `overfit_last.pt/overfit_frozen.json/history.jsonl/train_summary.json`，**不写 best.pt**；
  命令 `python scripts/train_mts_transport.py --config data/configs/mts_revision2_transport_overfit.yaml --output <run> --overfit-clips 8 --device auto --max-wall-seconds 900`。
  预计 GPU 时间与 profile 同量级（10–20 秒级）。
- 待用户选择：诊断 run 的门禁是 (a) 接受 data stage preflight（当前做法），还是 (b) 先实现一个
  monitor-only preflight 模式（新增 stage，要求 monitor manifest 而不是验证协议，配测试）再跑。

**E02 执行结果（exit 0，wall 12 秒，200/200 步）**

- 完成度：`optimizer_steps=planned_steps=200`、`steps_shortfall=0`、`completed=true`、`interrupted=false`；
  只写 `overfit_last.pt/overfit_frozen.json/history.jsonl/train_summary.json`，无 `best.pt`/`last.pt`，
  `summary.best_val_loss=null`、`validation_protocol_id=null`（诊断不声称 validation-best）。
- monitor 曲线（固定 mask，10 个点，step 20→200）：
  NLL **2.1380 → 2.1279 → 2.1138 → 2.1094 → 2.0821 → 2.0238 → 1.8707 → 1.8361 → 1.7129 → 1.6019**；
  accuracy 0.176 → **0.410**；同一 8 窗口同一 mask 的数据基线 = **2.2293** → 最终领先 **0.627 nat**。
- 逐 window（8/8 全部改善，且每一类都赢过基线）：random_coordinate +0.55/+0.77、stream +0.43/+0.48、
  temporal_span +0.91/+0.80、spatiotemporal_block +1.10/+0.70（nat，相对同类 unigram 基线）；
  没有单个窗口在支撑曲线。
- 输入泄漏检查（E02.3 要求）：把 hidden target 的 level 全部 +1 后重算，8 个窗口的 logits 变化**恰好为 0.0**
  ——隐藏目标没有进入模型输入（embedding 在 hidden 位置用 mask 向量替换 level 项）。
- 严格回读：`overfit_last.pt` 通过 `load_mts_checkpoint(..., require_tokenizer=True)`；tokenizer SHA 与
  文件/E02 identity 一致；action map 17 类；source digest 与 E02 identity 相同。
- 显存：device 峰值 3,091 MiB / 16,303 MiB，本进程峰值 1,874 MiB，利用率峰值 87%。
- 判据（E02.3）：5/5 通过 → **pass**；但限制写明：这 8 个是**训练窗口**，
  只能说明“优化链能学”，不构成泛化或质量证据。

### E03 / S2：2,000 步 base pilot — 已执行（pass / Go）

- 新目录：`outputs/mts_revision2/transport_pilot_s3407_20260918_1753`（此前不存在）。
- dry-run（exit 0）：**320 行**（5 kind × 64 行）冻结 val 协议，hash
  `82b0640a4b38df123ae69360c3dfbe30993b062db6d2338c7745f10073a256f4`，B16、T64、fp32、seed 3407、
  `planned_steps=2000`、`existing_artifacts=[]`；preflight（`--stage transport_train`，这次该 stage 适用）
  **ready=true、required_not_passed=[]、failed=[]、blocked=[]**。
- 数据基线（同 320 行、同 mask，只读）：hidden token 306,719，加权 unigram NLL **1.9857**
  （per-kind 1.9606–2.0423；action-conditioned 1.9339–2.0157）；baseline 脚本独立重算的协议 hash 与 dry-run 一致。
- **新增 base evaluator**`scripts/evaluate_mts_transport.py`（E03.1 第 4 项要求）：
  逐行/逐 kind 的 masked NLL 与 accuracy（用每行自己声明的 mask）、argmax/sample tokens、
  固定 case 的 decoded motion（.npy，64×248）、可选与 unigram 基线比较；拒绝对 store split 撒谎的行，
  并校验协议文件的自述形状与解析结果一致。**工具校验**：在 E01 profile checkpoint + 10 行协议上，
  它复现了训练器自己的 val objective（2.0435916400730725 vs 2.0435919792140043，差 3.4e-7），
  并写出 8 个 case 的 token/motion —— 证明“evaluator 与训练口径一致”。
- **step-0 参考**：pilot 在第 1 步之前写 `init.pt`（只有模型与 provenance，不含 optimizer 状态），
  因此 step 0 来自本次 run 自己的初始化，而不是按 seed 重建；`RUN_ARTIFACTS` 也把 `init.pt` 纳入
  “拒绝覆盖”的判定。step-0 的评测本身在 pilot 之后用同一 evaluator 跑（权重此刻还不存在）。
- 身份：digest `9d996fbdf60040e3…`（54 文件），相对 E02 只多出 evaluator、只改了 transport CLI（init.pt）
  与 training.py（RUN_ARTIFACTS）；tokenizer/store 绑定不变。测试：定向 99 passed、`-k mts` 258 passed。
- 预算申请（**尚未执行**）：2,000 steps（10 × 200），B16，T64，fp32，seed 3407，device auto→cuda，
  wall cap 3,600 秒，无 warm start，输出到上述目录；
  命令 `python scripts/train_mts_transport.py --config data/configs/mts_revision2_transport_pilot.yaml --output <run> --device auto --max-wall-seconds 3600`。
  成本估计：训练 200 步/epoch 约 12–18 秒（B16 ≈ 2× B8 单步成本）→ 10 epochs 约 2–3 分钟；
  320 行验证每 epoch 约 1–3 秒；checkpoint 每次 0.1–0.2 秒；合计约 3–5 GPU 分钟，3,600 秒是上限不是预期。

**E03 执行结果（exit 0，wall 2:09.66，2000/2000 步）**

- 完成度：`optimizer_steps=planned_steps=2000`、`steps_shortfall=0`、`completed=true`、`interrupted=false`；
  监督 token 28,416,092；max RSS 2.48 GB；显存 device 峰值 4,820 MiB / 16,303 MiB（本进程 3,102 MiB），利用率峰值 95%。
- 逐 epoch（train 约 12 秒、320 行 val 约 0.36 秒、checkpoint 约 0.15 秒）：
  val objective **1.9920 → 1.8589 → 1.7450 → 1.7012 → 1.6584 → 1.6227 → 1.5940 → 1.5636 → 1.5234 → 1.5331**；
  best = epoch 9（1.5234），last = 1.5331 → **best 确实优于 last**（差 0.0097），best.pt 为 epoch 9。
- 三个检查点在同一 320 行协议上（同一 evaluator，CPU）：
  **init（step 0）2.3717 → best 1.5234**（改善 0.8483 nat）；**相对数据基线 1.9857 改善 0.4623 nat**；
  last 1.5331。注意 init 本身**差于**数据基线（2.3717 > 1.9857），即“未训练模型不如 unigram”，训练后反超。
- 分 kind 相对基线改善：random_coordinate **+0.704**、spatiotemporal_block **+0.624**、
  temporal_span **+0.602**、stream **+0.370**、**full_generation 仅 +0.012**。
- 逐 case（8 个，2/kind，clip/start/action 均可定位）：
  context 类 NLL 1.07–1.61、accuracy 0.41–0.63，argmax 解码**跟随窗口**（root 速度比 0.15–2.39×target，
  无僵死/无 root 崩溃、无两个窗口解出同一结果）；full_generation 的 argmax 解码是“action 条件众数”
  （root 速度仅 target 的 1–27%，两次同 action 的 case 解出几乎同一 motion），采样解码能动但比 target
  粗 3.1–3.3× —— **2010 步的 base transport 还不能做可用的自由生成**，这与“operator 是编辑器不是生成器”一致。
- 动作反事实（固定可见 context，只换 action id）：每行都改变输出、同 action 对照变化**恰好 0.0**；
  但幅度较小（监督位置概率平均变化中位数约 0.9%、argmax 翻转 5–10%）——记录为观察，不作为缺陷
  （plan 明确只要求 action 能影响输出，不要求它单独决定完整 motion）。
- 身份：init/best/last 三个检查点都通过严格回读，tokenizer SHA、17 类 action map、source digest 与 E03 identity 一致，
  best/last 的 protocol hash 与 run 记录一致（init 因早于首次验证，改以 provenance 中的 protocol id 核对）。
- 判据（E03.4）：7/7 通过 → **pass / Go**（上游 best SHA `ef8d00a8d84a907d…`，协议 hash `82b0640a…`）。
- 已知限制（写入 stage 记录）：seen-action val subset（8,550/14,230 eligible，val 只有 6 类 action 有样本）；
  full_generation 自由生成弱、采样偏粗；action 影响偏弱；best/last 差距小；三 seed、unseen-style、part/NEF 均未测。
- 说明：run 本身在 digest `9d996fbdf60040e3…` 下执行；之后为“case 覆盖 kind / 逐行熵 / case 命名”
  扩展了验证工具（只读工件），不影响训练内容。

### E04 / S3：风格输入是否有用 — 已执行（pass / supported）

- 上游绑定：四份 operator 配方（style_id_logit / noref_logit / style_id_ctmc / reference_logit）与 experiment manifest
  都从模板占位目录改为**真实 pilot best**（`outputs/mts_revision2/transport_pilot_s3407_20260918_1753/best.pt`，
  SHA `ef8d00a8d84a907d…`）；两臂共用同一个上游 SHA。
- 共享协议：两臂 dry-run 得到的 213 行协议**逐行完全相同**（fingerprint `f04a5d4e71073794…`，
  per-kind 41–45 行，val only，每行都记录了 mask_config），满足 plan “同一完整行集/hash”的要求。
- **准备时发现并修掉三个真问题**（都有反例测试）：
  1. operator 协议行**从不记录 mask_config**（CLI 没把 mask_generator 传进 `ValidationProtocol.build`），
     preflight 因此无法与配方核对 → 现在每行带规范形 mask 块；
  2. val 行里混入 transport 17 类词表之外的 action（`Other` 76 行、`Sports` 31 行，共 107 行），
     验证第一批就会因 `ContentVocabulary` 拒绝借 id 而崩溃 → 现在按 kind 排除并计数，
     某类被排空则报错；preflight 只在“差量能被记录在案的排除解释”时放行，并在 reason 里说明；
  3. preflight 的 mask 比较改为规范形（配方扁平写法与行内嵌套写法是同一配置）。
  期间我自己的拼接误删过逐行 store split 校验，被测试立刻抓出并恢复（213/213 行已对 store 分表核验，0 行错 split）。
- preflight `--stage operator_train`：**两臂都 ready=true、exit 0**（transport/action_vocabulary/
  checkpoint_bindings/validation_protocol 全 pass）。
- 测试：定向 **100 passed**、`-k mts` **259 passed**。身份 digest `a1af1053da0b1781…`（54 文件）。
- 预算申请（**尚未执行**）：两臂各 1,000 steps（5 × 200），B=16，T=64，fp32，seed 3407，
  device auto→cuda，每臂 wall cap 1,800 秒；顺序为先 style-ID 后 constant（不并发）；
  命令与输出目录见 `stages/E04.json`。预计每臂 1–3 分钟。
- 判据（E04.4）：正确 style-ID 优于错误 ID（三个 style 分别报告）；style-ID 相对 constant 的收益
  不能只来自 neutral 或单一 mask；content/physics 不全面恶化；固定 case 能看到随 ID 的有方向响应。
  单 seed 筛查，不做显著性宣称。

**E04 执行结果（两臂各 exit 0；arm A wall 1:23.5，arm B wall 1:24.4）**

- 完成度：两臂各 1,000/1,000 步、`steps_shortfall=0`、未中断；协议 hash 两臂与 run 记录完全相同
  （`f04a5d4e71073794…`，213 行）；两臂上游 transport SHA 都是 pilot best `ef8d00a8d84a907d…`；
  style-ID 臂的 style map 恰为三个合格 style（injured leg / injured torso / neutral），constant 臂为空。
- 显存：arm A device 峰值 2,255 MiB（进程 1,056 MiB，util 86%），arm B 2,110 MiB（进程 1,056 MiB）。
- val 曲线（同一 213 行）：arm A 1.5615 → **1.5381**（best，5 个 epoch 单调下降）；
  arm B 1.5560 → **1.5438**。style-ID 臂比 constant 臂低 0.0057 nat。
- 冻结协议上的配对比较（同 213 行、同 mask、token 加权）：

  | 参照 | NLL | 说明 |
  |---|---|---|
  | base（strength 0） | 1.7468 | 冻结 transport 的预测 |
  | constant 臂 | 1.7203 | 无 style 输入的同容量 operator |
  | **correct style-ID** | **1.7131** | 用行自身 style 的 id |
  | wrong style-ID | 1.7273 | 换成其它 id 的平均 |

  → **correct − wrong = +0.0142**（correct − constant = +0.0073；correct − base = +0.0337）；
  213 行中 156 行（73%）correct 优于 wrong。
- 逐 style 全部为正：injured leg +0.0094/+0.0034（对 wrong/对 constant）、injured torso +0.0219/+0.0160、
  neutral +0.0122/+0.0032 → **不是只靠 neutral**。逐 kind 也全部为正：context 四类平均 +0.0119/+0.0053。
- strength 响应（style-ID 臂，correct id）：0 → 1.7468、0.5 → 1.7247、1.0 → 1.7131、**1.5 → 1.7111（最优）**、
  2.0 → 1.7177 —— 有峰值、可用，但 2.0 不是最佳设置；constant 臂 strength 0 与 1.0 为 1.7468 / 1.7203。
- 判据（E04.4）：7/7 通过 → **supported**。
- **如实记录的限制**：单 seed、单预算（1,000 步）只是筛查；margin 很小（0.007–0.014 nat/token）；
  **operator 的增益大部分与 style 无关**——constant 臂本身就比 base 好 0.0265 nat，style 分支的净贡献只有
  0.0073；73% 行而非全部；strength=2.0 反而变差；physics/FK 对比尚未运行
  （入口是 `scripts/evaluate_mts_operator.py`）。
- 决策建议：按 E04.4 条文 E05/E06 可以进入，但鉴于 style 净效应很小，
  更稳的下一步是先用更多步数或第二个 seed 验证 margin 是否随预算增长，再决定是否为 CTMC/reference 付费。

### E05 / S4A：style-ID CTMC 筛查 — 已执行（pass / **no_advantage**）

- 新目录 `outputs/mts_revision2/operator_styleid_ctmc_s3407_20260918_1854`（dry-run + `operator_train`
  preflight ready=true 后才跑）；协议与 E04 两臂**逐行相同**（fingerprint `f04a5d4e…`，213 行），
  同 base（pilot best `ef8d00a8…`）、同预算（1,000 步）、同 seed 3407；参数量 146,194（logit 臂 143,881，+2,313）。
- 运行：exit 0、1000/1000 步、`steps_shortfall=0`、wall 1:21.5；显存 device 峰值 2,082 MiB
  （进程 1,058 MiB，util 85%）。val 曲线 1.5656 → 1.5650 → 1.5639 → 1.5619 → **1.5566**
  （logit 臂同预算 **1.5381**）。
- **CTMC 自诊断（每 epoch 全部 finite、无截断、无掉出容差）**：
  `uniformization_terms` 9.6–14.7（上限 256，远未触顶）；`poisson_tail` ≤ 3.5e-11 ≤ tolerance 1e-10；
  `mass_error` ≈ 2.6e-7 ≤ fp32 容差 1e-5；`min_probability_before_clamp` ≈ +0.0017（无负质量）；
  实到 `max_up_rate` ≤ 1.489 / `max_down_rate` ≤ 1.504，都在 `max_rate=2.0` 内；
  `support_fraction` 0.607–0.662。**没有出现截断或 tolerance 错误，也没有调大 max_terms 掩盖。**
- 与 logit 臂的同条件比较（213 行、token 加权的 NLL）：

  | | correct id | wrong id | base | vs logit |
  |---|---|---|---|---|
  | logit | **1.5464** | 1.5594 | 1.7468 | — |
  | CTMC | 1.5655 | 1.5702 | 1.5756 | **+0.0191（更差）** |

  逐 style 全部更差（injured leg +0.0163、injured torso +0.0400、neutral +0.0014），逐 kind 全部更差
  （+0.0076…+0.0278）；style 条件效应也弱 2–4 倍（CTMC correct−wrong = +0.0039…+0.0053，
  logit = +0.0096…+0.0172）。
- **ordinal 邻近编辑度量**（同一批受监督位置，strength 1、correct id）：
  CTMC 的 adjacent-mass fraction 0.7679 < logit 0.7814，expected level displacement 1.2806 ≈ logit 1.2551，
  但 CTMC 相对 base 的 TV 只有 0.0306（logit 0.0672）——**CTMC 移动的 mass 不到 logit 的一半，
  而且并不更贴近相邻 level**，因此“ordinal 优势”不成立。
- strength 响应：CTMC 1.7468 → 1.7387 → 1.7352 → 1.7348 → 1.7368（几乎平）；
  logit 1.7468 → 1.7247 → 1.7131 → 1.7111 → 1.7177（同预算下强得多）。
- 判据：`ctmc_validity_holds=true`、`same_experiment=true`、`quality_does_not_collapse=true`
  （|Δ|=0.019 nat，未崩）、但 **nll_advantage=false、ordinal_advantage=false → no_advantage**。
- **决策（按 plan §8 的规则）**：不新增 shuffled adjacency / arbitrary kernel 配方，**回到 logit 主线**；
  CTMC 保留为 ablation 记录，不作为主方法。局限写明：单 seed、单预算，只排除“这一 birth-death 参数化在该预算下”的优势，
  不代表任何 CTMC 形式都无效。

### E06 / S4B：reference encoder 筛查 — 已执行（pass / **unsupported**）

- 新目录 `outputs/mts_revision2/operator_reference_logit_s3407_20260918_1906`；dry-run + `operator_train`
  preflight ready=true；协议与 E04/E05 **逐行相同**（`f04a5d4e…`，213 行）、同 base、同预算、同 seed。
  运行 exit 0、1000/1000 步、wall 1:42.7、显存 device 峰值 3,284 MiB（进程 2,266 MiB）。
  style_encoder=reference（无 style id），**可训练参数 5,777,705**（style-ID 臂 143,881 的 40 倍）。
- val：1.5528 → … → **1.5453**；同协议下 style-ID 1.5381 < constant 1.5438 < reference 1.5453 < CTMC 1.5566。
- **reference 交换（正确 = 行自身配对 reference；错误/随机 = 同 action、不同 style 的真实 val clip，
  已排除同 take/同 source/镜像）**：`nll_correct` 1.5330、`nll_wrong` 1.5330、`nll_random` 1.5330 ——
  **差值约 1e-8，换 reference 根本不改变预测**。
- **机制诊断（决定这一步的判据）**：在 64 个不同 reference 上，descriptor 的相对离散度只有 **1.1e-5**
  （模长 0.54、平均偏差 6e-6），且**层内离散 6.0e-6 > 层间离散 1.8e-6**；
  把 descriptor 置零，输出在受监督位置平均变化 3.8e-4，而**真实换 reference 只变 2.4e-7** ——
  即 operator 确实读 descriptor，但编码器把任何 reference 都映射到几乎同一个向量；
  该臂学到的编辑（`delta_abs_mean` 升到 0.194）由 target 与 mask 驱动，与 reference 无关。
- copying 检查通过（与 target token 一致率 0.392 > 与 reference 的 0.267），但在“忽略 reference”的情况下这项必然通过；
  retrieval 0.507 vs chance 0.333 只是层间微小位移的产物，不能当作可用 style descriptor。
- 判据（plan §9 继续条件）：correct reference 未在三个 style 上显示一致信号 → **unsupported**；
  按规则**不启动 reference CTMC，回到 encoder/pair 设计**（例如对比/检索辅助目标，或先冻结编码器），
  且在任何更多 reference 预算之前必须先修设计。局限：单 seed、单预算，只排除“该目标+配对设计在 1,000 步下”的可用性。

### E07：主实验决策 — `screening_decision.json` 已生成

```json
{"base_transport": "go", "style_input": "supported", "ctmc_vs_logit": "no_advantage",
 "reference_encoder": "unsupported", "requested_next_budget": null}
```

- 结论：**logit + style-ID 是唯一在单 seed 筛查下站得住的分支**；CTMC 降为 ablation；reference 需先改设计。
- 三个需要新授权的候选下一步（写进 `notes`）：(a) style-ID logit 加预算到 2,000–3,000 步或加第二个 seed；
  (b) reference encoder 改设计（对比/检索辅助目标或先冻结编码器）；(c) 建独立于 operator 自身 NLL 的
  style/content evaluator。**未经新批准不再启动任何训练。**
- Go/No-Go（E03.4）：完成且身份一致；固定 320 行相对 step-0 与 unigram 基线明确改善；
  train/val 无持续发散；decoded motion 不僵死/不高频抖动/root 不崩；action 反事实（换 action id 能改变输出）；
  best 优于 last 或有可解释证据。不满足则 fail/inconclusive，且**不启动 operator**。

### 本轮新增/修改的文件（E00）

- 新增：`scripts/mts_run_identity.py`、`scripts/evaluate_mts_token_baselines.py`、
  `outputs/mts_training_execution_20260918/**`、profile run 目录（协议 + preflight + 日志）。
- 未修改：任何 `stylized_motion/**` 实现、任何 `data/configs/**` 配方、任何既有 checkpoint/store 与历史 run 目录。

## Training Readiness（T00–T04，2026-09-18）

依据：[MTS_FSQ_Training_Readiness_and_Next_Plan_2026-09-18_zh.md](MTS_FSQ_Training_Readiness_and_Next_Plan_2026-09-18_zh.md)。
本轮只做 T00–T04，不重做 C00–C11，不 reset/clean/stash，不覆盖旧 checkpoint/store，
不启动 S0–S5 真实学习；所有研究效果维持“未验证”。

| 门槛 | 状态 | 工件 |
|---|---|---|
| **T00 验证划分与主配方** | 完成 | 本文件 T00 节；`outputs/training_readiness_audit_20260918/T00_dryrun_summary.json` |
| **T01 transport 绑定与按阶段预检** | 完成 | 本文件 T01 节；`T01/preflight_*/preflight.json` |
| **T02 调试预算 / 验证成本 / checkpoint 证据** | 完成 | 本文件 T02 节；`T02_transport_dryrun.log` |
| **T03 可执行实验配置 + no-reference 对照** | 完成 | 本文件 T03 节；`T03/`（6 份配方 dry-run 证据） |
| **T04 开训门槛（六项）** | **pass** | `outputs/training_readiness_audit_20260918/training_gate.json` |

六项门槛：`train_val_disjoint` / `content_chain_consistent` / `transport_sha_bound` /
`preflight_required_pass` / `budget_explicit` / `tiny_real_chain_pass` 全部 pass。
**这不是效果声明**：未训练 revision-2 transport/operator，style 效果与任何指标都未经验证。

### T00（P0）：验证划分与主配方 — 已完成

改动：
- `stylized_motion/learning/mts_operator/eval_protocol.py`
  · 新增 `store_split_of_clip`：协议每一行都对 store 自己的 split 表（v4 `clip_split` / v3 `split_ids`）逐行核验；
    store 不能报告 split 时直接拒绝，而不是相信调用者传进来的字符串。
  · `build_target_only_samples` 改为 `rows_per_kind`（与 `loader.batch_size` 无关），返回
    `(samples, selection)`；训练集窗口混入、val 为空、可用 clip 少于请求行数都会报错而不是回退/缩水。
  · `ValidationProtocol` 新增 `protocol_id` 与 `selection`，`describe()/write()` 一并落盘；
    新增 `fingerprint()`（当前覆盖 kind/权重/行数/split/seed，T02 扩展到完整行内容）。
  · `build_validation_samples`（算子 pair 路径）对每对 target 也做 store split 核验。
- `scripts/train_mts_transport.py`
  · `window_source` 取 **val** split（`windows_by_clip(store, "val")`），val 无 64 帧窗口时直接失败；
  · 新增 `resolve_validation_recipe`：`evaluation.protocol_id` / `validation_kinds` / `validation_rows_per_kind`
    三个字段必需；旧的 `validation_batches_per_kind`（此前被接受但从未使用）改为显式报错并给出迁移名；
    `--val-rows` 改为 SystemExit 迁移提示；
  · overfit 模式不再把冻结的 **训练** 窗口称作 validation：新增 `monitor_batches`，指标记为 `monitor_*`，
    不参与 best 选择。
- `stylized_motion/learning/mts_operator/training.py`：`TransportTrainer.fit` 增加 `monitor_batches`。
- `data/configs/mts_revision2_transport.yaml`：`data.content.kind: action_id`；
  `evaluation: {protocol_id: mts-transport-revision2-full-v1, validation_kinds: 5 类, validation_rows_per_kind: 64}`。
- `data/configs/mts_revision2_style.yaml`、`mts_revision2_style_smoke.yaml`：补 `evaluation.protocol_id`；
  `scripts/train_mts_operator.py` 接受并记录它（同时写入 provenance.validation_protocol_id）。
- 测试夹具 `tests/test_mts_cli.py::write_tiny_token_store` 修正一个真 bug：`split_ids` 形参被同名局部变量遮蔽，
  传进去的 split 表一直被静默丢弃（默认值恰好与既有用法一致，所以此前没被发现）。现在按传入值生效。

验收（全部实跑）：
1. 新反例测试（先失败后通过）：
   `tests/test_mts_cli_matrix.py::test_transport_validation_rows_come_from_the_val_split_only`（逐行核 store split、
   train∩val take group 为空、protocol 行集不随 `loader.batch_size` 变化）、
   `::test_transport_refuses_a_val_split_without_a_full_window`（空 val / val clip 只有 10 帧 → 报错，不回退 train）、
   `::test_transport_never_shrinks_the_protocol_to_the_available_clips`（请求行数 > 可用 clip → 报错）、
   `::test_transport_migrates_the_legacy_validation_field_explicitly`（旧字段 → ValueError 指明新名；`--val-rows` → SystemExit）。
2. `tests/test_mts_cli_matrix.py` → exit 0，38 passed；`tests/test_mts_cli.py` → exit 0，22 passed。
   MTS 相关集合（cli/cli_matrix/eval_protocol/transport/transport_training/operators/end_to_end/preflight/checkpoint/
   metrics/pairs/sampling/contract/nef_probe）→ exit 0，**205 passed, 1 warning**（warning 是既有的
   `test_logit_field_strength_scales_the_deviation` requires_grad 提示）。
3. 真实数据只读 dry-run（新目录，无训练）：
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_mts_transport.py --config data/configs/mts_revision2_transport.yaml
   --dry-run --device cpu --output outputs/training_readiness_audit_20260918/T00_transport_dryrun` → exit 0。
   证据：`outputs/training_readiness_audit_20260918/T00_transport_dryrun.log`、
   `T00_transport_dryrun/validation_protocol.json`、`T00_dryrun_summary.json`。
   · action 词表非空：17 类（train split）；protocol_id=`mts-transport-revision2-full-v1`；
   · protocol 320 行 = 5 kind × 64 行，`splits=["val"]`；
   · 独立复核（不经训练代码）：320/320 行的 `store.clip_split == 1`，train/val take group 交集为 0，320 个不同 seed；
   · selection 如实记录 val 有 14230 个满窗 clip，其中 8550 可用，**5680 个因 action 不在冻结词表而被排除**
     （Other 1837、Sports 3823、Martial Arts 20）——该事实写入工件，不再等到读窗口时才崩。
4. 配置级断言（`tests/test_mts_cli.py::test_revision2_style_recipe_has_the_expected_mixture_and_budget`）：
   transport 主配方 `content.kind=action_id`、protocol_id/5 类/64 行；style 主配方 action_id；
   smoke 配方保持无条件（明确命名的 debug 配方）。

限制与说明：
- 主配方首轮正式协议为 320 行；profile 配方（每类 2 行、10 行、protocol_id 不同）在 T03 生成，二者 best 数值不可比。
- 算子 pair 路径仍使用 `validation_batches_per_kind × loader.batch_size`（该字段在算子入口是真的生效的）；
  行数与 objective/hash 的统一在 T02 处理。

### T01（P0）：transport 绑定与按阶段预检 — 已完成

改动：
- `stylized_motion/learning/mts_operator/checkpoint.py`
  · `load_mts_checkpoint` 新增 `tokenizer_checkpoint` / `tokenizer_checkpoint_sha256` / `require_tokenizer`，
    内部调用 `require_tokenizer_checkpoint`：同结构异权重 tokenizer 在加载时被拒；不传文件也不允许静默跳过
    （`require_tokenizer=False` 仅供无 tokenizer 工件的单元夹具，且仍需 checkpoint 记录过 SHA）。
  · 新增 `store_identity_block` / `split_table_identity`：记录 schema、clip 数、motion_dim、声明的各类 hash，
    以及 split 表的 sha256 与 train/val/test 计数（store 路径不再是身份）。
- 调用点迁移：transport warm-start、operator 上游 transport、preflight 全部传实际 tokenizer 文件。
- `scripts/train_mts_transport.py`：provenance 的 `store_identity` 换成 `store_identity_block(...)`。
- `scripts/train_mts_operator.py`：store identity 同样换成 block；**dry-run 现在先生成并写出 frozen protocol**
  （output 目录与 protocol 构造移到 dry-run 返回之前，报告新增 `validation` 段），消除“先训才能验”的循环。
- `scripts/preflight_mts_revision2.py`（PREFLIGHT_VERSION=2）
  · 新增 `--stage {data,transport_train,operator_train,evaluate}`，旧 `--level` 显式映射
    （data→data，experiment→operator_train）并在工件中记录；
  · 每个 stage 有 `REQUIRED_CHECKS` 表：`ready = 所有 required 检查 pass`；required blocked/fail → ready=false 且 exit 1；
    可选项用 `not_applicable`（不再制造 blocked）；`ok` 保留为“无 fail”的诊断字段，与 ready 明确区分；
  · `check_validation_protocol`：接收实际文件（`--validation-protocol`），核对 protocol_id/kinds/每类行数/权重/split，
    并**逐行用 store 自己的 split 表复核**（“文件存在”不再算证据）；缺文件时给出无需训练的生成命令；
  · 新增 `check_transport_recipe`（transport_train 必需：action_id + protocol_id + kinds + rows_per_kind，旧字段报错）；
  · 新增 `check_evaluate_model`（evaluate 必需：真正重建 operator bundle 或 transport，绑定 tokenizer 文件）；
  · `check_checkpoint_bindings` 支持 operator 或 transport 两种工件，并校验记录 SHA；“尚未产生的工件”归为 blocked；
  · budget 的“新输出目录”规则细化为：仓库内路径必须在 `outputs/mts_revision2/` 下，且
    best.pt/last.pt/train_summary.json 任一存在即拒绝；仓库外（测试/tmp）路径不套用前缀规则。

验收（全部实跑）：
1. 反例测试（先失败后通过）：`tests/test_mts_checkpoint.py` 新增
   `test_load_mts_checkpoint_refuses_a_same_structure_other_tokenizer`（正确文件通过；同结构异权重被拒；
   两者都不传 → 报错；未记录 SHA → 报错）与 `test_store_identity_block_records_the_split_table_and_schema`
   （不同 split 表 digest 不同）；`tests/test_mts_preflight.py` 新增 5 项：
   operator_train 正路径（tiny transport + 真实冻结协议，ready=true，exit 0，且逐行 store 复核 0 错）、
   换 tokenizer 的负路径（exit 1，store_identity fail 且报“same structure is not the same weights”）、
   缺 transport 的 blocked 负路径（ready=false 且 transport 在 required_not_passed）、
   transport_train 无需 transport checkpoint 即可 ready=true、
   以及“协议文件谎报 split”被拒（`rows_with_wrong_store_split == 行数`）；
   `tests/test_mts_cli_matrix.py` 的 transport reload 改为传真实 tokenizer 文件。
   `tests/test_mts_transport_training.py` 的纯单元夹具显式 `require_tokenizer=False`（该测试不持有 tokenizer 工件）。
2. 测试集合：`tests/test_mts_cli_matrix.py + test_mts_cli.py + test_mts_checkpoint.py + test_mts_transport_training.py +
   test_mts_eval_protocol.py + test_mts_end_to_end.py + test_mts_operators.py + test_mts_transport.py + test_mts_metrics.py +
   test_mts_pairs.py + test_mts_sampling.py + test_mts_contract.py` → exit 0，**226 passed, 1 warning**；
   `tests/test_mts_preflight.py` → exit 0，**13 passed**。
3. 真实数据只读预检（证据 `outputs/training_readiness_audit_20260918/T01/`）：
   · `--stage data` → exit 0，ready=true，ok=true，failed=[]（13 项检查，非必需项均 not_applicable）；
   · `--stage transport_train`（配 T00 dry-run 写出的真实协议）→ **exit 0，ready=true**：
     transport_recipe pass、validation_protocol pass（320 行、5 类、val）、budget pass；transport/checkpoint_bindings
     均 not_applicable（该 stage 不得要求已训练 transport）；
   · `--stage operator_train` → exit 1，ready=false，required_not_passed=[transport, action_vocabulary, validation_protocol]，
     而 **ok 仍为 true**——正是 B4 要求的“ok 与 ready 区分”；
   · `--stage evaluate` → exit 1，ready=false，blocked=[checkpoint_bindings, evaluate_model, manifest_leakage]。

### T02（P1）：调试预算、验证成本与 checkpoint 证据 — 已完成

改动：
- `stylized_motion/learning/mts_operator/eval_protocol.py`
  · `fingerprint()` 改为覆盖**完整行内容**（kind/split/clip/start/seed/label/mask_config）、权重、selection
    （含 store identity）：改一行 start、改 seed、改 mask config、改数据身份都会改变摘要；
  · `ValidationProtocol.build/from_target_windows` 接受并保存 `identity/store_identity`，协议文件自描述数据身份；
  · 新增 `validation_evidence(protocol, report, seconds=...)`：两类 checkpoint 用同一函数产出同一组字段
    （val_objective / val_per_kind / val_counts / val_supervised_tokens / missing / invalid / usable /
    protocol_id / 完整 protocol_hash）。
- `stylized_motion/learning/mts_operator/training.py`
  · `fit()`：epoch 结尾由回调返回的指标合并进 history（一次验证同时在日志/history/payload/best）；
    history 增加 `optimizer_steps`/`global_step`；返回 `planned_steps`/`steps_shortfall`/`interrupted`/`interrupt_reason`；
  · 新增 `planned_step_budget` / `resolve_budget`（max_steps 或 epochs×steps_per_epoch，否则拒绝训练）、
    `RUN_ARTIFACTS` / `refuse_existing_output`（已存在 best/last/overfit_last/summary/history 时默认拒绝，exit 1）；
  · 独立 wall-clock 上限 `max_seconds`：超时即停并记 `interrupted`，不宣称完成。
- `scripts/train_mts_transport.py`
  · overfit：冻结窗口**循环读取到明确 step 预算**（此前一个 epoch 只走一个 batch，5 步实际只走 1 步）、
    写 `overfit_frozen.json`（窗口数/批数/目标步数/mask/condition/seed），**只写 `overfit_last.pt`**，不写 last/best；
  · 训练前置：显式预算（否则拒绝）、输出目录防覆盖、`--allow-existing-output`（显式 scratch 重跑）、
    `--max-wall-seconds`；
  · 校验走回调一次；payload 用 `validation_evidence`；epoch 记录 train/val/checkpoint/total 秒；
    全 epoch 序列写 `history.jsonl`；summary 记录 planned_steps/steps_shortfall/completed/interrupted/resume
    （warm-start 是新 run，不是 exact resume）；预算未达成时打印原因并 exit 1。
- `scripts/train_mts_operator.py`：同样接入预算/防覆盖/wall 上限/history.jsonl/timing；
  **删除“无协议时回退训练 loss 并写 best.pt”的分支**（overfit 分支只写 `overfit_last.pt`）；
  payload 的 val_nll 现在就是协议 objective（此前记录的是另一个数）。

验收（全部实跑）：
1. 新增/更新测试（先失败后通过）：`tests/test_mts_cli_matrix.py`
   · `test_overfit_runs_exactly_the_stated_steps_and_claims_no_best`（1 batch/epoch 的 loader + 5 步预算 → global_step=5，
     只写 overfit_last.pt，history 只有 monitor_* 没有 val_*，overfit_frozen.json 记录 windows=2/repeated_to_steps=5）；
   · `test_a_short_run_is_not_reported_as_the_stated_budget`（数据只够 2 步而预算是 6 → exit 1，
     summary completed=false、steps_shortfall=4 —— 旧行为是静默 2 步）；
   · `test_validation_runs_once_per_epoch_and_is_the_number_that_is_saved`（monkeypatch 计数：2 epoch = 2 次 evaluate；
     history/last.pt/summary 三处 objective 相等；protocol_id/hash/counts 齐全）；
   · `test_a_wall_time_cap_stops_the_run_without_claiming_completion`（interrupted=true、completed=false、exit 1）；
   · `test_an_existing_run_directory_is_refused_by_default`（第二次运行 exit 1；显式 flag 才允许）；
   · `test_operator_overfit_claims_no_best_and_meets_its_budget`（算子 overfit：4 步、只写 overfit_last.pt、val_objective=None）；
   · 算子正路径补充断言：val_objective==val_nll==summary.best_val_nll、protocol_hash 64 位、
     history.jsonl 两行且 timing 键齐全。
   `tests/test_mts_eval_protocol.py::test_protocol_fingerprint_changes_with_any_row_or_identity`（行/seed/mask/数据身份任一改变 → hash 变）。
2. 测试集合：`tests/test_mts_cli_matrix.py` → exit 0，**44 passed**；其余 MTS 集合
   （cli/preflight/eval_protocol/end_to_end/transport_training/checkpoint/operators/transport/metrics/pairs/
   sampling/contract/nef_probe）→ exit 0，**215 passed, 1 warning**。
3. 真实数据只读 dry-run（`outputs/training_readiness_audit_20260918/T02_transport_dryrun.log`）：
   exit 0；`budget.planned_steps=null` 且注明“配方未固定步数预算 → 训练会拒绝启动”，
   protocol 仍是 320 行 val、`existing_artifacts=[]`。即：旧配方现在**不能**直接开训（这正是 T03 要修的），
   而 dry-run 不再假装“可以直接开训”。

### T03（P1）：可执行实验配置与 no-reference 对照 — 已完成

改动：
- 新增 6 份独立配方（每份只含自己需要的字段；family 不合法字段会被入口拒绝）：
  · `data/configs/mts_revision2_transport_profile.yaml`（S0：20 步、protocol `...-profile-v1`、每类 2 行）
  · `data/configs/mts_revision2_transport_pilot.yaml`（S2 首段：2,000 步、protocol `...-pilot-v1`、每类 64 行）
  · `data/configs/mts_revision2_style_id_logit.yaml`（logit + style_id，1,000 步）
  · `data/configs/mts_revision2_style_id_ctmc.yaml`（birth_death + style_id，1,000 步，CTMC 字段只在这份）
  · `data/configs/mts_revision2_reference_logit.yaml`（logit + reference，1,000 步）
  · `data/configs/mts_revision2_noref_logit.yaml`（**logit + constant**，1,000 步，no-reference 对照）
  四份 operator 配方共用同一 base checkpoint 路径、同一 pairs/mask/采样协议与同一个
  `protocol_id=mts-operator-r2-shared-val-v1`（行集相同才可比较）。
- 新增 `ConstantStyleEncoder`（`style_encoder.kind: constant`）：一个学出来的常量描述子，
  不读 reference token、不读 style id、不读 action；宽度与对照臂相同，参数量如实暴露。
  `MtsStyleOperator._reference_embedding`、`load_operator_bundle`、算子入口
  （`--style-encoder-kind constant`，且拒绝携带 reference/style-id 的 encoder 键）全部接通。
- `data/configs/mts_revision2_experiment_manifest.yaml` 改为 manifest_version 2：逐 arm 的配方/预算
  （整数）/解析后的输出目录/依赖与 blocked_on/命令；无 `<run_id>` 占位符；
  所有 arm 共用 `identity`（tokenizer、store、base transport、eval manifest、seed 3407）。
- `docs/mts_revision2_experiment_request_zh.md`：新增“T00–T04 之后的更正”提示块 + 按 arm 的配方/预算表 +
  no-reference 对照的含义与参数差 + 新的吞吐测量方法（读 `history.jsonl` 的四个计时字段）。

验收（全部实跑）：
1. 新增测试：`tests/test_mts_cli_matrix.py::test_every_operator_recipe_carries_only_its_own_family_fields`
   （逐份断言 family 字段、整数预算、互不覆盖的输出目录、共享 base、constant 只带 width）；
   `::test_operator_cli_dry_runs_the_control_arm_and_refuses_stray_encoder_keys`
   （真实入口 dry-run：`style_encoder=constant`、`style_index=None`、40 行 val 协议；
   再给 constant 加 reference 键 → ValueError）；
   `tests/test_mts_end_to_end.py::test_the_constant_descriptor_reads_no_style_input`
   （真实/乱序/置零/缺席 reference 与正确/错误/缺席 style id 概率完全相等；同时确认
   style-ID 与 reference 两臂在扰动权重后**会**随输入变化）；
   `::test_the_constant_control_can_still_learn`（常量仍收到梯度）；
   `tests/test_mts_checkpoint.py` 往返参数化加入 `constant`（bundle 重建后仍是常量、参数量=width）。
2. 测试集合：matrix 44 passed（T03 两项在内）、checkpoint 19 passed、end_to_end 16 passed；
   其余 MTS 集合不回归（见 T04 的汇总复跑）。
3. 真实配方 dry-run（证据 `outputs/training_readiness_audit_20260918/T03/`）：
   · **transport profile**（真实数据）→ exit 0；protocol `mts-transport-revision2-profile-v1`、
     10 行（2/类）、`splits=[val]`、budget=20（training.max_steps）、content=action_id；
   · **transport pilot**（真实数据）→ exit 0；protocol `mts-transport-revision2-pilot-v1`、
     320 行（64/类）、`splits=[val]`、budget=2000；
   · 四份 operator 配方逐个经真实入口 dry-run（用 tiny tokenizer/store/transport 作占位工件，
     配方本身的 operator/encoder/data/masking/evaluation 段全部生效）→ 全部 exit 0，
     40 行 val 协议、budget=1000：
     constant 81,929 可训练参数 / style_id 83,209 / style-CTMC 85,522 / reference 5,716,265
     （no-reference 对照与它对照的臂之间的参数差如实可见）；
   · style-ID 两臂在 tiny 数据上先被拒绝（配方写 `num_styles: 3`，tiny 数据有 6 个可训练 style），
     用 `--num-styles 6` 覆盖后通过——这正是“声明不一致就拒绝”的预期行为。
     真实数据的 3 已由 T04 的 pair 审计确认：8 个 style 中恰好 3 个（`injured leg` / `injured torso` / `neutral`）
     具备 same-style/different-content 证据，即风格轴的可用词表就是这 3 个。
4. manifest 一致性检查（脚本）：每份 arm 的 config 存在、`output_dir` 已解析且与配方一致、
   `protocol_id` 与配方一致、`max_steps` 为正整数且 `epochs × steps_per_epoch >= max_steps`。

## 当前状态（Integration Closure 收口）

| 门槛 | 要求 | 当前 | 依据 |
|---|---|---|---|
| **G-entry** | C01–C05（R 计划 R01–R09）完成，真实 tiny CLI、数据绑定与固定协议可执行 | **通过** | C09 矩阵 A/B/C1–C6/F 全部实跑；`tests/test_mts_cli_matrix.py` 34 passed |
| **G-code** | R00–R12 必要路径通过（C06–C09） | **通过** | C06–C09 完成，R12 的"不偷看未提交 token"反例本轮补齐；全量 542 passed / 1 skipped / 1 failed（既有 100STYLE 布局） |
| **G-ready** | R13–R14（C10–C11）完成，真实小样本入口通过、预算与覆盖输出 | **data 级通过 / experiment 级 blocked** | R13 公平性探针与 R14 预检实跑（含 mask 分布与 token 预算）；experiment 级因未训练 revision-2 transport 而 blocked，不伪造性能 |

任务状态：C00–C11 全部落地。C04/C05/C07/C08 此前的 pending 尾巴已收口：
- C04：映射输出（tokens/valid_mask/content_condition/sample_metadata）、`--dry-run`、
  resolved config 回写（provenance.resolved_config 用 CLI override 后的段）、在线/缓存数值比对（见 C09-D）。
- C05：`eval_rows.jsonl` 逐样本数值行（NLL base/styled/delta、retrieval rank/hit）、dataset 透传。
- C07：evaluator 接上 source/base/styled 三对比（base 取冻结 transport 分布，`generate_edit(use_base=True)`，
  与 styled 共用同一 CRN cell），`physics_per_comparison` 三块分开。
- C08：协议 v2 落地（signed pulse / fixed signed span / 共享合法 support / ±1 与 ±d 分层 /
  完整后尾与 `temporal_probe_complete` / feature 与 world 两个 tail 分开 / 新产物 probe_stratified.*）。

验证基线（本轮实跑）：
- 全量套件（无 deselect）`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`
  → exit 1，**542 passed / 1 skipped / 1 failed**；唯一失败是既有的
  `tests/test_preprocess_pipeline.py::test_100style_fsq_excludes_last_ten_styles`
  （本机 100STYLE 目录为扁平布局，与代码无关）。
- C09 矩阵：`tests/test_mts_cli_matrix.py` → exit 0，34 passed
  （outputs/mts_revision2_closure/C09/{c09_matrix.txt,c09_matrix_cases.txt}），
  期间发现并修复 16 条真实入口缺陷（c09_defects.txt）。
- C10 预检：真实 holdout tokenizer+tokenstore → exit 0、ok=true
  （C10/preflight_ah/preflight.json）；旧 1h tokenizer + 同一 store → exit 1、
  store_identity fail（C10/preflight_stale_tokenizer/preflight.json）。
- C08 分层探针真实数据实跑（2 窗口 × 13 坐标 × 4 扰动 = 104 行）：C08/stratified/
  （tail_frames_needed_for_complete_probe=99，rows_with_incomplete_tail=0）。
- 未运行任何 GPU 训练、全量重建或全矩阵；未删除/覆盖任何旧 checkpoint、store 或历史配置。
- 最终全量复跑证据：outputs/mts_revision2_closure/final_full_suite.txt
  （exit 1；542 passed / 1 skipped / 1 failed；失败原因为
   `data/raw/100style` 为扁平目录、测试期望 `Aeroplane/Aeroplane_BR.bvh` 嵌套布局，属外部数据）。

仍未验证的事实（不得当作结论）：
- 真实 SEED 上的 revision-2 transport/operator 训练、评估、生成的**成功路径**（预算未批准）；
- 模型性能、风格效果、跨内容泛化；experiment 级预检的 pass 分支；
- 旧 MTS 结果在新协议下的可比性（旧模型全部被 schema/revision 拒绝，需重训）。
## R02 记录（已通过）

```text
任务状态：passed
实际修改：
  operators.py
    · 拆开 bool `eligible`（= hard_mask & editability & valid，plan §1.1 的 effective_edit）
      与 float `support`（= eligible × strength）；三族输出统一经 region_identity()
      以 `where(eligible, transformed, base)` 收尾，区域外逐位等于 base。
    · AdditiveLogitField：区域外直接取 base_logits（delta 头部在区域外的取值不再泄漏）。
    · ArbitraryKernelOperator：默认模式 λ 只缩放 kernel logits（区域内 λ=0 即 uniform，
      区域外 base）；identity_mix 模式改为真正的混合权重语义（λ=0 全 base、λ=1 学习到的
      kernel），并显式校验 0≤λ≤1；kernel_offdiagonal_mass 改为真正的非对角元素求和，
      另报 kernel_offdiagonal_ratio = mass / levels；support_fraction 语义修正为
      eligible 比例，新增 strength_mean。
    · OperatorOutput：先做 isfinite 检查（NaN 之前会因所有比较为 False 而漏过）。
    · uniformization_expm_apply：非 finite 的 generator/probabilities 直接报错；
      Poisson 截断 tail > tolerance 时显式失败（不再靠 renormalize 掩盖）；
      inactive（row scale 恰为 0）元素的 Poisson 权重修正为 (1,0,0,…)（此前是 1/k!，
      使质量误差诊断失真）；mass error / 负值按 dtype 容差（float64 1e-9、其余 1e-5）
      检查，并把理论 Poisson tail 与浮点质量误差分别记录。
    · BirthDeathCTMCOperator._level_generator：level_order 的置换改为用**逆置换**
      （此前用 order 本身，level 空间图仍是 i→i+1，只是把速率贴错了边，几何对照失效）；
      forward/reference_expm 都做区域外 base 收尾。
  model.py
    · OperatorBatch 新增 anchor_mask（区域内显式观察点）与
      effective_edit_mask(spec, require_visible=True)（唯一的编辑掩码定义：
      hard_mask & ~visible & valid）；supervision_mask 改为其别名。
    · visible_mask=None 在 loss/forward 编辑路径直接报带字段名的 ValueError（此前
      静默当作“全部可见”→监督为空）；生成路径 require_visible=False 时默认
      visible = ~hard_mask（局部编辑默认）。
    · operator_inputs/_base 统一用同一掩码，并把 transport 的可见集设为
      ~edit_mask（锁定的 token 是证据，待编辑的 token 不是）。
    · generate_edit 用 effective_edit_mask 决定锁定位置：空区域逐位返回 source tokens。
  masking.py
    · apply_hard_support 的“至少监督一个位置”兜底改为在该模式允许的区域内部取点，
      区域为空时明确报错；此前 restrict 模式会偷偷在 support 外制造监督。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
    tests/test_mts_operators.py tests/test_mts_contract.py tests/test_mts_transport.py \
    tests/test_mts_end_to_end.py tests/test_mts_metrics.py
结果：exit 0；R02 相关 62 passed（operators 19 / contract 17 / transport 17 / end_to_end 11 /
  metrics 9，含重叠文件的一次汇总）
  全量回归：414 passed / 1 skipped / 1 failed（仅既有的 100STYLE 数据布局问题；
  test_nef_probe 的 float32 容差问题按 R13 处理，本轮 deselect）
反例前后对照（outputs/audit_20260917/r02_probe.py → r02_probe_fixed.py，JSON 同名）：
  | 反例 | 修复前 | 修复后 |
  |---|---|---|
  | arbitrary 空 hard region 与 base 的最大差 | 0.7124 | 0.0（逐位相等） |
  | arbitrary 空 edit set 与 base 的最大差 | 0.7124 | 0.0 |
  | invalid 帧被编辑的最大差 | 0.5773 | 0.0 |
  | identity_mix λ=1.5 | 以“negative probabilities”间接失败 | 明确的 strength 范围错误 |
  | kernel_offdiagonal_mass vs 真实非对角和 | 5.9436 vs 7.4730 | 7.4730 vs 7.4730 |
  | NaN probabilities | 被接受 | ValueError: must be finite |
  | max_terms=3、tol=1e-14 | 静默截断（tail 0.61、mass error 0.61、值误差 4e-2） | 显式失败并给出 tail/terms |
  | level_order 边（order=[2,0,3,1,4,6,5,8,7]，up=1..8） | 只有 4/8 条边存在且速率错位 | 8/8 边 = up_rate[i]，非边全 0 |
  | loss 路径缺 visible_mask | 静默全可见 → 空监督 | 带字段名的 ValueError |
  | CTMC vs matrix_exp（float64，值/梯度） | 值 5.6e-8、梯度 4.5e-8（float32） | 值 5.4e-12、rate 空间梯度 7e-11 |
限制与说明：
  · CTMC 的 uniformization 算法本身经核验是正确的（值、rate 空间梯度、半群、mass
    conservation 全部对得上 matrix_exp），因此未改算法，只加了显式失败与诊断。
  · row scale 恰为 0 的元素走 identity 分支：值逐位等于 base（plan 要求的全遮蔽语义），
    其梯度对 Q 为 0（与 exp 在 0 处的导数不同），测试里显式区分这两种语义。
  · 旧的 MTS checkpoint 不含 anchor_mask/新掩码语义，其生成结果不可与新协议比较（R08 处理）。
下个任务：R03
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
```

当前任务：R03
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/sampling.py
    · 新增 uniform_seed(seed, sample_id, step_id, shape)：SHA-256 派生，跨进程/平台稳定。
    · CommonRandomNumbers 的缓存 key 改为 (sample_id, step_id, shape)（显式、与 device 无关）；
      uniforms 一律在 CPU 生成后按需 .to(device)，不再用 id(device) 参与 key，也不再以
      缓存插入顺序决定种子；draws/device_transfers 计数；sample()/sample_tokens()/
      paired_comparison() 全部接受 sample_id/step_id。
    · inverse_cdf_sample 改为“取第一个 CDF > u 的 level”：u=0 不再命中零质量 bin；
      u 规范到 [0,1)（u=1 用 nextafter 压到小于 1 的最大可表示值），u<0 或 u>1 直接报错。
  stylized_motion/learning/mts_operator/model.py
    · generate_edit 增加 sample_id/step_id（CRN cell），空 edit 区域直接返回 source 且不采样，
      sampler 钩子按 sampler(probabilities, generator=…) 真正生效（crn 优先）。
  scripts/generate_mts_operator.py
    · 显式构造 region/visible_mask（visible = ~region），不再依赖隐式默认；
      --no-locked-edit 仅在 whole_body 区域下允许，否则报错（新 helper
      resolve_generation_region 抽出该判据，可单测）；每个 draw 用独立 sample_id；
      输出改为 source/base/styled 三分：source_tokens.npy、base_tokens.npy（冻结 transport
      的 argmax，不用 λ=0 的 arbitrary kernel 当 base）、tokens.npy、support/edit_mask/
      visible_mask.npy，generation.json 记录 sample_ids、crn_seed、crn_keys 与三类 change ratio。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_sampling.py tests/test_mts_end_to_end.py tests/test_mts_operators.py     tests/test_mts_metrics.py tests/test_mts_transport.py tests/test_mts_contract.py
结果：exit 0；相关测试全绿（sampling 13 / end_to_end 13 / operators 19 / metrics 9 /
  transport 17 / contract 17）
  全量回归：421 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
反例前后对照（outputs/audit_20260917/r03_probe.py → r03_probe_fixed.json）：
  | 反例 | 修复前 | 修复后 |
  |---|---|---|
  | one-hot@level7，u=0 抽到 | level 0 | level 7（u=0/0.5/0.9999/1.0 全部 7） |
  | 零质量 bin 在 u=0 被选中 | 是（level 0） | 否（抽到有质量的 level 3） |
  | u>1 / u<0 | 静默接受 | ValueError（提示 [0,1) 规范） |
  | CRN 缓存 key | `(shape, id(device))`，同分布两次调用可能 2 个 key | `(0,0,(2,6,40))` 稳定 1 个 key |
  | 同分布两次 sample 逐位一致 | 依赖 CPython 对象复用，时好时坏 | 恒定一致 |
  | sample_id 区分独立抽样 | 不存在 | 4 个 sample → 4 个 key，值互不相同且可复现 |
  | device 对象变化影响取值 | 会（id 变化→新种子） | 不会 |
限制与说明：
  · CRN 的 uniforms 固定 float32（仅用于反查 CDF，精度足够）；需要更细的 u 时由调用方决定。
  · 生成 CLI 的完整 smoke（真实 checkpoint + store）归 R14；本轮只固化契约与纯逻辑单测。
下个任务：R04
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R04
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/pairs.py
    · 新增 STYLE100_ACTIONS（BR/BW/FR/FW/ID/SR/SW/TR1/TR2/TR3）、STYLE100_ACTOR
      （=100style_actor_0）、STYLE100_ACTOR_SOURCE、split_style100_action、
      style100_action_family（TR1/2/3→TR）、is_style100_name。
    · labels_from_name(name, *, dataset=None)：仅当来源明确为 100STYLE（`100style/` 前缀或
      dataset="100style"）时把后缀当 action、把演员记为数据集级单演员标记；其他数据集
      只按名称分组 style，actor 保持 unknown（旧代码会把 FW/BR 当成演员名）。
    · clip_records_from_store(store, *, dataset=None)：显式 style/action/actor 列优先；
      style 标签不再被反解出演员；删除 source_group_names 冒充演员的回退。
    · build_pair_audit：performer_axis/style_axis 提前到使用它们的 warning 之前
      （修 UnboundLocalError 反例）；新增 style_vocabulary（configured/eligible/sampled 与
      三者计数）、same_style_evidence（含 single_content_label 原因）、pair_report、
      label_provenance、target_sampling/window_frames/dataset/held_out_styles 回填。
    · StylePairSampler：新增 window_frames、target_sampling、rejections/targets_skipped 计数、
      _reject_reason（same_clip/same_take/cross_split/heldout_style/window_unavailable/
      模式证据不足）、has_window、_target_reason、eligible_targets、pair_report、
      _target_order；held-out 过滤对所有 stage/mode/显式 targets 生效，target 的 split
      也会被校验；无窗口 clip 不再进入配对。
    · target_sampling：clip_uniform（全体 eligible clip 等概率）与 style_uniform
      （轮转调度：每轮给每个 style 一个槽位，再在该 style 内抽 clip），不复制稀有 clip。
  scripts/audit_style_pairs.py：--dataset/--window-frames/--held-out-styles/--target-sampling，
    输出 style_vocabulary 计数、pair_report、performer_analysis 与 warnings，并写
    label_provenance（actor 来源）。
  scripts/train_mts_operator.py：dataset 透传、window_frames=frames、target_sampling 从
    data.pairs 读取并透传到 PairedBatchSource，训练开始打印 pair coverage（style/action/
    actor 覆盖 + 拒绝原因），summary 记录 target_sampling 与 pair_report。
  data/configs/mts_operator_style.yaml、mts_operator_style_ah.yaml：data.pairs 增加
    target_sampling: style_uniform。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_pairs.py tests/test_mts_metrics.py tests/test_mts_end_to_end.py
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python     scripts/audit_style_pairs.py --feature-database data/processed/seed_soma_pruned_v4_ah     --window-frames 64 --output outputs/mts_pairs/audit_seed_ah_rev2 --sample-pairs 256
结果：exit 0；tests/test_mts_pairs.py 20 passed（含 8 个新验收用例）
  全量回归：429 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
  真实数据（SEED actor-holdout store，256 pairs，stage=train，same_style）：
    · clip_uniform：pairs_per_style={injured leg:7, injured torso:9, neutral:240}（多数风格吃掉 94%）
    · style_uniform：pairs_per_style={injured leg:85, injured torso:86, neutral:85}
    · style_vocabulary：configured=8、eligible=8、sampled=3（其余 5 个风格只有单一 content 标签，
      无法构成 same-style/different-content 证据 —— 审计明确给出原因，不再用 len(style_split) 冒充）
    · 拒绝原因计数：same_content 20311、same_clip 260、same_take 260、attempts_exhausted 301、
      targets_skipped.window_unavailable=1；unique_targets=256
    · performer_analysis=overlap_reported；警告含“neutral 占 92% 片段”“5/8 风格单一 content”
      “没有只属于留出演员的风格，因此仅支持 unseen-performer 轴”
反例前后对照：
  | 反例 | 修复前 | 修复后 |
  |---|---|---|
  | `100style/Flapping_FW` 的演员 | "FW"（把动作当演员） | 100style_actor_0（附 provenance） |
  | `Aeroplane_BR`（未知数据集） | 演员 "BR" | 演员 ""（未知保持未知） |
  | 风格 unset 的 store 里 style 名被反解演员 | `style_names` 再拆一遍拿 performer | 只读显式 actor 列，不反解 style |
  | source_group_names 当演员 | 会 | 不再使用 |
  | performer_axis 使用在其赋值之前 | UnboundLocalError | 提前构建并输出 overlap 警告 |
  | held-out style 出现在 train 参考 | 只在 split 解析成功时过滤 | 所有 mode/stage/显式 targets 统一过滤 |
  | 显式 targets 跨 split | 直接使用 | 拒绝并计数 cross_split |
  | 无窗口 clip | 参与配对 | 排除并计数 window_unavailable |
  | 实际采样的风格数 | 用 len(style_split) 代替 | sampled 计数来自真实抽样 |
  | TR1/TR2/TR3 统计 | 分散 | pairs_per_action_family 归为 TR |
限制与说明：
  · 本地 `combined_pruned_90/fsq_window_index` 只有 motion/ 没有 manifest，
    `feature_cache` 是 v3 预切分缓存（缺 split_manifest_hash），两者都不是可审计的 store；
    100STYLE 真实 store 的完整审计与训练接线留到 R14 用配置指向的 store 执行。
  · 100STYLE 的旧 10-style heldout 机制（manifest.unseen_style_names）与新的
    held_out_styles 是两套东西，R04 不合并它们，只在审计里并列展示。
下个任务：R05
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R05
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/windows.py
    · 从训练脚本搬入共用批来源（不再每个脚本复制一份）：TokenSource（clip→window，
      含 history/valid/meta）、PairedBatchSource（pair→对齐 batch）、WindowSample。
      每个字段都与“接受这个 pair”的同一个分支成对 append，窗口缺失只会消耗一个
      skipped 计数，不会让 style/action id 错位。
    · ContentVocabulary：content.kind=none|action_id + 训练集构建的排序动作词典；
      id_for 对未知/空动作直接报错（不借用 id 0）；as_dict/from_dict 往返；
      unconditional 属性标记“无条件”而不是假装指定了内容。
    · read_window_tokens 的在线编码路径改为“只用 clip 内可用前缀当上下文”：
      context = min(history, target_start - clip_offset)，取 tokens[context:context+frames]。
      旧实现用“重复首帧”补齐 history，与 token store 构建者（从 clip 首帧开始编码、
      由 encoder 自身处理序列前导）不一致，导致 clip 起点附近在线 token ≠ 存储 token。
      window 元数据记录 history_frames 真实使用了多少。
  stylized_motion/learning/mts_operator/transport.py
    · MotionTransportTransformer 接受 content_vocabulary 并写入 config()；与 content_classes
      交叉校验；operator/evaluator 从冻结 transport 继承词典而不是按当前 split 重建。
  stylized_motion/learning/mts_operator/model.py：OperatorBatch 增加 sample_metadata（非张量）。
  stylized_motion/data/loader.py：新增 loader.return_metadata（打开已有 dataset 能力），
    不新建第二套 DataLoader。
  scripts/train_mts_transport.py
    · build_sources 返回 (sources, vocabulary)：按 data.content.kind 决定是否打开
      return_metadata，从训练 split 记录构建词典；TokenSource 产出
      {tokens, content_condition}（动作来自 target clip，未知动作报错）；
      checkpoint 记录 content_vocabulary，resume 时校验一致。
  scripts/train_mts_operator.py：改用共用 TokenSource/PairedBatchSource（删除脚本内副本）；
    data.content.kind 校验、词典从训练集构建、对照冻结 transport 的词典并强制继承
    （不一致直接报错，缺 conditioner 也报错）；日志打印无条件/动作条件与类别数，
    summary 记录 content_condition（含 unconditional 标志）。
  scripts/evaluate_mts_operator.py：改用共用来源；content_condition 来自 transport 词典
    （删除 `index % 4` 的假 id）；新增 --content-kind，与 transport 的 kind 不一致时报错。
  scripts/generate_mts_operator.py：改用共用 TokenSource；动作条件来自词典与实际
    target clip 的动作标签，非 100STYLE/未标注时明确不条件化。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_pairs.py tests/test_mts_transport.py tests/test_mts_transport_training.py     tests/test_mts_end_to_end.py
结果：exit 0；tests/test_mts_pairs.py 25 passed（含 5 个 R05 新用例）
  全量回归：434 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
关键反例（独立判断，不依赖同一实现的 wrapper 互比）：
  · 对齐：fixture 中中间一个 pair 的 target 窗口缺失 → 仍得 2 行，tokens=[clip0,clip4]、
    reference=[clip1,clip5]、style_ids=[0,0]、content ids=[action0,action1]、
    metadata.reference_clip_id=[1,5]；skipped_pairs 计数 1→2。旧实现在 continue 处
    丢 pair 但用完整 pairs 列表生成 style_ids/ids → 错位。
  · 词典：同一字符串在两个实例/两个进程/序列化往返后 id 相同；未知动作 ValueError；
    content.kind=none 时 id_for 报错并标记 unconditional；kind=none 携带 classes 报错。
  · 条件真实生效：小训练 fixture 下同一 tokens、不同 action id 的 logits 最大差 > 1e-3，
    同一 id 两次前向逐位相同（条件不来自 reference）。
  · transport 词典随 checkpoint 往返；content_classes 与词典长度不一致时报错。
  · 在线编码 vs 全 clip 编码：window start=0 与 start=40 都与 whole[0:16]/whole[40:56] 逐位一致；
    “取前 T 帧”会不同；history_frames 元数据在起点窗口为 0、在 start=40 时为 40。
限制与说明：
  · content.kind 默认 none：任何“指定内容生成”的结论都必须在 kind=action_id 且词典
    一致的前提下才成立，输出里以 unconditional 显式标注。
  · 本轮不接 trajectory/phase 条件（首版决策）；unseen-action 泛化留待新的条件表征。
下个任务：R06
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R06
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/embeddings.py
    · StreamTokenEmbedding 重写为 plan 固定的结构：level embedding [B,T,40,E]
      → 仅 hidden token 用 learned mask embedding 替换 level 项（coordinate identity
      对 visible/hidden 都保留）→ 加 coordinate identity → 每个 stream 按布局顺序
      flatten K_s*E → 该 stream 自己的 Linear(K_s*E, D) + stream embedding → [B,T,13,D]。
      E 由 token_embed_dim 显式配置（DEFAULT_TOKEN_EMBED_DIM=16），不再借用 D。
    · 新增 StreamLevelHead：每个 stream 一个 Linear(D, K_s*9)，按 canonical slice
      index_copy 回 [B,T,40,9]，另保留每 coordinate 的静态 bias。
      全部 slice/索引来自 layout adapter（新增 stream_coordinate_indices/stream_sizes），
      不手写 40-coordinate 分组。
  stylized_motion/learning/mts_operator/transport.py
    · 使用新的 embedding 与 per-stream head（删除共享 output_head 与统一 coordinate_bias
      的用法）；新增 token_embed_dim/architecture_revision 构造参数与 config 字段；
      传入不匹配的 architecture_revision 直接报错（不 strict=False 静默加载）。
  stylized_motion/learning/mts_operator/layout_adapter.py：新增只读
    stream_coordinate_indices() 与 stream_sizes()。
  scripts/train_mts_transport.py、data/configs/mts_operator_transport.yaml：
    token_embed_dim 进入 TRANSPORT_KEYS 与配置（16）；summary 记录 token_embed_dim、
    architecture_revision、参数量。
  data/configs/mts_operator_style*.yaml：data.content.kind: none 显式写出（默认无条件）。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_transport.py tests/test_mts_end_to_end.py tests/test_mts_transport_training.py     tests/test_mts_pairs.py tests/test_mts_metrics.py
结果：exit 0；tests/test_mts_transport.py 22 passed（含 5 个 R06 新用例）
  全量回归：439 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
反例前后对照（GENO layout，dim=32，dropout=0，eval）：
  | 读数 | 修复前 | 修复后 |
  |---|---|---|
  | 同 stream 两 coordinate 交换后的 embedding 差 | 1.9e-09 | 8.5e-03 |
  | 同 stream 两 coordinate 交换后的 logits 差 | 1.19e-07 | 6.7e-02 |
  | hidden 位置的 level 改变 | — | 0.0（逐位相同） |
  | 改变“哪一个 coordinate 被 mask” | — | 6.2e-03（可区分） |
  | 两种 hidden context 下同 stream 两 coordinate 的 logits 差 | 仅等于静态 bias | 0.053 / 0.035（动态；静态 bias 为 0） |
  结构断言（非阈值）：_TwoStreamAdapter 下用确定性权重（identity 投影）验证
  “flatten+投影”的精确解析值，交换两个 level 得到精确的另一向量；hidden level 不可读但
  mask 位置可区分；per-stream head 在零静态 bias 下仍产生逐坐标差异。
  参数记录：E=16、revision=2；小模型 embedding 22112 / head 12240 / 合计 47120 参数；
  另有按 layout stream_sizes 推导的精确参数量断言。
限制与说明：
  · 架构 revision 升为 2：旧 MTS transport/operator checkpoint 的权重 shape 不同，
    不能加载（构造即报错）；R08 会在 checkpoint schema 层再固化一次。
  · 本轮不做 family tying（plan 指定），左右两侧各有自己的投影与 head。
下个任务：R07
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R07
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/temporal.py（新增，plan 允许的两个新小模块之一）
    · sinusoidal_positions(length, dim, *, device, dtype) -> [1, T, 1, D]：float64 计算后
      转目标 dtype，按需生成长度（无 64 帧上限，4096 帧可构造），只依赖窗口位置。
    · SinusoidalPositionEncoding(dim, kind="sinusoidal")：把表加到 [B,T,S,D]；
      kind 仅支持 sinusoidal（RoPE/可学习长表/temporal conv 按 plan 暂不引入）。
  stylized_motion/learning/mts_operator/transport.py、style_encoder.py
    · 两者在 temporal attention 前加共享位置编码；config() 记录 position_encoding；
      GlobalStyleEncoder 增加同名构造参数（可配置，默认 sinusoidal）。
  scripts/train_mts_transport.py、scripts/train_mts_operator.py：
    TRANSPORT_KEYS / REFERENCE_ENCODER_KEYS 接受 position_encoding。
  data/configs/mts_operator_transport.yaml、mts_operator_transport_ah.yaml、
  mts_operator_style.yaml、mts_operator_style_ah.yaml：显式写 position_encoding: sinusoidal。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_transport.py tests/test_mts_end_to_end.py tests/test_mts_transport_training.py     tests/test_mts_pairs.py tests/test_mts_metrics.py tests/test_mts_operators.py tests/test_mts_contract.py
结果：exit 0；121 passed（tests/test_mts_transport.py 28 项，含 5 个 R07 新用例）
  全量回归：445 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
反例前后对照：
  | 读数（deterministic seed，dropout=0，eval） | 修复前 | 修复后 |
  |---|---|---|
  | 全遮蔽时各帧 logits 的两两最大差 | 5.96e-08（结构上必然相等） | 1.083 |
  | 全遮蔽时各帧 logits 的帧间 std | 4.06e-10 | 0.183 |
  | hidden 位置换 token 数值 | 0.0（不变） | 0.0（仍然不变，符合设计） |
  | 200 帧长序列 | — | 正常前向（无硬上限） |
  | reference 描述符对帧重排的差 | 6.08e-06（≈数值噪声） | 6.57e-04（相对变化 > 1e-3） |
  | 描述符加入 masked padding 后的差 | — | 5.96e-08（有效帧统计不受影响） |
  | transport 时间置换等变性误差 | 3.58e-07（完全等变=位置无信息） | 0.954（不再等变） |
  | causal 前缀对未来 token 改动的差 | — | 0.0（严格前缀不变），未来帧 1.7e-02 |
  全相同帧的重排仍是合法例外（描述符逐位不变，容差 1e-6），没有写“任意重排必变”的断言。
限制与说明：
  · 位置编码是加性正弦表，不参与 pooling 的掩码逻辑；padding 仍由 key-padding mask 与
    valid 帧统计排除（已测：追加 padding 后有效输出/描述符不变，float32 容差 1e-5）。
  · 该改动同时改变了 transport 与 reference encoder 的权重语义 → revision 2 的 checkpoint
    与旧权重不兼容（R06/R08 的 revision 与 schema 检查负责拒绝）。
下个任务：R08
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R08
任务状态：passed
实际修改：
  stylized_motion/learning/mts_operator/checkpoint.py
    · schema_version=2、metrics_version=2（彼此独立）；保存时写 metrics_version；
      schema 1 一律拒绝，错误信息明确“旧结果仅供审计、需重训”。
    · save_mts_checkpoint 改为同目录临时文件 + os.replace 原子替换，失败清理临时文件
      （不再可能留下半个 best.pt）。
    · 新增 file_sha256 / code_identity / build_provenance：provenance 记录
      store_identity（feature/normalization/split/skeleton/representation_id + 路径）、
      action_to_id、style_to_id、resolved_config、seed、upstream_transport_sha256、
      training_protocol_id、metrics_version、code_commit、working_tree_dirty。
    · 新增 load_operator_bundle(path, *, adapter, tokenizer_identity, device)：
      从 operator 自身的 model_config + state_dict 重建整套 MTS 模型（transport 权重就在
      operator 文件里，推理不再依赖外部 transport 文件仍在原路径）；严格
      load_state_dict（无 strict=False）；校验 token alphabet（结构 + representation id）、
      layout_hash、encoder output_dim 与 operator style_dim/hidden_dim 的一致性；
      加载后保持 frozen 子模块 eval。
    · 新增 checkpoint_style_index / checkpoint_action_vocabulary（style-ID 映射与动作词典
      从 checkpoint 恢复，不由新 eval 数据排列重建）。
    · 新增 validate_store_binding(store, *, tokenizer_identity, expected_data_identity)：
      逐项核对 store 能报告的 identity；store 报不出某 identity 时直接报错，不默认放行。
  scripts/train_mts_operator.py、train_mts_transport.py：checkpoint 写入 provenance 与
    tokenizer_checkpoint_sha256。
  scripts/evaluate_mts_operator.py、generate_mts_operator.py：改用 load_operator_bundle
    （删除脚本内的模型重建白名单与外部 transport 依赖）；--transport-checkpoint 若给出则
    与记录的 SHA 交叉校验；动作条件来自 bundle 内 transport 的词典。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_checkpoint.py tests/test_mts_transport_training.py tests/test_mts_end_to_end.py
结果：exit 0；tests/test_mts_checkpoint.py 14 passed（新增文件，6 组验收）
  全量回归：459 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
验收覆盖（3 算子 × 2 encoder 的 6 组 round-trip）：
  · 同一 batch 的 probability 在 CPU eval 下 allclose；birth_death 的 level_order
    （identity 或 shuffled）与 shuffled_adjacency 逐项一致。
  · 把训练时的 transport 路径设为不存在（fixture 内断言 not exists）仍能加载并前向，
    frozen transport 保持 eval。
  · 错 tokenizer（receptive_field 不同）/ 跨骨架 layout / schema 1 / 缺 operator 权重的
    部分 state_dict 全部明确报错（后者触发 strict 的 RuntimeError，而非静默加载）。
  · style-ID 映射从 checkpoint 恢复：同一 batch 重排（含 ids 一起重排）后逐行输出不变。
  · store binding：三项 hash 与 representation_id 各自不匹配时报出字段名；
    store 缺 identity 时报“does not report”，不假定一致。
  · 原子写：state_dict 抛错时目标文件不存在且无残留临时文件。
  另有 provenance 字段完整性与 tokenizer_checkpoint_sha256 的独立断言。
限制与说明：
  · 权重 identity 首版用 checkpoint 文件 SHA-256（plan 指定，不另造 tensor hash）；
    仅重新封装导致 SHA 变化也会被拒，错误信息给出原因。
  · 旧 schema/旧 architecture 的 checkpoint 全部拒绝；不提供 strict=False、默认值补齐或
    路径替换等绕过方式。
下个任务：R09
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
当前任务：R09
任务状态：passed
实际修改：
  R09a 固定验证
    · 新增 stylized_motion/learning/mts_operator/eval_protocol.py（plan 允许的第二个新模块）：
      ValidationSample（split/clip/window start/mask kind/seed/content/style/actor/sample_id）、
      build_validation_samples（用独立的 validation RNG 抽 pair，窗口取首个可用起点，
      每项一个 SHA-256 派生的 mask seed）、ValidationProtocol（固定每类权重、counts、
      missing_kinds、只接受 finite+有监督的 objective）、ValidationBatchBuilder
      （按 start 读窗口、按 seed 造 mask、把 sample_metadata 带进 batch）、write() 落盘
      validation_protocol.json。
    · training.py：TransportTrainer._unpack 接受显式 mask/visible_mask（冻结协议不能每
      epoch 重抽 mask）。
    · train_mts_operator.py：val_batches 改为“冻结协议 + 固定 kinds/权重”，不再每 epoch
      抽一个随机 batch；best 只依据协议的 objective（finite、有监督计数>0），全空则不动
      best.pt 并打印 missing kinds；validation_protocol.json 与 summary 记录预算与种类。
  R09b 配置真执行
    · 新增 data/configs/mts_revision2_style.yaml / mts_revision2_transport.yaml（不覆盖旧配方）：
      style 配方固定 mixture（full_generation .30 / random_coordinate .30(coordinate_ratio .80) /
      stream .15 / temporal_span .10 / spatiotemporal_block .15）、required_data_schema_version 4、
      reference_frames==frames、precision fp32、target_sampling style_uniform、
      evaluation.validation_batches_per_kind 4、输出到 outputs/mts_revision2/。
    · validate_revision2_training：val_every_steps 只支持 0、precision 只支持 fp32、amp 只能
      False、未知 training 字段直接报错；evaluation 段未知字段/非法 kind 报错。
    · 未实现的 operator 选项不再打印 “ignoring options …” 后继续，而是报错。
    · CLI override 抽成 resolve_operator_config 并可单测：YAML→CLI→推导→校验→构建；
      多个 override 合并到当前 resolved 段（修掉“--hidden-dim 之后 --style-encoder-kind
      style_id 把宽度退回 YAML 值”的顺序缺陷）。
    · 新增 --dry-run：打印模型/参数量/style 与 action 映射/store identity/mask mixture/预算/
      evaluation 预算/output，写 dry_run.json，不训练不写 best。
  R09c warm-start
    · 新增 --warm-start（apply_warm_start 可单测）：只加载同 revision 权重，optimizer/step/
      best/RNG 全部从零；缺 key 或 shape 不符直接报错，不做部分加载。
    · 训练脚本不含 --resume / optimizer.load_state_dict，"resumed" 字样已清除；文档说明
      exact resume（RNG/sampler cursor/optimizer/scaler）本轮不实现、不伪造。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q     tests/test_mts_eval_protocol.py tests/test_mts_cli.py tests/test_mts_checkpoint.py     tests/test_mts_transport_training.py tests/test_mts_end_to_end.py
结果：exit 0；新增 tests/test_mts_eval_protocol.py 6 项、tests/test_mts_cli.py 7 项
  全量回归：472 passed / 1 skipped / 1 failed（仅既有 100STYLE 数据布局问题）
验收覆盖：
  · 同一 checkpoint 两次评估：objective/per-kind/counts 完全一致；不同模型不同值。
  · 协议构建不消耗训练 RNG：显式 generator 的序列与“未构建协议”的对照运行逐步一致。
  · 每类有固定子集与固定权重；空类进 missing_kinds，权重不转移；全空时 objective=None 且
    usable=False（best 不会被更新）。
  · 验证 batch 逐位可复现（tokens/visible_mask/sample_id 两次调用相同）。
  · CLI override 确实改变 resolved config（hidden_dim/batch_size/kind/num_styles），
    YAML 本身不被修改，缺 --num-styles 报错。
  · val_every_steps!=0、precision=amp、amp=True、未知 training 字段、未实现 operator 选项、
    reference_frames!=frames 全部报错。
  · warm-start 权重确实加载（权重逐位相等且确有变化），不同架构 revision 报 “do not fit”，
    脚本内不存在 resume 语义或 “resumed” 输出。
限制与说明：
  · 本轮只支持 epoch-end 验证；没有实现第二套 checkpoint 节奏。
  · AMP 未支持（明确报错）；precision=fp32 是首轮唯一选项。
  · dry-run 不写 best.pt，只写 dry_run.json。
下个任务：R10
```

```text
代码起点：e9aaceb（工作区仅有两个未跟踪文档：本次的 plan 与 audit，均保留）
## Integration Closure（MTS_FSQ_Agent_Integration_Closure_Plan_zh.md）

执行规范：[MTS_FSQ_Agent_Integration_Closure_Plan_zh.md](MTS_FSQ_Agent_Integration_Closure_Plan_zh.md)。
本轮不重做 R00，保留 R01–R11 的全部工作区改动（未提交，禁止 reset/clean/stash）。

### 门槛状态

| 门槛 | 要求 | 当前 |
|---|---|---|
| G-entry | C01–C05 完成，真实 tiny CLI、数据绑定与固定协议可执行 | **通过**（C09 矩阵 A/B/C1–C6） |
| G-code | 再完成 C06–C09，所有相关反例通过 | **通过** |
| G-ready | C10–C11 完成，预检与训练申请包齐全 | **data 级通过；experiment 级 blocked（无 revision-2 模型）** |

### 任务清单与状态

| ID | 内容 | 状态 |
|---|---|---|
| C00 | 起点基线与入口反例 | **passed** |
| C01 | 配置、构建顺序与 CLI 前置条件 | **passed** |
| C02 | 权重/数据/词表绑定真正执行 | **passed** |
| C03 | 逐样本固定协议、style-ID validation、完整 objective | **passed** |
| C04 | transport 训练入口与数据读取补齐 | **passed** |
| C05 | eval/generate 完整条件与 style-ID 接入 | **passed** |
| C06 | 正确 base 对照与多步 monotonic filling | **passed** |
| C07 | 物理评估接线、逐样本输出与 plot 版本检查 | **passed** |
| C08 | geometry/temporal probe 协议修订 | **passed** |
| C09 | 真 CLI 集成矩阵与回归收口 | **passed**（34 项矩阵；16 条真实缺陷修复） |
| C10 | 真实数据预检与曝光审计 | **passed**（data 级通过、experiment 级 blocked） |
| C11 | 状态文档与实验申请包 | **passed** |

### C00 记录

```text
任务：C00
状态：passed
原计划映射：R00/R14
修复事实：无实现改动。建立基线并写下四条真实入口反例（现为失败用例，C01/C02/C05 修复）。
反例（修改前失败 = 现在就是失败的，修复后必须通过）：
  F01 tests/test_mts_cli.py::test_revision2_configs_pass_the_real_masking_validator
      实测 ValueError: Unknown masking options ['span_frames']（两份 revision2 YAML 都命中）
  F03 tests/test_mts_cli.py::test_manifest_only_exits_zero_without_any_checkpoint
      实测 argparse 退出码 2：`the following arguments are required: --checkpoint, --tokenizer-checkpoint`
  F04 tests/test_mts_cli.py::test_row_batch_keeps_the_target_condition
      实测 build_row_batch 签名是 (model, row, target_tokens, candidates)，只拿 tokens 重建 batch；
      另以 OperatorBatch.supervision_mask 证明缺 visible_mask 的 batch 无法被评分
  F05 tests/test_mts_cli.py::test_bundle_load_rejects_a_same_shape_tokenizer_with_a_different_sha
      实测 TypeError: load_operator_bundle() got an unexpected keyword argument 'tokenizer_checkpoint'
      （即：SHA 只被写进 payload，加载时没有任何入口比对；同结构不同权重可被放行）
  F02（sampler 先用后定义）只有代码路径证据，其可执行反例归 C01（需要 tiny fixture），本轮未声称已复现。
验证：
  · outputs/mts_revision2_closure/C00/baseline_git.txt —— HEAD e9aaceb、33 个已修改文件、
    11 个未跟踪文件（含 plan/audit/progress 与两个新模块），工作区未清理
  · outputs/mts_revision2_closure/C00/baseline_41_tests.txt —— 计划 §1 的 41 项基线：
    exit 0；41 passed in 2.55s（与计划所述一致）
  · outputs/mts_revision2_closure/C00/c00_counterexamples.txt —— C00 反例运行：
    exit 1；4 failed / 7 passed，失败原因逐条指向 F01/F03/F04/F05
  · 全量套件（无 deselect）exit 1：6 failed / 478 passed / 1 skipped。六个失败 =
    C00 的 4 条反例（待 C01/C02/C05 修复）+ test_100style_fsq_excludes_last_ten_styles
    （本机外部数据布局，与代码无关）+ test_level_probe_reports_kinematics_and_rejects_bad_inputs
    （float32 FK 用 abs=0.0 比较，按计划归 C08）。
仍未验证：任何真实 CLI 的端到端执行（需要 tiny fixture，属 C01/C09）；GPU/CUDA；训练效果。
下一任务：C01
```

### C01 记录

```text
任务：C01
状态：passed
原计划映射：R05/R09/R10
修复事实：
  C01a 配置（data/configs/）
    · mts_revision2_style.yaml / mts_revision2_transport.yaml：删掉 plan 不支持的 span_frames，
      改用已有 span_ratio: 0.25（= 64 帧中的 16 帧，与原意一致，并加注释说明）。
    · tokenizer.checkpoint 改指 outputs/nef_fsq_soma_packed_40x9_ah/best.pt（holdout tokenizer）。
    · operator 的 transport.checkpoint 改指 outputs/mts_revision2/mts_transport/best.pt，
      不再引用旧 schema-1 的 outputs/mts_transport/seed_ah/best.pt（缺文件时直接报错，无回退）。
    · 主 style 配方 data.content.kind 明确为 action_id；另建
      mts_revision2_style_smoke.yaml（kind=none、epochs=1、validation_batches_per_kind=1），
      只作无条件 smoke，不用于“指定内容”的主张。
  C01b 入口（scripts/train_mts_operator.py）
    · 构建顺序改为：解析配置→store/records→style_split→pairs 配置→StylePairSampler→
      eligible targets→动作词表→style 映射→batch source→trainer，消除 F02 的
      “sampler 先用后定义”（现在 sampler 在 499 行、词表在 520 行）。
    · 动作词表以冻结 transport 的词典为准：允许 operator 只用其子集，但任一使用的 action
      必须在 transport 词典中，否则报错（不重新编号，不新增 id）。
    · style-ID 映射改由 _trainable_style_index 构建：仅包含既能训练（非 held-out）
      又能形成合法 pair 的 style；num_styles 与实际可训练 style 数不一致时报错。
  C01b 入口（scripts/evaluate_mts_operator.py）
    · --checkpoint/--tokenizer-checkpoint 改为条件必需；--build-manifest-only 在任何
      tokenizer/模型加载之前分支，只用 store 的 clip 表与窗口元数据（不编码 motion），
      缺 store 时给出明确 error。
    · 自动构造 manifest 时 region 用 list(args.support or [])，不再有 list(None) 风险；
      --support 的语义在 help 里写明“省略或空 = whole body，不是空编辑”。
反例（修改前失败 → 修改后通过）：
  · F01 test_revision2_configs_pass_the_real_masking_validator：改前
    ValueError: Unknown masking options ['span_frames']；改后通过。
  · F03 test_manifest_only_exits_zero_without_any_checkpoint：改前 argparse 退出码 2
    （required: --checkpoint, --tokenizer-checkpoint）；改后通过。
  · 新增 test_manifest_only_writes_a_manifest_from_a_real_store：真实 tiny token store +
    无任何 checkpoint，subprocess 退出码 0 并写出 eval_manifest.json/.jsonl（“no model was loaded”）。
  · F02：sampler 先用后定义的代码路径已消除（顺序重排）。可执行的 tiny-fixture 版本
    要求 tokenizer/transport checkpoint fixture，属 C09 case C1–C6，本任务不声称已执行。
验证：
  · outputs/mts_revision2_closure/C00/c01_progress.txt —— tests/test_mts_cli.py：
    exit 1；10 passed / 2 failed（余下两条正是 F04/F05，归 C05/C02）
  · 定向回归 exit 0：tests/test_mts_pairs.py + test_mts_eval_protocol.py + test_mts_metrics.py
    + test_mts_checkpoint.py = 59 passed
  · 全量套件（无 deselect）exit 1：4 failed / 481 passed / 1 skipped。四个失败 =
    F04（C05）+ F05（C02）+ test_100style（外部数据布局）+ test_nef_probe 容差（C08）。
仍未验证：operator/transport 的真实两步训练（需要 C09 的 fixture）；GPU/CUDA；训练效果。
下一任务：C02
```

### C02 记录

```text
任务：C02
状态：passed
原计划映射：R08
修复事实：
  · bundle 侧（R08 的补缺）：checkpoint.py 新增 require_tokenizer_checkpoint(...)，
    load_operator_bundle 增加 tokenizer_checkpoint / tokenizer_checkpoint_sha256 /
    require_tokenizer 参数并默认强制校验；缺 SHA 记录或缺参数都报错，不一致时报出两个 SHA。
    train_mts_operator 的 apply_warm_start / --warm-start 传入冻结 tokenizer 的路径。
  · store 侧：新增 require_token_store_binding(store|path, *, tokenizer_checkpoint |
    tokenizer_checkpoint_sha256, checkpoint=None, where)：token store 的
    manifest.checkpoint_sha256、真实 tokenizer 文件 SHA、(加载模型时) MTS checkpoint 记录的
    SHA 三方一致；store 没记录 SHA 或缺参数都报错。新增 STORE_IDENTITY_FIELDS 按 store kind
    区分必查身份：token store 需 feature/normalization/split/representation_id，
    feature store 需 feature/normalization/split/skeleton（不强制 tokenizer 身份，
    避免把无关字段变成新阻塞）。
  · 四个入口接线：train_mts_operator / train_mts_transport / evaluate_mts_operator /
    generate_mts_operator 在打开 store 后立即调用（token store 走三方校验，feature store 走
    身份校验），并在失败时于任何前向/训练步之前退出。
  · 曝光记录（第 6 条）：checkpoint.py 新增 training_exposure(...)，两个训练脚本把
    trainable_styles / held_out_styles / actions / pairs_per_style 写进 provenance；
    无可报告信息时 exposure_unknown=True（不是空集合冒充）。evaluator 的 --held-out-styles
    现在与 checkpoint 记录交叉核对：记录缺失时报“records no training exposure”，
    不一致时报“disagrees with the checkpoint's recorded held-out”。
  · code identity（第 7 条）：新增 CODE_DIGEST_PATTERNS + source_digest(root, patterns)，
    对 mts_operator/*.py、data/*.py、scripts/*mts*.py、revision2 配置逐文件 SHA-256 并给出
    合并 digest（42 个文件），code_identity() 一并写入 provenance；只读显式 glob，
    不收集无关/敏感文件，不要求先 commit。
反例（修改前失败 → 修改后通过）：
  · F05 test_bundle_load_rejects_a_same_shape_tokenizer_with_a_different_sha：改前
    TypeError（不接受 tokenizer_checkpoint，同结构不同权重可放行）；改后 ValueError 拒绝，
    用训练时的文件可通过。
  · test_token_store_binding_checks_the_three_way_hash：改前 ImportError（无 helper）；
    改后 store 记录的 SHA 与 tokenizer 文件不符时报“same structure is not the same weights”，
    与记录一致时通过；feature store 不被强制要求 representation_id。
  · test_source_digest_distinguishes_two_uncommitted_revisions：改前 helper 不存在；
    改后同一文件两次内容的 combined digest 不同，声明外的文件（secrets.env）不被读取。
  · test_training_exposure_is_recorded_and_checked：改前无曝光记录；
    改后 exposure_unknown 与记录字段可区分，evaluator 的声明必须与记录一致。
验证：
  · outputs/mts_revision2_closure/C00/c02_final.txt —— 全量套件 exit 1：
    3 failed / 485 passed / 1 skipped。三个失败 = F04（C05）+ test_100style（外部数据布局）
    + test_nef_probe 容差（C08）。F01/F03/F05 全部转为通过。
  · 定向：tests/test_mts_cli.py + test_mts_checkpoint.py + test_mts_eval_protocol.py +
    test_mts_pairs.py + test_mts_metrics.py = 73 passed / 1 failed（仅 F04）。
仍未验证：真实 store 的三方一致（真实 tokenizer/token store 需 C10 预检执行）；GPU/CUDA；
  训练效果。feature store 的 model-space 一致性只做了身份字段校验，未做数值对齐测试。
下一任务：C03
```

### C03 记录

```text
任务：C03
状态：passed
原计划映射：R09/R10
修复事实：
  · 逐样本 mask：ValidationBatchBuilder 现在每个 sample 用自己的 seed、以 B=1 调 mask generator
    再 stack（不再“首行 seed 生成整个 batch”），因此 batch_size 不再改变任何一行的 mask。
  · 行语义：ValidationSample 增加 mask_config（完整 ratio/block 配置），manifest 保存它；
    build_batch 在 row 的 config 与本次运行的 config 不一致时明确报错（不静默覆盖）。
  · style-ID validation：builder 接 style_index / encoder_kind，按 sample.style 生成 style_ids，
    未知 style 报 KeyError（列出可用集合）；valid_mask 来自 WindowSample（新增
    TokenSource.window_at 与 valid_mask_for，短片段/尾部窗口的真实有效性不再被当成整帧有效）。
  · objective：正权重 kind 必须全部存在、supervised>0 且 nll_sum finite 才 usable=True；
    否则 objective=None 并列出 missing_kinds / invalid_kinds（缺一类不再让指标变小、更容易被
    best 选中）；权重必须 finite 非负且至少一个正值，构建时归一化一次并固化（sum=1）。
  · metrics._row 改为按字段语义切分（_BATCH_AXIS_FIELDS + dataclasses.replace）：
    hard_mask [T,K] 不会因 T==B 被误当 batch 轴，metadata/strength 同步切分。
  · train_mts_transport 的 train_records 为空时直接报错，不再 `or records` 把 val/test 混入词表。
反例（修改前失败 → 修改后通过）：
  · test_a_manifest_row_means_the_same_thing_at_any_batch_size：改前 TypeError（builder 不接受
    style_index）+ 首行 seed 生成整批 mask；改后 B=1/2/4 下每行 visible_mask 与 tokens 逐位相同。
  · test_validation_builder_carries_style_ids_and_valid_masks：改前无 style_ids、valid_mask 恒为
    全 True；改后 style_ids 逐行等于映射值，padding 窗口的 valid 位正确。
  · test_objective_is_unusable_when_a_weighted_kind_is_missing_or_not_finite：改前缺一类仍返回
    较小的 objective 且 usable=True；改后 missing/invalid（NaN、inf、零计数）一律 objective=None。
  · test_row_slicing_keeps_field_semantics：改前 T==B 时 hard_mask 被切成 [1,40]；改后保持 [T,40]。
验证：
  · outputs/mts_revision2_closure/C00/c03_progress.txt —— 全量套件 exit 1：
    3 failed / 490 passed / 1 skipped（失败仍是 F04、100STYLE 外部布局、nef_probe 容差）
  · 定向：tests/test_mts_eval_protocol.py 15 passed；tests/test_mts_metrics.py 全绿
仍未验证：真实 store 上按 manifest 重放（B 改变）需要 C09 的 fixture；GPU；训练效果。
下一任务：C04
```

### C04/C05 记录（本轮完成部分）

```text
任务：C04
状态：passed（验证协议、恢复语义、冻结精确性；pending 尾巴已在同一轮收口：
  loader 固定 mapping 输出、transport --dry-run、resolved config 即实际配置、在线/缓存数值比对见 C09-D）
原计划映射：R05/R09
修复事实：
  · eval_protocol.build_target_only_samples：base 模型（transport）的冻结验证集不再要求
    same-style pair，窗口-only、逐行 mask；train_mts_transport 用它构造 ValidationProtocol
    并写 validation_protocol.json。
  · transport 每 epoch 只评估一次：on_epoch_end 里的 validation_report 同时供日志、
    checkpoint metrics（val_objective/val_per_kind/val_counts/val_missing_kinds/protocol_hash）
    与 best 决策使用，不再二次评估。
  · best 只由 frozen protocol 的 objective 决定：不可用（缺失/NaN 类、无监督 token）时不更新
    best.pt 并打印原因，绝不回退 train loss；无验证的 overfit 模式写 overfit_last.pt 并声明
    不是 validation-best。
  · --checkpoint 改为明确迁移错误（exact resume 未实现），新增 --warm-start：只载权重，
    optimizer/step/best/RNG 从零；校验 action 词表一致；脚本内不再出现 resumed /
    optimizer.load_state_dict。
  · TokenSource.freeze(clips=N) 精确截断到 N 个窗口（含随附字段），不足时报错；
    不再把跨越阈值的整批（默认 128 窗）留下。
  · --val-batches 改为 --val-rows（冻结协议的行数）。
反例：test_transport_cli_refuses_checkpoint_and_offers_warm_start（改前 --checkpoint 静默
  “resumed transport”；改后非零退出并指向 --warm-start）；
  test_transport_freeze_keeps_exactly_the_requested_windows（改前 freeze(7) 留下 10 窗及以上，
  改后恰好 7 且随附 content_condition 同步截断）。
验证：outputs/mts_revision2_closure/C00/c05_progress.txt（全量 exit 1：2 failed / 493 passed /
  1 skipped，只剩 nef_probe 容差与 100STYLE 外部布局）

任务：C05
状态：passed（完整 batch 复用、条件、style-ID；pending 尾巴已收口：逐样本数值行、dataset 透传、
  style label 不再充当 content condition）
原计划映射：R10/R11
修复事实：
  · build_row_batch(model, row, batch, candidates) 改为复用完整 batch（[_row] 切一行，
    只替换 reference 与 reference_valid），不再用 tokens 重建无 visible/condition 的 batch。
  · style-ID 模型不再进入 reference retrieval 循环（而不是跑完再标 N/A）；
    generate 新增 --style-label（经 checkpoint 的 style_to_id 映射，未知标签报错），
    style-ID 模式不要求 --style-clip，reference 模式必须给。
  · content_condition 始终来自 target/source 的 action；style_ids 始终来自 style 映射；
    evaluator 默认从 checkpoint 读 content kind，--content-kind 只作对账（不一致报错）。
  · manifest 的 region/radius/frame_range 在构建 batch 时生效（hard_mask + visible=~region），
    同一 batch 的所有指标共用同一条件；同一 batch 混两个 region 直接报错。
反例：F04 test_row_batch_keeps_the_target_condition（改前 signature 只有 tokens，重建的 batch
  无法被 supervision_mask 接受；改后要求 row+batch 并复用完整条件）。
仍未验证（继续做）：eval_rows.jsonl 的实际数值（当前仍只写描述字段与计数）、dataset 标签在
  正式评估路径的透传、E 的 shuffled CTMC 集成用例。
下一任务：C06
```

### C06/C07/C08 记录（C07/C08 的 pending 尾巴已在同轮收口，见下）

```text
任务：C06
状态：passed
原计划映射：R03/R12
修复事实：
  · C06a：strength_response 以 model(batch).base_probabilities 为基准（不再用 operator λ=0 的
    分布；arbitrary kernel 的 λ=0 是 uniform，用它当基准等于自比），并新增命名明确的
    base_argmax_vs_styled_argmax_ratio 诊断与 refresh_only_tv（λ=0 相对真正 base 的 TV）。
    曲线每点带 comparison="base_transport_vs_styled"，λ 之间不混平均。
  · C06b：sampling.monotonic_fill_steps(remaining, steps) 给出每步提交块（每样本取
    ceil(剩余/剩余步数)，按 (time, coordinate) 顺序），并保证每个位置恰好提交一次、步数大于
    位置数也合法；fill_remaining 按 (sample_id, step) 分格共享 uniforms。
  · C06b：model.generate_edit 支持 steps>1 的迭代生成——每一步用已提交 token 重跑模型
    （visible = 已提交 ∪ 区域外锁定），只在该步的提交块写入，逐位保证区域外/初始 visible/
    padding 不变；return_trace=True 返回每步提交掩码。
  · C06c：generate CLI 新增 --steps，保存 commit_trace.npy，summary 记录 steps。
反例：test_monotonic_fill_commits_every_position_exactly_once（改前无该 helper；改后
  1/2/3/5/40 步都覆盖全部位置且无重复、后面步不大于前面步）；
  test_monotonic_fill_draws_are_shared_per_step_and_ordered（同 sample 同 step 逐位可复现、
  每步一个 CRN cell、不同 sample_id 独立）。
仍未验证：真实模型上 steps>1 与 steps=1 的一致性（需 C09 的 fixture）；迭代生成的效果指标。

任务：C07
状态：passed（comparison 拆分、plot 版本门槛；CLI 三对比接线已收口，见该节末尾）
修复事实：
  · metrics.comparison_physics(source, base, styled) 分别输出 source_to_base / base_to_styled /
    source_to_styled 三组物理指标并各自带 comparison 字段，禁止把它们平均成一个数。
  · plot_mts_figures 新增 EXPECTED_METRICS_VERSION/check_artifact_version/load_operator_artifact：
    没有 metrics_version 或版本不符的旧结果被明确拒绝并给出原因与文件名，不再悄悄拼进图。
反例：test_plot_refuses_artifacts_from_another_metrics_revision（改前无门槛，旧 JSON 可直接进入
  图表数据；改后 payload=None + 明确 reason）。
收口（同轮）：evaluator 现在对同一 batch 生成 source（目标 tokens）/base（冻结 transport 分布，
  `generate_edit(use_base=True)`）/styled（配置 strength，可多步）三条动作并分别解码，
  `comparison_physics` 的三块写进 physics.csv 与 summary.physics_per_comparison；
  base/styled 共用同一 CRN cell，逐样本行同时记录 base_vs_styled 的 changed_token_ratio 与 TV。
  仍属 C10 范围：解码 warmup/prefix 只在真实长序列评估时需要，本轮未启用。

任务：C08
状态：passed（容差按量纲修复；protocol v2 已收口，见该节末尾）
修复事实：
  · tests/test_nef_probe.py 的 fk_offtarget_max 从 abs=0.0 改为按量纲的 1e-4 m，并加
    “小于 owned 影响 0.1%” 的相对约束：world-space FK 是 float32 流水线，实测底噪约
    1e-6 m；feature 层的结构性零仍保留严格检查（该用例其余断言未放宽）。
反例：该用例本身（改前 8.99e-07 触发 abs=0.0 失败；改后 9 passed），并保留独立正确性断言。
收口（同轮）：协议 v2 落地——`signed_pulse_perturbations` / `signed_span_perturbations` /
  `shared_legal_support` / `influence_profile` / `stratified_span_probe`，near 与 far 在同一
  source/坐标/时间 support 且只在共同合法帧上比较；±1 与 ±d 分层；`temporal_probe_complete`
  按 decoder RF 与窗口尾部判定；feature 影响与 root 积分 world tail 分开上报（单位分别为
  mean|Δdecode| 与 m）；probe artifact 记录 protocol_revision=2、support/排除计数、RF、frame range、
  单位与 seed，并写入**新文件** probe_stratified.{json,csv}，不覆盖历史 probe_geometry.*。
  真实数据实跑 2 窗口 × 13 坐标 × 4 扰动（C08/stratified/）：far 的 world tail（实例 0.0099 m）
  明显大于同坐标 near（0.0019 m），且 far 的 feature 影响按距离分层。
```

### R12–R14 复核记录（按 R 计划逐条核对当前代码）

```text
复核方式：不是读旧标签，而是逐条把 R12/R13/R14 的验收项在**当前代码**上跑一遍；
本轮补上缺口后记录。R01–R11 的结论仍由全量套件（542 passed）与本文件的历史记录支撑。

R12（一致的多步 masked generation）—— passed
  已有：monotonic_fill_steps（每样本 ceil(remaining/steps_left)，固定 (time, coordinate) 顺序）、
  generate_edit(steps>1) 每步用已提交 token 重跑模型（visible = 已提交 ∪ 区域外锁定）、
  steps=1 等价单步、超出可编辑位置数的 steps 合法、区域外/初始 visible/anchor 不变、
  CRN 按 (sample_id, step_id) 分格，generate CLI 的 --steps 与 commit_trace.npy。
  本轮新增：
  · tests/test_mts_end_to_end.py::test_iterative_filling_never_peeks_at_uncommitted_tokens
    —— 两个只在"最后一步才提交"的位置上不同的 batch，第 1 步提交逐位相同（若模型提前看到答案，
    第 1 步就会变化）；N=1/2/8 各自同 seed 可复现；steps=8/200 时区域外仍与 source 逐位相同。
  · evaluator summary 新增 protocols 块，把 fixed teacher-forced mask NLL 与 iterative
    generation（schedule=monotonic_frame_coordinate, steps=n）分开命名。
  证据命令：pytest -q tests/test_mts_end_to_end.py -k iterative（1 passed）

R13（geometry/temporal 测量公平性）—— passed
  已有（C08 轮）：协议 v2（signed pulse / fixed signed span / 共享合法 support / ±1 与 ±d）、
  temporal_probe_complete、feature 与 world tail 分开、新产物 probe_stratified.* 不覆盖历史。
  本轮新增：
  · temporal_influence_width 读模型自身的 decoder_receptive_field 计算影响宽度（不再写死 33），
    并输出 decoder_receptive_field / temporal_probe_complete；新增 decoder_influence_frames() helper。
  · geometry 报告标注 protocol_revision=1、far_perturbation="random_per_frame_stress"，
    并写明 ordinal 比值只是 screen，公平比较在协议 v2 的 stratified 探针。
  · evaluate_nef_locality.py 支持 --graph-radius（默认 [0,1]）：radius 0 与 radius 1 的
    support 内 edit magnitude 与 support 外影响并排输出（flat/part 无邻接契约 → 记 not_applicable
    并给出原因，不假装 r1 等于 r0）；artifact 记录 protocol_revision=2 与 graph_radii。
  真实数据实跑（2 窗口，left_arm，edit[16,48)）：
    left_arm@r0 support=10% 坐标，edit_feature_mean=0.831，off_target=0，non_target_joint=0
    left_arm@r1 support=15% 坐标，edit_feature_mean=0.703，off_target=0，non_target_joint=0
  证据：outputs/mts_revision2_closure/R13/locality/{locality.json,locality.csv}

R14（集成 smoke、只读预检、文档收口）—— passed
  R14a：tests/test_mts_cli_matrix.py（34 项，见 C09 记录）+ tests/test_mts_cli.py。
  R14b：scripts/preflight_mts_revision2.py（见 C10 记录）。本轮补齐两项：
    · mask_distribution：真按配方采样 500 个 mask，报告 realised 每类比例与 hidden fraction
      （style 配方：declared {full 0.30, random 0.30, block 0.15, stream 0.15, span 0.10}；
      realised {0.302, 0.322, 0.144, 0.156, 0.076}，hidden_fraction=0.630）
    · budget 增加有效 token 预算：supervised_tokens_per_step = hidden_fraction × B × T × 40
      = 51606.8；steps 未在配方中固定时写 null + reason，不打印 0 冒充估计。
      同时报告 code_commit 与 source_digest（44 个文件）。
  R14c：README 增补能力边界表（unseen style / multi-style / content loss / AMP / exact resume
    逐项写明"拒绝或未实现"），阶段文档增补勘误，主入口链接新 config/protocol 与进度记录。
  证据：outputs/mts_revision2_closure/C10/preflight_experiment/preflight.json
```

### C09 记录（真 CLI 集成矩阵）

```text
任务：C09
状态：passed
新增：tests/test_mts_cli_matrix.py（34 项）、tests/test_mts_cli.py 的 tiny store 扩展
  （可绑定真实 tokenizer SHA / motion_dim / style_ids / action_ids / clip_lengths / split_ids）、
  tests/test_mts_pairs.py 的输入空间 ground truth 修正。
矩阵覆盖（每条都实跑，不是搜索文本）：
  A  --build-manifest-only：真实 tiny token store，无任何 MTS checkpoint
  B  transport：2 步 → 冻结 val → 保存 → 回读；--dry-run 只做身份/构建预检；
     错 tokenizer SHA（文件）/ 错 store SHA 各一条反例
  C1–C6  3 operator × {reference, style-ID}：2 步 → 验证 → 保存/回读 → eval（共用同一 manifest）
      → generate（steps=3 + commit_trace）；同一 manifest 以 batch_size=2 与 1 各评一次，
      逐行 NLL 1e-4 内一致，mask seed/窗口/候选集合完全相同
  D  真实 holdout tokenizer/tokenstore 三方绑定；feature 在线读 vs token store 同窗口数值比对
      （CPU 27/20480=0.0013 边界量化差异；CUDA 上逐位一致）
  E  shuffled CTMC：level_order 是置换、保存加载后 probabilities 逐位相同、生成元行和为零
  F  错 hash / 空 val（val 划分无法配对）/ 非法 style label / 未知配置键 / content-kind 冲突
      → 非零退出且不写 best.pt，也不留下伪 metrics
fixture 特性：窗口起点非 0（含 tail 窗口）、hard mask 有 anchor、style id 与 action id 数值刻意不同、
  词表刻意不同（style 4 个 / action 2 个，最后一个 style id 超出动作范围以暴露字段混用）。
本轮发现并修复的真实缺陷：16 条，逐条写进 outputs/mts_revision2_closure/C09/c09_defects.txt
  （最严重一条：packed feature store 的在线编码输入空间错误，同一窗口 71% token 不一致）。
验证：
  · <py> -m pytest -q tests/test_mts_cli_matrix.py → exit 0；34 passed
  · <py> -m pytest -q → exit 1；541 passed / 1 skipped / 1 failed（既有 100STYLE 外部布局）
  · 证据：outputs/mts_revision2_closure/C09/{README.txt,c09_matrix.txt,c09_matrix_cases.txt,
    c09_defects.txt,c09_full_suite.txt}
```

### C10 记录（真实数据预检）

```text
任务：C10
状态：passed（data 级通过、experiment 级 blocked 且如实标注）
新增：scripts/preflight_mts_revision2.py、tests/test_mts_preflight.py（6 项）
两级检查（每项 status/reason/evidence，异常不会中断报告，逐项落进 preflight.json）：
  data：tokenizer 可加载与 identity、store 三方 SHA 绑定、schema、归一化/骨架/划分身份、
        train/val/test 在 actor/take/mirror 三轴隔离、标签表完整、style×action 配对矩阵
        （含拒绝原因与 single-content styles）、真实窗口读取（≤ --max-windows）、
        manifest 泄漏（same take / 同一 crop / 合法负例不足单独计数）
  experiment：transport 加载与绑定、动作词表 ⊆ transport 词表、冻结协议（版本/权重和/样本数/哈希）、
        operator 曝光（与上游 tokenizer/transport 分开报告）、operator checkpoint 上游绑定、
        预算与输出目录（不得覆盖历史 run）
真实数据验收：
  · holdout tokenizer + v4-ah tokenstore → exit 0；ok=true；actor 标签经 **clip 名对齐**目录表
    得到 142220 个 actor 标签，train/test actor 重叠 0（exposure reported）
  · 旧 1h tokenizer + 同一 store → exit 1；store_identity fail
    （“the tokens were created by tokenizer e45…, but the supplied tokenizer is 485…;
      same structure is not the same weights”）
  · 窗口上限：--max-windows 2 → clips_read=2；--max-windows 0 → exit 1
  · 预检不写 best.pt、不启动训练（测试断言 tmp 目录内无 best.pt）
  · experiment 级在无 revision-2 transport 时全部 blocked（不伪造 pass）
证据：outputs/mts_revision2_closure/C10/{preflight_ah,preflight_stale_tokenizer,preflight_experiment}/
```

### C11 记录（叙述修正与实验申请包）

```text
任务：C11
状态：passed
1. 历史 passed 全部保留（R00–R11 记录未删改），顶部换成 Integration Closure 的实际收口状态。
2. 更正以下表述（原文保留在各自记录里，仅顶部说明现状）：
   · “R11 除 plot 均完成”→ plot 的 metrics_version 门槛在 C07 落地；R11 的物理指标在 C07 才真正接进 CLI。
   · “transport resume 已消除”→ 是，`--checkpoint` 明确报迁移错误、`--warm-start` 只载权重（C04）。
   · “固定协议已 batch 无关”→ C03 之前不是（按 batch 造 mask），C03 起逐行 mask，C09 用 B=2/1 复验。
   · “SHA 已绑定”→ C02 之前只写入不校验，C02 起加载必校验，C09 在真实数据上验证。
   · “历史配方未动”→ 不成立：4 个历史配置在上一轮被追加字段（见
     docs/mts_revision2_experiment_request_zh.md 的差异表），不能直接 checkout 覆盖。
3. 旧工件分类：NEF tokenizer 与既有 representation/physics 结果按原协议保留（不因 MTS 的 NLL 变化
   笼统作废）；但**feature 在线编码**路径的旧数值需按修正后的实现重跑；旧 MTS transport/operator
   （schema 1 / 旧 architecture revision）只能作审计材料，需按 revision 2 重训。
4. 新实验清单：data/configs/mts_revision2_experiment_manifest.yaml（family/encoder/data/protocol/
   hash/seed/步数/output + 命令），只生成不执行；申请包与状态模板见
   docs/mts_revision2_experiment_request_zh.md。
5. 吞吐测量方法：以 20 步短跑读 train_summary.json 的 history.seconds 与 global_step 得到 step/s，
   再乘计划步数；不沿用旧的“3 分钟一模型”估计（架构与有效验证量都已变化）。
6. README / mts_operator_stage_status.md 增补勘误条目，指向本文件与申请包。
```

## R01 记录（已通过）

```text
实际修改：
  stylized_motion/learning/mts_operator/contract.py
    · 新增 _supervision_mask / _reduce_selected / _selected_targets / _compute_dtype 四个共用 helper；
      mask 只在这一个地方生成，计数是 bool mask 的元素数而不是平均比例。
    · masked_cross_entropy(logits, targets, *, valid_mask, coordinate_mask, reduction="mean"|"sum")
      只接 logits，空集合返回 logits.sum()*0（保留梯度）；targets 先筛选再 gather，
      invalid sentinel 在未监督位置不再触发越界，监督位置仍做范围检查。
    · 新增 masked_nll_from_probs(probabilities, targets, *, …, clamp_min=1e-12)：-log(p_target)。
  stylized_motion/learning/mts_operator/model.py
    · MtsStyleOperator.loss 全部走 probability 路径（masked_nll_from_probs），
      metrics 提供 nll / nll_sum / supervised_tokens / correct_tokens / supervision_fraction；
      content_weight != 0 现在显式报错（冻结 base 的旧 content 项已删除）。
  stylized_motion/learning/mts_operator/training.py
    · TransportTrainer.loss: 同一监督 mask 产出 correct_tokens 与 supervised_tokens，并返回 nll_sum。
    · TransportTrainer.evaluate / OperatorTrainer.evaluate: 按 sum(nll_sum)/sum(supervised_tokens)
      聚合，空输入返回 loss=None + 计数（不再返回 NaN）。
    · OperatorTrainer.train_step: supervised_tokens == 0 的 batch 不 backward、不 step，
      记 skipped=1；操作数计数写入 record。
    · 两个 fit(): epoch loss 用 token 加权精确值（不再平均 batch means），并输出
      supervised_tokens / correct_tokens / skipped_steps 累计计数；日志对 None 安全。
  stylized_motion/learning/mts_operator/metrics.py
    · content_preservation 迁移到 masked_nll_from_probs（base_nll / styled_nll / nll_increase 语义不变）。
  stylized_motion/learning/mts_operator/__init__.py
    · 导出 masked_nll_from_probs。
  scripts/train_mts_operator.py、scripts/train_mts_transport.py
    · on_epoch_end 的 best 记录在 val 指标缺失/None 时跳过并打印，空验证集不能成为 best；
      checkpoint metrics 额外记录 supervised_tokens / skipped_steps。
  tests/test_mts_contract.py、tests/test_mts_transport_training.py、tests/test_mts_end_to_end.py
    · 新增 R01 验收测试（见下），删除「content_weight 使标量变大即通过」的旧断言。
验证命令：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python -m pytest -q \
    tests/test_mts_contract.py tests/test_mts_transport_training.py tests/test_mts_end_to_end.py \
    tests/test_mts_metrics.py
结果：exit 0；45 passed（R01 定向测试全绿）
  全量回归：404 passed / 1 skipped / 1 failed
    - 预先存在的失败：tests/test_preprocess_pipeline.py::test_100style_fsq_excludes_last_ten_styles（本机 100STYLE 扁平布局）
    - 预先存在的失败：tests/test_nef_probe.py::test_level_probe_reports_kinematics_and_rejects_bad_inputs
      （float32 FK 比较用 abs=0.0，实测 8.99e-07；已用 git stash 验证与本次改动无关，留待 R13 探针协议 v2 处理）
关键反例（R01 独立验收，全部落在测试里）：
  · p_target=.99 → NLL 0.01005033585；logits-CE 路径同一输入给 1.3803597688674927（审计反例固化）
  · 9 类 uniform → NLL = log 9
  · 随机 logits：CE(logits) 与 NLL(softmax(logits)) 数值相等，梯度最大差 < 1e-11（float64）
  · 恒定每 token NLL=2、监督比例 5%，evaluate 仍报 2（旧实现报 0.1）
  · 拼接/拆分 batch、追加无效 padding 帧：总 NLL 与总计数完全不变
  · uniform 算子（arbitrary kernel 零初始化）在 5% 与 95% 监督比例下 evaluate 都报 log 9
  · 全空监督：不 step optimizer、参数逐位不变、global_step 不变、evaluate 返回 None + counts、best 不更新
限制与说明：
  · 旧 MTS checkpoint/指标的 NLL 不可与新协议比较（旧 operator loss 是 probability-as-logits 的 CE，
    且 validation 是 batch-mean 平均）；R01 不重训、不删除旧 checkpoint。
  · 算子侧指标仍名为 content_preservation，但它度量的是「相对冻结 base 的 NLL 代价」，不是 content 保真；
    名称与输出字段的收口归 R10/R11。
下个任务：R02
```

## R00.1 审计反例清单（执行时的独立 oracle）

`outputs/audit_20260917/minimal_probes.json` 的读数与归口任务：

| 读数 | 值 | 说明 | 归口 |
|---|---|---|---|
| `probability_as_logits_loss` | 1.3804 | 把 probability 当 logits 送进 CE | R01 |
| `true_nll` | 0.010050 | p_target=.99 的解析 NLL | R01 |
| `reported_val_for_true_loss_2_and_mask_fraction_005` | loss=nll=0.1 | 真 NLL=2、监督比例 5% 却报 0.1（按 batch mean 平均） | R01 |
| `within_stream_swap_logit_max_difference` | 1.19e-07 | 同一 stream 内两个 coordinate 交换几乎不改变 logits（pooling 顺序无关） | R06 |
| `embedding_difference` | 1.86e-09 | 同上，embedding 层 | R06 |
| `default_generation_active_support` | 0.0 | 默认 mask 下 generation 的可编辑支持为 0 | R02/R03 |
| `crn_identical_distribution_same_draw` | false / cache=2 | 同一分布两次 `crn.sample` 结果不同（cache key 不稳定） | R03 |
| `all_masked_logits_frame_max_diff` | 0.0 | 全遮蔽时各帧 logits 完全相同（无时间位置信息） | R07 |
| `style_descriptor_time_permutation_diff` | 1.39e-06 | style descriptor 对时间重排近乎不变 | R07 |
| `transport_time_permutation_equivariance_error` | 4.77e-07 | transport 对时间置换等变（同样是缺位置编码的后果） | R07 |
| `actor_overlap_audit_error` | UnboundLocalError | `build_pair_audit` 引用未赋值的 `performer_axis` | R04 |
| `same_content_heldout_pairs` | [['b','a'],['b','a']] | `held_out_styles` 未覆盖 target，held-out style 仍被采样 | R04 |

## R00.2 旧 MTS 工件失效清单（只记录，不移动/删除/覆写）

| 工件 | 失效原因 | 处置 |
|---|---|---|
| `outputs/mts_operator/sandbox/{logit_field,arbitrary_kernel,birth_death}`、`sandbox_birth_death_shuffled`、`reference/{birth_death,logit_field}` | CLI 覆盖静默失效（style-ID/shuffled/dim/batch 未生效）→ 期望的 style-ID sandbox 与 shuffled 对照根本不存在；且 loss/计数协议在本轮修订中被替换（R01），其指标不可比较 | 原地保留；新入口默认拒绝旧 MTS schema（R08）；文档标注“仅用于审计，需重训” |
| `outputs/mts_transport/seed_ah`、`outputs/nef_fsq_soma_packed_40x9_ah`、`data/processed/seed_soma_pruned_v4_ah*` | 训练协议（loss/mask/计数）将在本轮变更；tokenizer/store 的 40×9/13-stream contract 不变，其**表征身份**仍可复用，但**训练指标**不可与 revision 2 直接比较 | 原地保留；R14 预检里区分“可复用的数据/表征绑定”与“不可复用的训练指标” |
| `outputs/stage3_eval*`、`outputs/mts_figures`、`outputs/mts_matrix`（空） | 基于旧 loss/指标协议 | 原地保留；plot 侧在 R11 加 metrics_version 校验 |

### T04（Gate）：开训门槛复核与源码冻结 — 已完成

**六项门槛（全部 pass；工件 `outputs/training_readiness_audit_20260918/training_gate.json`）**

| 门槛 | 状态 | 依据 |
|---|---|---|
| `train_val_disjoint` | pass | 真实协议 320 行全为 val（独立复核：`clip_split==1`、train/val take group 交集 0）；T00 反例测试与预检的逐行 store 复核 |
| `content_chain_consistent` | pass | transport/operator 主配方均 `action_id`；算子继承冻结词表（子集可用、不重编号）；真实 train 词表 17 类，val 中 5680 个 clip 的 action 不在词表内 → 排除并计数 |
| `transport_sha_bound` | pass | `load_mts_checkpoint` 三处调用点全部绑定 tokenizer 文件 SHA；同结构异权重反例；真实 SHA 左右相等 |
| `preflight_required_pass` | pass | 真实四 stage 实跑（data/transport_train ready=true exit 0；operator_train/evaluate ready=false exit 1 且 `ok` 仍为 true）；tiny 正/负路径 + 说谎协议反例 |
| `budget_explicit` | pass | 9 份配方全部固定整数预算（config 测试断言）；无预算拒绝启动；shortfall/wall-cap 记录且不宣称完成；输出目录防覆盖 |
| `tiny_real_chain_pass` | pass | dry-run → preflight(ready) → 2 步 style-ID 训练 → eval → generate 全链路 exit 0（默认设备 CUDA）；矩阵 6 组算子/encoder 全绿 |

**T04 复核内容与证据**

1. 测试集合（证据 `T04/gate_set.txt`、`T04/regression_set.txt`、`T04/full_suite.txt`）：
   · gate set（matrix + preflight + eval_protocol + operators）→ exit 0，**95 passed**（计划文档中的
     “73 passed”是本轮之前的口径，T00–T03 的新用例使数量上升）；
   · 回归集合（cli/checkpoint/end_to_end/transport/transport_training/metrics/pairs/sampling/contract/nef_probe）
     → exit 0，**172 passed**；
   · 全量套件 → exit 1，**570 passed / 1 skipped / 1 failed**，唯一失败仍是既有的
     `tests/test_preprocess_pipeline.py::test_100style_fsq_excludes_last_ten_styles`（本机 100STYLE 扁平布局）。
2. 真实数据只读审计（≤8 窗口；`T04/real_data_audit.py` → `T04/real_data_audit.json`）：
   · tokenizer SHA `e4507ff2…` 与 store 记录的 `checkpoint_sha256` 相等；store representation 一致；
   · split：train 113,883 / val 14,232 / test 14,105 clips；
   · **训练词表 17 类（不是 20）**；val 有 3 类 action 不在词表（Other 1837、Sports 3823、Martial Arts 20），
     test 有 2 类；这些 clip 被排除并记录，不借用 id；
   · **style 覆盖：8 个 style 中只有 3 个具备 same-style/different-content 证据**
     （`injured leg` / `injured torso` / `neutral`），另 5 个只有单一 content 标签；
     `trainable_pairable_styles` 正是这 3 个 —— 与计划预期一致，其余 style 不得写成“已训练”；
   · `held_out_styles` 为空 → unseen-style 轴目前**不受支持**（不是“已测为零”）；
   · pair 审计警告：token store 无 actor 表（performer 分析需 catalogue 对齐）、neutral 占 92%（多数 style 支配）、
     5/8 style 单 content、无 style 专属 held-out actor；
   · val 窗口读取 8 个（cap 生效），形状均为 [64,40]、store split 均为 val。
3. tiny 全链路（`T04/chain/`，产物 `outputs/mts_revision2/t04_tiny_chain/operator_chain/`）：
   dry-run(exit 0，写出协议) → `preflight --stage operator_train`(exit 0，ready=true) → 2 步训练(exit 0，
   global_step=2，best_val_nll=2.2826) → `evaluate_mts_operator.py`(exit 0，逐样本 NLL base/styled/delta、
   physics 三对比、style-ID 的 retrieval=N/A) → `generate_mts_operator.py`(exit 0，tokens/motion/commit_trace)。
   期间发现并修复一个真实设备缺陷：evaluator 的 style-ID 分支把 `style_ids` 建在 CPU 上，在 CUDA 上
   embed 直接崩；已改为建在 compute device 并复跑通过（矩阵用 `--device cpu` 才没暴露它）。
   另：style-ID checkpoint 未传 `--style-label` 时的报错改为明确提示（此前是模型层泛化的
   “needs batch.style_ids”）。
4. 源码冻结（`T04/freeze/`）：`worktree.patch`（tracked 变更）、`untracked_files.json`（24 个未跟踪源文件 +
   sha256）、`source_digest.json`（50 个文件的逐文件 sha256 与合并摘要）、`code_identity.json`
   （commit `d2be657`、dirty=true）、`config_sha256.json`（10 份配方）、`environment.json`
   （python 3.13.15 / torch、numpy 版本、CUDA 可用与设备名、解析设备）。是否提交留给用户；
   “未提交”不阻塞，因为工件绑定的是这套 dirty 源码的逐文件摘要。

**研究声明**：本轮没有任何研究效果声明 —— 未训练 revision-2 transport/operator（只有 tiny 2 步链路与 CPU 矩阵），
不声称 style 效果、content 保持数值或任何 per-kind objective。链路上 1e-3 量级的 NLL 差是 2 步模型，不构成证据。

**下一步**：`outputs/training_readiness_audit_20260918/training_gate.json` 六项全 pass，可申请 GPU 预算
开跑 S0（transport profile，20 步）与 S2（transport pilot，2,000 步）；所有 operator arm 依赖 pilot checkpoint。

### 勘误（2026-09-18 晚，NEF-FSQ 可视化检查发现，已修复）

症状：角色渲染的蒙皮/骨骼错位（颈部塌陷、头陷进胸腔、手只剩残端），以及所有 **世界坐标** 物理指标
建立在一个退化骨架上。

定位与证据：
- `MotionFeatureStats.ref_pos` 是**数据集 local positions 的均值**；packed store 由原始 + 镜像 clip 构成，
  镜像翻转横向轴，任何“恒定且沿轴”的偏移在均值里抵消。实测 `data/processed/seed_soma_pruned_v4_ah` 的
  `ref_pos`：`Spine1/Spine2/Chest/Neck2/Head` ≈ 0（真值 0.05–0.08 m），`Neck1` ≈ 0.005（真值 0.263 m），
  而 `LeftShin` 0.433、`LeftFoot` 0.423 等镜像下不变号的关节保持正确。
- 逐 clip 对照：`store.clip_position_sum[clip] / clip_length` 与 `_process_motion_data()` 的输出
  在 <1e-6 m 内一致（`Hips` 除外，它会动）→ store 的**逐 clip** 数据是对的，坏的只是聚合出的 `ref_pos`。
- 与 `data/assets/somaview/SOMA_bind.bvh`（厘米，÷100 后为米）逐关节对照：26 个非 Simulation 关节
  全部一致到 <1e-6 m → bind 骨架即正确骨架。
- 影响面：渲染（`build_database_from_feature_array`、本 viz）、FK 世界指标
  （`KinematicContext.ref_pos` → `evaluate_nef_locality.py`、`evaluate_mts_operator.py` 的 physics、
  `probe_nef_geometry.py`、`nef_eval` 的 world positions）。token/特征空间指标不受影响。
  NEF 训练不受影响：`nef_fsq_soma_packed_40x9_ah.yaml` 的 `joint_weight=0.0`（未用关节 FK loss）。

修复：
- 新增 `stylized_motion.anim.features.bind_reference_positions(names)` 与
  `stats_with_reference_skeleton(stats, names)`（缺失关节跳过、不可用返回 `None`，绝不臆造骨架）；
- 新增 `stylized_motion.learning.nef_probe.reference_positions_for_fk(feature_stats)`，
  并给 `KinematicContext.from_feature_stats` 增加 `reference_positions=` 参数（默认保持原值，
  合成夹具不受影响）；上述脚本全部改为传入 bind 骨架；
- `scripts/visualize_nef_fsq.py` 默认使用用户既有渲染资源
  （`--resources-root data/assets/somaview_quinn` + Quinn 贴图），并修正相机目标（改用 FK 后的**全局**位置，
  此前从 local positions 取“身高”，而 SOMA 的脊柱沿局部 x 轴，导致相机对准膝盖）；
- `anim/features.build_motion_feature_components` 增加 contacts 宽度校验（契约是 [T,2] 左右脚趾；
  传入 [T,J] 会静默产出错误宽度向量）。

复核（全部实跑）：
- `tests/test_nef_probe.py::test_bind_skeleton_replaces_the_mirror_averaged_reference`：
  镜像平均骨架下 Hips→Head 距离为 0，bind 骨架下恢复到 authored 0.605 m；`tests/test_nef_probe.py` 24 passed。
- MTS 全套（cli/matrix/preflight/checkpoint/eval_protocol/end_to_end/transport/transport_training/
  operators/metrics/pairs/sampling/contract/nef_probe）→ exit 0，**268 passed**。
- R13 locality 用修正骨架重跑：`outputs/mts_revision2_closure/R13/locality_bindfix/`。
  所有“0”结论（descendant / non-target / root / contact flip）不变；`target_joint_change` 0.18897→0.18893
  （相对 2e-4）；`edit_feature_mean`（特征空间）不变。
- 可视化重跑：`outputs/nef_fsq_viz/{locomotion,jumping_jack,sweep40}/`，数字以
  `docs/nef_fsq_visual_check.md`（已同步修正）为准：缓速 clip 关节误差均值 0.054 m（root 0.009），
  大幅 clip 0.100 m（root 0.006），36-clip 抽样均值 0.093 / 中位数 0.052 / p90 0.220 / 最差 max 0.834 m。
- T04 tiny chain 的 physics 数字不受影响（该夹具的 stats 是合成 2 关节骨架，helper 返回 `None` 走回退路径），
  复核后逐项一致（`outputs/training_readiness_audit_20260918/T04/chain/eval_bindfix/`）。

## 收尾（Final Delivery，2026-09-18）

本轮（Training Readiness T00–T04 + NEF-FSQ 可视化与勘误）到此结束。最终验证与交付工件如下。

**验证命令与结果（全部实跑，证据在 `outputs/training_readiness_audit_20260918/T04/final/`）**

| 集合 | 命令 | 结果 |
|---|---|---|
| gate set | `pytest -q tests/test_mts_cli_matrix.py tests/test_mts_preflight.py tests/test_mts_eval_protocol.py tests/test_mts_operators.py` | exit 0，**95 passed** |
| regression set | `pytest -q tests/test_mts_cli.py tests/test_mts_checkpoint.py tests/test_mts_end_to_end.py tests/test_mts_transport.py tests/test_mts_transport_training.py tests/test_mts_metrics.py tests/test_mts_pairs.py tests/test_mts_sampling.py tests/test_mts_contract.py tests/test_nef_probe.py` | exit 0，**173 passed** |
| full suite | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q` | exit 1，**571 passed / 1 skipped / 1 failed**（唯一失败为既有的 `test_100style_fsq_excludes_last_ten_styles`，本机 100STYLE 扁平布局，与代码无关） |
| 真实数据 preflight（按 stage） | `--stage data` / `transport_train` / `operator_train` / `evaluate` | data、transport_train：ready=true exit 0；operator_train、evaluate：ready=false exit 1（required blocked，`ok` 仍为诊断字段） |
| 真实数据只读审计（≤8 窗口） | `outputs/.../T04/real_data_audit.py` | exit 0；tokenizer/store SHA 相等；train 词表 17 类；8 个 style 中 3 个有 same-style 证据；val 窗口 8 个形状 [64,40] |
| tiny 真实链路 | dry-run → preflight(ready) → 2 步 style-ID 训练 → eval → generate | 全链 exit 0（默认设备 CUDA） |
| NEF-FSQ 可视化 | `scripts/visualize_nef_fsq.py`（两个 clip + 36-clip sweep） | exit 0；工件在 `outputs/nef_fsq_viz/` |
| R13 locality（修正骨架后） | `scripts/evaluate_nef_locality.py ... --output .../R13/locality_bindfix` | exit 0；结论不变（见「勘误」） |

**门槛**：`outputs/training_readiness_audit_20260918/training_gate.json` → `gate: pass`，
六项 `train_val_disjoint / content_chain_consistent / transport_sha_bound / preflight_required_pass /
budget_explicit / tiny_real_chain_pass` 全 pass；工件已刷新为**当前** dirty 源码
（commit `e9aaceb`、source digest `5ad610b7…`、50 个文件、8,750 insertions / 1,114 deletions 的 tracked diff 与
28 个未跟踪源文件哈希），并记录首次写入后的两处勘误（渲染资源、ref_pos 骨架）。

**交付分级**
- **G-code：通过** —— 真实入口（transport/operator/preflight/eval/generate/可视化/探针）在 CPU 与 CUDA 上实跑通过；
  全部反例（错 SHA、错 split、缺依赖、说谎协议、短跑、超时、覆盖、constant 臂带 reference 键）都有测试。
- **G-ready（data 级）：通过** —— 数据身份、split、词表、style 证据、窗口、mask 分布、预算与协议冻结全部有证据。
- **G-ready（experiment 级）：blocked** —— 未训练 revision-2 transport；六个 operator arm 依赖它。
- **研究效果：未验证** —— 不存在任何风格/内容/物理效果的声明。

**未完成项（等待用户决定，本 agent 不自行启动）**
1. S0 transport profile（20 步）与 S2 transport pilot（2,000 步）的 GPU 预算授权；
2. 之后的 style-ID logit vs no-reference logit（S3）、style-ID CTMC 与 reference logit（S4）；
3. unseen-style 轴（需要一个被排除出算子训练的 style，目前 `held_out_styles` 为空）；
4. 100STYLE 本机目录布局（外部数据问题，不影响 SEED 主线）。

### 勘误 2（同日更晚）：镜像 clip 需要镜像的参考骨架

背景：修复 `ref_pos` 之后，非镜像 clip 渲染正确，但**镜像 clip**（`clip_mirror=True`，名字带 `_M`，
store 里约一半）仍然错位——躯干折下、头落到髋部以下，视觉上就是"蒙皮/骨骼不对"。

定位与证据（可复现）：
- 该 store 的镜像 clip 链偏移，与 `_process_motion_data(..., mirror=True)` 的一致约定是
  **左右名字互换 + 横向 x 取反**：脊柱链（`Spine1/Spine2/Chest/Neck2/Head`）x 变号，
  四肢链（`LeftArm/LeftShin/LeftToeBase/...`）因为名字已经互换而**保持符号**。
- 判定用 pipeline 自己的 FK 真值（`_process_motion_data(mirror=True)` 的 rotations/positions）：
  头 y 1.571、脚 y 0.014；
  · 用未镜像参考重建 → 头 y 0.38（倒过来），对真值误差 0.648 m；
  · 用镜像参考重建 → 头 y 1.57、脚 y 0.01，对真值误差 0.230 m——与非镜像 clip 用未镜像参考的误差**相同**。
- store 的镜像 clip 128115（jumping_jack_002__A033_M）修正后：头 y 1.58、脚 y 0.01（此前 0.44 / 0.24）。

修复：
- `anim.features.bind_reference_positions(names, mirror=True)`（左右伙伴名 + x 取反，**不是**整体 x 翻转）、
  `mirror_partner(name)`、`stats_with_reference_skeleton(..., mirror=...)`、
  `nef_probe.reference_positions_for_fk(..., mirror=...)`；
- `scripts/visualize_nef_fsq.py`：按 clip 的 `clip_mirror` 选择参考骨架（summary 记录
  `clip_mirror` 与 `reference_skeleton=bind_bvh|bind_bvh_mirrored`，montage 标注 `[mirrored clip]`）；
  sweep 每行记录 `mirror` 并分组统计（36 clip 中 15 个镜像）；
- `scripts/evaluate_nef_locality.py`：按窗口选择上下文（artifact 记录 `mirror_flags`）；
- `scripts/evaluate_mts_operator.py`：physics 按 batch 内的镜像分组分别 FK，每行记录 `mirrored` 与 `samples`
  （原先整批共用一个上下文，镜像行会得到错误的世界坐标）。

复核（全部实跑）：
- 回归测试新增 `tests/test_nef_probe.py::test_mirrored_clips_need_the_mirrored_reference_skeleton`
  （断言"伙伴名 + x 取反"而非整体翻转，脊柱取反、四肢保号）；`test_nef_probe.py` 全部通过。
- 可视化重跑：`outputs/nef_fsq_viz/{locomotion,jumping_jack,sweep40}/`。修正后读数：
  缓速 clip 均值 0.054 m；镜像 jumping jack 0.096 m（p95 0.231 / max 0.396）；
  36-clip 抽样均值 0.094 / 中位数 0.049 / p90 0.229 / 最差 max 0.798（镜像组 0.129、非镜像组 0.068）。
  与上一版（未处理镜像）相比，抽样均值 0.0936→0.0936、中位数 0.0516→0.0494、p90 0.2202→0.2293、
  最差样本 0.834→0.798：**差异主要来自镜像样本的关节被正确摆放**，结论方向不变（误差集中在末端、且是位姿而非轨迹）。
- R13 locality 重跑 `outputs/mts_revision2_closure/R13/locality_mirrorfix/`（该次两个窗口都是镜像 clip）：
  `target_joint_change` 0.1890（上一版 0.1889），其余（descendant / non-target / root / contact flip）仍为 0。
- T04 tiny chain 的 physics 与改动前逐项**完全一致**（该夹具无 `clip_mirror`，走回退路径），
  证据 `outputs/training_readiness_audit_20260918/T04/chain/eval_mirrorfix/`。

补充（渲染 vs 指标）：镜像参考骨架修好了**姿势**，但把镜像骨架摆到未镜像的 SOMA 网格上仍会拧皮肤——
链式骨骼沿镜像轴（局部 x）排列，镜像后它们绕自身轴滚 180°（证据：修复前的
`outputs/nef_fsq_viz/jumping_jack/source_frame_042.png`，躯干/颈部被拧细）。这不是数据错误，
是"镜像姿势 + 未镜像蒙皮"的固有冲突。处理：
- `scripts/visualize_nef_fsq.py` 改为渲染**同一 take 的未镜像变体**（`source_group` 相同、`clip_mirror=False`、
  长度一致；summary 记录 `requested_clip`/`unmirrored_variant_of`），画面与数字一致；例如
  clip 128115（`_M`）→ 渲染 140158（同一 take 的原始变体）；
- **指标**不受影响：它们在关节/矩阵域，用镜像参考骨架即可（R13 locality、sweep 的镜像行即如此）。
  重跑后的 jumping jack（未镜像变体）读数：token 定点率 0.856、特征误差 0.148、
  关节误差均值 0.098 m / p95 0.231 / max 0.382，最差关节 LeftFoot/LeftToeBase/LeftHand。

## 追加交付（同日更晚）：9 段序列可视化 + 勘误 4（误差分解）

用户要求“再提供一些其他序列的可视化结果”。新渲染 7 段 test clip（加上此前的 `locomotion`、`jumping_jack`
共 9 段），坐标 `outputs/nef_fsq_viz/`；读数与看图结论写在 `docs/nef_fsq_visual_check.md`
（“更多序列”一节）。

**新增工件（本轮）**

| 工件 | 内容 |
|---|---|
| `outputs/nef_fsq_viz/{walk_forward,dance_hiphop,gesture_no_speak,household_blender,injured_leg_kick,sit_legs_crossed,stunts_postmortem}/` | 每段 `compare_frames.png` / `compare.gif` / `token_map.png` / `coordinate_error.png` / `joint_error.png` / `summary.json` / `source_raw.npy` / `recon_raw.npy` |
| `outputs/nef_fsq_viz/contact_sheet.png` | 9 段总览（三列：source / round trip / difference×4，行标题带该次运行读数） |
| `outputs/nef_fsq_viz/error_split.json` | 9 段的 pose / placement 分解 |
| `scripts/nef_fsq_contact_sheet.py`（新） | 从各 run 的 `summary.json` + still 拼总览图（每行按自身角色裁剪缩放） |
| `scripts/nef_fsq_error_split.py`（新） | 用逐帧 Kabsch 对齐把关节误差拆成 pose 与 placement；读已存 `source_raw.npy`/`recon_raw.npy`，不需重渲染 |

**读数（test split，窗口 64 帧，同一 holdout tokenizer）**

- 关节误差均值：`household_blender` 0.014、`gesture_no_speak` 0.021、`walk_forward` 0.033、
  `locomotion` 0.054、`injured_leg_kick` 0.080、`dance_hiphop` 0.090、`jumping_jack` 0.098、
  `sit_legs_crossed` 0.138、`stunts_postmortem` 0.388 m（max 0.823）。
- 长片段的**第一个窗口常常是静止起势**，因此舞蹈/走路/家务等 5 段改用片段中段窗口（`--start`）；
  用默认窗口会得到“站着不动”的四张 still，读数也偏低（`dance_hiphop` 特征误差 0.016 → 0.199）。
- 两个镜像请求（`sit_legs_crossed` 128118、`stunts_postmortem` 128161）按勘误 3 渲染成同 take 的未镜像变体
  （137113 / 132177），`summary.json` 记录 `requested_clip`/`unmirrored_variant_of`。
- 9 段渲染**全部**没有蒙皮/骨架异常（无扭曲、无塌陷、无缺头缺手）。

### 勘误 4：`root/local` 分解分不开“整体朝向错”与“位姿错”

`anim` 侧 `joint_error_stats` 的 local = 逐关节误差减去 root 关节自身的位移，**只去掉平移**。
整段身体被重建到错误朝向时，误差会落进 “body-local”，被误读为“40×9 码本表达不了这个姿态”。
本次 9 段用逐帧 Kabsch 最优刚体对齐重新分解，结果（`outputs/nef_fsq_viz/error_split.json`）：

| run | raw | pose（对齐后残差） | placement（占比） | 骨盆高 src→rec | 躯干主轴与竖直夹角 src→rec |
|---|---|---|---|---|---|
| `stunts_postmortem` | 0.388 | 0.077 | 0.311（80%） | 0.192 → 0.383 m | 94.4° → 121.9° |
| `sit_legs_crossed` | 0.138 | 0.052 | 0.086（62%） | 0.141 → 0.105 m | 20.5° → 5.5° |
| `jumping_jack` | 0.098 | 0.040 | 0.057（59%） | 0.992 → 0.971 m | 5.9° → 7.1° |
| `dance_hiphop` | 0.090 | 0.037 | 0.053（58%） | 0.945 → 0.934 m | 25.5° → 25.5° |
| `injured_leg_kick` | 0.080 | 0.036 | 0.044（55%） | 0.981 → 0.980 m | 23.7° → 19.9° |
| `locomotion` | 0.054 | 0.042 | 0.013（23%） | 0.974 → 0.996 m | 3.1° → 4.4° |
| `walk_forward` | 0.033 | 0.015 | 0.018（55%） | 0.983 → 0.983 m | 11.3° → 11.7° |
| `gesture_no_speak` | 0.021 | 0.020 | 0.001（5%） | 0.996 → 0.997 m | 4.8° → 4.6° |
| `household_blender` | 0.014 | 0.012 | 0.002（15%） | 0.996 → 0.996 m | 4.7° → 5.4° |

结论修正：本批 9 段的 **pose 残差都 ≤ 0.077 m**，所以“码本表达不了极端姿态”在这批里不成立；
误差大的片段是 **root 朝向流**的问题（躺地抽搐的躯干被转了 +27°，骨盆抬高 0.19 m）。
`docs/nef_fsq_visual_check.md` 的读数要点 1 已就地更正并指向本条。
对既有的 sweep 结论无影响（sweep 的 `mirror` 列、分组统计、以及“误差集中在末端”的关节分布都不变），
但**绝对归因**要按本条读。

**门槛刷新**：`outputs/training_readiness_audit_20260918/training_gate.json` 的
`source_freeze`（未跟踪文件哈希 28 → 30，新增两个可视化脚本；代码 digest `04f7ccc7…` 不变——
`CODE_DIGEST_PATTERNS` 不含可视化脚本）与 `errata_after_first_write`（追加第 4 条）已刷新；
测试证据 `T04/final/{gate_set,regression_set,full_suite}.txt` 复跑覆盖。

---

## 第二轮复核与推进（N00–N03、N05a，2026-09-18 夜）

依据：`docs/MTS_FSQ_First_Round_Results_and_Next_Plan_zh.md`。本轮**没有启动任何优化步骤**：所有结论来自只读重算、几何 oracle、数据审计、生成基准与单批梯度诊断。所有新工件在 `outputs/mts_next_round_20260918/`，旧 run、旧 verification JSON、旧 checkpoint 一律保留。

交付物索引：

| 工件 | 路径 |
|---|---|
| 统一只读评分入口 | `scripts/summarize_mts_operator.py` + `stylized_motion/learning/mts_operator/summary.py` |
| 四臂 213 行重算 | `outputs/mts_next_round_20260918/N00/four_arms_{summary.json,rows.jsonl}` |
| 物理真值 oracle | `scripts/check_mts_physics_truth.py` + `N01/physics_truth.json` |
| 共享 FK context | `stylized_motion/learning/mts_operator/physics_context.py`（`physical_metric_version=1`） |
| content schema | `data/configs/mts_content_schema_v1.yaml` + `content_schema.py` |
| 覆盖表与 dev benchmark | `scripts/audit_mts_content_style.py` + `N02/{coverage.json,benchmark_manifest.json}` |
| editing benchmark | `scripts/benchmark_mts_editing.py` + `N03/{benchmark.json,benchmark_rows.csv,flipbooks/}` |
| reference 机制诊断 | `scripts/diagnose_mts_reference_encoder.py` + `N05a/reference_diagnosis.json` |
| 阶段记录 / 决策 | `stages/{N00,N01,N02,N03,N05a}.json`、`screening_decision_v2.json` |
| 训练申请（未批准） | `N04_application.json`、`N05b_application.json` |
| 回归反例 | `tests/test_mts_{summary,physics_context,content_schema}.py`（30 例） |

### N00：统一只读评估入口

E04–E06 的三套临时脚本已收敛到 `scripts/summarize_mts_operator.py`。修正的语义缺陷（每条都有回归反例）：

1. **漏传 action**：旧脚本从 `model_config["content_vocabulary"]` 读词表——该键不存在，于是所有臂都在无条件下评分。现在词表取自 `model.transport.content_vocabulary`；条件存在而 condition 为空时**直接拒绝**，只有显式 `--omit-action-ablation` 才允许，并把协议标签写成 `action_omitted`。
2. **三种聚合混用**：micro / macro_row / protocol_weighted 现在各自成键，禁止跨口径拼表；`nll_improvement_vs_*` 正数表示前者更优（旧 `correct_minus_wrong` 实为 `wrong-correct`）。
3. **行集不一致**：wrong-reference 聚合只覆盖 143/213 行，却与 213 行的数字并列。新增 `paired_style_input_comparison`，只在两键都存在的行上比较，并报逐行分布。
4. **NaN**：写出前 `require_finite` 拒绝非有限值，未计算项写 `null`。
5. **seed/预算**：从 checkpoint provenance/metrics 读取，删掉 hardcoded `same_seed=True`。

四臂 213 行、正确条件（protocol weighted，行集全部 213）：

| 臂 | styled | wrong style input | base（冻结 transport） | micro | macro_row |
|---|---:|---:|---:|---:|---:|
| style-ID logit | **1.538102** | 1.550529 | 1.566107 | 1.708839 | 1.543687 |
| constant | 1.543817 | — | 1.566107 | 1.715610 | 1.549477 |
| reference logit | 1.545247 | （143 行）| 1.566107 | 1.716649 | 1.550843 |
| style-ID CTMC | 1.556606 | 1.561002 | 1.566107 | 1.730293 | 1.561992 |

`objective_reproduction` 对四臂全部通过（重算与 checkpoint 记录的 `val_objective` 最大差 7.6e-6，容差 1e-5），四臂 transport 权重逐 tensor 相同，逐行 base 分布一致。**行集对齐后的 style-input 效应**：

| 臂 | 行数 | 正确 vs 错误输入（nat/token） | 逐行 \|Δ\| 均值 | 正确更优的行 |
|---|---:|---:|---:|---:|
| style-ID | 213 | **+0.012427** | 1.90e-2 | 157/213 |
| CTMC | 213 | +0.004396 | 8.22e-3 | 157/213 |
| reference | 143 | **−9.4e-11** | **2.54e-08** | 12/143 |

reference 臂的"错误 reference 反而更好"是行集拼接的假象：逐行交换真实错风格 reference 只改变 1e-7 量级的 NLL。

### N01：FK / 镜像 / bind 真值与物理度量

`PhysicsContext` 记录 bind asset（`data/assets/somaview/SOMA_bind.bvh`，sha256 `d844abc9…`）、单位（cm→m，`unit_scale=0.01`）、joint order digest、镜像规则与 `physical_metric_version=1`。oracle 用源 BVH 走同一预处理重建真值（feature 完全一致，max\|Δ\|=0.0），4 组真实镜像对（8 个 clip，162–436 帧）：

| 度量（root-relative，均值 cm） | clip 8/9 | 18/19 | 56/57 | 86/87 |
|---|---:|---:|---:|---:|
| bind skeleton（正确镜像） | 0.005 | 0.159 | 0.020 | 0.021 |
| 旧 `stats['ref_pos']`（数据集均值） | 30.68 | 28.83 | 29.52 | 30.71 |
| 错误镜像骨架 | 61.38 | 57.67 | 59.07 | 61.44 |
| tokenizer 往返（去 warmup） | 5.09 | 9.50 | 25.74 | 4.86 |

- 特征→FK 的**坐标重建误差是 0.005–0.16 cm**（随窗口长度增长，来自 root 速度积分）；旧 ref_pos 路径偏 ~30 cm，错误镜像 ~60 cm。旧米制结论（NEF physics）因此**不能**与修正后的数字并列。
- **通道归因**（把一组通道换回无量化值后重测）：把 root 通道还原，clip 56 的误差从 25.74 → 2.53 cm；把全部 FK 相关通道还原 → 0.02 cm。即 tokenizer 的物理误差在不同片段由 root 流与关节旋转流分摊，逐片段不同，不能一概说"root 主导"。
- 接触/滑步（真实 token vs 重建）：几何接触率 1.000→0.110（clip 56），脚滑 0.011→0.071 m/s。tokenizer 重建丢失了接触。
- `scripts/evaluate_nef_physics.py` 已改为走 bind context 并写出 `physical_metric_version`/`skeleton_source`；旧路径不再是主评估入口。

### N02：内容标签捷径与 dev benchmark

213 行协议的 content×style 表（旧标签）：`Basic Locomotion Styles` 143 行 100% 非 neutral，`Basic Locomotion Neutral` 38 行 100% neutral——**内容标签独自完美区分风格**。v1 映射只合并这两类，合并后同样的 181 行落进 `Basic Locomotion`（非 neutral 占比 0.79）。训练集上 `Basic Locomotion` 的多数类占比 0.785、熵 1.06 bit（旧 `Styles` 熵 1.72、`Neutral` 0）。仅用 content 预测 style 的准确率（val）：raw 0.5609 → canonical 0.5303（三个可配对 style 上）。未知标签（`Martial Arts`/`Sports`/`Other`）不改写、保留原标签并计数报告。

Dev benchmark：18 行、fingerprint `354e4b75…`、`eval_seed=20260918`，两个子任务——同 style 跨 content 参考一致性（6 行）与固定 source content 换 target style（12 行）；query 内容必须在训练词表内（否则报告 unavailable），reference 可来自未声明内容；所有行以 take 分组、镜像/来源隔离。actor 轴不可测（store 无 performer 表），unseen-style 无 checkpoint，均记 `not_evaluated`。

### N03：真实 editing benchmark

网格：12 个锁定 case × {style-ID, constant, reference} × {whole_body, left_arm(radius 1)} × {λ=0,1,1.5} × {steps=1,4} × 2 draws，共 864 行 + 72 张固定投影 flipbook。base 与 styled 共用同一 CRN cell **且同一 schedule**。本轮为此返工三次，前两次都作为失败案例保留：

1. `N03/run1_independent_draws`：base 与 styled 是**独立 draw**，~48% 的 token 差异只是耦合方式造成的假差异；
2. `N03/run2_schedule_mismatch`：修好耦合并行 CRN 后，`use_base=True` 仍忽略 schedule，于是 4 步 styled draw 对 1 步 base draw 在**任何** λ（包括恒等锚点 λ=0、TV=0）都差 0.4915——那是 schedule，不是 style；
3. `N03/run3_iterative_unstyled`：**发现真实实现缺陷**。多步生成的实现对**当前要采样的 block 先标记为 visible**再采样，而 operator 只编辑 hidden 位置，于是每一步采出的 block 实际都来自冻结 base 分布——`steps=4` 的 styled draw 与 base draw **逐位相同**（changed=0.0000，而 TV 是 0.046–0.068）。这是"多步生成静默地不带风格"的 bug，只有把 steps=1/4 都跑一遍才会暴露。已在 `stylized_motion/learning/mts_operator/model.py` 改为 `visible_now = ~remaining`，并加回归反例 `tests/test_mts_end_to_end.py::test_a_multi_step_styled_draw_is_not_silently_the_base_draw`（同时断言 λ=0 时两者**必须**相同，否则测试无法区分"恒等锚点"与"没生效"）。修复后 steps=1 路径不变，steps=4 重新成为带风格的迭代生成。

风格响应（生成侧，同一 inputs 换 style 输入、同 CRN）：

| 臂 | 可测行 | TV | NLL 改善 | 换输入后 draw 改变的 token 比例 | 改变非零的行 |
|---|---:|---:|---:|---:|---:|
| style-ID | 144 | 0.0248 | +0.00833 | **0.066** | 96/144 |
| reference | 120 | 1.28e-08 | −2.1e-09 | **0.000** | **0/120** |
| constant | — | — | — | 无 style 输入 | — |

- **泄漏为 0**：局部编辑（left_arm，144 行）外部 token 改变比例 **0.0000**；support 内 0.155。whole-body 区域无 support 概念。
- 强度单调（style-ID，whole_body，steps=1）：TV 0.00000（λ=0，恒等锚点且与 base draw 逐位一致）→0.0816（λ=1）→0.1197（λ=1.5）；耦合 token 改变比例 0.000→0.252→0.343。steps=4 修复后同样单调（0.000→0.346→0.441）。
- 物理代价（N01 的 context，反归一化后）：source→base 的 FK 变化 0.407 m；base→styled 在 λ=1 时为 0.205 m（whole-body）/0.037 m（left arm）。**jerk 代价显著**：真实 motion 的 \|jerk\| 均值 **622 m/s³**，冻结 base 的采样 draw **54131 m/s³**，风格编辑只在此之上再加约 8%——"采样本身"带来约 87× 的三阶差分放大，风格编辑不是主因。这是 NLL 与 token 统计都看不到的可见质量代价，也是下一轮 base 改进的主要证据。
- flipbook 只是给人看的固定投影序列（source/base/styled 三行，相同坐标范围），**不作为分数**。

### N05a：reference 机制诊断（无训练）

24 条真实 reference（3 style）逐层相对离散（跨 reference）：

| 层 | trained | fresh（同架构，seed 3407） |
|---|---:|---:|
| embedding | 1.18e-01 | 1.25e-01 |
| temporal | 3.46e-04 | 3.85e-03 |
| graph | 2.97e-04 | 3.85e-03 |
| pooled | 2.34e-04 | 3.43e-03 |
| descriptor | **1.09e-05** | **3.39e-03** |

- trained 的 descriptor 对真实交换 reference 的相对变化 1.42e-05，fresh 为 5.10e-03（360×）；时间打乱对 trained descriptor 只有 5.7e-06，而关掉 position encoding 改变 4.5%——**descriptor 主要由位置信号决定，而非 reference 内容**。
- 梯度：单 batch、不 step，encoder 6 个层组全部收到有限非零梯度（97/103 个 tensor 非零），optimizer 含 103 个 encoder 张量 → **不是"梯度没到/实现错误"**。
- 逐层阻尼：fresh 的 embedding→temporal 相对离散已被压到 0.031，pooled→descriptor 0.984；fresh descriptor 的 between/within style 距离比仅 1.070。即**输入路径本身已经强烈压制 reference 差异**，且三类风格在 descriptor 上几乎不可分。
- 判定：**主因 `trained_into_a_constant`**（训练把输入依赖抹平），**次因 `the_input_path_already_damps_reference_differences`**（初始就不敏感 + 风格不可分）。二者都要在 N05b 里处理，不能只加一项辅助 loss。

### 门槛与决策

`screening_decision_v2.json`：`base=go_for_context_infill`、`content_schema=v1_merge_only_the_two_locomotion_labels`、`style_input={likelihood: weak_positive_row_matched, generation: small_visible_style_id_response_no_reference_response, perceptual: not_evaluated}`、`ctmc=no_advantage`、`reference_encoder=reference_invariant_trained_into_a_constant`、`motion_style_quality=not_evaluated`。六项状态标签（implementation/measurement/likelihood/motion_quality/style_fidelity/generalization）在各 stage JSON 内分别标注，未测项一律 `not_evaluated`。

**NLL 改善没有被写成已验证的可见风格迁移**：本轮所有"风格"结论都分成似然（weak_positive）、生成（style-ID 有可见但很小的响应；reference 为零）、感知（未评估）三层报告。

---

## N04：canonical-content 复现实验（2026-09-19，含训练）

授权：用户「继续，按照plan执行」（在 `N04_application.json` 报出预算之后）。本轮首批真实训练：**5 个 run、5,020 optimizer steps、全部 GPU wall < 10 分钟**（RTX 5070 Ti 16 GB，torch 2.13.0+cu132；pilot 10 epoch ≈ 107 s）。

### 实现与配方

- `data/configs/mts_content_schema_v1.yaml` 接入两个 trainer：词表、pair 条件、验证行标签全部走 canonical 标签；`strict: true` 只约束 **train split**（val/test 的 `Other`/`Sports`/`Martial Arts` 仍按词表过滤排除并计数，与第一轮同规则）。
- 新配方：`mts_revision2_transport_canonical.yaml`、`mts_revision2_style_id_logit_canonical.yaml`、`mts_revision2_noref_logit_canonical.yaml` + 请求清单 `mts_revision2_canonical_manifest.yaml`。除 content map 外，架构/loss/mask/batch/步数全部不动。
- checkpoint 现在记录 `content_schema`（版本、文件 SHA、重命名行数、strict 标志），transport 与四个 operator arm 均已写入。
- 新协议：transport `mts-transport-r2-canonical-v1`（320 行，hash `3cd39baa…`）、operator `mts-operator-r2-canonical-v1`（213 行，hash `03840b553b9e…`）。**旧协议的 hash/行集未被复用**，两轮数值不可直接比较。

### 结果（213 行 canonical 协议，冻结 base 全臂一致 = 1.575901）

| 臂 | protocol weighted | 相对 base 改善 | full_generation |
|---|---:|---:|---:|
| constant s3407 | 1.551681 | 0.024219 | 1.9789 |
| style-ID s3407 | **1.544235** | 0.031665 | 1.9689 |
| constant s3408 | 1.548915 | 0.026986 | 1.9803 |
| style-ID s3408 | **1.539890** | 0.036011 | 1.9671 |
| （数据基线）action-conditioned unigram | 1.902487 | — | 1.9486 |
| （数据基线）per-coordinate unigram | 1.960491 | — | 1.9966 |

**风格净效应（style-ID − constant，逐行配对，按 take 聚类 bootstrap）**：

| seed | 改善 | 95% CI | 区间排除 0 | 逐行更优 | 三风格方向 | 五种 mask 方向 |
|---|---:|---|---|---|---|---|
| 3407 | +0.007492 | [0.004685, 0.010124] | 是 | 145/213 | 全正 | 全正 |
| 3408 | +0.009070 | [0.006134, 0.011810] | 是 | 153/213 | 全正 | 全正 |

correct-vs-wrong style id：+0.016005（CI [0.012217, 0.019489]，168/213）与 +0.018594（CI [0.014705, 0.022312]，168/213）。两个 seed 的逐行 NLL 相关 0.999。**结论：content map 修正后风格信号没有消失，且两个 seed 一致**——达到文档"一致且可见"的继续条件；是否扩到 3k 步/第三 seed 属于新预算，未自行启动。

### 仍未通过的门槛

- **full-generation 仍低于 action-conditioned unigram**（base 2.0103、最好 arm 1.9671 vs 1.9486），与第一轮同向；context 各 kind 则远低于基线。这条短板没有被 content map 修正。
- 生成侧可见风格响应：style-ID 在 144 行中 96 行换 style 输入会改变 token（均值 7.4%），TV 0.0276；泄漏仍为 0。但**感知层面仍未评估**。

### 本轮发现并修复的缺陷（都带回归反例）

1. **验证行跟着训练 seed 走**：seed 3408 采出 201 行、seed 3407 采出 213 行——"换 seed"同时换了测量。新增 `evaluation.seed`（默认仍为训练 seed，保持旧行为），canonical 配方固定 3407；回归反例 `tests/test_mts_cli_matrix.py::test_the_validation_rows_do_not_follow_the_training_seed`。
2. **batch metadata 的 raw action 未过 map**：canonical profile 首次运行在 `Basic Locomotion Neutral` 上直接崩（拒绝借 id），随后在 batch source 里接上 schema。
3. **preflight 拿 raw 标签对 canonical 词表**：会误报（也会放过"map 没生效"的 run）；现在先应用配方 schema 并把 schema 证据写进检查项。
4. **N03 探针只认 raw 标签**：现在按 checkpoint 记录的 map 选 label（canonical arm 用 v1、旧 arm 用 raw），并在每行标注 `conditioning_map`。
5. **operator provenance 未记录 content map**：四个 arm 重跑（旧目录保留为 `*_norecord` 并附 NOTE，数值逐位相同）。
6. `outputs/mts_revision2/` 前缀守卫是仓库约定，N04 run 目录遵循它（而不是绕开）。

### 工件

`outputs/mts_next_round_20260918/N04/`（四臂 summary/rows、`canonical_style_effect.json`、`probe/`、两份 token baseline、`run_n04.sh`）、`stages/N04.json`、决策更新写入 `screening_decision_v2.json` 的 `n04_update`，`manifest.json` 现含 18 个工件与 6 个 checkpoint 的 SHA。

---

## N05b：reference 分支的受控修复（2026-09-19，含训练）

授权：用户「继续，按照plan执行」（在 `N05b_application.json` 报出预算之后）。训练量：**1,000 步 encoder 监督任务 + 1,000 步冻结 encoder 的 operator**，共 2,000 步；encoder 任务 GPU wall 约 91 s。

### 机制判定被细化（N05a 之后）

封闭式 ridge 探针（无优化器、encoder 逐位不变）在**三个可配对 style**上：

| 载体 | 平衡准确率（未见 take） | 未见 content | 打乱标签对照 | descriptor 范数 | 相对 L2 between/within |
|---|---:|---:|---:|---:|---:|
| round-1 reference encoder | 0.650 [0.611, 0.688] | 0.707 | 0.320 | 0.540 | 1.017 |
| fresh 同架构初始化 | **0.920** [0.898, 0.940] | 0.909 | 0.302 | 2.440 | 1.024 |
| 监督任务训练后 | **0.952** [0.934, 0.968] | 0.954 | 0.262 | 21.05 | **2.939** |

真实 reference 交换的相对变化：round-1 encoder **1.6e-05**、fresh 4.9e-03、监督版**0.56（同 style 跨 content）/ 1.28（不同 style）**；同输入重复编码为 0（数值地板）。

**结论修正**：N05a 的"输入路径本身已压制 reference 差异"过于绝对——信息在初始化时就存在（92%），训练用纯 NLL 目标把它*衰减*（65%、范数缩小 4.5×）而不是抹掉；operator 随后选择忽略它。因此"训练成常量"应表述为 **"目标没有使用 reference 的压力，训练把可读信号衰减到操作者忽略的程度"**。

### 冻结分支修复

用同一架构（未加宽）训练 seen-style 分类任务（1,000 步，三类平衡采样，val 为未见 take/content），随后**冻结 encoder** 重训 logit operator（1,000 步，canonical base、canonical 协议 `03840b55…`）：

| 臂 | protocol NLL | 相对 base | 逐行配对 style-input 改善 | 换 reference 后 draw 改变的行 |
|---|---:|---:|---:|---:|
| constant | 1.551681 | 0.024219 | —（无 style 输入） | — |
| **reference（冻结监督 encoder）** | **1.547040** | 0.028860 | **+0.008256**（143 行，|Δ| 0.0157） | **80/120** |
| style-ID | 1.544235 | 0.031665 | +0.015950 | 96/144 |
| round-1 reference（对照） | — | — | −9.4e-11 | **0/120** |

reference 的净效应（相对 constant）0.004641，约为 style-ID（0.007446）的 62%；泄漏仍为 0.0000，强度单调（λ=0 时 TV=0、changed=0）。**plan 的"冻结分支恢复 reference 响应"门槛通过**：真实 reference 交换现在会改变 80/120 行的生成结果，而 round-1 是 0/120。

### 代码改动

- `train_mts_operator.py` 新增 `style_encoder.checkpoint`：加载监督 encoder（校验 tokenizer 绑定、宽度与类别，记录 SHA/step/classes 到 dry-run 与 provenance）。
- 新脚本 `scripts/train_mts_reference_encoder.py` + `data/configs/mts_reference_encoder_v1.yaml`（seen-style 任务，≤1,000 步）与 `mts_revision2_reference_frozen_encoder.yaml`（冻结臂）。
- 探针 `scripts/probe_mts_style_identifiability.py` 同时接受 operator bundle 与 `style_encoder` checkpoint，并内置交换响应对照数值地板。
- 修掉两个只在 GPU 上暴露的 device bug（padding mask 在 CPU、logits 在 CUDA 上转 numpy）。

### 明确未做（都需要新预算）

- plan 提到的后续 ablation：低 LR encoder fine-tuning、单独一个辅助 CE 项（一次只改一个；不加 contrastive/adversarial/cycle/time-warp）。
- reference 臂的第二 seed（分离 encoder 任务随机性与 operator 随机性）。
- 任何感知层面的风格评价：本轮仍只有 token/TV/探针三类证据。

### N05b 消融：两个单变量实验（同日，2 × 1,000 步）

plan 要求"冻结分支恢复 reference 响应后，再单独 ablate 低 LR encoder fine-tuning 和一个辅助 CE 项"。两个 recipe 各只改一处：

| 变体 | 单一改动 | protocol NLL | 相对 base | 逐行配对 style-input 改善 | 换 reference 后 draw 改变 |
|---|---|---:|---:|---:|---:|
| 冻结分支（N05b-3，对照） | — | 1.547040 | 0.028860 | +0.008256 | 80/120 |
| **A1 低 LR 微调** | supervised encoder 解冻，lr=6e-6（operator 的 1/50） | **1.546078** | **0.029823** | **+0.017514** | 80/120 |
| **A2 辅助 CE** | fresh encoder 端到端 + 一项 seen-style CE（w=0.5） | 1.548242 | 0.027659 | +0.010881 | 80/120 |
| constant（对照） | — | 1.551681 | 0.024219 | —（无 style 输入） | — |

- **两个机制都保住了修复**：生成网格里 80/120 行换 reference 会改变 draw（round-1 是 0/120），泄漏仍为 0.0000，λ=0 恒等锚点逐位成立。
- **A1 最强**：三个 reference 变体中似然最好（1.546078），逐行 style-input 改善是冻结分支的 2.1 倍——但它在 143 个配对行上，与 style-ID 臂的 213 行不可直接比较。
- **A2 单项即足够**：fresh encoder 只加一项辅助 CE 就没有塌回 round-1 的状态，把"塌缩机制"隔离到了"目标里没有使用 reference 的压力"。
- 实现：`training.style_encoder_lr`（分参数组）、`training.aux_style_ce_weight` + `OperatorBatch.reference_style_ids`（配对源填充 reference 自己的 style id；验证批不带，辅助项自动跳过）、`model.loss` 的 metrics 暴露 `style_embedding`（带梯度）。回归面：`pytest -k mts` 291 passed。

### N06（决策准备，无训练）

N01/N03/N04 的证据已覆盖 N06 的判据：canonical base 的 full-mask 仍高于 action-conditioned unigram（2.0103 vs 1.9486）而 context kinds 远低于基线 → **base 暂不扩预算**；CTMC 无 matched style-fidelity 候选收益 → **不重启**；tokenizer 在 N01 的 oracle 里是重建误差的主导（4.9–26.6 cm，坐标上限 0.005–0.16 cm），但 operator 编辑的位移量级更大，**重训 tokenizer 仍不满足"证明它是当前主瓶颈"的条件**。N06 正式决策留到独立 style evaluator 与感知证据之后。

## 追加交付（同日更晚 2）：generator 结果的角色渲染

用户问“当前各个 generator 的结果能否可视化”。答案与工件见 **`docs/mts_generation_visualization_zh.md`**。
要点：N04 筛选轮已有 token/火柴棍 flipbook（48 张）与 576 行指标；本轮给
`scripts/benchmark_mts_editing.py` 加 `--save-motions`（把 source/base/styled 解码特征窗存 npz），
新增 `scripts/render_mts_generation.py` 用与 NEF-FSQ 检查相同的 somaview + Quinn + bind 骨架路径
渲染角色对比图，并实跑 2 case × 2 臂（`outputs/mts_generation_demo/`）。

实跑结论（筛选轮观察，非效果声明）：对比目前被 screening transport 的 argmax 失真淹没
（source→base max ≈ 1.2 m）；算子的 locked edit 本身工作正常（base→styled 均值 7–10 mm、
root 不动、改动集中在左臂）；style_id 臂比 constant 臂改动略大，与 N04 读数一致但量级很小。
另外实跑确认第一轮生成脚本 `generate_mts_operator.py` 与 revision-2 checkpoint 不兼容
（steps=1 的 return_trace 契约、constant encoder 未处理、不应用 content schema v1）——
待立项修复；在此之前生成入口是 `benchmark_mts_editing.py`。

### 独立 style evaluator（N03 未完项）与 N06 决策（同日，无训练）

**N03 遗留的"用原始 motion 而非 token-NLL 的独立评价基线"已建成**：18 个固定运动学特征（全部经共享 `PhysicsContext` 反归一化，米制单位）+ 封闭式 ridge；reference encoder 永远不当裁判。

- **真值上限**：val（未见 take）平衡准确率 **0.587** [0.551, 0.623]，chance 0.333——真实 motion 可分但很弱（injured leg 召回仅 0.284，多被读成 injured torso）。
- **token 往返的 source 窗口**：0.667（12/12 案例里 8 个正确）。
- **生成窗口**：**四个臂（含无 style 输入的 constant）× 两个区域（whole_body、left_arm）× 12 案例 = 96/96 全部被判成同一类**。局部编辑（~85% token 是真实 motion）也一样。

结论：**裁判对"生成 motion"无效**——采样分布主导了运动学特征（与 N03 的 87× jerk 一致），所以它今天既不能证明也不能否证风格保真。这直接封死了两类过度声明：NLL 改善依旧只是似然证据；"可见风格迁移"维持 `not_evaluated`。

**N06 决策（全部"不"，均有证据与改变条件）**：

| 决策 | 结论 | 关键证据 |
|---|---|---|
| 扩 base 预算 | **否** | full-generation 仍高于 action-conditioned unigram（2.0103/1.9671 vs 1.9486）；独立裁判显示采样分布 OOD；87× jerk |
| 重启 CTMC | **否** | plan 前提是 matched style fidelity 下有候选收益，而该裁判尚不存在 |
| 重训 tokenizer | **否** | N01 证明它是重建误差主导（4.9–26.6 cm vs 坐标上限 0.005–0.16 cm）、接触特征丢失，但 operator 编辑位移更大且采样才是 OOD 主因；重训会使全部 token 语义失效 |

下一轮建议（按价值排序）：给生成 motion 有效的裁判（在采样分布上拟合特征/限定到采样可存活的特征）；对已有 flipbook 做人工评分；A1 第二 seed。

### 勘误 3（生成渲染）：首版 `render_mts_generation.py` 漏了 bind 骨架替换

用户指出上一节的角色渲染图蒙皮扭曲。原因与可视化勘误 2 同源：渲染脚本直接用 tokenizer
checkpoint 的 stats 做 FK，没有调 `stats_with_reference_skeleton`——`ref_pos` 是镜像平均均值，
脊柱链塌缩（case000 source 窗 `Hips→Head` 仅 0.005 m，头缩进骨盆）。已修复（镜像感知：
`--record-npz` 读记录 meta 的 `mirror`，`--generation` 用 `--mirror`），四套 montage 重渲，
summary 记录 `reference_skeleton`。**读数几乎不变**（骨架误差共模，逐关节差分相消；
case000 style_id base→styled 0.0101 m / max 0.526 → 0.525）——变的是画面，不是相对数字。
详见 `docs/mts_generation_visualization_zh.md` 修正记录。

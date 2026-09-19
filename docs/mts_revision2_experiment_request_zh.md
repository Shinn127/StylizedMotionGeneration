# MTS revision-2 交付状态与下一轮实验申请（C11）

执行规范：[MTS_FSQ_Agent_Integration_Closure_Plan_zh.md](MTS_FSQ_Agent_Integration_Closure_Plan_zh.md)。
进度记录：[MTS_FSQ_Code_Revision_Progress_zh.md](MTS_FSQ_Code_Revision_Progress_zh.md)。
实验清单：`data/configs/mts_revision2_experiment_manifest.yaml`（只生成命令，未执行）。

## 交付状态

```text
G-entry：通过。
  依据：C01–C05 的真实 tiny CLI 可执行，且数据绑定与固定协议真的被执行：
  · OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <python> -m pytest -q tests/test_mts_cli_matrix.py
    → exit 0；34 passed（输出：outputs/mts_revision2_closure/C09/c09_matrix.txt）
  · A: --build-manifest-only 无任何 MTS checkpoint 即可落盘 manifest
  · B: transport 2 步 → 冻结 val → 保存 → 回读；dry-run 只做身份/构建预检
  · C1–C6: 3 operator × {reference, style-ID} 训练 2 步 → 验证 → 保存/回读 → eval → generate
  · F: 错 hash / 非法 style / 未知配置 / 空 val / content-kind 冲突 → 非零退出且不写 best

G-code：通过。
  依据：C06–C09 完成，C00 的入口反例全部转绿，另有 16 条本轮新发现的真实缺陷修好并留痕
  （outputs/mts_revision2_closure/C09/c09_defects.txt）。全量套件：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <python> -m pytest -q
  → exit 1；541 passed / 1 skipped / 1 failed
  （唯一失败 tests/test_preprocess_pipeline.py::test_100style_fsq_excludes_last_ten_styles
   是既有的本机 100STYLE 目录布局问题，非本次改动引入；证据 c09_full_suite.txt）
  未声称：真实 SEED 全量训练/评估/生成的非零退出路径之外的成功路径（预算未批准）。

G-ready：data 级通过，experiment 级 blocked（未训练 revision-2 模型）。
  data 级（真实数据，只读，最多 8 窗口）：
  · 正确组合（holdout tokenizer + v4-ah tokenstore）
    → exit 0；ok=true，全部 data 检查 pass
    证据：outputs/mts_revision2_closure/C10/preflight_ah/preflight.json
  · 旧 1h tokenizer + 同一 tokenstore
    → exit 1；store_identity fail（"...same structure is not the same weights"）
    证据：outputs/mts_revision2_closure/C10/preflight_stale_tokenizer/preflight.json
  · 窗口上限受测：--max-windows 2 → clips_read=2；--max-windows 0 → exit 1
  · 预检不创建 best.pt、不启动训练（tests/test_mts_preflight.py 断言）
  experiment 级：transport / action_vocabulary / validation_protocol / checkpoint_bindings
  全部 blocked（尚未训练 revision-2 transport），budget 通过；不得据此宣称模型性能。
```

> **T00–T04 之后的更正（2026-09-18，以本文档下方“下一轮实验申请”与 manifest v2 为准）**
>
> - 预检已改为按 stage（`data / transport_train / operator_train / evaluate`）判定，`ready` 才是有无
>   准入资格；上面 C10 时写的“ok=true 而 experiment 全 blocked”已不再可能出现在 `ready` 为真的情形
>   （required blocked → ready=false、exit 1；`ok` 只是“无 fail”的诊断字段）。
> - transport 的验证集此前取的是 **train** split，且 `--val-rows` 决定行数；现在取 val split、
>   行数由配方 `evaluation.validation_rows_per_kind` 决定、逐行对 store 的 split 表核验。
> - 主 transport 配方此前是 `content.kind: none`，现已改为 `action_id`；无条件只保留在明确命名的
>   smoke/debug 配方里。
> - 旧的 `protocol_id: mts_r2_closure_v1` 与 `<run_id>` 占位符已被逐 arm 的协议 id 与解析后的输出目录取代。

## 修改文件与主要行为

C09（真 CLI 矩阵）新增/修复的实现（详见 `c09_defects.txt`）：

| 文件 | 行为 |
|---|---|
| `scripts/train_mts_transport.py` | 冻结点验证集改用 `windows.TokenSource`；loader 段在 main 解析；`data.frames` 生效；固定 mapping 输出（tokens/valid_mask/content_condition/sample_metadata）；新增 `--dry-run` |
| `scripts/train_mts_operator.py` | `output` 前置；`training_exposure` 入 provenance；style-ID validation 传 `style_index`/`encoder_kind` |
| `scripts/evaluate_mts_operator.py` | bundle 加载带 tokenizer 文件；`build_row_batch` 保留完整条件；逐样本数值行（NLL/retrieval rank）；三对比物理；style label 不再充当 content condition |
| `scripts/generate_mts_operator.py` | bundle 加载带 tokenizer 文件；`output` 前置；区域 mask 按样本展开；style-ID 模式拒绝 `--style-clip` |
| `stylized_motion/data/loader.py` | v3 数据集透传 `return_metadata` |
| `stylized_motion/data/token_data.py` | v3 元数据附带 clip 标签（style/action/performer） |
| `stylized_motion/learning/mts_operator/pairs.py` | 新增 `clip_label_from_tables`（单行版，与 `clip_records_from_store` 同约定） |
| `stylized_motion/learning/mts_operator/style_encoder.py` | `GlobalStyleEncoder.config()` 带 `kind` |
| `stylized_motion/learning/mts_operator/eval_protocol.py` | `build_target_only_samples(labels=...)`；`transport_items()` |
| `stylized_motion/learning/mts_operator/model.py` | `generate_edit(use_base=...)`（base 对照取冻结 transport 分布） |
| `stylized_motion/learning/mts_operator/metrics.py` | `token_likelihood_per_sample`；`refresh_only_tv` detach |
| `stylized_motion/learning/nef_data.py` | 新增 `store_normalized_window`（packed=原始帧→store 归一化空间） |
| `stylized_motion/learning/mts_operator/windows.py`、`scripts/probe_nef_geometry.py`、`scripts/evaluate_nef_locality.py` | 在线编码输入空间修正后共用 |
| `tests/test_mts_cli.py`、`tests/test_mts_cli_matrix.py` | tiny fixture 可绑定真实 tokenizer/标签表；C09 矩阵 34 项 |

C10：新增 `scripts/preflight_mts_revision2.py`（data/experiment 两级、逐项 status/reason/evidence、
`preflight.json`、非零退出）与 `tests/test_mts_preflight.py`（6 项）。

C08（协议 v2）：`stylized_motion/learning/nef_probe.py` 新增
`PROBE_PROTOCOL_REVISION=2`、`signed_span_perturbations`、`signed_pulse_perturbations`、
`shared_legal_support`、`influence_profile`、`stratified_span_probe`；
`scripts/probe_nef_geometry.py` 新增 `--stratified*`，产物写 `probe_stratified.json/csv`
（不覆盖历史 `probe_geometry.*`）。真实数据实跑（2 窗口 × 13 坐标 × 4 扰动 = 104 行）：
`outputs/mts_revision2_closure/C08/stratified/`，其中
`tail_frames_needed_for_complete_probe=99`、`rows_with_incomplete_tail=0`、
far 的 world tail 大于 near（这正是 feature influence 与 root 积分 tail 必须分开报的原因）。

## 定向 / 全量 / CUDA 测试：实际命令与结果

| 命令 | 结果 |
|---|---|
| `<py> -m pytest -q tests/test_mts_cli_matrix.py` | exit 0；34 passed |
| `<py> -m pytest -q tests/test_mts_preflight.py` | exit 0；6 passed |
| `<py> -m pytest -q tests/test_nef_probe.py` | exit 0；13 passed |
| `<py> -m pytest -q tests/test_mts_pairs.py tests/test_packed_downstream.py` | exit 0 |
| `<py> -m pytest -q`（全量，无 deselect） | exit 1；541 passed / 1 skipped / 1 failed（既有 100STYLE 布局） |
| CUDA：`test_online_feature_reader_is_exact_on_the_device_that_built_the_store` | 本机有 CUDA，实跑通过（在线编码与 token store 逐位一致）；无 CUDA 时该用例 skip |

前缀均为 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/shinn/miniforge3/envs/mcc/bin/python`。

## 旧结果仍可使用的范围

- **NEF tokenizer 与其 representation/physics 结果**：MTS 的 loss/协议修订不改变 tokenizer 的
  重建与物理含义，旧 tokenizer 结果按当时的实现与协议继续有效。唯一需要重新解释的是
  **feature 在线编码**路径：`read_window_tokens` 此前把 packed store 的原始帧当成已归一化帧，
  该路径产出的 token 数值不成立（同一窗口 71% token 不一致，C09 已修）。凡是用这条路径
  （MTS 的 feature 在线训练、probe_nef_geometry / evaluate_nef_locality 的 token 编码）
  得到的结果都需要按修正后的实现重跑。
- **旧 MTS transport/operator checkpoint**：schema 1 与旧 architecture revision 继续被拒绝，
  只能作为审计材料，不能作为新主模型；需要按 revision 2 重训。
- **旧 100STYLE 相关结果**：数据布局问题未变（本机为扁平目录），与本次改动无关。
- **历史配置文件**：本轮之前已被修改（见下），不能再说“历史配方未动”。

## 历史配置文件的实际差异（不 checkout 覆盖）

`git diff data/configs/mts_operator_*.yaml`（4 个文件，+22 行，只增不删）：

| 文件 | 新增 |
|---|---|
| `mts_operator_style.yaml` | `style_encoder.position_encoding: sinusoidal`；`data.content.kind: none`（显式无条件）；`data.pairs.target_sampling: style_uniform` |
| `mts_operator_style_ah.yaml` | 同上三项 |
| `mts_operator_transport.yaml` | `position_encoding: sinusoidal`；`data.content.kind: none`；`data.pairs.target_sampling: style_uniform`；`transport.token_embed_dim: 16` |
| `mts_operator_transport_ah.yaml` | `transport.token_embed_dim: 16`；`position_encoding: sinusoidal` |

含义：这些字段以前“缺省即隐式默认”，现在被写进配方，因此**旧配方文件描述的运行与当时实际
执行的可能不同**（例如未声明 content 时旧代码按无条件跑，未声明 position_encoding 时旧代码
按无位置编码跑）。需要复现历史运行时，用上述 diff 逆推当时的隐式默认，不要直接 checkout 覆盖。

## 下一轮实验申请（需要预算；T03 之后已改为可执行配方）

清单见 `data/configs/mts_revision2_experiment_manifest.yaml`（manifest_version 2）。要点：

- **一条 arm 一个配方文件**，每份只含该 family 自己的字段，预算与输出目录都在文件里：
  | arm | 配方 | 预算 | 协议 |
  |---|---|---|---|
  | transport profile（S0） | `mts_revision2_transport_profile.yaml` | 20 步 | `mts-transport-revision2-profile-v1`（10 行） |
  | transport pilot（S2 首段） | `mts_revision2_transport_pilot.yaml` | 2,000 步 | `mts-transport-revision2-pilot-v1`（320 行） |
  | style-ID logit（S3） | `mts_revision2_style_id_logit.yaml` | 1,000 步 | `mts-operator-r2-shared-val-v1` |
  | **no-reference logit（S3 对照）** | `mts_revision2_noref_logit.yaml` | 1,000 步 | 同上 |
  | style-ID CTMC（S4） | `mts_revision2_style_id_ctmc.yaml` | 1,000 步 | 同上 |
  | reference logit（S4） | `mts_revision2_reference_logit.yaml` | 1,000 步 | 同上 |
- 数据：`seed_soma_pruned_v4_ah` + `_tokens`，holdout tokenizer
  `outputs/nef_fsq_soma_packed_40x9_ah/best.pt`（SHA 由预检在运行时核对，不写死在文档里）；
- base：所有 operator arm 共用 `outputs/mts_revision2/transport_pilot_seed3407/best.pt`，
  相同 pairs/mask/采样协议与同一个 validation protocol id（行集相同才可比）；
- seed：3407（先单 seed 筛查；三 seed 要等 S3 出信号后另行申请）；
- **no-reference 对照的含义**：同一个 logit operator、同宽、同预算，style descriptor 是一个学出来的
  **常量**（`style_encoder.kind: constant`），完全不读 reference token、不读 style id；参数差如实报告
  （tiny dry-run：constant 81,929 / style-ID 83,209 / reference 5,716,265 可训练参数）。
  不得用“reference 乱序/置换”冒充“模型不看 reference”。
- 每个 stage 前先跑 `scripts/preflight_mts_revision2.py` 的对应 stage
  （`transport_train` / `operator_train` / `evaluate`；旧 `--level` 只是别名）并留存 `preflight.json`；
  operator 配方的 dry-run 会先写出 frozen protocol，不需要先训练才能预检。

**吞吐测量方法（不沿用旧的“3 分钟一模型”）**：跑 profile 配方（20 步）实测，读其
`history.jsonl`：每个 epoch 记录 `train_seconds` / `val_seconds` / `checkpoint_seconds` /
`total_seconds`，对应真实 batch size 与窗口长度；再乘以计划步数。batch size 改变时要重测。
`train_summary.json` 现在给出 `planned_steps` / `steps_shortfall` / `completed` / `interrupted`，
预算未达成或被 wall-clock cap 截断的 run 会以非零退出并明确标注，不会伪装成完成。

**未批准前不会执行**：GPU 训练、上千步过拟合、正式 sweep、evaluator 训练；当前只交付配方、
命令、dry-run 与预检证据。

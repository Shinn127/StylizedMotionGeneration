# MTS 算子阶段执行状态（P0–P5）

日期：2026-09-17。本文是**当前状态快照**：哪些阶段已完成并有可复现数字、哪些产物需要重做、
哪些尚未开始，以及恢复执行的命令与预计耗时。研究方法与判据见
[MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md](MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md)，
前面各阶段的具体记录见 [seed_1h_run_report.md](seed_1h_run_report.md)、
[seed_stage2_promotion.md](seed_stage2_promotion.md)、[seed_actor_holdout.md](seed_actor_holdout.md)。

## 0. 一句话状态

数据（actor-holdout 全量 store）→ 对应 tokenizer → packed token store → base transport **都已跑完并有数字**；
算子阶段产出的 6 个 checkpoint **全部是 reference 条件算子**（可用，但其中 4 个本该是 style-ID sandbox 与
shuffled 对照，因 CLI 覆盖失效而训错），需要重训；P5 矩阵 0/26（脚本与评估器的缺陷已修好并单 cell 验证），
图表与最终报告未产出。恢复执行约需 45 分钟 GPU 时间。

## 1. 链路与产物总表

| 阶段 | 产物 | 关键数字 | 状态 |
|---|---|---|---|
| 数据 catalog | `data/processed/seed_ah_catalog` | 142,220 clip；52 演员冻结进 test（14,105 clip）；train 113,883 / val 14,232 | ✅ |
| 数据 store | `data/processed/seed_soma_pruned_v4_ah`（60 GB） | units_reused=142,220；`train∩test 演员 = 0`；0 group 跨界；norm hash `a7d2fb6e` | ✅ |
| 风格审计 | `outputs/seed_style_audit_ah/style_pair_audit.json` | `performer_axis` 支持（overlap 0）；`style_axis: zero_shot_style_supported = false`（结构性）；4 条 warnings | ✅ |
| 旧 ckpt 在新 split 的重评估 | `outputs/stage3_eval_ah/` | R1 结论不变；v1.1 接触优势保持；FK/脚高排序跨 split 翻转（方差告警） | ✅ |
| P0 等预算对照 | `outputs/nef_fsq_soma_packed_40x9_1h_ctrl/`、`outputs/stage3_eval/physics_p0/` | 多练 11,350 步 recon+delta 几乎不动指标 → 物理目标的因果归因成立 | ✅ |
| P1b tokenizer | `outputs/nef_fsq_soma_packed_40x9_ah/best.pt` | 14,000 步 / epoch 72；val_full loss **0.15349**（recon 0.09124、delta 0.02075） | ✅ |
| P2 token store | `data/processed/seed_soma_pruned_v4_ah_tokens`（2.5 GB） | 142,220 clip、19 分片、**0 失败**；吞吐 168 clip/s；绑定 ckpt `e4507ff2` / norm `a7d2fb6e` / schema `f7d840c5` | ✅ |
| P3 base transport | `outputs/mts_transport/seed_ah/best.pt` | 20,000 步；best val **1.4274**；train acc 0.3915 / val acc 0.3264；1,860,977 参数（dim 128、depth 8、graph_depth 2、bidirectional） | ✅ |
| P4 算子 | `outputs/mts_operator/**` | 6 个 checkpoint：logit_field / arbitrary_kernel / birth_death 各 2 份（重复），全部 **reference encoder、dim 256** | ⚠️ 需重训期望的那 4 个 |
| P5 矩阵 | `outputs/mts_matrix/` | **0/26 个 cell**（脚本与评估器已修好） | ❌ |
| P5 图表 | `outputs/mts_figures/` | 已能出 fig3（泄漏）、fig4（level 几何）与两张 CSV（当前无算子行） | ⚠️ 缺算子图 |
| P5 报告 | — | 未撰写 | ❌ |

## 2. 数据集：actor holdout 与两条泛化轴

划分：`--actor-holdout-ratio 0.10` → **52 个演员整组进 test**；其余组按 train:val 划分。
store 层校验通过：train∩test 演员 0、val∩test 演员 0、0 个 take group 跨 split。

审计结论（`seed_style_audit_ah`）：

```json
"performer_axis": {"train_actors": 468, "test_actors": 52, "train_test_actor_overlap": 0,
                   "zero_shot_performer_supported": true},
"style_axis":     {"styles_in_test_only": [], "held_out_styles": [],
                   "zero_shot_style_supported": false}
```

- **unseen performer 轴成立**：held-out 演员的片段从不参与任何参数学习。
- **unseen style 轴在 SEED 上不成立**（结构性）：没有任何 style 只属于被冻结的演员。要造这条轴必须在算子
  训练时显式排除 style（`data.pairs.held_out_styles`，采样器与配置键都已就位）。
- 标签层面的两条警告仍然有效：`neutral` 占 92%；8 个 style 中 5 个只有单一 content 标签。

## 3. 已完成阶段的数字

**P0：物理目标的等预算归因**（同一起点、同样 11,350 步，128 个 test 窗口，解码侧测量）

| checkpoint | FK mean | root 漂移 | foot slide | 接触准确率 | 接触召回 | 脚高误差 |
|---|---|---|---|---|---|---|
| v1 (14.0k, recon+delta) | 8.22 cm | 0.93 cm | 0.3298 m/s | 0.526 | 0.379 | 4.27 cm |
| v1-ctrl (25.4k, recon+delta) | 8.15 cm | 0.93 cm | 0.3284 m/s | 0.533 | 0.390 | 4.23 cm |
| v1.1 (25.4k, +physical) | 8.71 cm | 2.05 cm | **0.2765 m/s** | **0.620** | **0.505** | 4.02 cm |

结论：接触/脚部改善来自 **staged physical objective**，不是训练更久；FK +7% 与 root 漂移代价同样归因于它。

**在 held-out 演员 split 上的重评估**（4 个旧 checkpoint，注意它们**训练时**仍见过这些演员）

R1 局部性（64 窗口）：flat 支持 100%、区域外关节最大变化 **1.37 m**；part 12.5–17.5%、**0.0**；
NEF v1/v1.1 10%、**0.0**，且腿编辑接触翻转 0.10–0.14（part 0.33–0.39）→ 结构结论跨 split 稳定。

解码侧物理（128 窗口）：NEF v1 FK 11.42 cm / 召回 0.425；v1.1 11.63 cm / **0.571**；flat 20.46 cm；
part **9.08 cm**。**排序在该 split 上变化**（part 的 FK 反超 NEF），已记为方差注意点。

## 4. P4：现状与必须重做的部分

**已定位的缺陷**：`scripts/train_mts_operator.py` 里 `--style-encoder-kind`、`--shuffled-adjacency`、
`--hidden-dim`、`--batch-size` 只被 argparse 接受、应用代码在改写中丢失，于是**静默无效**。证据是 6 个
checkpoint 的元数据完全一致：`style_encoder_kind: reference`、`enc_dim: 256`、无 `level_order`。

| checkpoint | 实际是什么 | 是否可用 |
|---|---|---|
| `sandbox/logit_field`、`reference/logit_field` | 同一配置的重复：reference encoder + logit_field | ✅ 可作 reference 条件的 logit baseline（dim 256） |
| `sandbox/arbitrary_kernel` | reference encoder + arbitrary kernel | ✅ 可作对照二 |
| `sandbox/birth_death`、`sandbox_birth_death_shuffled`、`reference/birth_death` | 同一配置的重复：reference encoder + birth-death（**无**乱序邻接） | ✅ 主方法；shuffled 对照缺失 |
| 6 个共同 | 8,000 步训练（best 落在 step 6,000），val_nll 0.090–0.094 | |

**需要重做**：
1. **style-ID sandbox ×3**（Phase 3 的目的：把"算子没能力"与"encoder 没提取风格"分开）——已验证可跑：
   60 步试跑打印 `style encoder: style_id`、`style ids`（8 个 SEED style）、`trainable_parameters: 17362`；
   8,000 步约 **3 分钟/个**。
2. **shuffled-adjacency 对照 ×1**（geometry 臂）——已验证 `operator/shuffled_adjacency: 1.0` 落进指标；
   约 3 分钟。
3. 若要求严格同预算，**reference 两臂在 dim 128 重训**（各约 6 分钟）；否则接受 dim 256 并在论文里注明容量。

## 5. P5：矩阵与图表

矩阵 26 个 cell（injection ×3、locality radius0/1/whole_body ×3、multi_region、temporal_span、
seen/unseen performer、shuffled、reference ×2 等）。**首次运行 0/26 成功**，两个原因都已修复：

- 我的矩阵脚本传了评估器不存在的 `--operator`（算子从 checkpoint 元数据读取）→ 已删除该参数；
- 评估器本身在 GPU 上有三处缺陷（见 §6）→ 已修，且用一个 cell 端到端验证通过
  （写出 `label/split/regions/graph_radius/frame_range`，区域外泄漏 0、`support_fraction 0.15`、
  `leakage 0`）。

图表：`scripts/plot_mts_figures.py` 已能消费 locality/physics/probe/operator 四类产物，输出
`fig3_off_target_leakage.png`、`fig4_level_geometry.png` 与 `table_locality.csv`、`table_physics.csv`
（算子相关图与 `table_operator.csv` 需要矩阵结果）。§8.3 的图 2（区域视觉对比）与图 6（失败案例）需要
`generate_mts_operator.py` + `anim/somaview` 渲染，尚未开始。

## 6. 本轮修复的缺陷（7 项，全部已提交）

| 缺陷 | 症状 | 修复 | 证据 |
|---|---|---|---|
| 段落候选集物化 | 每个目标扫描整条 style 桶（SEED `neutral` 约 105k 记录）→ **7 分钟/epoch**，GPU 空转 | 拒绝采样，`attempts` 有界；`candidates()` 保留供检查 | 64 对采样 1 ms；新增回归测试 |
| 算子每 epoch 供给 batch 数 | 长训练被压到 192–512 步 | 改为供给 `steps_per_epoch` 个 batch | epoch 内 1,000 步 ✓ |
| **CLI 覆盖静默失效** | style-ID/shuffled/dim/batch 全部无效 → sandbox 训成 reference | 覆盖统一在构建前应用到 config；两种 encoder 的键取并集校验、只传对应键 | 试跑打印 style_id + level_order ✓ |
| style-ID 缺 batch 级 style id | style-ID 模型无法前向 | 配对源建立 style→id 映射并写入 batch | `style ids: {...8 个 SEED style}` ✓ |
| token store limited-build 校验 | `--limit-clips` 构建永远无法通过校验 | 按 `(source clip, variant)` 身份对齐；full build 仍查行数 | 200 clip 探测通过 ✓ |
| v3-only 的 token 打开/加载 | 打包 token store 无法用于训练 | `open_any_token_store` 分派 + `PackedTokenDataset` 接入 loader + 读取器版本无关 | transport 在 packed token store 上训练 ✓ |
| GPU-only 三类错误 | 评估器在 CUDA 上崩溃/给出空指标 | `sample_tokens`/`paired_comparison` 的 generator 设备规则；评估器 kinematic context 迁移设备；`aggregate` 容忍标签列 | CUDA 守卫回归测试；单 cell 验证 ✓ |

测试：**396 通过**（唯一失败仍是既有的 100STYLE 数据布局问题）。

## 7. 恢复执行步骤

```bash
# 1) style-ID sandbox ×3 + shuffled 对照（约 12 分钟）
bash /tmp/run_sandbox.sh

# 2) P5 矩阵 26 个 cell（约 15–20 分钟；脚本已去掉 --operator）
bash /tmp/run_matrix.sh

# 3) 图表（CPU，秒级）
python scripts/plot_mts_figures.py --output outputs/mts_figures \
  --operator outputs/mts_matrix/*/*/operator_metrics.json

# 4) （可选，严格同预算）reference 两臂以 dim 128 重训
bash /tmp/run_reference.sh   # 脚本内 --hidden-dim 128 现已被正确应用
```

## 8. 已知限制（写论文时必须写明）

1. **旧 4 个 NEF/flat/part checkpoint 是在旧划分上训练的**（它们见过现在被冻结的演员），因此
   `stage3_eval_ah` 的数字**不是** zero-shot performer 结果；holdout 专用的 tokenizer 已经训好
   （`nef_fsq_soma_packed_40x9_ah`），但尚未用它重训 flat/part 或 NEF 的 v1.1 变体。
2. **128 窗口样本的方差不可忽略**：FK 与脚高的排序在两个 test split 间发生翻转，任何"某表征更好"的
   结论都应在更多窗口、两个 split 方向复核。
3. **SEED 无法支撑 unseen-style 轴**（§2），该轴需以 `held_out_styles` 显式排除，或改用 100STYLE。
4. **flat 的接触指标是退化解**（解码足部几乎静止 → "永远接触" 得高准确率），不能读作 flat 接触建模更好。
5. **P4 的 reference 算子容量为 dim 256**（覆盖失效时期的产物），与 transport 的 128 不一致，需在论文中
   注明或按 §7.4 重训。

## 9. 勘误（revision-2 收口后补充，2026-09-17）

本节只追加，不修改上面的历史记录。历史读数仍按当时的口径成立，但以下表述已被后续工作更正：

1. **“恢复执行步骤”里的 `/tmp/*.sh` 脚本不再是入口**：transport 的 `--checkpoint` 现在是明确的
   迁移错误，恢复语义由 `--warm-start` 承担（只载权重，optimizer/step/best 从零）。
2. **固定协议此前并非 batch 无关**：C03 起每个验证项一行 mask，C09 用同一 manifest 在 B=2 与 B=1
   各评一次复验（逐行 NLL 1e-4 内一致）。
3. **“SHA 已绑定”此前不成立**：C02 之前 checkpoint 只是把 tokenizer SHA 写进 payload，加载时不比较；
   现在 `load_operator_bundle` / 训练入口 / 评估入口都要求文件级比对，真实数据上已验证
   （旧 1h tokenizer 配 holdout store 会被 store_identity 拒绝）。
4. **在线编码路径的旧数值需重跑**：packed feature store 的原始帧曾被当成已归一化帧送入 tokenizer，
   同一窗口约 71% token 不一致；修正后 CUDA 上逐位一致、CPU 边界量化差异 0.13%
   （`nef_data.store_normalized_window`，C09-D 证据）。用该路径得到的 MTS/probe 读数按修正后的实现重跑。
5. **物理指标（R11）**：旧的 `boundary_jerk_max` 已更名 `feature_delta_change_max`（单位不同），
   新的 jerk 是 FK world position 的三阶差分（m/s³），source/base/styled 三个对比分开报告。
6. **`/tmp/run_matrix.sh` 的 26-cell 矩阵**：`--operator` 覆盖在旧版本被接受但未生效，已修；
   但矩阵里的旧算子 checkpoint 属 schema 1，不能作为 revision-2 结果引用。

revision 2 的门槛状态、逐任务证据与下一轮实验申请见
[MTS_FSQ_Code_Revision_Progress_zh.md](MTS_FSQ_Code_Revision_Progress_zh.md) 与
[mts_revision2_experiment_request_zh.md](mts_revision2_experiment_request_zh.md)。

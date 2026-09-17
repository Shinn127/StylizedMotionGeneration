# SEED 第二阶段：performer 标签、物理 fine-tune、R1 表征对照

日期：2026-09-17。承接 [seed_1h_run_report.md](seed_1h_run_report.md)（全量数据构建 + 1 小时基线）。
本文记录三条推进线的结果：② performer 列与 zero-shot 审计、① 物理 staged objective 的 v1 vs v1.1、
③ flat/part 对照与 R1 局部性判据。所有评估都在同一个 packed store、同一批 test 窗口上完成。

## ② performer 列：重建 + 真实 zero-shot 审计

`packed-feature-store --overwrite` 复用全部 unit 缓存重建（**units_reused = 142,220**，无重新解析，约 7 分钟含全量校验），
clip 表新增 `clip_performer_id`，manifest 新增 `performer_names`（来自 catalogue 的 `take_actor`，522 个演员）。
重建后 `train-stats` 复算出同一个归一化 hash（`278da45c…`），1 小时 checkpoint 依旧通过校验——
说明补标签对已训练的模型完全透明。

审计结果（`outputs/seed_style_audit/`）：

- `performer_analysis: overlap_reported`（此前为 `unavailable_no_actor_table`）。
- style 划分 train 4 / val 2 / unseen 2，抽样 512 对，`same_clip_leakage = 0`。
- **每个 unseen style 的演员都出现在训练 style 里**（`injured torso to neutral` 14 人、`old` 4 人重叠）。
  结论：SEED 的 8 个 style 标签**无法构成演员不相交的 zero-shot 划分**，style 层面的 holdout 会把
  风格与演员身份混在一起。zero-shot 声明必须用 catalogue 级的 actor holdout
  （`seed-catalog --actor-holdout`），审计现在会把这个警告直接写进 `warnings`。
- 另外两条标签层面的警告：`neutral` 占 92% 的 clip；8 个 style 中 5 个只有单一 content 标签。

## ① 物理 staged objective：v1 vs v1.1

预算与调度（新加的 runner 语义）：

| | v1 基线 | v1.1 物理 fine-tune |
|---|---|---|
| 起点 | 全新训练 | 从 v1 的 best.pt **续跑** epoch 72 / step 14,000 |
| warmup / ramp | — | 2 / 6 epoch（**相对续跑起点**计数） |
| 物理权重 | 全 0（recon + 3·delta） | root_pos .05 / root_rot .05 / joint .10 / contact .03 / foot_slide .05 / foot_height .02 |
| LR 调度 | cosine 80 epoch | **重启** cosine（`reset_scheduler_on_resume`），optimizer 动量沿用 |
| best 判定 | 全量验证 | 继承的 best 被清除（`reset_best_on_resume`，目标不同不可比） |
| 步数 | 14,000 | 共 25,350（14,000 + 11,350） |

全量验证（val_full，v1.1 打 epoch 130）：

| 指标 | v1 | v1.1 | |
|---|---|---|---|
| recon | 0.09978 | **0.09723** | v1.1 更好 |
| delta | 0.02403 | 0.02422 | 持平 |
| joint | 未计算 | 0.00234 | v1 权重为 0，无法报告 |
| contact | 未计算 | **0.03418**（epoch 80 时 0.272） | 持续下降 |
| foot_slide | 未计算 | **0.12370** | |
| foot_height | 未计算 | 0.05298 | |
| root_pos / root_rot | 未计算 | 0.00919 / 0.00996 | |

**解码侧物理指标**（`scripts/evaluate_nef_physics.py`，128 test 窗口；与训练权重无关，可跨 checkpoint 比较）：

| checkpoint | FK mean | FK median | 最差关节 | root 漂移 | root 旋转 | foot slide | 接触准确率 | 接触召回 | 脚高误差 |
|---|---|---|---|---|---|---|---|---|---|
| NEF v1 (14k) | **8.22 cm** | 6.27 | RightToeBase 15.1 cm | **0.93 cm** | **0.41°** | 0.330 m/s | 0.526 | 0.379 | 4.27 cm |
| NEF v1.1 (25.4k) | 8.71 cm | 6.74 | RightToeBase 15.6 cm | 2.05 cm | 0.54° | **0.277 m/s** | **0.620** | **0.505** | **4.02 cm** |
| flat (14k) | 23.60 cm | 21.47 | RightHand 40.0 cm | 16.47 cm | 4.87° | 0.004 m/s† | 0.760† | 1.000† | 2.83 cm |
| part (14k) | 13.39 cm | 11.49 | RightHand 20.8 cm | 10.44 cm | 4.87° | 0.200 m/s | 0.699 | 0.614 | 2.20 cm |

† flat 的解码足部几乎静止（foot slide 0.004 m/s），于是"永远接触"就能拿到高准确率与召回 1.000：
这是退化解，不能读作 flat 的接触建模更好（precision 0.760 恰等于目标接触率即为证据）。

**①的结论**：物理目标确实作用在预期的通道上——接触召回 +33%、准确率 +18%、脚滑 −16%、脚高 −6%，
代价是 FK 平均误差 +6%、root 漂移 0.93 → 2.05 cm。**注意这里有一个未排除的混淆**：v1.1 比 v1 多练了
11,350 步，所以"目标变化"与"训练更久"尚未分离。下一步的对照是让 v1 用同一预算继续跑 recon+delta
（约 39 分钟）再比一次：

```bash
# 用 1h 配置从 v1 best.pt 续跑 11,350 步、同样重启 LR、同样清除继承 best
# 只改 training：max_steps 25350、epochs 141、scheduler_epochs 61、
#                reset_scheduler_on_resume/reset_best_on_resume: true
```

## ③ R1：flat / part / NEF 的局部编辑对照

配置 `flat_fsq_soma_packed_40x9_1h.yaml`、`part_fsq_soma_packed_40x9_1h.yaml` 与 NEF 基线
**数据、采样、seed、batch、预算、目标完全相同**，只有表征不同（part 因 SOMA 含 LeftEye/RightEye/Jaw，
在配置里显式声明了 `part_membership`）。三者都在 test 上完成 14,000 步（flat 37 分钟、part 23 分钟）。

同一批 64 个 test 窗口上的局部编辑（`outputs/stage3_eval/locality/`，token 交换到 `left_arm` 等区域）：

| 表征 | 支持集 | 支持内 token 改动率 | 区域外关节最大变化 | 腿编辑的接触翻转 |
|---|---|---|---|---|
| flat | **100%**（无区域契约，只能整体编辑） | 0.112 | **1.37 / 1.11 m** | 0.086 |
| part | 12.5%（臂）/ 17.5%（腿） | 0.34–0.40 | **0.0** | 0.35 / 0.38 |
| NEF v1 | 10% | 0.23–0.38 | **0.0** | 0.098 / 0.074 |
| NEF v1.1 | 10% | 0.23–0.37 | **0.0** | 0.108 / 0.116 |

三条可审稿的读数：

1. **flat 的编辑就是全身编辑**：区域外关节位移 1.37 m。flat 的 `off_target_feature_max` 为 0 只是因为
   它的支持集覆盖全部 40 个 coordinate，泄漏只能从关节侧看到。
2. **part 与 NEF 都做到了精确局部性**（区域外关节变化 0.0）。所以"局部"本身不是 NEF 独有的，
   NEF 的差别在下面两点。
3. **NEF 用更小的支持集达成同样的目标变化**（10% vs part 的 12.5–17.5%），并且**腿编辑对接触的副作用小 3–5 倍**
   （0.074–0.116 vs part 的 0.35–0.38）——这正是"contacts 归 global stream、腿流独立"的结构后果。
   同时 NEF 的重建质量明显更好（FK 8.2 cm vs part 13.4 cm vs flat 23.6 cm）。

R1 判据（方案 §3）因此部分成立：**NEF 的空间所有权确实带来更小的编辑支持集与更小的接触副作用**；
但"更局部"这一条对 part 同样成立，论文不能把 exact locality 当作 NEF 的独有贡献，
应写成"更小的支持集 + 更少的接触副作用 + 更好的重建质量"。

## 本轮修掉的问题

| 问题 | 影响 | 修复 |
|---|---|---|
| 物理调度按绝对 epoch 计数 | 续跑时 warmup/ramp 被跳过，直接满权重 | 改为相对本次运行起点计数 |
| 续跑继承旧 LR 调度 | fine-tune 以约 0.5% 峰值 LR 训练，几乎不动 | `reset_scheduler_on_resume`（新阶段重启 cosine） |
| 续跑继承旧 best_val | 目标不同、数值不可比 → `best.pt` 永远不写 | `reset_best_on_resume` |
| `epochs`/`max_steps` 是绝对预算 | fine-tune 的步数/调度长度无法表达 | 补 `scheduler_epochs`，配置里用绝对总量 |
| 我的物理评估器阈值化了归一化后的接触特征 | 接触/脚滑指标全 0 或矛盾 | 先反归一化再阈值，并输出目标/推断接触率 |
| `validate_checkpoint_store` 只认 NEF layout | flat/part 无法进入 R1 对照 | 抽出与 family 无关的 `validate_checkpoint_against_store` |

## 复现命令

```bash
# ② 重建带 performer 的 store（复用 unit 缓存）+ 审计
python -m stylized_motion.run --mode preprocess --pipeline packed-feature-store \
  --catalog data/processed/seed_full_catalog --output data/processed/seed_soma_pruned_v4 \
  --unit-cache data/processed/seed_soma_pruned_v4_units --workers 8 --verify full --overwrite
python -m stylized_motion.run --mode preprocess --pipeline train-stats --store data/processed/seed_soma_pruned_v4
python scripts/audit_style_pairs.py --feature-database data/processed/seed_soma_pruned_v4 --output outputs/seed_style_audit

# ① 物理 fine-tune
python -m stylized_motion.run --mode train --pipeline representation --representation nef-fsq \
  --config data/configs/nef_fsq_soma_packed_40x9_physical_ft.yaml \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_1h/best.pt --device cuda

# ③ R1 对照训练
python -m stylized_motion.run --mode train --pipeline representation --representation flat-fsq \
  --config data/configs/flat_fsq_soma_packed_40x9_1h.yaml --device cuda
python -m stylized_motion.run --mode train --pipeline representation --representation part-fsq \
  --config data/configs/part_fsq_soma_packed_40x9_1h.yaml --device cuda

# 评估：表征局部性对照 + 解码侧物理
python scripts/evaluate_nef_locality.py \
  --checkpoint outputs/flat_fsq_soma_packed_40x9_1h/best.pt \
  --checkpoint outputs/part_fsq_soma_packed_40x9_1h/best.pt \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_1h/best.pt \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_physical_ft/best.pt \
  --feature-database data/processed/seed_soma_pruned_v4 --split test \
  --parts left_arm right_arm left_leg right_leg --max-clips 64 --device cuda \
  --output outputs/stage3_eval/locality
python scripts/evaluate_nef_physics.py \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_1h/best.pt \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_physical_ft/best.pt \
  --checkpoint outputs/flat_fsq_soma_packed_40x9_1h/best.pt \
  --checkpoint outputs/part_fsq_soma_packed_40x9_1h/best.pt \
  --feature-database data/processed/seed_soma_pruned_v4 --split test --windows 128 \
  --device cuda --output outputs/stage3_eval/physics
```

## 待办

1. **① 的等预算对照**：v1 继续跑 11,350 步 recon+delta，把"目标变化"与"训练更久"分离（约 39 分钟）。
2. **actor holdout**：`seed-catalog --actor-holdout` 冻结整批演员，重建 catalog/store（或只在 token 层面过滤），
   才能给出真正的 zero-shot style 划分。
3. **flat/part 的 token store 与 MTS 接入**：R1 只完成表征侧；算子侧（同一 global style descriptor 作用于
   flat/part/NEF）仍待 token store。

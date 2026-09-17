# SEED actor holdout：zero-shot 泛化的两条轴

日期：2026-09-17。P1 决策选 A（SEED + actor holdout）。本文记录划分设计、产物、验证与在
held-out 演员 test split 上的重评估结果。

## 1. 为什么要 actor holdout

第一轮审计发现 SEED 的 8 个 style 标签在**任何** 10% 演员抽样下都被完整覆盖：把 style 分成
train/unseen 两半时，每个 unseen style 的演员都出现在训练集里。也就是说 style 层面的 holdout
无法把"风格"与"演员身份"分开。正确做法是把泛化拆成两条互不混淆的轴：

| 轴 | 定义 | 需要什么 | SEED 是否支持 |
|---|---|---|---|
| **unseen performer** | 目标/参考片段来自训练中从未出现的演员 | catalogue 级 actor holdout（整组进 test） | ✅ 现在支持（52 演员 / 14,105 clip） |
| **unseen style** | 某个 style 在算子参数学习中完全没出现过 | 显式从训练 pairs 中排除该 style（`data.pairs.held_out_styles`） | ✅ 机制已就位；SEED 上可用 style 只有 4 个有规模，主实验建议放 100STYLE |

两者的混淆关系是单向的：unseen style 若其演员也在训练集里**不是**混淆（模型从未见过该 style，
不可能靠演员反推它的语义）；而 unseen performer 若其 style 在训练集里则是真混淆（模型可能记住
"演员 A = style X"）。

## 2. 划分与产物

```bash
python -m stylized_motion.run --mode preprocess --pipeline seed-catalog \
  --seed-root data/raw/seed --output data/processed/seed_ah_catalog \
  --mirror-policy official --actor-holdout-ratio 0.10 --overwrite
python -m stylized_motion.run --mode preprocess --pipeline packed-feature-store \
  --catalog data/processed/seed_ah_catalog --output data/processed/seed_soma_pruned_v4_ah \
  --unit-cache data/processed/seed_soma_pruned_v4_units --workers 8 --shard-mib 256 --verify full
python -m stylized_motion.run --mode preprocess --pipeline train-stats --store data/processed/seed_soma_pruned_v4_ah
```

- holdout 比例 0.10 → **52 个演员整组冻结进 test**（`max(1, round(522 × 0.1))`，按 `seed + actor` 哈希确定性选取）。
- 演员 clip 数分布很均匀（2–899，中位 190），所以 0.10 恰好对应 **14,105 clip（9.9%）**。
- 校验：`train ∩ test 演员 = 0`、`val ∩ test 演员 = 0`、**0 个 take group 跨 split**（catalog 与 store 两层都查过）。
- split：train 113,883 / val 14,232 / test 14,105 clip；`train-stats` 的归一化 hash 变为 `a7d2fb6e…`
  （train 集合变了，特征字节未动）。
- **unit 缓存全部复用（units_reused = 142,220）**：unit 签名只含预处理配置，不含 split，所以整组重建约 8 分钟。
- 旧 store（`data/processed/seed_soma_pruned_v4`）与旧 checkpoint 全部保留，两者可并存比较。

## 3. 审计输出（两条轴）

`outputs/seed_style_audit_ah/style_pair_audit.json`：

```json
"performer_axis": {"train_actors": 468, "val_actors": 416, "test_actors": 52,
                   "train_test_actor_overlap": 0, "zero_shot_performer_supported": true},
"style_axis":     {"styles_in_test_only": [], "held_out_styles": [],
                   "zero_shot_style_supported": false}
```

`zero_shot_style_supported: false` 是**结构性事实**而非缺陷：没有 style 只属于被冻结的演员。
审计会给出提示——要造出 unseen-style 轴，必须在算子训练时排除 style
（`data.pairs.held_out_styles`，采样器已实现：train 阶段既不选该 style 作 target 也不作 reference）。

采样器的 stage 语义也随之升级：有数据划分时 `stage` 直接选 `record.split`
（train 目标与参考都来自训练演员，test 全部来自被冻结演员）；没有划分信息的旧 store 仍按
style 词表选（`use_data_splits=False` 是显式回退）。

## 4. 在 held-out 演员 test split 上的重评估

**注意**：现有 4 个 checkpoint 是在**旧划分**上训练的（那时这 52 个演员还在训练集里），
所以下表是"换到新 test split 的测量"，**不是** zero-shot performer 结果。

R1 局部性（64 窗口，结构结论与旧 split 一致）：

| 表征 | 支持集 | 区域外关节最大变化 | 腿编辑接触翻转 |
|---|---|---|---|
| flat | 100% | 1.37 m | 0.141 |
| part | 12.5% / 17.5% | **0.0** | 0.33–0.39 |
| NEF v1 | 10% | **0.0** | 0.108 / 0.102 |
| NEF v1.1 | 10% | **0.0** | 0.125 / 0.141 |

解码侧物理（128 窗口）：

| checkpoint | FK mean | root 漂移 | foot slide | 接触准确率 | 接触召回 | 脚高误差 |
|---|---|---|---|---|---|---|
| NEF v1 | 11.42 cm | 0.76 cm | 0.224 m/s | 0.483 | 0.425 | 9.68 cm |
| NEF v1.1 | 11.63 cm | 1.52 cm | **0.174 m/s** | **0.613** | **0.571** | 9.37 cm |
| flat | 20.46 cm | 9.47 cm | 0.004 m/s† | 0.898† | 1.000† | 6.46 cm |
| part | **9.08 cm** | 6.65 cm | 0.133 m/s | 0.666 | 0.632 | 2.17 cm |

† flat 仍是"永远接触"的退化解（见 seed_stage2_promotion.md §③）。

两点值得记录：**① 的接触/脚滑优势在未见演员上保持**（召回 0.425 → 0.571、脚滑 −22%）；
但 **FK 与脚高的排序在新 split 上变了**（part 的 FK 9.08 cm 反超 NEF 的 11.42 cm，NEF 的脚高误差
从 4.27 cm 升到 9.68 cm），说明 128 窗口样本上这些指标方差不小，且新 test 演员的动作分布不同。
任何"某表征更好"的结论都必须在同一 split 上、用足够窗口数、并在 (表征 × split) 两个方向都复核。

## 5. 下一步（按依赖顺序）

1. **在 holdout store 上重训 tokenizer**（NEF 先做，v1.1 配方）——只有 tokenizer 也尊重 holdout，
   operator 的 zero-shot performer 结果才干净：
   ```bash
   python -m stylized_motion.run --mode train --pipeline representation --representation nef-fsq \
     --config data/configs/nef_fsq_soma_packed_40x9_physical_ft.yaml \
     --checkpoint outputs/nef_fsq_soma_packed_40x9_physical_ft/best.pt --device cuda
   # 配置里把 data.fsq_window_index 换成 data/processed/seed_soma_pruned_v4_ah
   ```
2. **等预算 v1 对照**（上一阶段遗留）：把"物理目标"与"训练更久"分离，约 39 分钟。
3. **packed token store**（NEF/flat/part）→ Phase 2 transport → Phase 3/4 operator，
   评估时 train 阶段用 `stage="train"`（训练演员）、报告用 `stage="test"`（冻结演员），
   unseen-style 臂再用 `held_out_styles`。

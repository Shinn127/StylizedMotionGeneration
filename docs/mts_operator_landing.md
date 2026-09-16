# MTS-FSQ 代码落地说明

本文记录 [MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md](MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md) 的代码落地情况：
每个阶段有什么入口、能跑出什么、哪些结论**还没有**被验证。方案本身未改动，仍是验收依据。

## 1. 落地对照表

| 方案条目 | 代码 | 入口 | 状态 |
|---|---|---|---|
| §4.1 研究级 token contract | `nef_layout.py`、`nef_fsq.py`、`representation.py` | — | 已落地 |
| §7 Phase 0 探针 | `nef_probe.py` | `scripts/probe_nef_geometry.py`、`scripts/evaluate_nef_locality.py` | 已落地，见 §2 |
| §4.2 物理 warmup 目标 | `runner.py`（`physical_schedule_scale`） | `data/configs/nef_fsq_40x9_physical.yaml` | 已落地 |
| §5.1 Tensor contract | `mts_operator/contract.py` | — | 已落地 |
| §5 目录与 layout adapter | `mts_operator/layout_adapter.py` | — | 已落地 |
| §7 Phase 1 style pair audit | `mts_operator/pairs.py` | `scripts/audit_style_pairs.py` | 已落地 |
| §7 Phase 2 base transport | `embeddings.py`、`graph.py`、`masking.py`、`transport.py`、`training.py` | `scripts/train_mts_transport.py` | 已落地 |
| §6 operator 三族 | `operators.py` | `--operator {logit_field,arbitrary_kernel,birth_death}` | 已落地 |
| §7 Phase 3/4 参考条件训练 | `style_encoder.py`、`model.py` | `scripts/train_mts_operator.py` | 已落地 |
| §7 Phase 5 / §8 评估与出图输入 | `metrics.py` | `scripts/evaluate_mts_operator.py`、`scripts/generate_mts_operator.py` | 已落地 |
| §5 `checkpoint.py` | `mts_operator/checkpoint.py` | — | 已落地 |

## 2. 四个必须先看的读数

1. **`summary.ordinal_geometry_supported`（probe_geometry.json）** 只是筛查：中位 `adjacent_to_far_ratio < 1`
   且平均 `direction_consistency > 0`。成立才值得实现 birth-death；不成立按方案 §11 失败条件 A 走
   coordinate-aware kernel。它不构成“FSQ level 有动作语义”的结论。
2. **`adjacent_to_far_ratio` 的分母是 feature_l1**（对全部 motion 特征取均值），不是求和；跨 checkpoint
   比较时只看比值，不看绝对值。
3. **`support_token_change_fraction`（locality.json）** 是读所有局部性数字的前提：它为 0 时，
   `off_target_* = 0` 只说明 donor 与 target 在该区域本来就相同。
4. **`changed_token_ratio` 是 coupling 统计量**：`paired_comparison()` 同时返回与耦合无关的
   `total_variation`，并在 `note` 里写明这一点。报告里引用哪一个必须说明。

## 3. 一条可执行的路径

```bash
# Phase 0：先证明/否证表征可用
python scripts/probe_nef_geometry.py --checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --split test --max-clips 256 --output outputs/nef_probe/v1
python scripts/evaluate_nef_locality.py --checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --split test --parts left_arm right_arm left_leg right_leg --output outputs/nef_locality/v1

# Phase 1：style pair 审计（无 same-style/different-content 证据就此停下）
python scripts/audit_style_pairs.py --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --output outputs/mts_pairs/audit

# Phase 2：无 style 的 base transport
python scripts/train_mts_transport.py --config data/configs/mts_operator_transport.yaml \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --overfit-clips 8 --epochs 2            # 先小批过拟合，再全量
python scripts/train_mts_transport.py --config data/configs/mts_operator_transport.yaml \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt --output outputs/mts_transport/seed3407

# Phase 3：style-ID sandbox（把“算子不行”与“encoder 没提取到风格”分开）
python scripts/train_mts_operator.py --config data/configs/mts_operator_style.yaml \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --transport-checkpoint outputs/mts_transport/seed3407/best.pt \
  --operator birth_death --output outputs/mts_operator/style_id/seed3407

# Phase 4/5：参考条件训练 + 评估（三种 operator 同一套配置，只换 --operator）
python scripts/evaluate_mts_operator.py --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --split test --output outputs/mts_eval/main
python scripts/generate_mts_operator.py --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --content-clip 0 --style-clip 3 --regions left_arm --graph-radius 1 \
  --frame-range 80 140 --strength 1.0 --seed 1234 --locked-edit \
  --output outputs/mts_samples/example
```

`style_encoder.kind: style_id` 走 Phase 3（固定 style id），`kind: reference` 走 Phase 4（参考片段编码）；
两者输出同一宽度的全局 descriptor，接口不变。

## 4. 已用真实数据验证的部分

- **表征契约**：`layout_hash` 固定（Geno/SOMA 各一个），`stream_graph` 与 `make_region_mask` 的半径语义
  有测试锁住；`encode_indices/decode_indices` 与旧接口逐位一致。
- **translocality**：编辑 `left_arm_node` 时，图深度 0/1/2 分别只影响 `{left_arm}`、
  `{left_arm, left_shoulder_edge}`、`{+torso_node}`——这也是算子 support 论证要用的同一句话。
- **解码器时间影响**：单帧 token 编辑最多影响其后 33 帧（= decoder RF−1），探针给出 `within_contract`。
- **局部编辑**：NEF 局部 token 交换后，区域外解码特征与区域外 FK 变化为**精确 0**；
  flat/part/NEF 的对照由 `evaluate_nef_locality.py` 在同一批窗口上给出。
- **style pair 审计（真实 100STYLE manifest）**：101 个 style、每个 style 8 个不同 content、
  抽样 256 对 0 泄漏、train/val/unseen = 61/20/20。
- **CTMC 数值**：uniformization 与 `torch.matrix_exp` 一致（<1e-5），质量守恒、非负性、
  零生成元严格恒等、半群性质 `exp(Qs)exp(Qt)=exp(Q(s+t))` 都有测试；`strength=0`（或 hard mask 全 0）
  返回**逐位相同**的 base 分布。
- **训练链路**：transport 与 operator 的脚本都在真实（合成）store + 真实 checkpoint 上端到端跑通，
  checkpoint 记录 token/layout 指纹并在不匹配时拒绝加载。

## 5. 明确没有做的部分

- **没有跑全量训练**：仓库里没有 NEF checkpoint，Phase 0 的探针脚本只在合成 store 上做过
  端到端冒烟；`outputs/` 下现有的 checkpoint 属于其它 family。因此本文不提供任何
  adjacent/far 几何、style response 的真实数值结论。
- **没有 style 分类器 / FID / 文本对齐**：`metrics.unavailable_metrics()` 明确列出，
  不用替身凑数。
- **没有实现 basis/gate/composition**：方案 §6.3 要求先比较 direct CTMC 与 logit baseline。
- **style encoder 未做 region-specific 输出**：descriptor 是全局的（方案 §5.3 要求）；
  区域控制由 operator 的 hard mask 提供。
- **`windows.py` 的 token store 路径**只覆盖 v3 `TokenStore.read_indices`；
  v4 packed token store 需要各自的读取分支（当前走 feature store 在线编码）。
- **SEED 的 style 标签很粗**：8 个 `content_uniform_style`，其中 `neutral` 占 92%，5 个 style 只有单一
  content 标签——`audit_style_pairs.py` 会把这两点写进 `warnings`。做 SEED 上的算子实验前需要先决定
  风格来源（class balance、`content_all_rigplay_styles`，或换用 100STYLE）。
- **当前 SEED store 没有 performer 列**（`performer_analysis: unavailable_no_actor_table`）：
  zero-shot style 无法排除演员混淆。构建侧已支持（`take_actor` → `clip_performer_id` +
  manifest `performer_names`），重跑 `packed-feature-store --overwrite` 即可补上。
- **`evaluate_mts_operator.py` 的 representation × method 全矩阵**未展开：脚本按 checkpoint 评估
  单个方法，矩阵由多次调用 + 外部汇总构成（方案 §6.5 的对照配置已就位，缺的是算力与数据）。

## 6. 数据前提

`--feature-database` 支持 v3（`feature_data.py`）与 v4（`packed_store.py`）两代 store。
本机 `data/processed/combined_pruned_90/feature_cache` 是缺少 `split_manifest_hash` 的旧 store，
当前代码会拒绝打开——这是既有的数据布局问题，与本次改动无关（`docs/bones_seed_pipeline_implementation.md` §5 已记录）。

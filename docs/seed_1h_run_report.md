# BONES-SEED 全量数据 + NEF-FSQ 一小时训练记录

日期：2026-09-17。本文记录一次**跑通完整链路**的运行结果：数据构建 → 训练 → 评估。
目标是初步效果与链路正确性，不是收敛模型；所有数字都标明了预算与 split。

## 1. 数据：schema-v4 packed store（全量）

| 阶段 | 命令 | 结果 |
|---|---|---|
| catalog | `preprocess seed-catalog --mirror-policy official` | 142,220 clip / 62.3M 帧（60fps） |
| inventory | `preprocess seed-inventory --workers 16` | 142,220 OK，0 错误，0 帧数不符，1 个骨架 hash，帧率全部 120/60 |
| store | `preprocess packed-feature-store --workers 12 --shard-mib 256 --verify full` | 231 分片，60 GB 特征 + 1.7 GB 根通道 |
| 统计 | `preprocess train-stats` | 49,749,048 训练帧，归一化 hash `278da45c…` |

- 划分：train 113,784 / val 14,216 / test 14,220 clip（take group 整组同 split，0 泄漏）。
- 构建耗时：catalog+inventory ≈ 1 分钟，store ≈ 27 分钟（unit 处理 ~0.13 s/文件，12 worker；打包 ~836 clip/s）。
- 产物位置：`data/processed/seed_full_catalog`、`data/processed/seed_soma_pruned_v4`（含 `_units` 缓存 61 GB，可删）。
- `window_coverage`：train 42.6M / val 5.35M / test 5.42M 个 64 帧窗口（store 自校验输出）。

## 2. 训练：一小时预算

配置 `data/configs/nef_fsq_soma_packed_40x9_1h.yaml`（representation/data/sampling 与 canonical packed 配置完全一致，
只有预算不同；outputs/nef_fsq_soma_packed_40x9_1h）。预算来自实测：

| 指标 | 实测值 |
|---|---|
| 训练步时间 | 0.217 s（batch 512，64 帧，RTX 5070 Ti） |
| 吞吐 | 2,510 samples/s（137k frames/s） |
| 验证 | 0.078 s/batch，178 batch = 一次全量扫描 ≈ 14 s |
| 预算 | 14,000 步（195 步/epoch，72 epoch）≈ 51 min + 8 次全量验证 |

结果（best.pt 由全量验证选出，`best_metric_source: val_full`）：

| | epoch 10 | epoch 30 | epoch 50 | epoch 72 |
|---|---|---|---|---|
| val_full loss | 0.2099 | 0.1818 | 0.1747 | **0.1719** |
| recon | 0.1236 | 0.1066 | 0.1019 | **0.0998** |
| delta | 0.0288 | 0.0251 | 0.0243 | **0.0240** |

训练 loss 0.450（epoch 1）→ 0.186（epoch 72）。曲线仍在下降但已明显变缓：这是**约 8.5% 数据通过量**的
早期快照（7.2M 样本 / 42.6M 训练窗口），不是收敛结果。

## 3. 评估（test split，128 窗口 / 8,192 帧）

### 3.1 重建与物理量（`nef_report_test.json`）

| 指标 | 值 |
|---|---|
| overall recon（加权） | 0.351 |
| delta | 0.0263 |
| root 位置误差 | 0.0023 |
| root 旋转误差 | 0.0050 rad（0.29°） |
| mean FK 世界误差 | 0.156 m |
| Head 关节 FK / 旋转 | 0.033 m / 0.342 rad |

按关节看（64 窗口，`/tmp/perjoint.py` 复算）：**mean 9.6 cm / median 3.8 cm**；
最差是脚趾/手/脚（14.5–16.0 cm），最好在 root/颈部（0.8–7.4 cm）；
root 轨迹漂移 0.8 cm/64 帧、旋转 0.41°。误差沿运动链累积，且集中在角速度通道（见 §3.4）。

### 3.2 Level 几何（`nef_geometry/probe_geometry.json`）

- 40/40 coordinate 的 ±1 level 变化小于远距离跳跃；中位 `adjacent_to_far_ratio` **0.542**，
  平均 `direction_consistency` **0.945** → `ordinal_geometry_supported: True`（方案 §13 第 1 个决策点通过）。
- 按流看，Edge 流的 ratio 最差（shoulder_edge 0.79–0.82、hips_edge 0.67），Node 流最好（leg 0.51、torso 0.55）；
  direction consistency 反而 Edge 更高（0.93–0.99）。也就是说：**相邻 level 的“动作语义”在 Node 流上更清晰，
  在只承载单个关节旋转的 Edge 流上更弱**——这正好是 birth-death 需要谨慎的地方。
- 解码器时间影响：单帧 token 编辑最多影响其后 **31 帧**（≤ RF−1 = 33，`within_contract: True`）。

### 3.3 局部性（`nef_locality/locality.json`，64 窗口）

| 编辑区域 | support | donor 改动率 | 区域外特征变化 | 区域外关节变化 | contact 翻转 |
|---|---|---|---|---|---|
| left_arm | 4/40 | 0.23 | **0.0** | **0.0** | 0.000 |
| right_arm | 4/40 | 0.24 | **0.0** | **0.0** | 0.000 |
| left_leg | 4/40 | 0.33 | **0.0** | **0.0** | 0.098 |
| right_leg | 4/40 | 0.38 | **0.0** | **0.0** | 0.074 |
| whole_body | 40/40 | 0.27 | 0.0 | 0.0（全部为目标） | 0.145 |

token 交换（`nef_transfer_left_arm.json`）：`exact_stream_isolation: true`、区域外特征变化**精确 0**、
影响区间 `[63,127]` 且窗口之后逐位不变。腿部编辑会经接触通道产生 7–10% 的 contact 翻转，
正是方案 §2.2 提醒的“contacts 属于 global stream”的副作用，需要按 facet 报告。

### 3.4 训练中暴露的两个表征问题

1. **加权 loss 被 Hips 的少数通道主导**。按 `err × feature_weight` 排序，前 4 名是
   `hips_edge` 的 Hips 角速度（idx 168/169，weight 7.04，误差≈目标量级 → 基本预测为 0）、
   Hips 竖直速度（idx 166，误差 2.45 vs 目标 2.64）、Hips 旋转分量（idx 13）。
   `hips_edge` 的加权 recon 因此高达 2.05，而 `head_node` 只有 0.03。
   这也是“mean FK 9.6 cm”的主要来源：手/脚/脚趾的误差就是这些角速度通道没拟合好的表现。
2. **level 使用是健康的，不是塌缩**：平均 perplexity 6.32/9，40 个 coordinate 中 0 个接近二值化，
   9 个 level 在每个流里都被用到（3 个 coordinate 的 0/8 极端质量 >50%，集中在 hips_edge 与腿部 Node 流）。

结论：一小时快照下，**空间所有权与因果性已经完全成立**（局部性、RF、路由都可复现），
**重建精度仍受角速度通道与数据通过量限制**——下一步应优先做方案 §4.2 的物理 warmup 版本或延长训练，
而不是改 representation。

## 4. 本轮修掉的链路缺陷

| 缺陷 | 影响 | 修复 |
|---|---|---|
| `nef_eval.py` 只认 v3 store | SEED 上无法出 canonical 报告与 transfer | 抽出 `nef_data.py`（两代 store 统一读取），eval/probe/windows 共用 |
| packed store 无 performer 列 | zero-shot style 无法排除演员混淆 | 新增可选列 `clip_performer_id` + manifest `performer_names`（来自 `take_actor`），老 store 仍可打开 |
| 审计把 group id 当演员 | `val_styles` 静默为 0，重叠分析形同虚设 | 无演员表时明确报 `unavailable_no_actor_table`，改为 style-only 划分 |
| 探针把子孙关节算成泄漏 | Edge 流“泄漏”1.4–2.0 m 的假象 | 泄漏只看非子孙关节；新增 `fk_descendant_mean` / `fk_influence_mean`；global 流不再声明泄漏 |
| 探针无法在 CUDA 上跑 | GPU 评估直接崩 | kinematic context 随模块设备迁移；随机数在 generator 设备上生成后再搬运 |
| SEED style 标签很粗 | 算子实验的证据不足 | 审计输出 `warnings`：neutral 占 92%、8 个 style 中 5 个只有单一 content |

## 5. 下一步建议（按方案顺序）

1. **延长或物理化训练**：`nef_fsq_soma_packed_40x9_physical.yaml` 的 staged objective（warmup 10 + ramp 20）
   直接针对 §3.4 的角速度/接触问题；也可用 canonical packed 配置从当前 checkpoint 续跑。
2. **补 performer 列**：`packed-feature-store --overwrite` 复用 unit 缓存即可（约 3 分钟打包），
   之后 `audit_style_pairs.py` 就能给出真实的演员重叠与 zero-shot 划分。
3. **补 flat / part 对照 checkpoint**：`evaluate_nef_locality.py` 已支持，缺的是同一数据上的对照训练，
   这是方案 R1 的判据（NEF 是否真的更局部）。
4. **token store**：MTS 阶段需要（`packed-token-store`），当前 transport/operator 脚本可在线编码但更慢。

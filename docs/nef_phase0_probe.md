# NEF Phase 0 探针使用说明

本文对应 [MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md](MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md) 的 Phase 0：
在训练任何 style operator 之前，先用两个**可否证伪**的测量决定后续路线。代码位于
`stylized_motion/learning/nef_probe.py`，命令行入口是 `scripts/probe_nef_geometry.py` 与
`scripts/evaluate_nef_locality.py`。

`--feature-database` 同时支持两代 store：v3（`feature_data.py`）与 v4（`packed_store.py`）。
脚本只接受 NEF-FSQ checkpoint；`evaluate_nef_locality.py` 也接受 flat / part-FSQ checkpoint，
以便在相同窗口上做对照。

## 1. FSQ level 几何探针

```bash
python scripts/probe_nef_geometry.py \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --split test --max-clips 256 --output outputs/nef_probe/v1
```

对真实 token 的每个 coordinate 做 `±1` 与「远距离」level 扰动，解码后比较：

- `adjacent_distance` / `far_distance` / `adjacent_to_far_ratio`：主指标是解码特征的 L1 变化
  （`feature_l1`，分母是全部 motion 特征数）；
- `direction_consistency`：`+1` 与 `-1` 的平均解码变化向量之反平行程度，1.0 表示两个方向严格反向，
  即该 coordinate 存在单调的 ordinal 轴；
- 逐项子指标：`feature_l2`、`stream_feature_l1`、`offtarget_feature_max`、`fk_owned_mean`、
  `fk_offtarget_max`、`root_pos_change`、`root_rot_change`、`contact_flip_rate`、
  `velocity_change`、`jerk_change`（后六项需要 `--no-kinematics` 之外的模式）。

输出 `probe_geometry.json`（完整报告 + `temporal_influence`）与 `probe_geometry.csv`（每 coordinate 一行）。

**怎么读**：`summary.ordinal_geometry_supported` 只是筛查用的启发式（中位 ratio < 1 且平均
direction consistency > 0）。它成立只说明「可以为 birth-death 写代码」，不构成“FSQ level 有动作语义”
的结论。若不成立，按方案 §11 失败条件 A 转向 coordinate-aware arbitrary kernel。

## 2. 局部性探针

```bash
python scripts/evaluate_nef_locality.py \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --split test --parts left_arm right_arm left_leg right_leg \
  --max-clips 64 --output outputs/nef_locality/v1
```

把 donor 窗口的 region token 换进 target 窗口（半开区间 `[--edit-start, --edit-stop)`），解码后报告：

- `support_coordinates` / `support_fraction`：这次编辑动了多少 coordinate（NEF `left_arm` strict 是 4/40）；
- `support_token_change_fraction`：support 内真正发生变化的 token 比例。**读数的前提**：这个值为 0 时，
  后面的 0 只说明 donor 与 target 在该区域本来就相同，不说明模型局部性好；
- `off_target_feature_max/mean`：区域外的解码特征变化（NEF 上应为精确 0，作为对照的 flat 为全局编辑）；
- `target_joint_change` / `descendant_joint_change` / `non_target_joint_change_*`：世界坐标 FK 变化，
  按「目标关节 / 运动链后代 / 其余关节」分开，避免把合法的链式传播误报成泄漏；
- `contact_flip_rate`、`root_position_change`、`boundary_velocity_step_*`、
  `max_velocity_step_in_influence`：接触与边界代价；
- `pre_edit_unchanged` / `post_influence_unchanged`：因果性检查（编辑前与影响窗口之后必须逐位不变）。

`whole_body` 区域是全身对照（support = 全部 40 个 coordinate），此时所有关节都是目标关节。

## 3. 共同的约定

- 版本无关读取：`read_probe_window()` 在 v3 / v4 store 上都按「窗口不跨 clip、不足的历史用 clip 内首帧
  左填充」读取，填充帧是复制而非编造动作。
- 特征空间：窗口先由 store 统计量反归一化，再用 checkpoint 的 `feature_stats` 归一化，
  因此 store 与 checkpoint 的统计版本不同也不会污染结论。
- 编辑只做合法的 `±1` / far level 跳变；越界的帧被 `valid` 掩码排除而不是截断成「无变化」样本。
- 探针不修改模型、不写 checkpoint，只读 token 与解码结果。

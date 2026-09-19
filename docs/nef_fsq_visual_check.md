# NEF-FSQ 可视化检查（round trip）

工具：`scripts/visualize_nef_fsq.py`（只读真实 tokenizer + store，不训练、不写 checkpoint）。
它把子任务串成一条可复现的检查：真实窗口 → 编码成 40×9 token → `decode(encode(x))` → 回到特征空间 →
用生产渲染路径出图 + 统计误差。

> 本文档在 2026-09-18 修正过：第一版渲染用了错误的角色资源；随后发现 store 的 `ref_pos` 是镜像平均
> 而非骨架；再随后发现**镜像 clip 需要镜像的参考骨架**。三处都在“修正记录”里，数字已按修正后的实现重算。
> 追加：又渲染了 9 段不同动作的序列（“更多序列”一节 + `contact_sheet.png`），并因此发现原
> `root/local` 误差分解分不开“整体朝向错”与“位姿错”（“勘误 4”）。

## 运行方式

```bash
# 单个 clip：角色渲染 + token 图 + 误差图 + 对比 GIF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/visualize_nef_fsq.py \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
  --feature-database data/processed/seed_soma_pruned_v4_ah \
  --split test --action "Basic Locomotion Neutral" --frames 64 \
  --white-background --video-frames 24 --camera-distance 3.2 \
  --output outputs/nef_fsq_viz/locomotion

# 分层抽样：每个 action 取若干 test clip，输出 sweep.csv / sweep.png / sweep.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/visualize_nef_fsq.py \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
  --feature-database data/processed/seed_soma_pruned_v4_ah \
  --frames 64 --sweep 40 --device cpu --output outputs/nef_fsq_viz/sweep40
```

渲染默认走**用户既有**的 SOMA/Quinn 渲染路径（`stylized_motion.anim.render_stills`）：

* `--resources-root data/assets/somaview_quinn`（Quinn 蒙皮到 SOMA 的网格），
* `--base-color-map / --normal-map` 默认取该目录下的 Quinn 贴图，
* 相机固定在世界点（`--camera-target`），否则根节点跟随会让差值图全是地面视差。

产物（`outputs/` 已被 gitignore）：

| 目录 | 内容 |
|---|---|
| `outputs/nef_fsq_viz/contact_sheet.png` | 9 段序列总览（source / round trip / difference×4 + 每次运行的读数标题） |
| `outputs/nef_fsq_viz/error_split.json` | 9 段的 pose / placement 误差分解（`scripts/nef_fsq_error_split.py`） |
| `outputs/nef_fsq_viz/locomotion/` | 缓速 clip（idle）：`compare_frames.png`（source / round trip / difference×4）、`compare.gif`、`token_map.png`、`coordinate_error.png`、`joint_error.png`、`summary.json`、`source_raw.npy`、`recon_raw.npy`、`database_*.npz` |
| `outputs/nef_fsq_viz/jumping_jack/` | 大幅动作 clip（Sports 128115）：同上，能看到明显位姿差异 |
| `outputs/nef_fsq_viz/{walk_forward,dance_hiphop,gesture_no_speak,household_blender,injured_leg_kick,sit_legs_crossed,stunts_postmortem}/` | 追加的 7 段（见“更多序列”），同样的产物 |
| `outputs/nef_fsq_viz/sweep40/` | 36 个 test clip（18 个 action × 2）：`sweep.csv`、`sweep.png`、`sweep.json` |

## 读数（修正后；holdout tokenizer `nef_fsq_soma_packed_40x9_ah/best.pt`，test split）

都在 **held-out actor 的 test clip** 上测，不挑样本；窗口 64 帧，编码时模型自带 63 帧历史。

| 指标 | 缓速 clip（idle，未镜像） | 大幅 clip（jumping jack，同 take 的未镜像变体） | 36-clip 抽样（其中 15 个镜像） |
|---|---|---|---|
| token 定点率（把 decode 结果重新编码，与原 token 相同的比例） | 0.995 | 0.856 | 均值 0.957 |
| 关节误差（米，bind 骨架 FK 后世界坐标） | 均值 0.054 / p95 0.155 / max 0.185 | 均值 0.098 / p95 0.231 / max 0.382 | 均值 0.094 / 中位数 0.049 / p90 0.229 / 最差样本 max 0.798 |
| 其中 root 轨迹误差 | 0.009 | 0.006 | 最差两例 0.003–0.009 |
| 其中 body-local（位姿）误差 | 0.046 | 0.096 | 最差两例 0.449 |

镜像与非镜像分开看（36 clip：15 镜像 / 21 非镜像）：镜像组均值 0.129 m、非镜像组均值 0.068 m。
样本很少，不能据此说“镜像 clip 更难”；但它提醒镜像与非镜像要分开报告。

要点：

1. **误差是“位姿”，不是“轨迹漂移”。** 误差最大的两个 clip（`come_up_50cm_box_R_004` 0.449 m、
   `sitting_legs_crossed_arm_side_stop_R_002` 0.448 m）root 误差只有 3–22 mm，几乎全部误差在
   body-local 关节上，即 40×9 的码本在这些极端姿态上表达不出来。
   **（2026-09-18 更正：这条分解只去掉 root 关节的平移，不能把“整段身体朝向错了”从“位姿错了”里
   分出来；对上述第二个 clip 用 Kabsch 分解实测，62% 的误差其实是整体摆放/朝向——见“勘误 4”。）**
2. **token 定点率高 ≠ 动作对。** 上述两个 clip 的定点率仍有 0.94–0.97：重新编码回同一批 token，
   但解出来的姿态本身已经偏了。定点率只说明 token 映射自洽，不能当重建质量用。
3. **误差集中在肢体末端。** 最差关节稳定是 `RightToeBase`/`LeftToeBase`/feet/hands，
   符合“链式 FK 把上游小角度误差放大到末端”的预期。
4. **可见的差异**（jumping_jack，同 take 的未镜像变体，frame 42/63）：source 双臂对称上举；round trip
   左臂抬得明显偏低、下肢开度也不同，与 0.382 m 的末端误差（最差关节 LeftFoot/LeftToeBase/LeftHand）一致
   （frame 0/21 双臂下放时几乎无差别）。
5. **per-action 平均不可当结论。** sweep 每个 action 只有 2 个 clip，同 action 内两例可差 10 倍
   （`Basic Locomotion Neutral`：0.046 与 0.451 m），`sweep.png` 右侧散点比左侧柱状更可信。

## 更多序列（9 段总览，2026-09-18 追加）

为覆盖不同动作类型，又渲染了 9 段 test clip（同一 checkpoint / store，窗口 64 帧，默认
`--camera-distance 4.0`、每段 4 张 still + 32 帧 GIF）。总览图由
`scripts/nef_fsq_contact_sheet.py` 生成——每行 `source | NEF-FSQ round trip | difference×4`，
行标题直接抄该次运行 `summary.json` 的读数（每行按自身角色裁剪缩放到同尺寸，所以**行间大小不是尺度对比**）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/nef_fsq_contact_sheet.py \
  --root outputs/nef_fsq_viz \
  --runs locomotion walk_forward dance_hiphop gesture_no_speak household_blender \
         injured_leg_kick sit_legs_crossed stunts_postmortem jumping_jack \
  --output outputs/nef_fsq_viz/contact_sheet.png
```

产物：`outputs/nef_fsq_viz/contact_sheet.png`（总览）、`outputs/nef_fsq_viz/error_split.json`
（下表的分解读数），以及每段自己的目录（`compare_frames.png` / `compare.gif` / `token_map.png` /
`coordinate_error.png` / `joint_error.png` / `summary.json` / `source_raw.npy` / `recon_raw.npy`）。

| 目录 | clip | action / style | 窗口起点 | token 定点率 | 关节误差 mean / p95 / max (m) |
|---|---|---|---|---|---|
| `locomotion` | 128116 `idle_right_R_001__A420` | Basic Locomotion Neutral | 50988 | 0.995 | 0.054 / 0.155 / 0.185 |
| `walk_forward` | 128163 `walk_ff_start_000_L_002__A033` | Basic Locomotion Neutral | 72572 | 0.953 | 0.033 / 0.075 / 0.145 |
| `dance_hiphop` | 128170 `dance_hiphop_guess_R_fast_002__A317` | Dancing | 75185 | 0.811 | 0.090 / 0.220 / 0.383 |
| `gesture_no_speak` | 128155 `no_speak_002__A187` | Gestures | 69081 | 0.998 | 0.021 / 0.061 / 0.095 |
| `household_blender` | 128213 `operating_blender_R_001__A298` | Household | 93486 | 0.994 | 0.014 / 0.032 / 0.068 |
| `injured_leg_kick` | 128157 `injured_trail_leg_kick_trash_001__A057` | Other / injured leg | 70108 | 0.968 | 0.080 / 0.184 / 0.320 |
| `sit_legs_crossed` | 137113（请求 128118 `_M`）`sitting_legs_crossed_arm_side_stop_R_002__A033` | Basic Locomotion Neutral | 173314 | 0.979 | 0.138 / 0.259 / 0.321 |
| `stunts_postmortem` | 132177（请求 128161 `_M`）`postmortem_convulsions_side_loop_R_001__A470` | Stunts | 23333 | 0.904 | 0.388 / 0.707 / 0.823 |
| `jumping_jack` | 140158（请求 128115 `_M`）`jumping_jack_002__A033` | Sports | 249299 | 0.856 | 0.098 / 0.231 / 0.382 |

窗口注意：`locomotion`/`jumping_jack` 用的是工具默认的“片段第一个完整窗口”；其余 5 段长片段的开头
是静止起势（站着不动），所以显式给了 `--start` 取片段中段，否则四张 still 几乎一样。两个镜像请求
（`sit_legs_crossed`、`stunts_postmortem`）按“勘误 3”自动换成同 take 的未镜像变体渲染，`summary.json`
里记了 `requested_clip`/`unmirrored_variant_of`。

读数：

1. **日常动作在厘米级。** `household_blender` 0.014 m、`gesture_no_speak` 0.021 m、
   `walk_forward` 0.033 m，与 sweep 的中位数 0.049 m 同量级；渲染图上 source 与 round trip 的
   手指/手肘差异是仅有的可见区别。
2. **表达性动作中等。** `dance_hiphop` 0.090 m、`injured_leg_kick` 0.080 m：四肢摆动明显更大，
   末端（手/脚）差得最多，躯干基本跟得住。
3. **躺地抽搐（这批最差）不是“码本表达不了姿态”，而是整体朝向错。** `stunts_postmortem`
   0.388 m（max 0.823 m）里，Kabsch 对齐后只剩 0.077 m 的位姿误差，另外 0.311 m（80%）是整体
   摆放/朝向：躯干主轴从 **94° 被重建到 122°**（source 是躺平，round trip 把身体又转了 27°），
   骨盆从 0.19 m 抬到 0.38 m，而 Hips→Head 链长两边的 0.593/0.594 m 一致。即失败在
   **root 朝向流**，不是 40×9 的关节码本。
4. **坐姿盘腿同理但更轻**：躯干 20.5° → 5.5°（重建把身体“坐正了”），骨盆 0.141 → 0.105 m，
   placement 占 62%；渲染图上 round trip 的躯干明显更直。
5. **缓速 idle 会被重建得略高**：骨盆 0.974 → 0.996 m（+22 mm）、tilt 3.1° → 4.4°，
   肉眼几乎看不出，但这是 sweep 里非镜像组 floor-level 误差的一部分来源。
6. **蒙皮/骨架这 9 段全部正常**（无扭曲、无塌陷、无缺头缺手）；镜像变体都走“同 take 未镜像”渲染，
   图中标题带 `[shown unmirrored: a -> b]`。

### 勘误 4：`root/local` 分解分不开“整体朝向错”与“位姿错”

`joint_error_stats` 里的 local = 逐关节误差减去 root 关节自身的位移，**只去掉平移**。整段身体被
重建到错误朝向时，这部分误差会落进 “body-local”，读起来就像“码本表达不了这个姿态”。上面第 3、4 条
正是这种情况（那个 clip 的 root 关节自身误差只有 10 mm）。

`scripts/nef_fsq_error_split.py` 改用逐帧 **Kabsch 最优刚体对齐**重新分解，读已存的
`source_raw.npy`/`recon_raw.npy`，1 秒/段，不需要重渲染：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/nef_fsq_error_split.py \
  --checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
  --root outputs/nef_fsq_viz --output outputs/nef_fsq_viz/error_split.json
```

* `pose_aligned_mean_m`：对齐掉平移+旋转后的残差，即码本真正要表达的关节角度/体形；
* `placement_mean_m`：其余部分，即身体在世界里的位置与朝向。

| 目录 | raw | pose | placement（占比） | 骨盆高 src→rec (m) | 躯干主轴与竖直夹角 src→rec |
|---|---|---|---|---|---|
| `household_blender` | 0.014 | 0.012 | 0.002 (15%) | 0.996 → 0.996 | 4.7° → 5.4° |
| `gesture_no_speak` | 0.021 | 0.020 | 0.001 (5%) | 0.996 → 0.997 | 4.8° → 4.6° |
| `walk_forward` | 0.033 | 0.015 | 0.018 (55%) | 0.983 → 0.983 | 11.3° → 11.7° |
| `locomotion` | 0.054 | 0.042 | 0.013 (23%) | 0.974 → 0.996 | 3.1° → 4.4° |
| `injured_leg_kick` | 0.080 | 0.036 | 0.044 (55%) | 0.981 → 0.980 | 23.7° → 19.9° |
| `dance_hiphop` | 0.090 | 0.037 | 0.053 (58%) | 0.945 → 0.934 | 25.5° → 25.5° |
| `jumping_jack` | 0.098 | 0.040 | 0.057 (59%) | 0.992 → 0.971 | 5.9° → 7.1° |
| `sit_legs_crossed` | 0.138 | 0.052 | 0.086 (62%) | 0.141 → 0.105 | 20.5° → 5.5° |
| `stunts_postmortem` | 0.388 | 0.077 | 0.311 (80%) | 0.192 → 0.383 | 94.4° → 121.9° |

`walk_forward` 的 placement 是水平残留（骨盆高度两边一致），与“走远”的 root 漂移一致；
`locomotion`/`stunts_postmortem` 则是**竖直方向**的整体抬高，说明是朝向而不是平移。
`pose` 一列在 9 段里都 ≤ 0.077 m，因此本批没有“码本表达不了”的极端姿态；误差大的片段都是
root 朝向流的问题——这也是唯一值得继续盯的失败模式。

## 修正记录（三处真实缺陷）

### 1. 渲染资源：用了占位角色而不是 Quinn

第一版用 `render_stills` 的默认 `--resources-root data/assets/somaview`（旧的 SOMA 占位网格），
且没传贴图，于是出现“橙色、无贴图、无手指”的角色。用户既有渲染命令（`docs/quinn_to_soma_conversion_plan.md`
“视觉验证”一节）用的是：

```bash
python -m stylized_motion.anim.render_stills --pipeline somaview \
  --resources-root data/assets/somaview_quinn \
  --base-color-map data/assets/somaview_quinn/quinn_base_color.jpg \
  --normal-map data/assets/somaview_quinn/quinn_normal.jpg ...
```

`scripts/visualize_nef_fsq.py` 现在默认就用这套资源，可用 `--resources-root/--base-color-map/--normal-map` 覆盖。

### 2. 骨架：store 的 `ref_pos` 是镜像平均，不是骨架

`MotionFeatureStats.ref_pos` 是**整个数据集 local positions 的均值**。packed store 由原始 clip 与
**镜像** clip 共同构成，镜像会翻转横向轴，于是任何“恒定且沿轴”的偏移在均值里相互抵消：
SEED SOMA store 的 `Spine1/Spine2/Chest/Neck1/Neck2/Head` 的 `ref_pos` 因此几乎为 0（真值 0.05–0.26 m）。
用这个均值当骨架做 FK，脊柱被压扁、末端关节贴在父关节上 —— 这就是渲染里“脖子/头陷进胸腔、
手只剩残端”的原因，也是所有基于 `KinematicContext.ref_pos` 的**世界坐标**指标的系统性错误来源。

正确来源是 **bind 骨架**（`data/assets/somaview/SOMA_bind.bvh`，厘米；特征空间是米）：
它与 store 每个 clip 的常量 local positions 在 <1e-6 m 内一致（逐关节核对过，`Hips` 除外——髋部会动）。
新增两个 helper：

* `stylized_motion.anim.features.bind_reference_positions(names)`：按名字取 bind 偏移（米），
  缺失关节（如流水线生成的 `Simulation`）跳过，返回 `None` 表示不可用；
* `stats_with_reference_skeleton(stats, names)`：替换 `ref_pos`，保留会动的 `Simulation/Hips`。

已接入的调用点：本可视化脚本（渲染 + 米制误差）、`KinematicContext.from_feature_stats(..., reference_positions=...)`
以及 `scripts/evaluate_nef_locality.py`、`scripts/evaluate_mts_operator.py`、`scripts/probe_nef_geometry.py`。

### 3. 镜像 clip（`_M`）需要**镜像的**参考骨架

store 里约一半 clip 是镜像变体（`clip_mirror=True`，名字带 `_M`）。特征向量只带旋转、不带链偏移，
而镜像 clip 的链偏移是“**左右名字互换 + 横向（x）取反**”（与 `_process_motion_data` 的 mirror 分支一致）：
脊柱链 `Spine1/Chest/Head` 的 x 变号，四肢链因为名字已互换而保持符号。
消费者若用**未镜像**的参考骨架去摆镜像窗口，躯干会折下来（头落到髋部以下）——这正是 jumping jack 那张图
里“没有头/蒙皮错位”的原因。

判定依据（都可复现）：
* `_process_motion_data(bvh, mirror=True)` 的 FK 真值：头 y 1.571、脚 y 0.014；
  用 plain 参考重建 → 头 y 0.38（倒过来），用镜像参考 → 头 y 1.57、脚 y 0.01，且两者对真值的误差相同（0.230 m）；
* store 的镜像 clip 128115 现在重建为头 y 1.58、脚 y 0.01（先前是头 y 0.44、脚 0.24）。

**渲染与指标分开处理。** 指标在关节/矩阵域计算，用镜像参考骨架即可（差值不受网格影响）；
但**渲染**镜像 clip 需要镜像的网格：SOMA 的链式骨骼沿镜像轴（局部 x）排列，把镜像骨架摆到未镜像的
网格上会让这些骨骼绕自身轴滚 180°，皮肤被拧住（见本节修复前的 `source_frame_042.png`：躯干/颈部被拧细）。
`scripts/visualize_nef_fsq.py` 因此改为渲染**同一 take 的未镜像变体**
（`--clip` 指定任一变体，工具在 store 里找 `source_group` 相同、`clip_mirror=False`、长度一致的那一行，
summary 记录 `requested_clip` 与 `unmirrored_variant_of`），保持画面与数字一致；
镜像 clip 的镜像感知指标仍由 sweep（每行 `mirror`）与 locality（`mirror_flags`）覆盖。

修复：`bind_reference_positions(names, mirror=True)`（左右伙伴 + x 取反，**不是**整体 x 翻转）、
`stats_with_reference_skeleton(..., mirror=...)`、`reference_positions_for_fk(..., mirror=...)`；
可视化按 clip 选择、locality 探针按窗口选择（artifact 里记录 `mirror_flags`）、
MTS evaluator 的 physics 按 batch 内的镜像分组分别做 FK（每行记录 `mirrored`）。

**对既有结论的影响**：这些指标多是“同一骨架下的差值”。
R13 的 locality 记录用修正后的骨架重跑过（`outputs/mts_revision2_closure/R13/locality_bindfix/`）：
`descendant_joint_change`、`non_target_joint_change_*`、`root_*`、`contact_flip_rate` 全部不变（仍为 0），
`target_joint_change` 从 0.18897 → 0.18893（相对变化 2e-4），`edit_feature_mean`（特征空间）完全不变。
结论不变，但绝对米制数字以修正后的为准。

## 局限

- 只看了 tokenizer 的**自重建**（`decode(encode(x))`），与算子/风格无关；不构成任何风格效果结论。
- 渲染对比图的 `difference ×4` 行含抗锯齿边缘与影子，会显得比实际误差大；判断位姿请看 `joint_error.png`
  和并排的上两行。
- 相机固定在 clip 中段的世界点（见上），所以画面里角色的水平位移不完整；`summary.json` 里的
  `root_*` 才是轨迹读数。
- sweep 每个 action 只有 2 个 clip，`sweep.png` 左侧柱状只是粗略分组；不要据此做 per-action 结论。
- 镜像与非镜像样本要分开报告（`sweep.csv` 里带 `mirror` 列）；两组样本量都小，差异不宜当结论。

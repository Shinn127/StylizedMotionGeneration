# MTS 生成结果可视化（generator → 角色）

回答“当前各个 generator 的结果能不能可视化”：**能，两种粒度都有**。token/火柴棍层面的
flipbook 在 N04 筛选轮就已经有了；本轮补上了**角色层面**的渲染 —— 把生成器解码出的特征窗
（source / base / styled）用与 NEF-FSQ 检查完全相同的 somaview + Quinn + bind 骨架路径渲染成
对比图。

> 前提与定性：当前存在的是**预筛选轮（screening）的 1000 步算子**（`operator_styleid_logit_canonical_s3407`
> 与对照 `operator_noref_logit_canonical_s3407`），上游 transport 同样只有 screening 预算。
> 渲染只是把生成的东西看一眼，**不构成任何风格效果声明**（研究效果仍未验证）。

## 1. 已有的可视化（token / 火柴棍）

`scripts/benchmark_mts_editing.py`（N04 生成探针）对每个 (case, arm, region) 输出一张
`flipbooks/caseNNN_<arm>_<region>.png`：三行（source / base / styled）× 8 帧，固定投影的火柴棍图。
N04 探针全量跑过一次：12 case × 2 臂 × 2 region × strengths × steps × 2 draws = 576 行指标 + 48 张
flipbook，见 `outputs/mts_next_round_20260918/N04/probe/`（`benchmark.json`、`benchmark_rows.csv`、
`flipbooks/`）。这是 token 编辑与物理指标的可视化，但**不是角色**。

## 2. 新增：角色渲染（本轮）

改动（两个文件，均不影响训练入口）：

* `scripts/benchmark_mts_editing.py --save-motions`：把 figure 行（strength 1.0 / draw 0 / steps 1）
  已经解码好的 `source / base / styled` 特征窗存成
  `motions/caseNNN_<arm>_<region>.npz`（附 tokens 与 case 元数据）；
* `scripts/render_mts_generation.py`（新）：读生成目录（`generate_mts_operator.py` 的
  `motion.npy` 三件套）或上面的 npz，用 tokenizer checkpoint 自带的 `feature_stats` 反归一化，
  以 bind 骨架建 viewer database，逐帧走生产渲染 `stylized_motion.anim.render_stills`，
  输出 source / base / styled 三行 + 两行 difference×4 的
  `compare_generation.png`，并把 `source→base`、`base→styled` 的关节误差写进 `summary.json`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_mts_editing.py \
  --manifest outputs/mts_next_round_20260918/N02/benchmark_manifest.json \
  --arm style_id_s3407=outputs/mts_revision2/operator_styleid_logit_canonical_s3407/best.pt \
  --arm constant_s3407=outputs/mts_revision2/operator_noref_logit_canonical_s3407/best.pt \
  --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
  --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
  --steps 1 --strengths 1.0 --draws 1 --regions left_arm --graph-radius 1 \
  --max-cases 3 --eval-seed 20260918 --save-motions \
  --output outputs/mts_generation_demo/benchmark

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/render_mts_generation.py \
  --record-npz outputs/mts_generation_demo/benchmark/motions/case000_style_id_s3407_left_arm.npz \
  --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
  --label-styled "styled (style_id arm)" \
  --output outputs/mts_generation_demo/renders/case000_style_id_s3407_left_arm
```

本轮工件：`outputs/mts_generation_demo/benchmark/`（3 case × 2 臂的生成 + motions npz + flipbook）
与 `outputs/mts_generation_demo/renders/<case>_<arm>_left_arm/`（4 套角色对比图 + summary.json）。
case000/001 是 injured leg、case002 是 injured torso（same_style_cross_content：query 是
Basic Locomotion Styles 的伤腿/伤躯 clip，reference 是另一内容、同 style 的 clip——与 N04 筛选
完全同口径）。

## 3. 渲染读数（6 套 = 3 case × 2 臂，全部逐张目检过；case000/002 injured leg·injured torso，case001 injured leg）

| 记录 | source→base（transport argmax 往返） | base→styled（算子编辑，left_arm） | by-side base→styled（left / right / mid） |
|---|---|---|---|
| case000 style_id | 0.063 m（p95 0.694 / **max 1.233**） | 0.0101 m（p95 0.062 / max 0.525） | 0.0303 / 0 / 0 |
| case000 constant | 同上（base 与 arm 无关） | 0.0076 m（p95 0.032 / max 0.468） | 0.0229 / 0 / 0 |
| case001 style_id | 0.078 m（max 1.26） | 0.0084 m（max 0.42） | 0.0251 / 0 / 0 |
| case001 constant | 同上 | 0.0082 m（max 0.78） | 0.0245 / 0 / 0 |
| case002 style_id | 0.077 m（p95 0.740 / **max 1.224**） | 0.0086 m（p95 0.063 / max 0.538） | 0.0258 / 0 / 0 |
| case002 constant | 同上 | 0.0066 m（p95 0.052 / max 0.490） | 0.0199 / 0 / 0 |

目检（6 张 montage 逐张看过，2026-09-19）：蒙皮/骨架全部正常（无塌陷、无扭曲、无缺头缺手）；
montage 底部的 movers 标注在 6 套里全部只列出 `Left*` 关节；base 行（transport 失真）与
styled 行（编辑后）的差别集中在左臂，腿/躯干逐帧不动——locked edit 在角色层面成立；
by-side 统计 right/mid 严格为 0（区域外逐帧复制）。

看图与读数一致的三个结论（均为筛选轮观察，非效果声明）：

1. **当前的对比被上游 transport 的失真淹没。** screening transport（1000 步）的 argmax 重建
   本身就把这个 clip 的手臂摆到了完全不同的位置（p95 ≈ 0.7 m、max ≈ 1.2 m）。要读“风格化了什么”，
   现在只能看 base→styled 这一差分，不能拿 styled 直接和 source 比。
2. **锁定编辑在动且只在该动的地方动。** base→styled 均值只有 7–10 mm，但 max ≈ 0.5 m
   （左臂摆动的那几帧）；`root_mean = 0`，渲染里腿和躯干逐帧不动，改动集中在手臂——
   locked edit 的语义在角色层面可验证。
3. **style_id 臂比 constant 对照改动略大**（0.0101 vs 0.0076、0.0086 vs 0.0066），方向与 N04
   筛选读数一致（tv 0.028、changed-token 0.074），但量级很小——1000 步的筛选模型本就只够回答
   “管道是否通”，不够回答“风格效果如何”。

### 左右判定（“编辑的是不是右手？”——不是）

固定相机 + 角色朝向让角色的**左手**落在画面右侧，看起来像“编辑到了右手”。三层证据（case000/002，
`left_arm` 区域，均可复现）：

1. **区域映射**：`adapter.hard_mask(['left_arm'])` 只触及 6 个坐标，对应
   `left_arm_node` / `left_shoulder_edge` 流——按名字定义在左臂侧；
2. **逐关节 FK**：styled−base 的改动全部在 `LeftHand`（0.15–0.18 m）、`LeftForeArm`（0.07 m）、
   `LeftArm`（0.01–0.02 m），`Right*` 与躯干/根为 0（`summary.json` 的
   `base_to_styled_by_side`：left ≫ right/mid）；transport 自身的失真（source−base，max 1.2 m）
   也同样落在左臂（`LeftHand` 0.90/1.13 m）——两件事在同一只手臂上，不存在错位；
3. **骨架覆盖渲染**（`--skeleton`）：base 与 styled 帧里移动的关节簇就在画面右侧那条手臂的
   网格里——网格与骨架左右一致，动的就是解剖学左臂。

即：画面右侧的手臂 = 角色的左手（相机从角色前方拍摄，像面对面看人）。为免再误读，
`render_mts_generation.py` 现在把“base→styled movers”的关节名和数值自动标注在 montage 底部，
并在 `summary.json` 记录 `base_to_styled_by_side`（left/right/mid 三组的逐关节均值 + 前 4 名 mover）。

## 4. 发现的兼容性问题（未修，待立项）

第一轮的生成脚本 `scripts/generate_mts_operator.py` 与 revision-2 checkpoint **不兼容**，本轮实跑确认：

1. `steps=1` 时 `model.generate_edit(..., return_trace=True)` 走单步分支，**忽略 `return_trace`**
   只返回一个值 → 脚本解包 `(drawn, commits)` 崩溃（`ValueError: not enough values to unpack`）；
   多步分支正常。修法是让单步分支与多步分支的返回契约一致（一行 + 测试）。
2. style encoder kind 为 `constant` 的 checkpoint 被当成 reference 模型，脚本强制要求
   `--style-clip` → constant 臂无法用该脚本生成。修法是在脚本里显式处理 `constant`。
3. 该脚本不应用 content schema v1：canonical 臂的词表里是 "Basic Locomotion"，而 store 的原始
   label 是 "Basic Locomotion Neutral/Styles" → 这类 clip 会被拒（错误信息正确地拒绝借用 id）。
   这正是筛选轮改用 `benchmark_mts_editing.py`（按 checkpoint 记录的 schema 取
   `content_canonical`）的原因。

在此之前，**驱动当前 checkpoint 的生成入口只有 `benchmark_mts_editing.py`**；角色渲染走它的
`--save-motions` 输出即可，不受影响。

## 5. 不能角色渲染的生成工件

`outputs/training_readiness_audit_20260918/T04/chain/generate/`（tiny chain 的 `motion.npy` 等）
**不走这条渲染路径**：它是 tiny token env（230 维特征、独立的 fixture 骨架）+ 2 步 dry-run 模型的
输出，骨架与 SOMA/Quinn bind 骨架不同，而且 gate 里已注明它不构成研究证据。

## 修正记录：首版角色渲染漏了 bind 骨架替换（2026-09-19）

用户指出渲染图的蒙皮扭曲。原因与 tokenizer 检查的勘误 2 同源：`render_mts_generation.py` 首版
直接用 tokenizer checkpoint 里的 stats 做 FK，**没有调 `stats_with_reference_skeleton`**——而这套
stats 的 `ref_pos` 是镜像平均的均值，脊柱链偏移≈0。后果可量化（case000 source 窗）：
`Hips→Head` 只有 **0.005 m**（头缩进骨盆，整条脊柱 telescoped，头 y = 颈 y = 骨盆 y ≈ 1.0 m），
四张 montage 全部是“缩成一团”的角色。

修复：与 tokenizer 检查一致，渲染/FK/相机目标/误差统计全部改用 bind 骨架
（`stats_with_reference_skeleton(..., mirror=...)`；`--record-npz` 从记录 meta 读镜像标志，
`--generation` 用 `--mirror` 显式给），summary 记录 `reference_skeleton`/`clip_mirror`。
四套 montage 已用修正后的骨架重渲。

**对读数的影响：几乎为零。** 误差是“同一骨架下两窗之差”，骨架错误是共模的，在逐关节差分里
相消——修正前后数字在千分位一致（如 case000 style_id base→styled 0.0101 m / max 0.526 → 0.525）。
变的是**画面**（绝对骨架位置），不是相对读数；上表数字以修正后重渲的 summary 为准。


# BONES-SEED 数据管线落地记录

日期：2026-09-15。本文记录 [bones_seed_data_pipeline_plan.md](bones_seed_data_pipeline_plan.md) 的代码落地范围、已验证内容与尚未执行的部分。方案本身未改动，仍是验收依据。

流程与产物的速查图见 [seed_pipeline_overview.md](seed_pipeline_overview.md)。

## 1. 新增模块

| 模块 | 对应方案章节 | 职责 |
|---|---|---|
| `stylized_motion/data/seed_catalog.py` | §4.1 | CSV 权威目录、帧契约、镜像策略、take group split、时间标签换算 |
| `stylized_motion/data/seed_build.py` | §3.1/§4.3 | inventory、worker unit、resume、打包 v4 store、发布前校验 |
| `stylized_motion/data/resume.py` | §4.3 | work unit / 指纹 / journal / 有界调度 / 原子写 |
| `stylized_motion/data/normalization.py` | §4.2/§4.3.6 | float64 可合并 Welford 统计、独立版本化 artifact |
| `stylized_motion/data/packed_store.py` | §4.2 | schema v4 packed store 读写、CPU 批次归一化、v3 兼容入口 |
| `stylized_motion/data/packed_token.py` | §3.4/§4.6 | 按逻辑 clip 编码的 packed token store |
| `stylized_motion/data/packed_trajectory.py` | §4.6 | clip-local 轨迹通道、validity mask、train-only 统计 |
| `stylized_motion/data/benchmark.py` | §5 | sampler / loader / resident / end-to-end 分层 benchmark |

重写而非新增的部分：`stylized_motion/data/sampling.py`（紧凑索引与采样策略）、`seed_build` 相关的 `preprocess.py` CLI、`loader.py` 的 store/sampling 接线、`runner.py` 的训练预算与采样覆盖率。

## 2. 关键契约

- **帧契约**：`FrameContract` 明确 `raw_frames`（120 fps 行数）与 `target_frames`（60 fps，`ceil(raw/step)`）。元数据 `move_duration_frames` 不能当作 60 fps `stop`；`SeedClip` 会在构造时拒绝这种不一致。镜像少一帧属于合法的 variant 长度差异。
- **镜像策略**：`official`（保留全部官方文件）/ `generate`（只留原始、预处理生成镜像）/ `none`，三者互斥。SEED 不再走 v3 那条“每个文件无条件生成 original+mirror”的路径，`--dataset seed` 会直接报错并指向新入口。
- **group split**：group key 由 `take_date / take_actor / take_org_name / canonical move name` 组成，同一 take 的原始、官方镜像、派生片段必然同 split；支持 actor holdout 冻结测试集；不再使用“最后十个 style 自动 heldout”。
- **物理/逻辑解耦**：约 256 MiB（可配置 128/256/512）未归一化 float32 分片，clip 表记录 shard/offset/length/source_group/variant/split。采样窗口只在逻辑 clip 内取，跨片段一律 `IndexError`。
- **归一化**：特征只存一份未归一化版本；`normalization.npz/.json` 独立版本化，`train-stats` 可在不改动特征字节的前提下重算。统计数据只来自 train split。
- **hash 分离**：`skeleton_hash`、`feature_schema_hash`、`split_manifest_hash`、`normalization_hash` 分开保存；token/trajectory store 分别绑定它们，checkpoint 通过 feature schema 绑定同一组值。
- **可恢复**：unit id = hash(signature, clip, variant, 输入指纹)，成功原子提交到 journal 并记录输出 checksum；签名、指纹或 checksum 任一变化都会重建该 unit。失败的 unit 记入报告并可重试。
- **训练预算**：`steps_per_epoch` / `max_steps` / `eval_every_steps` / `full_eval_every_epochs`；DDP 每 rank 步数相等（`tail=drop|pad`），采样由 (seed, epoch, rank, ordinal) 决定，支持 epoch 中途恢复。

## 3. 验证情况

已在本机用真实 BONES-SEED 文件端到端跑通（小规模子集）：

- catalog discovery、inventory（78 节点、`Frame Time: 0.008333`、`Frames:` 与 CSV 一致）、group split。
- v4 store 构建：16 个 clip → 3 个分片；原始 96/69 帧正确降采样为 48/35 帧；官方镜像 variant 标记正确；`train/val/test` 划分与 group 一致。
- 重复构建命中 resume：第二次运行 `reused=16`，不重新解析任何 BVH。
- 采样：64 帧窗口不跨 clip、不跨 split；coverage 报告 group/interval 覆盖率与重复率。
- token：用真实 NEF-FSQ SOMA encoder 编码，clip 内分块（chunk=97）与整段编码逐字节一致；token 表与特征表逐行对齐；checkpoint/feature/normalization/split 四个 hash 全部绑定并校验。
- trajectory：`root_relative_future` 与直接重算一致；validity mask 恰为 `[0, length-horizon)`；统计只取 train。
- 训练接线：用真实 v4 store + `nef_fsq_soma_packed_40x9.yaml` 跑通 `train` 两轮（`steps_per_epoch=2, max_steps=3` 生效，产出 `last.pt`/`best.pt`），日志含 `data_wait_fraction` 与逐 epoch 采样覆盖率。
- 训练接线（复审修正后重验）：`--resume` 从 checkpoint 的 epoch 3 / step 8 精确续跑，日志显示「resuming at sample 8」；停在 `max_steps` 时 `sampler_state` 记录的是**实际训过的样本数**（8）而不是预取后的采样器计数；`best.pt` 由全量扫描决定（日志同时给出 `val_loss` 与 `val_full_loss`）；`normalize_on: none` 走通 CLI 并在 checkpoint 中记录；真实 store 上 `clip_uniform` 与 `group_balanced` 在 `mirror_probability=1.0` 时镜像占比均为 1.000。
- 自动化测试：`tests/test_seed_pipeline_v4.py`（56 项）与 `tests/test_packed_downstream.py`（28 项）全绿；全套 279 通过 / 1 跳过，唯一失败项为既有的 100STYLE 数据布局问题。

三处迁移过程中暴露并修掉的真实缺陷：

1. **短片段崩溃**：`_compute_simulation_root` 使用固定 31/61 帧 Savitzky-Golay 窗口，SEED 存在短于 61 帧（60 fps）的 take，会让整个构建立即失败。现按片段长度收缩窗口，长片段结果与历史实现逐位一致。
2. **CLI seed root 推断错误**：以前由 metadata CSV 的父目录推断原始数据根，`--metadata-csv` 指向别处时会去错误目录找 BVH。改为在 catalog manifest 中记录 `seed_root`。
3. **子集后 clip_id 陈旧**：tier 子集保留了裁剪前的 `clip_id/group_id`，保存后索引与 `clips.jsonl` 不一致。新增 `renumber_clips` 稠密重编号，`save()`/`load()` 均做一致性校验。
4. **跨 store 窗口偏移**：token/trajectory store 行对齐但物理偏移不同，条件数据集曾把 token 的绝对帧偏移直接喂给 trajectory store。现按 clip 行重新定位，并在 `verify_token_store` 中把行对齐列为硬校验。

未执行（按方案要求不得在 P0/P1 验收前进行）：全量 142,220 文件的预处理、正式训练、GPU 吞吐测量。因此本文不提供任何全量构建耗时、训练速度或 GPU 利用率结论。

## 3.1 验收意见修正（2026-09-15 复审后）

针对复审提出的四项问题，逐条修正：

1. **配置打不开文档所构建的 store。** 统一目录约定：**store 自身就是配置里写的目录**（`data/processed/seed_soma_pruned_v4`，内含 `manifest.json`），catalog、unit 缓存、token/trajectory store 一律放在它外面。构建时若 `--unit-cache` 落在 store 目录内会直接报错，因为 publish 会整体替换该目录并删掉缓存。另外 reader 在找不到 `manifest.json` 时会列出下属的 store 目录，避免把父子路径写反后只看到一句 missing manifest。
2. **“全量验证”只是受限子集。** `sampling.eval_limit` 现在只约束**监控子集**（每 epoch 或每 N step 的廉价评估）；`DataLoaders.full_val` 是独立的无上限 loader，`test` 从不设限。`evaluation.full_eval_every_epochs` 控制全量扫描周期（最后一轮必扫），且 `best.pt` **只由全量扫描决定**：未跑全量扫描的 epoch 直接不参与比较，不会用不同量纲的子集 loss 混判。日志与 checkpoint 分别记录 `val_loss`、`val_full_loss`、`best_metric_source`。
3. **训练断点续跑未接通。** checkpoint 现在保存 `sampler_state`（epoch、样本序号、是否完成）、`best_val`、`best_metric_source` 与 `normalization_on`；`--resume` 会自动加载 `<output_dir>/last.pt`（位置在 DDP 包装与 loss 构建之前解析，避免模块对象错位），也可继续用 `--checkpoint` 显式指定。`training.checkpoint_every_steps` 支持 epoch 中途落盘，中断后可精确续到下一个样本。位置以**训练器实际训过的样本数**为准：带 worker 的 DataLoader 会预取，采样器自身的计数会超前，不能用来续跑。epoch 是否结束用显式标记记录，因此被 `steps_per_epoch` 截断的 epoch 不会被误判为中断；续跑后的 epoch 最多再跑 `steps_per_epoch` 步，`max_steps` 仍是全局上限。
4. **可选模式行为偏差。** `group_balanced` 现在与 `clip_uniform` 共用同一套 variant-slot 选择，`mirror_probability` 在三套策略下含义一致（真实 store 上 p=1.0 时镜像占比 1.000，修正前 `group_balanced` 约 0.53）；镜像窗口不足时仍回退到有窗口的 variant。`normalize_on: none` 在训练运行器中通过对 device batch 施加 store 统计量真正完成 GPU 侧归一化（`_to_device` → `apply_batch_normalization`），实测与 CPU 路径产出的批次逐元素一致，checkpoint 记录 `normalization_on` 以便追溯。

## 4. 执行入口

目录约定：**store 自身就是配置里写的目录**，catalog 与 unit 缓存是它的同级兄弟目录（publish 会整体替换 store 目录，缓存若在内部会被构建直接拒绝）。

```bash
# P0：目录与划分（输出到 *_catalog）
python -m stylized_motion.run --mode preprocess --pipeline seed-catalog \
  --seed-root data/raw/seed --output data/processed/seed_soma_pruned_v4_catalog \
  --mirror-policy official --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# P0：header / fps / 骨架 / 帧数核对（一次性，结果落盘）
python -m stylized_motion.run --mode preprocess --pipeline seed-inventory \
  --catalog data/processed/seed_soma_pruned_v4_catalog --workers 16

# P1：全量特征构建（可中断续跑）；输出目录即 store 根目录
python -m stylized_motion.run --mode preprocess --pipeline packed-feature-store \
  --catalog data/processed/seed_soma_pruned_v4_catalog \
  --output data/processed/seed_soma_pruned_v4 \
  --unit-cache data/processed/seed_soma_pruned_v4_units \
  --workers 12 --shard-mib 256 --verify full

# P1：split 或统计变化时只重算统计
python -m stylized_motion.run --mode preprocess --pipeline train-stats \
  --store data/processed/seed_soma_pruned_v4

# P2：分层 benchmark
python -m stylized_motion.run --mode benchmark --pipeline data \
  --store data/processed/seed_soma_pruned_v4 --layers sampler,loader,resident,end_to_end

# P3：下游（先 token，再 trajectory）
python -m stylized_motion.run --mode preprocess --pipeline packed-token-store \
  --feature-store data/processed/seed_soma_pruned_v4 \
  --checkpoint outputs/nef_fsq_soma_40x9/best.pt \
  --output data/processed/seed_soma_pruned_v4_tokens
python -m stylized_motion.run --mode preprocess --pipeline packed-trajectory-store \
  --feature-store data/processed/seed_soma_pruned_v4 \
  --output data/processed/seed_soma_pruned_v4_trajectory

# 训练（--resume 从 <output_dir>/last.pt 续跑；不带则全新开始）
python -m stylized_motion.run --mode train --pipeline representation \
  --representation nef-fsq --config data/configs/nef_fsq_soma_packed_40x9.yaml --resume
```

`seed-catalog` 支持 `--tier-groups N` 生成 benchmark 档位：按 package × 长度分位 × 是否有官方镜像分层挑选**完整 take group**（方案 §5 要求覆盖长尾长度、镜像差帧与主要 package）。`--max-groups/--max-clips` 保留为快速截断，同样只切整组。`packed-feature-store` 的 `--verify quick|checksum|full` 控制发布前校验强度，默认 `full`；校验会拒绝“某个 split 有 clip 但没有任何可用 64 帧窗口”的 store。

训练侧使用 `data/configs/nef_fsq_soma_packed_40x9.yaml`（或 `args/nef_fsq_soma_packed_args.txt`）：`required_data_schema_version: 4` 会校验 store 版本，`normalize_on: cpu` 走 CPU 批次归一化，改 `none` 可对比 GPU 批次归一化。

## 5. 待办与风险

- **全量构建未运行**：本地 142,220 文件、约 57.6 GiB 特征的构建耗时、临时空间峰值、worker RSS 需要按 §5 benchmark 设计实测后再定 worker 数。代码按“unit 缓存 + journal”设计，中断可续。
- **镜像逐帧等价性未验证**：官方镜像与程序镜像是否等价、差一帧的时间对齐方式仍待 P0 小样本确认。当前实现按“保留官方镜像、各 variant 可有不同长度”处理，不假设逐帧等价。
- **训练家族未确定**：本仓库现有 SOMA 接线为 NEF-FSQ（248D）。其他 family 需要各自验证 layout 与 loss 后再接手，不能只改 `motion_dim`。
- **旧 100STYLE 数据布局**：`tests/test_preprocess_pipeline.py::test_100style_fsq_excludes_last_ten_styles` 依赖 `100style/<Style>/<Style>_<clip>.bvh` 子目录布局，本机 `data/raw/100style` 是扁平布局，因此该用例在改动前后均失败，与本次迁移无关。

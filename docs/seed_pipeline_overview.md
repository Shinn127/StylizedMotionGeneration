# BONES-SEED 数据流程总览（精炼版）

本文只讲「数据从哪来、经过谁、落到哪、训练怎么读」。设计依据是
[bones_seed_data_pipeline_plan.md](bones_seed_data_pipeline_plan.md)，实现范围与验收见
[bones_seed_pipeline_implementation.md](bones_seed_pipeline_implementation.md)。

## 1. 一张图

```text
  ┌─ ① 目录            stylized_motion/data/seed_catalog.py
  │
  │   输入   data/raw/seed/{metadata/seed_metadata_v004.csv, soma_uniform/bvh/**}
  │          （CSV 为权威目录；temporal_labels.jsonl / parquet 为可选扩展）
  │   动作   · 帧契约：raw_frames(120fps) 与 target_frames(60fps) 分开记录
  │          · 镜像策略：official | generate | none（互斥）
  │          · group key = take_date + take_actor + take_org_name + 规范动作名
  │          · 整组划分 train/val/test（原始、官方镜像、派生片段必同 split）
  │   产出   data/processed/<name>_catalog/{manifest.json, clips.jsonl, index/*.npy}
  │          index/*.npy 是可 mmap 的数字索引，clips.jsonl 存字符串与标签
  │
  ├─ ② 核对            stylized_motion/data/seed_build.py :: inventory_catalog
  │
  │   只读每个 BVH 的头部（Frames / Frame Time / 骨架），逐文件写一行 JSONL。
  │   帧数与 CSV 不符、帧率不是 120/60、骨架哈希不一致 → 立即失败，不进入构建。
  │   产出   <catalog>/inventory.jsonl
  │
  ├─ ③ 特征构建（可中断续跑）  seed_build.py :: run_units → pack_clips
  │
  │   每个 (clip, variant) 是一个 unit，worker 内完成全部重活：
  │     解析 BVH → 降采样 60fps → SOMA 约定（丢静态 Root、剪末端与手指）
  │     → mirror（仅 generate）→ 仿真根 → 速度/角速度/接触 → 248D 特征
  │
  │   一个 unit 产出三个文件（写在 unit 缓存目录，不进 store）：
  │     <unit_id>.npy        未归一化特征            [N,248]
  │     <unit_id>.root.npy   根位置 + 朝向           [N,7]   ← 轨迹通道，顺带产出
  │     <unit_id>.meta.json  names/parents/position_sum/frames
  │
  │   unit_id = hash(预处理签名, clip, variant, 输入指纹)
  │   journal.jsonl 记录 commit/failure 与输出 checksum：
  │     签名 + 指纹 + checksum 三者同时匹配才跳过，否则重建该 unit
  │
  │   打包：split 连续、分片内混来源（按 group 哈希排序，避免片内偏置）
  │
  ├─ ④ STORE  v4 packed            stylized_motion/data/packed_store.py
  │
  │   路径   data/processed/<name>/     ← 就是配置里的 data.fsq_window_index
  │
  │   motion/shard_00000.npy   约 256 MiB/片（128/256/512 可调），未归一化 float32 [N,248]
  │   root/shard_00000.npy     同分片、同 offset、同 length 的 [N,7]（并行数组）
  │   clips/*.npy              clip 表：shard / offset / length / source_group /
  │                            variant / split / mirror / style / action / package /
  │                            position_sum
  │   normalization.npz/.json  独立版本化的 train-only 统计量
  │   manifest.json            四组 hash 分开保存：
  │                            skeleton_hash · feature_schema_hash ·
  │                            split_manifest_hash · normalization_hash
  │
  ├─ train-stats        → 只重算统计（normalization_hash 变），特征字节不动
  │
  ├─ packed-token-store → data/processed/<name>_tokens/
  │     按**逻辑 clip** 逐个编码（chunk 只在 clip 内回放历史，不跨 clip），
  │     token 表与特征表逐行对齐；绑定 checkpoint / feature / normalization / split
  │
  └─ packed-trajectory-store → data/processed/<name>_trajectory/
        读 root/，每 clip 算未来根位移与朝向；validity = [0, length-horizon)；
        train-only 流式统计；表与特征表逐行对齐

  └─ ⑤ TRAIN               stylized_motion/learning/runner.py
  │
  │   sampler        sample(seed, epoch, rank, ordinal) 决定采样顺序
  │                  group 等概率 → variant（镜像概率，无窗口则回退）→ 窗口起点
  │                  窗口只在逻辑 clip 内；跨片段/跨 split 直接报错
  │                  DDP 每 rank 步数相等（tail=drop|pad）；支持 epoch 中途恢复
  │
  │   dataset        按 shard 分组读 mmap → 拼 [B,64,248]
  │                  normalize_on=cpu  → 数据集内就地归一化（默认）
  │                  normalize_on=none → 附带 offset/scale，交给设备侧
  │
  │   device         移到设备 → apply_batch_normalization() → forward → loss
  │                  （offset/scale 来自 store 的版本化统计量）
  │
  │   预算           steps_per_epoch / max_steps / checkpoint_every_steps
  │                  eval_every_steps（监控子集）/ full_eval_every_epochs（全量）
  │                  best.pt 只由全量验证决定；--resume 从 <output_dir>/last.pt 续跑
```

## 2. 产物与目录约定

| 产物 | 路径 | 里面有 | 能删吗 |
|---|---|---|---|
| catalog | `data/processed/<name>_catalog/` | `manifest.json`、`clips.jsonl`、`index/*.npy`、`labels_*.json` | 可重建，但重建会换 split hash |
| inventory | `<catalog>/inventory.jsonl` | 每个文件的 header 核对结果 | 可删，重跑约一次全量 stat |
| unit 缓存 | `data/processed/<name>_units/` | `<unit_id>.npy` / `.root.npy` / `.meta.json` / `journal.jsonl` | **可删**；删后重跑会重新解析 BVH |
| **特征 store** | `data/processed/<name>/` | `motion/`、`root/`、`clips/`、`normalization.*`、`manifest.json` | 重新构建代价最大 |
| token store | `data/processed/<name>_tokens/` | `tokens/`、`clips/`、`manifest.json` | 有 checkpoint 即可重建 |
| trajectory store | `data/processed/<name>_trajectory/` | `trajectory/`、`valid/`、`clips/`、`manifest.json` | 可重建 |

`<name>` 就是训练配置里 `data.fsq_window_index` 写的那个路径——**store 自己就是目录，不套一层 `store/`**。catalog 与 unit 缓存是它的同级兄弟；构建时若 `--unit-cache` 落在 store 目录内部会直接报错，因为 publish 是整体原子替换，缓存会被静默删掉。

## 3. 阶段对照表

| 阶段 | 命令 | 输入 | 输出 | 代码 |
|---|---|---|---|---|
| 目录与划分 | `preprocess seed-catalog` | CSV | catalog | `data/seed_catalog.py` |
| 头信息核对 | `preprocess seed-inventory` | catalog + BVH 头 | inventory.jsonl | `data/seed_build.py` |
| 特征构建 | `preprocess packed-feature-store` | catalog | store + units | `data/seed_build.py`、`data/packed_store.py` |
| 统计重算 | `preprocess train-stats` | store | `normalization.*` | `data/normalization.py` |
| 校验 | `preprocess validate-data --packed-store …` | store | 报告 | `data/seed_build.py` |
| benchmark | `benchmark data` | store | 分层报告 | `data/benchmark.py` |
| token | `preprocess packed-token-store` | store + checkpoint | token store | `data/packed_token.py` |
| 轨迹 | `preprocess packed-trajectory-store` | store | trajectory store | `data/packed_trajectory.py` |
| 训练 | `train representation --config … [--resume]` | store | checkpoint | `learning/runner.py` |

三段构建都支持中断续跑，共用 `data/resume.py` 的 unit id / 输入指纹 / journal / 有界调度。

## 4. 训练时的数据路径

```text
  ┌─ sampler：TrainWindowSampler（data/sampling.py）
  │
  │   构造时   一次 lexsort 建紧凑索引：group → variant slot → interval
  │            （成本 O(n log n)，与 group 数无关；v3 是 O(source × interval)）
  │   采样时   每个 ordinal 一个独立 RNG（可精确恢复）
  │            group 等概率（或 frame_uniform 按有效起点加权 / group_balanced 封顶加权）
  │            → 选 variant（镜像概率，无可用窗口则回退到有窗口的 variant）
  │            → 选区间 → 选窗口起点
  │
  ├─ SampleRequest(shard_idx, target_start, 64, variant_idx)
  │
  ┌─ dataset：PackedFeatureDataset / PackedTokenDataset
  │
  │   按 request.shard_idx 分组，逐个窗口从 mmap 读出 → 拼 [B,64,248]
  │   normalize_on=cpu  → 就地做 (x-offset)/scale
  │   normalize_on=none → 原样返回并附带 normalization{offset,scale,hash}
  │
  ├─ dict{motion, loss_mask[, normalization][, metadata]}
  │
  ├─ DataLoader（num_workers/prefetch_factor/pin_memory，worker 内清空 mmap 缓存）
  │
  ├─ 主进程：_to_device() → apply_batch_normalization() → 模型 forward → loss
  │
  └─ 评估：sampling.eval_limit 只约束监控子集
           DataLoaders.full_val 是无上限 loader（未配置 limit 时不存在，val 即全量）
           best.pt 只在跑完全量扫描的 epoch 之间比较，日志并列 val_loss / val_full_loss
```

## 5. 两代契约的分工

| | v3（既有） | v4（SEED） |
|---|---|---|
| 命令 | `feature-cache` → `fsq-window-index` | `seed-catalog` → `packed-feature-store` |
| 物理布局 | 每个 (来源, 镜像) 一个文件，约 28 万个小文件 | 约 256 MiB 分片，14.2 万 clip 装进约 231 片 |
| 逻辑与物理 | 一一对应 | clip 表（shard/offset/length）解耦 |
| 归一化 | 烧进特征字节，换 split 就要重写 | 独立版本化，`train-stats` 只改统计 |
| 镜像 | 每文件无条件生成 original+mirror | 官方镜像直接复用，策略显式三选一 |
| 划分 | 随机窗口 | take group 整组进同一 split |
| 下游 | `token-database` / `trajectory-*`（大 NPZ、整 shard 上下文） | `packed-token-store`（clip 内编码）/ `packed-trajectory-store`（clip-local + mask） |

`open_any_feature_store` 按 manifest 的 `data_schema_version` 分派，v3 store 不会被就地改写；`--dataset seed` 会明确拒绝走 v3 路径（那会重复生成 7.1 万个官方镜像）。

## 6. 各接缝处的硬保证

- **帧**：`raw_frames`（120fps 行数）与 `target_frames`（60fps，`ceil(raw/2)`）分开记录；元数据帧数当 60fps 用会直接报错。镜像少一帧是合法 variant 长度差。
- **不越界**：窗口只能在单个逻辑 clip 的 `[offset, offset+length)` 内；跨片段、跨 split 一律 `IndexError`。轨迹未来帧同理，尾部由 validity mask 标为无效。
- **统计只来自 train**：`train-stats` 只扫 train 区间，`val`/`test` 帧不参与；换 split 或统计不重写特征字节。
- **可恢复**：unit 命中复用需签名 + 输入指纹 + 输出 checksum 三者同时匹配；训练续跑以「实际训过的样本数」为准（worker 预取会让采样器计数超前，不能用来续跑）。
- **可追溯**：四组 hash 分开保存，token/trajectory store 各自绑定它们，checkpoint 也记录 `normalization_on` 与 `best_metric_source`。
- **performer 列可选**：`clip_performer_id` + manifest 的 `performer_names` 来自 catalogue 的 `take_actor`（缺失时回退 `actor_uid`）。
  没有这列出厂的老 store 仍可打开，`clip_label()` 的 `performer` 返回空串——风格审计据此报告
  “演员表不可用”，而不是把 group id 当成演员。（当前 `data/processed/seed_soma_pruned_v4` 就是这种老表；
  重跑一次 `packed-feature-store --overwrite` 即可补上，unit 缓存会复用。）

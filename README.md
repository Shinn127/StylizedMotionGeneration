# Stylized Motion Generation

面向风格化角色动作的离散 motion representation 与生成实验项目。当前主线是五种共享数据、训练、checkpoint 和生成接口的 canonical FSQ representation：

| CLI | family / variant | 语义 |
| --- | --- | --- |
| `flat-fsq` | `flat_fsq / flat` | 所有 motion feature 进入一个 flat stream |
| `part-fsq` | `part_fsq / hierarchical` | global、sync、torso、双腿、双臂的层次化 stream |
| `residual-part-fsq` | `residual_part_fsq / default` | holistic base 加 feature-space local residual |
| `latent-residual-fsq` | `latent_residual_fsq / v2` | holistic base 加显式互斥子空间中的 latent residual（旧实现，用于对照） |
| `latent-residual-fsq-v2` | `latent_residual_fsq_v2 / v2` | 非结构化 base latent 加全维 part residual projection，支持 compensated part edit |

五条 representation 使用同一 canonical contract：`motion_dim=230`、`40` 个 FSQ coordinate、每个 coordinate `9` 个 level、`frame_rate=60`、`receptive_field=64`、`lookahead_frames=0`。训练窗口长度为 64 帧，窗口内每一帧只访问当前帧及其历史，不再携带 `context_left`。

数据侧规范见 [data_pipeline_spec.md](docs/data_pipeline_spec.md)，representation 和 learning 的跨层规范见 [refactor_spec.md](docs/refactor_spec.md)。当前只读取 data schema v3 和 checkpoint schema v2，不提供旧数据兼容层。

## 1. 目录

```text
args/                         MimicKit 风格命令预设
data/configs/                 representation、generator、legacy VQ-VAE 配置
stylized_motion/
  run.py                      唯一应用级 dispatch 入口
  data/                       Store、Dataset、sampling、loader、preprocess
  learning/                   五类 FSQ、统一 runner、loss、generator
    nets/                     causal CNN、Transformer、quantizer
  anim/                       BVH/feature、GenoView、序列和实时可视化
  util/                       参数文件和路径工具
tests/                        contract 与回归测试
docs/
  data_pipeline_spec.md       数据侧唯一规范
  refactor_spec.md            representation/learning 重构规范
```

数据模块保持平级组织：`feature_data.py`、`token_data.py`、`trajectory_data.py`、`sampling.py`、`loader.py`、`preprocess.py`。没有额外的 `datasets/`、`stores/`、`samplers/` 或 workflow wrapper 子包。`learning/vqvae.py` 和 `data/configs/vqvae_*.yaml` 是隔离的 legacy baseline。

## 2. 环境

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
pip install -r requirements.txt
```

运行支持 `--device auto|cuda|mps|cpu`。GPU 训练通常使用 `--device cuda`。

## 3. 数据构建

完整数据链为：

```text
BVH / 原始动作
  -> motion database（可选）
  -> FeatureStore（230D 标准化 motion）
  -> TokenStore（40 个 uint8 FSQ index）
  -> TrajectoryStore（可选的对齐条件轨迹）
```

### 3.1 可选的原始 motion database

用于轨迹输入构建和低层动作检查：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline motion-database \
  --dataset 100style \
  --output data/processed/100style/motion_database.npz \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --workers 8
```

支持的数据集入口是 `lafan`、`100style` 和 `combined`；`--styles` 可筛选 100style 样式，`--max-styles` 适合小规模试跑。LAFAN 当前只保留 source clip，不推断 style/action 标签。

### 3.2 FSQ feature cache and window index

FSQ 数据构建只保留两个阶段：`feature-cache` 把完整 source sequence 转为未归一化的帧级 `motion_feature_v2`；`fsq-window-index` 在自己的 staging 目录中生成归一化 shard、切分不重叠的 64 帧 window，并按固定 seed 以 8:1:1 分配。100style 最后 10 个 style 会写入 cache，但排除出标准训练 index，记录在 `unseen_style_names`。

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-cache \
  --dataset 100style \
  --output data/processed/100style_pruned_90/feature_cache \
  --prune-ends-and-fingers \
  --workers 8 \
  --overwrite

python -m stylized_motion.run \
  --mode preprocess --pipeline fsq-window-index \
  --feature-cache data/processed/100style_pruned_90/feature_cache \
  --output data/processed/100style_pruned_90/fsq_window_index \
  --seed 3407 \
  --overwrite
```

联合构建时，LAFAN 使用完整 source clip，100style 使用 `Frame_Cuts.csv` 的 `[START, STOP)` 区间；推荐同样先生成 cache，再生成 FSQ window index：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-cache \
  --dataset combined \
  --output data/processed/combined_pruned/feature_cache \
  --prune-ends-and-fingers \
  --workers 1 \
  --overwrite

python -m stylized_motion.run \
  --mode preprocess --pipeline fsq-window-index \
  --feature-cache data/processed/combined_pruned/feature_cache \
  --output data/processed/combined_pruned/fsq_window_index \
  --seed 3407 \
  --overwrite
```

旧的 `feature-database` 入口仍保留用于兼容旧的 token/trajectory 构建；FSQ 训练只使用上面的两阶段入口。
TokenStore 可以直接继承 `fsq-window-index` 的最终 FeatureStore metadata 和 split，避免再次构建一套 feature database。
Trajectory 目前仍从 raw `motion-database.npz` 生成；它使用整段 source clip range，与 FSQ 的 64 帧 window range 尚未统一，暂不要混用两者。
如需 trajectory inputs，仍可显式添加 `--motion-database-output` 或单独运行 `motion-database`。

combined Store 的 `range_names` 使用 `lafan/<clip>` 和 `100style/<style>_<clip>` 前缀，避免 source clip 冲突。

物理布局：

```text
feature_cache/
    manifest.json
    index.npz
    motion/shard_00000.npy
    ...

fsq_window_index/
    manifest.json
    index.npz
    unseen_index.npz
    motion/shard_00000.npy
    ...
```

manifest 保存 data schema、相对 shard 路径、split manifest、feature schema、skeleton metadata 和 hash；index 保存 shard/range/split 数值索引及 normalization arrays。运行时使用 mmap 和 lazy shard cache，不会一次加载全部动作。

### 3.3 数据验证

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline validate-data \
  --feature-store data/processed/100style_pruned_90/fsq_window_index \
  --full
```

验证会检查 schema、manifest、hash、shard shape、range bounds、split 隔离、normalization 和完整 shard 内容。TokenStore、TrajectoryStore 可通过对应参数加入同一次验证。

## 4. Representation 数据管线

训练时调用链为：

```text
FeatureStore
  -> SplitManifest
  -> TrainWindowSampler / FixedWindowSampler
  -> FeatureDataset
  -> batch-oriented collate
  -> DataLoader
  -> move_batch_to_device()
  -> RepresentationRunner
  -> representation + losses
```

### 4.1 Sampling

- train 先均匀采样 source clip，再在该 clip 的有效窗口内随机采样，支持 `samples_per_epoch` 和 mirror probability；因此 clip 长度不会直接改变总体采样概率；
- val/test 使用确定性的完整 64 帧窗口，默认 stride 为 64，尾部不足窗口的部分丢弃；
- DDP 下 sampler 按 rank 分配互不重叠的 sample positions；
- 相同 Store、seed、epoch、world size 产生可复现采样序列；
- Dataset 只消费 `SampleRequest`，不自行决定 split 或随机窗口。

### 4.2 Representation batch

```text
motion     float32 [B, 64, 230]
loss_mask  bool    [B, 64]
```

64 帧全部是 supervised target，canonical batch 的 `loss_mask` 默认全为 true。模型保持 causal receptive field 64；第 `t` 帧只访问窗口内不晚于 `t` 的帧，不跨 clip 或 split 拼接历史。

### 4.3 Loader

`stylized_motion.data.loader.build_data_loaders()` 统一负责 batch read、contiguous tensor、worker seed、pin memory、persistent workers、prefetch、drop-last 和 shard cache 限制。learning 层只接收 CPU tensor batch，不直接打开 shard。

典型配置：

```yaml
loader:
  batch_size: 512
  num_workers: 4
  pin_memory: auto
  persistent_workers: true
  prefetch_factor: 2
  drop_last_train: true
  max_open_shards: 32
  prefetch_memory_limit_mb: 512
```

## 5. 五类 FSQ

五类模型均由 `stylized_motion/learning/representation.py` registry 构造。训练、验证、测试和 checkpoint 流程通过 `RepresentationAdapter` 使用统一接口，不在 workflow 中直接实例化具体模型类。

| family | coordinate layout |
| --- | --- |
| `flat_fsq` | `flat:40` |
| `part_fsq` | `global:6, sync:4, torso:6, left_leg:7, right_leg:7, left_arm:5, right_arm:5` |
| `residual_part_fsq` | `base:20, torso:6, left_leg:4, right_leg:4, left_arm:3, right_arm:3` |
| `latent_residual_fsq` | 与 residual part 相同；互斥 latent slice，`architecture_version=2` |
| `latent_residual_fsq_v2` | 与 residual part 相同；全维 latent projection，`architecture_version=3` |

各模型的核心差异如下：

- Flat-FSQ 使用一个完整 motion encoder、一个 40-coordinate quantizer 和一个 decoder。
- Part-FSQ 将 feature 划分为 global 与五个 body part，并额外学习 sync stream；每个 part 具有独立的 code group 和解码写入区域。
- Residual Part-FSQ 先通过 base code 重建完整 motion，再把五个 part decoder 产生的 feature residual 加到对应 feature 区域。
- Latent Residual-FSQ 旧实现从局部 feature state 与 base predictor 的差中量化 part residual，将五个 residual 投影到互斥的 base-latent slice 后求和，只调用共享 decoder。
- Latent Residual-FSQ V2 保持 base latent 和 decoder 输入为非结构化 128D latent。每个 part residual 投影到完整 128D latent，最终使用 `base + sum(part_delta)` 解码；迁移 donor part 时根据 donor base 恢复 donor part state，再相对 target base 做 compensation。

后四类使用固定的 body-part 集合：`torso`、`left_leg`、`right_leg`、`left_arm`、`right_arm`。模型差异封装在 encoder、quantizer、decoder 和 representation-specific loss 内部；共同 loss 由统一层计算 feature reconstruction、temporal delta、root trajectory、FK joint、contact、foot slide 和 foot height。

## 6. 训练、验证、测试

五条主线共用 `RepresentationRunner`。每个训练 batch 执行一次 forward、一次 loss 计算和一次 backward/update；同一个 device batch 被 model 和 loss 复用。

### 6.1 训练

```bash
python -m stylized_motion.run \
  --mode train \
  --pipeline representation \
  --representation flat-fsq \
  --config data/configs/flat_fsq_40x9.yaml \
  --device cuda
```

其他主线配置：

```text
part-fsq             data/configs/part_fsq_40x9.yaml
residual-part-fsq    data/configs/residual_part_fsq_40x9.yaml
latent-residual-fsq  data/configs/latent_residual_fsq_40x9.yaml
latent-residual-fsq-v2 data/configs/latent_residual_fsq_v2_40x9.yaml
```

也可以直接使用 MimicKit 风格的参数预设：

```bash
python -m stylized_motion.run --arg-file args/latent_residual_fsq_v2_args.txt
```

配置边界是：`data` 描述 Store，`sampling` 描述窗口和采样，`loader` 描述 batch/worker/host memory，`training` 描述 optimizer、epoch、precision、clip、seed 和 output。默认输出目录是 `outputs/<family>_40x9/`。

V2 训练会始终监督 base reconstruction。batch size 大于 1 时，runner 每个 step 轮换一个 body part，并用 batch 内循环移位构造 donor，联合计算 part transfer 和 non-part preserve loss。base、final reconstruction 和 edited reconstruction 的 latent 会合并后通过一次共享 decoder 调用。

### 6.2 训练效率

- `evaluation.metrics_interval` 控制昂贵 FSQ usage/perplexity 统计的采样频率，当前配置默认为每 100 step 一次；loss 仍在每个 step 完整计算。
- 没有外部 metric callback 时，runner 使用 `compact_output=True`，丢弃训练 loss 不需要的诊断 tensor。
- epoch scalar 在 device 上累加，epoch 结束时统一传到 CPU；DDP 使用一次 packed all-reduce 聚合。
- `precision` 支持 `fp32` 和 CUDA-only `amp`；当前五份 representation 配置默认使用 `fp32`。

### 6.3 验证和测试

```bash
python -m stylized_motion.run \
  --mode validate --pipeline representation \
  --representation flat-fsq \
  --config data/configs/flat_fsq_40x9.yaml \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --device cuda

python -m stylized_motion.run \
  --mode test --pipeline representation \
  --representation flat-fsq \
  --config data/configs/flat_fsq_40x9.yaml \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --device cuda
```

输出 JSON，包含 metrics、有效帧、样本数、data wait、step time、target frames/s 和 samples/s。训练控制台还会打印 train/val total loss、reconstruction loss 和 samples/s。

## 7. Loss 与 TensorBoard

统一 loss 对 64 帧窗口按 `loss_mask` 聚合；epoch 结果按有效帧数加权平均，而不是简单平均 batch。当前 common loss 字段为：

```text
loss, recon, delta, root_pos, root_rot, joint, contact, foot_slide, foot_height
```

representation-specific loss 由各模型的 `compute_representation_losses()` 提供并参与总 loss。

当前各 family 的额外 loss 为：

| family | representation-specific loss |
| --- | --- |
| `flat_fsq` | 无 |
| `part_fsq` | `reuse` |
| `residual_part_fsq` | `base_reuse` |
| `latent_residual_fsq` | `latent_energy` |
| `latent_residual_fsq_v2` | `base_recon`，训练时另有 `part_edit_transfer`、`part_edit_preserve` |

这些字段会进入 epoch 聚合结果和 checkpoint metrics。TensorBoard 当前只写入总 loss、八个 common loss component，以及按 `metrics_interval` 采样的 `train/step_loss`；representation-specific 字段尚未建立独立 tag。

训练日志位置：

```text
outputs/<family>_40x9/tensorboard/
```

当前 tags：

```text
train/step_loss
epoch/train_loss          epoch/val_loss
epoch/train_recon         epoch/val_recon
epoch/train_delta         epoch/val_delta
epoch/train_root_pos      epoch/val_root_pos
epoch/train_root_rot      epoch/val_root_rot
epoch/train_joint         epoch/val_joint
epoch/train_contact       epoch/val_contact
epoch/train_foot_slide    epoch/val_foot_slide
epoch/train_foot_height   epoch/val_foot_height
```

查看：

```bash
tensorboard --logdir outputs/flat_fsq_40x9/tensorboard
```

当前 generator training loop 保存 NLL、perplexity、coordinate accuracy 到 checkpoint 和控制台，但尚未接入 `SummaryWriter`。

## 8. Checkpoint

representation 默认输出：

```text
outputs/<family>_40x9/
  last.pt
  best.pt
  tensorboard/
```

schema v2 checkpoint 包含 model family、model config、state dict、canonical `representation` metadata、top-level `feature_schema`、feature normalization statistics、epoch/global step、optimizer/scheduler state 和 train/val metrics。

加载时会校验 family、variant、`40x9`、coordinate layout、feature schema、motion dimension、causal metadata 和 normalization contract，不会根据旧字段猜测 representation。

## 9. TokenStore 与 Generator

### 9.1 构建 TokenStore

Token database 必须使用已训练的 representation checkpoint。`run.py` 在 composition root 加载 checkpoint 并注入 encoder，data 层不构造 representation。

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline token-database \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --feature-store data/processed/100style_pruned_90/fsq_window_index \
  --output data/processed/100style_pruned_90/flat_fsq_40x9 \
  --device cuda --chunk-size 1024 --save-codes
```

TokenStore 继承最终 FeatureStore 的 range、source clip、split、frame rate 和 feature schema，不重新生成 split。每个 token frame 保存 `[40]` 个 `uint8` index；可选 FSQ code 以 `float16` 保存。分块编码时会读取所需的 63 帧历史，保持 chunk 边界的 causal 语义。

### 9.2 Generator batch

generator 使用 65 个 token frame：

```text
indices          uint8 [B, 65, 40]
inputs           indices[:, :-1]  -> [B, 64, 40]
next-token target indices[:, 1:]  -> [B, 64, 40]
```

一次 generator loss 对 64 个 target frame、共 `64 * 40` 个 coordinate-level token 计算 cross entropy。条件 generator 还读取 `trajectory [B,64,C]` 和 `trajectory_valid [B,64]`。

### 9.3 训练与生成

```bash
python -m stylized_motion.run \
  --mode train --pipeline generator \
  --config data/configs/fsq_generator.yaml

python -m stylized_motion.run \
  --mode generate --pipeline motion \
  --token-database data/processed/100style_pruned_90/flat_fsq_40x9 \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --seed-indices outputs/seed_indices.npy \
  --steps 60 --output outputs/generated_indices.npy --greedy
```

generator checkpoint 绑定 tokenizer checkpoint SHA256、TokenStore representation metadata 和 feature schema。生成器从 TokenStore metadata 获取 coordinate 数、level 数和 layout，不把 Part/Residual token 当作 Flat token 解释。

## 10. 可视化与实时生成

### 10.1 representation source/reconstruction

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline motion \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --range-idx 0 --start 0 --length 240 \
  --view compare --device cuda
```

`--view` 可选 `source`、`recon`、`compare`；`--dry-run` 只执行加载和重建检查；`--save-debug` 保存 source/recon features、tokens 和 metadata。

### 10.2 Part edit

`part-edit` 从同一 FeatureStore 读取 target 与 donor 片段，将 donor 的指定 body-part code 或 latent residual 迁移到 target，并使用 GenoView 并排播放参考动作和编辑结果。它支持 `part_fsq`、`residual_part_fsq`、`latent_residual_fsq` 和 `latent_residual_fsq_v2`；Flat-FSQ 没有 part layout，不支持该 pipeline。

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline part-edit \
  --checkpoint outputs/latent_residual_fsq_v2_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --target-range-idx 0 --target-start 0 \
  --donor-range-idx 16 --donor-start 0 \
  --length 240 \
  --part left_arm \
  --compare-with target \
  --device cuda
```

可编辑部位为 `torso`、`left_leg`、`right_leg`、`left_arm` 和 `right_arm`。`--compare-with target` 用于观察非目标部位是否保持，`--compare-with donor` 用于观察目标部位是否接近 donor。

不同 family 保留各自的编辑语义：Part-FSQ 和 Residual Part-FSQ 替换对应 part 的 code group；旧 Latent Residual-FSQ 迁移 donor latent residual；V2 使用 donor base-conditioned state 相对 target base 重新计算 compensated residual。pipeline 会按模型的 63 帧历史需求读取 target/donor 上下文，并在展示前裁掉历史帧。

无窗口检查与 debug 输出：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline part-edit \
  --checkpoint outputs/latent_residual_fsq_v2_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --target-range-idx 0 --target-start 0 \
  --donor-range-idx 16 --donor-start 0 \
  --length 240 --part left_arm \
  --device cuda --dry-run --save-debug
```

默认输出位于：

```text
outputs/latent_residual_fsq_v2_40x9/part_edit_visualization/
  target_000_donor_016_left_arm/
    target_source_features.npy
    donor_source_features.npy
    target_recon_features.npy
    donor_recon_features.npy
    edited_features.npy
    target_indices.npy
    donor_indices.npy
    metadata.json
```

Part-FSQ 和 Residual Part-FSQ 还会保存 `edited_indices.npy`。终端输出中的 `target_part_change` 表示被编辑 feature partition 相对 target reconstruction 的平均变化，`non_target_change` 表示其他 partition 的平均变化；二者用于快速辅助观察，不是训练或正式评估指标。

`target-range-idx` 和 `donor-range-idx` 是 FeatureStore range index，`target-start` 和 `donor-start` 是各自 range 内的相对帧。checkpoint 与 FeatureStore 必须具有相同的 structural feature schema 和 skeleton；如果 normalization statistics 不同，pipeline 会先还原 raw feature，再按 checkpoint statistics 重新标准化。

### 10.3 SomaView

SomaView 用 GenoView 的同一渲染管线播放 BONES-SEED 数据集的 SOMA 骨架动作。SOMA rig 与 Geno 共享 simulation-root 约定(同为 `Hips`/`Spine2` 关节、厘米 BVH、ZYX 欧拉角),因此查看器直接复用,只需替换资产。

资产需要从 BONES-SEED 的 `soma_shapes` 生成一次(gated 数据集,需先在 HuggingFace 上接受 license;只下载 `soma_shapes` 子目录,约 5MB,不含动作数据):

```bash
hf download bones-studio/seed --repo-type dataset --include "soma_shapes/*" \
  --local-dir data/raw/bones_seed

python -m stylized_motion.run \
  --mode preprocess --pipeline soma-assets \
  --usd data/raw/bones_seed/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd \
  --bvh data/raw/bones_seed/soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh \
  --output-dir data/assets/somaview --overwrite
```

转换器从 USD 提取 78 关节骨架和 18056 顶点蒙皮网格(每顶点 8 影响,截断为 top-4 重归一),bind pose 采用 USD `bindTransforms` 的蒙皮 rest pose(与 bind BVH 的 T-pose 是不同姿态),写出 GenoView 二进制契约的 `SOMA.bin` 并拷贝着色器。资产与 Geno 相同被 gitignore,删库后用上面命令再生成。

播放 soma_uniform 动作(SOMA 采集帧率 120fps):

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline somaview \
  --bvh data/raw/bones_seed/soma_uniform/bvh/<clip>.bvh --fps 120
```

如果需要将 SOMA 的 120 FPS BVH 下采样为 60 FPS，可使用仓库内的批处理脚本。脚本保留原始骨架层级和通道布局，默认每隔一帧取一帧，并自动将 `Frame Time` 加倍：

```bash
# 单个文件
python scripts/downsample_bvh.py \
  data/raw/bones_seed/bvh/soma_uniform/210531/jump_and_land_heavy_001__A001_M.bvh \
  data/raw/bones_seed/bvh/soma_uniform_60fps/210531/jump_and_land_heavy_001__A001_M.bvh

# 整个目录，递归保留相对目录结构
python scripts/downsample_bvh.py \
  data/raw/bones_seed/bvh/soma_uniform \
  data/raw/bones_seed/bvh/soma_uniform_60fps
```

输出文件已存在时加 `--overwrite`；其他倍率可用 `--factor N` 指定。例如 `--factor 4` 表示保留每 4 帧中的第 1 帧。

在下载动作数据之前,可以先检查两个标准姿态(转换时一并生成,均无窗口冒烟通过):

```bash
# T-pose:bind BVH 的第 0 帧,定义骨架层级
python -m stylized_motion.run --mode visualize --pipeline somaview \
  --bvh data/assets/somaview/SOMA_bind.bvh

# A-pose:USD bindTransforms 的蒙皮 rest pose,由转换器导出
python -m stylized_motion.run --mode visualize --pipeline somaview \
  --bvh data/assets/somaview/SOMA_apose.bvh
```

A-pose BVH 从 USD 世界 bind 变换反解局部旋转和平移再写回 ZYX 欧拉,经 FK 闭环校验(与 USD 逐关节零误差)。

也支持 `--database` npz;`--features` 需要尚不存在的 SOMA 特征库,为预留接口。关节名校验要求数据库关节是 SOMA bind 骨架(78 关节)的子集,缺失部位保持 bind pose。

### 10.4 GenoView

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline genoview \
  --database data/processed/100style/motion_database.npz
```

也可以直接播放单个、且骨骼名称与 Geno bind skeleton 对齐的 BVH 文件，无需先生成
`motion_database.npz`：

```bash
conda run -n mcc python -m stylized_motion.run \
  --mode visualize --pipeline genoview \
  --bvh data/raw/lafan/walk1_subject1.bvh
```

该入口会沿用项目的 BVH 解析、米制缩放和 Simulation root 约定；BVH 的骨骼名称
必须与 Geno bind skeleton 兼容，否则会在启动时报告不匹配的关节。

也可以查看 230D features：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline genoview \
  --features outputs/sequence_debug/recon_features.npy \
  --stats-source data/processed/100style_pruned_90/feature_database
```

GenoView 支持播放、暂停、逐帧移动、速度调整、时间轴拖拽和相机交互。渲染默认使用 legacy 光照模型；`--shading pbr` 显式启用 Cook-Torrance 光照和 ACES tone mapping。使用 `--debug-view albedo|normal|depth|ssao|lighting` 可查看中间缓冲，用于定位颜色和光照问题。

### 10.5 实时 FSQ rollout

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline realtime \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --fsq-checkpoint outputs/flat_fsq_40x9/best.pt \
  --token-database data/processed/100style_pruned_90/flat_fsq_40x9 \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --seed-source reencode --range-idx 0 --seed-frames 64 --device cuda
```

无窗口 smoke test：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline realtime \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --fsq-checkpoint outputs/flat_fsq_40x9/best.pt \
  --token-database data/processed/100style_pruned_90/flat_fsq_40x9 \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --dry-run --dry-run-frames 120
```

实时控制器默认使用 64 帧 seed，并维护 causal pose buffer。`--seed-source token-db` 可复用已保存 token，但 tokenizer checkpoint、TokenStore 和 FSQ checkpoint 必须匹配。

## 11. 参数预设、测试与排错

```bash
python -m stylized_motion.run --arg-file args/flat_fsq_args.txt
python -m pytest -q
python -m compileall -q stylized_motion
```

建议大规模训练前先执行 package/registry 测试和 `validate-data --full`。常见错误包括：CLI representation 与 config family 不一致、Store 不是 schema v3、checkpoint 与 feature schema 不匹配、generator 使用了不同 tokenizer checkpoint、窗口长度不是 canonical 64/65，或传入包含 `context_left` 的历史 metadata。

## 12. 当前边界

当前主训练路径是五条 FSQ representation，其中 `latent_residual_fsq` 保留为旧实现对照，新增实验应优先使用独立的 `latent_residual_fsq_v2` family。Generator 已有独立训练和生成路径，但尚未与 representation runner 共用 TensorBoard writer；legacy VQ-VAE 仅用于 baseline。data 层不负责 model、optimizer、loss、AMP、checkpoint 或 device transfer；learning 层不直接访问 shard 文件。

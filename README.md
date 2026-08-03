# Stylized Motion Generation

面向风格化角色动作的离散 motion representation 与生成实验项目。当前主线是四种共享数据、模型、训练和 checkpoint contract 的 FSQ representation：

| CLI | family / variant | 语义 |
| --- | --- | --- |
| `flat-fsq` | `flat_fsq / flat` | 所有 motion feature 进入一个 flat stream |
| `part-fsq` | `part_fsq / hierarchical` | global、sync、torso、双腿、双臂的层次化 stream |
| `residual-part-fsq` | `residual_part_fsq / default` | holistic base 加 feature-space local residual |
| `latent-residual-fsq` | `latent_residual_fsq / v2` | holistic base 加互斥 latent residual |

四条 representation 使用同一 canonical contract：`motion_dim=230`、`40` 个 FSQ coordinate、每个 coordinate `9` 个 level、`frame_rate=60`、`receptive_field=64`、`lookahead_frames=0`。训练窗口长度为 64 帧，窗口内每一帧只访问当前帧及其历史，不再携带 `context_left`。

数据侧唯一规范是 [docs/data_pipeline_spec.md](/Users/shinn/Documents/Projects/StylizedMotionGeneration/docs/data_pipeline_spec.md)，representation 和 learning 的跨层规范是 [docs/refactor_spec.md](/Users/shinn/Documents/Projects/StylizedMotionGeneration/docs/refactor_spec.md)。当前只读取 data schema v3 和 checkpoint schema v2，不提供旧数据兼容层。

## 1. 目录

```text
args/                         MimicKit 风格命令预设
data/configs/                 representation、generator、legacy VQ-VAE 配置
stylized_motion/
  run.py                      唯一应用级 dispatch 入口
  data/                       Store、Dataset、sampling、loader、preprocess
  learning/                   四类 FSQ、统一 runner、loss、generator
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

支持的数据集入口是 `lafan` 和 `100style`；`--styles` 可筛选样式，`--max-styles` 适合小规模试跑。

### 3.2 FeatureStore

representation 训练直接消费 FeatureStore。构建过程会读取 BVH、计算 `motion_feature_v2`、按 source clip 建立 train/val/test split、只使用 train split 拟合 normalization statistics，并在 full validation 成功后原子发布 Store。

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-database \
  --dataset 100style \
  --output data/processed/100style_pruned/feature_database \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --workers 8 \
  --overwrite
```

物理布局：

```text
feature_database/
  manifest.json
  index.npz
  motion/shard_00000.npy
  motion/shard_00001.npy
  ...
```

manifest 保存 data schema、相对 shard 路径、split manifest、feature schema、skeleton metadata 和 hash；index 保存 shard/range/split 数值索引及 normalization arrays。运行时使用 mmap 和 lazy shard cache，不会一次加载全部动作。

### 3.3 数据验证

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline validate-data \
  --feature-database data/processed/100style_pruned/feature_database \
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

- train 使用 epoch-aware 随机 frame crop，支持 `samples_per_epoch` 和 mirror probability；
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

## 5. 四类 FSQ

四类模型均由 [representation.py](/Users/shinn/Documents/Projects/StylizedMotionGeneration/stylized_motion/learning/representation.py) registry 构造。训练、验证、测试、checkpoint 和 decoder 不直接依赖具体模型类。

| family | coordinate layout |
| --- | --- |
| `flat_fsq` | `flat:40` |
| `part_fsq` | `global:6, sync:4, torso:6, left_leg:7, right_leg:7, left_arm:5, right_arm:5` |
| `residual_part_fsq` | `base:20, torso:6, left_leg:4, right_leg:4, left_arm:3, right_arm:3` |
| `latent_residual_fsq` | 与 residual part 相同，latent fusion 固定 `architecture_version=2` |

模型差异在 representation encoder、quantizer、decoder 和 representation-specific loss 内部；共同 loss 由统一层计算：feature reconstruction、temporal delta、root trajectory、FK joint、contact、foot slide、foot height。

## 6. 训练、验证、测试

四条主线共用 `RepresentationRunner`。每个训练 batch 执行一次 forward、一次 loss 计算和一次 backward/update；同一个 device batch 被 model 和 loss 复用。

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
```

配置边界是：`data` 描述 Store，`sampling` 描述窗口和采样，`loader` 描述 batch/worker/host memory，`training` 描述 optimizer、epoch、precision、clip、seed 和 output。默认输出目录是 `outputs/<family>_40x9/`。

### 6.2 验证和测试

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

representation-specific loss 由各模型的 `compute_representation_losses()` 提供并参与总 loss，但目前不单独写入 TensorBoard。

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
  --feature-database data/processed/100style_pruned/feature_database \
  --output data/processed/100style_pruned/flat_fsq_40x9 \
  --device cuda --chunk-size 1024 --save-codes
```

TokenStore 继承 FeatureStore 的 range、source clip、split、frame rate 和 feature schema，不重新生成 split。每个 token frame 保存 `[40]` 个 `uint8` index；可选 FSQ code 以 `float16` 保存。分块编码时会读取所需的 63 帧历史，保持 chunk 边界的 causal 语义。

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
  --token-database data/processed/100style_pruned/flat_fsq_40x9 \
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
  --feature-database data/processed/100style_pruned/feature_database \
  --range-idx 0 --start 0 --length 240 \
  --view compare --device cuda
```

`--view` 可选 `source`、`recon`、`compare`；`--dry-run` 只执行加载和重建检查；`--save-debug` 保存 source/recon features、tokens 和 metadata。

### 10.2 GenoView

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline genoview \
  --database data/processed/100style/motion_database.npz
```

也可以查看 230D features：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline genoview \
  --features outputs/sequence_debug/recon_features.npy \
  --stats-source data/processed/100style_pruned/feature_database
```

GenoView 支持播放、暂停、逐帧移动、速度调整、时间轴拖拽和相机交互。

### 10.3 实时 FSQ rollout

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline realtime \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --fsq-checkpoint outputs/flat_fsq_40x9/best.pt \
  --token-database data/processed/100style_pruned/flat_fsq_40x9 \
  --feature-database data/processed/100style_pruned/feature_database \
  --seed-source reencode --range-idx 0 --seed-frames 64 --device cuda
```

无窗口 smoke test：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline realtime \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --fsq-checkpoint outputs/flat_fsq_40x9/best.pt \
  --token-database data/processed/100style_pruned/flat_fsq_40x9 \
  --feature-database data/processed/100style_pruned/feature_database \
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

当前主训练路径是四条 FSQ representation。Generator 已有独立训练和生成路径，但尚未与 representation runner 共用 TensorBoard writer；legacy VQ-VAE 仅用于 baseline。data 层不负责 model、optimizer、loss、AMP、checkpoint 或 device transfer；learning 层不直接访问 shard 文件。

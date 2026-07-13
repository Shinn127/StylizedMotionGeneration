# Stylized Motion Generation

本项目训练离散运动表征模型，并在连续动作上评估重建质量与离散 code 的时序结构。当前主线是 frame-level FSQ motion tokenizer；仓库同时保留 VQ-VAE baseline。

## 环境与数据

项目默认使用 conda 环境 `mcc`：

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
```

原始数据通过软链接放入 `data/raw/`：

```bash
mkdir -p data/raw data/processed
ln -s /Users/shinn/Documents/DATASETS/100style data/raw/100style
ln -s /Users/shinn/Documents/DATASETS/lafan data/raw/lafan
```

期望的数据布局：

```text
data/raw/
├── 100style/
│   ├── Frame_Cuts.csv
│   └── <Style>/<Style>_<Clip>.bvh
└── lafan/
    └── *.bvh
```

## 完整流程

最短可运行流程使用 100STYLE 前 5 个 style 和 pruned skeleton：

```bash
# 1. 构建 database 与训练 features
python preprocess/build_data.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_test5_pruned

# 2. 训练 FSQ tokenizer
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml

# 3. 检查连续片段重建
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 \
  --start 128 \
  --length 256 \
  --view compare
```

## 数据处理

数据准备只有一个入口：[preprocess/build_data.py](preprocess/build_data.py)。它对每个 BVH 读取一次，并在同一遍预处理中生成可视化数据库和训练 feature database。

```text
BVH + Frame_Cuts.csv
  -> skeleton prune
  -> original + mirrored motion
  -> simulation root / velocity / angular velocity / foot contacts
     ├── disk-backed writer -> database.npz
     └── 230D feature shards -> train-only statistics -> mmap normalization
                              -> feature_database/
```

database 使用磁盘预分配数组顺序写入；并行预处理最多保留 `workers` 个待消费结果，因此内存不会随整个数据集线性增长。构建期间需要额外磁盘空间保存临时 database 数组，成功写出 `database.npz` 后会自动清理。

### 常用构建命令

指定 style：

```bash
python preprocess/build_data.py \
  --dataset 100style \
  --styles Neutral,Angry,Old \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_selected_pruned
```

完整 100STYLE：

```bash
python preprocess/build_data.py \
  --dataset 100style \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_pruned
```

LAFAN：

```bash
python preprocess/build_data.py \
  --dataset lafan \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/lafan_pruned
```

主要参数：

| 参数 | 说明 |
| --- | --- |
| `--styles` | 逗号分隔的 100STYLE style 名称 |
| `--max-styles` | 按 `Frame_Cuts.csv` 顺序选取前 N 个 style |
| `--prune-ends-and-fingers` | 删除 End joints 和手指链；当前 FSQ 配置需要此选项 |
| `--window-size` | train/val/test 窗口长度；必须与模型 receptive field 一致 |
| `--seed` | 窗口划分随机种子，默认 `3407` |
| `--workers` | BVH 并行处理进程数 |

### 输出结构

```text
data/processed/100style_test5_pruned/
├── database.npz
└── feature_database/
    ├── metadata.npz
    └── motion/
        ├── motion_00000.npy
        ├── motion_00001.npy
        └── ...
```

`database.npz` 保存 local position/rotation、linear/angular velocity、contact、skeleton、range 和 tag metadata，用于 Geno 与 trajectory 可视化。

`feature_database/metadata.npz` 保存：

- train/val/test window 索引；
- train frames 计算得到的 `offset/scale/dist`；
- feature loss weights；
- skeleton `names/parents/ref_pos`；
- motion shard 路径与 original/mirror metadata。

每个 `motion/*.npy` 是一个完整连续动作的 normalized features，Dataset 通过 mmap 按窗口读取，不会把所有 features 一次载入内存。

### 230D Motion Feature

pruned skeleton 包含 simulation root 和 24 个角色 joints：

```text
root local linear velocity           3
root local angular velocity          3
hips local position                  3
non-root joint rotations (6D)      144
hips local velocity                  3
non-root joint angular velocities   72
left/right toe contacts              2
total                              230
```

窗口在每个 clip 内按约 80/10/10 划分，original 与 mirror 使用相同 split。该划分用于 tokenizer reconstruction 开发；同一 clip 的不同窗口可能分布在多个 split，不能把它当作跨动作或跨 style 泛化协议。

## FSQ 训练

推荐配置是 [configs/fsq_pruned_frame_causal_cnn.yaml](configs/fsq_pruned_frame_causal_cnn.yaml)：

```text
normalized motion [B,T,230]
  -> frame causal CNN encoder
  -> FSQ codes [B,T,20], 9 levels per coordinate
  -> frame causal CNN decoder
  -> reconstructed motion [B,T,230]
```

启动训练：

```bash
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml
```

CLI 参数会覆盖 YAML 中的同名配置。例如使用另一份 feature database：

```bash
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml \
  --feature-database data/processed/100style_pruned/feature_database \
  --outdir outputs/fsq_100style_pruned \
  --run-name fsq_100style_pruned
```

从 `last.pt` 续训：

```bash
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml \
  --resume outputs/fsq_pruned_frame_causal_cnn_20x9/last.pt
```

训练自动选择 CUDA，否则使用 CPU。`--data-parallel` 仅在多 CUDA GPU 时启用。数据的 `window_size` 必须等于模型 receptive field，当前推荐配置为 64 帧。

### Loss 与监控

当前 FSQ objective 包含：

- weighted feature L1 与相邻帧 delta L1；
- integrated root position/rotation error；
- differentiable FK joint-position loss；
- toe-contact BCE；
- ground-truth-contact-gated foot sliding 与 foot-height loss。

训练同时记录 level perplexity/usage、tuple unique ratio、tuple change rate 和 coordinate change rate。

产物结构：

```text
outputs/fsq_pruned_frame_causal_cnn_20x9/
├── best.pt
├── last.pt
├── <run_name>.yaml
└── tensorboard/<run_name>/
```

查看 TensorBoard：

```bash
tensorboard \
  --logdir outputs/fsq_pruned_frame_causal_cnn_20x9/tensorboard
```

### VQ-VAE Baseline

```bash
python train_vqvae.py \
  --config configs/vqvae_pruned.yaml
```

其他 baseline 配置位于 `configs/vqvae_pruned_frame_causal_cnn.yaml` 和 `configs/vqvae_pruned_causal_transformer.yaml`。FSQ 与 VQ-VAE checkpoint 均可使用 `view_motion_sequence.py`。

## Checkpoint 评估

在共同的 feature database 上评估一个或多个 FSQ checkpoint：

```bash
python evaluate_fsq.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --split test \
  --device auto \
  --output outputs/evaluations/fsq_test5.json
```

重复传入 `--checkpoint` 可比较多个模型。评估报告包含 feature/delta error、MPJPE、root drift、contact precision/recall/F1、foot slide/height，以及 FSQ usage 和 temporal code statistics。评估器会先恢复物理 features，再使用各 checkpoint 自己的 normalization stats，因此可以比较训练统计不同但 skeleton 相同的模型。

## Dynamic FSQ Style Gate

冻结 FSQ token database，只训练 clip-dynamic coordinate-level mask 和 style classifier：

```bash
conda run -n mcc python train_fsq_style_gate.py \
  --config configs/fsq_style_gate.yaml
```

Gate 读取 `indices [B,T,20]`，输出 `mask [B,20,9]`。Mask 使用 Hard Concrete 自动学习激活数量，不使用预设 Top-K；训练沿用 token database 的 window splits，并同时训练 full-token 与 matched-random 性能基线。

评估并导出每个 window 的二值 mask、mask probability 和聚合结果：

```bash
conda run -n mcc python evaluate_fsq_style_gate.py \
  --checkpoint outputs/fsq_style_gate/best.pt \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --split test \
  --output outputs/evaluations/fsq_style_gate_test5.json
```

Gate checkpoint 会记录冻结 tokenizer 的 SHA256；评估时若 token database 来自不同 checkpoint，会直接拒绝运行。

## 可视化

### 原始数据库

直接打开预处理后的动作：

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz
```

需要未来轨迹显示时先构建 trajectory：

```bash
python preprocess/build_trajectory_inputs.py \
  --dataset 100style \
  --database data/processed/100style_test5_pruned/database.npz \
  --tags all \
  --future-frames 20,40,60 \
  --workers 8 \
  --output data/processed/100style_test5_pruned/trajectory.npz

python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --trajectory data/processed/100style_test5_pruned/trajectory.npz
```

### Source 与重建对比

`view_motion_sequence.py` 从完整 feature shard 截取连续片段，并自动加入 causal model 所需的左侧 context：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 \
  --start 128 \
  --length 256 \
  --view compare \
  --device auto
```

`--range-idx` 对应 `feature_database/metadata.npz` 中的 `motion_files/range_names/range_mirror` 索引。可视化模式：

| 参数 | 效果 |
| --- | --- |
| `--view source` | 只显示输入动作 |
| `--view recon` | 只显示模型重建 |
| `--view compare` | 并排显示 Source 与 Recon |
| `--dry-run` | 完成加载和推理检查，不打开窗口 |
| `--save-debug` | 保存 source/recon/indices 和 metadata 到 debug 目录 |

无显示环境下建议先运行：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 --start 128 --length 256 \
  --view compare --dry-run --save-debug
```

### 独立 Feature 文件

`Genoview.py` 也可以读取 `[T,D]` 的 `.npy/.npz` feature 文件。normalized features 必须提供训练 stats：

```bash
python Genoview.py \
  --features outputs/sequence_debug/recon_features.npy \
  --stats-source outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --normalized \
  --range-name recon
```

## FSQ Token Database

冻结训练好的 tokenizer，将完整 motion shards 编码为无重叠 token 序列：

```bash
python encode_fsq_database.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --output data/processed/100style_test5_pruned/fsq_20x9 \
  --chunk-size 1024 \
  --device auto \
  --save-codes
```

输出包含完整 `indices [T,20]`、可选 float16 codes、style/action/mirror 标签、split、checkpoint hash 和模型配置。重复写同一目录时必须显式添加 `--overwrite`。

## 验证

```bash
python -m pytest -q
```

测试覆盖 causal receptive field、FSQ codes/indices roundtrip、STE gradient、6D rotation/FK、kinematic losses，以及 database/feature pipeline。

## 代码结构

```text
configs/                         FSQ 与 VQ-VAE 训练配置
datasets/feature_dataset.py      mmap feature store 和 window Dataset
models/fsq.py                    FSQ tokenizer
models/vqvae.py                  VQ-VAE baselines
models/causal_cnn.py             causal encoder/decoder
models/losses.py                 reconstruction 与 kinematic losses
motion_features.py               230D schema、normalization、motion reconstruction
preprocess/build_data.py         唯一数据构建入口
preprocess/build_database.py     BVH motion processing 与 disk-backed database writer
preprocess/build_feature_database.py  split、feature stats 与 pipeline orchestration
preprocess/build_trajectory_inputs.py trajectory 构建
train_fsq.py                     FSQ 训练
evaluate_fsq.py                  checkpoint 定量评估
view_motion_sequence.py          连续片段重建可视化
Genoview.py                      database/feature viewer
encode_fsq_database.py           完整 shard token 编码
```

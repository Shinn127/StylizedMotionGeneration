# StylizedMotionGeneration

这是一个本地化的风格化动作生成研究项目。目前项目重点包括：ControlOperators 风格的 motion database、trajectory inputs、Geno 高质量可视化，以及基于 causal CNN 的 VQ-VAE 运动表征重建模型。

本仓库仅用于本地项目管理，暂时不绑定 remote。

## 环境

项目默认使用 conda 环境 `mcc`：

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
```

## 项目结构

```text
StylizedMotionGeneration/
├── data/                         # 原始数据软链接和处理后的数据，已 git-ignore
├── preprocess/                   # BVH 读取、quat 工具、database/trajectory 构建
├── resources/                    # 项目自管的 Geno shader/resources
├── configs/vqvae_pruned.yaml     # 当前 VQ-VAE 默认训练配置
├── models/                       # causal CNN、VQ-VAE、quantizer
├── motion_features.py            # 230D motion feature 打包与重建
├── train_vqvae.py                # VQ-VAE 训练入口
├── train_fsq.py                  # FSQ 训练入口
├── view_motion_sequence.py       # checkpoint 连续片段直接 Genoview 可视化入口
├── Visualization.py              # 轻量骨架/trajectory 可视化
├── Genoview.py                   # Geno 高质量渲染可视化
└── features_to_database.py       # 230D feature 到 database 的 roundtrip/export 工具
```

## 数据软链接

项目默认从 `data/raw/` 下读取本地数据软链接：

```bash
mkdir -p data/raw data/processed
ln -s /Users/shinn/Documents/DATASETS/100style data/raw/100style
ln -s /Users/shinn/Documents/DATASETS/lafan data/raw/lafan
```

也可以直接把数据复制到 `data/raw/`，但推荐使用软链接，避免在项目目录里重复存放大数据。

## 构建 Database

5 种 style 的小测试版本：

```bash
python preprocess/build_database.py \
  --dataset 100style \
  --max-styles 5 \
  --output data/processed/100style_test5
```

5 种 style 的 pruned 测试版本，也是当前 VQ-VAE 默认使用的数据格式：

```bash
python preprocess/build_database.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --output data/processed/100style_test5_pruned
```

全量 100STYLE pruned 版本：

```bash
python preprocess/build_database.py \
  --dataset 100style \
  --prune-ends-and-fingers \
  --workers 8 \
  --output data/processed/100style_pruned
```

`--prune-ends-and-fingers` 会剔除所有 `*End` 末端关节和手指链。当前 pruned skeleton 包含 25 个 joints，对应完整 motion feature 维度为 `230D`。

## 构建 Trajectory Inputs

Trajectory inputs 参考 ControlOperators 的输入格式，默认 future offsets 为 `[20, 40, 60]`。

```bash
python preprocess/build_trajectory_inputs.py \
  --dataset 100style \
  --database data/processed/100style_test5_pruned/database.npz \
  --tags all \
  --workers 8 \
  --output data/processed/100style_test5_pruned/trajectory.npz
```

生成的 `trajectory.npz` 包含：

- `indices`：源 pose 的 frame index
- `T`：展平后的 `[future_positions, future_directions]`
- `Tpos`：未来位置，位于当前 root local space
- `Tdir`：未来朝向，位于当前 root local space
- `future_frames`：用于构建 trajectory target 的未来帧偏移

## 可视化

轻量级骨架和 trajectory 可视化：

```bash
python Visualization.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --trajectory data/processed/100style_test5_pruned/trajectory.npz
```

Geno 高质量渲染可视化：

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --trajectory data/processed/100style_test5_pruned/trajectory.npz
```

控制方式：

- `Space`：暂停/继续播放
- `Left` / `Right`：存在 trajectory input 时切换 sample
- 默认以 `60 FPS` 播放

`Genoview.py` 支持 full database 和 pruned database。对于 pruned 输入，它会重建到完整 Geno skeleton 再渲染。如果本机需要 GLSL 4.10 shader header，可以执行：

```bash
python tools/patch_glsl_version.py
```

## Motion Representation

当前 pruned database 使用 ControlOperators 风格的 motion state：

```text
D_full_motion = 230
```

特征顺序如下：

```text
root local linear velocity          3
root local angular velocity         3
hips local position                 3
non-root joint rotations 6D         24 * 6 = 144
hips local velocity                 3
non-root joint angular velocities   24 * 3 = 72
foot contacts                       2
total                               230
```

Dataset 的 train/val/test 划分是在每个 clip 内进行的，固定比例为 `8/1/1`，默认 seed 为 `3407`。窗口之间不重叠。

归一化统计只使用 train split 计算，并会保存进 checkpoint，避免 val/test 信息泄漏。

## VQ-VAE 训练

使用默认 config 训练：

```bash
python train_vqvae.py --config configs/vqvae_pruned.yaml
```

当前训练入口直接读取离线 `feature_database/` 目录，不再在训练启动时从 `database.npz` 现场构建特征。

推荐显式指定 run name 和输出目录：

```bash
python train_vqvae.py \
  --config configs/vqvae_pruned.yaml \
  --run-name vqvae_pruned_test5 \
  --outdir outputs/vqvae_pruned
```

命令行参数会覆盖 config 中的同名配置：

```bash
python train_vqvae.py \
  --config configs/vqvae_pruned.yaml \
  --epochs 20 \
  --batch-size 16
```

当前默认训练参数：

```text
feature_database: data/processed/100style_test5_pruned/feature_database
epochs: 100
batch_size: 256
lr: 2e-4
min_lr: 1e-5
warmup_epochs: 2
scheduler: warmup + cosine decay
optimizer: AdamW
grad_clip_norm: 1.0
save_every: 0
seed: 3407
```

训练产物：

```text
outputs/vqvae/
├── best.pt
├── last.pt
├── <run_name>.yaml
└── tensorboard/<run_name>/
```

`save_every: 0` 表示关闭周期性 `epoch_XXXX.pt` checkpoint 保存。`last.pt` 和 `best.pt` 始终会更新。

## 构建训练表征 Database

训练前先直接基于原始 BVH 构建离线 `feature_database/` 目录：

```bash
python preprocess/build_feature_database.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --window-size 64 \
  --seed 3407 \
  --workers 8 \
  --output data/processed/100style_test5_pruned/feature_database
```

输出 `feature_database/` 包含：

- `motion`：归一化后的 `230D` full motion feature
- `train_windows` / `val_windows` / `test_windows`：固定 split 的窗口索引表
- `offset` / `scale` / `dist` / `weights` / `ref_pos`：train split 统计得到的 feature stats
- `names` / `parents` / `joint_subset` / `range_names` / `range_mirror`

### Root Condition 机制

`feature_database` 只保存完整 `230D` 特征，不再单独落盘 root condition。是否将前 `6D` 作为 root condition 拆给 decoder，是训练/模型层的选择，不属于 feature database schema。

参数说明：

- `--dataset 100style`：使用 100style 数据集
- `--max-styles 5`：只取前 5 个 style 做测试
- `--prune-ends-and-fingers`：使用 pruned skeleton
- `--window-size 64`：固定窗口长度
- `--seed 3407`：窗口划分随机种子
- `--workers 8`：并行 worker 数
- `--output ...`：feature database 输出目录

```text
完整 motion state: 230D
前 6D:             root local linear velocity + root local angular velocity
剩余 224D:         body/style features
```

当前无 root condition 的训练链路：

```text
encoder input: 230D full motion feature
decoder input: quantized latent
decoder output: 230D full motion reconstruction
loss target: 230D full motion feature
```

连续片段可视化链路：

```text
feature_database normalized 230D -> source/recon_230D -> Genoview side-by-side
```

## 连续片段推理与 Genoview 可视化

从 `feature_database` 中取连续 `L` 帧，用 FSQ 或 VQ-VAE checkpoint 重建，并直接启动 Genoview 横向对照。该入口不会默认导出 `.npy` 或 `clip_features`：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 \
  --start 128 \
  --length 256 \
  --context-left 63 \
  --view compare
```

同一个入口也适配 VQ-VAE checkpoint：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/vqvae_pruned_frame_causal_cnn/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 \
  --start 128 \
  --length 256 \
  --context-left 63 \
  --view compare
```

`--view` 支持 `source`、`recon`、`compare`。`compare` 模式左侧显示 database 原始片段，右侧显示模型重建片段；`--save-debug` 才会额外保存 source/recon features 和 token 文件。

## Feature Roundtrip 工具

使用 database 自身的 motion features 验证 `230D -> database.npz` 重建链路：

```bash
python features_to_database.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --output data/processed/100style_test5_pruned_roundtrip/database.npz \
  --use-database-motion
```

导出外部模型预测的 features：

```bash
python features_to_database.py \
  --features path/to/features.npy \
  --database data/processed/100style_test5_pruned/database.npz \
  --output data/processed/custom_recon/database.npz
```

## 本地 Git 管理

本仓库用于本地项目管理：

```bash
git status
git add -A
git commit -m "describe local change"
```

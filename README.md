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
├── infer_vqvae.py                # checkpoint 推理与 database 导出入口
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
epochs: 100
batch_size: 32
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

### Root Condition 机制

默认配置 `use_root_cond: true` 时，模型不会直接重建完整 `230D` 向量。

```text
完整 motion state: 230D
root condition:    6D   = root local linear velocity + root local angular velocity
模型重建目标:       224D = 剩余 body/style features
```

训练链路：

```text
encoder input: 224D body/style motion
decoder input: quantized latent + 6D root_cond
decoder output: 224D body/style reconstruction
loss target: 224D body/style motion
```

推理/导出链路：

```text
recon_224D + root_cond_6D -> pack 回 230D -> reconstructed database.npz
```

如果使用 `--no-use-root-cond`，模型会回到直接重建完整 `230D` feature vector。

## VQ-VAE 推理与导出

从 checkpoint 导出重建后的 database：

```bash
python infer_vqvae.py \
  --checkpoint outputs/vqvae_pruned/best.pt \
  --split test \
  --outdir outputs/vqvae_pruned/infer \
  --tag test_recon \
  --export-trajectory
```

输出目录：

```text
outputs/vqvae_pruned/infer/test_recon/
├── database.npz
├── trajectory.npz
└── recon_windows.npz
```

可视化重建结果：

```bash
python Genoview.py \
  --database outputs/vqvae_pruned/infer/test_recon/database.npz \
  --trajectory outputs/vqvae_pruned/infer/test_recon/trajectory.npz
```

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

如果输入 feature 已经是当前项目统计下的 normalized features，需要加上 `--normalized`。

## 本地 Git 管理

本仓库用于本地项目管理：

```bash
git status
git add -A
git commit -m "describe local change"
```


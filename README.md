# Stylized Motion Generation

本项目研究风格化角色运动的离散表征与可控生成。当前主线是一个 frame-level FSQ motion tokenizer：先从 style-unlabeled 或基础运动中学习可复用的离散运动坐标，再研究风格如何通过这些坐标的时空组合得到表达。

当前阶段的目标是训练并评估基础 tokenizer，不包含 style composer 或 style-transfer generator。

## Quick Start

项目默认使用 conda 环境 `mcc`：

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
```

建立原始数据软链接：

```bash
mkdir -p data/raw data/processed
ln -s /Users/shinn/Documents/DATASETS/100style data/raw/100style
ln -s /Users/shinn/Documents/DATASETS/lafan data/raw/lafan
```

构建 5-style 测试 feature database：

```bash
python preprocess/build_feature_database.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_test5_pruned/feature_database
```

训练当前 FSQ tokenizer：

```bash
python train_fsq.py --config configs/fsq_pruned_frame_causal_cnn.yaml
```

## Current FSQ Tokenizer

当前推荐配置位于 [configs/fsq_pruned_frame_causal_cnn.yaml](/Users/shinn/Documents/Projects/StylizedMotionGeneration/configs/fsq_pruned_frame_causal_cnn.yaml)。它使用 causal CNN、frame-level 量化和 64 帧感受野：

```text
motion [B,T,230]
  -> causal encoder [B,128,T]
  -> FSQ coordinates [B,T,20], each with 9 levels
  -> causal decoder
  -> reconstruction [B,T,230]
```

`20 × 9-level` 的理论容量是约 `63.4 bits/frame`，在 60 FPS 下约为 `3.8 kbit/s`。模型同时输出：

- `fsq_codes [B,T,20]`：带 straight-through gradient 的量化坐标，用于后续 pattern/composer 研究。
- `indices [B,T,20]`：每个坐标的整数 level index，值域 `0..8`，用于存储、统计和离散生成。
- `recon_state [B,T,230]`：重建后的 normalized motion features。

`FSQMotionAutoencoder` 还提供：

```python
codes, indices = model.encode_to_codes(motion)
recon = model.decode_from_codes(codes)
recon = model.decode_from_indices(indices)
```

新训练产物默认写入：

```text
outputs/fsq_pruned_frame_causal_cnn_20x9/
├── best.pt
├── last.pt
├── fsq_pruned_frame_causal_cnn_20x9_test5.yaml
└── tensorboard/fsq_pruned_frame_causal_cnn_20x9_test5/
```

## Motion Features

pruned skeleton 有 25 个 joints，motion state 为 `230D`：

```text
root local linear velocity          3
root local angular velocity         3
hips local position                 3
non-root joint rotations (6D)       24 × 6 = 144
hips local velocity                 3
non-root joint angular velocities   24 × 3 = 72
left/right toe contacts             2
total                               230
```

feature database 会保存 normalized motion windows、train-only normalization stats、skeleton `names/parents/ref_pos` 和窗口索引。模型 checkpoint 同样保存这些 stats，以便重建与可视化使用一致的特征空间。

## Reconstruction Objective

FSQ tokenizer 优化 feature fidelity 与运动学质量：

```text
feature reconstruction: weighted feature L1
temporal consistency:   adjacent-frame delta L1
root trajectory:        integrated root position and rotation error
joint quality:          root-relative differentiable FK joint-position loss
contact quality:        toe-contact BCE
foot quality:           target-contact-gated horizontal foot sliding
                         and target-relative foot-height error
```

joint 和 foot losses 在反归一化后的物理单位中计算。6D joint rotation 会先正交化为 rotation matrix，再通过 skeleton hierarchy 做可微 forward kinematics。foot sliding 只在 ground-truth contact 的相邻帧激活，避免模型通过关闭预测 contact 逃避约束。

当前 loss 权重可直接在 YAML 中调整。首次训练建议先观察 TensorBoard 中各项未收敛前的数值量级，避免 foot loss 压制上身动作重建。

## Monitoring

训练会记录重建、root、joint、contact 和 foot losses，并记录 FSQ 表征统计：

```text
level perplexity / usage
per-coordinate min / max perplexity and usage
tuple unique ratio
tuple change rate
coordinate change rate
```

这些指标用于判断 20 个坐标是否被充分使用，以及相邻帧的离散组合是否存在稳定的时序 pattern。高重建质量本身不等价于可复用的运动基元。

## Data Preparation

如果还需要用于 Geno 或 trajectory 可视化的完整 database，可单独构建：

```bash
python preprocess/build_database.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --output data/processed/100style_test5_pruned
```

trajectory inputs：

```bash
python preprocess/build_trajectory_inputs.py \
  --dataset 100style \
  --database data/processed/100style_test5_pruned/database.npz \
  --tags all \
  --workers 8 \
  --output data/processed/100style_test5_pruned/trajectory.npz
```

当前 `feature_database` 训练窗口在每个 clip 内划分为 train/val/test。它适合 tokenizer reconstruction 的开发验证，但同一原始动作序列会出现在不同 split 中；不要将该 split 上的 style classifier 结果视为跨动作泛化证据。正式 style/composer 评估应按完整 action 或 clip 留出。

## Reconstruction Visualization

从 feature database 中取得连续片段，用 checkpoint 重建并在 Genoview 中比较：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 \
  --start 128 \
  --length 256 \
  --view compare
```

可选项：

```text
--view source|recon|compare
--dry-run                  # 只做 checkpoint/reconstruction 检查
--save-debug               # 保存 source、recon、FSQ indices 和 metadata
--device auto|cuda|mps|cpu
```

Geno 原始 database 可直接使用：

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --trajectory data/processed/100style_test5_pruned/trajectory.npz
```

## VQ-VAE Baselines

仓库仍保留 VQ-VAE baseline：

```bash
python train_vqvae.py --config configs/vqvae_pruned.yaml
```

FSQ 与 VQ-VAE 可以通过同一个 `view_motion_sequence.py` 入口可视化。新实验应使用不同 `outdir`，避免覆盖已有 checkpoint。

## Verification

运行单元测试：

```bash
pytest -q
```

测试覆盖 causal receptive field、FSQ indices/codes roundtrip、STE gradient、6D rotation/FK、joint loss 和 contact-gated foot losses。

## Checkpoint Evaluation

使用共同的 feature database 对一个或多个 FSQ checkpoint 进行评估：

```bash
python evaluate_fsq.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9_full_loss/best.pt \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9_none_loss/best.pt \
  --feature-database data/processed/100style_test20_pruned/feature_database \
  --split test \
  --device cuda \
  --output outputs/evaluations/fsq_20x9_loss_ablation.json
```

评估器将 feature database 先恢复到原始物理特征，再按每个 checkpoint 自带的 normalization stats 重归一化执行推理。报告的共同指标包括 weighted feature/delta error、MPJPE、root drift、contact precision/recall/F1、foot slide、foot height，以及 FSQ usage 和 temporal code statistics。因此即使 checkpoint 使用不同训练统计，也可以在同一个评估集合上比较。

## FSQ Token Database

冻结训练好的 tokenizer，将完整连续 motion shards 编码为无重叠 FSQ 序列：

```bash
python encode_fsq_database.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9_full_loss/best.pt \
  --feature-database data/processed/100style_test20_pruned/feature_database \
  --output data/processed/100style_test20_pruned/fsq_20x9_full_loss \
  --chunk-size 1024 \
  --device cuda \
  --save-codes
```

编码器会为每个 chunk 自动补充模型要求的左侧 causal context，只保存目标帧对应的输出。token database 包含完整 `indices [T,20]`、可选的 float FSQ codes、style/action/mirror 标签、原始窗口 split、checkpoint hash 和模型配置。

## FSQ Pattern Probes

Pattern suite 使用相同的 standard split 与 held-out-action 协议，分别检验：level occupancy、转移方向、多帧 n-gram、持续时间、pattern 位置、频谱周期，以及 coordinate 间的同步和滞后协调。

```bash
python analyze_fsq_patterns.py \
  --token-database data/processed/100style_test20_pruned/fsq_20x9_full_loss \
  --representations histogram transition ngram3 ngram4 run_length \
    position_histogram spectrum coordinate_covariance coordinate_cooccurrence \
    lagged_coordination histogram_ngram3 histogram_run_length histogram_position \
    histogram_spectrum histogram_cooccurrence histogram_coordination \
  --output outputs/evaluations/fsq_20x9_pattern_probes.json
```

用反转、逐帧打乱和 block shuffle 做顺序消融：

```bash
python analyze_fsq_patterns.py \
  --token-database data/processed/100style_test20_pruned/fsq_20x9_full_loss \
  --representations transition reversed_transition shuffled_transition block_shuffled_transition \
  --block-size 8 \
  --output outputs/evaluations/fsq_20x9_order_ablations.json
```

超过 tokenizer 训练窗口的结构需要从完整 shard 重新切窗。此模式为避免重叠窗口造成 train/test 泄漏，只报告 held-out-action：

```bash
python analyze_fsq_patterns.py \
  --token-database data/processed/100style_test20_pruned/fsq_20x9_full_loss \
  --window-size 256 --window-stride 128 \
  --representations position_histogram spectrum ngram4 lagged_coordination \
  --output outputs/evaluations/fsq_20x9_long_256.json
```

非线性关系用完全相同的 `raw_sequence` 输入分别运行 linear 与 MLP probe，并比较两份报告。MLP 的 held-out-action 增益才可视为固定窗口内的非线性时序证据：

```bash
python analyze_fsq_patterns.py \
  --token-database data/processed/100style_test20_pruned/fsq_20x9_full_loss \
  --representations raw_sequence --probe-model mlp --max-iter 200 \
  --output outputs/evaluations/fsq_20x9_raw_sequence_mlp.json
```

`histogram_*` 是关键的 nested comparison：它回答对应 pattern 特征在 level occupancy 已知后是否仍有增量，而不是比较两个维度、容量都不同的独立分类器。`coordinate_covariance`、`coordinate_cooccurrence` 与 `lagged_coordination` 描述的是潜在 FSQ coordinate 的协同，不能直接称为身体部位协调。要得到身体部位相位差，需要在原始关节运动上定义身体部位和步态事件，或先完成 coordinate-to-joint attribution。

## Repository Map

```text
configs/                    training configurations
datasets/                   feature store and window dataset
models/fsq.py               FSQ tokenizer and raw-code API
models/losses.py            reconstruction and differentiable kinematic losses
models/causal_cnn.py        causal encoder/decoder backbones
motion_features.py          230D feature schema and NumPy reconstruction
preprocess/                 BVH, database, feature database and trajectory tools
train_fsq.py                FSQ training entry point
train_vqvae.py              VQ-VAE baseline entry point
view_motion_sequence.py     continuous reconstruction visualization
tests/                      model and loss regression tests
```

## Next Research Step

在 FSQ tokenizer 具备稳定重建和可解释 code statistics 后，下一步是建模 code sequence 的选择、共现、持续时间和转移规律，并在 style data 上学习轻量的 style-conditioned composer。该阶段将直接检验“motion style 是基础运动 pattern 的特定时空表达”这一核心假设。

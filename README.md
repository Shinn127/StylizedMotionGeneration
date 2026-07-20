# Stylized Motion Generation

本项目研究基于离散运动表征的风格化角色动作生成。当前主线由两个阶段组成：

1. 训练 frame-level FSQ motion tokenizer，把每帧 230D motion feature 编码为 20 个 9-level 离散坐标；
2. 冻结 FSQ tokenizer/decoder，在离散 token 空间训练 causal Transformer generator，并通过 style 与 future trajectory 实现实时控制。

仓库同时保留 VQ-VAE baseline、FSQ style-information probe 和 dynamic style gate，均属于辅助实验，不是当前 generator 主线。

## 当前主线

```text
BVH motion
  │
  ├─ preprocess/build_data.py
  │    ├─ database.npz                 可视化与 trajectory 来源
  │    └─ feature_database/            normalized 230D motion shards
  │
  ├─ frozen FSQ tokenizer
  │    └─ token database               indices [T,20], levels 0...8
  │
  ├─ future trajectory database        18D root-local control
  │
  └─ conditional causal Transformer
       ├─ token embedding
       ├─ trajectory embedding
       ├─ block-level causal dynamic FiLM style conditioning
       └─ next-frame logits [B,T,20,9]
             │
             └─ frozen FSQ decoder → motion features → articulated pose
```

Generator 训练期间不加载 FSQ encoder/decoder，也不回传到 tokenizer；它直接读取提前编码好的离散 token。条件 generator 的完整 Transformer 从零训练，不继承无条件 generator 权重。

## 环境

项目默认使用 conda 环境 `mcc`：

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
pip install -r requirements.txt
```

自动设备选择顺序为 CUDA、MPS、CPU。部分训练入口支持多 CUDA GPU 的 `DataParallel`，generator 默认使用单设备。

## 数据布局

原始数据建议通过软链接放入 `data/raw/`：

```bash
mkdir -p data/raw data/processed
ln -s /Users/shinn/Documents/DATASETS/100style data/raw/100style
ln -s /Users/shinn/Documents/DATASETS/lafan data/raw/lafan
```

```text
data/raw/
├── 100style/
│   ├── Frame_Cuts.csv
│   └── <Style>/<Style>_<Clip>.bvh
└── lafan/
    └── *.bvh
```

## 最小端到端流程

下面使用 100STYLE 前 5 个 style 和 pruned skeleton。完整数据训练时应重新构建 `100style_pruned`，并通过 CLI 或 YAML 将 tokenizer、token database 和 trajectory database 全部切换到完整数据路径；不要只修改输出目录。

### 1. 构建 motion database

```bash
python preprocess/build_data.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_test5_pruned
```

输出：

```text
data/processed/100style_test5_pruned/
├── database.npz
└── feature_database/
    ├── metadata.npz
    └── motion/motion_*.npy
```

### 2. 训练 FSQ tokenizer

```bash
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml
```

默认 checkpoint：

```text
outputs/fsq_pruned_frame_causal_cnn_20x9/
├── best.pt
├── last.pt
├── <run_name>.yaml
└── tensorboard/<run_name>/
```

评估与可视化：

```bash
python evaluate_fsq.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --split test \
  --device auto \
  --output outputs/evaluations/fsq_test5.json

python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 --start 128 --length 256 \
  --view compare --device auto
```

### 3. 编码完整 FSQ token database

```bash
python encode_fsq_database.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --output data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --chunk-size 1024 \
  --device auto \
  --save-codes
```

Token database 保存：

- 每个 motion shard 的 `indices [T,20]`；
- 可选 float16 FSQ codes；
- style、action、mirror 和 range metadata；
- train/val/test windows；
- FSQ checkpoint 路径、配置和 SHA256。

重复写入同一路径时必须显式添加 `--overwrite`。

### 4. 构建 trajectory database

```bash
python preprocess/build_trajectory_inputs.py \
  --dataset 100style \
  --database data/processed/100style_test5_pruned/database.npz \
  --tags all \
  --future-frames 20,40,60 \
  --workers 8 \
  --output data/processed/100style_test5_pruned/trajectory.npz

python preprocess/build_fsq_trajectory_database.py \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --trajectory-input data/processed/100style_test5_pruned/trajectory.npz \
  --output data/processed/100style_test5_pruned/fsq_20x9_full_loss_trajectory_20_40_60
```

每帧 trajectory 是 18D root-local future control：

```text
[pos(+20), pos(+40), pos(+60), dir(+20), dir(+40), dir(+60)]
```

每个 position/direction 都是 xyz。无有效 future control 的帧保存为零向量并令 `valid=false`。对齐后的 trajectory database 必须与 token database 使用完全相同的 shard 顺序、帧数和 tokenizer SHA。

### 5. 训练 conditional generator

默认配置指向完整 100STYLE。使用 test5 数据时通过 CLI 覆盖路径：

```bash
python train_fsq_conditional_generator.py \
  --config configs/fsq_generator_conditional.yaml \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --trajectory-database data/processed/100style_test5_pruned/fsq_20x9_full_loss_trajectory_20_40_60 \
  --outdir outputs/fsq_generator_conditional_dynamic_film_test5
```

快速 smoke test：

```bash
python train_fsq_conditional_generator.py \
  --config configs/fsq_generator_conditional.yaml \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --trajectory-database data/processed/100style_test5_pruned/fsq_20x9_full_loss_trajectory_20_40_60 \
  --outdir outputs/fsq_generator_conditional_smoke \
  --max-samples 512 --epochs 1 --num-workers 0
```

### 6. 检查条件是否被模型使用

```bash
python evaluate_fsq_conditional_generator.py \
  --checkpoint outputs/fsq_generator_conditional_dynamic_film_test5/best.pt \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --trajectory-database data/processed/100style_test5_pruned/fsq_20x9_full_loss_trajectory_20_40_60 \
  --split test \
  --output outputs/evaluations/fsq_conditional_test5.json
```

评估报告并列：

- 正确 style 与 trajectory；
- 循环错置的 style；
- zero/invalid trajectory；
- batch 内打乱的 trajectory。

条件 ablation 的 NLL 差异用于判断模型是否真正依赖控制分支。

## 数据与时间对齐

### 230D motion feature

Pruned skeleton 包含 simulation root 和 24 个角色 joints：

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

`feature_database/metadata.npz` 保存 train-only normalization stats、loss weights、skeleton metadata、motion shard 路径和 split windows。Motion shard 通过 mmap 读取。

窗口在 clip 内近似按 80/10/10 划分，original 与 mirror 使用相同 split。当前划分服务于 reconstruction 和 generator 开发，不应视为严格的跨动作或跨 style 泛化协议。

### Next-token control alignment

对一个长度为 `W` 的 token window：

```text
indices:    x0, x1, ..., x(W-1)       [W,20]
inputs:     x0, x1, ..., x(W-2)       [W-1,20]
targets:    x1, x2, ..., x(W-1)       [W-1,20]
trajectory: c1, c2, ..., c(W-1)       [W-1,18]
```

`ct` 表示目标帧 `xt` 自身坐标系下的 future trajectory。因此模型位置 `t` 的关系是：

```text
input xt + control c(t+1) + style → predict x(t+1)
```

Trajectory normalization 只从 train windows 覆盖的有效目标帧拟合。当前训练不做 trajectory dropout。

## FSQ tokenizer

推荐配置：[configs/fsq_pruned_frame_causal_cnn.yaml](configs/fsq_pruned_frame_causal_cnn.yaml)。

```text
normalized motion [B,T,230]
  → frame-causal CNN encoder
  → FSQ indices [B,T,20], each in 0...8
  → dequantization
  → frame-causal CNN decoder
  → reconstructed motion [B,T,230]
```

Encoder/decoder receptive field 为 64 帧，只依赖当前帧及左侧历史，没有 lookahead。当前 objective 包含：

- weighted feature L1 与 adjacent-frame delta L1；
- integrated root position/rotation error；
- differentiable FK joint-position loss；
- toe-contact BCE；
- ground-truth-contact-gated foot sliding 与 foot-height loss。

训练同时记录 level perplexity/usage、tuple unique ratio、tuple change rate 和 coordinate change rate。

## Generator 实现

核心实现位于 [models/fsq_generator.py](models/fsq_generator.py)。无条件和条件 generator 共享同一个 causal Transformer 主体。

### Token embedding

每个 FSQ coordinate 使用独立的 level vocabulary 区间：

```text
indices [B,T,20]
  → coordinate-aware level embedding [B,T,20,16]
  → concatenate [B,T,320]
  → Linear + RMSNorm
  → frame hidden [B,T,256]
```

Transformer 最终一次性输出 20 个 9-way categorical distributions；同一帧内的 coordinate 不进行额外自回归。

### Trajectory conditioning

```text
[trajectory_18D, valid_flag]
  → Linear(19,128)
  → SiLU
  → Linear(128,256)
  → RMSNorm
  → add to frame hidden
```

### Causal dynamic block FiLM

当前 style 架构标识为：

```text
causal_dynamic_block_film_v1
```

`style_id` 先映射为 128D embedding。在每个 Transformer block 中，attention 和 FFN 前各有一套独立 Dynamic FiLM：

```text
u = RMSNorm(h)
condition = RMSNorm(u + style_projection(style_embedding))
gamma, beta = MLP(condition)
u_style = u * (1 + 0.5 * tanh(gamma)) + beta
```

FiLM 参数 shape 为 `[B,T,256]`，因此即使一个 training clip 使用同一个 style，调制仍会随当前 causal hidden state 逐帧变化。

默认条件模型配置：

| 项目 | 值 |
| --- | ---: |
| context | 64 frames |
| model dim | 256 |
| Transformer blocks | 6 |
| query / KV heads | 8 / 4 |
| FFN dim | 768 |
| style embedding | 128 |
| trajectory hidden | 128 |
| optimizer | AdamW |
| initial LR | 3e-4 |

### Output 与 loss

```text
hidden [B,T,256]
  → RMSNorm
  → Linear(256,20×9)
  → logits [B,T,20,9]
```

训练 loss 是所有时间位置、所有 FSQ coordinate 上的 cross entropy。完整条件 Transformer、style embedding、FiLM 和 trajectory encoder 都从零训练。

## KV cache 与实时切换

每个 Transformer block 保存最近 64 个输入帧的 K/V；`next_position` 持续增长，因此 RoPE 仍使用绝对位置。cache 不包含 style prefix，`prefix_length=0`。

正常生成步骤：

```text
cache 已处理到 xt，并持有预测 x(t+1) 的 logits
  → sample x(t+1)
  → 取 control c(t+2)
  → decode_step(x(t+1), c(t+2), current_style)
  → append 新 K/V，超出 64 帧时丢弃最旧 K/V
  → 得到预测 x(t+2) 的 logits
```

Checkpoint 中的 style cache policy 为 `append_only`：历史 K/V 保留其生成时的 style。实时切换 style 时，为了立即影响下一次采样，controller 只做一次局部 replay：

1. 从 cache 移除最新输入 token 的 K/V；
2. 保留更早的历史 K/V；
3. 用新 style 和同一 trajectory 重放最新输入 token；
4. 更新最新 K/V 与 staged next-token logits。

因此不会重建整个 64-frame cache；更早的旧 style K/V 会在后续生成中自然滑出窗口。

Trajectory command 更新目前会替换最新 control 并重新 prefill rolling history，以保证新的 command 立即进入下一次采样。

## Generator 训练与评估

### 无条件 generator

```bash
python train_fsq_generator.py \
  --config configs/fsq_generator.yaml
```

评估 teacher-forced NLL、rollout、decoded motion 和增量推理延迟：

```bash
python evaluate_fsq_generator.py \
  --checkpoint outputs/fsq_generator/best.pt \
  --token-database data/processed/100style_pruned/fsq_20x9_full_loss \
  --fsq-checkpoint <MATCHING_FSQ_CHECKPOINT> \
  --split test \
  --output outputs/evaluations/fsq_generator_test.json
```

离线生成无条件 continuation：

```bash
python generate_fsq_motion.py \
  --checkpoint outputs/fsq_generator/best.pt \
  --token-database data/processed/100style_pruned/fsq_20x9_full_loss \
  --fsq-checkpoint <MATCHING_FSQ_CHECKPOINT> \
  --range-idx 0 --start 128 \
  --seed-frames 64 --generate-frames 120 \
  --sample --temperature 0.8 \
  --output-dir outputs/generated/fsq_unconditional
```

### 条件 generator checkpoint 约束

加载条件 checkpoint 时会校验：

- `model_family == fsq_conditional_generator`；
- `style_conditioning == causal_dynamic_block_film_v1`；
- `style_cache_policy == append_only`；
- checkpoint、token database、trajectory database 的 tokenizer SHA 一致；
- style vocabulary、FSQ coordinates/levels 和 shard layout 一致。

旧 style-prefix、输入层静态 FiLM 或缺少 cache-policy metadata 的 checkpoint 不兼容当前实现。

## 实时控制

`realtime_fsq_controller.py` 同时支持无条件与当前条件 generator checkpoint。

Headless dry run：

```bash
python realtime_fsq_controller.py \
  --generator-checkpoint outputs/fsq_generator_conditional_dynamic_film_test5/best.pt \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --fsq-checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --trajectory-database data/processed/100style_test5_pruned/fsq_20x9_full_loss_trajectory_20_40_60 \
  --style-id 0 \
  --range-idx 0 --start 128 --seed-frames 64 \
  --dry-run --dry-run-frames 120
```

交互模式可省略 `--dry-run`。键盘控制：

| 键位 | 功能 |
| --- | --- |
| `W/S` | 前进 / 后退 |
| `A/D` | 左右侧移 |
| `Q/E` | 左右转向 |
| `J/K` | 上一个 / 下一个 style |
| `Space` | 暂停 / 继续 |
| `R` | 重置 |

建议实时生成使用 `--sample --temperature 0.8`，并通过 `--move-speed`、`--turn-speed` 调整键盘 trajectory。

程序化接口：

```python
controller.set_style(style_id)
controller.set_trajectory_control(raw_18d, valid=True)
controller.set_trajectory_control(None)  # 恢复 reference trajectory
```

外部 trajectory 输入使用未归一化的 18D root-local layout；controller 使用 checkpoint 中保存的 train normalization 自动归一化。

## Token 到动作

生成新 token 后，controller 使用最近最多 64 帧 token：

```text
FSQ indices
  → frozen FSQ dequantization
  → frozen causal CNN decoder
  → normalized 230D feature
  → denormalization / root integration / FK
  → local joint positions and rotations
```

FSQ decoder 是 causal model，因此 rolling 64-frame token history 足以恢复当前最后一帧 feature。

## 可视化工具

打开原始 motion database：

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz
```

显示 future trajectory：

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --trajectory data/processed/100style_test5_pruned/trajectory.npz
```

查看 tokenizer 连续片段重建：

```bash
python view_motion_sequence.py \
  --checkpoint outputs/fsq_pruned_frame_causal_cnn_20x9/best.pt \
  --feature-database data/processed/100style_test5_pruned/feature_database \
  --range-idx 0 --start 128 --length 256 \
  --view compare
```

无显示环境可添加 `--dry-run --save-debug`。`Genoview.py` 也可以读取独立 `[T,D]` `.npy/.npz` feature 文件；normalized feature 必须同时提供 `--stats-source`。

## 辅助实验

### Dynamic FSQ Style Gate

```bash
python train_fsq_style_gate.py \
  --config configs/fsq_style_gate.yaml

python evaluate_fsq_style_gate.py \
  --checkpoint outputs/fsq_style_gate/best.pt \
  --token-database data/processed/100style_test5_pruned/fsq_20x9_full_loss \
  --split test \
  --output outputs/evaluations/fsq_style_gate_test5.json
```

Gate 在冻结 token 上学习 coordinate-level Hard Concrete mask，并与 full-token、matched-random baseline 比较。

### Style information probes

`analyze_fsq_patterns.py` 使用 histogram、transition、n-gram、run length、spectrum、coordinate co-occurrence 等表示，分析 style 信息来自 level occupancy 还是 temporal structure。

### VQ-VAE baselines

```bash
python train_vqvae.py --config configs/vqvae_pruned.yaml
```

其他配置：

- `configs/vqvae_pruned_frame_causal_cnn.yaml`
- `configs/vqvae_pruned_causal_transformer.yaml`

## 验证

```bash
python -m pytest -q
```

测试覆盖：

- causal CNN receptive field；
- FSQ codes/indices roundtrip 与 STE gradient；
- motion feature、6D rotation、FK 和 kinematic losses；
- preprocessing 与 mmap database；
- unconditional/conditional generator cache/full-forward 一致性；
- dynamic FiLM、style-switch replay 和 bounded KV cache；
- realtime keyboard trajectory mapping。

## 代码结构

```text
configs/
  fsq_pruned_frame_causal_cnn.yaml       FSQ tokenizer 主配置
  fsq_generator.yaml                     无条件 generator
  fsq_generator_conditional.yaml         style + trajectory 条件 generator
  fsq_style_gate.yaml                    dynamic style gate

models/
  fsq.py                                 FSQ encoder/quantizer/decoder
  fsq_generator.py                       causal Transformer、FiLM、KV cache
  fsq_style_gate.py                      coordinate-level style gate
  causal_cnn.py                          frame-causal convolution modules
  losses.py                              reconstruction 与 kinematic losses

datasets/
  feature_dataset.py                     mmap motion feature windows
  fsq_token_dataset.py                   frozen FSQ token windows
  fsq_trajectory_dataset.py              shifted next-token trajectory controls

preprocess/
  build_data.py                          BVH → database + feature database
  build_database.py                      motion processing 与 disk-backed writer
  build_feature_database.py              split、stats 与 pipeline orchestration
  build_trajectory_inputs.py             raw future trajectory
  build_fsq_trajectory_database.py       trajectory → FSQ shard alignment

train_fsq.py                             FSQ tokenizer 训练
encode_fsq_database.py                   完整 motion shard tokenization
train_fsq_generator.py                   无条件 generator 训练
train_fsq_conditional_generator.py       当前条件 generator 训练
evaluate_fsq.py                          tokenizer 定量评估
evaluate_fsq_generator.py                无条件 generator 评估
evaluate_fsq_conditional_generator.py    条件 ablation
generate_fsq_motion.py                   无条件离线 continuation
realtime_fsq_controller.py               实时 token rollout 与 GenoView
view_motion_sequence.py                  source/reconstruction 对比
Genoview.py                              database、feature 与 realtime viewer
```

## 实验注意事项

- Token database 与 generator checkpoint 强绑定 FSQ checkpoint SHA；不要混用不同 tokenizer 产生的 token。
- Trajectory database 只保存 raw values；normalization 由条件训练脚本从 train targets 拟合并写入 checkpoint。
- `window_size`、FSQ causal receptive field 和 generator context 应保持一致；当前主线使用 64 帧。
- 输出目录命名不是兼容性依据，应以 checkpoint 内的 `model_family`、model config、style metadata 和 tokenizer SHA 为准。
- 工作树中的旧 checkpoint 可能来自已淘汰架构；当前 loader 会主动拒绝不兼容的 conditional checkpoint。

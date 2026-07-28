# Stylized Motion Generation

面向风格化角色动作的离散 motion tokenizer 研究项目。当前工作的主线是 **Hierarchical Part-FSQ**：在不做时间下采样、不使用图消息传递的前提下，将 230D frame-level motion feature 编码为按身体部位组织的 40 个 FSQ 坐标，并以 9 个离散 level 量化每个坐标。

当前阶段只落地 tokenizer 的训练和定量评估。Generator、token database、实时控制和可视化工具仍是旧的 Flat-FSQ 路线，尚未接入 Part-FSQ，不能和 Part-FSQ checkpoint 混用。

## 当前状态

| 模块 | 状态 | 入口 |
| --- | --- | --- |
| Hierarchical Part-FSQ 训练 | 当前主线 | `train_part_fsq.py` |
| Part-FSQ 断点续训 | 支持 | `--resume <checkpoint>` |
| Part-FSQ 定量评估 | 支持 | `evaluate_fsq.py` |
| Flat-FSQ tokenizer | 保留的基线 | `train_fsq.py` |
| FSQ token database / generator / realtime | 仅 Flat-FSQ | 见“未接入功能” |
| GraphFSQ | 已放弃 | 不在代码路径中 |

## 环境

项目默认使用 conda 环境 `mcc`：

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
pip install -r requirements.txt
```

训练设备自动按 CUDA、MPS、CPU 选择。Part-FSQ 训练固定使用 **FP32**：没有 AMP、`autocast` 或 `GradScaler` 路径。CUDA 多卡可通过 `--data-parallel` 启用 PyTorch `DataParallel`。

## 数据

### 原始 BVH 布局

原始数据可通过软链接放入 `data/raw/`：

```bash
mkdir -p data/raw data/processed
ln -s /Users/shinn/Documents/DATASETS/100style data/raw/100style
```

```text
data/raw/100style/
├── Frame_Cuts.csv
└── <Style>/<Style>_<Clip>.bvh
```

构建 pruned skeleton 的 feature database：

```bash
python preprocess/build_data.py \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --window-size 64 \
  --workers 8 \
  --output data/processed/100style_pruned_test5
```

输出中 tokenizer 直接使用 `feature_database/`：

```text
data/processed/100style_pruned_test5/
├── database.npz
└── feature_database/
    ├── metadata.npz
    └── motion/motion_*.npy
```

`FeatureDataset` mmap 读取 motion shards。它兼容两种 metadata：

- 旧格式：直接读取 `train_windows` / `val_windows` / `test_windows`；
- V2 格式：从 split-safe intervals 派生固定 **64 帧**窗口。每段 interval 按 64 帧步长取窗，并补充末尾对齐窗口。

Part-FSQ 训练要求 window size 恰好为 64；全部 64 帧均参与重建和 loss，不存在 context/target 的切分或时间下采样。

### 230D motion feature

pruned skeleton 的 feature layout 为：

```text
root local linear velocity            3
root local angular velocity           3
hips local position                   3
non-root joint rotations (6D)       144
hips local velocity                   3
non-root joint angular velocities    72
left/right toe contacts               2
total                               230
```

`metadata.npz` 同时保存 train-only normalization stats、feature loss weights、joint names、parents 与 split 信息。Part-FSQ 假设 root 位于 joint index 0，joint 按 parent order 排列，并且 skeleton 能划分为 torso、左右腿、左右臂五个互斥部分。

## Hierarchical Part-FSQ

### 表征规格

```text
输入：normalized motion [B, 64, 230]
输出：FSQ indices / continuous codes [B, 64, 40]
每个坐标：9 levels，index ∈ {0, ..., 8}
时间尺度：frame-level，无 temporal downsampling
因果性：左侧 context 63 帧，无 lookahead，receptive field = 64
```

40 个坐标按以下顺序拼接，checkpoint 和后续消费代码必须保持此顺序：

| group | coordinates | 来源 / 职责 |
| --- | ---: | --- |
| `global` | 6 | root、hips、contact 等全局 feature stream |
| `sync` | 4 | 融合 global 与五个 part state 的跨身体同步信息 |
| `torso` | 6 | 躯干 joints |
| `left_leg` | 7 | 左腿 joints |
| `right_leg` | 7 | 右腿 joints；与左腿共享参数 |
| `left_arm` | 5 | 左臂 joints |
| `right_arm` | 5 | 右臂 joints；与左臂共享参数 |
| **total** | **40** | **40 × 9 FSQ** |

### 完整数据流

```text
normalized motion [B,T,230]
  │
  ├─ static feature partition
  │    global | torso | left/right leg | left/right arm
  │
  ├─ per-group input projection + stream embedding
  │
  ├─ shared frame-causal encoder
  │    six streams fold into batch dimension: [B×6,C,T]
  │
  ├─ quantization hierarchy
  │    global state ───────────────→ global FSQ (6)
  │    all six encoded states ─────→ sync FSQ (4)
  │    part state + global + sync ─→ torso FSQ (6)
  │                                → shared leg FSQ (7 × 2)
  │                                → shared arm FSQ (5 × 2)
  │
  ├─ decode hierarchy
  │    [global, sync] ─────────────→ global stream
  │    [part, global, sync] ───────→ each part stream
  │
  ├─ shared frame-causal decoder [B×6,C,T]
  │
  └─ per-group output heads + static scatter
       reconstructed motion [B,T,230]
```

这是 dense static routing，不使用 GNN、scatter/index-add 聚合、动态图或 virtual nodes。六条 temporal stream 共用同一个 causal encoder 和 decoder：通过折叠到 batch 维实现，避免 GraphFSQ 的图 kernel 开销。左右腿共享 projection、fusion、quantizer、output head；左右臂同理。

`global` 和 `sync` 是跨 group 的唯一共享通道。part token 不直接读其他 part 的私有 token，从而保持身体部位 token 的语义隔离。

### 训练 objective

训练直接使用完整 loss，不采用先训 reconstruction、再微调 reuse 的阶段式流程：

```text
L = L_recon
  + 3.0 L_delta
  + 0.1 L_root_pos
  + 0.1 L_root_rot
  + 0.5 L_joint
  + 0.1 L_contact
  + 0.1 L_foot_slide
  + 0.1 L_foot_height
  + 0.01 L_reuse
```

其中：

- `L_recon`：按 feature weights 加权的 L1 reconstruction；
- `L_delta`：相邻帧 feature delta L1；
- `L_root_pos` / `L_root_rot`：由 root velocity 积分得到的误差；
- `L_joint`：differentiable FK joint-position loss；
- `L_contact`：toe contact BCE；
- `L_foot_slide` / `L_foot_height`：ground-truth contact gating 下的物理一致性项；
- `L_reuse`：对量化后的连续 FSQ codes 的相邻帧差异施加 Charbonnier penalty。

reuse gate 根据对应身体区域 target feature 的创新度计算：慢速段 gate 接近 1，快速运动时趋向 0。contact 切换时关闭 `global`、`sync` 和受影响腿的 reuse gate，避免接触事件被错误平滑。默认 `reuse_weight=0.01`，各 group 的创新度阈值均为 1.0，可通过 YAML 或 CLI 调整。

### 训练

当前配置文件是 [configs/part_fsq_pruned.yaml](configs/part_fsq_pruned.yaml)。其中默认路径为完整数据集的 `data/processed/100style_pruned/feature_database`；若使用仓库当前 test5 数据，请显式覆盖：

```bash
python train_part_fsq.py \
  --config configs/part_fsq_pruned.yaml \
  --feature-database data/processed/100style_pruned_test5/feature_database
```

默认超参数：

| item | value |
| --- | ---: |
| batch size | 256 |
| epochs | 100 |
| learning rate | 2e-4 |
| warmup | 2 epochs |
| stream dimension | 64 |
| FSQ levels | 9 |
| DataLoader workers | 8 |
| prefetch factor | 4 |
| metrics interval | 100 steps |

训练输出位于 `outputs/part_fsq_pruned_40x9/`：

```text
outputs/part_fsq_pruned_40x9/
├── best.pt
├── last.pt
├── part_fsq_pruned_40x9.yaml
└── tensorboard/part_fsq_pruned_40x9/
```

续训只接受 `model_family == "part_fsq"` 且 `model_config` 完全一致的 checkpoint：

```bash
python train_part_fsq.py \
  --config configs/part_fsq_pruned.yaml \
  --feature-database data/processed/100style_pruned_test5/feature_database \
  --resume outputs/part_fsq_pruned_40x9/last.pt
```

每个 epoch 会在终端打印 train/validation loss、reconstruction/kinematic 子项、reuse、representation metrics、吞吐量和 learning rate。训练中只有第一个 step 及每 `metrics_interval` 个 step 计算 `bincount` / `unique` 类 representation metrics；它们仅用于观测，不参与反传或 loss。这避免每个 batch 同步计算高开销的 level usage、token tuple unique ratio 等统计。

### 定量评估

`evaluate_fsq.py` 能按 checkpoint 的 `model_family` 自动载入 Flat-FSQ 或 Part-FSQ。Part-FSQ 评估示例：

```bash
python evaluate_fsq.py \
  --checkpoint outputs/part_fsq_pruned_40x9/best.pt \
  --feature-database data/processed/100style_pruned_test5/feature_database \
  --split test \
  --batch-size 128 \
  --device auto \
  --output outputs/evaluations/part_fsq_test5.json
```

报告包含 feature reconstruction、root trajectory、FK joint error、contact precision/recall/F1、foot slide/height，以及 FSQ representation metrics。`--checkpoint` 可重复传入，以同一数据集切分比较多个 tokenizer。

## Latent Residual Part-FSQ

`latent_residual_part_fsq` 保持 Residual Part-FSQ 的 `20 base + 20 part`、40×9 token 布局，但将五个 part embedding 投影到 holistic base latent，在单一 causal decoder 前完成融合。训练时通过 `decode_base=True` 将 base/fused latent 拼到 batch 维共同解码；`decode_from_indices` 和 `decode_from_codes` 的推理路径只运行一次 decoder。

```bash
python train_latent_residual_part_fsq.py \
  --config configs/latent_residual_part_fsq_pruned.yaml
```

可通过 `--init-from-residual-checkpoint` 选择性加载 feature-side Residual Part-FSQ 的 encoder、quantizer 与 base decoder。新旧模型虽然坐标布局相同，但 `model_family` 和 part token 语义不同，checkpoint/token database 不能直接互换。`evaluate_fsq.py` 和 `evaluate_part_editing.py` 均支持新 family。

## Flat-FSQ 基线

旧的 frame-level Flat-FSQ 仍可用于对照：20 个 9-level 坐标，causal CNN encoder/decoder，receptive field 同为 64。

```bash
python train_fsq.py \
  --config configs/fsq_pruned_frame_causal_cnn.yaml
```

其配置见 [configs/fsq_pruned_frame_causal_cnn.yaml](configs/fsq_pruned_frame_causal_cnn.yaml)。Flat-FSQ 的训练、token database 与生成器是旧路线；Flat 和 Part checkpoint 的 `model_family`、model config、坐标数均不同，不能互换。

## 未接入 Part-FSQ 的功能

以下脚本目前硬性要求 `model_family == "fsq"` 或只理解 20×9 Flat-FSQ token layout，不能输入 `part_fsq` checkpoint：

- `encode_fsq_database.py`
- `view_motion_sequence.py`
- `train_fsq_generator.py`
- `train_fsq_conditional_generator.py`
- `evaluate_fsq_generator.py`
- `evaluate_fsq_conditional_generator.py`
- `generate_fsq_motion.py`
- `realtime_fsq_controller.py`

因此当前不要为 Part-FSQ 调用 token database 构建、generator 训练、离线 token rollout 或实时控制。将 Part-FSQ 接入这些模块需要单独设计 40-coordinate token schema、checkpoint metadata 校验和 decoder loading 路径。

## 测试

```bash
python -m pytest -q
```

测试覆盖 causal receptive field、FSQ index/code roundtrip、STE gradient、kinematic losses、feature-database window adapter、Part-FSQ layout/参数共享/因果性，以及 adaptive reuse contact gating。

## 代码结构

```text
configs/
  part_fsq_pruned.yaml                 Part-FSQ 主配置
  residual_part_fsq_pruned.yaml        feature-side Residual Part-FSQ
  latent_residual_part_fsq_pruned.yaml single-decoder latent residual 配置
  fsq_pruned_frame_causal_cnn.yaml     Flat-FSQ 基线配置

datasets/
  feature_dataset.py                   mmap feature windows；V1/V2 metadata adapter

models/
  part_layout.py                       skeleton → static body-part feature partition
  part_fsq.py                          dense causal Hierarchical Part-FSQ
  residual_part_fsq.py                 feature-side Residual Part-FSQ
  latent_residual_part_fsq.py          single-decoder latent Residual Part-FSQ
  part_fsq_losses.py                   adaptive latent reuse loss
  fsq.py                               Flat-FSQ quantizer / tokenizer
  causal_cnn.py                        shared causal CNN modules
  losses.py                            reconstruction 与 kinematic losses

train_part_fsq.py                      Part-FSQ train / resume / logging
train_residual_part_fsq.py             feature-side Residual Part-FSQ trainer
train_latent_residual_part_fsq.py      latent Residual Part-FSQ trainer
train_fsq.py                           Flat-FSQ baseline training
evaluate_fsq.py                        unified tokenizer evaluation
evaluate_part_editing.py               body-part token editing evaluation
preprocess/build_data.py               BVH → database + feature database
tests/test_part_fsq.py                 Part-FSQ unit tests
tests/test_residual_part_fsq.py        feature-side residual tests
tests/test_latent_residual_part_fsq.py latent-side residual tests
```

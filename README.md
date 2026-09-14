# Stylized Motion Generation

用于风格化角色动作表示、生成和可视化的研究代码库。应用层只有一个入口：

```bash
python -m stylized_motion.run --mode <mode> --pipeline <pipeline> [options]
```

当前可运行的主线是五种 canonical FSQ motion representation；它们共用数据读取、训练/验证/测试、checkpoint 和 token 接口。实验代码以可复现和快速验证为目标，不是通用训练平台。

## 当前实现

| CLI representation | family | 用途 |
| --- | --- | --- |
| `flat-fsq` | `flat_fsq` | 单一 40-coordinate motion stream |
| `part-fsq` | `part_fsq` | global、sync、torso、双腿、双臂的层次化 stream |
| `residual-part-fsq` | `residual_part_fsq` | holistic base 加 feature-space 局部 residual |
| `latent-residual-fsq` | `latent_residual_fsq` | 旧版 latent-residual 对照 |
| `latent-residual-fsq-v2` | `latent_residual_fsq_v2` | 全维 part residual projection 与 compensated part edit |

五条主线固定使用 `motion_dim=230`、每帧 40 个 9-level FSQ coordinate、60 FPS、64 帧 causal receptive field 和 0 帧 lookahead。representation 训练窗口为 64 帧；generator 使用前 64 帧 token 预测接下来的 64 帧 token。checkpoint 会校验 family、coordinate layout、feature schema、normalization 和 causal metadata，不能跨 family 混用。

`latent_residual_fsq_v2` 是当前的局部编辑实验主线：base latent 与 part projection 均为稠密 latent，不预先分配 body-part channel；编辑时会将 donor 的 part state 重新表示到 target base 条件下。模型细节、损失和限制见 [docs/latent_residual_part_fsq_v2_spec.md](docs/latent_residual_part_fsq_v2_spec.md)。

旧 VQ-VAE 配置仍保留在 `data/configs/vqvae_*.yaml`，仅作历史 baseline；不属于上述 canonical representation dispatch 路径。

## 目录

```text
args/                         五条 FSQ 训练预设
data/configs/                 representation 与 generator YAML 配置
data/assets/                  GenoView / SomaView 着色器及可再生资产位置
stylized_motion/
  run.py                      唯一应用级 dispatcher
  data/                       preprocess、Store、sampling、DataLoader
  learning/                   FSQ representation、runner、generator、checkpoint
  anim/                       BVH、GenoView、SomaView、离屏渲染、实时 rollout
  util/                       参数文件和路径工具
tests/                        contract 与回归测试
docs/
  latent_residual_part_fsq_v2_spec.md
  assets/pbr_baseline/        受版本控制的离屏渲染回归图像
```

`data/raw/`、`data/processed/` 和 `outputs/` 都是本地数据或生成结果，不纳入版本控制。`data/processed/` 的 FeatureStore/TokenStore 及 `outputs/` 的 checkpoint 均可从原始数据、配置和训练命令重建。

## 环境

```bash
conda activate mcc
pip install -r requirements.txt
```

运行时通过 `--device auto|cuda|mps|cpu` 选择设备。`amp` 仅在 CUDA 上可用；提交的 canonical representation 配置默认使用 `fp32`。

## 数据构建

原始 BVH 数据由 `--dataset lafan|100style|combined` 选择。FSQ 训练所需的数据链为：

```text
raw BVH
  -> feature-cache          # 未归一化的逐帧 motion_feature_v2
  -> fsq-window-index       # 归一化 shard、64 帧 window 与 train/val/test split
  -> token-database         # 使用已训练 representation 编码（generator 可选）
```

以 100Style 为例，先构建 FeatureStore：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-cache \
  --dataset 100style \
  --output data/processed/100style_pruned_90/feature_cache \
  --prune-ends-and-fingers --workers 8

python -m stylized_motion.run \
  --mode preprocess --pipeline fsq-window-index \
  --feature-cache data/processed/100style_pruned_90/feature_cache \
  --output data/processed/100style_pruned_90/fsq_window_index \
  --seed 3407
```

已有同名输出时，再明确加 `--overwrite`。可用以下命令检查 Store 的 manifest、shard、range、split 和 normalization：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline validate-data \
  --feature-store data/processed/100style_pruned_90/fsq_window_index --full
```

`motion-database`、`feature-database`、`trajectory-inputs` 和 `trajectory-database` 仍可由 preprocess pipeline 调用，分别服务原始动作检查和 trajectory 条件输入；canonical FSQ representation 训练读取的是 `fsq-window-index`。

## Representation 训练

各默认 YAML 指向 `data/processed/100style_pruned_90/fsq_window_index`；使用其他数据位置时，先修改该配置的 `data.fsq_window_index`，再启动训练。

```bash
# Flat-FSQ
python -m stylized_motion.run --arg-file args/flat_fsq_args.txt

# 当前局部编辑主线
python -m stylized_motion.run --arg-file args/latent_residual_fsq_v2_args.txt
```

也可以显式调用：

```bash
python -m stylized_motion.run \
  --mode train --pipeline representation \
  --representation latent-residual-fsq-v2 \
  --config data/configs/latent_residual_fsq_v2_40x9.yaml \
  --device cuda
```

验证或测试需要匹配的 checkpoint：

```bash
python -m stylized_motion.run \
  --mode validate --pipeline representation \
  --representation latent-residual-fsq-v2 \
  --config data/configs/latent_residual_fsq_v2_40x9.yaml \
  --checkpoint outputs/latent_residual_fsq_v2_40x9/best.pt \
  --device cuda
```

训练输出默认为配置中的 `training.output_dir`，包含 `last.pt`、`best.pt` 和 `tensorboard/`。完整训练是耗时操作；开始前应确认数据路径、输出目录、seed 与 device。TensorBoard 日志记录通用重建/运动学损失；各 family 的专用 loss 会写入 checkpoint metrics 与控制台汇总。

## TokenStore 与 Generator

TokenStore 由匹配的 representation checkpoint 构建。它继承 FeatureStore 的 range、split 和 feature schema；不会重新划分数据。

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline token-database \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --feature-store data/processed/100style_pruned_90/fsq_window_index \
  --output data/processed/100style_pruned_90/flat_fsq_40x9 \
  --device cuda --chunk-size 1024 --save-codes
```

生成器独立训练和采样：

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

Generator 配置中的 `data.token_database` 必须指向实际生成的 TokenStore；generator checkpoint 也会绑定 tokenizer checkpoint SHA256 和 representation metadata。

## 可视化

`visualize` 提供 motion reconstruction、part edit、GenoView、SomaView 和 realtime FSQ rollout。下例在不打开窗口时检查 V2 part edit 的加载、上下文和解码路径：

```bash
python -m stylized_motion.run \
  --mode visualize --pipeline part-edit \
  --checkpoint outputs/latent_residual_fsq_v2_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --target-range-idx 0 --target-start 0 \
  --donor-range-idx 16 --donor-start 0 \
  --length 240 --part left_arm --device cuda --dry-run
```

GenoView 和 SomaView 共用 renderer。PBR 路径当前使用单张 depth shadow map 与 PCF、G-buffer、SSAO、HDR/tonemap 和可选 IBL；`--shading legacy` 使用保留的旧光照路径。`docs/assets/pbr_baseline/` 保存离屏渲染的回归图像和再生成方法。

## 检查与清理

```bash
# 不需要数据或 GUI 的代码检查
python -m compileall -q stylized_motion

# 在已安装测试依赖的环境中运行测试
python -m pytest -q
```

本地生成的 `outputs/`、`data/processed/`、`__pycache__/`、`.pytest_cache/` 和 `.zcode/` 均不应作为源码提交。清理它们会删除本地 checkpoint、训练日志和预处理数据；如需保留实验结论，请先将所需结果归档到仓库外的位置。

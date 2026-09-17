# Stylized Motion Generation

用于风格化角色动作表示、生成和可视化的研究代码库。应用层只有一个入口：

```bash
python -m stylized_motion.run --mode <mode> --pipeline <pipeline> [options]
```

当前可运行的主线是六种 canonical FSQ motion representation；它们共用数据读取、训练/验证/测试、checkpoint 和 token 接口。实验代码以可复现和快速验证为目标，不是通用训练平台。

## 当前实现

| CLI representation | family | 用途 |
| --- | --- | --- |
| `flat-fsq` | `flat_fsq` | 单一 40-coordinate motion stream |
| `part-fsq` | `part_fsq` | global、sync、torso、双腿、双臂的层次化 stream |
| `residual-part-fsq` | `residual_part_fsq` | holistic base 加 feature-space 局部 residual |
| `latent-residual-fsq` | `latent_residual_fsq` | 旧版 latent-residual 对照 |
| `latent-residual-fsq-v2` | `latent_residual_fsq_v2` | 全维 part residual projection 与 compensated part edit |
| `nef-fsq` | `nef_fsq` | Node–Edge Factorized：13 个独立量化 stream，Geno/SOMA 各自 layout |

前五条主线固定使用 `motion_dim=230`、每帧 40 个 9-level FSQ coordinate、60 FPS、64 帧 causal receptive field 和 0 帧 lookahead。representation 训练窗口为 64 帧；generator 使用前 64 帧 token 预测接下来的 64 帧 token。checkpoint 会校验 family、coordinate layout、feature schema、normalization 和 causal metadata，不能跨 family 混用。

`nef_fsq` 的 motion width 由 skeleton 决定（Geno `9J+5=230`、SOMA `9J+5=248`），coordinate order 固定为 13 个 Node/Edge stream，skeleton、layout、feature schema 或 coordinate order 不匹配时拒绝加载和 token 交换。设计与验收标准见 [docs/NEF-FSQ_Design.md](docs/NEF-FSQ_Design.md)。

`latent_residual_fsq_v2` 是当前的局部编辑实验主线：base latent 与 part projection 均为稠密 latent，不预先分配 body-part channel；编辑时会将 donor 的 part state 重新表示到 target base 条件下。模型细节、损失和限制见 [docs/latent_residual_part_fsq_v2_spec.md](docs/latent_residual_part_fsq_v2_spec.md)。

旧 VQ-VAE 配置仍保留在 `data/configs/vqvae_*.yaml`，仅作历史 baseline；不属于上述 canonical representation dispatch 路径。

## 目录

```text
args/                         FSQ 训练预设
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
  MTS_FSQ_SIGGRAPH_Implementation_Plan_zh.md  MTS-FSQ 研究方案
  mts_operator_landing.md                     MTS-FSQ 落地对照与未验证项
  nef_phase0_probe.md                         NEF Phase 0 探针使用说明
  seed_1h_run_report.md                       BONES-SEED 全量数据 + 1 小时训练记录
  seed_stage2_promotion.md                    performer 标签 / 物理 fine-tune / R1 表征对照
  bones_seed_data_pipeline_plan.md          SEED 数据管线迁移方案
  bones_seed_pipeline_implementation.md     落地范围、验证情况与待办
  assets/pbr_baseline/        受版本控制的离屏渲染回归图像
```

`stylized_motion/data/` 同时提供两代数据契约：v3（`feature_data.py`、`token_data.py`、`trajectory_data.py`，每来源一分片并预归一化）与 v4（`seed_catalog.py`、`seed_build.py`、`packed_store.py`、`packed_token.py`、`packed_trajectory.py`，物理分片与逻辑 clip 解耦、统计独立版本化）。`packed_store.open_any_feature_store` 按 manifest 的 `data_schema_version` 分派，v3 store 不会被就地改写。

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

### BONES-SEED（schema v4 packed store）

一张图看清数据从原始 BVH 到训练的全部阶段：[docs/seed_pipeline_overview.md](docs/seed_pipeline_overview.md)。

SEED 不走上面那条“每个来源生成 original/mirror 两个文件”的路径：官方已有约 7.1 万个镜像文件，重复生成会让特征量翻倍。SEED 使用目录化入口，特征只存一份未归一化版本，统计与 split 各自版本化：

```text
CSV metadata
  -> seed-catalog           # clip 目录、帧契约、镜像策略、take-group split
  -> seed-inventory         # header / fps / 骨架 / 帧数一次性核对
  -> packed-feature-store   # ~256 MiB 未归一化分片 + clip 表 + 统计（可中断续跑）
  -> packed-token-store     # 按逻辑 clip 编码（generator 需要）
  -> packed-trajectory-store# clip-local 未来控制量 + validity mask
```

目录约定：**store 自身就是配置里写的那个目录**，catalog 与 unit 缓存放在它外面（publish 会整体替换 store 目录，缓存若在内部会被一并删除，构建会直接拒绝这种路径）：

| 产物 | 路径 |
| --- | --- |
| 特征 store（`data.fsq_window_index`） | `data/processed/seed_soma_pruned_v4` |
| catalog + inventory | `data/processed/seed_soma_pruned_v4_catalog` |
| unit 缓存（可删，删后重跑） | `data/processed/seed_soma_pruned_v4_units` |
| token store | `data/processed/seed_soma_pruned_v4_tokens` |
| trajectory store | `data/processed/seed_soma_pruned_v4_trajectory` |

```bash
python -m stylized_motion.run --mode preprocess --pipeline seed-catalog \
  --seed-root data/raw/seed --output data/processed/seed_soma_pruned_v4_catalog \
  --mirror-policy official

python -m stylized_motion.run --mode preprocess --pipeline packed-feature-store \
  --catalog data/processed/seed_soma_pruned_v4_catalog \
  --output data/processed/seed_soma_pruned_v4 \
  --unit-cache data/processed/seed_soma_pruned_v4_units \
  --workers 12 --shard-mib 256

# split 或统计变化时，只重算统计，不重写特征字节
python -m stylized_motion.run --mode preprocess --pipeline train-stats \
  --store data/processed/seed_soma_pruned_v4
```

对应训练配置 `data/configs/nef_fsq_soma_packed_40x9.yaml`（`required_data_schema_version: 4`）。验证预算与评估节奏分开：`sampling.eval_limit` 限制的是**监控子集**，`evaluation.full_eval_every_epochs` 控制**全量验证**周期，`best.pt` 只依据全量验证（没有配置 `eval_limit` 时两者是同一个 loader）。训练预算由 `training.steps_per_epoch` / `max_steps` / `checkpoint_every_steps` 控制，`--checkpoint` 指向已有 checkpoint 即从该处续跑。分层 benchmark：

```bash
python -m stylized_motion.run --mode benchmark --pipeline data \
  --store data/processed/seed_soma_pruned_v4 --layers sampler,loader,resident,end_to_end
```

落地范围、已验证内容与待办见 [docs/bones_seed_pipeline_implementation.md](docs/bones_seed_pipeline_implementation.md)。
全量数据构建与 1 小时 NEF-FSQ 训练的实测数字见 [docs/seed_1h_run_report.md](docs/seed_1h_run_report.md)；performer 标签、物理 fine-tune（v1 vs v1.1）与 flat/part/NEF 的 R1 局部性对照见 [docs/seed_stage2_promotion.md](docs/seed_stage2_promotion.md)。

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

NEF-FSQ 使用 `args/nef_fsq_args.txt`（Geno）或 `args/nef_fsq_soma_args.txt`（SOMA），分别对应 `data/configs/nef_fsq_40x9.yaml` 与 `data/configs/nef_fsq_soma_40x9.yaml`。训练后的 per-stream 报告与编辑评估：

```bash
python -m stylized_motion.run \
  --mode evaluate --pipeline nef-fsq \
  --metric transfer \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_90/fsq_window_index \
  --target-range-idx 0 --target-start 0 \
  --donor-range-idx 40 --donor-start 0 --length 64 \
  --part left_arm --edit strict --edit-start 10 --edit-stop 40 \
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

### MTS style operator（研究分支 `feat/mts-fsq-siggraph`）

`stylized_motion/learning/mts_operator/` 是方案 §5 的算子包：token/掩码 contract（`contract.py`）、
layout 只读索引（`layout_adapter.py`）、无 style 的 base transport（`embeddings.py`、`graph.py`、
`masking.py`、`transport.py`）、三族风格算子（`operators.py`：logit field / arbitrary kernel /
birth-death CTMC）、耦合采样（`sampling.py`）、style pair 审计（`pairs.py`）、参考风格编码
（`style_encoder.py`）、顶层模型与训练循环（`model.py`、`training.py`）、指标（`metrics.py`）。

```bash
python scripts/probe_nef_geometry.py --checkpoint <nef.pt> --feature-database <store> --split test
python scripts/audit_style_pairs.py --feature-database <store> --output outputs/mts_pairs/audit
python scripts/train_mts_transport.py --config data/configs/mts_operator_transport.yaml \
  --tokenizer-checkpoint <nef.pt> --output outputs/mts_transport/seed3407
python scripts/train_mts_operator.py --config data/configs/mts_operator_style.yaml \
  --tokenizer-checkpoint <nef.pt> --transport-checkpoint <transport.pt> --operator birth_death
python scripts/evaluate_mts_operator.py --checkpoint <operator.pt> --tokenizer-checkpoint <nef.pt> \
  --feature-database <store> --split test
python scripts/generate_mts_operator.py --checkpoint <operator.pt> --tokenizer-checkpoint <nef.pt> \
  --feature-database <store> --content-clip 0 --style-clip 3 --regions left_arm --locked-edit
```

落地范围、已用真实数据验证的读数、以及**尚未验证**的部分见
[docs/mts_operator_landing.md](docs/mts_operator_landing.md)；Phase 0 两个探针怎么读见
[docs/nef_phase0_probe.md](docs/nef_phase0_probe.md)。

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

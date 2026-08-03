# Stylized Motion Generation

面向风格化角色动作的离散 motion tokenizer 研究项目。当前唯一主线是四条共享 contract 的 FSQ representation：

- `flat_fsq`：单一 motion stream，`flat` variant；
- `part_fsq`：global/sync/part hierarchical stream，`hierarchical` variant；
- `residual_part_fsq`：holistic base + feature-space local residual，`default` variant；
- `latent_residual_fsq`：holistic base + disjoint latent residual，`v2` variant。

四条主线都使用 `motion [B,T,230]`、`K=40`、`L=9`、frame-level causal contract。representation 的模型和 checkpoint 规则见 [主线重构 spec](docs/refactor_spec.md)，数据侧的 split、采样、batch 和 loader 规则见 [Data Pipeline Spec](docs/data_pipeline_spec.md)。

## 目录

```text
args/                          MimicKit 风格命令预设
data/configs/                  representation、generator、VQ-VAE 配置
stylized_motion/data/          flat feature、token、trajectory data modules
stylized_motion/learning/      representation、统一 runner、loss、metrics
stylized_motion/learning/nets/ 可复用 causal CNN/Transformer/quantizer
stylized_motion/anim/          BVH、motion feature、GenoView 和可视化
stylized_motion/run.py         唯一应用级 dispatch 入口
tests/                         contract 与回归测试
docs/refactor_spec.md           FSQ representation 与跨层集成 spec
docs/data_pipeline_spec.md     data schema、sampling、batch、loader spec
```

`learning/vqvae.py` 和 `data/configs/vqvae_*.yaml` 是隔离的 baseline，不属于 FSQ representation registry。

## 环境与数据

数据侧严格使用 Draft 0.1 的 schema v3 contract；旧 `metadata.npz` 数据不再读取，已有
数据需要重新构建。canonical output 为 `manifest.json + index.npz + mmap shards`。

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
pip install -r requirements.txt
```

准备原始数据并构建 feature database：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-database \
  --dataset 100style \
  --max-styles 5 \
  --prune-ends-and-fingers \
  --workers 8 \
  --output data/processed/100style_pruned_test5/feature_database
```

feature database 使用 data schema v3 的 `manifest.json + index.npz + mmap motion shards`，并包含 `motion_feature_v2` 的 joint names、parents、normalization stats、`230D` schema hash、source-clip split manifest 和 intervals。

## Representation 训练

四条主线使用同一个 `RepresentationRunner`，通过 `--representation` 选择 family：

```bash
python -m stylized_motion.run \
  --mode train --pipeline representation \
  --representation flat-fsq \
  --config data/configs/flat_fsq_40x9.yaml

python -m stylized_motion.run \
  --mode train --pipeline representation \
  --representation part-fsq \
  --config data/configs/part_fsq_40x9.yaml

python -m stylized_motion.run \
  --mode train --pipeline representation \
  --representation residual-part-fsq \
  --config data/configs/residual_part_fsq_40x9.yaml

python -m stylized_motion.run \
  --mode train --pipeline representation \
  --representation latent-residual-fsq \
  --config data/configs/latent_residual_fsq_40x9.yaml
```

验证和测试只替换 workflow mode，并显式提供 canonical checkpoint：

```bash
python -m stylized_motion.run \
  --mode validate --pipeline representation \
  --representation part-fsq \
  --config data/configs/part_fsq_40x9.yaml \
  --checkpoint outputs/part_fsq_40x9/best.pt

python -m stylized_motion.run \
  --mode test --pipeline representation \
  --representation part-fsq \
  --config data/configs/part_fsq_40x9.yaml \
  --checkpoint outputs/part_fsq_40x9/best.pt
```

统一 runner 负责 device、FP32/AMP policy、optimizer、scheduler、gradient clipping、TensorBoard、best/last checkpoint 和吞吐量统计；DataLoader 由 data spec 规定的统一 data loader 负责。每个 checkpoint 必须包含 schema v2、canonical `representation`、top-level `feature_schema`、model config 和 state dict。

## 坐标布局

`flat_fsq` 使用 `flat:40`。`part_fsq` 的固定顺序是 `global:6, sync:4, torso:6, left_leg:7, right_leg:7, left_arm:5, right_arm:5`。`residual_part_fsq` 与 `latent_residual_fsq` 共享 `base:20, torso:6, left_leg:4, right_leg:4, left_arm:3, right_arm:3`，但属于不同 family；latent residual 强制 `architecture_version=2`。

所有 representation 都要求 `temporal_downsample=1`、`receptive_field=64`、`lookahead_frames=0`；history frames 由 receptive field contract 推导为 63。Residual Part-FSQ 的 inference decoder pass 为 2，其余为 1。

## Token database

使用 registry-driven encoder 生成 token shards：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline token-database \
  --checkpoint outputs/flat_fsq_40x9/best.pt \
  --feature-database data/processed/100style_pruned_test5/feature_database \
  --output data/processed/100style_pruned_test5/flat_fsq_40x9 \
  --save-codes
```

`TokenStore` 只接受 data schema v3，并校验 checkpoint SHA256、family/variant/id、`K/L`、coordinate order/counts、feature schema、split manifest、frame rate 和 causal metadata。完整 shard value 校验由 preprocess full validation 执行，训练启动只做 runtime contract 校验。

trajectory database 由 token database 派生，并保留相同的 shard、representation 和 feature schema contract：

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline trajectory-database \
  --token-database data/processed/100style_pruned_test5/flat_fsq_40x9 \
  --trajectory-input data/processed/100style_pruned_test5/trajectory_inputs.npz \
  --output data/processed/100style_pruned_test5/flat_fsq_40x9_trajectory
```

## Generator 与可视化

Generator 从 `TokenStore` 动态读取坐标数、level 数和 layout，不在模型代码中写死 `K`。训练与生成入口分别是：

```bash
python -m stylized_motion.run \
  --mode train --pipeline generator \
  --config data/configs/fsq_generator.yaml

python -m stylized_motion.run \
  --mode generate --pipeline motion \
  --token-database data/processed/100style_pruned_test5/flat_fsq_40x9 \
  --generator-checkpoint outputs/generator_flat_fsq_40x9/best.pt \
  --seed-indices outputs/seed_indices.npy \
  --output outputs/generated_indices.npy
```

generator autoregressive training、trajectory conditioning 和 realtime consumer 都通过统一 TokenStore/TrajectoryStore public API 读取数据；不会把 Part/Residual token 当作 Flat token 解释。representation decoder、motion viewer 和 realtime loader 都通过 `load_representation_checkpoint()` 构造统一 adapter。

## 命令预设与测试

```bash
python -m stylized_motion.run --arg-file args/part_fsq_args.txt
conda run -n mcc python -m pytest -q
```

`args/` 只包含四个 canonical representation preset。实现代码全部位于 `stylized_motion/`，旧根目录脚本、旧 workflow wrapper、具体 representation evaluator 和旧 token store 模块不再是项目 API。

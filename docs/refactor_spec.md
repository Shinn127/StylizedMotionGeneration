# Stylized Motion Generation

## FSQ 主线重构 Spec

状态：Draft 0.4
日期：2026-08-03
范围：representation、checkpoint、generator、评估、统一命令入口和跨层集成

执行口径：representation checkpoint 只注册和读取 canonical schema v2。迁移表中的旧
`model_family` 值只作为 `model_family_legacy` 序列化字段保留，不提供旧
checkpoint、旧参数名或旧命令 alias 的读取路径；所有 workflow 必须通过
canonical representation spec 构造和校验。

数据管线的 schema、sampling、batch、loader 和离线构建规范已独立到
[`data_pipeline_spec.md`](data_pipeline_spec.md)。该文件是数据侧唯一权威来源；本文件
只定义 representation/learning 与 data 的跨层接口。checkpoint schema 继续使用 v2。

## 1. 目标

项目后续以四条 FSQ representation 线为主线：

1. `Flat-FSQ`
2. `Hierarchical Part-FSQ`
3. `Residual Part-FSQ`
4. `Latent Residual-FSQ`

四条线共享同一套 motion feature、训练窗口、checkpoint 元数据、token database 和下游消费 contract。模型内部可以有不同的编码层级和 residual/fusion 策略，但下游不应再通过导入具体 Python 类来判断 token 格式。

本轮重构的结果应满足：

- 新增 representation 时只需要注册模型、配置和能力，不需要复制 encode/evaluate/generate 流程；
- checkpoint、token database 和 generator 能通过显式 metadata 校验 contract；
- 四条 FSQ representation 均能走同一个 representation API；
- 四条 representation 的训练、验证、测试共用同一个 runner、metric suite、checkpoint manager 和数据 contract；
- representation、runner 和 generator 只通过独立 data spec 的 public API 消费 tensor batch；
- 根目录继续保持 MimicKit 风格：配置在 `data/`，预设在 `args/`，实现代码在 `stylized_motion/`，应用流程由 `run.py` 调度；
- 不为历史 checkpoint、历史命令名或历史模块路径保留兼容入口；历史实验产物不属于当前代码 API。

## 2. 当前状态与问题

### 2.1 当前 family

当前代码中存在四个 tokenizer checkpoint family，正好对应四条主线：

| 当前 `model_family` | 归属主线 | 说明 |
| --- | --- | --- |
| `fsq` | `flat_fsq` | 单一 frame-causal encoder/FSQ/decoder |
| `part_fsq` | `part_fsq` | global/sync/part hierarchical FSQ |
| `residual_part_fsq` | `residual_part_fsq` | holistic base + feature-space local residual |
| `latent_residual_part_fsq` | `latent_residual_fsq` | latent residual V2，单 decoder |

四条主线都进入新的 representation registry。`vqvae` 保留为 legacy baseline，不纳入新的 FSQ representation contract。

### 2.2 Draft 0.3 已建立的基线

当前实现已经完成以下 canonical 收口，Draft 0.4 不重新设计这些边界：

- 四条 FSQ family 统一注册在 `learning/representation.py`，并使用 `40 x 9` token contract；
- checkpoint schema v2 已包含 family、variant、coordinate layout、causal metadata 和 feature schema；
- TokenStore、FeatureStore 和 TrajectoryStore 均由独立 data spec 定义为 schema v3，并校验 representation/checkpoint/feature contract；
- representation train/validate/test 已共用 `learning/runner.py`；
- generator 已从 TokenStore metadata 获取 `K/L/layout`，不直接导入具体 representation class；
- 旧 checkpoint、旧命令 alias 和旧模块路径不属于 canonical API。

Draft 0.4 保持 `40 x 9` representation/checkpoint 语义不变，替换其下方的 runtime
data、sampling 和 DataLoader contract。

### 2.3 Data pipeline 状态

Draft 0.3 data pipeline 的问题、目标设计和迁移方案由独立
[`data_pipeline_spec.md`](data_pipeline_spec.md) 维护。主 spec 不复制这些实现细节。

## 3. 四条主线定义

### 3.1 Flat-FSQ

规范 ID：`flat_fsq_40x9`  
旧 family：`fsq`  
坐标语义：单一 motion stream 的 frame-level coordinates  
当前实现：`FSQMotionAutoencoder`

```text
normalized motion [B, T, 230]
  -> shared frame-causal encoder
  -> one MotionFSQ quantizer [K=40, L=9]
  -> shared frame-causal decoder
  -> reconstructed motion [B, T, 230]
```

Flat-FSQ 的 `K` 必须来自 checkpoint/config，当前 contract 固定为 40；任何 `20 x 9` token database 都不属于当前 API。

### 3.2 Hierarchical Part-FSQ

规范 ID：`part_fsq_40x9`  
旧 family：`part_fsq`  
当前实现：`HierarchicalPartFSQMotionAutoencoder`

坐标顺序固定为：

| group | coordinates |
| --- | ---: |
| `global` | 6 |
| `sync` | 4 |
| `torso` | 6 |
| `left_leg` | 7 |
| `right_leg` | 7 |
| `left_arm` | 5 |
| `right_arm` | 5 |
| total | 40 |

`global` 和 `sync` 是跨 part 的共享通道；左右腿共享参数，左右臂共享参数。该语义必须进入 checkpoint 的 `coordinate_layout`，不能只依赖 Python 常量。

### 3.3 Residual Part-FSQ

规范 ID：`residual_part_fsq_40x9`  
旧 family：`residual_part_fsq`  
当前实现：`ResidualPartFSQMotionAutoencoder`

token 坐标顺序固定为：

| group | coordinates | 说明 |
| --- | ---: | --- |
| `base` | 20 | holistic base stream |
| `torso` | 6 | local residual |
| `left_leg` | 4 | local residual |
| `right_leg` | 4 | local residual |
| `left_arm` | 3 | local residual |
| `right_arm` | 3 | local residual |
| total | 40 | 40 x 9 |

Residual Part-FSQ 在 feature/output space 中重建 local residual；base stream 与 part residual 分别解码，推理阶段使用两路 decoder。

### 3.4 Latent Residual-FSQ

规范 ID：`latent_residual_fsq_40x9`  
旧 family：`latent_residual_part_fsq`  
当前实现：`LatentResidualPartFSQMotionAutoencoder`

Latent Residual-FSQ 与 Residual Part-FSQ 共享 `base + torso/leg/arm` 的 token coordinate layout，但属于独立 representation family。它把 local residual 投影到 base latent 的互斥子空间，再通过单一 causal decoder 重建：

```text
motion
  -> quantized holistic base [20 coordinates]
  -> quantized local residuals [20 coordinates]
  -> fixed disjoint latent projectors
  -> q_base + sum(delta_z_part)
  -> one causal decoder
```

`part_latent_dims` 是 latent fusion 的维度分配，不是 token 坐标顺序；二者必须分开记录。`architecture_version=2` 是该 family 的强制 representation version。

## 4. 统一 Representation Layer

四条 FSQ 主线都实现同一组 representation 接口。Representation layer 负责“motion ↔ FSQ representation”的语义和模型构造；训练、验证、测试和 generator 不直接依赖具体模型类。第一阶段可以通过 Protocol/adapter 约束，不要求立刻把所有模型改成同一个继承层次。

```python
class RepresentationProtocol(Protocol):
    family: str
    variant: str
    representation_id: str
    motion_dim: int
    num_coordinates: int
    num_levels: int
    receptive_field: int
    lookahead_frames: int

    def forward(self, motion: Tensor) -> dict[str, Tensor]: ...
    def encode_to_indices(self, motion: Tensor) -> Tensor: ...
    def encode_to_codes(self, motion: Tensor) -> tuple[Tensor, Tensor]: ...
    def decode_from_indices(self, indices: Tensor) -> Tensor: ...
    def decode_from_codes(self, codes: Tensor) -> Tensor: ...
    def representation_metadata(self) -> dict[str, object]: ...
    def compute_representation_losses(self, output: dict, batch: dict) -> dict[str, Tensor]: ...
```

输入/输出约束：

- motion 输入形状为 `[B, T, motion_dim]`；模型 forward 不写死序列长度；
- 训练 batch shape、context 和 mask 由 data spec 定义；common loss 和 representation-specific loss 必须正确消费其 `loss_mask`；
- `indices` 形状为 `[B, T, K]`，dtype 为整数，取值范围 `[0, num_levels)`；
- `codes` 形状为 `[B, T, K]`，dtype 为 float；
- `K`、`num_levels`、coordinate order 只从 representation metadata 获取；
- `lookahead_frames` 必须为 0；当前四条主线 receptive field 为 64；
- 所有 encode/decode 路径必须保持 frame alignment，不能隐式做 temporal downsample。

### 4.1 Registry 与 builder

`learning/representation.py` 中的 `build_representation()` 对齐 MimicKit 的 `agent_builder.py`，根据配置中的 `representation.family` 和 `representation.variant` 构造模型：

```text
representation config
  -> build_representation()
  -> FlatFSQRepresentation
  -> PartFSQRepresentation
  -> ResidualPartFSQRepresentation
  -> LatentResidualFSQRepresentation
```

builder 是唯一允许知道四个具体 representation class 的地方。checkpoint loader、训练器、验证器、测试器和可视化只接收 `RepresentationProtocol`。

### 4.2 统一输出与 loss 边界

每个 representation 的 `forward()` 至少返回：

```text
recon_state
indices
codes
representation_metrics
```

通用 loss 由 `learning/losses.py` 计算：feature reconstruction、delta、root trajectory、FK joint、contact、foot slide/height。representation 特有项由 `compute_representation_losses()` 返回，例如 Part 的 reuse gate、Residual 的 base reconstruction、Latent Residual 的 edit/base/latent ratio。这样四个模型可以共享训练循环，同时保留必要的结构性 loss。

representation layer 不负责 optimizer、DataLoader、epoch loop、日志或 checkpoint 文件写入；这些职责属于统一 learning runner。

## 5. Checkpoint Metadata Contract

checkpoint 顶层保留现有字段，但新增统一 `schema_version` 和 `representation` 区块：

```yaml
schema_version: 2
model_family: fsq                  # serialized source value; not a loader selector
model_config: {...}
representation:
  family: flat_fsq                  # flat_fsq | part_fsq | residual_part_fsq | latent_residual_fsq
  variant: flat                     # flat | hierarchical | default | v2
  representation_id: flat_fsq_40x9
  coordinate_order: [flat]
  coordinate_counts: {flat: 40}
  num_coordinates: 40
  num_levels: 9
  temporal_downsample: 1
  frame_rate: 60
  receptive_field: 64
  lookahead_frames: 0
  decoder_passes_inference: 1
feature_schema:
  name: motion_feature_v2
  motion_dim: 230
  joint_subset: pruned
  names_sha256: ...
  stats_sha256: ...
```

序列化字段映射（仅用于记录来源，不提供 alias 读取）：

| 旧 `model_family` | 新 canonical family | 新 variant |
| --- | --- | --- |
| `fsq` | `flat_fsq` | `flat` |
| `part_fsq` | `part_fsq` | `hierarchical` |
| `residual_part_fsq` | `residual_part_fsq` | `default` |
| `latent_residual_part_fsq` | `latent_residual_fsq` | `v2` |

新训练产生的 checkpoint 使用新 `representation` 区块，并保留
`model_family_legacy` 作为来源记录。loader 只接受完整 schema v2 checkpoint，
不会根据旧字段猜测 representation。

## 6. Data Pipeline Integration Boundary

数据侧完整 contract 见 [`data_pipeline_spec.md`](data_pipeline_spec.md)。跨层边界固定为：

- `stylized_motion.data` 禁止导入 `stylized_motion.learning`；
- `stylized_motion.run` 是 composition root，负责构造 representation 并向 data preprocess 注入 token encoder；
- data loader 只返回 CPU tensor batch，不执行 device transfer；
- learning runner 每 step 只执行一次 device transfer，model、loss 和 metrics 共用同一个 device batch；
- representation common loss 与 family-specific loss 必须消费 data batch 提供的 `loss_mask`；
- generator 只消费 TokenStore/Dataset/Loader public API，不解释 split index 或直接打开 shard；
- FeatureStore、TokenStore、TrajectoryStore 的 data schema 和兼容策略完全由独立 data spec 决定；
- checkpoint schema、representation metadata 和 coordinate layout 仍由本文件定义。

## 7. Data Contract Reference

以下内容不得在本文件重复定义：

- Store 物理布局与 data schema；
- split manifest 与 source-clip policy；
- train/val/test sampling；
- representation/generator/conditional batch shape 与 dtype；
- mmap cache、collate、worker、prefetch 和 DataLoader policy；
- data preprocess、public exports、实施阶段和 data acceptance tests。

发生冲突时，数据侧行为以 [`data_pipeline_spec.md`](data_pipeline_spec.md) 为准；
representation/checkpoint 行为以本文件为准。

## 8. 目标目录结构

延续 MimicKit 的组织风格，同时保持当前项目的领域边界：

```text
args/
  flat_fsq_args.txt
  part_fsq_args.txt
  residual_part_fsq_args.txt
  latent_residual_fsq_args.txt

data/
  configs/
    flat_fsq_40x9.yaml
    part_fsq_40x9.yaml
    residual_part_fsq_40x9.yaml
    latent_residual_fsq_40x9.yaml
    fsq_generator.yaml
    fsq_generator_conditional.yaml
    vqvae_*.yaml
  assets/
  raw/
  processed/

stylized_motion/
  anim/
  data/                       # 见 data_pipeline_spec.md
  learning/
    representation.py
    fsq.py
    part_fsq.py
    residual_part_fsq.py
    latent_residual_fsq.py
    part_layout.py
    nets/
      causal_cnn.py
      causal_transformer.py
      resnet.py
      quantizer.py
    losses.py
    metrics.py
    checkpoint.py
    runner.py
    generate.py
  util/
  run.py

tools/
tests/
docs/
```

迁移原则：

- 不为每条 FSQ 复制一套 data loader、checkpoint loader 或 token store；
- data 目录和迁移规则以 [`data_pipeline_spec.md`](data_pipeline_spec.md) 为准；
- `learning/representation.py` 负责公共接口、representation spec、builder 和 metadata；
- 四个 representation 实现与 `part_layout.py` 平铺在 `learning/`，保持 MimicKit 的 direct-module taste；
- `learning/nets` 放 causal CNN、Transformer、ResNet、quantizer 等可复用网络；
- `learning/runner.py` 是四条 FSQ 共用的 train/validate/test 核心，不为每个模型复制一套生命周期脚本；
- `stylized_motion/run.py` 只负责 CLI/config 解析和 runner 装配，不再额外创建 `train.py`、`validate.py`、`test.py`；
- `anim` 只负责 motion/BVH/GenoView 和可视化，不直接决定 representation family；
- generator 只依赖 `TokenStore` 和 representation metadata，不导入具体 representation class；
- `vqvae.py` 保留在 `learning/`，但从主线 representation builder 中隔离并标记为 legacy。

### 8.1 与 MimicKit 的对应关系

| MimicKit | 本项目 |
| --- | --- |
| `agent_builder.py` | `learning/representation.py` 中的 `build_representation()` |
| `base_agent.py` / `base_model.py` | `learning/representation.py` + `learning/runner.py` |
| `learning/nets/` | `learning/nets/` |
| agent train/test mode | `runner.py` 的 `run(mode="train"/"validate"/"test")` |
| `run.py` config dispatch | `stylized_motion/run.py` + `build_representation()` |

MimicKit 的 taste 是“直接模块承载清晰领域对象、builder 选择具体实现、base class 定义公共生命周期”。本项目对应为“representation builder 选择 FSQ 主线，单一 runner 定义 train/validate/test 生命周期”；data 侧的对应关系在独立 spec 中定义。

## 9. 统一 Train / Validate / Test

### 9.1 单一 Runner 生命周期

四条 FSQ 都进入同一套生命周期：

```text
config + representation family
  -> build dataset / dataloader
  -> build_representation()
  -> RepresentationRunner.run(mode="train")
       train split: forward + common loss + representation loss + backward
       val split:   no_grad forward + common metrics + representation metrics
       checkpoint:  best val + last + config + representation metadata
  -> RepresentationRunner.run(mode="validate", split="val")
  -> RepresentationRunner.run(mode="test", split="test")
```

`runner.py` 内部用同一套 `fit()` / `evaluate(split, report)` 逻辑完成训练、验证和测试。训练每个 epoch 自动执行 validation；独立 validate/test 只改变 split 和报告策略。验证和测试都不能更新模型参数或 normalization stats。

### 9.2 统一 Runner API

```python
runner = RepresentationRunner(
    representation=representation,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    loss_fn=compute_losses,
    metric_suite=metric_suite,
    checkpoint_manager=checkpoint_manager,
    config=config,
)
runner.run(mode="train")
runner.run(mode="validate", split="val")
runner.run(mode="test", split="test")
```

`RepresentationRunner` 统一处理：device、DataParallel、seed、AMP/FP32 policy、optimizer、scheduler、gradient clipping、logging、TensorBoard、resume、best/last checkpoint 和吞吐量统计。四个 representation 只提供模型构造、forward 输出和 representation-specific loss/metric hooks。

### 9.3 统一配置

训练配置使用同一层级：

```yaml
representation:
  family: part_fsq
  variant: hierarchical
  config: {...}

data:
  feature_database: data/processed/100style_pruned/feature_database
  required_data_schema_version: 3

sampling: {...}  # defined by data_pipeline_spec.md
loader: {...}    # defined by data_pipeline_spec.md

training:
  epochs: 100
  lr: 0.0002
  precision: fp32

evaluation:
  metrics_interval: 100
  root_dt: 0.016666666666666666
```

旧配置不属于当前 runner API；新配置必须显式提供 representation family/variant，
不通过文件名推断语义。`data/sampling/loader` 三个配置区块由 data spec 定义，本文件
只要求 runner 透传并消费构造结果。

## 10. 统一入口设计

当前 `stylized_motion.run` 的 `--mode/--pipeline` 继续保留，新增统一 representation workflow：

```text
train       + representation   --representation flat-fsq
train       + representation   --representation part-fsq
train       + representation   --representation residual-part-fsq
train       + representation   --representation latent-residual-fsq
validate    + representation   --representation <family>
test        + representation   --representation <family>
preprocess  + token-database
preprocess  + validate-data --full
train       + generator
generate    + motion
visualize   + motion
```

每次 dispatch 前先解析 representation spec，再把 `--config`、`--checkpoint` 和 workflow 参数转发给统一 runner。workflow 不应再通过文件名猜 family。

## 11. Draft 0.4 实施计划

Draft 0.3 的 representation、checkpoint、TokenStore 和统一 runner 主线视为既有基础。
数据侧按照 [`data_pipeline_spec.md`](data_pipeline_spec.md) 的实施顺序独立落地；本文件
只跟踪以下跨层工作：

1. `run.py` 成为 data/learning composition root，并注入 token encoder；
2. representation loss 与 metric 接受 data batch 的 mask contract；
3. runner 只消费统一 loader 输出，并保证每 step 单次 device transfer；
4. generator 只消费 TokenStore/Dataset/Loader public API；
5. checkpoint、representation metadata 与 data Store contract 做集成校验；
6. README、args 和 config 同时引用两份 spec 的各自配置边界。

验收：data package 与 learning package 依赖方向正确，四条 representation 和 generator
都通过统一 data public API 完成真实数据 smoke workflow。

## 12. 测试与验收标准

必须新增或改造以下测试：

- representation builder 覆盖四条 canonical FSQ 主线；
- 四条主线的 `[B, 64, 230] -> [B, 64, K] -> [B, 64, 230]` functional roundtrip；
- 四条主线均验证 `lookahead_frames == 0` 和 receptive field；
- 四条主线 checkpoint metadata 的 family、variant、layout、feature schema 完整；
- canonical `model_family_legacy` 字段与 family 的序列化一致性；
- token database 与 checkpoint hash/family/layout 不匹配时 fail fast；
- generator 不硬编码 20 或 40 坐标；
- `run.py` 的 train/validate/test canonical pipeline 都能 dispatch 到同一个 runner；
- 任何核心模块不再从根目录旧路径或具体 legacy wrapper 导入；
- data package 不导入 learning，token encoder 只通过 `run.py` 注入；
- runner 和 generator 能消费 data spec 定义的 canonical batch，不重新解释 Store/split；
- loss/metric 正确消费 mask，metric 聚合不依赖 batch size，每个 tensor 每 step 恰好一次 device transfer；
- data 内部验收完整通过 [`data_pipeline_spec.md`](data_pipeline_spec.md) 第 14 节。

## 13. 暂不做的事情

- 不在本阶段重新设计 motion feature 230D schema；
- 不把四条 FSQ 强行合并为一个内部网络；
- 不在没有 token metadata contract 的情况下接入新的 generator architecture；
- 不删除已有 output、processed data 或旧 checkpoint；
- 不把 VQ-VAE 扩展成新的 FSQ representation 主线；
- data pipeline 非目标见 [`data_pipeline_spec.md`](data_pipeline_spec.md) 第 2 节。

## 14. 已确认决策

1. Flat-FSQ、Residual Part-FSQ 和 Latent Residual-FSQ 都固定使用 `40 x 9`；`20 x 9` 不进入当前 TokenStore contract。
2. 四条 FSQ 统一使用 `temporal_downsample=1`、`receptive_field=64`、`lookahead_frames=0`；训练 window/supervision 由 data spec 定义。
3. generator 通过 TokenStore metadata 动态构造 autoregressive token model；conditional generator 通过 TrajectoryStore 与统一 conditional batch contract 接入 trajectory conditioning。
4. VQ-VAE baseline 配置继续留在 `data/configs/`，但不进入 FSQ representation builder；canonical generator 配置独立于 representation 训练配置。
5. data schema、sampling、batch、loader、目录和兼容策略以 [`data_pipeline_spec.md`](data_pipeline_spec.md) 为唯一权威来源。

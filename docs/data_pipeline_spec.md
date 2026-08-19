# Stylized Motion Generation

## Data Pipeline Spec

状态：Draft 0.3
日期：2026-08-03
依赖：`docs/refactor_spec.md` Draft 0.4
范围：offline data build、Store、Dataset、sampling、collate、DataLoader 和训练数据交付

本文件是数据侧的唯一权威 spec。FSQ family、representation metadata 和 checkpoint
语义由主重构 spec 定义；主 spec 只引用本文件，不重复定义 data schema、sampling 或
loader policy。

执行口径：checkpoint schema 保持 v2；FeatureStore、TokenStore 和 TrajectoryStore
只接受 data schema v3。data schema v2 不兼容读取，不提供迁移 wrapper，已有 processed
data 必须通过 canonical preprocess 重新构建。

## 1. 目标

数据管线首先服务于模型训练：保证监督语义正确、采样有效、吞吐可测量，然后才考虑
更复杂的 storage backend。

必须满足：

- feature、token、trajectory 共用一个 source-clip split manifest；
- train 使用 epoch-aware random crop，val/test 使用固定窗口；
- causal representation 使用 64-frame window；每个 frame 只访问当前帧及其窗口内历史；
- generator 每个 sample 提供 64 个 next-token target；
- 默认 batch 只包含 contiguous tensor，不包含字符串、路径或 Python object；
- runtime 使用 lazy mmap shard cache，不在 Store/Dataset 构造时打开或扫描全部 shard；
- representation 与 generator 共用 sampling、loader 和 worker policy；
- 同一 seed/epoch/world size 可复现，DDP rank 消费互不重叠的全局 sample sequence position；
- loader 的内存占用、data wait 和吞吐可观测；
- data package 保持 MimicKit 风格的平级 direct modules。

## 2. 非目标

- 不重新设计 230D motion feature schema；
- 不定义 FSQ family、coordinate layout、loss 公式或 generator 网络；
- 不在本阶段引入 WebDataset、LMDB、HDF5、数据库服务或远程 object storage；
- 不让 data layer 管理 optimizer、AMP、backward、checkpoint 保存或 model device；
- 不保留 Draft 0.3 data module 路径或 data schema v2 reader；
- 不自动删除旧 processed data、output 或 checkpoint。

## 3. 实现边界

### 3.1 目标模块

```text
stylized_motion/data/
  __init__.py
  feature_data.py
  token_data.py
  trajectory_data.py
  sampling.py
  loader.py
  preprocess.py
```

禁止增加 `datasets/`、`samplers/`、`loaders/`、`stores/` 或 `builders/` 子包。
禁止继续保留 `feature_dataset.py`、`token_store.py`、`trajectory_store.py` 和多个
`build_*.py` workflow 模块。

### 3.2 模块所有权

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `feature_data.py` | FeatureStore、FeatureDataset、feature schema v3、motion slice | BVH 解析、split 生成、DataLoader |
| `token_data.py` | TokenStore、TokenDataset、token schema v3、index slice | representation 构造、next-token loss |
| `trajectory_data.py` | TrajectoryStore、ConditionalTokenDataset、normalization、对齐读取 | generator 架构、split/sampler |
| `sampling.py` | SplitManifest、SampleRequest、train/fixed sampler、sampling weights | shard I/O、collate、device transfer |
| `loader.py` | dataset/sampler 装配、batch read/collate、worker seed、pin/prefetch/drop-last | model、loss、optimizer、device transfer |
| `preprocess.py` | 数据库构建、写入、full validation、CLI 子命令实现 | representation registry、训练 lifecycle |
| `data/__init__.py` | 稳定 public exports | 业务逻辑、兼容 alias |

### 3.3 依赖方向

```text
anim + util
    ^
    |
data  <--------- learning
  ^                 ^
  |                 |
  +------ run.py ---+
```

规则：

- `stylized_motion.data` 可以导入标准库、NumPy、PyTorch、`stylized_motion.anim` 和 `stylized_motion.util`；
- `stylized_motion.data` 禁止导入 `stylized_motion.learning`；
- `stylized_motion.learning` 只通过 `stylized_motion.data` public API 消费 CPU batch；
- `stylized_motion.run` 是 composition root，可以同时导入 data 与 learning；
- token database 编码由 `run.py` 构造 representation encoder，再以 Protocol/callback 注入 `preprocess.py`；
- viewer/realtime 可以读取 Store，但不得修改 split、normalization 或 sampling contract。

### 3.4 跨层 Protocol

data layer 只要求 token encoder 满足最小注入接口，不认识具体 FSQ class：

```python
class TokenEncoderProtocol(Protocol):
    num_coordinates: int
    num_levels: int

    def encode_to_codes(self, motion: Tensor) -> tuple[Tensor, Tensor]: ...
    def representation_metadata(self) -> Mapping[str, object]: ...
```

`run.py` 负责 checkpoint 加载、feature schema 校验和 Protocol 注入；`preprocess.py`
只调用接口并写 TokenStore。

## 4. Data Schema v3

### 4.1 物理布局

三种 Store 统一使用：

```text
<store>/
  manifest.json
  index.npz
  <typed-shard-dir>/
    shard_00000.npy
    ...
```

- `manifest.json` 保存 schema version、字符串、相对路径、feature/representation metadata 和 hash；
- `index.npz` 只保存定长 numeric/bool arrays，必须使用 `allow_pickle=False` 读取；
- shard 使用 uncompressed `.npy`，允许 `mmap_mode="r"`；
- manifest 内路径必须相对 Store 根目录，禁止依赖构建机器的绝对路径；
- JSON hash 使用 UTF-8、sorted keys、紧凑 separators 的 canonical serialization；
- 所有 frame interval 使用半开区间 `[start, stop)`。

### 4.2 公共 manifest 字段

```yaml
data_schema_version: 3
store_type: feature | token | trajectory
frame_rate: 60
num_shards: 0
shard_files: []
shard_sha256: []
split_manifest_hash: ...
feature_schema_hash: ...
created_by: stylized_motion.data.preprocess
```

`created_at`、本机绝对路径等不稳定字段不得参与 contract hash。

### 4.3 公共 index 字段

```text
shard_num_frames       int64 [S]
clip_ids               int32 [R]
source_clip_ids        int32 [R]
range_shard_indices    int32 [R]
range_starts           int64 [R]
range_stops            int64 [R]
range_mirror           bool  [R]
split_ids              uint8 [R]    # train=0, val=1, test=2
```

字符串 clip/style/action 表放在 manifest；index 只保存对应 integer id。source clip 及
其 mirror 的 `split_ids` 必须相同。

### 4.4 FeatureStore

FeatureStore manifest 额外包含：

```text
motion_dim: 230
joint_subset
names
parents
names_sha256
stats_sha256
feature_schema_hash
```

FeatureStore index 额外包含 normalization 的 `offset`、`scale`、`weights`、`ref_pos`。
这些统计量只由 train split frame 计算。motion shard dtype 固定为 float32，shape 为
`[N, 230]`，写入前完成 normalization。

### 4.5 TokenStore

TokenStore manifest 额外包含：

```text
representation_family
representation_variant
representation_id
model_family_legacy
checkpoint_sha256
num_coordinates: 40
num_levels: 9
coordinate_order
coordinate_counts
temporal_downsample: 1
receptive_field: 64
lookahead_frames: 0
decoder_passes_inference
```

index shard dtype 固定为 uint8，shape 为 `[N, 40]`。可选 code shard 使用 float16，
shape 同为 `[N, 40]`。TokenStore 必须逐项继承 FeatureStore 的 clip/range/split index，
不得重新生成 split。

### 4.6 TrajectoryStore

TrajectoryStore manifest 额外包含：

```text
trajectory_dim
future_frames
feature_order
checkpoint_sha256
normalization_valid_frames
```

trajectory shard 为 float32 `[N,C]`，valid shard 为 bool `[N]`。TrajectoryStore 必须
逐项继承 TokenStore 的 clip/range/split index；normalization 只使用 train split 中
valid frame。

### 4.7 校验层级

`runtime` 校验：

- schema version、store type 和 required fields；
- manifest/hash/representation/feature contract；
- 文件存在、npy header、dtype、rank 和 shape；
- split index、range bounds 和 shard frame count；
- 不读取全部 shard value。

`full` 校验：

- runtime 全部项目；
- finite value、token range、normalization finite/std positive；
- 全 shard checksum；
- feature/token/trajectory frame-level alignment；
- split 无 source clip 交集；
- train-only statistics coverage。

full 校验只在 database build 完成后或 `preprocess validate-data --full` 显式执行。

## 5. Split Contract

canonical 配置：

```yaml
split:
  policy: source_clip
  algorithm_version: 1
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 3407
  stratify_keys: [style, action]
```

规则：

1. `source_clip_id` 是 split 原子，mirror/augmentation 不得跨 split；
2. 使用 seed 驱动的稳定 hash，不依赖文件枚举顺序；
3. 有 label 时做 grouped stratification，使 style/action 分布尽量接近；
4. manifest 记录 policy、ratio、seed、stratify keys 和最终 source clip id 列表；
5. manifest 记录 `split_algorithm_version`，相同 version/config/source ids 必须生成相同结果；
6. `split_manifest_hash` 由 canonical manifest 计算；
7. FeatureStore 创建 split，TokenStore/TrajectoryStore 只能继承；
8. train stats、trajectory stats 和 sampling weights 只能读取 train split。

canonical benchmark 禁止同一 source clip 的窗口跨 split。实验性 within-clip split 必须
使用不同 policy 名称并保持 source clip 边界隔离，不得覆盖 canonical manifest。

## 6. Sampling Contract

### 6.1 SampleRequest

```python
@dataclass(frozen=True)
class SampleRequest:
    shard_idx: int
    target_start: int
    target_frames: int
    variant_idx: int
```

Sampler 只返回 request，不读取 shard。Dataset 只消费 request，不自行随机选择 split 或
window start。

### 6.2 TrainWindowSampler

```yaml
sampling:
  strategy: frame_uniform
  target_frames: 64
  samples_per_epoch: 100000
  seed: 3407
  mirror_probability: 0.5
  balance_key: null
```

- 不 materialize 全部 train window；
- `frame_uniform` 按 source clip 的有效 target start 数量加权，再均匀采样 start；
- 先采样 source clip group，再按 mirror probability 选择 variant；
- `group_balanced` 可按 style/action/label 平衡，但必须记录最终 sampling weight；
- sampling 允许 replacement，epoch 长度只由 `samples_per_epoch` 决定；
- `(base_seed, epoch)` 生成带稳定 ordinal 的全局 request sequence，再按 rank 确定性分片；
- rank 之间的 ordinal 不重叠；sampling with replacement 导致的 request value 偶然碰撞不属于分片错误；
- request sequence 不随 `num_workers` 变化；
- augmentation 使用独立 `(base_seed, epoch, rank, worker_id)` 随机流；
- runner 每个 epoch 必须调用 `set_epoch(epoch)`；resume 后 sample sequence 可复现。

### 6.3 FixedWindowSampler

- val/test 禁止 replacement；
- 默认 target length 64、stride 64；
- 尾部默认丢弃；`include_tail=true` 只纳入仍满足完整 64-frame contract 的尾部窗口；
- 派生 index 使用紧凑 `int32/int64 [N,4]` array，不构造 Python dataclass list；
- 相同 Store/manifest/config 必须得到相同 sample 数量和顺序；
- training workflow 不得读取 test sampler 结果来选择 checkpoint。

## 7. Batch Contract

### 7.1 Representation

```text
motion     float32 [B,64,230]
loss_mask bool    [B,64]
```

representation batch 不携带额外 context，64 帧全部是 supervised target，默认
`loss_mask` 全为真。模型保持 `receptive_field=64`、`lookahead_frames=0`，第 `t` 帧
只能访问窗口内的 `max(0,t-63)...t`。窗口外历史由模型内部 causal zero padding 提供，
禁止从其他 clip/split 读取。

representation history frames 不作为独立字段保存，由
`receptive_field - 1 - lookahead_frames` 推导。canonical contract 的 history frames 为
63。窗口外历史由模型内部 causal zero padding 提供。

### 7.2 Generator

```text
indices uint8 [B,65,40]
```

`indices[:, :-1]` 是 64-frame input，`indices[:, 1:]` 是 64-frame next-token target。
uint8 保持到 host-to-device copy 完成，再在 device 上转换 long。

### 7.3 Conditional generator

```text
indices          uint8   [B,65,40]
trajectory       float32 [B,64,C]
trajectory_valid bool    [B,64]
```

trajectory frame `t` 与 `indices[:, 1:][:, t]` 对齐。

### 7.4 Metadata

默认 train/val/test batch 禁止包含：

```text
range_name, style_name, action_name, path,
mirror, shard_idx, start_idx, end_idx
```

这些字段只在 `return_metadata=true` 的 inspect/visualize workflow 返回，不进入 canonical
training collate。

## 8. Runtime I/O Contract

### 8.1 Lazy shard cache

- Store/Dataset 构造不打开全部 shard；
- 每个 worker 维护独立 LRU cache；
- `max_open_shards` 控制上限，eviction 释放 mmap reference；
- worker pickle/state 不包含 mmap handle；
- train/val/test dataset 不重复执行 full value scan；
- Store 提供显式 `close()`，进程退出不能依赖全局 singleton。

### 8.2 Batch-oriented read

Dataset 使用 `__getitems__` 或等价 batch reader：

1. 按 shard 对 request 分组；
2. 一次分配最终 contiguous CPU batch；
3. 从 mmap 直接复制到 batch 对应 slice；
4. 恢复 sampler request 顺序；
5. 不生成中间 Python tensor/object list。

### 8.3 DataLoader policy

```yaml
loader:
  batch_size: 128
  num_workers: 4
  pin_memory: auto
  persistent_workers: true
  prefetch_factor: 2
  drop_last_train: true
  max_open_shards: 32
  prefetch_memory_limit_mb: 512
```

- 所有 loader 只通过 `build_data_loaders()` 构造；
- `num_workers=0` 时不传 persistent/prefetch worker 参数；
- CUDA 才自动启用 pin memory；MPS/CPU 不沿用 CUDA 判断；
- train 默认 drop last，val/test 不 drop；
- 估算 `batch_bytes * num_workers * prefetch_factor`，超过 memory limit 时 fail fast，除非显式 override；
- 128/4/2 是保守初始 baseline，最终值由目标硬件 benchmark 决定；
- loader 返回 CPU tensor，不调用 `.to(device)`。

统一装配 API：

```python
def build_data_loaders(
    kind: Literal["representation", "generator", "conditional_generator"],
    store: FeatureStore | TokenStore,
    *,
    trajectory_store: TrajectoryStore | None = None,
    sampling_config: Mapping[str, object],
    loader_config: Mapping[str, object],
    rank: int = 0,
    world_size: int = 1,
) -> DataLoaders: ...
```

`DataLoaders` 暴露 train/val/test loader 及对应 sampler，使 runner 可以调用
`set_epoch()`，并暴露本次装配的 `prefetch_bytes`；`kind` 只选择数据 contract，不允许接收
representation family。

### 8.4 Typed read API

Store 的 runtime read 只接受已校验的 `SampleRequest`：

```python
FeatureStore.read_motion(request) -> np.ndarray
TokenStore.read_indices(request, sequence_frames) -> np.ndarray
TrajectoryStore.read_aligned(request, target_frames) -> tuple[np.ndarray, np.ndarray]
```

返回 array 的 shape/dtype 由第 7 节规定。Store 不做随机采样、不返回 label string，
Dataset 不绕过 typed read 直接访问内部 mmap list。

## 9. Learning Integration Contract

### 9.1 Runner

`learning/runner.py` 负责而 data layer 不负责：

- 调用 `sampler.set_epoch(epoch)`；
- 使用一次 `move_batch_to_device()`；
- model、loss、metrics 共用同一个 device batch；
- CUDA uint8 token 在 transfer 后转换 long；
- batch mean metric 以 numerator/valid-count 聚合；
- DDP all-reduce numerator 与 denominator；
- 报告 samples/s、target_frames/s、data wait time 和 step time。

### 9.2 Loss 与 metric

所有 representation common loss 和 representation-specific loss 必须接受 `loss_mask`。
禁止先对完整 `[B,64]` 做 mean 再忽略无效 frame。metric 的有效计数以 mask 中 true frame
数量为准，不以 batch 数为准。

### 9.3 Generator

`learning/generate.py` 只消费 TokenStore/Dataset/Loader public API，不直接实例化
DataLoader，不解释 split index，不打开 token shard。

## 10. Preprocess Contract

`data/preprocess.py` 暴露统一子命令实现：

```text
motion-database
feature-database
token-database
trajectory-inputs
trajectory-database
validate-data
```

写入流程使用 staging directory；所有 shard、manifest、index 和 full validation 成功后
再原子发布最终 Store。失败不得留下看似完整的 `manifest.json`。

构建 FeatureStore 时创建 split manifest 并拟合 train stats；构建 TokenStore 和
TrajectoryStore 时继承 manifest。`run.py` 负责 CLI dispatch 与 TokenEncoderProtocol
注入。

`feature-database --dataset combined` 联合构建 LAFAN 和 100style。LAFAN 使用完整 source
clip，100style 使用 `Frame_Cuts.csv` 中的半开区间 `[START, STOP)`；两个数据集共同参与
source-clip split 和 train normalization。combined range name 使用
`lafan/<clip>` 或 `100style/<style>_<clip>` 前缀。LAFAN 的 style/action taxonomy 暂不从
文件名推断，使用 `__unknown__` 占位并作为后续 TODO。

```bash
python -m stylized_motion.run \
  --mode preprocess --pipeline feature-database \
  --dataset combined \
  --output data/processed/combined_pruned/feature_database \
  --prune-ends-and-fingers \
  --workers 1 \
  --overwrite
```

`feature-database` 默认不生成 raw `database.npz`。需要 trajectory preprocessing 时，显式传入
`--motion-database-output <path>`，或单独运行 `motion-database`。

## 11. Public API

`stylized_motion.data` 只导出：

```python
FeatureStore
FeatureDataset
TokenStore
TokenDataset
TrajectoryStore
ConditionalTokenDataset
SplitManifest
SampleRequest
TrainWindowSampler
FixedWindowSampler
DataLoaders
build_data_loaders
open_feature_store
open_token_store
open_trajectory_store
```

offline builder helper、manifest parser internal type、shard cache 和 collator 实现不导出。

## 12. 配置边界

```yaml
data:
  feature_database: data/processed/100style_pruned/feature_database
  required_data_schema_version: 3

sampling:
  strategy: frame_uniform
  target_frames: 64
  samples_per_epoch: 100000
  seed: 3407
  mirror_probability: 0.5

loader:
  batch_size: 128
  num_workers: 4
  pin_memory: auto
  persistent_workers: true
  prefetch_factor: 2
  drop_last_train: true
  max_open_shards: 32
  prefetch_memory_limit_mb: 512
```

- `data` 只描述 Store 与 schema requirement；
- `sampling` 只描述 request 分布与 sequence；
- `loader` 只描述 batching/worker/host memory；
- `training` 中不再保存 batch size、worker 或 prefetch 参数；
- representation/generator model config 不得覆盖 data/sampling/loader contract。

## 13. 实施顺序

### Phase 0：正确性

- 修复 runner 重复 device transfer；
- 修复 metric numerator/denominator 聚合；
- loss/metric 接入 mask；
- generator 改为 65-frame token sample。

### Phase 1：schema 与平级模块

- 创建七个目标模块（含 `__init__.py`）和 public exports；
- 实现 manifest.json + numeric index.npz data schema v3；
- 删除旧 data modules，不保留 wrapper；
- 建立 runtime/full validation。

### Phase 2：split 与 sampling

- source-clip grouped stratified split；
- train/fixed sampler；
- representation 使用 64-frame causal window；generator 使用 65-frame token sample；
- DDP rank partition 与 resume reproducibility。

### Phase 3：runtime I/O 与 loader

- lazy LRU mmap cache；
- batch-oriented read；
- unified loader policy；
- prefetch memory estimate 与 timing metrics。

### Phase 4：重建与 benchmark

- 重建 canonical FeatureStore 和四条 TokenStore；
- full validation；
- 真实数据 smoke epoch；
- 扫描 batch/worker/prefetch，按目标硬件固化 baseline；
- 更新 README、args 和 configs。

## 14. 验收标准

- 目标 data 目录与第 3.1 节完全一致，旧模块路径不存在；
- `data` package 的 import graph 不包含 `stylized_motion.learning`；
- manifest JSON 不含绝对路径，index NPZ 可用 `allow_pickle=False` 读取；
- 三种 Store data schema v3、split manifest 和 frame index 完全对齐；
- source clip/mirror 不跨 split，train/val/test source id 交集为空；
- train request 跨 epoch 变化，相同 seed/epoch/world size 可复现；
- request sequence 不随 num_workers 变化，不同 rank 的 global sequence ordinal 不重叠；
- representation batch 为 `[B,64,230]`，64 帧全部进入 loss/metric；
- generator batch 为 uint8 `[B,65,40]`，产生 64 个 target；
- conditional trajectory 与 64 个 next-token target 对齐；
- canonical batch 不包含字符串、路径或调试 object；
- Store/Dataset 构造不打开全部 shard，不执行 full value scan；
- 每 worker 打开 shard 数不超过 `max_open_shards`；
- metric 结果不依赖 batch size，每个 tensor 每 step 只 transfer 一次；
- loader 输出 prefetch bytes、data wait、step time、samples/s、target_frames/s；
- full validation、完整测试和真实数据 smoke epoch 通过。

## 15. 已确认决策

1. data schema v3 使用 `manifest.json + index.npz + mmap .npy shards`。
2. data package 不依赖 learning；token encoder 由 composition root 注入。
3. source clip 是 canonical split 原子，mirror 不改变 source clip sampling weight。
4. train random crop，val/test fixed windows。
5. representation 使用 64-frame causal window；generator 使用 65 token frame。
6. default batch tensor-only；token 在 CPU 保持 uint8。
7. DataLoader 只由 `data/loader.py` 构造，实际 device transfer 归 learning runner。
8. 先优化 mmap pipeline，不提前引入新的 storage backend。

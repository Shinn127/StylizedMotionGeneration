# BONES-SEED 数据管线迁移方案

日期：2026-09-15。本文为代码与本地数据审计后的实施规划，未修改训练代码或启动全量预处理。

## 1. 结论

保留现有 NumPy mmap、按请求批量组装张量、DataLoader 预取和 CPU→GPU 异步传输；重点改造数据接入、物理分片、逻辑索引、采样和预处理恢复机制。当前规模适合单机 NVMe 上的分片二进制数据，不需要先引入分布式存储系统。

首期使用 soma_uniform，保持现有 60 fps、SOMA pruned 27 joints / 248D 契约。默认保留官方原始与镜像文件，每个文件只提取一次特征；镜像关系作为元数据，不再自动翻倍。

## 2. 本地规模与证据范围

已遍历文件并统计大小，读取完整 CSV，检查全部 CSV 动作路径存在性，随机抽查 128 个 BVH 文件头；没有解析全量动作或测量训练吞吐。

| 项目 | 实测或估算 |
|---|---|
| 原始数据路径 | `data/raw/seed` 软链接至 `/home/shinn/桌面/Projects/data/seed` |
| soma_uniform BVH | 142,220 个，276.28 GiB |
| 对比 100style BVH | 810 个，26.71 GiB；SEED 文件数约 176 倍，体积约 10.3 倍 |
| CSV | 142,220 行，所有引用的 uniform 动作文件存在 |
| 原始 / 镜像 | 71,132 / 71,088 个 |
| 原始动作配对 | 44 个找不到对应 `_M` 路径；7,653 对帧数不同，其中 7,645 对镜像少 1 帧、8 对多 1 帧 |
| 帧率抽查 | 128/128 为 `Frame Time: 0.008333`，78 个 ROOT/JOINT 节点；抽查帧数均与 CSV 一致 |
| 原始动作时长 | 按元数据和 120 fps 估算约 144.22 小时 |
| 含官方镜像时长 | 同上约 288.32 小时 |
| 60 fps 帧数 | 假定全体均为 120 fps 且沿用 `[::2]`：62,313,039 帧 |
| 248D float32 特征 | 含官方镜像约 57.57 GiB；仅原始约 28.80 GiB，均未计索引和文件头 |
| 固定 64 帧非重叠窗口 | 含镜像约 903,834 个，切分前、丢弃尾部的估算 |
| 主机 | 约 30 GiB RAM；数据与处理目录同在 NVMe 分区，可用约 1.3 TiB |

原始动作共有 522 个 actor_uid。`content_uniform_style` 中 neutral 为 65,458/71,132，约 92%；package 分布包括 Locomotion 37,260、Communication 10,749、Interactions 7,322 等。迁移后训练分布不能直接沿用 100STYLE 的标签假设。

GPU 查询未成功，因此本文不提供未经实测的训练速度或 GPU 利用率结论。帧数差异不等于动作内容不同；需要对齐后检查镜像等价性，不能直接据此认定冗余或删除数据。

## 3. 当前管线与瓶颈

主路径：BVH → feature-cache（未归一化、每个来源生成 original/mirror 两个文件）→ fsq-window-index（固定窗口划分、统计、另写归一化特征）→ FeatureStore → TrainWindowSampler / FixedWindowSampler → FeatureDataset → DataLoader → runner。

### 3.1 接入与正确性，必须先改

- `data/preprocess.py:_discover_source_clips` 与 CLI 仅支持 lafan、100style、combined，尚无 SEED 全量入口。
- `_normalize_to_60fps` 已支持 120→60；`_drop_static_rig_root` 已处理 SOMA 静态 Root；已有 SOMA 测试与 248D NEF 配置，可以复用。
- 但 `_SourceClip.start/stop` 随后用于裁剪降采样后的数组。SEED 适配器必须明确原始帧、目标帧和时间区间，不能直接把 120 fps 元数据帧数作为 60 fps stop。
- `preprocess_worker.py:process_motion_pair` 无条件生成两个版本，接入官方镜像后会重复增广：若照搬到所有文件，特征量约 115.1 GiB，cache 加归一化副本约 230.3 GiB。
- `build_fsq_window_index` 隐含 shard 偶数为原始、奇数为镜像，并随机分配同一动作内的 64 帧窗口。SEED 镜像长度不同且存在缺失，必须解除这种隐含关系。
- 随机窗口划分虽无直接窗口重叠，同一 take 的相邻动作与镜像仍可能跨集合，泛化评估会偏乐观。token 编码还需避免上下文跨 split。
- `ref_pos` 目前从整个 feature cache 聚合。新版本应明确归一化与参考骨架统计只依赖训练集，或使用独立固定参考骨架。

### 3.2 预处理

- 进程池已有最多 workers 个待处理任务，内存并非随全量线性累积；但按提交顺序等待结果，长片段会阻塞后续完成任务。
- worker 返回原始/镜像的多组大数组，主进程再裁剪、提特征和写文件，有进程间传输成本及串行瓶颈。
- 失败后删除 staging，缺乏逐任务恢复；对十几万个文件的一次性作业代价高。
- `_window_stats` 按随机 train_records 读取窗口，配合 32 项 LRU 容易反复开文件。
- `build_fsq_window_index` 实际会重写全部归一化特征，并非只建索引。改变 split 又复制一遍约 57.6 GiB。

### 3.3 训练热路径

- `FeatureDataset.__getitems__` 已一次分配整批数组，并按 shard 分组读取；保留。
- `MMapShardCache` 已是进程本地 LRU，支持 worker 序列化清空缓存；保留。mmap 不意味着数据全驻内存，LRU 也不等于 OS 页缓存大小。
- `TrainWindowSampler` 初始化对每个 source 扫描全部 interval，约 O(source 数 × interval 数)。在约 7 万 source、近百万窗口上不合适。
- 每个样本都调用带完整概率数组的 `rng.choice`；即使 clip 等概率，也付出随 source 数增长的工作。应使用整数均匀采样；带权模式预计算 CDF 或 alias table。
- FixedWindowSampler 先构造完整 Python tuple 列表，训练索引也保留重复字符串和多个窗口列表，应改成紧凑数组与区间索引。
- 开库会逐一检查文件并 mmap 读取形状；十几万个小文件会使启动很慢，即使没有读全量数据。
- DDP 已接入 rank/world_size，但每个 rank 仍生成完整采样序列后丢弃其他 rank 的请求。还需明确 global/per-rank 样本预算，保证各 rank 训练步数相等。
- runner 每 epoch 完整 validation。100,000 个训练窗口并不意味着遍历全库，评估成本与训练规模必须单独控制。

### 3.4 后续 token / trajectory 路径

- token 已分块编码，但以整个物理 shard 为序列上下文；合并多个 clip 后必须以逻辑 clip 为边界编码。
- trajectory-inputs 使用大 NPZ，拼接全局数组；trajectory-database 又保留所有 shard 的 values/valid 列表，不能照搬到全量 SEED。
- 当前 64 帧 FSQ 窗口索引不能直接作为要求 65 帧的 generator 训练区间；下一版应共享 clip/split 索引，由任务指定窗口长度。

## 4. 目标设计

### 4.1 数据登记与划分

建立 SEED adapter：以 CSV 为首期权威目录，保留 parquet 和时间标签扩展入口。输出紧凑 clip catalog，记录 dataset、相对路径、source/variant/group ID、actor、take、package、style、fps、有效帧数、骨架与预处理版本。时间标签以秒转目标帧区间，需注明端点规则和版本。

镜像策略显式配置为 `official`（首期）、`generate`、`none`，三者互斥。`official` 模式保留所有官方文件，按 group 关联但允许各 variant 有不同长度；不因缺少镜像而丢弃原始动作。

主验证集按 source/take group 划分，原始、官方镜像、同 take 派生片段必须同 split。group key 先用日期、actor、原始 take 标识构建并检查碰撞/派生关系，不能只剥离 `_M` 或只用文件 stem。可另外冻结 actor-heldout 测试集；比例按动作覆盖与预算决定，不盲目把全部演员随机拆散。旧随机窗口协议只用于历史对照。

不要沿用“最后十个 style 自动 heldout”的隐式规则。保留原始多标签字段和 unknown 值，明确 SEED 的 style/action 含义。

### 4.2 物理存储与逻辑片段解耦

首期采用约 256 MiB 的未压缩 float32 `.npy` 分片，benchmark 比较 128/256/512 MiB。当前 57.6 GiB 约对应 231 个 256 MiB 分片，远少于 14.2 万文件。

每片存 `[N,248]`，clip 表存 shard_id、offset、length、source_group、variant、split。采样器永远只在逻辑 clip 有效区间内取窗，不能跨片段连接处。物理分片优先让同 split 数据集中，片内混合不同来源，避免只按日期打包造成批次偏置。

只持久化一份未归一化特征；statistics 单独版本化。读取选定窗口后在批次 CPU 数组上归一化，和 GPU 批次归一化做实测对比。改变 split 或统计不重写原始特征。float16 存储作为后续精度/吞吐实验，不作为首期默认。

轻量 manifest 记录 schema、分片列表、hash 和构建状态；clip/range 数字索引使用可 mmap 的 `.npy`，不把重复字符串塞入窗口级 JSON。骨架/特征定义 hash、normalization hash、split hash 分开保存，checkpoint 明确绑定三者。

定义 schema v4，新 reader 保留 v3 兼容。不得通过修改 v3 manifest 数字伪装格式兼容，也不覆盖旧数据集。新目录建议 `data/processed/seed_soma_pruned_v4/`。

### 4.3 可恢复预处理

1. inventory 阶段检查 header、fps、骨架、原始/镜像配对和元数据；持久化轻量结果。
2. 按估计帧数分组调度 worker，worker 内完成解析、降采样、SOMA 处理与特征提取，写独立临时分片，仅返回小型描述记录。
3. 用完成顺序接收任务，控制在途任务数和总字节预算；限制 BLAS/OMP 线程，避免进程数乘线程数过量。
4. 每个工作单元具有稳定 ID、输入指纹、预处理版本、输出 checksum；成功原子提交，失败记入报告并可重试。恢复时复用已完成单元，改变骨架或预处理配置则失效重建。
5. 合并分片需限制临时空间；最终 manifest 只引用完整文件。发布前完整校验，不能静默把失败数据当成功。
6. 统计按 shard 顺序扫描，仅取 train 区间，以 float64 Welford/可合并统计累计；归一化规则保持与旧实现一致，单独测试数值等价性。

### 4.4 采样与加载

- 先重写 sampler 复杂度，再调 worker 数。source→variant→interval 用一次排序/分组构造紧凑 offset 表，避免逐 source 全量扫描。
- 首期以 source group 等概率，再选择官方 variant；镜像概率在有可用窗口的 variant 中执行回退，避免因多一个版本使 source 权重翻倍。
- 增加独立 `frame_uniform` 模式，按有效起点数加权，不能继续把 frame_uniform 别名映射到 clip_uniform。
- 提供 package/style 混合分布作为后续消融实验；neutral 占比很高，简单反频率会极度重复只有十几个片段的稀有类。使用权重上限、自然分布混合，并记录采样覆盖率。
- 基线先保留全局随机采样，利用大分片和现有 batch 分组读取。只有实测 I/O 等待较高时，再引入多个 shard 的活跃池和 shuffle buffer；检验边际采样概率与批次相关性。
- 采样顺序由 seed、step/epoch、rank 决定，支持恢复；DDP 明确相等步数和尾部策略。局部性优化不应牺牲可复现与 split 隔离。
- 现有 SOMA 配置 B=512、T=64、D=248：每批 motion 约 31 MiB，4 workers × prefetch 2 的队列约 248 MiB；实际还包含 worker、临时数组、页缓存与验证 loader。以 4 workers / prefetch 2 为测量起点，测试 0/2/4/8 workers。

### 4.5 训练预算与评估

引入 max_steps/steps_per_epoch、eval_every_steps、固定验证子集与周期性全量验证。当前 B=512、100,000 samples、drop_last 时每 epoch 为 195 个训练 batch；扩大数据集并不会自动扩大训练预算。

记录源动作覆盖率、训练窗口数、有效帧数与重复采样率。100,000 个 clip-uniform 样本不等于一次全量训练，正式预算根据学习曲线决定。

SOMA 使用 248D，不能直接加载 Geno 的 230D checkpoint/token/stats。已有 NEF SOMA 配置作为接入样例；其他模型 family 各自验证 layout、骨架、重建和 loss 后再开始全量训练，不能只改 motion_dim。

### 4.6 token 和轨迹

共享 clip catalog/split，token 按逻辑片段重置上下文、分块编码和落盘，并绑定 checkpoint、feature 与 normalization hash。按实际 encoder lookahead 验证分块与整段一致性。

轨迹在特征构建时顺带按 clip 输出所需根位置/朝向或轨迹通道；保留 clip 尾部 validity mask，禁止跨 clip 或 split 取未来帧。train-only 流式计算统计，取消全库 NPZ 中转。两者均支持任务恢复。

## 5. 分阶段实施与验收

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| P0 正确性 | SEED adapter、时间/帧契约、官方镜像、group split、SOMA 审计 | 代表性短/长/运动类别及缺镜像样本可构建；原始/镜像不跨 split；单位、根运动、contact、重建验证通过 |
| P1 全量基础 | v4 packed store、worker 写出、resume、流式 stats、紧凑 sampler | 1k→10k→全量元数据规模下无 source×window 扫描；中断恢复等价；换 split 不复制 features |
| P2 吞吐 | loader 与 sampler benchmark、训练预算、验证节奏 | 达到下述端到端目标，再启动正式全量训练 |
| P3 下游 | token 与 trajectory 流式构建 | 分块等价、无上下文越界、全量内存有界、checkpoint 绑定有效 |

P0/P1 是正式全量训练的前置工作；P3 在开始 generator/conditional generator 前完成。

### Benchmark 设计

- 数据档位：约 1,000 个代表性动作、10,000 个动作、全量；必须覆盖长尾长度、镜像差帧与主要 package。
- 分开测纯 sampler、纯 DataLoader、固定 GPU resident batch 的模型基线、实际端到端训练。
- 指标：构建 frames/s 和 wall time；主进程/worker RSS、系统可用内存、磁盘字节与空间峰值；开库/首 batch 时间；稳态 batch/s、数据等待 p50/p95、step time、GPU 利用率和采样覆盖率。
- 对比：v3 小集基线、packed store 全局随机、不同 worker/分片配置；有必要才测试局部性采样和半精度存储。
- 冷读与热读分别报告，不以热缓存成绩冒充磁盘性能；不主动清空整个系统 page cache影响其他作业。
- 建议门槛（目标而非测量结论）：端到端吞吐达到同模型 GPU resident batch 基线的 90% 以上，稳态数据等待占比小于 10%；全量预处理工作进程与协调器总 RSS 目标不超过 12–16 GiB，系统无持续 swap；复测 10 倍 catalog 规模不能出现近百倍初始化增长。
- 正确性验收：原始/镜像 group 不跨集合；采样不跨 clip；统计只用 train；60 fps 与奇数帧边界正确；resume 与完整构建内容一致；采样权重符合定义；DDP 步数相等；新旧小集特征/归一化在容差内一致。

## 6. 资源预算与暂缓事项

248D float32 官方全量主特征预计约 57.6 GiB。首轮为最终分片、worker 临时文件和重打包预留约 120–180 GiB，具体取决于是否同时保留旧产物和临时分片。token、轨迹、checkpoint 额外计算，不能把此数视为完整项目上限。

首期不更换存储框架、不在线解析 BVH、不预生成所有重叠窗口、不通过提高 worker 数掩盖 sampler 算法问题。保留原始软链接即可；软链接本身不是吞吐瓶颈。

待 P0 小样本确认：官方镜像与程序镜像是否逐帧等价、差一帧的时间对齐方式、全部骨架/fps 是否统一、具体训练 family，以及可用 GPU 环境。若考虑仅原始+在线镜像以节省约一半空间，须先验证旋转、速度、contact 和标签方向变换的完整等价性，再单独立项。

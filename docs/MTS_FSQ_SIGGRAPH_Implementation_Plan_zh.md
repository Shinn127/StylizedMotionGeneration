# MTS-FSQ / NEF-FSQ 面向 SIGGRAPH 的科研与代码实施方案

版本：v1.0  
日期：2026-09-16  
适用仓库：`StylizedMotionGeneration`

本文不是对原工程计划的逐条翻译，也不是一份“把所有模块都实现”的任务清单。它先回答一个科研问题：**当前 NEF-FSQ 是否需要改，改到什么程度，才有可能支撑一篇 SIGGRAPH 论文？** 然后给出一条可以在当前仓库中执行、可以随时停止、可以被实验否证的代码路线。

## 1. 先给结论

### 1.1 NEF-FSQ 不需要推倒重来，但需要做面向研究的改进

当前 NEF-FSQ 已经具备一个很适合研究局部 motion style operator 的基础：

- 40 个 FSQ scalar、9 levels、逐帧对齐；
- 13 个有明确空间所有权的 Node/Edge stream；
- 左右区域共享 family projection/FSQ/output head，同时保留 side embedding；
- encoder 和 decoder 的 ownership 是可测试的；
- 已有 `encode_to_indices()` / `decode_from_indices()`；
- 已有 TokenStore 的 canonical metadata 和跨 skeleton 校验；
- 已有 receptive-field、stream swap、局部 decode 测试。

这些性质本身就是后续论文的一部分，不应为了方便 generator 而改成一个全身 latent。

但它还有四个科研缺口：

1. **局部 token locality 不等于物理 motion locality。** 当前 decoder 的 receptive field 是 34 帧，因此一个 token 编辑会向未来传播；必须测量并明确这个传播，而不能把“逐帧 token 对齐”误写成“逐帧独立影响”。
2. **FSQ level 的顺序是否具有动作意义尚未验证。** birth-death scattering 依赖相邻 level 是有意义的邻近状态；这应由 decoder perturbation probe 证明，不能只由 FSQ 的整数编号推断。
3. **当前 tokenizer 是严格独立 stream。** 这有利于 ownership 和局部性，但不在 tokenizer 内建模全身协调。协调应由后续 graph transport 负责；如果重写 tokenizer 加 full-body fusion，会失去论文中最清楚的 locality 因果链。
4. **当前训练配置只使用重建和 delta 目标。** [nef_fsq_40x9.yaml](/Users/shinn/Documents/Projects/StylizedMotionGeneration/data/configs/nef_fsq_40x9.yaml) 将 root、joint、contact、foot-slide 等物理相关权重设为 0。作为最小表征基线可以接受，但作为风格编辑的冻结 decoder，这会使局部风格和接触/根轨迹的一致性缺少保障。

因此建议：**保留当前 40×9 / 13-stream canonical alphabet，先做审计和小幅 v1.1 改进；不要在第一轮实验中加入 cross-stream tokenizer fusion、holistic token、第二套 style tokenizer 或 temporal compression。**

### 1.2 真正值得冲 SIGGRAPH 的主张

原方案中最有潜力的 idea 不是“CTMC 很漂亮”，而是：

> **Motion style 可以被表示为作用在结构化离散动作字母表上的、跨内容复用的生成分布算子；NEF-FSQ 的空间所有权使同一个全局 style operator 可以被限制在指定身体区域和时间范围内。**

形式化为：

\[
p_s(Z\mid C,H)=\mathcal T_{s,\lambda,M}\big[p_0(Z\mid C,H)\big]
\]

其中：

- `p0` 是不接收 style 的冻结 content generator；
- `s` 是一个全局 reference style descriptor；
- `H` 是当前 motion context；
- `lambda` 是 style strength；
- `M` 是 Node/Edge 和时间组成的 hard support；
- `T` 是在 FSQ probability 上工作的 style operator。

论文要证明的是：这个算子在未见过的 style-content 组合上仍然有效，并且 NEF 的空间结构确实带来比 flat/part 表征更好的局部性、内容保持和组合控制。

CTMC / birth-death 只能作为第一种 operator parameterization。它不是论文的全部，也不能在没有对照实验时直接当成创新结论。

## 2. 当前 NEF-FSQ 的具体评估

### 2.1 已有实现的优点

当前实现中的以下事实应视为稳定边界：

- [NEFMotionAutoencoder](/Users/shinn/Documents/Projects/StylizedMotionGeneration/stylized_motion/learning/nef_fsq.py) 使用 `NEFLayout` 导出 stream 与 feature ownership；
- 13 个 stream 共享 temporal module，但通过 fold-to-batch 保持 stream 独立；
- `MotionFSQ.dequantize()` 提供 index 到 quantized code 的路径；
- [NEFLayout](/Users/shinn/Documents/Projects/StylizedMotionGeneration/stylized_motion/learning/nef_layout.py) 已持久化 stream 顺序和 coordinate count；
- [TokenStore.validate_contract](/Users/shinn/Documents/Projects/StylizedMotionGeneration/stylized_motion/data/token_data.py) 已校验 40×9、skeleton、representation id 和 layout metadata；
- [tests/test_nef_fsq.py](/Users/shinn/Documents/Projects/StylizedMotionGeneration/tests/test_nef_fsq.py) 已覆盖 round-trip、input routing、decoder ownership、causality 和 token swap 的传播范围。

这意味着新代码不应复制一套 `FSQStreamSpec` 或新建并行 tokenizer cache。应在现有接口上加只读 metadata、probe 和 style operator adapter。

### 2.2 必须改善但不应改变的部分

| 项目 | 当前状态 | 处理方式 |
|---|---|---|
| canonical token layout | 已稳定 | 不改 40×9 和 stream 顺序 |
| stream ownership | 很强 | 保留，作为局部性实验自变量 |
| temporal causality | 已明确，RF=64、lookahead=0 | 不改；把传播范围纳入指标 |
| 物理重建 | 当前只强调 recon/delta | 增加可选 staged physical loss，做 ablation |
| level ordinal meaning | 未验证 | 加 probe；只有 probe 支持时才启用 birth-death 主张 |
| cross-stream coordination | tokenizer 中没有 | 放到 graph transport，不在 NEF v1.1 内融合 |
| contacts ownership | 在 global stream | 局部腿部 style 评估必须报告 contact side effect |

### 2.3 推荐的 NEF-FSQ v1.1 目标

v1.1 不是新 representation family，而是同一 canonical alphabet 的研究增强版本：

1. layout 提供 stream graph、region mask、coordinate metadata 和 stable hash；
2. 模型公开统一 `encode_indices` / `decode_indices` aliases，并增加 `token_layout()` 只读接口；
3. 训练配置支持 physical-loss warmup，但保留 v1 baseline；
4. 新增 level perturbation、stream locality、temporal influence、FK/contact 影响报告；
5. 不允许新接口修改 index 的 shape、范围、顺序和 decoder ownership。

## 3. 研究问题和验收门槛

整个项目拆成三个可否证伪命题：

### R1：NEF 的空间所有权是否让 style support 真正可控？

比较 flat-FSQ、已有 part-FSQ 和 NEF-FSQ，在相同 operator 容量和相同生成设置下只作用 `left_arm`。验收不是“左臂看起来变了”，而是：

\[
Leak_{token}=\frac{\#\{z'_{off}\neq z_{off}\}}{\#\{z_{off}\}}
\]

以及 decode 后的 `off-target FK error`、contact violation、boundary jerk。

如果 NEF 不比其他表征更局部，论文不应把 representation-operator coupling 作为主贡献。

### R2：受限 probability operator 是否比普通 adapter 更能跨内容迁移？

至少比较：

- additive logit field；
- AdaLN/FiLM 或轻量 feature adapter；
- arbitrary row-softmax kernel；
- birth-death CTMC kernel。

训练和参数量尽量匹配。测试使用 style-content compositional split，而不是普通随机 split。

如果 CTMC 不能在少样本、未见组合或内容保持上稳定胜出，CTMC 只能作为实现细节；论文主张应收缩为 NEF 上的局部 style operator。

### R3：同一个 global style descriptor 是否能支持不同区域的不同 response？

不训练 `left_arm_style` 等局部 style embedding。测试：

- whole body；
- Node-only radius 0；
- Node+incoming Edge radius 1；
- selected frame range；
- disjoint multi-style regions。

成功标准是：style semantics 仍由 reference 决定，support 由 layout mask 决定，区域外内容保持显著优于全身注入 baseline。

## 4. NEF-FSQ 具体改进方案

### 4.1 改进 A：公开研究级 token contract

**文件：**

- `stylized_motion/learning/nef_layout.py`
- `stylized_motion/learning/nef_fsq.py`
- `stylized_motion/learning/representation.py`
- `tests/test_nef_token_contract.py`

在 `NEFLayout` 上增加只读方法：

```python
def layout_hash(self) -> str: ...
def stream_graph(self) -> tuple[tuple[str, str, str], ...]: ...
def coordinate_metadata(self) -> tuple[dict[str, object], ...]: ...
def make_region_mask(
    self,
    regions: Sequence[str],
    *,
    graph_radius: int = 0,
    frame_range: tuple[int, int] | None = None,
    length: int,
    device: torch.device | None = None,
) -> torch.Tensor: ...  # [T, 40] bool
```

`stream_graph()` 不要把 directed ownership 宣称为严格 causal graph。返回 relation type，例如：

```python
(
    ("global", "torso_node", "global_to_root"),
    ("torso_node", "head_edge", "node_to_edge"),
    ("head_edge", "head_node", "edge_to_child"),
    ("torso_node", "left_shoulder_edge", "node_to_edge"),
    ("left_shoulder_edge", "left_arm_node", "edge_to_child"),
)
```

`make_region_mask()` 的 region 语义直接复用现有 `NEF_EDIT_PARTS`。radius 0 只返回 Node stream；radius 1 返回 Node 加 incoming Edge。不要让调用者手写 coordinate slice。

在 `NEFMotionAutoencoder` 增加兼容 aliases，不改变旧接口：

```python
def get_token_layout(self) -> NEFLayout:
    return self.layout

@torch.no_grad()
def encode_indices(self, motion, *, lengths=None):
    return self.encode_to_indices(motion)

@torch.no_grad()
def decode_indices(self, indices, *, lengths=None):
    return self.decode_from_indices(indices)
```

`lengths` 第一版只用于验证和未来 padding API；不要在没有 mask 语义时静默截断数据。

**测试：** layout completeness、stable hash、region radius、左右不串、encode/decode alias 与旧接口 bitwise 一致。

### 4.2 改进 B：增加物理质量的 staged objective

当前配置已经通过 `compute_motion_reconstruction_losses()` 支持多个物理项，因此优先复用现有 loss，不新增一套 loss registry。

新增配置：

```yaml
training:
  objective_variant: recon_delta_physical_warmup
  delta_weight: 3.0
  root_pos_weight: 0.05
  root_rot_weight: 0.05
  joint_weight: 0.10
  contact_weight: 0.03
  foot_slide_weight: 0.05
  foot_height_weight: 0.02
  physical_warmup_epochs: 10
  physical_ramp_epochs: 20
```

训练开始时仍使用当前 `recon + 3*delta`，然后线性增加 physical terms。目的不是让 tokenizer 直接学习 style，而是让冻结 decoder 在 token 编辑后更少出现根轨迹、脚滑和接触异常。

必须保留两个独立 checkpoint：

- `nef_fsq_v1_recon_delta`：原始基线；
- `nef_fsq_v1_1_physical`：物理增强版。

如果 v1.1 改善 FK/foot metrics，但显著损害 token locality 或 level usage，则不作为主模型，只作为 ablation。不能用更强的物理 loss 直接掩盖 representation locality 下降。

### 4.3 改进 C：做 level geometry probe，不先加正则

**文件：** `stylized_motion/learning/nef_probe.py`、`scripts/probe_nef_geometry.py`、`tests/test_nef_probe.py`。

对真实 token `z`，逐个 coordinate 做合法的 `+1/-1` perturbation，解码并计算：

```text
decoded feature L1/L2
FK joint position change
root trajectory change
foot contact change
temporal velocity / jerk change
```

报告：

```text
adjacent_distance[k]
far_distance[k]
adjacent_to_far_ratio[k]
direction_consistency[k]
```

如果相邻 level 的动作影响明显小于随机远距离 level，才把 birth-death 作为主 operator。否则：

- 先使用 arbitrary bounded kernel 作为科学基线；
- 研究“FSQ coordinate identity”而不是“ordinal adjacent level”；
- 不对外宣称 scalar level 的动作语义。

第一轮不要贸然加入 ordinal regularizer。先知道现有 tokenizer 的几何事实，再决定是否需要重训。

### 4.4 改进 D：保留 independent tokenizer，把协调交给 graph transport

当前 NEF 的每个 stream 在 tokenizer 内独立编码/解码。这是局部性最干净的证据来源。不要在 v1.1 中加入 full-body latent 或 unrestricted cross-stream fusion。

全身协调放在后续 generator：

```text
13 stream token embeddings
        ↓
temporal attention per stream
        ↓
local graph message passing
        ↓
40×9 output head
```

如果 tokenizer 内部加入 cross-stream context，单个 input stream 的变化会改变其他 stream token，之后无法判断 locality 是 representation 还是 coupling 学出来的。

## 5. 代码目录和接口设计

建议新增目录，但复用现有 representation/data/checkpoint 体系：

```text
stylized_motion/learning/mts_operator/
├── __init__.py
├── contract.py          # Tensor shapes, dataclasses, checkpoint metadata
├── layout_adapter.py    # thin adapter over NEFLayout; no duplicated slices
├── embeddings.py        # 13-stream token embedding
├── graph.py             # relation-indexed graph message passing
├── masking.py           # random/stream/span/block/full masks
├── transport.py         # style-free masked content generator
├── style_encoder.py     # global reference style descriptor
├── operators.py         # logit baseline and probability-kernel operators
├── sampling.py          # inverse-CDF and common-random-number sampling
├── pairs.py             # style pair audit and split-safe sampler
├── metrics.py           # locality, preservation, style and physics metrics
├── model.py             # top-level wrapper
└── checkpoint.py        # tokenizer/layout fingerprint validation

scripts/
├── probe_nef_geometry.py
├── audit_style_pairs.py
├── train_mts_transport.py
├── train_mts_operator.py
├── generate_mts_operator.py
└── evaluate_mts_operator.py

tests/
├── test_nef_token_contract.py
├── test_nef_probe.py
├── test_mts_contract.py
├── test_mts_masking.py
├── test_mts_transport.py
├── test_mts_operators.py
├── test_mts_sampling.py
├── test_mts_pairs.py
└── test_mts_end_to_end.py
```

不建议立即把这些脚本全部注册到 `stylized_motion/run.py`。先以独立 script 验证，接口稳定后再接入现有 CLI router。

### 5.1 Tensor contract

统一使用：

```text
tokens              LongTensor [B, T, 40], values 0..8
visible_mask        BoolTensor [B, T, 40]
style_mask          BoolTensor [B, T, 40]
stream_hidden       FloatTensor [B, T, 13, D]
base_logits         FloatTensor [B, T, 40, 9]
style_embedding     FloatTensor [B, Ds]
styled_probs        FloatTensor [B, T, 40, 9]
```

padding 不能用合法 FSQ level 0 伪装。第一版使用 explicit `valid_mask: [B,T]`，所有 CE、pooling、style metrics 都必须应用它。

### 5.2 `TransportOutput`

```python
@dataclass
class TransportOutput:
    stream_hidden: torch.Tensor       # [B,T,13,D]
    logits: torch.Tensor              # [B,T,40,9]
    valid_mask: torch.Tensor | None
```

`MotionTransportTransformer` 不接 style 输入：

```python
def forward(
    self,
    tokens: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    content_condition: object | None = None,
    valid_mask: torch.Tensor | None = None,
) -> TransportOutput:
    ...
```

第一版 content condition 可以复用仓库现有 trajectory/text adapter；如果没有可靠 text encoder，先使用 action/trajectory condition，不能把新文本模型混入 operator 贡献。

### 5.3 Style encoder

```python
class GlobalStyleEncoder(nn.Module):
    def forward(
        self,
        reference_tokens: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, Ds]
        ...
```

输入仍经过 13-stream embedding 和小型 temporal/graph encoder，最后使用 masked mean/std pooling。不要输出 region-specific style vector。

## 6. Operator 的最小科学实现

### 6.1 对照一：Additive logit field

这是必须实现的 baseline：

```python
delta = delta_head(torch.cat([stream_context, style_context], dim=-1))
styled_logits = base_logits + strength * hard_mask[..., None] * visibility[..., None] * delta
styled_probs = styled_logits.softmax(dim=-1)
```

初始化 `delta_head` 为零，使 strength=0 严格回到 base。

### 6.2 对照二：bounded arbitrary kernel

让每个 coordinate 预测一个 row-stochastic kernel，作为 CTMC 的表达能力对照：

```python
kernel_logits = kernel_head(context).reshape(B, T, 40, 9, 9)
kernel = kernel_logits.softmax(dim=-1)
styled_probs = torch.einsum("btki,btkij->btkj", base_probs, kernel)
```

它不具备 identity-at-zero 和 semigroup 结构，正好用来检验这些限制是否真的有益。

### 6.3 主方法：birth-death CTMC

只在 level 相邻位置放 rate：

```text
up_rate[i]   >= 0 for i < 8
down_rate[i] >= 0 for i > 0
Q[i, i+1] = up_rate[i]
Q[i, i-1] = down_rate[i]
Q[i, i]   = -sum(off_diagonal row i)
```

用 bounded sigmoid：

```python
rate = max_rate * torch.sigmoid(raw_rate)
rate = rate * strength[..., None, None]
rate = rate * hard_mask[..., None, None]
```

**第一版不使用 universal basis。** 只有 direct CTMC 与 logit baseline 通过跨内容测试后，才增加 basis ablation。

固定 `Q` 时可以用 `torch.matrix_exp` 作为 reference；批量优化用自适应 uniformization：

1. `nu = max(-diag(Q))`；
2. `mu = nu * lambda`；
3. 根据 Poisson tail 选择最小 `N`，而不是固定 6；
4. 如果 `nu == 0`，直接返回 `p0`；
5. 对每次迭代检查质量误差和非负性。

不要把截断后的结果直接称为严格概率守恒；测试要求 `abs(sum-1) < tolerance`，并记录截断尾概率。

### 6.4 hard mask 与 visibility 的数学约束

```python
effective_rate = strength * hard_mask * visibility * raw_rate
```

必须满足：

```python
hard_mask == 0 -> Q == 0 -> styled_probs == base_probs
```

visibility 不能突破 hard mask。完整 iterative generation 若要保证区域外 token 不变，应提供 `locked_edit=True` 模式：先固定区域外 token，不重新采样，而不是只依赖当轮 operator mask。

## 7. 数据、训练和实验实施顺序

### Phase 0：冻结基线并做 NEF 研究审计（2–3 周）

**新增代码：**

- `nef_probe.py`
- `scripts/probe_nef_geometry.py`
- `scripts/evaluate_nef_locality.py`
- `tests/test_nef_probe.py`

**运行：**

```bash
python scripts/probe_nef_geometry.py \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --split test \
  --max-clips 256 \
  --output outputs/nef_probe/v1

python scripts/evaluate_nef_locality.py \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --split test \
  --parts left_arm right_arm left_leg right_leg \
  --output outputs/nef_locality/v1
```

**退出标准：**

- round-trip 与现有 tests 通过；
- stream token swap 的 off-target features 仍 exact zero；
- 报告 decoder temporal influence；
- level adjacent/far geometry 有明确统计；
- 没有可靠的 style pair 时，明确记录为阻塞项。

如果这里发现相邻 FSQ level 没有平滑性，不要强行进入 birth-death；先进入 Phase 0B：训练一个仅改变 quantizer/decode geometry 的 ablation checkpoint，再重新测量。

### Phase 0B：可选 NEF physical checkpoint（2–4 周）

只在当前重建或 foot/contact 指标明显不够时执行。修改配置，不改 canonical contract：

```bash
python -m stylized_motion.run \
  --workflow-mode train \
  --representation nef-fsq \
  --config data/configs/nef_fsq_40x9_physical.yaml
```

比较 `v1` 与 `v1_1_physical`：

```text
feature recon
FK position error
root trajectory error
foot slide
contact consistency
stream locality
level adjacent geometry
```

只有当 physical checkpoint 不显著损害 locality，才允许它进入后续 generator 主实验。

### Phase 1：建立 style pair audit（1–2 周）

**文件：** `pairs.py`、`scripts/audit_style_pairs.py`。

输出 `style_pair_audit.json`，至少包含：

```json
{
  "style_groups": 0,
  "clips_per_style": {},
  "content_diversity": {},
  "performer_overlap": {},
  "same_clip_leakage": 0,
  "train_styles": [],
  "val_styles": [],
  "test_unseen_styles": []
}
```

划分原则：

- transport 可以使用 broad motion；
- operator train 需要 same-style / different-content evidence；
- zero-shot style 必须在 operator 参数学习中排除；
- style split 与 content split 分开管理；
- reference 与 target 不能是同一个 clip 的相邻 crop。

### Phase 2：训练无 style 的 base transport（3–5 周）

**文件：** `embeddings.py`、`graph.py`、`masking.py`、`transport.py`、`scripts/train_mts_transport.py`。

训练目标：

\[
L_{base}=CE\big(L_0,Z^*\big)\quad\text{only on masked valid positions}
\]

建议初始配置：

```yaml
mts_operator:
  tokenizer:
    representation: nef-fsq
    checkpoint: outputs/nef_fsq_40x9/best.pt
    freeze: true
  transport:
    dim: 256
    depth: 8
    heads: 8
    dropout: 0.1
    graph_mode: local_relational
  masking:
    random_coordinate: 0.20
    stream: 0.25
    temporal_span: 0.20
    spatiotemporal_block: 0.20
    full_generation: 0.15
```

先在 1–8 个 clip 上 overfit；再在完整 train split 训练。记录 mask 类型分别的 CE 和 masked completion 质量。

**退出标准：** base generator 在没有 style 输入时可以完成 infill / full generation，并且 motion quality 不低于现有 FSQ generator baseline。

### Phase 3：style-ID operator sandbox（2–3 周）

先用 fixed style ID 替代 reference encoder：

```text
style_id -> embedding -> operator
```

分别训练 logit field、arbitrary kernel、birth-death CTMC。transport 完全冻结。

目的：把“operator 没有能力”与“style encoder 没提取出风格”分开。

最小测试：

- strength 0 identity；
- strength sweep；
- same content, different style id；
- same style id, different content；
- local mask；
- flat/part/NEF representation comparison。

**退出标准：** style ID 确实改变风格且不破坏 content；若 CTMC 不优于 logit field，不把 CTMC 作为主贡献。

### Phase 4：reference style encoder（3–4 周）

**文件：** `style_encoder.py`、`pairs.py`、`scripts/train_mts_operator.py`。

训练流程：

```text
target tokens + mask + content
        ↓
frozen transport -> base logits, H

reference tokens
        ↓
global style encoder -> s

(base logits, H, s, hard mask, strength)
        ↓
operator -> styled probabilities
        ↓
masked target-token NLL
```

训练 batch 必须同时保存：

```text
target_tokens
reference_tokens
content_condition
same_style_evidence
different_content_evidence
valid_mask
```

为防止 target visible tokens 泄露 style，mask mixture 中至少 30% 使用 full generation mask，至少 30% 使用高遮盖率；剩余部分覆盖真实编辑场景。报告正确 reference、错误 reference、随机 reference 的差异。

**退出标准：** 替换 reference 后 style score 改变，content score 保持；随机 reference 明显下降；style encoder 不能只靠 performer/session identity。

### Phase 5：SIGGRAPH 主实验（4–6 周）

必须完成以下矩阵：

| 实验 | 方法 | 目的 |
|---|---|---|
| Injection | logit field / AdaLN / kernel / CTMC | 判断 operator 是否有优势 |
| Representation | flat / part / NEF | 判断空间所有权是否关键 |
| Geometry | birth-death / shuffled adjacency / full kernel | 判断 FSQ ordinal bias 是否有效 |
| Generalization | seen content / unseen content / unseen style | 判断是否学到可迁移规律 |
| Locality | radius 0 / radius 1 / whole body | 判断局部控制精度 |
| Composition | disjoint styles / overlap conflict | 判断组合和冲突 |
| Temporal | all frames / selected span | 判断时空 support |

每个主结果都要有同容量对照和至少 3 个随机 seed；先做小规模 seed 筛查，再对最终配置跑完整实验。

## 8. 评价指标和报告格式

### 8.1 NEF 表征指标

```text
feature reconstruction
FK joint position error
root trajectory error
contact precision/recall
foot slide ratio
per-coordinate level perplexity
adjacent/far perturbation ratio
token temporal change rate
decoder temporal influence width
```

### 8.2 Operator 指标

```text
style recognition accuracy / retrieval
content recognition or text-motion alignment
motion FID / diversity
foot skate ratio
strength-response curve
changed-token ratio
transition distance
off-target token leakage
off-target FK deviation
boundary jerk
contact violation
```

`changed-token ratio` 使用 common random numbers 时必须标注这是 coupling 下的配对指标，不要把它当成 kernel 的唯一统计性质。

### 8.3 必须画的图

1. 同一 content、同一 random seed，strength 从 0 到 2 的 motion strip；
2. whole-body / left-arm / upper-body / disjoint multi-style 的视觉对比；
3. flat、part、NEF 的 off-target leakage；
4. adjacent level、far level、shuffled adjacency 的动作影响；
5. correct/wrong/random reference 的 style response；
6. failure cases：contact 冲突、节奏风格、decoder temporal spill。

## 9. 推荐的文件级实施清单

### Commit 1：NEF contract probe

修改：

- `nef_layout.py`
- `nef_fsq.py`

新增：

- `nef_probe.py`
- `tests/test_nef_token_contract.py`
- `tests/test_nef_probe.py`
- `scripts/probe_nef_geometry.py`

验收：现有测试全通过，新增 probe 可生成 JSON/CSV。

### Commit 2：NEF physical ablation

新增：

- `data/configs/nef_fsq_40x9_physical.yaml`
- `tests/test_nef_objective_config.py`

只修改可选配置和训练 schedule；不修改 representation id。

### Commit 3：MTS contract and layout adapter

新增：

- `stylized_motion/learning/mts_operator/contract.py`
- `layout_adapter.py`
- `tests/test_mts_contract.py`

所有 shape、levels、layout hash、checkpoint fingerprint 先固定。

### Commit 4：stream embedding and graph transport

新增：

- `embeddings.py`
- `graph.py`
- `transport.py`
- `masking.py`
- `tests/test_mts_transport.py`

先不接 style。

### Commit 5：base transport training

新增：

- `scripts/train_mts_transport.py`
- base config

验收：small-batch overfit、masked CE、full-mask generation smoke test。

### Commit 6：operator sandbox

新增：

- `operators.py`
- `sampling.py`
- `tests/test_mts_operators.py`
- `tests/test_mts_sampling.py`

实现 logit field、arbitrary kernel、birth-death CTMC 三个版本；不要先实现 basis/gate/composition。

### Commit 7：style pair audit and style encoder

新增：

- `pairs.py`
- `style_encoder.py`
- `scripts/audit_style_pairs.py`
- `tests/test_mts_pairs.py`

验收：无 same-clip leakage、split provenance 可追踪。

### Commit 8：reference-conditioned training

新增：

- `model.py`
- `scripts/train_mts_operator.py`
- `tests/test_mts_end_to_end.py`

验收：correct/wrong/random reference 对照成立。

### Commit 9：evaluation and visuals

新增：

- `metrics.py`
- `scripts/evaluate_mts_operator.py`
- `scripts/generate_mts_operator.py`

输出统一 JSON、CSV、NPY 和渲染输入，确保论文图可复现。

## 10. 可执行命令模板

### NEF baseline

```bash
python -m stylized_motion.run \
  --workflow-mode test \
  --representation nef-fsq \
  --config data/configs/nef_fsq_40x9.yaml \
  --checkpoint outputs/nef_fsq_40x9/best.pt \
  --split test
```

### Transport

```bash
python scripts/train_mts_transport.py \
  --config data/configs/mts_operator_transport.yaml \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --output outputs/mts_transport/seed3407
```

### Style operator

```bash
python scripts/train_mts_operator.py \
  --config data/configs/mts_operator_style.yaml \
  --transport-checkpoint outputs/mts_transport/seed3407/best.pt \
  --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
  --operator birth_death \
  --output outputs/mts_operator/birth_death/seed3407
```

### Evaluation

```bash
python scripts/evaluate_mts_operator.py \
  --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
  --split test \
  --methods logit_field arbitrary_kernel birth_death \
  --representations flat part nef \
  --output outputs/mts_eval/main
```

### Generation

```bash
python scripts/generate_mts_operator.py \
  --checkpoint outputs/mts_operator/birth_death/seed3407/best.pt \
  --content-id CONTENT_ID \
  --style-reference STYLE_CLIP \
  --regions left_arm \
  --graph-radius 1 \
  --frame-range 80 140 \
  --strength 1.0 \
  --seed 1234 \
  --locked-edit \
  --output outputs/mts_samples/example
```

## 11. 失败时如何停止或转向

### 失败条件 A：NEF adjacent level 没有动作几何

停止 birth-death 主线。保留 NEF locality，改测 coordinate-aware arbitrary kernel；论文主张转为“structured alphabet support”，不再强调 ordinal FSQ transport。

### 失败条件 B：NEF 不比 flat/part 更局部

停止 style operator 开发，回到 representation。可能的问题是 stream feature ownership、global contact、decoder temporal coupling 或数据预处理，而不是 style model。

### 失败条件 C：style-ID 有效，reference-style 无效

算子本身没有问题，style pair 或 style encoder 有问题。检查 content leakage、performer identity、same-clip crop 和 unseen split；不要加更多 operator 参数。

### 失败条件 D：reference 改变，style 不改变

说明模型在 target visible tokens 上作弊，或 operator 接近 identity。提高 full-mask 比例，做 wrong-reference 诊断；如果仍失败，停止 reference claim。

### 失败条件 E：CTMC 不优于 logit field

不要为了保住原叙事继续加 basis、gate 和 auxiliary loss。把 CTMC 降级为一个稳定的参数化，主论文写 NEF-conditioned local style operator。

## 12. 最终论文叙事

论文不应写成：

```text
NEF-FSQ + Transformer + Style Encoder + CTMC + Local API
```

推荐叙事：

1. 现有 motion style transfer 主要在 feature/latent 上注入 style；局部控制通常需要 region-specific style representation 或额外分支。
2. 我们提出把 style 表示成作用在 factorized discrete motion alphabet 上的生成 operator。
3. NEF-FSQ 的 Node/Edge ownership 让同一个 global style descriptor 可以被限制在明确的 spatial-temporal support。
4. 我们证明这种表征-算子耦合在未见 style-content 组合上改善 style transfer、content preservation 和 local leakage。
5. CTMC/birth-death 是一个受限 operator family，用于研究 strength、identity 和 local transition 的归纳偏置；它的价值由对照实验决定。

论文必须明确限制：

- token locality 不等于 world-space exact locality；
- causal decoder 会造成 temporal spill；
- global contacts 可能影响 leg-local style；
- 不能把 semigroup 直接解释成完整生成过程的 semigroup；
- adjacent FSQ levels 只有经过 probe 验证后才有 ordinal interpretation。

## 13. 建议的研究决策点

在投入完整训练之前，只需先完成两个小实验：

1. `probe_nef_geometry.py`：相邻 FSQ level 是否对应更小、更稳定的动作变化；
2. `evaluate_nef_locality.py`：NEF 是否在 decode 后仍比 flat/part 更少 off-target 影响。

若两者都成立，继续实现 transport 和两个 operator；若只有第二个成立，继续做 structured local operator，但不用把 birth-death 写成核心；若两者都不成立，应该先改 NEF 或暂停 MTS-FSQ，而不是继续堆模块。

本方案的成功标准不是“所有模块都实现”，而是得到一条可审稿的因果链：

```text
NEF spatial ownership
        ↓
operator support locality
        ↓
cross-content style transfer
        ↓
strength / composition / authoring control
```

只有这条链在实验中成立，MTS-FSQ 才值得以 SIGGRAPH 主线推进。

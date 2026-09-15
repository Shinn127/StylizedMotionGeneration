# NEF-FSQ v3：面向代码实现的 Node–Edge Factorized FSQ 设计

本文档定义 NEF-FSQ v3 的首版实现规格。它用于指导代码、配置、测试和实验落地；文中标为“保证”的性质必须能由计算图或接口测试直接验证，模型质量相关内容均视为实验假设。

对应的现有代码基础：

- motion feature：`stylized_motion/anim/features.py`
- Part-FSQ 与布局：`stylized_motion/learning/part_fsq.py`、`stylized_motion/learning/part_layout.py`
- frame-causal 网络：`stylized_motion/learning/nets/causal_cnn.py`
- FSQ：`stylized_motion/learning/fsq.py`
- reconstruction loss：`stylized_motion/learning/losses.py`
- representation registry：`stylized_motion/learning/representation.py`
- token store：`stylized_motion/data/token_data.py`

“FSQ v3”是本方案的研究名称，不等同于已有 `latent_residual_fsq_v2` 的 `architecture_version: 3`。本方案注册为新的 representation family。

## 1. 目标与首版边界

NEF-FSQ 将逐帧人体动作拆成 Global、Node 和有向 Edge streams，并分别量化。首版目标是验证两点：

1. 在不使用 holistic base、Sync token 或跨区域 latent 融合的情况下，能否得到可用的重建质量；
2. 将 attachment joint 从身体内部关节中单独量化，能否改善局部 token 的跨动作替换与解释性。

首版固定以下约束：

- tokenizer 不使用 style、text、action 等条件；
- 时间不下采样，每个 motion frame 对应一组 40 维 FSQ coordinates；
- 所有 coordinates 均为 9 levels；
- 每个 stream 独立编码、独立量化、独立解码；
- Edge 有方向且只有一个 feature 所有者；
- decoder 输出通过确定性 scatter 写回 motion feature；
- 训练目标仅为现有 weighted L1 reconstruction 与 weighted delta；
- Geno 与 SOMA 分别训练、分别保存 checkpoint 和 token store；
- 首版不实现 Edge conditioning、Masked Graph Transformer、直接解码 `MASK`、IK、自动重定时或编辑边界修补。

Node–Edge factorization、较好的 donor transfer 和生成便利性是待验证假设。首版只从结构上保证 feature 路由，不宣称各 stream 统计独立，也不保证任意 token 组合均为自然动作。

## 2. Representation contract

| 属性 | 首版值 |
| --- | --- |
| family | `nef_fsq` |
| variant | `independent` |
| representation id | `nef_fsq_independent_40x9` |
| architecture version | `1` |
| frame rate | `60` |
| 训练窗口 | `64` frames |
| 输入/输出 | `[B, T, D]`，其中 `D=9J+5` |
| FSQ indices/codes | `[B, T, 40]` |
| index 范围 | `0..8` |
| temporal downsample | `1` |
| encoder receptive field | `31` frames |
| decoder receptive field | `34` frames |
| 完整 autoencoder receptive field | `64` frames |
| lookahead | `0` |

公共模型接口保持与现有 representation adapter 一致：

```python
forward(motion: Tensor[B, T, D]) -> dict
encode_to_indices(motion) -> LongTensor[B, T, 40]
encode_to_codes(motion) -> (Tensor[B, T, 40], LongTensor[B, T, 40])
decode_from_indices(indices: LongTensor[B, T, 40]) -> Tensor[B, T, D]
decode_from_codes(codes: Tensor[B, T, 40]) -> Tensor[B, T, D]
```

`forward` 至少返回 `recon_state`、`fsq_codes`、`codes`、`indices`、`commit_loss` 以及现有 FSQ utilization metrics。`commit_loss` 保持为零。

同为 40×9 只说明张量形状相同。不同 family、不同 skeleton、不同 feature schema 或不同 coordinate order 的 token 不兼容，加载和交换时必须拒绝。

## 3. 当前 motion feature contract

### 3.1 通用排列

设裁剪并加入 `Simulation` 后的关节数为 (J)，关节按 feature schema 的 `names`、`parents` 排列，且 joint 0 为 `Simulation`、joint 1 为 `Hips`。每帧 feature 为：

\[
X=[v^{sim},\omega^{sim},p^{hips},R^{1:J}_{6D},v^{hips},\omega^{1:J},c^L,c^R].
\]

| 分量 | slice | 维度 |
| --- | --- | ---: |
| Simulation local linear velocity | `[0:3]` | 3 |
| Simulation local angular velocity | `[3:6]` | 3 |
| Hips local position | `[6:9]` | 3 |
| non-root local rotation 6D | `[9 : 9+6(J-1)]` | `6(J-1)` |
| Hips local velocity | `[9+6(J-1) : 12+6(J-1)]` | 3 |
| non-root local angular velocity | `[12+6(J-1) : 12+9(J-1)]` | `3(J-1)` |
| left/right contact | `[9J+3 : 9J+5]` | 2 |

因此 (D=9J+5)。对任意非根 joint (j\in[1,J-1])，定义其 9D joint feature：

\[
F(j)=R_{6D}(j)\cup\omega(j),
\]

```text
rotation(j) = [9 + 6(j-1) : 9 + 6j]
angular(j)  = [12 + 6(J-1) + 3(j-1) : 12 + 6(J-1) + 3j]
```

两个 slice 在原始 feature 中不连续，layout 应以 index tensor 表示，不能假设每个 stream 对应连续输入。

### 3.2 物理含义与限制

- 非根 rotation 是相对 parent 的 local rotation，首版不再增加 world-position canonicalization。
- 非根 angular velocity 由相邻 local rotation 差分得到，是输入 feature 的一个物理量。
- delta loss 比较整个 normalized feature vector 的相邻帧差分；它不是 angular velocity，也不替代 angular velocity feature。
- `Simulation` transform 由身体数据派生：平移参考 `Spine2`，朝向参考 `Hips`；Global 与身体 streams 在数据统计上相关。
- contact 由 `LeftToeBase`、`RightToeBase` 的 global velocity 派生；contact 与腿部 streams 在数据统计上相关。

输入隔离只约束 encoder 能读取哪些张量，不能推出互信息为零或统计独立。

## 4. Skeleton 与 feature 分区

### 4.1 解析和校验规则

layout 必须按 skeleton names 查找 joints，再使用 parents 校验拓扑。禁止把 Geno 的数值 joint indices 复用于 SOMA。

1. skeleton 必须只有一个根，且为 index 0 的 `Simulation`；
2. `Hips` 必须为 index 1，parent 为 `Simulation`；
3. 下表列出的关节必须各出现一次，不允许缺失、重名或别名猜测；
4. 每条 chain 的 parent 关系必须与本文一致；
5. 每个非根 joint 的 9D feature 恰好归属一个 Node 或 Edge；
6. Hips position/velocity 只归属 `Simulation→Hips` Edge；
7. root velocities 和 contacts 只归属 Global；
8. 所有 feature indices 的并集必须恰好为 `[0,D)`，交集必须为空。

遇到未知 skeleton 或拓扑不匹配时直接报错。首版不做自动重映射。

### 4.2 Geno：25 joints / 230D

| Stream | joint/features | feature 维度 |
| --- | --- | ---: |
| Global | Simulation linear/angular velocity + contacts | 8 |
| Torso Node | Spine, Spine1, Spine2, Spine3 | 36 |
| Head Node | Neck1, Head | 18 |
| Left Arm Node | LeftArm, LeftForeArm, LeftHand | 27 |
| Right Arm Node | RightArm, RightForeArm, RightHand | 27 |
| Left Leg Node | LeftLeg, LeftFoot, LeftToeBase | 27 |
| Right Leg Node | RightLeg, RightFoot, RightToeBase | 27 |
| Simulation→Hips Edge | Hips position/velocity + `F(Hips)` | 15 |
| Torso→Head Edge | Neck | 9 |
| Torso→Left Arm Edge | LeftShoulder | 9 |
| Torso→Right Arm Edge | RightShoulder | 9 |
| Hips→Left Leg Edge | LeftUpLeg | 9 |
| Hips→Right Leg Edge | RightUpLeg | 9 |
| **合计** |  | **230** |

```text
Simulation → Hips
Hips → Spine → Spine1 → Spine2 → Spine3
Spine3 → Neck → Neck1 → Head
Spine3 → LeftShoulder → LeftArm → LeftForeArm → LeftHand
Spine3 → RightShoulder → RightArm → RightForeArm → RightHand
Hips → LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase
Hips → RightUpLeg → RightLeg → RightFoot → RightToeBase
```

### 4.3 SOMA：27 joints / 248D

SOMA 首先删除静态 BVH `Root`，再加入 `Simulation` 并执行相同裁剪：

| Stream | joint/features | feature 维度 |
| --- | --- | ---: |
| Global | Simulation linear/angular velocity + contacts | 8 |
| Torso Node | Spine1, Spine2, Chest | 27 |
| Head Node | Neck2, Head, Jaw, LeftEye, RightEye | 45 |
| Left Arm Node | LeftArm, LeftForeArm, LeftHand | 27 |
| Right Arm Node | RightArm, RightForeArm, RightHand | 27 |
| Left Leg Node | LeftShin, LeftFoot, LeftToeBase | 27 |
| Right Leg Node | RightShin, RightFoot, RightToeBase | 27 |
| Simulation→Hips Edge | Hips position/velocity + `F(Hips)` | 15 |
| Torso→Head Edge | Neck1 | 9 |
| Torso→Left Arm Edge | LeftShoulder | 9 |
| Torso→Right Arm Edge | RightShoulder | 9 |
| Hips→Left Leg Edge | LeftLeg | 9 |
| Hips→Right Leg Edge | RightLeg | 9 |
| **合计** |  | **248** |

```text
Simulation → Hips
Hips → Spine1 → Spine2 → Chest
Chest → Neck1 → Neck2 → Head
Head → Jaw / LeftEye / RightEye
Chest → LeftShoulder → LeftArm → LeftForeArm → LeftHand
Chest → RightShoulder → RightArm → RightForeArm → RightHand
Hips → LeftLeg → LeftShin → LeftFoot → LeftToeBase
Hips → RightLeg → RightShin → RightFoot → RightToeBase
```

SOMA Head 用 2 个 coordinates 表示 45D feature，而 Geno Head 用相同容量表示 18D。训练和评估必须单独报告 SOMA Head reconstruction，不得从全身平均误差推断其容量足够。

### 4.4 有向 Edge 与所有权

Edge 方向为 parent region 到 child region：`Simulation→Hips/Torso`、`Torso→Head`、`Torso→Arm`、`Hips→Leg`。“单一归属”表示 attachment joint 只由对应 Edge decoder 写入。Parent Node 不读取出边，Child Node 首版也不读取入边。

Shoulder Edge 只包含 `Shoulder`，不包含 `Arm`。因此 strict Arm Node edit 会替换 `Arm`、`ForeArm`、`Hand`，能够改变上臂朝向，同时保持 shoulder attachment joint 的 target token。

显式 Edge 用于检验 attachment control 与 chain internal motion 的分离效果。FK 本身已保证骨骼连接，Edge 不是用于防止骨架“断开”。

## 5. Token layout

coordinate 顺序是持久化接口的一部分：

| 顺序 | stream | coordinates | token slice |
| ---: | --- | ---: | --- |
| 0 | Global | 4 | `[0:4]` |
| 1 | Torso Node | 4 | `[4:8]` |
| 2 | Head Node | 2 | `[8:10]` |
| 3 | Left Arm Node | 4 | `[10:14]` |
| 4 | Right Arm Node | 4 | `[14:18]` |
| 5 | Left Leg Node | 4 | `[18:22]` |
| 6 | Right Leg Node | 4 | `[22:26]` |
| 7 | Simulation→Hips Edge | 4 | `[26:30]` |
| 8 | Torso→Head Edge | 2 | `[30:32]` |
| 9 | Torso→Left Arm Edge | 2 | `[32:34]` |
| 10 | Torso→Right Arm Edge | 2 | `[34:36]` |
| 11 | Hips→Left Leg Edge | 2 | `[36:38]` |
| 12 | Hips→Right Leg Edge | 2 | `[38:40]` |
|  | **合计** | **40** | `[0:40]` |

所有 coordinate 使用 9 levels，每帧理论容量为 (40\log_2 9\approx126.8\) bits。这是首版公平对比所需的固定容量。后续若改变容量或 levels，必须创建新的 representation id 和 token schema。

## 6. Tokenizer architecture

```text
normalized motion [B,T,D]
        │
        ├─ deterministic feature partition → 13 stream tensors
        ├─ family input projection + learned stream embedding
        ├─ fold stream into batch → one shared FrameCausalEncoder1D
        ├─ family FSQ, with left/right symmetry sharing → [B,T,40]
        ├─ one shared FrameCausalDecoder1D
        └─ family output head → deterministic scatter → [B,T,D]
```

不存在 full-body latent、base addition、Sync token、cross-stream attention、message passing 或 learnable assemble。

默认参数：`stream_dim=128`、ReLU、`norm=None`；复用 `FrameCausalEncoder1D`/`FrameCausalDecoder1D`；FSQ 使用 `scale=None`、`preserve_symmetry=false`、`noise_dropout=0.0`。每个 stream 添加独立 learned stream embedding，temporal module 不改变时间长度。

所有 13 个 streams 共用同一个 temporal encoder 和同一个 temporal decoder。不同输入维度先通过 family input projection 映射到 `stream_dim`，量化后的各 stream embedding 也具有相同 `stream_dim`，因此可以将 stream 维折入 batch 维执行这两个共享网络。

input projection、FSQ 和 output head 的 family sharing 如下：

| family | 共享 streams |
| --- | --- |
| global | Global |
| torso_node | Torso Node |
| head_node | Head Node |
| arm_node | Left/Right Arm Node |
| leg_node | Left/Right Leg Node |
| hips_edge | Simulation→Hips Edge |
| head_edge | Torso→Head Edge |
| shoulder_edge | Torso→Left/Right Arm Edge |
| leg_edge | Hips→Left/Right Leg Edge |

左右对称 streams 共享 input projection、FSQ 和 output head，通过不同 stream embedding 区分左右。只有 I/O 维度相同的 streams 才能共享 projection/head；Geno 与 SOMA 分别实例化，因此跨 checkpoint 不要求参数 shape 一致。

对 stream (s)：

\[
Z_s=Q_s(E_s(X_s)),\qquad \hat X_s=D_s(Z_s).
\]

`E_s` 只读取 `X_s`，`D_s` 只读取 `Z_s` 并只写 layout 指定 indices。由此在 local feature tensor 上保证：

\[
\frac{\partial\hat X_{s'}}{\partial Z_s}=0,\quad s'\neq s.
\]

该式不适用于 FK 后的 world-space positions。Assemble 仅执行无重叠 scatter；FK 只用于 loss/metrics、动作还原和可视化。

## 7. Training objective

训练沿用 `compute_motion_reconstruction_losses` 的现有定义。设 normalized target 为 (X)，重建为 (hat X)，feature weights 为 (w_d)，有效帧 mask 为 (m_t)：

\[
L_{recon}=\frac{\sum m_{b,t}w_d|\hat X_{b,t,d}-X_{b,t,d}|}{\sum m_{b,t,d}}.
\]

分母沿用 `_masked_weighted_mean`：mask 展开后计数，不除以 weights 之和。相邻帧 mask 为 (m^\Delta_t=m_t\land m_{t-1})：

\[
L_{delta}=\frac{\sum m^\Delta_{b,t}w_d|\Delta\hat X_{b,t,d}-\Delta X_{b,t,d}|}{\sum m^\Delta_{b,t,d}}.
\]

最终目标固定为：

\[
\boxed{L=L_{recon}+3.0L_{delta}}.
\]

delta 不跨 clip 计算，不额外除以 (dt)。长度小于 2 或没有有效 pair 时 `L_delta=0`，不得产生 NaN。

配置必须显式关闭其他损失，避免继承 runner 非零默认值：

```yaml
training:
  delta_weight: 3.0
  root_pos_weight: 0.0
  root_rot_weight: 0.0
  joint_weight: 0.0
  contact_weight: 0.0
  foot_slide_weight: 0.0
  foot_height_weight: 0.0
  reuse_weight: 0.0
  base_reuse_weight: 0.0
  latent_energy_weight: 0.0
  base_recon_weight: 0.0
  edit_weight: 0.0
  edit_preserve_weight: 0.0
```

FSQ commitment weight 为零。关闭独立 contact/root/FK/foot loss 不删除对应 reconstruction features。训练复用 60 fps、64-frame window、mirror sampling、当前 feature normalization 和 weights；Geno/SOMA 使用各自训练集统计、配置和输出目录。

## 8. Metadata、checkpoint 与 token store

metadata 至少持久化：family、variant、representation id、architecture version、motion dim、frame rate、时间 contract、40×9 contract、coordinate order/count/slices、skeleton names/parents、feature schema/joint subset，以及各 stream 的 feature 所有权。NEF 骨架兼容性按这些语义字段逐项比较，不依赖 skeleton SHA256。

实现时需要：

1. 在 representation registry 和 checkpoint metadata 路径注册 `nef_fsq`；
2. restore 时校验 family、version、layout 与 feature schema；
3. token export 按固定顺序写出 `[T,40]`；
4. 将 TokenStore 的 motion width 从固定 230 改为匹配 schema 的 `9J+5`；
5. 保持已有 Geno 230D / 40×9 store 可读；
6. SOMA 248D store 使用自身 schema；
7. generator 加载时校验 representation id 和 coordinate order。

现有 generator 的 coordinate-wise 9-way 分类头可以复用，但不会自动获得 graph-local attention 或 masked spatial generation。

## 9. 编辑语义

对 child region (r)，Node 为 (N_r)，唯一入边为 (E_r)。Strict edit 只替换 (Z^{N_r})；full-part edit 同时替换 (Z^{N_r}) 和 (Z^{E_r})。

- strict Left Arm 替换 LeftArm/LeftForeArm/LeftHand，保留 LeftShoulder；
- full Left Arm 同时替换 Left Shoulder Edge；
- strict Left Leg 替换膝以下 chain，保留 hip/upper-leg attachment；
- full Left Leg 同时替换 Hips→Left Leg Edge。

donor 与 target 必须来自同一 checkpoint、相同 skeleton/schema，token shape 一致。首版不做跨长度对齐。

时间区间采用半开区间 `[start,end)`，只替换区间内指定 stream token。首版没有可直接解码的 `MASK` 值。decoder 为 causal 且 RF=34，因此潜在 feature 影响区间为：

```text
[start, min(T, end + 33))
```

编辑不会影响 `start` 之前的 feature，但不保证 `end` 后立刻恢复。修改 Global root velocity 时，后续 world trajectory 还会因积分持续变化。

评估必须区分：

1. local feature preservation：非目标 decoder-owned features 是否逐值不变；
2. kinematic influence：修改关节后，其 descendants 的 world transform 变化；
3. world-space non-target change：排除预期后代后的其他区域变化。

修改 Torso local rotations 会移动 Head 和 Arms 的 world positions，即使这些区域的 local features 未变。保留 contact token 也不保证 donor leg 与 target contact phase 相容或无 foot sliding。

## 10. 测试与验收

### 10.1 Layout

- Geno 得到 `J=25,D=230`，SOMA 得到 `J=27,D=248`；
- 两者均生成 13 streams、40 个连续 coordinates；
- feature indices 合并、排序后严格等于 `arange(D)`，无重复；
- 每个 joint 均落入本文指定 stream；
- 缺失/重名 joint、错误 parent、额外未归属 joint、错误 motion dim 均拒绝；
- feature partition 后直接 assemble 与输入逐值相同。

### 10.2 模型与路由

- `forward` 输出 `[B,T,D]` motion 和 `[B,T,40]` token；
- indices/codes 解码与 forward reconstruction 逐值一致；
- 修改一个输入 stream 只允许该 stream encoded indices 改变；
- 修改一个 token slice 只允许其 decoder-owned local features 改变；
- 左右 family modules 为同一参数对象，stream embeddings 不共享；
- 小 batch forward/backward 梯度有限，FSQ STE 能传回对应 encoder；
- Geno 和 SOMA 分别完成 round trip。

### 10.3 时间与 loss

- encoder RF=31、decoder RF=34、完整 RF=64，均为 lookahead 0；
- future input/token 不影响当前输出；
- chunked causal encoding 与 full sequence 在有效区间一致；
- 时间 edit 的 feature 影响不早于 `start`、不晚于 `end+33`；
- 手算 fixture 验证 weighted recon、weighted delta、现有分母和 pair mask；
- 总 loss 严格等于 `recon + 3.0 * delta`；
- 无有效 pair 时 delta 为有限零值；所有额外 loss 不进入总 loss。

### 10.4 持久化

- Geno/SOMA checkpoint restore 保持 layout、schema 与 reconstruction；
- token store export/read round trip 保持 `[T,40]`；
- 旧 Geno 230D store 可读，SOMA 248D store 按自身 schema 通过；
- family、representation id、skeleton names/parents、feature schema 或 coordinate order 不匹配时拒绝；
- 禁止 Geno/SOMA 交叉 donor decode。

## 11. 实验报告与实现顺序

训练后按 skeleton 和 stream 报告 weighted reconstruction/delta、rotation/FK/root error、token usage/entropy/change rate、strict/full donor transfer、local preservation、排除 FK 后代后的 leakage、edit boundary velocity、foot sliding/contact mismatch，以及 SOMA/Geno Head 的独立 reconstruction。

不预设未经实验支持的质量阈值。结构测试通过只表示路由正确，不表示 NEF-FSQ 已优于 baseline。公平对比包括：当前 Part-FSQ 40×9、去掉 Global/Sync 且不拆 Edge 的 independent Part-FSQ、本文 NEF-FSQ；三者保持相同数据、sampling、width、FSQ、loss 和训练步数。

实现顺序固定为：

1. **分区与接口**：skeleton-aware layout、feature indices、token slices、metadata 和 partition/scatter tests；
2. **Tokenizer**：独立 stream encoder/FSQ/decoder、左右 family sharing、round-trip 与路由 tests；
3. **训练接入**：注册 `nef_fsq`、增加 Geno/SOMA 配置、关闭额外损失、小 batch 前后向；
4. **持久化**：接入 checkpoint、token store/export 与 generator compatibility checks；
5. **编辑评估**：实现同骨架 strict/full-part 和时间区间 swap，报告 transfer、preservation、边界和足部指标。

完成前四步后，NEF-FSQ 才具备可训练、可恢复和可导出 token 的首版实现；第五步用于验证局部可组合这一研究目标。

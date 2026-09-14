# NEF-FSQ：面向生成与空间编辑的 Node-Edge Factorized FSQ 表征方案

## 1. 项目目标

NEF-FSQ（Node-Edge Factorized Finite Scalar Quantization）是一种面向人体动作生成与局部编辑的离散运动表征方案。它的目标不是追求极高压缩率，也不是在表征阶段显式学习“风格”或“内容-风格解耦”，而是构建一种：

- 适合后续生成模型建模；
- 支持明确的身体空间分区；
- 能够进行局部区域编辑；
- 时间维度与原始动作逐帧对齐；
- 表征学习阶段尽量只依赖 reconstruction loss；
- 空间解耦主要由网络结构和信息路由保证，而非大量辅助 loss；
- 可以在后续生成阶段方便地接入 style conditioning、text conditioning、motion completion、masked generation 等任务。

NEF-FSQ 的核心思想可以概括为：

> **不再将全身动作压缩进一个统一 latent 或“base + residual”表示，而是把人体看作一个运动学图，将运动分解为 Global、Node 和 Edge 三类变量，并分别进行 FSQ 离散化。**

因此，NEF-FSQ 更接近一种结构化 motion alphabet，而不是一个普通的 VQ/FSQ autoencoder。

---

# 2. 核心设计约束

## 2.1 表征阶段不学习风格

NEF-FSQ 的 tokenizer 不显式构造：

- style latent；
- content latent；
- style token；
- style/content disentanglement loss；
- style cycle consistency；
- adversarial style removal；
- style classifier。

表征阶段只学习动作本身：

\[
X \rightarrow Z \rightarrow \hat X
\]

其中：

\[
Z = \{Z^G, Z^N, Z^E\}
\]

分别表示全局运动、局部身体节点运动和区域间接口运动。

后续如果需要风格生成，则在生成器阶段学习：

\[
p_\theta(Z \mid y,s)
\]

其中：

- \(y\)：文本、动作语义、轨迹等条件；
- \(s\)：风格条件。

风格只改变 token 的生成分布，不改变 tokenizer 本身的定义。

---

## 2.2 不进行时间压缩

NEF-FSQ 严格保持：

\[
\boxed{T_{token}=T_{motion}}
\]

即：

- 不做 temporal pooling；
- 不做 stride > 1 的时间下采样；
- 不做 \(T\rightarrow T/2,T/4,T/8\)；
- 不使用 multi-rate temporal token；
- 所有 Global / Node / Edge stream 都与原始动作逐帧对齐。

Encoder / Decoder 可以拥有 temporal receptive field，例如使用 temporal convolution、dilated convolution、local Transformer、temporal mixer 或 state-space block，但必须始终保持：

\[
T_{out}=T_{in}
\]

即：

\[
\boxed{\text{temporal context} \neq \text{temporal compression}}
\]

每个 FSQ timestep 都对应一个明确的动作帧。

---

## 2.3 空间解耦主要通过结构实现

NEF-FSQ 不依赖复杂的：

\[
L_{disentangle},L_{transfer},L_{cycle},L_{preserve}
\]

来学习空间解耦。

核心原则是：

\[
\boxed{\text{Spatial disentanglement is an architectural constraint}}
\]

通过：

1. 输入 feature partition；
2. 局部 encoder 的受限输入；
3. Node / Edge 分离；
4. block-sparse information routing；
5. 局部 decoder；
6. deterministic kinematic assembly；

从 computation graph 上限制某个区域 token 能影响的输出范围。

目标是尽可能得到：

\[
\frac{\partial \hat X_{r'}}{\partial Z_r}=0,\qquad r'\neq r
\]

而不是仅通过 loss 鼓励该值变小。

---

# 3. 为什么不继续使用 Base + Residual

很多结构化运动 tokenizer 可以写成：

\[
z=z_{base}+\sum_r \Delta z_r
\]

其中 \(z_{base}\) 是全身 base representation，\(\Delta z_r\) 是局部身体区域 residual。

这种设计的核心问题是：只要 base 容量足够大，它就可以编码大量局部信息：

\[
I(z_{base};X_{left-arm})>0
\]

甚至最终退化为：

\[
z_{base}\approx \text{full-body representation}
\]

于是局部 residual 更像是“base 没有表示好的剩余信息”，而不是稳定、独立的区域运动变量。

进一步地，局部 residual 通常条件于 base：

\[
Z_r=f(X_r,z_{base})
\]

因此同一个局部 code 在不同 base 下未必有一致的含义。这会直接影响：

- part swapping；
- 局部编辑；
- masked generation；
- token compositionality；
- 跨动作局部迁移。

NEF-FSQ 因此取消可以直接重建完整全身运动的 holistic base latent。

---

# 4. 人体运动图建模

NEF-FSQ 将人体表示成 skeleton region graph：

\[
\mathcal G=(\mathcal V,\mathcal E)
\]

其中：

- \(\mathcal V\)：身体区域节点；
- \(\mathcal E\)：身体区域之间的连接边。

推荐第一版节点划分：

\[
\mathcal V=\{Torso,Head,LArm,RArm,LLeg,RLeg\}
\]

对应边可以设计为：

\[
\mathcal E=\{
Root\text{-}Torso,
Torso\text{-}Head,
Torso\text{-}LArm,
Torso\text{-}RArm,
Pelvis\text{-}LLeg,
Pelvis\text{-}RLeg
\}
\]

工程实现中 Pelvis 可以视为 Torso 的下边界，也可以独立成一个节点。第一版建议保持图结构简单，优先验证 Node-Edge factorization 本身。

---

# 5. 三类离散变量

NEF-FSQ 的离散表示为：

\[
\boxed{Z=\{Z^G,Z^N_1,\ldots,Z^N_R,Z^E_1,\ldots,Z^E_M\}}
\]

其中：

- \(Z^G\)：Global FSQ；
- \(Z^N_r\)：第 \(r\) 个 Node FSQ；
- \(Z^E_e\)：第 \(e\) 条 Edge FSQ。

所有 stream 都与原动作逐帧对齐：

\[
Z^G\in\mathcal Z_G^T
\]

\[
Z^N_r\in\mathcal Z_r^T
\]

\[
Z^E_e\in\mathcal Z_e^T
\]

因此在任意时刻 \(t\)，完整离散状态为：

\[
Z_t=\{Z_t^G,Z_{t,1:R}^N,Z_{t,1:M}^E\}
\]

这实际上构成一个：

\[
\boxed{T\times \text{Skeleton Graph}}
\]

的离散 token lattice。

---

# 6. Global 表征

## 6.1 Global 只表示真正的全局变量

Global stream 不应该看到全身关节旋转。建议输入只包含真正与整体移动有关的变量，例如：

- root planar velocity；
- root vertical state；
- root yaw velocity；
- root height；
- root orientation 中必要的全局部分；
- contact state；
- 必要的 COM / root dynamics。

例如：

\[
X_t^G=[v_{x,t}^{root},v_{z,t}^{root},\omega_{y,t}^{root},h_t^{root},c_t^{left},c_t^{right}]
\]

这里“Global”是空间意义上的 global，而不是时间频率意义上的 low-frequency。

---

## 6.2 Global 不允许成为隐式 full-body base

Global Encoder：

\[
H^G=E_G(X^G)
\]

只能访问 \(X^G\)，不能访问 Torso、Arm、Leg 等局部 joint features。

否则 \(Z^G\) 会重新退化成 holistic base，这会破坏 NEF-FSQ 的基本出发点。

---

# 7. Node 表征

## 7.1 Node 表示身体区域内部运动

每个 Node stream 表示某一个 articulated region 的内部自由度。

例如 LeftArm Node 可以包含：

- elbow rotation；
- wrist rotation；
- hand orientation；
- arm internal angular dynamics。

而不包含：

- root motion；
- torso orientation；
- shoulder attachment rotation。

因为 shoulder 更适合作为 Edge variable。

类似地：

### LeftLeg Node

包含 knee、ankle、foot 以及 leg internal motion。

### Torso Node

包含 spine、chest、upper-body internal articulation。

### Head Node

包含 neck / head local articulation。

---

# 8. Edge 表征

## 8.1 为什么必须有 Edge

人体空间分区最难处理的是 region boundary。

以 LeftArm 与 Torso 为例，连接发生在 shoulder。

如果 shoulder 完全属于 LeftArm：

- LeftArm 编辑会改变肩部；
- 容易破坏 torso 与 arm 的连接关系。

如果 shoulder 完全属于 Torso：

- LeftArm 又无法完整控制整条手臂。

因此 NEF-FSQ 显式定义：

\[
Z^E_{Torso-LArm}
\]

用来表示两个 region 之间的接口运动。

---

## 8.2 Edge 可以表示什么

一个 Edge stream 可以编码：

- attachment joint rotation；
- boundary angular velocity；
- parent-child relative transform；
- boundary frame orientation；
- 必要的局部协调状态。

例如：

\[
X^E_{Torso-LArm}=[R_{shoulder},\omega_{shoulder}]
\]

LeftLeg 的 Edge 可以表示：

\[
X^E_{Pelvis-LLeg}=[R_{hip},\omega_{hip}]
\]

最终：

\[
\boxed{\text{Region motion}=\text{Interface motion}+\text{Internal motion}}
\]

以 LeftArm 为例：

\[
LeftArm=Z^E_{Torso-LArm}+Z^N_{LeftArm}
\]

这里的加号表示“组成关系”，不要求一定是 latent 向量相加。

---

# 9. Local Canonicalization

即使 encoder 只能看到局部区域，如果直接使用 world-space position，也可能间接泄漏大量全局信息。

例如 LeftArm 的 world-space position 会受到：

- root translation；
- torso orientation；
- shoulder position；

影响。

因此 Node 输入应该进行 local canonicalization。

对于 region \(r\)，设 attachment frame 为 \(T_r\)，则：

\[
\tilde X_r=T_r^{-1}X_r
\]

Node Encoder 实际处理：

\[
Z^N_r=Q_N(E_N(Canonicalize_r(X)))
\]

例如：

- LeftArm：相对于 shoulder frame；
- LeftLeg：相对于 hip frame；
- Head：相对于 neck / upper-spine frame。

这样 Node token 更接近表达：

> 该 articulated chain 相对于自身 attachment point 如何运动。

而不是：

> 它在世界坐标中位于哪里。

这是降低跨区域信息泄漏的关键设计。

---

# 10. Encoder 设计

## 10.1 总体形式

分别定义：

\[
H^G=E_G(X^G)
\]

\[
H^N_r=E_N(X^N_r)
\]

\[
H^E_e=E_E(X^E_e)
\]

基本要求：

- 不改变时间长度；
- 有一定 temporal receptive field；
- 不进行 unrestricted cross-region mixing；
- 输入域严格受限。

---

## 10.2 Temporal Encoder

推荐结构：

```text
Input Projection
      ↓
Temporal Residual Block × N
      ↓
LayerNorm / RMSNorm
      ↓
FSQ Input Projection
      ↓
FSQ
```

所有 temporal convolution：

\[
stride=1
\]

并通过 padding 保持：

\[
T_{out}=T_{in}
\]

第一版优先选择简单、稳定的 temporal block，例如：

- Conv1D；
- depthwise temporal convolution；
- gated MLP；
- residual temporal block。

不建议 tokenizer 第一版就使用非常大的 Transformer，因为此阶段更重要的是稳定、局部、易控制。

---

# 11. 参数共享

不建议为每个身体区域建立完全独立的 encoder / decoder。

推荐 family sharing：

\[
E_{LArm}=E_{RArm}
\]

\[
D_{LArm}=D_{RArm}
\]

\[
Q_{LArm}=Q_{RArm}
\]

同理：

\[
E_{LLeg}=E_{RLeg}
\]

左右 shoulder Edge 共用一套参数；左右 hip Edge 共用一套参数。

通过 region / side embedding 区分：

\[
e_{left},e_{right},e_{region}
\]

这样可以：

- 提升样本效率；
- 强化左右对称性；
- 降低参数量；
- 提高左右肢体 code space 的一致性。

---

# 12. FSQ 设计

## 12.1 基本形式

对某一 stream 的 latent：

\[
H_t\in\mathbb R^d
\]

先做：

\[
u_t=W_{in}H_t
\]

得到：

\[
u_t\in\mathbb R^K
\]

然后逐 coordinate 使用 FSQ：

\[
q_t^k=Q_{FSQ}(u_t^k)
\]

最终：

\[
Z_t=[q_t^1,\ldots,q_t^K]
\]

每个 scalar coordinate 具有有限 levels。

---

## 12.2 空间异构、时间同构

NEF-FSQ 不要求所有 stream 使用相同容量。

建议第一版：

| Stream | 建议 FSQ Levels |
|---|---|
| Global | `[9,9,9,7]` |
| Torso Node | `[9,9,9,7]` |
| Arm Node | `[9,9,7]` |
| Leg Node | `[9,9,9,7]` |
| Head Node | `[7,7]` |
| Shoulder Edge | `[7,7]` |
| Hip Edge | `[7,7]` |
| Root-Torso Edge | `[9,7]` |

但所有 stream 都满足：

\[
T_{token}=T
\]

因此整个表示是：

\[
\boxed{\text{spatially heterogeneous, temporally homogeneous}}
\]

---

## 12.3 容量分配原则

不同区域的自由度和运动复杂度不同：

- Torso 和腿通常需要更高容量；
- Arm 次之；
- Head 相对较低；
- Edge 只承担接口信息，应刻意保持较小容量。

第一阶段应控制总理论 bitrate 与 Flat-FSQ / Part-FSQ baseline 接近，以便公平比较。

Edge 容量尤其不能过大，否则它可能重新吸收完整 region 信息。

---

# 13. Decoder 设计

## 13.1 禁止重新融合成 full-body latent

不建议：

```text
All tokens
   ↓
Concatenate / Full Attention
   ↓
Shared full-body latent
   ↓
Full-body decoder
```

否则空间分解会重新变成软约束。

---

## 13.2 Factorized Decoder

分别定义：

\[
\hat X^G=D_G(Z^G)
\]

\[
\hat X^E_e=D_E(Z^E_e)
\]

\[
\hat X^N_r=D_N(Z^N_r,Z^E_{\partial r})
\]

其中 \(\partial r\) 表示和 Node \(r\) 相邻的 Edge。

Node decoder 可以访问：

- 自己的 Node token；
- 自己相邻的 Edge token。

但不能访问：

\[
Z^N_{r'},\qquad r'\neq r
\]

---

## 13.3 Block-Sparse Information Routing

例如 LeftArm Decoder 允许读取：

\[
Z^N_{LArm}
\]

以及：

\[
Z^E_{Torso-LArm}
\]

但不能读取：

\[
Z^N_{RArm},Z^N_{LLeg},Z^N_{RLeg}
\]

这种约束应该直接体现在 computation graph，而不是依赖额外 loss。

---

# 14. Kinematic Assemble

各 decoder 输出后，通过尽可能确定性的 Assemble 模块恢复完整 motion：

\[
\hat X=A(\hat X^G,\{\hat X^N_r\},\{\hat X^E_e\})
\]

Assemble 负责：

- 将 rotation feature 写回对应 joints；
- 拼接 root motion；
- 合并 attachment joint；
- 必要时通过 FK 计算 joint positions；
- 恢复统一 motion feature vector。

最好：

\[
A
\]

不存在 learnable full-body mixing。

---

# 15. 空间解耦性质

理想情况下，如果 LeftArm Node decoder 只写 LeftArm 内部 joints：

\[
\frac{\partial \hat X_{RArm}}{\partial Z^N_{LArm}}=0
\]

\[
\frac{\partial \hat X_{LLeg}}{\partial Z^N_{LArm}}=0
\]

\[
\frac{\partial \hat X_{Torso}}{\partial Z^N_{LArm}}=0
\]

这种 locality 是 architecture-induced，而不是 loss-induced。

需要注意：Edge token 本身就是共享边界变量，因此编辑 Edge 时允许影响该 Edge 两侧的关联区域，这是设计的一部分而不是泄漏。

---

# 16. 两级区域编辑

Node / Edge factorization 自然支持两种编辑粒度。

## 16.1 Strict Local Edit

只修改：

\[
Z^N_r
\]

保持：

\[
Z^E_{\partial r}
\]

不变。

例如：

\[
Z^N_{LArm}\leftarrow Z^{N,donor}_{LArm}
\]

肩部接口保持 target 状态。

其含义是：保持 region attachment 不变，只修改内部运动。

---

## 16.2 Full-Part Edit

同时修改：

\[
Z^N_r
\]

和：

\[
Z^E_{\partial r}
\]

例如：

\[
\{Z^N_{LArm},Z^E_{Torso-LArm}\}
\]

一起替换。

其含义是：替换完整左臂，包括肩部连接运动。

---

# 17. 时间局部编辑

由于：

\[
T_{token}=T_{motion}
\]

可以精确编辑任意 frame range。

例如只编辑 \(t\in[80,120]\) 的 LeftArm：

\[
Z^N_{LArm}[80:120]\leftarrow MASK
\]

或：

\[
Z^N_{LArm}[80:120]\leftarrow Z^{donor}_{LArm}[80:120]
\]

因此编辑 mask 可以统一写成：

\[
M\in\{0,1\}^{T\times R}
\]

实现：

\[
\boxed{\text{spatiotemporal editing}}
\]

---

# 18. Reconstruction Objective

## 18.1 核心原则

表征学习阶段尽可能只使用：

\[
\boxed{L=L_{recon}}
\]

不主动加入：

- commitment loss；
- part transfer loss；
- edit preservation loss；
- style loss；
- contrastive disentanglement loss；
- latent energy loss；
- token reuse loss。

FSQ 本身也不需要传统 learned VQ codebook 的 commitment objective。

---

## 18.2 Reconstruction Space

可以定义统一 reconstruction mapping：

\[
\Phi(X)
\]

例如：

\[
\Phi(X)=[R_{6D},v_{root},\omega_{root},p_{joint},c_{foot}]
\]

则：

\[
L_{recon}=\rho(W\Phi(\hat X),W\Phi(X))
\]

其中：

- \(\rho\)：SmoothL1 / Charbonnier / L1；
- \(W\)：feature normalization。

推荐：

\[
W_{ii}=\frac{1}{\sigma_i}
\]

即根据训练集标准差进行归一化。

这样依然只有一个 reconstruction objective，而不是多个手调 loss 权重。

---

## 18.3 最简版本

如果原始 motion feature 已经包含合理的 rotation、root velocity、joint position、contact，第一版甚至可以直接：

\[
L_{recon}=SmoothL1(\hat X,X)
\]

先验证 architecture。

只有出现明显 FK drift、foot sliding 等问题时，再考虑将确定性 FK feature 纳入统一 \(\Phi\)，而不是马上增加很多独立 loss。

---

# 19. 时间稳定性

由于逐帧 FSQ，可能出现相邻帧 index 抖动，例如：

```text
4 4 5 4 5 4 4 5 ...
```

第一阶段不建议增加 temporal reuse loss。

优先通过 architecture 处理：

## 19.1 Temporal receptive field

\[
H_t=E(X_{t-k:t+k})
\]

使 FSQ 输入天然具有时间上下文。

## 19.2 Smooth latent architecture

使用 temporal convolution、residual temporal block、depthwise conv 和 normalization，降低高频 latent noise。

## 19.3 稳定 FSQ projection

在 FSQ 前使用 LayerNorm / RMSNorm、合理初始化和 bounded projection，降低 latent 长时间停留在量化边界附近造成的 index flicker。

---

# 20. 与后续生成器的接口

Tokenizer 训练完成后冻结：

\[
E,Q,D
\]

得到离散表示：

\[
Z=\{Z^G,Z^N,Z^E\}
\]

后续生成器学习：

\[
p_\theta(Z\mid text,style,trajectory,mask,\ldots)
\]

Tokenizer 本身不知道 style。

---

# 21. Style 的位置

Style 只属于生成阶段：

\[
p_\theta(Z\mid y,s)
\]

同一个语义：

\[
y=\text{walk forward}
\]

在不同 style condition 下生成不同 token sequence：

\[
Z_1\sim p(Z\mid y,s_1)
\]

\[
Z_2\sim p(Z\mid y,s_2)
\]

但最后都通过同一个 frozen NEF-FSQ decoder 还原动作。

因此：

\[
\boxed{\text{style changes token distribution, not tokenizer semantics}}
\]

---

# 22. 后续生成器建议

NEF-FSQ 很适合 Masked Graph Transformer。

token lattice 是：

\[
T\times\mathcal G
\]

例如：

```text
              time →
Global      ● ● ● ● ● ● ●

Torso       ● ● ● ● ● ● ●
Head        ● ● ● ● ● ● ●
L-Arm       ● ● ● ● ● ● ●
R-Arm       ● ● ● ● ● ● ●
L-Leg       ● ● ● ● ● ● ●
R-Leg       ● ● ● ● ● ● ●

T-H         ● ● ● ● ● ● ●
T-LA        ● ● ● ● ● ● ●
T-RA        ● ● ● ● ● ● ●
P-LL        ● ● ● ● ● ● ●
P-RL        ● ● ● ● ● ● ●
```

生成器可以显式区分：

- temporal interaction；
- graph-local spatial interaction。

对于空间编辑，只需要 mask 指定 region / edge 和时间范围。

---

# 23. Factorized FSQ Prediction

如果一个 stream 使用：

\[
[9,9,7]
\]

没有必要将它展开成：

\[
9\times9\times7=567
\]

类 vocabulary。

生成器可以为三个 scalar coordinate 分别预测：

\[
p(q_1|h),\quad p(q_2|h),\quad p(q_3|h)
\]

对应：

- 9-way classification；
- 9-way classification；
- 7-way classification。

这更符合 FSQ 的内部结构，也便于未来进行 coordinate-level masking。

---

# 24. 推荐的第一版网络

## 24.1 Node Encoder

```text
Local Motion Features
        ↓
Linear Projection
        ↓
Temporal Residual Block × N
        ↓
LayerNorm / RMSNorm
        ↓
FSQ Input Projection
        ↓
FSQ
```

建议隐藏维度：

\[
d=64\sim128
\]

---

## 24.2 Edge Encoder

```text
Edge Features
    ↓
Linear
    ↓
Temporal Block × 2~4
    ↓
Normalization
    ↓
FSQ
```

建议隐藏维度：

\[
d=32\sim64
\]

Edge 的表示能力应有意小于 Node。

---

## 24.3 Global Encoder

结构与 Edge / Node 类似，但输入域严格限制为 global root / contact variables。

---

## 24.4 Decoder

所有 decoder 保持：

\[
T\rightarrow T
\]

Node Decoder 输入：

\[
[Emb(Z^N_r),Emb(Z^E_{\partial r})]
\]

但输出只能写入对应 region feature。

---

# 25. 第一版推荐 token 配置

| Group | FSQ Levels |
|---|---|
| Global | `[9,9,9,7]` |
| Torso | `[9,9,9,7]` |
| Head | `[7,7]` |
| Left / Right Arm | `[9,9,7]` |
| Left / Right Leg | `[9,9,9,7]` |
| Root-Torso Edge | `[9,7]` |
| Torso-Head Edge | `[7,7]` |
| Torso-Arm Edge | `[7,7]` |
| Pelvis-Leg Edge | `[7,7]` |

这只是合理初始化方案，不应被视为最终固定配置。

正式实验建议通过 bitrate-controlled ablation 选择容量。

---

# 26. 推荐训练流程

## Stage 1：Feature Definition

先确定：

- Global features；
- Node features；
- Edge features；
- local coordinate frame；
- Assemble 规则。

这一阶段的重要性高于网络深度。

---

## Stage 2：Pure Reconstruction Tokenizer

训练：

\[
X\rightarrow E\rightarrow FSQ\rightarrow D\rightarrow\hat X
\]

只优化：

\[
L_{recon}
\]

---

## Stage 3：Representation Evaluation

评估：

- reconstruction；
- FSQ utilization；
- spatial locality；
- temporal stability；
- local edit preservation。

这些指标用于评价 representation，不需要作为训练 loss。

---

## Stage 4：Freeze Tokenizer

固定：

\[
E,Q,D
\]

并对训练数据离线生成 token。

---

## Stage 5：Conditional Generator

训练：

\[
p_\theta(Z\mid condition)
\]

之后再加入：

- text；
- style；
- trajectory；
- region mask；
- partial motion。

---

# 27. Representation Evaluation

NEF-FSQ 不应该只看 reconstruction。

## 27.1 Reconstruction Error

建议报告：

- rotation reconstruction；
- MPJPE / FK joint position；
- root trajectory；
- velocity error。

---

## 27.2 Token Utilization

对每个 FSQ coordinate 统计：

- level usage；
- entropy；
- perplexity；
- dead level ratio。

例如：

\[
H(q_k)=-\sum_l p(q_k=l)\log p(q_k=l)
\]

---

## 27.3 Spatial Leakage

固定其它 token，只改变一个 region token。

定义：

\[
Leak(r\rightarrow r')=
\|\hat X_{r'}^{edit}-\hat X_{r'}^{original}\|
\]

理想情况下，对非相邻区域：

\[
Leak(r\rightarrow r')\approx0
\]

---

## 27.4 Edit Preservation

例如编辑 LeftArm：

\[
Preserve=d(X_{\neg LeftArm}^{edit},X_{\neg LeftArm}^{target})
\]

越低越好。

---

## 27.5 Edit Transfer

若使用 donor：

\[
Transfer=d(X_{LeftArm}^{edit},X_{LeftArm}^{donor})
\]

用于衡量区域替换能力。

这些 edit 指标只用于 evaluation，不进入 tokenizer loss。

---

## 27.6 Temporal Token Stability

统计：

\[
P(Z_t\neq Z_{t-1})
\]

以及：

\[
\mathbb E[|q_t-q_{t-1}|]
\]

用于分析逐帧 token 是否过度抖动。

---

# 28. Ablation 设计

## A. Flat-FSQ

完整动作统一量化：

\[
X\rightarrow FSQ\rightarrow\hat X
\]

作为无空间结构 baseline。

---

## B. Part-FSQ

只使用：

\[
Global+BodyPart
\]

没有 Edge，用于验证 Node-Edge factorization 的必要性。

---

## C. NEF-FSQ without Canonicalization

验证 local canonicalization 是否显著降低 information leakage。

---

## D. NEF-FSQ without Edge

验证 shoulder / hip boundary 是否变差，以及 Edge 是否确实改善区域组合和编辑稳定性。

---

## E. NEF-FSQ Full

完整：

\[
Global+Node+Edge+Canonicalization
\]

---

# 29. 与现有 FSQ 方案的对比

| 方法 | 表征结构 | 空间解耦机制 | 全身 Base | 时间压缩 | 核心 Loss |
|---|---|---|---|---|---|
| Flat-FSQ | 单流 | 无 | 否 | 可有 | Recon |
| Part-FSQ | Global + Part | Feature partition | 否 | 可有 | Recon + reuse |
| Residual-Part | Base + Part residual | Feature residual | 是 | 可有 | Recon + regularizer |
| Latent Residual | Base + latent slice | Latent partition | 是 | 可有 | Recon + energy |
| Latent Residual V2 | Base + full latent residual | Learned editing | 是 | 可有 | Recon + edit losses |
| **NEF-FSQ** | **Global + Node + Edge** | **Architectural routing** | **否** | **否** | **Recon** |

---

# 30. NEF-FSQ 的主要创新点

## 30.1 Node-Edge Motion Factorization

不是简单 body-part partition，而是：

\[
\boxed{\text{Internal Motion}+\text{Boundary Motion}}
\]

显式建模 articulated body 的接口。

---

## 30.2 Spatial Disentanglement by Construction

不依赖复杂 disentanglement objective，而通过网络拓扑直接定义信息流。

---

## 30.3 Frame-Aligned Structured Discrete Representation

所有 token：

\[
T\rightarrow T
\]

支持精确的逐帧、逐区域编辑。

---

## 30.4 Recon-Only Tokenizer Training

尽可能保持：

\[
L=L_{recon}
\]

让 representation 的结构性质主要由 architecture 决定。

---

## 30.5 Generation-Oriented Representation

NEF-FSQ 不是为了极致 compression，而是让后续生成器能够直接操作：

\[
\text{time}\times\text{body graph}
\]

的 token lattice。

---

# 31. 潜在风险与应对

## 31.1 分区过强导致全身协调不足

如果 Node 完全独立，可能出现：

- 手臂与 torso 节奏不一致；
- gait coordination 下降；
- 局部动作组合不自然。

优先依靠：

- Edge token；
- 后续 generator 中的 graph interaction；

解决，而不是重新引入 full-body base。

---

## 31.2 Edge 容量过大

如果 Edge token 太强，可能偷偷承担整个 region 的信息。

因此 Edge FSQ 应保持较低容量，例如：

\[
[7,7]
\]

并在实验中检查 Edge-only reconstruction 能力，防止 Edge 成为新的 leakage 通道。

---

## 31.3 Global 泄漏

如果 Global 输入包含全身 joint features，就会重新退化成 base。

因此 Global 的输入域必须严格限制。

---

## 31.4 Canonicalization 设计困难

局部坐标系选取不稳定可能带来：

- orientation discontinuity；
- noisy input；
- anchor frame jitter。

需要仔细定义 shoulder、pelvis、torso 等 anchor frame，并尽量避免使用不连续欧拉角。

---

## 31.5 Reconstruction 与 Editable Representation 的权衡

空间 factorization 限制了信息共享，因此 reconstruction error 可能略高于 unrestricted Flat-FSQ。

这是预期 trade-off：

\[
\boxed{\text{reconstruction optimality}\neq\text{generation/editability optimality}}
\]

NEF-FSQ 的目标不是单纯达到最低 reconstruction loss。

---

# 32. 推荐实现优先级

## Priority 1：定义表示

实现：

- feature partition；
- Global / Node / Edge 定义；
- local canonicalization；
- deterministic Assemble。

---

## Priority 2：实现 tokenizer

实现：

- stride-1 temporal encoder；
- FSQ；
- factorized decoder。

---

## Priority 3：验证 representation

重点检查：

- reconstruction；
- token utilization；
- leakage；
- temporal stability。

---

## Priority 4：验证编辑能力

实现：

- strict node editing；
- node + edge editing；
- arbitrary time-range editing。

---

## Priority 5：训练生成器

冻结 tokenizer，再训练 masked / autoregressive / diffusion-style discrete generator。

---

# 33. 建议的第一版完整数据流

```text
                         Motion X[1:T]
                               │
                   Spatial Feature Partition
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
      Global Features      Node Features        Edge Features
          │                    │                    │
          E_G                  E_N                  E_E
          │                    │                    │
     Temporal Blocks      Temporal Blocks      Temporal Blocks
       stride = 1          stride = 1           stride = 1
          │                    │                    │
         FSQ                  FSQ                  FSQ
          │                    │                    │
         Z_G                  Z_N                  Z_E
          │                    │                    │
         D_G             D_N(Z_N,Z_E)             D_E
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    Deterministic Assemble
                               │
                          Reconstructed X
```

训练：

\[
\boxed{L=L_{recon}}
\]

全程：

\[
\boxed{T\rightarrow T}
\]

---

# 34. 最终定义

NEF-FSQ 可以形式化为：

\[
X_{1:T}
\rightarrow
\{X^G,X^N,X^E\}
\rightarrow
\{Z^G,Z^N,Z^E\}
\rightarrow
\hat X_{1:T}
\]

其中：

\[
Z^G=FSQ(E_G(X^G))
\]

\[
Z^N_r=FSQ(E_N(X^N_r))
\]

\[
Z^E_e=FSQ(E_E(X^E_e))
\]

并严格满足：

\[
\boxed{T_{token}=T_{motion}}
\]

空间解耦依靠：

\[
\boxed{
Feature\ Partition
+
Local\ Canonicalization
+
Node/Edge\ Factorization
+
Restricted\ Information\ Routing
+
Factorized\ Decoder
}
\]

训练目标尽可能保持：

\[
\boxed{L=L_{recon}}
\]

最终得到一种：

\[
\boxed{
Spatially\ Factorized
+
Frame\text{-}Aligned
+
Generation\text{-}Oriented
+
Editable
}
\]

的离散人体运动表征。

NEF-FSQ 不试图在 tokenizer 阶段定义“风格是什么”。它只提供一个稳定、结构化、局部可操作的 motion token space。后续任何风格注入、文本控制、局部生成、动作补全和区域编辑，都由后续生成模型在同一个 token space 上完成。

---

# 35. 一句话概括

> **NEF-FSQ 将人体动作表示为逐帧对齐的 Global、Node 和 Edge FSQ token，通过运动学图上的结构化信息路由获得空间可编辑性，并尽可能仅以 reconstruction loss 完成 tokenizer 学习。**

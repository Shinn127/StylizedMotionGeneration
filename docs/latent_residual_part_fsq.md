# Latent Residual Part-FSQ

## 1. 方法概述

本文给出 **Latent Residual Part-FSQ（LRP-FSQ）** 的理论表述。该表示面向离散人体运动建模，其目标是在保持统一、非结构化 decoder latent 的同时，将局部身体运动编码为可独立替换的离散 residual token。

方法所处理的核心矛盾是：holistic latent 有利于建模全身协调，但通常缺乏可解释的局部编辑接口；显式拆分 decoder latent 可以增强局部性，却会限制全身运动因素在 latent space 中的自由组织。LRP-FSQ 不划分 decoder latent channel，而是将局部表示定义为相对于 holistic base 的条件残差，并通过部位相关的稠密投影将其注入共享 latent space。

对于 motion sequence \(x\)，方法学习一个量化的 holistic base \(q_b\) 和一组量化的局部 residual \(\{q_p\}_{p\in\mathcal P}\)，最终 latent 为

\[
z(x)=q_b(x)+\sum_{p\in\mathcal P}A_pq_p(x),
\tag{1}
\]

其中 \(\mathcal P\) 是身体部位集合，\(A_p\) 将部位 residual 投影到完整 decoder latent。单一 decoder 根据 \(z\) 重建 motion。局部编辑通过替换式 residual composition 完成，但 donor residual 在注入 target 前需要从 donor-base 条件重表达到 target-base 条件。

In mechanism terms, the model **learns** a holistic base, **predicts** base-explainable local states, **quantizes** the remaining innovations, and **composes** their projected corrections in a shared latent space. This factorization enables local token replacement without requiring a part-structured decoder latent.

## 2. 问题定义

设 motion sequence 为

\[
x=(x_1,\ldots,x_T)\in\mathbb R^{T\times D},
\tag{2}
\]

其中 \(T\) 为序列长度，\(D\) 为每帧运动特征维度。身体被划分为

\[
\mathcal P=\{\mathrm{torso},\mathrm{lleg},\mathrm{rleg},
\mathrm{larm},\mathrm{rarm}\}.
\tag{3}
\]

令 \(I_p\subseteq\{1,\ldots,D\}\) 表示部位 \(p\) 对应的局部特征索引，并定义

\[
x_p=S_p(x)=x[...,I_p].
\tag{4}
\]

局部索引满足

\[
I_p\cap I_{p'}=\varnothing,
\qquad p\neq p',
\tag{5}
\]

但它们不必覆盖 root、global trajectory 和 contact 等全局特征。这些信息由 holistic stream 统一建模。

我们希望学习离散表示

\[
\mathcal C(x)=\left(c_b,c_{p_1},\ldots,c_{p_{|\mathcal P|}}\right),
\tag{6}
\]

使其同时支持：

1. **重建性**：完整 token tuple 能够重建输入运动；
2. **基础完整性**：base token 单独保留全身运动的粗粒度结构；
3. **局部可替换性**：替换部位 \(p\) 的 token 主要改变该部位；
4. **统一解码性**：重建和编辑均由同一个 decoder 从单个稠密 latent 解码；
5. **时序因果性**：时刻 \(t\) 的表示和输出不依赖未来帧。

## 3. 有限标量量化

令 \(Q_g\) 表示 group \(g\) 的 Finite Scalar Quantizer（FSQ）。对连续状态 \(h_g\in\mathbb R^{T\times d_g}\)，FSQ 先投影到 \(K_g\) 个标量坐标，逐坐标量化到有限 level 集合，再投影回 group latent：

\[
a_g=W_g^{\mathrm{in}}h_g,
\qquad
c_g=\mathcal Q_{L}(a_g),
\qquad
q_g=W_g^{\mathrm{out}}c_g.
\tag{7}
\]

其中

\[
c_g\in\mathcal A_L^{T\times K_g},
\qquad |\mathcal A_L|=L,
\tag{8}
\]

\(L\) 是每个标量坐标的 level 数，\(K_g\) 是该 group 的坐标数。每帧 group \(g\) 的离散容量上界为

\[
|\mathcal C_g|=L^{K_g}.
\tag{9}
\]

LRP-FSQ 使用一个 base group 和五个 part groups：

\[
K=K_b+\sum_{p\in\mathcal P}K_p.
\tag{10}
\]

当前实例采用

\[
(K_b,K_{torso},K_{lleg},K_{rleg},K_{larm},K_{rarm})
=(20,6,4,4,3,3),
\tag{11}
\]

因此 \(K=40\)，且 \(L=9\)。这里的 40 个 FSQ 坐标定义离散传输接口，而不是 decoder latent 的 channel partition。

## 4. Base-Conditioned Residual Factorization

### 4.1 Holistic base

Holistic causal encoder \(E_b\) 从完整 motion 提取 base state：

\[
h_b=E_b(x).
\tag{12}
\]

经过 base quantizer 得到

\[
(c_b,q_b)=Q_b(h_b),
\qquad q_b\in\mathbb R^{T\times d_z}.
\tag{13}
\]

\(q_b\) 是完整的、非结构化的 decoder latent，而不是若干 part latent 的拼接。

### 4.2 局部状态

每个部位使用 causal encoder \(E_p\) 将局部 motion 映射到局部状态空间：

\[
u_p=E_p(x_p),
\qquad u_p\in\mathbb R^{T\times d_p}.
\tag{14}
\]

局部状态 \(u_p\) 表示从真实局部运动中提取的时序信息，但它本身不是 residual。

### 4.3 条件 residual

部位 predictor \(P_p\) 根据量化 base latent 估计 base 已能解释的局部状态：

\[
\hat u_p=P_p(\operatorname{sg}(q_b)),
\tag{15}
\]

其中 \(\operatorname{sg}(\cdot)\) 表示 stop-gradient。局部 residual 定义为

\[
r_p=u_p-\hat u_p.
\tag{16}
\]

Part-FSQ 量化该 residual：

\[
(c_p,q_p)=Q_p(r_p),
\qquad q_p\in\mathbb R^{T\times d_p}.
\tag{17}
\]

因此，\(q_p\) 不是无条件 part code，而是 **base-conditioned residual code**。它回答的是“在给定 \(q_b\) 后，部位 \(p\) 还需要补充什么”，而不是“部位 \(p\) 的完整状态是什么”。

Stop-gradient 隔离了 residual prediction 路径对 base encoder 的条件梯度：

\[
\frac{\partial P_p(\operatorname{sg}(q_b))}{\partial q_b}=0.
\tag{18}
\]

但 \(q_b\) 在最终重建路径中不被截断，因此仍由 reconstruction objective 优化。

## 5. 非结构化 Latent Residual Composition

每个量化 part residual 通过独立的稠密映射投影到共享 decoder latent：

\[
\Delta z_p=A_pq_p,
\qquad
A_p:\mathbb R^{d_p}\rightarrow\mathbb R^{d_z}.
\tag{19}
\]

所有局部 correction 与 base 相加：

\[
z=q_b+\sum_{p\in\mathcal P}\Delta z_p.
\tag{20}
\]

共享 causal decoder \(D\) 产生最终重建：

\[
\hat x=D(z).
\tag{21}
\]

式（20）是方法的核心 composition rule。由于融合是加法，part correction 对排列顺序不敏感：

\[
q_b+\sum_{p\in\mathcal P}\Delta z_p
=q_b+\sum_{p\in\pi(\mathcal P)}\Delta z_p,
\tag{22}
\]

其中 \(\pi\) 是任意部位排列。该性质使各 part correction 可以被独立加入、移除或替换。

需要注意，\(A_p\) 是 full-latent projector，而非固定子空间注入。一般情况下允许

\[
\operatorname{Im}(A_p)\cap\operatorname{Im}(A_{p'})\neq\{0\}.
\tag{23}
\]

因此，方法不要求 decoder latent 具有显式身体拓扑；局部性通过 token factorization 和编辑监督学习，而不是由 latent channel disjointness 硬编码。

## 6. Compensated Residual Editing

### 6.1 为什么不能直接交换 residual

考虑 target motion \(x^t\) 和 donor motion \(x^d\)。它们对应的 part residual 分别满足

\[
q_p^t\approx u_p^t-P_p(q_b^t),
\qquad
q_p^d\approx u_p^d-P_p(q_b^d).
\tag{24}
\]

若直接以 \(q_p^d\) 替换 \(q_p^t\)，则 donor residual 仍相对于 \(q_b^d\) 定义，而最终组合使用的是 \(q_b^t\)。这种操作忽略了条件基准变化：

\[
P_p(q_b^d)\neq P_p(q_b^t).
\tag{25}
\]

### 6.2 Donor state recovery

利用 donor base prediction 和 donor residual，可近似恢复 donor 的完整局部状态：

\[
\tilde u_p^d
=P_p(\operatorname{sg}(q_b^d))+q_p^d.
\tag{26}
\]

这里的 \(q_p^d\) 已是量化后 residual state，故 \(\tilde u_p^d\) 是量化精度下的 donor local state estimate。

### 6.3 Target-conditioned re-expression

将 donor local state 改写为相对于 target base 的 residual：

\[
\tilde r_p^{d\rightarrow t}
=\tilde u_p^d-P_p(\operatorname{sg}(q_b^t)).
\tag{27}
\]

对应的 edited latent correction 为

\[
\widetilde{\Delta z}_p^{d\rightarrow t}
=A_p\tilde r_p^{d\rightarrow t}.
\tag{28}
\]

式（27）不经过第二次 FSQ。编辑是在已解码的量化 residual state 上完成条件坐标变换，再直接投影到 decoder latent。

### 6.4 Residual replacement

Target 的完整 latent 为

\[
z^t=q_b^t+\sum_{j\in\mathcal P}A_jq_j^t.
\tag{29}
\]

仅编辑部位 \(p\) 时，移除 target correction 并加入 compensated donor correction：

\[
z_{p}^{d\rightarrow t}
=z^t-A_pq_p^t
+A_p\left[
P_p(q_b^d)+q_p^d-P_p(q_b^t)
\right].
\tag{30}
\]

编辑结果为

\[
\hat x_{p}^{d\rightarrow t}=D(z_{p}^{d\rightarrow t}).
\tag{31}
\]

这一步只改变一个 part correction，并保持 target base 和其余 part correction 不变。

## 7. 精炼 Pipeline

### 7.1 表示学习与重建

\[
\boxed{
\begin{aligned}
q_b &= Q_b(E_b(x)),\\
q_p &= Q_p\!\left(E_p(S_p(x))-P_p(\operatorname{sg}(q_b))\right),
&&p\in\mathcal P,\\
z &= q_b+\sum_{p\in\mathcal P}A_pq_p,\\
\hat x &= D(z).
\end{aligned}}
\tag{32}
\]

### 7.2 局部编辑

\[
\boxed{
\begin{aligned}
\tilde u_p^d
&=P_p(\operatorname{sg}(q_b^d))+q_p^d,\\
\tilde r_p^{d\rightarrow t}
&=\tilde u_p^d-P_p(\operatorname{sg}(q_b^t)),\\
z_p^{d\rightarrow t}
&=z^t-A_pq_p^t+A_p\tilde r_p^{d\rightarrow t},\\
\hat x_p^{d\rightarrow t}
&=D(z_p^{d\rightarrow t}).
\end{aligned}}
\tag{33}
\]

### 7.3 端到端结构

```text
                                     q_base
x --> holistic encoder --> Base-FSQ -----+------------------------+
                                         |                        |
S_p(x) --> part encoder --> u_p           |                        |
                              P_p(sg(q_base))                       |
                                      |                            |
u_p - P_p(sg(q_base)) --> Part-FSQ --> q_p --> A_p --> delta_z_p  |
                                                                  |
                           z = q_base + sum_p(delta_z_p) <---------+
                                                                  |
                                      shared causal decoder --> x_hat
```

## 8. 学习目标

### 8.1 Final reconstruction objective

最终 latent 重建由组合损失监督：

\[
\mathcal L_{final}
=\lambda_x\mathcal L_x
+\lambda_\Delta\mathcal L_\Delta
+\lambda_r\mathcal L_{root}
+\lambda_j\mathcal L_{joint}
+\lambda_c\mathcal L_{contact}
+\lambda_f\mathcal L_{foot}.
\tag{34}
\]

特征重建项为

\[
\mathcal L_x
=\frac{1}{TD}\sum_{t=1}^{T}
\left\|w\odot(\hat x_t-x_t)\right\|_1,
\tag{35}
\]

时序差分项为

\[
\mathcal L_\Delta
=\frac{1}{(T-1)D}\sum_{t=2}^{T}
\left\|
(\hat x_t-\hat x_{t-1})-(x_t-x_{t-1})
\right\|_1.
\tag{36}
\]

其余项分别约束 root trajectory、前向运动学关节位置、接触状态、支撑脚滑移和足部高度。它们作用于最终重建 \(\hat x\)，共同训练 base、part encoders、projectors 和共享 decoder。

### 8.2 Base sufficiency objective

为防止 part streams 承担全部信息，base latent 单独解码为

\[
\hat x_b=D(q_b),
\tag{37}
\]

并施加低权重重建监督：

\[
\mathcal L_{base}
=\frac{1}{TD}\sum_{t=1}^{T}
\left\|w\odot(\hat x_{b,t}-x_t)\right\|_1.
\tag{38}
\]

该目标使 \(q_b\) 成为具有独立解释能力的全身基础表示，而 part tokens 学习在此基础上的局部修正。

### 8.3 Editability objective

对随机或 batch 内匹配的 target-donor pair \((x^t,x^d)\)，选取部位 \(p\)，并由式（33）得到编辑结果 \(\hat x_p^{d\rightarrow t}\)。目标部位迁移损失为

\[
\mathcal L_{tr}^{p}
=\frac{1}{T|I_p|}
\left\|
S_p(\hat x_p^{d\rightarrow t})-S_p(x^d)
\right\|_1.
\tag{39}
\]

非目标区域保持损失为

\[
\mathcal L_{pr}^{p}
=\frac{1}{T|\bar I_p|}
\left\|
S_{\bar p}(\hat x_p^{d\rightarrow t})-S_{\bar p}(x^t)
\right\|_1,
\tag{40}
\]

其中 \(\bar I_p\) 是 \(I_p\) 的特征补集。前者要求目标部位趋近 donor，后者抑制对 target 其余运动的扰动。

### 8.4 Overall objective

完整训练目标为

\[
\boxed{
\mathcal L
=\mathcal L_{final}
+\lambda_b\mathcal L_{base}
+\lambda_e\mathbb E_{p,t,d}
\left[
\mathcal L_{tr}^{p}
+\lambda_{pr}\mathcal L_{pr}^{p}
\right].}
\tag{41}
\]

该目标只使用重建、base sufficiency 和 editability 三类监督。方法不依赖显式 latent orthogonality、cross-part decorrelation 或 residual-energy penalty。

## 9. 方法性质

### 9.1 同源编辑恒等性

当 donor 与 target 相同时，\(q_b^d=q_b^t\) 且 \(q_p^d=q_p^t\)。由式（27）有

\[
\tilde r_p^{t\rightarrow t}=q_p^t.
\tag{42}
\]

进而

\[
z_p^{t\rightarrow t}
=z^t-A_pq_p^t+A_pq_p^t=z^t.
\tag{43}
\]

因此，同源编辑严格退化为原始重建。这是 compensated replacement 的代数性质，不依赖训练效果。

### 9.2 表示级保持性

编辑部位 \(p\) 时，target base 与其他部位 correction 在 latent 构造中严格保持：

\[
q_b^{edit}=q_b^t,
\qquad
\Delta z_j^{edit}=\Delta z_j^t,
\quad j\neq p.
\tag{44}
\]

因此，方法在 representation level 上实现单部位 replacement。

### 9.3 输出局部性不是结构定理

尽管式（44）成立，一般不能推出

\[
S_{\bar p}(D(z_p^{d\rightarrow t}))
=S_{\bar p}(D(z^t)).
\tag{45}
\]

原因是 \(A_p\) 作用于完整 latent，且共享 decoder 通常是非线性的。对小编辑 \(\delta z_p\)，一阶近似为

\[
D(z^t+\delta z_p)-D(z^t)
\approx J_D(z^t)\delta z_p,
\tag{46}
\]

其中 decoder Jacobian \(J_D\) 不必具有 body-part block structure。非目标保持依赖 \(\mathcal L_{pr}\) 对 \(J_D(z)A_p\) 的训练约束，而不是架构硬保证。

### 9.4 条件补偿的必要性

直接交换与补偿交换的 latent 差异为

\[
\delta z_{comp}-\delta z_{swap}
=A_p\left[P_p(q_b^d)-P_p(q_b^t)\right].
\tag{47}
\]

该项显式修正 donor 与 target 的 base-conditioned local-state offset。当二者 base prediction 相同，补偿编辑自然退化为直接 residual swap。

### 9.5 因果性

若 \(E_b\)、\(E_p\) 和 \(D\) 均为 causal temporal operators，而 \(Q_g\)、\(P_p\) 和 \(A_p\) 均逐帧作用，则

\[
\hat x_t=f(x_{\le t}),
\tag{48}
\]

编辑情况下有

\[
\hat x_{p,t}^{d\rightarrow t}
=f_p(x^t_{\le t},x^d_{\le t}).
\tag{49}
\]

因此，aligned target-donor 编辑不引入未来信息。当前实例的有效 receptive field 为 64 帧，lookahead 为 0。

### 9.6 单次解码

重建和编辑都先在 latent side 完成组合：

\[
\hat x=D(z),
\qquad
\hat x_p^{d\rightarrow t}=D(z_p^{d\rightarrow t}).
\tag{50}
\]

每个推理结果只需一次 decoder evaluation。该性质来自“先融合 latent、后统一解码”，也是相较 feature-side residual reconstruction 的主要计算优势。

## 10. 理论定位

LRP-FSQ 可以从三个互补视角理解。

### 10.1 条件残差编码

Base stream 建模共享全身上下文，part stream 只编码条件创新量：

\[
\text{part information}
=\text{local state}-\text{base-predictable local state}.
\tag{51}
\]

这使其不同于独立编码完整局部状态的 Part-FSQ。

### 10.2 加性 latent composition

局部信息不占据预定义 channel，而是作为对统一 decoder latent 的可组合更新：

\[
\text{unified latent}
=\text{holistic base}+\text{part-specific updates}.
\tag{52}
\]

该形式保留了 residual model 的优化路径和替换接口，同时允许 decoder 自主组织 latent factors。

### 10.3 条件坐标迁移

Part residual 不能脱离其 base condition 解释。Compensated editing 先恢复 donor local state，再相对于 target base 重新定义 residual：

\[
\text{donor residual under donor base}
\rightarrow
\text{donor local state}
\rightarrow
\text{donor residual under target base}.
\tag{53}
\]

因此，局部编辑不是简单 token substitution，而是 base-conditioned residual transport。

## 11. 可验证假设与实验对应

理论机制应通过分离式实验验证，而不应仅以总体 reconstruction error 支撑。

### H1：Residual factorization 提高 base 完整性与 part 增量性

比较完整模型、去除 \(\mathcal L_{base}\) 和无条件 part encoding：

\[
q_p=Q_p(E_p(x_p)).
\tag{54}
\]

应分别报告 base-only reconstruction、full reconstruction 和 part-code utilization。

### H2：Compensation 优于直接 residual swap

直接交换基线为

\[
z_{swap}=z^t-A_pq_p^t+A_pq_p^d.
\tag{55}
\]

与式（30）比较 target-part transfer error 和 non-target leakage。差异应在 donor/target 全身上下文差异较大时更明显。

### H3：Preserve objective 负责输出局部性

移除 \(\mathcal L_{pr}\) 后，式（44）的表示级保持仍成立，但式（45）的输出级保持预计恶化。该 ablation 用于区分“token replacement 正确”与“decoder 响应局部”两个不同性质。

### H4：Full-latent projection 在重建自由度与局部泄漏之间存在权衡

将稠密 \(A_p\) 与显式 disjoint projection、固定 mask 或 concat-based latent 比较。至少报告：

\[
E_{transfer}^{p}
=\left\|S_p(\hat x_p^{d\rightarrow t})-S_p(x^d)\right\|,
\tag{56}
\]

\[
E_{leak}^{p}
=\left\|S_{\bar p}(\hat x_p^{d\rightarrow t})-S_{\bar p}(x^t)\right\|,
\tag{57}
\]

以及整体 reconstruction、temporal consistency 和 inference cost。

## 12. 适用于论文写作的贡献表述

在实验结果支持后，可将方法贡献概括为：

1. 一种 base-conditioned residual tokenization，将 holistic motion context 与可替换的局部离散修正统一在同一表示中；
2. 一种不显式切分 decoder latent 的加性 composition mechanism，使每个 part token 通过独立稠密投影更新共享 latent，并保持一次解码；
3. 一种 compensated residual editing 操作，将 donor residual 从 donor-base 条件重表达到 target-base 条件，以支持上下文一致的局部运动迁移；
4. 一组简洁的 base-sufficiency 与 editability objectives，在不引入显式正交约束的情况下学习局部可编辑性。

这些表述应使用“supports”或“enables”而不是“guarantees”。方法在 representation level 保证单 part correction replacement，但 output-space locality 仍需 transfer/leakage 实验验证。

## 13. 实例化参数

当前实现对上述一般形式采用以下实例：

| 符号 | 含义 | 数值 |
| --- | --- | ---: |
| \(D\) | motion feature dimension | 230 |
| \(T\) | training window | 64 |
| \(d_z\) | base/decoder latent dimension | 128 |
| \(d_p\) | part residual-state dimension | 64 |
| \(K\) | FSQ coordinates per frame | 40 |
| \(L\) | levels per coordinate | 9 |
| \(\lambda_b\) | base reconstruction weight | 0.1 |
| \(\lambda_e\) | edit objective weight | 0.25 |
| \(\lambda_{pr}\) | non-target preserve multiplier | 1.0 |

左右腿共享 part encoder 与 quantizer，左右臂共享 part encoder 与 quantizer；五个部位使用独立的 predictor 和 full-latent projector。所有 temporal encoder/decoder 均为 frame-causal network，表示不执行 temporal downsampling。

## 14. 适用范围与限制（Limitations and Failure Modes）

LRP-FSQ 是离散运动表示和编辑模型，而不是物理控制器。它可作为后续 autoregressive motion model、motion prior 或 physics-based controller 的 token interface，但当前目标本身不保证动力学可行性、扰动恢复或闭环控制能力。

其主要限制包括：

1. 局部 residual 定义在 learned part-state space，不等价于真实 motion feature reconstruction residual；
2. 稠密 projector 与 nonlinear decoder 不提供输出局部性的结构保证；
3. 基于逐帧 L1 的 donor supervision 假设 target 与 donor 在时间上具有可比较的相位；
4. 随机 donor pair 可能同时包含风格、速度和全局状态差异，使局部迁移目标存在歧义；
5. 部位划分依赖预定义 skeleton semantics，尚未学习层级化或动态 part decomposition；
6. 离散表示的生成质量、长时稳定性和物理可执行性需要在下游模型中独立评估。

因此，论文中的核心经验主张应聚焦于 reconstruction、discrete representation quality、local transfer、non-target preservation 和 inference efficiency，并将物理控制能力留给明确接入 simulator/controller 后的后续验证。

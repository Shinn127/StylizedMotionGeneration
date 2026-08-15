# Latent Residual Part-FSQ V2 技术说明

本文档描述仓库中 `latent_residual_fsq_v2` 的**当前代码实现**，用于解释模型的技术路线、核心公式、训练目标、局部编辑机制和实现边界。它不是脱离代码的候选设计。

对应实现：

- 模型：`stylized_motion/learning/latent_residual_fsq_v2.py`
- FSQ：`stylized_motion/learning/fsq.py`
- 部位与特征布局：`stylized_motion/learning/part_layout.py`
- 损失：`stylized_motion/learning/losses.py`
- 训练调度：`stylized_motion/learning/runner.py`
- 注册与 checkpoint contract：`stylized_motion/learning/representation.py`
- 默认配置：`data/configs/latent_residual_fsq_v2_40x9.yaml`

## 1. 设计目标

V2 要同时满足以下目标：

1. `q_base` 和 decoder 接收的 latent 都是普通的稠密向量，不按身体部位切分 latent channel。
2. 每个部位具有独立、可替换的离散 token。
3. 部位信息以 latent residual 的形式叠加到 holistic base latent 上。
4. 标准重建和单部位编辑在推理时都只执行一次共享 decoder。
5. 局部编辑不仅迁移目标部位，还显式约束非目标部位保持不变。
6. V2 作为独立 representation family 与旧版 `latent_residual_fsq` 共存，便于直接比较。

因此，V2 的核心不是构造显式 part-structured decoder latent，而是学习五个从局部状态空间到同一个 decoder latent 空间的稠密投影：

\[
z = q_b + \sum_{p \in \mathcal P} \Delta z_p,
\qquad
\Delta z_p = A_p q_p
\]

其中：

- \(\mathcal P=\{\text{torso},\text{left leg},\text{right leg},\text{left arm},\text{right arm}\}\)；
- \(q_b\in\mathbb R^{128}\) 是 holistic base 的量化 latent；
- \(q_p\in\mathbb R^{64}\) 是部位 \(p\) 的量化 residual state；
- \(A_p:\mathbb R^{64}\rightarrow\mathbb R^{128}\) 是该部位独立的无 bias 线性投影；
- \(z\in\mathbb R^{128}\) 是共享 decoder 的输入。

`q_b`、每个 `A_p q_p` 和最终 `z` 都覆盖完整的 128 维 decoder latent。代码中不存在 `part_latent_slices`、固定 channel mask、part concat 或预定义正交基。

## 2. Representation Contract

| 属性 | 当前值 |
| --- | --- |
| family | `latent_residual_fsq_v2` |
| variant | `v2` |
| representation id | `latent_residual_fsq_v2_40x9` |
| architecture version | `3` |
| 输入 | `[B, T, 230]`，默认训练时 `T=64` |
| base latent 维度 | `128` |
| part state 维度 | `64` |
| 时间下采样 | `1`，每帧一个 token tuple |
| receptive field | `64` 帧 |
| lookahead | `0`，严格 causal |
| FSQ levels | 每坐标 `9` 级 |
| token coordinates | 每帧共 `40` 个 |
| 推理 decoder passes | `1` |

40 个 FSQ 坐标的顺序与数量固定为：

| group | base | torso | left_leg | right_leg | left_arm | right_arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coordinates | 20 | 6 | 4 | 4 | 3 | 3 |

总坐标数为：

\[
20+6+4+4+3+3=40.
\]

这里必须区分两个概念：

- `40 x 9` 描述离散 token contract，即 40 个标量坐标、每个坐标 9 个 level；
- `128` 和 `64` 描述 FSQ `project_out` 后的连续 latent/state 维度。

40 个 token 坐标没有被直接当作 decoder latent channel。

## 3. Motion Feature 与部位划分

输入记为：

\[
x\in\mathbb R^{B\times T\times D},\qquad D=230.
\]

`PartFSQLayout` 将所有非 root joint 唯一地划分到五个部位。每个 \(x_p\) 只抽取该部位关节的 6D rotation 和 angular velocity 特征：

\[
x_p = \operatorname{Select}(x, I_p).
\]

五个 \(I_p\) 互不重叠，但它们不覆盖 root/global 和 contact 特征；这些信息由 holistic base stream 处理。局部编辑损失中的“目标部位”也使用同一组 \(I_p\)，而其补集被视为需要保持的特征。

左右腿共享一套 part encoder 和一套 FSQ；左右臂同样共享；torso 单独使用一套。五个部位仍各自拥有独立的 predictor 和 latent projector：

| 模块 | torso | left/right leg | left/right arm |
| --- | --- | --- | --- |
| part encoder | 独立 | 左右共享 | 左右共享 |
| part FSQ | 独立 | 左右共享 | 左右共享 |
| base predictor | 每个部位独立 | 每个部位独立 | 每个部位独立 |
| latent projector | 每个部位独立 | 每个部位独立 | 每个部位独立 |

共享 encoder/FSQ 让镜像肢体使用相同的编码规则；独立 predictor/projector 则允许左右侧在 holistic base 条件和 decoder latent 中学习不同映射。

## 4. 编码路径

### 4.1 Holistic base stream

完整 motion 先经过 frame-causal encoder：

\[
h_b = E_b(x),
\qquad h_b\in\mathbb R^{B\times T\times 128}.
\]

Base-FSQ 使用 20 个 9-level 坐标量化 \(h_b\)：

\[
(c_b,i_b,q_b)=Q_b(h_b).
\]

其中 \(c_b\) 是量化后的标量 code，\(i_b\) 是离散 level index，\(q_b\) 是 FSQ `project_out` 后的 128 维 latent。

### 4.2 Local residual stream

对每个部位 \(p\)，先从真实 motion 中抽取局部特征并编码成 64 维局部状态：

\[
u_p = E_p(x_p),
\qquad u_p\in\mathbb R^{B\times T\times 64}.
\]

再由量化后的 base latent 预测该部位在 base 条件下应具有的局部状态：

\[
\hat u_p=P_p(\operatorname{sg}(q_b)).
\]

`sg` 表示 `detach/stop-gradient`。局部 residual state 定义为：

\[
r_p = u_p-\hat u_p.
\]

随后量化 residual state：

\[
(c_p,i_p,q_p)=Q_p(r_p).
\]

最后将 64 维量化 residual state 投影到完整的 128 维 decoder latent：

\[
\Delta z_p=A_pq_p.
\]

### 4.3 “latent residual”的准确含义

当前实现的 residual 不是 motion feature space 中的：

\[
x_p-D(q_b)_p.
\]

它是在学习到的局部状态空间中形成的：

\[
E_p(x_p)-P_p(\operatorname{sg}(q_b)),
\]

然后被量化并投影成 decoder latent residual \(\Delta z_p\)。因此更准确的描述是：

> V2 在 learned part-state space 中估计 residual，在 shared decoder-latent space 中执行 residual fusion。

这种设计避免为了获得 feature residual 而先运行一次 base decoder，同时仍保留“base 表示公共信息、part token 表示 base 条件下局部增量”的 residual 语义。

`q_b.detach()` 还阻止 part predictor 分支通过条件输入反向改写 base encoder；但最终重建路径中的 \(q_b\rightarrow z\rightarrow D(z)\) 没有 detach，base stream 仍会从最终重建损失获得梯度。

## 5. Latent 融合与重建

五个部位 residual 使用加法累积：

\[
\Delta z=\sum_{p\in\mathcal P}A_pq_p,
\qquad
z=q_b+\Delta z.
\]

最终重建为：

\[
\hat x=D(z).
\]

加法有三项直接意义：

1. 保留标准 residual 形式，`base + correction` 的职责明确；
2. 每个部位 correction 可以独立移除和替换；
3. decoder 只接收一个普通的 128 维 latent，不需要知道 part layout。

项目没有对 \(A_p\) 施加显式正交约束。不同部位可以使用重叠的 latent 方向：

\[
\operatorname{span}(A_p)\cap\operatorname{span}(A_{p'})\neq\varnothing.
\]

这不是实现错误，而是“非结构化 decoder latent”的直接代价。当前代码依赖输入部位划分、独立 projector 以及 edit transfer/preserve loss 学出语义局部性。

## 6. 完整 Pipeline

### 6.1 标准重建

```text
x [B,T,230]
|
+--> Base causal encoder --> h_base [B,T,128]
|                         --> Base-FSQ (20 x 9)
|                         --> q_base [B,T,128]
|
+--> select torso features ----> Torso encoder --> residual against P_torso(sg(q_base))
|                                                --> Torso-FSQ (6 x 9) --> q_torso [B,T,64]
|
+--> select left-leg features -> shared Leg encoder --> residual against P_left_leg(sg(q_base))
|                                                   --> shared Leg-FSQ (4 x 9) --> q_left_leg [B,T,64]
|
+--> select right-leg features -> shared Leg encoder --> residual against P_right_leg(sg(q_base))
|                                                    --> shared Leg-FSQ (4 x 9) --> q_right_leg [B,T,64]
|
+--> select left-arm features -> shared Arm encoder --> residual against P_left_arm(sg(q_base))
|                                                   --> shared Arm-FSQ (3 x 9) --> q_left_arm [B,T,64]
|
`--> select right-arm features -> shared Arm encoder --> residual against P_right_arm(sg(q_base))
                                                     --> shared Arm-FSQ (3 x 9) --> q_right_arm [B,T,64]

q_part --> independent dense projector A_part: R^64 -> R^128 --> delta_z_part

z = q_base
    + delta_z_torso
    + delta_z_left_leg
    + delta_z_right_leg
    + delta_z_left_arm
    + delta_z_right_arm

z [B,T,128] --> one shared causal decoder --> x_hat [B,T,230]
```

### 6.2 单部位编辑

```text
target tokens --> target q_base + all target part residuals --> z_target
donor tokens  --> donor q_base and donor q_part

donor q_base + donor q_part
        |
        `--> recover donor part state under donor base
             --> re-express it relative to target base
                  --> project to edited delta_z_part

z_edit = z_target - target delta_z_part + edited delta_z_part

z_edit --> one shared causal decoder --> edited motion
```

## 7. 补偿式局部编辑

Part token \(q_p\) 表示的是“相对于自身 base 的 residual”，所以不能直接把 donor 的 \(q_p^d\) 替换进 target。直接替换隐含地假设：

\[
P_p(q_b^d)=P_p(q_b^t),
\]

这通常不成立。

V2 先恢复 donor 的完整局部状态：

\[
u_p^d\approx P_p(\operatorname{sg}(q_b^d))+q_p^d.
\]

再将它改写为 target base 条件下的 residual：

\[
\tilde r_p^{d\rightarrow t}
=u_p^d-P_p(\operatorname{sg}(q_b^t)).
\]

投影后的编辑 correction 为：

\[
\widetilde{\Delta z}_p^{d\rightarrow t}
=A_p\tilde r_p^{d\rightarrow t}.
\]

最后只替换目标部位的 latent correction：

\[
z_{edit}
=z_t-\Delta z_p^t+\widetilde{\Delta z}_p^{d\rightarrow t}.
\]

其余部位 residual 和 target base 均保持不变。这里的“补偿”指 donor residual 先还原到 donor local state，再减去 target predictor，从 donor base 坐标系转换到 target base 坐标系。

## 8. 训练目标

总损失由通用最终重建损失和 V2 专用损失组成：

\[
\mathcal L
=\mathcal L_{final}
+\lambda_b\mathcal L_{base}
+\lambda_e\mathcal L_{transfer}
+\lambda_e\lambda_{preserve}\mathcal L_{preserve}.
\]

### 8.1 最终重建损失

`recon_state = D(z)` 使用项目统一的 motion reconstruction loss：

\[
\mathcal L_{final}
=\mathcal L_{recon}
+\lambda_\Delta\mathcal L_\Delta
+\lambda_{root\_pos}\mathcal L_{root\_pos}
+\lambda_{root\_rot}\mathcal L_{root\_rot}
+\lambda_{joint}\mathcal L_{joint}
+\lambda_{contact}\mathcal L_{contact}
+\lambda_{slide}\mathcal L_{foot\_slide}
+\lambda_{height}\mathcal L_{foot\_height}.
\]

其中 `recon` 是按 feature weight 加权的 L1，`delta` 约束相邻帧运动增量，其余项约束 root trajectory、FK joint position、contact 与 foot behavior。FSQ 的 `commit_loss` 为零，训练器的 commit weight 也固定为零。

默认权重为：

| 项 | 权重 |
| --- | ---: |
| delta | `3.0` |
| root position | `0.1` |
| root rotation | `0.1` |
| joint | `0.5` |
| contact | `0.1` |
| foot slide | `0.1` |
| foot height | `0.1` |

### 8.2 Base reconstruction supervision

训练器始终请求：

\[
\hat x_b=D(q_b).
\]

并计算低权重的 feature-weighted L1：

\[
\mathcal L_{base}
=\operatorname{mean}\left(w\odot|\hat x_b-x|\right),
\qquad \lambda_b=0.1.
\]

它要求 base stream 自身保留完整 motion 的粗粒度表达，防止所有信息都迁移到 part streams。它不是与 `final` 相同的一整套运动学损失。

### 8.3 Edit transfer 与 preserve

训练时每一步按 `global_step mod 5` 轮换一个编辑部位，batch 内 donor 使用循环移位：

```text
donor_permutation = roll([0, 1, ..., B-1], shifts=1)
```

设 \(I_p\) 是当前编辑部位的 feature mask，则：

\[
\mathcal L_{transfer}
=\operatorname{mean}\left(
|\hat x_{edit}[I_p]-x_d[I_p]|
\right),
\]

\[
\mathcal L_{preserve}
=\operatorname{mean}\left(
|\hat x_{edit}[\neg I_p]-x_t[\neg I_p]|
\right).
\]

默认 \(\lambda_e=0.25\)，\(\lambda_{preserve}=1.0\)。当 batch size 为 1 时，runner 不构造 donor edit，因此该 step 没有 edit loss；默认 batch size 为 512，不触发此退化路径。

V2 没有 latent energy、projector orthogonality、cross-part decorrelation 或额外 code reuse loss。局部性主要由简单的 transfer/preserve 监督提供。

## 9. Decoder 计算口径

“一次 decoder”需要区分模块调用次数与实际 batch 计算量：

| 场景 | decoder 输入 | 模块调用次数 | 等效 batch 计算量 |
| --- | --- | ---: | ---: |
| 标准重建推理 | `z` | 1 | `B` |
| 单部位编辑推理 | `z_edit` | 1 | `B` |
| base-only decode | `q_base` | 1 | `B` |
| 默认训练 | concat(`q_base`, `z`, `z_edit`) | 1 | `3B` |
| validation/test | concat(`q_base`, `z`) | 1 | `2B` |

训练实现把多个 latent 沿 batch 维拼接后统一调用 decoder，再将输出拆回 `base/recon/edit`。这减少 Python 和 module invocation 开销，但训练时 decoder FLOPs/显存仍近似按 `3B` 计算。用户侧标准重建和局部编辑 API 均只解码一个 latent，满足推理一次 decoder 的设计目标。

## 10. 推理与 Token API

### 编码

- `encode_to_indices(x)`：返回 `[B,T,40]` 的离散 FSQ indices。
- `encode_to_codes(x)`：返回 `[B,T,40]` 的 quantized codes 及对应 indices。

### 解码

- `decode_from_indices(indices)`：反量化六组 token、融合 residual、一次 decoder 重建。
- `decode_from_codes(codes)`：从 quantized scalar codes 执行同一路径。
- `decode_base_from_indices(indices)`：只解码 target base group，part token 被忽略。
- `decode_base_from_codes(codes)`：上述接口的 code 版本。
- `decode_from_indices_with_part_edit(target, donor, part)`：执行补偿式 part edit，并只解码一次编辑 latent。

编辑 API 要求 target 与 donor 的 base embedding shape 相同。当前公开实现提供 index-based part edit，没有对应的 `decode_from_codes_with_part_edit`。

## 11. 设计原理与取舍

### 11.1 为什么不是显式 part latent concat

如果预先规定 decoder latent 的某些 channel 只能属于某个身体部位，局部性具有更强的结构保证，但同时会限制 base 和 decoder 自主组织运动因素。V2 选择稠密 \(A_p\)，让 base 和 decoder latent 保持非结构化，并把 part 语义放在 encoder、token group、projector 和训练目标上。

### 11.2 为什么仍然是 residual，而不是退化成 Part-FSQ

Part token 编码的不是无条件局部状态 \(u_p\)，而是：

\[
u_p-P_p(q_b).
\]

最终 decoder latent 也不是 part latent concat，而是：

\[
q_b+\sum_p A_pq_p.
\]

因此 base 是主表示，part 是 base-conditioned correction；编辑时还必须完成 donor-base 到 target-base 的 residual 重表达。这三点共同构成 residual 语义。

### 11.3 为什么不先解码 base 再计算 feature residual

feature residual 路线需要先得到 \(D(q_b)\)，再计算局部输出误差并重新编码。V2 用 predictor \(P_p(q_b)\) 直接在 learned state space 估计 base 可解释部分，避免把 base decoder 放在 part encoder 前面，也避免标准推理中的额外 decoder pass。

代价是 \(P_p(q_b)\) 不等于 \(E_p(D(q_b)_p)\)，更不等于真实 feature-space base reconstruction；residual 的质量取决于 predictor 与局部状态空间的联合学习。

### 11.4 局部性的保证强度

当前结构对“哪个 token 可被替换”有明确保证，但不对“哪个 latent channel 或输出 feature 只受该 token 影响”提供硬保证。局部编辑能力是软约束学习结果：

- 有利因素：部位输入 mask、独立 predictor/projector、逐部位 edit transfer、非目标 preserve；
- 风险因素：五个 projector 的输出子空间可重叠，共享 nonlinear decoder 可把任意 latent 方向传播到全身。

因此评估不能只看 reconstruction loss，还必须测量目标部位 transfer 和非目标部位 leakage。

## 12. 当前实现审查结论

当前代码与上述 V2 技术路线一致：

1. base latent 和最终 decoder latent 均为非结构化 `[B,T,128]`；
2. 五个 part residual 均通过独立的 full-latent projector 加入同一 base；
3. 融合采用显式逐项加法，没有 concat 或固定 latent slice；
4. donor edit 使用了 base-conditioned compensation，而不是直接交换 residual token；
5. 训练同时监督 final reconstruction、低权重 base reconstruction 和局部编辑；
6. 推理重建及编辑路径各只执行一次 decoder；
7. V2 通过独立 family、`architecture_version=3` 和 representation id 与旧 checkpoint 隔离。

需要持续关注的限制：

- projector 子空间没有正交或去相关约束，part leakage 只能通过 edit loss 被软抑制；
- predictor residual 定义在 learned state space，不等价于真实 feature reconstruction residual；
- edit supervision 是随机 batch donor 的逐特征 L1，可能惩罚 donor 与 target 在时序相位或全局条件上的合理不一致；
- 当前单元测试验证了维度契约、一次 decoder 调用、causality 和 target base 不变，但局部 transfer/leakage 的效果需要在训练后 checkpoint 上做数据集级评估。

## 13. 默认配置与启动方式

默认训练配置：

```text
data/configs/latent_residual_fsq_v2_40x9.yaml
```

关键训练参数：

| 参数 | 值 |
| --- | ---: |
| epochs | `100` |
| batch size | `512` |
| learning rate | `5e-4` |
| precision | `fp32` |
| samples per epoch | `100000` |
| window | `64` frames |
| base reconstruction weight | `0.1` |
| edit weight | `0.25` |
| edit preserve weight | `1.0` |
| output directory | `outputs/latent_residual_fsq_v2_40x9` |

训练命令：

```bash
python -m stylized_motion.run \
  --arg-file args/latent_residual_fsq_v2_args.txt
```

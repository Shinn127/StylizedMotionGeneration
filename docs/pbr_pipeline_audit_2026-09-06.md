# GenoView / SomaView PBR 渲染管线审计与推进方案

日期：2026-09-06  
审计范围：`stylized_motion/anim`、`data/assets/genoview`、`data/assets/somaview`、PBR 基线与相关测试。  
审计基线：`main` at `7ac81be`，包含当前工作区未提交的 PBR 修订。

## 1. 结论

当前实现已经不是 PBR 原型，而是一条可运行的 minimal deferred PBR 管线。GenoView 和 SomaView 均通过同一个 Python `Renderer` 调度以下链路：

```text
3-cascade CSM / 4-moment EVSM
  -> GBuffer (baseColor+metallic, normal+roughness, material AO, depth)
  -> horizon-based GTAO
  -> bilateral AO blur
  -> Cook-Torrance direct light + IBL
  -> RGBA16F HDR
  -> ACESApprox / Reinhard / AgX
  -> FXAA
  -> persistent final target / window / PNG / video
```

结构拆分、资源销毁、离屏输出和 debug view 已基本建立，Geno/SOMA 的运行时 shader 也保持一致。实际离屏运行中，两套 viewer 的 shader 全部编译成功，FBO 创建和正常退出清理成功，当前输出与提交中的对应基线逐像素一致。

但还不能把“材质纹理和 IBL 已完整落地”视为完成。最高优先级缺口是索引网格切线生成不正确，SOMA 的 face-varying UV 又在资产转换时被平均成 per-vertex UV；两者共同使 normal map 和一般材质纹理在 SOMA 上没有可靠语义。IBL 的 GGX 预过滤与 lighting 端也缺少关键能量权重。现有测试大多检查源码字符串，能证明接口存在，不能证明像素或物理行为正确。

建议把当前状态定义为：**PBR 主链路完成，纹理材质与 IBL 正确性未封板，工程健壮性和自动化验收仍需补齐。**

## 2. 实现盘点

| 子系统 | 状态 | 现状 | 结论 |
| --- | --- | --- | --- |
| Pass 调度 | 已落地 | `Renderer.render_frame()` 串联 shadow、GBuffer、SSAO、lighting、tone map、presentation | 顺序正确，viewer 主循环已基本脱离裸 pass 调度 |
| GBuffer | 已落地 | RGBA8 albedo/metallic、RGBA16 normal/roughness、R8 material AO、depth texture | 契约清晰；仍缺 attachment 能力检查和失败回滚 |
| Direct PBR | 已落地 | GGX NDF、Smith、Schlick、metallic workflow、directional light | 主公式正确，roughness/NdotV 有数值保护 |
| Shadow | 部分落地 | 3 级 CSM、texel snapping、4-moment EVSM、半分辨率 separable blur、级联混合 | 可用，但计算出的 bias 未消费，caster coverage 和动态范围缺测试 |
| SSAO | 已落地 | 4 方向双侧、每侧 4 步 horizon search，双边模糊 | 职责已和 shadow 分离；默认效果很弱，尚无定量验收 |
| HDR / tone map | 已落地 | lighting 输出 RGBA16F；独立 tone-map pass；FXAA 在其后 | 颜色管线边界成立 |
| IBL | 部分落地 | 程序化 HDR sky、irradiance、GGX prefilter mip、BRDF LUT、fallback | 资源链完整，但预过滤积分和间接漫反射能量项需修正 |
| Material | 部分落地 | uniform + base color / normal / packed MRA texture，per-object material | 仅 CLI 默认材质入口；纹理校验、过滤、UV/tangent 正确性不足 |
| Debug | 已落地 | 12 种视图，lighting 与 GBuffer debug 分流 | 可诊断；背景行为与文档旧描述不一致，AO 未显示 combined AO |
| Geno/SOMA 共用 | 已落地 | 共用 `GenoView`/`Renderer`，SOMA 仅替换 rig 和资源；shader 有字节一致测试 | 逻辑共用成立；shader 仍以复制文件分发 |
| 生命周期 | 部分落地 | target/IBL/model/shader 有 cleanup，重复 cleanup 有测试 | 初始化失败不是事务式，交互入口的 try/finally 起点过晚 |
| 回归工具 | 部分落地 | 14 张基线图、离屏渲染、mean absolute diff | 能锁定现状，不能验证 BRDF、纹理、级联和局部回归 |

## 3. 缺口与风险

### P0：索引网格的 tangent 生成不成立

`load_geno_model()` 对带 index buffer 的模型直接调用 raylib `GenMeshTangents()`。实际运行时 raylib 对 Geno 地面和 SOMA 角色输出：

```text
WARNING: MESH: vertexCount expected to be a multiple of 3. Expect uninitialized values.
```

SOMA.bin 为 18,056 vertices / 36,108 triangles，顶点数不是 3 的倍数。更根本的问题是该警告对应的生成路径按连续三个 vertex 处理三角形，而当前 Geno/SOMA 都是 indexed mesh。即使 Geno 的 10,329 个顶点碰巧能被 3 整除，也不等于连续三个顶点就是 index buffer 指定的一个三角形。

同时，SOMA 的 USD UV 是 face-varying，`_vertex_texcoords()` 当前把同一 position 的多个 corner UV 求平均。UV seam 因而被抹掉，base-color/normal/MRA map 会跨 seam 拉伸，切线手性也无法保持。

影响：无 normal map 时几何法线路径可用；一旦传入 `--normal-map`，Geno 结果未经正确性保证，SomaView 结果明确不可靠。`--base-color-map` 和 `--metallic-roughness-map` 在 SOMA 的 seam 附近同样不可靠。

修复要求：

1. 在 SOMA 资产转换阶段按 `(position index, uv index, normal/smoothing group)` 拆点，保留 face-varying UV seam，并同步复制 skin weights。
2. 用 index-aware tangent generator 计算每三角形 tangent/bitangent，按顶点累加，Gram-Schmidt 正交化并计算 handedness `w`。
3. 将 tangent 写入 `.bin`，或在 CPU load 阶段从 indices 稳定生成；不要继续调用当前 `GenMeshTangents()` 处理 indexed mesh。
4. 为 Geno 和 SOMA 各增加 flat normal、方向性 normal、UV checker 三张纹理用例；flat normal 的最终图必须与无 normal map 路径近似一致。

相关代码：`stylized_motion/anim/genoview.py:335-390`、`stylized_motion/anim/soma_assets.py:221-284`、`data/assets/genoview/pbr.fs:63-77`。

### P0：IBL 预过滤和间接光未完整守恒能量

`_prefilter_environment()` 当前生成反射方向后直接 `colors.mean(axis=1)`。它没有排除 `NdotL <= 0` 的样本，也没有按 `NdotL` 累积和归一化。对均匀环境不容易看出问题，对有局部亮源的 HDRI 会把表面背后的辐射混入粗糙反射，造成粗糙金属偏亮、lobe 形状错误。

lighting 端的 `diffuseIBL` 只乘 `(1 - metallic)`，没有乘 `kD = (1 - F) * (1 - metallic)`；IBL Fresnel 也没有 roughness-aware 版本。结果是掠射角的间接漫反射和镜面反射可能重复计能。

修复要求：

1. prefilter 对有效半球执行 `sum(L * NdotL) / sum(NdotL)`；引入真实 HDRI 后再加基于 sample solid angle 的 source mip 选择。
2. lighting 使用 roughness-aware Fresnel，按 `kD * diffuseIBL + specularIBL` 合成。
3. 升级 IBL cache version，防止旧的错误 prefilter 被复用。
4. 增加常量环境、单方向亮源、上下半球异色三类解析/数值测试；常量环境的所有 prefilter mip 必须保持常量。

相关代码：`stylized_motion/anim/environment.py:187-269`、`data/assets/genoview/pbrLighting.fs:235-245`。

### P0：显式纹理槽超出 OpenGL 4.1 的最低可移植保证

当前 lighting/debug/EVSM blur 使用固定槽 13-26。当前 Apple M5 / OpenGL 4.1 实测能够运行，但 fragment shader texture unit 的跨设备最低保证不足以支持槽 26。代码也没有查询运行时上限或在不足时失败。

一个 pass 的峰值采样器数量约为 12，完全可以压缩到 0-11。应建立 per-pass binding layout：lighting 统一显式绑定 GBuffer、AO、shadow、IBL；debug 使用 0-4；单输入 blur 使用 0。初始化时查询上限并验证布局，避免把 GL texture object id 与 texture unit 混为同一命名空间。

相关代码：`stylized_motion/anim/genoview.py:942-990`、`stylized_motion/anim/render_targets.py:163-176`。

### P1：Shadow bias 代码已失联

CSM 为每一级计算 `shadow_biases`，并一路传到 `render_lighting()`，但 shader 调用固定传入 `baseBias = 0.0`，函数内部也不使用 `normal` 或 `baseBias`。`shadow_bias_ptr` 同样只分配、不上传。当前 EVSM 画面可用不代表不需要 receiver-plane/depth/normal offset；动画姿态、远级联和掠射地面仍可能出现 acne 或 peter-panning。

应先做参数扫查，再明确选择：删除整条死 bias 契约并用实测证明 EVSM 无需偏移，或恢复 slope-aware receiver bias / normal offset。级联 debug view 需要显示 cascade id 和 blend band，测试相机平移、旋转、远近裁剪面及快速肢体运动。

相关代码：`stylized_motion/anim/renderer.py:64-86, 255-284, 529-580`、`data/assets/genoview/pbrLighting.fs:118-166`。

### P1：资源初始化不是失败原子操作

`GenoView.run()` 在进入 `try/finally` 前调用 `_initialize_rendering()`。shader、IBL、target 或模型任一步失败都会绕过 `_cleanup()` 和 `CloseWindow()`。`RenderTargets.initialize()` 也会逐项分配资源，但中途失败不会回收已经创建的 attachment；FBO 和 cubemap 的关键检查使用 `assert`，`python -O` 下会消失。

应把窗口初始化后的全部步骤纳入 `try/finally`，让 `initialize()` 在异常时调用幂等 cleanup，再抛出带资源名和尺寸的 `RuntimeError`。所有 `rlFramebufferComplete`、texture id、shader id 和输入纹理 id 都要成为运行时校验。

相关代码：`stylized_motion/anim/genoview.py:1024-1275, 1475-1497`、`stylized_motion/anim/render_targets.py:25-90, 179-213, 294-330`。

### P1：纹理 API 声明了 mipmap，但没有完成采样契约

`load_material_texture()` 只调用 `LoadTexture()` 和 `GenTextureMipmaps()`；没有验证文件/texture id，也没有显式设置 mipmapped minification filter、wrap 或各语义的采样策略。`Material.base_color` 没有验证恰好四个有限分量，texture 的 `uses_*` 判断也只检查 `None`，id=0 仍会被当作有效资源。

应为每类纹理设置明确 sampler：base color 与 MRA 使用 trilinear；normal map 的 mip 需要归一化策略；所有加载失败立即报错。若继续使用当前自定义 MRA 布局，文档和文件名必须明确其通道为 R=metallic/G=roughness/B=AO；若目标是 glTF 兼容，应另设适配层，因为 glTF 的 metallic-roughness 通道约定不同。

相关代码：`stylized_motion/anim/materials.py:17-74`、`stylized_motion/anim/renderer.py:17-44`。

### P1：现有图像回归不能证明渲染正确

`compare_stills` 只用全图 mean absolute difference 判定。角色只占画面较小区域，局部整块错误可能被大面积不变背景稀释到 0.02 阈值以下。pytest 中对 shader 的测试主要是字符串包含检查；material grid 没有提交基线；纹理入口、IBL on/off、级联边界和 normal map 都没有真实 GPU 回归。

应把回归场景改为矩阵：Geno character、SOMA character、material grid、textured Geno、textured SOMA、IBL off、legacy。指标至少包含全图 MAE、前景 mask MAE、P95/P99 和最大差；允许平台容差时可加 SSIM。GPU smoke test 必须作为独立进程运行并检查 shader/FBO 日志。

相关代码：`stylized_motion/anim/compare_stills.py:17-26`、`tests/test_somaview.py:32-105, 294-331`、`tests/test_render_tools.py`。

### P2：Renderer 仍与 GenoView 状态强耦合

`Renderer` 虽已抽出文件，但仍通过 `self.view` 读取大量裸字段，并在文件末尾反向 import `genoview`，形成循环依赖。`Material` 直接持有 raylib `Texture`，也与 spec 中的 backend boundary 目标不一致。这会增加 resize、多 camera、headless backend 和 pass 单测的成本。

后续应引入小型 `RenderContext`、`FrameInputs`、`GpuMaterial`/resource handle，把 viewer 只保留播放、相机和 UI。此项应在 P0/P1 正确性修复后进行，避免重构与像素变化混在同一批提交。

### P2：调试显示和 overlay 的语义需要收口

`debug.fs` 对所有背景像素固定输出白色，没有读取 `whiteBackground`；这与旧 spec 中“debug 背景为黑”的描述以及 final 默认天空不一致。AO debug 只显示 SSAO，不显示 `SSAO * materialAO`。trajectory/skeleton 在 HDR target 上用 LDR 颜色绘制，且 HDR target 没有 depth attachment，因此 overlay 不受场景深度遮挡并再次经过 tone map。

应明确产品语义：debug 背景随 `whiteBackground` 切换黑/白；AO 至少拆成 `ssao`、`material_ao`、`combined_ao`；overlay 若是 UI/x-ray，应在 tone map 后合成，若要被几何遮挡，则给 overlay pass 提供 depth。

## 4. 建议推进顺序

### Milestone A：正确性封板（3-5 个工作日）

目标：让“纹理材质和 IBL 已落地”成为可验证事实。

1. 完成 seam-aware SOMA 拆点与 index-aware tangent；更新资产格式或 loader；加入 flat/directional normal 和 UV checker 测试。
2. 修正 GGX prefilter 的有效半球和 `NdotL` 权重；修正 IBL `kD`/roughness Fresnel；升级 cache key。
3. 将每个 pass 的 texture units 压缩到运行时上限内，统一显式 sampler layout。
4. 清理 shadow bias 死契约，增加 cascade-id/blend debug，并确定 bias 策略。

完成标准：Geno/SOMA 不再出现 tangent warning；flat normal 与无 map 前景 MAE 小于 1/255；常量环境的所有 roughness mip 保持常量；目标 OpenGL 设备无 invalid texture unit；级联扫查无明显断层和 acne。

### Milestone B：工程与回归封板（2-4 个工作日）

1. 将 window/render resource 初始化改成失败可回滚；替换关键 `assert`。
2. 加强 Material/texture 校验与 sampler 设置。
3. 建立 7 场景 GPU golden matrix，采用前景指标和分位数阈值。
4. 在 CI 或专用图形 runner 上运行最小 GPU smoke；无 GPU 环境继续运行 CPU 数值测试和 shader 编译检查。
5. 增加每 pass GPU/CPU timing，记录 1280x720、1024 shadow 的预算。

建议预算：总帧时间 <= 8.33 ms 为 120 fps 目标；shadow、GTAO、lighting、post 分别记录，不只记录总 FPS。当前文档记录的 EVSM 约 120.1 fps 应作为基线重新测量并机器标注。

### Milestone C：资产与环境能力（4-7 个工作日）

1. 支持外部 HDR equirect/cubemap 输入和离线预计算资源；程序化 sky 保留为 deterministic test fixture。
2. 使用精确 sRGB transfer function，统一 authored color、linear texture、HDR 和 display transform。
3. 决定自定义 MRA 与 glTF metallic-roughness 的边界，增加显式 importer/adapter。
4. 增加 per-mesh material 和多材质模型，不再只覆盖 `materials[0]`。

完成标准：同一 HDRI 在 reference renderer 与本项目的 diffuse/specular probe 趋势一致；纹理通道和色彩空间由自动测试覆盖；多材质模型不串材质。

### Milestone D：结构和性能（持续优化）

1. 用 `RenderContext`/`FrameInputs` 解耦 viewer 与 renderer，移除循环 import。
2. 支持 resize 时重建 screen-sized targets；shadow 资源保持独立。
3. 评估 half-resolution GTAO + depth-aware upsample、shadow atlas、prefilter 离线化。
4. 只有在相机/动画抖动成为实际问题时再考虑 temporal AO/TAA；不要在正确性封板前引入时间域状态。

## 5. 验收矩阵

| 场景 | 必查输出 | 主要断言 |
| --- | --- | --- |
| Geno character | final/shadow/AO/HDR | 无 NaN/黑条；shadow 仅影响 direct；输出可复现 |
| SOMA character | final/normal + flat normal map | 无 tangent warning；flat normal 接近几何法线 |
| 5x5 grid | diffuse/specular/IBL/final | metallic=1 无 diffuse；roughness 增大时高光变宽、峰值降低 |
| UV checker | base color/normal | seam 连续且无平均 UV 拉伸；切线手性正确 |
| IBL off | final/IBL | 不生成或绑定 IBL 资源；fallback 可用 |
| constant HDRI | IBL probes | 所有方向和 roughness mip 保持常量 |
| CSM sweep | shadow/cascade id | 平移旋转不闪烁；blend band 连续；远级联覆盖场景 |
| legacy | final | 与冻结基线一致 |

建议把每次像素语义变更分成三步提交：先加入会失败的数值/图像测试，再修实现，最后单独更新并说明基线差异。这样可以区分正确性修复与纯视觉调参。

## 6. 本次验证记录

已执行：

```text
python -m compileall -q stylized_motion/anim                         PASS
conda run -n mcc python -m pytest -q                                 146 passed
focused PBR tests                                                     28 passed
GenoView PBR offscreen, Apple M5 / OpenGL 4.1 / GLSL 4.10             PASS
SomaView PBR offscreen, Apple M5 / OpenGL 4.1 / GLSL 4.10             PASS
Geno/SOMA current final vs committed baseline, threshold 0.001        MAE 0.00000
material grid IBL debug offscreen                                     PASS (manual inspection)
```

限制：本次运行只覆盖当前 Apple M5；没有 Linux/Windows GPU 结果，没有带真实材质纹理的 committed fixture，也没有对动画全片做 CSM/EVSM sweep。因此当前 PASS 证明的是主链路可执行和结果可复现，不等于上述 P0/P1 风险已关闭。

# Stylized Motion Generation

## PBR 渲染管线修订 Spec

状态：Draft 0.1  
日期：2026-08-31  
范围：GenoView/SomaView 的 Minimal Deferred PBR、材质契约、Shadow/SSAO、HDR、Tone Mapping、IBL、调试视图和渲染层拆分

参考设计：[GenoViewPython_Minimal_PBR_技术路线.md](/home/shinn/下载/GenoViewPython_Minimal_PBR_技术路线.md)

执行口径：本 spec 以当前项目已有的 Deferred Renderer 为基础，保留动画、Skinning、相机、资源格式和 viewer 入口。修订分阶段完成；第一阶段只修复职责边界和颜色管线，不一次性实现全部 IBL 或现代实时渲染技术。

当前实施记录（2026-08-31）：Phase 1、Phase 2 已完成，Phase 3 已落地程序化 environment/irradiance 资源、environment/prefilter mip 链、GGX split-sum BRDF LUT 以及独立 IBL 采样路径。Phase 4 仍负责 normal map 的 TBN 应用、AO map 的材质级间接光接入和材质测试场景；legacy 地面网格调制保留在 PBR GBuffer 中作为视觉兼容项。

实施更新（2026-09-03）：Debug View 已落地，Phase 0 的 debug 参数交付项关闭。模式为 final/base_color/metallic/roughness/normal/depth/ao/shadow/diffuse/specular/ibl/hdr；运行时 `V`（`Shift+V` 反向）循环，`--debug-view` 指定初始模式。shadow/diffuse/specular/ibl 由 `pbrLighting.fs` 的 `debugMode` 输出到 HDR target，其余模式由新增 `debug.fs` 显示 pass 直接读取 GBuffer/SSAO/HDR 纹理；调试显示只做 exposure + linear→sRGB，不经过 ACES，且跳过 FXAA。

实施更新（2026-09-03，Phase 4 前半）：normal mapping 与材质 AO 已落地。顶点切线管线（含蒙皮切线变换）此前已就绪；`pbr.fs` 现构建 TBN 并采样 `normalMap.rg`（z 分量由 xy 重建，兼容标准 RGB 法线贴图），退化切线回退几何法线。GBuffer 新增第三个 R8 attachment（`gbufferMaterialAO`），材质 ao（uniform 或 `metallicRoughnessMap.B`）经此进入 lighting，与 SSAO 相乘后仅作用于间接光。viewer CLI 新增 `--normal-map` 与 `--metallic-roughness-map`，应用到角色默认材质。材质测试场景（material grid）仍属 Phase 4 未完成项。

实施更新（2026-09-03，阶段 B 视觉升级）：Shadow 升级为 3×3 PCF（`shadowTexelSize` uniform，spec 8.2 关闭）；Lighting Pass 背景从 RAYWHITE 清屏色改为程序化天空（沿视线采样 environment cubemap，`--disable-ibl` 时回退 skyColor 平色；调试视图背景仍由 debug.fs 按 GBuffer 深度判定为黑）；`tonemap.fs` 支持 `--tone-curve aces|reinhard|agx`（默认 aces 保持历史公式不变）；顺带移除 IBL 分支中未使用的 `environment` 采样死代码（environmentMap 现由天空背景真正消费）。

实施更新（2026-09-03，阶段 C2 IBL 资源真实化）：IBL 链从手工单色 cubemap 升级为从程序化天空梯度（冷天顶/亮地平线/暗地面）出发的完整 split-sum 资源管线：irradiance 由 Fibonacci 球余弦卷积生成（全求积覆盖、对称性 0.0）；prefilter 六级由 GGX importance sampling 生成并按 GL mip 链规范（逐级减半）打包为一张 cubemap，shader 的 `textureLod(prefilterMap, r, roughness*maxLod)` 现在采样的是精确 IS 级别而非 box-filter 近似。修掉两个 CPU 数学 bug（Hammersley 双重反转使 xi≈0、Fibonacci 采样缺 2x 拉伸只覆盖上半球）。prefilter 查表走 8² 代理 cubemap，初始化 CPU 成本从 514s 降到 ~6s。

实施更新（2026-09-03，阶段 A 基线工具）：新增 `stylized_motion.anim.render_stills`（离屏渲染单帧到 PNG，直读 render-target 纹理，绕开 macOS 隐藏窗口 `take_screenshot` 抓黑帧问题；final 视图为 FXAA 前显示目标，其余与交互路径一致）与 `stylized_motion.anim.compare_stills`（阈值化像素对比，超差退出码 1）。`docs/assets/pbr_baseline/` 提交了 14 张基线图（12 个调试视图 + legacy + SomaView final）与再生成/回归说明；Phase 0 的基线截图交付项关闭。

## 1. 目标

本轮修订完成后，GenoView/SomaView 应具备一个结构清晰、可验证、可扩展的 Minimal Deferred PBR 管线：

- Geometry Pass 只输出几何和材质数据，不执行光照；
- Lighting Pass 使用标准 Cook-Torrance BRDF；
- Shadow、SSAO、Direct Light、Indirect Light 职责分离；
- Lighting 输出线性 HDR，Tone Mapping 独立为后处理 Pass；
- Material 支持 uniform 默认材质，并为纹理材质保留稳定契约；
- GenoView 与 SomaView 共用渲染实现，资源目录只承载 rig-specific 资产；
- Python 负责每帧/每对象/每 Pass 调度，像素级计算全部在 GPU Shader 中完成；
- 保留 `legacy` shading，PBR 作为明确的独立模式；
- 增加 Material Test Scene 和自动化检查，使 PBR 结果可以回归。

## 2. 当前基线与问题

当前实现已经具备 Deferred、双 MRT GBuffer、Position Reconstruction、Cook-Torrance、Shadow Map、SSAO、Blur 和 FXAA。相关实现位于：

- `stylized_motion/anim/genoview.py`
- `data/assets/genoview/pbr.fs`
- `data/assets/genoview/pbrLighting.fs`
- `data/assets/genoview/ssao.fs`

本轮需要修复的主要问题：

| 问题 | 当前行为 | 修订要求 |
| --- | --- | --- |
| Shadow/SSAO 耦合 | `ssao.fs` 同时计算 AO 和 Shadow，并把两者写入同一纹理 | Shadow 在 Lighting Pass 独立采样；SSAO 只输出 AO |
| Shadow Blur 耦合 | AO 的 bilateral blur 同时作用于 shadow 通道 | Shadow 使用独立 PCF 策略，不经过 SSAO blur |
| HDR 缺失 | Lighting shader 内部已经 ACES/Gamma，`lighted` 不是明确的 HDR 中间目标 | 增加显式 `RGBA16F` HDR target，Lighting 只输出线性 HDR |
| Tone Mapping 位置错误 | `pbrLighting.fs` 同时承担 BRDF 和 Tone Mapping | 新增独立 `tonemap.fs` |
| Ambient 非物理 | 使用 `skyColor`、`groundStrength`、`ambientStrength` 人工近似环境光 | 第一阶段保留兼容参数，后续替换为 Diffuse/Specular IBL |
| 材质粒度不足 | metallic/roughness 是 viewer 级 uniform，无 Material/Texture 契约 | 引入 Material、RenderObject、默认值和纹理通道约定 |
| Debug 不完整 | 只能查看 albedo、normal、depth、ssao、lighting | 增加 metallic、roughness、shadow、direct、specular、ibl、hdr |
| Python 每帧分配 | 大量 `ffi.new("float*")` 和临时参数 | 初始化阶段创建可复用参数 buffer |
| 渲染逻辑集中 | `GenoView.run()` 包含所有 Pass 和资源绑定 | 先建立 Pass 边界，再逐步抽出 Renderer/Scene/Material 模块 |

## 3. 非目标

本轮明确不实现：

- Clustered Lighting、Forward+、GPU Driven Rendering、Mesh Shader；
- CSM、VSM、EVSM；
- SSR、Ray Tracing、GI；
- TAA、Temporal Upscaling；
- SSS、Transmission、Clear Coat、Sheen、Area Light；
- 重写动画、BVH、Skinning 或 Geno/SOMA 资产格式；
- 为历史 checkpoint 或历史命令增加额外兼容层。

## 4. 目标渲染架构

每帧的规范顺序固定为：

```text
Animation / Playback Update
        ↓
Skinning Pose Update
        ↓
Shadow Pass
        ↓
GBuffer Pass
        ↓
SSAO Pass
        ↓
SSAO Blur Pass
        ↓
PBR Lighting Pass → HDR RGBA16F
        ↓
Tone Mapping Pass
        ↓
FXAA Pass
        ↓
Screen
```

职责边界：

| Pass | 输入 | 输出 | 禁止承担 |
| --- | --- | --- | --- |
| Shadow | skinned/static geometry、light camera | depth texture | AO、BRDF、tone mapping |
| GBuffer | geometry、Material | albedo/metallic、normal/roughness、depth | direct/indirect lighting |
| SSAO | GBuffer、camera | AO texture | Shadow Map 采样、颜色合成 |
| SSAO Blur | AO、GBuffer depth/normal | filtered AO | Shadow filtering |
| PBR Lighting | GBuffer、AO、Shadow、lights、environment | linear HDR；背景像素输出程序化天空（沿视线采样 environment cubemap，`--disable-ibl` 时回退 skyColor 平色） | sRGB 输出、tone mapping |
| Tone Mapping | HDR texture、exposure | display-linear/sRGB output | BRDF、shadow、FXAA |
| FXAA | tone-mapped output | final screen image | HDR 运算 |

## 5. GBuffer 与 RenderTarget 契约

### 5.1 GBuffer

第一阶段保持当前的轻量布局：

```text
GBuffer0: RGBA8
  RGB = Base Color
  A   = Metallic

GBuffer1: RGBA16F
  RGB = encoded world normal = normal * 0.5 + 0.5
  A   = Roughness

GBuffer2: R8（Phase 4 修订新增）
  R   = material AO，baked 材质级遮蔽项

Depth: depth texture
  R   = linear depth，沿用当前 Position Reconstruction 契约
```

约束：

- 不增加 Position、Velocity、Material ID、Emission attachment；
- SSAO 结果不写入 GBuffer；材质 AO 使用独立的 R8 attachment，语义见 20.5 决策记录；
- metallic、roughness 必须 clamp 到 `[0, 1]`，roughness 最小值暂定 `0.04`；
- normal 在写入前必须 normalize，在 Lighting 解码后再次 normalize；
- GBuffer 的 attachment 数量和 draw buffer 状态必须在 `begin_gbuffer/end_gbuffer` 中成对恢复。

### 5.2 颜色空间

统一采用：

```text
Base Color / vertex color / base-color texture: sRGB authoring space
Metallic / Roughness / AO / Normal: linear data space
Lighting intermediate: linear HDR
Tone Mapping output: display sRGB
```

第一阶段沿用当前 uniform/vertex color 路径：GBuffer shader 将 Base Color 转换到 linear 后写入 GBuffer。引入纹理后必须显式区分：

- Base Color texture 使用 sRGB 采样或显式 sRGB→linear；
- Metallic/Roughness/AO texture 禁止 sRGB 解码；
- Normal texture 禁止 sRGB 解码；
- 不得在 Lighting Pass 对已转换的 albedo 再次执行 sRGB→linear。

### 5.3 HDR Target

新增独立 HDR render target：

```text
HDR Target: RGBA16F
RGB = linear scene radiance after direct + indirect lighting
A   = 1.0
```

要求：

- 不使用默认 `LoadRenderTexture` 作为 HDR 规范实现，必须显式创建浮点颜色附件；
- HDR target 的尺寸与 GBuffer 一致；
- Lighting shader 不调用 ACES、Reinhard 或 LinearToSRGB；
- HDR target 只由 Tone Mapping Pass 消费，不直接显示到屏幕。

## 6. Material 契约

引入最小 Material 数据模型：

```python
@dataclass
class Material:
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.58
    ao: float = 1.0
    base_color_map: Texture | None = None
    normal_map: Texture | None = None
    metallic_roughness_map: Texture | None = None
```

纹理约定：

```text
base_color_map: RGBA, sRGB
metallic_roughness_map.R = Metallic
metallic_roughness_map.G = Roughness
metallic_roughness_map.B = AO
normal_map: tangent-space normal, linear
```

默认值：

- 无 base color map 时使用 `base_color`；
- 无 metallic/roughness map 时使用 uniform 值；
- 无 AO map 时使用 `ao = 1.0`；
- 无 normal map 时使用几何 normal；
- alpha 暂不参与透明渲染，第一阶段统一按 opaque 处理。

viewer CLI 的 `--metallic` 和 `--roughness` 继续保留，作为默认 Material 的覆盖值，而不是未来 Material API 的替代品。

## 7. Shader 契约

目标 Shader 职责如下：

```text
pbr_gbuffer.vs / skinned_pbr_gbuffer.vs
  vertex transform、skinning、world position、normal、UV、future tangent

pbr_gbuffer.fs
  material resolve、sRGB/linear、TBN 构建与 normal map 采样、GBuffer packing、material AO 输出、linear depth

shadow.vs / skinnedShadow.vs
  light-space transform、shadow skinning

shadow.fs
  depth-only output

ssao.fs
  只计算 AO，输出单通道语义；不得读取 shadowMap

blur.fs
  只 blur AO

pbrLighting.fs
  position reconstruction、GGX、Smith、Schlick、direct light、shadow、SSAO×material AO、IBL
  输出 linear HDR；debugMode 输出 shadow/diffuse/specular/ibl 调试量

tonemap.fs
  exposure、tone mapping、linear→sRGB

fxaa.fs
  FXAA

debug.fs
  仅调试显示：读取 GBuffer/SSAO/HDR 纹理与 lighting debugMode 输出，
  exposure + linear→sRGB 显示，不参与生产画面路径
```

PBR Lighting 必须遵守：

```text
f0 = mix(vec3(0.04), baseColor, metallic)
kS = F
kD = (1 - kS) * (1 - metallic)
direct = (diffuse + specular) * radiance * NdotL * shadow
indirectDiffuse = irradiance * baseColor / PI * kD * ao
```

其中 AO 只能影响间接光；Shadow 只能影响受光方向上的 direct light。二者不可复用同一个纹理通道作为长期契约。

## 8. Shadow 规范

### 8.1 第一阶段

- 保留单方向光和现有正交 light camera；
- Shadow Pass 仍同时支持 ground、static model 和 skinned model；
- PBR Lighting 新增 `shadowMap`、`lightViewProj`、shadow clip/bias uniform；
- Shadow factor 在 Lighting Pass 中计算；
- 默认先使用单次比较采样，保证职责拆分后结果稳定。

### 8.2 第二阶段 PCF（已落地 2026-09-03）

`pbrLighting.fs` 的 `ShadowFactor` 使用 3×3 PCF：

- 使用 `shadowTexelSize` uniform（1/shadowResolution，viewer 初始化后缓存）；
- bias 保持 normal offset（世界空间 `0.01 * normal`）+ 逐样本常数 bias（5e-5 线性深度）；
- shadow sampling 不经过 SSAO blur；
- 超出 shadow projection 范围时返回 `shadow = 1.0`；
- `ssao.fs` 不计算或输出 shadow。

## 9. SSAO 规范

- SSAO 只输出 AO，推荐单通道 R8 或现有 RGBA target 的 `.r` 通道；
- AO 范围、bias、sample count 先保持当前视觉基线；
- AO bilateral blur 只读取 GBuffer normal/depth 和 AO 输入；
- Lighting 中使用 `indirect *= ao`；
- 不对 direct light 乘 AO；
- 不把 sky/ground/IBL 分量拆成不同的 AO 语义；
- 增加 AO 强度 uniform，默认值保持当前效果附近。

## 10. PBR Lighting 规范

第一阶段只实现 Directional Light：

```python
@dataclass
class DirectionalLight:
    direction: Vector3
    color: Vector3
    intensity: float
```

Lighting Pass 必须完成：

1. 从 depth 和 `camInvViewProj` 重建 world position；
2. 解码 normal、metallic、roughness；
3. 计算 `V/L/H` 和 `Ndot*`；
4. 使用 GGX NDF、Smith Geometry、Schlick Fresnel；
5. 计算 direct diffuse/specular；
6. 应用独立 shadow factor；
7. 计算间接光，并应用 AO；
8. 将结果写入线性 RGBA16F HDR target。

必须保留 roughness 的数值保护：

```glsl
roughness = clamp(roughness, 0.04, 1.0);
NdotV = max(NdotV, 1e-4);
NdotL = max(NdotL, 0.0);
```

## 11. Ambient 到 IBL 的迁移

当前 `skyColor`、`skyStrength`、`groundStrength`、`ambientStrength` 先作为兼容的 fallback，不再扩展新的人工 ambient 参数。

### IBL-1：Diffuse IBL

- 使用程序化 sky 或固定 HDR environment 生成 environment cubemap；
- 生成 irradiance map；
- Lighting 增加 `irradianceMap`；
- 用 diffuse IBL 替换现有 sky/ground diffuse ambient；
- 保留一个明确的 fallback 开关，便于 A/B 对比。

### IBL-2：Specular IBL

- 生成 prefiltered environment cubemap；
- 使用 roughness 选择 mip level；
- 增加环境反射方向和镜面 IBL；
- 继续使用 direct light + diffuse IBL + specular IBL 的组合。

### IBL-3：BRDF LUT

- 离线或初始化阶段生成 BRDF LUT；
- 使用 `NdotV`、roughness 查询 LUT；
- 将 specular IBL 的 Fresnel/visibility 近似纳入 LUT；
- IBL 资源生命周期由 Environment/Renderer 管理，不由单个模型管理。

## 12. Tone Mapping 规范

新增 `tonemap.fs`，输入为 HDR 线性颜色：

```text
hdr
  → exposure
  → tone mapping
  → clamp
  → linear to sRGB
```

第一阶段使用当前 ACES fitted curve 作为兼容实现，并在代码中命名为 `ACESApprox`，避免误称为完整 ACES 色彩管理。Tone Mapping 必须与 PBR Lighting 分离，以便后续替换 Reinhard、ACES 或 AgX。

Tone curve 已可选化（2026-09-03）：`tonemap.fs` 的 `toneCurve` uniform 支持 `aces`（默认，保持历史拟合公式逐位不变）/ `reinhard` / `agx`（MinifiedAgX 拟合），viewer CLI 通过 `--tone-curve` 选择。FXAA 必须放在 Tone Mapping 之后，因为 FXAA 依赖最终显示空间的边缘亮度。

## 13. Python 渲染层

### 13.1 目标模块

保持 `genoview.py` 和 `somaview.py` 作为命令入口，逐步抽出：

```text
stylized_motion/anim/
  renderer.py          # Renderer、Pass 调度、生命周期
  render_targets.py    # GBuffer、ShadowMap、AO、HDR target
  materials.py         # Material、texture binding、默认材质
  scene.py             # Scene、RenderObject、DirectionalLight、Environment
  genoview.py          # Geno 数据/播放适配和 viewer UI
  somaview.py          # SOMA rig/资源适配
```

第一阶段允许暂时保留部分实现于 `genoview.py`，但必须先形成同名职责边界，不能继续增加新的裸 texture/uniform 操作散落在主循环中。

### 13.2 Renderer API

目标接口：

```python
class Renderer:
    def initialize(self, width: int, height: int) -> None: ...
    def render_shadow(self, scene: Scene) -> None: ...
    def render_gbuffer(self, scene: Scene, camera: Camera) -> None: ...
    def render_ssao(self, camera: Camera) -> None: ...
    def render_lighting(self, scene: Scene, camera: Camera) -> None: ...
    def render_tonemap(self, exposure: float) -> None: ...
    def render_fxaa(self) -> None: ...
    def cleanup(self) -> None: ...
```

主循环只负责：播放更新、场景更新、调用 Renderer、绘制 UI。

### 13.3 参数与 CFFI

初始化时分配并保存：

- near/far float pointers；
- exposure/light intensity pointers；
- light color/direction vectors；
- inverse texture resolution vectors；
- shadow texture slot pointer。

每帧只更新 buffer 内容，不重复 `ffi.new()`。所有 uniform location 在 shader 初始化后缓存。

### 13.4 Backend 边界

第一阶段继续使用 raylib/RLGL/OpenGL，但 Renderer 不应在业务代码中直接依赖每个 RLGL 调用。至少通过 RenderTarget 和 Pass helper 隔离：

```text
GenoView/SomaView
      ↓
Renderer / Passes
      ↓
Raylib/RLGL backend
```

暂不实现 Vulkan 或多后端，但不能让材质和场景对象绑定 raylib 的裸结构体。

## 14. Material Test Scene

> 2026-09-03 已落地：`--scene grid`（GenoView/SomaView/render_stills 均支持）。5×5 球体网格，行=metallic {0,.25,.5,.75,1}（向 -Z），列=roughness {0,.25,.5,.75,1}（向 +X），共享 UV 球 mesh、每格独立 Material（中性 albedo、微弱金属梯度着色以便辨认）。grid 模式下角色隐藏、相机与 shadow 光目标固定于网格中心。验收（离屏渲染 + 调试视图）：metallic=1 行 diffuse 调试视图为黑（金属无漫反射）；非金属行 diffuse 随 roughness 单调；直接光高光随 roughness 增大变宽变弱；投影正确落在球与地面。五种标准 probe 材质定义见 `scene.MATERIAL_GRID_PROBES`（供后续 per-object 材质实验引用）。

增加开发专用测试场景，至少包含 5×5 material grid：

```text
rows    = metallic:  0.0, 0.25, 0.5, 0.75, 1.0
columns = roughness: 0.0, 0.25, 0.5, 0.75, 1.0
```

必须覆盖：

- white diffuse；
- plastic；
- rough plastic；
- gold-like metal；
- chrome-like metal。

测试场景用于验证 Fresnel、energy conservation、roughness 单调性、metallic 行为、shadow 和 IBL，不属于生产 viewer 默认场景。

## 15. 实施阶段

### Phase 0：基线与诊断

> 2026-09-03 进度：已完成关闭。debug view 全套 + 基线截图（`docs/assets/pbr_baseline/`，14 张，由 `render_stills` 生成、`compare_stills` 可回归对比）均已交付。

- 固定当前 PBR 输出截图/参数作为 baseline；
- 增加 metallic、roughness、shadow debug；
- 修正 README 与代码默认 shading 不一致的问题；
- 为 GenoView 和 SomaView 验证相同 shader contract。

交付物：基线截图、debug 参数、无资源泄漏的 viewer smoke test。

### Phase 1：职责拆分与 HDR

- 从 `ssao.fs` 移除 Shadow 计算；
- 在 `pbrLighting.fs` 独立采样 Shadow Map；
- SSAO 只输出 AO，blur 只处理 AO；
- 创建 RGBA16F HDR target；
- 新增 `tonemap.fs`；
- 将 FXAA 移到 Tone Mapping 之后；
- 保持最终画面尽量接近当前基线。

这是本 spec 的最高优先级阶段。

### Phase 2：Material 与结构化 Pass

- 引入 Material、RenderObject、Scene、DirectionalLight；
- 将 uniform 默认材质接入 Material；
- 缓存 CFFI buffers 和 uniform locations；
- 抽出 RenderTarget/Pass helper；
- 保留 Geno/SOMA 现有动画和模型加载接口。

### Phase 3：Diffuse/Specular IBL

- 先实现程序化 sky 或固定 HDR environment；
- 加入 irradiance map；
- 再加入 prefilter map；
- 最后加入 BRDF LUT；
- 每一步都保留 debug view 和 fallback 对照。

### Phase 4：Normal Mapping 与测试场景

> 2026-09-03 进度：已全部完成关闭。tangent 管线、TBN/normal map 采样、材质 AO 间接光接入、`--scene grid` 材质测试场景（5×5 grid + 验收）均已交付；`--normal-map` / `--metallic-roughness-map` 提供纹理验证入口。

- 扩展模型/mesh 数据以提供可靠 tangent；
- 在 GBuffer 中构造 TBN 并采样 normal map；
- 完成 material grid；
- 加入 normal map、纹理颜色空间和 mipmap 验证。

## 16. 验收标准

### 17.1 功能验收

- `--shading legacy` 仍可启动并渲染；
- `--shading pbr` 完成 Shadow → GBuffer → SSAO → HDR Lighting → Tone Mapping → FXAA；
- GenoView 和 SomaView 均可使用同一套 PBR Pass；
- compare mode、trajectory overlay、播放和相机交互不回归；
- 无模型区域不产生 NaN、黑色条带或深度污染。

### 17.2 渲染正确性验收

- metallic=0 的材质具有 diffuse 响应和约 0.04 的非金属 F0；
- metallic=1 时 diffuse 项消失；
- roughness 增大时高光扩散且峰值降低；
- Shadow 只影响 direct light；
- AO 只影响 indirect light；
- 曝光改变不会改变 GBuffer 或 Lighting 的材质语义；
- Tone Mapping 可替换而不需要修改 BRDF shader。

### 17.3 工程验收

- Shader uniform location 不在每帧查询；
- 新增渲染参数不引入每帧 `ffi.new()`；
- GBuffer/HDR/Shadow/SSAO 资源均有成对 cleanup；
- `python -m compileall -q stylized_motion` 通过；
- 现有测试集通过；
- 至少存在一条 viewer smoke test 和一组 material/debug 截图。

## 17. 测试计划

静态/单元测试：

- 检查 PBR shader 文件存在且关键 uniform/输出契约完整；
- 检查 Geno/SOMA resource directory 均包含相同 shader 集合；
- 检查 Material 默认值和 clamp 规则；
- 检查 Renderer cleanup 覆盖所有 RenderTarget。

运行时 smoke test：

```bash
python -m pytest -q
python -m compileall -q stylized_motion
python -m stylized_motion.run --mode visualize --pipeline genoview --bvh <small_bvh> --shading pbr
python -m stylized_motion.run --mode visualize --pipeline somaview --bvh <small_bvh> --shading pbr
```

若 CI 没有图形环境，至少执行 shader/source contract 检查；有图形环境时补充 Xvfb 或等价的短时窗口 smoke test。

## 18. 迁移与回滚

- Phase 1 前保留旧 shader 文件副本或通过 git 提交点回滚；
- 旧 `legacy` 路径不依赖新的 Material/IBL 资源；
- PBR 新增资源缺失时，允许回退到默认 uniform material 和人工 ambient fallback；
- 不修改 Geno.bin、SOMA.bin、BVH 和 motion database 格式；
- 不删除现有 debug view 名称，只扩展选项；
- 每个阶段都应可独立运行，不能要求 IBL 资源存在才能启动基础 PBR。

## 19. 风险与决策记录

### 20.1 RGBA8 albedo 精度

当前 GBuffer0 使用 RGBA8。第一阶段保持该格式以降低改动面；引入高对比度 HDR 材质或纹理后，应评估使用 sRGB attachment、RGB10A2 或更高精度 albedo target 的成本。

### 20.2 Shadow 与 SSAO 资源尺寸

第一阶段保持全分辨率 AO 和当前 shadow resolution。只有在正确性稳定后才调整 half-resolution AO、采样数量或 shadow atlas。

### 20.3 ACES 命名

当前实现是常见的 ACES fitted approximation，不是完整色彩管理系统。新 shader 使用 `ACESApprox` 命名，避免把 tone curve 与色彩管理混为一谈。

### 20.4 当前默认值

代码中的 `shading` 默认值目前是 `pbr`，README 仍描述为 legacy。Phase 0 必须统一文档、CLI help 和实际默认值；本 spec 选择保留代码现状，将 PBR 作为默认 viewer 模式，legacy 作为显式回退模式。

### 20.5 材质 AO attachment（2026-09-03）

Phase 4 落地材质 AO 时，GBuffer0/1 的既有布局没有空闲通道。评估了两个方案：albedo 预乘（零布局变更，但材质 AO 会错误地压暗 direct light，违反"AO 只影响间接光"验收项）与新增第三个 R8 attachment（布局最小增量，语义与 glTF occlusion texture 一致）。选择后者：GBuffer2 为 R8，仅存材质 AO；SSAO 结果仍不进入 GBuffer；lighting 中 `ao = ssao * materialAO`，仅作用于间接光。5.1 中原"不把 AO 挤入 GBuffer"约束按此口径修订为 SSAO/Material AO 语义分离。

### 20.6 纹理槽绑定（2026-09-03）

raylib 的 `SetShaderValueTexture` 以 `texture.id` 作为采样器 unit 键值；当渲染目标的 GL 纹理名恰好与手工管理的槽位（shadow map=10、environment/irradiance/prefilter=11-13、BRDF LUT=14）重合时发生冲突。新增 material AO attachment 使后续纹理 id 整体移位后，debug 显示 pass 的 `texLighted`（id=12 撞 irradiance 槽 12）实际采样到错误纹理，表现为 shadow/diffuse/ibl 调试视图内容异常。处理：多纹理 pass 一律改用显式槽绑定（新增 `render_targets.set_shader_value_texture_slot`），材质 AO=15、debug 五张输入=16-20、lighting SSAO=21（加固，防止未来再增纹理时 id 移位破坏 AO）。lighting 其余 sampler（gbufferColor/Normal/Depth）经验证输出逐像素不变，维持原绑定。此后新增 attachment 或纹理只需避开 10-21 已占槽位。

### 20.7 PBR 灯光 rig 默认值（2026-09-03）

从 legacy 迁移时沿用的灯光参数（sunStrength=0.25、ambientStrength=1.0、灰白 sun/sky 等）在 Cook-Torrance + IBL 路径下形成环境光主导的平淡画面：角色背光/受光线性比 ≈0.43，远高于真实室外晴天（约 0.15-0.25），且缺少暖阳光/冷天光的色温对比。处理：viewer 按 shading 模式选择灯光 rig（`LEGACY_LIGHT_RIG` 完整冻结 GenoViewPython 数值；`PBR_LIGHT_RIG` 重调）：sun 0.55、暖阳光 `(255,240,214)`、光向 `(0.45,-0.8,-0.35)`（仰角约 55°，拉长投影）、sky fallback `(150,180,220)`、fallback 权重 ambient/sky/ground = 0.15/0.35/0.25、地面反照率 `(215,215,215)`（浅灰白整体色调，同日微调；渲染后约 0.72 显示亮度，低于背景 0.89 保持层次）；IBL environment/irradiance/prefilter 色盘重调为"冷天顶、暗地面"结构。`--sun-strength` 作为直接光强的 CLI 覆盖（缺省取当前模式的 rig 值）。设计落点：白反照率上直接:间接 ≈3.5:1，角色背光/受光线性比 ≈0.2，曝光 0.9 下受光白不削波（ACES 后 ≈0.81 sRGB）。

### 20.8 白背景哨兵（2026-09-03）

论文图像可视化需要天空背景为精确纯白（255,255,255），且不能影响角色 PBR 光照（往环境贴图塞白天气会污染 IBL）。方案：`--white-background`（或 `white_background=True`）下，pbrLighting.fs 的背景分支输出哨兵辐射度 `BACKGROUND_SENTINEL=6e4`，tonemap.fs 检测 `hdr.r > 3e4` 时绕过 exposure 与 tone curve 直接输出显示白，因此任意 tone curve（aces/reinhard/agx）下背景都是精确 255；debug 显示 pass 的背景分支同步改为白底。两个实现约束记录在案：(1) lighting 全屏 quad 实际运行在 `(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)` 混合下——`rlDisableColorBlend` 在 raylib 5.5 的批次刷新时不生效——因此背景标记必须带 alpha=1 完整替换 clear 色，不能依赖 alpha 通道语义；(2) RGBA16F 存储为半精度（上限 65504），哨兵不能超出该范围，6e4 与场景辐射度（≤ ~1.3）相距四个数量级，误检风险可忽略。关闭该开关时逐像素与原输出一致（已验证）。

### 20.10 锚定反演校准（2026-09-03）

前两轮天空/光照调参失败的复盘结论：在无量化目标的情况下凭渲染结果试错、且同时改动存储/色盘/能量比多个变量，效果不佳时无法归因。改用锚定反演方法：先从真实室外参考确定 5 个显示锚点（天空、地板亮部/阴影、角色亮部/阴影的 sRGB 值），对每个锚点反演 `exposure → ACES → sRGB` 得到线性辐射度目标，再从 shader 能量模型（direct = albedo·sunLin·sun·nDotL，indirect = albedo·irradiance·ibl）反解 rig 数值。反解过程暴露的硬约束：上法向的余弦叶重权高仰角天区，天顶蓝（B>1.6）会把地板辐照染到无法用标量 iblStrength 映射回中性阴影。最终色盘形态：中性白蓝基底（zenith `(0.40,0.55,0.72)` → mid `(0.55,0.66,0.80)`）保证辐照 B/R≈1.7，饱和蓝只以地平线高斯环带呈现（ring `(0.50,0.90,1.60)`，σ 上 0.12/下 0.10，同时覆盖有限地板造成的 -4.3°..0° 可见带），地面半区暗中性。解算结果：sun 1.88、ibl 0.45、直接:间接 ≈5:1（影子作为形体语言变实）。配套工具：`render_stills` 导出翻转到窗口方向（修复方向误读），`sample_anchors` 用颜色掩码（角色 body mask 按亮度分位拆 lit/shadow、蓝灰 blob 取投影核心）+ 固定矩形（天空/地板亮部）自动比对 5 锚点，容差 8。已知边界：mid-sky 本身不是照片级蓝天（锚点取实际达成值 (169,177,187)）——要 18-22° 仰角也呈蓝天需 B≥1.5 的 mid，会把辐照 B 拉到地板锚点无法兼容；解锁该形态需要太阳盘提供暖色主导（后续步骤）。`--white-background` 不受影响。

## 20. 完成定义

当 Phase 1 完成并满足以下条件时，称为“PBR 管线结构修订完成”：

```text
Shadow 独立于 SSAO
SSAO 只影响间接光
Lighting 输出 RGBA16F 线性 HDR
Tone Mapping 为独立 Pass
FXAA 位于 Tone Mapping 之后
GenoView/SomaView 共用契约
```

Phase 2 至 Phase 4 是在该稳定基线上的能力扩展，不应反过来阻塞基础 PBR 的验收。

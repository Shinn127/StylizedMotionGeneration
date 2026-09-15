# Quinn mesh 绑定至 SOMA 骨骼：转换执行方案

## 目标

将 `ue5_female_mannequin_-_quinn_-_web_ready.glb` 中的 Quinn 外观网格转换为由 SOMA 的 78 个骨骼驱动的可视化资源。最终资源须能被 SomaView 直接加载，并播放现有 SOMA BVH、数据库或模型生成的动作。

转换首版采用离线 Python 脚本，不依赖 Blender。Blender 仅作为后续需要交互式检查、局部改权重或调姿势时的辅助工具。

## 已确认的输入约束

Quinn GLB 位于：

`/Users/shinn/Downloads/ue5_female_mannequin_-_quinn_-_web_ready.glb`

其内容为一个 skinned mesh、45,979 个顶点、87,280 个三角形、80 个 skin joints、每顶点最多四个骨骼影响、两张内嵌 JPEG 纹理和一段单帧 T-pose 动画。顶点权重索引指向 glTF skin 的 joint 列表，而非 glTF node 编号。

项目中的 SOMA 可视化资源位于 `data/assets/somaview/`。其中：

- `SOMA.bin` 保存 skinned mesh、78 个骨骼、真实 mesh bind pose。
- `SOMA_bind.bvh` 定义播放 SOMA 动作所需的层级和通道。
- SomaView 可通过 `--resources-root` 加载独立资源目录。

`SOMA.bin` 的格式要求一个 mesh、每顶点最多四个影响、`uint8` 骨骼编号及 `uint16` 三角索引。转换后顶点数必须低于 65,536；UV seam 如需拆点时应重新检查此上限。

## 输出目录和文件

首版输出到独立目录，例如：

```text
data/assets/somaview_quinn/
├── SOMA.bin
├── SOMA_bind.bvh
├── quinn_base_color.jpg
├── quinn_normal.jpg
├── conversion_report.json
└── <SomaView shader files>
```

不得覆盖 `data/assets/somaview/` 中的现有资源。`SOMA_bind.bvh` 与 shader 文件从现有 SOMA 资源复制；`SOMA.bin` 由 Quinn 网格、SOMA 骼骨架和 SOMA bind pose 组成。

## 实现模块

新增 `stylized_motion/anim/quinn_to_soma_assets.py`，并将它注册为 `preprocess` 子命令。模块应提供以下功能：

1. `load_glb(path)`：读取 GLB JSON、binary chunk、accessor 与 bufferView。
2. `load_quinn_mesh(path)`：提取 positions、normals、UV、indices、`JOINTS_0`、`WEIGHTS_0`、skin joints、inverse bind matrices 和纹理。
3. `load_soma_bind(resource_dir)`：读取现有 `SOMA.bin` 中的 78 个 bone names、parents、global bind positions 和 rotations。
4. `map_quinn_weights(...)`：根据固定映射表合并 Quinn 权重并映射到 SOMA bone indices。
5. `align_mesh_to_soma_bind(...)`：应用明确记录的坐标、比例和可选的局部身体段落对齐。
6. `write_soma_bin(...)`：复用 `stylized_motion.anim.soma_assets.write_soma_bin` 的二进制写入契约。
7. `extract_textures(...)`：从 GLB buffer views 写出两张纹理，并写入报告。
8. `validate_conversion(...)`：执行所有文件级和数值级检查。

脚本 CLI 建议为：

```bash
python -m stylized_motion.anim.quinn_to_soma_assets \
  --glb /Users/shinn/Downloads/ue5_female_mannequin_-_quinn_-_web_ready.glb \
  --soma-resources data/assets/somaview \
  --output-dir data/assets/somaview_quinn
```

## 数据读取与坐标约定

### GLB 读取

实现须支持 accessor 的 `byteOffset`、bufferView 的 `byteOffset` 和可选 `byteStride`。读取结果使用显式 little-endian dtype，并根据 accessor 的 component type 与 shape 还原数组。

读取后立即检查：

- primitive 为三角形；
- `POSITION`、`NORMAL`、`TEXCOORD_0`、`JOINTS_0`、`WEIGHTS_0` 数量一致；
- index 在顶点范围内；
- joint index 在 skin joint 范围内；
- 所有 weight 有限且非负；
- 每顶点的原始权重和接近 1；
- 只有一个目标 mesh primitive，或在遇到多个 primitive 时显式失败并报告。

### Bind pose

不要使用 SOMA BVH 第 0 帧作为 mesh 的真实 bind pose。项目的 `soma_assets.py` 已说明：SOMA mesh 的静止姿势由 USD 的 `bindTransforms` 决定，和 BVH 第 0 帧不完全相同。

输出 `SOMA.bin` 时，bone names、parents、bind positions 和 bind rotations 必须直接来自现有 `data/assets/somaview/SOMA.bin`。SomaView 运行时增加的 `Simulation` root 不写入 mesh bone list。

### 坐标和比例

Quinn GLB 存在 Sketchfab 导出的上层坐标变换。脚本必须将该变换只应用一次，并将最终 mesh 写入 SomaView 使用的米制坐标。

第一版应输出下列对齐统计：

- 对齐前后 mesh AABB 与身高；
- Hips、Chest、左右肩、左右肘、左右腕、左右髋、左右膝、左右踝的目标关节位置；
- 网格顶点的整体 affine 变换；
- 每个可选局部变换的身体区域、参数及最大位移。

首轮只使用全局刚体变换和统一比例。若动作检查显示关节中心明显偏离，再引入以骨盆、躯干、左右臂、左右腿为单位的局部对齐；所有局部对齐必须可重复、可记录。

## 骨骼与权重映射

映射应写成显式的、可测试的 Python 常量，而不使用模糊名称匹配。每条规则按 Quinn skin joint 名称映射至 SOMA bone 名称；转换时再解析为 SOMA bone index。

### 主体映射

| Quinn joint | SOMA bone |
| --- | --- |
| `pelvis` | `Hips` |
| `spine_01` | 初始映射 `Spine1`，见脊柱规则 |
| `spine_02` | `Spine1` |
| `spine_03` | `Spine2` |
| `spine_04` | `Chest` |
| `spine_05` | `Chest`，当前文件权重为零 |
| `neck_01` | `Neck1` |
| `neck_02` | `Neck2` |
| `head` | `Head` |
| `clavicle_l/r` | `LeftShoulder` / `RightShoulder` |
| `upperarm_l/r` | `LeftArm` / `RightArm` |
| `lowerarm_l/r` | `LeftForeArm` / `RightForeArm` |
| `hand_l/r` | `LeftHand` / `RightHand` |
| `thigh_l/r` | `LeftLeg` / `RightLeg` |
| `calf_l/r` | `LeftShin` / `RightShin` |
| `foot_l/r` | `LeftFoot` / `RightFoot` |
| `ball_l/r` | `LeftToeBase` / `RightToeBase` |

### Twist bone 合并

Quinn 的主要四肢主骨骼本身几乎没有顶点权重；权重大多由 twist 子骨骼承担。当前 GLB 的 16 个 twist bones 没有独立动画通道。在 Quinn 的 bind pose 与文件中的 T-pose 下，将 twist 权重合并给父骨骼的最大顶点差约为 `1.4e-7`，可视为浮点误差。

因此首版规则为：

| Quinn joints | SOMA bone |
| --- | --- |
| `upperarm_twist_01_l`、`upperarm_twist_02_l` | `LeftArm` |
| `upperarm_twist_01_r`、`upperarm_twist_02_r` | `RightArm` |
| `lowerarm_twist_01_l`、`lowerarm_twist_02_l` | `LeftForeArm` |
| `lowerarm_twist_01_r`、`lowerarm_twist_02_r` | `RightForeArm` |
| `thigh_twist_01_l`、`thigh_twist_02_l` | `LeftLeg` |
| `thigh_twist_01_r`、`thigh_twist_02_r` | `RightLeg` |
| `calf_twist_01_l`、`calf_twist_02_l` | `LeftShin` |
| `calf_twist_01_r`、`calf_twist_02_r` | `RightShin` |

前臂 twist 权重不应合并到手掌，避免手腕旋转错误带动前臂。

这条规则保留不了独立 twist bone 能产生的扭转分布；若真实 SOMA 动作下出现明显糖纸式扭曲，后续需要在 Soma 骨架可用骨骼之间重新分配局部权重，或增加 shader 层面的 corrective deformation。首版先以实际 Soma 动作验证问题是否存在。

### 手指映射

Quinn 的非拇指结构为 `metacarpal -> 01 -> 02 -> 03`，SOMA 对应为编号 `1 -> 2 -> 3 -> 4`。因此每根非拇指都使用：

| Quinn joint | SOMA bone |
| --- | --- |
| `<finger>_metacarpal_l/r` | `LeftHand<finger>1` / `RightHand<finger>1` |
| `<finger>_01_l/r` | `LeftHand<finger>2` / `RightHand<finger>2` |
| `<finger>_02_l/r` | `LeftHand<finger>3` / `RightHand<finger>3` |
| `<finger>_03_l/r` | `LeftHand<finger>4` / `RightHand<finger>4` |

`finger` 为 `Index`、`Middle`、`Ring`、`Pinky`。拇指的 `thumb_01`、`thumb_02`、`thumb_03` 分别映射至 `Thumb1`、`Thumb2`、`Thumb3`。

SOMA 的 `*End`、`Jaw`、`LeftEye`、`RightEye` 等骨骼可以为零权重；mesh 的每个 SOMA 骨骼都不需要有顶点影响。

### 脊柱规则

Quinn 有五节脊柱，而 SOMA 的可变形主体为 `Hips -> Spine1 -> Spine2 -> Chest`。初始映射采用 `spine_02 -> Spine1`、`spine_03 -> Spine2`、`spine_04 -> Chest`。

`spine_01` 的权重在首版全部合并至 `Spine1`，并在报告中单独列出。如果腰腹测试出现明显折痕，第二版按顶点相对于 Hips 和 Spine1 的纵向位置分配该权重：靠近髋部的部分给 Hips，靠近上躯干的部分给 Spine1；每个顶点仍须保持总权重为 1。

## 权重计算过程

对每个 Quinn 顶点执行以下过程：

1. 读出四个 Quinn skin joint indices 和权重。
2. 将每个源 joint 通过显式映射表转为 SOMA bone index。
3. 若多个源影响映射到同一 SOMA bone，累加权重。
4. 删除零权重项并按权重从大到小排序。
5. 保留前四项；如果多于四项，记录被丢弃权重的总和。
6. 将保留权重除以其和，确保每顶点权重和为 1。
7. 为少于四项的顶点用零权重和任意有效 bone id 填充到四个槽位。

转换报告应包含：

- 每个 Quinn joint 的总输入权重；
- 每个 SOMA bone 的总输出权重；
- 每条映射规则传递的总权重；
- 每顶点丢弃权重的最大值、平均值与总和；
- 零权重顶点数；
- 最终影响数为 1、2、3、4 的顶点数量。

若任一顶点无法映射到有效 SOMA bone，脚本应失败并列出源 joint 名称和受影响顶点数。不要静默把未知骨骼映射至 Hips。

## Mesh 和材质写入

Quinn 当前已经是三角网格。首版应保留 positions、normals、UV 和 indices，不重新三角化。若需要按 UV seam 扩展顶点，必须同步复制 position、normal、bone ids 和 bone weights。

法线采用 GLB 中的 `NORMAL`，让 `GenMeshTangents` 在加载 `SOMA.bin` 后生成切线。写入纹理时：

- base-color 贴图作为 sRGB；
- normal 贴图作为 linear；
- 初始 `metallic=0`、`roughness=0.58`、`ao=1`，因为 GLB 未提供项目所需的 packed metallic/roughness/AO 贴图。

当前 PBR 渲染路径会将 `DrawModel` 的 tint 乘到 base-color 贴图上。Quinn 使用贴图时应提供一个最小的 `--character-tint` 参数，默认白色 `(255, 255, 255, 255)`；否则当前角色橙色 tint 会污染 Quinn 原始外观。

## 文件级验证

脚本必须在写入前、写入后都执行以下检查：

1. 输出 bone count 为 78，bone names 和 parents 与源 `SOMA.bin` 完全一致。
2. 输出 bind positions 和 bind rotations 与源 `SOMA.bin` 一致，允许浮点序列化误差。
3. 顶点数低于 65,536，所有 triangle index 在范围内。
4. 每个 `bone_id` 在 `[0, 77]`。
5. 每个 weight 有限、非负；每顶点权重和与 1 的误差小于 `1e-5`。
6. 每顶点至少有一个大于零的权重。
7. positions、normals、UV、bone ids、weights 的顶点维度一致。
8. 输出文件大小和各段 offset 与 `load_geno_model` 的二进制布局一致。
9. 将输出重新读回，检查其 bone metadata 和数组内容。

新增单元测试应使用一个小型合成 GLB fixture，覆盖 accessor offset/stride、twist 合并、手指偏移映射、重复影响合并、四权重截断、归一化及无效 joint 的失败路径。另加一个可选的真实 Quinn 集成测试，仅在输入文件存在时运行。

## 视觉验证

数值验证通过后，使用独立资源目录做两类渲染。

静止姿势检查：

```bash
python -m stylized_motion.anim.render_stills \
  --pipeline somaview \
  --bvh data/assets/somaview/SOMA_apose.bvh \
  --resources-root data/assets/somaview_quinn \
  --base-color-map data/assets/somaview_quinn/quinn_base_color.jpg \
  --normal-map data/assets/somaview_quinn/quinn_normal.jpg \
  --skeleton \
  --output docs/assets/quinn_soma/apose.png
```

动作检查使用一个短 SOMA clip 或训练输出，至少采样：站立、肘膝大幅弯曲、躯干前屈/侧弯/扭转、手腕旋转、手指弯曲及脚底接地帧。输出每个检查动作的代表帧和一段短视频。

验收时重点检查：

- 骨架关节是否位于合理的视觉关节中心；
- 肩、肘、膝、踝处是否塌陷、爆开或有显著 candy-wrapper 扭转；
- 腰腹是否有不合理折痕；
- 手指是否错节，手掌和拇指是否穿插；
- 脚底在站立和接触时是否明显偏移；
- UV seam、base-color、normal map 是否连续且颜色未被 tint 污染；
- 动画帧间是否跳变。

文件级检查通过只表明转换格式和绑定关系正确；上述动作检查通过后，才能确认 Quinn 外观可以用于 SOMA 可视化。

## 问题定位和迭代策略

| 现象 | 优先排查 | 处理方向 |
| --- | --- | --- |
| 整条肢体相对骨架偏移 | 坐标变换、缩放、bind pose | 修正全局或身体段落对齐 |
| 肘、膝附近塌陷 | 关节中心、twist 合并后的权重 | 先校正关节位置，再调整局部权重分布 |
| 手指错节 | metacarpal 与第一指节映射 | 检查手指四节偏移映射表 |
| 腰腹折痕 | `spine_01` 分配 | 在 Hips 与 Spine1 间按位置分配该权重 |
| 前臂或小腿扭曲 | twist 合并限制 | 重新分配相邻主骨骼权重；必要时引入 corrective deformation |
| 颜色偏橙或发黑 | draw tint、贴图色彩空间 | 默认白色 tint，确认 base-color 为 sRGB、normal 为 linear |
| 法线凹凸方向错误 | 切线空间或贴图绿色通道 | 检查 normal map Y 通道并提供显式翻转选项 |

若第一版在关节对齐和局部权重修正后仍无法接受，再安装 Blender，用它只处理问题区域：查看骨架与 mesh、weight paint、保存可复现的局部修正数据。转换主流程仍保留为 Python 脚本，以便实验和论文图的可重复生成。

## 许可与归属

输入 GLB 的 metadata 指向 Sketchfab 的 Quinn Web Ready 模型，作者为 SketchPunk，许可证为 CC BY 4.0。任何公开发布的截图、视频、模型或衍生资源应保留合理署名、链接至许可证，并标明已将模型重新绑定至 SOMA 骨骼。

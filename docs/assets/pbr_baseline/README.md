# PBR 回归图像

此目录保存由 `stylized_motion.anim.render_stills` 离屏渲染得到的受版本控制图像，供人工检查或 `compare_stills` 像素比较使用。它们不是渲染架构说明；当前实现以 `stylized_motion/anim/` 和 `data/assets/*/*.fs` 为准。

离屏工具直接读取与交互显示相同的最终 render target。final/legacy 图像包含 FXAA；调试视图按约定跳过 tonemap 和 FXAA；导出前统一垂直翻转。离屏相机与交互 viewer 均使用 `0.01–50.0` 裁剪面。

## 再生成

先准备对应的 Geno/SOMA 可再生资产，再运行：

```bash
# GenoView：final 和 11 个调试视图
for mode in final base_color metallic roughness normal depth ao shadow diffuse specular ibl hdr; do
  python -m stylized_motion.anim.render_stills \
    --bvh data/assets/genoview/Geno_bind.bvh \
    --debug-view $mode \
    --output docs/assets/pbr_baseline/genoview_$mode.png
done

# legacy 对照与 SomaView final
python -m stylized_motion.anim.render_stills \
  --bvh data/assets/genoview/Geno_bind.bvh \
  --shading legacy \
  --output docs/assets/pbr_baseline/genoview_legacy.png

python -m stylized_motion.anim.render_stills \
  --pipeline somaview \
  --bvh data/assets/somaview/SOMA_bind.bvh \
  --output docs/assets/pbr_baseline/somaview_final.png
```

当前 PBR renderer 使用一张 depth shadow map 的 PCF 路径；修改光照、阴影、材质、后处理或相机约定后，应先重新生成受影响的图像，再将其用作新的视觉基线。

## 对比

```bash
python -m stylized_motion.anim.compare_stills \
  --reference docs/assets/pbr_baseline/genoview_final.png \
  --current /tmp/current_final.png
```

均值差超过阈值（默认 `0.02`）时命令以退出码 1 结束。

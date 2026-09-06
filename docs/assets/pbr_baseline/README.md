# PBR 基线截图

由 `stylized_motion.anim.render_stills` 离屏渲染生成（单帧、不依赖窗口合成，
绕开 macOS 隐藏窗口 `take_screenshot` 抓黑帧的平台问题）。所有图像直读与
交互显示相同的最终 render target：final/legacy 包含 FXAA，调试视图按契约跳过
FXAA；导出前统一做垂直翻转，保持文件为顶部朝上。离屏相机使用与交互 viewer
相同的 `0.01–50.0` 裁剪面，使 CSM split 和级联过渡也进入回归覆盖。

## 再生成

```bash
# 12 个 PBR 调试视图（GenoView）
for mode in final base_color metallic roughness normal depth ao shadow diffuse specular ibl hdr; do
  python -m stylized_motion.anim.render_stills \
    --bvh data/assets/genoview/Geno_bind.bvh \
    --debug-view $mode \
    --output docs/assets/pbr_baseline/genoview_$mode.png
done

# legacy 对照 + SomaView
python -m stylized_motion.anim.render_stills --bvh data/assets/genoview/Geno_bind.bvh \
  --shading legacy --output docs/assets/pbr_baseline/genoview_legacy.png
python -m stylized_motion.anim.render_stills --pipeline somaview \
  --bvh data/assets/somaview/SOMA_bind.bvh \
  --output docs/assets/pbr_baseline/somaview_final.png
```

## 回归对比

```bash
python -m stylized_motion.anim.compare_stills \
  --reference docs/assets/pbr_baseline/genoview_final.png \
  --current /tmp/current_final.png
```

均值差超过阈值（默认 0.02）时以退出码 1 失败。2026-09-05 后的最终 target
导出修复了历史基线的上下倒置，并把 final/legacy 切换为 FXAA 后像素；首次更新
应整体重生成基线。此后有意修改渲染外观时，重新生成对应基线并一并提交；无意的
差异即回归信号。

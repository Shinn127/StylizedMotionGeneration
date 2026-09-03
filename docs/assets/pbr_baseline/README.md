# PBR 基线截图

由 `stylized_motion.anim.render_stills` 离屏渲染生成（单帧、不依赖窗口合成，
绕开 macOS 隐藏窗口 `take_screenshot` 抓黑帧的平台问题）。final 视图为
FXAA 之前的显示目标，其余与交互路径逐像素一致。

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

均值差超过阈值（默认 0.02）时以退出码 1 失败。**有意**修改渲染外观时，
重新生成对应基线并一并提交；无意的差异即回归信号。

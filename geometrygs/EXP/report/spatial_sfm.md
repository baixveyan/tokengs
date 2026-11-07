# spatial_sfm：完整 TokenGS 上游 + 新空间高斯生成器

状态：实现及工程冒烟通过，正式训练未启动。

- 1024 维，12 层官方图像编码、12 层官方 scene latent 聚合，保留多尺度/归一化/LayerScale。
- 2 层恒等初始化的 SfM 场景融合，4 层新空间 GS decoder，1024 维局部 child MLP。
- 4096 scene latents、4096 GS tokens、每 token 64 高斯，最多 1024 测量锚点，其余为自由 token。
- 总参数 454.51M，严格从官方 release 迁移上游 340.78M，新增 113.73M；原 GS query/decoder/head 不加载，无宽度压缩。
- 默认 RGB/SSIM/可见性/激光渲染深度；无贴面，软链关闭。`spatial_sfm_chain` 仅增加软链损失，`spatial_free` 关闭 SfM，均同规模、同上游初始化。
- 新模块 lr 2e-4，上游 2e-5；默认所有参数训练，100 epoch（预算尚未经收敛验证）。正式协议为 50 验证场景、6 输入/8 目标、FP32。

运行：`bash EXP/run_train.sh spatial_sfm`。当前 75M 版本仅作为 `spatial_small_sfm` 显式消融保留。

实现细节及验证范围见 `docs/SPATIAL_ARCHITECTURE.md`。`EXP/spatial_tokengs_smoke` 只是在验证 demo 场景上更新一步的工程测试，不是此实验的质量结果。

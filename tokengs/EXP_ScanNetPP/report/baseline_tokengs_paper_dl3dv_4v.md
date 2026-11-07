# baseline_tokengs_paper_dl3dv_4v

- 模型：官方论文版 TokenGS `checkpoints_hf/checkpoints/dl3dv_4v.safetensors`（3 层编码器 + 12 层解码器、无场景隐变量、4096 token × 64 高斯、4 视角训练），SHA256 `0d6012d133e2488be82da4c747aae9e7cf7f2320d4d72f6bc224387f65e11980`。它是 LocusGS 论文表 1 的 TokenGS 基线，与 LocusGS 结构对等。用 `4DGS/LocusGS` 仓库的 `tokengs` 模型类加载（严格匹配）。
- 协议：同 `baseline_locusgs_dl3dv_4v.md`（first_cam 归一化，其余与 `baseline_pretrained_6view` 相同）。脚本 `scripts/evaluate_scannetpp_query_models.py --model tokengs_paper_4v`。未导出 Demo，未登记进总表。
- 结果（`summary.json`）：PSNR 28.13、SSIM 0.898、LPIPS 0.237；AbsRel 0.188、δ<1.05 0.401；视线厚度中位 / p90 20.5 / 48.2 cm；法线中位 39.4°；Accuracy 19.9 cm、Completeness 11.0 cm、F@3 0.272、F@5 0.397；尺度门限中位比 1.036。
- 分析见 `EXP_DL3DV/diag_locusgs_report.md`。

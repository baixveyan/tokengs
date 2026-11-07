# baseline_locusgs_dl3dv_4v

- 模型：LocusGS（arXiv 2608.12825）DL3DV 微调权重，4 视角训练、4096 token × 64 高斯、256×448；`/mnt/disk/xveyan/checkpoints/locusgs/checkpoints/dl3dv/finetune/model.safetensors`，SHA256 `a554f552505f97122ae7e745090d37cefd023b4d9c7f87de3ce22e5683e09ec0`。用 `4DGS/LocusGS` 仓库的模型类加载（多出的 `anchor_head_cross_attn` 未使用）。
- 协议：与 `baseline_pretrained_6view` 相同（50 个官方验证场景、6 张输入、窗口 18 内 8 个插值目标、FP32、无 TF32、激光网格 G1–G4、0 点云、0 优化步），只是相机和网格按 first_cam 归一化（与该模型训练一致），缩放 0.3。脚本 `scripts/evaluate_scannetpp_query_models.py --model locusgs_finetune_4v`。没有导出 Demo，也未登记进 `all_experiments_table`（它的入口不是 `evaluate_scannetpp.py`）。
- 结果（`summary.json`）：PSNR 29.52、SSIM 0.922、LPIPS 0.195；AbsRel 0.131、δ<1.05 0.510；视线厚度中位 / p90 12.9 / 35.0 cm；法线中位 30.4°；Accuracy 16.3 cm、Completeness 8.7 cm、F@3 0.351、F@5 0.493；尺度门限中位比 1.020。
- 对照：对等的论文版 TokenGS 见 `baseline_tokengs_paper_dl3dv_4v.md`；分析见 `EXP_DL3DV/diag_locusgs_report.md`。

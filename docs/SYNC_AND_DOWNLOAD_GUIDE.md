# 双服务器同步与大文件（权重 / 数据集 / Demo 点云）下载指南

本仓库（[`baixveyan/tokengs`](https://github.com/baixveyan/tokengs)）已同步包含：
- 全部核心源码（`tokengs/`）、实验与评估脚本（`scripts/`）、单元测试（`tests/`）、研究文档与记忆库（`docs/`、`AGENTS.md`、`TokenGS.pdf`）；
- `EXP_TokenGS/` 下全部 13 组 140 场景实验的完整指标、总表（`all_experiments_table.{md,csv}`）、四阶段报告（`reports/01_... ~ 04_...`）、终端日志（`console.log`）；
- `EXP_TokenGS/demo/scene_2f3e1c0f688c/` 下的对比表（`comparison.md`）、微调曲线图（`tuning_curve.png`）、渲染对比条（`render_strip.png`）、COLMAP 监督点云（`colmap_*.ply`）以及 10/20 个高亮 Token 点云（`*_only{10,20}tokens*.ply`）。

为避免 Git 仓库体积膨胀（超出 GitHub 单文件 100 MB 限制），以下三类大体积二进制文件未直接纳入 Git，在新服务器上按下方命令即可一键准备完毕：

---

## 一、下载预训练模型权重（`checkpoints_hf/`）

所有实验默认使用的官方 Release 2026.6 权重为：
- `checkpoints_hf/checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors`（1.38 GB，SHA256: `5913da347475049cb58e552634ee1c527293c2571576806db0d4da702ce978c3`）

### 在新服务器上下载命令（支持国内 `hf-mirror` 镜像加速）：
```bash
pip install -U huggingface_hub

# 1. 仅下载本实验必需的 6v SSIM 权重（仅 1.38 GB，推荐）：
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  nvidia/TokenGS \
  checkpoints_26.06/dl3dv_latent_6v_ssim.safetensors \
  --local-dir checkpoints_hf

# 2. 若需下载官方仓库全部权重（约 14.4 GB）：
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  nvidia/TokenGS \
  --local-dir checkpoints_hf
```

---

## 二、准备或链接 DL3DV-10K Benchmark 数据集（`data/dl3dv_eval`）

评估脚本默认从 `data/dl3dv_eval/<scene_id>.zip` 读取 140 个测试场景（每个 `.zip` 内含图像、`transforms.json` 及 `gaussian_splat/sparse/0/points3D.bin` COLMAP 点云）。

### 1. 若新服务器上已有解压/下载好的 `DL3DV-10K-Benchmark` 目录：
直接建立软链接即可：
```bash
mkdir -p data
ln -snf /你的数据集路径/DL3DV-10K-Benchmark data/dl3dv_eval
ln -snf /你的数据集路径/DL3DV-10K-Benchmark data/dl3dv
```

### 2. 若新服务器尚未下载 `DL3DV-10K-Benchmark`：
可从 HuggingFace 官方仓库 `DL3DV/DL3DV-10K-Benchmark` 下载（或从第一台服务器的 `/mnt/disk/xveyan/4dgs/DL3DV-10K-Benchmark` 通过 `rsync` 同步）：
```bash
# 方式 A：从第一台服务器直接 rsync 同步
rsync -avP jms@<源服务器IP>:/mnt/disk/xveyan/4dgs/DL3DV-10K-Benchmark/ /目标路径/DL3DV-10K-Benchmark/
ln -snf /目标路径/DL3DV-10K-Benchmark data/dl3dv_eval

# 方式 B：使用 huggingface-cli 从镜像下载
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  DL3DV/DL3DV-10K-Benchmark \
  --repo-type dataset \
  --local-dir data/dl3dv_eval
```

---

## 三、（可选）在新服务器上重新生成 Demo 场景的完整 26.2 万点 `.ply` 与 `.pt`

`EXP_TokenGS/demo/scene_2f3e1c0f688c/<实验名>/` 下每个方法的完整未裁剪点云（`<实验名>.ply`、`<实验名>_slotcolor.ply`、`gaussians.pt`、`decoder_inputs.pt`，单个方法约 180 MB）可通过以下命令在闲置 GPU 上十几秒内重新生成：

```bash
# 重新生成某个具体实验的完整 Demo PLY（例如去噪 COLMAP 方案，约 12 秒）：
rm -rf EXP_TokenGS/demo/scene_2f3e1c0f688c/scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning
bash EXP_TokenGS/run_demo.sh scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning
```

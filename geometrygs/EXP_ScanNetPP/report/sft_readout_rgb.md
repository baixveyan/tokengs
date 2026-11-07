# 读出链路纯 RGB 后训练对照（`sft_readout_rgb`）

状态：待运行（`bash EXP_ScanNetPP/run_50_scenes.sh sft_readout_rgb` 启动；训练用启动时全部空闲 GPU，评测用 1 张）。

## 一、动机

注意力局部化 / 解码器读取诊断（`docs/DIAG_latent_attention_geometry.md`）表明：阶段 ④ 记忆格与阶段 ⑤ 解码读取已把每个 token 的表面位置定到 3 cm 以内，几何缺陷集中在阶段 ⑥.1 的线性读出头把子球摊散、10–20% 子球离面。R 系列阶梯（`docs/AGENT_MEMORY.md` 15.12）因此只解冻**记忆格之后的读出链路**，每步只加一样东西。本实验不加任何几何监督，是 `sft_readout_rgb_depth` 及之后各步的真正对照：它回答“仅仅解冻读出链路并用光度损失训练，几何会不会自己变好”。

## 二、设计

- **初始化**：官方 `dl3dv_latent_6v_ssim.safetensors`（strict 加载，第 0 步与基线逐位一致）。
- **解冻（读出链路，21.91M 参数）**：`gs_tokens`、`enc_dec_backbone.latent_kv_proj` + `k_proj_norm`（记忆格 → 解码器 K/V）、`enc_dec_backbone.decoder_blocks`（1 层）、`activation_head`（`deconv`）。记忆格 `latents` 与其之前全部冻结。
- **读出头学习率缩放**：`activation_head` 参数的学习率 × 0.1（`sft_readout_head_lr_scale=0.1`）。原因：Adam 早期步长近似符号步，对扇入 1024 的末端线性层会产生相干的输出位移，直接拽动高斯位置（DL3DV SFT-3 放开 `deconv` 掉 0.88 dB 的机制）。
- **训练配置**：Preset `sft_scannetpp_readout_rgb`；853 个训练场景（repeat=4），6 输入 + 4 目标视角（`use_input_supervision=True`），`256×448`，**10 Epochs**，峰值学习率 `4e-5`（OneCycleLR，warmup 200 步），FP32，seed=42，每卡 batch 1、梯度累积 2。
- **损失**：仅 `0.8·L1 + 0.2·SSIM + LPIPS + Visibility`，0 几何监督。
- **验证**：每 Epoch 在 50 个验证场景报 PSNR 与**深度 AbsRel**（`val_depth_metric=True`，目标视角真值由网格投射）；验证加载器在主进程运行（forked worker 内 Open3D 投射会死锁）。
- **评测**：最优验证 PSNR 权重 → 50 场景 PSNR/SSIM/LPIPS + G1–G4 + **G5**（`gs_opacity_binarized`：存活高斯中 α>0.9 比例；`gs_opacity_midrange`：0.2≤α≤0.8 比例；`gs_floater_rate`：α>0.5 的高斯中离真值表面 >10 cm 的比例）。

## 三、预期与判读

- 预期：PSNR 略升（域适应），G1–G4 与 G5 基本不变（同 `sft_latenttail_rgb` 的结论）。
- 若本实验自己就明显改善几何：说明读出链路在光度下能自我修正，需重新解释问题来源。

## 四、结果

待运行完成后自动写入 `../sft_readout_rgb/` 与 `../all_experiments_table.md`。

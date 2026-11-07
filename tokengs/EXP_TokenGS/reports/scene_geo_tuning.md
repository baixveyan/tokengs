# Geometry loss inside scene latent tuning (FP32, 140 scenes): report

## Per group (140-scene mean)

| group | all PSNR | all SSIM | all LPIPS | novel PSNR | edge L1 | HF ratio | d4+ PSNR |
|---|---|---|---|---|---|---|---|
| no_tuning | 25.194 | 0.8081 | 0.2549 | 25.007 | 0.0224 | 0.759 | 24.108 |
| scene_lr1e-2_tuning | 27.400 | 0.8683 | 0.1831 | 26.845 | 0.0183 | 0.852 | 25.325 |
| scene_lr1e-2_geo_tuning | 26.734 | 0.8525 | 0.2122 | 26.251 | 0.0197 | 0.829 | 24.816 |

## Geometry / needles (scene median)

| group | SW | packing ratio / target | inter d | too close | too far | eff. children | aniso p50 | long axis ∥ ray >0.9 |
|---|---|---|---|---|---|---|---|---|
| no_tuning | 0.7132 | 470.4 / 6.08 | 0.472 | 0.919 | 0.026 | 15.4 | 27.2 | 0.274 |
| scene_lr1e-2_tuning | 0.7123 | 517.9 / 5.83 | 0.482 | 0.917 | 0.028 | 14.5 | 28.1 | 0.264 |
| scene_lr1e-2_geo_tuning | 0.3699 | 180.6 / 5.67 | 1.031 | 0.788 | 0.050 | 13.9 | 26.1 | 0.243 |

## Paired comparisons (delta = this - reference, 95% CI, scene wins this/ref)

The last two are frozen in `EXP_TokenGS/reports/scene_tuning_noise_floor.json` (extra runs of scene_lr1e-2_tuning, deleted after freezing).

### scene_tuning_gain: scene_lr1e-2_tuning - no_tuning

| key | Δ mean | 95% CI | wins | |Δ| p95 |
|---|---|---|---|---|
| all_psnr | +2.2061 | [+2.0592, +2.3531] | 140/0 | 3.8877 |
| all_ssim | +0.0602 | [+0.0562, +0.0642] | 140/0 | 0.1038 |
| all_lpips | -0.0718 | [-0.0742, -0.0694] | 140/0 | 0.0967 |
| all_edge_l1 | -0.0041 | [-0.0044, -0.0038] | 139/1 | 0.0076 |
| all_hf_ratio | +0.0937 | [+0.0862, +0.1011] | 140/0 | 0.1765 |
| novel_psnr | +1.8381 | [+1.6729, +2.0033] | 138/2 | 3.6633 |
| novel_ssim | +0.0554 | [+0.0515, +0.0592] | 140/0 | 0.0955 |
| novel_lpips | -0.0687 | [-0.0710, -0.0663] | 140/0 | 0.0919 |
| novel_edge_l1 | -0.0038 | [-0.0041, -0.0035] | 138/2 | 0.0074 |
| novel_hf_ratio | +0.0936 | [+0.0861, +0.1010] | 140/0 | 0.1769 |
| bucket_d0_psnr | +4.9051 | [+4.7747, +5.0355] | 140/0 | 6.0298 |
| bucket_d1_psnr | +2.7601 | [+2.5893, +2.9309] | 139/1 | 4.5270 |
| bucket_d2-3_psnr | +1.8404 | [+1.6641, +2.0167] | 137/3 | 3.6959 |
| bucket_d4+_psnr | +1.2175 | [+1.0576, +1.3774] | 129/11 | 2.9563 |

### geo_loss_effect: scene_lr1e-2_geo_tuning - scene_lr1e-2_tuning

| key | Δ mean | 95% CI | wins | |Δ| p95 |
|---|---|---|---|---|
| all_psnr | -0.6653 | [-0.7406, -0.5899] | 1/139 | 1.5004 |
| all_ssim | -0.0158 | [-0.0166, -0.0150] | 0/140 | 0.0234 |
| all_lpips | +0.0290 | [+0.0279, +0.0302] | 0/140 | 0.0408 |
| all_edge_l1 | +0.0014 | [+0.0013, +0.0015] | 0/140 | 0.0024 |
| all_hf_ratio | -0.0233 | [-0.0245, -0.0221] | 0/140 | 0.0346 |
| novel_psnr | -0.5941 | [-0.6706, -0.5177] | 4/136 | 1.4494 |
| novel_ssim | -0.0156 | [-0.0165, -0.0148] | 0/140 | 0.0236 |
| novel_lpips | +0.0288 | [+0.0276, +0.0299] | 0/140 | 0.0403 |
| novel_edge_l1 | +0.0013 | [+0.0012, +0.0014] | 1/139 | 0.0025 |
| novel_hf_ratio | -0.0233 | [-0.0246, -0.0221] | 0/140 | 0.0347 |
| bucket_d0_psnr | -1.1871 | [-1.2725, -1.1017] | 1/139 | 2.1682 |
| bucket_d1_psnr | -0.7141 | [-0.7933, -0.6348] | 0/140 | 1.6586 |
| bucket_d2-3_psnr | -0.5953 | [-0.6716, -0.5190] | 5/135 | 1.3713 |
| bucket_d4+_psnr | -0.5090 | [-0.5880, -0.4300] | 8/132 | 1.2326 |

### scene_geo_tuning_vs_no_tuning: scene_lr1e-2_geo_tuning - no_tuning

| key | Δ mean | 95% CI | wins | |Δ| p95 |
|---|---|---|---|---|
| all_psnr | +1.5408 | [+1.4414, +1.6403] | 140/0 | 2.4749 |
| all_ssim | +0.0444 | [+0.0406, +0.0482] | 140/0 | 0.0852 |
| all_lpips | -0.0428 | [-0.0457, -0.0398] | 140/0 | 0.0694 |
| all_edge_l1 | -0.0027 | [-0.0030, -0.0025] | 139/1 | 0.0056 |
| all_hf_ratio | +0.0704 | [+0.0635, +0.0773] | 139/1 | 0.1484 |
| novel_psnr | +1.2440 | [+1.1301, +1.3578] | 137/3 | 2.2997 |
| novel_ssim | +0.0397 | [+0.0362, +0.0433] | 139/1 | 0.0772 |
| novel_lpips | -0.0399 | [-0.0427, -0.0370] | 139/1 | 0.0659 |
| novel_edge_l1 | -0.0025 | [-0.0027, -0.0022] | 137/3 | 0.0054 |
| novel_hf_ratio | +0.0702 | [+0.0634, +0.0771] | 139/1 | 0.1486 |
| bucket_d0_psnr | +3.7180 | [+3.5949, +3.8412] | 140/0 | 5.1609 |
| bucket_d1_psnr | +2.0460 | [+1.9308, +2.1613] | 139/1 | 2.9968 |
| bucket_d2-3_psnr | +1.2451 | [+1.1222, +1.3680] | 136/4 | 2.3305 |
| bucket_d4+_psnr | +0.7085 | [+0.5923, +0.8247] | 121/19 | 1.8635 |

### repeat_run_noise_floor: scene_lr1e-2_tuning (repeat run) - scene_lr1e-2_tuning

| key | Δ mean | 95% CI | wins | |Δ| p95 |
|---|---|---|---|---|
| all_psnr | +0.0003 | [-0.0034, +0.0039] | 66/74 | 0.0150 |
| all_ssim | -0.0000 | [-0.0000, +0.0000] | 65/75 | 0.0003 |
| all_lpips | +0.0000 | [-0.0000, +0.0000] | 69/71 | 0.0005 |
| all_edge_l1 | +0.0000 | [-0.0000, +0.0000] | 62/78 | 0.0000 |
| all_hf_ratio | +0.0001 | [-0.0000, +0.0003] | 85/55 | 0.0006 |
| novel_psnr | -0.0008 | [-0.0027, +0.0011] | 67/73 | 0.0182 |
| novel_ssim | -0.0000 | [-0.0000, +0.0000] | 68/72 | 0.0004 |
| novel_lpips | +0.0000 | [-0.0000, +0.0000] | 68/72 | 0.0004 |
| novel_edge_l1 | +0.0000 | [-0.0000, +0.0000] | 62/78 | 0.0000 |
| novel_hf_ratio | +0.0001 | [-0.0000, +0.0003] | 84/56 | 0.0006 |
| bucket_d0_psnr | +0.0080 | [-0.0112, +0.0272] | 65/75 | 0.0301 |
| bucket_d1_psnr | -0.0002 | [-0.0037, +0.0034] | 66/74 | 0.0269 |
| bucket_d2-3_psnr | -0.0007 | [-0.0028, +0.0014] | 67/73 | 0.0194 |
| bucket_d4+_psnr | -0.0015 | [-0.0033, +0.0004] | 70/70 | 0.0262 |

### reimplemented_loop_vs_official_loop: scene_lr1e-2_tuning - scene_lr1e-2_tuning (official loop)

| key | Δ mean | 95% CI | wins | |Δ| p95 |
|---|---|---|---|---|
| all_psnr | +0.0002 | [-0.0038, +0.0042] | 81/59 | 0.0187 |
| all_ssim | +0.0000 | [-0.0000, +0.0000] | 71/69 | 0.0003 |
| all_lpips | -0.0000 | [-0.0000, +0.0000] | 70/70 | 0.0004 |
| novel_psnr | +0.0014 | [-0.0007, +0.0036] | 80/60 | 0.0181 |
| novel_ssim | +0.0000 | [-0.0000, +0.0000] | 75/65 | 0.0003 |
| novel_lpips | -0.0000 | [-0.0001, +0.0000] | 70/70 | 0.0004 |

## Geometry loss effect vs repeat-run noise floor

| key | geo Δ | geo CI95 | noise Δ | noise CI95 | CI excl. 0 | disjoint from noise CI |
|---|---|---|---|---|---|---|
| all_psnr | -0.6653 | [-0.7406, -0.5899] | +0.0003 | [-0.0034, +0.0039] | True | True |
| all_ssim | -0.0158 | [-0.0166, -0.0150] | -0.0000 | [-0.0000, +0.0000] | True | True |
| all_lpips | +0.0290 | [+0.0279, +0.0302] | +0.0000 | [-0.0000, +0.0000] | True | True |
| all_edge_l1 | +0.0014 | [+0.0013, +0.0015] | +0.0000 | [-0.0000, +0.0000] | True | True |
| all_hf_ratio | -0.0233 | [-0.0245, -0.0221] | +0.0001 | [-0.0000, +0.0003] | True | True |
| novel_psnr | -0.5941 | [-0.6706, -0.5177] | -0.0008 | [-0.0027, +0.0011] | True | True |
| novel_ssim | -0.0156 | [-0.0165, -0.0148] | -0.0000 | [-0.0000, +0.0000] | True | True |
| novel_lpips | +0.0288 | [+0.0276, +0.0299] | +0.0000 | [-0.0000, +0.0000] | True | True |
| novel_edge_l1 | +0.0013 | [+0.0012, +0.0014] | +0.0000 | [-0.0000, +0.0000] | True | True |
| novel_hf_ratio | -0.0233 | [-0.0246, -0.0221] | +0.0001 | [-0.0000, +0.0003] | True | True |
| bucket_d0_psnr | -1.1871 | [-1.2725, -1.1017] | +0.0080 | [-0.0112, +0.0272] | True | True |
| bucket_d1_psnr | -0.7141 | [-0.7933, -0.6348] | -0.0002 | [-0.0037, +0.0034] | True | True |
| bucket_d2-3_psnr | -0.5953 | [-0.6716, -0.5190] | -0.0007 | [-0.0028, +0.0014] | True | True |
| bucket_d4+_psnr | -0.5090 | [-0.5880, -0.4300] | -0.0015 | [-0.0033, +0.0004] | True | True |

## Demo scene

- 2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd: gain +2.212 dB (median +2.215 dB) — scene_tuning - no_tuning all_frames PSNR gain closest to the 140-scene median; ties -> lexicographically first scene_id (pre-registered)
- next: `bash EXP_TokenGS/run_demo.sh` (every finished experiment, one folder per experiment in `EXP_TokenGS/demo/scene_<id[:12]>/`)

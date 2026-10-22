# Tuned parameters x geometry loss (140 scenes): report

Scene latents lr 1e-2, Gaussian tokens lr 1e-4 or 1e-2, 50 steps; geometry loss weights sw / packing / inter = 0.1 / 0.001 / 0.01. Token tuning lr 1e-4 without geometry loss = the earlier official-loop run.

## all PSNR grid (140-scene mean; no tuning 25.194)

| tuned | render loss only | render + geometry loss | geometry loss Δ (paired, 95% CI) |
|---|---|---|---|
| scene latents (lr 1e-2) | 27.400 | 26.734 | -0.665 [-0.741, -0.590] |
| Gaussian tokens (lr 1e-4) | 26.810 | 26.606 | -0.204 [-0.229, -0.178] |
| scene latents (lr 1e-2) + Gaussian tokens (lr 1e-4) | 27.481 | 26.940 | -0.541 [-0.603, -0.479] |

## Per group (140-scene mean)

| group | all PSNR | all SSIM | all LPIPS | novel PSNR | HF ratio | d4+ PSNR |
|---|---|---|---|---|---|---|
| no_tuning | 25.194 | 0.8081 | 0.2549 | 25.007 | 0.759 | 24.108 |
| scene_lr1e-2_tuning | 27.400 | 0.8683 | 0.1831 | 26.845 | 0.852 | 25.325 |
| scene_lr1e-2_geo_tuning | 26.734 | 0.8525 | 0.2122 | 26.251 | 0.829 | 24.816 |
| token_lr1e-4_tuning | 26.810 | 0.8539 | 0.2153 | 26.439 | 0.767 | 25.167 |
| token_lr1e-4_geo_tuning | 26.606 | 0.8474 | 0.2296 | 26.264 | 0.744 | 25.039 |
| scene_lr1e-2_token_lr1e-4_tuning | 27.481 | 0.8707 | 0.1795 | 26.901 | 0.852 | 25.357 |
| scene_lr1e-2_token_lr1e-4_geo_tuning | 26.940 | 0.8578 | 0.2030 | 26.409 | 0.838 | 24.916 |

## Geometry / needles (scene median)

| group | SW | packing ratio / target | inter d | too close | too far | aniso p50 | long axis ∥ ray >0.9 |
|---|---|---|---|---|---|---|---|
| no_tuning | 0.7132 | 470.4 / 6.08 | 0.472 | 0.919 | 0.026 | 27.2 | 0.274 |
| scene_lr1e-2_tuning | 0.7123 | 517.9 / 5.83 | 0.482 | 0.917 | 0.028 | 28.1 | 0.264 |
| scene_lr1e-2_geo_tuning | 0.3699 | 180.6 / 5.67 | 1.031 | 0.788 | 0.050 | 26.1 | 0.243 |
| token_lr1e-4_tuning | 0.7122 | 475.1 / 6.08 | 0.538 | 0.910 | 0.030 | 27.3 | 0.275 |
| token_lr1e-4_geo_tuning | 0.6103 | 413.2 / 6.16 | 0.973 | 0.819 | 0.052 | 27.0 | 0.270 |
| scene_lr1e-2_token_lr1e-4_tuning | 0.7107 | 522.0 / 5.89 | 0.508 | 0.914 | 0.030 | 28.0 | 0.266 |
| scene_lr1e-2_token_lr1e-4_geo_tuning | 0.3493 | 188.8 / 5.73 | 1.110 | 0.763 | 0.056 | 25.1 | 0.243 |

## Paired comparisons (Δ = this - reference, mean over scenes; PSNR with 95% CI and scene wins this/ref)

| comparison | all PSNR | novel PSNR | d4+ PSNR | LPIPS | HF ratio | SW | packing ratio | inter d | aniso p50 | ∥ ray >0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| geo_loss_effect_in_scene_tuning | -0.665 [-0.741, -0.590] 1/139 | -0.594 [-0.671, -0.518] 4/136 | -0.509 [-0.588, -0.430] 8/132 | +0.02905 | -0.02328 | -0.3358 | -427.9 | +0.5246 | -2.386 | -0.01811 |
| geo_loss_effect_in_token_tuning_lr1e-4 | -0.204 [-0.229, -0.178] 2/138 | -0.175 [-0.201, -0.150] 11/129 | -0.127 [-0.153, -0.102] 24/116 | +0.01432 | -0.02295 | -0.1007 | -90.52 | +0.4197 | -0.1893 | -0.003403 |
| geo_loss_effect_in_scene_token_tuning_token_lr1e-4 | -0.541 [-0.603, -0.479] 0/140 | -0.492 [-0.555, -0.428] 3/137 | -0.442 [-0.511, -0.372] 7/133 | +0.02349 | -0.01458 | -0.3544 | -438.3 | +0.5778 | -3.407 | -0.02011 |
| token_tuning_lr1e-4_vs_scene_tuning | -0.590 [-0.648, -0.531] 5/135 | -0.406 [-0.473, -0.339] 24/116 | -0.159 [-0.224, -0.094] 52/88 | +0.03218 | -0.08517 | +2.976e-06 | -54.4 | +0.06665 | -1.049 | +0.01111 |
| adding_tokens_lr1e-4_to_scene_tuning | +0.081 [+0.068, +0.094] 127/13 | +0.056 [+0.043, +0.069] 123/17 | +0.032 [+0.018, +0.046] 109/31 | -0.003649 | -5.065e-05 | -0.001751 | +13.36 | +0.03292 | -0.05635 | +0.002422 |
| adding_tokens_lr1e-4_to_scene_geo_tuning | +0.205 [+0.186, +0.224] 139/1 | +0.158 [+0.139, +0.178] 134/6 | +0.099 [+0.079, +0.119] 116/24 | -0.009201 | +0.008648 | -0.02038 | +2.936 | +0.08609 | -1.078 | +0.0004204 |
| adding_scene_latents_to_token_tuning_lr1e-4 | +0.671 [+0.608, +0.733] 136/4 | +0.462 [+0.390, +0.534] 119/21 | +0.191 [+0.120, +0.261] 88/52 | -0.03583 | +0.08512 | -0.001754 | +67.76 | -0.03372 | +0.9931 | -0.008685 |
| adding_scene_latents_to_token_geo_tuning_lr1e-4 | +0.333 [+0.274, +0.393] 122/18 | +0.146 [+0.081, +0.210] 84/56 | -0.124 [-0.193, -0.054] 47/93 | -0.02666 | +0.09349 | -0.2555 | -280 | +0.1243 | -2.224 | -0.02539 |
| scene_lr1e-2_tuning_vs_no_tuning | +2.206 [+2.059, +2.353] 140/0 | +1.838 [+1.673, +2.003] 138/2 | +1.217 [+1.058, +1.377] 129/11 | -0.0718 | +0.0937 | -0.0009355 | +61.5 | +0.0178 | +1.092 | -0.01178 |
| scene_lr1e-2_geo_tuning_vs_no_tuning | +1.541 [+1.441, +1.640] 140/0 | +1.244 [+1.130, +1.358] 137/3 | +0.708 [+0.592, +0.825] 121/19 | -0.04276 | +0.07042 | -0.3367 | -366.4 | +0.5424 | -1.294 | -0.02989 |
| token_lr1e-4_tuning_vs_no_tuning | +1.617 [+1.522, +1.711] 140/0 | +1.432 [+1.329, +1.536] 140/0 | +1.059 [+0.959, +1.159] 140/0 | -0.03962 | +0.008526 | -0.0009325 | +7.097 | +0.08445 | +0.04246 | -0.0006722 |
| token_lr1e-4_geo_tuning_vs_no_tuning | +1.413 [+1.338, +1.488] 140/0 | +1.257 [+1.175, +1.339] 140/0 | +0.931 [+0.851, +1.012] 140/0 | -0.0253 | -0.01442 | -0.1016 | -83.42 | +0.5042 | -0.1468 | -0.004075 |
| scene_lr1e-2_token_lr1e-4_tuning_vs_no_tuning | +2.287 [+2.138, +2.436] 140/0 | +1.894 [+1.725, +2.063] 138/2 | +1.249 [+1.085, +1.414] 129/11 | -0.07545 | +0.09365 | -0.002687 | +74.86 | +0.05072 | +1.036 | -0.009357 |
| scene_lr1e-2_token_lr1e-4_geo_tuning_vs_no_tuning | +1.746 [+1.633, +1.860] 140/0 | +1.402 [+1.273, +1.532] 136/4 | +0.808 [+0.677, +0.938] 122/18 | -0.05196 | +0.07907 | -0.3571 | -363.4 | +0.6285 | -2.371 | -0.02947 |
| repeat_run_noise_floor | +0.000 [-0.003, +0.004] 66/74 | -0.001 [-0.003, +0.001] 67/73 | -0.001 [-0.003, +0.000] 70/70 | +2.279e-06 | +0.0001424 | +8.46e-06 | -0.1597 | +5.008e-05 | -0.002238 | -7.221e-06 |

Comparison definitions (this - reference): geo_loss_effect_in_scene_tuning = scene_lr1e-2_geo_tuning - scene_lr1e-2_tuning; geo_loss_effect_in_token_tuning_lr1e-4 = token_lr1e-4_geo_tuning - token_lr1e-4_tuning; geo_loss_effect_in_scene_token_tuning_token_lr1e-4 = scene_lr1e-2_token_lr1e-4_geo_tuning - scene_lr1e-2_token_lr1e-4_tuning; token_tuning_lr1e-4_vs_scene_tuning = token_lr1e-4_tuning - scene_lr1e-2_tuning; adding_tokens_lr1e-4_to_scene_tuning = scene_lr1e-2_token_lr1e-4_tuning - scene_lr1e-2_tuning; adding_tokens_lr1e-4_to_scene_geo_tuning = scene_lr1e-2_token_lr1e-4_geo_tuning - scene_lr1e-2_geo_tuning; adding_scene_latents_to_token_tuning_lr1e-4 = scene_lr1e-2_token_lr1e-4_tuning - token_lr1e-4_tuning; adding_scene_latents_to_token_geo_tuning_lr1e-4 = scene_lr1e-2_token_lr1e-4_geo_tuning - token_lr1e-4_geo_tuning; scene_lr1e-2_tuning_vs_no_tuning = scene_lr1e-2_tuning - no_tuning; scene_lr1e-2_geo_tuning_vs_no_tuning = scene_lr1e-2_geo_tuning - no_tuning; token_lr1e-4_tuning_vs_no_tuning = token_lr1e-4_tuning - no_tuning; token_lr1e-4_geo_tuning_vs_no_tuning = token_lr1e-4_geo_tuning - no_tuning; scene_lr1e-2_token_lr1e-4_tuning_vs_no_tuning = scene_lr1e-2_token_lr1e-4_tuning - no_tuning; scene_lr1e-2_token_lr1e-4_geo_tuning_vs_no_tuning = scene_lr1e-2_token_lr1e-4_geo_tuning - no_tuning; repeat_run_noise_floor = scene_lr1e-2_tuning (repeat run) - scene_lr1e-2_tuning

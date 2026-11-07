# Same-scene comparison 2f3e1c0f688c84cec67f9a1ea219c54c14ffabf31a046e620dacc690cac2f1bd

One folder per method next to this file, named after the method: `<method>.ply` (true colour), `<method>_slotcolor.ply` (one colour per token), `cameras.ply` (input red, evaluation green), `<method>_minus_no_tuning.ply` (blue = Gaussian did not move, red = moved most).

- no_tuning: `no_tuning/`
- scene_lr1e-2_tuning: `scene_lr1e-2_tuning/`
- scene_lr1e-2_geo_tuning: `scene_lr1e-2_geo_tuning/`
- token_lr1e-4_tuning: `token_lr1e-4_tuning/`
- token_lr1e-4_geo_tuning: `token_lr1e-4_geo_tuning/`
- scene_lr1e-2_token_lr1e-4_tuning: `scene_lr1e-2_token_lr1e-4_tuning/`
- scene_lr1e-2_token_lr1e-4_geo_tuning: `scene_lr1e-2_token_lr1e-4_geo_tuning/`
- scene_lr1e-2_token_lr1e-4_vis_xyz_tuning: `scene_lr1e-2_token_lr1e-4_vis_xyz_tuning/`
- scene_lr1e-2_token_lr1e-4_vis_opac_tuning: `scene_lr1e-2_token_lr1e-4_vis_opac_tuning/`
- scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning: `scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning/`
- scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning: `scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning/`
- scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning: `scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning/`
- scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning: `scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning/`

## Metrics (this scene)

| | no_tuning | scene_lr1e-2_tuning | scene_lr1e-2_geo_tuning | token_lr1e-4_tuning | token_lr1e-4_geo_tuning | scene_lr1e-2_token_lr1e-4_tuning | scene_lr1e-2_token_lr1e-4_geo_tuning | scene_lr1e-2_token_lr1e-4_vis_xyz_tuning | scene_lr1e-2_token_lr1e-4_vis_opac_tuning | scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning | scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning | scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning | scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all PSNR | 23.591 | 25.802 | 25.520 | 25.149 | 25.049 | 25.957 | 25.735 | 25.958 | 25.973 | 25.932 | 26.253 | 26.196 | 25.910 |
| novel PSNR | 23.454 | 25.316 | 25.095 | 24.869 | 24.786 | 25.449 | 25.271 | 25.448 | 25.463 | 25.425 | 25.820 | 25.757 | 25.405 |
| d0 PSNR | 24.595 | 29.365 | 28.631 | 27.206 | 26.970 | 29.689 | 29.135 | 29.704 | 29.709 | 29.654 | 29.430 | 29.415 | 29.616 |
| d4+ PSNR | 22.654 | 23.587 | 23.622 | 23.713 | 23.672 | 23.695 | 23.679 | 23.691 | 23.707 | 23.685 | 24.261 | 24.175 | 23.740 |
| all SSIM | 0.7890 | 0.8482 | 0.8388 | 0.8354 | 0.8312 | 0.8510 | 0.8424 | 0.8511 | 0.8511 | 0.8507 | 0.8574 | 0.8565 | 0.8502 |
| all LPIPS | 0.3235 | 0.2372 | 0.2623 | 0.2762 | 0.2882 | 0.2320 | 0.2535 | 0.2320 | 0.2330 | 0.2331 | 0.2325 | 0.2331 | 0.2398 |
| HF ratio | 0.713 | 0.839 | 0.817 | 0.730 | 0.711 | 0.842 | 0.828 | 0.842 | 0.841 | 0.840 | 0.836 | 0.838 | 0.839 |
| edge L1 | 0.0222 | 0.0186 | 0.0194 | 0.0193 | 0.0195 | 0.0184 | 0.0191 | 0.0184 | 0.0184 | 0.0184 | 0.0181 | 0.0182 | 0.0184 |
| SW | 0.526 | 0.532 | 0.273 | 0.528 | 0.451 | 0.532 | 0.259 | 0.535 | 0.539 | 0.101 | 0.528 | 0.528 | 0.542 |
| packing ratio | 367.0 | 406.6 | 111.7 | 361.5 | 301.3 | 402.4 | 119.6 | 402.6 | 413.4 | 5.6 | 407.0 | 397.0 | 379.8 |
| inter d | 0.810 | 0.875 | 1.480 | 1.021 | 1.509 | 0.957 | 1.585 | 0.949 | 0.951 | 1.458 | 0.953 | 0.947 | 0.957 |
| too close | 0.831 | 0.815 | 0.606 | 0.769 | 0.603 | 0.791 | 0.563 | 0.791 | 0.792 | 0.652 | 0.797 | 0.795 | 0.786 |
| aniso p50 | 23.57 | 24.99 | 25.51 | 23.80 | 23.52 | 24.99 | 25.37 | 25.08 | 25.07 | 25.08 | 23.08 | 23.64 | 26.73 |
| needle>0.9 | 0.161 | 0.166 | 0.177 | 0.161 | 0.161 | 0.168 | 0.176 | 0.166 | 0.168 | 0.168 | 0.160 | 0.163 | 0.170 |

## Gaussian-level difference to no_tuning (same token, same child)

| | scene_lr1e-2_tuning | scene_lr1e-2_geo_tuning | token_lr1e-4_tuning | token_lr1e-4_geo_tuning | scene_lr1e-2_token_lr1e-4_tuning | scene_lr1e-2_token_lr1e-4_geo_tuning | scene_lr1e-2_token_lr1e-4_vis_xyz_tuning | scene_lr1e-2_token_lr1e-4_vis_opac_tuning | scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning | scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning | scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning | scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| centre displacement median | 0.0195 | 0.1193 | 0.0145 | 0.0481 | 0.0199 | 0.1241 | 0.0216 | 0.0234 | 0.0230 | 0.0229 | 0.0226 | 0.0327 |
| centre displacement p90 | 0.3797 | 0.6469 | 0.2750 | 0.3511 | 0.4182 | 0.6868 | 0.4156 | 0.4152 | 0.4138 | 0.5998 | 0.5844 | 0.5325 |
| median displacement / scene extent | 0.0047 | 0.0289 | 0.0035 | 0.0116 | 0.0048 | 0.0300 | 0.0052 | 0.0057 | 0.0056 | 0.0055 | 0.0055 | 0.0079 |
| token centroid displacement median | 0.0215 | 0.1196 | 0.0153 | 0.0507 | 0.0233 | 0.1221 | 0.0248 | 0.0278 | 0.0326 | 0.0277 | 0.0268 | 0.0407 |
| mean abs opacity change | 0.0312 | 0.0694 | 0.0061 | 0.0075 | 0.0313 | 0.0742 | 0.0316 | 0.0337 | 0.0346 | 0.0337 | 0.0337 | 0.0356 |
| median abs log-scale change | 0.154 | 0.332 | 0.028 | 0.056 | 0.149 | 0.340 | 0.154 | 0.162 | 0.161 | 0.157 | 0.157 | 0.167 |
| mean abs RGB change | 0.0411 | 0.0599 | 0.0094 | 0.0097 | 0.0394 | 0.0602 | 0.0396 | 0.0416 | 0.0416 | 0.0410 | 0.0408 | 0.0437 |

## Re-decoding check (decoder_inputs.pt -> Gaussians, max abs diff)

- no_tuning: 0.000e+00
- scene_lr1e-2_tuning: 0.000e+00
- scene_lr1e-2_geo_tuning: 0.000e+00
- token_lr1e-4_tuning: 0.000e+00
- token_lr1e-4_geo_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_geo_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_vis_xyz_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_vis_opac_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_vis_opac_core_gate2.5_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_colmap_depth0.01_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_colmap_clean_depth0.01_tuning: 0.000e+00
- scene_lr1e-2_token_lr1e-4_vggt_omega_depth0.01_tuning: 0.000e+00

# 外推视角评测（同 6 输入，目标视角在 18 帧窗口之外）

## 窗口内插值（官方，距最近输入 1 帧）

| 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | 厚度p90(m) ↓ | F@5cm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pretrained_6view | 30.503 | 0.9270 | 0.1801 | 0.1744 | 0.2568 | 0.4453 |
| sft_latenttail_rgb | 31.187 | 0.9305 | 0.1760 | 0.1645 | 0.2323 | 0.4446 |
| sft_readout_rgb | 30.968 | 0.9292 | 0.1782 | 0.1741 | 0.2482 | 0.4443 |
| sft_readout_rgb_depth | 30.502 | 0.9269 | 0.1855 | 0.1031 | 0.2804 | 0.5568 |
| sft_readout_rgb_depth_thick | 29.845 | 0.9169 | 0.2058 | 0.1096 | 0.0710 | 0.4999 |
| sft_readout_rgb_depth_thick_sh1 | 29.852 | 0.9169 | 0.2056 | 0.1117 | 0.0717 | 0.4962 |
| sft_readout_rgb_depth_thick_posmlp | 29.789 | 0.9165 | 0.2075 | 0.1105 | 0.0718 | 0.4956 |
| sft_readout_rgb_depth_thick_posmlp_surfel_binar | 29.166 | 0.9069 | 0.2233 | 0.1104 | 0.0801 | 0.4767 |
| sft_readout_rgb_depth_surface3d | 30.013 | 0.9214 | 0.1973 | 0.1021 | 0.2331 | 0.5513 |
| sft_readout_rgb_depth_surface3d_posmlp | 30.211 | 0.9239 | 0.1923 | 0.0991 | 0.2241 | 0.5516 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp | 30.631 | 0.9270 | 0.1850 | 0.0710 | 0.1386 | 0.6261 |
| sft_readout_rgb_depth_surface3d_posmlp_ep20 | 30.295 | 0.9242 | 0.1925 | 0.0896 | 0.1997 | 0.5648 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20 | 30.690 | 0.9276 | 0.1837 | 0.0663 | 0.1358 | 0.6414 |

## 近外推（窗口外 3–12 帧）

| 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | 厚度p90(m) ↓ | F@5cm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pretrained_6view | 20.230 | 0.7950 | 0.3760 | 0.3112 | 0.3783 | 0.3341 |
| sft_latenttail_rgb | 20.464 | 0.7968 | 0.3720 | 0.3014 | 0.3548 | 0.3269 |
| sft_readout_rgb | 20.386 | 0.7960 | 0.3738 | 0.3199 | 0.3720 | 0.3278 |
| sft_readout_rgb_depth | 20.399 | 0.7971 | 0.3710 | 0.2245 | 0.4055 | 0.3949 |
| sft_readout_rgb_depth_thick | 19.884 | 0.7820 | 0.3983 | 0.2225 | 0.1298 | 0.3670 |
| sft_readout_rgb_depth_thick_sh1 | 19.884 | 0.7821 | 0.3984 | 0.2236 | 0.1306 | 0.3657 |
| sft_readout_rgb_depth_thick_posmlp | 19.882 | 0.7820 | 0.3980 | 0.2233 | 0.1312 | 0.3645 |
| sft_readout_rgb_depth_thick_posmlp_surfel_binar | 19.305 | 0.7735 | 0.4172 | 0.2138 | 0.1376 | 0.3493 |
| sft_readout_rgb_depth_surface3d | 19.678 | 0.7890 | 0.3941 | 0.1940 | 0.3219 | 0.4072 |
| sft_readout_rgb_depth_surface3d_posmlp | 19.809 | 0.7919 | 0.3884 | 0.1912 | 0.3138 | 0.4082 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp | 20.227 | 0.7967 | 0.3823 | 0.1557 | 0.2227 | 0.4658 |
| sft_readout_rgb_depth_surface3d_posmlp_ep20 | 19.789 | 0.7914 | 0.3917 | 0.1738 | 0.2919 | 0.4182 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20 | 20.302 | 0.7971 | 0.3814 | 0.1501 | 0.2231 | 0.4762 |

## 远外推（窗口外 18–36 帧）

| 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | AbsRel ↓ | 厚度p90(m) ↓ | F@5cm ↑ |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pretrained_6view | 15.836 | 0.7117 | 0.5086 | 0.3819 | 0.4719 | 0.2501 |
| sft_latenttail_rgb | 15.916 | 0.7128 | 0.5043 | 0.3823 | 0.4657 | 0.2435 |
| sft_readout_rgb | 15.919 | 0.7118 | 0.5063 | 0.3944 | 0.4771 | 0.2465 |
| sft_readout_rgb_depth | 16.056 | 0.7167 | 0.4986 | 0.2960 | 0.4889 | 0.2878 |
| sft_readout_rgb_depth_thick | 15.617 | 0.7025 | 0.5252 | 0.2915 | 0.2275 | 0.2801 |
| sft_readout_rgb_depth_thick_sh1 | 15.632 | 0.7026 | 0.5255 | 0.2924 | 0.2293 | 0.2785 |
| sft_readout_rgb_depth_thick_posmlp | 15.634 | 0.7028 | 0.5253 | 0.2937 | 0.2329 | 0.2764 |
| sft_readout_rgb_depth_thick_posmlp_surfel_binar | 15.275 | 0.6978 | 0.5396 | 0.2832 | 0.2248 | 0.2673 |
| sft_readout_rgb_depth_surface3d | 15.568 | 0.7091 | 0.5197 | 0.2603 | 0.3957 | 0.2997 |
| sft_readout_rgb_depth_surface3d_posmlp | 15.662 | 0.7119 | 0.5155 | 0.2633 | 0.4009 | 0.2961 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp | 15.863 | 0.7159 | 0.5135 | 0.2088 | 0.3291 | 0.3410 |
| sft_readout_rgb_depth_surface3d_posmlp_ep20 | 15.618 | 0.7112 | 0.5188 | 0.2474 | 0.3791 | 0.3059 |
| sft_latenttail_readout_rgb_depth_surface3d_posmlp_ep20 | 15.906 | 0.7161 | 0.5117 | 0.2053 | 0.3297 | 0.3479 |


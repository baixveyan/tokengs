# Same-scene comparison 7b6477cb95

One folder per method next to this file, named after the method: `<method>.ply` (true colour), `<method>_slotcolor.ply` (one colour per token), `cameras.ply` (input red, evaluation green), `<method>_minus_baseline_pretrained_6view.ply` (blue = Gaussian did not move, red = moved most).

- baseline_pretrained_6view: `baseline_pretrained_6view/`

## Metrics (this scene)

| | baseline_pretrained_6view |
|---|---|
| all PSNR | 29.477 |
| novel PSNR | 29.477 |
| d0 PSNR | - |
| d4+ PSNR | - |
| all SSIM | 0.9426 |
| all LPIPS | 0.1489 |
| HF ratio | - |
| edge L1 | - |
| SW | 0.789 |
| packing ratio | 392.1 |
| inter d | 0.408 |
| too close | 0.936 |
| aniso p50 | 26.72 |
| needle>0.9 | 0.335 |

## Re-decoding check (decoder_inputs.pt -> Gaussians, max abs diff)

- baseline_pretrained_6view: not saved

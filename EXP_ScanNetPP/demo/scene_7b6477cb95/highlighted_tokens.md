# 高亮 token 图例：scene_7b6477cb95

- 选法：在 `baseline_pretrained_6view` 中按 token 的不透明度总量（64 个子 Gaussian 的不透明度之和）取前一半（≥ 中位数 2.872，2048/4096 个），固定随机种子 0 随机抽 20 个；前 k 个就是 k 个 token 的文件，所以颜色在各文件里一致。
- 所有方法用同一批 token 编号；选中的 Gaussian 保留原位置、形状和不透明度，只换颜色。
- `_highlight{k}tokens.ply` 保留其他 Gaussian，但将其设为灰色、不透明度 ×0.1；`_only{k}tokens.ply` 完全删除其他 Gaussian，只显示选中的 token。
- Points 模式下可以按字段 `selected_token_rank` 着色（0…k−1 为选中，−1 为其余；only 文件没有 −1）。
- 存活 = 不透明度 > 1/255 的子 Gaussian 数（满分 64）；中心 = 按不透明度加权的平均位置；散开半径 = 按不透明度加权的到中心的均方根距离；中心位移 = 与参考方法同一 token 中心的距离。

## 文件

- `baseline_pretrained_6view`：`baseline_pretrained_6view/baseline_pretrained_6view_highlight10tokens.ply`、`baseline_pretrained_6view/baseline_pretrained_6view_only10tokens.ply`、`baseline_pretrained_6view/baseline_pretrained_6view_highlight20tokens.ply`、`baseline_pretrained_6view/baseline_pretrained_6view_only20tokens.ply`

## 图例与各方法的 token 统计

| 序号 | token | 颜色 | 在 10 个里 | baseline_pretrained_6view 存活 / 不透明度总量 / 散开半径 / 中心位移 |
|---|---|---|---|---|
| 0 | 3671 | red `#e6194b` | 是 | 50 / 4.04 / 0.120 / 0.000 |
| 1 | 867 | green `#3cb44b` | 是 | 51 / 4.89 / 0.114 / 0.000 |
| 2 | 755 | yellow `#ffe119` | 是 | 51 / 4.53 / 0.100 / 0.000 |
| 3 | 3786 | blue `#4363d8` | 是 | 49 / 3.59 / 0.117 / 0.000 |
| 4 | 1739 | orange `#f58231` | 是 | 50 / 4.19 / 0.099 / 0.000 |
| 5 | 2205 | purple `#911eb4` | 是 | 51 / 6.16 / 0.121 / 0.000 |
| 6 | 1989 | cyan `#42d4f4` | 是 | 51 / 3.43 / 0.099 / 0.000 |
| 7 | 3938 | magenta `#f032e6` | 是 | 50 / 4.65 / 0.085 / 0.000 |
| 8 | 2325 | lime `#bfef45` | 是 | 45 / 4.27 / 0.085 / 0.000 |
| 9 | 2423 | pink `#fabed4` | 是 | 51 / 5.04 / 0.106 / 0.000 |
| 10 | 491 | teal `#469990` | 否 | 50 / 4.22 / 0.091 / 0.000 |
| 11 | 3196 | lavender `#dcbeff` | 否 | 53 / 8.10 / 0.114 / 0.000 |
| 12 | 2324 | brown `#9a6324` | 否 | 50 / 4.62 / 0.087 / 0.000 |
| 13 | 43 | beige `#fffac8` | 否 | 53 / 5.83 / 0.136 / 0.000 |
| 14 | 1921 | maroon `#800000` | 否 | 52 / 4.55 / 0.093 / 0.000 |
| 15 | 210 | mint `#aaffc3` | 否 | 53 / 8.35 / 0.081 / 0.000 |
| 16 | 3357 | olive `#808000` | 否 | 39 / 3.25 / 0.103 / 0.000 |
| 17 | 1722 | apricot `#ffd8b1` | 否 | 50 / 3.16 / 0.136 / 0.000 |
| 18 | 2797 | navy `#000075` | 否 | 46 / 3.02 / 0.111 / 0.000 |
| 19 | 118 | black `#000000` | 否 | 51 / 3.20 / 0.116 / 0.000 |

# 末层组外 28 个通道的单独和联合阻断

沿用 36 张 dSprites 图像和六个冻结权重。在开发条件对原末层名单以外的 28 个通道排序，冻结后对留出条件逐个阻断，同时遍历全部 378 种双通道组合。在上游四通道自由和已阻断的两个背景下，测量最终三阶得分差及三组全选的端点得分。

## 从仓库根目录运行

在 Python 3.12 环境执行以下命令。

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/outside_channels_20261126/reproduce.py --device cpu
```

默认写入 `local_output/outside-channels-时间戳/`，用 `--out` 可指定新目录。CUDA 选项为 `--device cuda`。入口重选开发名单、双背景阻断、汇总、数值比对及绘图。计算所需的权重、面板、旧规则和旧位置名单都在仓库内。[冻结设计](DESIGN.md) · [包含全部 28 个编号的实验报告](../../reports/dSprites_末层组外28通道的承接与抵消_2026-11-26.md)

## 从哪份结果取数

- [channel_scores.csv](../../runs/dsprites_outside_channels_20261126/channel_scores.csv) 每模型每强度 28 行。`channel` 是末层编号，`dev_rank` 是开发名次，`projection_F` 是对上游四通道总作用的有符号投影，`share_outside` 是该通道占组外投影的份额。`output_change_rms` 和 `output_remaining_ratio` 分别给出阻断使整个三阶输出移动多少 logit 及阻断前后 RMS 比。
- [pair_scores.csv](../../runs/dsprites_outside_channels_20261126/pair_scores.csv) 收录每一对的同号、异号次数与抵消比例。`route_` 开头的列对应经该对传递的上游四通道作用，`output_` 开头的列对应完整三阶输出。[candidate_cases.csv](../../runs/dsprites_outside_channels_20261126/candidate_cases.csv) 保留两对开发端候选在每条留出记录中的单阻断、双阻断与端点得分。
- [set_scores.csv](../../runs/dsprites_outside_channels_20261126/set_scores.csv) 列出冻结的 20 个集合。[selection.json](../../runs/dsprites_outside_channels_20261126/selection.json) 记录各模型的开发排名、两对候选、原四通道及两组随机排列。[replay_comparison.json](../../runs/dsprites_outside_channels_20261126/replay_comparison.json) 给出逐例复跑误差。

## 读取某一例的原始得分

每个 `model_模型种子.npz` 有 144 条记录。`keys` 的四列是种子、强度乘 100、配对编号、三组掩码。`singles`、`pairs` 和 `groups` 的后两轴都是上游背景（0 自然，1 已阻断）和读数（0 三阶差分，1 三组全选端点）。在仓库根目录可以这样查模型 291 的配对 63。

```python
from pathlib import Path
import numpy as np

folder = Path("reproduction/final/runs/dsprites_outside_channels_20261126")
with np.load(folder / "model_202609291.npz", allow_pickle=False) as data:
    key = [202609291, 100, 63, 7]
    row = np.flatnonzero(np.all(data["keys"] == key, axis=1)).item()
    pair = np.flatnonzero(np.all(data["channel_pairs"] == [17, 27], axis=1)).item()
    base = data["group_names"].tolist().index("base")
    before = data["groups"][row, base, 0, 0]
    after = data["pairs"][row, pair, 0, 0]
    print(f"三阶得分 {before:.4f} → {after:.4f}")
```

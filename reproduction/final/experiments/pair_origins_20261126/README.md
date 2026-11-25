# 模型 291 的两对末层通道如何获得同向和反向作用

固定 dSprites 的 36 张图和模型 202609291，追踪末层 23、30 以及 17、27 的上游输入。每种强度在 18 个留出配对上评价 4 种三组选择。实验从上轮的 64 点开关设置、原四通道和模型权重开始。

## 从仓库根目录复跑

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/pair_origins_20261126/reproduce.py --device cpu
```

使用 Python 3.12。CUDA 可传 `--device cuda`。默认生成 `local_output/pair-origins-时间戳/`，`--out` 可指定新的目录。

1. `select_paths.py` 在开发端实测 block2 输出通道、block3 第一处 ReLU 的激活恢复，选择两层组合、空间坐标及符号阈值。
2. `measure.py` 先封存上游方向判定，再对每条留出记录实际运行 66 种状态或干预，核对上一轮的单通道、双通道和整网得分。
3. `local_trace.py` 在各例定位开闭发生变化的位置。对符号翻转且前后成员贡献绝对值均达 0.001 logit 的记录，逐点核验末层恢复原符号所需的最少位置数。
4. `coverage.py` 在开发端用固定末层开关的上游卷积斜率给 32 个输入通道排序。留出端逐预算独立重算，记录开关固定和自由重算两种读出。
5. `score.py` 汇总方向、开关、通道及最终输出。`compare.py` 比较两次复跑的名单、坐标、阈值、真实输出与逐例表。`figure.py` 从实测记录生成图。

## 读取结果

[完整实验报告](../../reports/dSprites_模型291两对末层通道的上游与开关_2026-11-26.md) · [主方案](DESIGN.md) · [逐例开关方案](LOCAL_DESIGN.md) · [上游覆盖对照](COVERAGE.md)

- [selection.json](../../runs/dsprites_pair_origins_20261126/selection.json) 存放 H（block2 输出）、K（block3 第一处 ReLU）、两处固定坐标、随机对照和方向阈值。[coverage_order.json](../../runs/dsprites_pair_origins_20261126/coverage_order.json) 是补充对照的开发排名。
- [arm_scores.csv](../../runs/dsprites_pair_origins_20261126/arm_scores.csv) 中 `total_change_rms`、`endpoint_change_rms` 是相对八态原背景的三阶、端点得分改变量 RMS。`y_from_h_rms` 从 H 已阻断的状态起算。`h_signs_restored` 是 H 翻转后回到原符号的案例数。`extra_flips` 是此前保持原方向、本次又翻转的案例数。
- [case_effects.csv](../../runs/dsprites_pair_origins_20261126/case_effects.csv) 可搜索具体配对和掩码。`a,b` 是两个末层通道对完整三阶得分的贡献。`t_a,t_b` 只计能追溯到原四通道 U 的部分。`y` 是整网三阶得分，`endpoint` 是三组全选时供体形状与受体形状的得分差。
- [local_coordinates.csv](../../runs/dsprites_pair_origins_20261126/local_coordinates.csv) 的坐标依次为通道、行、列，均从 0 编号。`original_bits`、`h_bits` 从低位到高位分别记录空组、三个单组、三个双组、三组全选的开闭。以源掩码 13 为例，八位依次对应子集掩码 0、1、4、8、5、9、12、13。`original_z,h_z` 对应改变最大的角点，`peak_corner` 是它的八态索引。
- [responses.npz](../../runs/dsprites_pair_origins_20261126/responses.npz) 保存所有 66 种条件的原始读数。轴为留出记录、末层通道对、干预名称、测量字段。`keys` 的四列依次为模型、强度乘 100、配对、源掩码。`arms` 和 `columns` 给出第三及第四轴的名字。[跨设备核对](../../runs/dsprites_pair_origins_20261126/replay_comparison.json)

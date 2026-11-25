# 追踪两处开关阻断后留下的三阶响应

从 36 张 dSprites 图、六个冻结权重、11 月 1 日的拟合与 11 月 25 日的八状态位置名单开始，重算最后一个残差块的两条支路、两处 ReLU、各通道及读出位置。完整的六个模型共 864 条留出记录。全强度和半强度各 432 条。

## 从仓库根目录复跑

使用 Python 3.12，在克隆或解压的仓库根目录运行以下两行。

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/remaining_routes_20261126/reproduce.py --device cpu
```

输出目录为 `local_output/remaining-routes-时间戳/`。用 `--out` 可指定待创建的目录。CUDA 入口为 `--device cuda`。CPU 逐通道重跑卷积，CUDA 用分组卷积同时计算候选连接。两种入口均另用 CPU 替换单条连接的卷积消息，并比较归档结果。核对容差为绝对误差 1e-8。

## 从开发名单到输出

1. `experiment.py select` 只在开发图像条件上排序，每模型、每处各选 4 个主分析通道，保存两个匹配预算的随机排列和 SHA-256 记录。
2. `experiment.py measure` 读取已封存名单，在每个三组的八个状态实施同一干预，测量全强度、半强度的输出。每一条首先与上轮两处各锁 64 个开关的结果对齐。整通道干预将各状态的激活图换为对应空组的激活图。
3. `summarize.py` 按模型求均方根、带符号投影、抵消率和各预算的剩余比。`audit_links.py` 对 110,592 个记录级通道连接效应做独立前向核对，其中 3,456 次还在 BN 前直接替换卷积输出通道。
4. `verify.py` 比较开发排名、逐例表、逐位置表、全部通道记录以及六模型汇总。`figure.py` 绘制支路、通道预算及模型 291 的通道连接图。

## 对应文件

- [设计与指标](DESIGN.md) · [完整结果报告](../../reports/dSprites_两层开关阻断后的剩余传递路径_2026-11-26.md)
- [开发端通道顺序](../../runs/dsprites_remaining_routes_20261126/rankings.json) · [两种强度汇总](../../runs/dsprites_remaining_routes_20261126/summary.json)
- [逐例支路与开关](../../runs/dsprites_remaining_routes_20261126/cases.csv) · [通道组干预](../../runs/dsprites_remaining_routes_20261126/group_interventions.csv) · [各模型通道对](../../runs/dsprites_remaining_routes_20261126/link_scores.csv)
- [单连接核验](../../runs/dsprites_remaining_routes_20261126/isolated_links.json) · [CPU 与 GPU 独立复跑](../../runs/dsprites_remaining_routes_20261126/replay_comparison.json)

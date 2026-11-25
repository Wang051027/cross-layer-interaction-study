# 复跑三组通道的开关预测与阻断实验

[实验结果](../../reports/dSprites_三组通道的开关预测与八状态阻断_2026-11-25.md) · [主实验协议](DESIGN.md) · [八状态阻断](EIGHT_CORNERS.md) · [逐例复跑核对](../../runs/dsprites_gate_forecast_20261125/replay_comparison.json)

从公开仓库根目录运行下面两行命令。使用 Python 3.12，输入为仓库内 36 张 64 × 64 图像、6 个权重和 6 份已冻结的通道名单。

```text
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/gate_forecast_20261125/reproduce.py --device cpu
```

在原研究项目根目录，入口为 `python -B -X utf8 experiments/gate_forecast_20261125/reproduce.py --assets . --device cpu`。两种入口都在 `local_output/final-时间戳/` 下保存重算数据。可用 `--out 路径` 指定新的输出目录，用 `--device cuda` 切换主实验设备。独立梯度和选点核对在 CPU 上运行。

入口先复跑 11 月 10 日的通道覆盖实验，再生成 11 月 25 日的开关与干预预测、封存名单、执行两种强度的真实干预并生成主图。打开输出目录的 `comparison.json` 查看 `status`、逐表最大差异，以及两次运行的开关和名单比对。公开研究仓库同时保留了 11 月 1 日判别实验的入口。[公开仓库](https://github.com/Wang051027/cross-layer-interaction-study)

| 文件 | 对应步骤 |
| --- | --- |
| [gates.py](gates.py) | 七状态外推、开关投影、四通道定向选点、两层阻断 |
| [run.py](run.py) | 预测与测量两个进程，检查原始图像、模型、名单及运行指纹 |
| [eight_corners.py](eight_corners.py) | 对相同位置执行八状态阻断，计算三阶差分 |
| [analyze.py](analyze.py) | 逐模型的翻转精确率、召回率、输出误差与干预强度 |
| [audit_predictions.py](audit_predictions.py) | 用完整卷积、子集差分和自动微分独立检查开关、点号和干预预测 |
| [verify_runs.py](verify_runs.py) | 比较 CPU 和 CUDA 表格、布尔开关、位置编号及上一轮结果 |
| [figure.py](figure.py) | 根据逐模型表重绘三幅对照 |
| [reproduce.py](reproduce.py) | 连续完成两轮复现和逐例核对 |

逐例记录用 `seed, strength, case, mask` 对齐。干预表再加 `arm, layer`。开关和位置保存在 `forecast_sites.npz` 中，`keys` 里的强度以 100 或 50 记录。`sites[记录, 名单, 层]` 的三份名单依次为定向、随机一、随机二，两层为 block3 前后两处 ReLU。编号按 `channel * 1024 + y * 32 + x` 展平，数组维度是 32 通道 × 32 × 32。开关用小端位序打包，`np.unpackbits(array, axis=-1, bitorder="little")` 可还原每个位置。用相同记录索引在 `observed_gates.npz` 读取真实开关。

# dSprites 训练与评价记录

2026 年 9 月 29 日的实验包含 6 个模型和 24 种评价设置。独立审计核对了 38,016 个中介选择集合。

| 文件 | 内容 |
| --- | --- |
| [model_freeze.json](model_freeze.json) | 六个模型的训练种子、检查点和哈希 |
| [descriptive_analysis.json](descriptive_analysis.json) | 各选择方法的恢复、移植和逐模型统计 |
| [independent_audit.json](independent_audit.json) | 评价响应与中介选择的核验结果 |
| [independent_data_audit.json](independent_data_audit.json) | 像素、标签和因素划分的核验结果 |

权重保存在[六个模型的种子目录](../../reproduction/held_condition/runs/dsprites_20260929)。在仓库根目录运行 `python -B run_dsprites_demo.py`，可重算三个形状类别各一条样本的指标。

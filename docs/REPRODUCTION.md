# 复现范围、数据依赖与已知问题

## 可以直接运行

在仓库根目录运行 README 中的两个示例、`check_repository.py` 和 `run_checks.py`。这些入口仅访问本仓库；不会训练、下载数据或写入原研究目录。

`run_dsprites_demo.py` 取首个模型、主评价面板中每一类别的第一条记录，共三条。选择没有使用成绩，示例仍不代表六模型汇总。它以独立矩阵运算重算两种目标、44 种方案的逐图指标，并检查四阶段分解之和。

`run_checks.py` 覆盖已知加性／阈值／高阶例子、Walsh 与直接 Hadamard 运算、完美／无效／有害中介控制，以及三类 64×64 随机网络上的真实 `forward` 钩子核验。随机网络只用于实现检查，不能提供任务机制证据。

## 完整历史实验需要什么

当前仓库是一份**研究源码与可运行示例**，不是全部原始证据档案。`source/` 对应原研究项目根目录，保留了代码间的相对关系；历史入口仍需要 `data/`、`runs/`、`audit/` 内的原始文件。

完整 dSprites 增量运行所需条件：

- 由 `prepare_dsprites.py` 获取的官方固定版本，以及本研究受控划分；解压后的图像数组约 3.02 GB。
- 三份旧交付清单及其所有被核验文件：`runs/cross_layer_20260928/delivery_manifest.json`、`runs/adaptive_mediator_20260928/delivery_manifest.json`、`audit/joint_response_manifest_20260925.json`。
- 原 Fashion 模型、验证图和选点记录，供 `fashion_portability()` 在同一模型上比较新旧公式。
- 与原协议匹配的**冻结原件**。注释副本和公开路径替换会改变字节哈希，不能把本仓库的阅读源码覆盖到原冻结目录后继续声称原哈希不变。

这些历史依赖未包含在仓库中；缺少它们时，完整训练／评价入口会明确失败。不要删除完整性检查来伪装复现成功。若开发独立的新实验，应另建协议与运行目录，并明确它是新运行。

## 最新三组规则实验（2026 年 11 月 1 日）

最新数值实现见 [LATEST_EXPERIMENT.md](LATEST_EXPERIMENT.md)。公开源码保持原运行的相对目录关系。完整执行还需要 `source/runs/dsprites_20260929/` 中六个模型及交付清单、`source/runs/dsprites_shape_donor_20261010/` 中固定面板及交付清单，以及协议列出的冻结源码。报告中的事后诊断另依赖 `source/runs/dsprites_layer_transfer_20261013/` 的已保存中间场。

`results/rule_discrimination/` 提供原设计、协议和统计副本，供阅读核对。公开源码含注释和路径调整，其哈希与原运行可能不同。原协议检查应使用完整冻结原件；新运行应在独立目录重新锁定源码、输入和预测。

具备上述依赖时，在 `source/runs/` 下创建新运行目录并放入 `design.json`。下面每个入口均传同一 `--run-dir` 绝对路径，依次执行：

```text
run_dsprites_triple_rule.py --stage predict
run_dsprites_triple_rule.py --stage measure
audit_dsprites_triple_rule.py
analyze_dsprites_triple_rule.py
report_dsprites_triple_rule.py --stage report
report_dsprites_triple_rule.py --stage seal
report_dsprites_triple_rule.py --stage verify
```

入口位于 `source/scripts/`，使用 `python -B -X utf8` 调用。`predict` 包含协议建立、开发拟合和预测冻结。完整历史数组没有收入本仓库，以上是依赖齐备时的阶段顺序。

## 早期 dSprites 执行顺序（2026 年 9 月 29 日）

下面是具备上述原档案时的阶段顺序说明，**不是刚克隆本仓库就能从头执行的一键命令**：

```text
prepare_dsprites.py --stage fetch
prepare_dsprites.py --stage extract
prepare_dsprites.py --stage prepare
dsprites_teacher.py --stage init
dsprites_workflow.py --stage train-safe
run_dsprites_mechanism.py --stage init
run_dsprites_mechanism.py --stage develop
run_dsprites_mechanism.py --stage freeze
run_dsprites_mechanism.py --stage select
run_dsprites_mechanism.py --stage seal
run_dsprites_mechanism.py --stage evaluate
run_dsprites_mechanism.py --stage summarize
run_dsprites_mechanism.py --stage verify
audit_dsprites_results.py
audit_dsprites_data.py
summarize_dsprites_factors.py
tabulate_dsprites_results.py
```

不要在已有结果目录重复初始化。报告与打包脚本可能写研究状态，只读核验入口与报告生成器的用途不同。

## 环境与已知问题

- 发布核验使用 Python 3.12.14 和当前本机依赖，实际版本见 `checks/release_checks.json`。`requirements.txt` 给出安装范围，不承诺范围内每个版本已测试；历史训练环境为 PyTorch 2.12.1+cu132，训练 float32、干预 float64、关闭 TF32。
- 首版 `dsprites_teacher.py` 最后一条日志的 `utc` 键重名。六个检查点、训练历史和模型冻结已在日志异常前保存。保留该历史源码；`dsprites_workflow.py --stage train-safe` 仅重命名附加日志字段，不改变训练计算。记录见 `results/dsprites/training_logging_incident.json`。
- 早期报告绘图使用 Windows 中文字体路径；其他系统需要配置可用字体。数据维度的核心迁移检查不依赖这些字体。
- 六个历史报告／打包脚本的个人路径改为相对占位路径；相关可视化和文献私有档案未随仓库发布，不能直接执行这些档案打包入口。
- dSprites 的核心投影和修补按张量尺寸适配。早期实验固定四个源点与 64 候选，后续实验使用四个源通道组。各阶段仍使用三类输出、三残差块及各自固定的候选规则。

`results/dsprites/` 保存的是历史结果副本；核验报告中的源文件哈希指原运行，不能解释成对刚克隆仓库的一次新审计。完整数据未收入 Git，已保存的 PASS 也不等于读者独立重跑了全部实验。

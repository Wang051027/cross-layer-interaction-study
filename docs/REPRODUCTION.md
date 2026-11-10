# 复现范围、数据依赖与已知问题

## 可以直接运行

在仓库根目录运行 README 中的两个示例、`check_repository.py` 和 `run_checks.py`。这些入口仅访问本仓库；不会训练、下载数据或写入原研究目录。

`run_dsprites_demo.py` 取首个模型、主评价面板中每一类别的第一条记录，共三条。选择没有使用成绩，示例仍不代表六模型汇总。它以独立矩阵运算重算两种目标、44 种方案的逐图指标，并检查四阶段分解之和。

`run_checks.py` 覆盖已知加性／阈值／高阶例子、Walsh 与直接 Hadamard 运算、完美／无效／有害中介控制，以及三类 64×64 随机网络上的真实 `forward` 钩子核验。随机网络只用于实现检查，不能提供任务机制证据。

## 从空目录复跑最新三组规则实验

在克隆或解压后的仓库根目录，使用 Python 3.12 虚拟环境。安装依赖后，实验本身可以离线运行：

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduce_latest.py
```

自动选择 CUDA 或 CPU。要指定设备，传 `--device cpu` 或 `--device cuda`。可以用 `--output` 提供一个尚未创建的输出目录。默认在 `local_output/` 中为每次复跑创建新目录，避免把旧缓存误作本次计算。

1. 入口校验发布文件，将源码和输入复制到独立工作目录，核对原协议里的七份代码指纹和七个输入指纹。其中 `residual_teacher.py` 的阅读副本加过注释，入口会使用随包提供的原始冻结版本。
2. 从 `design.json` 开始，重新划分条件、拟合六组参数并封存，随后生成预测和阻断名单。参考参数、既有预测和答案表均不作为拟合或预测输入。
3. 测量全部八个子集角上的阻断和只保留，独立核验真实模型前向，计算两种强度下的统计、事后诊断、图表和报告。程序构建并校验新的交付清单。
4. 与原记录核对划分、六组参数、四项主判定、逐案例及阻断结果和模型级统计。浮点比较的相对和绝对容差分别为 `1e-8`、`1e-9`。计数与标识精确核验。参数、预测和实测文件的 SHA-256 匹配数另行记录。

`comparison.json` 在本次输出目录顶层。报告位于 `experiment/source/reports/`，逐例结果位于 `experiment/source/runs/dsprites_rule_discrimination_20261031/tables/`。预测和实测数组位于同一运行目录的 `predictions/` 和 `measurements/`。

2026 年 11 月 10 日的 Windows 复跑使用 Python 3.12.14。CUDA 环境重新生成的 6 个拟合、12 个预测和 12 个实测文件全部与原件逐字节相同。全新 CPU 虚拟环境的数值比较通过，四项判定一致。两次均核对 864 条案例、4,320 条阻断与只保留记录和 516 行模型级统计。[验证记录](../checks/held_condition_replay.json)

[复跑输入](../reproduction/held_condition/runs) 含六个 `teacher.pt`、36 张图的 `panel/data.npz`、原始哈希及 24 个历史参考数组文件。10 月 10 日的十二个参考文件用于比对模型原始 logit，10 月 13 日的十二个文件用于事后梯度投影和中间响应范数。9 月 29 日的交付清单使用公开摘录，保留原清单哈希及所需条目的原始哈希。模型冻结记录、两轮参考数组与清单、[本实验的冻结记录](../reproduction/held_condition/reference) 均保留原文件字节。只读取这些输入即可从冻结模型重新运行整个判别实验。

## 9 月训练及更早历史实验的额外依赖

最新实验的复跑包只使用固定的 36 张图像。`source/` 对应原研究项目根目录，保留了代码间的相对关系；历史入口仍需要 `data/`、`runs/`、`audit/` 内的原始文件。

完整 dSprites 增量运行所需条件：

- 由 `prepare_dsprites.py` 获取的官方固定版本，以及本研究受控划分；解压后的图像数组约 3.02 GB。
- 三份旧交付清单及其所有被核验文件：`runs/cross_layer_20260928/delivery_manifest.json`、`runs/adaptive_mediator_20260928/delivery_manifest.json`、`audit/joint_response_manifest_20260925.json`。
- 原 Fashion 模型、验证图和选点记录，供 `fashion_portability()` 在同一模型上比较新旧公式。
- 与原协议匹配的**冻结原件**。注释副本和公开路径替换会改变字节哈希，不能把本仓库的阅读源码覆盖到原冻结目录后继续声称原哈希不变。

这些历史依赖未包含在仓库中；缺少它们时，完整训练／评价入口会明确失败。不要删除完整性检查来伪装复现成功。若开发独立的新实验，应另建协议与运行目录，并明确它是新运行。

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

# 神经元组合效应如何跨层传递？

在残差卷积网络中，同时改变三个通道组会产生多大的三阶响应？实验追踪这部分响应经过的 ReLU 开关，并测量它对两种形状得分差的影响。

在 36 张 dSprites 图像和六个冻结模型上，用空组、单组和双组的七种状态提前预测三组共同替换时的响应。11 月 25 日，在留出的大小与角度条件上，两处下游 ReLU 翻转的 F1 为 **0.907 和 0.816**。同一个四通道中间预测，用固定梯度估计最终三阶输出时误差比为 **1.0031**，用预测开关为 **0.4355**，重算下游开关为 **0.4175**。

将每层预选的 64 个位置锁回空组开关，三组端点的得分改变量 RMS 为 **4.187 logit**。在全部八个子集状态锁相同位置，三阶改变量 RMS 为 **0.661 logit**，阻断后六模型的三阶 RMS 平均剩余 **91.14%**。[完整实验报告](reproduction/final/reports/dSprites_三组通道的开关预测与八状态阻断_2026-11-25.md) · [逐模型结果](reproduction/final/runs/dsprites_gate_forecast_20261125/teacher_scores.csv)

11 月 26 日继续追踪这 91.14% 经哪里传到输出。只让最后一个残差块的卷积分支变化，保留上轮余量 RMS 的 **94.20%**，只让捷径变化则保留 **7.05%**。在开发条件为两处 ReLU 各选定 4 个通道，在留出条件将这些通道的激活恢复到空组值后，余量降至 **47.25%**（原始三阶 RMS 的 **42.97%**）。两个同预算随机对照留下 **76.42% 和 81.72%**。第一处四通道作用的 **72.63%**（有符号投影）经预选末层四通道以外的通道表达。[支路、通道连接与抵消报告](reproduction/final/reports/dSprites_两层开关阻断后的剩余传递路径_2026-11-26.md) · [通道连接表](reproduction/final/runs/dsprites_remaining_routes_20261126/link_scores.csv)

将这 72.63% 逐通道拆开，在开发条件排名前 4、8、14 的组外通道分别承接组外投影的 **36.01%、61.61%、83.17%**。只阻断新选的末层四通道，全强度最终三阶输出 RMS 剩 **82.37%**。六个模型各自测完 28 个单通道与全部 378 个双通道组合。模型 291 的一个留出案例中，通道 17 和 27 的得分贡献分别为 −0.2235 和 +0.2374 logit，合计仅 +0.0139。[完整通道编号与联合阻断报告](reproduction/final/reports/dSprites_末层组外28通道的承接与抵消_2026-11-26.md) · [逐例候选通道对](reproduction/final/runs/dsprites_outside_channels_20261126/candidate_cases.csv)

在模型 291 的 72 条全强度留出记录中，末层 23、30 有 66 条对三阶输出同向贡献，17、27 有 38 条。阻断第二残差块 29、30 后，23、30 的符号关系翻转 14 次，在第三块第一处 ReLU 恢复 13、16 号激活后，这 14 次均回到原方向。对 17、27，阻断第二块 27、30 引起 23 次翻转，恢复第三块 16、20 号挽回 11 次。两种上游阻断分别使三组全选时的形状得分差移动 **1.464 和 1.894 logit RMS**。在一个留出案例中，仅关闭末层 27 号的一个位置就使两条贡献恢复同向，得分移动 **+0.00743 logit**。[模型 291 上游机制报告](reproduction/final/reports/dSprites_模型291两对末层通道的上游与开关_2026-11-26.md) · [逐例开关坐标](reproduction/final/runs/dsprites_pair_origins_20261126/local_coordinates.csv)

## 复现模型 291 的上游通道与开关干预

从原模型和 36 张图重新筛选上游通道，测量留出条件的方向与得分，再核对逐位置干预及通道覆盖对照。[复现说明与结果字段](reproduction/final/experiments/pair_origins_20261126/README.md)

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/pair_origins_20261126/reproduce.py --device cpu
```

## 复现末层组外通道阻断

在 Python 3.12 环境从仓库根目录重选名单、重算两种强度的 864 条留出记录并核对各通道的作用。[复现说明与原始数组读取示例](reproduction/final/experiments/outside_channels_20261126/README.md)

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/outside_channels_20261126/reproduce.py --device cpu
```

## 复现剩余效应的路径干预

在 Python 3.12 环境从仓库根目录运行以下命令。程序从冻结权重和上轮 64 点名单重选开发端通道，重算 864 条留出记录、两组随机对照及单连接干预。[复现说明](reproduction/final/experiments/remaining_routes_20261126/README.md) · [跨设备复跑记录](reproduction/final/runs/dsprites_remaining_routes_20261126/replay_comparison.json)

```bash
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/remaining_routes_20261126/reproduce.py --device cpu
```

## 复现开关预测与八状态阻断

在 Python 3.12 环境从仓库根目录运行以下命令。入口复跑 11 月 10 日及 11 月 25 日两轮实验，核对位置编号、布尔开关、端点得分和八状态差分。输出写入 `local_output/final-时间戳/`。[复现说明](reproduction/final/experiments/gate_forecast_20261125/README.md) · [CPU 与 CUDA 核对记录](reproduction/final/runs/dsprites_gate_forecast_20261125/replay_comparison.json) · [11 月 1 日判别实验协议](docs/LATEST_EXPERIMENT.md)

```text
python -m pip install -r reproduction/held_condition/requirements.txt
python -B -X utf8 reproduction/final/experiments/gate_forecast_20261125/reproduce.py --device cpu
```

## 怎样把“层与层的关系”变成实验

早期实验在 Fashion-MNIST 上用四个空间位置建立交互测量方法；随后用 dSprites 的方形、椭圆、心形做受控配对。六个 dSprites 模型分别初始化，架构相同，各含三个残差块。它们在 24,576 张图上训练，在 3,072 张图的受控验证集上三分类准确率均为 100%。以下实验冻结权重和归一化统计量，测量内部干预怎样改变得分。

1. **替换前层通道。** 取一张受体图和一张生成大小、角度、位置相同的供体图，例如方形和椭圆。把受体第一残差块（block1）的 16 个通道固定分成四组，每组四个。全强度时把选中组的整张激活图换成供体值，半强度时换成两者的中点。模型后半段照常运行。读出是“供体类得分减受体类得分”（logit 差）；它能测量偏好移动，即使预测类别没有翻转。
2. **算出三组共同变化多出的部分。** 对任意三组 A、B、C，先测空组、三个单组和三个双组。若只存在单组和双组效应，三组响应应为 `Q = X_AB + X_AC + X_BC − X_A − X_B − X_C + X_空组`。实测 `X_ABC − Q` 就是**三阶响应**。对 block2 的整层激活和最终 logit 差各算一次；二者回答不同问题。
3. **先预测，再干预。** 在开发条件上，每个模型选定 block2 第一处 ReLU 的四个通道，以及 block2 输出的四个通道。用 ReLU 是否跨过零点和真实卷积权重预测中间层修正，再用空组状态的输出梯度估计它能传到输出多少。冻结名单和留出预测后，测量真实三组响应。随后在三组的全部八个子集状态上分别做两种操作：**阻断**这四个输出通道（恢复到空组值），或**只保留**它们变化（其余通道固定为空组值）。每种操作都重新运行 block3、池化和分类头，得到实际输出效应。

## 留出条件上的预测结果

固定面板有 36 张原生 64×64 图像：6 种大小 × 2 个角度 × 3 种形状。这里匹配的是生成参数，不同形状占的像素面积仍可能不同。按大小和角度把 12 个条件交错分成开发、留出各 6 个条件，各含 18 张图。每个条件只用方形→椭圆、椭圆→心形、心形→方形三个方向。每种强度共有 **6 个模型 × 18 个留出配对 × 4 种三组组合 = 432 条记录**。这些记录共享图像和模型；本轮留出的是拟合条件，图像面板本身在前期已经探索过。

以下为 2026 年 11 月 1 日的结果。**误差比 = 预测误差的均方根（RMS）÷ 实测目标的 RMS**，先在每个模型内计算，再对六个模型等权平均。越低越好；预测三阶作用为零时，比值是 1。事前规定：全强度均值 ≤ 0.95，且至少 4/6 个模型低于 1，单项才通过；完整机制要求四项都通过。半强度使用同一规则，只作补充检验。

| 预测对象 | 全强度误差比（主检验） | 低于 1 的模型 | 半强度误差比 | 主检验 |
| --- | ---: | ---: | ---: | --- |
| block2 全部 32 个通道的三阶响应 | **0.8840** | 6/6 | 0.8656 | 通过 |
| 最终输出的三阶差分 | 1.0031 | 5/6 | 0.9868 | 未通过 |
| 阻断选中四通道实际移除的输出作用 | 1.0435 | 2/6 | 0.9753 | 未通过 |
| 只保留这四通道实际重现的输出作用 | 0.9693 | 4/6 | 0.8520 | 未通过 |

这四个通道确实影响输出：阻断它们实际移除的作用，按模型取 RMS 后平均为 **1.990 logit**；同预算的两组随机通道分别为 **1.207、1.085 logit**。难点在于预报这份作用：规则给“阻断移除”和“只保留重现”预测了相同的带符号贡献，实测却在 **124/432** 条全强度记录中符号相反。

另一个简单对照，只按开发条件中的“形状转换 × 三组编号”取输出三阶均值，误差比为 **0.6185**，优于当前规则的 1.0031。输出差异中有可预测的部分；本轮的四通道选择和固定梯度读出没有充分捕捉它。

[冻结设计与判据](results/rule_discrimination/design.json) · [未取整的结果](results/rule_discrimination/analysis.json) · [逐例干预记录](reproduction/held_condition/reference/tables/patches.csv) · [阻断与只保留的事后对照](reproduction/held_condition/reference/posthoc.json)

## 前面各阶段发现了什么

早期实验替换单个空间位置，后续实验替换整张通道激活图。

| 阶段 | 做了什么 | 观察到的结果 |
| --- | --- | --- |
| 9/28 · Fashion-MNIST | 在 64 张测试图上改变前层四个位置，再恢复或移植 16 个固定中介位置。 | 恢复后仍剩原交互 RMS 的 **87.73%**；仅移植这些位置的重建误差为 **89.55%**。少量固定位置不足以解释大部分交互。[原报告](source/reports/跨层交互_首轮结果_2026-09-28.md) |
| 9/29 · dSprites 标量干预 | 在三分类模型上比较全局固定、按类别固定、逐图选择 16 个中介位置。 | 三者恢复后分别剩 **71.20%、62.88%、61.16%** 的交互。12,960 次主面板非空干预中，预测类别改变 **0 次**；这阶段测到的是得分交互。[数值附表](source/reports/dSprites_逐项结果表_2026-09-29.md) |
| 10/9–10 · 梯度与形状配对 | 先测标量干预如何改变局部梯度，再沿受体激活逐渐换为供体激活的路径重建输出变化。 | 全强度下，四个源坐标的梯度相对基准平均变化 **120.38%**，中间层梯度变化 **14.43%**。在形状配对上，仅用起点梯度预测端点变化的相对 RMS 误差为 **151.60%**，沿路径做 32 点积分后为 **0.46%**。 |
| 10/11–13 · 三阶抵消与层间传递 | 枚举八个中间通道中的全部 56 个三元组；另从 block1 施加组干预，抹去 block2 的二阶及以上组合项再计算输出。 | 纯三阶之和与四至八阶余项的余弦在恢复、移植时分别为 **−0.930、−0.872**，存在明显抵消。抹去中间组合项后，输出条件成对交互的 RMS 仍剩 **63.80%**，六个模型均比原来弱。 |
| 10/17 · 原组合为何失效 | 把中间层拆成只含空组与单组效应的基准 A，以及剩余的组合项 R。在从半强度切到全强度后原组合不再触发、基准仍在阈值原侧的 539 个状态上，分别只换 A 或只换 R。 | 只换 A，原组合在 **533/539** 个状态不能再跨过 ReLU 阈值；只换 R，则是 **15/539**。在这批失败状态里，基准移动比组合项改变更能重现失效。 |
| 10/18–25 · 联合触发与组外背景 | 在上述状态中寻找上游干预，要求整组重新触发 ReLU，且去掉任一成员都不再触发；再分别切换组内与组外状态。 | 在移动基准、固定原基准和实际组外背景下都成立的定向恢复共 **26 条记录，去重为 15 个状态，分布在 5 个模型**。在组外也改变的 14 条中，组外使原组的输出效应增强 **8** 条、削弱 **6** 条；另外 12 条组外未变。 |
| 10/25 · 单点与联合阻断 | 在上述 26 条记录上，按同一案例的实测输出作用排序，阻断 block3 第一处 ReLU 的位置。 | 阻断最强单点后，原组输出作用的 RMS 仍剩 **97.99%**；联合阻断最多 256 个位置后剩 **37.63%**，同预算两组随机对照剩 **93.49%、93.19%**。此处的整组条件效应包含可加成分。 |

[十月结果的逐模型数值、计数与原始记录指纹](results/stage_evidence.json) · [早期 dSprites 的模型、数据划分与对照](source/reports/dSprites_旧研究再评估_2026-09-29.md)

## 复现 11 月 1 日的四通道规则实验

**复跑从六个已训练的模型出发。** 仓库提供了[六个权重](reproduction/held_condition/runs/dsprites_20260929)、[36 张原图及配对索引](reproduction/held_condition/runs/dsprites_shape_donor_20261010/panel/data.npz)、[原始冻结记录](reproduction/held_condition/reference)和[固定版本依赖](reproduction/held_condition/requirements.txt)。安装依赖后，实验可离线在 CPU 上运行。先克隆仓库，或下载 ZIP 并进入解压后的仓库根目录：

```bash
git clone https://github.com/Wang051027/cross-layer-interaction-study.git
cd cross-layer-interaction-study
```

**Windows PowerShell**（Python 3.12）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r reproduction/held_condition/requirements.txt
.\.venv\Scripts\python.exe -B -X utf8 reproduce_latest.py --device cpu
```

**Linux / macOS**（Python 3.12，同一入口）：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r reproduction/held_condition/requirements.txt
.venv/bin/python -B -X utf8 reproduce_latest.py --device cpu
```

入口会重新拟合规则，先保存预测，再测量两种强度下的真实响应和两项干预，最后与冻结记录比对。有兼容 CUDA 环境可改用 `--device cuda`。每次运行新建一个 `local_output/held-condition-*/` 目录：

| 想核对什么 | 在本次输出目录中打开 |
| --- | --- |
| 是否复跑成功 | `comparison.json` 中 `status` 为 `PASS`，`decisions` 中只有 `middle_third_all` 为 `true`。 |
| 实验报告和逐例数字 | `experiment/source/reports/` 是报告；`experiment/source/runs/dsprites_rule_discrimination_20261031/tables/` 是结果表。 |

**已验证的环境：** Windows、Python 3.12.14。独立 CPU 环境复跑核对了 864 条案例、4,320 条干预名单记录和 516 行模型级统计，数值在容差内一致；CUDA 复跑的 6 个拟合文件、12 个预测文件、12 个实测文件还与原件逐字节相同。[复跑核验记录](checks/held_condition_replay.json) · [完整复现说明](docs/REPRODUCTION.md)

安装 `requirements-demo.txt` 后，运行 `python -B run_dsprites_demo.py` 可重算三张 dSprites 图像的指标，运行 `python -B run_demo.py` 可比较四张 Fashion-MNIST 图像的单点和成对预测。`python -B check_repository.py` 校验文件哈希和 Python 语法。

[源码索引](docs/SCRIPT_INDEX.md) · [实现与公式](source/scripts/dsprites_triple_rule.py) · [独立核验代码](source/scripts/audit_dsprites_triple_rule.py) · [文件来源](PROVENANCE.json) · [第三方数据与许可](THIRD_PARTY.md)

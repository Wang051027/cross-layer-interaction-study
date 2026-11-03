# 留出图像条件的三组规则判别实验

实验完成于 2026 年 11 月 1 日，代码发布于 11 月 3 日。

研究问题是同层通道组合的非加和作用能否通过一条事先固定的层间规则，预测到中间层和输出，并经通道干预验证。

## 输入和冻结顺序

沿用 36 张 dSprites 图像与六个冻结的同架构模型。12 个大小×角度条件按奇偶交错分为 6 个开发条件和 6 个留出条件，两侧各 18 张图。每个条件只取方形→椭圆、椭圆→心形、心形→方形三个循环方向，避免反向互补状态重用。所有强度、组合和模型遵循同一划分。

源层 block1 的 16 个通道分成四个连续的四通道组。每个目标含三个源组，组合掩码为 7、11、13、14。规则只读取目标的空组、三个单组和三个双组状态，四个目标共需 11 个低阶状态，以及空组状态的三个类别梯度。拟合只用开发条件的全强度数据。通道名单、参数、留出预测和阻断名单全部冻结后，才读取三组实测响应。

这里是在已经探索过的图像面板内部留出因素条件，不能视为全新输入上的独立验证。

## 组合规则与输出

二阶基线为 `Q(X) = Xab + Xac + Xbc - Xa - Xb - Xc + X0`。三阶残差为真实三组响应减去 Q。

规则从 block2 第一 ReLU 的阈值变化估计缺失的三阶项，在开发条件选择四个上游通道和四个中间输出通道，使用模型的卷积权重和归一化斜率传播修正。其余中间通道修正为零。输出预测使用空组状态的梯度乘以中间层修正。任务读出为供体类别 logit 减受体类别 logit。

阻断将选中通道恢复为空组激活，只保留操作则仅允许选中通道随组合变化，两者均重新运行原模型后续层。全部八个子集状态使用一致的操作，再求三阶差分。

## 主结果

全强度留出记录为每模型 72 条，共 432 条。下表为先在模型内归一化，再对六模型等权的误差比，越低越好。二阶基线对三阶残差预测为零，对应误差比 1。每项要求均值不大于 0.95，且至少四模型小于 1。

| 预测目标 | 误差比 | 改善模型数 | 预设判据 |
|---|---:|---:|---|
| 完整中间层三阶响应 | 0.884 | 6/6 | 通过 |
| 输出三阶差分 | 1.003 | 5/6 | 未通过 |
| 阻断四通道移除的输出作用 | 1.044 | 2/6 | 未通过 |
| 只保留四通道重现的输出作用 | 0.969 | 4/6 | 未通过 |

同一规则的半强度结果为次要检验。它不能替代全强度判定。完整机制要求四项全部通过，本轮仅支持部分中间层响应的预测。按开发条件的形状转换和组合编号取输出三阶均值，其误差比为 0.619，优于当前物理规则的输出预测。

## 代码入口

| 文件 | 用途 |
|---|---|
| [dsprites_triple_rule.py](../source/scripts/dsprites_triple_rule.py) | 二阶预测、稀疏阈值修正与空组梯度读出 |
| [run_dsprites_triple_rule.py](../source/scripts/run_dsprites_triple_rule.py) | 条件划分、开发拟合、冻结预测与实测干预 |
| [audit_dsprites_triple_rule.py](../source/scripts/audit_dsprites_triple_rule.py) | 用独立运算路径核对响应、导数和阻断 |
| [analyze_dsprites_triple_rule.py](../source/scripts/analyze_dsprites_triple_rule.py) | 逐例统计、模型内归一化与预设判据 |
| [report_dsprites_triple_rule.py](../source/scripts/report_dsprites_triple_rule.py) | 汇总、图表、封存及只读校验 |

已保存的 [设计](../results/rule_discrimination/design.json)、[协议](../results/rule_discrimination/protocol.json)、[分析](../results/rule_discrimination/analysis.json)、[独立核验](../results/rule_discrimination/independent_audit.json) 和 [模型级统计](../results/rule_discrimination/tables/teacher_scores.csv) 随源码发布。它们来自原运行，完整模型、面板、预测和实测数组另存。复算依赖和阶段说明见 [REPRODUCTION.md](REPRODUCTION.md)。

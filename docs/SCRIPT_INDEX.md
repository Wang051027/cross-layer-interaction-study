# 全部研究源码索引

原项目 169 个研究脚本与 9 个历史依赖，共 178 个。根目录另有发布专用示例和核验入口。

| 文件 | 用途 |
|---|---|
| [adaptive_mediator.py](../source/scripts/adaptive_mediator.py) | 第二轮：在相同候选空间内比较固定/逐输入选择与梯度/有限干预评分。 |
| [analyze_dsprites_coalition_context.py](../source/scripts/analyze_dsprites_coalition_context.py) | 计算组内通道和组外背景在四个状态下的输出差及交互项。 |
| [analyze_dsprites_downstream_routes.py](../source/scripts/analyze_dsprites_downstream_routes.py) | 提取具体下游位置、通道、联合阻断余量、残差支路及抵消。 |
| [analyze_dsprites_triple_rule.py](../source/scripts/analyze_dsprites_triple_rule.py) | 按冻结判据汇总留出预测，保留逐案例误差、阻断方向和失败案例。 |
| [ar_factorial_mechanism.py](../source/scripts/ar_factorial_mechanism.py) | 分开控制基准A和组合残差R，解释冻结通道组的阈值失效及输出变化。 |
| [audit_dsprites_ar_factorial.py](../source/scripts/audit_dsprites_ar_factorial.py) | 用原模型前向运算重建 A/R 四个状态，以 32 个单通道卷积和自动微分核对阈值及传递量。 |
| [audit_dsprites_channel_gates.py](../source/scripts/audit_dsprites_channel_gates.py) | 用稠密容斥矩阵核对通道配对系数。用 block3 前向运算和自动微分核对指定 ReLU 位置的干预。 |
| [audit_dsprites_coalition_context.py](../source/scripts/audit_dsprites_coalition_context.py) | 从模型前向激活重建四个状态。用卷积和 BatchNorm 的仿射差分核对逐成员恢复及分层余项。 |
| [audit_dsprites_data.py](../source/scripts/audit_dsprites_data.py) | 按官方索引核对 dSprites 像素、标签和因素划分，逐像素验证平移对照。 |
| [audit_dsprites_downstream_routes.py](../source/scripts/audit_dsprites_downstream_routes.py) | 按九种空间位移累加单点作用，用原模型前向运算复核前 16 个位置、目标开关、边界点及联合阻断。 |
| [audit_dsprites_factor_transfer.py](../source/scripts/audit_dsprites_factor_transfer.py) | 由原始响应、差分矩阵和增广最小二乘重算因素迁移结果，核对划分中的重复图像。 |
| [audit_dsprites_gradient.py](../source/scripts/audit_dsprites_gradient.py) | 用稠密对比矩阵重算交互和单点收益。用四层卷积的伴随运算重建中间层和源层的雅可比矩阵。 |
| [audit_dsprites_joint_relu.py](../source/scripts/audit_dsprites_joint_relu.py) | 以 32 个单通道卷积重建预激活，用原模型前向运算和自动微分核对开关干预。 |
| [audit_dsprites_joint_window.py](../source/scripts/audit_dsprites_joint_window.py) | 用 13×13 区域上的原卷积和 BatchNorm 复核候选解，在 block2、block3 上检验所有真子集的输出。 |
| [audit_dsprites_layer_transfer.py](../source/scripts/audit_dsprites_layer_transfer.py) | 逐组重建源激活，用稠密容斥矩阵核算中间场，并沿卷积和残差支路反向计算梯度。 |
| [audit_dsprites_residual_routes.py](../source/scripts/audit_dsprites_residual_routes.py) | 独立核验中间组合残差路由：原尾部、稠密差分、autograd及手推路径梯度。 |
| [audit_dsprites_results.py](../source/scripts/audit_dsprites_results.py) | 从 dSprites 的逐例输出重算中介评分。用差分矩阵和类别投影矩阵核对交互量。 |
| [audit_dsprites_shape_donor.py](../source/scripts/audit_dsprites_shape_donor.py) | 用卷积伴随重算方向梯度，逐通道重建配对修补，并用稠密差分矩阵核算交互。 |
| [audit_dsprites_triple_gates.py](../source/scripts/audit_dsprites_triple_gates.py) | 核对 56 个三元组的纯三阶和条件三阶系数、符号抵消及真实开关干预。 |
| [audit_dsprites_triple_rule.py](../source/scripts/audit_dsprites_triple_rule.py) | 在预测与实测封存之后，用原模块、独立容斥和手算尾部伴随重建数值。 |
| [audit_dsprites_upstream_rescue.py](../source/scripts/audit_dsprites_upstream_rescue.py) | 复核 22 种上游干预。用 5×5 卷积重计算第一处 ReLU 的候选得分，以原网络前向运算和自动微分核对恢复效果。 |
| [audit_external_head_activity.py](../source/scripts/audit_external_head_activity.py) | Summarize head activations from the stored training fits. |
| [audit_residual_affine_runtime.py](../source/scripts/audit_residual_affine_runtime.py) | Post-test, explicitly separate audit of an algebraically fused affine control. |
| [audit_structure_interpretation.py](../source/scripts/audit_structure_interpretation.py) | Post-result explanation checks; no fitting, selection, or result replacement. |
| [audit_upstream.py](../source/scripts/audit_upstream.py) | Re-score/replay v0.7 without modifying its imported archive contents. |
| [benchmark_heads.py](../source/scripts/benchmark_heads.py) | Descriptive matched-backend diagnostic, separate from historical timings. |
| [benchmark_pilot.py](../source/scripts/benchmark_pilot.py) | CPU cost diagnostic on four pilot validation images, after other jobs finish. |
| [channel_gate_mechanism.py](../source/scripts/channel_gate_mechanism.py) | 在 block2 通道上枚举 43 个状态，分解 block3 两处 ReLU 的带符号贡献，并运行门控干预。 |
| [coalition_context_mechanism.py](../source/scripts/coalition_context_mechanism.py) | 分别替换原通道组 C 和组外通道，计算四个中间状态及其输出差。 |
| [cross_layer_mechanism.py](../source/scripts/cross_layer_mechanism.py) | 从 block1 注入组合干预，沿 block2 和 block3 的卷积及残差支路分解响应，并测量中介恢复与移植。 |
| [diagnose_dsprites_channel_combinations.py](../source/scripts/diagnose_dsprites_channel_combinations.py) | 从半强度下 32 个单通道和 19 个联合修补响应计算高阶余项、端点重建误差和类别变化。 |
| [downstream_route_mechanism.py](../source/scripts/downstream_route_mechanism.py) | 定位原C的下游传递：实际单点阻断、联合阻断、保留恢复与残差支路。 |
| [dsprites_mechanism.py](../source/scripts/dsprites_mechanism.py) | 计算 dSprites 形状分类的条件差分和中介修补。支持类别 margin、中心化输出向量与平移坐标。 |
| [dsprites_teacher.py](../source/scripts/dsprites_teacher.py) | 在 64×64 的 dSprites 图像上训练六个三类残差网络，按照验证损失选取检查点。 |
| [dsprites_triple_rule.py](../source/scripts/dsprites_triple_rule.py) | 用低阶状态预测三组联合响应。训练和打分在独立入口中进行。 |
| [dsprites_workflow.py](../source/scripts/dsprites_workflow.py) | 接续 dSprites 的训练、选点、冻结、评价与核验。train-safe 将训练完成时间写入独立日志字段。 |
| [external_analysis.py](../source/scripts/external_analysis.py) | Prespecified seed-level analysis and saved-array audit, without model fitting. |
| [external_experiment.py](../source/scripts/external_experiment.py) | Shared, frozen implementation for the six-seed external experiment. |
| [factor_mediator.py](../source/scripts/factor_mediator.py) | 在开发集上拟合类别、大小和角度与单点收益的关系，从 64 个候选中选取前 16 个进行联合恢复和移植。 |
| [finalize_dsprites_ar_factorial.py](../source/scripts/finalize_dsprites_ar_factorial.py) | 生成基准 A 与组合残差 R 四种干预状态的文件清单。 |
| [finalize_dsprites_channel_gates.py](../source/scripts/finalize_dsprites_channel_gates.py) | 审计和报告完成后封存通道与ReLU研究，本轮交付不覆盖历史证据。 |
| [finalize_dsprites_coalition_context.py](../source/scripts/finalize_dsprites_coalition_context.py) | 封存新四状态交付；已完成旧目录及旧根说明保持原样。 |
| [finalize_dsprites_downstream_routes.py](../source/scripts/finalize_dsprites_downstream_routes.py) | 封存下游定位交付；不覆盖旧交付或自动发布GitHub。 |
| [finalize_dsprites_gradient.py](../source/scripts/finalize_dsprites_gradient.py) | 保存梯度实验的审计、报告和代码快照，并计算 SHA-256。 |
| [finalize_dsprites_joint_relu.py](../source/scripts/finalize_dsprites_joint_relu.py) | 封存本轮独立交付；仅CURRENT_RESEARCH.md的日期快照进入清单。 |
| [finalize_dsprites_joint_window.py](../source/scripts/finalize_dsprites_joint_window.py) | 封存联合必要性恢复；不改十一份历史交付及旧根文件。 |
| [finalize_dsprites_layer_transfer.py](../source/scripts/finalize_dsprites_layer_transfer.py) | 汇总层间传递实验的输出文件和哈希。 |
| [finalize_dsprites_residual_routes.py](../source/scripts/finalize_dsprites_residual_routes.py) | 封存中间组合效应路由研究，保留全部七轮旧交付。 |
| [finalize_dsprites_shape_donor.py](../source/scripts/finalize_dsprites_shape_donor.py) | 生成形状配对实验的代码、统计表、报告及文件哈希清单。 |
| [finalize_dsprites_triple_gates.py](../source/scripts/finalize_dsprites_triple_gates.py) | 全部独立核验和报告完成后封存三阶研究，保留前五轮交付。 |
| [finalize_dsprites_upstream_rescue.py](../source/scripts/finalize_dsprites_upstream_rescue.py) | 汇总上游基准定位与恢复实验的文件哈希。 |
| [finalize_joint_response.py](../source/scripts/finalize_joint_response.py) | Build the joint-response results index and checksum manifest. |
| [gradient_mechanism.py](../source/scripts/gradient_mechanism.py) | 计算类别 logit 对源层和中间层激活的梯度，用各干预状态的局部斜率预测输出变化。 |
| [joint_relu_mechanism.py](../source/scripts/joint_relu_mechanism.py) | 对第一处 ReLU 的 32 个通道贡献按目标方向排序，计算触发阈值所需的最小通道数，再重算输出。 |
| [joint_response.py](../source/scripts/joint_response.py) | Fit additive and pairwise responses from empty, singleton, and pair masks. A set bit scales its activation by 1 − alpha. The Walsh transform uses the full cube. |
| [joint_window_measure.py](../source/scripts/joint_window_measure.py) | 整图重算冻结的39个端点，区分分解变化、实际传递、C消融与最终任务输出。 |
| [joint_window_mechanism.py](../source/scripts/joint_window_mechanism.py) | 在移动基准、固定基准和实测组外背景下搜索触发 ReLU 的上游干预，并逐一检验原通道组的必要成员。 |
| [layer_transfer_mechanism.py](../source/scripts/layer_transfer_mechanism.py) | 枚举 block1 四组通道的 16 个状态，用容斥分离 block2 的组合项，再经 block3 测量擦除后的输出。 |
| [make_audit_figure.py](../source/scripts/make_audit_figure.py) | Standalone scientific figure of local replay accuracy and matched CPU timing. |
| [package_snapshot.py](../source/scripts/package_snapshot.py) | Package the completed research without changing existing research artifacts. |
| [plot_external_results.py](../source/scripts/plot_external_results.py) | Standalone figure; primary accuracy and descriptive matched-budget cost. |
| [plot_residual_results.py](../source/scripts/plot_residual_results.py) | Presentation-only scientific figure for the completed residual experiment. |
| [plot_scale_results.py](../source/scripts/plot_scale_results.py) | Scientific figure from completed, frozen evaluation and timing records. |
| [plot_structure_controls.py](../source/scripts/plot_structure_controls.py) | Plot the completed follow-up and the quality limitation of the GAP control. |
| [prepare_dsprites.py](../source/scripts/prepare_dsprites.py) | 从官方固定提交获取dSprites，安全解压数字数组并建立受控形状分类划分。 |
| [prepare_external_split.py](../source/scripts/prepare_external_split.py) | Use the upstream fixed grouping algorithm and seed; write outside upstream. |
| [prepare_fashion_mnist.py](../source/scripts/prepare_fashion_mnist.py) | Download Fashion-MNIST and verify IDX headers, dimensions, and checksums. |
| [report_adaptive_mediator.py](../source/scripts/report_adaptive_mediator.py) | 生成第二轮中文报告、图表和交付索引；只读取已冻结、已独立核验的结果。 |
| [report_cross_layer_mechanism.py](../source/scripts/report_cross_layer_mechanism.py) | 汇总跨层恢复和移植实验。误差线为六个模型的标准差，匹配随机对照先在模型内取平均。 |
| [report_dsprites_ar_factorial.py](../source/scripts/report_dsprites_ar_factorial.py) | 分别计算 A/R 四格在每个模型内的 RMS 和投影，再对六个模型等权汇总。 |
| [report_dsprites_channel_gates.py](../source/scripts/report_dsprites_channel_gates.py) | 汇总通道配对、ReLU 位置、各阶系数和干预结果，比较半强度与全强度的排名。 |
| [report_dsprites_coalition_context.py](../source/scripts/report_dsprites_coalition_context.py) | 生成中文四状态研究报告与可导出的科学图；所有数字来自已保存逐行结果。 |
| [report_dsprites_downstream_routes.py](../source/scripts/report_dsprites_downstream_routes.py) | 生成下游路径定位报告、通道/位置图和联合阻断后的正负余量图。 |
| [report_dsprites_factor_transfer.py](../source/scripts/report_dsprites_factor_transfer.py) | 按模型和因素条件列出配对误差，比较候选排序在不同干预强度下的表现。 |
| [report_dsprites_gradient.py](../source/scripts/report_dsprites_gradient.py) | 从逐例响应计算模型内梯度变化和交互误差，生成逐模型结果表和图。 |
| [report_dsprites_joint_relu.py](../source/scripts/report_dsprites_joint_relu.py) | 把冻结实验整理为逐案例证据表、按教师汇总和可独立保存的科学图。 |
| [report_dsprites_joint_window.py](../source/scripts/report_dsprites_joint_window.py) | 原组精确必要性恢复的证据表、共同参照筛查、任务输出和中文科学报告。 |
| [report_dsprites_layer_transfer.py](../source/scripts/report_dsprites_layer_transfer.py) | 从层间传递数组计算各模型的指标，并绘制六模型等权平均结果。 |
| [report_dsprites_residual_routes.py](../source/scripts/report_dsprites_residual_routes.py) | 将固定通道名单的传递、阈值与抵消结果写入 CSV 和图表。 |
| [report_dsprites_shape_donor.py](../source/scripts/report_dsprites_shape_donor.py) | 将形状配对的恢复、移植和路径积分误差按模型汇总并绘图。 |
| [report_dsprites_triple_gates.py](../source/scripts/report_dsprites_triple_gates.py) | 汇总八通道内的纯三阶、条件三阶和四阶以上余项，比较两种强度的开关与符号。 |
| [report_dsprites_triple_rule.py](../source/scripts/report_dsprites_triple_rule.py) | 制作判别实验报告，复核逐例统计，并提供封存后的只读校验。 |
| [report_dsprites_upstream_rescue.py](../source/scripts/report_dsprites_upstream_rescue.py) | 上游基准移动与救回的逐项证据、六模型汇总和可导出科学图。 |
| [report_joint_response.py](../source/scripts/report_joint_response.py) | Render the completed, independently verified descriptive analysis. |
| [report_switch_results.py](../source/scripts/report_switch_results.py) | Presentation derived only from completed exact-switch experiment records. |
| [research_runtime.py](../source/scripts/research_runtime.py) | Matched CPU runtimes for the unchanged v0.7 estimators. |
| [residual_context.py](../source/scripts/residual_context.py) | Local finite interventions inside the last branch of a residual CNN. |
| [residual_route_mechanism.py](../source/scripts/residual_route_mechanism.py) | 中间组合残差R=H−H≤1的通道、两次ReLU与抵消机制。 |
| [residual_route_path.py](../source/scripts/residual_route_path.py) | 预定例子的分段梯度积分：先找第一ReLU根，再找每段第二ReLU根。 |
| [residual_teacher.py](../source/scripts/residual_teacher.py) | Small residual CNN and locked train/validation-only development stage. |
| [run_adaptive_mediator.py](../source/scripts/run_adaptive_mediator.py) | 第二轮的显式阶段入口：开发、锁定规则、半强度选择、锁定名单、完整强度评价。 |
| [run_cross_layer_mechanism.py](../source/scripts/run_cross_layer_mechanism.py) | 跨层机制第一轮的阶段入口：先开发冻结，再评价和只读核验。 |
| [run_dsprites_ar_factorial.py](../source/scripts/run_dsprites_ar_factorial.py) | 在冻结通道名单上交叉组合 A0、A1 和 R0、R1，运行四种干预。 |
| [run_dsprites_channel_gates.py](../source/scripts/run_dsprites_channel_gates.py) | 在八通道名单中按半强度效应选取通道对和开关位置，再与两组随机名单一起评价全强度响应。 |
| [run_dsprites_coalition_context.py](../source/scripts/run_dsprites_coalition_context.py) | 在已选恢复端点上分别替换组内和组外通道，运行四个状态。 |
| [run_dsprites_downstream_routes.py](../source/scripts/run_dsprites_downstream_routes.py) | 同输入下游定位：先锁定案例/规则，单点定位封存后再做联合阻断。 |
| [run_dsprites_factor_transfer.py](../source/scripts/run_dsprites_factor_transfer.py) | 开发数据拟合因素规则，冻结后在未用过的dSprites像素上评价。 |
| [run_dsprites_gradient.py](../source/scripts/run_dsprites_gradient.py) | 梯度变化接续实验：init → select → seal → evaluate → summarize。 |
| [run_dsprites_joint_relu.py](../source/scripts/run_dsprites_joint_relu.py) | 按半强度响应选定通道与 ReLU 坐标，在全强度下验证联合触发。 |
| [run_dsprites_joint_window.py](../source/scripts/run_dsprites_joint_window.py) | 按阈值余量搜索联合必要的上游干预，再运行完整网络测量输出。 |
| [run_dsprites_layer_transfer.py](../source/scripts/run_dsprites_layer_transfer.py) | 枚举源通道组，擦除中间层组合项，计算剩余输出交互。 |
| [run_dsprites_mechanism.py](../source/scripts/run_dsprites_mechanism.py) | dSprites迁移评估的阶段入口，先锁定规则和半强度名单，再评价。 |
| [run_dsprites_residual_routes.py](../source/scripts/run_dsprites_residual_routes.py) | 同批输入的中间组合残差路由：冻结→半强度定位→封存→全强度验证。 |
| [run_dsprites_shape_donor.py](../source/scripts/run_dsprites_shape_donor.py) | 在 144 张开发图形成的 288 个有向配对上拟合固定名单，用半强度响应选出 4 或 8 个中介通道，再评价全强度响应。 |
| [run_dsprites_triple_gates.py](../source/scripts/run_dsprites_triple_gates.py) | 用半强度响应选取三元组及正、反向开关位置，在两个强度上测量同一名单。 |
| [run_dsprites_triple_rule.py](../source/scripts/run_dsprites_triple_rule.py) | 按协议、拟合、预测、实测的顺序封存。每一阶段可在文件边界续跑。 |
| [run_dsprites_upstream_rescue.py](../source/scripts/run_dsprites_upstream_rescue.py) | 同36图六模型的基准上游定位：先封存协议，再定位，最后评价救回与输出。 |
| [run_external.py](../source/scripts/run_external.py) | Train and evaluate six Fashion-MNIST models using the frozen development split and test protocol. |
| [run_fashion_pilot.py](../source/scripts/run_fashion_pilot.py) | First fixed teacher, train/validation feasibility only; no official test access. |
| [run_joint_response.py](../source/scripts/run_joint_response.py) | Protocol-locked descriptive reanalysis of saved coalition responses. |
| [run_residual_controls.py](../source/scripts/run_residual_controls.py) | Frozen residual-block context experiment, including independent references. |
| [run_scale_controls.py](../source/scripts/run_scale_controls.py) | Locked size/position follow-up; no prior numerical artifact is overwritten. |
| [run_structure_controls.py](../source/scripts/run_structure_controls.py) | Compare head activity and GAP/linear readouts in the Fashion-MNIST structural controls. |
| [run_switch_controls.py](../source/scripts/run_switch_controls.py) | Frozen exact-switch-screening validation, timing, and new intervention batch. |
| [scale_context.py](../source/scripts/scale_context.py) | Variable source sizes, spatial positions, and a conservative affine fallback. |
| [shape_donor_mechanism.py](../source/scripts/shape_donor_mechanism.py) | 同条件形状状态替换、通道中介修补与沿路径梯度。 |
| [structure_runtime.py](../source/scripts/structure_runtime.py) | Activity-only selection controls and exact GAP/linear structural controls. |
| [summarize_dsprites_factors.py](../source/scripts/summarize_dsprites_factors.py) | 按模型、形状、大小和角度汇总 dSprites 评价响应，绘制因素分层结果。 |
| [summarize_scale_results.py](../source/scripts/summarize_scale_results.py) | Descriptive synthesis and shared-validation consistency audit; no selection. |
| [switch_context.py](../source/scripts/switch_context.py) | Exact local responses after folding states stable over the intervention box. |
| [tabulate_dsprites_results.py](../source/scripts/tabulate_dsprites_results.py) | 将已审计的完整结果生成中文数值附表；文字结论在主报告中另行判断。 |
| [test_adaptive_mediator.py](../source/scripts/test_adaptive_mediator.py) | 新增选择器的已知答案与原网络核验；验证数据不进入模型选择。 |
| [test_cross_layer_mechanism.py](../source/scripts/test_cross_layer_mechanism.py) | 用已知差分和模型前向钩子核对跨层干预。 |
| [test_dsprites_ar_factorial.py](../source/scripts/test_dsprites_ar_factorial.py) | 协议前的已知四格答案和原模型核验；不寻找新的图像或坐标。 |
| [test_dsprites_channel_gates.py](../source/scripts/test_dsprites_channel_gates.py) | 在开发图像上核对通道配对和 ReLU 开关干预。 |
| [test_dsprites_coalition_context.py](../source/scripts/test_dsprites_coalition_context.py) | 数值预检：原forward、四状态边界、抵消恒等式和旧端点复现。 |
| [test_dsprites_downstream_routes.py](../source/scripts/test_dsprites_downstream_routes.py) | 新下游阻断的数值预检：原finish、边界核方向、联合控制与末层可加性。 |
| [test_dsprites_gradient.py](../source/scripts/test_dsprites_gradient.py) | 在验证图上比较 block3 反向卷积、逐样本自动微分与小扰动差商，核对雅可比矩阵的批次轴。 |
| [test_dsprites_joint_relu.py](../source/scripts/test_dsprites_joint_relu.py) | 选择前的已知答案和原网络核验；只用上一轮已指定的第一个案例。 |
| [test_dsprites_joint_window.py](../source/scripts/test_dsprites_joint_window.py) | 冻结协议前验证联合必要性逻辑、固定参照不变性和局部/原整图一致。 |
| [test_dsprites_layer_transfer.py](../source/scripts/test_dsprites_layer_transfer.py) | 在开发集首个形状配对上核对门控前向结果，用手算例子验证上下游交互和抵消。 |
| [test_dsprites_mechanism.py](../source/scripts/test_dsprites_mechanism.py) | 迁移核验：旧十类计算的一致性、三类真实模型钩子及动态维度已知答案。 |
| [test_dsprites_residual_routes.py](../source/scripts/test_dsprites_residual_routes.py) | 新通道与开关路由的前置检查，使用旧面板预定案例。 |
| [test_dsprites_shape_donor.py](../source/scripts/test_dsprites_shape_donor.py) | 用开发图检查通道替换、梯度内积和仿射路径积分。 |
| [test_dsprites_triple_gates.py](../source/scripts/test_dsprites_triple_gates.py) | 在手算多项式、阈值抵消例子和开发图上核对三阶差分，用自动微分复核第一层传播权重。 |
| [test_dsprites_upstream_rescue.py](../source/scripts/test_dsprites_upstream_rescue.py) | 定位前检查：原网络锚点、单开关精确分数、联动端点及两层恒等式。 |
| [test_joint_response.py](../source/scripts/test_joint_response.py) | Meaningful numerical controls for the finite-game analysis. |
| [triple_gate_mechanism.py](../source/scripts/triple_gate_mechanism.py) | 枚举八个通道的 56 个三元组。八角差分给出空背景的纯三阶系数，补全其他通道后的差分还含四阶以上项。 |
| [upstream_rescue_measure.py](../source/scripts/upstream_rescue_measure.py) | 冻结原通道组C和下游点j，比较A单独修复与上游联动重算的阈值和输出。 |
| [upstream_rescue_mechanism.py](../source/scripts/upstream_rescue_mechanism.py) | 用空组和单组响应重建基准 A，再按对基准的修复量排序上游干预，并同时重算 A 与 R。 |
| [verify_adaptive_mediator.py](../source/scripts/verify_adaptive_mediator.py) | 用 24×16 差分矩阵和 10 维目标向量重算中介评分与通道名单。 |
| [verify_cross_layer_mechanism.py](../source/scripts/verify_cross_layer_mechanism.py) | 用稠密差分矩阵和输出方向的内积，从保存的响应重算跨层指标。 |
| [verify_dsprites_ar_factorial_delivery.py](../source/scripts/verify_dsprites_ar_factorial_delivery.py) | 只读核验全部四格、输出分解、正文统计、交付来源和九份旧交付。 |
| [verify_dsprites_channel_gates_delivery.py](../source/scripts/verify_dsprites_channel_gates_delivery.py) | 只读核验交付、协议、选择名单；深核验覆盖全部冻结输入和旧交付。 |
| [verify_dsprites_coalition_context_delivery.py](../source/scripts/verify_dsprites_coalition_context_delivery.py) | 只读核验四状态交付：逐行表格、独立差分、统计分母、图和文件SHA。 |
| [verify_dsprites_downstream_routes_delivery.py](../source/scripts/verify_dsprites_downstream_routes_delivery.py) | 逐行展开 67 种干预名单，核对输出余量、残差支路与抵消恒等式。 |
| [verify_dsprites_factor_delivery.py](../source/scripts/verify_dsprites_factor_delivery.py) | 核对因素迁移实验的清单和代码快照。--deep-inputs 额外检查原始输入和源码哈希。 |
| [verify_dsprites_gate_sites.py](../source/scripts/verify_dsprites_gate_sites.py) | 预定例子的逐坐标反事实补充：先封存规则，再逐点固定开关。 |
| [verify_dsprites_gradient_delivery.py](../source/scripts/verify_dsprites_gradient_delivery.py) | 核对梯度实验的清单、历史输入与冻结记录。深度核验还检查原始数据。 |
| [verify_dsprites_joint_relu_delivery.py](../source/scripts/verify_dsprites_joint_relu_delivery.py) | 只读核验：来源/时序、全部证据表、汇总分母和八份旧交付。 |
| [verify_dsprites_joint_window_delivery.py](../source/scripts/verify_dsprites_joint_window_delivery.py) | 只读复核逐行证据、解选择引用、教师统计、共同开关恒等式与全部交付SHA。 |
| [verify_dsprites_layer_transfer_delivery.py](../source/scripts/verify_dsprites_layer_transfer_delivery.py) | 核对层间传递实验的文件、汇总表与历史输入。 |
| [verify_dsprites_residual_routes_delivery.py](../source/scripts/verify_dsprites_residual_routes_delivery.py) | 只读核验新路由交付、固定时序、数值恒等式和七轮旧研究。 |
| [verify_dsprites_route_sixth_order.py](../source/scripts/verify_dsprites_route_sixth_order.py) | 在固定的六通道和目标位置上枚举 64 个子集，计算六阶激活和输出差分。 |
| [verify_dsprites_route_threshold.py](../source/scripts/verify_dsprites_route_threshold.py) | 固定例子的事后联合阈值验证，独立于半强度名单的主评价。 |
| [verify_dsprites_shape_donor_delivery.py](../source/scripts/verify_dsprites_shape_donor_delivery.py) | 核对形状配对的实验清单、协议和封存记录。深度核验还检查数据和模型权重。 |
| [verify_dsprites_triple_gates_delivery.py](../source/scripts/verify_dsprites_triple_gates_delivery.py) | 只读核验三阶交付、协议、名单和表格；深核验包含全部历史输入。 |
| [verify_dsprites_upstream_rescue_delivery.py](../source/scripts/verify_dsprites_upstream_rescue_delivery.py) | 只读检查清单、冻结时序、CSV逐行证据、教师统计和全部历史文件。 |
| [verify_external_delivery.py](../source/scripts/verify_external_delivery.py) | Read-only experiment checks, then a separate September 11 delivery snapshot. |
| [verify_joint_response.py](../source/scripts/verify_joint_response.py) | Check saved local coefficients, aggregate scores and projections with an explicit polynomial and parity basis. |
| [verify_residual_delivery.py](../source/scripts/verify_residual_delivery.py) | Verify residual-study file hashes and rescore the stored response arrays. |
| [verify_scale_delivery.py](../source/scripts/verify_scale_delivery.py) | Final preservation and delivery checks for the size/position study. |
| [verify_structure_delivery.py](../source/scripts/verify_structure_delivery.py) | Read-only preservation checks and a separate snapshot for structure controls. |
| [verify_switch_delivery.py](../source/scripts/verify_switch_delivery.py) | Preservation, provenance, and delivery audit for the exact-switch study. |
| [write_external_report.py](../source/scripts/write_external_report.py) | Human-readable report from the completed frozen experiment only. |
| [write_followup_report.py](../source/scripts/write_followup_report.py) | Build the handoff from saved, actually executed results. |
| [write_residual_report.py](../source/scripts/write_residual_report.py) | Presentation-only synthesis of the frozen study and separate affine audit. |
| [write_scale_report.py](../source/scripts/write_scale_report.py) | Presentation-only Chinese report and project-status update. |
| [write_structure_report.py](../source/scripts/write_structure_report.py) | Presentation-only report generator for the completed frozen controls. |
| [upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py](../source/upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py) | Acquire official Fashion-MNIST binaries, verify published MD5 and local SHA-256. |
| [upstream/CAT_MT_phase7_v0.7/code/analyze_and_verify.py](../source/upstream/CAT_MT_phase7_v0.7/code/analyze_and_verify.py) | 历史依赖模块 |
| [upstream/CAT_MT_phase7_v0.7/code/context_module.py](../source/upstream/CAT_MT_phase7_v0.7/code/context_module.py) | Restricted background selection. Original weights are retained; train-only masks |
| [upstream/CAT_MT_phase7_v0.7/code/prepare_external.py](../source/upstream/CAT_MT_phase7_v0.7/code/prepare_external.py) | Hard-gated Fashion-MNIST preparation; no missing-file fallback to another task. |
| [upstream/CAT_MT_phase7_v0.7/code/run_development.py](../source/upstream/CAT_MT_phase7_v0.7/code/run_development.py) | 历史依赖模块 |
| [upstream/CAT_MT_phase7_v0.7/code/runtime_backend_check.py](../source/upstream/CAT_MT_phase7_v0.7/code/runtime_backend_check.py) | Post-primary backend control: same selected-background formula in Torch. |
| [upstream/CAT_MT_phase7_v0.7/code/write_report.py](../source/upstream/CAT_MT_phase7_v0.7/code/write_report.py) | 历史依赖模块 |
| [upstream/CAT_MT_phase7_v0.7/vendor/confirmatory_core.py](../source/upstream/CAT_MT_phase7_v0.7/vendor/confirmatory_core.py) | 历史依赖模块 |
| [upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py](../source/upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py) | Phase VI. Exact context-conditioned local compilation and conservative interaction groups. |

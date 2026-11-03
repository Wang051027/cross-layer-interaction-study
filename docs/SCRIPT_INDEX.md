# 全部研究源码索引

原项目 169 个研究脚本与 9 个历史依赖，共 178 个。根目录另有发布专用示例和核验入口。

| 文件 | 文档字符串中的用途 |
|---|---|
| [adaptive_mediator.py](../source/scripts/adaptive_mediator.py) | 第二轮：在相同候选空间内比较固定/逐输入选择与梯度/有限干预评分。 |
| [analyze_dsprites_coalition_context.py](../source/scripts/analyze_dsprites_coalition_context.py) | 四状态效应逐行提取：保留失败、随机及分解对照，不把交互项分配成唯一份额。 |
| [analyze_dsprites_downstream_routes.py](../source/scripts/analyze_dsprites_downstream_routes.py) | 提取具体下游位置、通道、联合阻断余量、残差支路及抵消。 |
| [analyze_dsprites_triple_rule.py](../source/scripts/analyze_dsprites_triple_rule.py) | 按冻结判据汇总留出预测，保留逐案例误差、阻断方向和失败案例。 |
| [ar_factorial_mechanism.py](../source/scripts/ar_factorial_mechanism.py) | 分开控制基准A和组合残差R，解释冻结通道组的阈值失效及输出变化。 |
| [audit_dsprites_ar_factorial.py](../source/scripts/audit_dsprites_ar_factorial.py) | 独立审计A/R四格：原forward、原开关注入、32单通道卷积及自动微分。 |
| [audit_dsprites_channel_gates.py](../source/scripts/audit_dsprites_channel_gates.py) | 独立核验配对、容斥各阶、ReLU坐标选择和真实门控干预。 |
| [audit_dsprites_coalition_context.py](../source/scripts/audit_dsprites_coalition_context.py) | 独立审计：不调用四状态数值核心，从原模型forward与实际激活重建。 |
| [audit_dsprites_data.py](../source/scripts/audit_dsprites_data.py) | 独立检查 dSprites 来源行、标签、因素分组与跨划分像素重合。 |
| [audit_dsprites_downstream_routes.py](../source/scripts/audit_dsprites_downstream_routes.py) | 独立审计下游定位与阻断：原forward、不同邻域算法、全部联合干预。 |
| [audit_dsprites_factor_transfer.py](../source/scripts/audit_dsprites_factor_transfer.py) | 独立回读因素迁移实验：矩阵差分、增广最小二乘及数据去重检查。 |
| [audit_dsprites_gradient.py](../source/scripts/audit_dsprites_gradient.py) | 独立回读梯度接续实验，不调用新的梯度或汇总实现。 |
| [audit_dsprites_joint_relu.py](../source/scripts/audit_dsprites_joint_relu.py) | 独立审计：原网络32个单通道forward、原mid/finish开关干预、自动微分。 |
| [audit_dsprites_joint_window.py](../source/scripts/audit_dsprites_joint_window.py) | 独立审计有限搜索与真实恢复：原卷积/原forward、所有真子集及共同点控制。 |
| [audit_dsprites_layer_transfer.py](../source/scripts/audit_dsprites_layer_transfer.py) | 独立核验层间传递，不调用新核心的数值函数。 |
| [audit_dsprites_residual_routes.py](../source/scripts/audit_dsprites_residual_routes.py) | 独立核验中间组合残差路由：原尾部、稠密差分、autograd及手推路径梯度。 |
| [audit_dsprites_results.py](../source/scripts/audit_dsprites_results.py) | 独立回读 dSprites 原始输出，重算交互/中介分数，并检查冻结与历史完整性。 |
| [audit_dsprites_shape_donor.py](../source/scripts/audit_dsprites_shape_donor.py) | 独立核验形状配对、通道修补、方向梯度及路径积分。 |
| [audit_dsprites_triple_gates.py](../source/scripts/audit_dsprites_triple_gates.py) | 独立核验全部56纯三阶、条件三阶、符号抵消与真实门控反事实。 |
| [audit_dsprites_triple_rule.py](../source/scripts/audit_dsprites_triple_rule.py) | 在预测与实测封存之后，用原模块、独立容斥和手算尾部伴随重建数值。 |
| [audit_dsprites_upstream_rescue.py](../source/scripts/audit_dsprites_upstream_rescue.py) | 独立检查上游修复：原forward注入、裁剪卷积候选分数及自动微分。 |
| [audit_external_head_activity.py](../source/scripts/audit_external_head_activity.py) | Post-test descriptive diagnostic from already frozen training fits only. |
| [audit_residual_affine_runtime.py](../source/scripts/audit_residual_affine_runtime.py) | Post-test, explicitly separate audit of an algebraically fused affine control. |
| [audit_structure_interpretation.py](../source/scripts/audit_structure_interpretation.py) | Post-result explanation checks; no fitting, selection, or result replacement. |
| [audit_upstream.py](../source/scripts/audit_upstream.py) | Re-score/replay v0.7 without modifying its imported archive contents. |
| [benchmark_heads.py](../source/scripts/benchmark_heads.py) | Descriptive matched-backend diagnostic, separate from historical timings. |
| [benchmark_pilot.py](../source/scripts/benchmark_pilot.py) | CPU cost diagnostic on four pilot validation images, after other jobs finish. |
| [channel_gate_mechanism.py](../source/scripts/channel_gate_mechanism.py) | 定位block2通道组合在block3两次ReLU中产生的非加和效应。 |
| [coalition_context_mechanism.py](../source/scripts/coalition_context_mechanism.py) | 四状态交换：原C的变化、组外背景及其输出交互，全部沿用冻结干预。 |
| [cross_layer_mechanism.py](../source/scripts/cross_layer_mechanism.py) | 跨层交互的逐层计算、固定门控分解与中介恢复/移植。 |
| [diagnose_dsprites_channel_combinations.py](../source/scripts/diagnose_dsprites_channel_combinations.py) | 事后只读诊断通道联合非加和性，不新增模型查询或重新选择名单。 |
| [downstream_route_mechanism.py](../source/scripts/downstream_route_mechanism.py) | 定位原C的下游传递：实际单点阻断、联合阻断、保留恢复与残差支路。 |
| [dsprites_mechanism.py](../source/scripts/dsprites_mechanism.py) | 将旧跨层计算迁移到任意图片尺寸/类别数；不修改旧冻结源码。 |
| [dsprites_teacher.py](../source/scripts/dsprites_teacher.py) | dSprites受控迁移：保留原残差架构，原生64×64输入，重新训练3类输出。 |
| [dsprites_triple_rule.py](../source/scripts/dsprites_triple_rule.py) | 用低阶状态预测三组联合响应。训练和打分在独立入口中进行。 |
| [dsprites_workflow.py](../source/scripts/dsprites_workflow.py) | 从已有冻结阶段接续 dSprites 评估，或在全新目录中启动训练阶段。 |
| [external_analysis.py](../source/scripts/external_analysis.py) | Prespecified seed-level analysis and saved-array audit, without model fitting. |
| [external_experiment.py](../source/scripts/external_experiment.py) | Shared, frozen implementation for the six-seed external experiment. |
| [factor_mediator.py](../source/scripts/factor_mediator.py) | 只用开发集学习中介排序的因素规则，完全不调用评价图的网络。 |
| [finalize_dsprites_ar_factorial.py](../source/scripts/finalize_dsprites_ar_factorial.py) | 封存A/R四格实验，不修改九份历史交付。 |
| [finalize_dsprites_channel_gates.py](../source/scripts/finalize_dsprites_channel_gates.py) | 审计和报告完成后封存通道与ReLU研究，本轮交付不覆盖历史证据。 |
| [finalize_dsprites_coalition_context.py](../source/scripts/finalize_dsprites_coalition_context.py) | 封存新四状态交付；已完成旧目录及旧根说明保持原样。 |
| [finalize_dsprites_downstream_routes.py](../source/scripts/finalize_dsprites_downstream_routes.py) | 封存下游定位交付；不覆盖旧交付或自动发布GitHub。 |
| [finalize_dsprites_gradient.py](../source/scripts/finalize_dsprites_gradient.py) | 保存本轮交付快照并生成哈希清单；数值记录与旧交付不修改。 |
| [finalize_dsprites_joint_relu.py](../source/scripts/finalize_dsprites_joint_relu.py) | 封存本轮独立交付；仅CURRENT_RESEARCH.md的日期快照进入清单。 |
| [finalize_dsprites_joint_window.py](../source/scripts/finalize_dsprites_joint_window.py) | 封存联合必要性恢复；不改十一份历史交付及旧根文件。 |
| [finalize_dsprites_layer_transfer.py](../source/scripts/finalize_dsprites_layer_transfer.py) | 封存本轮层间传递结果；历史交付保持只读。 |
| [finalize_dsprites_residual_routes.py](../source/scripts/finalize_dsprites_residual_routes.py) | 封存中间组合效应路由研究，保留全部七轮旧交付。 |
| [finalize_dsprites_shape_donor.py](../source/scripts/finalize_dsprites_shape_donor.py) | 独立审计和报告完成后，封存本轮交付；历史实验原件不修改。 |
| [finalize_dsprites_triple_gates.py](../source/scripts/finalize_dsprites_triple_gates.py) | 全部独立核验和报告完成后封存三阶研究，保留前五轮交付。 |
| [finalize_dsprites_upstream_rescue.py](../source/scripts/finalize_dsprites_upstream_rescue.py) | 封存上游基准定位与恢复实验；十份历史交付只读保全。 |
| [finalize_joint_response.py](../source/scripts/finalize_joint_response.py) | Publish local navigation and hash the reviewed analysis artifacts. |
| [gradient_mechanism.py](../source/scripts/gradient_mechanism.py) | 冻结网络的状态梯度、局部线性修补预测及梯度变化诊断。 |
| [joint_relu_mechanism.py](../source/scripts/joint_relu_mechanism.py) | 多通道联合阈值的跨案例检验；只操作既有中间激活，不训练网络。 |
| [joint_response.py](../source/scripts/joint_response.py) | Small finite-intervention games: predictive fits and full-cube diagnostics. |
| [joint_window_measure.py](../source/scripts/joint_window_measure.py) | 整图重算冻结的39个端点，区分分解变化、实际传递、C消融与最终任务输出。 |
| [joint_window_mechanism.py](../source/scripts/joint_window_mechanism.py) | 在固定名单内找原C每一成员都必要的上游恢复，几何搜索不读取任务输出。 |
| [layer_transfer_mechanism.py](../source/scripts/layer_transfer_mechanism.py) | 源层四组组合到中间层再到输出的传递计算。 |
| [make_audit_figure.py](../source/scripts/make_audit_figure.py) | Standalone scientific figure of local replay accuracy and matched CPU timing. |
| [package_snapshot.py](../source/scripts/package_snapshot.py) | Package the completed research without changing existing research artifacts. |
| [plot_external_results.py](../source/scripts/plot_external_results.py) | Standalone figure; primary accuracy and descriptive matched-budget cost. |
| [plot_residual_results.py](../source/scripts/plot_residual_results.py) | Presentation-only scientific figure for the completed residual experiment. |
| [plot_scale_results.py](../source/scripts/plot_scale_results.py) | Scientific figure from completed, frozen evaluation and timing records. |
| [plot_structure_controls.py](../source/scripts/plot_structure_controls.py) | Plot the completed follow-up and the quality limitation of the GAP control. |
| [prepare_dsprites.py](../source/scripts/prepare_dsprites.py) | 从官方固定提交获取dSprites，安全解压数字数组并建立受控形状分类划分。 |
| [prepare_external_split.py](../source/scripts/prepare_external_split.py) | Use the upstream fixed grouping algorithm and seed; write outside upstream. |
| [prepare_fashion_mnist.py](../source/scripts/prepare_fashion_mnist.py) | Download official Fashion-MNIST bytes and audit them without running models. |
| [report_adaptive_mediator.py](../source/scripts/report_adaptive_mediator.py) | 生成第二轮中文报告、图表和交付索引；只读取已冻结、已独立核验的结果。 |
| [report_cross_layer_mechanism.py](../source/scripts/report_cross_layer_mechanism.py) | 把完成的跨层实验整理成中文报告、静态图和可导航入口。 |
| [report_dsprites_ar_factorial.py](../source/scripts/report_dsprites_ar_factorial.py) | 从A/R四格原始数据整理失效条件、逐层输出及教师级统计和科学图。 |
| [report_dsprites_channel_gates.py](../source/scripts/report_dsprites_channel_gates.py) | 从封存数组提炼通道配对、开关坐标、各阶项和反事实干预结果。 |
| [report_dsprites_coalition_context.py](../source/scripts/report_dsprites_coalition_context.py) | 生成中文四状态研究报告与可导出的科学图；所有数字来自已保存逐行结果。 |
| [report_dsprites_downstream_routes.py](../source/scripts/report_dsprites_downstream_routes.py) | 生成下游路径定位报告、通道/位置图和联合阻断后的正负余量图。 |
| [report_dsprites_factor_transfer.py](../source/scripts/report_dsprites_factor_transfer.py) | 将已完成且独立审计通过的因素迁移实验整理为中文报告和可追溯表格。 |
| [report_dsprites_gradient.py](../source/scripts/report_dsprites_gradient.py) | 从已审计原始数据提取梯度结论，生成中文报告、逐教师表和静态图。 |
| [report_dsprites_joint_relu.py](../source/scripts/report_dsprites_joint_relu.py) | 把冻结实验整理为逐案例证据表、按教师汇总和可独立保存的科学图。 |
| [report_dsprites_joint_window.py](../source/scripts/report_dsprites_joint_window.py) | 原组精确必要性恢复的证据表、共同参照筛查、任务输出和中文科学报告。 |
| [report_dsprites_layer_transfer.py](../source/scripts/report_dsprites_layer_transfer.py) | 从已独立核验的层间传递数组提取证据表和科研图。 |
| [report_dsprites_residual_routes.py](../source/scripts/report_dsprites_residual_routes.py) | 从已审计路由结果生成CSV证据与科研图；不重新挑选输入或干预名单。 |
| [report_dsprites_shape_donor.py](../source/scripts/report_dsprites_shape_donor.py) | 只读提炼已独立审计的形状配对实验，生成中文报告、逐教师表和图。 |
| [report_dsprites_triple_gates.py](../source/scripts/report_dsprites_triple_gates.py) | 读取封存三阶实验，提炼具体三元组、开关模式及正负贡献抵消。 |
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
| [run_dsprites_ar_factorial.py](../source/scripts/run_dsprites_ar_factorial.py) | 固定上轮所有名单，独立运行A0/A1×R0/R1；不扩大输入，不训练。 |
| [run_dsprites_channel_gates.py](../source/scripts/run_dsprites_channel_gates.py) | 通道配对与ReLU定位：init → discover → seal → evaluate → summarize。 |
| [run_dsprites_coalition_context.py](../source/scripts/run_dsprites_coalition_context.py) | 锁定既有解后执行四状态验证；不扩大输入、不训练、不新增输出驱动搜索。 |
| [run_dsprites_downstream_routes.py](../source/scripts/run_dsprites_downstream_routes.py) | 同输入下游定位：先锁定案例/规则，单点定位封存后再做联合阻断。 |
| [run_dsprites_factor_transfer.py](../source/scripts/run_dsprites_factor_transfer.py) | 开发数据拟合因素规则，冻结后在未用过的dSprites像素上评价。 |
| [run_dsprites_gradient.py](../source/scripts/run_dsprites_gradient.py) | 梯度变化接续实验：init → select → seal → evaluate → summarize。 |
| [run_dsprites_joint_relu.py](../source/scripts/run_dsprites_joint_relu.py) | 冻结协议→半强度选点选通道→封存→全强度，不增加输入或训练。 |
| [run_dsprites_joint_window.py](../source/scripts/run_dsprites_joint_window.py) | 同面板联合必要性恢复：协议→几何搜索冻结→整图输出；不训练不增加图像。 |
| [run_dsprites_layer_transfer.py](../source/scripts/run_dsprites_layer_transfer.py) | 层间传递阶段：init → evaluate → summarize；不扩大输入、不重新选名单。 |
| [run_dsprites_mechanism.py](../source/scripts/run_dsprites_mechanism.py) | dSprites迁移评估的阶段入口，先锁定规则和半强度名单，再评价。 |
| [run_dsprites_residual_routes.py](../source/scripts/run_dsprites_residual_routes.py) | 同批输入的中间组合残差路由：冻结→半强度定位→封存→全强度验证。 |
| [run_dsprites_shape_donor.py](../source/scripts/run_dsprites_shape_donor.py) | 形状配对与梯度传递：init → fit → freeze → select → seal → evaluate → summarize。 |
| [run_dsprites_triple_gates.py](../source/scripts/run_dsprites_triple_gates.py) | 三阶通道与门控接续：协议、半强度定位、统一封存、两强度评价、汇总。 |
| [run_dsprites_triple_rule.py](../source/scripts/run_dsprites_triple_rule.py) | 按协议、拟合、预测、实测的顺序封存。每一阶段可在文件边界续跑。 |
| [run_dsprites_upstream_rescue.py](../source/scripts/run_dsprites_upstream_rescue.py) | 同36图六模型的基准上游定位：先封存协议，再定位，最后评价救回与输出。 |
| [run_external.py](../source/scripts/run_external.py) | Stage-gated six-seed Fashion-MNIST experiment; never tune on official test. |
| [run_fashion_pilot.py](../source/scripts/run_fashion_pilot.py) | First fixed teacher, train/validation feasibility only; no official test access. |
| [run_joint_response.py](../source/scripts/run_joint_response.py) | Protocol-locked descriptive reanalysis of saved coalition responses. |
| [run_residual_controls.py](../source/scripts/run_residual_controls.py) | Frozen residual-block context experiment, including independent references. |
| [run_scale_controls.py](../source/scripts/run_scale_controls.py) | Locked size/position follow-up; no prior numerical artifact is overwritten. |
| [run_structure_controls.py](../source/scripts/run_structure_controls.py) | Frozen follow-up: activity controls and GAP/linear architecture controls. |
| [run_switch_controls.py](../source/scripts/run_switch_controls.py) | Frozen exact-switch-screening validation, timing, and new intervention batch. |
| [scale_context.py](../source/scripts/scale_context.py) | Variable source sizes, spatial positions, and a conservative affine fallback. |
| [shape_donor_mechanism.py](../source/scripts/shape_donor_mechanism.py) | 同条件形状状态替换、通道中介修补与沿路径梯度。 |
| [structure_runtime.py](../source/scripts/structure_runtime.py) | Activity-only selection controls and exact GAP/linear structural controls. |
| [summarize_dsprites_factors.py](../source/scripts/summarize_dsprites_factors.py) | dSprites 实验的描述性后处理：教师层面比较、因素分层、图像与结果图。 |
| [summarize_scale_results.py](../source/scripts/summarize_scale_results.py) | Descriptive synthesis and shared-validation consistency audit; no selection. |
| [switch_context.py](../source/scripts/switch_context.py) | Exact local responses after folding states stable over the intervention box. |
| [tabulate_dsprites_results.py](../source/scripts/tabulate_dsprites_results.py) | 将已审计的完整结果生成中文数值附表；文字结论在主报告中另行判断。 |
| [test_adaptive_mediator.py](../source/scripts/test_adaptive_mediator.py) | 新增选择器的已知答案与原网络核验；验证数据不进入模型选择。 |
| [test_cross_layer_mechanism.py](../source/scripts/test_cross_layer_mechanism.py) | 跨层机制的已知答案控制与原网络钩子核验，不训练或读取测试/校准集。 |
| [test_dsprites_ar_factorial.py](../source/scripts/test_dsprites_ar_factorial.py) | 协议前的已知四格答案和原模型核验；不寻找新的图像或坐标。 |
| [test_dsprites_channel_gates.py](../source/scripts/test_dsprites_channel_gates.py) | 新通道和门控计算的前置检查，只访问旧开发图，不访问新定位结果。 |
| [test_dsprites_coalition_context.py](../source/scripts/test_dsprites_coalition_context.py) | 数值预检：原forward、四状态边界、抵消恒等式和旧端点复现。 |
| [test_dsprites_downstream_routes.py](../source/scripts/test_dsprites_downstream_routes.py) | 新下游阻断的数值预检：原finish、边界核方向、联合控制与末层可加性。 |
| [test_dsprites_gradient.py](../source/scripts/test_dsprites_gradient.py) | 梯度实验的数值前置检查，使用旧验证图，不读取新的评价像素。 |
| [test_dsprites_joint_relu.py](../source/scripts/test_dsprites_joint_relu.py) | 选择前的已知答案和原网络核验；只用上一轮已指定的第一个案例。 |
| [test_dsprites_joint_window.py](../source/scripts/test_dsprites_joint_window.py) | 冻结协议前验证联合必要性逻辑、固定参照不变性和局部/原整图一致。 |
| [test_dsprites_layer_transfer.py](../source/scripts/test_dsprites_layer_transfer.py) | 层间传递前置检查：只读取旧开发集第一形状配对。 |
| [test_dsprites_mechanism.py](../source/scripts/test_dsprites_mechanism.py) | 迁移核验：旧十类计算的一致性、三类真实模型钩子及动态维度已知答案。 |
| [test_dsprites_residual_routes.py](../source/scripts/test_dsprites_residual_routes.py) | 新通道与开关路由的前置检查，使用旧面板预定案例。 |
| [test_dsprites_shape_donor.py](../source/scripts/test_dsprites_shape_donor.py) | 形状替换的前置数值检验，只读取旧开发图，不读取新评价输出。 |
| [test_dsprites_triple_gates.py](../source/scripts/test_dsprites_triple_gates.py) | 三阶前置核验：手算多项式、阈值抵消和旧开发图原网络正向。 |
| [test_dsprites_upstream_rescue.py](../source/scripts/test_dsprites_upstream_rescue.py) | 定位前检查：原网络锚点、单开关精确分数、联动端点及两层恒等式。 |
| [test_joint_response.py](../source/scripts/test_joint_response.py) | Meaningful numerical controls for the finite-game analysis. |
| [triple_gate_mechanism.py](../source/scripts/triple_gate_mechanism.py) | 56个空背景三元组、两层ReLU分解及冻结坐标的反事实计算。 |
| [upstream_rescue_measure.py](../source/scripts/upstream_rescue_measure.py) | 冻结原通道组C和下游点j，比较A单独修复与上游联动重算的阈值和输出。 |
| [upstream_rescue_mechanism.py](../source/scripts/upstream_rescue_mechanism.py) | 追踪block2如何移动构造基准A，并冻结仅依据基准修复量的上游名单。 |
| [verify_adaptive_mediator.py](../source/scripts/verify_adaptive_mediator.py) | 第二轮的独立数组核验器：不调用执行模块的差分、目标或指标函数。 |
| [verify_cross_layer_mechanism.py](../source/scripts/verify_cross_layer_mechanism.py) | 独立回读跨层实验：用稠密差分矩阵与向量内积重算全部指标。 |
| [verify_dsprites_ar_factorial_delivery.py](../source/scripts/verify_dsprites_ar_factorial_delivery.py) | 只读核验全部四格、输出分解、正文统计、交付来源和九份旧交付。 |
| [verify_dsprites_channel_gates_delivery.py](../source/scripts/verify_dsprites_channel_gates_delivery.py) | 只读核验交付、协议、选择名单；深核验覆盖全部冻结输入和旧交付。 |
| [verify_dsprites_coalition_context_delivery.py](../source/scripts/verify_dsprites_coalition_context_delivery.py) | 只读核验四状态交付：逐行表格、独立差分、统计分母、图和文件SHA。 |
| [verify_dsprites_downstream_routes_delivery.py](../source/scripts/verify_dsprites_downstream_routes_delivery.py) | 只读核验下游定位交付：原始数组、全部表格、统计分母和封存文件。 |
| [verify_dsprites_factor_delivery.py](../source/scripts/verify_dsprites_factor_delivery.py) | 只读核验本轮交付；不查询模型，不重写报告，也不更改审计记录。 |
| [verify_dsprites_gate_sites.py](../source/scripts/verify_dsprites_gate_sites.py) | 预定例子的逐坐标反事实补充：先封存规则，再逐点固定开关。 |
| [verify_dsprites_gradient_delivery.py](../source/scripts/verify_dsprites_gradient_delivery.py) | 只读核验本轮交付；可选核对全部旧交付与冻结输入。 |
| [verify_dsprites_joint_relu_delivery.py](../source/scripts/verify_dsprites_joint_relu_delivery.py) | 只读核验：来源/时序、全部证据表、汇总分母和八份旧交付。 |
| [verify_dsprites_joint_window_delivery.py](../source/scripts/verify_dsprites_joint_window_delivery.py) | 只读复核逐行证据、解选择引用、教师统计、共同开关恒等式与全部交付SHA。 |
| [verify_dsprites_layer_transfer_delivery.py](../source/scripts/verify_dsprites_layer_transfer_delivery.py) | 只读核验本轮文件、汇总与历史研究输入，不运行实验或写事件。 |
| [verify_dsprites_residual_routes_delivery.py](../source/scripts/verify_dsprites_residual_routes_delivery.py) | 只读核验新路由交付、固定时序、数值恒等式和七轮旧研究。 |
| [verify_dsprites_route_sixth_order.py](../source/scripts/verify_dsprites_route_sixth_order.py) | 六通道阈值例子的全部64子集，验证局部六阶激活与输出传递。 |
| [verify_dsprites_route_threshold.py](../source/scripts/verify_dsprites_route_threshold.py) | 固定例子的事后联合阈值验证，独立于半强度名单的主评价。 |
| [verify_dsprites_shape_donor_delivery.py](../source/scripts/verify_dsprites_shape_donor_delivery.py) | 只读核验形状配对实验，可选核对全部旧交付和冻结输入。 |
| [verify_dsprites_triple_gates_delivery.py](../source/scripts/verify_dsprites_triple_gates_delivery.py) | 只读核验三阶交付、协议、名单和表格；深核验包含全部历史输入。 |
| [verify_dsprites_upstream_rescue_delivery.py](../source/scripts/verify_dsprites_upstream_rescue_delivery.py) | 只读检查清单、冻结时序、CSV逐行证据、教师统计和全部历史文件。 |
| [verify_external_delivery.py](../source/scripts/verify_external_delivery.py) | Read-only experiment checks, then a separate September 11 delivery snapshot. |
| [verify_joint_response.py](../source/scripts/verify_joint_response.py) | Independent parity-basis and explicit-polynomial audit of the saved analysis. |
| [verify_residual_delivery.py](../source/scripts/verify_residual_delivery.py) | Preserve prior evidence and verify the completed residual study delivery. |
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

# 全部研究源码索引

原项目 66 个接续脚本与 9 个历史依赖，共 75 个。根目录另有发布专用示例和核验入口。

| 文件 | 原文档字符串中的用途 |
|---|---|
| [adaptive_mediator.py](../source/scripts/adaptive_mediator.py) | 第二轮：在相同候选空间内比较固定/逐输入选择与梯度/有限干预评分。 |
| [audit_dsprites_data.py](../source/scripts/audit_dsprites_data.py) | 独立检查 dSprites 来源行、标签、因素分组与跨划分像素重合。 |
| [audit_dsprites_results.py](../source/scripts/audit_dsprites_results.py) | 独立回读 dSprites 原始输出，重算交互/中介分数，并检查冻结与历史完整性。 |
| [audit_external_head_activity.py](../source/scripts/audit_external_head_activity.py) | Post-test descriptive diagnostic from already frozen training fits only. |
| [audit_residual_affine_runtime.py](../source/scripts/audit_residual_affine_runtime.py) | Post-test, explicitly separate audit of an algebraically fused affine control. |
| [audit_structure_interpretation.py](../source/scripts/audit_structure_interpretation.py) | Post-result explanation checks; no fitting, selection, or result replacement. |
| [audit_upstream.py](../source/scripts/audit_upstream.py) | Re-score/replay v0.7 without modifying its imported archive contents. |
| [benchmark_heads.py](../source/scripts/benchmark_heads.py) | Descriptive matched-backend diagnostic, separate from historical timings. |
| [benchmark_pilot.py](../source/scripts/benchmark_pilot.py) | CPU cost diagnostic on four pilot validation images, after other jobs finish. |
| [cross_layer_mechanism.py](../source/scripts/cross_layer_mechanism.py) | 跨层交互的逐层计算、固定门控分解与中介恢复/移植。 |
| [dsprites_mechanism.py](../source/scripts/dsprites_mechanism.py) | 将旧跨层计算迁移到任意图片尺寸/类别数；不修改旧冻结源码。 |
| [dsprites_teacher.py](../source/scripts/dsprites_teacher.py) | dSprites受控迁移：保留原残差架构，原生64×64输入，重新训练3类输出。 |
| [dsprites_workflow.py](../source/scripts/dsprites_workflow.py) | 从已有冻结阶段接续 dSprites 评估，或在全新目录中启动训练阶段。 |
| [external_analysis.py](../source/scripts/external_analysis.py) | Prespecified seed-level analysis and saved-array audit, without model fitting. |
| [external_experiment.py](../source/scripts/external_experiment.py) | Shared, frozen implementation for the six-seed external experiment. |
| [finalize_joint_response.py](../source/scripts/finalize_joint_response.py) | Publish local navigation and hash the reviewed analysis artifacts. |
| [joint_response.py](../source/scripts/joint_response.py) | Small finite-intervention games: predictive fits and full-cube diagnostics. |
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
| [report_joint_response.py](../source/scripts/report_joint_response.py) | Render the completed, independently verified descriptive analysis. |
| [report_switch_results.py](../source/scripts/report_switch_results.py) | Presentation derived only from completed exact-switch experiment records. |
| [research_runtime.py](../source/scripts/research_runtime.py) | Matched CPU runtimes for the unchanged v0.7 estimators. |
| [residual_context.py](../source/scripts/residual_context.py) | Local finite interventions inside the last branch of a residual CNN. |
| [residual_teacher.py](../source/scripts/residual_teacher.py) | Small residual CNN and locked train/validation-only development stage. |
| [run_adaptive_mediator.py](../source/scripts/run_adaptive_mediator.py) | 第二轮的显式阶段入口：开发、锁定规则、半强度选择、锁定名单、完整强度评价。 |
| [run_cross_layer_mechanism.py](../source/scripts/run_cross_layer_mechanism.py) | 跨层机制第一轮的阶段入口：先开发冻结，再评价和只读核验。 |
| [run_dsprites_mechanism.py](../source/scripts/run_dsprites_mechanism.py) | dSprites迁移评估的阶段入口，先锁定规则和半强度名单，再评价。 |
| [run_external.py](../source/scripts/run_external.py) | Stage-gated six-seed Fashion-MNIST experiment; never tune on official test. |
| [run_fashion_pilot.py](../source/scripts/run_fashion_pilot.py) | First fixed teacher, train/validation feasibility only; no official test access. |
| [run_joint_response.py](../source/scripts/run_joint_response.py) | Protocol-locked descriptive reanalysis of saved coalition responses. |
| [run_residual_controls.py](../source/scripts/run_residual_controls.py) | Frozen residual-block context experiment, including independent references. |
| [run_scale_controls.py](../source/scripts/run_scale_controls.py) | Locked size/position follow-up; no prior numerical artifact is overwritten. |
| [run_structure_controls.py](../source/scripts/run_structure_controls.py) | Frozen follow-up: activity controls and GAP/linear architecture controls. |
| [run_switch_controls.py](../source/scripts/run_switch_controls.py) | Frozen exact-switch-screening validation, timing, and new intervention batch. |
| [scale_context.py](../source/scripts/scale_context.py) | Variable source sizes, spatial positions, and a conservative affine fallback. |
| [structure_runtime.py](../source/scripts/structure_runtime.py) | Activity-only selection controls and exact GAP/linear structural controls. |
| [summarize_dsprites_factors.py](../source/scripts/summarize_dsprites_factors.py) | dSprites 实验的描述性后处理：教师层面比较、因素分层、图像与结果图。 |
| [summarize_scale_results.py](../source/scripts/summarize_scale_results.py) | Descriptive synthesis and shared-validation consistency audit; no selection. |
| [switch_context.py](../source/scripts/switch_context.py) | Exact local responses after folding states stable over the intervention box. |
| [tabulate_dsprites_results.py](../source/scripts/tabulate_dsprites_results.py) | 将已审计的完整结果生成中文数值附表；文字结论在主报告中另行判断。 |
| [test_adaptive_mediator.py](../source/scripts/test_adaptive_mediator.py) | 新增选择器的已知答案与原网络核验；验证数据不进入模型选择。 |
| [test_cross_layer_mechanism.py](../source/scripts/test_cross_layer_mechanism.py) | 跨层机制的已知答案控制与原网络钩子核验，不训练或读取测试/校准集。 |
| [test_dsprites_mechanism.py](../source/scripts/test_dsprites_mechanism.py) | 迁移核验：旧十类计算的一致性、三类真实模型钩子及动态维度已知答案。 |
| [test_joint_response.py](../source/scripts/test_joint_response.py) | Meaningful numerical controls for the finite-game analysis. |
| [verify_adaptive_mediator.py](../source/scripts/verify_adaptive_mediator.py) | 第二轮的独立数组核验器：不调用执行模块的差分、目标或指标函数。 |
| [verify_cross_layer_mechanism.py](../source/scripts/verify_cross_layer_mechanism.py) | 独立回读跨层实验：用稠密差分矩阵与向量内积重算全部指标。 |
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
| [acquire_fashion.py](../source/upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py) | Acquire official Fashion-MNIST binaries, verify published MD5 and local SHA-256. |
| [analyze_and_verify.py](../source/upstream/CAT_MT_phase7_v0.7/code/analyze_and_verify.py) | 历史依赖；结合调用处阅读。 |
| [context_module.py](../source/upstream/CAT_MT_phase7_v0.7/code/context_module.py) | Restricted background selection. Original weights are retained; train-only masks |
| [prepare_external.py](../source/upstream/CAT_MT_phase7_v0.7/code/prepare_external.py) | Hard-gated Fashion-MNIST preparation; no missing-file fallback to another task. |
| [run_development.py](../source/upstream/CAT_MT_phase7_v0.7/code/run_development.py) | 历史依赖；结合调用处阅读。 |
| [runtime_backend_check.py](../source/upstream/CAT_MT_phase7_v0.7/code/runtime_backend_check.py) | Post-primary backend control: same selected-background formula in Torch. |
| [write_report.py](../source/upstream/CAT_MT_phase7_v0.7/code/write_report.py) | 历史依赖；结合调用处阅读。 |
| [confirmatory_core.py](../source/upstream/CAT_MT_phase7_v0.7/vendor/confirmatory_core.py) | 历史依赖；结合调用处阅读。 |
| [local_modules.py](../source/upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py) | Phase VI. Exact context-conditioned local compilation and conservative interaction groups. |

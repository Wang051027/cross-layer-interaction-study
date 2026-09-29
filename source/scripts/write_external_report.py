# 中文阅读说明｜用途：生成早期六教师实验的中文报告和根目录状态。
# 数据流：读取既有汇总；这是历史报告生成器，会写根目录状态，不应覆盖当前版本。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Human-readable report from the completed frozen experiment only."""
from __future__ import annotations
import csv
import json
import statistics
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
RUN=BASE / "runs/fashion_external_20260911"
SEEDS=list(range(202609071,202609077))
NAMES={"greedy":"背景选择","weight":"权重范数","pca_fused":"融合 PCA"}


# 用途：读取 UTF-8 JSON，返回其中的协议、配置或结果对象。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read_json(path):return json.loads(path.read_text(encoding="utf-8"))
# 用途：读取 CSV 表格为记录列表，供报告查找和汇总。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：生成早期六教师实验的中文报告和根目录状态。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    primary=read_json(RUN / "analysis/primary_statistics.json")
    verification=read_json(RUN / "analysis/verification.json")
    completion=read_json(RUN / "test/completion.json")
    assert verification["status"]=="passed" and completion["status"]=="completed"
    summary=read_csv(RUN / "analysis/test_summary.csv")
    selected=read_csv(RUN / "analysis/selected_summary.csv")
    cost=read_csv(RUN / "analysis/cost_summary.csv")
    cost_seeds=read_csv(RUN / "analysis/cost_and_accuracy_seeds.csv")
    results=read_csv(RUN / "test/all_results.csv")
    classification=read_csv(RUN / "test/teacher_classification.csv")
    activity=read_csv(BASE / "audit/head_activity_20260911.csv")
    choices=read_json(RUN / "frozen_before_test.json")["validation_choices"]
    # 用途：按层、方法和预算定位报告需要的准确性汇总。
    # 参数：layer、method、k；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：next((r for r in summary if r['layer'] == str(layer) and r['alpha'] == '1.0' and (r['me…。
    def find_summary(layer,method,k):
        return next(r for r in summary if r["layer"]==str(layer) and r["alpha"]=="1.0" and r["method"]==method and r["k"]==str(k))
    # 用途：按层、方法、预算规则和成本口径定位保存的成本记录。
    # 参数：layer、method、rule、scope；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：next((r for r in cost if r['layer'] == str(layer) and r['method'] == method and (r['rul…。
    def find_cost(layer,method,rule,scope="from_image"):
        return next(r for r in cost if r["layer"]==str(layer) and r["method"]==method and r["rule"]==rule and r["queries"]=="256" and r["workload"]==scope)
    fixed_rows=[]
    for layer in (1,2):
        for k in (16,32):
            values=[find_summary(layer,m,k) for m in NAMES]
            fixed_rows.append(f"| L{layer} | {k}/64 | "+" | ".join(f"{100*float(r['nrmse_mean']):.2f}% ± {100*float(r['nrmse_sd']):.2f}%" for r in values)+f" | {values[0]['seeds_meet_10pct']}/6 |")
    raw_primary=[]
    for seed,difference in zip(SEEDS,primary["differences"]):
        pair=[next(r for r in results if r["seed"]==str(seed) and r["layer"]=="1" and r["alpha"]=="1.0" and r["k"]=="16" and r["method"]==method) for method in ("greedy","weight")]
        raw_primary.append(f"| {seed} | {float(pair[0]['nrmse'])*100:.2f}% | {float(pair[1]['nrmse'])*100:.2f}% | {difference*100:+.2f} |")
    adaptive_rows=[]
    for layer in (1,2):
        for method in NAMES:
            r=next(r for r in selected if r["layer"]==str(layer) and r["method"]==method and r["alpha"]=="1.0")
            c=find_cost(layer,method,"validation_selected")
            adaptive_rows.append(f"| L{layer} | {NAMES[method]} | {r['k_values']} | {100*float(r['nrmse_mean']):.2f}% | {r['seeds_meet_10pct']}/6 | {float(c['paired_speedup_mean']):.3f}× | {100*float(c['background_reduction_mean']):.1f}% |")
    time_rows=[]
    for layer in (1,2):
        for method in NAMES:
            fresh=find_cost(layer,method,"fixed32")
            cached=find_cost(layer,method,"fixed32","cached")
            time_rows.append(f"| L{layer} | {NAMES[method]} | {float(fresh['milliseconds_mean']):.3f} | {float(fresh['paired_speedup_mean']):.3f} ± {float(fresh['paired_speedup_sd']):.3f} | {float(cached['paired_speedup_mean']):.3f}× | {fresh['seeds_meet_10pct']}/6 |")
    teacher_rows=[]
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        t=read_json(RUN / f"seed_{seed}/teacher_summary.json")
        c=next(r for r in classification if r["seed"]==str(seed))
        teacher_rows.append(f"| {seed} | {t['best_epoch']} | {t['best_validation_accuracy']*100:.2f}% | {float(c['accuracy'])*100:.2f}% |")
    control_rows=[]
    for method,k,label in (("random",16,"随机选择 16"),("random",32,"随机选择 32"),
                           ("boundary_only",0,"精确边界＋均值门控"),("mean_head",0,"均值头部"),("pool_ridge",0,"池化响应 Ridge")):
        values=[find_summary(layer,method,k) for layer in (1,2)]
        control_rows.append("| "+label+" | "+" | ".join(f"{100*float(r['nrmse_mean']):.2f}% ± {100*float(r['nrmse_sd']):.2f}%" for r in values)+" |")
    activity_rows=[]
    for r in activity:
        if r["layer"]=="1":
            activity_rows.append(f"| {r['seed']} | {r['positive_on_some_training_inputs']} | {r['never_positive_on_training_inputs']} |")
    p=primary["sign_flip_two_sided_p"]
    inference=("主比较在预先指定的双侧符号翻转检验下达到 0.05 阈值。" if p<.05 else
               "主比较在预先指定的双侧符号翻转检验下未达到 0.05 阈值。")
    first16=find_summary(1,"greedy",16)
    first32=find_summary(1,"greedy",32)
    calibrated=[r for r in selected if r["method"]=="greedy" and r["alpha"]=="1.0"]
    match_time=[r for r in cost_seeds if r["method"]=="greedy" and r["rule"]=="validation_selected" and r["workload"]=="from_image" and r["queries"]=="256" and r["meets_test_10pct"]=="True" and r["joint_pipeline_amortization_input_bundles"]]
    amort=[int(r["joint_pipeline_amortization_input_bundles"]) for r in match_time]
    amort_text=(f"在测试误差达标且计时点估计更快的背景选择设置中，用本次完整开发流程耗时除以每个输入的在线节省，得到约 {min(amort):,}–{max(amort):,} 个输入批次的摊销量；每批为同一输入的 256 个组合请求。" if amort else "本轮没有可同时满足该误差条件及正在线节省的设置用于计算此项摊销。")
    report=f"""# 六种子外部测试：结果与下一步

2026-09-11。此报告接续 [首轮复核与实验](复核与首轮外部实验_2026-09-10.md)。

**原六种子实验已完成：32/64 的公平比较、其余五个教师、测试前冻结、官方测试及数组回读审计均已执行。** 第一层主预算 16/64 的背景选择平均 NRMSE 为 {100*float(first16['nrmse_mean']):.2f}%，达到总体 10% 门槛的教师有 {first16['seeds_meet_10pct']}/6；固定 32/64 时均值为 {100*float(first32['nrmse_mean']):.2f}%，达标 {first32['seeds_meet_10pct']}/6。

原主比较的平均差为 **{100*primary['mean_difference']:+.2f} 个百分点**（背景选择减权重范数，负值有利于背景选择），双侧精确符号翻转 p={p:.5f}。{inference} 相对基线的误差改善、10% 误差门槛和计算收益需要分别判断。

## 实验保持了哪些约束

首个教师和其已拟合参数沿用 9 月 10 日的版本，剩余种子为 202609072–202609076。架构、数据划分、优化器、学习率、训练轮数上限、早停及验证交叉熵选择规则保持原样，没有舍弃或替换种子。

在新种子训练前，登记了执行细节、k=32 次要分析、融合 PCA 和各方法的验证预算选择规则。所有教师、选择器、预算、代码、数据哈希及计时在首次对官方测试集执行模型评价之前冻结；原主终点仍是 **L1、完全置零、k=16、背景选择减权重范数选择**。9 月 10 日对测试文件的哈希、标签计数和精确重复检查不涉及模型评价。

每个教师在相同的 256 张预留官方测试图像上，分别检查两层和两种干预强度，每个输入独立执行全部 256 个组合的原网络尾部。主要误差统计 219 个三阶及以上组合，分母是对应真实 logit 变化的平方和。共执行 **1,572,864 个干预输出向量**；这些向量共享输入和教师，不是同样数量的独立样本。

教师分类准确率另在完整 10,000 张官方测试图上记录。5000 张校准图像仍未参与模型评价。本轮干预分数针对固定的 256 张测试图像，不代表已经对全部 10,000 张做组合干预，也不代表跨网络结构泛化。

## 固定预算的测试结果

以下为六个教师的平均 NRMSE ± 种子间标准差，完全置零，数值越低越好。标准差不是置信区间。

| 层 | 预算 | 背景选择 | 权重范数 | 融合 PCA | 背景选择达标数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(fixed_rows)}

PCA 使用相同数量的编码值。它的线性编码器已与前一全连接层融合，完整近似头部的重构及非线性响应仍被计入。k=32 的比较在本轮执行前登记为次要分析，没有取代较难达到误差门槛的 k=16 主设置。

![六种子误差与成本](six_seed_external_results.png)

图中的点为各教师，误差条为种子间标准差。成本图为单独采样的验证输入，未使用测试误差来挑计时样本。

其余固定控制也全部保留，以下为同一完全置零、高阶组合上的平均误差 ± SD；表中各控制在两层均为 0/6 达到 10% 门槛。

| 控制 | L1 | L2 |
|---|---:|---:|
{chr(10).join(control_rows)}

## 为什么部分 k=32 结果几乎精确

测试后的描述性检查发现，六个教师的有效头部规模差异很大。以下只回读已冻结拟合文件中 512 张训练上下文的 ReLU 门控频率，没有重新拟合或读取新输入。

| 教师种子 | 训练上下文中曾激活的头部单元 | 从未激活的单元 |
|---|---:|---:|
{chr(10).join(activity_rows)}

后三个教师只有 32、27、29 个单元在这些训练上下文中曾经激活；背景选择和权重范数选择的前 32 项均覆盖了这些单元，两种方法在测试上的 k=32 误差也都接近浮点精度。这与大量不活跃头部单元造成的冗余一致，不能把这些近乎精确的结果全部解释为背景选择独有的优势。训练样本上的不活跃不等于对所有输入和干预都永久不活跃。此项为观察结果后开展的诊断，不增加确认性统计结论。原始计数见 [头部活动核查](../audit/head_activity_20260911.json)。

## 原主比较与统计边界

| 教师种子 | 背景选择 k16 | 权重范数 k16 | 差值（百分点） |
|---|---:|---:|---:|
{chr(10).join(raw_primary)}

均值差的 95% Student t 区间为 **[{100*primary['t95_interval'][0]:+.2f}, {100*primary['t95_interval'][1]:+.2f}] 个百分点**。精确符号翻转枚举了全部 64 种符号分配，采用双侧平均差统计量；主 p 值及区间与 SciPy 独立实现交叉核对。配对符号翻转依赖零假设下的符号交换／对称条件，t 区间依赖相应近似；这里仅有 6 个种子，统计能力有限，推断条件于这组固定测试输入。[SciPy 配对置换说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)、[配对 t 区间接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)。

只有上述一项是原定主推断。其余层、预算、强度、PCA 及计时比较均按描述性结果报告。即使主比较显著，也不能据此推导某个输入一定满足 10% 误差要求，或模型会在其他结构、数据集上保持相同收益。

## 按验证集选择预算后，测试上是否仍然达标

每个教师、层、方法均从预先固定的 `0,1,2,4,8,16,32,64` 中选择验证集 NRMSE≤10% 的最小预算，未找到时选 64；同一选择用于两个测试强度。下表为完全置零。预算序列按种子 202609071→202609076 排列，实际输出规模与误差一起展示。

| 层 | 方法 | 验证集选定预算 | 测试平均误差 | 测试达标 | 从图像开始加速 | 平均背景减少 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(adaptive_rows)}

64/64 的方法在运行时使用相同的局部精确实现，汇总时共享其计时观测，加速比记为 1；不把相同实现的微小计时噪声解释为压缩收益。预算由验证集选定后未再根据测试结果更改。总体误差门槛仍不构成逐输入保证，逐输入第 95 百分位及近零效果输入数量保存在明细中。

## 达到误差要求时的成本

所有计时采用 CPU float64、单线程，Q=37／256；覆盖从原始图像开始、源层已计算、完整背景已缓存三种场景。每个教师—层使用前 4 个固定验证输入，预热 3 次，15 轮随机方法顺序，每轮 3 次调用；先取输入内中位数，再对教师汇总。计时与 GPU 训练串行执行，模型训练使用 float32。

下表为固定 k=32、Q=256。加速比为同教师精确耗时／该方法耗时，再对六个教师取均值；大于 1 表示该次计时点估计更快。它不是速度保证。

| 层 | 方法 | 从图像开始 ms | 配对加速均值 ± SD | 已有缓存加速 | 测试误差达标数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(time_rows)}

总背景计数包含完整的卷积／池化边界及头部状态；另有 8 个源值。具体字节数针对实际执行数组，未把它宣称为 Python 进程或 GPU 的峰值内存。不同日期、线程、后端、精度及缓存场景之间的绝对耗时不直接比较。四个计时输入和毫秒级差异只提供成本诊断。

{amort_text} 这个分子包含所有控制方法、验证和部分文件写出，是本次完整实验流程的耗时，并非部署所需的最小拟合成本。单独的拟合、参数构建和全部原始计时均已保存。对在线没有正节省或退回完整精确执行的设置，不报告有限摊销数。

## 教师本身的分类表现

| 种子 | 选中轮次 | 验证准确率 | 官方测试准确率 |
|---|---:|---:|---:|
{chr(10).join(teacher_rows)}

六教师测试准确率平均 {statistics.mean(float(r['accuracy']) for r in classification)*100:.2f}%。这些指标用于说明教师质量，不能代替干预预测误差。

## 核查与下一步

保存数组独立回读重算 {verification['rescored_arrays']} 组指标，最大 NRMSE 差 {verification['max_nrmse_difference']:.3g}；所有冻结模型和拟合参数在测试后保持原哈希。局部精确执行与真实尾部的最大绝对差为 {completion['max_exact_error']:.3g}。每个案例的预测承诺先于参考结果承诺，源码快照、事件记录和哈希均保留。

建议下一阶段聚焦两项工作：

1. **结构控制。** 先增加按训练激活情况筛选的简单基线，分离不活跃单元带来的冗余与残差选择的额外贡献；增加 GAP＋线性读出的负对照，检查收益是否依赖当前的非线性全连接头。随后再决定是否扩展到残差块内部。
2. **误差约束下的预算与基线。** 研究更细预算或有明确校准目标的预算规则，并比较利用输出响应信息的低秩方法。任何进一步改法都要标明开发性质，使用新的预留评价条件验证，避免在本次已读取的测试输出上反复优化后仍声称独立确认。

本轮已经给出一个可核查的外部实验结果。论文或申请项目应同时呈现准确率、误差、总状态、完整成本、种子差异和适用结构，暂不作语义机制、分布无关可靠性或特定期刊录用水平的主张。

详细证据：[执行前协议](../runs/fashion_external_20260911/execution_protocol.json)、[测试前冻结清单](../runs/fashion_external_20260911/frozen_before_test.json)、[主统计](../runs/fashion_external_20260911/analysis/primary_statistics.json)、[逐种子误差与成本](../runs/fashion_external_20260911/analysis/cost_and_accuracy_seeds.csv)、[回读核查](../runs/fashion_external_20260911/analysis/verification.json)。运行入口见 [RUNBOOK](../RUNBOOK.md)。
"""
    (BASE / "reports/六种子外部测试与推进建议_2026-09-11.md").write_text(report,encoding="utf-8")
    state=read_json(BASE / "RESEARCH_STATUS.json")
    state.update(date="2026-09-11",external_teacher_training_runs=6,external_teacher_seeds=SEEDS,
        external_intervention_stage="six-seed official test completed",external_official_test_runs=6,
        external_official_test_batches=1,external_model_layer_cases=12,official_test_model_access=True,
        calibration_model_access=False,original_primary_contrast_unchanged=True,
        current_run="runs/fashion_external_20260911",primary_mean_difference=primary["mean_difference"],
        primary_sign_flip_p=p,primary_t95_interval=primary["t95_interval"],
        next_steps=["activity-aware simple baseline and GAP plus linear-head architecture control","finer or calibrated budgets and response-aware low-rank controls with fresh evaluation conditions"],
        latest_report="reports/六种子外部测试与推进建议_2026-09-11.md")
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (BASE / "RESEARCH_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

# 中文阅读说明｜用途：组织活动筛选和 GAP 结构对照报告。
# 数据流：从既有结果生成表格和结论文本；属于历史文档生成器。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Presentation-only report generator for the completed frozen controls."""
from pathlib import Path
import csv
import json
import statistics as st

BASE=Path(__file__).resolve().parents[1]
RUN=BASE / "runs/fashion_structure_20260911"
SEEDS=list(range(202609071,202609077))
LABELS={"greedy":"训练残差贪心选择","weight":"权重范数","pca_fused":"融合 PCA",
        "random":"原随机选择","active_weight":"活跃优先＋权重范数","gate_weight":"门控不确定性×权重",
        "activity":"激活频率","active_random_mean":"活跃单元随机（4 次均值）"}


# 用途：读取 UTF-8 JSON，返回其中的协议、配置或结果对象。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read_json(path):return json.loads(path.read_text(encoding="utf-8"))


# 用途：读取 CSV 表格为记录列表，供报告查找和汇总。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：组织活动筛选和 GAP 结构对照报告。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    audit=read_json(RUN / "analysis/verification.json")
    complete=read_json(RUN / "test/completion.json")
    assert audit["status"]=="passed" and complete["status"]=="completed"
    summary=read_csv(RUN / "analysis/summary.csv")
    seeds=read_csv(RUN / "analysis/seed_nrmse.csv")
    differences=read_csv(RUN / "analysis/paired_differences.csv")
    timings=read_csv(RUN / "timing/summary.csv")
    gaps=read_csv(RUN / "test/gap_classification.csv")
    old_classes=read_csv(BASE / "runs/fashion_external_20260911/test/teacher_classification.csv")
    interpretation=read_json(BASE / "audit/structure_interpretation_20260911.json")
    # 用途：筛选满足当前表格条件的结果行；并查集中的同名函数另有专门说明。
    # 参数：method、layer、k、architecture、stratum；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：next((r for r in summary if r['stratum'] == stratum and r['architecture'] == architectu…。
    def find(method,layer,k=16,architecture="cnn",stratum="all"):
        return next(r for r in summary if r["stratum"]==stratum and r["architecture"]==architecture and r["layer"]==str(layer)
            and r["method"]==method and r["k"]==str(k) and r["alpha"]=="1.0")
    # 用途：把结果记录格式化为可读的表格单元，缺失值按本报告规则展示。
    # 参数：r；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：f"{100 * float(r['nrmse_mean']):.2f}% ± {100 * float(r['nrmse_sd']):.2f}%"。
    def cell(r):return f"{100*float(r['nrmse_mean']):.2f}% ± {100*float(r['nrmse_sd']):.2f}%"
    fixed=[]
    for method,label in LABELS.items():
        fixed.append("| "+label+" | "+" | ".join(cell(find(method,layer,k)) for layer,k in ((1,16),(1,32),(2,16),(2,32)))+" |")
    strata=[]
    for method in ("greedy","weight","gate_weight","active_random_mean","activity"):
        strata.append("| "+LABELS[method]+" | "+" | ".join(cell(find(method,layer,k,stratum="more_than_32_training_active_head_units")) for layer,k in ((1,16),(1,32),(2,16),(2,32)))+" |")
    contrasts=[]
    for layer in (1,2):
        for k in (16,32):
            for method in ("weight","gate_weight","active_random_mean"):
                d=[float(r["greedy_minus_control"])*100 for r in differences if r["layer"]==str(layer) and r["k"]==str(k) and r["alpha"]=="1.0" and r["control"]==method]
                contrasts.append(f"| L{layer} | {k} | {LABELS[method]} | {st.mean(d):+.2f} | {sum(v < -1e-10 for v in d)}／{sum(abs(v) <= 1e-10 for v in d)}／{sum(v > 1e-10 for v in d)} |")
    identical=0
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        for row in read_json(RUN / f"cnn_seed_{seed}/completion.json")["diagnostics"]:
            identical+=int(row["active_weight_equals_weight_at16"] and row["active_weight_equals_weight_at32"])
    gap_rows=[]
    for layer in (1,2,3):
        state=[read_json(RUN / f"test/gap_{seed}_L{layer}/completion.json")["boundary_scalars"] for seed in SEEDS]
        exact=find("exact",layer,0,"gap")
        gap_rows.append(f"| L{layer} | {min(state)}–{max(state)} | 0 | {100*float(exact['nrmse_mean']):.3g}% | {cell(find('mean_boundary',layer,0,'gap'))} | {cell(find('original_region',layer,0,'gap'))} |")
    class_rows=[]
    for r in gaps:
        t=read_json(RUN / f"gap_seed_{r['seed']}/teacher_summary.json")
        old=next(v for v in old_classes if v["seed"]==r["seed"])
        class_rows.append(f"| {r['seed']} | {t['best_epoch']} | {100*t['best_validation_accuracy']:.2f}% | {100*float(r['accuracy']):.2f}% | {100*float(old['accuracy']):.2f}% |")
    costs=[]
    for layer in (1,2):
        for method in ("greedy","weight","gate_weight","activity","pca_fused"):
            values={}
            for workload in ("from_image","cached"):
                ms=[];ratios=[]
                for seed in SEEDS:
                    one=next(float(r["seconds"]) for r in timings if r["architecture"]=="cnn" and r["seed"]==str(seed) and r["layer"]==str(layer) and r["workload"]==workload and r["method"]==method+"_k32")
                    exact=next(float(r["seconds"]) for r in timings if r["architecture"]=="cnn" and r["seed"]==str(seed) and r["layer"]==str(layer) and r["workload"]==workload and r["method"]=="exact_k64")
                    ms.append(one*1000);ratios.append(exact/one)
                values[workload]=(st.mean(ms),st.mean(ratios),st.stdev(ratios))
            fresh,cached=values["from_image"],values["cached"]
            costs.append(f"| L{layer} | {LABELS[method]} | {fresh[0]:.3f} | {fresh[1]:.3f} ± {fresh[2]:.3f} | {cached[1]:.3f} |")
    c1=find("greedy",1);c2=find("greedy",2);gate1=find("gate_weight",1);gate2=find("gate_weight",2)
    gap_accuracy=st.mean(float(r["accuracy"]) for r in gaps)*100
    cnn_accuracy=st.mean(float(r["accuracy"]) for r in old_classes)*100
    report=fr"""# 背景收益来自哪里：激活基线与线性读出对照

2026-09-11。接续 [原六种子正式测试](六种子外部测试与推进建议_2026-09-11.md)。

本轮完成原六个 CNN 教师的新增基线、六个新 GAP＋线性读出教师、三个 GAP 干预位置、测试前冻结、完整成本计时和保存数组审计。测试共有 **{complete['cases']} 个模型—层设置、{complete['full_tail_output_vectors']:,} 个实际尾网络干预输出向量**；这些不是独立重复数量。

在新的固定干预输入批次上，原 CNN 的贪心选择 k=16 在 L1／L2 的平均 NRMSE 为 **{100*float(c1['nrmse_mean']):.2f}%／{100*float(c2['nrmse_mean']):.2f}%**；简单门控加权基线为 **{100*float(gate1['nrmse_mean']):.2f}%／{100*float(gate2['nrmse_mean']):.2f}%**。下文保留全部基线、预算和种子分组，用于判断额外选择收益。

## 1. 这次解决哪个疑问

上一轮发现，后三个 CNN 教师在 512 张训练上下文中只有 32、27、29 个头部单元曾激活，k=32 时贪心与权重选择都近乎精确。本轮区分两件事：排除不活跃单元是否足以得到同样结果；有限干预的背景需求是否依赖干预之后还剩哪些非线性运算。

本轮是看过旧结果后设计的后续诊断。原模型和拟合器复用且保持原哈希；新增方法只依据训练激活与原权重，不使用新批次参考结果选预算。正式主实验的 p=0.03125 仍属于旧的原主比较，本轮没有增加或替换确认性 p 值。

## 2. 数据与冻结

方法训练上下文、验证上下文、教师训练／验证划分保持原样。新增组合干预测试固定为官方测试索引 **256–511**，与原批次 0–255 不重叠。但原 CNN 已在第一轮对全部 10,000 张计算过分类输出，因此本次是同一数据集上的新增干预评价，不能称为新外部数据集或完全未访问的测试池。5000 张校准图像仍未用于模型评价。

新模型训练前冻结代码、种子、方法定义、固定 16/32 预算、随机重复种子、评价输入及分析规则；全部教师、选择顺序、窗口均值和计时完成后再次冻结。每个模型—层先保存预测及哈希，再独立执行原网络尾部得到干预参考，没有测试后修改参数。

每个教师、每层只研究一个由训练激活预先选定的 8 坐标局部源集合；不是遍历全部空间位置和全部可能模块。下文表格聚焦完全置零，减半干预的全部结果也保存在汇总文件中。

## 3. 加入了哪些简单控制

- **活跃优先＋权重范数：**训练上下文中曾开启的单元排在前面，内部仍按原权重范数顺序。
- **激活频率：**只按训练门控频率 \(g\) 降序，稳定处理并列值。
- **门控不确定性×权重：**按 \(g(1-g)\|W_j\|_2\|U_{{:j}}\|_2\) 排序，不使用训练干预响应。
- **活跃单元随机：**先在训练样本中曾激活的单元里随机排，再排其余单元，使用 4 个事先固定的随机重复。先在每个教师内平均四个 NRMSE，再跨教师汇总，未选最优随机结果。

这些方法都使用原教师权重、相同局部边界、相同预算及相同的遗漏单元平均门控近似；差异是选择顺序。原随机、权重、融合 PCA、贪心及完整精确控制也保留。

## 4. 同预算的误差比较

下表为完全置零、高阶组合的六教师平均 NRMSE ± 种子间 SD，越低越好。SD 不是置信区间。

| 方法 | L1 k16 | L1 k32 | L2 k16 | L2 k32 |
|---|---:|---:|---:|---:|
{chr(10).join(fixed)}

在全部 {identical}/12 个 CNN 模型—层设置中，活跃优先筛选都没有改变原权重排序的前 16 或前 32 项。这说明原权重基线在这些预算下已经做到了相同的活跃单元优先效果；该新增控制与原权重选择重合，不能把两项当成两份独立支持证据。

![结构控制与任务质量](structure_controls.png)

## 5. 将头部冗余较少的教师单独列出

下表仅为训练中曾激活单元大于 32 的三个教师（种子末位 071、072、073）。这一分组在本轮测试前根据旧训练诊断固定；全部六教师仍是上表的主展示，没有删除其他种子。三个教师的结果属于描述性分组，不能据此声明总体统计显著。

| 方法 | L1 k16 | L1 k32 | L2 k16 | L2 k32 |
|---|---:|---:|---:|---:|
{chr(10).join(strata)}

逐种子差值也保留。下表“负值”有利于贪心；胜出数把差异小于 \(10^{{-10}}\) 个百分点的值视为数值并列。这里只用于显示方向，没有新的显著性检验。

| 层 | k | 对照 | 贪心减对照：平均百分点 | 贪心较低／并列／较高 |
|---|---:|---|---:|---:|
{chr(10).join(contrasts)}

k=16 时，贪心相对门控加权在两层的六个教师中方向都较好，支持它在本设置中提供了超出简单活跃筛选的额外信息。k=32 时则需收窄说法：L1 为 2 个较好、4 个并列；L2 为 1 个较好、3 个并列、2 个较差。六种子均值较低不能改写为逐教师稳定优越。前三个较少冗余教师的结果仍有差异，因此优势也不能完全归因于排除样本中不活跃的单元。

## 6. GAP＋线性读出告诉我们什么

保持两层卷积、ReLU 和 MaxPool，移除 64 单元非线性全连接头，改用每个通道的空间均值与线性分类器。重新训练六个模型，同样最多 15 轮，按最低验证交叉熵选中；未因准确率偏低而替换种子或追加训练。

分别干预：L1 为 ReLU1 后；L2 为 ReLU2 后、最大池化前；L3 为最大池化后，此后只剩平均与线性读出。精确方法保留完整受影响窗口；均值边界方法用训练窗口均值替代当前值；原区域公式固定初始 ReLU 分支与池化胜者。

| 位置 | 精确边界标量范围 | 头部背景标量 | 精确 NRMSE | 均值边界 NRMSE | 原区域公式 NRMSE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(gap_rows)}

L3 的响应可直接写成 \(\Delta F=B\Delta h\)，不变背景和偏置从差分中消去，因而不需要额外背景。L1／L2 仍经过可能切换的非线性操作；不能因为最终分类器线性，就省掉更早的窗口竞争信息。所有位置的局部精确最大绝对误差合并上限为 {complete['max_exact_error']:.3g}（包含原 CNN 与 GAP）。

L2 的原区域误差超过 100% 并非分类错误率超过 100%，而是分数变化的预测误差超过真实变化尺度。固定旧池化胜者，会错过其他值接替成为最大值的情况，从而过度预测下降。结果出来后，另用首个 GAP 教师的四个既有验证输入、三个位置和两种强度，将原区域公式与 PyTorch 自动微分的独立 Jacobian 计算核对；24 个设置的最大差为 {interpretation['original_region_autograd']['max_difference']:.3g}。这是实现排错记录，没有重拟合或替换测试输出，原始响应与误差尺度保存在 [补充诊断](../audit/structure_interpretation_20260911.json)。

## 7. GAP 控制的任务能力明显更弱

| 种子 | GAP 选中轮次 | GAP 验证准确率 | GAP 测试准确率 | 原 CNN 测试准确率 |
|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

GAP 测试均值 **{gap_accuracy:.2f}%**，原 CNN 为 **{cnn_accuracy:.2f}%**。本轮训练出的浅层 GAP 模型不构成任务能力相当的现代网络替代。结构恒等式不依赖分类准确率，但不能用此对照声称压缩方法已经在强残差网络上失效、成功或无必要。两种架构学到的函数和真实效应尺度也不同，NRMSE 不能直接跨架构排名。

## 8. 完整成本仍需与误差一起看

本轮采用单线程 CPU float64，4 个固定验证输入、Q=256、3 次预热、15 轮随机方法顺序、每轮 3 次调用，分别记录从图像开始与已有完整背景缓存的场景。先在输入内取中位数，再平均输入和教师。以下为固定 k=32；加速比为同教师精确耗时除以方法耗时。

| 层 | 方法 | 从图像开始 ms | 配对加速均值 ± SD | 已缓存配对加速 |
|---|---|---:|---:|---:|
{chr(10).join(costs)}

原始计时、执行数组字节和全部预算均保存。毫秒级点估计与四个计时输入不能当作普遍速度保证。GAP 中直接尾网络的计时仅为原始执行参考，未将它宣称为最快的精确实现，也未用它推广大型模型加速结论。

第一层的背景总量尤其容易误读：六个 CNN 均需 576 个窗口标量，完整头部再需 64 个。k=32 把这两类背景从 640 减为 608，实际只减少 5%；还不包括 8 个源值、权重及计算临时数组。所谓“头部保留一半”并不等于“总背景或总运算减半”，这也解释了为何要报告完整计时。

## 9. 核查与研究决定

本轮从保存数组回读重算 **{audit['arrays_rescored']} 组指标**，最大 NRMSE 差为 {audit['max_nrmse_difference']:.3g}；模型与选择器冻结哈希未变，六个 GAP 检查点均对应最低验证交叉熵，预测承诺先于参考承诺。

当前证据支持继续研究“有限组合干预下的背景选择”：在较紧的 k=16 预算，残差选择的额外收益通过了这一轮简单基线检查；在误差较低的 k=32 预算，收益变小且受教师影响，完整速度优势也有限。线性尾部控制进一步明确了适用边界：关键是干预后可能切换的非线性运算，而不是固定要求一种 64 单元头部背景。

下一步应在任务能力足够的网络中选择更早的非线性块，保留当前门控基线、最快局部精确基线和完整成本口径，再检验选择的额外价值。需要预留新的评价条件；当前不宣称已经找到最少背景、人类语义模块或跨任务机制迁移方法。

完整的原始设想—数学—算法—实验讲解见 [从原始设想到当前实验](从原始设想到当前实验_2026-09-11.md)。证据入口：[协议](../runs/fashion_structure_20260911/protocol.json)、[测试前冻结](../runs/fashion_structure_20260911/freeze.json)、[全部汇总](../runs/fashion_structure_20260911/analysis/summary.csv)、[核验](../runs/fashion_structure_20260911/analysis/verification.json)。
"""
    (BASE / "reports/激活基线与线性读出对照_2026-09-11.md").write_text(report,encoding="utf-8")
    story=BASE / "reports/从原始设想到当前实验_2026-09-11.md"
    content=story.read_text(encoding="utf-8")
    start="<!-- STRUCTURE_RESULTS -->";end="<!-- END_STRUCTURE_RESULTS -->"
    if end in content:
        before=content.split(start)[0];after=content.split(end,1)[1]
    else:
        before,after=content.split(start)
    update=f"""

最新这轮对照已经跑完：在新的固定干预批次中，贪心选择 k=16 的 L1／L2 平均 NRMSE 为 {100*float(c1['nrmse_mean']):.2f}%／{100*float(c2['nrmse_mean']):.2f}%，简单门控加权为 {100*float(gate1['nrmse_mean']):.2f}%／{100*float(gate2['nrmse_mean']):.2f}%。这一预算下两层均为六个教师方向一致。k=32 的贪心平均误差降至 {100*float(find('greedy',1,32)['nrmse_mean']):.2f}%／{100*float(find('greedy',2,32)['nrmse_mean']):.2f}%，但相对门控加权不再逐教师占优。

“少一半头部状态”也没有带来一半总成本：第一层还有 576 个窗口背景值必须保留，64 减到 32 使两类背景的合计从 640 变成 608，仅减少 5%。把这点与完整计时一起看，就能理解为什么准确近似成立，而加速仍有限。

线性尾部控制验证了“最后一个非线性之后不需额外背景”的结构边界；但新 GAP 模型的测试分类准确率只有 {gap_accuracy:.2f}%，因此它不是强现代 CNN 的替代验证。本轮完整执行 {complete['full_tail_output_vectors']:,} 个参考输出向量，{audit['arrays_rescored']} 组保存指标回读一致。具体结果与限制见 [激活基线与线性读出对照报告](激活基线与线性读出对照_2026-09-11.md)。

"""
    story.write_text(before+start+update+end+after,encoding="utf-8")
    state=read_json(BASE / "RESEARCH_STATUS.json")
    state.update(current_run="runs/fashion_structure_20260911",
        latest_followup_run="runs/fashion_structure_20260911",followup_stage="activity and GAP controls completed",
        additional_gap_teachers=6,total_fashion_teachers=12,followup_model_layer_cases=30,
        followup_intervention_ids=[256,511],calibration_model_access=False,
        external_official_test_batches=2,official_test_classification_teacher_runs=12,
        completed_fashion_intervention_cases=42,
        followup_full_tail_output_vectors=complete["full_tail_output_vectors"],
        original_primary_run="runs/fashion_external_20260911",
        latest_report="reports/激活基线与线性读出对照_2026-09-11.md",
        methods_explained="reports/从原始设想到当前实验_2026-09-11.md",
        next_steps=["evaluate response-aware selection beyond activity/gate controls in a task-competent nonlinear architecture",
                    "study input-specific background uncertainty with genuinely reserved evaluation conditions"])
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (BASE / "RESEARCH_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    readme_path=BASE / "README.md"
    readme=readme_path.read_text(encoding="utf-8")
    readme=readme.replace("先读 [六种子外部测试与推进建议](reports/六种子外部测试与推进建议_2026-09-11.md)。",
        "先读 [从原始设想到当前实验](reports/从原始设想到当前实验_2026-09-11.md)，再看最新的 [激活基线与线性读出对照](reports/激活基线与线性读出对照_2026-09-11.md)。原主实验见 [六种子外部测试与推进建议](reports/六种子外部测试与推进建议_2026-09-11.md)。")
    readme=readme.replace("本轮已完成：","原研究包复核与第一轮正式实验已完成：")
    readme=readme.replace("这一冗余因素需要在下一阶段分离。", "这一冗余因素已在下述新增结构对照中进一步检查。")
    old="接下来：先做激活筛选基线及 GAP＋线性头结构控制，再考虑更细或校准后的预算和输出响应低秩基线。新增方法需要新的预留评价条件，不把已读取的测试输出重新当作未见数据。"
    new=(f"新增结构对照现已完成：原六个 CNN 的活跃筛选、门控加权与活跃随机基线，另训练六个 GAP＋线性读出教师，在非线性之前／之后共三个位置检验背景需求。新的固定干预索引为 256–511，共 30 个模型—层设置、{complete['full_tail_output_vectors']:,} 个真实干预参考输出；660 组指标回读一致。它是同一数据集上的后续诊断，不替代原主检验。GAP 平均分类准确率为 {gap_accuracy:.2f}%，明显低于原 CNN 的 {cnn_accuracy:.2f}%，因此只支持结构边界判断。5000 张校准图仍未用于模型评价。\n\n"
         "接下来应把响应感知选择的额外收益放到任务能力足够、仍有非线性背景需求的结构中检验，并保留新的评价条件。当前不把背景压缩等同于语义机制发现。")
    if old in readme:readme=readme.replace(old,new)
    if "- `runs/fashion_structure_20260911/`" not in readme:
        readme=readme.replace("- `reports/`", "- `runs/fashion_structure_20260911/`：激活基线与 GAP 控制的独立协议、六个新教师、冻结、37440 条原始计时、预测与参考及描述性分析。\n- `audit/structure_manifest_20260911.json`：新增实验与最新讲解的独立交付快照。\n- `reports/`")
    readme_path.write_text(readme,encoding="utf-8")


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

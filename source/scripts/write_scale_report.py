"""Presentation-only Chinese report and project-status update."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import json
import statistics as st

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/"runs/fashion_scale_20260914"
LABEL={"center":"中心","upper_left":"左上"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 组织规模/位置变化、近似接受率及精确回退结果的中文报告。
def main():
    data=read(BASE/"audit/scale_interpretation_20260914.json")
    done=read(RUN/"test/completion.json")
    verification=read(RUN/"analysis/verification.json")
    protocol=read(RUN/"protocol.json")
    assert data["status"]=="passed" and verification["status"]=="passed"
    conditions=[(p,q) for p in LABEL for q in (4,8,12)]
    # 从规模/位置结果中取出指定方法和强度的记录。
    def metric(pos,q,m,f,alpha=1.):
        return next(r for r in data["summaries"] if r["position"]==pos and int(r["q"])==q and r["method"]==m
                    and float(r["fraction"])==f and float(r["alpha"])==alpha)
    # 取当前设置的报告数值，供统一格式化与文字插入。
    def value(pos,q,m,f,alpha=1.):
        return 100*float(metric(pos,q,m,f,alpha)["nrmse_mean"])
    # 把结果记录格式化为可读的表格单元，缺失值按本报告规则展示。
    def cell(pos,q,m,f,alpha=1.):
        r=metric(pos,q,m,f,alpha)
        return f"{100*float(r['nrmse_mean']):.2f} ± {100*float(r['nrmse_sd']):.2f}"
    # 查询所指定位置、规模与强度的上界/回退统计。
    def bound(pos,q,alpha=1.):
        return next(r for r in data["bound_summary"] if r["position"]==pos and r["q"]==q and r["alpha"]==alpha)
    # 按工作场景和查询集合读取已有加速比，避免混用不同计时口径。
    def speed(pos,q,m,workload="from_image",query_set="evaluation"):
        return next(r for r in data["timing"] if r["position"]==pos and r["q"]==q and r["method"]==m
                    and r["query_set"]==query_set and r["workload"]==workload)
    primary="\n".join(f"| {LABEL[p]} | {q} | "+" | ".join(cell(p,q,m,f) for m,f in
            [("gate_weight",.25),("greedy",.25),("pca_fused",.25),("affine_fused",1.)])+" |" for p,q in conditions)
    half_budget="\n".join(f"| {LABEL[p]} | {q} | "+" | ".join(cell(p,q,m,.5) for m in
                         ("gate_weight","greedy","pca_fused"))+" |" for p,q in conditions)
    half_strength="\n".join(f"| {LABEL[p]} | {q} | "+" | ".join(cell(p,q,m,f,.5) for m,f in
            [("greedy",.25),("pca_fused",.25),("affine_fused",1.),("bounded_affine",1.)])+" |" for p,q in conditions)
    bound_table=[];zero_table=[];tail_table=[];time_table=[]
    for p,q in conditions:
        r=bound(p,q)
        bound_table.append(f"| {LABEL[p]} | {q} | {100*r['acceptance_usable_mean']:.2f}% | {100*r['hindsight_affine_usable_rate_mean']:.2f}% | {value(p,q,'bounded_affine',1.):.3f}% |")
        zero_table.append(f"| {LABEL[p]} | {q} | {r['zero_effect_sum']} | {r['usable_sum']} | {r['accepted_all_sum']} | {r['accepted_usable_sum']} |")
        tail_table.append(f"| {LABEL[p]} | {q} | "+" | ".join(
            f"{100*float(metric(p,q,m,f)['input_fraction_over_10pct_mean']):.2f}%" for m,f in
            [("greedy",.25),("pca_fused",.25),("affine_fused",1.),("bounded_affine",1.)])+" |")
        time_table.append(f"| {LABEL[p]} | {q} | "+" | ".join(
            f"{speed(p,q,m,w)['speed_mean']:.3f}" for m,w in
            [("greedy_f0.25","from_image"),("pca_fused_f0.25","from_image"),
             ("affine_fused_f1.0","from_image"),("bounded_affine_f1.0","from_image"),
             ("bounded_affine_f1.0","cached")])+" |")
    directions=[]
    for alpha in (1.,.5):
        for f in (.25,.5):
            for control in ("gate_weight","pca_fused"):
                chosen=[r for r in data["paired"] if r["alpha"]==alpha and r["fraction"]==f and r["control"]==control]
                directions.append(f"| {alpha:g} | {f:g} | {control} | {sum(r['greedy_better'] for r in chosen)}／{sum(r['tied'] for r in chosen)}／{sum(r['greedy_worse'] for r in chosen)} |")
    bd=rows(RUN/"analysis/bound_diagnostics.csv")
    candidate_table=[]
    for p,q in conditions:
        group=[r for r in bd if r["position"]==p and int(r["q"])==q and r["alpha"]=="1.0"]
        r=bound(p,q)
        candidate_table.append(f"| {LABEL[p]} | {q} | {st.mean(int(r['potentially_switching_states']) for r in group):.2f} | "
                               f"{100*r['switching_states_mean_mean']:.2f}% | {100*r['switching_queries_mean_mean']:.2f}% |")
    affine_center=value("center",12,"affine_fused",1.)
    affine_upper=value("upper_left",12,"affine_fused",1.)
    report=f"""# 干预规模、空间位置与精确回退

2026-09-14。接续 [残差块内部实验](残差块内部干预与背景选择_2026-09-13.md)。

本轮完成了六个已冻结残差教师、两个空间位置、4／8／12 个源坐标，共 **36 个设置**。新增官方干预索引为 768–895，共 128 张图像；没有重新训练教师，没有新增分类测试，也没有访问保留的 5000 张校准图。

核心结果是：局部线性近似对位置敏感，12 坐标时中心平均误差为 **{affine_center:.2f}%**，左上为 **{affine_upper:.2f}%**。四分之一预算下 PCA 比状态选择更准确，两个预算下状态选择均优于门控加权。但加入解析上界和精确回退后，当前实现虽然控制住了接受输入的误差，却比直接精确计算更慢。

![规模、位置与回退结果](scale_position_results.png)

## 1. 本轮具体改变了什么

教师继续使用 9 月 13 日的六个小型残差 CNN，平均分类准确率 92.335% 是上一轮测得的结果。干预位置仍是最后残差块第一层卷积、BN、ReLU 之后，第二层卷积之前；旁路取原输入的值。

本轮的“位置”指 14×14 特征图上以 (7,7) 和 (3,3) 为中心的两个 3×3 窗口，坐标从零开始。每个窗口内按 512 个训练上下文的平均激活选前 q 个坐标，排序稳定且 q=4、8、12 的集合嵌套。它覆盖空间位置变化，尚未覆盖更早网络层、多层非线性链或其他数据集。

状态预算固定为受影响状态数 J 的 1/4 和 1/2，向上取整。J 范围为 480–800；36 个设置中全部受影响状态均在训练上下文中曾开启也曾关闭。选择器使用低阶组合的真实局部响应拟合；PCA 将相关背景联合编码，编码直接融合进卷积与旁路映射。四分之一状态数量和四分之一编码维数是这里的预算比较，不等于相同读取量或相同运算量。

| 源坐标 q | 拟合的低阶组合 | 实际评价组合 | 用于评分的高阶组合 |
|---|---:|---:|---:|
| 4 | 11 | 16，全部组合 | 5 |
| 8 | 37 | 256，全部组合 | 219 |
| 12 | 79 | 335：79 个低阶＋256 个高阶 | 256，预先固定抽样 |

q=12 不是 4096 组合的完整测试。扩大 q 同时改变了可干预坐标和组合阶数分布，因此也保存了 [按阶数分层的全部指标](../runs/fashion_scale_20260914/analysis/order_results.csv)，不将总体误差变化解释成只受 q 一个因素影响。

每个设置拟合用 512 个训练上下文、验证用 128 个上下文。正式测试共有 **{done['full_tail_output_vectors']:,} 个原网络参考输出向量**，验证另有同样数量。36 个设置共享六个教师和同一批图像；独立训练重复仍为六个，不能把设置数或输出向量数当作独立样本数。

这是同一 Fashion-MNIST 上的第四批干预。整个官方分类池此前已被访问，新干预索引与旧批次不重叠，但不构成新的外部数据集或完全未访问的测试池。

## 2. 精度：收益存在，但比较结果有条件

下表为完全置零干预，六教师 NRMSE 的均值 ± 种子间 SD，单位为百分比。前面三种方法使用 1/4 维数预算；局部线性公式获取完整当前门控，再合并为源到输出矩阵。

| 位置 | q | 门控加权 1/4 | 背景选择 1/4 | PCA 1/4 | 局部线性 |
|---|---:|---:|---:|---:|---:|
{primary}

同样条件下，1/2 预算结果如下：

| 位置 | q | 门控加权 1/2 | 背景选择 1/2 | PCA 1/2 |
|---|---:|---:|---:|---:|
{half_budget}

四分之一预算下，两种干预强度中，PCA 都在全部 36 个设置上优于状态选择；一半预算下则有例外。完全置零时，状态选择有 5 个设置比 PCA 更准，31 个更差；减半时为 8 个更准、28 个更差。状态选择相对门控加权则在两个预算、两种强度下都保持 36 个设置同向。

| 强度 α | 预算 | 对照 | 状态选择较好／数值并列／较差 |
|---|---:|---|---:|
{chr(10).join(directions)}

这些是共享六教师的描述性比较，没有新的确认性 p 值。这里的 SD 也不是置信区间。

减半干预的主要结果也完整报告，单位仍为百分比：

| 位置 | q | 背景选择 1/4 | PCA 1/4 | 局部线性 | 上界判断＋精确回退 |
|---|---:|---:|---:|---:|---:|
{half_strength}

## 3. “什么时候失效”要看逐输入结果

完全置零时，局部线性在中心三个规模下的总体均值仍低于 10%；左上三个规模的均值均超过 10%。但总体合格不等于每个输入都合格。

下表为有效输入中 NRMSE 超过 10% 的比例，先在教师内计算，再对六教师取均值：

| 位置 | q | 背景选择 1/4 | PCA 1/4 | 局部线性 | 上界＋回退 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tail_table)}

例如左上 q=12 时，局部线性约 48.44% 的有效输入超过 10%，而 1/4 PCA 约为 2.47%。不能仅凭上一轮中心位置约 8% 的平均误差，将局部线性近似推广到其他位置。

位置改变时，选中的通道、源激活幅度和背景也一起改变。左上位置的源范数及开关改变比例较高，是误差差异的相关诊断，不足以将变化完全归因于几何位置本身。

## 4. 回退规则的可靠性与保守性

规则和 10% 门槛在开发前固定。先用源变化的区间限制每个 ReLU 相对原切线的遗漏项，再推出查询集合的输出误差范数上界。若该上界足够小就采用线性近似，否则执行精确局部响应。具体推导、适用范围和数值限制见 [数学说明](局部线性回退规则_数学说明_2026-09-14.md)。

完全置零时的结果：

| 位置 | q | 有效输入中允许近似的比例 | 事后实际线性误差≤10% 的比例 | 回退方法总体 NRMSE |
|---|---:|---:|---:|---:|
{chr(10).join(bound_table)}

“事后实际合格比例”需要真实干预结果才能知道，只用于衡量上界有多保守，不能作为可部署的决策规则。中心 q=12 实际约 80.29% 的有效输入可以达到门槛，上界规则仅接受约 13.61%；左上 q=12 对应约 51.56% 和 2.47%。

验证和正式测试均未发现被接受的有效输入超过 10% 门槛。正式测试的相对门槛例外计数为 **{done['accepted_relative_threshold_exceptions']}**。解析上界检查的最大浮点超出为 **{done['max_bound_excess']:.3g}**，低于预先规定的绝对容差 10⁻⁹。推导针对实数运算，代码使用普通 float64，未实现向外舍入，因此不称为形式化验证软件。

近零效应必须单列。下表是每种设置六教师×128 图像的 768 个教师—输入配对计数，不是 768 张独立图像：

| 位置 | q | 近零效应配对 | 有效配对 | 总接受配对 | 接受的有效配对 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(zero_table)}

有效输入定义沿用真实响应平方和大于 10⁻²⁴。接受零效应输入并不能证明方法对有意义的干预普遍有用，所以正文接受率使用教师内有效输入比例的均值。

## 5. 完整成本：这版回退规则没有加速

全部计时在正式干预测试前完成。使用四个固定验证输入、单线程 CPU float64、两次预热、七轮随机顺序、每轮三次调用，共 **48,384 条记录**。原网络 GPU 参考执行只用于生成真值，不参加速度比较。每个设置保留三种精确实现，以其中实际最快者作基线。

表中数值为“最快精确耗时／方法耗时”的六教师配对均值；超过 1 才较快。q=4／8／12 的完整评价查询量分别为 16／256／335。前四列包含从图像开始的前缀和准备成本，最后一列只计算已准备背景的查询。

| 位置 | q | 选择 1/4：完整 | PCA 1/4：完整 | 线性：完整 | 上界＋回退：完整 | 上界＋回退：缓存 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(time_table)}

六种规模—位置条件下，上界回退的完整加速比均约 0.71–0.80；六个教师在每个条件下均更慢。它要计算完整背景、源到输出矩阵和上界，拒绝后还要继续执行精确响应。误差降低主要来自大量回退，不能将其当作高效近似方法已经成立的证据。

未经回退保护的局部线性公式在较大查询量下有完整速度优势，但左上位置的精度不足。两者分别回答速度和误差控制问题，不能把一个方法的速度与另一个方法的精度拼成共同收益。

低阶查询、其他预算和完整种子级 SD 保存在 [全部配对计时](../runs/fashion_scale_20260914/timing/summary.csv) 与 [描述性汇总](../audit/scale_interpretation_20260914.json)。缓存张量和保留的构造映射也计入 [存储记录](../runs/fashion_scale_20260914/timing/cost_counts.csv)，它不是进程峰值内存统计。

## 6. 下一项最值得检验的改进

现有上界虽然保守，但给出了“哪些状态可能跨过 ReLU 开关”的候选集合。完全置零的诊断为：

| 位置 | q | 每输入可能切换的状态数均值 | 实际每组合切换状态比例 | 至少改变一个开关的高阶组合比例 |
|---|---:|---:|---:|---:|
{chr(10).join(candidate_table)}

下一项可以检验：获取当前背景后，对区间范围内始终不切换的状态合并线性响应，只对可能切换的状态做逐组合精确修正。按本轮单层尾部的数学结构，这种分解可保持实数运算下的等价性；是否能在本机获得完整加速仍须另行固定方案并计时，本轮没有实现或宣称这一收益。

这比继续微调 10% 接受阈值更直接地对应当前瓶颈。同时，更早层的多次非线性传播仍是未完成的外推边界，不能用本轮两个空间位置替代。

## 7. 数值与文件核验

协议先于开发固定，数值实现通过独立实现检查后锁定，450 个开发文件及计时产物在开启新批次前冻结。每个病例先保存预测、回退决定及哈希，之后才由原始未融合教师生成独立参考。

保存数组已重算 **720 组验证指标＋720 组测试指标**，最大 NRMSE 差 **{verification['max_nrmse_difference']:.3g}**，逐输入第 95 分位也独立重算。精确局部响应与原网络参考最大绝对差为 **{done['max_exact_error']:.3g}**。

中心 q=8 与上一轮共享教师、拟合和验证输入；108 组共享预测再次一致，最大差 **{data['shared_validation_replay']['max_prediction_difference']:.3g}**，原网络参考完全一致。这是实现连续性核查，不是 108 组新的独立证据。

证据入口：[协议](../runs/fashion_scale_20260914/protocol.json)、[数值源码锁定](../runs/fashion_scale_20260914/implementation.json)、[测试前冻结](../runs/fashion_scale_20260914/freeze.json)、[数组核验](../runs/fashion_scale_20260914/analysis/verification.json)。旧研究原件、旧数值运行和历史清单继续保留；本轮没有生成 ZIP。
"""
    report_path=BASE/"reports/规模位置变化与精确回退_2026-09-14.md"
    report_path.write_text(report,encoding="utf-8")
    state=read(BASE/"RESEARCH_STATUS.json")
    if "residual_followup_summary" not in state:
        state["residual_followup_summary"]={k:state[k] for k in
            ("followup_stage","followup_model_layer_cases","followup_intervention_ids","followup_full_tail_output_vectors")}
    state.update(date="2026-09-14",current_run="runs/fashion_scale_20260914",
        latest_followup_run="runs/fashion_scale_20260914",latest_report=report_path.relative_to(BASE).as_posix(),
        followup_stage="size and spatial position study with fixed analytic-bound fallback completed",
        followup_model_layer_cases=36,followup_intervention_ids=[768,895],
        followup_full_tail_output_vectors=done["full_tail_output_vectors"],
        scale_position_cases=36,scale_source_sizes=[4,8,12],scale_spatial_positions=["center","upper_left"],
        scale_reused_residual_teachers=6,scale_new_teacher_training_runs=0,
        scale_bound_threshold=.1,scale_accepted_relative_threshold_exceptions=done["accepted_relative_threshold_exceptions"],
        scale_affine_center_q12_nrmse=affine_center/100,scale_affine_upper_left_q12_nrmse=affine_upper/100,
        scale_scope="same last residual branch; spatial positions, not earlier-layer generalization",
        completed_fashion_intervention_cases=84,completed_fashion_intervention_settings=84,
        independent_fashion_teacher_runs=18,total_fashion_teachers=18,official_test_classification_teacher_runs=18,
        external_official_test_batches=4,calibration_model_access=False,
        next_steps=["freeze and benchmark exact corrections only at states that may switch over the source-subset interval",
                    "extend beyond the final residual branch with explicit multi-layer nonlinear background accounting"])
    (BASE/"RESEARCH_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    readme=BASE/"README.md";text=readme.read_text(encoding="utf-8")
    marker="<!-- SCALE_LATEST -->";end="<!-- END_SCALE_LATEST -->"
    latest=f"""{marker}

最新结果见 [规模位置变化与精确回退](reports/规模位置变化与精确回退_2026-09-14.md)。复用六个残差教师，完成两个空间位置、4／8／12 个源坐标的 36 个设置。12 坐标时，局部线性误差由中心的 {affine_center:.2f}% 变为左上的 {affine_upper:.2f}%；四分之一预算下，PCA 在所有设置上比状态选择更准确。

预先固定的解析上界＋精确回退规则，没有出现被接受有效输入超过 10% 门槛的情况，但当前实现完整流程比最快精确基线更慢。1440 组指标回读一致，5000 张校准图仍未用于模型评价。两个空间窗口不代表更早网络层。当前状态以本报告和 RESEARCH_STATUS 为准。

{end}
"""
    if marker in text:text=text.split(marker)[0]+latest+text.split(end,1)[1]
    else:text=text.replace("<!-- RESIDUAL_LATEST -->",latest+"\n<!-- RESIDUAL_LATEST -->",1)
    text=text.replace("更新：2026-09-13。","更新：2026-09-14。")
    text=text.replace("最新阶段见 [残差块内部干预与背景选择]","9 月 13 日阶段见 [残差块内部干预与背景选择]")
    text=text.replace("本轮 720 组保存指标回读一致，5000 张校准数据仍未用于模型评价。原两阶段报告保留历史结果；当前总体状态以新报告和 RESEARCH_STATUS 为准。",
                      "9 月 13 日该阶段的 720 组保存指标回读一致。上述数字保留其历史实验含义，当前总体状态见本页最新报告和 RESEARCH_STATUS。")
    text=text.replace("- `audit/final_residual_delivery_20260913.json`：当前研究阶段的最终保全与交付核查。",
                      "- `audit/final_residual_delivery_20260913.json`：9 月 13 日残差研究阶段的保全与交付核查。")
    if "- `runs/fashion_scale_20260914/`" not in text:
        text=text.replace("- `reports/`",
            "- `runs/fashion_scale_20260914/`：36 个规模与空间位置设置、固定上界回退、冻结与完整计时。\n"
            "- `audit/final_scale_delivery_20260914.json`：当前阶段的最终保全与交付核查。\n"
            "- `audit/scale_manifest_20260914.json`：规模位置实验与当前报告的独立交付清单。\n"
            "- `reports/`")
    text=text.replace("接下来应改变干预规模与位置，检验仿射近似、PCA 与状态选择的失效条件。",
                      "这些规模与空间位置检查也已完成，当前推进方向见最新报告。")
    readme.write_text(text,encoding="utf-8")
    story=BASE/"reports/从原始设想到当前实验_2026-09-11.md"
    text=story.read_text(encoding="utf-8");marker="<!-- SCALE_EXTENSION -->"
    if marker in text:text=text.split(marker)[0]
    text+=f"""

{marker}
## 15. 9 月 14 日：从平均有效，走到何时失效、何时回退

你最初想知道，能否通过局部节点组合得到整体输出。现在我们不仅能比较局部公式与完整网络，还开始检查公式在什么条件下会失效。

这一轮复用同样六个残差网络，在中心与左上两个窗口，分别选取 4、8、12 个激活坐标。12 坐标时，中心的局部线性平均误差约为 {affine_center:.2f}%，左上却达到 {affine_upper:.2f}%。所以之前约 8% 的结果有明确的位置边界。两个窗口的所选通道与激活幅度也不同，不能把差异仅归因于几何坐标。

我们随后用预先固定的数学规则检查误差上界：足够小时采用近似，不够小时做精确计算。这一规则在本批次没有接受超过 10% 相对误差门槛的有效输入，但上界太保守、判断又有开销，导致完整流程比直接精确计算更慢。

这让研究的问题更具体：不是只证明“能写出一个局部公式”，还要同时说明它在哪些输入上足够准确、要付出多少背景读取与计算成本。四分之一预算下 PCA 仍比逐状态选择更准确，因此论文必须保留这个强对照；一半预算下也要诚实保留状态选择更好的那些例外。

误差上界还给出下一条可检验路线：始终不切换的状态可合并成线性响应，只对可能切换的状态做精确修正。它尚未在本轮实现或计时，不能先宣布获得加速。具体结果见 [规模与位置报告](规模位置变化与精确回退_2026-09-14.md)，规则从局部响应到整体误差的推导见 [数学说明](局部线性回退规则_数学说明_2026-09-14.md)。
"""
    story.write_text(text,encoding="utf-8")


if __name__=="__main__":main()

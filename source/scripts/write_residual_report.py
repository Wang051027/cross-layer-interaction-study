# 中文阅读说明｜用途：组织残差块实验的结果、限制和成本表。
# 数据流：读取冻结实验汇总及单独仿射审计；输出历史报告和状态。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Presentation-only synthesis of the frozen study and separate affine audit."""
from pathlib import Path
import csv
import json
import statistics as st

BASE=Path(__file__).resolve().parents[1]
RUN=BASE / "runs/fashion_residual_20260913"
SEEDS=list(range(202609071,202609077))
LABELS={"greedy":"残差贪心选择","weight":"权重范数","gate_weight":"门控方差×权重",
        "activity":"激活频率","active_random_mean":"活跃随机（4 次均值）","pca_fused":"融合 PCA"}


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read(path):return json.loads(path.read_text(encoding="utf-8"))


# 用途：读取保存的表格记录；不在这里重新运行实验。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def rows(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：组织残差块实验的结果、限制和成本表。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    audit=read(RUN / "analysis/verification.json")
    affine_audit=read(RUN / "affine_runtime_audit/completion.json")
    complete=read(RUN / "test/completion.json")
    assert audit["status"]=="passed" and affine_audit["status"]=="completed"
    summary=rows(RUN / "analysis/summary.csv");raw=rows(RUN / "test/all_results.csv")
    paired=rows(RUN / "analysis/paired_differences.csv")
    half_pca_differences=[float(r["greedy_minus_control"]) for r in paired
                          if r["alpha"]=="0.5" and r["control"]=="pca_fused"]
    assert len(half_pca_differences)==18
    half_pca_better=sum(d > 1e-12 for d in half_pca_differences)
    half_greedy_better=sum(d < -1e-12 for d in half_pca_differences)
    classes=rows(RUN / "test/classification.csv")
    quality=read(RUN / "training_complete.json")
    # 用途：筛选满足当前表格条件的结果行；并查集中的同名函数另有专门说明。
    # 参数：method、fraction、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：next((r for r in summary if r['method'] == method and float(r['fraction']) == fraction …。
    def find(method,fraction,alpha=1.):
        return next(r for r in summary if r["method"]==method and float(r["fraction"])==fraction and float(r["alpha"])==alpha)
    # 用途：读取指定设置下已经汇总的平均误差，供报告表格使用。
    # 参数：method、fraction、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：100 * float(find(method, fraction, alpha)['nrmse_mean'])。
    def mean(method,fraction,alpha=1.):return 100*float(find(method,fraction,alpha)["nrmse_mean"])
    # 用途：把结果记录格式化为可读的表格单元，缺失值按本报告规则展示。
    # 参数：method、fraction、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：f"{100 * float(r['nrmse_mean']):.2f}% ± {100 * float(r['nrmse_sd']):.2f}%"。
    def cell(method,fraction,alpha=1.):
        r=find(method,fraction,alpha)
        return f"{100*float(r['nrmse_mean']):.2f}% ± {100*float(r['nrmse_sd']):.2f}%"
    accuracy_tables={}
    for alpha in (1.,.5):
        accuracy_tables[alpha]="\n".join("| "+label+" | "+" | ".join(cell(method,f,alpha) for f in (.125,.25,.5))+" |" for method,label in LABELS.items())
    class_table=[];zero_table=[]
    for c in classes:
        seed=int(c["seed"]);teacher=read(RUN / f"seed_{seed}/teacher_summary.json")
        fit=read(RUN / f"seed_{seed}/fit_complete.json")
        class_table.append(f"| {seed} | {teacher['best_epoch']} | {100*teacher['best_validation_accuracy']:.2f}% | {100*float(c['accuracy']):.2f}% | {fit['affected_states']} | {fit['training_active_units']} | {fit['always_open_training_units']} |")
        zero=int(next(r["zero_effect_inputs"] for r in raw if r["seed"]==str(seed)))
        zero_table.append(f"| {seed} | {zero} | {256-zero} |")
    contrasts=[]
    for f in (.125,.25,.5):
        for control in ("gate_weight","pca_fused"):
            differences=[100*float(r["greedy_minus_control"]) for r in paired if float(r["fraction"])==f and r["alpha"]=="1.0" and r["control"]==control]
            contrasts.append(f"| {f:.3g} | {LABELS[control]} | {st.mean(differences):+.2f} | {sum(d < -1e-10 for d in differences)}／{sum(abs(d)<=1e-10 for d in differences)}／{sum(d>1e-10 for d in differences)} |")
    cost_specs=[("greedy_f0.25","残差选择 1/4"),("greedy_f0.5","残差选择 1/2"),
                ("pca_fused_f0.125","融合 PCA 1/8"),("pca_fused_f0.25","融合 PCA 1/4"),
                ("original_region_f1.0","原区域公式"),("affine_fused","合并后的原区域公式")]
    # 用途：把不同工作场景的成本数据整理为表格，可单独包含融合仿射审计。
    # 参数：folder、allow_fused；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：('\n'.join(table), values)。
    def cost_tables(folder,allow_fused):
        timing=rows(folder / "summary.csv");exact=rows(folder / "fastest_exact.csv")
        table=[];values={}
        for method,label in cost_specs:
            if not allow_fused and method=="affine_fused":continue
            cells=[]
            for q,w in ((37,"from_image"),(256,"from_image"),(256,"cached")):
                ratios=[];milliseconds=[]
                # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
                for seed in SEEDS:
                    same=lambda r:r["seed"]==str(seed) and r["queries"]==str(q) and r["workload"]==w
                    one=next(float(r["seconds"]) for r in timing if same(r) and r["method"]==method)
                    full=next(float(r["seconds"]) for r in exact if same(r))
                    ratios.append(full/one);milliseconds.append(one*1000)
                values[(method,q,w)]=(st.mean(ratios),st.stdev(ratios),st.mean(milliseconds))
                cells.append(f"{st.mean(ratios):.3f} ± {st.stdev(ratios):.3f}")
            table.append("| "+label+" | "+" | ".join(cells)+" |")
        return "\n".join(table),values
    frozen_costs,_=cost_tables(RUN / "timing",False)
    new_costs,new_cost_values=cost_tables(RUN / "affine_runtime_audit",True)
    tails=[]
    for method,f,label in [("greedy",.25,"残差选择 1/4"),("greedy",.5,"残差选择 1/2"),
                           ("pca_fused",.125,"融合 PCA 1/8"),("pca_fused",.25,"融合 PCA 1/4"),
                           ("pca_fused",.5,"融合 PCA 1/2"),("original_region",1.,"原区域公式／合并公式")]:
        selected=[r for r in raw if r["method"]==method and float(r["fraction"])==f and r["alpha"]=="1.0"]
        tails.append(f"| {label} | {mean(method,f):.2f}% | {st.mean(100*float(r['input_p95']) for r in selected):.2f}% | {st.mean(100*float(r['input_fraction_over_10pct']) for r in selected):.2f}% |")
    switches=rows(RUN / "affine_runtime_audit/switch_diagnostics.csv")
    switch_states=st.mean(float(r["mean_fraction_states_switched"]) for r in switches if r["alpha"]=="1.0")*100
    switch_queries=st.mean(float(r["fraction_coalitions_any_switch"]) for r in switches if r["alpha"]=="1.0")*100
    accuracy=100*st.mean(float(r["accuracy"]) for r in classes)
    speed=new_cost_values[("affine_fused",256,"from_image")][0]
    cached_speed=new_cost_values[("affine_fused",256,"cached")][0]
    report=fr"""# 残差块内部的组合干预与背景选择

2026-09-13。接续 [激活基线与线性读出对照](激活基线与线性读出对照_2026-09-11.md)。

本轮完成六个小型残差 CNN、三个固定状态预算、原始网络独立干预参考、测试前冻结、完整 CPU 计时与保存数组重算。模型平均测试分类准确率为 **{accuracy:.3f}%**；局部模块包含 640–768 个状态，训练上下文中全部出现过开启与关闭。因此，本轮选择收益不能再由“选掉一批样本中始终关闭的头部单元”来解释。

结果同时改变了方法判断：**完全置零时，残差选择在所有预算和六个种子上都优于简单门控加权，但在同预算下也全部落后于融合 PCA。** 1/4 预算的平均 NRMSE 分别为残差选择 **{mean('greedy',.25):.2f}%**、门控加权 **{mean('gate_weight',.25):.2f}%**、融合 PCA **{mean('pca_fused',.25):.2f}%**。原区域线性公式约 **{mean('original_region',1.):.2f}%**；其可进一步合并计算，另行审计见第 7 节。

## 1. 为什么换到残差块内部

上一轮浅层 GAP 模型测试准确率仅 69.06%，不能充当任务能力足够的网络控制；原全连接头又存在不活跃单元冗余。本轮采用 stem 卷积与三个残差块，最后经过 GAP 和线性读出，共 38,266 个参数。残差连接思想依据 [He 等人的原始论文](https://arxiv.org/abs/1512.03385)，本机实现是用于受控实验的小网络，不能称为标准 ResNet-18 或当前大型模型验证。

干预位于最后一个残差块的分支内部：第一层卷积、BatchNorm 和 ReLU 之后，第二层卷积之前。置零／减半的是该分支的指定激活，旁路输入保持该图片原来的值。两条路径在最后一个 ReLU 前相加。

固定评价模式下，BatchNorm 是仿射变换；融合使用 [PyTorch 2.12 的官方接口](https://docs.pytorch.org/docs/2.12/generated/torch.nn.utils.fuse_conv_bn_eval.html)，没有把训练模式的 BatchNorm 当作固定线性层。独立参考仍执行原始未融合网络尾部。

记分支源值为 $h$，旁路值为 $r$，受影响状态为 $z=Wh+b+r$，源变化为 δh，分类器到相应位置的线性权重为 $U$，则有限干预响应是：

\[
\Delta F=U[\operatorname{{ReLU}}(z+W\delta h)-\operatorname{{ReLU}}(z)].
\]

这里 $W$ 在响应计算中只保留被干预的 8 个源坐标对应列；在背景获取时仍需要完整卷积邻域和旁路。该公式允许分支切换。最初的“棱形”继续表现为局部计算与完整尾网络两条路线的一致性。

## 2. 教师质量、状态活跃性与数据边界

训练仍采用同一 Fashion-MNIST 的 50,000／5,000／5,000 训练、验证、校准划分。六个种子全部保留；AdamW、20 轮固定余弦学习率、无数据增强，严格选择最低验证交叉熵检查点。预设质量门槛为平均验证准确率至少 90%、最差至少 88%；实际平均 **{100*quality['mean_validation_accuracy']:.2f}%**、最低 **{100*quality['minimum_validation_accuracy']:.2f}%**。

| 种子 | 选中轮次 | 验证分类准确率 | 测试分类准确率 | 受影响状态 J | 训练中曾开启 | 训练中始终开启 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_table)}

新干预测试固定为官方索引 **512–767**，与旧批次 0–511 不重叠；但整个官方分类池此前已被其他教师访问。这仍是同一数据集的后续研究，不能称为第三个外部数据集或完全未碰过的测试池。5000 张校准图像仍未用于模型评价。

每教师只研究一个训练平均激活选出的中央 3×3 区域内的 8 坐标集合；方法训练用 512 个上下文、37 个低阶组合，验证用 128 个上下文。测试为 256 输入×256 组合×2 强度×6 教师，实际执行 **{complete['full_tail_output_vectors']:,} 个参考输出向量**，评分只使用 219 个高阶组合。独立重复数量是六个教师，不能用这些向量数代替。

## 3. 同预算精度比较

预算将 J/8、J/4、J/2 分别向上取整。选择方法精确保留相应当前状态，其余单元采用训练平均门控线性响应；融合 PCA 使用相同数量的编码坐标，重构全部受影响状态后计算响应。四个活跃随机重复先在教师内平均误差，未挑最优重复。

下表为完全置零、六教师平均 NRMSE ± 种子间 SD；SD 不是置信区间。

| 方法 | 1/8 预算 | 1/4 预算 | 1/2 预算 |
|---|---:|---:|---:|
{accuracy_tables[1.]}

全平均门控无需当前状态，NRMSE 为 **{mean('mean_gate',0.):.2f}%**。完整精确计算与独立参考的最大绝对误差为 **{complete['max_exact_error']:.3g}**。

| 预算比例 | 对照 | 贪心减对照：平均百分点 | 贪心较好／数值并列／较差 |
|---|---|---:|---:|
{chr(10).join(contrasts)}

这里没有新的确认性 p 值；并列阈值为 $10^{{-10}}$ 个百分点。原先的唯一主检验属于 9 月 11 日的原实验。

减半干预也完整保留：

| 方法 | 1/8 预算 | 1/4 预算 | 1/2 预算 |
|---|---:|---:|---:|
{accuracy_tables[.5]}

减半干预的三个预算共 18 个教师—预算设置，融合 PCA 在 {half_pca_better} 个中比残差选择更准确，残差选择在另外 {half_greedy_better} 个中更准确；不能把完全置零下的全部同向结果延伸到每种强度。

![残差块结果与冻结计时](residual_context_results.png)

## 4. 总体达标不代表每张输入达标

1/4 残差选择只有 {find('greedy',.25)['seeds_meet_10pct']}/6 个教师总体 NRMSE ≤10%；1/2 残差选择和 1/8 PCA 均为 6/6。10% 仍是沿用的经验门槛，不是理论保证或校准后的逐输入承诺。

下表先在教师内计算，再对六教师取均值；逐输入统计只包括真实干预效应平方和大于 $10^{{-24}}$ 的输入。

| 方法 | 总体 NRMSE 均值 | 逐输入 NRMSE 第 95 分位的均值 | 逐输入超过 10% 的比例均值 |
|---|---:|---:|---:|
{chr(10).join(tails)}

近零效应计数不可省略，特别是种子 072 的固定局部源集合在这一批输入上有大量近零效应。总体 NRMSE 以效应能量归一，没有把零效应输入当作一批零误差分数来平均；但这仍限制了结论覆盖的有效输入。

| 种子 | 近零效应输入 | 用于逐输入统计的输入 |
|---|---:|---:|
{chr(10).join(zero_table)}

## 5. 背景维数、读取成本与内存要区分

状态选择只计算所选卷积行的当前值，并加上对应旁路值。PCA 编码预先融合进“卷积＋旁路”的仿射映射，直接由局部特征与旁路形成低维码，没有先物化全部 z 的人为限制。但 PCA 仍需读取编码涉及的特征、重构状态并计算全部非线性响应。

双方都要计算共同的分支与旁路前缀。传递 k 个状态／编码，不等于只读 k 个原始特征，也不等于总计算、内存或原网络参数减少同样比例。

`timing/cost_counts.csv` 同时记录活动执行数组字节、保留的非教师张量存储以及教师状态字节。构造器保留的共享映射也单列计入，未把它隐藏成零。该统计仍不等于操作系统测得的峰值进程内存。

## 6. 测试前冻结的完整成本

四个固定验证输入，单线程 CPU float64，Q=37／256，两种场景，3 次预热、9 轮随机方法顺序、每轮 3 次调用。保存 27,648 条原始记录。精确方法同时实现索引卷积行、受影响区域裁剪卷积、完整末层卷积，逐教师／查询规模／场景取其中实际最快者作为基线。

加速比为精确耗时／方法耗时，超过 1 才较快。表内为六教师配对加速均值 ± SD。

| 方法 | 从图像开始 Q37 | 从图像开始 Q256 | 已缓存背景 Q256 |
|---|---:|---:|---:|
{frozen_costs}

精度达到门槛所需的 1/2 残差选择，并没有稳定的完整流程加速；PCA 在本实现中更准确，但完整计算通常也没有胜过最快精确方法。1/4 选择的查询加速较好，却不能把其约 12% 的误差省略不报。

## 7. 测试后发现并单独审计的仿射融合

原区域公式保留当前 ReLU 开关而不更新分支。它在本批次达到约 {mean('original_region',1.):.2f}% 平均误差，这促使我们检查其实现是否已经充分优化。当前开关给定后，可以先计算：

\[
M(x)=U\operatorname{{diag}}(z(x)>0)W,\qquad
\widehat{{\Delta F}}=M(x)\delta h.
\]

对本研究的 10 个输出、8 个源坐标，M 只有 **80 个系数**。准备阶段计算当前背景、当前门控与 M；查询阶段只用 8 个源值和 80 个系数。这与原区域近似在数学上等价，不是新的非线性预测模型，也不是精确有限干预。

这是**明确标注的测试后实现审计**：先记录补充方案与源码，再核对全部 12 组既有原区域预测，并在同一次新计时中重跑全部原方法与三种精确基线。没有重训、选预算或覆盖原测试与原计时。最大预测差为 **{affine_audit['max_prediction_difference']:.3g}**，新增计时 28,512 条。

| 方法 | 从图像开始 Q37 | 从图像开始 Q256 | 已缓存摘要 Q256 |
|---|---:|---:|---:|
{new_costs}

合并后的原区域公式从图像开始 Q256 约 **{speed:.2f}×**，已缓存摘要 Q256 约 **{cached_speed:.1f}×**；两种工作负载不能混称。前者计入前缀、完整当前门控获取与 80 系数的形成；后者只表示同一输入与同一源集合下复用已准备摘要的查询成本。四个计时输入及本机 CPU float64 不提供普遍部署加速保证。

补充门控统计显示，完全置零时平均约 **{switch_states:.2f}%** 的受影响状态发生开关改变，约 **{switch_queries:.2f}%** 的高阶组合至少改变一个开关。这说明原区域近似并未严格留在原区域；较小总体输出误差是当前设置的经验结果，不能由链式法则推出有限干预精确性，也不能直接保证其他位置与更大源集合。

## 8. 核验与本轮研究决定

模型训练规则于 12:57 UTC 固定；数值实现于 13:08 UTC 固定；78 个开发产物于 13:11 UTC 冻结，随后才对新残差教师开启官方测试。每个验证／测试病例均先保存预测及哈希，再执行未融合原网络得到参考。

验证参考与正式参考使用 CUDA float64，仅用于缩短真实尾网络重算时间；所有计时方法与计时基线均为相同 CPU float64。四个验证输入的 CPU／CUDA 独立尾部、BatchNorm 融合、三个获取实现、批量／单个调用及自动微分核对已通过，不能把 GPU 参考执行时间用作方法加速证据。

保存数组已回读重算 **360 组验证指标＋360 组测试指标**，最大 NRMSE 差为 {audit['max_nrmse_difference']:.3g}，源码、教师与冻结文件哈希未变。新增仿射审计也保留独立方案、来源快照、等价核查和完整计时。

本轮支持的结论是：选择当前非线性背景，确实比简单激活排序更有针对性；但逐状态选择并非本设置下最准确的表示，完全置零时融合 PCA 在所有固定预算与六个种子上更准，减半时则有一个设置方向相反。若同时要求总体误差约 10% 和完整速度优势，输入相关的局部仿射摘要是必须保留的强对照。当前不能把“选择了更少状态”作为普遍最优或普遍更快的证据。

下一步应优先扩大干预规模与位置，寻找局部仿射、PCA 和状态选择各自失效的条件，再研究如何按输入在近似与精确计算之间选择。逐输入可靠性可利用仍未访问的校准数据，但需要先固定准则。仅继续增加相似教师或突出缓存查询加速，不足以解决当前方法边界。

证据入口：[协议](../runs/fashion_residual_20260913/protocol.json)、[源码锁定](../runs/fashion_residual_20260913/implementation.json)、[测试前冻结](../runs/fashion_residual_20260913/freeze.json)、[完整指标](../runs/fashion_residual_20260913/analysis/summary.csv)、[数组核验](../runs/fashion_residual_20260913/analysis/verification.json)、[仿射补充方案](../runs/fashion_residual_20260913/affine_runtime_audit/plan.json)。
"""
    # In a raw f-string the display-math wrappers need only one backslash.
    report=report.replace("\\\\[","\\[").replace("\\\\]","\\]").replace("\\\\", "\\")
    report_path=BASE / "reports/残差块内部干预与背景选择_2026-09-13.md"
    report_path.write_text(report,encoding="utf-8")
    state=read(BASE / "RESEARCH_STATUS.json")
    if "structure_followup_summary" not in state:
        state["structure_followup_summary"]={
            "run":"runs/fashion_structure_20260911",
            "stage":state["followup_stage"],
            "model_layer_cases":state["followup_model_layer_cases"],
            "intervention_ids":state["followup_intervention_ids"],
            "full_tail_output_vectors":state["followup_full_tail_output_vectors"]}
    state.update(date="2026-09-13",current_run="runs/fashion_residual_20260913",
        latest_followup_run="runs/fashion_residual_20260913",
        followup_stage="residual block controls and separate affine runtime audit completed",
        followup_model_layer_cases=6,followup_intervention_ids=[512,767],
        followup_full_tail_output_vectors=complete["full_tail_output_vectors"],
        latest_report=report_path.relative_to(BASE).as_posix(),residual_stage="six-seed residual study and separate affine runtime audit completed",
        residual_teachers=6,total_fashion_teachers=18,official_test_classification_teacher_runs=18,
        completed_fashion_intervention_cases=48,external_official_test_batches=3,
        residual_intervention_ids=[512,767],residual_test_classification_accuracy=accuracy/100,
        residual_affected_state_range=[640,768],residual_quarter_greedy_nrmse=mean("greedy",.25)/100,
        residual_quarter_pca_nrmse=mean("pca_fused",.25)/100,
        residual_affine_nrmse=mean("original_region",1.)/100,
        residual_affine_audit_is_post_test=True,calibration_model_access=False,
        next_steps=["vary source-set size and intervention position to characterize affine/PCA/selection failure boundaries",
                    "freeze an input-dependent approximation/exact decision rule before using reserved calibration data"])
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (BASE / "RESEARCH_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    readme_path=BASE / "README.md";text=readme_path.read_text(encoding="utf-8")
    text=text.replace("更新：2026-09-11。","更新：2026-09-13。")
    marker="<!-- RESIDUAL_LATEST -->";end="<!-- END_RESIDUAL_LATEST -->"
    update=f"""{marker}

最新阶段见 [残差块内部干预与背景选择](reports/残差块内部干预与背景选择_2026-09-13.md)。六个残差教师平均测试分类准确率 {accuracy:.3f}%，受影响状态 640–768 个，训练中全部曾开启也曾关闭。完全置零时，1/4 预算的残差选择／门控加权／融合 PCA 误差分别为 {mean('greedy',.25):.2f}%／{mean('gate_weight',.25):.2f}%／{mean('pca_fused',.25):.2f}%；PCA 同预算更准，逐状态选择不能主张普遍最优。测试后单独核验的局部仿射融合保持约 {mean('original_region',1.):.2f}% 误差，从图像开始 Q256 约 {speed:.2f}×；已缓存摘要约 {cached_speed:.1f}×，两者口径不同。

本轮 720 组保存指标回读一致，5000 张校准数据仍未用于模型评价。原两阶段报告保留历史结果；当前总体状态以新报告和 RESEARCH_STATUS 为准。研究包 ZIP 已按用户要求删除，本轮没有重新打包。

{end}
"""
    if marker in text:text=text.split(marker)[0]+update+text.split(end,1)[1]
    else:text=text.replace("来源：",update+"\n来源：",1)
    text=text.replace("再看最新的 [激活基线与线性读出对照]","再看前一阶段的 [激活基线与线性读出对照]")
    text=text.replace("接下来应把响应感知选择的额外收益放到任务能力足够、仍有非线性背景需求的结构中检验，并保留新的评价条件。当前不把背景压缩等同于语义机制发现。",
        "9 月 11 日提出的任务能力与非线性背景检查，现已在上述残差阶段完成。接下来应改变干预规模与位置，检验仿射近似、PCA 与状态选择的失效条件。当前不把背景压缩等同于语义机制发现。")
    if "- `runs/fashion_residual_20260913/`" not in text:
        text=text.replace("- `reports/`","- `runs/fashion_residual_20260913/`：六个残差教师、局部干预与冻结指标、三种精确基线、独立仿射补充审计。\n- `reports/`")
    readme_path.write_text(text,encoding="utf-8")
    story=BASE / "reports/从原始设想到当前实验_2026-09-11.md";text=story.read_text(encoding="utf-8")
    text=text.replace("最新这轮对照已经跑完：","9 月 11 日的这轮对照结果为：")
    marker="<!-- RESIDUAL_EXTENSION -->"
    if marker in text:text=text.split(marker)[0]
    text+=f"""

{marker}
## 14. 9 月 13 日：强一些的结构让方法判断进一步改变

我们把干预移到小型残差网络最后一个残差块内部。新的六个教师平均测试分类准确率为 {accuracy:.3f}%，解决了前一轮浅层 GAP 控制偏弱的问题；640–768 个受影响状态都在训练样本中出现过开关变化，不能靠排除一批始终关闭的单元取得收益。

完全置零时，残差选择依然比简单门控加权准确，但同预算下融合 PCA 在六个种子上都更准确。保留四分之一状态时，二者平均误差是 {mean('greedy',.25):.2f}% 和 {mean('pca_fused',.25):.2f}%。减半干预时 PCA 在 18 个教师—预算设置中的 {half_pca_better} 个更准，另外 {half_greedy_better} 个由残差选择胜出。这说明“保留哪些单元”只是背景表示的一种办法；把相关状态共同编码也可能更合适。

更接近你最初“由局部变化直接联系整体输出”直觉的原区域线性公式，在这批次约有 {mean('original_region',1.):.2f}% 误差。当前开关给定后，它可以预先合并成每张输入的 10×8 系数矩阵，之后各组合只需一次小矩阵乘法。这个结果来自既有局部线性性质，并非新定理；约 {switch_queries:.2f}% 的高阶组合实际上跨过了至少一个 ReLU 分支，所以仍需真实干预来检验有限变化的误差。

合并实现通过了与原保存预测的等价核验，完整 Q256 流程约 {speed:.2f}×，已缓存摘要约 {cached_speed:.1f}×。它是看过结果后单独做的实现审计，原测试、旧计时和参数保持原样。更完整的有效输入计数、逐输入尾部误差与限制见 [残差块报告](残差块内部干预与背景选择_2026-09-13.md)。

研究接下来需要回答不同干预规模、不同位置下，局部线性、联合编码、状态选择分别何时失效，以及怎样决定何时退回精确计算。实验允许我们调整最初的方法判断；不能为了维持某个算法名称而忽略表现更强的对照。
"""
    story.write_text(text,encoding="utf-8")


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

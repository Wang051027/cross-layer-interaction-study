"""Presentation derived only from completed exact-switch experiment records."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import csv
import json
import os
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE/"runs/fashion_switch_20260914"
REPORT = "可翻转状态筛选与精确组合计算_2026-09-17.md"
MATH = "稳定状态合并与精确修正_数学说明_2026-09-17.md"
AUDIT = BASE/"audit/switch_interpretation_20260917.json"
POSITIONS = ("center", "upper_left")
LABELS = dict(center="中心", upper_left="左上")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


# 把已经生成的报告或记录文本写到指定路径。
def write(path, value):
    path.write_text(value,encoding="utf-8")


def stats(values):
    return float(np.mean(values)), float(np.std(values,ddof=1))


# 按模型和输入汇总误差与计算成本。
def summarize():
    numerical = rows(RUN/"analysis/results.csv")
    groups = defaultdict(list)
    keys = ("phase","position","q","method","alpha")
    for row in numerical:groups[tuple(row[k] for k in keys)].append(row)
    accuracy = []
    for key, values in groups.items():
        assert len(values) == 6
        mean,sd = stats([float(v["nrmse"]) for v in values])
        accuracy.append(dict(zip(keys,key),seeds=6,nrmse_mean=mean,nrmse_sd=sd,
            max_abs=max(float(v["all_queries_max_abs"]) for v in values),
            input_p95_mean=float(np.mean([float(v["input_p95"]) for v in values])),
            zero_effect_mean=float(np.mean([int(v["zero_effect_inputs"]) for v in values]))))
    screening = rows(RUN/"analysis/screening.csv")
    keys = ("phase","seed","position","q")
    groups = defaultdict(list)
    for row in screening:groups[tuple(row[k] for k in keys)].append(row)
    by_seed = []
    for key, values in groups.items():
        J = int(values[0]["affected_states"])
        record = dict(zip(keys,key),inputs=len(values),affected_states=J,
                      candidate_min=min(int(v["candidates"]) for v in values),
                      candidate_max=max(int(v["candidates"]) for v in values))
        for k in ("candidates","stable_on","stable_off","candidate_fraction","switch_cache_bytes","exact_cache_bytes"):
            record[k] = float(np.mean([float(v[k]) for v in values]))
        record["cache_ratio"] = record["switch_cache_bytes"]/record["exact_cache_bytes"]
        by_seed.append(record)
    groups = defaultdict(list)
    keys = ("phase","position","q")
    for row in by_seed:groups[tuple(row[k] for k in keys)].append(row)
    screen_summary = []
    for key, values in groups.items():
        assert len(values) == 6
        record = dict(zip(keys,key),seeds=6,affected_min=min(v["affected_states"] for v in values),
                      affected_max=max(v["affected_states"] for v in values),
                      candidate_min=min(v["candidate_min"] for v in values),
                      candidate_max=max(v["candidate_max"] for v in values))
        for k in ("candidates","candidate_fraction","switch_cache_bytes","exact_cache_bytes","cache_ratio"):
            record[k+"_mean"],record[k+"_sd"] = stats([v[k] for v in values])
        screen_summary.append(record)
    groups = defaultdict(list)
    keys = ("position","q","query_set","queries","workload","method")
    for row in rows(RUN/"analysis/paired_timings.csv"):groups[tuple(row[k] for k in keys)].append(row)
    timing = []
    for key, values in groups.items():
        assert len(values) == 6
        ratios = [float(v["speedup"]) for v in values]
        mean,sd = stats(ratios)
        timing.append(dict(zip(keys,key),seeds=6,speedup_mean=mean,speedup_sd=sd,
            faster_seeds=sum(v>1. for v in ratios),minimum_speedup=min(ratios),maximum_speedup=max(ratios),
            method_ms=float(np.mean([float(v["seconds"])*1000 for v in values])),
            exact_ms=float(np.mean([float(v["exact_seconds"])*1000 for v in values]))))
    memory = rows(RUN/"timing/cost_counts.csv")
    result = dict(utc=datetime.now(timezone.utc).isoformat(),status="passed",accuracy=accuracy,
        screening_by_seed=by_seed,screening=screen_summary,timing=timing,
        retained_non_teacher_tensor_bytes_range=[min(int(v["retained_non_teacher_tensor_bytes"]) for v in memory),
            max(int(v["retained_non_teacher_tensor_bytes"]) for v in memory)],
        test=read(RUN/"test/completion.json"),development=read(RUN/"development_complete.json"),
        verification=read(RUN/"analysis/verification.json"))
    write(AUDIT,json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status="summarized",accuracy_groups=len(accuracy),timing_groups=len(timing),screen_groups=len(screen_summary))))


# 按当前报告固定条件筛选记录，供图表使用。
def select(records, **labels):
    return next(r for r in records if all(str(r[k]) == str(v) for k,v in labels.items()))


def plot():
    os.environ["MPLCONFIGDIR"] = str(BASE/"runs/residual_figure_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = read(AUDIT)
    fig,axes = plt.subplots(2,2,figsize=(12.4,8.8))
    for position,color,label in (("center","#147a68","Center"),("upper_left","#b27634","Upper left")):
        ss = [select(d["screening"],phase="test",position=position,q=q) for q in (4,8,12)]
        axes[0,0].errorbar((4,8,12),[100*r["candidate_fraction_mean"] for r in ss],
            yerr=[100*r["candidate_fraction_sd"] for r in ss],fmt="o-",color=color,capsize=4,label=label)
        axes[1,1].errorbar((4,8,12),[r["cache_ratio_mean"] for r in ss],
            yerr=[r["cache_ratio_sd"] for r in ss],fmt="o-",color=color,capsize=4,label=label)
        for ax,workload in ((axes[0,1],"from_image"),(axes[1,0],"cached")):
            ts = [select(d["timing"],position=position,q=q,query_set="evaluation",workload=workload,method="switch_exact") for q in (4,8,12)]
            ax.errorbar((4,8,12),[r["speedup_mean"] for r in ts],
                yerr=[r["speedup_sd"] for r in ts],fmt="o-",color=color,capsize=4,label=label)
    titles = ["A. States kept nonlinear", "B. Complete cost from image",
              "C. Query cost after preparation", "D. Actual per-input cache size"]
    ylabels = ["Candidates / affected states (%)", "Exact baseline time / screened exact time",
               "Exact baseline time / screened exact time", "Screened cache bytes / exact cache bytes"]
    for ax,title,ylabel in zip(axes.flat,titles,ylabels):
        ax.set_title(title,loc="left",fontweight="bold",fontsize=12)
        ax.set_xlabel("Number of source coordinates")
        ax.set_ylabel(ylabel,fontsize=10)
        ax.set_xticks((4,8,12))
        ax.grid(axis="y",color="#dddddd",linewidth=.6)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=9)
        if ax is not axes[0,0]:ax.axhline(1,color="#888888",ls=":",lw=1)
    axes[0,0].set_ylim(bottom=0)
    axes[0,1].set_ylim(bottom=0)
    axes[1,0].set_ylim(bottom=0)
    axes[1,1].set_ylim(bottom=0)
    fig.suptitle("Exact local intervention responses by screening ReLU switches",fontsize=16,x=.06,ha="left")
    fig.text(.06,.924,"Six frozen teachers; mean ± between-teacher SD. Numerical equality checked against the original unfused network.",fontsize=10,color="#444444")
    fig.text(.06,.045,"Timing: CPU float64, one thread, four fixed validation inputs; best of three matched exact engines.\nEvaluation queries: 16 / 256 / 335. The q=12 exhaustive 4096-query results are reported separately. Cache includes gathered weights.",fontsize=9,color="#444444")
    fig.tight_layout(rect=(0,.085,1,.91),h_pad=2,w_pad=2)
    for ext in ("png","pdf"):fig.savefig(BASE/f"reports/switch_exact_results.{ext}",dpi=190)
    plt.close(fig)
    print(json.dumps(dict(status="plotted")))


# 把表头和数据行组织为报告表格；Word/Markdown 的输出形式由本文件决定。
def table(headers, values):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in values)


# 把已有结果填入中文报告，保留该阶段实验范围与限制。
def report():
    d = read(AUDIT)
    # 在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
    def timing(position,q,query_set="evaluation",workload="from_image",method="switch_exact"):
        return select(d["timing"],position=position,q=q,query_set=query_set,workload=workload,method=method)
    # 按阶段和干预设置读取准确性汇总，区分验证与测试记录。
    def acc(position,q,method="switch_exact",alpha=1.,phase="test"):
        return select(d["accuracy"],position=position,q=q,method=method,alpha=alpha,phase=phase)
    exact_max = max(r["max_abs"] for r in d["accuracy"] if r["method"] == "switch_exact")
    screen_table = table(["位置","源数 q","受影响 J 范围","候选 K，均值±SD","候选比例","实际缓存比"],[
        [LABELS[p],q,f"{s['affected_min']}–{s['affected_max']}",f"{s['candidates_mean']:.2f} ± {s['candidates_sd']:.2f}",
         f"{100*s['candidate_fraction_mean']:.2f}%",f"{s['cache_ratio_mean']:.2f}×"]
        for p in POSITIONS for q in (4,8,12) for s in [select(d["screening"],phase="test",position=p,q=q)]])
    accuracy_table = table(["位置","q","精确筛选最大绝对差","原区域线性 NRMSE","上界回退 NRMSE"],[
        [LABELS[p],q,f"{acc(p,q)['max_abs']:.2e}",f"{100*acc(p,q,'affine_fused')['nrmse_mean']:.2f}%",
         f"{100*acc(p,q,'bounded_affine')['nrmse_mean']:.2f}%"] for p in POSITIONS for q in (4,8,12)])
    strength_table = table(["位置","q","α=0.5 精确最大绝对差","线性 NRMSE","回退 NRMSE"],[
        [LABELS[p],q,f"{acc(p,q,alpha=.5)['max_abs']:.2e}",f"{100*acc(p,q,'affine_fused',.5)['nrmse_mean']:.2f}%",
         f"{100*acc(p,q,'bounded_affine',.5)['nrmse_mean']:.2f}%"] for p in POSITIONS for q in (4,8,12)])
    timing_table = table(["位置","q / 查询数","完整流程加速±SD","完整流程胜出","已有背景后重新准备","已缓存查询加速"],[
        [LABELS[p],f"{q} / {timing(p,q)['queries']}",f"{timing(p,q)['speedup_mean']:.3f} ± {timing(p,q)['speedup_sd']:.3f}×",
         f"{timing(p,q)['faster_seeds']}/6",f"{timing(p,q,workload='from_context')['speedup_mean']:.3f}×",
         f"{timing(p,q,workload='cached')['speedup_mean']:.2f}×"] for p in POSITIONS for q in (4,8,12)])
    full_table = table(["位置","完整流程加速±SD","完整流程胜出","精确基线 ms","筛选精确 ms","已缓存加速"],[
        [LABELS[p],f"{t['speedup_mean']:.2f} ± {t['speedup_sd']:.2f}×",f"{t['faster_seeds']}/6",f"{t['exact_ms']:.3f}",f"{t['method_ms']:.3f}",
         f"{timing(p,12,'exhaustive','cached')['speedup_mean']:.2f}×"] for p in POSITIONS for t in [timing(p,12,"exhaustive")]])
    low_table = table(["位置","q / 低阶查询数","完整流程加速","已缓存加速"],[
        [LABELS[p],f"{q} / {t['queries']}",f"{t['speedup_mean']:.3f}×",f"{timing(p,q,'low_order','cached')['speedup_mean']:.2f}×"]
        for p in POSITIONS for q in (4,8,12) for t in [timing(p,q,"low_order")]])
    control_table = table(["位置","q","精确筛选","线性近似","上界＋回退"],[
        [LABELS[p],q,*[f"{timing(p,q,method=m)['speedup_mean']:.3f}×" for m in ("switch_exact","affine_fused","bounded_affine")]]
        for p in POSITIONS for q in (4,8,12)])
    freeze = read(RUN/"freeze.json")
    text = f"""# 可翻转状态筛选与精确组合计算

更新：2026-09-17。协议及实现于 9 月 14 日固定，本轮继续完成验证。这是沿用六个冻结残差教师的后续检查；当前方法将区间内稳定的 ReLU 合并为线性响应，精确执行其余状态。全部验证与新批次中，该方法相对原网络的最大绝对差为 **{exact_max:.3e}**，处于 float64 数值误差量级。

本轮的关键判断是：在指定干预范围内，可以精确减少每个组合需要计算的非线性状态。完整流程的速度取决于查询量；下表同时保留准备成本、缓存成本和未加速的设置。这个结论限于当前最后一个残差分支。

计入准备后，q=4 的全部 16 组合在两个位置都更慢，平均速度比约 {timing('center',4)['speedup_mean']:.2f}×／{timing('upper_left',4)['speedup_mean']:.2f}×；q=8 的 256 组合为 {timing('center',8)['speedup_mean']:.2f}×／{timing('upper_left',8)['speedup_mean']:.2f}×。q=12 的全部 4096 组合在固定验证输入上达到 {timing('center',12,'exhaustive')['speedup_mean']:.2f}×／{timing('upper_left',12,'exhaustive')['speedup_mean']:.2f}×，两个位置均为 6/6 教师加速。这里依次为中心／左上，速度比均相对同轮最快精确基线。

## 1. 从上一步的失败得到新方法

上一轮先问“线性近似的整体误差能否被上界保证”，不能保证时回退完整计算。上界偏保守，又增加准备开销，导致完整流程变慢。它同时给出一个更直接的信息：对当前图片及固定源集合，有些 ReLU 在所有允许组合下始终同号，有些可能触零或翻转。

本轮把始终开启的状态乘入一个 10×q 线性响应矩阵，始终关闭的状态贡献为零，其余 K 个状态保留原始 ReLU 差分。输入换了，K 和矩阵会重新计算；同一图片、源集合和 α∈[0,1] 范围内的所有组合可以共用一次准备。它不依赖上一轮的贪心拟合或 PCA，也不使用 10% 接受门槛。

完整公式与逐步推导见 [数学说明]({MATH})。这是已有分段线性性质和区间筛选的具体计算分解；本报告不据此主张新定理或首次提出。

## 2. 固定范围与数据访问

- 复用 6 个冻结残差教师，平均已有分类准确率 92.335%；没有重训或新增分类评估。
- 复用 36 组源坐标：中心与左上 2 个窗口，q=4／8／12，源集合由前轮 512 个训练上下文固定。新方法没有参数拟合。
- 128 个原验证上下文复用上一轮原网络参考，共 1,864,704 个参考向量；属于已见开发证据。
- 固定实现与 {len(freeze['artifacts'])} 个开发文件后，开启新干预索引 **896–1023**，共 128 张。每病例先保存预测和筛选状态，再运行原始未融合网络得到参考，共 **1,864,704 个新批次参考输出向量**。
- q=4 全部 16 组合，q=8 全部 256 组合；q=12 的新批次使用此前固定的 335 组合，其中 79 个低阶、256 个高阶抽样。不能说 128 张新图片均已遍历 4096 组合。
- 单独在每个教师、位置的前 4 个既有验证输入上，检查 q=12 的全部 **4096 组合**与两种强度，共 48 个教师—位置—输入配对、393,216 个新生成参考向量。

新索引与旧四批 0–895 不重叠，但整个官方分类池已有访问；它仍是同一 Fashion-MNIST 的描述性后续检查。项目仍为 18 个独立训练教师，六残差教师重复使用。五批累计 120 个实验设置。5000 张校准图仍未进入模型，本轮没有新的确认性 p 值。

## 3. 是否仍然精确

完全置零 α=1 的新批次结果如下。最大绝对差覆盖所有已查询组合及 10 个输出；NRMSE 仍按阶数大于 2 的组合计算，为六教师均值。

{accuracy_table}

减半干预 α=0.5 使用同一个按最大强度 α=1 准备的候选集合：

{strength_table}

主实现之外，已检查 24 个合成情形：随机、全开、全关、全部候选、区间端点及邻近浮点数、零源值，覆盖 q=1／4／8／12 和 α=0／0.25／0.5／1。另在原网络中比较 CPU 与 CUDA 的未融合执行、三种背景获取路径及新准备／已缓存结果。

保存后用另一套 NumPy 算式重新计算每个输入的区间、筛选集合与预测。**624 组数组指标回读一致**：验证 288、完整 4096 组合补充 48、新测试 288。NumPy 重算的最大预测差为 {d['verification']['max_numpy_split_difference']:.3e}，所有省去状态的非线性余项最大为 {d['verification']['max_omitted_remainder']:.3e}。实数代数保证与这些浮点核对相互补充；代码没有采用向外舍入，因此不是形式验证证书。

## 4. 真正减少了哪些计算

新测试批次中，每个教师先对 128 个输入取均值，再对六教师计算均值及标准差。K 是可能触零或翻转而保留的状态数，不等于每个实际组合真的翻转的状态数。

{screen_table}

一次准备仍然获取全部 J 个当前背景值，检查所有 J×q 个贡献，并构造稳定响应矩阵和抽取候选权重。节省发生在之后各组合的非线性计算。原精确查询的大致计算量随 QJ(q+10) 增长；新查询随 QK(q+10)+10Qq 增长，同时增加一次 O(Jq+10Jq) 的筛选与合并成本。这里省略常数、内存搬运和框架开销，因此实际收益必须以计时判断。

“候选更少”也不自动代表缓存更小。当前缓存含 q 个源值、10q 个系数，以及 K(q+11) 个候选系数／状态，总计 8[11q+K(q+11)] 字节；原精确缓存只需 8(q+J) 字节。上表缓存比计入为每个输入抽取的 W、U 和 z，较大 K 时可能超过原缓存。完整局部构造映射仍被保留，非教师张量占用范围 {d['retained_non_teacher_tensor_bytes_range'][0]/1024/1024:.2f}–{d['retained_non_teacher_tensor_bytes_range'][1]/1024/1024:.2f} MiB；另有教师权重。该内存口径不含 Python／NumPy 开销及查询临时量，不是峰值进程内存，也不是模型压缩率。

## 5. 包括准备之后，究竟快不快

所有方法同为单线程 CPU float64，用相同四个验证输入、随机顺序、2 次预热、7 轮、每轮 3 次调用，共 **42,336 条原始计时**。9 月 14 日计时进程中断，未落盘完整结果；9 月 17 日保留中断记录并以同一冻结实现重跑全部配对计时，不拼接两次运行。先按输入取七轮中位数，再对四输入取均值；每个设置选择同一轮三种精确获取实现中最快者作分子，最后汇总六个教师的配对速度比。大于 1 表示加速。CPU／CUDA 只用于参考一致性核对，没有把两种设备的耗时相除。

“完整流程”从图像开始，计入共同前缀、完整背景获取、筛选、线性合并、权重抽取及所有查询。“已有背景后重新准备”假定 h,z 已获取，但每次仍执行方法准备。“已缓存查询”只计已完成全部准备后的查询成本。固定教师与源坐标对应的 W、U 和获取索引作为所有方法共享的一次性结构构建，不包含在逐输入计时中；本轮记录其保留存储，但没有测量改变源集合后的结构构建摊销。精确基线直接复用已获 h,z，没有额外的无用上下文复制。

{timing_table}

低阶小查询量也按预先固定的方案完整保留：

{low_table}

q=12 的全部 4096 组合计时：

{full_table}

上表毫秒数分别对六个教师平均；加速比是逐教师配对比值的平均，因此不应直接用两列均值相除来回算。4096 组合的数值验证及计时均使用四个固定验证输入，不代表 128 个新测试输入的耗时测量。

同一次计时中的旧方法控制，均为从图像开始的评价查询集：

{control_table}

线性近似速度必须与它自己的误差一起读；上界回退也必须与其自身误差和耗时一起读。新方法的精确性与这些近似控制的速度不能拼成同一种方法的表现。PCA 与贪心的前轮比较仍然有效，本轮没有重新计时它们，也不主张新法在所有预算、内存限制和模型结构下优于它们。

![筛选比例、完整耗时、缓存查询与实际缓存](switch_exact_results.png)

## 6. 对原设想与下一步的含义

你最初希望从相邻节点的组合变化，追踪到最终输出，并减少逐组合运行大网络的开销。现在可具体实现为：保留当前图片提供的背景，把不会因这些组合改变开关的状态合并，再精确处理会改变分支的部分。固定卷积连接提供局部作用范围，分段线性结构提供可合并的部分，完整干预参考检验结果是否正确。

这使原先“哪些背景值得近似保留”的问题，进一步变成“在规定干预范围内，哪些背景必须保留非线性计算”。它仍然不是可读语义规则或跨输入通用的树，也没有消除组合枚举本身：Q=2^q 仍可能指数增长，当前减少的是每个查询的后续计算量。

下一步优先把干预向更早一层移动，使变化连续经过至少两组非线性操作，检查这种筛选能否逐层传播，并分别记录边界膨胀、候选数量和完整成本。另需在独立数据集与其他结构上验证外推，并在单独冻结的实验中比较更多实现、精度与硬件。现在的结论限定在最后一个残差分支的一个受影响 ReLU 层和线性读出。

## 7. 可复核文件

- [执行协议](../runs/fashion_switch_20260914/protocol.json)、[实现快照记录](../runs/fashion_switch_20260914/implementation.json)、[测试前冻结](../runs/fashion_switch_20260914/freeze.json)。
- [测试完成记录](../runs/fashion_switch_20260914/test/completion.json)、[数组与独立算式核验](../runs/fashion_switch_20260914/analysis/verification.json)、[六教师配对计时](../runs/fashion_switch_20260914/analysis/paired_timings.csv)。
- [逐输入筛选记录](../runs/fashion_switch_20260914/analysis/screening.csv)、[机器可读结果](../audit/switch_interpretation_20260917.json)、[独立交付检查](../audit/final_switch_delivery_20260917.json)。
- [从原始设想到当前实验](从原始设想到当前实验_2026-09-11.md)。

原始来件、全部历史冻结数值与旧清单继续保留；本阶段不重建已删除的 ZIP。
"""
    write(BASE/"reports"/REPORT,text)
    math_text = r"""# 稳定状态合并与精确修正：数学说明

固定一张图片、最后一个残差分支内部的 q 个源激活 h，以及保持原值的其他激活和恒等支路。将推理时 BatchNorm 融入卷积，把受影响的 J 个 ReLU 前值记为 z，源到这些状态的线性映射为 W∈R^{J×q}，其后 GAP 与线性读出合并为 U∈R^{10×J}。

对掩码 b∈{0,1}^q，使用 δh=−α(h⊙b)，α∈[0,1]。精确输出变化为

\[
\Delta F=U\{\operatorname{ReLU}(z+W\delta h)-\operatorname{ReLU}(z)\}.
\]

其他不受影响的空间坐标及读出偏置在差分中消去；恒等支路没有被一起干预。这个前提决定当前公式的适用范围。

## 1. 一次求出所有允许组合的状态范围

令 a_{ji}=−W_{ji}h_i，则

\[
\ell_j=\sum_i\min(a_{ji},0),\qquad u_j=\sum_i\max(a_{ji},0).
\]

每个状态的变化 t_j=αΣ_i a_{ji}b_i 满足 ℓ_j≤t_j≤u_j。因为 ℓ_j≤0≤u_j，按最大 α=1 计算的区间也覆盖所有 α∈[0,1]。在 α=1 时，每个单独状态的这两个端点都可以由相应的二进制掩码取得；不同状态取得端点所用的掩码不必相同。

定义始终开启集合 S₊={j:z_j+ℓ_j>0}，始终关闭集合 S₋={j:z_j+u_j<0}，其余为 C。端点触零的情况也放入 C，虽可能多算，却不损害恒等式。

在当前代码中，正负边界使用额外保护量

\[
\eta_j=64\,\epsilon_{64}\,(|z_j|+u_j-\ell_j+1),
\]

并把严格条件改为 z_j+ℓ_j>η_j 与 z_j+u_j<−η_j。因此非常接近边界的状态保留精确计算。实现采用 float64 保护余量，并用独立数值计算核对边界。

## 2. 把稳定部分合并

对 j∈S₊，ReLU(z_j+t_j)−ReLU(z_j)=t_j；对 j∈S₋，差分为零。因此

\[
M=U_{:,S_+}W_{S_+,:},\qquad
\Delta F=M\delta h+
U_{:,C}\{\operatorname{ReLU}(z_C+W_{C,:}\delta h)-\operatorname{ReLU}(z_C)\}.
\]

这是原精确表达式按三个不相交集合直接拆开的结果，没有丢弃候选状态，也没有用训练平均门控代替当前门控。第一项是固定背景下合并后的线性响应；第二项承担全部可能的非线性变化。在实现里为 Q 个掩码按行计算时，相应矩阵均使用转置。

S₊、S₋、C 与 M 都依赖当前图片及被允许变化的源集合。换图片、换源集合、扩大强度范围，都必须重新准备；当前实现拒绝 α<0 或 α>1。已有准备可覆盖同一盒内的全部二进制掩码，事实上也覆盖每个源的连续缩放系数位于 [0,1] 的情形；本轮主评价仍为二进制掩码与两种固定强度。

## 3. 一个可以手算的例子

只有一个源，h=1，W=(1,1,1)ᵀ，z=(2,0.2,−2)ᵀ，读出 U=(1,1,1)。源完全置零时 δh=−1，各状态变化范围均为 [−1,0]。

第一个状态始终开着，输出变化 −1；第三个始终关着，变化 0。只有第二个可能翻转，其精确变化为 ReLU(0.2−1)−ReLU(0.2)=−0.2。因此总变化为 −1.2。

若全部采用原区域切线，会把前两个状态都当成线性，得到 −2，发生误差。新方法把第一个状态合并为线性系数，保留第二个的 ReLU，仍得到原网络的 −1.2。每张图先获取背景状态，再减少后续查询中的 ReLU 计算量。

## 4. 成本与结论的边界

准备必须先获得所有 J 个受影响背景值，求 J×q 个区间贡献，再构造 M 和抽取 W_C、U_C、z_C。K=|C| 很小时，后续查询能更便宜；Q 很小时，准备开销可能大于节省。内存统计同时计入候选缓存的系数。

这是一个单层受影响 ReLU 后接线性读出的恒等分解。更早干预若继续跨越多层 ReLU 或最大池化，需要新的逐层传播与候选边界处理。组合个数 Q 自身也未减少。

完整数值证据、计时与限制见 [实验报告](可翻转状态筛选与精确组合计算_2026-09-17.md)。
"""
    write(BASE/"reports"/MATH,math_text)
    # Append the new stage while keeping all earlier stages explicitly historical.
    story_path = BASE/"reports/从原始设想到当前实验_2026-09-11.md"
    story = story_path.read_text(encoding="utf-8").split("<!-- SWITCH_EXTENSION -->")[0].rstrip()
    story += f"""

<!-- SWITCH_EXTENSION -->
## 16. 9 月 17 日：把可能翻转的部分精确算出来

上一步已经知道，线性近似何时容易错，以及整体误差上界为什么容易过于保守。继续往下，我们问得更细：到底是哪几个状态需要保留非线性？

对每张图片，我们先求所有允许源组合带来的取值范围。整个范围内都开着的状态，可以合并成线性系数；整个范围内都关着的状态，对差分没有贡献；会触零或翻转的状态继续计算原始 ReLU。这样即使某些开关改变，输出也能保持原网络的精确响应。

你原来“从局部组合连到整体结果”的想法，现在对应一条明确的计算路径：卷积给出局部影响矩阵 W，当前图片给出背景 z，稳定开关合并为小矩阵 M，可能变化的开关精确处理，最后由 U 汇总成十维输出变化。我们不必先把网络画成树，便能逐个组合检查这条路径是否正确。

六个教师、两个窗口和三种源数量的新批次验证中，差异保持浮点舍入量级；另在固定验证图片上遍历了 12 节点的全部 4096 组合。从图像开始，这个 4096 查询情形的六教师平均加速为中心 {timing('center',12,'exhaustive')['speedup_mean']:.2f}×、左上 {timing('upper_left',12,'exhaustive')['speedup_mean']:.2f}×。小查询量的准备成本仍须看完整表格，不能只报告缓存后的数字。

这个进展也让原目标更清楚：当前可以减少每个组合重复执行的非线性计算，但组合枚举本身还在，缓存也可能因保存候选系数而变大。更早层、多层传播、其他数据集和可读语义机制仍是后续问题。完整结果与算例见 [本轮报告]({REPORT}) 和 [数学说明]({MATH})。
"""
    write(story_path,story)
    readme_path = BASE/"README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("更新：2026-09-14。", "更新：2026-09-17。", 1)
    if "<!-- SWITCH_LATEST -->" in readme:
        start,end = readme.index("<!-- SWITCH_LATEST -->"),readme.index("<!-- END_SWITCH_LATEST -->")+len("<!-- END_SWITCH_LATEST -->")
        readme = readme[:start]+readme[end:]
    latest = f"""<!-- SWITCH_LATEST -->

最新结果见 [可翻转状态筛选与精确组合计算](reports/{REPORT})。六个冻结残差教师、两个位置、4／8／12 源节点共 36 个设置已完成新批次验证。区间内稳定的状态合并为线性响应，其余状态精确计算；相对原网络最大绝对差为 {exact_max:.2e}。

本轮包括完整背景获取、筛选与准备成本的配对计时，以及 12 节点全部 4096 组合的固定验证输入补充检查。4096 查询完整流程平均加速中心 {timing('center',12,'exhaustive')['speedup_mean']:.2f}×、左上 {timing('upper_left',12,'exhaustive')['speedup_mean']:.2f}×；小查询量的开销和实际缓存增长同时报告。624 组数组回读复核通过，5000 张校准图仍未进入模型。当前状态以本报告和 RESEARCH_STATUS 为准。

<!-- END_SWITCH_LATEST -->

"""
    readme = readme.replace("<!-- SCALE_LATEST -->",latest+"<!-- SCALE_LATEST -->",1)
    readme = readme.replace("最新结果见 [规模位置变化与精确回退]","前一阶段结果见 [规模位置变化与精确回退]")
    readme = readme.replace("当前状态以本报告和 RESEARCH_STATUS 为准。\n\n<!-- END_SCALE_LATEST -->","该段保留规模与位置阶段的历史结果。\n\n<!-- END_SCALE_LATEST -->")
    readme = readme.replace("`audit/final_scale_delivery_20260914.json`：当前阶段的最终保全与交付核查。","`audit/final_scale_delivery_20260914.json`：规模与位置阶段的历史保全与交付核查。")
    if "`runs/fashion_switch_20260914/`" not in readme:
        readme += "\n- `runs/fashion_switch_20260914/`：精确状态筛选、完整组合补充验证、第五批干预与配对计时。\n- `audit/final_switch_delivery_20260917.json`、`audit/switch_manifest_20260917.json`：本阶段交付核查与独立哈希快照。\n"
    write(readme_path,readme)
    status_path = BASE/"RESEARCH_STATUS.json"
    status = read(status_path)
    status["scale_followup_summary"] = dict(run="runs/fashion_scale_20260914",cases=36,intervention_ids=[768,895],full_tail_output_vectors=1864704)
    status.update(date="2026-09-17",current_run="runs/fashion_switch_20260914",latest_followup_run="runs/fashion_switch_20260914",
        latest_report="reports/"+REPORT,followup_stage="exact switch screening and matched full-cost validation completed",
        external_official_test_batches=5,completed_fashion_intervention_cases=120,completed_fashion_intervention_settings=120,
        followup_model_layer_cases=36,followup_intervention_ids=[896,1023],followup_full_tail_output_vectors=1864704,
        switch_screening_cases=36,switch_exact_max_abs=exact_max,switch_arrays_rescored=624,
        switch_supplementary_exhaustive_q=12,switch_supplementary_reference_vectors=393216,
        switch_exhaustive_from_image_speedup={p:timing(p,12,"exhaustive")["speedup_mean"] for p in POSITIONS},
        switch_new_teacher_training_runs=0,switch_scope="same final residual branch; single affected ReLU layer, q12 exhaustive only on4 fixed validation inputs",
        next_steps=["propagate exact stable/candidate decomposition through at least two downstream nonlinear layers after an earlier intervention",
                    "validate external datasets and architectures with explicit preparation, per-query and memory accounting"])
    write(status_path,json.dumps(status,ensure_ascii=False,indent=2))
    card_path = BASE/"data/fashion_mnist/DATA_CARD.md"
    card = card_path.read_text(encoding="utf-8").split("<!-- SWITCH_ACCESS -->")[0].rstrip()
    card += f"""

<!-- SWITCH_ACCESS -->
9 月 14 日固定可翻转状态筛选的协议与数值实现，9 月 17 日完成全部验证。复用六个残差教师及前轮 36 组源坐标，没有重训、方法拟合或新分类评估。128 个开发上下文复用既有独立参考；q=12 在每教师、位置的前四个验证输入上另遍历全部 4096 组合，两种强度共 393,216 个新参考向量。

冻结 {len(freeze['artifacts'])} 个开发文件后，开启第五批组合干预索引 **896–1023**。128 张图像与前四批不重叠，产生 1,864,704 个原始未融合网络参考向量。q=4／8 全枚举，q=12 仍使用原固定 335 组合。预测和筛选上下文在新参考生成前保存。624 组预测指标与独立 NumPy 算式核验通过。

项目仍有 18 个独立训练教师，五批累计 120 个实验设置；本轮 36 个设置复用旧教师，补充完整组合检查也使用旧验证输入。5000 张校准图仍未用于模型评价。官方分类池已有访问，全部批次仍来自同一数据集，均按描述性后续诊断报告。详见 [本轮报告](../../reports/{REPORT}) 与 [测试前冻结](../../runs/fashion_switch_20260914/freeze.json)。
"""
    write(card_path,card)
    book_path = BASE/"RUNBOOK.md"
    book = book_path.read_text(encoding="utf-8").split("<!-- SWITCH_RUNBOOK -->")[0].rstrip()
    book = book.replace("报告和状态当前由 `write_scale_report.py` 更新", "规模阶段报告和状态曾由 `write_scale_report.py` 更新；此命令现仅作历史记录，不再写回当前状态")
    book += r"""

<!-- SWITCH_RUNBOOK -->
## 可翻转状态筛选与精确组合计算

本轮数值实现为 `switch_context.py` 和 `run_switch_controls.py`，固定输出目录 `runs/fashion_switch_20260914/`。继承的规模、残差及更早数值脚本和运行文件均不修改。已执行的顺序为：

```powershell
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage init
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage smoke
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage lock
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage develop
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage timing
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage freeze
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage test
& $researchPython -B "$researchScripts\run_switch_controls.py" --stage analyze
& $researchPython -B "$researchScripts\report_switch_results.py" --stage summarize
& $researchPython -B "$researchScripts\report_switch_results.py" --stage plot
& $researchPython -B "$researchScripts\report_switch_results.py" --stage report
& $researchPython -B "$researchScripts\verify_switch_delivery.py"
```

上述命令是已完成实验的记录，数值阶段拒绝覆盖既有文件，也没有 `--run-dir` 参数。若需要重跑，先在独立副本指定新目录并形成新协议；不得修改冻结原件。报告生成与最终快照在未来阶段也转为历史命令，避免覆盖当时清单。

算法按 α∈[0,1] 的所有源子集做一次区间筛选，严格稳定的开关合并为线性响应，边界及其他状态保留精确 ReLU。全部获取、筛选、矩阵合并与权重抽取计入从图像开始的计时。固定评价查询是 q4=16、q8=256、q12=335；q12 的全部4096查询单独在每教师、位置的前4个验证输入上核验与计时。

新批次索引896–1023在开发冻结后开启，原网络参考使用CUDA float64。所有速度比较都是同一轮单线程CPU float64，42,336条计时，三种工作口径。624组数组指标另用NumPy区间及分解算式核对。新方法没有拟合，不新增教师、分类评价或ZIP，不访问5000张校准图像的模型输出。
"""
    write(book_path,book)
    print(json.dumps(dict(status="reported",report=REPORT),ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=("summarize","plot","report"))
    globals()[parser.parse_args().stage]()

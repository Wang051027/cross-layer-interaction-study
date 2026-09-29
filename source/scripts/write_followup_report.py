# 中文阅读说明｜用途：生成首轮导入复核和 Fashion-MNIST 验证阶段的历史报告。
# 数据流：输入复核、成本及首教师结果；执行时会写旧版首页和状态。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Build the handoff from saved, actually executed results."""
from __future__ import annotations
import csv
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]


# 用途：读取 CSV 表格为记录列表，供报告查找和汇总。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 用途：生成首轮导入复核和 Fashion-MNIST 验证阶段的历史报告。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    replay=BASE / "runs/upstream_replay_20260910_v2"
    timing=BASE / "runs/pca_cost_audit_20260910_v2"
    pilot=BASE / "runs/fashion_pilot_20260910"
    audit=json.loads((replay / "audit.json").read_text())
    teacher=json.loads((pilot / "teacher_summary.json").read_text())
    imported=json.loads((BASE / "audit/upstream_import.json").read_text(encoding="utf-8"))
    completed=json.loads((pilot / "completion.json").read_text())
    old=read_csv(replay / "quarter_summary.csv")
    oldtime=read_csv(timing / "timing_summary.csv")
    new=read_csv(pilot / "validation_results.csv")
    newtime=read_csv(pilot / "timing/timing_summary.csv")
    assert audit["status"]=="passed" and completed["status"]=="completed"
    # 用途：提取原包对应数据集、层和方法的误差，供接续报告对照。
    # 参数：ds、layer、method；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：100 * float(next((r for r in old if r['dataset'] == ds and r['layer'] == str(layer) and…。
    def old_error(ds,layer,method):
        return 100*float(next(r for r in old if r["dataset"]==ds and r["layer"]==str(layer) and r["method"]==method)["nrmse_mean"])
    # 用途：提取新增实验对应层、方法和预算的误差。
    # 参数：layer、method、k；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：100 * float(next((r for r in new if r['layer'] == str(layer) and r['method'] == method …。
    def new_error(layer,method,k):
        return 100*float(next(r for r in new if r["layer"]==str(layer) and r["method"]==method and r["k"]==str(k) and r["alpha"]=="1.0")["nrmse"])
    old_table="\n".join(f"| {ds.title()} L{layer} | {old_error(ds,layer,'greedy'):.2f}% | {old_error(ds,layer,'weight'):.2f}% | {old_error(ds,layer,'pca_fused'):.2f}% |"
                         for ds in ("digits","synthetic") for layer in (1,2))
    old_timing_rows=[]
    for ds in ("digits","synthetic"):
        for layer in (1,2):
            matched={r["method"]:r for r in oldtime if r["dataset"]==ds and r["layer"]==str(layer) and r["queries"]=="256" and r["workload"]=="from_image"}
            old_timing_rows.append(f"| {ds.title()} L{layer} | {float(matched['greedy']['ms_mean']):.3f} | {float(matched['pca_fused']['ms_mean']):.3f} | {float(matched['exact']['ms_mean']):.3f} | {float(matched['greedy']['paired_speedup_mean']):.3f}× |")
    new_table="\n".join(f"| L{layer} | {new_error(layer,'greedy',16):.2f}% | {new_error(layer,'weight',16):.2f}% | {new_error(layer,'pca_fused',16):.2f}% | {new_error(layer,'greedy',32):.2f}% |" for layer in (1,2))
    pilot_timing_rows=[]
    for layer in (1,2):
        matched={r["method"]:r for r in newtime if r["layer"]==str(layer) and r["queries"]=="256" and r["workload"]=="from_image"}
        pilot_timing_rows.append(f"| L{layer} | {float(matched['greedy']['ms_mean']):.3f} | {float(matched['pca_fused']['ms_mean']):.3f} | {float(matched['exact']['ms_mean']):.3f} |")
    report=f"""# 研究包复核与首轮 Fashion-MNIST 实验

2026-09-10。本报告对应实际执行结果；此前的接续方案保留为历史草案。

**判断：旧结果可以重现，PCA 的获取成本需要修正；首轮新任务显示，保留 25% 头部状态尚达不到原定误差门槛。下一步应补齐原六个种子的独立测试，重点考察误差要求下的完整成本。**

## 1. 收到的材料与复核范围

归档：`CAT_MT_background_selection_v0.7.zip`，98,417,930 字节。SHA-256：

`{imported['archive_sha256']}`

以 [导入记录](../audit/upstream_import.json) 中的完整 SHA-256 为准。压缩包按路径检查后解压；原清单 271/271 文件均匹配，原归档及解压内容保留。

重新载入 12 个冻结教师，覆盖两个数据集、六个种子和两个层，共 24 个设置。重新评分 72 组已保存主预测，重新拟合选择器，并对每个设置的全部 128 个历史留出输入执行两种强度、全部 256 种组合的原网络尾部，共 1,572,864 个干预输出向量。主 NRMSE 只统计 219 个三阶及以上组合。

- 保存预测的重放最大差异：{audit['maximum_errors']['prediction']:.3g}。
- 原网络尾部重放最大差异：{audit['maximum_errors']['full_tail']:.3g}。
- 局部精确执行与完整尾部最大差异：{audit['maximum_errors']['exact_local']:.3g}。
- 所有主预算的重新拟合预测均保持一致；4 个 Synthetic 设置在几乎零增益的排序末端有浮点交换，未影响主预算。第一次严格要求全排序逐项相等的检查因此中止，诊断及修正后的审计均保留。

这是关键数值结论的本机复核，没有重新训练旧教师，也没有重算原包全部辅助预算表。归档哈希可证明收到的文件一致，不能独立证明历史实验的执行时间或“预测先于参考答案”的时间顺序。历史 Digits/Synthetic 仍属已探索的开发数据。

详见 [完整复核记录](../runs/upstream_replay_20260910_v2/audit.json)、[逐种子结果](../runs/upstream_replay_20260910_v2/quarter_results.csv) 和 [排序差异诊断](../audit/refit_order_diagnostic.json)。

## 2. PCA 成本修正及其影响

源码明确对 FC1 的 ReLU 前预激活做 PCA。令预激活为 z = Wp + b，V 的每行是一个主成分，则：

```
c = V(z - μ) = (VW)p + V(b - μ)
z_hat = μ + V.T c
```

因此可以预先合并编码器，在线直接得到 k 个编码值，无需先计算全部 J 个原始预激活。这里利用的是线性代数等价关系，不是新增的研究创新点。

已实现并检查融合与未融合路径，历史 24 个设置、两种强度上的预测最大差异为 {audit['maximum_errors']['pca_fusion']:.3g}。PCA 的均值、基、拟合数据和预测公式都保持原样。

PCA 评价时仍须重构 J 个近似预激活，并计算完整的非线性响应头；完整响应矩阵、解码器及临时量不能省略。编码的获取维度可以是 k，并不意味着全部计算也缩到 k。当前代码统计执行数组与传递状态，不等同于部署后的峰值内存；共享教师、Python 对象及分配器开销未纳入数组字节数。

主预算为 J/4、完全置零时，六个历史教师的平均 NRMSE：

| 数据与层 | 背景选择 | 权重范数选择 | 融合 PCA |
|---|---:|---:|---:|
{old_table}

PCA 在 Digits 上误差更低，背景选择在 Synthetic 上更低。因此既不能忽略 PCA，也不能据单个数据集宣布某一种方法普遍更好。

## 3. 相同后端下的成本

Windows，PyTorch {audit['environment']['torch']}，NumPy {audit['environment']['numpy']}；CPU float64、单线程。每个模型—层取 4 个历史留出输入，Q=37 和 Q=256；每个输入预热后进行 15 轮随机方法顺序计时，每轮调用 3 次，先取输入内中位数，再按教师汇总。计时期间没有运行新教师训练。

覆盖三种场景：原始图像开始；源层已经计算；完整背景已有缓存。缓存场景允许精确基线直接复用完整 z，不添加不必要的复制。所有场景都计入其相应的选择／编码及输出计算。方法不进行逐输入动态选点或回退；离线拟合和矩阵融合不计入在线耗时，尚未计算摊销阈值。

以下为从图像开始、Q=256 的平均毫秒数；加速比按同一教师的精确耗时/选择耗时配对后取均值：

| 数据与层 | 背景选择 ms | 融合 PCA ms | 局部精确 ms | 背景选择加速比 |
|---|---:|---:|---:|---:|
{chr(10).join(old_timing_rows)}

总体收益有限。完整背景已经缓存时，选择法相对精确执行的均值加速比约为 0.91–1.04 倍，部分设置更慢。不同后端、硬件和调用规模不能直接沿用这些数字；微小耗时差异存在明显波动，独立测量的场景不能相减来估算某一个算子的成本。

![误差与完整在线耗时](accuracy_cost_audit.png)

图中误差条是六个教师的标准差，不是置信区间。所有原始重复计时、逐教师汇总和数组计数见 [计时目录](../runs/pca_cost_audit_20260910_v2/)。初版计时在修正精确缓存基线的冗余复制后被替代，最终报告使用 v2。

## 4. 已执行的新任务：一个固定教师的训练／验证实验

Fashion-MNIST 四个官方文件已通过发布方 MD5 和 IDX 检查。采用原包分组算法及种子 2026090717，划分 50,000 训练、5,000 验证、5,000 校准；每类分别为 5,000／500／500。70,000 张图像未发现精确像素重复；未做感知近重复排查。

本轮只让训练、验证输入进入模型。官方 10,000 张测试图像未做任何模型评价，校准数据也未进入模型。测试和校准索引已经预留。数据完整性检查与模型测试是不同的访问范围。

保留原外部协议的首个种子 202609071、网络结构、AdamW、批量 128、学习率 0.001、权重衰减 0.0001、最多 15 轮、验证交叉熵选择及耐心值 4。训练前记录了原协议未写明的执行细节和融合 PCA 修订，保存协议及源码哈希。没有更换种子、按结果淘汰教师或改动原主比较。

教师实际训练 {teacher['epochs']} 轮，选中第 {teacher['best_epoch']} 轮；5,000 个验证输入上的准确率为 **{teacher['best_validation_accuracy']*100:.2f}%**，交叉熵 {teacher['best_validation_cross_entropy']:.5f}。训练使用 GPU float32，干预审计使用 CPU float64。该准确率参与过 checkpoint 选择，不能当作独立测试准确率。

每层用固定 512 个训练输入拟合，用固定 128 个验证输入检查。源坐标在训练平均激活的中央 3×3 范围内选定，q=8。主强度为完全置零，次强度为减半；每个验证输入执行全部 256 个真实尾部输出，NRMSE 统计 219 个高阶组合。

完全置零的结果如下，NRMSE 越小越好：

| 层 | 选择 k=16 | 权重范数 k=16 | 融合 PCA k=16 | 选择 k=32 |
|---|---:|---:|---:|---:|
{new_table}

**两层的 k=16 均未达到 10% 的总体误差门槛。** 背景选择在这一个教师上优于同预算权重范数及 PCA，但单教师验证结果不能证明跨种子的稳定优势。与旧 Digits 的较低误差相比，新任务说明固定 25% 头部预算不能直接推广。

事先登记的验证预算集合选出 k=32。其总体 NRMSE 虽低于 10%，逐输入 NRMSE 的第 95 百分位仍约为 13.11%／18.18%，因此不构成逐输入误差保证。L2 的 128 个输入中有 40 个干预效果接近零，报告逐输入相对误差时排除了效果平方和 ≤1e-24 的输入；总体主分数仍使用全部输入。没有将无效干预当作额外独立证据。

背景状态包含卷积／池化边界和头部值。L1 保留 k=32 时仅由 640 降到 608 个背景标量，减少 5%；L2 从 88 降到 56，减少约 36.36%。另外两者都有 q=8 个源状态，不在上述背景数内。“头部减少一半”并不是总状态减少一半。

4 个验证输入上的同后端计时，从图像开始、Q=256、k=16：

| 层 | 背景选择 ms | 融合 PCA ms | 局部精确 ms |
|---|---:|---:|---:|
{chr(10).join(pilot_timing_rows)}

这些是单教师成本诊断，且 k=16 没有达到精度门槛，不能将相应速度写成“满足误差要求的加速”。**k=32 的成本和同预算 PCA 比较尚未执行**，需在后续测试前补齐。新教师的分类准确率也不能代替上述干预误差。

详见 [训练前协议](../runs/fashion_pilot_20260910/pilot_protocol.json)、[训练记录](../runs/fashion_pilot_20260910/training_history.json)、[完整验证指标](../runs/fashion_pilot_20260910/validation_results.csv)、[精确性核查](../runs/fashion_pilot_20260910/verification.json) 和 [新任务计时](../runs/fashion_pilot_20260910/timing/timing_summary.csv)。

## 5. 具体往下推进

1. **补齐达到误差门槛时的比较。** 在仍然保留官方测试集的前提下，补测 k=32 及相同预算 PCA 的误差、完整成本和离线摊销。保持 k=16 为原主终点，把更大预算或验证选择预算明确列为次要分析。
2. **完成原六种子实验。** 保留首个教师，补齐其余五个冻结种子。训练、选点、拟合和验证选择全部完成并保存后，再一次性执行六个教师的官方测试评价。沿用主层 L1、强度 1、k=16 的“选择法减权重范数法”种子级 NRMSE 差，按原约定做精确符号翻转检验及种子级区间；其他比较保持描述性，披露六种子的统计能力限制。
3. **用完整成本决定研究主张。** 在满足误差要求的预算下若速度优势消失，应将结果定位为背景依赖与压缩边界分析。若精度和成本均成立，再加入 GAP＋线性头结构控制，继而检查残差网络内部位置。当前结构只覆盖带非线性全连接头的小 CNN，尚无跨结构机制迁移证据。

目前不应宣称已经发现语义机制、获得分布无关可靠性保证、验证了 ResNet，或已达到某个期刊录用水平。PCA 融合是必要的公平基线修正；本轮最有价值的新证据，是固定预算在新任务上的误差上升及其成本含义。

## 6. 文件与复现

原包放在 `incoming/`，解压原件放在 `upstream/`；新脚本、运行记录和报告单独保存。本轮没有修改原包，也没有安装它包含大量无关 Linux 依赖的完整 freeze。

本机使用已存在的 Python 环境，仅调用 NumPy、PyTorch、threadpoolctl；绘图另用 Matplotlib。版本及执行路径见 [运行说明](../RUNBOOK.md)。脚本对已存在的运行目录主动报错，避免覆盖；复跑时需使用新的输出目录或显式调整脚本中的运行标识。

NRMSE 定义为 sqrt(所有被评分 logit 变化误差的平方和 / 对应真实变化的平方和)。这些组合共享输入、模型和坐标，不能把干预输出数量当作独立统计样本数。当前外部独立测试执行次数为 **0**。
"""
    (BASE / "reports/复核与首轮外部实验_2026-09-10.md").write_text(report,encoding="utf-8")
    state=dict(date="2026-09-10",upstream_archive_sha256=imported["archive_sha256"],
               imported_manifest_files=271,matched_manifest_files=271,
               replayed_teachers=12,replayed_model_layer_runs=24,
               replay_full_tail_outputs=1572864,pca_fusion_verified=True,
               external_teacher_training_runs=1,external_teacher_seeds=[202609071],
               external_intervention_stage="train/validation pilot",external_official_test_runs=0,
               calibration_model_access=False,official_test_model_access=False,
               original_primary_contrast_unchanged=True,
               next_steps=["budget 32 matched-cost controls on validation","remaining five locked teachers","frozen six-seed official test"],
               latest_report="reports/复核与首轮外部实验_2026-09-10.md")
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (BASE / "RESEARCH_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

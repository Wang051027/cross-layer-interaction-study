"""汇总跨层恢复和移植实验。误差线为六个模型的标准差，匹配随机对照先在模型内取平均。"""
import os
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/cross_layer_20260928'
REPORT=BASE/'reports/跨层交互_首轮结果_2026-09-28.md'
FIGURE=BASE/'reports/cross_layer_mechanism_20260928.png'
os.environ['MPLCONFIGDIR']=str(RUN/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS=list(range(202609071,202609077))
METHODS=('interaction','single_effect','activation','matched')
NAMES={'interaction':'交互候选排序','single_effect':'单独效应排序','activation':'激活排序','matched':'匹配随机组（四组均值）'}
ENGLISH={'interaction':'Interaction selector','single_effect':'Single-effect selector','activation':'Activation selector','matched':'Matched random'}
COLORS=('#1e638f','#b87428','#7e8490','#779845')


def read(path):
    """读取已保存记录，报告不另行定义实验数值。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def values(summary,method,budget,alpha,metric):
    """取六个教师的同口径指标；匹配随机重复先在教师内平均。"""
    rows=[r for r in summary['methods'] if r['alpha']==alpha and r['budget']==budget
          and (r['method']==method or method=='matched' and r['method'].startswith('matched_'))]
    return np.array([np.mean([r[metric] for r in rows if r['seed']==seed]) for seed in SEEDS])


def fmt(array,percent=True):
    """统一显示教师均值±样本标准差，百分比只用于已定义的比率。"""
    a=np.asarray(array)*(100 if percent else 1)
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'+('%' if percent else '')


def table(header,rows):
    """生成简单 Markdown 表格，避免把未知或无效量写成零。"""
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def make_figure(summary):
    """绘制固定四个面板；原始层激活尺度不用于跨层比较。"""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.2),layout='constrained')
    ax=axes[0,0]
    for alpha,marker,color in [(1.,'o',COLORS[0]),(.5,'s',COLORS[1])]:
        a=[100*next(r['interaction_to_effect'] for r in summary['baseline'] if r['seed']==s and r['alpha']==alpha) for s in SEEDS]
        ax.plot(range(1,7),a,marker=marker,color=color,label=f'Attenuation {alpha:g}')
    ax.set(xlabel='Teacher index',ylabel='Interaction RMS / response RMS (%)',title='A  Conditional pair interaction is measurable',xticks=range(1,7))
    ax.legend(frameon=False)
    ax=axes[0,1]
    stages=('b2_mid','b2_out','b3_mid','b3_out')
    means=[];sds=[]
    for i,stage in enumerate(stages):
        a=np.array([r['component_to_total'] for r in summary['generation'] if r['stage']==stage and r['alpha']==1.])
        means.append(a.mean());sds.append(a.std(ddof=1))
        ax.scatter(np.full(6,i)+np.linspace(-.1,.1,6),a,s=16,color=COLORS[0],alpha=.6)
    ax.errorbar(range(4),means,yerr=sds,fmt='D',color='#173b58',capsize=4)
    ax.set(xticks=range(4),xticklabels=['B2 first\nReLU','B2 output\nReLU','B3 first\nReLU','B3 output\nReLU'],
           ylabel='Component RMS / total interaction RMS',title='B  Baseline-gate algebraic components')
    ax.text(.01,.02,'Components can cancel; these are not causal shares.',transform=ax.transAxes,fontsize=8,color='#575c63')
    for ax,metric,title in [(axes[1,0],'restore_ratio','C  Interaction remaining after restoration'),
                            (axes[1,1],'transplant_error_ratio','D  Interaction reconstruction error after transplant')]:
        for method,color in zip(METHODS,COLORS):
            a=np.stack([values(summary,method,k,1.,metric) for k in (8,16,32)])*100
            ax.errorbar([8,16,32],a.mean(1),yerr=a.std(1,ddof=1),marker='o',markersize=4,
                        color=color,label=ENGLISH[method],capsize=3,linewidth=1.4)
        ax.axhline(100,color='#bfc3c7',linewidth=.8,linestyle='--')
        ax.set(xlabel='Number of mediator coordinates',ylabel='Relative RMS (%) — lower is better',
               title=title,xticks=[8,16,32],ylim=(65,108))
        ax.grid(axis='y',alpha=.15)
    axes[1,0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Cross-layer interaction: detectable, not captured by a small fixed mediator set',fontsize=14,fontweight='bold')
    fig.savefig(FIGURE,dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    """保存完整报告、阅读入口与新增分析源码；原历史文档保持原字节。"""
    summary=read(RUN/'summary.json');audit=read(RUN/'independent_verification.json')
    assert summary['status']==audit['status']=='PASS'
    make_figure(summary)
    primary=[]
    for method in METHODS:
        primary.append([NAMES[method],fmt(values(summary,method,16,1.,'restore_ratio')),
                        fmt(values(summary,method,16,1.,'transplant_error_ratio'))])
    primary_table=table(['方法，k=16、α=1','恢复后剩余交互 / 原交互','移植交互误差 / 原交互'],primary)
    by_seed=[]
    for i,seed in enumerate(SEEDS):
        by_seed.append([seed]+[f"{100*values(summary,m,16,1.,f)[i]:.2f}%"
            for f in ('restore_ratio','transplant_error_ratio') for m in ('interaction','single_effect','matched')])
    seed_table=table(['教师','恢复：候选','恢复：单独效应','恢复：匹配组','移植误差：候选','移植误差：单独效应','移植误差：匹配组'],by_seed)
    dose_table=table(['α','k','候选恢复剩余','单独效应恢复剩余','候选移植误差','单独效应移植误差'],[
        [alpha,k]+[f'{100*values(summary,m,k,alpha,f).mean():.2f}%'
            for f in ('restore_ratio','transplant_error_ratio') for m in ('interaction','single_effect')]
        for alpha in (1.,.5) for k in (8,16,32)])
    generation_table=table(['阶段','投影到输出的分量 RMS / 总交互 RMS（均值±SD）'],[
        [name,fmt([r['component_to_total'] for r in summary['generation'] if r['stage']==stage and r['alpha']==1.],False)]
        for stage,name in zip(('b2_mid','b2_out','b3_mid','b3_out'),('block2 第一次 ReLU','block2 输出 ReLU','block3 第一次 ReLU','block3 输出 ReLU'))])
    base1=[r for r in summary['baseline'] if r['alpha']==1.]
    basehalf=[r for r in summary['baseline'] if r['alpha']==.5]
    details=[];checkmax=0.
    for seed in SEEDS:
        record=read(RUN/'development'/f'seed_{seed}'/'selections.json')
        selected=next(r['nodes'] for r in record['selections'] if r['method']=='interaction' and r['budget']==16)
        single=next(r['nodes'] for r in record['selections'] if r['method']=='single_effect' and r['budget']==16)
        matches=[r for r in record['selections'] if r['method'].startswith('matched_') and r['budget']==16]
        details.append([seed,record['pool_size'],len(set(selected)&set(single)),
            f"{np.mean([r['mean_match_distance'] for r in matches]):.3f}",
            f"{max(r['max_match_distance'] for r in matches):.3f}"])
        checkmax=max(checkmax,max(read(RUN/'development'/f'seed_{seed}'/'numerical_checks.json')['errors'].values()))
    diagnostic_table=table(['教师','结构可达中介数','候选与单独排序重合数/16','匹配平均距离','最大距离'],details)
    baseline_table=table(['强度','输出交互 RMS 范围（logit）','交互 RMS / 干预效应 RMS（六教师均值±SD）'],[
        [alpha,f"{min(r['interaction_rms'] for r in rows):.5f}–{max(r['interaction_rms'] for r in rows):.5f}",
         fmt([r['interaction_to_effect'] for r in rows])] for alpha,rows in [(1.,base1),(.5,basehalf)]])
    wins={metric: int(np.sum(values(summary,'interaction',16,1.,metric)<values(summary,'single_effect',16,1.,metric)))
          for metric in ('restore_ratio','transplant_error_ratio')}
    report=f'''# 跨层交互：第一轮机制验证结果

2026-09-28｜六个已有冻结残差教师｜同数据集后续探索｜没有训练新模型

**已经观察到可测量的跨层组合效应，但当前训练集选出的少量固定中介，尚不足以构成紧凑、稳定的传递机制解释。** 主要预算 16 个坐标时，恢复候选中介后仍剩约 87.73% 的原交互 RMS；仅移植这些中介，重建原交互的相对 RMS 误差仍约 89.55%。恢复和移植都不能按“发现了约 90% 的机制”解释；这里两个比率越小才越好。

候选交互排序在完整强度下优于单独效应排序的教师数，恢复指标为 {wins['restore_ratio']}/6、移植指标为 {wins['transplant_error_ratio']}/6。半强度时，k=16 的平均恢复指标反而略差于单独效应排序。这一限制保留在主报告中，不选择性改预算、删教师或改目标。

## 本轮具体推进了什么

与此前只研究最后残差块内部不同，本轮固定从 **block1 完整输出** 施加四个空间坐标的衰减，追踪 block2 和 block3 的四个 ReLU 阶段。它研究同一次干预如何跨层传播，不把不同源层的不同坐标直接比较成“深度效应”。所有受影响的主分支与旁路自然变化。

每图像枚举 16 个组合；6 对源坐标各有 4 个其余坐标背景，共 24 个条件成对差分。主要目标为真实类别 logit 减其他九类平均；完整十维输出并行保存。训练上下文前 64 张用于中介排序，前 512 张用于源坐标选择，4 张既有验证输入用于实现核验。全部六教师的选择统一冻结后，评价固定官方测试索引 1024–1087 的 64 张图像和两种强度。

这一批之前未用于本项目的组合干预，但测试集已经用于历史分类评价；它不是新外部数据集。64 张图在所有教师、方法间复用；24 个差分也不是独立样本。5,000 张校准图仍未进入模型。

共保存 12,288 个自然干预输出向量和 516,096 个恢复/移植输出向量。中介候选池是 block2 输出中结构可达的空间标量，含 576–640 个坐标；原完整层有 6,272 个坐标。本轮没有做完整通道或视觉语义评价。

![跨层交互及恢复移植结果](cross_layer_mechanism_20260928.png)

图：A 的分母是真实源干预响应，C/D 的分母是原交互；它们不是同一种百分比。B 是固定基线门控下的代数分量，不能相加成因果百分比。误差线表示教师标准差，匹配随机组先在每个教师内平均。

## 数学逻辑如何落到实验

对源坐标 i,j 及其余坐标的背景 T，逐层计算 I_l=h_l(Tij)−h_l(Ti)−h_l(Tj)+h_l(T)。源干预本身是独立缩放，因此源层 I 为零。仿射变换满足 I_z=W I_h，残差旁路的对应项同时加入。

每张图的原 ReLU 开关记为 D0，定义每个组合的余项 r=ReLU(z)−D0z；其四项差分 N 满足 I_out=D0 I_z+N。于是可以区分固定门控下的传播和本阶段开关变化造成的修正。递推到输出时，四个阶段的贡献之和准确重建最终交互。

这是一项由网络结构推导的恒等式，不是新定理，也不是独立于基线选择的唯一机制分配。真实机制证据还需要恢复/移植和跨输入检验。

## 结果一：交互可测量，不能把模型整体视作线性

{baseline_table}

这些绝对效应远大于本轮约 10⁻¹⁴ 量级的数值差异，说明不是舍入噪声。但仍需额外检验其对具体分类行为或视觉概念的作用，不能把非零直接等同于具有语义重要性。这个“交互/效应”比率也不能和上一轮的 1.20% 二阶预测误差直接比较。

## 结果二：多个阶段共同改变交互，第一阶段的输出投影较大

{generation_table}

这里把各阶段余项通过后续固定基线门控投影到同一个输出尺度。不同分量可以互相抵消或同向叠加；表中 RMS 比率不可相加，不是“哪一层解释了多少百分比”。它支持跨层计算分解的可行性，尚未唯一识别语义路径。

## 结果三：固定的小中介集合只捕获有限作用

中介位于 block2 完整输出。**恢复**：让源干预照常发生，但将候选 M 还原为本图无干预值；**移植**：让同层其余状态保持无干预值，仅给 M 放入源干预后的值。每种操作在全部 16 个组合上执行，再重算交互。

{primary_table}

例如候选恢复比率 87.73% 表示剩余交互 RMS 仍接近原来，不能据此说已阻断大部分传递。移植误差 89.55% 同样表明这 16 个状态不足以在基线背景中重现原交互。候选的平均指标有改善，但相对强基线并不逐教师稳定，也没有新的确认性显著性结论。

{seed_table}

## 固定补充：预算与强度

{dose_table}

预算增加到 32 个时，候选恢复仍剩约 79.54%，移植误差仍约 83.06%。这些预算均在评价前固定，不能事后把最有利一格改称原主检验。半强度下与单独效应排序的差距更弱，不能主张普遍优势。

## 候选与对照的公平性边界

候选按开发数据平均 |原目标梯度×中介条件交互| 排序。它是局部梯度辅助的提名方法，不能保证选出对有限恢复/移植最优的组。单独效应基线采用同样的目标梯度，对齐这一辅助信息。

{diagnostic_table}

匹配距离来自开发数据激活大小与单独效应两个特征的标准化 log1p 空间。部分最大距离较大，因此“匹配随机组”只是近似对照；不能声称完全排除了主效应、激活或其他因素。候选与单独效应组也存在重叠。本轮选择固定于教师、跨输入复用，尚未测试按输入条件变化的中介。

## 数值核验与可复核记录

- 四个已知答案控制通过：加性零交互、阈值两两交互、背景相关三阶作用、隐藏交互在读出处抵消。
- 六教师各四张验证图上，原模型钩子、逐层展开、恢复/移植、空/全中介、零强度和自动微分投影检查通过；最大差 {checkmax:.3e}。
- 评价期间逐例检查仿射、ReLU 与输出分解；保存数组独立回读的输出分解最大差 {summary['independent_output_decomposition_error']:.3e}。
- 独立核验器使用 24×16 稠密差分矩阵和目标向量内积，回算 16,128 条逐输入记录与 252 条方法汇总，共 {audit['scalar_comparisons']:,} 项数值比较。原始量及按已核验分子/分母直接计算的比率最大差 {audit['maximum_primitive_or_direct_division_difference']:.3e}。
- 有 {audit['amplified_ratio_comparisons']} 次独立比率比较因较小分母放大了舍入差，最大为 {audit['maximum_independent_ratio_difference']:.3e}。核验器对原始量和保存的除法仍按 10⁻⁹ 检查，并记录独立运算次序的明确舍入界；没有改动研究的分母下限、指标或任何实验结果。极端逐输入比率应与绝对量一起阅读。
- 原研究交付清单中的 136 个文件保持原哈希，原模型和已交付注释包未修改。

## 这如何改变后续问题

当前可以支持“存在可测量的组合效应，且能在指定基线下准确分解其跨层传播”；不能支持“已经找到少量、跨输入稳定的充分中介”，更不能支持唯一语义回路。

下一步应先区分两个解释：**固定中介集合不适配不同输入**，还是**组合效应本身分散，当前小预算/排序方式捕捉不足**。还存在梯度提名不适合有限干预的可能，不能把本轮失败直接归因于网络必然分布式。

合理的下一轮是在开发数据上比较固定集合、输入条件选择与有限干预重评分，固定规则后另用新批次评价；保持预算、源坐标、目标和修补操作一致。只有条件选择确实外推，才进一步构造较小的机制模型。上述新方法本轮尚未实现或运行，也未提前消耗校准集。

跨层连接与干预已有 [GCC](https://arxiv.org/html/2508.01728v1)、[CLA](https://arxiv.org/html/2404.14349v1) 等先例；[activation patching](https://arxiv.org/abs/2309.16042) 已讨论方法选择对结论的影响。真正的[高层因果抽象](https://www.jmlr.org/papers/v26/23-0058.html)还需要明确变量与干预映射。本轮的价值是形成可被否定的机制检验，而不是将这些既有工具重新命名为创新。

## 文件入口

- [冻结实验协议](跨层交互_实验协议_2026-09-28.md)
- [运行与数据入口](../runs/cross_layer_20260928/README.md)
- [原始响应与修补输出](../runs/cross_layer_20260928/evaluation/)
- [全部结果汇总](../runs/cross_layer_20260928/summary.json)
- [逐输入指标](../runs/cross_layer_20260928/per_input.csv)
- [独立核验记录](../runs/cross_layer_20260928/independent_verification.json)
- [中文注释核心实现](../scripts/cross_layer_mechanism.py)
- [实验阶段入口](../scripts/run_cross_layer_mechanism.py)
'''
    REPORT.write_text(report,encoding='utf-8')
    entry='''# 跨层机制第一轮：运行与阅读入口

本轮已完成。先读 [中文结果报告](../../reports/跨层交互_首轮结果_2026-09-28.md)，再看 [协议](../../reports/跨层交互_实验协议_2026-09-28.md)。旧研究首页属于已冻结文件，因此本轮以此独立入口记录进度。

代码顺序：`../../scripts/cross_layer_mechanism.py`（公式、逐层追踪和干预）→ `test_cross_layer_mechanism.py`（已知答案和原网络核验）→ `run_cross_layer_mechanism.py`（阶段和数据）→ `verify_cross_layer_mechanism.py`（独立回读）→ `report_cross_layer_mechanism.py`（报告）。新代码附中文说明，数值代码快照位于 `source/scripts/`。

`development/` 保存训练数据决定的排序、候选集合和验证核验；`freeze.json` 为评价前承诺；`evaluation/` 保存全部原始和修补 logits。每份 responses.npz 中 natural 为 [64图像,16组合,10类别]；patched 为 [64图像,21方案,2操作,16组合,10类别]，操作轴 0=恢复、1=移植；contributions 为 [64图像,4阶段,24条件成对差分,10类别]。24 个差分的 i/j/T 和四项索引保存在同一文件。

21 个方案按 budgets=8/16/32 排列，各预算依次为 interaction、single_effect、activation、matched_0..3；精确坐标以该教师 development/selections.json 为准。stage_statistics 最后一维依次为交互 RMS、固定门控项 RMS、余项 RMS、发生开关变化的坐标数、该阶段总坐标数；不同层原始尺度不直接比较。

已完成目录不可通过 init/develop/evaluate 重建。原电脑可运行以下只读检查；独立核验器会刷新自身核验记录，不改变实验数组：

```powershell
$researchPython = 'python'
& $researchPython -B -X utf8 'scripts/run_cross_layer_mechanism.py' --stage verify
```

本目录不是可脱离原模型与数据独立重训的发行包。原数据与检查点的哈希保存在 protocol.json。要开展新方法，应新建明确版本与输入划分，保留本轮结果。
'''
    (RUN/'README.md').write_text(entry,encoding='utf-8')
    # 分析/报告脚本在结果完成后添加，单列保存，避免冒称为评价前冻结代码。
    for filename in ('verify_cross_layer_mechanism.py','report_cross_layer_mechanism.py'):
        dest=RUN/'post_analysis_source'/filename;dest.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE/'scripts'/filename,dest)
    paths=[p for p in RUN.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name!='delivery_manifest.json']
    paths.extend([REPORT,FIGURE,BASE/'reports/跨层交互_实验协议_2026-09-28.md'])
    paths.extend(BASE/'scripts'/f for f in ('cross_layer_mechanism.py','test_cross_layer_mechanism.py',
        'run_cross_layer_mechanism.py','verify_cross_layer_mechanism.py','report_cross_layer_mechanism.py'))
    manifest=dict(scope='completed cross-layer follow-up; original numeric sources frozen before evaluation; post-analysis sources named separately',
        files=[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))])
    (RUN/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',report=str(REPORT),figure=str(FIGURE),files=len(manifest['files'])),ensure_ascii=False))


if __name__=='__main__':main()

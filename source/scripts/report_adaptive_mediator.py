"""生成第二轮中文报告、图表和交付索引；只读取已冻结、已独立核验的结果。

该文件属于结果完成后的呈现代码，不参与候选选择或模型计算。
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/adaptive_mediator_20260928'
REPORT=BASE/'reports/自适应中介_第二轮结果_2026-09-28.md'
FIGURE=BASE/'reports/adaptive_mediator_20260928.png'
os.environ['MPLCONFIGDIR']=str(RUN/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'fixed_gradient':'固定集合＋梯度评分','input_gradient':'逐图集合＋梯度评分',
       'fixed_finite':'固定集合＋实际干预评分','input_finite':'逐图集合＋实际干预评分',
       'fixed_single':'固定单独效应排序','input_single':'逐图单独效应排序',
       'input_gradient_full':'逐图梯度，整个可达候选池','legacy_fixed':'上一轮原固定名单',
       'random':'四个固定随机组的教师内均值','all_shortlist':'整个64候选名单',
       'all_reachable':'整个结构可达池（576–640）'}
PRIMARY=('fixed_gradient','input_gradient','fixed_finite','input_finite')
METRICS=('restore_ratio','transplant_error_ratio','joint_loss')


def read(path):
    """读取中介选择和修补指标。"""
    return json.loads(path.read_text(encoding='utf-8'))


def values(summary,method,metric,alpha=1.,budget=16):
    """先按教师取一个值；随机4组先在教师内平均，避免伪重复。"""
    result=[]
    for seed in range(202609071,202609077):
        rows=[r for r in summary['methods'] if r['seed']==seed and r['alpha']==alpha
              and (r['method']==method or (method=='random' and r['method'].startswith('random_')))
              and (r['budget']==budget or method.startswith('all_'))]
        assert len(rows)==(4 if method=='random' else 1),(method,seed,len(rows))
        result.append(np.mean([r[metric] for r in rows]))
    return np.array(result)


def formatted(array,percent=True):
    """均值±六教师标准差；不写成置信区间或确认性统计。"""
    scale=100 if percent else 1
    return f'{np.mean(array)*scale:.2f} ± {np.std(array,ddof=1)*scale:.2f}'+('%' if percent else '')


def markdown_table(headers,rows):
    """统一生成简单表格，使指标单位和比较对象容易审阅。"""
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                     +['| '+' | '.join(map(str,row))+' |' for row in rows])


def method_table(summary,methods,alpha=1.,budget=16):
    """固定同一输入、强度和预算的比较；较大预算参照另行标明。"""
    return markdown_table(['方法','恢复后剩余交互','移植重建误差','联合损失'],
        [[NAMES[m],formatted(values(summary,m,METRICS[0],alpha,budget)),
          formatted(values(summary,m,METRICS[1],alpha,budget)),
          formatted(values(summary,m,METRICS[2],alpha,budget),False)] for m in methods])


def make_figure(summary):
    """并列呈现两个主指标和预算曲线，保留教师点与标准差。"""
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.8),layout='constrained')
    colors=['#6b7280','#277da1','#d68929','#2d936c']
    labels=['Fixed\ngradient','Input\ngradient','Fixed\nfinite','Input\nfinite']
    for ax,metric,title in zip(axes[:2],METRICS[:2],
        ['A  Interaction remaining after restoration','B  Interaction reconstruction error']):
        for i,method in enumerate(PRIMARY):
            v=values(summary,method,metric)*100
            ax.scatter(i+np.linspace(-.10,.10,6),v,color=colors[i],s=22,alpha=.6,zorder=3)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='D',color=colors[i],capsize=4,ms=6)
        ax.set_xticks(range(4),labels);ax.set_ylim(0,110)
        ax.axhline(100,color='#aeb5bf',linestyle='--',lw=1)
        ax.set_ylabel('Relative RMS (%) — lower is better');ax.set_title(title,fontsize=11)
    for method,color,label in zip(PRIMARY,colors,['Fixed gradient','Input gradient','Fixed finite','Input finite']):
        a=np.array([values(summary,method,'joint_loss',budget=k) for k in (8,16,32)])
        axes[2].errorbar([8,16,32],a.mean(1),yerr=a.std(1,ddof=1),fmt='o-',color=color,capsize=3,label=label)
    axes[2].set_xticks([8,16,32]);axes[2].set_ylim(0,1.1)
    axes[2].set_xlabel('Number of mediator coordinates')
    axes[2].set_ylabel('Mean of two squared RMS ratios')
    axes[2].set_title('C  Joint loss across budgets',fontsize=11)
    axes[2].legend(fontsize=8,loc='lower left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Select at half strength, evaluate at full strength | 6 teachers, 64 images\n'
                 'A/B: 16 mediators from the same 64 candidates; points: teachers; bars: SD',fontsize=12)
    fig.savefig(FIGURE,dpi=170);plt.close(fig)


def main():
    """生成可追溯报告和清单；原数值结果与历史文件保持原样。"""
    summary=read(RUN/'summary.json');audit=read(RUN/'independent_verification.json')
    assert summary['status']==audit['status']=='PASS'
    make_figure(summary)
    primary_table=method_table(summary,PRIMARY)
    control_table=method_table(summary,['fixed_single','input_single','random','legacy_fixed','input_gradient_full'])
    capacity_table=method_table(summary,['all_shortlist','all_reachable'])
    contrasts=[]
    for a,b in [('input_gradient','fixed_gradient'),('input_finite','fixed_finite'),
                ('fixed_finite','fixed_gradient'),('input_finite','input_gradient')]:
        va=[values(summary,a,k) for k in METRICS];vb=[values(summary,b,k) for k in METRICS]
        contrasts.append([f'{NAMES[a]} 对 {NAMES[b]}',
            f'{100*np.mean(va[0]-vb[0]):+.2f} 百分点',f'{100*np.mean(va[1]-vb[1]):+.2f} 百分点',
            f'{int(np.sum(va[0]<vb[0]))}/6、{int(np.sum(va[1]<vb[1]))}/6、{int(np.sum(va[2]<vb[2]))}/6'])
    contrast_table=markdown_table(['预定比较','恢复比率差','移植误差差','恢复／移植／联合损失胜出教师数'],contrasts)
    dose_table=markdown_table(['α','k','固定梯度：恢复／移植','逐图梯度：恢复／移植',
                               '固定实际评分：恢复／移植','逐图实际评分：恢复／移植'],
        [[alpha,k,*[f'{100*values(summary,m,METRICS[0],alpha,k).mean():.2f}% / '
                    f'{100*values(summary,m,METRICS[1],alpha,k).mean():.2f}%' for m in PRIMARY]]
         for alpha in (1.,.5) for k in (8,16,32)])
    seed_table=markdown_table(['教师','固定梯度','逐图梯度','固定实际评分','逐图实际评分'],
        [[seed,*[f'{100*values(summary,m,METRICS[0])[i]:.2f}% / '
                 f'{100*values(summary,m,METRICS[1])[i]:.2f}%' for m in PRIMARY]]
         for i,seed in enumerate(range(202609071,202609077))])
    overlap_table=markdown_table(['动态方法','与对应固定集合Jaccard','不同输入集合间Jaccard'],
        [[NAMES[m],*[formatted(np.array([r[key] for r in summary['overlaps'] if r['method']==m]),False)
                     for key in ('mean_jaccard_to_fixed','mean_jaccard_between_inputs')]]
         for m in ('input_gradient','input_finite')])
    numeric=max(max(read(RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'/'checks.json')['errors'].values())
                for seed in range(202609071,202609077) for alpha in (1.,.5))
    report=build_report(primary_table,contrast_table,control_table,capacity_table,dose_table,
                        seed_table,overlap_table,numeric,audit)
    REPORT.write_text(report,encoding='utf-8')
    write_delivery()
    print(json.dumps(dict(status='PASS',report=str(REPORT),figure=str(FIGURE)),ensure_ascii=False))


def build_report(primary_table,contrast_table,control_table,capacity_table,dose_table,
                 seed_table,overlap_table,numeric,audit):
    """明确区分已有证据、当前发现和仍待验证的解释。"""
    return f'''# 自适应中介：第二轮机制验证与旧研究的用途

2026-09-28｜六个已有冻结教师｜同数据集后续探索｜未训练新模型

**旧结果与代码仍有用；新结果支持继续研究逐输入的中介选择，但尚未找到少量、稳定且充分的跨层机制。**

本轮在相同64候选范围、同样16个中介预算下，逐图实际干预评分得到的完整强度恢复剩余为75.72%、移植误差为77.90%；固定梯度分别为89.32%、91.22%。改善存在，两个残差仍然较大。这些是6教师的比率均值，不是“解释了多少百分比的因果机制”。

## 之前的结果和代码具体还能做什么

| 已有资产 | 在当前研究中的用途 | 结论的适用范围 |
|---|---|---|
| 冻结CNN、检查点和数据划分 | 直接复用，保持模型不变后比较不同机制假设 | 6个同架构模型不等于跨模型家族验证 |
| 尾部局部响应与组合枚举 | 作为局部加性/低阶参照；保留单独、成对和高阶失效案例 | 中心3.56%降到1.20%是原局部预测误差，不是本轮的中介指标 |
| 稳定门控合并与精确修正代码 | 保留为特定尾部结构的精确计算工具、速度基线和数学依据 | 原公式只验证了限定的下游结构；多层ReLU需要重新推导/验证，不能直接套用 |
| 第一轮逐层追踪、恢复/移植和核验 | 本轮直接通过import复用；少量新增代码负责选点 | 代数恒等式验证计算，不等于唯一因果解释 |
| 第一轮固定中介的弱结果 | 提供可被否定的研究问题；原名单在本轮新批次重评 | 不能因后续更好就删掉，也不能拿不同批次的均值直接算提升 |
| 原中文注释代码包 | 保留作原阶段的阅读和复现入口 | 它代表当时版本；本轮新增代码放在新的独立目录 |

因此，旧实验可以在论文中承担“限定场景下的基线证据、计算工具验证、提出新问题的消融结果”。最终正文是否全部纳入，取决于论文主张；证明局部近似有效的结果，不应改写成证明整个网络线性。原始记录和代码应保留，呈现时再取舍。

相关旧报告：[局部联合干预](联合干预响应_首轮结果_2026-09-25.md)、[精确修正数学说明](稳定状态合并与精确修正_数学说明_2026-09-17.md)、[跨层首轮结果](跨层交互_首轮结果_2026-09-28.md)。

## 本轮如何把原因拆开

沿用block1的四个源标量、16种组合、24个条件成对差分，以及block2完整输出的中介切口。训练上下文64张确定候选和固定选择；新批次为官方测试索引1088–1151的64张图，全部6教师和方法复用它们。历史全测试分类结果已知，故这是同数据集后续探索。

先在半强度α=0.5下选点，锁定全部教师、全部输入的名单，再读取完整强度α=1的评价结果。主比较使用相同的64候选名单，来自训练均值的交互/单独效应前32并集并补齐，主预算16。

两条轴分别是：固定集合或逐图集合；基线梯度评分或实际单坐标恢复/移植评分。实际评分为 `mean(I²) − [mean(R_j²)+mean((T_j−I)²)]/2`，即两个平方误差的单坐标改善；不保证多个坐标一起选后的全局最优。

**逐图选择会使用真实标签、该图半强度状态和修补结果，是逐图诊断搜索。** 它没有访问完整强度结果来选点，但也不是只看原图就给出机制的现成解释器。方法规则可复用于新图，不代表每张图的搜索不需要额外计算。

![固定与逐图中介选择](adaptive_mediator_20260928.png)

图A/B：相同候选和k=16，α=1；点为教师，误差线为教师标准差。图C：预先固定的预算。所有指标越小越好。联合损失是两种相对RMS的平方取平均。

## 主结果：逐图选择改善，单纯替换固定评分没有稳定改善

{primary_table}

{contrast_table}

逐图实际评分相对逐图梯度评分，在两项指标上均为6/6教师改善；固定实际评分相对固定梯度，仅3/6改善，均值还略差。这更支持继续考察“输入/目标条件下的选点”，而不是认为更精确的单点评分一定能产生更好的通用固定集合。

## 不能忽略的强对照：候选范围也限制了结果

以下均为k=16、α=1；整个可达池方法的可选范围更大，因此不属于主2×2等范围比较。

{control_table}

从全部576–640个结构可达点中逐图按梯度选16个，平均恢复剩余71.43%、移植误差72.81%，比在64名单里做逐图实际评分还低。说明共同64名单可能排除了对部分输入有用的点；不能据本轮64名单内的比较宣布实际评分全面优于梯度方法。

上一轮原固定名单在这批新图上的恢复/移植指标为86.42%/87.70%。本批次按同一组图像比较两种方法。上一批图像的对应均值为87.73%/89.55%。

## 较大集合参照不等于理论上界

{capacity_table}

整个结构可达池可把恢复交互和移植误差降到数值误差量级，验证了完整切口和支持集实现。64名单仍有较大残差，与名单范围受限相容。

但恢复/移植误差对集合大小并不保证单调：加入新坐标会引入抵消、干扰和门控变化。因此“整个64名单”的结果只是一个实际整组参照，不是所有64名单内子集的最优下界。本轮未穷举最优子集，也未证明16个或更多状态必然不足。

## 固定补充：预算、强度和逐教师结果

表内每格为“恢复剩余 / 移植误差”，六教师平均。

{dose_table}

半强度也是选择评分使用的强度，所以它的较好表现含有选择适配，不能当作额外独立泛化证据。完整强度的改善仍存在，但幅度弱于选择强度，不能忽略这个差距。

下面是主设置α=1、k=16的逐教师结果；格内仍为“恢复剩余 / 移植误差”。

{seed_table}

## 选择会变化，但还不能给出唯一机制解释

{overlap_table}

Jaccard=交集数/并集数。集合重合较低只描述排序与选择的变化，也可能受近似分数、并列点或不稳定排序影响。没有单凭低重合推断动态回路。

本轮还存在一个要明确控制的因素：真实标签不同，目标logit对比也会改变。逐图优于跨类别平均的固定集合，可能包含“不同类别使用不同状态”的作用，不能全部归因为同类别内部的图像变化或ReLU门控。所有小集合仍留下较大误差；未验证视觉语义、唯一回路、跨架构泛化或新的数学定理。

## 下一步的可检验问题

优先检验：**按类别固定的中介，能否解释逐图选择的改善？同一类别内是否仍需要逐图改变中介？** 在开发数据上为每类固定集合，并与全局固定、逐图选择比较，再冻结规则后用新输入评价。类别样本量需要足够，不能拿这64图事后拟合又自评。

同时应控制候选范围，优先比较整个结构可达池中的梯度提名与有限干预重评分；必要时加入不依赖真实标签的固定输出方向。这样才能区分目标变化、候选遗漏与有限非线性修补三个因素。本轮尚未实现这下一步，也不能据当前结果直接命名某个语义传递机制。

## 实现与证据核验

- 保留6个检查点，没有重新训练；5,000张校准图未进入模型。
- 完美、无效、有害中介评分和零交互控制通过；新单点修补与原网络钩子、旧梯度特征实现核对通过。
- 全部评价图的逐层恒等式、全可达池端点以及半强度重放检查通过，最大差{numeric:.3e}。
- 独立回读核验{audit['finite_scores_verified']:,}个有限评分、{audit['selected_sets_verified']:,}个选择集合、456条教师方法汇总和29,184条逐输入记录；共{audit['scalar_comparisons']:,}项数值比较。教师比率最大差{audit['max_absolute_differences']['teacher_ratios']:.3e}，逐输入绝对量最大差{audit['max_absolute_differences']['input_primitives']:.3e}。
- 比率分母小于等于10⁻⁹时置空；核验逐输入比率使用已独立核验的保存分子分母。
- 保存1,572,864个单坐标修补向量用于开发/逐图选择、933,888个成组修补评价向量。它们不是独立样本。实际评分需要额外修补查询，本轮不主张同计算预算下的速度优势。
- 原跨层交付90个文件和更早交付136个文件，合计226个文件保持原哈希。

[Activation patching方法研究](https://arxiv.org/abs/2309.16042)讨论指标与修补设计如何影响结论；[AtP*](https://arxiv.org/abs/2403.00745)研究梯度近似可能遗漏的效应。本轮采用这类方法问题检验自己的CNN，没有复现或重新命名这些方法为创新。

## 阅读入口

- [评价前冻结的实验协议](自适应中介_实验协议_2026-09-28.md)
- [运行、数据和代码说明](../runs/adaptive_mediator_20260928/README.md)
- [新增选择器代码](../scripts/adaptive_mediator.py)
- [阶段执行代码](../scripts/run_adaptive_mediator.py)
- [原始评价数组](../runs/adaptive_mediator_20260928/evaluation/)
- [全部结果汇总](../runs/adaptive_mediator_20260928/summary.json)
- [独立核验记录](../runs/adaptive_mediator_20260928/independent_verification.json)
'''


def write_delivery():
    """建立新入口、分析源码快照和哈希清单；不更新已冻结的旧首页。"""
    entry='''# 第二轮中介验证：代码与数据入口

先读[结果及旧研究用途](../../reports/自适应中介_第二轮结果_2026-09-28.md)与[冻结协议](../../reports/自适应中介_实验协议_2026-09-28.md)。

本轮直接复用上一轮 cross_layer_mechanism.py 和 run_cross_layer_mechanism.py 中的模型计算、数据与辅助函数。新增代码附中文说明，建议按下面顺序阅读：

1. [adaptive_mediator.py](../../scripts/adaptive_mediator.py)：两类评分、公共候选、固定/逐图选点、指标。
2. [test_adaptive_mediator.py](../../scripts/test_adaptive_mediator.py)：已知答案与原模型钩子核验。
3. [run_adaptive_mediator.py](../../scripts/run_adaptive_mediator.py)：各阶段、源文件冻结、选点冻结及数据存储。
4. [verify_adaptive_mediator.py](../../scripts/verify_adaptive_mediator.py)：独立矩阵差分与选择回读。
5. [report_adaptive_mediator.py](../../scripts/report_adaptive_mediator.py)：结果呈现，不参与选点。

`development/seed_*/discovery.npz` 保存训练特征、半强度自然输出、64个单点修补原始输出和评分。`input_selection/seed_*/selection.npz` 保存新输入同类记录以及全部逐图名单。两者中 singleton_logits 均为[64图,64候选,2操作,16组合,10类]；操作0为恢复、1为移植。features 为[64图,3分数,6272中介坐标]，分数依次为交互梯度、单独效应梯度、激活大小。

selections 为[64图,38方案,可达池大小]，按 specs.json 的 budget 取前k个坐标，其余填-1。方案顺序为每个预算8/16/32分别12种方法，之后为整个64名单与整个可达池。两种评价强度使用完全相同的选择。evaluation/seed_*/alpha_*/responses.npz 中 natural=[64,16,10]，patched=[64,38,2,16,10]。

fixed_freeze.json 锁定六教师的开发规则与固定排序，adaptive_freeze.json 锁定六教师全部新输入名单；后者先于 evaluation_opened.json。source/ 为评价前数值代码和协议快照，post_analysis_source/ 为完成后的独立核验与报告代码，二者不混称。

原电脑上的只读检查：

```powershell
& 'python' -B -X utf8 'scripts/run_adaptive_mediator.py' --stage verify
```

阶段顺序为 smoke → init → develop → freeze → select → seal → evaluate → summarize → verify。已完成目录拒绝覆盖；当前文件不是脱离原项目模型和数据就可独立重训的发行包。新实验应另设目录和协议，不能改本轮冻结代码后沿用原结果。
'''
    (RUN/'README.md').write_text(entry,encoding='utf-8')
    post=RUN/'post_analysis_source';post.mkdir(exist_ok=True)
    for name in ('verify_adaptive_mediator.py','report_adaptive_mediator.py'):
        shutil.copy2(BASE/'scripts'/name,post/name)
    paths=[p for p in RUN.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name!='delivery_manifest.json']
    paths += [REPORT,FIGURE,BASE/'reports/自适应中介_实验协议_2026-09-28.md']
    paths += [BASE/'scripts'/n for n in ('adaptive_mediator.py','test_adaptive_mediator.py',
        'run_adaptive_mediator.py','verify_adaptive_mediator.py','report_adaptive_mediator.py')]
    files=[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))]
    manifest=dict(scope='second-round adaptive mediator follow-up; pre-evaluation and post-analysis source snapshots distinguished',files=files)
    (RUN/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()

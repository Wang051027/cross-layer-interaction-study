"""Render the completed, independently verified descriptive analysis."""
import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/joint_response_20260925'
os.environ['MPLCONFIGDIR']=str(RUN/'mpl_cache')
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

REPORT=BASE/'reports/联合干预响应_首轮结果_2026-09-25.md'
FIGURE=BASE/'reports/figures/joint_response_20260925.png'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
# 把报表字段转成浮点数；空值保留为 NaN，避免当作真实零。
def number(r,key):return np.nan if r[key] in ('',None) else float(r[key])
def stats(g,key,precision=2):return f"{100*g[key+'_mean']:.{precision}f}% ± {100*g[key+'_sd']:.{precision}f}%"

# 把已完成的联合干预分析生成中文报告及四面板图。
def main():
    summary=read(RUN/'summary.json');verification=read(RUN/'independent_verification.json')
    assert verification['status']=='PASS'
    with (RUN/'per_input_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    groups={(g['position'],g['q'],g['alpha']):g for g in summary['groups']}
    primary=[r for r in rows if r['q']=='8' and r['alpha']=='1.0']
    teachers=[r for r in summary['teachers'] if r['q']==8 and r['alpha']==1.]
    positions=['center','upper_left'];names={'center':'中心位置','upper_left':'左上位置'}
    palettes=['#245985','#d66b35'];poscolors={'center':'#245985','upper_left':'#53956b'}
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    # Panel A: both models evaluated on the same 219 unseen coalition masks.
    ax=axes[0,0]
    for j,(method,label) in enumerate([('single','单独干预：9 次查询'),('pair','单独＋成对：37 次查询')]):
        x=np.arange(2)+(j-.5)*.28
        mean=[100*groups[p,8,1.][f'{method}_scalar_pooled_nrmse_mean'] for p in positions]
        sd=[100*groups[p,8,1.][f'{method}_scalar_pooled_nrmse_sd'] for p in positions]
        ax.bar(x,mean,yerr=sd,width=.26,capsize=4,color=palettes[j],alpha=.8,label=label)
        for i,p in enumerate(positions):
            values=[100*t[f'{method}_scalar_pooled_nrmse'] for t in teachers if t['position']==p]
            ax.scatter(x[i]+np.linspace(-.065,.065,6),values,s=20,color='#273544',zorder=3)
    ax.set_xticks(range(2),[names[p] for p in positions]);ax.set_ylabel('留出组合 NRMSE（%）')
    ax.set_title('A  少量查询能否预测其余组合？',loc='left',pad=14)
    ax.legend(frameon=False,fontsize=9);ax.set_ylim(bottom=0)
    # Panel B: different domain and access to every response, explicitly labelled.
    ax=axes[0,1]
    for j,(degree,label) in enumerate([(1,'最佳一阶'),(2,'最佳二阶')]):
        x=np.arange(2)+(j-.5)*.28
        mean=[100*groups[p,8,1.][f'scalar_oracle_degree{degree}_pooled_nrmse_mean'] for p in positions]
        sd=[100*groups[p,8,1.][f'scalar_oracle_degree{degree}_pooled_nrmse_sd'] for p in positions]
        ax.bar(x,mean,yerr=sd,width=.26,capsize=4,color=palettes[j],alpha=.8,label=label)
        for i,p in enumerate(positions):
            values=[100*t[f'scalar_oracle_degree{degree}_pooled_nrmse'] for t in teachers if t['position']==p]
            ax.scatter(x[i]+np.linspace(-.065,.065,6),values,s=20,color='#273544',zorder=3)
    ax.set_xticks(range(2),[names[p] for p in positions]);ax.set_ylabel('全组合诊断 NRMSE（%）')
    ax.set_title('B  使用全部 256 个响应的最佳拟合',loc='left',pad=14)
    ax.legend(frameon=False,fontsize=9);ax.set_ylim(bottom=0)
    ax.text(.03,.96,'事后诊断；不可作为留出预测成绩',transform=ax.transAxes,ha='left',va='top',fontsize=9,color='#71442c')
    # Panel C: all finite input-level ratios, including the adverse tail.
    ax=axes[1,0]
    for p in positions:
        errors=np.array([number(r,'pair_scalar_nrmse') for r in primary if r['position']==p])
        errors=np.sort(errors[np.isfinite(errors)])*100
        ax.step(errors,np.arange(1,len(errors)+1)/len(errors),where='post',color=poscolors[p],lw=2,
                label=f'{names[p]}，有效记录 {len(errors)}')
    ax.axvline(10,color='#bd623f',ls='--',lw=1.2,label='10% 诊断线')
    ax.set_xscale('symlog',linthresh=.5);ax.set_xlim(0,100);ax.set_ylim(0,1.03)
    ax.set_xticks([0,.5,1,5,10,50,100],['0','0.5','1','5','10','50','100'])
    ax.set_xlabel('每张图像的二阶外推 NRMSE（%）');ax.set_ylabel('有效记录的累计比例')
    ax.set_title('C  平均误差小，仍有个别输入失效',loc='left',pad=14)
    ax.legend(frameon=False,fontsize=9,loc='lower right')
    # Panel D: all primary input records; no selected teacher examples.
    ax=axes[1,1]
    for p in positions:
        rs=[r for r in primary if r['position']==p]
        ax.scatter([number(r,'candidate_states') for r in rs],[number(r,'pair_scalar_rmse') for r in rs],
                   s=11,alpha=.25,color=poscolors[p],edgecolor='none',label=names[p])
    ax.set_yscale('symlog',linthresh=1e-6);ax.set_ylabel('二阶外推 RMSE（logit 对比单位）')
    ax.set_xlabel('可能跨零的下游状态数');ax.set_title('D  状态跨零与误差：描述性关联',loc='left',pad=14)
    ax.legend(frameon=False,fontsize=9)
    fig.suptitle('联合干预首轮诊断：整体接近低阶，个别输入仍需识别',fontsize=17,x=.07,ha='left',y=.97)
    fig.text(.07,.93,'8 个源坐标 · 完全置零 · 6 个冻结教师 · 每个教师/位置复用同一组 128 张测试图像',fontsize=10,color='#4c5968')
    fig.text(.07,.038,'A/B：先按教师汇总，柱为均值，误差线为标准差，点为教师。A 与 B 的评分域及响应访问权限不同。',fontsize=9,color='#4c5968')
    fig.text(.07,.018,'C/D：记录在教师间复用图像，不视为独立样本。零效应记录不计相对误差；全部保留绝对误差。历史数据的描述性再分析。',fontsize=9,color='#4c5968')
    fig.subplots_adjust(left=.08,right=.97,top=.86,bottom=.13,wspace=.27,hspace=.45)
    FIGURE.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGURE,dpi=220,facecolor='white');plt.close(fig)

    main_table=[]
    for p in positions:
        g=groups[p,8,1.]
        main_table.append(f"| {names[p]} | {stats(g,'single_scalar_pooled_nrmse')} | {stats(g,'pair_scalar_pooled_nrmse')} | 6/6 |")
    oracle_table=[]
    for p in positions:
        g=groups[p,8,1.]
        oracle_table.append(f"| {names[p]} | {stats(g,'scalar_oracle_degree1_pooled_nrmse',3)} | {stats(g,'scalar_oracle_degree2_pooled_nrmse',3)} | {100*g['scalar_high_order_variance_fraction_mean']:.5f}% |")
    extra_table=[]
    for q in (4,8,12):
        for alpha in (1.,.5):
            gs=[groups[p,q,alpha] for p in positions]
            extra_table.append(f"| {q} | {alpha:g} | {'4（验证补充）' if q==12 else '128（既有测试）'} | {100*gs[0]['pair_scalar_pooled_nrmse_mean']:.3f}% / {100*gs[1]['pair_scalar_pooled_nrmse_mean']:.3f}% | {100*gs[0]['scalar_oracle_degree2_pooled_nrmse_mean']:.3f}% / {100*gs[1]['scalar_oracle_degree2_pooled_nrmse_mean']:.3f}% |")
    tail_table=[]
    for p in positions:
        rs=[r for r in primary if r['position']==p]
        valid=[r for r in rs if np.isfinite(number(r,'pair_scalar_nrmse'))]
        worst=max(valid,key=lambda r:number(r,'pair_scalar_nrmse'))
        tail_table.append(f"| {names[p]} | {len(valid)}/768 | {sum(number(r,'pair_scalar_nrmse')>.1 for r in valid)}/{len(valid)} | {100*number(worst,'pair_scalar_nrmse'):.2f}% | {number(worst,'pair_scalar_rmse'):.6f} / {number(worst,'pair_scalar_effect_rms'):.6f} |")
    corr_table=[]
    for p in positions:
        g=groups[p,8,1.]
        corr_table.append(f"| {names[p]} | {g['candidate_states_mean_mean']:.2f} | {g['candidate_vs_pair_rmse_rho_mean']:.3f} ± {g['candidate_vs_pair_rmse_rho_sd']:.3f} | {g['candidate_vs_highorder_rmse_rho_mean']:.3f} |")
    text=REPORT_TEMPLATE.replace('{{MAIN}}','\n'.join(main_table)).replace('{{ORACLE}}','\n'.join(oracle_table))
    text=text.replace('{{EXTRA}}','\n'.join(extra_table)).replace('{{TAIL}}','\n'.join(tail_table)).replace('{{CORR}}','\n'.join(corr_table))
    text=text.replace('{{REFERENCE_MAX}}',f"{summary['maximum_reconstruction_difference']:.3e}")
    text=text.replace('{{IDENTITY_MAX}}',f"{summary['maximum_nonlinear_identity_difference']:.3e}")
    text=text.replace('{{AUDIT_MAX}}',f"{verification['maximum_difference']:.3e}")
    assert '{{' not in text
    # 把报告中的本地相对链接解析为实际完整路径，网络链接保持原样。
    def absolute_link(match):
        prefix,target=match.group(1),match.group(2)
        if target.startswith(('https://','http://','D:/')): return match.group(0)
        return prefix+(REPORT.parent/target).resolve().as_posix()+')'
    text=re.sub(r'(!?\[[^\]]*\]\()([^\n)]+)\)',absolute_link,text)
    REPORT.write_text(text,encoding='utf-8')
    print(json.dumps({'report':str(REPORT),'figure':str(FIGURE),'characters':len(text)},ensure_ascii=False))

REPORT_TEMPLATE='''# 联合干预响应：从文献问题到第一轮可证伪检查

2026-09-25｜文献综述之后的首轮描述性再分析。沿用历史数据，未训练新教师、未调用新的网络前向。此文不替代此前冻结的加速实验报告。

## 这次得到了什么

**现有局部干预的响应整体接近低阶结构，成对信息明显改善组合预测；三阶及以上成分存在，但总体能量很小。** 主要的 8 单元、完全置零设置中，只用单独干预时的误差为中心 3.56%、左上 4.67%；加入成对干预后为 1.20%、1.79%。六个教师在两个位置上都改善。然而，少数图像的相对误差较高，不能得出“所有输入都用二阶就够”的结论。

因此，当前优先研究的问题应收窄为：**在什么输入、干预背景和查询方式下，简单组合模型会失效，能否预先识别这些情况？** 这批证据还不足以支持“神经元普遍存在强高阶协同”，也没有发现可命名的跨层语义回路。

![联合干预响应首轮诊断](figures/joint_response_20260925.png)

## 你的设想如何变成这个实验

你的原始问题关注多个神经元如何共同起作用，以及上游变化如何通过下游共同影响输出。文献已经研究过单元重要性、分组贡献、联合消融和回路发现；仅把几个神经元放在一起，不能直接构成创新。综述据此提出，先把“共同作用”写成能被观测否定的问题。

这里的第一步是固定一张图像和几个源激活，把每一种干预组合都看作一次可回答的提问。例如八个坐标中，只改变第 1 个、只改变第 2 个、同时改变第 1 和第 2 个，输出分别怎样变化？如果前两个变化直接相加就能预测第三个，这一部分是加性的；如果还需要一个成对修正，就是二阶关系。再看三个以上一起改变时，前面的规则还能否成立。

| 原始想法 | 这次的可计算对象 | 什么结果会使我们收缩主张 |
|---|---|---|
| 多个神经元可能有组合效应 | 完整二元干预响应函数，以及单独/成对查询产生的预测 | 简单模型已准确，就不能先假设必须使用复杂高阶方法 |
| 个体重要性不等于组合机制 | 留出三个及以上同时干预的组合，直接预测它们的输出变化 | 成对模型整体准确时，研究重点应转向其适用边界 |
| 上游组合改变下游状态 | 当前段中 ReLU 是否跨零，以及非线性余项的精确分解 | 恒等式只能验证计算来源，不能直接命名语义机制 |
| 加速技术能帮助研究机制 | 复用先前保存的完整响应表和内部状态，减少本轮计算成本 | 计算等价或运行更快本身不等于机制创新 |

这里的“单元”是**通道与空间位置共同确定的标量激活坐标**，不是整个特征通道。两个位置位于同一残差块，不是两个网络层。本轮是通往跨层问题的局部诊断。

## 数据、目标和公平比较

复用六个冻结残差 CNN 教师、中心与左上两个源位置，组合规模为 4、8、12，强度 α 为 1 和 0.5。α=1 表示把所选源坐标置零，α=0.5 表示减半。未被选择的源坐标与残差恒等旁路保持原值，下游自然传播。

4、8 单元各使用官方测试索引 896–1023 的 128 张已保存图像；12 单元使用每个教师/位置原有的四张验证输入补充记录。4/8 的图像在教师间、位置间复用；12 的图像来源和数量不同。总计 36 个数据设置、72 个强度设置、6,240 条图像—教师—位置—规模—强度记录，复用 1,228,800 个十维输出向量。这些不是同等数量的独立样本。

对每张图像，固定标量目标为“真实类别 logit 的变化减去其他九类 logit 变化的平均值”。这衡量相对类别证据的变化，不是概率、准确率或分类翻转率。原始误分类图像也保留。十维输出向量另作配套检查。没有使用 5,000 张校准图像的模型输出。

令 b_i=1 表示干预第 i 个坐标，G(T) 为相对于原网络的输出变化。空联盟是原网络，所以 G(∅)=0（允许浮点误差）。定义：

$$a_i=G(\\{i\\})-G(\\varnothing),$$

$$a_{ij}=G(\\{i,j\\})-G(\\{i\\})-G(\\{j\\})+G(\\varnothing).$$

单独查询模型使用 G(∅)+Σ_i a_i b_i；成对模型再加 Σ_{i<j} a_{ij}b_i b_j。每张图像各自建模，无跨图像学习。q=8 时分别访问 9 和 37 个响应，统一在另外 **219 个 |T|>2 的组合**上评分。q=4、12 的查询数为 5/11、13/79，评分组合数分别为 5、4,017。

这里的留出只表示拟合多项式时没有访问那些组合值。本轮复用曾经分析过的数据，不是新获得的独立测试集。分析代码与协议在本轮计算前冻结，不能因此称为独立预注册。

误差以 RMSE 除以真实干预效应的 RMS，记作 NRMSE。例如 1.20% 是“预测残差相对于干预效应的大小”，不是分类准确率下降 1.20%。先在每个教师的相同评分域上汇总，再报告六教师均值和标准差；不把组合或重复图像当作独立重复实验。

## 结果一：成对查询能改善预测，但并非每张图像都改善

主分析：q=8、α=1。下表均为六教师均值 ± 标准差，标量目标。

| 位置 | 单独干预模型：9 次查询 | 加入成对干预：37 次查询 | 教师整体误差改善 |
|---|---:|---:|---:|
{{MAIN}}

十维向量也一致：中心误差从 4.14% 降至 1.47%，左上从 5.06% 降至 1.99%。因此，主结论不完全依赖真实类别对比这一标量投影。这里的改善伴随更多查询，**不是同查询预算下的方法优越性结论**。

有成对改善不等于发现新的交互理论。该成对系数是以原始输入为背景的二阶有限差分；它不是在所有联盟背景上平均的 Shapley 交互值，更不能直接按其正负命名语义协同或拮抗。

## 结果二：当前游戏的真正高阶残差很小

只用空、单、双联盟建立的模型可能因为查询集中在原始背景附近而外推不佳。这与“任何二阶模型都做不好”是两个命题。为区分它们，我们还使用全部组合响应，作均匀二元立方体上的 Walsh 正交投影：

$$G(b)=\\sum_A \\widehat G_A\\chi_A(b),\\qquad \\chi_A(b)=\\prod_{i\\in A}(1-2b_i).$$

保留 |A|≤2 是这个评分分布下均方意义上的最佳二阶近似；舍弃项的平方和给出三阶及以上的不可消除均方残差。Walsh 的低阶空间与二元变量中的低阶多项式空间相同。分解本身是既有数学工具，不是本研究提出的新归因指数。

| 位置 | 最佳一阶的全组合 NRMSE | 最佳二阶的全组合 NRMSE | 三阶及以上 / 去均值响应方差 |
|---|---:|---:|---:|
{{ORACLE}}

前两列以总响应均方为分母；末列以去均值方差为分母，避免平均响应很大掩盖随联盟变化的部分。三阶及以上仅占约 **0.00293% / 0.00525%** 的中心/左上方差（先按教师计算，再平均）。这支持“当前局部游戏整体以低阶成分为主”。它并不证明任意图像或其他网络层中高阶都不重要。

最佳二阶查看了全部 256 个响应，因此不具有 37 次查询预测器的访问成本。其评分域也包含拟合查询，和上一节不同。两个表不能直接作为等预算竞赛、性能差的精确分解或新的预测成绩。小的最佳残差只说明低阶函数空间有潜力；能否少量查询就找到好的低阶模型，需要另做实验。

## 结果三：保留失效尾部，避免总体均值掩盖问题

效应 RMS≤10⁻⁹ 的记录不计算相对误差，仍计入绝对误差。q=8、α=1 的全部记录如下；10% 只是沿用的诊断线，没有应用层面的统一合格含义。

| 位置 | 可计算相对误差的记录 | 二阶 NRMSE 超过 10% | 最大单输入 NRMSE | 该最大相对误差案例的 RMSE / 效应 RMS |
|---|---:|---:|---:|---:|
{{TAIL}}

中心、左上分别有 65、3 条近零效应记录，没有被写成“相对误差为零”。最高相对误差案例为教师 202609075、左上位置、测试图像 919；其绝对误差约 0.00318、效应约 0.00430，说明解释这类尾部时必须同时看分子和分母。中心最高案例为教师 202609076、图像 985。它们按统一规则保留，不作为事后挑选的新主检验。

在有效记录中，成对模型的绝对 RMSE 改善超过 10⁻⁹ 的数量为中心 615/703、左上 727/765。整体六教师都改善，并不意味着每张图像都改善。当前结果没有给出能预先筛选这些失败输入的可靠阈值。

## 结果四：状态变化与误差相关，但还不是新机制

当前受影响的尾部具有一个非线性阶段，可写为：

$$\\Delta F=U[\\operatorname{ReLU}(z+W\\delta h)-\\operatorname{ReLU}(z)].$$

用原始 z 的符号固定线性项，余下为门状态改变产生的非线性修正。若所有受影响状态在全部组合上保持同一符号，响应就是仿射的，不需要二阶或更高项。可能跨零的状态才需要保留非线性修正。我们核验了二阶及以上 Walsh 系数完全来自这个修正部分。

| 位置 | 每张输入的候选跨零状态数，教师均值 | 状态数与二阶预测绝对误差的 Spearman ρ，均值 ± SD | 状态数与最佳二阶残差的 ρ，均值 |
|---|---:|---:|---:|
{{CORR}}

目标权重加权的局部仿射余项上界与二阶预测 RMSE 也相关，两个位置的教师平均 ρ 为 0.840、0.784。**该上界约束的是局部仿射余项，不是二阶外推误差的已证明上界。** 相关性可能同时受激活幅度、目标权重和效应尺度影响；没有做确认性显著性检验，也没有将其用于本轮调参。

状态恒等式的吻合验证了这个特定计算段的来源，不能当作跨多层的语义因果机制。筛选使用浮点保护量，不是形式化的区间证明系统。

## 补充范围：其他规模与强度

下表均为六教师平均标量 NRMSE，斜线前后分别为中心/左上。完整标准差和单教师数据见结果目录。

| 源坐标数 | α | 每教师/位置输入数 | 成对模型留出 NRMSE，中心 / 左上 | 全表最佳二阶 NRMSE，中心 / 左上 |
|---|---:|---|---:|---:|
{{EXTRA}}

12 单元只有四张验证输入，不能与 4/8 的 128 张测试输入差别解释为纯规模规律，也不报告这四张上的相关系数。其中左上、半强度时，六教师中只有五个的成对模型整体优于单独模型。本文不隐藏这一反例。减半干预在这些汇总中通常误差更小，但它不是连续剂量反应验证。

## 这如何支撑下一步，而不是替创新点找理由

当前证据可保留的主张是：低阶组合响应有较好的整体可预测性，但它依赖输入、干预背景和下游非线性状态。应把候选创新从泛泛的“发现神经元组合/高阶交互”收缩到**组合响应的适用边界与可预测失效**。这个表述仍只是研究问题，不是已经建立的创新。

下一步优先检验同预算下的查询覆盖：在已用数据上开发并锁定一个 37 次查询设计，与空/单/双联盟的 37 次查询比较；以掩码设计矩阵条件数选择分散覆盖的组合，而不是根据测试响应挑点。随后在另行固定的新干预输入上，比较共同未访问联盟的误差；查询集合不同，评分集合须取各方法查询集合之外的交集，必要时同时报告全域成绩但不得混称留出。

为避免只对一个预算挑结果，可预设第二个预算，例如 64 次，但各方法都必须获得相同响应预算。模型阶数、正则化、数值稳定处理和失败阈值必须用开发数据确定；新干预评价前冻结。已有 5,000 校准图像继续保留。若查询覆盖改善足以解释差距，就收缩高阶方法的需求；若仍有稳定的困难输入，再检验状态指标能否在新输入上预测失败，以及更早、经过多个非线性层的干预是否改变结论。

这一方案还需要与真正拟合联合响应的相关方法比较。Neuron Shapley、GCC、PURE、CODEC 的目标与输出各不相同，本轮没有运行它们，不能声称优于它们。尤其 [CODEC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3b36297debce91932600be70800bc094-Abstract-Conference.html) 已涉及分组贡献，[GraphSHAP-IQ](https://proceedings.iclr.cc/paper_files/paper/2025/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html) 已利用结构计算交互；当前的数学工具、分组消融和加速均不能单独作为新颖性证据。原文证据与局限见 [中文综述](literature_review/神经元组合与跨层机制_证据综述.md)、[候选问题](literature_review/候选创新点与待验证问题.md)及 [逐项证据表](literature_review/逐项证据表.xlsx)。

本轮到此完成第一步诊断。尚未运行上述新查询设计，也没有新增跨多层干预、外部数据集或语义概念实验。

## 可复核性与文件

核对了 148 个输入文件，并与历史交付哈希及原始保存承诺核验；三个分析源码与协议均冻结。本轮独立 NumPy 状态重建相对历史未融合网络输出的最大绝对差为 **{{REFERENCE_MAX}}**；高阶系数来源恒等式差为 **{{IDENTITY_MAX}}**。这证明计算一致性，不是独立数据上的泛化证据。

独立核验器不调用本轮分析函数：检查全部 155,040 个低阶查询输出向量对应的保存系数；对 240 个确定性选取的输入—强度游戏，用显式多项式和直接奇偶基矩阵重算预测及最佳投影；回算全部 72 条教师汇总、12 条组汇总和 6,240 条 CSV 数据。最大差 **{{AUDIT_MAX}}**，均通过 10⁻⁹ 数值容差。另有一阶、二阶、纯三阶函数与稳定/跨零 ReLU 的已知答案控制。

- [分析协议](联合干预响应_分析协议_2026-09-25.md)：目的、指标、固定范围和限制。
- [汇总 JSON](../runs/joint_response_20260925/summary.json)：所有数值摘要。
- [逐图像指标 CSV](../runs/joint_response_20260925/per_input_metrics.csv)：含绝对误差、相对误差、零效应标记与状态指标。
- [逐教师汇总 CSV](../runs/joint_response_20260925/teacher_summary.csv)和 [分组汇总 CSV](../runs/joint_response_20260925/group_summary.csv)。
- [独立复核记录](../runs/joint_response_20260925/independent_verification.json)与 [源码和输入冻结清单](../runs/joint_response_20260925/protocol.json)。
- [运行说明](../RUNBOOK.md)：历史执行命令和只读保全核验入口。

每种方法的原始拟合系数与每阶能量保存在新运行目录的 cases 子目录。本轮未改写旧参考数组、教师模型或旧研究结果，也未重新生成研究包 ZIP。
'''

if __name__=='__main__':main()

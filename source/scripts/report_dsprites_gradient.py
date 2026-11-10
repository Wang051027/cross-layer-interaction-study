"""从逐例响应计算模型内梯度变化和交互误差，生成逐模型结果表和图。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import dsprites_teacher as teacher
import numpy as np
from scipy.stats import spearmanr

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_gradient_20261009'
REPORT=BASE/'reports/dSprites_梯度变化与交互传递_2026-10-09.md'
METHODS=('class_fixed','input_gradient','fixed_linear','state_linear','input_finite',
         'random_0','random_1','random_2','random_3','all_shortlist','all_reachable')
NAMES=dict(class_fixed='按类固定',input_gradient='原基线梯度交互评分',fixed_linear='基线梯度预测修补收益',
           state_linear='状态梯度预测修补收益',input_finite='实际半强度单点查询',
           all_shortlist='全部64候选',all_reachable='全部可达坐标')


def load(path):
    """原始NPZ只读，始终拒绝pickle。"""
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def matrix():
    """独立四项差分矩阵用于描述性分解，不读选择器的分数。"""
    import itertools
    rows=[]
    for a,b in itertools.combinations(range(4),2):
        for background in range(16):
            if background&((1<<a)|(1<<b)):continue
            row=np.zeros(16)
            row[[background|(1<<a)|(1<<b),background|(1<<a),background|(1<<b),background]]=[1,-1,-1,1]
            rows.append(row)
    return np.asarray(rows)


def margin(values,labels):
    """三类margin为正确类减其余类平均，对logit或交互都作同一投影。"""
    weights=np.full((len(labels),3),-.5);weights[np.arange(len(labels)),labels]=1.
    return np.einsum('n...c,nc->n...',values,weights)


def safe_spearman(a,b):
    """常数向量相关性未定义，保留None，不把它默认为0。"""
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    value=float(spearmanr(a,b).statistic)
    return value if np.isfinite(value) else None


def analyze():
    """保留全部教师、方法和强度，并明确主比较之外的分析为诊断。"""
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert not (RUN/'analysis.json').exists(),'Preserve completed analysis'
    rows=teacher.read(RUN/'summary.json')['rows']
    lookup={(r['seed'],r['alpha'],r['target'],r['method']):r for r in rows}
    panel=load(RUN/'panel/data.npz');labels=panel['labels'];contrasts=matrix()
    diagnostic=[];prediction=[];primary=[];transfer=[];task=[];sensitivity=[]
    rank_notes=teacher.read(RUN/'independent_audit.json')['floating_rank_notes']
    for seed in teacher.SEEDS:
        selection=load(RUN/f'selection/seed_{seed}/data.npz')
        for j,method in enumerate(('fixed_linear','state_linear')):
            truth=selection['finite_scores'][:,0];pred=selection['predicted_scores'][:,j,0]
            correlations=[safe_spearman(a,b) for a,b in zip(truth,pred)]
            finite=[v for v in correlations if v is not None]
            top=[len(set(a[:16])&set(b[:16]))/16 for a,b in
                 zip(np.argsort(-truth,axis=-1,kind='stable'),np.argsort(-pred,axis=-1,kind='stable'))]
            prediction.append(dict(seed=seed,method=method,
                score_nrmse=float(np.sqrt(np.mean((pred-truth)**2)/np.mean(truth**2))),
                median_spearman=float(np.median(finite)) if finite else None,
                mean_spearman=float(np.mean(finite)) if finite else None,
                defined_correlations=len(finite),undefined_correlations=36-len(finite),
                top16_overlap_fraction=float(np.mean(top)),truth_score_rms=float(np.sqrt(np.mean(truth**2)))))
        fixed=lookup[seed,1.,'margin','fixed_linear'];state=lookup[seed,1.,'margin','state_linear']
        primary.append(dict(seed=seed,restore_gain_pp=100*(fixed['restore_ratio']-state['restore_ratio']),
                            transplant_gain_pp=100*(fixed['transplant_error_ratio']-state['transplant_error_ratio'])))
        # 仅作事后数值敏感性检查：剔除独立审计认出的近并列截断病例。
        affected=sorted({r['image'] for r in rank_notes if r['seed']==seed})
        full_metrics=load(RUN/f'evaluation/seed_{seed}/alpha_1.0/metrics_margin.npz')
        keep=np.ones(36,dtype=bool);keep[affected]=False
        ratios=np.sqrt(np.c_[full_metrics['restore_mse'][keep].mean(0),
                            full_metrics['transplant_error_mse'][keep].mean(0)]/full_metrics['natural_mse'][keep].mean())
        sensitivity.append(dict(seed=seed,excluded_images=affected,remaining=int(keep.sum()),
            restore_gain_pp=float(100*(ratios[2,0]-ratios[3,0])),
            transplant_gain_pp=float(100*(ratios[2,1]-ratios[3,1]))))
        strengths={}
        for alpha in (.25,.5,.75,1.):
            raw=load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
            ni=margin(np.einsum('km,nmc->nkc',contrasts,raw['natural']),labels)
            li=margin(raw['linear_middle_interaction'],labels)
            tail=margin(raw['nonlinear_tail_interaction'],labels)
            signal=np.mean(ni**2)
            directional=margin(np.einsum('km,nmc->nkc',contrasts,raw['directional_derivative']),labels)
            diagnostic.append(dict(seed=seed,alpha=alpha,
                middle_relative_drift=float(raw['middle_drift'][:,:,0].mean()),
                source_relative_drift=float(raw['source_drift'][:,:,0].mean()),
                middle_direction_cosine=float(raw['middle_drift'][:,:,1].mean()),
                source_direction_cosine=float(raw['source_drift'][:,:,1].mean()),
                middle_base_norm_min=float(raw['base_gradient_norm'][:,0].min()),
                source_base_norm_min=float(raw['base_gradient_norm'][:,1].min()),
                tail_linearization_error_ratio=float(np.sqrt(np.mean(tail**2)/signal)),
                linear_interaction_ratio=float(np.sqrt(np.mean(li**2)/signal)),
                nonlinear_projection_share=float(np.sum(tail*ni)/np.sum(ni**2)),
                interaction_strength_derivative_rms=float(np.sqrt(np.mean(directional**2))),
                gate_flips_mean=raw['gate_flips'][:,1:].mean((0,1)).tolist(),
                zero_middle_base_gradients=int((raw['base_gradient_norm'][:,0]==0).sum()),
                zero_source_base_gradients=int((raw['base_gradient_norm'][:,1]==0).sum())))
            if alpha in (.5,1.):
                m=load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/metrics_margin.npz')
                strengths[alpha]=dict(jac=raw['middle_jacobian'],error=.5*(m['restore_mse'][:,4]+
                    m['transplant_error_mse'][:,4])/m['natural_mse'].mean())
            if alpha==1.:
                baseline=raw['natural'][:,0].argmax(-1)
                task.append(dict(seed=seed,accuracy=float(np.mean(baseline==labels)),
                    prediction_changes=int(np.sum(raw['natural'][:,1:].argmax(-1)!=baseline[:,None])),
                    nonempty_interventions=36*15,interaction_rms=float(np.sqrt(signal))))
        half,full=strengths[.5],strengths[1.]
        weights=np.full((36,3),-.5);weights[np.arange(36),labels]=1.
        jhalf=np.einsum('nscj,nc->nsj',half['jac'],weights)
        jfull=np.einsum('nscj,nc->nsj',full['jac'],weights)
        reference=np.linalg.norm(jhalf[:,0],axis=-1)
        drift=(np.linalg.norm(jfull[:,1:]-jhalf[:,1:],axis=-1)/reference[:,None]).mean(-1)
        correlation=safe_spearman(drift,full['error']-half['error'])
        transfer.append(dict(seed=seed,mean_gradient_drift_half_to_full=float(drift.mean()),
            error_contribution_drift_spearman=correlation,
            definition='Per-image mean state-gradient drift vs change in finite-list error contribution; normalized by teacher natural MSE at each strength',
            scope='exploratory association, no causal identification or independent-image significance test'))
    aggregate=[]
    for target in ('margin','vector'):
        for alpha in (.25,.5,.75,1.):
            for method in NAMES:
                values=[lookup[s,alpha,target,method] for s in teacher.SEEDS]
                aggregate.append(dict(target=target,alpha=alpha,method=method,
                    restore_percent=float(np.mean([r['restore_ratio'] for r in values])*100),
                    transplant_error_percent=float(np.mean([r['transplant_error_ratio'] for r in values])*100)))
            random=[lookup[s,alpha,target,f'random_{r}'] for s in teacher.SEEDS for r in range(4)]
            aggregate.append(dict(target=target,alpha=alpha,method='random_average',
                restore_percent=float(np.mean([r['restore_ratio'] for r in random])*100),
                transplant_error_percent=float(np.mean([r['transplant_error_ratio'] for r in random])*100)))
    result=dict(utc=teacher.now(),primary=primary,diagnostic=diagnostic,prediction=prediction,
                transfer=transfer,task=task,aggregate=aggregate,
                floating_rank_notes=rank_notes,numerical_sensitivity=sensitivity,
                sensitivity_mean_gain_pp={k:float(np.mean([r[k] for r in sensitivity])) for k in ('restore_gain_pp','transplant_gain_pp')},
                primary_mean_gain_pp={k:float(np.mean([r[k] for r in primary])) for k in ('restore_gain_pp','transplant_gain_pp')},
                primary_both_improved_teachers=sum(r['restore_gain_pp']>0 and r['transplant_gain_pp']>0 for r in primary))
    teacher.dump(RUN/'analysis.json',result)
    with (RUN/'method_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (RUN/'gradient_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnostic[0]));writer.writeheader();writer.writerows(diagnostic)
    return result


def plot(result):
    """标准静态研究图：六教师的配对差异可见，不绘制伪独立置信区间。"""
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.6),layout='constrained')
    strengths=np.array([.25,.5,.75,1.])
    diag={(r['seed'],r['alpha']):r for r in result['diagnostic']}
    for key,color,name in [('source_relative_drift','#bd4d35','Source (4 coordinates)'),
                            ('middle_relative_drift','#207ba3','Middle (reachable pool)')]:
        values=np.array([[diag[s,a][key]*100 for a in strengths] for s in teacher.SEEDS])
        for line in values:axes[0,0].plot(strengths,line,color=color,alpha=.18,linewidth=.9)
        axes[0,0].plot(strengths,values.mean(0),'-o',color=color,label=name)
    axes[0,0].set(title='A  State-gradient change',xlabel='Intervention strength',ylabel='Mean relative change (%)')
    axes[0,0].legend(fontsize=9)
    values=np.array([[diag[s,a]['tail_linearization_error_ratio']*100 for a in strengths] for s in teacher.SEEDS])
    for line in values:axes[0,1].plot(strengths,line,color='#725b9d',alpha=.25)
    axes[0,1].plot(strengths,values.mean(0),'-o',color='#725b9d')
    axes[0,1].set(title='B  Baseline-J interaction prediction error',xlabel='Intervention strength',ylabel='Error RMS / interaction RMS (%)')
    p={(r['seed'],r['method']):r for r in result['prediction']}
    for index,s in enumerate(teacher.SEEDS):
        values=[p[s,m]['median_spearman'] for m in ('fixed_linear','state_linear')]
        axes[1,0].plot([0,1],values,'-o',color=f'C{index}',label=str(s)[-1])
    axes[1,0].set_xticks([0,1],['Baseline gradient','State gradient'])
    axes[1,0].set(title='C  Half-strength singleton prediction',ylabel='Median Spearman correlation')
    axes[1,0].legend(title='Teacher',ncol=3,fontsize=8)
    names=['class_fixed','input_gradient','fixed_linear','state_linear','input_finite']
    agg={(r['target'],r['alpha'],r['method']):r for r in result['aggregate']}
    x=np.arange(5);ax=axes[1,1]
    for offset,key,color,label in [(-.18,'restore_percent','#207ba3','Restore residual'),(.18,'transplant_error_percent','#bd4d35','Transplant error')]:
        ax.bar(x+offset,[agg['margin',1.,m][key] for m in names],width=.34,color=color,label=label)
    ax.set_xticks(x,['Class\nfixed','Old\ngradient','Baseline\nprediction','State\nprediction','Finite\nreference'])
    ax.set(title='D  Full-strength joint patch, 16 coordinates',ylabel='RMS / natural interaction RMS (%)')
    ax.legend(fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.15)
    figure=BASE/'reports/figures/dsprites_gradient_20261009'
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure.with_suffix('.png'),dpi=200);fig.savefig(figure.with_suffix('.svg'));plt.close(fig)


def write_report(result):
    """报告根据预设比较给结论，保留失败证据和条件，避免改写创新点。"""
    gains=result['primary_mean_gain_pp'];both=result['primary_both_improved_teachers']
    agg={(r['target'],r['alpha'],r['method']):r for r in result['aggregate']}
    diag=result['diagnostic'];pred=result['prediction']
    def dm(alpha,key):return float(np.mean([r[key] for r in diag if r['alpha']==alpha]))
    def pm(method,key):return float(np.mean([r[key] for r in pred if r['method']==method]))
    def pair(alpha,method,target='margin'):
        r=agg[target,alpha,method];return f"{r['restore_percent']:.2f} / {r['transplant_error_percent']:.2f}"
    lines=[ '# dSprites 梯度变化与跨层交互传递', '',
        f"预设的全强度、16点、类别margin比较显示：相对于基线梯度预测，状态梯度预测的平均恢复改善为 {gains['restore_gain_pp']:+.2f} 个百分点，移植改善为 {gains['transplant_gain_pp']:+.2f} 个百分点；{both}/6 教师同时改善。两种规则都在半强度选点并封存后，才测全强度。",'',
        f"全强度下，中介梯度相对无干预梯度的平均变化为 {dm(1.,'middle_relative_drift')*100:.2f}%，四个源坐标梯度的平均变化为 {dm(1.,'source_relative_drift')*100:.2f}%。用基线中介梯度传播中介交互时，输出交互预测误差RMS与真实交互RMS之比平均为 {dm(1.,'tail_linearization_error_ratio')*100:.2f}%。这些量回答的是不同问题：梯度变化衡量局部斜率，后一个比率衡量固定斜率能否解释实际组合交互。",'',
        '## 从层关系转为可计算的传递问题','',
        '原问题是前一层的神经元组合怎样影响后一层。当前把它拆成三个可检验环节：源组合改变中介状态；中介状态改变尾部网络的局部斜率；该状态与斜率共同决定恢复和移植的有限输出效果。本轮保持权重冻结，研究logit对激活的导数，不研究训练中的参数更新或梯度下降。','',
        '设源状态为u，四坐标掩码为b，强度为α，则 u_S(α)=u−α(b⊙u)，中介 h_S(α)=block2(u_S)，输出 f_S(α)=F(h_S)。其中乘法只施加于四个源坐标。源层之后仍运行完整残差网络；梯度只是描述状态附近的导数，没有把整个网络假定为同一个线性函数。','',
        '三类margin固定为真实类别logit减其余两类的平均。每图16种源组合产生24个条件成对差分。对成对差分算子C，有 I(α)=C f(α)。四坐标的源改动本身可加，故源状态的C差分为零。','',
        '在不跨越ReLU折点的局部，有 d f_S/dα = −Σ_j b_Sj u_j J_source,Sj(α)，进而 dI/dα = C[df/dα]。因此需要考察梯度在实际干预方向上的作用；单独比较梯度范数大小并不能推出交互大小。代码保存所有三类方向导数，独立审计逐点核对链式法则。','',
        '## 基线梯度的精确误差分解','',
        '取无干预中介h0和尾部Jacobian J0。对每个组合定义 e_S=f_S−f0−J0(h_S−h0)，于是严格有 I=J0·C h + C e。第一项是固定基线斜率传递的中介交互，第二项是尾部非线性修正。这是加减同一项得到的代数恒等式，不是新的数学定理，也不保证得到唯一因果路径。','',
        '报告的尾部线性化误差为 ‖C e‖RMS/‖I‖RMS。线性分量和修正分量可能相互抵消，比率可能超过100%；不能把它直接叫作某层的因果贡献百分比。ReLU开关固定后，源之后成为仿射映射，四个源坐标的加性干预应不产生交互；实验另运行该控制网络并核对近零结果。','',
        '## 面板与提前冻结的比较','',
        '使用6个既有dSprites形状分类教师，不重训、不筛掉任何种子。新36图由6种大小、2个共同角度索引20与37、3种形状组成，位置索引固定15、15。官方弧度值用于角度覆盖选取；新面板与训练、验证、旧评价、旧平移以及10月7日72图面板均无像素重复。其他位置的这些角度可能已在训练出现，所以这不是全局旋转域外检验。','',
        '源层为block1完整输出，中介为block2完整输出。四个源标量、拓扑可达池、共同64候选均精确复用旧冻结坐标。预算16在候选名单内选点。主要比较仅替换预测修补收益时所用的恢复梯度：','',
        '| 方法 | 恢复单点的输出预测 | 移植单点的输出预测 |',
        '|---|---|---|',
        '| 基线梯度预测 | f_S−J0,j·δ_S,j | f0+J0,j·δ_S,j |',
        '| 状态梯度预测 | f_S−JS,j·δ_S,j | f0+J0,j·δ_S,j |','',
        'δ_S,j=h_S,j−h0,j。恢复从干预状态出发，因此状态法用该状态JS；移植从无干预状态出发，两种方法都用J0。它们有相同自然logit与中介变化，且不查询单点修补输出。预测收益沿用旧定义 s_j=mean(I²)−0.5[mean(R_j²)+mean((T_j−I)²)]，取前16名。实际单点查询只作为评分真值与逐图参照，不能回调预测公式。','',
        '半强度16组合的一次批次正向及3次类别反向即可得到全部候选梯度。实际单点参照每图另需2048个尾部输出向量。反向传播与正向向量的成本不同，本轮同时运行参照，不能用输出向量数直接宣称实际加速倍数。原基线梯度交互评分、按类固定和四个冻结随机名单是附加对照。所有稀疏名单冻结后，在α为0.25、0.5、0.75、1下测真实联合恢复与移植。','',
        '## 梯度变化与固定斜率误差','',
        '梯度变化为每图每个非空组合的 ‖gS−g0‖/‖g0‖，先平均36图与15组合，再平均六教师。源梯度限四个源坐标，中介梯度限全部拓扑可达坐标，两者空间不同，不比较它们的绝对范数。','',
        '| α | 源梯度变化百分比 | 中介梯度变化百分比 | 中介梯度方向余弦 | 固定J0交互预测误差百分比 |',
        '|---|---:|---:|---:|---:|']
    for alpha in (.25,.5,.75,1.):
        lines.append(f"| {alpha:.2f} | {dm(alpha,'source_relative_drift')*100:.2f} | {dm(alpha,'middle_relative_drift')*100:.2f} | {dm(alpha,'middle_direction_cosine'):.4f} | {dm(alpha,'tail_linearization_error_ratio')*100:.2f} |")
    lines += ['', '梯度方向余弦接近1并不保证组合交互预测准确：交互是四项相消后的量，局部梯度变化在具体方向上造成的误差可能与剩余交互同阶。当前只采四个强度，不能精确定位某个ReLU首次切换的强度。','',
        '全强度时，四阶段平均ReLU开关变化数分别为 '+
        '、'.join(f'{v:.2f}' for v in np.mean([r['gate_flips_mean'] for r in diag if r['alpha']==1.],axis=0))+
        '，依次对应block2首ReLU、block2输出ReLU、block3首ReLU、block3输出ReLU；每阶段有32768个标量单元。计数先按非空组合和图像求平均，不是每图所有组合的开关并集。不同开关变化可能在反向映射中相消，计数不能直接换算为输出贡献。','',
        '## 单点收益预测与联合修补','',
        '预测误差NRMSE为预测分数与真实分数的平方差均值开方，除以真实分数RMS，先在同教师36图与64候选内计算。Spearman为每图64候选的排序相关性，先取教师内中位数，再平均六教师；未定义的常数向量相关性单独记录。前16重合率以实际单点收益前16为参照。','',
        '| 方法 | 分数预测NRMSE | 排序相关性 | 前16重合率 |',
        '|---|---:|---:|---:|']
    for method in ('fixed_linear','state_linear'):
        lines.append(f"| {NAMES[method]} | {pm(method,'score_nrmse'):.3f} | {pm(method,'median_spearman'):.3f} | {pm(method,'top16_overlap_fraction')*100:.2f}% |")
    lines += ['', '全强度下的实际联合修补如下。所有比率都是百分比且越小越好；向量列使用同一套按margin选出的名单，输出只改为去类别均值的三类logit向量。按类和margin选择使用真实类别标签。','',
        '| 方法 | margin恢复和移植 | 向量恢复和移植 |','|---|---:|---:|']
    for method in ('class_fixed','input_gradient','fixed_linear','state_linear','input_finite','random_average','all_shortlist','all_reachable'):
        lines.append(f"| {NAMES.get(method,'随机四次平均')} | {pair(1.,method)} | {pair(1.,method,'vector')} |")
    state_full=agg['margin',1.,'state_linear'];class_full=agg['margin',1.,'class_fixed']
    lines += ['',f"与按类固定参照相比，状态梯度法平均恢复改善 {class_full['restore_percent']-state_full['restore_percent']:.2f} 个百分点，移植改善 {class_full['transplant_error_percent']-state_full['transplant_error_percent']:.2f} 个百分点。状态法与实际单点查询参照接近，说明局部梯度有助于替代昂贵的单点提名查询，但这里的实际查询也是单点排序，不是最优16点集合；目前没有单独测得计算加速倍数。",'',
              '预设主比较的逐教师改善为正时表示状态梯度法误差下降，单位为百分点。','',
              '| 教师 | 恢复改善 | 移植改善 |','|---|---:|---:|']
    for row in result['primary']:
        lines.append(f"| {row['seed']} | {row['restore_gain_pp']:+.2f} | {row['transplant_gain_pp']:+.2f} |")
    sensitive=result['sensitivity_mean_gain_pp']
    lines += ['',f"独立审计识别出 {len(result['floating_rank_notes'])} 例基线梯度预测在前16截断边界存在浮点近并列。两种等价四项差分的计算顺序使这些边界候选换位，预测收益分数的独立重算差最大约为10⁻¹⁷量级。原名单严格遵循冻结的原始分数，独立重算的排序损失也在按实际舍入差推导的界内，但这些候选的先后不能解释成明确机制差异。协议没有近零回退规则，本轮不补加规则或重选点。",'',
        f"事后数值敏感性检查仅剔除上述独立审计标记的两例输入与教师组合、保留全部六教师后，平均恢复改善为 {sensitive['restore_gain_pp']:+.2f} 个百分点，移植改善为 {sensitive['transplant_gain_pp']:+.2f} 个百分点。原主结果仍包含全部216个输入与教师组合；该检查仅评估计算顺序敏感性，不能代替新的验证面板。"]
    lines += ['', '## 同一名单跨强度的表现','',
        '| α | 基线梯度预测 | 状态梯度预测 | 实际单点参照 |','|---|---:|---:|---:|']
    for alpha in (.25,.5,.75,1.):lines.append(f"| {alpha:.2f} | {pair(alpha,'fixed_linear')} | {pair(alpha,'state_linear')} | {pair(alpha,'input_finite')} |")
    correlations=[r['error_contribution_drift_spearman'] for r in result['transfer']]
    lines += ['',
        '半强度名单到全强度的误差变化，同时受自然交互尺度、中介状态与尾部斜率影响。诊断还比较逐图梯度变化与实际查询名单的归一化误差贡献变化；六教师的Spearman依次为 '+', '.join('未定义' if v is None else f'{v:+.3f}' for v in correlations)+'。这只是同一图上的探索性关联，不构成梯度变化造成修补退化的因果证明。','',
        '## 对研究方向的含义','',
        '层关系可以落到“状态依赖的传递斜率”上，数学上已能定义并逐点核验。实验还必须区分梯度变化存在、单点预测改善、联合修补改善这三层证据。局部一阶法的成功或失败只检验当前近似；有限修补跨越ReLU开关以及多点共同作用，都可能使它与实际结果不同。','',
        f"本轮全强度 {sum(r['nonempty_interventions'] for r in result['task'])} 次非空源干预中，{sum(r['prediction_changes'] for r in result['task'])} 次改变预测类别；无干预准确率为 {min(r['accuracy'] for r in result['task'])*100:.2f}% 至 {max(r['accuracy'] for r in result['task'])*100:.2f}%。因此本轮机制结论首先属于logit交互，不能直接命名为决定形状识别的语义回路。",'',
        '下一步优先把同大小角度位置的不同形状状态作为源替换值，检查输出类别对比是否真被改变，再检验中介的阻断和重现效果。梯度方法则保留为单点提名工具，另做计算成本对照。若任务对照中的单点预测仍明显失准，再用开发图规定采样点数和积分规则，在新的因素面板检验沿修补路径累积梯度能否减少误差；局部预测精度提高未必能解决多点共同作用。本轮没有运行这些后续检验。','',
        '## 代码与独立核验','',
        '独立审计不调用新的梯度计算或汇总实现：以四层反向卷积和残差旁路伴随映射重算全部状态、全部三类的源与中介导数；用稠密24×16差分矩阵重算单点收益、名单和最终MSE；核对官方像素、因素、选择封存时序与旧交付哈希。原研究代码和结果仍保留，可继续复核。','',
        '[完整实验目录](../runs/dsprites_gradient_20261009/README.md) · [冻结协议](../runs/dsprites_gradient_20261009/protocol.json) · [逐教师分析](../runs/dsprites_gradient_20261009/analysis.json) · [梯度数据表](../runs/dsprites_gradient_20261009/gradient_diagnostics.csv) · [独立审计](../runs/dsprites_gradient_20261009/independent_audit.json)','',
        '![梯度变化与修补效果](figures/dsprites_gradient_20261009.png)','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    shutil_copy_report()


def shutil_copy_report():
    """保存运行内报告副本，报告相对链接以项目reports目录版本为准。"""
    import shutil
    (RUN/'source/reports').mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPORT,RUN/'source/reports'/REPORT.name)


if __name__=='__main__':
    result=analyze();plot(result);write_report(result)
    print(json.dumps(dict(status='PASS',report=str(REPORT),primary=result['primary_mean_gain_pp'],
                         both_improved=result['primary_both_improved_teachers']),ensure_ascii=False))

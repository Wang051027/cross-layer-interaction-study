"""将形状配对的恢复、移植和路径积分误差按模型汇总并绘图。"""
from __future__ import annotations

import csv
import argparse
import itertools
import json
import os
import dsprites_teacher as teacher
import numpy as np
from scipy.stats import spearmanr

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_shape_donor_20261010'
REPORT=BASE/'reports/dSprites_形状配对与通道中介_2026-10-10.md'
FIGURE=BASE/'reports/figures/dsprites_shape_donor_20261010'
METHODS=('global_fixed','pair_fixed','fixed_linear','state_linear','input_finite')
NAMES=dict(global_fixed='全局固定',pair_fixed='类别转换固定',fixed_linear='基线梯度',
           state_linear='状态梯度',input_finite='实际单通道查询',random_mean='四组随机均值')


def load(path):
    """数值原件只读，关闭句柄并拒绝pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def contrast_matrix():
    """独立枚举四源组的24个条件交互，不将端点变化当成交互。"""
    rows=[]
    for a,b in itertools.combinations(range(4),2):
        for background in range(16):
            if background&((1<<a)|(1<<b)):continue
            row=np.zeros(16)
            row[[background|(1<<a)|(1<<b),background|(1<<a),background|(1<<b),background]]=[1,-1,-1,1]
            rows.append(row)
    return np.asarray(rows)


def correlation(a,b):
    """常数排序未定义，保留None。"""
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    v=float(spearmanr(a,b).statistic)
    return v if np.isfinite(v) else None


def mean(values):
    """普通描述性均值，无独立性或显著性假定。"""
    return float(np.mean(values))


def analyze():
    """从已封存响应提取预设主结果，并标记补充诊断。"""
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert not (RUN/'analysis.json').exists(),'Preserve completed analysis'
    rows=teacher.read(RUN/'summary.json')['rows'];specs=teacher.read(RUN/'specs.json')
    lookup={(r['seed'],r['alpha'],r['target'],r['method'],r['budget']):r for r in rows}
    panel=load(RUN/'panel/data.npz');pairs=panel['pairs'];labels=panel['labels']
    rlabel=labels[pairs[:,0]];dlabel=labels[pairs[:,1]]
    primary=[];prediction=[];gradients=[];path_errors=[];categories=[];task=[]
    for seed in teacher.SEEDS:
        fixed=lookup[seed,1.,'task_endpoint','fixed_linear',4]
        state=lookup[seed,1.,'task_endpoint','state_linear',4]
        primary.append(dict(seed=seed,restore_gain_pp=100*(fixed['restore_ratio']-state['restore_ratio']),
            transplant_gain_pp=100*(fixed['transplant_error_ratio']-state['transplant_error_ratio'])))
        selected=load(RUN/f'selection/seed_{seed}/data.npz');truth=selected['finite_scores']
        for j,method in enumerate(('fixed_linear','state_linear')):
            pred=selected['predicted_scores'][:,j]
            corr=[correlation(a,b) for a,b in zip(truth,pred)];defined=[x for x in corr if x is not None]
            # 原始收益含共享E²项，另报去除逐配对均值后的误差，以免公共项掩盖排名误差。
            tc=truth-truth.mean(-1,keepdims=True);pc=pred-pred.mean(-1,keepdims=True)
            item=dict(seed=seed,method=method,
                score_nrmse=float(np.sqrt(np.mean((pred-truth)**2)/np.mean(truth**2))),
                centered_score_nrmse=float(np.sqrt(np.mean((pc-tc)**2)/np.mean(tc**2))),
                median_spearman=float(np.median(defined)) if defined else None,
                defined_correlations=len(defined),undefined_correlations=len(corr)-len(defined))
            for budget in (4,8):
                torder=np.argsort(-truth,axis=-1,kind='stable');porder=np.argsort(-pred,axis=-1,kind='stable')
                item[f'top{budget}_overlap_fraction']=mean([len(set(a[:budget])&set(b[:budget]))/budget for a,b in zip(torder,porder)])
            prediction.append(item)
        for alpha in (1.,.5):
            raw=load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz');weights=raw['weights']
            natural_class=raw['natural'].argmax(-1)
            task.append(dict(seed=seed,alpha=alpha,native_accuracy=mean(raw['native_logits'].argmax(-1)==labels),
                full_source_donor_rate=mean(natural_class[:,-1]==dlabel),
                nonempty_source_predictions_changed=int((natural_class[:,1:]!=rlabel[:,None]).sum()),
                nonempty_source_interventions=len(pairs)*15))
            for j,spec in enumerate(specs):
                pclass=raw['patched'][:,j,:,-1].argmax(-1)
                categories.append(dict(seed=seed,alpha=alpha,**spec,
                    recipient_return_rate=mean(pclass[:,0]==rlabel),donor_reproduction_rate=mean(pclass[:,1]==dlabel),
                    restore_donor_rate=mean(pclass[:,0]==dlabel),transplant_recipient_rate=mean(pclass[:,1]==rlabel),
                    restore_third_class_count=int(((pclass[:,0]!=rlabel)&(pclass[:,0]!=dlabel)).sum()),
                    transplant_third_class_count=int(((pclass[:,1]!=rlabel)&(pclass[:,1]!=dlabel)).sum())))
            drift=raw['gradient_drift'];norm=raw['baseline_gradient_norm']
            gradients.append(dict(seed=seed,alpha=alpha,
                source_relative_drift=mean(drift[:,1:,0,0]),middle_relative_drift=mean(drift[:,1:,1,0]),
                source_direction_cosine=mean(drift[:,1:,0,1]),middle_direction_cosine=mean(drift[:,1:,1,1]),
                source_endpoint_relative_drift=mean(drift[:,-1,0,0]),middle_endpoint_relative_drift=mean(drift[:,-1,1,0]),
                source_base_norm_min=float(norm[:,0].min()),middle_base_norm_min=float(norm[:,1].min()),
                zero_source_base_gradients=int((norm[:,0]==0).sum()),zero_middle_base_gradients=int((norm[:,1]==0).sum()),
                gate_flips_mean=raw['gate_flips'][:,1:].mean((0,1)).tolist()))
            if alpha==1.:
                path=load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/path.npz')
                effect=np.einsum('nc,nc->n',raw['natural'][:,-1]-raw['natural'][:,0],weights)
                directional=np.einsum('nsgc,nc->nsg',raw['source_effect'],weights).sum(-1)
                predictions=dict(baseline=directional[:,0],trapezoid=.5*(directional[:,0]+directional[:,-1]),
                    midpoint_8=path['integrated_groups'][:,0].sum(-1),midpoint_32=path['integrated_groups'][:,1].sum(-1))
                for name,pred in predictions.items():
                    residual=pred-effect
                    path_errors.append(dict(seed=seed,method=name,error_ratio=float(np.linalg.norm(residual)/np.linalg.norm(effect)),
                        effect_rms=float(np.sqrt(np.mean(effect**2))),effect_min=float(effect.min()),
                        maximum_absolute_error=float(np.abs(residual).max()),
                        median_relative_absolute_error=float(np.median(np.abs(residual/effect))),
                        direction_opposite_count=int((pred*effect<0).sum())))
    aggregate=[]
    for target in ('task_endpoint','interaction'):
        for alpha in (1.,.5):
            for budget in (4,8):
                for method in (*METHODS,'random_mean'):
                    names=[method] if method!='random_mean' else [f'random_{i}' for i in range(4)]
                    chosen=[r for r in rows if r['target']==target and r['alpha']==alpha and r['budget']==budget and r['method'] in names]
                    assert len(chosen)==6*len(names)
                    aggregate.append(dict(target=target,alpha=alpha,budget=budget,method=method,
                        restore_ratio=mean([r['restore_ratio'] for r in chosen]),
                        transplant_error_ratio=mean([r['transplant_error_ratio'] for r in chosen])))
    area=panel['pixel_areas'];ratios=area[pairs[:,1]]/area[pairs[:,0]]
    result=dict(utc=teacher.now(),primary=primary,aggregate=aggregate,prediction=prediction,
        gradients=gradients,path=path_errors,categories=categories,task=task,
        pixel_area=dict(minimum=int(area.min()),maximum=int(area.max()),
            unequal_pairs=int((area[pairs[:,0]]!=area[pairs[:,1]]).sum()),pairs=len(pairs),
            donor_recipient_area_ratio_min=float(ratios.min()),donor_recipient_area_ratio_max=float(ratios.max())),
        primary_mean_gain_pp=dict(restore=mean([r['restore_gain_pp'] for r in primary]),
            transplant=mean([r['transplant_gain_pp'] for r in primary])),
        primary_both_better=sum(r['restore_gain_pp']>0 and r['transplant_gain_pp']>0 for r in primary),
        units='six teacher replicates;72 directed pairs share36 images in12 factor conditions',
        centered_score_note='posthoc diagnostic, no reselection or tuning')
    teacher.dump(RUN/'analysis.json',result)
    # CSV保留全部六种子，避免仅交付平均值而隐藏反例。
    tables=dict(summary=rows,primary=primary,prediction=prediction,gradients=gradients,path=path_errors,
                categories=categories,task=task,aggregate=aggregate)
    folder=RUN/'tables';folder.mkdir()
    for name,values in tables.items():
        with (folder/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return result


def figures(a):
    """标准静态科研图，显示主指标、逐教师差异和路径积分误差。"""
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
    methods=(*METHODS,'random_mean');x=np.arange(len(methods));width=.34
    fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    main=[next(r for r in a['aggregate'] if r['target']=='task_endpoint' and r['alpha']==1 and r['budget']==4 and r['method']==m) for m in methods]
    axes[0,0].bar(x-width/2,[100*r['restore_ratio'] for r in main],width,label='恢复后的剩余变化')
    axes[0,0].bar(x+width/2,[100*r['transplant_error_ratio'] for r in main],width,label='移植重现误差')
    axes[0,0].set_xticks(x,[NAMES[m] for m in methods],rotation=20,ha='right')
    axes[0,0].set_ylabel('占自然端点变化RMS（%，越低越好）');axes[0,0].set_title('主结果：4通道，全强度，六教师均值');axes[0,0].legend(fontsize=10)
    rows=a['primary'];x2=np.arange(6)
    axes[0,1].bar(x2-width/2,[r['restore_gain_pp'] for r in rows],width,label='恢复改善')
    axes[0,1].bar(x2+width/2,[r['transplant_gain_pp'] for r in rows],width,label='移植改善')
    axes[0,1].axhline(0,color='black',lw=.8);axes[0,1].set_xticks(x2,[str(r['seed'])[-3:] for r in rows])
    axes[0,1].set_ylabel('状态梯度相对基线梯度改善（百分点）');axes[0,1].set_xlabel('教师种子末三位')
    axes[0,1].set_title('预设主比较：六教师分别保留');axes[0,1].legend()
    crows=[]
    for m in methods:
        names=[m] if m!='random_mean' else [f'random_{i}' for i in range(4)]
        crows.append([r for r in a['categories'] if r['alpha']==1 and r['budget']==4 and r['method'] in names])
    axes[1,0].bar(x-width/2,[100*mean([r['recipient_return_rate'] for r in c]) for c in crows],width,label='恢复后回到原形状')
    axes[1,0].bar(x+width/2,[100*mean([r['donor_reproduction_rate'] for r in c]) for c in crows],width,label='移植后判为对照形状')
    axes[1,0].set_xticks(x,[NAMES[m] for m in methods],rotation=20,ha='right');axes[1,0].set_ylim(0,105)
    axes[1,0].set_ylabel('配对比例（%，越高越好）');axes[1,0].set_title('分类结果与连续logit指标分开');axes[1,0].legend(fontsize=10)
    pmethods=('baseline','trapezoid','midpoint_8','midpoint_32')
    for seed in teacher.SEEDS:
        v=[next(r['error_ratio'] for r in a['path'] if r['seed']==seed and r['method']==m) for m in pmethods]
        axes[1,1].plot(np.arange(4),100*np.asarray(v),marker='o',alpha=.8,label=str(seed)[-3:])
    axes[1,1].set_xticks(np.arange(4),['起点梯度','端点梯形','8点中点积分','32点中点积分'],rotation=15)
    axes[1,1].set_yscale('log');axes[1,1].set_ylabel('端点变化重建误差RMS（%，对数轴）')
    axes[1,1].set_title('同一源状态直线路径；样本数事先固定');axes[1,1].legend(ncol=3,fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    FIGURE.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(FIGURE.with_suffix('.'+suffix),dpi=160)
    plt.close(fig)


def path_figure():
    """固定第一教师第一配对，直接显示路径输出与方向梯度，不按结果挑图。"""
    import matplotlib.pyplot as plt
    seed=teacher.SEEDS[0];folder=RUN/f'evaluation/seed_{seed}/alpha_1.0'
    raw=load(folder/'data.npz');path=load(folder/'path.npz');i=0;weight=raw['weights'][i]
    effect=(raw['natural'][i,-1]-raw['natural'][i,0])@weight;origin=raw['natural'][i,0]@weight
    x=np.concatenate(([0.],path['positions'][i],[1.]));order=np.argsort(x)
    y=np.concatenate(([0.],(path['path_logits'][i]@weight-origin)/effect,[1.]))
    endpoints=(raw['source_effect'][i]@weight).sum(-1)[[0,-1]]/effect
    d=np.concatenate(([endpoints[0]],path['path_group_derivative'][i].sum(-1)/effect,[endpoints[1]]))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    axes[0].plot(x[order],y[order],marker='.',label='实际任务对比变化 / 完整变化')
    axes[0].plot([0,1],[0,1],'--',label='端点间直线');axes[0].set_ylabel('归一化任务对比变化')
    axes[1].plot(x[order],d[order],marker='.',label='实际方向梯度 / 完整变化')
    axes[1].axhline(endpoints[0],ls='--',color='tab:orange',label='起点方向梯度')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('归一化路径方向梯度')
    for ax in axes:ax.set_xlabel('源状态替换比例 t');ax.grid(alpha=.2);ax.legend(fontsize=10)
    fig.suptitle('预定示例：第一教师，最小生成大小，角度17，方形 → 椭圆',fontsize=12)
    for suffix in ('png','svg'):fig.savefig(FIGURE.with_name(FIGURE.name+'_path').with_suffix('.'+suffix),dpi=160)
    plt.close(fig)


def percent(v):
    """比率转为百分数。"""
    return f'{100*v:.2f}%'


def write_report(a):
    """解释任务、数学机制和证据边界，所有数值由审计后的分析填入。"""
    def aggregate(target,alpha,budget,method):
        return next(r for r in a['aggregate'] if (r['target'],r['alpha'],r['budget'],r['method'])==(target,alpha,budget,method))
    def rates(method,budget=4):
        names=[method] if method!='random_mean' else [f'random_{i}' for i in range(4)]
        rows=[r for r in a['categories'] if r['alpha']==1 and r['budget']==budget and r['method'] in names]
        return mean([r['recipient_return_rate'] for r in rows]),mean([r['donor_reproduction_rate'] for r in rows])
    gain=a['primary_mean_gain_pp'];fixed=aggregate('task_endpoint',1.,4,'fixed_linear');state=aggregate('task_endpoint',1.,4,'state_linear')
    sr,st=rates('state_linear');pair=aggregate('task_endpoint',1.,4,'pair_fixed')
    path={m:mean([r['error_ratio'] for r in a['path'] if r['method']==m]) for m in ('baseline','trapezoid','midpoint_8','midpoint_32')}
    gradient=[r for r in a['gradients'] if r['alpha']==1.]
    score={m:[r for r in a['prediction'] if r['method']==m] for m in ('fixed_linear','state_linear')}
    audit=teacher.read(RUN/'independent_audit.json');area=a['pixel_area']
    combination=teacher.read(RUN/'combination_diagnostic.json') if (RUN/'combination_diagnostic.json').exists() else None
    partial=sum(r['nonempty_source_predictions_changed']-72 for r in a['task'] if r['alpha']==1.)
    lines=['# dSprites形状配对中的梯度变化与跨层通道传递','',
        '本轮把最初的“层与层之间是否存在组合关系”变成一个具体问题：大小、角度、位置相同的两张不同形状图，前层状态逐组替换后，哪些后层通道能够阻断或重现模型形状判断的变化？同一替换路径上的梯度变化，能否定量解释实际输出变化？','',
        f'主预算为4个完整中介通道。全强度下，状态梯度名单相对基线梯度名单的恢复剩余和移植误差平均改善{gain["restore"]:.2f}、{gain["transplant"]:.2f}个百分点，{a["primary_both_better"]}/6教师两项同时改善；状态梯度恢复后回到原形状的比例为{percent(sr)}，移植后判为对照形状的比例为{percent(st)}。',
        '',f'源状态路径上，起点梯度预测端点变化的平均误差为{percent(path["baseline"])}，预设32点中点积分降至{percent(path["midpoint_32"])}。这是已知路径积分性质的数值验证，不能当作新的归因定理。','',
        '## 从原设想走到当前实验','',
        '原设想没有把神经网络假定为线性。问题在于“层间关系”过宽，而此前只替换四个标量激活，虽然能测到输出交互和梯度变化，全强度3240次干预仍未改变预测类别。这些旧结果保留为方法检查，本轮增加直接指向任务的配对对照。','',
        '现在，源输入来自真实配对图像，源组混合状态和中介修补状态则是人为构造的内部状态。先观察自然源替换，再恢复少量中介通道到原状态，检验阻断；另从原中介状态出发移入这些通道的替换状态，检验重现。两个方向都重要，单独恢复有效不能自动证明移植充分。','',
        '## 数据和干预单位','',
        '使用原六个冻结的三分类残差CNN，预测方形、椭圆、心形。模型不重训，权重和BatchNorm统计量固定，干预和梯度计算采用float64。梯度始终指logit对内部激活的导数，不是训练时参数更新的梯度。','',
        '新面板为6种大小 × 2个角度 × 3种形状，共36张图。角度索引17、22约为157°、203°，位置固定15、15；排除全部旧训练、验证、评价及前两轮新面板后，所有36张图的像素都不重复。每个因素条件形成6个有向形状配对，共72对；反向配对和共享图像相互依赖，不能当作72个独立实验重复。统计保留六个教师分别的结果，不报告基于432个“独立样本”的显著性。','',
        f'同生成大小不等于同像素面积：{area["unequal_pairs"]}/{area["pairs"]}配对面积不同，面积范围{area["minimum"]}—{area["maximum"]}像素，对照/原图面积比为{area["donor_recipient_area_ratio_min"]:.3f}—{area["donor_recipient_area_ratio_max"]:.3f}。结果涉及模型的形状标签判断，也可能依靠面积、边缘或其他相关特征，尚不能称为纯形状语义通路。角度覆盖很窄，不能外推为广泛旋转泛化。','',
        '| 位置 | 本轮干预单位 | 大小 |','|---|---|---|',
        '| 源层 block1 | 16通道固定分成4组，每组连续4通道 | 每组16384个标量，整层65536 |',
        '| 中介 block2 | 候选为全部32个完整通道 | 每通道1024个标量 |',
        '| 主预算 | 4个中介通道 | 4096个标量，占整层12.5% |',
        '| 补充预算 | 8个中介通道 | 8192个标量，占整层25% |','',
        '固定源分组不等于发现了最优神经元组合。本轮通道预算也不能与前轮16个标量神经元预算直接比较；“4通道”涉及的标量更多。','',
        '## 可检验的数学关系','',
        '令原图和对照图的源激活为 u₀、u₁，固定四组掩码为 M₁…M₄。集合 S 的替换状态为 u_S(α)=u₀+α∑_{g∈S}M_g⊙(u₁−u₀)。这是源状态的仿射构造，源组之间的四项差分恒为零。后续网络输出写为 f(u)，中介状态为 h(u)，任务对比固定为 φ(f)=f_对照类−f_原类。','',
        '任务端点变化 E=φ(f(u_all))−φ(f(u_∅))。恢复名单 P 后的剩余 R=φ(f_restore(P))−φ(f(u_∅))；移植产生的变化 T=φ(f_transplant(P))−φ(f(u_∅))。主要评价分别为 RMS(R)/RMS(E)、RMS(T−E)/RMS(E)，两项越小越好。分类是否回到原类、是否重现对照类另外记录；RMS比率不能解释成百分之多少样本预测正确。','',
        '条件交互 I_ab|B=φ(f(u_{B∪{a,b}}))−φ(f(u_{B∪{a}}))−φ(f(u_{B∪{b}}))+φ(f(u_B))，共6对源组 × 4种背景=24项。它衡量组合相对于单组叠加的差异，与完整端点变化E是两个不同目标。本轮名单按E提名，再用I作补充评价，不能称为专门优化交互的名单。','',
        '冻结所有后续ReLU开关并固定BatchNorm时，源到logit映射为仿射，I应为零；实际网络可以跨越ReLU区域。实验通过这一对照确认，测得的组交互由当前网络的后续非线性产生，而不是源替换公式自带交互。它尚未把每次输出变化定位到某一组开关。','',
        '沿完整源替换直线 u(t)=u₀+t(u₁−u₀)，方向导数 D(t)=∇_uφ(f(u(t)))·(u₁−u₀)，端点变化满足 E=∫₀¹D(t)dt。分组积分 A_g=∫₀¹∇_uφ(f(u(t)))·[M_g⊙(u₁−u₀)]dt，求和重建E。这里的完整性已有[Integrated Gradients原论文](https://proceedings.mlr.press/v70/sundararajan17a.html)依据。连续ReLU网络沿路径分段仿射；有限8点、32点中点求积仍有数值误差。分组归因可相加不等于网络函数可相加，也不等于路径无关或唯一的因果分解。','',
        '等价地，E=D(0)+∫₀¹[D(t)−D(0)]dt。右侧第二项是沿实际替换方向累积的梯度变化，可以解释起点切线漏掉多少输出变化；这比仅看梯度范数增减更接近可检验的传递关系。它是同一积分公式的代数改写，并非另一个新定理。','',
        '## 名单如何形成','',
        '固定名单只在旧144张开发图形成的288个有向配对上拟合。开发图属于旧训练子集，因此它是拟合集，不是额外独立测试集。全局固定按全部配对的平均单通道收益排序；类别转换固定分别对六个有向类别对拟合。新面板的半强度输出用于逐配对提名，所有名单封存后才进行全强度评价。','',
        '每个候选通道j的单点收益为 E²−[R_j²+(T_j−E)²]/2，在半强度全源组端点计算。基线梯度把恢复输出近似为 f_all−J₀,jΔh_j，把移植近似为 f₀+J₀,jΔh_j；状态梯度仅把恢复导数改为 J_all,j。实际单通道查询作为参照，另有四组随机固定名单。联合修补始终重新运行真实后续网络，不能用单通道输出相加代替。','',
        '实际单通道参照每配对查询32通道 × 恢复/移植2操作，共64个尾部输出，不是最优联合集合。梯度方案需要反向计算；本轮没有计时支持实际加速结论。','',
        '## 任务端点结果','',
        '| 方法 | 恢复剩余 | 移植误差 | 恢复后回到原形状 | 移植后判为对照形状 |','|---|---:|---:|---:|---:|']
    for method in (*METHODS,'random_mean'):
        row=aggregate('task_endpoint',1.,4,method);rr,tr=rates(method)
        lines.append(f'| {NAMES[method]} | {percent(row["restore_ratio"])} | {percent(row["transplant_error_ratio"])} | {percent(rr)} | {percent(tr)} |')
    lines +=['',f'以上为主预算4、全强度的六教师均值。状态梯度优于基线梯度的平均量分别为{gain["restore"]:.2f}、{gain["transplant"]:.2f}个百分点；以下保留每个教师的方向，正值表示改善。','',
        '| 教师种子 | 恢复改善百分点 | 移植改善百分点 |','|---|---:|---:|']
    lines +=[f'| {r["seed"]} | {r["restore_gain_pp"]:.3f} | {r["transplant_gain_pp"]:.3f} |' for r in a['primary']]
    lines +=['',f'类别转换固定名单的两项为{percent(pair["restore_ratio"])}、{percent(pair["transplant_error_ratio"])}。它的平均结果优于状态梯度名单：当前证据支持更新梯度能改善局部提名，却不支持“逐输入梯度名单优于所有固定方法”。开发集上的类别转换特征仍是强基线。','',
        '整源层替换后的72个端点在六教师上均为对照类；整中介32通道恢复和移植精确对应原状态与当前状态。这些是网络切口的结构正对照，不能作为稀疏中介发现的证据。','',
        f'除去四源组全选这一结构正对照，14种部分源组合在六教师、72配对上的6048次干预中有{partial}次改变原预测类别。这包含可能的第三类，不全部等于对照形状；它表明当前部分源替换确实影响任务判断，但干预单位已比旧标量实验扩大。','',
        '## 单通道预测和梯度变化','',
        '| 半强度预测 | 原收益NRMSE | 逐配对中心化NRMSE | 候选排序相关中位数的教师均值 | 前4通道重合率 | 前8通道重合率 |','|---|---:|---:|---:|---:|---:|']
    for method in ('fixed_linear','state_linear'):
        rs=score[method]
        lines.append(f'| {NAMES[method]} | {mean([r["score_nrmse"] for r in rs]):.3f} | {mean([r["centered_score_nrmse"] for r in rs]):.3f} | {mean([r["median_spearman"] for r in rs]):.3f} | {percent(mean([r["top4_overlap_fraction"] for r in rs]))} | {percent(mean([r["top8_overlap_fraction"] for r in rs]))} |')
    lines +=['','逐配对中心化误差为事后诊断：原收益包含通道共享的E²项，因此同时报告去除候选均值后的误差，避免公共项使误差看起来过小。它不参与名单或参数调整。排序相关、截断重合和最终联合修补分别衡量不同环节，不能相互替代。','',
        f'全强度15个非空源组状态中，源整层梯度平均相对变化为{percent(mean([r["source_relative_drift"] for r in gradient]))}，中介整层梯度为{percent(mean([r["middle_relative_drift"] for r in gradient]))}；方向余弦分别为{mean([r["source_direction_cosine"] for r in gradient]):.3f}、{mean([r["middle_direction_cosine"] for r in gradient]):.3f}。这些范数统计不直接等于任务影响；路径重建使用梯度在实际替换方向上的点积。所有原始范数、逐教师结果及开关变化数都在表中保留。','',
        '## 路径梯度能否重建输出变化','',
        '| 计算方法 | 六教师平均重建误差RMS比 |','|---|---:|',
        f'| 只用起点梯度 D(0) | {percent(path["baseline"])} |',
        f'| 两端点梯形 [D(0)+D(1)]/2 | {percent(path["trapezoid"])} |',
        f'| 预设8点中点积分 | {percent(path["midpoint_8"])} |',
        f'| 预设32点中点积分 | {percent(path["midpoint_32"])} |','',
        f'补充方向诊断：在432个教师与配对案例中，起点梯度预测和真实端点变化方向相反的有{sum(r["direction_opposite_count"] for r in a["path"] if r["method"]=="baseline")}个。这里仍有共享图像和反向配对，432不是独立样本数；这项事后诊断不改变主比较或选择规则。','',
        '误差比超过100%完全可能：单个局部切线未必能预测跨越许多非线性区域的有限替换。积分结果说明沿实际状态路径累积梯度可以准确追踪当前输出变化。它没有证明积分归因最高的源组一定是最有效的中介通道，也没有将源路径积分用于重新选通道。源路径重建和中介名单提名是两项实验。','',
        f'![预定配对的路径输出与方向梯度](figures/{FIGURE.name}_path.png)','',
        '图中固定第一教师、第一配对，不根据效果挑选案例。两端都来自真实图像的源状态，中间为合成内部状态；曲线解释当前替换过程，不能据此声称所有自然图像变形都会沿这条路径。','',
        '## 组合交互和预算补充','',
        '| 预算和目标 | 方法 | 恢复剩余 | 移植误差 |','|---|---|---:|---:|']
    for budget,target in ((4,'interaction'),(8,'task_endpoint'),(8,'interaction')):
        for method in METHODS:
            row=aggregate(target,1.,budget,method);name='条件交互' if target=='interaction' else '任务端点'
            lines.append(f'| {budget}通道 {name} | {NAMES[method]} | {percent(row["restore_ratio"])} | {percent(row["transplant_error_ratio"])} |')
    lines +=['','交互表使用任务名单，在24项四项差分上单独求RMS。一个名单能较好改变端点，不保证能恢复全部条件交互；两个目标使用各自自然量作分母，不应直接比较百分数大小。半强度和随机各重复的全部结果在summary.csv中。','',
        f'主预算4的状态梯度名单仍保留{percent(aggregate("interaction",1.,4,"state_linear")["restore_ratio"])}的条件交互RMS，移植误差为{percent(aggregate("interaction",1.,4,"state_linear")["transplant_error_ratio"])}。较好的端点修补尚未构成对全部组合机制的充分解释。','']
    if combination is not None:
        assert combination['status']=='PASS' and combination['new_model_queries']==0 and combination['new_selections']==0
        lines +=['## 中介通道联合效果不能简单相加','',
            '以下是原主比较完成后的事后诊断，只使用已保存的半强度精确单通道响应和联合响应，不增加模型查询，也不更换名单。源组的条件交互与这里的中介通道联合余项是不同干预，不能混为同一个量。','',
            '在固定源端点下，令F_R(P)为恢复通道集合P后的任务对比，F_all为无恢复时的对比；联合余项 C_R(P)=F_R(P)−F_all−∑_{j∈P}[F_R({j})−F_all]。移植余项 C_T(P)=F_T(P)−F₀−∑_{j∈P}[F_T({j})−F₀]。如果单通道效果可以直接相加，余项为零。它包含该基线下二阶及以上的总联合贡献，不是某一对通道的交互值，也不是Shapley值。','',
            '| 半强度方法 | 预算 | 恢复非加和余项RMS比 | 移植非加和余项RMS比 |','|---|---:|---:|---:|']
        for method in ('state_linear','input_finite'):
            for budget in (4,8):
                cs=[r for r in combination['rows'] if r['method']==method and r['budget']==budget]
                lines.append(f'| {NAMES[method]} | {budget} | {percent(mean([r["restore_nonadditive_ratio"] for r in cs]))} | {percent(mean([r["transplant_nonadditive_ratio"] for r in cs]))} |')
        lines +=['','即使用实际单通道输出，联合效果仍不能由相加精确预测，因此剩余问题不只是梯度单点预测误差。余项可能加强也可能抵消单点叠加，这个RMS表不能单独确定它在某一案例中的因果方向。','',
            '| 强度 | 状态梯度预算 | 恢复剩余 | 移植误差 | 恢复后回到原形状 | 移植后判为对照形状 |','|---|---:|---:|---:|---:|---:|']
        for alpha in (.5,1.):
            for budget in (4,8):
                row=aggregate('task_endpoint',alpha,budget,'state_linear')
                cr=[r for r in a['categories'] if r['alpha']==alpha and r['budget']==budget and r['method']=='state_linear']
                lines.append(f'| {alpha} | {budget} | {percent(row["restore_ratio"])} | {percent(row["transplant_error_ratio"])} | {percent(mean([r["recipient_return_rate"] for r in cr]))} | {percent(mean([r["donor_reproduction_rate"] for r in cr]))} |')
        cs=[r for r in combination['rows'] if r['method']=='state_linear' and r['budget']==8]
        lines +=['',f'增加通道后分类结果改善，但logit匹配误差增大。半强度8通道状态名单中，恢复对比越过原基线的比例为{percent(mean([r["restore_overshoot_fraction"] for r in cs]))}，移植对比越过自然端点的比例为{percent(mean([r["transplant_overshoot_fraction"] for r in cs]))}。过冲可能强化正确类别，所以不能把更高的logit误差直接写成分类变差；精确重建输出与改变决策边界是不同研究目标。非加和余项与过冲都存在，也不能据此断言前者独自导致后者。','',
            f'该诊断保留114行逐教师结果；先在logit空间求和再投影与先投影再求和的最大差异为{combination["independent_projection_maximum_error"]:.3g}。[逐教师联合表](../runs/dsprites_shape_donor_20261010/tables/channel_combinations.csv)记录全部方法。','']
    lines +=[
        f'![任务端点、逐教师差异、分类和路径积分](figures/{FIGURE.name}.png)','',
        '## 数值核验和能支撑的结论','',
        f'独立审计通过。四层卷积伴随重算全部组合和积分点的梯度；逐通道另造全部开发与新半强度单点修补；补集赋值另造全部19名单、16组合、两操作的联合修补；真实模型forward钩子作切口抽查。最大记录误差为{max(audit["maximum_errors"].values()):.3g}，数值近并列排序记录{len(audit["floating_rank_notes"])}条。冻结源、拟合、名单的时间顺序通过核验，旧三轮交付675个文件条目的哈希保留。','',
        '目前能够支撑的是：在这六个模型和固定面板上，真实形状配对的源替换会改变任务判断；部分中介通道能阻断或重现这些变化；状态梯度提名相对起点梯度有一定改善；沿源状态路径积分显著减少跨状态输出预测误差。','',
        '目前不能把它写成“发现了通用形状神经元组合”，也不能把路径完整性或ReLU仿射区域当成原创数学证明。较合理的研究主线是：跨层有限状态传递为何使局部梯度失准，路径上的梯度及开关变化怎样帮助预测通道联合修补。现有结果提供这个问题的实验基础，创新性还需与已有归因和激活修补研究比较。','',
        '## 下一步优先检验','',
        '首先明确下一轮要重建输出值还是改变预测类别，然后沿中介通道的实际恢复和移植路径积分，检查局部单点预测误差是否来自路径跨区。联合余项现已用已有数据定量检验，后续应进一步定位哪些通道组合和开关变化使其产生，并评价直接选择组合是否胜过单点排名。最后引入面积、边缘和不同源分组的对照，扩大位置、角度与模型范围。这些新增实验尚未运行。','',
        '## 代码和证据入口','',
        '- [实验协议与复现](../runs/dsprites_shape_donor_20261010/README.md)',
        '- [源替换、梯度和通道修补代码](../scripts/shape_donor_mechanism.py)',
        '- [阶段运行代码](../scripts/run_dsprites_shape_donor.py)',
        '- [独立审计代码](../scripts/audit_dsprites_shape_donor.py)',
        '- [全部456行连续指标](../runs/dsprites_shape_donor_20261010/tables/summary.csv)',
        '- [分类结果](../runs/dsprites_shape_donor_20261010/tables/categories.csv)',
        '- [路径积分逐教师结果](../runs/dsprites_shape_donor_20261010/tables/path.csv)',
        '- [前轮梯度报告](dSprites_梯度变化与交互传递_2026-10-09.md)','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def current(a):
    """更新项目当前进度。"""
    gain=a['primary_mean_gain_pp']
    path=mean([r['error_ratio'] for r in a['path'] if r['method']=='midpoint_32'])
    contents=f'''# 神经元组合与跨层机制研究当前进度

更新于2026年10月10日。六个冻结dSprites形状CNN已完成新36图、72个同因素有向形状配对实验；两种强度、4/8中介通道预算及源状态8/32点路径积分全部完成，独立伴随和全部联合修补核验通过。

当前问题是：前层不同形状状态逐组替换后，哪些后层通道能阻断或重现任务输出变化，梯度沿有限状态路径的变化是否解释局部预测失准。梯度仍指logit对激活的导数，模型不重训。

主预算4个完整通道。状态梯度相对基线梯度的恢复剩余及移植误差平均改善{gain['restore']:.2f}、{gain['transplant']:.2f}个百分点，{a['primary_both_better']}/6教师两项同时改善。类别转换固定名单的平均结果更好，当前不能声称输入自适应名单优于固定方法。源路径32点中点积分的平均输出重建误差为{100*path:.2f}%，但这是已知积分性质的数值验证，没有新增归因定理。

整源层替换会得到对照形状预测；整中介层恢复/移植是结构正对照，稀疏名单结果另评。当前源组每组16384个标量，中介4通道4096个标量，不能与旧16标量预算直接比较。配对同生成大小仍有面积差异，角度只有约157°、203°，不能据此宣称纯形状语义或广泛旋转泛化。

下一步优先沿中介修补路径积分，再区分单点排序误差与多个通道的联合非加和误差，加入面积/边缘和源分组对照。这些扩展尚未运行。

- [本轮中文报告](reports/{REPORT.name})
- [复现协议与数组说明](runs/dsprites_shape_donor_20261010/README.md)
- [计算代码和中文注释](scripts/shape_donor_mechanism.py)
- [全部连续指标](runs/dsprites_shape_donor_20261010/tables/summary.csv)
- [前轮梯度报告](reports/dSprites_梯度变化与交互传递_2026-10-09.md)

本文件可更新；原根目录README、RUNBOOK和RESEARCH_STATUS及全部旧交付继续保留原件。9月29日275、10月7日226、10月9日174个文件条目哈希均保持一致。本轮交付另保存当日进度副本。
'''
    if (RUN/'combination_diagnostic.json').exists():
        contents=contents.replace('下一步优先沿中介修补路径积分，再区分单点排序误差与多个通道的联合非加和误差，加入面积/边缘和源分组对照。这些扩展尚未运行。',
            '使用既有半强度数据的事后诊断已确认联合非加和余项：状态梯度4通道恢复/移植余项为自然端点变化RMS的23.70%/18.80%，8通道为42.95%/39.63%。增加通道使分类恢复和重现比例提高，但logit匹配误差变大，两种目标应分开。该诊断没有新增查询或更换名单。下一步优先明确任务指标，沿中介修补路径积分并定位组合余项，加入面积/边缘和源分组对照；这些新增实验尚未运行。')
    (BASE/'CURRENT_RESEARCH.md').write_text(contents,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-report',action='store_true',help='只更新交付前报告及进度，不重算分析或图')
    args=parser.parse_args()
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve frozen report'
    if args.refresh_report:
        assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
        a=teacher.read(RUN/'analysis.json')
    else:a=analyze();figures(a);path_figure()
    write_report(a);current(a)
    print(json.dumps(dict(status='PASS',primary_mean_gain_pp=a['primary_mean_gain_pp'],report=str(REPORT)),ensure_ascii=False))

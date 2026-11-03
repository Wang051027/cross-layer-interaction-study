"""从封存数组提炼通道配对、开关坐标、各阶项和反事实干预结果。

本文件只读数值实验，不改名单或再训练。跨模型仅汇总指标，不把相同
通道编号看成相同特征；全强度排名只用于描述半强度名单的稳定性。
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_channel_gates_20261011'
PRIOR=BASE/'runs/dsprites_shape_donor_20261010'
REPORT=BASE/'reports/dSprites_通道组合与ReLU开关定位_2026-10-11.md'
OPS=('restore','transplant');OP_CN=('恢复','移植');METHODS=('selected','random_0','random_1')


def load(path):
    """读纯数值数组；图像与实验值不经过pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def rms(value):
    """RMS先在一个教师内部计算，再跨六教师取均值。"""
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def write_csv(name,rows):
    """保留每个教师和条件，使用Excel可读的UTF8 BOM。"""
    path=RUN/'tables'/name;path.parent.mkdir(exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def project(value,weights,axes):
    """任务输出为供体形状logit减接收体形状logit，不是概率。"""
    return np.einsum(axes,value,weights)


def ratio_parts(total,parts):
    """带符号投影可相消；RMS比不是互斥的因果百分比。"""
    denominator=float(np.mean(total**2));assert denominator>1e-20
    return [dict(rms_ratio=rms(part)/rms(total),signed_projection=float(np.mean(part*total)/denominator)) for part in parts]


def analyze():
    """独立审计PASS后才写分析，所有门控位置仍使用半强度名单。"""
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert not (RUN/'analysis.json').exists(),'Preserve completed analysis'
    panel=load(PRIOR/'panel/data.npz');design=load(RUN/'design.npz');summary=teacher.read(RUN/'summary.json')['rows']
    weights=np.zeros((72,3));weights[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1
    weights[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    primary=[r for r in summary if r['alpha']==1. and r['ranks']=='first' and r['budget']==128]
    aggregates=[]
    for op in OPS:
        for stage in (1,2):
            rows=[r for r in primary if r['operation']==op and r['stage']==stage]
            methods={m:[r['remaining_ratio'] for r in rows if r['method']==m] for m in METHODS}
            chosen=np.array(methods['selected']);random=np.mean([methods['random_0'],methods['random_1']],0)
            aggregates.append(dict(operation=op,stage=stage,means={m:float(np.mean(v)) for m,v in methods.items()},
                random_mean=float(random.mean()),gain_pp=float(100*np.mean(random-chosen)),
                better_than_random_teachers=int(np.sum(chosen<random)),below_natural_teachers=int(np.sum(chosen<1)),
                per_teacher=[dict(seed=s,selected=float(chosen[j]),random_mean=float(random[j]),gain_pp=float(100*(random[j]-chosen[j]))) for j,s in enumerate(teacher.SEEDS)]))
    layers=[];orders=[];pair_rows=[];top_pairs=[];coordinate_rows=[];examples=[];stability=[];gate_counts=[]
    for seed in teacher.SEEDS:
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz')
        half_pair=project(frozen['pair_logits'],weights,'nopmc,nc->nopm')[:,:,:,0]
        full=load(RUN/f'evaluation/seed_{seed}/alpha_1.0/data.npz')
        full_pair=project(full['pair_logits'],weights,'nopmc,nc->nopm')
        full_parts=project(full['pair_stageparts'],weights,'nopsc,nc->nops')
        partial=project(full['partial_logits'],weights,'norlbmsc,nc->norlbms')
        partial_i=partial[...,3]-partial[...,1]-partial[...,2]+partial[...,0]
        for op in range(2):
            indices=frozen['selected_pairs'][:,op,0];selected=full_pair[np.arange(72),op,indices]
            algebra=full_parts[np.arange(72),op,indices]
            ratios=ratio_parts(selected[:,0],algebra.T)
            layers.append(dict(seed=seed,operation=OPS[op],natural_rms=rms(selected[:,0]),
                hold_first_ratio=rms(selected[:,1])/rms(selected[:,0]),hold_second_ratio=rms(selected[:,2])/rms(selected[:,0]),
                hold_both_ratio=rms(selected[:,3])/rms(selected[:,0]),
                first_signed_projection=ratios[0]['signed_projection'],second_signed_projection=ratios[1]['signed_projection'],
                first_rms_ratio=ratios[0]['rms_ratio'],second_rms_ratio=ratios[1]['rms_ratio']))
            # 物理通道对只在确实进入该输入8通道名单时记录，缺失不是零效应。
            physical=defaultdict(list);head1=defaultdict(int);head3=defaultdict(int)
            for i in range(72):
                for p,(a,b) in enumerate(design['pairs']):
                    key=tuple(sorted((int(frozen['channels'][i,a]),int(frozen['channels'][i,b]))))
                    physical[key].append((half_pair[i,op,p],full_pair[i,op,p,0]))
                    head1[key]+=int(p==frozen['selected_pairs'][i,op,0]);head3[key]+=int(p in frozen['selected_pairs'][i,op])
            rows=[]
            for key,observed in physical.items():
                v=np.asarray(observed)
                rows.append(dict(seed=seed,operation=OPS[op],channel_a=key[0],channel_b=key[1],
                    observed_cases=len(v),half_energy=float(np.sum(v[:,0]**2)),half_rms=rms(v[:,0]),
                    full_rms=rms(v[:,1]),half_top1_cases=head1[key],half_top3_cases=head3[key]))
            rows.sort(key=lambda r:(-r['half_energy'],r['channel_a'],r['channel_b']))
            pair_rows.extend(rows);top_pairs.extend([dict(rank=j+1,**r) for j,r in enumerate(rows[:3])])
            overlap=[];corr=[]
            for i in range(72):
                now=np.argsort(-np.abs(full_pair[i,op,:,0]),kind='stable')[:3]
                overlap.append(len(set(now)&set(frozen['selected_pairs'][i,op]))/3)
                corr.append(float(spearmanr(np.abs(half_pair[i,op]),np.abs(full_pair[i,op,:,0])).statistic))
            stability.append(dict(seed=seed,operation=OPS[op],top3_overlap=float(np.mean(overlap)),abs_rank_spearman=float(np.mean(corr))))
            for stage in range(2):
                ids=frozen['selected_gates'][:,op,0,stage];z=full['gate_preactivations'][:,op,0,stage]
                valid=ids>=0;switched=np.any((z>0)!=(z[:,0:1]>0),axis=1)&valid
                gate_counts.append(dict(seed=seed,operation=OPS[op],stage=stage+1,
                    mean_actual_budget=float(np.mean(valid.sum(1))),min_actual_budget=int(valid.sum(1).min()),
                    mean_half_eligible=float(np.mean(frozen['eligible_counts'][:,op,0,stage])),
                    full_switch_retention=float(switched.sum()/valid.sum())))
            for alpha in (1.,.5):
                raw=full if alpha==1. else load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
                mobius=project(raw['mobius4'],weights,'nosmc,nc->nosm')[:,:,:,0]
                joint=project(raw['joint_logits'],weights,'nokmc,nc->nokm')[:,:,:,0]
                pair=project(raw['pair_logits'],weights,'nopmc,nc->nopm')[:,:,:,0]
                groups=np.array([sum(mobius[:,op,s] for s in range(16) if s.bit_count()==order) for order in (2,3,4)])
                assert np.max(np.abs(groups.sum(0)-joint[:,op,0]))<1e-9
                pieces=ratio_parts(joint[:,op,0],groups)
                for j,order in enumerate((2,3,4)):
                    orders.append(dict(seed=seed,alpha=alpha,operation=OPS[op],channels=4,order=str(order),
                        total_joint_rms=rms(joint[:,op,0]),component_rms=rms(groups[j]),**pieces[j]))
                low=pair[:,op].sum(1);high=joint[:,op,1]-low;pieces=ratio_parts(joint[:,op,1],[low,high])
                for j,order in enumerate(('2','3+')):
                    orders.append(dict(seed=seed,alpha=alpha,operation=OPS[op],channels=8,order=order,
                        total_joint_rms=rms(joint[:,op,1]),component_rms=rms((low,high)[j]),**pieces[j]))
            # 全案例只导出半强度排序的前8坐标；完整128坐标和值保留在NPZ。
            for i in range(72):
                for rank,p in enumerate(frozen['selected_pairs'][i,op]):
                    positions=design['pairs'][p];channels=sorted(int(frozen['channels'][i,j]) for j in positions)
                    for stage in range(2):
                        for j,unit in enumerate(frozen['selected_gates'][i,op,rank,stage,:8]):
                            if unit<0:continue
                            c,y,x=np.unravel_index(int(unit),(32,32,32));z=full['gate_preactivations'][i,op,rank,stage,:,j]
                            hz=frozen['selection_preactivations'][i,op,rank,stage,:,j]
                            coordinate_rows.append(dict(seed=seed,case=i,operation=OPS[op],pair_rank=rank+1,
                                channel_a=channels[0],channel_b=channels[1],stage=stage+1,half_unit_rank=j+1,
                                state_a_channel=int(frozen['channels'][i,positions[0]]),state_b_channel=int(frozen['channels'][i,positions[1]]),
                                gate_channel=int(c),y=int(y),x=int(x),unit_id=int(unit),
                                half_contribution=float(frozen['selection_contributions'][i,op,rank,stage,j]),
                                full_contribution=float(full['unit_contributions'][i,op,rank,stage,j]),
                                half_pattern=''.join('1' if v>0 else '0' for v in hz),
                                full_pattern=''.join('1' if v>0 else '0' for v in z),
                                z_empty=float(z[0]),z_a=float(z[1]),z_b=float(z[2]),z_ab=float(z[3])))
            # 预定例子：第一教师、第一配对，避免在全强度结果中挑最漂亮的个案。
            if seed==teacher.SEEDS[0]:
                for rank,p in enumerate(frozen['selected_pairs'][0,op]):
                    physical=sorted(int(frozen['channels'][0,j]) for j in design['pairs'][p])
                    value=float(full_pair[0,op,p,0]);held=partial_i[0,op,rank,:,1,0]
                    examples.append(dict(seed=seed,case=0,operation=OPS[op],rank=rank+1,channels=physical,
                        state_channels=[int(frozen['channels'][0,j]) for j in design['pairs'][p]],
                        half_effect=float(half_pair[0,op,p]),full_effect=value,
                        stage_parts=full_parts[0,op,p].tolist(),whole_layer_effects=full_pair[0,op,p,1:].tolist(),
                        selected128_effects=held.tolist(),selected128_ratios=[float(abs(v/value)) if abs(value)>1e-12 else None for v in held],
                        coordinates=[r for r in coordinate_rows if r['seed']==seed and r['case']==0 and r['operation']==OPS[op] and r['pair_rank']==rank+1]))
    write_csv('summary.csv',summary);write_csv('physical_channel_pairs.csv',pair_rows)
    write_csv('gate_coordinates_top8.csv',coordinate_rows);write_csv('order_decomposition.csv',orders)
    write_csv('layer_holds.csv',layers);write_csv('ranking_stability.csv',stability);write_csv('gate_budgets.csv',gate_counts)
    result=dict(status='PASS',utc=teacher.now(),primary=aggregates,primary_rows=primary,layer_holds=layers,
        order_decomposition=orders,top_physical_pairs=top_pairs,examples=examples,ranking_stability=stability,
        gate_budgets=gate_counts,coordinate_rows=len(coordinate_rows),physical_pair_rows=len(pair_rows),
        example_rule='first teacher and first panel case, each operation; selected pairs and all coordinates fixed at half strength',
        aggregation_rule='physical pair half-energy ranked within each teacher/operation, only observed 8-list membership; coverage reported; descriptive, no new intervention list')
    sites=teacher.read(RUN/'single_sites/result.json');assert sites['status']=='PASS'
    result['single_site_supplement']=sites
    teacher.dump(RUN/'analysis.json',result);return result


def figures(a):
    """静态研究图输出PNG和SVG，缓存限定到本轮目录。"""
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    folder=BASE/'reports/figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.2),sharey=True)
    for row,op in enumerate(OPS):
        for col,stage in enumerate((1,2)):
            ax=axes[row,col];item=next(r for r in a['primary'] if r['operation']==op and r['stage']==stage)
            for j,s in enumerate(teacher.SEEDS):
                v=[next(r['remaining_ratio'] for r in a['primary_rows'] if r['operation']==op and r['stage']==stage and r['seed']==s and r['method']==m) for m in METHODS]
                ax.plot(range(3),np.array(v)*100,color='0.68',lw=.8,marker='o',ms=3)
            ax.plot(range(3),[100*item['means'][m] for m in METHODS],color='#1864ab',lw=2.6,marker='o',ms=6,label='Mean of 6 teachers')
            ax.axhline(100,color='#b02a37',ls='--',lw=1,label='Natural interaction')
            ax.set_xticks(range(3),['Selected','Random 1','Random 2']);ax.set_title(f'{op.capitalize()}, ReLU {stage}')
            ax.set_ylabel('Remaining pair interaction RMS (%)');ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=8);fig.suptitle('Full strength: 128 fixed gate coordinates, half-strength top pair',y=.99)
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ('png','svg'):fig.savefig(folder/f'dsprites_channel_gates_20261011.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    ex=next(r for r in a['examples'] if r['operation']=='restore' and r['rank']==1)
    fig,axes=plt.subplots(1,2,figsize=(11.7,5.6))
    for stage,ax in enumerate(axes,1):
        rows=[r for r in ex['coordinates'] if r['stage']==stage];data=np.array([[r['z_empty'],r['z_a'],r['z_b'],r['z_ab']] for r in rows])
        limit=float(abs(data).max());im=ax.imshow(data,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        for y in range(len(rows)):
            for x in range(4):
                ax.text(x,y,f'{data[y,x]:.3f}',ha='center',va='center',fontsize=8,color='white' if abs(data[y,x])>.6*limit else 'black')
        ax.set_xticks(range(4),['Empty',f'A: ch {ex["state_channels"][0]}',f'B: ch {ex["state_channels"][1]}','A + B'])
        ax.set_yticks(range(len(rows)),[f'({r["gate_channel"]},{r["y"]},{r["x"]})  {r["full_contribution"]:+.3f}' for r in rows],fontsize=8)
        ax.set_title(f'ReLU {stage}: preactivation; blue=open, red=closed');fig.colorbar(im,ax=ax,shrink=.75,label='Preactivation')
    fig.suptitle(f'Fixed example: teacher 291, square to ellipse, channel pair {tuple(ex["channels"])}\nRows = (gate channel, y, x) and signed full-strength contribution',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ('png','svg'):fig.savefig(folder/f'dsprites_channel_gates_20261011_example.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def four_channel_terms():
    """将已审计的前4通道容斥系数还原为物理组合，不新增网络查询。"""
    path=RUN/'four_channel_terms.json'
    if path.exists():return teacher.read(path)
    panel=load(PRIOR/'panel/data.npz');weights=np.zeros((72,3))
    weights[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1
    weights[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    rows=[];examples=[]
    for seed in teacher.SEEDS:
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz')
        for alpha in (1.,.5):
            raw=load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
            values=project(raw['mobius4'],weights,'nosmc,nc->nosm')[:,:,:,0]
            total=project(raw['joint_logits'],weights,'nokmc,nc->nokm')[:,:,0,0]
            for i in range(72):
                for op in range(2):
                    subtotal=0.
                    for s in range(16):
                        if s.bit_count()<2:continue
                        channels=sorted(int(frozen['channels'][i,j]) for j in range(4) if s&(1<<j))
                        row=dict(seed=seed,alpha=alpha,case=i,operation=OPS[op],order=s.bit_count(),
                            subset_mask=s,channels=','.join(map(str,channels)),coefficient=float(values[i,op,s]))
                        rows.append(row);subtotal+=row['coefficient']
                        if seed==teacher.SEEDS[0] and i==0 and s.bit_count()>=3:examples.append(row)
                    assert abs(subtotal-total[i,op])<1e-9
    write_csv('four_channel_mobius.csv',rows)
    value=dict(status='PASS',utc=teacher.now(),rows=len(rows),example_terms=examples,
        rule='all11 subsets of order2+ in inherited first4 list; fixed first teacher/case higher-order example; numerical coefficients already independently audited')
    teacher.dump(path,value);return value


def table(headers,rows):
    """Markdown表格用于并列条件，不将解释段落塞进单元格。"""
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def report(a):
    """报告正文与数字一起生成，无法把预设的正面结论写进结果。"""
    audit=teacher.read(RUN/'independent_audit.json');lines=[]
    first=[next(r for r in a['primary'] if r['operation']==op and r['stage']==1) for op in OPS]
    lines.extend(['# dSprites通道组合与ReLU开关定位','',
        f'半强度定位后，在全强度固定第一层ReLU的128个选中坐标，恢复与移植配对效应的剩余RMS分别为{100*first[0]["means"]["selected"]:.2f}%、{100*first[1]["means"]["selected"]:.2f}%，同数量随机坐标分别为{100*first[0]["random_mean"]:.2f}%、{100*first[1]["random_mean"]:.2f}%。预定输入的通道对(12,20)已定位到两种单独变化保持打开、合并后关闭的具体开关，同时单点干预也发现下游抵消和效应增强。','',
        '本轮把中介通道的联合非加和效应具体化为通道配对、两次ReLU的带符号贡献和坐标级开关干预。实验沿用六个冻结CNN、36张形状图和72个同大小同角度的有向配对；模型不重训。半强度选出的配对和坐标统一封存后，在全强度检验。','',
        '## 固定128个开关的主结果','',
        '主结果针对每个输入、操作在半强度排名第一的通道对。每位教师先对72个配对计算干预后的效应RMS与自然效应RMS之比，再对六位教师取均值。100%表示联合效应的幅度保持不变，超过100%表示增强；不等价于分类性能变差。随机对照从同一半强度发生开关变化的坐标池取同样数量，两个固定随机名单均保留。','',
        table(['操作','固定层','选中坐标剩余','随机1','随机2','较随机均值降低','教师更优数'],[
            [OP_CN[OPS.index(r['operation'])],f'ReLU {r["stage"]}',f'{100*r["means"]["selected"]:.2f}%',f'{100*r["means"]["random_0"]:.2f}%',f'{100*r["means"]["random_1"]:.2f}%',f'{r["gain_pp"]:+.2f}个百分点',f'{r["better_than_random_teachers"]}/6'] for r in a['primary']]),'',
        '![固定开关主比较](figures/dsprites_channel_gates_20261011.png)','',
        '下表列出所有教师，避免仅凭均值判断稳定性。单元格为选中坐标剩余／两次随机的均值；数值越小，配对效应幅度越小。','',
        table(['教师尾号','恢复R1','恢复R2','移植R1','移植R2'],[
            [str(s)[-3:]]+[f'{100*r["selected"]:.1f}% / {100*r["random_mean"]:.1f}%' for op,st in [('restore',1),('restore',2),('transplant',1),('transplant',2)] for r in [next(v for g in a['primary'] if g['operation']==op and g['stage']==st for v in g['per_teacher'] if v['seed']==s)]] for s in teacher.SEEDS]),'',
        '半强度坐标到全强度的开关变化保留情况如下。保持全部开关模式相同的坐标在当前四状态中没有开关余项，说明坐标名单的跨强度定位能力也有边界。候选池和实际预算均针对半强度第一配对。','',
        table(['操作','层','实际坐标数均值','最少坐标数','半强度候选池均值','全强度仍有开关变化'],[
            [OP_CN[op],stage,f'{np.mean([r["mean_actual_budget"] for r in a["gate_budgets"] if r["operation"]==OPS[op] and r["stage"]==stage]):.1f}',min(r['min_actual_budget'] for r in a['gate_budgets'] if r['operation']==OPS[op] and r['stage']==stage),f'{np.mean([r["mean_half_eligible"] for r in a["gate_budgets"] if r["operation"]==OPS[op] and r["stage"]==stage]):.1f}',f'{100*np.mean([r["full_switch_retention"] for r in a["gate_budgets"] if r["operation"]==OPS[op] and r["stage"]==stage]):.2f}%'] for op in range(2) for stage in (1,2)]),'',
        '## 哪些通道对产生效应','',
        '每个输入继承10月10日冻结的8通道名单，枚举名单内全部28对。不同教师的通道编号没有语义对齐。下表按半强度效应平方和在每个教师、操作内排序，给出前3个物理通道对；覆盖数为该通道对实际进入8通道名单的配对数，未观测的组合不填零。这是条件内的描述排序，覆盖较多的组合更容易累计较大平方和，不是所有32通道对的普遍排名。全强度值仅在相同已观测条件内计算。','',
        table(['教师','操作','通道对','覆盖/72','半强度RMS','全强度RMS','半强度第1次数'],[
            [str(r['seed'])[-3:],OP_CN[OPS.index(r['operation'])],f'({r["channel_a"]}, {r["channel_b"]})',r['observed_cases'],f'{r["half_rms"]:.4f}',f'{r["full_rms"]:.4f}',r['half_top1_cases']] for r in a['top_physical_pairs']]),'',
        '完整物理通道对表：[physical_channel_pairs.csv](../runs/dsprites_channel_gates_20261011/tables/physical_channel_pairs.csv)。该排序不能说明同号通道在六个模型中共同编码了某种形状。','',
        '## 预定输入的配对和具体开关','',
        '例子固定为第一教师202609291、面板第0个配对：最小生成大小、角度索引17（约157°）、方形接收体和椭圆供体。位置为(15,15)。配对及开关均按半强度选择，没有按全强度结果挑选输入。下表效应是供体logit减接收体logit的二阶差分，可正可负；固定开关后的数值表示重新运行网络得到的同一差分。','',
        table(['操作','半强度名次','通道对','半强度I','全强度I','第一层代数项','第二层代数项','固定R1的128坐标后I','固定R2的128坐标后I'],[
            [OP_CN[OPS.index(r['operation'])],r['rank'],str(tuple(r['channels'])),f'{r["half_effect"]:+.5f}',f'{r["full_effect"]:+.5f}',f'{r["stage_parts"][0]:+.5f}',f'{r["stage_parts"][1]:+.5f}',f'{r["selected128_effects"][0]:+.5f}',f'{r["selected128_effects"][1]:+.5f}'] for r in a['examples']]),'',
        '开关坐标记为(C,Y,X)，均从0计数。C属于block3该次ReLU的32通道，不能与被替换的block2通道直接等同。状态顺序为：未替换、只替换A、只替换B、同时替换AB；1表示ReLU前激活大于0、开关打开，0表示关闭。A/B遵循冻结8通道名单的原顺序，可能与上述排序后的数字顺序不同，精确对应见数组或后面的README。',''])
    for ex in [r for r in a['examples'] if r['rank']==1]:
        lines.extend([f'### {OP_CN[OPS.index(ex["operation"])]}通道对{tuple(ex["channels"])}','',
            f'此处A为通道{ex["state_channels"][0]}，B为通道{ex["state_channels"][1]}。','',
            table(['ReLU','半强度坐标名次','(C,Y,X)','半强度开关模式','全强度开关模式','全强度带符号贡献'],[
                [r['stage'],r['half_unit_rank'],str((r['gate_channel'],r['y'],r['x'])),r['half_pattern'],r['full_pattern'],f'{r["full_contribution"]:+.6f}'] for r in ex['coordinates'] if r['half_unit_rank']<=5]),''])
    sites=a['single_site_supplement']['rows']
    lines.extend(['### 单个开关固定的净效应','',
        '这是看过主结果后加做的探索补充。执行前另行封存规则：同一个预定输入、半强度第一配对、每层冻结名单前8个坐标，逐点固定及8点一起固定，不按全强度表现挑坐标。净变化为自然效应减固定后的效应；负值表示固定该点反而增强联合效应。','',
        table(['操作','ReLU','坐标或组','代数贡献','实际净变化','固定后I'],[
            [OP_CN[OPS.index(r['operation'])],r['stage'],r['coordinates'] if r['frozen_unit_rank']!='group8' else '前8坐标一起',f'{r["algebra_contribution"]:+.6f}',f'{r["net_effect_removed"]:+.6f}',f'{r["held_effect"]:+.6f}'] for r in sites if r['frozen_unit_rank']=='group8' or int(r['frozen_unit_rank'])<=3 or (r['operation']=='restore' and r['stage']==1 and r['frozen_unit_rank']==7)]),'',
        '例如恢复操作的第一层(16,17,14)，代数贡献为'+f'{next(r["algebra_contribution"] for r in sites if r["operation"]=="restore" and r["stage"]==1 and r["frozen_unit_rank"]==2):+.6f}'+
        '，实际固定单点只减少'+f'{next(r["net_effect_removed"] for r in sites if r["operation"]=="restore" and r["stage"]==1 and r["frozen_unit_rank"]==2):+.6f}'+
        '。第一层(16,17,13)甚至会增强效应，说明下游重新开关和抵消确实影响单点净作用。第二层所有逐点净变化则与代数贡献一致，最大核验误差'+f'{a["single_site_supplement"]["maximum_reference_error"]:.2e}'+
        '。因此不能把一个坐标的第一层代数贡献直接命名为它的独立因果效应。完整36项补充干预见[single_sites.csv](../runs/dsprites_channel_gates_20261011/single_sites/single_sites.csv)。',''])
    lines.extend(['![预定恢复输入的两层开关前激活](figures/dsprites_channel_gates_20261011_example.png)','',
        '全部案例、三个冻结配对、每层前8坐标见[gate_coordinates_top8.csv](../runs/dsprites_channel_gates_20261011/tables/gate_coordinates_top8.csv)。完整128坐标、所有预激活和贡献保留在每教师的discovery与evaluation NPZ中，未用一个小表替代原始证据。','',
        '## 开关怎样产生联合效应','',
        '研究的截面是block2输出H，尾部block3有两次ReLU。设z₁=A₁H+b₁，h₁=ReLU(z₁)，z₂=A₂h₁+b₂+H，h₂=ReLU(z₂)。尾部全局平均池化和读出都是线性的；批归一化使用冻结推理参数，A₁、A₂分别是卷积与批归一化的线性部分。','',
        '固定空背景的开关g₁⁰、g₂⁰，写r₁=ReLU(z₁)−g₁⁰z₁，r₂=ReLU(z₂)−g₂⁰z₂。对通道对A、B，定义I=F(AB)−F(A)−F(B)+F(∅)，F为任务logit差；ΔΔ表示这个四项差分。若q为读出的空间权重，p=A₂ᵀ(g₂⁰q)，则','',
        '`I = Σ p · ΔΔr₁ + Σ q · ΔΔr₂`','',
        '对应的任务梯度为`∇_H F = A₁ᵀ D₁ A₂ᵀ D₂ q + D₂ q`，D₁、D₂是当前ReLU开关的对角矩阵。权重与推理批归一化参数固定，开关保持不变时梯度保持不变；开关改变时，传播到H的梯度也可能改变。这把此前观察到的梯度变化落实到了具体空间坐标，而不是仅比较两层统计相关性。','',
        '层内卷积仍然线性；两个通道的变化叠加后穿越ReLU阈值，即可产生联合非加和。例如上述恢复操作在(16,16,15)出现1110：两个单独变化都没有关闭开关，合并后关闭。因此通道差分不是两个单通道效应的简单相加。但出现该效应并不能独自证明这些通道形成纯形状语义组合或普适专用电路。','',
        '这是该冻结网络的精确代数展开。通道修补状态H对成员指标是仿射的，所以两次ReLU都固定时四项差分应为零；该零值是结构对照。第一项描述第一层开关余项在固定第二层开关下的传播；第二项使用实际自然上游产生的第二层预激活。两项会抵消，分配依赖基准开关。实际固定第一层开关会重新计算第二层，因此第一项不等于固定第一层后的净效应变化。固定第二层某些坐标，则其净效应变化精确等于这些第二层坐标贡献之和。','',
        '全层开关固定的剩余幅度及带符号代数投影如下。代数投影为E[I·项]/E[I²]，两项投影相加为1；负值或超过1代表抵消，不能当作互斥的因果份额。','',
        table(['操作','整层R1固定剩余RMS','整层R2固定剩余RMS','第一项带符号投影','第二项带符号投影'],[
            [OP_CN[op],f'{100*np.mean([r["hold_first_ratio"] for r in a["layer_holds"] if r["operation"]==OPS[op]]):.2f}%',f'{100*np.mean([r["hold_second_ratio"] for r in a["layer_holds"] if r["operation"]==OPS[op]]):.2f}%',f'{np.mean([r["first_signed_projection"] for r in a["layer_holds"] if r["operation"]==OPS[op]]):+.3f}',f'{np.mean([r["second_signed_projection"] for r in a["layer_holds"] if r["operation"]==OPS[op]]):+.3f}'] for op in range(2)]),'',
        '整层双固定的最大配对差分误差为'+f'{audit["maximum_errors"]["affine_pair_zero"]:.2e}'+'。只固定128坐标的主实验与整层固定应分开解释；每层总共有32768个标量开关。强制基准开关可能把自然关闭的负预激活传向后层，这是内部计算反事实，不是自然图像或普通二值消融。','',
        '## 二阶配对能否解释全部组合','',
        '前4通道的全部16子集已计算，可精确容斥得到二、三、四阶项。以下RMS比针对“同一阶全部项的带符号和”，不等于逐项绝对值之和。8通道只计算全部28个空背景配对和整8通道组合，能够得到三阶及以上的合并余项，不能指出8通道内具体是哪一个三元组或四元组。','',
        table(['操作','通道数','阶数','该阶和/联合余项RMS','带符号投影'],[
            [OP_CN[op],k,order,f'{100*np.mean([r["rms_ratio"] for r in a["order_decomposition"] if r["alpha"]==1. and r["operation"]==OPS[op] and r["channels"]==k and r["order"]==order]):.2f}%',f'{np.mean([r["signed_projection"] for r in a["order_decomposition"] if r["alpha"]==1. and r["operation"]==OPS[op] and r["channels"]==k and r["order"]==order]):+.3f}'] for op in range(2) for k,order in [(4,'2'),(4,'3'),(4,'4'),(8,'2'),(8,'3+')]]),'',
        '8通道的三阶及以上余项仍然很大，且平均带符号投影为负，说明它们总体上抵消了空背景配对和的一部分。因此二阶配对定位只解释机制的一部分，不能把28个配对差分简单相加当作整8通道的效果。','',
        '预定输入在原4通道名单中的具体三阶、四阶系数如下，全部列出而非挑选最大值。三阶系数通过该三通道的全部8个子集容斥计算，四阶通过16个子集计算，属于空背景下的组合余项，不等于某一对通道的坐标干预净变化。','',
        table(['操作','阶数','物理通道组合','半强度系数','全强度系数'],[
            [OP_CN[OPS.index(r['operation'])],r['order'],r['channels'],f'{next(v["coefficient"] for v in four_channel_terms()["example_terms"] if v["alpha"]==.5 and v["operation"]==r["operation"] and v["subset_mask"]==r["subset_mask"]):+.5f}',f'{r["coefficient"]:+.5f}'] for r in four_channel_terms()['example_terms'] if r['alpha']==1.]),'',
        '所有案例的具体二至四阶组合见[four_channel_mobius.csv](../runs/dsprites_channel_gates_20261011/tables/four_channel_mobius.csv)。8通道的三阶及以上尚未逐组合拆开，与这张4通道表的范围不同。','',
        '半强度到全强度的28对绝对效应排序Spearman相关均值为'+f'{np.mean([r["abs_rank_spearman"] for r in a["ranking_stability"]]):.3f}'+'，前3集合重合比例均值为'+f'{100*np.mean([r["top3_overlap"] for r in a["ranking_stability"]]):.2f}%'+'。这是同一面板沿强度变化的稳定性，不是新输入泛化。其余预算32、半强度重复和前3配对汇总见[summary.csv](../runs/dsprites_channel_gates_20261011/tables/summary.csv)。','',
        '## 证据范围与后续问题','',
        '结论限定于每个教师、具体因素配对、冻结通道名单和两种修补操作。本轮所有数据属于已经看过的36图机制复核；角度只有约157°、203°，各形状面积不同，不能据此断言纯形状语义、旋转泛化或一个普适神经元组合。同一图参与多个有向配对，72个配对也不是72个独立样本；本报告不据此给出独立样本显著性。','',
        '下一步应将稳定的通道配对与开关模式定义为候选机制，在未用于定位的图像与面积/边缘对照中验证，并区分任务logit重建与分类效果。若三阶及以上余项较大，需要以特定背景条件下的配对和三元组干预继续定位，而不能把空背景的28对当成全部机制。','',
        '## 可复核材料','',
        f'独立核验已通过：原block3正向、所有43个子集的四种模式、全部坐标固定干预、容斥各阶和伴随/autograd计算均核对。最大数值误差{max(audit["maximum_errors"].values()):.2e}；半强度开关排名浮点边界记录{len(audit["floating_unit_rank_notes"])}条。此前四轮共836个交付文件条目逐一保持SHA256一致。','',
        '- [协议、数组含义与复现入口](../runs/dsprites_channel_gates_20261011/README.md)',
        '- [精确开关分解代码](../scripts/channel_gate_mechanism.py)',
        '- [实验阶段代码](../scripts/run_dsprites_channel_gates.py)',
        '- [独立核验代码](../scripts/audit_dsprites_channel_gates.py)',
        '- [本报告分析代码](../scripts/report_dsprites_channel_gates.py)','',
        '数值定义和主比较在本轮评估前冻结；精确代数与坐标干预为当前网络的机制证据，未提出新的普遍归因定理。',''])
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-docs',action='store_true')
    args=parser.parse_args()
    # 报告修订只重绘文档和图，不覆盖已经生成的分析与数值证据。
    a=teacher.read(RUN/'analysis.json') if args.refresh_docs else analyze()
    assert a['status']=='PASS' and teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    four_channel_terms();figures(a);report(a)
    import json
    print(json.dumps(dict(status='PASS',report=str(REPORT),primary=a['primary'],coordinate_rows=a['coordinate_rows']),ensure_ascii=False))

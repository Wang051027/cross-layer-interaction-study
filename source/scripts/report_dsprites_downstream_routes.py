"""生成下游路径定位报告、通道/位置图和联合阻断后的正负余量图。"""
from __future__ import annotations
import csv
import json
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_downstream_routes import BASE,RUN
from run_dsprites_upstream_rescue import load

REPORT=BASE/'reports/dSprites_下游通道位置与联合阻断_2026-10-25.md'
FIGDIR=BASE/'reports/figures/dsprites_downstream_routes_20261025'


def read_table(name):
    """只读结果表；图不另造数据。"""
    with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def figures(claims,ex,summaries):
    """导出科学图PNG/SVG；比例注明RMS定义，不当成因果解释百分比。"""
    assert not (RUN/'delivery_manifest.json').exists();FIGDIR.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    blue,gold,gray,ink='#285F8F','#BE7826','#7E8792','#24282E';paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            path=FIGDIR/f'{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');paths.append(str(path.relative_to(BASE)).replace('\\','/'))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),sharey=True)
    budgets=np.array([1,4,16,64,256,1024])
    for layer,ax in enumerate(axes):
        for order,label,color,marker in ((0,'Effect ranked',blue,'o'),(1,'Random 0',gold,'s'),(2,'Random 1',gray,'^')):
            ys=[]
            for b in range(6):
                selected=[r for r in summaries if r['cohort']=='primary26' and r['context']=='1' and int(r['layer'])==layer and int(r['command'])==order*6+b and int(r['count'])>0]
                ys.append(np.mean([float(r['remaining_rms_ratio']) for r in selected]))
            ax.plot(budgets,ys,color=color,marker=marker,lw=1.6,label=label)
        ax.axhline(1,color=ink,lw=.7,ls=':');ax.set_xscale('log',base=2);ax.set_xticks(budgets,[str(x) for x in budgets]);ax.set_ylim(0,1.08)
        ax.set_xlabel('Maximum number of restored positions');ax.set_title('First ReLU, original skip' if layer==0 else 'Final ReLU output',loc='left',fontweight='bold')
        ax.grid(axis='y',color='#E4E7EB',lw=.6);ax.legend(frameon=False,loc='lower left',fontsize=9)
    axes[0].set_ylabel('Remaining C-effect RMS / original C-effect RMS')
    fig.text(.08,.015,'New outside context; 26 records, 5 nonempty teachers. Mean of within-teacher RMS ratios; no independent test set.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));save(fig,'01_blocking_budget')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for layer,ax in enumerate(axes):
        rows=sorted([r for r in ex['channels'] if r['context']==1 and r['layer']==layer],key=lambda r:-abs(r['whole_channel_removed']))[:8]
        values=[r['whole_channel_removed'] for r in rows];yy=np.arange(8)
        ax.barh(yy,values,color=[blue if x>=0 else gold for x in values],height=.6);ax.set_yticks(yy,[str(r['channel']) for r in rows]);ax.invert_yaxis();ax.axvline(0,color=ink,lw=.7)
        ax.set_ylabel('Channel index');ax.set_xlabel('Output effect removed (logit contrast)');ax.set_title('First ReLU channel restore' if layer==0 else 'Final ReLU channel restore',loc='left',fontweight='bold')
        ax.grid(axis='x',color='#E4E7EB',lw=.6);ax.set_axisbelow(True)
        for y,value in enumerate(values):ax.text(value+(0.012 if value>=0 else -.012),y,f'{value:+.3f}',va='center',ha='left' if value>=0 else 'right',fontsize=9)
        ax.margins(x=.22)
    fig.suptitle('Fixed example: teacher 292, case 70, mask 15, C = (7, 9)',fontsize=12)
    fig.text(.08,.015,'Each bar is a separate channel intervention. First-layer bars are not additive. Independent horizontal scales.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96));save(fig,'02_example_channels')
    record=ex['record']['record'];loc=load(RUN/f'discovery/record_{record:03d}/data.npz');cm=LinearSegmentedColormap.from_list('signed_effect',[gold,'#FFFFFF',blue])
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.8))
    arrays=[loc['point_effect'][1,0].reshape(32,32,32)[2],loc['point_effect'][1,1].reshape(32,32,32)[26]]
    for i,(ax,array) in enumerate(zip(axes,arrays)):
        limit=float(abs(array).max());im=ax.imshow(array,cmap=cm,vmin=-limit,vmax=limit,origin='upper',interpolation='nearest')
        ax.set_title('First ReLU: channel 2' if i==0 else 'Final ReLU: channel 26',loc='left',fontweight='bold');ax.set_xlabel('x coordinate');ax.set_ylabel('y coordinate');ax.set_xticks([0,8,16,24,31]);ax.set_yticks([0,8,16,24,31])
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Single-position effect removed')
    axes[0].scatter([9,17],[13,11],marker='x',s=50,c='#000000');axes[0].text(9,14.8,'Top point',fontsize=8,color=ink,ha='center');axes[0].text(18,10,'Original j',fontsize=8,color=ink)
    fig.suptitle('Spatial output effects in the fixed example / new outside context',fontsize=12)
    fig.text(.08,.01,'Indices start at 0. Independent color scales. Marks: strongest first point (2,13,9), original j (2,11,17).',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94));save(fig,'03_example_positions')
    order=[61,0,4,44,45,5,62];labels=['Original j','Top 1 position','Top 256 positions','Top 4 channels','Top 8 channels','Top 1024 positions','All changed first positions']
    rows=[next(r for r in ex['blocks'] if r['context']==1 and r['layer']==0 and r['command']==i) for i in order]
    pos=np.array([r['residual_positive'] for r in rows]);neg=np.array([r['residual_negative'] for r in rows]);net=pos+neg;yy=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(10.5,5.7));ax.barh(yy,pos,color=blue,height=.6,label='Positive terminal residual');ax.barh(yy,neg,color=gold,height=.6,label='Negative terminal residual');ax.scatter(net,yy,c=ink,marker='D',s=28,label='Net residual',zorder=4)
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.axvline(0,color=ink,lw=.7);ax.set_xlabel('Remaining output contrast relative to C-off')
    ax.set_title('A small net residual can hide opposing remaining effects',loc='left',pad=50,fontweight='bold');ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=9)
    ax.grid(axis='x',color='#E4E7EB',lw=.6);ax.set_axisbelow(True)
    fig.text(.1,.01,'Fixed example; first-layer replacements with native skip. Point and channel budgets are different intervention costs.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'04_residual_cancellation')
    return paths


def report():
    """以原固定例和全部主队列为主，明确幅度变化、开关变化与残差支路。"""
    assert not (RUN/'delivery_manifest.json').exists();claims=teacher.read(RUN/'claim_checks.json');ex=teacher.read(RUN/'examples.json');audit=teacher.read(RUN/'independent_audit.json');analysis=teacher.read(RUN/'analysis.json');assert audit['status']=='PASS'
    blocks=read_table('blocking');points=read_table('point_candidates');summaries=read_table('teacher_summary');reschannels=read_table('residual_channels')
    figs=figures(claims,ex,summaries)
    get=lambda layer,index:next(r for r in ex['blocks'] if r['context']==1 and r['layer']==layer and r['command']==index)
    agg=lambda layer,index:next(r for r in claims['teacher_equal'] if r['context']==1 and r['layer']==layer and r['command']==index)
    pair=ex['pairs'][1];top0=sorted([r for r in ex['points'] if r['context']==1 and r['layer']==0],key=lambda r:-abs(r['single_removed']))
    topchannels=[sorted([r for r in ex['channels'] if r['context']==1 and r['layer']==layer],key=lambda r:-abs(r['whole_channel_removed'])) for layer in (0,1)]
    stable_counts=[sum(r['switch']=='False' for r in points if r['primary']=='True' and r['context']=='1' and int(r['layer'])==layer and r['absolute_rank']=='1') for layer in (0,1)]
    example_rows=[]
    for index,label in ((61,'原j单点'),(0,'排名第一单点'),(3,'前64点'),(4,'前256点'),(5,'前1024点'),(44,'前4通道'),(45,'前8通道'),(62,'全部第一层变化')):
        r=get(0,index);example_rows.append(f"| {label} | {r['removed']:+.6f} | {r['remaining']:+.6f} | {r['remaining_without_skip']:+.6f} |")
    meanrows=[]
    for index,label in ((61,'原j'),(0,'排名第一单点'),(3,'最多64点'),(4,'最多256点'),(5,'最多1024点'),(44,'最多4通道'),(45,'最多8通道'),(63,'全部翻转位置'),(64,'全部未翻转但改变幅度的位置'),(62,'全部激活变化')):
        r=agg(0,index);meanrows.append(f"| {label} | {100*r['remaining_rms_ratio']:.2f}% | {r['spent_min']}–{r['spent_max']} |")
    pointrows='\n'.join(f"| ({r['channel']},{r['y']},{r['x']}) | {r['single_removed']:+.6f} | {'翻转' if r['switch'] else '保持开启、幅度改变'} |" for r in top0[:5])
    # 留下最大余量的具体末层通道；这是固定预算后的读出贡献，不是新增选点。
    rc=[r for r in reschannels if int(r['record'])==ex['record']['record'] and r['context']=='1' and r['layer']=='0' and r['command']=='4']
    pos=sorted([r for r in rc if float(r['residual_contribution'])>0],key=lambda r:-float(r['residual_contribution']))[:4]
    neg=sorted([r for r in rc if float(r['residual_contribution'])<0],key=lambda r:float(r['residual_contribution']))[:4]
    channel_detail='；'.join(f"通道{r['channel']}={float(r['residual_contribution']):+.6f}" for r in pos+neg)
    residual_example=next(r for r in ex['residual_points'] if r['context']==1 and r['layer']==0 and r['command']==4 and r['rank']==1)
    checks=dict(status='PASS',utc=teacher.now(),new_context_top_point_stable_counts=stable_counts,primary26=26,figures=figs,example_record=ex['record']['record'],residual_channel_example=pos+neg)
    teacher.dump(RUN/'report_checks.json',checks)
    text=f'''# dSprites：原组合的下游通道、ReLU位置与联合阻断

2026年10月25日。沿用原36图与六个同架构冻结模型，不增加输入或训练。保留上轮全部26条严格恢复记录（15状态、5个有成功的模型），另加4个仅分解对照、每模型首个非空失败源干预共6条、原参照依赖失败例1条，合计37条案例×方法记录。没有把26条当独立样本。

**原组合的输出作用分布在多个通道和空间位置；只盯开关翻转会遗漏大量保持开启的位置上的幅度传递。联合阻断能显著减少效应，但残差支路和正负抵消使“小净余量”不能直接证明路径已消失。** 以下结果是同面板输出指导定位，不是未见输入确认，也不声称最小或唯一电路。

## 1. 阻断的对象与参照

对旧/新两种组外背景，分别比较C-off→C-on（00→10、01→11）。本报告主表使用新组外背景，原固定例的组合效应为b={pair['C_effect']:+.6f}，不是完整上游干预效应3.371569。这里C效应是整组通道变化的条件输出差，包含可加部分，不等同于纯高阶交互项。

“阻断”指把选定下游位置的激活替换为**同组外背景下C-off的实测值**，不是一律置零。第一ReLU替换后，卷积、第二ReLU和读出真实重算。另做“只保留选定变化”的恢复，并分别保留或恢复残差支路。末层ReLU后只剩GAP和线性读出，因此末层阻断可加是结构恒等式，不作为新的协同发现。

每层32768点。先精确计算单点输出差并按绝对值排序，整通道用实际整通道阻断差排序；全部37条、两背景、两层的名单先封存，再计算联合阻断。点预算1/4/16/64/256/1024，通道预算1/2/4/8/16/32；候选不足则截断。两组随机名单来自相同变化位置或通道池，同预算比较。翻转、稳定、支持、反向名单另有自己的候选池。第一/末层的相同通道编号不是同一个节点，跨模型编号也不作功能对齐。

## 2. 哪些下游通道与位置有作用

原固定例：教师292、案例70、mask15，源通道7的既有回退恢复中间C=(7,9)。在block3第一ReLU后，效应最大的四个整通道为**{', '.join(str(r['channel']) for r in topchannels[0][:4])}**；分别阻断造成的输出差为{', '.join(f"{r['whole_channel_removed']:+.6f}" for r in topchannels[0][:4])}。它们联合阻断的输出差为{get(0,44)['removed']:+.6f}，不能把单通道效应简单相加。

末层ReLU读出贡献最大的通道为**{', '.join(str(r['channel']) for r in topchannels[1][:4])}**。这确定了两个层级的重要单元，但尚未通过逐边干预证明某个第一层通道唯一对应某个末层通道。

| 第一ReLU位置(channel,y,x)，0起始 | 单点阻断移除的效应 | 原配对中的开关状态 |
|---|---:|---|
{pointrows}

原j=(2,11,17)移除效应仅+0.004414；最大单点(2,13,9)为+0.054351，仍远小于整个组合的+2.920629。主26条中，第一层最大单点有{stable_counts[0]}/26处于保持开启状态；末层最大单点为{stable_counts[1]}/26。排名大的节点不必发生开/关翻转。

![具体通道](figures/dsprites_downstream_routes_20261025/02_example_channels.png)

![具体空间位置](figures/dsprites_downstream_routes_20261025/03_example_positions.png)

## 3. 单点与联合阻断的差别

固定例，保留原残差支路时：

| 阻断变化 | 移除的组合效应 | 剩余效应 | 同时恢复残差支路后的余量 |
|---|---:|---:|---:|
{chr(10).join(example_rows)}

前256点的单独阻断效应之和为{get(0,4)['sum_single_removed']:+.6f}，实际联合效应为{get(0,4)['removed']:+.6f}，差{get(0,4)['joint_minus_singles']:+.6f}。第一ReLU之后还存在第二ReLU，多点共同改变其响应，故单点之和不等于联合效应。主26条的前256点均出现大于1e−8的非加和差。只保留该256点并恢复残差支路，得到+{get(0,4)['keep_without_skip']:.6f}，也保留了恢复方向的证据。

全部26条主记录的剩余量级如下：每教师先计算剩余效应RMS/原C效应RMS，再对5个非空教师等权平均。这里不是“每案例解释百分比”，也不是移除量与余量可相加的百分比；残差可能反号或大于原量。

| 第一ReLU阻断 | 剩余RMS比例 | 实际位置/通道预算范围 |
|---|---:|---:|
{chr(10).join(meanrows)}

最多256点的两随机对照仍剩{100*agg(0,10)['remaining_rms_ratio']:.2f}%/{100*agg(0,16)['remaining_rms_ratio']:.2f}%；最多4通道随机对照剩{100*agg(0,50)['remaining_rms_ratio']:.2f}%/{100*agg(0,56)['remaining_rms_ratio']:.2f}%。这些对照支持定位在当前队列内捕捉了输出相关位置，但输出排序与测试使用相同案例，不能视为独立验证。整通道和单点覆盖的空间数量不同，不能把二者当相同成本比较。

![预算与余量](figures/dsprites_downstream_routes_20261025/01_blocking_budget.png)

## 4. 剩余效应来自哪里

残差块有卷积分支u和直接支路H。对每个C配对进一步交叉u-off/u-on与H-off/H-on。固定例：卷积分支单独带来{pair['convolution_only']:+.6f}，直接支路单独带来{pair['skip_only']:+.6f}，两路交互{pair['branch_interaction']:+.6f}，合成{pair['C_effect']:+.6f}。

第一层全部变化被阻断后，该例仍有{get(0,62)['remaining']:+.6f}，与直接支路单独效应完全一致；同时恢复该支路，余量为0。26条主记录都保留非零直接支路效应。两条记录的卷积单独与直接支路单独作用反向。全量阻断是实现/路径覆盖控制，不能称为稀疏电路发现。

前256点阻断后剩{get(0,4)['remaining']:+.6f}：其中直接支路单独为{pair['skip_only']:+.6f}，未阻断卷积变化在此支路背景下的条件作用为{get(0,4)['remaining_convolution_given_skip']:+.6f}。再看末层读出，正余量合计{get(0,4)['residual_positive']:+.6f}，负余量合计{get(0,4)['residual_negative']:+.6f}。具体主要通道：{channel_detail}。最大绝对余量位置为末层({residual_example['channel']},{residual_example['y']},{residual_example['x']})，读出贡献{residual_example['residual_contribution']:+.6f}。完整32通道和前128个余量位置见证据表。

前8个第一层通道阻断后净余量仅{get(0,45)['remaining']:+.6f}，但正余量仍有{get(0,45)['residual_positive']:+.6f}、负余量{get(0,45)['residual_negative']:+.6f}。因此净输出接近0有一部分来自抵消，不能只看净值宣称剩余路径已被删除。

![联合阻断后的正负余量](figures/dsprites_downstream_routes_20261025/04_residual_cancellation.png)

## 5. 开关翻转与保持开启的幅度变化

固定例第一层有{get(0,63)['spent_units']}个发生翻转的变化位置、{get(0,64)['spent_units']}个未翻转但幅度改变的位置。恢复全部翻转位置后仍剩{get(0,63)['remaining']:+.6f}；恢复全部稳定开启位置后剩{get(0,64)['remaining']:+.6f}。两个集合大小不同，这不是等预算算法比较，而是检查“只解释翻转是否足够”。

主队列恢复全部第一层翻转后仍剩{100*agg(0,63)['remaining_rms_ratio']:.2f}%的RMS量级；恢复稳定开启位置后剩{100*agg(0,64)['remaining_rms_ratio']:.2f}%。末层相应为{100*agg(1,63)['remaining_rms_ratio']:.2f}%和{100*agg(1,64)['remaining_rms_ratio']:.2f}%。这要求修正研究表述：联合开关解释了局部阈值条件，但本队列的输出传递还大量依赖原本开启位置的幅度变化及残差支路，不能把全部输出联合效应归于少数开关。

## 6. 审计、范围与文件

全部37条完成两背景、两层定位与阻断；4条仅分解对照的C效应仍严格为0。保存每层2,424,832个单点分数，通过另一种输出位移累加算法全量复核；另对{audit['sampled_original_single_points']}个选中前16点/边界/原j做完整原forward抽查，不能把抽查说成全部单点原forward枚举。所有{audit['first_joint_fields']}个第一层联合/恢复场、{audit['last_joint_fields']}个末层场和{audit['original_channel_blocks']}个整通道定位场均经独立实现核验。最大误差{max(audit['maximum_errors'].values()):.3g}。十三份历史交付1711条目与更早来源未改。

结论限同面板的探索性定位、匹配C-off参照和指定干预。激活混合可能离开自然输入流形；未见输入泛化、跨架构、电路最小性、唯一因果份额和分类收益均未建立。下一步如继续，可固定本轮名单，检验背景变化时这些路径是否保持作用，并针对多个第一层通道到末层通道的具体边做路径干预；本轮未执行这些新验证。

- [数据与复现入口](../runs/dsprites_downstream_routes_20261025/README.md)
- [全部阻断与恢复](../runs/dsprites_downstream_routes_20261025/tables/blocking.csv)、[残差两路径](../runs/dsprites_downstream_routes_20261025/tables/pairs.csv)
- [具体候选位置](../runs/dsprites_downstream_routes_20261025/tables/point_candidates.csv)、[全部通道定位](../runs/dsprites_downstream_routes_20261025/tables/channel_localization.csv)
- [阻断后余量通道](../runs/dsprites_downstream_routes_20261025/tables/residual_channels.csv)、[余量位置](../runs/dsprites_downstream_routes_20261025/tables/residual_points.csv)
'''
    REPORT.write_text(text,encoding='utf-8')
    (RUN/'README.md').write_text('''# 下游通道与位置阻断：数据、参照与复现

[中文报告](../../reports/dSprites_下游通道位置与联合阻断_2026-10-25.md)。沿用36图/6模型：主26案例×方法记录、4仅分解、6各教师首个非空失败源干预、1原参照依赖例，共37条。完整元数据在protocol.json的cohort，未训练或扩输入。

## 参照与含义

context0=旧组外C00→10；context1=新组外C01→11。阻断是将选中激活换成同背景C-off值；keep只在C-off上保留选中变化。第一ReLU用原第二ReLU重算，并分别取C-off/C-on的直接残差支路。末层ReLU之后读出线性，联合效应可加是结构事实。两背景各自定位，尚未做跨背景固定名单验证。

第一ReLU点排名用真实单点输出差的绝对值，整通道排名用整通道干预而非单点和。输出指导的同面板定位，不是未见输入验证。两随机与absolute用相同变化点/通道池及预算；switch/stable/supporting/opposing为各自子池。激活变化阈值1e−10，numerical tolerance1e−9。跨模型编号不作功能对齐。

## 轴与文件

discovery/record_000..036/data.npz：

- point_effect、delta_activation、switched：context2×layer2×32768；layer0第一ReLU，layer1最后ReLU。point=c*1024+y*32+x，0起始。
- channel_effect：context2×layer2×channel32。第一层通道效应通常不可加，末层线性可加。
- orders：context2×layer2×order7×1024，−1填充；order为absolute/random0/random1/switch_only/stable_only/supporting/opposing；order_counts记录各子池完整数量。
- channel_orders：context2×layer2×3名单×32；channel_counts相应数量。direction为每背景C效应符号，零用+1约定，仅分解控制无方向性解释。

evaluation/record_*/data.npz：

- native_logits：context2×[C-off,C-on]×class3。
- route_logits：context2×firstactivation-off/on2×skip-off/on2×class3；route_channels相同前3轴×末层32通道，为相对两路off的读出差。
- first_logits：context2×command67×skip2×operation2×class3；operation0=block，1=keep。first_channels末轴32，值为相对C-off整体输出的末层通道余量；first_pn末轴[空间正余量,空间负余量]。first_second_flips是相对对应C-off完整第二ReLU的翻转数。
- last_logits：context2×command67×operation2×class3；last_channels/last_pn同义，无skip轴。
- residual_top_ids/values：context2×layer2×[absolute point256,absolute channel4]×128，是原生skip阻断后末层读出余量绝对值最大的128位置。排序只用于描述余量，不进入主阻断选点。

command0..41=7点名单×预算1/4/16/64/256/1024；42..59=3通道名单×1/2/4/8/16/32；60..66=empty/original_j/all_active/all_switch/all_stable/all_supporting/all_opposing。末层original_j无对应定位含义，使用空控制。预算不足截断，数量在blocking表明确保存；special预算随案例变化。

## 证据表

pairs记录卷积/直接支路四状态。blocking记录全部9916种组合，比较单点和、真实联合、选定集合与补集交互、残差支路及读出正负余量。point_candidates是五个定位名单各前64的并集及原j，完整32768地图在NPZ。channel_localization保留全部32通道。residual_channels和residual_points只针对事前指定point256/channel4预算。teacher_summary按教师计算RMS和比值；主成功5教师、空教师留空，控制另列。RMS余量比例不是唯一因果份额，可能大于1或净值反号。

## 注释脚本及复现

../../scripts下八份新脚本：downstream_route_mechanism.py、run_dsprites_downstream_routes.py、test_dsprites_downstream_routes.py、audit_dsprites_downstream_routes.py、analyze_dsprites_downstream_routes.py、report_dsprites_downstream_routes.py、verify_dsprites_downstream_routes_delivery.py、finalize_dsprites_downstream_routes.py。

已完成目录只运行verify脚本的--deep-inputs只读检查。全流程需新命名运行路径，先test，再run --stage init → discover → evaluate，再audit → analyze → report → verify --before-manifest → finalize → verify --deep-inputs。Python为相邻medical_ai_reliability/.venv-gpu/Scripts/python.exe，参数-B -X utf8。

协议、输入SHA、所有定位封存、原模型独立审计、逐行检查、图PNG/SVG与当前进度快照一并保存；旧报告与源码保持不可变。
''',encoding='utf-8')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':report()

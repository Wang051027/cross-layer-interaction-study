"""生成中文四状态研究报告与可导出的科学图；所有数字来自已保存逐行结果。"""
from __future__ import annotations
import csv
import json
import os
from collections import Counter
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_coalition_context import BASE,RUN

REPORT=BASE/'reports/dSprites_组合与组外背景的四状态输出验证_2026-10-25.md'
FIGDIR=BASE/'reports/figures/dsprites_coalition_context_20261025'
LABELS={'source':'源通道','middle':'中间通道','first_activation':'第一ReLU后激活','second_activation':'第二ReLU后激活'}


def read_table(name):
    """只读已经生成的完整证据表。"""
    with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def ident(row):
    """图上保留教师末三位、案例、mask和方法，便于反查。"""
    short={'source':'S','middle':'M','first_activation':'R1','second_activation':'R2'}
    return f"{str(row['seed'])[-3:]}/{row['case']}/{row['mask']}/{short[row['family']]}"


def figures(primary,example):
    """沿用研究项目的独立PNG/SVG交付；不把比较中的分量大小画成因果份额。"""
    assert not (RUN/'delivery_manifest.json').exists();FIGDIR.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            p=FIGDIR/f'{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');paths.append(str(p.relative_to(BASE)).replace('\\','/'))
        plt.close(fig)
    blue,gold,ink='#285F8F','#BC7524','#262A30'
    y=np.arange(len(primary));a=np.array([float(r['C_before']) for r in primary]);b=np.array([float(r['C_after']) for r in primary]);t=np.array([float(r['total']) for r in primary])
    fig,ax=plt.subplots(figsize=(10,10))
    for yy,aa,bb in zip(y,a,b):ax.plot([aa,bb],[yy,yy],color='#A6ADB5',lw=1.4,zorder=1)
    ax.scatter(a,y,facecolors='white',edgecolors=blue,s=40,label='C effect with outside BEFORE',zorder=3)
    ax.scatter(b,y,c=gold,s=25,label='C effect with outside AFTER',zorder=4)
    ax.scatter(t,y,c=ink,s=16,marker='D',label='Whole intervention effect',zorder=2)
    ax.axvline(0,color='#767D87',lw=.8);ax.set_yticks(y,[ident(r) for r in primary]);ax.invert_yaxis()
    ax.set_xlabel('Donor minus recipient logit contrast: change');ax.set_ylabel('Teacher / case / mask / method')
    ax.set_title('Original C effect in two outside contexts',loc='left',pad=55,fontweight='bold')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=1,fontsize=9)
    ax.grid(axis='x',color='#E6E8EC',lw=.6);ax.set_axisbelow(True)
    fig.text(.12,.01,'26 fixed case-method records; 15 distinct states sharing 36 images. S: source, M: middle, R2: second activation.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));save(fig,'01_conditional_C_effects')
    states=example['states'];delta=np.array([s['F']-states[0]['F'] for s in states]);z=np.array([s['point_z'] for s in states]);names=['00: before','10: C only','01: outside only','11: both']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,values,title,unit in zip(axes,(delta,z),('Final output change','Frozen point j preactivation'),('Logit contrast change','Preactivation')):
        bars=ax.bar(np.arange(4),values,color=blue,width=.58);ax.axhline(0,color=ink,lw=.8)
        ax.set_xticks(np.arange(4),names,rotation=20,ha='right');ax.set_ylabel(unit);ax.set_title(title,loc='left',fontweight='bold')
        span=max(values.max()-values.min(),.1)
        for bar,val in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,val+(.035*span if val>=0 else -.045*span),f'{val:+.4f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=9)
        ax.margins(y=.22);ax.grid(axis='y',color='#E6E8EC',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Previously fixed example: teacher 292, case 70, mask 15, source intervention',fontsize=12,y=1.02)
    fig.text(.09,-.025,'Separate y-axis units. Outside changes lower the local j response while increasing C\'s final output effect.',fontsize=9)
    fig.tight_layout();save(fig,'02_fixed_example_four_states')
    active=[r for r in primary if float(r['outside_max_change'])>1e-10];yy=np.arange(len(active))
    first=np.array([float(r['interaction_first_relu']) for r in active]);second=np.array([float(r['interaction_second_relu']) for r in active]);net=np.array([float(r['interaction']) for r in active])
    fig,ax=plt.subplots(figsize=(10,6.7))
    ax.barh(yy,first,color=blue,height=.55,label='First ReLU remainder propagated')
    offset=np.where(first*second>0,first,0.)
    ax.barh(yy,second,left=offset,color=gold,height=.55,label='Second ReLU remainder')
    ax.scatter(net,yy,c=ink,s=24,marker='D',label='Net output interaction J',zorder=3)
    ax.axvline(0,color='#767D87',lw=.8);ax.set_yticks(yy,[ident(r) for r in active]);ax.invert_yaxis();ax.set_xlabel('Four-state interaction in logit contrast')
    ax.set_title('ReLU contributions to C–outside interaction',loc='left',pad=48,fontweight='bold')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=9,ncol=1)
    ax.grid(axis='x',color='#E6E8EC',lw=.6);ax.set_axisbelow(True)
    fig.text(.12,.01,'Outside channels changed in 14 records. Terms had opposing signs in 7 records.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'03_relu_interaction_cancellation')
    return paths


def report():
    """核验完成后写正文、轴说明与报告补充计算；封存后禁止重写。"""
    assert not (RUN/'delivery_manifest.json').exists();audit=teacher.read(RUN/'independent_audit.json');assert audit['status']=='PASS'
    claims=teacher.read(RUN/'claim_checks.json');analysis=teacher.read(RUN/'analysis.json');examples=teacher.read(RUN/'examples.json')
    rows=read_table('effects');primary=[r for r in rows if r['primary_success']=='True'];states=read_table('states');members=read_table('member_reversions')
    example=next(e for e in examples if e['effect']['seed']==202609292);er=example['effect'];active=[r for r in primary if float(r['outside_max_change'])>1e-10]
    oppose=[r for r in active if float(r['interaction_first_relu'])*float(r['interaction_second_relu'])<0 and abs(float(r['interaction_first_relu']))>1e-8 and abs(float(r['interaction_second_relu']))>1e-8]
    offset=[r for r in primary if r['standalone_offset']=='True'];first_opposed=oppose[0];first_offset=offset[0]
    assert len(active)==14 and len(oppose)==7 and len(offset)==2
    saved=figures(primary,example)
    supplements=dict(status='PASS',utc=teacher.now(),active_outside_records=len(active),opposed_relu_interactions=len(oppose),first_opposed_key={k:first_opposed[k] for k in ('seed','case','mask','endpoint')},first_offset_key={k:first_offset[k] for k in ('seed','case','mask','endpoint')},extra_examples='first byseed/case/mask/endpoint within computed category;posthoc explanatory,not validationselection',figures=saved)
    teacher.dump(RUN/'report_checks.json',supplements)
    table=[]
    for row in claims['teacher_equal']:
        if row['count']==0:continue
        values=[row[k+'_rms'] for k in ('C_before','C_after','outside_before','interaction','total')]
        table.append(f"| {LABELS[row['family']]} | {row['count']} / {row['models']} | "+' | '.join(f'{x:.4f}' for x in values)+' |')
    pointtable=[]
    for row in claims['teacher_equal']:
        if not row['count']:continue
        pointtable.append(f"| {LABELS[row['family']]} | {row['C_after_rms']:.6f} | {row['C_after_j_before_control_rms']:.6f} | {row['C_after_j_after_control_rms']:.6f} |")
    statetable='\n'.join(f"| {s['state']} | {s['F']:.6f} | {s['F']-example['states'][0]['F']:+.6f} | {s['point_z']:+.6f} |" for s in example['states'])
    maxerr=max(audit['maximum_errors'].values());tables=analysis['table_rows']
    text=f'''# dSprites：原组合与组外背景的四状态输出验证

2026年10月25日。沿用36幅图、72个共享这些图的有向配对、六个同架构冻结模型及源mask3/15。C、j、干预单元和强度取自上一轮冻结名单。本轮已完成全部四状态、原模型独立审计与输出分解。

**原C的激活变化确实能造成输出变化，组外背景既能增强也能削弱这种作用；被定位的单个联合开关只解释其中一部分。** 上轮26条严格恢复记录（15个状态）全部保留。14条同时改变C和组外，8条增强C的输出效应、6条削弱；其余12条组外没有变化，交互为结构性零，不证明一般独立。

## 1. 四个状态与可检验问题

H00是原完整强度状态，H11是上轮冻结干预后的实际中间层状态。C是原通道组合，O是其余通道。交换的是整张通道图，H10、H01是内部混合反事实，未必由自然输入产生。

| 状态 | C | O | 问题 |
|---|---|---|---|
| 00 | 干预前 | 干预前 | 原输出 |
| 10 | 干预后 | 干预前 | 仅改变C有什么影响 |
| 01 | 干预前 | 干预后 | 仅改变组外有什么影响 |
| 11 | 干预后 | 干预后 | 完整干预输出 |

F仍是供体类别logit减受体类别logit，三类原始logit全部保存。定义：

`a=F10−F00`，`o=F01−F00`，`J=F11−F10−F01+F00`。

`ΔF=a+o+J`；组外干预后C的条件作用为`b=F11−F01=a+J`。反向恢复C到原值移除的正是b。输出中的交互J来自真实非线性前向，不是把整个网络线性化。此处不把J任意分成两半，也不把a/ΔF当作唯一因果比例。

本轮所有四状态只用H00/H11，不使用重新构建的AI/RI。若实际H未变，四状态严格相同，因此无法靠修改分解制造效应。全部864状态、16个已选干预端点都运行：12个aligned端点（四族×定向/两随机），另加4个moving-A定向对照，共13824条记录。539状态仍是原失败队列。

## 2. 组合究竟带来多少输出变化

以下是主26条记录的量级。先在每个教师内计算该类成功记录的RMS，再对有成功记录的教师等权平均；空教师不记零。**RMS不能相加，下表不是贡献百分比。** 所有量均为logit对比变化。

| 干预 | 记录 / 有成功模型 | C作用a：旧组外 | C作用b：新组外 | 组外单独o | 交互J | 完整ΔF |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

源通道干预下，组外单独作用的RMS均值大于C单独作用；这不意味着每个案例都由组外主导。中间通道的11条成功记录都只改变C内激活，所以o和J严格为零。第二激活的 2 条恢复记录均来自同一模型。第一激活无上轮严格成功，保持缺失。

![C的条件输出效应](figures/dsprites_coalition_context_20261025/01_conditional_C_effects.png)

26条中18条j真实翻转，另8条开关原来已在目标侧。18条翻转中，只改变C都能将j推到目标侧；其中2条只改组外也能做到。**组合成员对旧A1参照的必要性、成员数值变化对恢复的必要性、对最终输出的必要性，是不同命题。** 把每个C成员恢复成干预前实际值后，完整11背景中的58个成员检验有25个使j退回相反侧；不能把上轮的“每个成员必要”直接解释为“每个成员都必须改变”。

## 3. 预先固定的恢复例：输出增强与局部抑制并存

教师292、案例70、mask15，原C=(7,9)，源通道7回退λ=0.89453125；这是沿用上轮固定示例，没有依据本轮输出挑选。下游j=(2,11,17)。

| 状态 | F | 相对00的变化 | j预激活 |
|---|---:|---:|---:|
{statetable}

这里，C单独带来 **+{er['C_before']:.6f}**；组外单独带来 **+{er['outside_before']:.6f}**；交互额外带来 **+{er['interaction']:.6f}**，合成完整变化 **+{er['total']:.6f}**。在新组外背景下恢复C原值，会移除 **+{er['C_after']:.6f}** 的输出效应。

但同一个组外变化把j从C单独时的+0.623797压到+0.187479，**削弱局部激活，却增强C对最终输出的作用**。这是整图C输出通路与单点j不能等同的直接例子。在旧组外背景，恢复通道9的原值后j仍为+0.387225；在新组外背景则变为−0.049094。组外背景同时改变了“通道9的变化是否必要”。

![预定示例四状态](figures/dsprites_coalition_context_20261025/02_fixed_example_four_states.png)

## 4. 增强和抵消发生在哪里

必须区分两类作用：J改变C的条件效应；o则是组外独立带来的输出变化。二者可以方向不同。按现有结果编号取首个独立抵消说明例（事后解释，不用于估计成功率）：教师{str(first_offset['seed'])[-3:]}、案例{first_offset['case']}、mask{first_offset['mask']}，a={float(first_offset['C_before']):+.6f}，o={float(first_offset['outside_before']):+.6f}，J={float(first_offset['interaction']):+.6f}，总变化={float(first_offset['total']):+.6f}。组外增强C的条件作用，同时又通过自己的负向通路抵消总输出。共有2条主记录出现a与o反向。

相对H00的两层门控，精确展开输出变化为：仿射传递、第一ReLU余项经后层传递、第二ReLU余项。四状态的仿射交互为0，因此J恰等于两层非线性余项之和；这是一种有声明参照的代数解释，不是唯一中介份额。14条非零交互记录中，7条的两层项异号，存在层间抵消。

按编号取首个异号说明例（事后解释）：教师{str(first_opposed['seed'])[-3:]}、案例{first_opposed['case']}、mask{first_opposed['mask']}，第一层项={float(first_opposed['interaction_first_relu']):+.6f}、第二层项={float(first_opposed['interaction_second_relu']):+.6f}，净J={float(first_opposed['interaction']):+.6f}。只看第一层项会判断错净交互方向。原32个末层通道及空间位置的正负读出贡献均保存，支持核验抵消，而非只展示净值。

![两层交互抵消](figures/dsprites_coalition_context_20261025/03_relu_interaction_cancellation.png)

## 5. 单个联合开关解释多少

把原j的第一ReLU输出固定为00或11实测的非负值，分别重算所有四状态，第二ReLU自然执行。定义控制差为自然效应减固定j后的效应。这个操作检验该点激活变化在指定背景下的作用，未必只反映开/关二值变化；仍是可能离开自然输入流形的内部干预。

| 干预 | C作用b的RMS均值 | 固定j为00时b减少的效应RMS | 固定j为11时b减少的效应RMS |
|---|---:|---:|---:|
{chr(10).join(pointtable)}

预定恢复例完整ΔF为+3.371569，固定j为原00的0激活后仅减少+0.004414；固定为11激活的对比差为+0.005377。这支持“C确实影响输出，但大部分变化经其他位置或路径到达读出”。两种控制参照均保存，不选较大的值作为唯一解释比例。

## 6. 失败对照、可靠性与边界

原预定失败例教师291/案例1/mask3的moving-A源干预，a=−0.099199、o=−0.519688、J=+0.001840、ΔF=−0.617047。实际网络有输出变化，但上轮共同参照必要性仍未成立。仅分解对照教师291/案例55/mask15的四状态全部相同，a、o、J、ΔF都为0；全部4条仅分解对照均如此。失败、随机及无候选记录未删除，随机沿用上轮同候选池/最大预算搜索，实际最优预算可不同。

独立审计{audit['native_fields']}个原前向四状态场、{audit['point_patches']}个非负点替换场及{audit['member_reversions']}个逐成员恢复场；最大误差{maxerr:.3g}。审计独立解码旧名单、重建源干预和四状态，通过原forward与hook核验，不调用新数值核心。十二份旧交付1650条目及更早来源保持一致。5张CSV共{sum(tables.values())}行；全部端点原始数组保留。

结论仍限同一36图面板、已有自适应搜索解、同架构六个模型。26 条记录对应 15 个源状态。下一步若继续同面板，应定位C作用通往多个下游位置的路径，并比较路径联合阻断与单点阻断；本轮未做这项后续搜索。

- [数据、代码与复现入口](../runs/dsprites_coalition_context_20261025/README.md)
- [全部四状态效应](../runs/dsprites_coalition_context_20261025/tables/effects.csv)、[四状态原始读出](../runs/dsprites_coalition_context_20261025/tables/states.csv)
- [逐成员恢复](../runs/dsprites_coalition_context_20261025/tables/member_reversions.csv)、[主记录与预定对照的末层通道贡献](../runs/dsprites_coalition_context_20261025/tables/channels.csv)
- [逐教师统计](../runs/dsprites_coalition_context_20261025/tables/teacher_summary.csv)、[独立审计](../runs/dsprites_coalition_context_20261025/independent_audit.json)
'''
    REPORT.write_text(text,encoding='utf-8')
    readme='''# 四状态输出验证：复现与数组说明

沿用36图、72依赖配对、六冻结模型和mask3/15。主26条案例×方法（15个状态）及对照全部继承。中文正文：[研究报告](../../reports/dSprites_组合与组外背景的四状态输出验证_2026-10-25.md)。

## 四状态及效应

00为完整强度干预前H；11为上轮已冻结干预后的H；10仅替换原C整张通道图；01仅替换组外。四场不用A/R。a=F10−F00，o=F01−F00，J=F11−F10−F01+F00；总差=a+o+J。C新组外作用b=a+J。F为供体减受体类别logit，正负不等于准确率收益。

## 原始数组轴

evaluation/seed_*/data.npz前3轴：case72、mask2（3/15）、endpoint16。
端点顺序为上轮25..36（aligned，四家族×定向/两随机）再1/4/7/10（moving-A定向四族）；没有本轮重选。

- logits后续state4（00,10,01,11）、mode3（自然、j固定00值、j固定11值）、class3。
- point_z/point_u后续state4；natural_flips为state4×两层ReLU，均相对同端点00。
- patch_second_flips为state4×两种j固定值，相对对应自然场第二层门控。
- channel_terms为mode3×term6×末层channel32；term依次a、o、J、总差、b、组外新C作用。通道和等于对应logit对比变化。
- spatial_pn为mode3×term6×[空间正贡献、空间负贡献]，相加等于净效应。
- branch_terms为term6×[固定00门控的仿射传递、第一ReLU余项、第二ReLU余项]。三项和等于自然效应，J仿射项为0；不是唯一因果份额。
- shifts为[C整图最大变化、组外整图最大变化]。
- drop_logits/drop_z的轴不同：case72、mask2、四个aligned定向家族、两种组外背景（10/11）、原C名单位置32填充、class3（仅logits）。前k位置对应C成员，后面填0，非物理通道号。恢复值为实际H00，区别于上一轮固定A1的消融。

## 表格与筛选

effects保留全部13824条及六差分/控制差/余项/抵消。states保留55296四状态的三类logit。member_reversions保留全部定向四族而非仅成功。channels仅列主26条记录及两条预定对照，其他所有端点的通道贡献在NPZ。teacher_summary先按教师计算RMS；空集留空。分析群体原focus539，primary_success字段标识原26条严格定向成功。

增强/削弱指C条件作用的绝对量变化（J与a同号/异号），不是分类改善。a和o异号是另一种独立路径抵消。空间/通道抵消为1−|净量|/绝对贡献总量；零总量留空。无独立配对p值或唯一贡献百分比。

## 注释代码与运行顺序

数值核心coalition_context_mechanism.py；协调器run_dsprites_coalition_context.py；预检test_dsprites_coalition_context.py；独立审计audit_dsprites_coalition_context.py；表格analyze_dsprites_coalition_context.py；正文和图report_dsprites_coalition_context.py；校验verify_dsprites_coalition_context_delivery.py；封存finalize_dsprites_coalition_context.py。全部位于../../scripts，含中文说明。

已完成目录只运行只读校验，不能覆盖。完整复现需另建明确的新运行路径，并先固定协议再运行：test → run --stage init → run --stage evaluate → audit → analyze → report → verify --before-manifest → finalize → verify --deep-inputs。Python使用旁边medical_ai_reliability/.venv-gpu/Scripts/python.exe，参数-B -X utf8。

交付保留协议及源码SHA、旧输入摘要、模型别结果、独立审计、图PNG/SVG、逐行检查、当前进度日期快照。封存之后不改旧结果；仅根CURRENT_RESEARCH.md可在未来更新。
'''
    (RUN/'README.md').write_text(readme,encoding='utf-8');print(json.dumps(supplements),flush=True)


if __name__=='__main__':report()

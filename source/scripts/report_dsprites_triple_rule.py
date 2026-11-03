"""制作判别实验报告，复核逐例统计，并提供封存后的只读校验。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import shutil
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_triple_rule import (BASE,DEFAULT,SHAPE,STRENGTHS,atomic_json,
                                      manifest_entries,verify_protocol,verify_seal)
from analyze_dsprites_triple_rule import TASKS,PRIMARY_METHOD,csv_table,rms

REPORT=BASE/'reports/dSprites_留出图像条件的三组规则判别实验_2026-11-01.md'
CODE=('dsprites_triple_rule.py','run_dsprites_triple_rule.py','audit_dsprites_triple_rule.py',
      'analyze_dsprites_triple_rule.py','report_dsprites_triple_rule.py')
LABELS=('全中间层三阶响应','输出三阶差分','阻断四通道移除的作用','只保留四通道重现的作用')


def report_path(run):
    """新运行使用自己的报告名，避免覆盖此前封存的默认报告。"""
    return REPORT if run.resolve()==DEFAULT.resolve() else BASE/'reports'/f'{run.name}_三组规则判别实验.md'


def read_table(run,name):
    with (run/'tables'/f'{name}.csv').open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def check_tables(run):
    """从CSV的逐例原值独立恢复主表，包括教师等权、空分母和四项判据。"""
    analysis=teacher.read(run/'analysis.json')
    cases=read_table(run,'cases')
    patch=read_table(run,'patches')
    assert len(cases)==864 and len(patch)==4320
    case_key=lambda r:(int(r['seed']),float(r['strength']),int(r['case']),int(r['mask']))
    assert len({case_key(r) for r in cases})==864
    assert len({(case_key(r),r['patch']) for r in patch})==4320
    by_case=defaultdict(list)
    by_patch=defaultdict(list)
    for r in cases: by_case[case_key(r)[:2]].append(r)
    for r in patch: by_patch[(*case_key(r)[:2],r['patch'])].append(r)
    computed=defaultdict(list)
    n_checked=0
    for r in read_table(run,'teacher_scores'):
        seed,strength=int(r['seed']),float(r['strength'])
        task,method=r['target'],r['method']
        group=by_case[seed,strength]
        assert len(group)==72
        def values(rows,key):
            return np.array([float(row[key]) for row in rows])
        if task=='middle_third_all':
            ratio=float(np.sqrt(values(group,'middle_error_sq_'+method).sum()/values(group,'middle_third_squared_norm').sum()))
        elif task in ('output_third','output_total'):
            target=values(group,'third_true')
            err=values(group,'third_'+method)-target
            denominator=target if task=='output_third' else values(group,'F_true')-values(group,'F_empty')
            ratio=rms(err)/rms(denominator)
        elif task in ('block_removed','keep_recovered'):
            group=by_patch[seed,strength,method]
            target=values(group,'removed' if task=='block_removed' else 'keep')
            ratio=rms(values(group,'predicted')-target)/rms(target)
        else:
            continue
        assert abs(ratio-float(r['ratio']))<1e-10, (seed,task,method)
        computed[strength,task,method].append(ratio)
        n_checked+=1
    for entry in analysis['aggregates']:
        key=entry['strength'],entry['target'],entry['method']
        if key in computed:
            vals=computed[key]
            assert len(vals)==6 and abs(np.mean(vals)-entry['ratio'])<1e-10
            assert sum(v<1 for v in vals)==entry['improved']
    for t,m in zip(TASKS,PRIMARY_METHOD):
        vals=computed[1.,t,m]
        assert analysis['decisions'][t]==bool(np.mean(vals)<=0.95 and sum(v<1 for v in vals)>=4)
    assert analysis['all_links_pass']==all(analysis['decisions'].values())
    return n_checked


def retrospective(run):
    """只在主结果确定后，用10月13日的真实中间场投影诊断误差来源。"""
    cases=read_table(run,'cases')
    patches=read_table(run,'patches')
    source=BASE/'runs/dsprites_layer_transfer_20261013'
    hashes=manifest_entries(source)
    inputs={}
    rows=[]
    for seed in teacher.SEEDS:
        selected=teacher.read(run/'fit'/f'{seed}.json')['middle']
        for strength in STRENGTHS:
            p=source/f'evaluation/seed_{seed}/alpha_{strength:.1f}/data.npz'
            name=p.relative_to(BASE).as_posix()
            inputs[name]=hashes[name]
            assert teacher.sha(p)==inputs[name]
            with np.load(p,allow_pickle=False) as z:
                projections=z['baseline_mobius_channel_projection']
                channel_norm=z['middle_mobius_channel_norm']
            group=[r for r in cases if int(r['seed'])==seed and float(r['strength'])==strength]
            group_p=[r for r in patches if int(r['seed'])==seed and float(r['strength'])==strength and r['patch']=='chosen4']
            target=np.array([float(r['third_true']) for r in group])
            index=([int(r['case']) for r in group],[int(r['mask']) for r in group])
            proj=projections[index]
            size=channel_norm[index]**2
            np.testing.assert_allclose(size.sum(-1),[float(r['middle_third_squared_norm']) for r in group],rtol=1e-10,atol=1e-9)
            blocked=np.array([float(r['removed']) for r in group_p])
            kept=np.array([float(r['keep']) for r in group_p])
            opposed=(abs(blocked)>1e-9)&(abs(kept)>1e-9)&(blocked*kept<0)
            rows.append(dict(seed=seed,strength=strength,
                             measured_H4_J0_error_ratio=rms(proj[:,selected].sum(-1)-target)/rms(target),
                             measured_H32_J0_error_ratio=rms(proj.sum(-1)-target)/rms(target),
                             selected_H3_energy_fraction=float(size[:,selected].sum()/size.sum()),
                             block_keep_gap_rms=rms(blocked-kept),opposite_sign=int(opposed.sum()),cases=len(group)))
    csv_table(run/'tables/posthoc_diagnostic.csv',rows)
    stats={}
    for strength in STRENGTHS:
        group=[r for r in rows if r['strength']==strength]
        stats[str(strength)]={key:float(np.mean([r[key] for r in group])) for key in
                              ('measured_H4_J0_error_ratio','measured_H32_J0_error_ratio',
                               'selected_H3_energy_fraction','block_keep_gap_rms')}
        stats[str(strength)]['opposite_sign']=sum(r['opposite_sign'] for r in group)
    atomic_json(run/'posthoc.json',dict(utc=teacher.now(),used_for_selection=False,inputs=inputs,stats=stats))
    return stats


def draw(run):
    """教师内归一化误差和配对阻断读数直接来自证据表。"""
    # 桌面沙箱的系统Temp有不同的ACL，字体缓存放在工作区共用的tmp目录。
    cache=BASE.parents[1]/'tmp'/'dsprites_matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    with matplotlib.rc_context():
        import matplotlib.pyplot as plt
        scores=read_table(run,'teacher_scores')
        patches=read_table(run,'patches')
        fig,(left,right)=plt.subplots(1,2,figsize=(10.6,4.6),layout='constrained')
        colors=['#3477AA','#B65929','#47785B','#805E91','#A2274A','#61727D']
        for i,seed in enumerate(teacher.SEEDS):
            y=[float(next(r['ratio'] for r in scores if int(r['seed'])==seed and r['strength']=='1.0'
                          and r['target']==t and r['method']==m)) for t,m in zip(TASKS,PRIMARY_METHOD)]
            left.scatter(np.arange(4)+(i-2.5)*.055,y,color=colors[i],s=25,label=str(seed)[-3:])
            points=[r for r in patches if int(r['seed'])==seed and r['strength']=='1.0' and r['patch']=='chosen4']
            right.scatter([float(r['predicted']) for r in points],[float(r['removed']) for r in points],
                          color=colors[i],s=11,alpha=.60,edgecolors='none')
        left.axhline(1,color='#444444',ls='--',lw=.9)
        left.set_xticks(range(4),['Middle','Output','Block','Keep'])
        left.set_ylabel('RMS prediction error / RMS true effect')
        left.set_title('Full-strength error by model')
        left.legend(title='Model seed (suffix)',ncols=3,fontsize=8,loc='upper left')
        left.grid(axis='y',alpha=.2)
        xlim=right.get_xlim()
        ylim=right.get_ylim()
        lo=min(xlim[0],ylim[0]);hi=max(xlim[1],ylim[1])
        right.plot([lo,hi],[lo,hi],lw=.9,ls='--',color='#444444')
        right.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Predicted removed interaction (logit contrast)',
                  ylabel='Measured removed interaction',title='Four-channel block on held-out cells')
        right.set_aspect('equal',adjustable='box')
        fig.savefig(run/'held_condition_results.png',dpi=170,facecolor='white')
        plt.close(fig)


def report(run):
    analysis=teacher.read(run/'analysis.json')
    audit=teacher.read(run/'independent_audit.json')
    assert audit['status']=='PASS'
    check_tables(run)
    post=retrospective(run)['1.0']
    draw(run)
    ag={(r['strength'],r['target'],r['method']):r for r in analysis['aggregates']}
    def ratio(t,m,s=1.):
        return ag[s,t,m]['ratio']
    def truth_rms(t,m):
        return ag[1.,t,m]['teacher_mean_truth_rms']
    def signs(t,m):
        r=ag[1.,t,m]
        return f"{r['sign_hits']}/{r['sign_eligible']}"
    fmt=lambda x:f'{x:.3f}'
    main_table=[]
    for label,t,m in zip(LABELS,TASKS,PRIMARY_METHOD):
        a=ag[1.,t,m]
        main_table.append(f"| {label} | {a['ratio']:.4f} | {a['improved']}/6 | {ratio(t,m,.5):.4f} | {'通过' if analysis['decisions'][t] else '未通过'} |")
    controls=[]
    labels={'quadratic':'真实二阶基线','rule':'冻结的阈值与传播规则','ridge':'相同低阶信息的岭回归',
            'threshold_only':'仅校正第二个阈值','carry_only':'传播第一处修正、固定第二处开关','class_mean':'开发格的类别转换×三组编号均值'}
    for method,label in labels.items():
        mid=fmt(ratio('middle_third_all',method)) if method!='class_mean' else '—'
        controls.append(f"| {label} | {mid} | {ratio('output_third',method):.3f} | {100*ratio('output_total',method):.2f}% |")
    models=[]
    individual=read_table(run,'teacher_scores')
    for seed in teacher.SEEDS:
        fit=teacher.read(run/'fit'/f'{seed}.json')
        values=[float(next(r['ratio'] for r in individual if int(r['seed'])==seed and float(r['strength'])==1.
                           and r['target']==t and r['method']==m)) for t,m in zip(TASKS,PRIMARY_METHOD)]
        models.append('| '+str(seed)+' | '+', '.join(map(str,fit['first']))+' | '+', '.join(map(str,fit['middle']))+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    strata=read_table(run,'strata')
    strong=[r for r in strata if r['subset']=='strong' and r['strength']=='1.0']
    weak=[r for r in strata if r['subset']=='weak' and r['strength']=='1.0']
    strong_value=np.array([float(r['removed_rms']) for r in strong])
    weak_value=np.array([float(r['removed_rms']) for r in weak])
    sample=analysis['first_example']; sample_patch=analysis['example_block']
    folder='../runs/'+run.name
    text=f"""# dSprites 留出图像条件的三组规则判别实验

2026年10月31日冻结实验并计算预测，11月1日完成报告。冻结的四通道规则在六个模型上都改善了中间层三阶响应的预测。它对最终输出、四通道阻断和只保留的预报均未达到事前标准。开发格中的形状转换与三组编号已经能预报相当一部分输出差异。

## 哪些信息可以进入预测

沿用36幅图、6种大小、2种角度、同一位置和6个冻结模型。按大小和角度索引奇偶交错，12个因素格分为6个开发格和6个留出格，各18幅图。开发格为(0,17)、(1,22)、(2,17)、(3,22)、(4,17)、(5,22)，其余6格留出。每格固定方形→椭圆、椭圆→心形、心形→方形三个方向。每个无向图像对只用一个方向，避免全强度时反向单组和正向互补三组共享内部状态。

每个模型、每种强度在留出格有18个配对，每对4种三组替换，共72条。全强度主检验共432条，半强度作为同一规则的迁移检验另有432条。所有图像都来自前期已经研究过的面板，此次留出发生在拟合本轮规则时。输入、输出和方法记录共享图像。统计先在各模型内计算，再等权平均。

对目标三组 a、b、c，二阶预测 QX = Xab + Xac + Xbc − Xa − Xb − Xc + X0。三阶响应是真实 Xabc 减去 QX。中间目标是 block2 输出的完整32×32×32激活场，输出目标是供体类别减受体类别的 logit，另存三个类别各自的预测。一个三组预测只读同方向本三组的七个真子集，四个三组共用11个低阶前向。输出规则还读取空组处三个头部梯度。二阶输出基线使用相同的低阶输出，稀疏规则额外读取低阶内部状态和导数。

## 规则及封存

每个源组包含block1的四个整通道。第一步用空组和单组的预激活推算block2第一处ReLU的三组预激活，经过阈值，减去该处的二阶预测，得到新增的三阶响应。在开发格按该响应的平方和锁定4个通道K。卷积和推理BN的斜率把K的修正传到第二处ReLU。第二处先以二阶预激活为底，再加这项修正，重新过阈值。在开发格按整层三阶平方误差的减少量锁定4个输出通道C，其余28个通道使用原二阶预测。全通道代数在一个开发配对的两种强度上与原block2一致到5×10⁻¹⁴以内。

规则把C的中间修正乘以空组处的读出导数，并加到二阶logit上。六个模型各自在全强度开发格拟合一套K、C与控制模型，随后统一封存。每个模型始终使用自己的名单，半强度也使用同一份参数。全部864条预测和阻断名单在计算本轮留出三组真值前已封存。[设计与判据]({folder}/design.json) · [划分与输入指纹]({folder}/protocol.json) · [冻结参数]({folder}/fit_freeze.json) · [预测封存]({folder}/prediction_freeze.json)

## 全强度主结果

表中数值是预测误差RMS除以实测三阶作用RMS，越小越好。预测额外三阶作用为零时，比值为1。事前标准要求每项的六模型均值至多0.95，且至少4个模型低于1。完整传递假设要求四项都通过。

| 预测对象 | 全强度误差比 | 低于1的模型 | 半强度误差比 | 全强度判定 |
|---|---:|---:|---:|---|
{chr(10).join(main_table)}

| 全强度对照 | 整个中间层三阶误差比 | 最终输出三阶误差比 | 总输出变化相对误差 |
|---|---:|---:|---:|
{chr(10).join(controls)}

选中4个通道内部的三阶误差比为{ratio('middle_third_selected','rule'):.3f}，整个场为{ratio('middle_third_all','rule'):.3f}。类别转换×三组编号基线使用开发格的平均三阶logit加到留出配对的真实二阶读数上。二阶基线在本次固定留出、固定配对方向的总变化误差为{100*ratio('output_total','quadratic'):.2f}%，这一分母是三组相对于空组的整个输出变化。上一轮23.82%来自72个有向配对的全量回顾。本轮主判据使用单独的三阶分量作分母，保留各模型归一化后的差别。

| 模型 | 第一处4通道K | 中间4通道C | 中间层 | 输出 | 阻断 | 只保留 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(models)}

![逐模型全强度误差和阻断实测对照]({folder}/held_condition_results.png)

## 实际阻断揭示的作用

在每个目标三组的八个角上，把相同四个C通道恢复为该配对空组的实际非负激活，其余通道自然变化，原block3与读出重新计算。原始输出三阶差分减阻断后三阶差分，就是被移除的带符号作用。反向操作从空组中间场出发，只让C采用各角的真实激活，再算输出三阶差分。两种操作各自给出独立的预测检验。

所选四通道实际移除作用的教师内RMS均值为{truth_rms('block_removed','chosen4'):.3f} logit，两个同预算组外随机对照为{truth_rms('block_removed','random0'):.3f}和{truth_rms('block_removed','random1'):.3f}。规则对四通道移除方向命中{signs('block_removed','chosen4')}，对只保留方向命中{signs('keep_recovered','chosen4')}。依开发格预测绝对值中位数划分的强组共{sum(int(r['count']) for r in strong)}条，弱组{sum(int(r['count']) for r in weak)}条。强弱组实际移除作用RMS的教师均值分别为{strong_value.mean():.3f}和{weak_value.mean():.3f}，{int((strong_value>weak_value).sum())}/6个模型的强组更大。区分影响大小与准确预报实际数值是两个评估指标。

固定顺序的首个留出例为模型291、案例{int(sample['case'])}、源掩码{int(sample['mask'])}。其真实输出三阶修正为{sample['third_true']:+.3f}，规则预测{sample['third_rule']:+.3f}。对同一四通道，阻断实际移除{sample_patch['removed']:+.3f}，只保留它们时重现{sample_patch['keep']:+.3f}。冻结规则给两种操作的预报均为{sample_patch['predicted']:+.3f}。

## 对失败环节的事后诊断

在全强度432条记录中，实际移除作用与只保留作用有{post['opposite_sign']}条符号相反，两者之差的教师等权RMS为{post['block_keep_gap_rms']:.3f} logit。这是同一组通道在其余通道自然变化和其余通道固定两种背景下的直接对照。使用一个固定的带符号贡献同时预报两种操作，在这些记录上会给出至少一个错误方向。

进一步只读10月13日保存的真实中间场三阶系数及其空组梯度投影。选中四通道约承载{100*post['selected_H3_energy_fraction']:.1f}%的中间场三阶平方范数（先按模型求比例再均值）。把四通道的**真实**三阶响应交给同一空组线性读出，最终三阶误差比为{post['measured_H4_J0_error_ratio']:.3f}。把全部32通道的真实三阶响应交给它，误差比为{post['measured_H32_J0_error_ratio']:.3f}。这两项诊断使用三组答案，专用于检查读出假设。它们与上表中事先作出的预测分开保存。[事后诊断明细]({folder}/tables/posthoc_diagnostic.csv)

## 复核与数据

独立实现用原block2完整卷积重建稀疏阈值公式，用转置卷积手算空组梯度，并用广播掩码重新执行全部69,120个阻断与只保留状态。3,240个原始状态的三维logit与既有归档一致，所核对数值的最大绝对误差为{max(audit['maximum_absolute_errors'].values()):.2e}。六个模型、所有失败和强弱组均可逐项查看。

- [逐案例预测]({folder}/tables/cases.csv) · [逐案例阻断与只保留]({folder}/tables/patches.csv)
- [逐模型得分]({folder}/tables/teacher_scores.csv) · [强弱条件]({folder}/tables/strata.csv) · [独立复核]({folder}/independent_audit.json)
- [低阶预测与干预代码](../scripts/dsprites_triple_rule.py) · [封存与运行入口](../scripts/run_dsprites_triple_rule.py) · [复现命令]({folder}/README.md)
"""
    output_report=report_path(run)
    output_report.write_text(text,encoding='utf-8')
    (run/'README.md').write_text(f"""# 三组规则判别实验

[研究报告](../../reports/{output_report.name}) · [协议](protocol.json) · [数学和判定设计](design.json)。

每个文件名标识模型种子和强度。两侧各6个因素格、18张图及18个循环方向配对。单个预测文件有18个case，三组掩码轴依次为7、11、13、14。空间轴是通道、y、x，编号从0开始。完整重建依赖本项目现有模型、面板及列于协议中的源码。

- fit中有每个模型的四个K、四个C、开发中位数、回归参数和两套对照通道。
- predictions中qh是完整中间层二阶预测，h0是空组。middle_corrections为案例×三组×方法4×选中通道4×32×32，方法依次为物理规则、岭回归、仅末阈值、仅上传修正。未选通道的修正取零。output_corrections对三个logit保存二阶、规则、岭回归、仅阈值、仅上传、类别均值六种三阶预测。加qlogits得到最终logit预测。channel_logits和patch_ids给出规则逐通道贡献及全部阻断名单。
- measurements中true_middle是真实三组中间场。natural_logits的掩码轴为0—14。patch_logits的轴为案例×三组×五名单×阻断/只保留×八角×三个logit。五名单依次为所选四通道、预测最强的一个通道、预测最弱的一个通道、随机0、随机1。八角为先空、按掩码递增的单组、按掩码递增的双组、三组。干预单位是整通道。
- tables中cases记录864个三组实例，patches记录4320个名单实例。teacher_scores给出教师内的RMS和归一化比。按条件回顾时须保留模型、因素格与共享图像的依赖。

已有交付在项目根目录执行最后一行即可只读校验。复算时在runs下新建目录，复制design.json，再把该目录的绝对路径作为 --run-dir 参数传给下面每个入口。环境为相邻medical_ai_reliability项目的.venv-gpu，Python 3.12，PyTorch float64。

```text
python -B -X utf8 scripts/run_dsprites_triple_rule.py --stage predict
python -B -X utf8 scripts/run_dsprites_triple_rule.py --stage measure
python -B -X utf8 scripts/audit_dsprites_triple_rule.py
python -B -X utf8 scripts/analyze_dsprites_triple_rule.py
python -B -X utf8 scripts/report_dsprites_triple_rule.py --stage report
python -B -X utf8 scripts/report_dsprites_triple_rule.py --stage seal
python -B -X utf8 scripts/report_dsprites_triple_rule.py --stage verify
```
""",encoding='utf-8')


def finalize(run):
    assert not (run/'delivery_manifest.json').exists()
    assert teacher.read(run/'independent_audit.json')['status']=='PASS'
    checked=check_tables(run)
    assert (run/'held_condition_results.png').is_file()
    for name in CODE:
        target=run/'source'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE/'scripts'/name,target)
    snapshot=run/'source'/'CURRENT_RESEARCH.md'
    shutil.copy2(BASE/'CURRENT_RESEARCH.md',snapshot)
    protected=[p for p in run.rglob('*') if p.is_file()]
    protected.extend([report_path(run)]+[BASE/'scripts'/n for n in CODE])
    payload=dict(utc=teacher.now(),files=[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,
                sha256=teacher.sha(p)) for p in sorted(protected)],verified_primary_table_ratios=checked)
    atomic_json(run/'delivery_manifest.json',payload)
    (run/'delivery_manifest.sha256').write_text(teacher.sha(run/'delivery_manifest.json')+'\n',encoding='ascii')


def verify(run):
    protocol=verify_protocol(run)
    for name,prev in (('fit_freeze','protocol'),('prediction_freeze','fit_freeze'),('measurement_freeze','prediction_freeze')):
        verify_seal(run,name,prev)
    entries=manifest_entries(run)
    for name,digest in entries.items(): assert teacher.sha(BASE/name)==digest,name
    for name,digest in teacher.read(run/'posthoc.json')['inputs'].items(): assert teacher.sha(BASE/name)==digest,name
    checked=check_tables(run)
    print(dict(status='PASS',files=len(entries),seeds=len(protocol['seeds']),verified_primary_table_ratios=checked),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=('report','seal','verify'),required=True)
    args=parser.parse_args()
    if args.stage=='verify':
        verify(args.run_dir)
    else:
        assert not (args.run_dir/'delivery_manifest.json').exists()
        if args.stage=='report':report(args.run_dir)
        else:finalize(args.run_dir)

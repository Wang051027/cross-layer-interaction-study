"""提取具体下游位置、通道、联合阻断余量、残差支路及抵消。

主26条记录固定；控制单独保留。单点效应之和不当成联合阻断效应。
"""
from __future__ import annotations
import csv
import json
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_downstream_routes import BASE,RUN,SHAPE,verify_protocol,verify_selection,verify_folder
from run_dsprites_upstream_rescue import load
from downstream_route_mechanism import commands


def rms(values):
    """空教师为缺失，零效应观测才为0。"""
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def cancellation(positive,negative):
    """正负贡献抵消率；零分母留空。"""
    den=positive-negative
    return None if den<=1e-10 else float(1-abs(positive+negative)/den)


def table(name,rows):
    """写可直接查看的证据CSV，完整地图另存NPZ。"""
    p=RUN/'tables'/f'{name}.csv';p.parent.mkdir(exist_ok=True);assert rows and not p.exists()
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    """逐项恒等式与完整控制核验后汇总；不以效果大小删失败案例。"""
    assert not (RUN/'analysis.json').exists();p=verify_protocol();verify_selection();panel=load(SHAPE/'panel/data.npz')
    pairs=[];blocks=[];points=[];channels=[];residual_points=[];residual_channels=[];summaries=[];maxerr=0.
    for record in p['cohort']:
        number=record['record'];sf=RUN/f'discovery/record_{number:03d}';ef=RUN/f'evaluation/record_{number:03d}';verify_folder(sf);verify_folder(ef)
        sel=load(sf/'data.npz');raw=load(ef/'data.npz');recipient,donor=panel['pairs'][record['case']];w=np.zeros(3);w[panel['labels'][donor]]=1.;w[panel['labels'][recipient]]=-1.
        base={k:record[k] for k in ('record','seed','case','mask','endpoint','family')};base.update(primary='primary26' in record['roles'],roles=';'.join(record['roles']))
        for ctx in range(2):
            key={**base,'context':ctx};s={k:v[ctx] for k,v in sel.items()};v={k:a[ctx] for k,a in raw.items()};route=v['route_logits']@w;f0=route[0,0];f1=route[1,1];total=f1-f0
            conv=route[1,0]-f0;skip=route[0,1]-f0;interaction=route[1,1]-route[1,0]-route[0,1]+f0
            maxerr=max(maxerr,abs(total-conv-skip-interaction),float(abs(v['route_channels'].sum(-1)-(route-f0)).max()))
            pairs.append(dict(**key,C=json.dumps(record['C']),point=record['point'],F_off=float(f0),F_on=float(f1),C_effect=float(total),convolution_only=float(conv),skip_only=float(skip),branch_interaction=float(interaction),convolution_given_skip=float(f1-route[0,1]),skip_given_convolution=float(f1-route[1,0]),branch_opposition=bool(conv*skip<0 and abs(conv)>1e-8 and abs(skip)>1e-8)))
            for layer in range(2):
                cmds=commands(s,layer,record['point']);scores=s['point_effect'][layer];change=s['delta_activation'][layer];switch=s['switched'][layer];active=abs(change)>1e-10
                # 位置表只列五个效应/开关名单前64的并集及原j，不冒充完整空间表。
                candidate=set([record['point']] if layer==0 else [])
                for order in (0,3,4,5,6):candidate.update(int(x) for x in s['orders'][layer,order,:64] if x>=0)
                for point in sorted(candidate):
                    c,y,x=np.unravel_index(point,(32,32,32));rank=np.flatnonzero(s['orders'][layer,0]==point)
                    points.append(dict(**key,layer=layer,point=point,channel=int(c),y=int(y),x=int(x),single_removed=float(scores[point]),activation_change=float(change[point]),switch=bool(switch[point]),active=bool(active[point]),absolute_rank=int(rank[0])+1 if len(rank) else None,original_j=bool(layer==0 and point==record['point'])))
                for channel in range(32):
                    ar=active.reshape(32,1024)[channel];sw=switch.reshape(32,1024)[channel];sc=scores.reshape(32,1024)[channel]
                    channels.append(dict(**key,layer=layer,channel=channel,whole_channel_removed=float(s['channel_effect'][layer,channel]),sum_single_points=float(sc.sum()),active_points=int(ar.sum()),switch_points=int((ar&sw).sum()),stable_points=int((ar&~sw).sum()),positive_single_sum=float(sc.clip(0).sum()),negative_single_sum=float(sc.clip(max=0).sum())))
                for index,cmd in enumerate(cmds):
                    ids=cmd['ids'];score_sum=float(scores[ids].sum());spent=len(cmd.get('channels',ids));active_n=int(active[ids].sum());switch_n=int(switch[ids].sum())
                    if layer==0:
                        f=v['first_logits'][index]@w;remaining=f[1,0]-f0;keep=f[1,1]-f0;skipoff_remaining=f[0,0]-f0;skipoff_keep=f[0,1]-f0
                        channel=v['first_channels'][index,1,0];pn=v['first_pn'][index,1,0]
                        complement=float(f1-f[1,0]-f[1,1]+route[0,1]);complement0=float(route[1,0]-f[0,0]-f[0,1]+f0)
                        maxerr=max(maxerr,float(abs(v['first_channels'][index].sum(-1)-(f-f0)).max()),float(abs(v['first_pn'][index].sum(-1)-(f-f0)).max()))
                        blogit=v['first_logits'][index,1,0]
                    else:
                        f=v['last_logits'][index]@w;remaining=f[0]-f0;keep=f[1]-f0;skipoff_remaining=None;skipoff_keep=None
                        channel=v['last_channels'][index,0];pn=v['last_pn'][index,0];complement=float(f1-f[0]-f[1]+f0);complement0=None;blogit=v['last_logits'][index,0]
                        maxerr=max(maxerr,abs(total-remaining-score_sum),float(abs(v['last_channels'][index].sum(-1)-(f-f0)).max()),float(abs(v['last_pn'][index].sum(-1)-(f-f0)).max()))
                    removed=total-remaining
                    row=dict(**key,layer=layer,command=index,kind=cmd['kind'],order=cmd['order'],budget=int(cmd['budget']),spent_units=spent,spatial_points=len(ids),active_points=active_n,switch_points=switch_n,
                        C_effect=float(total),removed=float(removed),remaining=float(remaining),keep_with_native_skip=float(keep),remaining_without_skip=None if skipoff_remaining is None else float(skipoff_remaining),keep_without_skip=None if skipoff_keep is None else float(skipoff_keep),
                        skip_only=float(skip) if layer==0 else None,remaining_convolution_given_skip=float(remaining-skip) if layer==0 else None,
                        sum_single_removed=score_sum,joint_minus_singles=float(removed-score_sum),selected_complement_interaction=complement,selected_complement_without_skip=complement0,
                        residual_positive=float(pn[0]),residual_negative=float(pn[1]),residual_cancellation=cancellation(*pn),channel_positive=float(channel.clip(0).sum()),channel_negative=float(channel.clip(max=0).sum()),
                        block_logit0=float(blogit[0]),block_logit1=float(blogit[1]),block_logit2=float(blogit[2]))
                    blocks.append(row)
                    if index in (4,44):
                        j=(4,44).index(index)
                        for c,value in enumerate(channel):residual_channels.append(dict(**key,layer=layer,command=index,channel=c,residual_contribution=float(value)))
                        for rank,(point,value) in enumerate(zip(v['residual_top_ids'][layer,j],v['residual_top_values'][layer,j])):
                            c,y,x=np.unravel_index(int(point),(32,32,32));residual_points.append(dict(**key,layer=layer,command=index,rank=rank+1,point=int(point),channel=int(c),y=int(y),x=int(x),residual_contribution=float(value)))
    # 同教师、同上下文及方法内RMS；跨教师比例以教师自己的原C效应RMS为分母。
    for seed in teacher.SEEDS:
        for ctx in range(2):
            for layer in range(2):
                for command in range(67):
                    for cohort in ('primary26','controls'):
                        group=[r for r in blocks if r['seed']==seed and r['context']==ctx and r['layer']==layer and r['command']==command and r['primary']==(cohort=='primary26')]
                        denom=rms([r['C_effect'] for r in group]);rec=dict(seed=seed,context=ctx,layer=layer,command=command,cohort=cohort,count=len(group),C_effect_rms=denom)
                        for metric in ('remaining','removed','joint_minus_singles','keep_with_native_skip','remaining_without_skip','keep_without_skip'):
                            values=[r[metric] for r in group if r[metric] is not None];value=rms(values);rec[metric+'_rms']=value;rec[metric+'_rms_ratio']=value/denom if value is not None and denom is not None and denom>1e-8 else None
                        summaries.append(rec)
    mainpairs=[r for r in pairs if r['primary'] and r['context']==1];mainblocks=[r for r in blocks if r['primary'] and r['context']==1]
    assert len(mainpairs)==26 and len(pairs)==74 and len(blocks)==9916 and maxerr<1e-9
    report_methods=(0,1,2,3,4,5,6,10,12,16,18,22,24,28,42,44,45,48,50,54,56,61,62,63,64,65,66)
    aggregates=[]
    for ctx in range(2):
        for layer in range(2):
            for command in report_methods:
                group=[r for r in summaries if r['context']==ctx and r['layer']==layer and r['command']==command and r['cohort']=='primary26' and r['count']>0]
                sample=next(r for r in blocks if r['context']==ctx and r['layer']==layer and r['command']==command)
                # special的实际预算随案例变化，不能把首个控制案例的数量当汇总预算。
                actual=[r['spent_units'] for r in blocks if r['primary'] and r['context']==ctx and r['layer']==layer and r['command']==command]
                rec=dict(context=ctx,layer=layer,command=command,kind=sample['kind'],order=sample['order'],budget=sample['budget'] if sample['kind']!='special' else None,
                    spent_min=min(actual),spent_max=max(actual),teachers=len(group),records=sum(r['count'] for r in group))
                for k in summaries[0]:
                    if k.endswith('_rms') or k.endswith('_ratio'):
                        vals=[r[k] for r in group if r[k] is not None];rec[k]=float(np.mean(vals)) if vals else None
                aggregates.append(rec)
    example_record=next(r for r in p['cohort'] if (r['seed'],r['case'],r['mask'],r['endpoint'])==(202609292,70,15,25))
    exid=example_record['record'];examples=dict(record=example_record,pairs=[r for r in pairs if r['record']==exid],blocks=[r for r in blocks if r['record']==exid],points=[r for r in points if r['record']==exid],channels=[r for r in channels if r['record']==exid],residual_points=[r for r in residual_points if r['record']==exid])
    claims=dict(status='PASS',utc=teacher.now(),records=37,primary_records=26,primary_states=len({(r['seed'],r['case'],r['mask']) for r in mainpairs}),primary_teachers=len({r['seed'] for r in mainpairs}),
        new_context_branch_opposition=sum(r['branch_opposition'] for r in mainpairs),
        first_all_block_nonzero_skip=sum(abs(r['skip_only'])>1e-8 for r in mainpairs),
        first_point256_joint_nonlinear=sum(abs(r['joint_minus_singles'])>1e-8 for r in mainblocks if r['layer']==0 and r['command']==4),
        decomp_max_abs_effect=max(abs(r['C_effect']) for r in pairs if 'decomposition_control' in r['roles']),maximum_identity_error=maxerr,
        teacher_equal=aggregates)
    tabs=dict(pairs=pairs,blocking=blocks,point_candidates=points,channel_localization=channels,residual_points=residual_points,residual_channels=residual_channels,teacher_summary=summaries)
    for name,rows in tabs.items():table(name,rows)
    teacher.dump(RUN/'claim_checks.json',claims);teacher.dump(RUN/'examples.json',examples)
    teacher.dump(RUN/'analysis.json',dict(status='PASS',utc=teacher.now(),table_rows={k:len(v) for k,v in tabs.items()},maximum_identity_error=maxerr,point_table_scope='union top64 absolute/switch/stable/supporting/opposing plusoriginalj;all32768maps in NPZ',residual_tables_scope='native-skip blocks at predeclared absolute point256/channel4;top128 terminal positions plusall32channels',aggregation='teacherRMS andteacher RMSratios;5 nonemptyprimary teachers,controls separate;no independentpairpvalue'))
    print(json.dumps({k:v for k,v in claims.items() if k!='teacher_equal'}),flush=True)


if __name__=='__main__':analyze()

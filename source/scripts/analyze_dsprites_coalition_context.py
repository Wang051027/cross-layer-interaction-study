"""四状态效应逐行提取：保留失败、随机及分解对照，不把交互项分配成唯一份额。"""
from __future__ import annotations
import csv
import json
from collections import Counter
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_coalition_context import BASE,RUN,PRIOR,AR,SHAPE,verify_protocol,verify_folder
from run_dsprites_upstream_rescue import load
from coalition_context_mechanism import ENDPOINTS,TERMS,CONTRAST,STATES,DROP_SLOTS

EPS=1e-8
FAMILIES=('source','middle','first_activation','second_activation')
VARIANTS=('selected','random0','random1')


def rms(values):
    """空子集是缺失，不是零；有观测才计算RMS。"""
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def modulation(a,b,j):
    """组外对C的条件效应调制；与组外单独输出效应分开描述。"""
    if abs(a)<=EPS: return 'context_enabled' if abs(b)>EPS else 'both_C_effects_zero'
    if abs(j)<=EPS:return 'unchanged'
    if abs(b)<=EPS:return 'canceled_to_zero'
    if a*b<0:return 'reversed'
    return 'enhanced' if abs(b)>abs(a) else 'attenuated'


def cancel(parts):
    """绝对分量总量为0时抵消比例未定义。"""
    den=float(np.abs(parts).sum())
    return None if den<=1e-10 else float(1-abs(np.sum(parts))/den)


def save_table(name,rows):
    """UTF-8 BOM便于本地表格软件阅读，JSON保留通道编号列表。"""
    path=RUN/'tables'/f'{name}.csv';path.parent.mkdir(exist_ok=True)
    assert rows and not path.exists()
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    """每个旧端点都提取六差分；主结论群体在新输出之前已冻结。"""
    assert not (RUN/'analysis.json').exists();protocol=verify_protocol();panel=load(SHAPE/'panel/data.npz')
    weights=np.zeros((72,3))
    for case,(r,d) in enumerate(panel['pairs']):weights[case,panel['labels'][d]]=1.;weights[case,panel['labels'][r]]=-1.
    effects=[];states=[];channels=[];members=[];summaries=[];examples=[];focus_total=0;max_identity=0.
    example_keys={(r['seed'],r['case'],r['mask'],r['endpoint']) for r in protocol['examples']}
    for seed in teacher.SEEDS:
        verify_folder(RUN/f'evaluation/seed_{seed}');raw=load(RUN/f'evaluation/seed_{seed}/data.npz')
        prior=load(PRIOR/f'evaluation/seed_{seed}/data.npz'); fixed=load(PRIOR/f'search/seed_{seed}/frozen.npz');search=load(PRIOR/f'search/seed_{seed}/data.npz')
        old=load(AR/f'evaluation/seed_{seed}/data.npz');focus=old['threshold'][:,:,3,0].astype(bool)&~old['threshold'][:,:,3,1].astype(bool);focus_total+=int(focus.sum())
        task=np.einsum('nmespk,nk->nmesp',raw['logits'],weights)
        terms=np.einsum('ts,nmesp->nmept',CONTRAST,task)
        max_identity=max(max_identity,float(abs(terms[...,3]-terms[...,0]-terms[...,1]-terms[...,2]).max()),float(abs(terms-raw['channel_terms'].sum(-1)).max()),float(abs(terms-raw['spatial_pn'].sum(-1)).max()),float(abs(terms[:,:,:,0]-raw['branch_terms'].sum(-1)).max()))
        seedrows=[]
        for case in range(72):
            for m,mask in enumerate((3,15)):
                C=fixed['channels'][case,m];C=C[C>=0];point=int(fixed['lists'][case,m,0,0]);direction=int(fixed['half_direction'][case,m])
                for slot,endpoint in enumerate(ENDPOINTS):
                    i=endpoint-1;family=FAMILIES[i%12//3];variant=VARIANTS[i%3];strategy='aligned' if endpoint>=25 else 'moving_A'
                    eligible=bool(search['choices'][case,m,i,2]);q=prior['margins'][case,m,endpoint].min(-1)
                    aligned=eligible and bool((q>=1e-6).all());own=eligible and bool(q[0]>=1e-6);physical=eligible and bool((q[1:]>=1e-6).all())
                    noH=bool(prior['shifts'][case,m,endpoint,0]<=1e-10)
                    status='not_met' if not (aligned if strategy=='aligned' else own) else 'decomposition_only' if noH else 'common_reference' if physical else 'reference_dependent'
                    primary=bool(focus[case,m] and strategy=='aligned' and variant=='selected' and aligned)
                    z=raw['point_z'][case,m,slot];t=terms[case,m,slot];a,o,j,total,ca,oa=t[0]
                    row=dict(seed=seed,case=case,mask=mask,endpoint=endpoint,slot=slot,strategy=strategy,family=family,variant=variant,C=json.dumps(C.tolist()),point=point,direction=direction,
                        eligible=eligible,focus539=bool(focus[case,m]),prior_aligned=aligned,prior_own=own,prior_physical=physical,prior_status=status,primary_success=primary,
                        C_max_change=float(raw['shifts'][case,m,slot,0]),outside_max_change=float(raw['shifts'][case,m,slot,1]),
                        modulation=modulation(a,ca,j),standalone_offset=bool(a*o<0 and abs(a)>EPS and abs(o)>EPS),
                        j_new_flip=bool((z[0]>0)!=(z[3]>0)),j_C_only_target_side=bool(direction*z[1]>EPS),j_outside_only_target_side=bool(direction*z[2]>EPS),
                        j_before_target_side=bool(direction*z[0]>EPS),j_after_target_side=bool(direction*z[3]>EPS),component_cancellation=cancel(np.array([a,o,j])))
                    for k,name in enumerate(TERMS):
                        row[name]=float(t[0,k]);row[f'{name}_j_before_control']=float(t[0,k]-t[1,k]);row[f'{name}_j_after_control']=float(t[0,k]-t[2,k])
                        for b,nameb in enumerate(('affine','first_relu','second_relu')):row[f'{name}_{nameb}']=float(raw['branch_terms'][case,m,slot,k,b])
                    for k in (0,2,3,4):
                        name=TERMS[k];pieces=raw['channel_terms'][case,m,slot,0,k];pn=raw['spatial_pn'][case,m,slot,0,k]
                        row[f'{name}_channel_positive']=float(pieces.clip(0).sum());row[f'{name}_channel_negative']=float(pieces.clip(max=0).sum())
                        row[f'{name}_channel_cancellation']=cancel(pieces);row[f'{name}_spatial_positive']=float(pn[0]);row[f'{name}_spatial_negative']=float(pn[1]);row[f'{name}_spatial_cancellation']=cancel(pn)
                    effects.append(row);seedrows.append(row)
                    for state in range(4):
                        r={k:row[k] for k in ('seed','case','mask','endpoint','slot','family','variant','strategy','primary_success')}
                        r.update(state=STATES[state],point_z=float(z[state]),point_u=float(raw['point_u'][case,m,slot,state]),F=float(task[case,m,slot,state,0]),F_j_before=float(task[case,m,slot,state,1]),F_j_after=float(task[case,m,slot,state,2]),
                            first_gate_flips=int(raw['natural_flips'][case,m,slot,state,0]),second_gate_flips=int(raw['natural_flips'][case,m,slot,state,1]),j_before_patch_second_flips=int(raw['patch_second_flips'][case,m,slot,state,0]),j_after_patch_second_flips=int(raw['patch_second_flips'][case,m,slot,state,1]))
                        for mode,name in enumerate(('natural','j_before','j_after')):
                            for label in range(3):r[f'{name}_logit_{label}']=float(raw['logits'][case,m,slot,state,mode,label])
                        states.append(r)
                    key=(seed,case,mask,endpoint)
                    if primary or key in example_keys:
                        for term in range(6):
                            for ch in range(32):
                                channels.append(dict(seed=seed,case=case,mask=mask,endpoint=endpoint,family=family,primary_success=primary,term=TERMS[term],channel=ch,
                                    contribution=float(raw['channel_terms'][case,m,slot,0,term,ch]),j_before_contribution=float(raw['channel_terms'][case,m,slot,0,term,ch]-raw['channel_terms'][case,m,slot,1,term,ch]),j_after_contribution=float(raw['channel_terms'][case,m,slot,0,term,ch]-raw['channel_terms'][case,m,slot,2,term,ch])))
                    if key in example_keys:
                        examples.append(dict(effect=row,states=states[-4:],reason=next(e['reason'] for e in protocol['examples'] if (e['seed'],e['case'],e['mask'],e['endpoint'])==key)))
                    if slot in DROP_SLOTS:
                        di=DROP_SLOTS.index(slot)
                        for context,state in enumerate((1,3)):
                            for pos,ch in enumerate(C):
                                dlog=raw['drop_logits'][case,m,di,context,pos];df=float(dlog@weights[case]);dz=float(raw['drop_z'][case,m,di,context,pos])
                                members.append(dict(seed=seed,case=case,mask=mask,endpoint=endpoint,family=family,primary_success=primary,context=STATES[state],channel=int(ch),full_point_z=float(z[state]),reverted_point_z=dz,
                                    change_needed_for_target_side=bool(direction*z[state]>EPS and direction*dz<-EPS),full_F=float(task[case,m,slot,state,0]),reverted_F=df,F_effect_of_member_change=float(task[case,m,slot,state,0]-df),logit0=float(dlog[0]),logit1=float(dlog[1]),logit2=float(dlog[2])))
        # 同一教师内先计算，随后跨有成功观测的教师等权；绝不把空教师当零。
        for slot,endpoint in enumerate(ENDPOINTS):
            base=[r for r in seedrows if r['slot']==slot]
            groups=dict(all864=lambda r:True,focus539=lambda r:r['focus539'],aligned_success539=lambda r:r['focus539'] and r['prior_aligned'],aligned_failed539=lambda r:r['focus539'] and not r['prior_aligned'])
            for group,predicate in groups.items():
                rows=[r for r in base if predicate(r)];sample=base[0]
                rec=dict(seed=seed,endpoint=endpoint,family=sample['family'],variant=sample['variant'],strategy=sample['strategy'],cohort=group,count=len(rows))
                for name in TERMS:
                    for suffix in ('','_j_before_control','_j_after_control','_affine','_first_relu','_second_relu'):
                        metric=name+suffix;rec[metric+'_rms']=rms([r[metric] for r in rows])
                summaries.append(rec)
    assert focus_total==539 and len(effects)==13824 and len(states)==55296 and max_identity<1e-9
    primary=[r for r in effects if r['primary_success']];assert len(primary)==26 and len({(r['seed'],r['case'],r['mask']) for r in primary})==15
    aggregate=[]
    for family in FAMILIES:
        rows=[r for r in primary if r['family']==family];summ=[r for r in summaries if r['family']==family and r['variant']=='selected' and r['strategy']=='aligned' and r['cohort']=='aligned_success539' and r['count']>0]
        rec=dict(family=family,count=len(rows),models=len(summ))
        for metric in [k for k in summaries[0] if k.endswith('_rms')]:rec[metric]=float(np.mean([r[metric] for r in summ])) if summ else None
        aggregate.append(rec)
    main_members=[r for r in members if r['primary_success'] and r['context']=='11_after']
    pointpatterns=Counter()
    for r in primary:
        if not r['j_new_flip']: pattern='already_on_target_side_or_same_gate'
        else:pattern=('C_yes' if r['j_C_only_target_side'] else 'C_no')+'_'+('outside_yes' if r['j_outside_only_target_side'] else 'outside_no')
        pointpatterns[pattern]+=1
    claims=dict(status='PASS',utc=teacher.now(),focus_states=539,primary_records=len(primary),primary_distinct_states=15,primary_models=len({r['seed'] for r in primary}),
        primary_family_counts=dict(Counter(r['family'] for r in primary)),modulation_counts=dict(Counter(r['modulation'] for r in primary)),standalone_offsets=sum(r['standalone_offset'] for r in primary),
        C_unchanged=sum(r['C_max_change']<=1e-10 for r in primary),outside_unchanged=sum(r['outside_max_change']<=1e-10 for r in primary),point_patterns=dict(pointpatterns),
        primary_member_reversions=len(main_members),member_changes_needed=sum(r['change_needed_for_target_side'] for r in main_members),
        decomposition_controls=sum(r['focus539'] and r['strategy']=='moving_A' and r['prior_status']=='decomposition_only' for r in effects),max_identity_error=max_identity,teacher_equal=aggregate)
    tables=dict(effects=effects,states=states,channels=channels,member_reversions=members,teacher_summary=summaries)
    for name,rows in tables.items():save_table(name,rows)
    teacher.dump(RUN/'claim_checks.json',claims);teacher.dump(RUN/'examples.json',examples)
    teacher.dump(RUN/'analysis.json',dict(status='PASS',utc=teacher.now(),table_rows={name:len(rows) for name,rows in tables.items()},axes='raw:case72,mask2,endpoint16;logits thenstate4,mode3,class3;channel_terms thenmode3,term6,channel32',channel_table_scope='all26 primarysuccessrecords plus2 priorcontrol examples;fullcasechannels inrawnpz',aggregation='withinteacherRMS thenequalnonemptyteachermean;no independentpair inferentialtest',maximum_identity_error=max_identity))
    print(json.dumps(claims,ensure_ascii=False),flush=True)


if __name__=='__main__':analyze()

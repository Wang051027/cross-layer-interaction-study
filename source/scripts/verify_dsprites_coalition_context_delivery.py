"""只读核验四状态交付：逐行表格、独立差分、统计分母、图和文件SHA。"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_coalition_context import BASE,RUN,PRIOR,AR,SHAPE,verify_protocol,verify_folder,preservation
from run_dsprites_upstream_rescue import load

ENDPOINTS=tuple(range(25,37))+(1,4,7,10)
TERMS=('C_before','outside_before','interaction','total','C_after','outside_after')
FAMILIES=('source','middle','first_activation','second_activation')
VARIANTS=('selected','random0','random1')
CODE=('coalition_context_mechanism.py','run_dsprites_coalition_context.py','test_dsprites_coalition_context.py','audit_dsprites_coalition_context.py',
      'analyze_dsprites_coalition_context.py','report_dsprites_coalition_context.py','verify_dsprites_coalition_context_delivery.py','finalize_dsprites_coalition_context.py')


def close(a,b):
    """不允许形状广播掩盖错误；容差与模型独立审计一致。"""
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape,(a.shape,b.shape)
    assert not a.size or float(abs(a.astype(float)-b.astype(float)).max())<1e-9


def truth(value):
    """CSV布尔值必须显式为True/False。"""
    assert value in ('True','False');return value=='True'


def optional(value,expected):
    """零分母和空群体用空值，不强制变成0。"""
    if expected is None:assert value==''
    else:close(float(value),expected)


def cancellation(value):
    """独立重算抵消定义。"""
    den=sum(abs(v) for v in value)
    return None if den<=1e-10 else 1-abs(sum(value))/den


def verify(before_manifest=False,deep=False):
    """封存前生成通过记录；封存后只读，保持完整交付字节不变。"""
    records=[]
    if not before_manifest:
        manifest=RUN/'delivery_manifest.json';assert teacher.sha(manifest)==manifest.with_suffix('.sha256').read_text().strip()
        records=teacher.read(manifest)['files'];assert len({r['path'] for r in records})==len(records)
        for r in records:
            path=(BASE/r['path']).resolve();assert path.is_relative_to(BASE.resolve())
            assert path.stat().st_size==r['bytes'] and teacher.sha(path)==r['sha256'],r['path']
    protocol=verify_protocol();audit=teacher.read(RUN/'independent_audit.json');analysis=teacher.read(RUN/'analysis.json');claims=teacher.read(RUN/'claim_checks.json');supp=teacher.read(RUN/'report_checks.json')
    assert all(v['status']=='PASS' for v in (audit,analysis,claims,supp))
    assert audit['native_fields']==55296 and audit['point_patches']==110592 and audit['member_reversions']==16304 and audit['old_delivery_entries']==1650
    assert max(audit['maximum_errors'].values())<1e-9
    tables={}
    for name,count in analysis['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);rows=list(reader)
            assert len(rows)==count and len(set(reader.fieldnames))==len(reader.fieldnames) and all(None not in r for r in rows)
            tables[name]=rows
    assert len(tables['effects'])==13824 and len(tables['states'])==55296 and len(tables['member_reversions'])==16304 and len(tables['teacher_summary'])==384
    panel=load(SHAPE/'panel/data.npz');weight=np.zeros((72,3))
    for i,(r,d) in enumerate(panel['pairs']):weight[i,panel['labels'][d]]=1.;weight[i,panel['labels'][r]]=-1.
    data={}
    for seed in teacher.SEEDS:
        verify_folder(RUN/f'evaluation/seed_{seed}');raw=load(RUN/f'evaluation/seed_{seed}/data.npz')
        assert raw['logits'].shape==(72,2,16,4,3,3) and raw['channel_terms'].shape==(72,2,16,3,6,32)
        assert raw['drop_logits'].shape==(72,2,4,2,32,3) and all(np.isfinite(v).all() for v in raw.values())
        f=(raw['logits']*weight[:,None,None,None,None,:]).sum(-1)
        f00,f10,f01,f11=[f[:,:,:,i] for i in range(4)]
        t=np.stack((f10-f00,f01-f00,f11-f10-f01+f00,f11-f00,f11-f01,f11-f10),-1)
        close(t[...,3],t[...,0]+t[...,1]+t[...,2]);close(t[...,4],t[...,0]+t[...,2]);close(t[...,5],t[...,1]+t[...,2])
        close(t,raw['channel_terms'].sum(-1));close(t,raw['spatial_pn'].sum(-1));close(t[:,:,:,0],raw['branch_terms'].sum(-1));close(raw['branch_terms'][:,:,:,2,0],np.zeros((72,2,16)))
        close(raw['logits'][:,:,:,0,0],raw['logits'][:,:,:,0,1]);close(raw['logits'][:,:,:,3,0],raw['logits'][:,:,:,3,2])
        z=raw['point_z'];close(z[...,3]-z[...,1]-z[...,2]+z[...,0],np.zeros((72,2,16)))
        old=load(AR/f'evaluation/seed_{seed}/data.npz');prior=load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        focus=old['threshold'][:,:,3,0].astype(bool)&~old['threshold'][:,:,3,1].astype(bool)
        close(raw['logits'][:,:,:,3,0],prior['natural'][:,:,list(ENDPOINTS),2])
        close(raw['logits'][:,:,:,0,0],np.repeat(prior['natural'][:,:,0:1,2],16,axis=2))
        fixed=load(PRIOR/f'search/seed_{seed}/frozen.npz');search=load(PRIOR/f'search/seed_{seed}/data.npz')
        data[seed]=dict(raw=raw,task=f,terms=t,focus=focus,prior=prior,fixed=fixed,search=search)
    indexed={}
    for row in tables['effects']:
        seed,case,mask,endpoint,slot=[int(row[k]) for k in ('seed','case','mask','endpoint','slot')];m=(3,15).index(mask);d=data[seed];key=(case,m,slot);i=endpoint-1
        assert endpoint==ENDPOINTS[slot] and row['family']==FAMILIES[i%12//3] and row['variant']==VARIANTS[i%3] and row['strategy']==('aligned' if endpoint>=25 else 'moving_A')
        assert truth(row['focus539'])==bool(d['focus'][case,m])
        C=d['fixed']['channels'][case,m];C=C[C>=0];assert json.loads(row['C'])==C.tolist() and int(row['point'])==int(d['fixed']['lists'][case,m,0,0])
        direction=int(d['fixed']['half_direction'][case,m]);assert int(row['direction'])==direction
        eligible=bool(d['search']['choices'][case,m,i,2]);q=d['prior']['margins'][case,m,endpoint].min(-1)
        aligned=eligible and bool(np.all(q>=1e-6));own=eligible and bool(q[0]>=1e-6);physical=eligible and bool(np.all(q[1:]>=1e-6))
        assert truth(row['eligible'])==eligible and truth(row['prior_aligned'])==aligned and truth(row['prior_own'])==own and truth(row['prior_physical'])==physical
        status='not_met' if not (aligned if endpoint>=25 else own) else 'decomposition_only' if d['prior']['shifts'][case,m,endpoint,0]<=1e-10 else 'common_reference' if physical else 'reference_dependent'
        assert row['prior_status']==status
        primary=bool(d['focus'][case,m] and endpoint>=25 and i%3==0 and aligned);assert truth(row['primary_success'])==primary
        close([float(row['C_max_change']),float(row['outside_max_change'])],d['raw']['shifts'][key])
        t=d['terms'][key]
        for k,name in enumerate(TERMS):
            close(float(row[name]),t[0,k]);close(float(row[name+'_j_before_control']),t[0,k]-t[1,k]);close(float(row[name+'_j_after_control']),t[0,k]-t[2,k])
            close([float(row[name+'_'+b]) for b in ('affine','first_relu','second_relu')],d['raw']['branch_terms'][key][k])
        for k in (0,2,3,4):
            name=TERMS[k];pieces=d['raw']['channel_terms'][key][0,k];pn=d['raw']['spatial_pn'][key][0,k]
            close([float(row[name+'_channel_positive']),float(row[name+'_channel_negative'])],[sum(x for x in pieces if x>0),sum(x for x in pieces if x<0)])
            close([float(row[name+'_spatial_positive']),float(row[name+'_spatial_negative'])],pn)
            optional(row[name+'_channel_cancellation'],cancellation(pieces));optional(row[name+'_spatial_cancellation'],cancellation(pn))
        a,o,j,total,b,oa=t[0];optional(row['component_cancellation'],cancellation((a,o,j)))
        expected='context_enabled' if abs(a)<=1e-8 and abs(b)>1e-8 else 'both_C_effects_zero' if abs(a)<=1e-8 else 'unchanged' if abs(j)<=1e-8 else 'canceled_to_zero' if abs(b)<=1e-8 else 'reversed' if a*b<0 else 'enhanced' if abs(b)>abs(a) else 'attenuated'
        assert row['modulation']==expected and truth(row['standalone_offset'])==bool(a*o<0 and abs(a)>1e-8 and abs(o)>1e-8)
        z=d['raw']['point_z'][key];assert truth(row['j_new_flip'])==bool((z[0]>0)!=(z[3]>0))
        for field,s in (('j_before_target_side',0),('j_C_only_target_side',1),('j_outside_only_target_side',2),('j_after_target_side',3)):assert truth(row[field])==bool(direction*z[s]>1e-8)
        if status=='decomposition_only':close(t,np.zeros_like(t))
        if float(row['outside_max_change'])<=1e-10:close(t[:,[1,2,5]],np.zeros((3,3)))
        fullkey=(seed,case,mask,endpoint);assert fullkey not in indexed;indexed[fullkey]=row
    def retrieve(row):
        seed,case,mask,endpoint=[int(row[k]) for k in ('seed','case','mask','endpoint')]
        m=(3,15).index(mask);slot=ENDPOINTS.index(endpoint)
        return data[seed],(case,m,slot),indexed[(seed,case,mask,endpoint)]
    seen=set()
    for row in tables['states']:
        d,key,e=retrieve(row);state=('00_before','10_C_only','01_outside_only','11_after').index(row['state']);k=key+(state,)
        sk=tuple(row[n] for n in ('seed','case','mask','endpoint','state'));assert sk not in seen;seen.add(sk)
        assert truth(row['primary_success'])==truth(e['primary_success'])
        close([float(row[n]) for n in ('F','F_j_before','F_j_after')],d['task'][k]);close([float(row['point_z']),float(row['point_u'])],[d['raw']['point_z'][k],d['raw']['point_u'][k]])
        for mode,name in enumerate(('natural','j_before','j_after')):close([float(row[f'{name}_logit_{c}']) for c in range(3)],d['raw']['logits'][k][mode])
        close([int(row['first_gate_flips']),int(row['second_gate_flips'])],d['raw']['natural_flips'][k]);close([int(row['j_before_patch_second_flips']),int(row['j_after_patch_second_flips'])],d['raw']['patch_second_flips'][k])
    seen=set()
    for row in tables['channels']:
        d,key,e=retrieve(row);t=TERMS.index(row['term']);ch=int(row['channel']);v=d['raw']['channel_terms'][key][:,t,ch]
        ck=tuple(row[n] for n in ('seed','case','mask','endpoint','term','channel'));assert ck not in seen;seen.add(ck)
        close([float(row['contribution']),float(row['j_before_contribution']),float(row['j_after_contribution'])],[v[0],v[0]-v[1],v[0]-v[2]])
    seen=set()
    for row in tables['member_reversions']:
        d,key,e=retrieve(row);case,m,slot=key;di=(0,3,6,9).index(slot);state=1 if row['context']=='10_C_only' else 3;context=0 if state==1 else 1
        C=json.loads(e['C']);pos=C.index(int(row['channel']));dl=d['raw']['drop_logits'][case,m,di,context,pos];dz=d['raw']['drop_z'][case,m,di,context,pos]
        dk=tuple(row[n] for n in ('seed','case','mask','endpoint','context','channel'));assert dk not in seen;seen.add(dk)
        assert truth(row['primary_success'])==truth(e['primary_success'])
        df=float(dl@weight[case]);full=float(d['task'][key][state,0]);z=d['raw']['point_z'][key][state];direction=int(e['direction'])
        close([float(row[n]) for n in ('full_point_z','reverted_point_z','full_F','reverted_F','F_effect_of_member_change')],[z,dz,full,df,full-df]);close([float(row[f'logit{c}']) for c in range(3)],dl)
        assert truth(row['change_needed_for_target_side'])==bool(direction*z>1e-8 and direction*dz<-1e-8)
    for row in tables['teacher_summary']:
        group=[r for r in tables['effects'] if r['seed']==row['seed'] and r['endpoint']==row['endpoint']]
        cohort=row['cohort']
        if cohort!='all864':group=[r for r in group if truth(r['focus539'])]
        if cohort=='aligned_success539':group=[r for r in group if truth(r['prior_aligned'])]
        if cohort=='aligned_failed539':group=[r for r in group if not truth(r['prior_aligned'])]
        assert len(group)==int(row['count'])
        for name,value in row.items():
            if name.endswith('_rms'):
                expected=float(np.sqrt(np.mean([float(r[name[:-4]])**2 for r in group]))) if group else None
                optional(value,expected)
    primary=[r for r in tables['effects'] if truth(r['primary_success'])];assert len(primary)==claims['primary_records']==26
    assert len({(r['seed'],r['case'],r['mask']) for r in primary})==claims['primary_distinct_states']==15
    assert len({r['seed'] for r in primary})==claims['primary_models']==5
    assert sum(float(r['C_max_change'])<=1e-10 for r in primary)==claims['C_unchanged']==0
    assert sum(float(r['outside_max_change'])<=1e-10 for r in primary)==claims['outside_unchanged']==12
    assert sum(truth(r['standalone_offset']) for r in primary)==claims['standalone_offsets']==2
    selected_members=[r for r in tables['member_reversions'] if truth(r['primary_success']) and r['context']=='11_after']
    assert len(selected_members)==claims['primary_member_reversions']==58
    assert sum(truth(r['change_needed_for_target_side']) for r in selected_members)==claims['member_changes_needed']==25
    assert sum(truth(r['focus539']) and r['strategy']=='moving_A' and r['prior_status']=='decomposition_only' for r in tables['effects'])==claims['decomposition_controls']==4
    patterns=Counter()
    for row in primary:
        name='already_on_target_side_or_same_gate' if not truth(row['j_new_flip']) else ('C_yes' if truth(row['j_C_only_target_side']) else 'C_no')+'_'+('outside_yes' if truth(row['j_outside_only_target_side']) else 'outside_no')
        patterns[name]+=1
    assert dict(patterns)==claims['point_patterns']
    assert dict(Counter(r['modulation'] for r in primary))==claims['modulation_counts'] and dict(Counter(r['family'] for r in primary))==claims['primary_family_counts']
    for rec in claims['teacher_equal']:
        selected=[r for r in tables['teacher_summary'] if r['family']==rec['family'] and r['variant']=='selected' and r['strategy']=='aligned' and r['cohort']=='aligned_success539' and int(r['count'])>0]
        assert rec['count']==sum(int(r['count']) for r in selected) and rec['models']==len(selected)
        for k,v in rec.items():
            if k.endswith('_rms'):
                if selected:close(v,np.mean([float(r[k]) for r in selected]))
                else:assert v is None
    assert sum(float(r['outside_max_change'])>1e-10 for r in primary)==supp['active_outside_records']==14
    assert sum(abs(float(r['interaction_first_relu']))>1e-8 and abs(float(r['interaction_second_relu']))>1e-8 and float(r['interaction_first_relu'])*float(r['interaction_second_relu'])<0 for r in primary)==supp['opposed_relu_interactions']==7
    visual=teacher.read(RUN/'visual_review.json');assert visual['status']=='PASS' and [r['path'] for r in visual['files']]==supp['figures'] and len(visual['files'])==6
    for r in visual['files']:assert teacher.sha(BASE/r['path'])==r['sha256']
    if deep:assert preservation()==protocol['preservation']
    result=dict(status='PASS',utc=teacher.now(),tables=len(tables),rows=sum(len(v) for v in tables.values()),native_fields=audit['native_fields'],point_patches=audit['point_patches'],member_reversions=audit['member_reversions'],old_entries=1650,files=len(records),bytes=sum(r['bytes'] for r in records),independent_maximum_error=max(audit['maximum_errors'].values()))
    if before_manifest:
        assert not (RUN/'delivery_manifest.json').exists();teacher.dump(RUN/'table_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--before-manifest',action='store_true');parser.add_argument('--deep-inputs',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.before_manifest,args.deep_inputs)),flush=True)

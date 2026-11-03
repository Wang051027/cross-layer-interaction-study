"""三阶通道与门控接续：协议、半强度定位、统一封存、两强度评价、汇总。

旧五轮交付943条文件只读。用已见36图定位当前机制，不宣称新泛化。
三元组及正反向开关子集都来自半强度，评价期间不换名单。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import dsprites_teacher as teacher
import numpy as np
import torch
import channel_gate_mechanism as previous
import triple_gate_mechanism as triple
from run_dsprites_channel_gates import middle_states
from test_dsprites_channel_gates import forward_override

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_triple_gates_20261011'
PRIOR=BASE/'runs/dsprites_channel_gates_20261011';SHAPE=BASE/'runs/dsprites_shape_donor_20261010'
CODE=('triple_gate_mechanism.py','run_dsprites_triple_gates.py','test_dsprites_triple_gates.py')
OLDER=('dsprites_20260929','dsprites_factor_transfer_20261007','dsprites_gradient_20261009','dsprites_shape_donor_20261010','dsprites_channel_gates_20261011')


def load(path):
    """读取纯数值文件，关闭文件句柄，禁止pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def event(kind,**details):
    """日志只属于本轮，不调用会写旧目录的teacher.event。"""
    value=dict(utc=teacher.now(),event=kind,**details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def finish(folder,started):
    """完整数据与核验落盘后才标记完成，未完成目录拒绝覆盖。"""
    teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seconds=time.perf_counter()-started,
        files={p.name:teacher.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))


def verify_folder(folder):
    """已完成阶段先逐文件验哈希再接续。"""
    for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name


def verify_protocol():
    """数值源码与输入由协议锁定，评价不允许临时改变算法。"""
    p=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    return p


def init(device):
    """冻结已见数据背景、主比较、符号分组规则和全部旧证据。"""
    assert not RUN.exists();preservation={}
    for name in OLDER:
        path=BASE/f'runs/{name}/delivery_manifest.json';old=teacher.read(path)
        assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip()
        for item in old['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[str(path.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(path),files=len(old['files']))
    legacy=teacher.verify_old();preflight=BASE/'reports/dsprites_triple_gates_preflight_20261011.json'
    assert teacher.read(preflight)['status']=='PASS'
    inputs=[SHAPE/'panel/data.npz',PRIOR/'protocol.json',PRIOR/'gate_freeze.json',PRIOR/'design.npz',preflight]
    inputs +=[PRIOR/f'discovery/seed_{s}/data.npz' for s in teacher.SEEDS]
    inputs +=[PRIOR/f'evaluation/seed_{s}/alpha_{a}/data.npz' for s in teacher.SEEDS for a in (1.,.5)]
    inputs +=[teacher.RUN/f'seed_{s}/teacher.pt' for s in teacher.SEEDS]
    helpers=('dsprites_teacher.py','residual_teacher.py','external_experiment.py','research_runtime.py',
        'channel_gate_mechanism.py','run_dsprites_channel_gates.py','test_dsprites_channel_gates.py',
        'shape_donor_mechanism.py','audit_dsprites_gradient.py')
    inputs +=[BASE/'scripts'/n for n in helpers]
    RUN.mkdir();shutil.copy2(preflight,RUN/'preflight.json')
    for name in CODE:
        destination=RUN/'source/scripts'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'scripts'/name,destination)
    np.savez_compressed(RUN/'design.npz',**triple.design())
    p=dict(utc=teacher.now(),version='triple-gates-v1',seeds=teacher.SEEDS,
        prior_knowledge='Oct11 pair effects,4-list coefficients,orders3+ aggregate and gate interventions already inspected; same-panel mechanistic follow-up',
        panel='same36 images,72 directed shape cases,6 frozen teachers; shared images not independent cases',
        shortlist='inherit Oct11 eight physical channel order; no semantic alignment across independently trained teachers',
        states='95 masks: all0..3-member subsets,old first4 complete mask15,full8mask255',
        pure_triples='all56 at empty middle-patch background; local state order empty,A,B,AB,C,AC,BC,ABC',
        signs=triple.SIGNS.tolist(),hierarchy='joint8=sum28pairs+sum56puretriples+orders4..8aggregate',
        selection='top3 absolute half-strength empty-background triple effects per case/operation; fixed before full evaluation',
        contexts='empty background and all other5 shortlisted channels patched; latter only same selected3 triples, not exhaustive conditional56',
        gate_selection='each selected triple/context/stage largest absolute coordinate contributions among half-switched sites,up to128;32is prefix',
        sign_groups='split selected128 by half-contribution times half-triple sign positive/negative; zero total usespositive sign; zeros in neither group',
        random='two fixed permutations of same half-switched-site pool and actual budget;RNGseed=teacher_seed+2026101120+100*case+operation',
        primary=dict(alpha=1.,context=0,rank=0,variant='selected128',metric='per-teacher RMS held triple effect / RMS natural triple effect,thenmean6',comparison='two same-budget fixed random lists,each ReLU stage and operation'),
        secondary='same selected3 in complement background;budget32;half replication;aligned/opposed subsets;all56 and hierarchical signed parts;actual single-site example',
        single_example='first teacher,case0,half rank0 triple,first8 frozen coordinates,each alone,both contexts/stages/operations/strengths',
        gate_values='held bit is baseline of current context/strength/operation; coordinates and sign-group membership frozen at half strength',
        interpretation='conditional triple differs from pure coefficient by strict-superset order4+ terms; first-stage algebra not net causal effect; last-stage removal exact; synthetic internal gate clamp can transmit negative activation',
        sources={f'scripts/{n}':teacher.sha(BASE/'scripts'/n) for n in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in inputs},
        preservation=preservation,legacy_manifests=legacy,
        environment=dict(torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii')
    event('protocol_locked',sha256=teacher.sha(RUN/'protocol.json'))


@torch.no_grad()
def calculate(model,base,reference,channels,weight,d,rng,frozen=None,example=False):
    """所有全局指标和局部干预共用同一中介状态；评价不调用选择函数。"""
    x=previous.patch_states(base,reference,channels,d['bits']);global_trace=previous.trace_modes(model,x)
    summary=triple.global_summary(global_trace,d)
    selected=np.argsort(-np.abs(summary['triple_logits'][:,0]@weight),kind='stable')[:3] if frozen is None else frozen['selected_triples']
    output={**summary,'selected_triples':selected};rank_records=[];single=[];maximum={}
    def check(key,value):
        err=float(np.max(np.abs(value)));assert err<1e-9,(key,err);maximum[key]=max(maximum.get(key,0.),err)
    check('affine_global',summary['triple_logits'][:,3]);check('order_reconstruction',summary['order_totals'].sum(0)-summary['joint_logits'][1])
    for rank,p in enumerate(selected):
        contexts=[];example_contexts=[]
        for context in range(2):
            t=triple.local_trace(model,base,reference,channels,d['triples'][p],context,d,global_trace)
            logits=t['logits'].cpu().numpy();effect=triple.contrast(logits)
            parts=np.stack((effect[2]-effect[3],effect[0]-effect[2]))
            unit=triple.unit_contributions(model,t,weight)
            check('unit_sum',unit.sum((1,2,3)).cpu().numpy()-parts@weight);check('affine_local',effect[3])
            if context==0:check('global_local_triple',effect-summary['triple_logits'][p])
            selection=triple.select_sites(t,unit,float(effect[0]@weight),rng) if frozen is None else {k:frozen[k][rank,context] for k in triple.SELECTION_KEYS}
            row=dict(context_logits=logits,context_effects=effect,context_stageparts=parts,
                **triple.details(t,unit,selection),**selection)
            if frozen is not None:
                row['partial_logits']=triple.partial_holds(model,t,selection)
                last=triple.contrast(np.moveaxis(row['partial_logits'][1],1,0))@weight
                check('last_gate_removal',last-(effect[0]@weight-row['variant_contribution_sums'][1]))
            if example and rank==0:
                site=[]
                for stage in range(2):
                    ids=selection['selected_gates'][stage,:8];values=[]
                    for unit_id in ids:
                        valid=np.array([unit_id]) if unit_id>=0 else np.array([],dtype=np.int64)
                        values.append(forward_override(model,t['x'],t['x'][:1],stage=stage,ids=valid).cpu().numpy())
                    site.append(values)
                example_contexts.append(site)
            contexts.append(row)
        rank_records.append({k:np.asarray([r[k] for r in contexts]) for k in contexts[0]})
        if example and rank==0:single=np.asarray(example_contexts)
    output.update({k:np.asarray([r[k] for r in rank_records]) for k in rank_records[0]})
    return output,maximum,single


def seal():
    """全部6模型定位结束后一次封存，不能边看全强度边选择。"""
    verify_protocol();assert not (RUN/'selection_freeze.json').exists();files={}
    for seed in teacher.SEEDS:
        folder=RUN/f'discovery/seed_{seed}';verify_folder(folder)
        for p in folder.iterdir():files[str(p.relative_to(RUN)).replace('\\','/')]=teacher.sha(p)
    teacher.dump(RUN/'selection_freeze.json',dict(utc=teacher.now(),protocol_sha256=teacher.sha(RUN/'protocol.json'),files=files))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n',encoding='ascii')
    event('selection_sealed',files=len(files))


def verify_seal():
    """封存必须与当前协议和每个定位文件一致。"""
    verify_protocol();p=RUN/'selection_freeze.json';assert teacher.sha(p)==p.with_suffix('.sha256').read_text().strip()
    assert teacher.read(p)['protocol_sha256']==teacher.sha(RUN/'protocol.json')
    for name,value in teacher.read(p)['files'].items():assert teacher.sha(RUN/name)==value,name


def run(device,evaluate=False):
    """半强度定位不运行部分干预；全强度和半强度评价都复用冻结名单。"""
    verify_seal() if evaluate else verify_protocol()
    if not evaluate:assert not (RUN/'selection_freeze.json').exists()
    stage='evaluation' if evaluate else 'discovery';(RUN/stage).mkdir(exist_ok=True)
    panel=load(SHAPE/'panel/data.npz');d=triple.design()
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for parameter in model.parameters():parameter.requires_grad_(False)
        old=load(PRIOR/f'discovery/seed_{seed}/data.npz');channels=old['channels']
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz') if evaluate else None
        for alpha in ((1.,.5) if evaluate else (.5,)):
            folder=RUN/f'{stage}/seed_{seed}'
            if evaluate:folder=folder/f'alpha_{alpha}'
            if (folder/'completion.json').exists():verify_folder(folder);continue
            assert not folder.exists();folder.mkdir(parents=True);started=time.perf_counter();values={};maxima={};example=None
            baseline,endpoint=middle_states(model,panel,device,alpha)
            for i,(recipient,donor) in enumerate(panel['pairs']):
                weight=np.zeros(3);weight[panel['labels'][donor]]=1;weight[panel['labels'][recipient]]=-1;case=[];single=[]
                for op in range(2):
                    base,reference=(endpoint[i:i+1],baseline[recipient:recipient+1]) if op==0 else (baseline[recipient:recipient+1],endpoint[i:i+1])
                    frozen_case={k:frozen[k][i,op] for k in ('selected_triples',*triple.SELECTION_KEYS)} if evaluate else None
                    rng=np.random.default_rng(seed+2026101120+100*i+op)
                    output,checks,sites=calculate(model,base,reference,channels[i],weight,d,rng,frozen_case,evaluate and seed==teacher.SEEDS[0] and i==0)
                    case.append(output);single.append(sites)
                    for key,value in checks.items():maxima[key]=max(maxima.get(key,0.),value)
                for key in case[0]:values.setdefault(key,[]).append(np.asarray([c[key] for c in case]))
                if evaluate and seed==teacher.SEEDS[0] and i==0:example=np.asarray(single)
                if (i+1)%24==0:event(stage+'_progress',seed=seed,alpha=alpha,cases=i+1,total=72)
            np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()},channels=channels,pairs=panel['pairs'])
            if example is not None:np.savez_compressed(folder/'example.npz',single_logits=example)
            teacher.dump(folder/'checks.json',dict(status='PASS',maximum=maxima));finish(folder,started)
            event(stage+'_complete',seed=seed,alpha=alpha,seconds=time.perf_counter()-started)


def summarize():
    """每模型内部RMS和带符号投影；符号子集只作机制验证不作等预算排名。"""
    verify_seal();assert not (RUN/'summary.json').exists();panel=load(SHAPE/'panel/data.npz')
    w=np.zeros((72,3));w[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1;w[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    rows=[];hierarchy=[]
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz')
            natural=np.einsum('norcmk,nk->norcm',raw['context_effects'],w)[...,0]
            partial=np.einsum('norcsvtk,nk->norcsvt',raw['partial_logits'],w)
            held=np.einsum('norcsvt,t->norcsv',partial,triple.SIGNS)
            joint=np.einsum('nobmk,nk->nobm',raw['joint_logits'],w)[:,:,1,0]
            pieces=np.einsum('nobmk,nk->nobm',raw['order_totals'],w)[...,0]
            for op in range(2):
                den=float(np.mean(joint[:,op]**2));assert den>1e-20
                for j,name in enumerate(('pairs','pure_triples','orders4plus')):
                    part=pieces[:,op,j];hierarchy.append(dict(seed=seed,alpha=alpha,operation=('restore','transplant')[op],order=name,
                        rms_ratio=float(np.sqrt(np.mean(part**2)/den)),signed_projection=float(np.mean(part*joint[:,op])/den)))
                for ranks in ((0,),(0,1,2)):
                    for context in range(2):
                        denominator=float(np.mean(natural[:,op,list(ranks),context]**2));assert denominator>1e-20
                        for stage in range(2):
                            for variant,name in enumerate(triple.VARIANTS):
                                response=held[:,op,list(ranks),context,stage,variant]
                                original=natural[:,op,list(ranks),context]
                                rows.append(dict(seed=seed,alpha=alpha,operation=('restore','transplant')[op],ranks='first' if len(ranks)==1 else 'three',
                                    context=('empty','complement')[context],stage=stage+1,variant=name,
                                    natural_rms=float(np.sqrt(denominator)),remaining_ratio=float(np.sqrt(np.mean(response**2)/denominator)),
                                    signed_remaining=float(np.mean(original*response)/denominator),
                                    mean_gate_count=float(np.mean(raw['variant_counts'][:,op,list(ranks),context,stage,variant]))))
    teacher.dump(RUN/'summary.json',dict(utc=teacher.now(),rows=rows,hierarchy=hierarchy));event('summary_complete',rows=len(rows),hierarchy=len(hierarchy))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('init','discover','seal','evaluate','summarize'))
    args=parser.parse_args();device=teacher.configure()
    if args.stage=='init':init(device)
    elif args.stage=='discover':run(device)
    elif args.stage=='seal':seal()
    elif args.stage=='evaluate':run(device,True)
    else:summarize()

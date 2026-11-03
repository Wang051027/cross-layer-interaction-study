"""同批输入的中间组合残差路由：冻结→半强度定位→封存→全强度验证。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import residual_route_mechanism as core
from residual_route_path import integrate
from layer_transfer_mechanism import design,lower_fields
from shape_donor_mechanism import replace_groups,pair_weights

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_residual_routes_20261013'
PRIOR=BASE/'runs/dsprites_layer_transfer_20261013';SHAPE=BASE/'runs/dsprites_shape_donor_20261010'
CODE=('residual_route_mechanism.py','residual_route_path.py','run_dsprites_residual_routes.py','test_dsprites_residual_routes.py')
OLDER=('dsprites_20260929','dsprites_factor_transfer_20261007','dsprites_gradient_20261009','dsprites_shape_donor_20261010',
    'dsprites_channel_gates_20261011','dsprites_triple_gates_20261011','dsprites_layer_transfer_20261013')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def event(name,**values):
    row=dict(utc=teacher.now(),event=name,**values)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def verify_protocol():
    p=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    for name,value in p['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    return p


def verify_folder(folder):
    for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name


def init():
    assert not RUN.exists();pre=BASE/'reports/dsprites_residual_routes_preflight_20261013.json';assert teacher.read(pre)['status']=='PASS'
    preservation={}
    for name in OLDER:
        path=BASE/f'runs/{name}/delivery_manifest.json';old=teacher.read(path)
        assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip()
        for item in old['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[str(path.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(path),files=len(old['files']))
    teacher.verify_old();RUN.mkdir();shutil.copy2(pre,RUN/'preflight.json')
    inputs=[pre,SHAPE/'panel/data.npz']+[teacher.RUN/f'seed_{s}/teacher.pt' for s in teacher.SEEDS]
    inputs +=[PRIOR/f'evaluation/seed_{s}/alpha_{a}/data.npz' for s in teacher.SEEDS for a in (.5,1.)]
    inputs +=[BASE/'scripts'/n for n in ('dsprites_teacher.py','residual_teacher.py','external_experiment.py','research_runtime.py',
        'shape_donor_mechanism.py','layer_transfer_mechanism.py','audit_dsprites_gradient.py','dsprites_mechanism.py','cross_layer_mechanism.py','test_dsprites_layer_transfer.py')]
    for name in CODE:
        destination=RUN/'source/scripts'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'scripts'/name,destination)
    np.savez_compressed(RUN/'design.npz',**design())
    p=dict(utc=teacher.now(),version='residual-routes-v1',scope='same seen36images/72directedpairs/6frozen teachers;no new input or training',
        question='which physical middle channels and downstream ReLU coordinates carry source-combination residual to output;how signed effects cancel',
        residual='R(S)=H(S)-H<=1(S);baseline A(S) is source-state-specific,not shared empty-state',
        identity='deltaF(S)=J_A(S)R(S)+p_A(S)*first_ReLU_remainder(S)+q*second_ReLU_remainder(S)',
        gate_reference='hold selected coordinates to gates at A(S) for each source state;first-stage intervention recomputes second-stage naturally',
        selection_strength=.5,test_strength=1.,prior_knowledge='previous36input results all seen;current route ranking only half strength;full not reranked',
        primary=['selected8 channel restoration error fordeltaI vs two random8 lists','selected128 gate holds at each ReLU,deltaI remainingRMS vs two random lists from same meaningful remainder pool'],
        secondary=['channel4/8 erasure andrestore;positive/negative8','gatepositive/negative128 and3one-coordinate holds','full first/second/both holds','per-coordinate and layer cancellation'],
        scores='projection of24conditional single-channel restore or gate direct terms onto actual24deltaI;stable physical-index tie breaks;abs score<1e-10 setzero',
        gate_pool='any source state abs(unweighted ReLU remainder)>1e-10 at half strength;fixed pool,actual budget=min128,poolsize',
        channel_pool='all32 physical middle channels;two fixed random permutations',random_seed='teacher_seed+20261013000+case*97',
        path_examples='teacher202609291,case0,masks3(G0/G1) and15(all4),both strengths;no extra cases',
        path='analytic first roots then second roots within each first interval;midpoint original autograd;merge tolerance1e-12;integral/error<1e-9',
        effect='deltaI=contrast(F(H)-F(A));not natural whole contrast;nonunique baseline-dependent computational decomposition',
        limitations='seen narrow panel;A cannegative/off-manifold;coordinate/channel scores not unique causal shares;half signs may change;no universal circuit/semantic/generalization claims',
        sources={f'scripts/{n}':teacher.sha(BASE/'scripts'/n) for n in CODE},inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in inputs},preservation=preservation)
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii');event('protocol_locked',sha256=teacher.sha(RUN/'protocol.json'))


def frame(seed,alpha,device,selection=None):
    panel=load(SHAPE/'panel/data.npz');d=design();model=teacher.load_teacher(seed,device)
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():sources=torch.cat([model.block1(model.stem(x)) for x in torch.as_tensor(panel['images'][:,None],device=device,dtype=torch.float64).split(16)])
    bits=torch.as_tensor(d['bits'],device=device);values={};selected={};paths={};fields={}
    old=load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
    for case,(r,donor) in enumerate(panel['pairs']):
        with torch.no_grad():
            h=model.block2(replace_groups(sources[r:r+1],sources[donor:donor+1],bits,alpha));a=lower_fields(h,d)[1][0]
        w=pair_weights(panel['labels'][r],panel['labels'][donor]);fixed=None if selection is None else {k:v[case] for k,v in selection.items()}
        out,chosen,details=core.compute(model,a,h,w,d['stencils'],fixed,seed+20261013000+case*97,seed==teacher.SEEDS[0] and case==0)
        assert np.max(abs(out['modes'][:,0]-old['logits'][case,:,0]))<1e-9
        assert np.max(abs(out['baseline']-old['lower_logits'][case,0]))<1e-9
        for key,val in out.items():values.setdefault(key,[]).append(val)
        for key,val in chosen.items():selected.setdefault(key,[]).append(val)
        if details:
            fields=details
            for mask in (3,15):paths[mask]=integrate(model,a[mask:mask+1],h[mask:mask+1],w)
        if (case+1)%24==0:event('frame_progress',seed=seed,alpha=alpha,cases=case+1)
    return {k:np.asarray(v) for k,v in values.items()},{k:np.asarray(v) for k,v in selected.items()},fields,paths


def save_frame(folder,seed,alpha,result):
    assert not folder.exists();folder.mkdir(parents=True);raw,selection,fields,paths=result
    np.savez_compressed(folder/'data.npz',**raw);np.savez_compressed(folder/'selection.npz',**selection)
    if fields:np.savez_compressed(folder/'example_fields.npz',**fields)
    for mask,value in paths.items():np.savez_compressed(folder/f'path_mask{mask}.npz',**value)
    teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seed=seed,alpha=alpha,files={p.name:teacher.sha(p) for p in folder.iterdir() if p.is_file()}))


def discovery(device):
    verify_protocol();assert not (RUN/'selection_freeze.json').exists()
    for seed in teacher.SEEDS:
        folder=RUN/f'discovery/seed_{seed}'
        if folder.exists():verify_folder(folder);continue
        started=time.perf_counter();save_frame(folder,seed,.5,frame(seed,.5,device));event('discovery_complete',seed=seed,seconds=time.perf_counter()-started)


def seal():
    verify_protocol();assert not (RUN/'selection_freeze.json').exists();files={}
    for seed in teacher.SEEDS:
        folder=RUN/f'discovery/seed_{seed}';verify_folder(folder)
        for p in folder.iterdir():files[str(p.relative_to(RUN)).replace('\\','/')]=teacher.sha(p)
    teacher.dump(RUN/'selection_freeze.json',dict(utc=teacher.now(),protocol_sha256=teacher.sha(RUN/'protocol.json'),files=files))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n',encoding='ascii');event('selection_locked')


def verify_seal():
    verify_protocol();f=teacher.read(RUN/'selection_freeze.json');assert teacher.sha(RUN/'selection_freeze.json')==(RUN/'selection_freeze.sha256').read_text().strip()
    for name,value in f['files'].items():assert teacher.sha(RUN/name)==value,name
    return f


def evaluate(device):
    verify_seal()
    for seed in teacher.SEEDS:
        selected=load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            if folder.exists():verify_folder(folder);continue
            if alpha==.5:
                folder.mkdir(parents=True);origin=RUN/f'discovery/seed_{seed}'
                for p in origin.iterdir():
                    if p.name!='completion.json':shutil.copy2(p,folder/p.name)
                teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seed=seed,alpha=alpha,reuse='sealed half-strength discovery',files={p.name:teacher.sha(p) for p in folder.iterdir()}))
            else:save_frame(folder,seed,alpha,frame(seed,alpha,device,selected))
            event('evaluation_complete',seed=seed,alpha=alpha)


def stats(target,response):
    den=float(np.mean(target**2));root=np.sqrt(den)
    return dict(target_rms=float(root),remaining_ratio=None if den<1e-20 else float(np.sqrt(np.mean(response**2)/den)),
        error_ratio=None if den<1e-20 else float(np.sqrt(np.mean((target-response)**2)/den)),signed_removal=None if den<1e-20 else float(np.mean(target*(target-response))/den))


def summarize():
    verify_seal();assert not (RUN/'summary.json').exists();d=design();panel=load(SHAPE/'panel/data.npz');w=np.array([pair_weights(panel['labels'][r],panel['labels'][s]) for r,s in panel['pairs']]);rows=[];parts=[]
    cp=lambda x:x[:,d['stencils'][:,0]]-x[:,d['stencils'][:,1]]-x[:,d['stencils'][:,2]]+x[:,d['stencils'][:,3]]
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz');base=np.einsum('nsk,nk->ns',raw['baseline'],w)
            modes=np.einsum('nsmk,nk->nsm',raw['modes'],w)-base[:,:,None];target=cp(modes[:,:,0])
            for j,name in enumerate(('natural','hold_first_all','hold_second_all','hold_both_all')):rows.append(dict(seed=seed,alpha=alpha,family='full',method=name,operation='hold',**stats(target,cp(modes[:,:,j]))))
            ch=np.einsum('nvosk,nk->nvos',raw['channel_logits'],w)-base[:,None,None,:]
            for v,name in enumerate(core.CHANNEL_SPECS):
                for op,index in (('erase',0),('restore',1)):rows.append(dict(seed=seed,alpha=alpha,family='channel',method=name,operation=op,**stats(target,cp(ch[:,v,index]))))
            gate=np.einsum('navsk,nk->navs',raw['gate_logits'],w)-base[:,None,None,:]
            for stage in (0,1):
                for v,name in enumerate(core.GATE_SPECS):rows.append(dict(seed=seed,alpha=alpha,family=f'gate{stage+1}',method=name,operation='hold',**stats(target,cp(gate[:,stage,v]))))
            linear=cp(raw['linear_channel']);g1=cp(raw['gate_channel'][:,0]);g2=cp(raw['gate_channel'][:,1]);components=np.stack((linear.sum(2),g1.sum(2),g2.sum(2)),2)
            assert np.max(abs(components.sum(2)-target))<1e-9
            den=float(np.mean(target**2));abs_units=np.abs(linear).sum()+raw['gate_signed_mass'][...,0].sum()-raw['gate_signed_mass'][...,1].sum()
            parts.append(dict(seed=seed,alpha=alpha,rms=rms(target),component_rms_ratios=[float(rms(components[:,:,j])/np.sqrt(den)) for j in range(3)],
                component_projection=[float(np.mean(components[:,:,j]*target)/den) for j in range(3)],
                coordinate_cancellation=float(1-np.abs(target).sum()/abs_units),layer_cancellation=float(1-np.abs(target).sum()/np.abs(components).sum())))
    teacher.dump(RUN/'summary.json',dict(utc=teacher.now(),rows=rows,decomposition=parts));event('summary_complete',rows=len(rows))


def rms(x):return float(np.sqrt(np.mean(x**2)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('init','discovery','seal','evaluate','summarize'));args=parser.parse_args()
    if args.stage=='init':init()
    elif args.stage=='seal':seal()
    elif args.stage=='summarize':summarize()
    else:
        device=teacher.configure()
        if args.stage=='discovery':discovery(device)
        else:evaluate(device)

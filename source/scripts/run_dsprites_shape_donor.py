"""形状配对与梯度传递：init → fit → freeze → select → seal → evaluate → summarize。

旧数据、权重与交付只读。新候选是32个完整中介通道，旧标量名单不混用。
固定规则只在旧144开发图形成的288有向配对上拟合，半强度名单封存后
才读取新面板全强度响应。主要比较预算4，预算8只作预设补充。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time

import dsprites_teacher as teacher
import numpy as np
import torch
import dsprites_mechanism as mechanism
import gradient_mechanism as old_gradient
import shape_donor_mechanism as donor

BASE,DATA,OLD=teacher.BASE,teacher.DATA,teacher.RUN
RUN=BASE/'runs/dsprites_shape_donor_20261010'
FULL=BASE/'data/dsprites'
PREVIOUS=('dsprites_20260929','dsprites_factor_transfer_20261007','dsprites_gradient_20261009')
SEEDS=teacher.SEEDS
STRENGTHS=(1.,.5)
METHODS=('global_fixed','pair_fixed','fixed_linear','state_linear','input_finite',
         'random_0','random_1','random_2','random_3')
BUDGETS=(4,8)
SPECS=[dict(method=m,budget=k) for k in BUDGETS for m in METHODS]+[dict(method='all_channels',budget=32)]
CODE=('scripts/shape_donor_mechanism.py','scripts/run_dsprites_shape_donor.py','scripts/test_dsprites_shape_donor.py')


def event(kind,**details):
    """新日志不会写入旧events.jsonl，也不会进入评分规则。"""
    value=dict(utc=teacher.now(),event=kind,**details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def complete(folder,started):
    """完成标记只在所有原始文件落盘后写入；中断残留不自动覆盖。"""
    teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seconds=time.perf_counter()-started,
        files={p.name:teacher.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))


def verify_folder(folder):
    """续跑只接受完成且未改动的阶段。"""
    for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,str(folder/name)


def verify_protocol():
    """核对冻结代码、教师、开发数据和面板，拒绝运行途中换规则。"""
    p=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['sources'].items():
        assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    verify_folder(RUN/'panel');return p


def freeze(name,stage):
    """六教师的该阶段全部完成后，统一封存，下一阶段才能读取新响应。"""
    verify_protocol();assert not (RUN/f'{name}.json').exists();files={}
    for seed in SEEDS:
        folder=RUN/stage/f'seed_{seed}';verify_folder(folder)
        files.update({str(p.relative_to(RUN)).replace('\\','/'):teacher.sha(p) for p in folder.iterdir()})
    teacher.dump(RUN/f'{name}.json',dict(utc=teacher.now(),protocol_sha256=teacher.sha(RUN/'protocol.json'),files=files))
    (RUN/f'{name}.sha256').write_text(teacher.sha(RUN/f'{name}.json')+'\n',encoding='ascii');event(name,files=len(files))


def verify_freeze(name):
    """名单及固定模型封存后每次用前只读复核。"""
    verify_protocol();p=teacher.read(RUN/f'{name}.json')
    assert teacher.sha(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
    for path,value in p['files'].items():assert teacher.sha(RUN/path)==value,path


def digest(pixel):
    """像素身份去重，排除不同因素标签对应同一旋转对称图像的情况。"""
    return hashlib.sha256(np.ascontiguousarray(pixel).tobytes()).hexdigest()


def prepare_panel():
    """未访问模型输出，选6大小×2共同未用角度×3形状，固定位置15、15。

    角度仍由0和π的周期方向覆盖代价选取。排除前两轮新增面板之后，
    合格共同角度仅17、22，角度覆盖局限会在报告中明确披露。
    """
    folder=RUN/'panel';folder.mkdir();started=time.perf_counter();previous=set()
    for split in ('train','validation','evaluation','translated'):
        previous.update(digest(x) for x in np.load(DATA/f'{split}_images.npy',mmap_mode='r',allow_pickle=False))
    for name in PREVIOUS[1:]:
        with np.load(BASE/f'runs/{name}/panel/data.npz',allow_pickle=False) as z:previous.update(digest(x) for x in z['images'])
    images=np.load(FULL/'imgs.npy',mmap_mode='r',allow_pickle=False)
    latent=np.load(FULL/'latents_classes.npy',mmap_mode='r',allow_pickle=False)
    physical=np.load(FULL/'latents_values.npy',mmap_mode='r',allow_pickle=False)
    def ids(scale,angle):return (((np.arange(3)*6+scale)*40+angle)*32+15)*32+15
    eligible=[[a for a in range(40) if not {digest(images[i]) for i in ids(s,a)}&previous] for s in range(6)]
    common=sorted(set.intersection(*(set(x) for x in eligible)));candidates=[]
    for angles in itertools.combinations(common,2):
        index=np.concatenate([ids(s,a) for s in range(6) for a in angles])
        if len({digest(images[i]) for i in index})!=36:continue
        theta=np.array([physical[ids(0,a)[0],3] for a in angles])
        cost=min(float(np.square(np.angle(np.exp(1j*(theta[list(order)]-[0.,np.pi])))).sum()) for order in itertools.permutations(range(2)))
        candidates.append((round(cost,12),angles))
    assert candidates,'No pixel-disjoint matched panel'
    cost,angles=min(candidates);index=np.concatenate([ids(s,a) for s in range(6) for a in angles])
    pixels=np.asarray(images[index]);labels=np.asarray(latent[index,1]);factors=np.asarray(latent[index])
    pairs=donor.matched_pairs(factors,labels);hashes=[digest(x) for x in pixels]
    assert len(pairs)==72 and len(set(hashes))==36 and not set(hashes)&previous
    np.savez_compressed(folder/'data.npz',images=pixels,labels=labels,factors=factors,physical=np.asarray(physical[index]),
        official_ids=index,pairs=pairs,pixel_areas=pixels.sum((1,2)))
    teacher.dump(folder/'audit.json',dict(status='PASS',images=36,directed_pairs=72,conditions=12,
        chosen_angles=list(angles),common_angles=common,eligible_by_scale=eligible,candidate_panels=len(candidates),
        coverage_cost=cost,pixel_overlap=0,hashes=hashes,
        scope='two new center orientations and pixels, narrow angular coverage, not globally rotation-OOD'))
    complete(folder,started);event('panel_prepared',images=36,pairs=72,angles=list(angles))


def initialize(device):
    """保全三轮交付，记录已知旧结果，冻结新的干预单位与主要指标。"""
    assert not RUN.exists(),'Preserve existing run';teacher.verify_protocol();preservation={}
    for name in PREVIOUS:
        manifest=BASE/f'runs/{name}/delivery_manifest.json';previous=teacher.read(manifest)
        for item in previous['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[str(manifest.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(manifest),files=len(previous['files']))
    legacy=teacher.verify_old();inputs={}
    paths=[DATA/f'development_{kind}.npy' for kind in ('images','labels','factors')]
    paths +=[DATA/'manifest.json',FULL/'arrays.json',FULL/'latents_values.npy',FULL/'latents_classes.npy']
    paths +=[BASE/f'runs/{n}/panel/data.npz' for n in PREVIOUS[1:]]
    paths +=[OLD/f'seed_{s}/teacher.pt' for s in SEEDS]
    paths +=[BASE/'scripts'/n for n in ('dsprites_teacher.py','dsprites_mechanism.py','gradient_mechanism.py',
        'cross_layer_mechanism.py','residual_teacher.py','external_experiment.py','research_runtime.py')]
    inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in paths}
    RUN.mkdir()
    for name in CODE:
        path=RUN/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,path)
    p=dict(utc=teacher.now(),version='matched-shape-donor-v1',seeds=SEEDS,
        prior_knowledge='Oct9 scalar gradient results and absence of prediction changes read; new follow-up, not original preregistration',
        source_groups=[list(range(k*4,k*4+4)) for k in range(4)],source_group_scalar_count=16384,
        middle_candidates='all32 complete block2 output channels, 1024 scalar coordinates each',
        budgets=BUDGETS,primary_budget=4,selection_strength=.5,evaluation_strengths=STRENGTHS,
        primary=dict(target='donor_logit-recipient_logit endpoint change',strength=1.,comparison='state_linear vs fixed_linear',
            metrics=['restored endpoint contrast change RMS / natural endpoint change RMS',
                     'transplant endpoint error RMS / natural endpoint change RMS'],aggregation='per teacher ratios, mean all6; all paired results'),
        selection_score='E²-0.5*(R²+(T-E)²), endpoint of all4 groups at alpha0.5; signed unnormalized utility',
        fixed_fit='all144 old development images;288 directed pairs;global and six ordered-class-pair mean scores',
        secondary=['recipient prediction return and donor prediction reproduction',
                   '24 conditional interactions in same label-pair contrast using task-selected sets',
                   'budget8','source/middle state gradient direction drift','full source and full middle exact controls'],
        source_path=dict(counts=[8,32],rule='midpoint',target='donor-recipient logit',strength=1.,
                         attribution='group sum of gradient times full source delta, known integrated-gradient completeness; not new theorem'),
        specs=SPECS,random='four seeded fixed32-channel permutations shared across all pairs',
        tuning='none on new outputs; all fixed fits freeze before new queries, all0.5 lists seal before full strength',
        model='same frozen6 shape CNNs, eval BN, float64, no retraining',
        sources={n:teacher.sha(BASE/n) for n in CODE},inputs=inputs,preservation=preservation,legacy_manifests=legacy,
        environment=dict(torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii')
    teacher.dump(RUN/'specs.json',SPECS);prepare_panel();teacher.dump(RUN/'known_answers.json',donor.known_answers())
    event('protocol_locked',sha256=teacher.sha(RUN/'protocol.json'))


def panel_arrays():
    """新面板仅保存纯数值，不反序列化Python对象。"""
    with np.load(RUN/'panel/data.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


@torch.no_grad()
def sources_and_native(model,pixels,device):
    """每个真实图像仅计算一次冻结前缀；另存完整原模型分类正对照。"""
    states=[];outputs=[]
    for start in range(0,len(pixels),16):
        images=torch.as_tensor(pixels[start:start+16,None],dtype=torch.float64,device=device)
        states.append(mechanism.source_activation(model,images));outputs.append(model(images))
    return torch.cat(states),torch.cat(outputs).cpu().numpy()


def fit(device):
    """只读旧开发数据，拟合global和六种有向类别转换的固定通道顺序。"""
    verify_protocol();assert not (RUN/'fit_freeze.json').exists();(RUN/'fitted').mkdir(exist_ok=True)
    pixels=np.load(DATA/'development_images.npy',allow_pickle=False);labels=np.load(DATA/'development_labels.npy',allow_pickle=False)
    factors=np.load(DATA/'development_factors.npy',allow_pickle=False);pairs=donor.matched_pairs(factors,labels)
    bt=torch.as_tensor([[0]*4,[1]*4],dtype=torch.float64,device=device)
    for seed in SEEDS:
        folder=RUN/f'fitted/seed_{seed}'
        if (folder/'completion.json').exists():verify_folder(folder);continue
        assert not folder.exists();folder.mkdir();started=time.perf_counter();model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        sources,native=sources_and_native(model,pixels,device);natural=[];singletons=[];scores=[]
        for i,(r,d) in enumerate(pairs):
            with torch.no_grad():
                states=donor.replace_groups(sources[r:r+1],sources[d:d+1],bt,.5)
                middle=model.block2(states);output=mechanism.tail_from_middle(model,middle).cpu().numpy()
                singleton=donor.singleton_channels(model,middle).cpu().numpy()
            natural.append(output);singletons.append(singleton)
            scores.append(donor.endpoint_scores(output,singleton,donor.pair_weights(labels[r],labels[d])))
            if (i+1)%72==0:event('fit_progress',seed=seed,pairs=i+1,total=len(pairs))
        scores=np.asarray(scores);rl=labels[pairs[:,0]];dl=labels[pairs[:,1]]
        means={f'{r}>{d}':scores[(rl==r)&(dl==d)].mean(0).tolist() for r in range(3) for d in range(3) if r!=d}
        orders=dict(global_fixed=donor.rank_channels(scores.mean(0)).tolist(),
            pair_fixed={key:donor.rank_channels(np.asarray(value)).tolist() for key,value in means.items()})
        for repeat in range(4):orders[f'random_{repeat}']=np.random.default_rng(seed+2026101000+repeat).permutation(32).tolist()
        teacher.dump(folder/'orders.json',orders);teacher.dump(folder/'means.json',dict(pair_means=means,global_mean=scores.mean(0).tolist()))
        np.savez_compressed(folder/'data.npz',natural=np.asarray(natural),singleton_logits=np.asarray(singletons),scores=scores,
            pairs=pairs,recipient_labels=rl,donor_labels=dl,native_logits=native)
        complete(folder,started);event('fit_complete',seed=seed,seconds=time.perf_counter()-started)


def select(device):
    """只用新配对半强度信息选通道，实际单点分数不供梯度规则读取。"""
    verify_freeze('fit_freeze');assert not (RUN/'selection_freeze.json').exists();(RUN/'selection').mkdir(exist_ok=True)
    panel=panel_arrays();labels=panel['labels'];bt=torch.as_tensor([[0]*4,[1]*4],dtype=torch.float64,device=device)
    for seed in SEEDS:
        folder=RUN/f'selection/seed_{seed}'
        if (folder/'completion.json').exists():verify_folder(folder);continue
        assert not folder.exists();folder.mkdir();started=time.perf_counter();model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        sources,native=sources_and_native(model,panel['images'],device)
        fixed=teacher.read(RUN/f'fitted/seed_{seed}/orders.json');values={k:[] for k in (
            'natural','source_effect','baseline_middle_effect','state_middle_effect','gradient_drift','baseline_gradient_norm',
            'singleton_logits','finite_scores','predicted_scores','selections')}
        for i,(r,d) in enumerate(panel['pairs']):
            sr,sd=sources[r:r+1],sources[d:d+1];states=donor.replace_groups(sr,sd,bt,.5)
            v=donor.state_effects(model,states,sd-sr,int(labels[r]),int(labels[d]))
            arrays={k:(x.cpu().numpy() if torch.is_tensor(x) else x) for k,x in v.items() if k!='middle'}
            singleton=donor.singleton_channels(model,v['middle']).cpu().numpy();weight=donor.pair_weights(labels[r],labels[d])
            predicted=donor.predicted_singletons(arrays['natural'],arrays['baseline_middle_effect'],arrays['state_middle_effect'])
            ps=np.stack([donor.endpoint_scores(arrays['natural'],p,weight) for p in predicted])
            actual=donor.endpoint_scores(arrays['natural'],singleton,weight)
            orders={**fixed,'pair_fixed':fixed['pair_fixed'][f'{labels[r]}>{labels[d]}'],
                'fixed_linear':donor.rank_channels(ps[0]),'state_linear':donor.rank_channels(ps[1]),'input_finite':donor.rank_channels(actual)}
            selections=[np.arange(32) if s['method']=='all_channels' else np.asarray(orders[s['method']])[:s['budget']] for s in SPECS]
            arrays.update(singleton_logits=singleton,finite_scores=actual,predicted_scores=ps,selections=mechanism.encode(selections,32))
            for key,value in arrays.items():values[key].append(value)
            if (i+1)%24==0:event('selection_progress',seed=seed,pairs=i+1,total=len(panel['pairs']))
        np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()},pairs=panel['pairs'],
            recipient_labels=labels[panel['pairs'][:,0]],donor_labels=labels[panel['pairs'][:,1]],native_logits=native)
        teacher.dump(folder/'cost.json',dict(pairs=72,endpoint_gradient_forward_states=144,class_backward_batches=216,
            finite_reference_tail_vectors=72*64,note='Forward-vector counts are not backward costs; no measured speedup claimed'))
        complete(folder,started);event('selection_complete',seed=seed,seconds=time.perf_counter()-started)


def metrics(raw,weights,stencil,target):
    """任务端点与24个条件交互分别求平方误差，不混为一种证据。"""
    n=np.einsum('nsc,nc->ns',raw['natural'],weights)
    p=np.einsum('nmosc,nc->nmos',raw['patched'],weights)
    if target=='task_endpoint':
        effect=n[:,-1]-n[:,0];response=p[:,:,:,-1]-n[:,None,None,0]
        return effect**2,response[:,:,0]**2,(response[:,:,1]-effect[:,None])**2
    contrast=lambda x:x[...,stencil[:,0]]-x[...,stencil[:,1]]-x[...,stencil[:,2]]+x[...,stencil[:,3]]
    ni=contrast(n);pi=contrast(p)
    return np.mean(ni**2,-1),np.mean(pi[:,:,0]**2,-1),np.mean((pi[:,:,1]-ni[:,None])**2,-1)


def evaluate(device):
    """名单统一封存后检验任务阻断、重现、交互及路径积分。"""
    verify_freeze('selection_freeze');panel=panel_arrays();labels=panel['labels'];bits,stencil,metadata=mechanism.design()
    bt=torch.as_tensor(bits,dtype=torch.float64,device=device);weights=np.stack([donor.pair_weights(labels[r],labels[d]) for r,d in panel['pairs']])
    (RUN/'evaluation').mkdir(exist_ok=True)
    for seed in SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        sources,native=sources_and_native(model,panel['images'],device)
        with np.load(RUN/f'selection/seed_{seed}/data.npz',allow_pickle=False) as z:selected=z['selections'].copy()
        for alpha in STRENGTHS:
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            if (folder/'completion.json').exists():verify_folder(folder);continue
            assert not folder.exists();folder.mkdir(parents=True);started=time.perf_counter();values={k:[] for k in (
                'natural','source_effect','baseline_middle_effect','state_middle_effect','gradient_drift','baseline_gradient_norm',
                'patched','frozen_gate_logits','gate_flips')};paths=[]
            checks=dict(forward=0.,donor_endpoint=0.,frozen_gate_interaction=0.,source_interaction=0.,fixed_source_derivative=0.)
            for i,(r,d) in enumerate(panel['pairs']):
                sr,sd=sources[r:r+1],sources[d:d+1];states=donor.replace_groups(sr,sd,bt,alpha)
                v=donor.state_effects(model,states,sd-sr,int(labels[r]),int(labels[d]))
                traced=mechanism.trace(model,states);arrays={k:(x.cpu().numpy() if torch.is_tensor(x) else x) for k,x in v.items() if k!='middle'}
                selections=[x[x>=0] for x in selected[i]]
                frozen=old_gradient.frozen_gate_output(model,states,traced).cpu().numpy()
                arrays.update(patched=donor.patch_channels(model,v['middle'],selections).cpu().numpy(),frozen_gate_logits=frozen,
                    gate_flips=np.stack([((z>0)!=(z[:1]>0)).flatten(1).sum(1).cpu().numpy() for z in traced['z']],-1))
                for key,value in arrays.items():values[key].append(value)
                checks['forward']=max(checks['forward'],float(np.max(np.abs(arrays['natural']-traced['output'].cpu().numpy()))))
                checks['source_interaction']=max(checks['source_interaction'],float(mechanism.pair_difference(states,stencil).abs().max()))
                checks['frozen_gate_interaction']=max(checks['frozen_gate_interaction'],float(np.max(np.abs(mechanism.pair_difference(frozen,stencil)))))
                fixed_change=alpha*arrays['source_effect'][0].sum(0)
                checks['fixed_source_derivative']=max(checks['fixed_source_derivative'],float(np.max(np.abs(frozen[-1]-frozen[0]-fixed_change))))
                if alpha==1.:
                    checks['donor_endpoint']=max(checks['donor_endpoint'],float(np.max(np.abs(arrays['natural'][-1]-native[d]))))
                    paths.append(donor.integrate_source_path(model,sr,sd,int(labels[r]),int(labels[d])))
                if (i+1)%24==0:event('evaluation_progress',seed=seed,alpha=alpha,pairs=i+1,total=72)
            raw={k:np.asarray(v) for k,v in values.items()}
            np.savez_compressed(folder/'data.npz',**raw,pairs=panel['pairs'],recipient_labels=labels[panel['pairs'][:,0]],
                donor_labels=labels[panel['pairs'][:,1]],weights=weights,stencil=stencil,pair_metadata=metadata,native_logits=native)
            if paths:np.savez_compressed(folder/'path.npz',**{k:np.asarray([p[k] for p in paths]) for k in paths[0]},counts=np.array([8,32]))
            for target in ('task_endpoint','interaction'):
                n,r,t=metrics(raw,weights,stencil,target)
                np.savez_compressed(folder/f'metrics_{target}.npz',natural_mse=n,restore_mse=r,transplant_error_mse=t)
            assert max(checks.values())<1e-9,checks
            teacher.dump(folder/'checks.json',dict(status='PASS',maximum=checks));complete(folder,started)
            event('evaluation_complete',seed=seed,alpha=alpha,seconds=time.perf_counter()-started,checks=checks)


def summarize():
    """先求教师内RMS之比，保留所有种子、预算及任务与交互指标。"""
    verify_freeze('selection_freeze');assert not (RUN/'summary.json').exists();rows=[]
    for seed in SEEDS:
        for alpha in STRENGTHS:
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder)
            for target in ('task_endpoint','interaction'):
                with np.load(folder/f'metrics_{target}.npz',allow_pickle=False) as m:
                    denominator=float(m['natural_mse'].mean());assert denominator>1e-20
                    for j,spec in enumerate(SPECS):rows.append(dict(seed=seed,alpha=alpha,target=target,**spec,
                        natural_rms=float(np.sqrt(denominator)),restore_ratio=float(np.sqrt(m['restore_mse'][:,j].mean()/denominator)),
                        transplant_error_ratio=float(np.sqrt(m['transplant_error_mse'][:,j].mean()/denominator))))
    teacher.dump(RUN/'summary.json',dict(utc=teacher.now(),rows=rows));event('summary_complete',rows=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,
        choices=['init','fit','freeze','select','seal','evaluate','summarize'])
    args=parser.parse_args();device=teacher.configure()
    if args.stage=='init':initialize(device)
    elif args.stage=='fit':fit(device)
    elif args.stage=='freeze':freeze('fit_freeze','fitted')
    elif args.stage=='select':select(device)
    elif args.stage=='seal':freeze('selection_freeze','selection')
    elif args.stage=='evaluate':evaluate(device)
    else:summarize()

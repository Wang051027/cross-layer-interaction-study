"""层间传递阶段：init → evaluate → summarize；不扩大输入、不重新选名单。

原六轮1059条交付只读。新任务直接擦除中间层的源组合项，保留空及
单组响应；评价前冻结源组、旧通道名单、随机排列和完整数值实现。
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
import dsprites_teacher as teacher
import numpy as np
import torch
import layer_transfer_mechanism as core
from shape_donor_mechanism import replace_groups,pair_weights

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_layer_transfer_20261013'
SHAPE=BASE/'runs/dsprites_shape_donor_20261010';TRIPLE=BASE/'runs/dsprites_triple_gates_20261011'
CODE=('layer_transfer_mechanism.py','run_dsprites_layer_transfer.py','test_dsprites_layer_transfer.py')
OLDER=('dsprites_20260929','dsprites_factor_transfer_20261007','dsprites_gradient_20261009',
       'dsprites_shape_donor_20261010','dsprites_channel_gates_20261011','dsprites_triple_gates_20261011')


def load(path):
    """复制数值数组后关闭文件，不使用pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def event(kind,**values):
    """当前阶段日志不写入任何旧目录。"""
    row=dict(utc=teacher.now(),event=kind,**values)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def verify_folder(folder):
    """已经完成的数组逐一核对后才允许跳过。"""
    for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name


def verify_protocol():
    """主数值实现、旧输入和旧名单在整个评价中保持冻结。"""
    p=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    assert teacher.sha(RUN/'channel_orders.npz')==p['channel_orders_sha256']
    return p


def init(device):
    """冻结用户确认的范围：当前面板直接验证源→中间→输出的传递。"""
    assert not RUN.exists();preservation={}
    for name in OLDER:
        path=BASE/f'runs/{name}/delivery_manifest.json';old=teacher.read(path)
        assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip()
        for item in old['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[str(path.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(path),files=len(old['files']))
    legacy=teacher.verify_old();preflight=BASE/'reports/dsprites_layer_transfer_preflight_20261013.json'
    assert teacher.read(preflight)['status']=='PASS'
    inputs=[SHAPE/'panel/data.npz',preflight]
    inputs +=[TRIPLE/f'discovery/seed_{s}/data.npz' for s in teacher.SEEDS]
    inputs +=[SHAPE/f'evaluation/seed_{s}/alpha_{a}/data.npz' for s in teacher.SEEDS for a in (1.,.5)]
    inputs +=[teacher.RUN/f'seed_{s}/teacher.pt' for s in teacher.SEEDS]
    helpers=('dsprites_teacher.py','residual_teacher.py','external_experiment.py','research_runtime.py',
        'shape_donor_mechanism.py','dsprites_mechanism.py','cross_layer_mechanism.py','audit_dsprites_gradient.py')
    inputs +=[BASE/'scripts'/name for name in helpers]
    RUN.mkdir();shutil.copy2(preflight,RUN/'preflight.json')
    for name in CODE:
        path=RUN/'source/scripts'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'scripts'/name,path)
    np.savez_compressed(RUN/'design.npz',**core.design())
    selected=np.array([load(TRIPLE/f'discovery/seed_{s}/data.npz')['channels'] for s in teacher.SEEDS])
    random=np.array([[np.random.default_rng(s+2026101300+v).permutation(32) for v in range(2)] for s in teacher.SEEDS])
    np.savez_compressed(RUN/'channel_orders.npz',selected=selected,random=random,seeds=teacher.SEEDS)
    p=dict(utc=teacher.now(),version='source-middle-transmission-v1',seeds=teacher.SEEDS,
        user_scope='focus on layer-to-layer transmission;do not expand image inputs',
        prior_knowledge='all previous pair/triple and gate results inspected;new same-panel mechanism follow-up,not new-input validation',
        panel='same36 images and72 directed shape pairs;no retraining,no newly selected cases',
        source_groups=[list(range(4*g,4*g+4)) for g in range(4)],source='block1 output16x64x64;four groups,not four individual neurons',
        middle='block2 output32x32x32;all16 source coalitions observed',
        goal='test whether source-combination nonadditivity is transmitted through middle state and modified by downstream nonlinear gates',
        erase='H<=1 keeps empty and each single-source-group middle response exactly;remove only source orders2..4 in middle,then actual original block3',
        primary=dict(alpha=1.,method='all32 middle interaction erasure',metric='RMS of24 conditional pair contrasts over72cases perteacher;meanall6',
            results='remaining RMS,natural-minus-erased RMS,signed removal projection;can enhance rather than reduce due cancellation'),
        secondary='H<=2/3;actual inherited4/8channel erase and restore;two same-budget random lists;two full residual channel permutations with equal norm;rectified H<=1 sensitivity',
        restore_target='output contrast natural-minus-H<=1;restore error compares partial correction to that operational mediated effect,not whole output',
        modes=core.MODES,gate_control='only both block2 andblock3 gates fixed makes source-to-output affine;fixed downstream can carry upstream interactions',
        gradients='task donor-minus-recipient;autograd source andmiddle percoalition,global and local-context middle Jacobian projections;no optimizer gradients',
        channel_selection='inherit Oct11 percase8physical channels and4prefix;no new rankings;random permutations fixed before evaluation',
        random_seed='teacher_seed+2026101300+repeat',strengths=[1.,.5],
        example='first teacher,case0,both strengths;store full source/middle coefficients and gradients;no result-selected example',
        limitations='seen narrow panel;internal erasure may create negative middle values;rectification introduces new interaction;baselinedecomposition not unique causal attribution;group size arbitrary',
        sources={f'scripts/{n}':teacher.sha(BASE/'scripts'/n) for n in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in inputs},
        channel_orders_sha256=teacher.sha(RUN/'channel_orders.npz'),preservation=preservation,legacy_manifests=legacy,
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii')
    event('protocol_locked',sha256=teacher.sha(RUN/'protocol.json'))


def evaluate(device):
    """全部原72配对、16源状态、两强度；完成文件拒绝覆盖。"""
    verify_protocol();panel=load(SHAPE/'panel/data.npz');d=core.design();orders=load(RUN/'channel_orders.npz')
    bits=torch.as_tensor(d['bits'],dtype=torch.float64,device=device)
    for j,seed in enumerate(teacher.SEEDS):
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        with torch.no_grad():
            images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
            sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            if (folder/'completion.json').exists():verify_folder(folder);continue
            assert not folder.exists();folder.mkdir(parents=True);started=time.perf_counter();values={};maxima={}
            for i,(r,donor) in enumerate(panel['pairs']):
                src=replace_groups(sources[r:r+1],sources[donor:donor+1],bits,alpha)
                weight=pair_weights(panel['labels'][r],panel['labels'][donor])
                out,checks,fields=core.calculate(model,src,weight,alpha*(sources[donor:donor+1]-sources[r:r+1]),
                    orders['selected'][j,i],orders['random'][j],d,seed==teacher.SEEDS[0] and i==0)
                for key,value in out.items():values.setdefault(key,[]).append(value)
                for key,error in checks.items():maxima[key]=max(maxima.get(key,0.),error)
                if fields:np.savez_compressed(folder/'example_fields.npz',**fields)
                if (i+1)%24==0:event('evaluation_progress',seed=seed,alpha=alpha,cases=i+1,total=72)
            np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()},pairs=panel['pairs'],channels=orders['selected'][j],random_orders=orders['random'][j])
            teacher.dump(folder/'checks.json',dict(status='PASS',maximum=maxima))
            teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seconds=time.perf_counter()-started,files={p.name:teacher.sha(p) for p in folder.iterdir() if p.is_file()}))
            event('evaluation_complete',seed=seed,alpha=alpha,seconds=time.perf_counter()-started)


def contrast_axis1(values,d):
    """案例后状态轴四项差分，保存最后的模式/类别轴。"""
    a,b,c,e=d['stencils'].T;return values[:,a]-values[:,b]-values[:,c]+values[:,e]


def statistics(original,response):
    """保留分母与原RMS；近零分母返回None而非编造百分比。"""
    denominator=float(np.mean(original**2));rms=float(np.sqrt(denominator))
    return dict(natural_rms=rms,remaining_ratio=None if denominator<1e-20 else float(np.sqrt(np.mean(response**2)/denominator)),
        changed_ratio=None if denominator<1e-20 else float(np.sqrt(np.mean((original-response)**2)/denominator)),
        signed_removal=None if denominator<1e-20 else float(np.mean(original*(original-response))/denominator))


def summarize():
    """主擦除和恢复修正分别用对应目标归一化，不偷换作用对象。"""
    verify_protocol();assert not (RUN/'summary.json').exists();d=core.design();panel=load(SHAPE/'panel/data.npz')
    w=np.stack([pair_weights(panel['labels'][r],panel['labels'][s]) for r,s in panel['pairs']]);rows=[];decomposition=[]
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz')
            modes=np.einsum('nsmk,nk->nsm',raw['logits'],w);lower=np.einsum('nqsk,nk->nqs',raw['lower_logits'],w)
            patch=np.einsum('nvosk,nk->nvos',raw['patched_logits'],w)
            perm=np.einsum('nvsk,nk->nvs',raw['permuted_logits'],w);rect=np.einsum('nsk,nk->ns',raw['rectified_logits'],w)
            for metric in ('conditional_pairs','joint_all4'):
                change=lambda x:contrast_axis1(x,d) if metric=='conditional_pairs' else np.einsum('s,ns...->n...',d['joint'],x)
                nat=change(modes[:,:,0]);add=change(lower[:,0]);target=nat-add
                for mode,name in enumerate(core.MODES):rows.append(dict(seed=seed,alpha=alpha,metric=metric,method=name,operation='forward',**statistics(nat,change(modes[:,:,mode]))))
                for degree in (1,2,3):rows.append(dict(seed=seed,alpha=alpha,metric=metric,method=f'middle_le{degree}',operation='erase',**statistics(nat,change(lower[:,degree-1]))))
                for v,(method,budget) in enumerate(core.SPECS):
                    erase=change(patch[:,v,0]);restore=change(patch[:,v,1])-add
                    rows.append(dict(seed=seed,alpha=alpha,metric=metric,method=f'{method}{budget}',operation='erase',**statistics(nat,erase)))
                    # 这里remaining_ratio是恢复误差/待恢复修正RMS，all32应精确为0。
                    row=statistics(target,restore);row['restore_error_ratio']=None if row['natural_rms']<1e-10 else float(np.sqrt(np.mean((restore-target)**2))/row['natural_rms'])
                    rows.append(dict(seed=seed,alpha=alpha,metric=metric,method=f'{method}{budget}',operation='restore',**row))
                for v in range(2):rows.append(dict(seed=seed,alpha=alpha,metric=metric,method=f'permuted{v}',operation='forward',**statistics(nat,change(perm[:,v]))))
                rows.append(dict(seed=seed,alpha=alpha,metric=metric,method='rectified_additive',operation='erase',**statistics(nat,change(rect))))
            nat=contrast_axis1(modes[:,:,0],d);den=float(np.mean(nat**2))
            for name,projection in [('global',raw['conditional_baseline_channel_projection'].sum(2)),('context',raw['conditional_context_channel_projection'].sum(2))]:
                remainder=nat-projection
                decomposition.append(dict(seed=seed,alpha=alpha,baseline=name,natural_rms=float(np.sqrt(den)),
                    transmitted_rms_ratio=None if den<1e-20 else float(np.sqrt(np.mean(projection**2)/den)),
                    downstream_remainder_rms_ratio=None if den<1e-20 else float(np.sqrt(np.mean(remainder**2)/den)),
                    transmitted_projection=None if den<1e-20 else float(np.mean(nat*projection)/den),
                    downstream_projection=None if den<1e-20 else float(np.mean(nat*remainder)/den)))
    teacher.dump(RUN/'summary.json',dict(utc=teacher.now(),rows=rows,decomposition=decomposition));event('summary_complete',rows=len(rows),decomposition=len(decomposition))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('init','evaluate','summarize'))
    args=parser.parse_args();device=teacher.configure()
    if args.stage=='init':init(device)
    elif args.stage=='evaluate':evaluate(device)
    else:summarize()

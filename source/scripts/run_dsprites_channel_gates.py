"""在八通道名单中按半强度效应选取通道对和开关位置，再与两组随机名单一起评价全强度响应。"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import dsprites_teacher as teacher
import channel_gate_mechanism as gates
import numpy as np
import torch

BASE=teacher.BASE;PRIOR=BASE/'runs/dsprites_shape_donor_20261010'
RUN=BASE/'runs/dsprites_channel_gates_20261011'
CODE=('scripts/channel_gate_mechanism.py','scripts/run_dsprites_channel_gates.py','scripts/test_dsprites_channel_gates.py')
OLDER=('dsprites_20260929','dsprites_factor_transfer_20261007','dsprites_gradient_20261009','dsprites_shape_donor_20261010')


def load(path):
    """纯数值读取，禁止pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def event(kind,**details):
    """只在新目录记录，不向旧教师events文件追加。"""
    value=dict(utc=teacher.now(),event=kind,**details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def complete(folder,started):
    """所有数据落盘后写完成标记；中断残留拒绝覆盖。"""
    teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seconds=time.perf_counter()-started,
        files={p.name:teacher.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))


def verify_folder(folder):
    """只接续完成且未改动的阶段。"""
    for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name


def verify_protocol():
    """冻结数值代码、原名单和面板，防止运行途中改变解释对象。"""
    p=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['sources'].items():
        assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    return p


def initialize(device):
    """核验前序文件，并记录本轮配对、选点规则和开关干预协议。"""
    assert not RUN.exists(),'Preserve existing run';preservation={}
    for old in OLDER:
        manifest=BASE/f'runs/{old}/delivery_manifest.json';record=teacher.read(manifest)
        for item in record['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[str(manifest.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(manifest),files=len(record['files']))
    legacy=teacher.verify_old();test=BASE/'reports/dsprites_channel_gates_preflight_20261011.json'
    assert teacher.read(test)['status']=='PASS'
    paths=[PRIOR/'panel/data.npz',PRIOR/'protocol.json',PRIOR/'selection_freeze.json',test]
    paths +=[PRIOR/f'selection/seed_{s}/data.npz' for s in teacher.SEEDS]
    paths +=[teacher.RUN/f'seed_{s}/teacher.pt' for s in teacher.SEEDS]
    paths +=[PRIOR/f'evaluation/seed_{s}/alpha_{a}/data.npz' for s in teacher.SEEDS for a in (1.,.5)]
    paths +=[BASE/'scripts'/n for n in ('dsprites_teacher.py','residual_teacher.py','external_experiment.py','research_runtime.py')]
    RUN.mkdir()
    for name in CODE:
        destination=RUN/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,destination)
    shutil.copy2(test,RUN/'preflight.json')
    d=gates.design();np.savez_compressed(RUN/'design.npz',**d)
    p=dict(utc=teacher.now(),version='channel-pairs-relu-gates-v1',seeds=teacher.SEEDS,
        purpose='locate middle-channel nonadditivity after block2; only two block3 ReLUs are downstream of this cut',
        prior_knowledge='Oct10 endpoints and combination residuals inspected; same panel mechanistic follow-up, not new independent generalization test',
        panel='same36 official pixels and72 directed matched-shape pairs, same six frozen teachers',
        channels='frozen Oct10 state_linear eight-channel order; first4 prefix is old primary list; channel identities not aligned across teachers',
        source_strengths=[.5,1.],gate_fit_strength=.5,
        coalitions='43 union: complete first4 subsets, all28 eight-list pairs at empty background, singles, empty, full8',
        high_order='exact first4 Mobius coefficients; full8 orders3+ aggregate=joint residual minus sum28 empty-background pair terms',
        modes=gates.MODES,
        algebra='split signed pair or joint residual into propagation of first ReLU remainder under baseline second gate and second ReLU remainder',
        selection='top3 absolute half-strength pair effects per input/operation; per stage largest absolute unit contributions among half-strength switched sites',
        budgets=gates.GATE_BUDGETS,random_repeats=2,
        random='two seeded permutations of same half-strength switched-site pool, same actual budget; no full-strength resampling',
        primary=dict(strength=1.,pair_rank=0,gate_budget=128,
            metric='RMS pair effect after selected-unit gate holding / RMS natural pair effect, per teacher then mean6',
            comparisons='selected units vs both fixed random repetitions, each ReLU stage and restore/transplant separately'),
        secondary='three frozen pairs; budget32; half-strength replication; layer holds; signed decomposition; high orders; stage coordinate contributions',
        gate_values='coordinates frozen; held values are baseline gate at current strength and operation, not old half-strength gate bits',
        causality='first-gate holding recalculates second gate; signed algebra allocation depends on baseline, not unique causal attribution; clamp may generate synthetic negative activation',
        sources={n:teacher.sha(BASE/n) for n in CODE},
        inputs={str(x.relative_to(BASE)).replace('\\','/'):teacher.sha(x) for x in paths},
        preservation=preservation,legacy_manifests=legacy,
        environment=dict(torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii')
    event('protocol_locked',sha256=teacher.sha(RUN/'protocol.json'))


@torch.no_grad()
def middle_states(model,panel,device,alpha):
    """从原图和全源形状替换构建中介端点，不对每个子集重新算前缀。"""
    images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
    sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
    base=torch.cat([model.block2(x) for x in sources.split(16)])
    endpoints=[]
    for r,d in panel['pairs']:
        x=(1-alpha)*sources[r:r+1]+alpha*sources[d:d+1];endpoints.append(model.block2(x))
    return base,torch.cat(endpoints)


def validate_checks(summary,units,weight):
    """门控分解、四阶展开和仿射对照必须回到同一原始输出。"""
    values=dict(frozen_pairs=float(np.abs(summary['pair_logits'][:,3]).max()),
        frozen_joint=float(np.abs(summary['joint_logits'][:,3]).max()),
        pair_decomposition=float(np.abs(summary['pair_stageparts'].sum(1)-summary['pair_logits'][:,0]).max()),
        joint_decomposition=float(np.abs(summary['joint_stageparts'].sum(1)-summary['joint_logits'][:,0]).max()),
        unit_decomposition=float(np.abs(units.sum((2,3,4)).cpu().numpy()-summary['pair_stageparts']@weight).max()))
    assert max(values.values())<1e-9,values;return values


def discover(device):
    """半强度定位所有配对和开关，并保存全阶段数据后统一封存。"""
    verify_protocol();assert not (RUN/'gate_freeze.json').exists();(RUN/'discovery').mkdir(exist_ok=True)
    panel=load(PRIOR/'panel/data.npz');labels=panel['labels'];d=gates.design()
    for seed in teacher.SEEDS:
        folder=RUN/f'discovery/seed_{seed}'
        if (folder/'completion.json').exists():verify_folder(folder);continue
        assert not folder.exists();folder.mkdir();started=time.perf_counter();values={};maximum={}
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        baseline,endpoint=middle_states(model,panel,device,.5)
        old=load(PRIOR/f'selection/seed_{seed}/data.npz');channels=old['selections'][:,12,:8]
        assert np.array_equal(channels[:,:4],old['selections'][:,3,:4])
        for i,(r,donor) in enumerate(panel['pairs']):
            weight=np.zeros(3);weight[labels[donor]]=1;weight[labels[r]]=-1;case=[]
            for op in range(2):
                base,reference=(endpoint[i:i+1],baseline[r:r+1]) if op==0 else (baseline[r:r+1],endpoint[i:i+1])
                x=gates.patch_states(base,reference,channels[i],d['bits']);traced=gates.trace_modes(model,x)
                units=gates.pair_unit_contributions(model,traced,weight,d['stencils']);s=gates.summaries(traced,d,weight)
                for name,v in validate_checks(s,units,weight).items():maximum[name]=max(maximum.get(name,0.),v)
                score=s['pair_logits'][:,0]@weight
                selection=gates.select_pairs_and_gates(traced,units,score,d['stencils'],seed+2026101100+100*i+op)
                case.append({**s,**selection})
            for key in case[0]:values.setdefault(key,[]).append(np.asarray([c[key] for c in case]))
            if (i+1)%24==0:event('discovery_progress',seed=seed,pairs=i+1,total=72)
        np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()},channels=channels,pairs=panel['pairs'])
        teacher.dump(folder/'checks.json',dict(status='PASS',maximum=maximum));complete(folder,started)
        event('discovery_complete',seed=seed,seconds=time.perf_counter()-started)


def seal():
    """半强度配对、单位坐标和随机对照统一封存，全强度禁止重新选。"""
    verify_protocol();assert not (RUN/'gate_freeze.json').exists();files={}
    for seed in teacher.SEEDS:
        folder=RUN/f'discovery/seed_{seed}';verify_folder(folder)
        for p in folder.iterdir():files[str(p.relative_to(RUN)).replace('\\','/')]=teacher.sha(p)
    teacher.dump(RUN/'gate_freeze.json',dict(utc=teacher.now(),protocol_sha256=teacher.sha(RUN/'protocol.json'),files=files))
    (RUN/'gate_freeze.sha256').write_text(teacher.sha(RUN/'gate_freeze.json')+'\n',encoding='ascii');event('gate_lists_sealed',files=len(files))


def verify_seal():
    """阶段入口复核封存，而不是仅信任文件存在。"""
    verify_protocol();assert teacher.sha(RUN/'gate_freeze.json')==(RUN/'gate_freeze.sha256').read_text().strip()
    for name,value in teacher.read(RUN/'gate_freeze.json')['files'].items():assert teacher.sha(RUN/name)==value,name


def evaluate(device):
    """全强度为主，半强度作补充；坐标选择始终来自同一个封存文件。"""
    verify_seal();panel=load(PRIOR/'panel/data.npz');labels=panel['labels'];d=gates.design()
    (RUN/'evaluation').mkdir(exist_ok=True)
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz');channels=frozen['channels']
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            if (folder/'completion.json').exists():verify_folder(folder);continue
            assert not folder.exists();folder.mkdir(parents=True);started=time.perf_counter();values={};maximum={}
            baseline,endpoint=middle_states(model,panel,device,alpha)
            for i,(r,donor) in enumerate(panel['pairs']):
                weight=np.zeros(3);weight[labels[donor]]=1;weight[labels[r]]=-1;case=[]
                for op in range(2):
                    base,reference=(endpoint[i:i+1],baseline[r:r+1]) if op==0 else (baseline[r:r+1],endpoint[i:i+1])
                    x=gates.patch_states(base,reference,channels[i],d['bits']);traced=gates.trace_modes(model,x)
                    units=gates.pair_unit_contributions(model,traced,weight,d['stencils']);s=gates.summaries(traced,d,weight)
                    for name,v in validate_checks(s,units,weight).items():maximum[name]=max(maximum.get(name,0.),v)
                    selection={k:frozen[k][i,op] for k in ('selected_pairs','selected_gates','random_gates')}
                    case.append({**s,**gates.selected_details(traced,units,d['stencils'],selection),
                        'partial_logits':gates.partial_holds(model,traced,d['stencils'],selection)})
                for key in case[0]:values.setdefault(key,[]).append(np.asarray([c[key] for c in case]))
                if (i+1)%24==0:event('evaluation_progress',seed=seed,alpha=alpha,pairs=i+1,total=72)
            np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()},channels=channels,pairs=panel['pairs'])
            teacher.dump(folder/'checks.json',dict(status='PASS',maximum=maximum));complete(folder,started)
            event('evaluation_complete',seed=seed,alpha=alpha,seconds=time.perf_counter()-started)


def summarize():
    """主结果为半强度第一配对、128开关的全强度剩余RMS比；保留所有补充。"""
    verify_seal();assert not (RUN/'summary.json').exists();panel=load(PRIOR/'panel/data.npz');rows=[]
    labels=panel['labels'];weights=np.zeros((72,3));weights[np.arange(72),labels[panel['pairs'][:,1]]]=1;weights[np.arange(72),labels[panel['pairs'][:,0]]]=-1
    for seed in teacher.SEEDS:
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz')
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz')
            pair=np.einsum('nopmc,nc->nopm',raw['pair_logits'],weights)
            partial=np.einsum('norlbmsc,nc->norlbms',raw['partial_logits'],weights)
            pi=partial[...,-1]-partial[...,1]-partial[...,2]+partial[...,0]
            for op in range(2):
                for ranks in ((0,),(0,1,2)):
                    natural=np.asarray([pair[i,op,frozen['selected_pairs'][i,op,list(ranks)],0] for i in range(72)])
                    denominator=float(np.mean(natural**2));assert denominator>1e-20
                    for stage in range(2):
                        for budget_index,budget in enumerate(gates.GATE_BUDGETS):
                            for method in range(3):
                                response=pi[:,op,list(ranks),stage,budget_index,method]
                                rows.append(dict(seed=seed,alpha=alpha,operation=('restore','transplant')[op],
                                    ranks='first' if len(ranks)==1 else 'three',stage=stage+1,budget=budget,
                                    method=('selected','random_0','random_1')[method],natural_rms=float(np.sqrt(denominator)),
                                    remaining_ratio=float(np.sqrt(np.mean(response**2)/denominator))))
    teacher.dump(RUN/'summary.json',dict(utc=teacher.now(),rows=rows));event('summary_complete',rows=len(rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('init','discover','seal','evaluate','summarize'))
    args=parser.parse_args();device=teacher.configure()
    if args.stage=='init':initialize(device)
    elif args.stage=='discover':discover(device)
    elif args.stage=='seal':seal()
    elif args.stage=='evaluate':evaluate(device)
    else:summarize()

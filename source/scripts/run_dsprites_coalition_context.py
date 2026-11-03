"""锁定既有解后执行四状态验证；不扩大输入、不训练、不新增输出驱动搜索。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import coalition_context_mechanism as core
from run_dsprites_joint_window import OLDER as OLD, RUN as PRIOR, SHAPE, AR
from run_dsprites_upstream_rescue import load, sources_for, two_traces
from joint_window_mechanism import intervention_specs

BASE = teacher.BASE
RUN = BASE/'runs/dsprites_coalition_context_20261025'
OLDER = OLD+('dsprites_joint_window_20261018',)
CODE = ('coalition_context_mechanism.py','run_dsprites_coalition_context.py','test_dsprites_coalition_context.py')


def event(name, **details):
    """事件日志只记录进度。"""
    row = dict(utc=teacher.now(), event=name, **details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def preservation():
    """十二份已封存交付1650条目及更早三个来源全量只读比对。"""
    records = {}; total = 0
    for name in OLDER:
        path = BASE/f'runs/{name}/delivery_manifest.json'; rows = teacher.read(path)['files']
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        for row in rows:
            p = BASE/row['path']; assert teacher.sha(p)==row['sha256'] and p.stat().st_size==row['bytes'], row['path']
        records[str(path.relative_to(BASE)).replace('\\','/')] = dict(sha256=teacher.sha(path), files=len(rows)); total += len(rows)
    assert total == 1650; teacher.verify_old()
    return records


def verify_protocol():
    """每一阶段都验证协议和原始输入；已冻结数值代码不能中途改写。"""
    path = RUN/'protocol.json'; p = teacher.read(path)
    assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    for name,digest in p['sources'].items(): assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==digest, name
    for name,digest in p['inputs'].items(): assert teacher.sha(BASE/name)==digest, name
    return p


def verify_folder(folder):
    """已完成子目录仅核验，不能覆盖。"""
    for name,digest in teacher.read(folder/'completion.json')['files'].items(): assert teacher.sha(folder/name)==digest


def initialize():
    """结果之前锁定16个端点、四场、统计口径与实例，不按新输出选择。"""
    assert not RUN.exists()
    pre = BASE/'reports/dsprites_coalition_context_preflight_20261025.json'; assert teacher.read(pre)['status']=='PASS'
    preserved = preservation(); inputs = [pre,SHAPE/'panel/data.npz',PRIOR/'selection_freeze.json',PRIOR/'selection_freeze.sha256']
    dependencies = tuple(teacher.read(PRIOR/'protocol.json')['inputs'])
    inputs += [BASE/p for p in dependencies if p.startswith('scripts/')]
    inputs += [BASE/'scripts'/n for n in ('joint_window_mechanism.py','run_dsprites_joint_window.py')]
    for seed in teacher.SEEDS:
        inputs += [teacher.RUN/f'seed_{seed}/teacher.pt',AR/f'evaluation/seed_{seed}/data.npz']
        inputs += [PRIOR/f'search/seed_{seed}/{name}' for name in ('data.npz','units.npz','frozen.npz')]
        inputs += [PRIOR/f'evaluation/seed_{seed}/data.npz']
    RUN.mkdir(); shutil.copy2(pre,RUN/'preflight.json')
    for name in CODE:
        target=RUN/'source/scripts'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(BASE/'scripts'/name,target)
    p=dict(version='coalition-context-v1',utc=teacher.now(),client_date='2026-10-25',
        scope='same36images/72dependentpairs/6frozen samearchitecture teachers/masks3,15;no training/newinput/newsearch',
        endpoints=list(core.ENDPOINTS),selection='12 aligned solutions targeted+2random in4families, plus4 movingA targeted controls;all864 states,including failures/noops;no newtask-output selection',
        four_states=list(core.STATES),definition='H00=before originalfulltarget;H11=after frozenI;H10=Cafter+outsidebefore;H01=Cbefore+outsideafter;wholechannel maps;no A/R in these fields',
        modes=list(core.MODES),point_control='frozen originalj postfirstReLU value patched to nonnegative nativeu00 oru11;secondReLU originalforward;offmanifoldpossible',
        terms=list(core.TERMS),contrast=core.CONTRAST.tolist(),
        equations='a=F10-F00;c=F01-F00;J=F11-F10-F01+F00;total=a+c+J;C_after=a+J;outside_after=c+J',
        no_unique_share='report both conditionalC effects andJ;no fraction allocation to C, no Shapley split, no accuracygain claim',
        branches='exact H00-gate anchored decomposition:affine transport,propagated firstReLUremainder,secondReLUremainder;interaction affinepart0;accounting not unique causal mediation',
        cancellations='signed readout-channel and spatial contributions;1-abs(net)/sumabs when denominator>1e-10,otherwise undefined',
        drops='each originalC member atalignedtargeted4 endpoints revertedtoactualH00 inH10/H11;this asks necessityofvaluechange,distinct frompreviousA1membernecessity',
        primary='previousfocus539 andprioralignedtargeted26records/15distinctstates frozen;all864 andrandom/failed/movingcontrols retained',
        summaries='within each teacher selectedrecordsRMS thenequalvalidteacher averages;missing successes null;no independentpair tests;counts reportrecords vsstates',
        modulation='epsilon1e-8;JalignedwithC_before enhances;opposed attenuates orreverses;C_beforezero distinguishedfrom contextenabled;outside standaloneoffsetreportedseparately',
        examples=[dict(seed=202609292,case=70,mask=15,endpoint=25,reason='prior firstpositive fixedbefore newoutputs'),dict(seed=202609291,case=1,mask=3,endpoint=1,reason='prior plannedreference-dependent'),dict(seed=202609291,case=55,mask=15,endpoint=10,reason='priorfirstdecompositiononly')],
        tolerance=1e-9,epsilon=1e-8,limitations='samepanel prioradaptive selection;wholeCgroup swap not proof everymemberchange needed;j isonepointnotwholecircuit;hybridactivations mayoffmanifold;no universalmechanism orunique contributionpercent',
        sources={f'scripts/{name}':teacher.sha(BASE/'scripts'/name) for name in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in set(inputs)},preservation=preserved)
    teacher.dump(RUN/'protocol.json',p); (RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii')
    event('protocol_locked',old_entries=1650,sha256=teacher.sha(RUN/'protocol.json'))


def evaluate(device):
    """整图计算所有已冻结端点，不按成功与否减少评价范围。"""
    verify_protocol(); panel=load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        folder=RUN/f'evaluation/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start=time.perf_counter(); model=teacher.load_teacher(seed,device)
        for p in model.parameters(): p.requires_grad_(False)
        source=sources_for(model,panel,device)
        search=load(PRIOR/f'search/seed_{seed}/data.npz'); units=load(PRIOR/f'search/seed_{seed}/units.npz'); frozen=load(PRIOR/f'search/seed_{seed}/frozen.npz')
        prior=load(PRIOR/f'evaluation/seed_{seed}/data.npz'); values={}; errors=[]
        for case,(recipient,donor) in enumerate(panel['pairs']):
            half,full=two_traces(model,source,recipient,donor); row={}
            weight=np.zeros(3); weight[panel['labels'][donor]]=1.; weight[panel['labels'][recipient]]=-1.
            for m,mask in enumerate((3,15)):
                u={k:v[case,m] for k,v in units.items()}; s={k:v[case,m] for k,v in search.items()}; f={k:v[case,m] for k,v in frozen.items()}
                specs=intervention_specs(u,s); chosen=[specs[e] for e in core.ENDPOINTS]
                before,after=core.endpoints(model,half,full,mask,chosen)
                C=f['channels']; C=C[C>=0]; point=int(f['lists'][0,0])
                out=core.measure(model,before,after,C,point,weight)
                expected=prior['natural'][case,m,list(core.ENDPOINTS),2]
                error=max(float(abs(out['logits'][:,3,0]-expected).max()),float(abs(out['logits'][:,0,0]-prior['natural'][case,m,0,2]).max()))
                assert error<1e-9; errors.append(error)
                for k,v in out.items(): row.setdefault(k,[]).append(v)
            for k,v in row.items(): values.setdefault(k,[]).append(v)
            if (case+1)%12==0: event('evaluation_progress',seed=seed,cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in values.items()})
        teacher.dump(folder/'completion.json',dict(utc=teacher.now(),seed=seed,seconds=time.perf_counter()-start,prior_max_error=max(errors),files={'data.npz':teacher.sha(folder/'data.npz')}))
        event('evaluation_complete',seed=seed,seconds=time.perf_counter()-start,prior_max_error=max(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=('init','evaluate'),required=True)
    args=parser.parse_args()
    if args.stage=='init': initialize()
    else: evaluate(teacher.configure())

"""同输入下游定位：先锁定案例/规则，单点定位封存后再做联合阻断。"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import numpy as np
import torch
import dsprites_teacher as teacher
import downstream_route_mechanism as core
from run_dsprites_coalition_context import RUN as PRIOR,OLDER as OLD
from run_dsprites_joint_window import RUN as WINDOW,SHAPE
from run_dsprites_upstream_rescue import load,sources_for,two_traces
from joint_window_mechanism import intervention_specs
from coalition_context_mechanism import endpoints,four_fields

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_downstream_routes_20261025'
OLDER=OLD+('dsprites_coalition_context_20261025',)
CODE=('downstream_route_mechanism.py','run_dsprites_downstream_routes.py','test_dsprites_downstream_routes.py')


def event(name,**details):
    """记录阶段与完成情况，不参与选点。"""
    row=dict(utc=teacher.now(),event=name,**details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def cohort():
    """继承26成功、4仅分解、原失败例，加每教师首个非空失败源干预。

不按本轮输出筛案例；所有下游随机对照与主名单在同一固定队列运行。
"""
    with (PRIOR/'tables/effects.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    selected={}
    def add(r,role):
        key=tuple(int(r[k]) for k in ('seed','case','mask','endpoint'))
        if key not in selected:selected[key]=dict(zip(('seed','case','mask','endpoint'),key),roles=[],C=json.loads(r['C']),point=int(r['point']),family=r['family'])
        selected[key]['roles'].append(role)
    for r in rows:
        if r['primary_success']=='True':add(r,'primary26')
        if r['focus539']=='True' and r['strategy']=='moving_A' and r['prior_status']=='decomposition_only':add(r,'decomposition_control')
        if tuple(int(r[k]) for k in ('seed','case','mask','endpoint'))==(202609291,1,3,1):add(r,'prior_reference_failure')
    for seed in teacher.SEEDS:
        r=next(r for r in rows if int(r['seed'])==seed and int(r['endpoint'])==25 and r['focus539']=='True' and r['prior_aligned']=='False' and r['eligible']=='True' and max(float(r['C_max_change']),float(r['outside_max_change']))>1e-10)
        add(r,'first_failed_source_per_teacher')
    records=[selected[k] for k in sorted(selected)]
    for i,r in enumerate(records):r['record']=i
    assert len(records)==37 and sum('primary26' in r['roles'] for r in records)==26
    return records


def preservation():
    """十三份旧交付1711条目与更早来源只读核验。"""
    out={};total=0
    for name in OLDER:
        path=BASE/f'runs/{name}/delivery_manifest.json';records=teacher.read(path)['files']
        assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip()
        for r in records:
            p=BASE/r['path'];assert teacher.sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],r['path']
        out[str(path.relative_to(BASE)).replace('\\','/')]=dict(sha256=teacher.sha(path),files=len(records));total+=len(records)
    assert total==1711;teacher.verify_old();return out


def verify_protocol():
    """锁定新数值核心、旧模型、输入和案例来源。"""
    path=RUN/'protocol.json';p=teacher.read(path);assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip()
    for name,digest in p['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==digest
    for name,digest in p['inputs'].items():assert teacher.sha(BASE/name)==digest,name
    return p


def verify_folder(folder):
    """完成目录不能重写。"""
    for name,digest in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==digest


@torch.no_grad()
def four_for_record(model,source,panel,record,search,units):
    """复原上轮实际四状态，只读取被冻结的单个干预端点。"""
    case=record['case'];m=(3,15).index(record['mask']);recipient,donor=panel['pairs'][case]
    half,full=two_traces(model,source,recipient,donor)
    specs=intervention_specs({k:v[case,m] for k,v in units.items()},{k:v[case,m] for k,v in search.items()})
    before,after=endpoints(model,half,full,record['mask'],[specs[record['endpoint']]])
    fields=four_fields(before,after,record['C'])[0]
    weight=np.zeros(3);weight[panel['labels'][donor]]=1.;weight[panel['labels'][recipient]]=-1.
    return fields,weight


def initialize():
    """看本轮定位结果前冻结队列、预算、排名及未见输入限制。"""
    assert not RUN.exists();pre=BASE/'reports/dsprites_downstream_routes_preflight_20261025.json';assert teacher.read(pre)['status']=='PASS'
    preserved=preservation();records=cohort();inputs=[pre,PRIOR/'tables/effects.csv',PRIOR/'protocol.json',SHAPE/'panel/data.npz']
    inputs += [BASE/p for p in teacher.read(PRIOR/'protocol.json')['inputs'] if p.startswith('scripts/')]
    inputs += [BASE/'scripts'/n for n in ('coalition_context_mechanism.py','run_dsprites_coalition_context.py')]
    for seed in teacher.SEEDS:
        inputs += [teacher.RUN/f'seed_{seed}/teacher.pt',PRIOR/f'evaluation/seed_{seed}/data.npz']
        inputs += [WINDOW/f'search/seed_{seed}/{n}' for n in ('data.npz','units.npz','frozen.npz')]
    RUN.mkdir();shutil.copy2(pre,RUN/'preflight.json')
    for name in CODE:
        dst=RUN/'source/scripts'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/'scripts'/name,dst)
    p=dict(version='downstream-routes-v1',utc=teacher.now(),client_date='2026-10-25',cohort=records,
        scope='same36images/6frozen samearchitecture teachers;37 inherited case-method records incl26 primary/4decomposition/6failedsource/1priorreferencefailure;no training or inputexpansion',
        contexts=['old outside:C00to10','new outside:C01to11'],reference='matchedC-off realnonnegativeactivations within eachoutsidecontext;not oldA1 or crosscontext00clamp',
        localization='exact singlepoint outputloss on restoringC-off value;firstReLU local3x3 secondReLU recompute;lastReLU outputlinear;wholechannel first nativefinish;abs outputloss ranks',
        selection_caveat='output-guided samepanel exploratory localization;notheldout confirmation,no minimalunique circuitclaim',
        point_budgets=list(core.POINT_BUDGETS),channel_budgets=list(core.CHANNEL_BUDGETS),point_orders=list(core.ORDERS),specials=list(core.SPECIALS),
        random='two deterministic uniform permutations of samechangedpoint/changedchannel pool;matched budget within pool;specialswitch/stable/sign pools haveowncap',
        sign='support/opposing relative tosign ofpairedC effect;zeroeffect uses+1 convention;no directionalclaimfornull controls',
        freeze='all37 records two-context/twolayer rankedlists frozen before anyjointpatch evaluation',
        operations='block selected changes inC-on;keep onlyselectedchanges inC-off;firstReLU each run withskipoff andskipon;67commands perlayer/context;lastpostReLU linearreadout has no skipoption',
        residual_path='2x2 native firstReLUoff/on versusidentityskipoff/on before finalReLU;fullfirstblockleavesskiponly;bothpathsblockedzero positivecontrol',
        nonadditivity='first jointremoved-minus-sumsingles and selected/complement interaction are secondReLU effects;terminal jointadditivity byconstruction',
        residual_map='top128 terminalsignedreadout residualpoints after predeclared point256/channel4 blocks,fullchannelandspatialpositive/negative residuals for allcommands',
        epsilon=core.EPS,tolerance=1e-9,summaries='primary26 with teacher-awareRMS;5successfulteachers and1empty;controlsseparate;do notpool dependentrecords asindependent samples',
        example=dict(seed=202609292,case=70,mask=15,endpoint=25,reason='previous fixedpositive,notchosenfromnewresults'),
        limitations='hiddenhybrids mayoffmanifold;allchangedfirstactivationclamp exhaustivecontrol notsparsecircuit;selection usesoutputs;necessaryandsufficientrelative topatchspecification;no accuracy/unseeninput/unique mediationfraction',
        sources={f'scripts/{n}':teacher.sha(BASE/'scripts'/n) for n in CODE},inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in set(inputs)},preservation=preserved)
    teacher.dump(RUN/'protocol.json',p);(RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n',encoding='ascii');event('protocol_locked',records=len(records),old_entries=1711)


def selection_seed(record,context):
    """固定随机名单种子，不使用模型输出。"""
    return record['seed']+2026102500+record['case']*97+record['mask']*13+record['endpoint']*1009+context*11


def discover(device):
    """单点定位完成后整体封存，联合阻断不得再优化名单。"""
    p=verify_protocol();assert not (RUN/'selection_freeze.json').exists();panel=load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for v in model.parameters():v.requires_grad_(False)
        source=sources_for(model,panel,device);search=load(WINDOW/f'search/seed_{seed}/data.npz');units=load(WINDOW/f'search/seed_{seed}/units.npz')
        for record in [r for r in p['cohort'] if r['seed']==seed]:
            folder=RUN/f"discovery/record_{record['record']:03d}"
            if folder.exists():verify_folder(folder);continue
            fields,weight=four_for_record(model,source,panel,record,search,units);out={}
            for context,(off,on) in enumerate(((0,1),(2,3))):
                result=core.locate(model,core.trace(model,fields[off:off+1]),core.trace(model,fields[on:on+1]),weight,selection_seed(record,context))
                for key,v in result.items():out.setdefault(key,[]).append(v)
            folder.mkdir(parents=True);np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in out.items()})
            teacher.dump(folder/'completion.json',dict(utc=teacher.now(),files={'data.npz':teacher.sha(folder/'data.npz')}));event('located',record=record['record'],seed=seed)
    paths=[p for p in (RUN/'discovery').rglob('*') if p.is_file()]
    teacher.dump(RUN/'selection_freeze.json',dict(utc=teacher.now(),files={str(p.relative_to(RUN)).replace('\\','/'):teacher.sha(p) for p in paths}))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n',encoding='ascii');event('all_localization_frozen')


def verify_selection():
    """全部定位结果与随机名单的不可变承诺。"""
    path=RUN/'selection_freeze.json';assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip();p=teacher.read(path)
    for name,digest in p['files'].items():assert teacher.sha(RUN/name)==digest
    return p


def evaluate(device):
    """同名单执行阻断/恢复、残差支路分离，保存失败和仅分解控制。"""
    p=verify_protocol();verify_selection();panel=load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for v in model.parameters():v.requires_grad_(False)
        source=sources_for(model,panel,device);search=load(WINDOW/f'search/seed_{seed}/data.npz');units=load(WINDOW/f'search/seed_{seed}/units.npz');prior=load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        for record in [r for r in p['cohort'] if r['seed']==seed]:
            folder=RUN/f"evaluation/record_{record['record']:03d}"
            if folder.exists():verify_folder(folder);continue
            fields,weight=four_for_record(model,source,panel,record,search,units);selection=load(RUN/f"discovery/record_{record['record']:03d}/data.npz");out={}
            for context,(off,on) in enumerate(((0,1),(2,3))):
                result=core.evaluate(model,core.trace(model,fields[off:off+1]),core.trace(model,fields[on:on+1]),weight,{k:v[context] for k,v in selection.items()},record['point'])
                for key,v in result.items():out.setdefault(key,[]).append(v)
                slot=(tuple(range(25,37))+(1,4,7,10)).index(record['endpoint']);expected=prior['logits'][record['case'],(3,15).index(record['mask']),slot,[off,on],0]
                assert abs(result['native_logits']-expected).max()<1e-9
            folder.mkdir(parents=True);np.savez_compressed(folder/'data.npz',**{k:np.asarray(v) for k,v in out.items()})
            teacher.dump(folder/'completion.json',dict(utc=teacher.now(),files={'data.npz':teacher.sha(folder/'data.npz')}));event('evaluated',record=record['record'],seed=seed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('init','discover','evaluate'),required=True);args=parser.parse_args()
    if args.stage=='init':initialize()
    elif args.stage=='discover':discover(teacher.configure())
    else:evaluate(teacher.configure())

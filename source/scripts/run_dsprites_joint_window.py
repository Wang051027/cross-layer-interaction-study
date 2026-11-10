"""按阈值余量搜索联合必要的上游干预，再运行完整网络测量输出。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import joint_window_mechanism as core
from joint_window_measure import measure
from run_dsprites_upstream_rescue import OLDER as OLD, load, sources_for, two_traces

BASE = teacher.BASE
RUN = BASE/'runs/dsprites_joint_window_20261018'
PRIOR = BASE/'runs/dsprites_upstream_rescue_20261017'
AR = BASE/'runs/dsprites_ar_factorial_20261017'
SHAPE = BASE/'runs/dsprites_shape_donor_20261010'
OLDER = OLD+('dsprites_upstream_rescue_20261017',)
CODE = ('joint_window_mechanism.py', 'joint_window_measure.py', 'run_dsprites_joint_window.py', 'test_dsprites_joint_window.py')


def event(name, **values):
    """日志不参与选点；只记录阶段次序、耗时与完成情况。"""
    row = dict(utc=teacher.now(), event=name, **values)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def verify_protocol():
    """冻结数值代码、名单、模型、旧结果和协议。"""
    p = teacher.read(RUN/'protocol.json'); assert teacher.sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    for name, digest in p['sources'].items(): assert teacher.sha(BASE/name) == teacher.sha(RUN/'source'/name) == digest, name
    for name, digest in p['inputs'].items(): assert teacher.sha(BASE/name) == digest, name
    return p


def verify_folder(folder):
    """校验已完成阶段的文件清单。"""
    for name, digest in teacher.read(folder/'completion.json')['files'].items(): assert teacher.sha(folder/name) == digest


def initialize():
    """看新搜索结果前锁定联合必要性定义、三个参照、预算、强度和选解规则。"""
    assert not RUN.exists(); pre = BASE/'reports/dsprites_joint_window_preflight_20261018.json'; assert teacher.read(pre)['status'] == 'PASS'
    preservation = {}; total = 0
    for name in OLDER:
        path = BASE/f'runs/{name}/delivery_manifest.json'; rows = teacher.read(path)['files']
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        for row in rows: assert teacher.sha(BASE/row['path']) == row['sha256'], row['path']
        preservation[str(path.relative_to(BASE)).replace('\\', '/')] = dict(sha256=teacher.sha(path), files=len(rows)); total += len(rows)
    assert total == 1558; teacher.verify_old(); RUN.mkdir(); shutil.copy2(pre, RUN/'preflight.json')
    inputs = [pre, SHAPE/'panel/data.npz', PRIOR/'selection_freeze.json', PRIOR/'selection_freeze.sha256']
    for seed in teacher.SEEDS:
        inputs += [teacher.RUN/f'seed_{seed}/teacher.pt', PRIOR/f'discovery/seed_{seed}/selection.npz',
                   PRIOR/f'discovery/seed_{seed}/frozen_prior.npz', AR/f'evaluation/seed_{seed}/data.npz']
    dependencies = ('dsprites_teacher.py', 'residual_teacher.py', 'external_experiment.py', 'research_runtime.py',
                    'run_dsprites_upstream_rescue.py', 'upstream_rescue_mechanism.py', 'upstream_rescue_measure.py',
                    'run_dsprites_ar_factorial.py', 'ar_factorial_mechanism.py', 'joint_relu_mechanism.py',
                    'residual_route_mechanism.py', 'audit_dsprites_gradient.py', 'layer_transfer_mechanism.py',
                    'shape_donor_mechanism.py', 'dsprites_mechanism.py', 'cross_layer_mechanism.py')
    inputs += [BASE/'scripts'/n for n in dependencies]
    for name in CODE:
        dst = RUN/'source/scripts'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(BASE/'scripts'/name, dst)
    p = dict(version='joint-window-v1', utc=teacher.now(), scope='same36images/72dependentpairs/6frozen samemodel teachers;source masks3/15;no newimages/training',
        fixed='original C,firstpointj,direction and upstreamselected/randomlists inheritedverbatim',
        interventions=list(core.FAMILIES), activation_rule='(1-lambda)full+lambda correspondinghalf atfixedunits;source/middlechannels orfirst/secondnonnegativeactivationcoordinates;subsequent originalReLUs rerun',
        paths='4families*3variants*4prefixes=48;source/mid1,2,3,4;gate1,4,8,16;clamp toprioractualbudget;duplicateprefix invalidated',
        grid=core.GRID.tolist(), refine='17pointswithinone gridstep ofbest coarsepathpoint;reselectwithin samepath;notexhaustivecontinuousoptimization',
        contexts=list(core.CONTEXTS), definitions='Cexactnecessary iffbaselineoriginalside,allCcross,andallleaveoneoutfail;this impliesallproperCsubsetsfail',
        fixed_C='A1+P_C(HI-A1);doesnotdependonAI;actualcontext=HI withCclampedtooriginalA1 forbaseline/deletions',
        objectives=list(core.STRATEGIES), selection='movingA;min(fixedA1,actualcontext);min(allthree);maxsafety q=min(base/full/worstdrop margins);feasibleprefersfewerunits;otherwisehighestfailedscore;nooutputtaskscore',
        guard=core.GUARD, epsilon=core.EPS, neighbor='center +/-1/256 clipped0..1;testedstability not certifiedcontinuousinterval',
        random='samefrozeneligiblepool/maxbudget/prefix/strengthsearch;optimalactualbudgetmaydiffer;notclaimsame selectedunitcount',
        separation='HIchanged vsHIunchanged;rawF andcommonoriginalA1 firstjcontrol;decompositiononly cannotchangefixedreference geometry',
        endpoints='natural+36solutions+allsource16/allmiddle32positivecontrols=39;fullforward afterall6selections sealed',
        actual_C_drops='natural andalignedselected0fourfamilies;originalA1 channelclamp;raw3logits plusjpreactivation',
        primary='previous539samebase/Cnotcross;all864retained;teacher-equal summaries;noindependentpairpvalues',
        example=dict(seed=202609291, case=1, mask=3, reason='samepredefined priorfailure,notchosenbynewsuccess'),
        limitations='samepanel adaptiveinternal search;no unseeninputorcrossarchitectureoraccuracybenefit;fixedA1 maynegative andpointgateholddiagnostic;no unique mediationfraction',
        sources={f'scripts/{n}': teacher.sha(BASE/'scripts'/n) for n in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in inputs}, preservation=preservation)
    teacher.dump(RUN/'protocol.json', p); (RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n', encoding='ascii')
    event('protocol_locked', sha256=teacher.sha(RUN/'protocol.json'), old_entries=total)


def search(device):
    """仅局部精确几何搜索；任务读出直到全部名单封存后才计算。"""
    verify_protocol(); assert not (RUN/'selection_freeze.json').exists(); panel = load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        folder = RUN/f'search/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device); old = load(PRIOR/f'discovery/seed_{seed}/selection.npz'); fixed = load(PRIOR/f'discovery/seed_{seed}/frozen_prior.npz'); values = {}
        for case, (recipient, donor) in enumerate(panel['pairs']):
            half, full = two_traces(model, source, recipient, donor); row = {}
            for m, mask in enumerate((3, 15)):
                chosen = {k: v[case, m] for k, v in old.items()}; frozen = {k: v[case, m] for k, v in fixed.items()}
                c = core.cache(model, half, full, int(frozen['lists'][0, 0]), mask, frozen); out = core.search(c, chosen)
                for k, v in out.items(): row.setdefault(k, []).append(v)
            for k, v in row.items(): values.setdefault(k, []).append(v)
            if (case+1) % 12 == 0: event('search_progress', seed=seed, cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'data.npz', **{k: np.asarray(v) for k, v in values.items()})
        for name, src in (('units.npz', 'selection.npz'), ('frozen.npz', 'frozen_prior.npz')): shutil.copy2(PRIOR/f'discovery/seed_{seed}/{src}', folder/name)
        teacher.dump(folder/'completion.json', dict(utc=teacher.now(), seed=seed, seconds=time.perf_counter()-start,
                     files={p.name: teacher.sha(p) for p in folder.iterdir()})); event('search_complete', seed=seed, seconds=time.perf_counter()-start)
    teacher.dump(RUN/'selection_freeze.json', dict(utc=teacher.now(), files={f'search/seed_{s}/{n}': teacher.sha(RUN/f'search/seed_{s}/{n}') for s in teacher.SEEDS for n in ('data.npz', 'units.npz', 'frozen.npz', 'completion.json')}))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n', encoding='ascii'); event('all_solutions_frozen')


def verify_selection():
    """评价阶段禁止修改任何模型的强度、路径或名单。"""
    path = RUN/'selection_freeze.json'; assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    p = teacher.read(path)
    for name, digest in p['files'].items(): assert teacher.sha(RUN/name) == digest
    return p


def evaluate(device):
    """39端点整图重算及C逐成员消融，保留没有满足条件的最佳失败解。"""
    verify_protocol(); freeze = verify_selection(); panel = load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        folder = RUN/f'evaluation/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device); result = load(RUN/f'search/seed_{seed}/data.npz')
        units = load(RUN/f'search/seed_{seed}/units.npz'); fixed = load(RUN/f'search/seed_{seed}/frozen.npz'); values = {}
        for case, (recipient, donor) in enumerate(panel['pairs']):
            half, full = two_traces(model, source, recipient, donor); row = {}
            weight = np.zeros(3); weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
            for m, mask in enumerate((3, 15)):
                chosen = {k: v[case, m] for k, v in units.items()}; frozen = {k: v[case, m] for k, v in fixed.items()}; answer = {k: v[case, m] for k, v in result.items()}
                c = core.cache(model, half, full, int(frozen['lists'][0, 0]), mask, frozen)
                seven, unique = core.full_fields(model, half, full, core.intervention_specs(chosen, answer))
                out = measure(model, seven, full, mask, frozen, weight, c); out['unique_count'] = np.array(unique)
                assert np.max(abs(out['margins'][1:37]-answer['selected_margins'])) < 1e-9
                assert np.all(out['margins'][37:, 0] > core.EPS)
                for k, v in out.items(): row.setdefault(k, []).append(v)
            for k, v in row.items(): values.setdefault(k, []).append(v)
            if (case+1) % 12 == 0: event('evaluation_progress', seed=seed, cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'data.npz', **{k: np.asarray(v) for k, v in values.items()})
        teacher.dump(folder/'completion.json', dict(utc=teacher.now(), selection_utc=freeze['utc'], seed=seed, seconds=time.perf_counter()-start,
                     files={p.name: teacher.sha(p) for p in folder.iterdir()})); event('evaluation_complete', seed=seed, seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', choices=('init', 'search', 'evaluate'), required=True)
    args = parser.parse_args()
    if args.stage == 'init': initialize()
    elif args.stage == 'search': search(teacher.configure())
    else: evaluate(teacher.configure())

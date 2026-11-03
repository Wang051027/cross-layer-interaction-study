"""同36图六模型的基准上游定位：先封存协议，再定位，最后评价救回与输出。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import upstream_rescue_mechanism as core
from upstream_rescue_measure import measure
from run_dsprites_ar_factorial import OLDER as PRE_AR, load

BASE = teacher.BASE
RUN = BASE/'runs/dsprites_upstream_rescue_20261017'
PRIOR = BASE/'runs/dsprites_ar_factorial_20261017'
JOINT = BASE/'runs/dsprites_joint_relu_stability_20261017'
SHAPE = BASE/'runs/dsprites_shape_donor_20261010'
OLDER = PRE_AR+('dsprites_ar_factorial_20261017',)
CODE = ('upstream_rescue_mechanism.py', 'upstream_rescue_measure.py',
        'run_dsprites_upstream_rescue.py', 'test_dsprites_upstream_rescue.py')


def event(name, **details):
    """阶段日志仅用于进度和先后核验，不参与选择。"""
    row = dict(utc=teacher.now(), event=name, **details)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def verify_protocol():
    """数值代码与全部冻结输入逐项比对，任何阶段都不能静默改协议。"""
    p = teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    for name, digest in p['sources'].items(): assert teacher.sha(BASE/name) == teacher.sha(RUN/'source'/name) == digest, name
    for name, digest in p['inputs'].items(): assert teacher.sha(BASE/name) == digest, name
    return p


def verify_folder(folder):
    """完成子目录只读核验，拒绝覆盖结果。"""
    for name, digest in teacher.read(folder/'completion.json')['files'].items(): assert teacher.sha(folder/name) == digest


def init():
    """先锁定问题、候选池、预算、对照、例子及所有旧来源。"""
    assert not RUN.exists()
    pre = BASE/'reports/dsprites_upstream_rescue_preflight_20261017.json'; assert teacher.read(pre)['status'] == 'PASS'
    preservation = {}; total = 0
    for name in OLDER:
        path = BASE/f'runs/{name}/delivery_manifest.json'; m = teacher.read(path)
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        for r in m['files']:
            p = BASE/r['path']; assert teacher.sha(p) == r['sha256'] and p.stat().st_size == r['bytes'], r['path']
        preservation[str(path.relative_to(BASE)).replace('\\', '/')] = dict(sha256=teacher.sha(path), files=len(m['files']))
        total += len(m['files'])
    assert total == 1467; teacher.verify_old(); RUN.mkdir(); shutil.copy2(pre, RUN/'preflight.json')
    inputs = [pre, SHAPE/'panel/data.npz', PRIOR/'protocol.json', PRIOR/'protocol.sha256',
              JOINT/'selection_freeze.json', JOINT/'selection_freeze.sha256']
    for seed in teacher.SEEDS:
        inputs += [teacher.RUN/f'seed_{seed}/teacher.pt', PRIOR/f'evaluation/seed_{seed}/data.npz',
                   JOINT/f'discovery/seed_{seed}/selection.npz']
    dependencies = ('dsprites_teacher.py', 'residual_teacher.py', 'external_experiment.py', 'research_runtime.py',
                    'residual_route_mechanism.py', 'audit_dsprites_gradient.py', 'layer_transfer_mechanism.py',
                    'shape_donor_mechanism.py', 'dsprites_mechanism.py', 'cross_layer_mechanism.py',
                    'run_dsprites_ar_factorial.py', 'ar_factorial_mechanism.py', 'joint_relu_mechanism.py')
    inputs += [BASE/'scripts'/n for n in dependencies]
    for name in CODE:
        dst = RUN/'source/scripts'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(BASE/'scripts'/name, dst)
    protocol = dict(version='upstream-rescue-v1', utc=teacher.now(),
        scope='same36images/72dependent directed pairs/6frozen samearchitecture teachers;source masks3/15;noinput expansion/training',
        states=list(core.STATES), methods=list(core.METHODS), epsilon=core.EPS,
        definitions='A=sum(single anchor H)-(k-1)Hempty;R=Htarget-A;allI applied consistently to7source states before reconstructingA/R',
        reference='halfstrength corresponding source state for each row,not globalempty;fullstrength is intervention start',
        fixed='original C/block3first j/direction copiedfrom priorhalfselection;never reranked',
        ranking='only s*(bI-b1), b=block3firstpreactivation(A) at frozenj;no residual contribution/grouprescue/tasklogit used',
        candidate_pools='source8or16 wholechannels;middle32 wholechannels;gate1 valid5x5x32/gate2 valid3x3x32,any anchor remainderabs>1e-8',
        selection='positive repairscore>1e-8,top<=4source/middle,<=16gates;physicalindex ties;zero positive gives no-op retained',
        random='two controls eachsamefamily,eligiblepool,actualbudget;default_rng(seed+20261017000+case*97+mask)',
        interventions='source/middle amplitudes rolledto correspondinghalf;gate fixedto correspondinghalfmask times currentz;subsequent layers recomputed',
        regimes=['A_only: AI+P_C R1 / AI+R1', 'coherent: AI+P_C RI / HI=AI+RI'],
        threshold='same baseline side;originalC cross;all32 single-channel solos safe;strictjoint=cross andsolossafe;fixed originalC not necessarily minimumcurrentgroup',
        primary='prior539 baselineunchanged/Cno-cross states;all864 retained;targeted familiesvs ownrandom;A_only vscoherent',
        decomposition='Delta b=sumanchors[Jhalf DeltaU+propagatedfirstReLUremainder+secondReLUremainder];linear/source16 andgate32 signedparts',
        outputs='F=donor-recipient logit;store raw3logitsF(AI),F(X),heldpoint;d=F(X)-F(AI);coherentall32 is fullupstream network effect',
        denominators='same original fullall32 d11 RMS within teacher/mask;equal6teacher means;noindependentpair pvalues',
        planned_example=dict(seed=202609291, case=1, mask=3, reason='first ofprior539 byfixed ordering,notselectedby newrescue'),
        limitations='samepanel exploratory intervention;internaloffmanifold holds maynegative;noaccuracybenefit/unseeninputs/crossarchitecture/permanentcircuit/uniquecausalshare',
        sources={f'scripts/{n}': teacher.sha(BASE/'scripts'/n) for n in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in inputs}, preservation=preservation)
    teacher.dump(RUN/'protocol.json', protocol); (RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n', encoding='ascii')
    event('protocol_locked', sha256=teacher.sha(RUN/'protocol.json'), old_entries=total)


def sources_for(model, panel, device):
    """只复用已冻结36图，不读取其他图像或标签。"""
    with torch.no_grad():
        pixels = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
        return torch.cat([model.block1(model.stem(x)) for x in pixels.split(16)])


def two_traces(model, source, recipient, donor):
    """相同七状态的半/全强度，整个后续流程复用它们。"""
    return [core.upstream_trace(model, core.source_fields(source[recipient:recipient+1], source[donor:donor+1], alpha)) for alpha in (.5, 1.)]


def discover(device):
    """全部模型的基准名单完成并哈希封存后，才允许运行下游评价。"""
    verify_protocol(); panel = load(SHAPE/'panel/data.npz')
    assert not (RUN/'selection_freeze.json').exists()
    for seed in teacher.SEEDS:
        folder = RUN/f'discovery/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device); frozen = load(JOINT/f'discovery/seed_{seed}/selection.npz'); values = {}
        for case, (recipient, donor) in enumerate(panel['pairs']):
            half, full = two_traces(model, source, recipient, donor); row = {}
            for m, mask in enumerate(core.MASKS):
                old = {k: v[case, m] for k, v in frozen.items()}
                out = core.discover(model, half, full, mask, old, seed+20261017000+case*97+mask)
                for k, v in out.items(): row.setdefault(k, []).append(v)
            for k, v in row.items(): values.setdefault(k, []).append(v)
            if (case+1) % 24 == 0: event('discovery_progress', seed=seed, cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'selection.npz', **{k: np.asarray(v) for k, v in values.items()})
        shutil.copy2(JOINT/f'discovery/seed_{seed}/selection.npz', folder/'frozen_prior.npz')
        teacher.dump(folder/'completion.json', dict(utc=teacher.now(), seed=seed, seconds=time.perf_counter()-start,
                     files={p.name: teacher.sha(p) for p in folder.iterdir()}))
        event('discovery_complete', seed=seed, seconds=time.perf_counter()-start)
    teacher.dump(RUN/'selection_freeze.json', dict(utc=teacher.now(), seeds=teacher.SEEDS,
                 files={f'discovery/seed_{s}/{n}': teacher.sha(RUN/f'discovery/seed_{s}/{n}') for s in teacher.SEEDS
                        for n in ('selection.npz', 'frozen_prior.npz', 'completion.json')}))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n', encoding='ascii')
    event('all_baseline_selections_frozen', sha256=teacher.sha(RUN/'selection_freeze.json'))


def verify_selection():
    """防止评价期间改变排名名单。"""
    path = RUN/'selection_freeze.json'; assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    p = teacher.read(path)
    for name, digest in p['files'].items(): assert teacher.sha(RUN/name) == digest
    return p


def evaluate(device):
    """22方法×两种R控制×两种通道范围，全部案例都保留。"""
    verify_protocol(); freeze = verify_selection(); panel = load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        folder = RUN/f'evaluation/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        source = sources_for(model, panel, device)
        sel = load(RUN/f'discovery/seed_{seed}/selection.npz'); frozen = load(RUN/f'discovery/seed_{seed}/frozen_prior.npz')
        old = load(PRIOR/f'evaluation/seed_{seed}/data.npz'); values = {}
        for case, (recipient, donor) in enumerate(panel['pairs']):
            half, full = two_traces(model, source, recipient, donor); row = {}
            weight = np.zeros(3); weight[int(panel['labels'][donor])] = 1.; weight[int(panel['labels'][recipient])] = -1.
            for m, mask in enumerate(core.MASKS):
                chosen = {k: v[case, m] for k, v in sel.items()}; original = {k: v[case, m] for k, v in frozen.items()}
                a, r = core.intervene(model, half, full, mask, chosen)
                _, r1 = core.reconstruct(full['h'], mask)
                out = measure(model, a, r, r1, weight, original)
                # 三个自然端点须复现上一轮00/01/11，避免改了问题却没有察觉。
                assert np.max(abs(out['natural'][0, 0]-old['natural'][case, m, :, 3])) < 1e-9
                for n in (5, 10):
                    assert np.max(abs(out['natural'][n, 0]-old['natural'][case, m, :, 1])) < 1e-9
                    assert np.max(abs(out['natural'][n, 1]-old['natural'][case, m, :, 0])) < 1e-9
                ids = np.array(core.anchors(mask))-1
                assert abs(out['base_z'][21]-chosen['base_z'][0]-chosen['source_linear'][ids].sum()) < 1e-9
                for k, v in out.items(): row.setdefault(k, []).append(v)
            for k, v in row.items(): values.setdefault(k, []).append(v)
            if (case+1) % 12 == 0: event('evaluation_progress', seed=seed, cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'data.npz', **{k: np.asarray(v) for k, v in values.items()})
        teacher.dump(folder/'completion.json', dict(utc=teacher.now(), selection_utc=freeze['utc'], seed=seed,
                     seconds=time.perf_counter()-start, files={p.name: teacher.sha(p) for p in folder.iterdir()}))
        event('evaluation_complete', seed=seed, seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', required=True, choices=('init', 'discover', 'evaluate'))
    args = parser.parse_args()
    if args.stage == 'init': init()
    elif args.stage == 'discover': discover(teacher.configure())
    else: evaluate(teacher.configure())

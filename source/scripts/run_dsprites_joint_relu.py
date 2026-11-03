"""冻结协议→半强度选点选通道→封存→全强度，不增加输入或训练。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import joint_relu_mechanism as core
from layer_transfer_mechanism import design, lower_fields
from shape_donor_mechanism import replace_groups, pair_weights
from residual_route_mechanism import trace

BASE = teacher.BASE
RUN = BASE / 'runs/dsprites_joint_relu_stability_20261017'
SHAPE = BASE / 'runs/dsprites_shape_donor_20261010'
PRIOR = BASE / 'runs/dsprites_residual_routes_20261013'
CODE = ('joint_relu_mechanism.py', 'run_dsprites_joint_relu.py', 'test_dsprites_joint_relu.py')
OLDER = ('dsprites_20260929', 'dsprites_factor_transfer_20261007', 'dsprites_gradient_20261009',
         'dsprites_shape_donor_20261010', 'dsprites_channel_gates_20261011', 'dsprites_triple_gates_20261011',
         'dsprites_layer_transfer_20261013', 'dsprites_residual_routes_20261013')


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def event(name, **details):
    row = dict(utc=teacher.now(), event=name, **details)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def verify_protocol():
    p = teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    for name, value in p['sources'].items():
        assert teacher.sha(BASE/name) == teacher.sha(RUN/'source'/name) == value, name
    for name, value in p['inputs'].items(): assert teacher.sha(BASE/name) == value, name
    return p


def verify_folder(folder):
    for name, value in teacher.read(folder/'completion.json')['files'].items():
        assert teacher.sha(folder/name) == value, name


def init():
    assert not RUN.exists(), '新运行不得覆盖旧交付'
    pre = BASE/'reports/dsprites_joint_relu_preflight_20261017.json'
    assert teacher.read(pre)['status'] == 'PASS'
    preservation = {}
    for name in OLDER:
        path = BASE/f'runs/{name}/delivery_manifest.json'; old = teacher.read(path)
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        for item in old['files']: assert teacher.sha(BASE/item['path']) == item['sha256'], item['path']
        preservation[str(path.relative_to(BASE)).replace('\\', '/')] = dict(sha256=teacher.sha(path), files=len(old['files']))
    teacher.verify_old(); RUN.mkdir(); shutil.copy2(pre, RUN/'preflight.json')
    # 依赖和旧输出都固定哈希，复核旧自然/基准端点，而不沿用旧选点。
    inputs = [pre, SHAPE/'panel/data.npz'] + [teacher.RUN/f'seed_{s}/teacher.pt' for s in teacher.SEEDS]
    inputs += [PRIOR/f'evaluation/seed_{s}/alpha_{a}/data.npz' for s in teacher.SEEDS for a in (.5, 1.)]
    dependencies = ('dsprites_teacher.py', 'residual_teacher.py', 'external_experiment.py', 'research_runtime.py',
                    'shape_donor_mechanism.py', 'layer_transfer_mechanism.py', 'residual_route_mechanism.py',
                    'audit_dsprites_gradient.py', 'dsprites_mechanism.py', 'cross_layer_mechanism.py',
                    'test_dsprites_residual_routes.py')
    inputs += [BASE/'scripts'/n for n in dependencies]
    for name in CODE:
        destination = RUN/'source/scripts'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/'scripts'/name, destination)
    p = dict(version='joint-relu-stability-v1', utc=teacher.now(),
             scope='same36images/72dependent directedpairs/6frozen teachers;no newinput ortraining',
             source_masks=list(core.MASKS), source='block1 four4channel groups;mask3=G0G1;mask15=all4',
             middle='A(S)=H<=1(S);R(S)=H(S)-A(S);32binary whole32x32 residualchannel additions',
             target='endpoint d(S)=F(H)-F(A),F=donor minus recipient logit;not24source contrasts or pure source orders',
             gate='block3 firstReLU only;second nonlinear layer recomputed for allactual outputinterventions',
             census='all32768first coordinates at each fixed endpoint;robust natural flips and exactminimum count by affine32term sorting;fullcensus descriptiveonly',
             epsilon=core.EPS, proof='directional k-1 topterms bound stays before zero byEPS;ktopterms crosses byEPS;ties physicalindex',
             half_pool='robust endpointflip and proven minimumk>=2;ambiguous threshold bounds excluded andcounted',
             discovery='alpha=.5;selecttop32 abs(baseline-gradient-weighted firstReLU remainder),score<1e-12 zero;physicalindex ties;no actualnet rerank',
             random='two fixed permutations ofsamejoint pool;seed=teacher_seed+20261017000+case*97+mask;budget=min32,pool;top1=first listpoint',
             channel_selection='forselected top1 only,kmost favorable half-strength wholechannels;freezecoordinates,lists,pools,k,direction,channels beforefull',
             full='alpha1 doesnotrerank anygate/channel;currentminimum atfrozen point isdiagnostic,notnew interventionchoice',
             holds=list(core.HOLD_NAMES), patches=['A', 'half_frozen_first_kminus1', 'half_frozen_first_k', 'H_all32'],
             primary=['fullfrozen top1 same-direction endpointjoint rate','fullfrozen channelset joint threshold rate withall32solos safe',
                      'actual singlepoint and32point outputnet RMS/dRMS andsignedprojection versus2samepoolrandom lists'],
             secondary=['all-half-joint andall-firstholds','fullcensus minimum histogram','direct firstremainder andsecondresponse cancellation',
                        'patchkminus1 to k outputjump vs pointnet change','threeclass argmax changes under internal intervention'],
             effect_thresholds='report rawlogit absnet>=.01 and>=.1;relative absnet>=.1absd onlywhenabsd>=.1;descriptive scales,not scientific significance',
             aggregation='perteacher andsourcemask first;6teacher equalmean;all432directedcases/864states retained;noindependentpair pvalues',
             known='previousfull36panel results alreadyseen;thisisnewfrozen strength validation onsameinputs,notunseen-input confirmation',
             limitations='samearchitecture6seeds;selectionlocal to eachcase/model;A andopen-gatehold mayoffmanifold;minchannel notminoutputcircuit;no universal channelidentity orclassificationbenefit',
             sources={f'scripts/{n}': teacher.sha(BASE/'scripts'/n) for n in CODE},
             inputs={str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in inputs}, preservation=preservation)
    teacher.dump(RUN/'protocol.json', p)
    (RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n', encoding='ascii')
    event('protocol_locked', sha256=teacher.sha(RUN/'protocol.json'))


def frame(seed, alpha, device, selection=None):
    panel = load(SHAPE/'panel/data.npz'); model = teacher.load_teacher(seed, device); d = design()
    for p in model.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        x = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
        source = torch.cat([model.block1(model.stem(v)) for v in x.split(16)])
    old = load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
    values = {}; selected = {}; bits = torch.as_tensor(d['bits'], device=device)
    for case, (recipient, donor) in enumerate(panel['pairs']):
        with torch.no_grad():
            h = model.block2(replace_groups(source[recipient:recipient+1], source[donor:donor+1], bits, alpha))
            a = lower_fields(h, d)[1][0]
        w = pair_weights(panel['labels'][recipient], panel['labels'][donor]); outputs = {}; choices = {}
        for m, mask in enumerate(core.MASKS):
            t = trace(model, a[mask:mask+1], h[mask:mask+1], w); population = core.census(model, t)
            fixed = None if selection is None else {k: v[case, m] for k, v in selection.items()}
            chosen = core.select(t, population, seed+20261017000+case*97+mask) if fixed is None else fixed
            out = core.measure(model, a[mask:mask+1], h[mask:mask+1], w, population, chosen)
            assert np.max(abs(out['natural']-old['modes'][case, mask, 0])) < 1e-9
            assert np.max(abs(out['baseline']-old['baseline'][case, mask])) < 1e-9
            for key, val in out.items(): outputs.setdefault(key, []).append(val)
            for key, val in chosen.items(): choices.setdefault(key, []).append(val)
        for key, val in outputs.items(): values.setdefault(key, []).append(val)
        for key, val in choices.items(): selected.setdefault(key, []).append(val)
        if (case+1) % 12 == 0: event('frame_progress', seed=seed, alpha=alpha, cases=case+1)
    return {k: np.asarray(v) for k, v in values.items()}, {k: np.asarray(v) for k, v in selected.items()}


def save_frame(folder, seed, alpha, result):
    assert not folder.exists(); folder.mkdir(parents=True)
    raw, selected = result
    np.savez_compressed(folder/'data.npz', **raw); np.savez_compressed(folder/'selection.npz', **selected)
    teacher.dump(folder/'completion.json', dict(utc=teacher.now(), seed=seed, alpha=alpha,
                 files={p.name: teacher.sha(p) for p in folder.iterdir() if p.is_file()}))


def discovery(device):
    verify_protocol(); assert not (RUN/'selection_freeze.json').exists()
    for seed in teacher.SEEDS:
        folder = RUN/f'discovery/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); save_frame(folder, seed, .5, frame(seed, .5, device))
        event('discovery_complete', seed=seed, seconds=time.perf_counter()-start)


def seal():
    verify_protocol(); assert not (RUN/'selection_freeze.json').exists(); files = {}
    for seed in teacher.SEEDS:
        folder = RUN/f'discovery/seed_{seed}'; verify_folder(folder)
        for p in folder.iterdir(): files[str(p.relative_to(RUN)).replace('\\', '/')] = teacher.sha(p)
    teacher.dump(RUN/'selection_freeze.json', dict(utc=teacher.now(), protocol_sha256=teacher.sha(RUN/'protocol.json'), files=files))
    (RUN/'selection_freeze.sha256').write_text(teacher.sha(RUN/'selection_freeze.json')+'\n', encoding='ascii')
    event('selection_locked')


def verify_seal():
    verify_protocol(); freeze = teacher.read(RUN/'selection_freeze.json')
    assert teacher.sha(RUN/'selection_freeze.json') == (RUN/'selection_freeze.sha256').read_text().strip()
    for name, value in freeze['files'].items(): assert teacher.sha(RUN/name) == value, name
    return freeze


def evaluate(device):
    verify_seal()
    for seed in teacher.SEEDS:
        selected = load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (1., .5):
            folder = RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            if folder.exists(): verify_folder(folder); continue
            if alpha == .5:
                folder.mkdir(parents=True); origin = RUN/f'discovery/seed_{seed}'
                for p in origin.iterdir():
                    if p.name != 'completion.json': shutil.copy2(p, folder/p.name)
                teacher.dump(folder/'completion.json', dict(utc=teacher.now(), seed=seed, alpha=alpha,
                             reuse='sealed half discovery', files={p.name: teacher.sha(p) for p in folder.iterdir()}))
            else: save_frame(folder, seed, alpha, frame(seed, alpha, device, selected))
            event('evaluation_complete', seed=seed, alpha=alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('init', 'discovery', 'seal', 'evaluate'))
    args = parser.parse_args()
    if args.stage == 'init': init()
    elif args.stage == 'seal': seal()
    elif args.stage == 'discovery': discovery(teacher.configure())
    else: evaluate(teacher.configure())

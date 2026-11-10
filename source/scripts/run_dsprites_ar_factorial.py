"""在冻结通道名单上交叉组合 A0、A1 和 R0、R1，运行四种干预。"""
from __future__ import annotations
import argparse
import json
import shutil
import time
import numpy as np
import torch
import dsprites_teacher as teacher
import ar_factorial_mechanism as core
from layer_transfer_mechanism import design, lower_fields
from shape_donor_mechanism import replace_groups, pair_weights

BASE = teacher.BASE
RUN = BASE/'runs/dsprites_ar_factorial_20261017'
PRIOR = BASE/'runs/dsprites_joint_relu_stability_20261017'
SHAPE = BASE/'runs/dsprites_shape_donor_20261010'
MASKS = (3, 15)
CODE = ('ar_factorial_mechanism.py', 'run_dsprites_ar_factorial.py', 'test_dsprites_ar_factorial.py')
OLDER = ('dsprites_20260929', 'dsprites_factor_transfer_20261007', 'dsprites_gradient_20261009',
         'dsprites_shape_donor_20261010', 'dsprites_channel_gates_20261011', 'dsprites_triple_gates_20261011',
         'dsprites_layer_transfer_20261013', 'dsprites_residual_routes_20261013', 'dsprites_joint_relu_stability_20261017')


def load(path):
    """只读载入已有数组，禁止pickle对象。"""
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def event(name, **details):
    """记录时间和完成状态，不让日志参与任何定位。"""
    row = dict(utc=teacher.now(), event=name, **details)
    with (RUN/'events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def verify_protocol():
    """每阶段都检查数值脚本和全部冻结来源。"""
    protocol = teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json') == (RUN/'protocol.sha256').read_text().strip()
    for name, value in protocol['sources'].items():
        assert teacher.sha(BASE/name) == teacher.sha(RUN/'source'/name) == value, name
    for name, value in protocol['inputs'].items(): assert teacher.sha(BASE/name) == value, name
    return protocol


def verify_folder(folder):
    """校验已完成阶段的文件清单。"""
    for name, digest in teacher.read(folder/'completion.json')['files'].items(): assert teacher.sha(folder/name) == digest


def init():
    """先封存分析问题、四格定义和既有名单；全部旧交付保持一致。"""
    assert not RUN.exists()
    pre = BASE/'reports/dsprites_ar_factorial_preflight_20261017.json'; assert teacher.read(pre)['status'] == 'PASS'
    preservation = {}
    for name in OLDER:
        path = BASE/f'runs/{name}/delivery_manifest.json'; manifest = teacher.read(path)
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        for item in manifest['files']: assert teacher.sha(BASE/item['path']) == item['sha256'], item['path']
        preservation[str(path.relative_to(BASE)).replace('\\', '/')] = dict(sha256=teacher.sha(path), files=len(manifest['files']))
    teacher.verify_old(); RUN.mkdir(); shutil.copy2(pre, RUN/'preflight.json')
    inputs = [pre, SHAPE/'panel/data.npz', PRIOR/'selection_freeze.json', PRIOR/'selection_freeze.sha256']
    for seed in teacher.SEEDS:
        inputs += [teacher.RUN/f'seed_{seed}/teacher.pt', PRIOR/f'discovery/seed_{seed}/selection.npz']
        inputs += [PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz' for alpha in (.5, 1.)]
    dependencies = ('dsprites_teacher.py', 'residual_teacher.py', 'external_experiment.py', 'research_runtime.py',
                    'layer_transfer_mechanism.py', 'shape_donor_mechanism.py', 'residual_route_mechanism.py',
                    'joint_relu_mechanism.py', 'audit_dsprites_gradient.py', 'dsprites_mechanism.py',
                    'cross_layer_mechanism.py', 'test_dsprites_residual_routes.py')
    inputs += [BASE/'scripts'/name for name in dependencies]
    for name in CODE:
        destination = RUN/'source/scripts'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE/'scripts'/name, destination)
    protocol = dict(version='ar-factorial-v1', utc=teacher.now(),
        scope='same36images/72dependentpairs/6frozen samearchitecture teachers;source masks3/15;no inputexpansion ortraining',
        selection='all previous halfstrength frozenpoint/channel/lists/pools copiedverbatim;no newranking orselection',
        definitions='Ai=H<=1 atalpha .5or1;Rj=H-H<=1 atalpha .5or1;Xij=Ai+Rj;XCij=Ai+P_C Rj',
        cells=list(core.CELLS), contexts=list(core.CONTEXTS), target='dij=F(Xij)-F(Ai),F=donor-recipient logit',
        primary=['joint00/joint01/joint10/joint11 ofsamefrozen channelset;explicit8Boolean patterns',
                 'prior539 samebaseline frozen-group failures:threshold shift vsresidual incoming changes',
                 'd11-d00=(d10-d00)+(d01-d00)+(d11-d10-d01+d00);RMS,signed projection andcancellation',
                 'point/selected32 actualnet,firstdirect,secondresponse andfinal32channel transmission across4cells'],
        geometry='zCij=bi+sumC Djc;firstpreactivation ARcontrastzero;ReLU/outputmaynonzero',
        gate_reference='main holds useAi firstgate;control usescommonA0 gate andsubtracts itsown baseline effectatAi',
        common_control='Erefij=[F(Xij)-Fhold_g0(Xij)]-[F(Ai)-Fhold_g0(Ai)]',
        probe='add exactly+1 tosame firstactivation coordinate atnaturalXij;originalsecondfinish/readout;standardized downstreamresponse,notactualgate attribution',
        exact_net='pointnet=firstremainder*secantgain whenremainder>1e-8;undefined gains masked,notzero observations',
        epsilon=1e-8, aggregation='6teacher equalmeans;separate masks andpooled;no independentpair pvalues',
        denominators='factorial terms vsRMS(y11-y00);crosscell effects also normalized to original all32 d11 RMS;no mixing old24contrasts',
        limitations='hybridA/R andfixedgates are internal off-manifold controls;factorial allocation baseline/order dependent;no unseeninputs/crossarchitecture/uniquecausalshare claims',
        sources={f'scripts/{name}': teacher.sha(BASE/'scripts'/name) for name in CODE},
        inputs={str(p.relative_to(BASE)).replace('\\', '/'): teacher.sha(p) for p in inputs}, preservation=preservation)
    teacher.dump(RUN/'protocol.json', protocol)
    (RUN/'protocol.sha256').write_text(teacher.sha(RUN/'protocol.json')+'\n', encoding='ascii')
    event('protocol_locked', sha256=teacher.sha(RUN/'protocol.json'))


@torch.no_grad()
def middle_fields(model, sources, recipient, donor, device):
    """同16源状态重建两强度A/R，只返回两个预定mask。"""
    d = design(); bits = torch.as_tensor(d['bits'], device=device); bases = []; residuals = []
    for alpha in (.5, 1.):
        h = model.block2(replace_groups(sources[recipient:recipient+1], sources[donor:donor+1], bits, alpha))
        a = lower_fields(h, d)[1][0]
        bases.append(a[list(MASKS)]); residuals.append((h-a)[list(MASKS)])
    return torch.stack(bases), torch.stack(residuals)


def evaluate(device):
    """每个模型只写一个完成目录；旧结果的两个对角格逐项复现。"""
    verify_protocol(); panel = load(SHAPE/'panel/data.npz')
    for seed in teacher.SEEDS:
        folder = RUN/f'evaluation/seed_{seed}'
        if folder.exists(): verify_folder(folder); continue
        start = time.perf_counter(); model = teacher.load_teacher(seed, device)
        for p in model.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            images = torch.as_tensor(panel['images'][:, None], device=device, dtype=torch.float64)
            source = torch.cat([model.block1(model.stem(v)) for v in images.split(16)])
        selection = load(PRIOR/f'discovery/seed_{seed}/selection.npz')
        old = [load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz') for alpha in (.5, 1.)]
        values = {}
        for case, (recipient, donor) in enumerate(panel['pairs']):
            bases, residuals = middle_fields(model, source, recipient, donor, device)
            w = pair_weights(panel['labels'][recipient], panel['labels'][donor]); case_values = {}
            for m, mask in enumerate(MASKS):
                fixed = {k: v[case, m] for k, v in selection.items()}
                out = core.compute(model, bases[:, m], residuals[:, m], w, fixed)
                for j, cell in ((0, 0), (1, 3)):
                    assert np.max(abs(out['natural'][1, cell]-old[j]['natural'][case, m])) < 1e-9
                    assert np.max(abs(out['held'][1, cell]-old[j]['held'][case, m])) < 1e-9
                    assert np.max(abs(out['natural'][0, cell]-old[j]['patch_logits'][case, m, 2])) < 1e-9
                    assert np.max(abs(out['held'][0, cell, 0]-old[j]['patch_held'][case, m, 2])) < 1e-9
                    assert out['threshold'][cell, 3] == old[j]['point_meta'][case, m, 6]
                    assert np.max(abs(out['incoming'][j]-old[j]['point_change'][case, m])) < 1e-9
                for key, value in out.items(): case_values.setdefault(key, []).append(value)
            for key, value in case_values.items(): values.setdefault(key, []).append(value)
            if (case+1) % 12 == 0: event('frame_progress', seed=seed, cases=case+1)
        folder.mkdir(parents=True); np.savez_compressed(folder/'data.npz', **{k: np.asarray(v) for k, v in values.items()})
        shutil.copy2(PRIOR/f'discovery/seed_{seed}/selection.npz', folder/'selection.npz')
        teacher.dump(folder/'completion.json', dict(utc=teacher.now(), seed=seed, seconds=time.perf_counter()-start,
                     files={p.name: teacher.sha(p) for p in folder.iterdir()}))
        event('evaluation_complete', seed=seed, seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage', required=True, choices=('init', 'evaluate'))
    args = parser.parse_args()
    if args.stage == 'init': init()
    else: evaluate(teacher.configure())

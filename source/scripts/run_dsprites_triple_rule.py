"""按协议、拟合、预测、实测的顺序封存。每一阶段可在文件边界续跑。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import dsprites_teacher as teacher
import dsprites_triple_rule as rule

BASE = teacher.BASE
DEFAULT = BASE/'runs/dsprites_rule_discrimination_20261031'
SHAPE = BASE/'runs/dsprites_shape_donor_20261010'
SOURCES = ('dsprites_triple_rule.py', 'run_dsprites_triple_rule.py', 'dsprites_teacher.py',
           'residual_teacher.py', 'cross_layer_mechanism.py', 'shape_donor_mechanism.py',
           'dsprites_mechanism.py')
STRENGTHS = (1., 0.5)


def relative(path):
    return path.relative_to(BASE).as_posix()


def atomic_json(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    part = path.with_suffix('.part')
    teacher.dump(part, value)
    os.replace(part, path)


def atomic_npz(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    part = path.with_suffix('.part')
    with part.open('wb') as stream:
        np.savez_compressed(stream, **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
    os.replace(part, path)


def panel_arrays():
    with np.load(SHAPE/'panel/data.npz', allow_pickle=False) as source:
        return {key: source[key] for key in ('images','pairs','labels','factors','official_ids')}


def stage_path(run, stage, seed, strength):
    return run/stage/f'{seed}_{strength:.1f}.npz'


def seal(run, name, paths, previous):
    path = run/(name+'.json')
    if path.exists():
        verify_seal(run, name, previous)
        return
    payload = dict(utc=teacher.now(), previous=teacher.sha(run/(previous+'.json')),
                   files={p.relative_to(run).as_posix(): teacher.sha(p) for p in paths})
    atomic_json(path, payload)
    path.with_suffix('.sha256').write_text(teacher.sha(path)+'\n', encoding='ascii')
    print(name, len(paths), flush=True)


def verify_seal(run, name, previous):
    path = run/(name+'.json')
    assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    payload = teacher.read(path)
    assert payload['previous'] == teacher.sha(run/(previous+'.json'))
    for filename, digest in payload['files'].items():
        assert teacher.sha(run/filename) == digest, filename
    return payload


def manifest_entries(folder):
    path = folder/'delivery_manifest.json'
    assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    return {entry['path']:entry['sha256'] for entry in teacher.read(path)['files']}


def initialize(run):
    """固定分格、数据来源、两份数值程序和事前判据。"""
    path = run/'protocol.json'
    if path.exists():
        return verify_protocol(run)
    assert (run/'design.json').is_file()
    old_weights = manifest_entries(teacher.RUN)
    old_shape = manifest_entries(SHAPE)
    filenames = [relative(SHAPE/'panel/data.npz')]
    filenames.extend(relative(teacher.RUN/f'seed_{seed}/teacher.pt') for seed in teacher.SEEDS)
    old = {**old_weights, **old_shape}
    inputs = {n:old[n] for n in filenames}
    for name, expected in inputs.items():
        assert teacher.sha(BASE/name) == expected, name
    panel = panel_arrays()
    split = rule.split_panel(panel)
    cells = {}
    for side, cases in split.items():
        images = sorted(set(panel['pairs'][cases].flatten().tolist()))
        cells[side] = dict(cases=cases, images=images, official_ids=panel['official_ids'][images].tolist(),
                           cells=sorted({tuple(row) for row in panel['factors'][images, 2:4].tolist()}))
    payload = dict(utc=teacher.now(), design=teacher.sha(run/'design.json'),
                   source={f'scripts/{n}':teacher.sha(BASE/'scripts'/n) for n in SOURCES}, inputs=inputs,
                   split=cells, seeds=teacher.SEEDS, strengths=STRENGTHS,
                   software=dict(numpy=np.__version__,torch=rule.torch.__version__))
    atomic_json(path, payload)
    path.with_suffix('.sha256').write_text(teacher.sha(path)+'\n',encoding='ascii')
    return verify_protocol(run)


def verify_protocol(run):
    path = run/'protocol.json'
    assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    payload = teacher.read(path)
    assert teacher.sha(run/'design.json') == payload['design']
    for name, digest in {**payload['source'], **payload['inputs']}.items():
        assert teacher.sha(BASE/name) == digest, name
    return payload


def load_model(seed, device):
    model = teacher.load_teacher(seed, device)
    return model.requires_grad_(False)


def fit_all(run, panel, protocol, device):
    if (run/'fit_freeze.json').exists():
        verify_seal(run, 'fit_freeze', 'protocol')
        return
    paths = []
    for seed in teacher.SEEDS:
        path = run/'fit'/f'{seed}.json'
        if not path.exists():
            model = load_model(seed, device)
            result = rule.fit_model(model, panel, protocol['split']['development']['cases'], device, seed)
            atomic_json(path, result)
            print('fit', seed, flush=True)
        paths.append(path)
    seal(run, 'fit_freeze', paths, 'protocol')


def forecast_all(run, panel, protocol, device):
    verify_seal(run, 'fit_freeze', 'protocol')
    if (run/'prediction_freeze.json').exists():
        verify_seal(run, 'prediction_freeze', 'fit_freeze')
        return
    cases = protocol['split']['test']['cases']
    paths = []
    for seed in teacher.SEEDS:
        fit = teacher.read(run/'fit'/f'{seed}.json')
        model = load_model(seed, device)
        bank = rule.source_bank(model, panel, cases, device)
        for strength in STRENGTHS:
            path = stage_path(run, 'predictions', seed, strength)
            if not path.exists():
                rows = []
                for case in cases:
                    r, d = panel['pairs'][case]
                    row = rule.predict_pair(model,bank[r],bank[d],int(panel['labels'][r]),int(panel['labels'][d]),strength,fit)
                    rows.append(dict(case=case, **row))
                atomic_npz(path, rows)
                print('forecast', seed, strength, flush=True)
            paths.append(path)
    seal(run, 'prediction_freeze', paths, 'fit_freeze')


def measure_all(run, panel, protocol, device):
    verify_seal(run, 'fit_freeze', 'protocol')
    verify_seal(run, 'prediction_freeze', 'fit_freeze')
    if (run/'measurement_freeze.json').exists():
        verify_seal(run, 'measurement_freeze', 'prediction_freeze')
        return
    cases = protocol['split']['test']['cases']
    paths = []
    for seed in teacher.SEEDS:
        model = load_model(seed, device)
        bank = rule.source_bank(model, panel, cases, device)
        for strength in STRENGTHS:
            path = stage_path(run, 'measurements', seed, strength)
            if not path.exists():
                with np.load(stage_path(run,'predictions',seed,strength),allow_pickle=False) as pred:
                    assert pred['case'].tolist() == cases
                    patch_ids = pred['patch_ids']
                rows = []
                for i, case in enumerate(cases):
                    r,d = panel['pairs'][case]
                    rows.append(dict(case=case, **rule.measure_pair(model,bank[r],bank[d],strength,patch_ids[i])))
                atomic_npz(path, rows)
                print('measured', seed, strength, flush=True)
            paths.append(path)
    seal(run, 'measurement_freeze', paths, 'prediction_freeze')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=('predict','measure'),required=True)
    args = parser.parse_args()
    assert (args.run_dir/'design.json').is_file()
    protocol = initialize(args.run_dir) if args.stage=='predict' else verify_protocol(args.run_dir)
    panel = panel_arrays()
    device = teacher.configure()
    if args.stage=='predict':
        fit_all(args.run_dir,panel,protocol,device)
        forecast_all(args.run_dir,panel,protocol,device)
    else:
        measure_all(args.run_dir,panel,protocol,device)

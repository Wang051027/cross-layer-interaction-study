"""只读核验：来源/时序、全部证据表、汇总分母和八份旧交付。"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_joint_relu import BASE, RUN, SHAPE, load, verify_seal, verify_folder
from finalize_dsprites_joint_relu import REPORT


def verify(deep=False):
    path = RUN/'delivery_manifest.json'; records = teacher.read(path)['files']
    assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
    assert len(set(r['path'] for r in records)) == len(records)
    for r in records:
        p = (BASE/r['path']).resolve(); assert p.is_relative_to(BASE.resolve())
        assert p.stat().st_size == r['bytes'] and teacher.sha(p) == r['sha256'], r['path']
    freeze = verify_seal(); protocol = teacher.read(RUN/'protocol.json')
    assert protocol['utc'] < freeze['utc']
    analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == audit['status'] == 'PASS' and audit['endpoints'] == 1728
    assert max(audit['maximum_errors'].values()) < 1e-9 and audit['old_delivery_files'] == 1286
    tables = {}
    for name, count in analysis['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader)
            assert len(set(reader.fieldnames)) == len(reader.fieldnames)
            assert len(rows) == count and all(None not in row for row in rows), name
            tables[name] = rows
    assert len(tables['states']) == 1728 and len(tables['hold_effects']) == 13824
    assert len(tables['teacher_effects']) == 288 and len(tables['teacher_stability']) == 36
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for i, (r, s) in enumerate(panel['pairs']): weights[i, panel['labels'][s]] = 1.; weights[i, panel['labels'][r]] = -1.
    for seed in teacher.SEEDS:
        half = load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (.5, 1.):
            folder = RUN/f'evaluation/seed_{seed}/alpha_{alpha}'; verify_folder(folder)
            assert teacher.read(folder/'completion.json')['utc'] > freeze['utc']
            selection = load(folder/'selection.npz'); assert all(np.array_equal(selection[k], half[k]) for k in selection)
            raw = load(folder/'data.npz')
            assert raw['natural'].shape == raw['baseline'].shape == (72, 2, 3)
            assert raw['held'].shape == (72, 2, 8, 3) and raw['point_meta'].shape == (72, 2, 11)
            assert raw['patch_logits'].shape == raw['patch_held'].shape == (72, 2, 4, 3)
            assert selection['pool'].shape == (72, 2, 32768)
            assert all(np.isfinite(v).all() for v in raw.values())
            cc = raw['counts']; assert np.all(cc[:, :, 0] == 32768)
            assert np.array_equal(cc[:, :, 2], cc[:, :, 3:6].sum(2))
            assert np.array_equal(cc[:, :, 4:6].sum(2), raw['histogram'].sum(2))
            assert np.all(cc[:, :, 1] >= cc[:, :, 2])
            if alpha == .5:
                assert np.array_equal(cc[:, :, 5], selection['pool'].sum(2))
                valid = selection['budget'] > 0
                assert np.all(raw['point_meta'][:, :, 5][valid] == 1) and np.all(raw['point_meta'][:, :, 6][valid] == 1)
            target = np.einsum('nmk,nk->nm', raw['natural']-raw['baseline'], weights).flatten()
            net = np.einsum('nmvk,nk->nmv', raw['natural'][:, :, None]-raw['held'], weights).reshape(144, 8)
            assert np.max(abs(net-raw['direct'].reshape(144, 8)-raw['response'].reshape(144, 8))) < 1e-9
            index = np.arange(144).reshape(72, 2)
            for group, ids in (('3', index[:, 0]), ('15', index[:, 1]), ('pooled', index.flatten())):
                for j, method in enumerate(('selected1', 'random0_1', 'random1_1', 'selected32', 'random0_32', 'random1_32', 'all_half_joint', 'all_first')):
                    row = next(r for r in tables['teacher_effects'] if int(r['seed']) == seed and float(r['alpha']) == alpha and r['group'] == group and r['method'] == method)
                    den = np.mean(target[ids]**2); nn = net[ids, j]
                    checks = dict(d_rms=np.sqrt(den), net_rms=np.sqrt(np.mean(nn**2)),
                                  net_ratio=np.sqrt(np.mean(nn**2)/den), remaining_ratio=np.sqrt(np.mean((target[ids]-nn)**2)/den),
                                  projection=np.mean(target[ids]*nn)/den)
                    for key, value in checks.items(): assert abs(float(row[key])-value) < 1e-10, (seed, alpha, group, method, key)
    for row in analysis['overview_effects']:
        rows = [r for r in tables['teacher_effects'] if float(r['alpha']) == row['alpha'] and r['group'] == row['group'] and r['method'] == row['method']]
        assert len(rows) == 6
        for key in ('d_rms', 'net_ratio', 'remaining_ratio', 'projection', 'median_abs_net'):
            assert abs(row[key]-np.mean([float(r[key]) for r in rows])) < 1e-10
    # 单点的效应表和状态表、CSV与NPZ之间的主要量逐行核对。
    states = {(int(r['seed']), float(r['alpha']), int(r['case']), int(r['mask'])): r for r in tables['states']}
    for r in tables['hold_effects']:
        if r['method'] != 'selected1': continue
        s = states[(int(r['seed']), float(r['alpha']), int(r['case']), int(r['mask']))]
        assert abs(float(s['point_net'])-float(r['net'])) < 1e-12
    claims = teacher.read(RUN/'claim_checks.json')
    full_states = [r for r in tables['states'] if float(r['alpha']) == 1.]
    for name, rows in (('all', full_states),
                       ('stable_endpoint', [r for r in full_states if r['same_direction_endpoint_joint'] == 'True']),
                       ('stable_frozen_channels', [r for r in full_states if r['frozen_channels_joint'] == 'True'])):
        values = np.array([abs(float(r['point_net'])) for r in rows]); checked = claims['subgroups'][name]
        assert checked['states'] == len(rows)
        assert abs(checked['median_abs_net']-float(np.median(values))) < 1e-12
        assert abs(checked['p90_abs_net']-float(np.quantile(values, .9))) < 1e-12
        assert abs(checked['max_abs_net']-float(values.max())) < 1e-12
    categories = [0, 0, 0, 0]
    for r in full_states:
        if -int(r['half_direction'])*float(r['za']) <= 1e-8: categories[0] += 1
        elif r['frozen_channels_flip'] != 'True': categories[1] += 1
        elif r['frozen_channels_joint'] != 'True': categories[2] += 1
        else: categories[3] += 1
    assert categories == list(claims['frozen_failure_categories'].values())
    for p in (REPORT, RUN/'README.md', BASE/'CURRENT_RESEARCH.md'):
        text = p.read_text(encoding='utf-8'); width = None
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if not target.startswith(('http://', 'https://', '#')):
                assert (p.parent/target.strip('<>').split('#')[0]).resolve().exists(), (p, target)
        for line in text.splitlines():
            if line.startswith('|'):
                current = len(line.split('|'))
                if width is None: width = current
                assert current == width, (p, line)
            else: width = None
    for p in (RUN/'source/scripts').glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    old_count = 0
    if deep:
        for name, row in protocol['preservation'].items():
            p = BASE/name; assert teacher.sha(p) == row['sha256'] == p.with_suffix('.sha256').read_text().strip()
            old = teacher.read(p)['files']; assert len(old) == row['files']
            for r in old:
                p = BASE/r['path']; assert p.stat().st_size == r['bytes'] and teacher.sha(p) == r['sha256'], r['path']; old_count += 1
        assert old_count == 1286; teacher.verify_old()
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records),
                tables=len(tables), table_rows=sum(len(rows) for rows in tables.values()), old_entries=old_count,
                independent_maximum_error=max(audit['maximum_errors'].values()), manifest_sha256=teacher.sha(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--deep-inputs', action='store_true')
    args = parser.parse_args(); print(json.dumps(verify(args.deep_inputs)))

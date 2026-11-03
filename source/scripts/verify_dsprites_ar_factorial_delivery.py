"""只读核验全部四格、输出分解、正文统计、交付来源和九份旧交付。"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_ar_factorial import BASE, RUN, PRIOR, SHAPE, load, verify_protocol, verify_folder
from finalize_dsprites_ar_factorial import REPORT


def close(a, b, tolerance=1e-9):
    assert np.asarray(a).shape == np.asarray(b).shape
    assert float(np.max(abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float)))) < tolerance


def verify(deep=False):
    manifest = RUN/'delivery_manifest.json'; digest = teacher.sha(manifest)
    assert digest == manifest.with_suffix('.sha256').read_text().strip()
    records = teacher.read(manifest)['files']; assert len({r['path'] for r in records}) == len(records)
    for row in records:
        p = (BASE/row['path']).resolve(); assert p.is_relative_to(BASE.resolve())
        assert p.stat().st_size == row['bytes'] and teacher.sha(p) == row['sha256'], row['path']
    protocol = verify_protocol(); analysis = teacher.read(RUN/'analysis.json'); audit = teacher.read(RUN/'independent_audit.json')
    claims = teacher.read(RUN/'claim_checks.json')
    assert analysis['status'] == audit['status'] == claims['status'] == 'PASS'
    assert audit['endpoint_fields'] == 6912 and audit['old_delivery_files'] == 1394 and max(audit['maximum_errors'].values()) < 1e-9
    tables = {}
    for name, count in analysis['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader)
            assert len(rows) == count and len(set(reader.fieldnames)) == len(reader.fieldnames)
            assert all(None not in row for row in rows)
            tables[name] = rows
    assert len(tables['threshold_cells']) == 3456 and len(tables['holds']) == 55296
    assert len(tables['failures']) == 864 and len(tables['factorial_cases']) == 10368
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (r, donor) in enumerate(panel['pairs']): weights[n, panel['labels'][donor]] = 1.; weights[n, panel['labels'][r]] = -1.
    observed = {}; hold_observed = {}
    factorial_rows = {tuple([int(r['seed']), int(r['case']), int(r['mask']), r['context'], r['target']]): r for r in tables['factorial_cases']}
    for seed in teacher.SEEDS:
        folder = RUN/f'evaluation/seed_{seed}'; verify_folder(folder)
        assert teacher.read(folder/'completion.json')['utc'] > protocol['utc']
        sel = load(folder/'selection.npz'); prior = load(PRIOR/f'discovery/seed_{seed}/selection.npz')
        assert all(np.array_equal(sel[k], prior[k]) for k in sel)
        raw = load(folder/'data.npz')
        assert raw['natural'].shape == (72, 2, 2, 4, 3) and raw['held'].shape == (72, 2, 2, 4, 8, 3)
        assert raw['channel_net'].shape == (72, 2, 2, 4, 8, 32) and raw['threshold'].shape == (72, 2, 4, 8)
        assert all(np.isfinite(v).all() for v in raw.values())
        close(raw['group_z'][:, :, 3]-raw['group_z'][:, :, 2]-raw['group_z'][:, :, 1]+raw['group_z'][:, :, 0], np.zeros((72, 2)))
        net = np.einsum('nmqvgk,nk->nmqvg', raw['natural'][:, :, :, :, None]-raw['held'], weights)
        task = np.einsum('nmqvk,nk->nmqv', raw['natural'], weights)
        baseline = np.einsum('nmik,nk->nmi', raw['baseline'], weights)
        d = task-baseline[:, :, None, [0, 0, 1, 1]]
        note = next(r for r in claims['raw_output_and_baseline'] if r['seed'] == seed)
        delta_base = baseline[:, :, 1]-baseline[:, :, 0]
        assert abs(note['baseline_change_rms']-np.sqrt(np.mean(delta_base**2))) < 1e-10
        for q, context in enumerate(('frozen_C', 'all32')):
            raw_delta = task[:, :, q, 3]-task[:, :, q, 0]
            assert abs(note[context+'_raw_output_change_rms']-np.sqrt(np.mean(raw_delta**2))) < 1e-10
            assert abs(note[context+'_residual_output_change_rms']-np.sqrt(np.mean((raw_delta-delta_base)**2))) < 1e-10
        close(net, raw['direct']+raw['response']); close(net, raw['channel_net'].sum(-1))
        close(d, raw['parts'].sum((-1, -2))); close(d, raw['d_channel'].sum(-1))
        common_hold = np.einsum('nmqvgk,nk->nmqvg', raw['common_held'], weights)
        common_base = np.einsum('nmigk,nk->nmig', raw['common_baseline_held'], weights)
        common = task[..., None]-common_hold-(baseline[..., None]-common_base)[:, :, None, [0, 0, 1, 1]]
        close(common[:, :, :, :2], net[:, :, :, :2])
        probe = np.einsum('nmqvk,nk->nmqv', raw['probe']-raw['natural'], weights)
        targets = dict(d=d, point_net=net[..., 0], selected32_net=net[..., 3], common_point_net=common[..., 0],
                       common_selected32_net=common[..., 3], probe_delta=probe)
        for n in range(72):
            for m, mask in enumerate((3, 15)):
                observed[(seed, n, mask)] = raw['threshold'][n, m]
                hold_observed[(seed, n, mask)] = dict(net=net[n, m], common=common[n, m], direct=raw['direct'][n, m], response=raw['response'][n, m])
                for q, context in enumerate(('frozen_C', 'all32')):
                    for target, array in targets.items():
                        row = factorial_rows[(seed, n, mask, context, target)]
                        y = array[n, m, q]; values = np.array([float(row[k]) for k in ('y00', 'y01', 'y10', 'y11')])
                        close(values, y)
                        terms = np.array([y[2]-y[0], y[1]-y[0], y[3]-y[2]-y[1]+y[0]])
                        close(np.array([float(row[k]) for k in ('A', 'R', 'AR')]), terms)
                        assert abs(float(row['total'])-terms.sum()) < 1e-9
    for row in tables['threshold_cells']:
        key = tuple(int(row[k]) for k in ('seed', 'case', 'mask')); cell = ('00', '01', '10', '11').index(row['cell'])
        state = observed[key][cell]
        for j, name in enumerate(('same_base', 'group_cross', 'all_singles_safe', 'joint')):
            assert (row[name] == 'True') == bool(state[j])
        assert int(row['minimum_direction']) == int(state[4])
        close(np.array([float(row['before_margin']), float(row['after_margin'])]), state[5:7])
    methods = ('selected1', 'random0_1', 'random1_1', 'selected32', 'random0_32', 'random1_32', 'all_half_joint', 'all_first')
    for row in tables['holds']:
        key = tuple(int(row[k]) for k in ('seed', 'case', 'mask')); data = hold_observed[key]
        q = ('frozen_C', 'all32').index(row['context']); cell = ('00', '01', '10', '11').index(row['cell']); g = methods.index(row['method'])
        for name, column in (('net', 'net'), ('common', 'common_corrected_net'), ('direct', 'direct'), ('response', 'response')):
            assert abs(float(row[column])-float(data[name][q, cell, g])) < 1e-9
    # 按CSV重新计算教师级RMS/投影，区别于报告生成函数。
    for row in tables['teacher_factorial']:
        selected = [r for r in tables['factorial_cases'] if r['seed'] == row['seed'] and r['context'] == row['context'] and r['target'] == row['target'] and (row['group'] == 'pooled' or r['mask'] == row['group'])]
        total = np.array([float(r['total']) for r in selected]); den = float(np.mean(total**2))
        assert abs(float(row['change_rms'])-np.sqrt(den)) < 1e-10
        for key in ('A', 'R', 'AR'):
            values = np.array([float(r[key]) for r in selected])
            assert abs(float(row[key+'_rms'])-np.sqrt(np.mean(values**2))) < 1e-10
            if den > 1e-20: assert abs(float(row[key+'_projection'])-np.mean(values*total)/den) < 1e-10
    truth = lambda v: v == 'True'
    focus = {tuple(int(r[k]) for k in ('seed', 'case', 'mask')) for r in tables['failures'] if r['prior_category'] == 'group_not_cross'}
    assert len(focus) == 539
    for cell in ('00', '10', '01', '11'):
        for scope, expected in (('all', claims['cell_counts'][cell]), ('focus', claims['focus539_counts'][cell])):
            counts = [0, 0, 0, 0]
            for row in tables['threshold_cells']:
                key = tuple(int(row[k]) for k in ('seed', 'case', 'mask'))
                if row['cell'] != cell or (scope == 'focus' and key not in focus): continue
                which = 0 if not truth(row['same_base']) else 1 if not truth(row['group_cross']) else 2 if not truth(row['all_singles_safe']) else 3
                counts[which] += 1
            assert counts == expected
    for group in ('3', '15', 'pooled'):
        for row in [r for r in tables['cohort_patterns'] if r['group'] == group]:
            selected = []
            for r in tables['failures']:
                if group != 'pooled' and r['mask'] != group: continue
                if row['cohort'] != 'all' and not (row['cohort'] == 'failed' and r['joint11'] == '0') and r['prior_category'] != row['cohort']: continue
                selected.append(r)
            assert int(row['denominator']) == len(selected)
            assert int(row['count']) == sum(r['pattern'] == row['pattern'] for r in selected)
    for p in (REPORT, RUN/'README.md', BASE/'CURRENT_RESEARCH.md'):
        text = p.read_text(encoding='utf-8'); width = None
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('http://', 'https://', '#')): assert (p.parent/link.strip('<>').split('#')[0]).resolve().exists(), (p, link)
        for line in text.splitlines():
            if line.startswith('|'):
                n = len(line.split('|'))
                if width is None: width = n
                assert n == width, (p, line)
            else: width = None
    for p in (RUN/'source/scripts').glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    count = 0
    if deep:
        for name, record in protocol['preservation'].items():
            p = BASE/name; assert teacher.sha(p) == record['sha256'] == p.with_suffix('.sha256').read_text().strip()
            rows = teacher.read(p)['files']; assert len(rows) == record['files']
            for r in rows:
                p = BASE/r['path']; assert p.stat().st_size == r['bytes'] and teacher.sha(p) == r['sha256'], r['path']; count += 1
        assert count == 1394; teacher.verify_old()
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records),
                tables=len(tables), table_rows=sum(len(rows) for rows in tables.values()), old_entries=count,
                independent_maximum_error=max(audit['maximum_errors'].values()), manifest_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--deep-inputs', action='store_true')
    args = parser.parse_args(); print(json.dumps(verify(args.deep_inputs)))

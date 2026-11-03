"""只读复核逐行证据、解选择引用、教师统计、共同开关恒等式与全部交付SHA。"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_joint_window import BASE, RUN, AR, SHAPE, load, verify_protocol, verify_selection, verify_folder
from audit_dsprites_joint_window import own_paths
from finalize_dsprites_joint_window import CODE
from report_dsprites_joint_window import REPORT

STRATEGIES = ('moving_A', 'fixed_actual', 'aligned')
FAMILIES = ('source', 'middle', 'first_activation', 'second_activation')
VARIANTS = ('selected', 'random0', 'random1')
FIELDS = ('moving_C', 'fixed_C', 'actual_H', 'actual_no_C')


def close(a, b, tolerance=1e-9):
    """形状也检查，禁止隐式广播产生假相等。"""
    a, b = np.asarray(a), np.asarray(b); assert a.shape == b.shape
    assert not a.size or float(abs(a.astype(float)-b.astype(float)).max()) < tolerance


def verify(deep=False, before_manifest=False):
    """封存前可检查全部表；封存后连同文件字节与旧交付一起检查。"""
    records = []; digest = None
    if not before_manifest:
        manifest = RUN/'delivery_manifest.json'; digest = teacher.sha(manifest)
        assert digest == manifest.with_suffix('.sha256').read_text().strip()
        records = teacher.read(manifest)['files']; assert len({r['path'] for r in records}) == len(records)
        for r in records:
            p = (BASE/r['path']).resolve(); assert p.is_relative_to(BASE.resolve())
            assert p.stat().st_size == r['bytes'] and teacher.sha(p) == r['sha256'], r['path']
    protocol = verify_protocol(); freeze = verify_selection(); analysis = teacher.read(RUN/'analysis.json')
    claims = teacher.read(RUN/'claim_checks.json'); audit = teacher.read(RUN/'independent_audit.json')
    assert analysis['status'] == claims['status'] == audit['status'] == 'PASS'
    assert audit['endpoint_fields'] == audit['point_holds'] == 134784 and audit['old_delivery_entries'] == 1558
    assert max(audit['maximum_errors'].values()) < 1e-9
    visual = teacher.read(RUN/'visual_review.json'); assert visual['status'] == 'PASS'
    assert [r['path'] for r in visual['files']] == analysis['figures'] and len(visual['files']) == 8
    for r in visual['files']: assert teacher.sha(BASE/r['path']) == r['sha256']
    tables = {}
    for name, count in analysis['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader)
            assert len(rows) == count and len(set(reader.fieldnames)) == len(reader.fieldnames) and all(None not in r for r in rows)
            tables[name] = rows
    assert len(tables['solutions']) == 31104 and len(tables['transmission']) == 134784
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (r, d) in enumerate(panel['pairs']): weights[n, panel['labels'][d]] = 1.; weights[n, panel['labels'][r]] = -1.
    data = {}; truth = lambda v: v == 'True'
    for seed in teacher.SEEDS:
        sf = RUN/f'search/seed_{seed}'; ef = RUN/f'evaluation/seed_{seed}'; verify_folder(sf); verify_folder(ef)
        assert teacher.read(sf/'completion.json')['utc'] > protocol['utc'] and teacher.read(ef/'completion.json')['utc'] > freeze['utc']
        raw = load(ef/'data.npz'); search = load(sf/'data.npz'); units = load(sf/'units.npz'); frozen = load(sf/'frozen.npz'); old = load(AR/f'evaluation/seed_{seed}/data.npz')
        assert raw['natural'].shape == (72, 2, 39, 4, 3) and raw['margins'].shape == (72, 2, 39, 3, 3)
        assert search['coarse_margins'].shape == (72, 2, 48, 33, 3, 3)
        assert all(np.isfinite(v).all() for v in raw.values()) and all(np.isfinite(v).all() for v in search.values())
        task = np.einsum('nmeqk,nk->nmeq', raw['natural'], weights); held = np.einsum('nmeqk,nk->nmeq', raw['held'], weights)
        base = np.einsum('nmek,nk->nme', raw['baseline'], weights); net = task-held
        df = task[:, :, :, 2]-task[:, :, :1, 2]; de = net[:, :, :, 2]-net[:, :, :1, 2]
        close(net, raw['direct']+raw['response']); close(net, raw['point_channel_net'].sum(-1)); close(df, raw['channel_change'].sum(-1))
        close(df, held[:, :, :, 2]-held[:, :, :1, 2]+de)
        close(raw['margins'][:, :, 1:37], search['selected_margins'])
        assert (raw['margins'][:, :, 37:, 0] > 1e-8).all()
        focus = old['threshold'][:, :, 3, 0].astype(bool) & ~old['threshold'][:, :, 3, 1].astype(bool)
        oldF = np.einsum('nmqvk,nk->nmqv', old['natural'], weights)[:, :, 1, 3]
        oldA = np.einsum('nmik,nk->nmi', old['baseline'], weights)[:, :, 1]
        data[seed] = dict(raw=raw, search=search, units=units, frozen=frozen, task=task, held=held, net=net, base=base,
                          df=df, de=de, focus=focus, old_d=oldF-oldA)
        valid = np.zeros((72, 2, 48), bool)
        for n in range(72):
            for m in range(2): valid[n, m] = [p[3] for p in own_paths({k: v[n, m] for k, v in units.items()})]
        data[seed]['path_valid'] = valid
    assert sum(int(v['focus'].sum()) for v in data.values()) == 539
    indexed = {}; actual_rows = {}
    for row in tables['solutions']:
        seed, case, mask, endpoint = [int(row[k]) for k in ('seed', 'case', 'mask', 'endpoint')]; m = (3, 15).index(mask); i = endpoint-1
        d = data[seed]; raw = d['raw']; key = (case, m, endpoint)
        assert row['strategy'] == STRATEGIES[i//12] and row['family'] == FAMILIES[i % 12//3] and row['variant'] == VARIANTS[i % 3]
        assert truth(row['focus539']) == d['focus'][case, m]
        answer = d['search']['choices'][case, m, i]; assert int(row['path']) == int(answer[0]); assert abs(float(row['amount'])-answer[1]) < 1e-14
        assert truth(row['eligible']) == bool(answer[2]); pindex = int(answer[0])
        pp = own_paths({k: v[case, m] for k, v in d['units'].items()}); ids = [] if pindex < 0 else pp[pindex][2].tolist()
        assert json.loads(row['ids']) == ids and int(row['unit_count']) == len(ids)
        margin = raw['margins'][key].min(-1); good = margin >= 1e-6
        own, physical, aligned = bool(good[0]), bool(good[1:].all()), bool(good.all())
        passed = bool(answer[2]) and (own if i//12 == 0 else physical if i//12 == 1 else aligned)
        close([float(row[k]) for k in ('own_q', 'fixed_q', 'actual_q')], margin)
        assert truth(row['own_joint']) == own and truth(row['physical_joint']) == physical and truth(row['aligned_joint']) == aligned and truth(row['passed']) == passed
        noH = raw['shifts'][key][0] <= 1e-10; deltaF = float(d['df'][key])
        status = 'not_met' if not passed else 'decomposition_only' if noH else 'physical_with_task_effect' if physical and abs(deltaF) > 1e-8 else 'physical_task_contrast_null' if physical else 'reference_dependent_H_changed'
        assert row['status'] == status
        close([float(row[k]) for k in ('H_max_change', 'H_rms_change', 'A_max_change', 'R_max_change')], raw['shifts'][key])
        close([float(row['actual_F_change']), float(row['point_E_change']), float(row['point_E'])], [deltaF, d['de'][key], d['net'][key][2]])
        assert abs(float(row['C_output_effect'])-(d['task'][key][2]-d['task'][key][3])) < 1e-9
        assert abs(float(row['C_point_control_contrast'])-(d['net'][key][2]-d['net'][key][3])) < 1e-9
        assert truth(row['actual_j_flip']) == raw['actual_point_flip'][key]
        neighbor = d['search']['neighbor_margins'][case, m, i].min(-1)
        qn = neighbor[:, 0] if i//12 == 0 else neighbor[:, 1:].min(-1) if i//12 == 1 else neighbor.min(-1)
        assert truth(row['stable_tested_neighbors']) == (passed and bool((qn >= 1e-6).all()))
        k = (seed, case, mask, endpoint); assert k not in indexed; indexed[k] = row
        actual_rows.setdefault(seed, []).append(row)
        if noH:
            close(raw['natural'][key][2], raw['natural'][case, m, 0, 2], 1e-8)
            close(raw['margins'][key][1:], raw['margins'][case, m, 0, 1:], 1e-8)
    for row in tables['transmission']:
        seed, case, mask, endpoint = [int(row[k]) for k in ('seed', 'case', 'mask', 'endpoint')]; m = (3, 15).index(mask); q = FIELDS.index(row['context']); key = (case, m, endpoint, q); d = data[seed]
        close([float(row[k]) for k in ('F', 'held_F', 'point_net', 'j_z', 'point_direct', 'second_response')],
              [d['task'][key], d['held'][key], d['net'][key], d['raw']['point_z'][key], d['raw']['direct'][key], d['raw']['response'][key]])
    for row in tables['member_drops']:
        seed, case, mask, endpoint, channel = [int(row[k]) for k in ('seed', 'case', 'mask', 'endpoint', 'channel')]; m = (3, 15).index(mask); d = data[seed]
        C = d['frozen']['channels'][case, m]; C = C[C >= 0]; j = int(np.flatnonzero(C == channel)[0]); k = (0, 25, 28, 31, 34).index(endpoint)
        drop = float(d['raw']['drop_logits'][case, m, k, j]@weights[case]); full = d['task'][case, m, endpoint, 2]
        close([float(row['all_F']), float(row['drop_F']), float(row['drop_F_change']), float(row['drop_j_z'])], [full, drop, drop-full, d['raw']['drop_z'][case, m, k, j]])
    for row in tables['intervention_units']:
        seed, case, mask = [int(row[k]) for k in ('seed', 'case', 'mask')]; i = STRATEGIES.index(row['strategy'])*12+FAMILIES.index(row['family'])*3; source = indexed[(seed, case, mask, i+1)]
        coordinate = json.loads(source['ids'])[int(row['rank'])-1]; assert int(row['id']) == coordinate
        if FAMILIES.index(row['family']) >= 2: close([int(row[k]) for k in ('channel', 'y', 'x')], np.unravel_index(coordinate, (32, 32, 32)))
    for row in tables['teacher_summary']:
        seed = int(row['seed']); masks = (3, 15) if row['group'] == 'pooled' else (int(row['group']),)
        rr = [r for r in actual_rows[seed] if truth(r['focus539']) and int(r['mask']) in masks and r['strategy'] == row['strategy'] and r['family'] == row['family'] and r['variant'] == row['variant']]
        good = [r for r in rr if truth(r['passed'])]
        assert int(row['n']) == len(rr) and int(row['success']) == len(good) and abs(float(row['success_rate'])-len(good)/len(rr)) < 1e-12
        for name, predicate in (('decomposition_only', lambda r: r['status'] == 'decomposition_only'), ('reference_dependent', lambda r: r['status'] == 'reference_dependent_H_changed'),
                                ('new_j_flips', lambda r: truth(r['actual_j_flip'])), ('stable_neighbors', lambda r: truth(r['stable_tested_neighbors']))):
            subset = good if name in ('new_j_flips', 'stable_neighbors') else rr
            assert int(row[name]) == sum(predicate(r) for r in subset)
        if not good: assert row['success_F_rms'] == row['success_E_change_rms'] == row['E_change_ratio'] == ''
        else:
            F = np.array([float(r['actual_F_change']) for r in good]); E = np.array([float(r['point_E_change']) for r in good])
            den = np.sqrt(np.mean(data[seed]['old_d'][:, [0] if masks == (3,) else [1] if masks == (15,) else [0, 1]]**2))
            close([float(row[k]) for k in ('success_F_rms', 'success_E_change_rms', 'E_change_ratio', 'original_full_d_rms')],
                  [np.sqrt(np.mean(F**2)), np.sqrt(np.mean(E**2)), np.sqrt(np.mean(E**2))/den, den])
    for row in tables['budget_search']:
        seed = int(row['seed']); family = FAMILIES.index(row['family']); variant = VARIANTS.index(row['variant']); strategy = STRATEGIES.index(row['strategy'])
        pindex = family*12+variant*4+int(row['prefix_slot']); d = data[seed]
        margins = d['search']['coarse_margins'][:, :, pindex].min(-1)
        q = margins[..., 0] if strategy == 0 else margins[..., 1:].min(-1) if strategy == 1 else margins.min(-1)
        eligible = d['focus'] & d['path_valid'][:, :, pindex]
        assert int(row['focus_n']) == int(d['focus'].sum()) and int(row['eligible_cases']) == int(eligible.sum())
        assert int(row['grid_success']) == int((eligible & (q[:, :, 1:] >= 1e-6).any(-1)).sum())
    for row in tables['teacher_output_channels']:
        seed, ch = int(row['seed']), int(row['channel'])
        good = [r for r in actual_rows[seed] if truth(r['focus539']) and r['strategy'] == 'aligned' and r['variant'] == 'selected' and r['family'] == row['family'] and truth(r['passed'])]
        assert int(row['n']) == len(good)
        if not good: assert row['task_change_mean'] == row['task_change_rms'] == ''
        else:
            values = np.array([data[seed]['raw']['channel_change'][int(r['case']), (3, 15).index(int(r['mask'])), int(r['endpoint']), ch] for r in good])
            close([float(row['task_change_mean']), float(row['task_change_rms'])], [values.mean(), np.sqrt(np.mean(values**2))])
    all_rows = [r for r in tables['solutions'] if truth(r['focus539'])]
    unique = set(); total = 0
    for family in FAMILIES:
        c = claims['families'][family]
        moving = [r for r in all_rows if r['strategy'] == 'moving_A' and r['variant'] == 'selected' and r['family'] == family and truth(r['passed'])]
        physical = [r for r in all_rows if r['strategy'] == 'fixed_actual' and r['variant'] == 'selected' and r['family'] == family and truth(r['passed'])]
        aligned = [r for r in all_rows if r['strategy'] == 'aligned' and r['variant'] == 'selected' and r['family'] == family and truth(r['passed'])]
        assert c['moving_success'] == len(moving) and c['physical_success'] == len(physical) and c['aligned_success'] == len(aligned)
        assert c['moving_decomposition_only'] == sum(r['status'] == 'decomposition_only' for r in moving)
        assert c['moving_reference_dependent'] == sum(r['status'] == 'reference_dependent_H_changed' for r in moving)
        assert c['aligned_task_effectful'] == sum(abs(float(r['actual_F_change'])) > 1e-8 for r in aligned)
        assert c['aligned_new_j_flips'] == sum(truth(r['actual_j_flip']) for r in aligned)
        unique.update((int(r['seed']), int(r['case']), int(r['mask'])) for r in aligned); total += len(aligned)
    assert claims['aligned_unique_states'] == len(unique) and claims['aligned_case_method_count'] == total
    for row in tables['positive_controls']: assert truth(row['original_C_exactly_necessary']) and float(row['moving_q']) > 1e-8
    for path in (REPORT, RUN/'README.md', BASE/'CURRENT_RESEARCH.md'):
        text = path.read_text(encoding='utf-8'); width = None
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('https://', 'http://', '#')): assert (path.parent/link.strip('<>').split('#')[0]).resolve().exists(), (path, link)
        for line in text.splitlines():
            if line.startswith('|'):
                n = len(line.split('|'))
                if width is None: width = n
                assert n == width, (path, line)
            else: width = None
    for name in CODE: ast.parse((BASE/'scripts'/name).read_text(encoding='utf-8'))
    old_count = 0
    if deep:
        for name, record in protocol['preservation'].items():
            path = BASE/name; assert teacher.sha(path) == record['sha256'] == path.with_suffix('.sha256').read_text().strip()
            rows = teacher.read(path)['files']; assert len(rows) == record['files']
            for r in rows: assert teacher.sha(BASE/r['path']) == r['sha256'] and (BASE/r['path']).stat().st_size == r['bytes']; old_count += 1
        assert old_count == 1558; teacher.verify_old()
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), tables=len(tables),
                rows=sum(len(r) for r in tables.values()), old_entries=old_count, independent_maximum_error=max(audit['maximum_errors'].values()), manifest_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--deep-inputs', action='store_true'); parser.add_argument('--before-manifest', action='store_true')
    args = parser.parse_args(); print(json.dumps(verify(args.deep_inputs, args.before_manifest)))

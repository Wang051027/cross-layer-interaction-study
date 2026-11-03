"""只读检查清单、冻结时序、CSV逐行证据、教师统计和全部历史文件。"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_upstream_rescue import BASE, RUN, PRIOR, SHAPE, load, verify_protocol, verify_selection, verify_folder
from finalize_dsprites_upstream_rescue import CODE
from report_dsprites_upstream_rescue import REPORT

NAMES = ('natural', 'src_selected4', 'src_random0', 'src_random1', 'src_selected1', 'src_all',
         'mid_selected4', 'mid_random0', 'mid_random1', 'mid_selected1', 'mid_all',
         'g1_selected16', 'g1_random0', 'g1_random1', 'g1_selected1', 'g1_all',
         'g2_selected16', 'g2_random0', 'g2_random1', 'g2_selected1', 'g2_all', 'g12_all')


def close(a, b, tolerance=1e-9):
    """逐项核对，不接受形状广播导致的假相等。"""
    a, b = np.asarray(a), np.asarray(b); assert a.shape == b.shape, (a.shape, b.shape)
    assert not a.size or float(abs(a.astype(float)-b.astype(float)).max()) < tolerance


def verify(deep=False, before_manifest=False):
    """封存前也能检查正文与表，避免先封存错误后再改交付。"""
    records = []; digest = None
    if not before_manifest:
        path = RUN/'delivery_manifest.json'; digest = teacher.sha(path)
        assert digest == path.with_suffix('.sha256').read_text().strip()
        records = teacher.read(path)['files']; assert len({r['path'] for r in records}) == len(records)
        for r in records:
            p = (BASE/r['path']).resolve(); assert p.is_relative_to(BASE.resolve())
            assert p.stat().st_size == r['bytes'] and teacher.sha(p) == r['sha256'], r['path']
    protocol = verify_protocol(); freeze = verify_selection(); analysis = teacher.read(RUN/'analysis.json')
    audit = teacher.read(RUN/'independent_audit.json'); claims = teacher.read(RUN/'claim_checks.json')
    assert analysis['status'] == audit['status'] == claims['status'] == 'PASS'
    assert audit['endpoint_fields'] == audit['point_holds'] == 76032 and audit['old_delivery_files'] == 1467
    assert max(audit['maximum_errors'].values()) < 1e-9
    visual = teacher.read(RUN/'visual_review.json'); assert visual['status'] == 'PASS'
    assert [r['path'] for r in visual['files']] == analysis['figures'] and len(visual['files']) == 8
    for row in visual['files']: assert teacher.sha(BASE/row['path']) == row['sha256']
    tables = {}
    for name, count in analysis['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader)
            assert len(rows) == count and len(set(reader.fieldnames)) == len(reader.fieldnames)
            assert all(None not in r for r in rows)
            tables[name] = rows
    assert len(tables['baseline_parts']) == 864 and len(tables['thresholds']) == 38016 and len(tables['interventions']) == 76032
    panel = load(SHAPE/'panel/data.npz'); w = np.zeros((72, 3))
    for n, (r, d) in enumerate(panel['pairs']): w[n, panel['labels'][d]] = 1.; w[n, panel['labels'][r]] = -1.
    raw_all = {}; part_all = {}; focus_all = {}; values_all = {}; fixed_all = {}; selected_all = {}; within_all = {}
    for seed in teacher.SEEDS:
        discovery = RUN/f'discovery/seed_{seed}'; folder = RUN/f'evaluation/seed_{seed}'
        verify_folder(discovery); verify_folder(folder)
        assert teacher.read(discovery/'completion.json')['utc'] > protocol['utc']
        assert teacher.read(folder/'completion.json')['utc'] > freeze['utc']
        raw = load(folder/'data.npz'); sel = load(discovery/'selection.npz'); fixed = load(discovery/'frozen_prior.npz')
        prior = load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        assert raw['natural'].shape == (72, 2, 22, 2, 2, 3) and raw['threshold'].shape == (72, 2, 22, 2, 5)
        assert raw['parts'].shape == (72, 2, 22, 2, 2, 3, 32)
        assert all(np.isfinite(v).all() for v in raw.values()) and all(np.isfinite(v).all() for v in sel.values())
        task = np.einsum('nmersk,nk->nmers', raw['natural'], w)
        base = np.einsum('nmek,nk->nme', raw['baseline'], w)
        net = np.einsum('nmersk,nk->nmers', raw['natural']-raw['held'], w)
        d = task-base[..., None, None]; delta = task-task[:, :, :1, :1]
        dchange = d-d[:, :, :1, :1]; basechange = base-base[:, :, :1]
        close(d, raw['parts'].sum((-1, -2))); close(d, raw['d_channel'].sum(-1))
        close(net, raw['direct']+raw['response']); close(net, raw['channel_net'].sum(-1))
        close(delta, dchange+basechange[..., None, None]); close(raw['natural'][:, :, 0, 0], prior['natural'][:, :, :, 3])
        close(raw['incoming'][:, :, :, 0], np.repeat(raw['incoming'][:, :, :1, 0], 22, axis=2))
        for method in (5, 10):
            close(raw['natural'][:, :, method, 0], prior['natural'][:, :, :, 1]); close(raw['natural'][:, :, method, 1], prior['natural'][:, :, :, 0])
        focus = prior['threshold'][:, :, 3, 0].astype(bool) & ~prior['threshold'][:, :, 3, 1].astype(bool)
        parts = np.empty((72, 2, 3))
        for m, k in enumerate((2, 4)):
            parts[:, m, 0] = sel['source_linear'][:, m, :k].sum((1, 2))
            parts[:, m, 1:] = sel['gate_parts'][:, m, :, :k].sum((2, 3))
        close(parts.sum(-1), sel['base_z'][:, :, 1]-sel['base_z'][:, :, 0])
        close(raw['base_z'][:, :, 21]-sel['base_z'][:, :, 0], parts[:, :, 0])
        raw_all[seed] = raw; part_all[seed] = parts; focus_all[seed] = focus; fixed_all[seed] = fixed; selected_all[seed] = sel
        values_all[seed] = dict(F=task, baseline_F=base, d=d, point_net=net, raw_F_change=delta, d_change=dchange, baseline_F_change=basechange)
        within = np.zeros((72, 2, 22, 2), bool)
        for n in range(72):
            for m in range(2):
                C = fixed['channels'][n, m]; C = C[C >= 0]; direction = int(fixed['half_direction'][n, m])
                solos = direction*(raw['base_z'][n, m, :, None, None]+raw['incoming'][n, m][:, :, C])
                within[n, m] = raw['threshold'][n, m, :, :, 1] & (solos.max(-1) < -1e-8)
        within_all[seed] = within
    assert sum(int(v.sum()) for v in focus_all.values()) == claims['scope']['focus539'] == 539
    reg_index = ('A_only', 'coherent'); ctx_index = ('frozen_C', 'all32')
    for row in tables['thresholds']:
        seed, n, mask = [int(row[k]) for k in ('seed', 'case', 'mask')]; m = (3, 15).index(mask); method = NAMES.index(row['method']); reg = reg_index.index(row['regime'])
        raw = raw_all[seed]; key = (n, m, method, reg); flags = raw['threshold'][key]
        for j, name in enumerate(('same_base', 'group_cross', 'all_solos_safe', 'joint', 'all_cross')): assert (row[name] == 'True') == flags[j]
        assert (row['focus539'] == 'True') == focus_all[seed][n, m]
        close([float(row[k]) for k in ('b', 'group_z', 'all_z')], [raw['base_z'][n, m, method], raw['group_z'][key], raw['all_z'][key]])
        direction = int(fixed_all[seed]['half_direction'][n, m]); C = fixed_all[seed]['channels'][n, m]; C = C[C >= 0]
        assert (row['within_C_joint'] == 'True') == within_all[seed][key]
        safe = max(direction*(raw['base_z'][n, m, method]+raw['incoming'][key][C])) < -1e-8
        assert (row['C_solos_safe'] == 'True') == safe
        expected_a = direction*(raw['base_z'][n, m, method]-raw['base_z'][n, m, 0])
        expected_r = direction*(raw['incoming'][key][C]-raw['incoming'][n, m, 0, 0, C]).sum()
        close([float(row['A_repair']), float(row['R_group_repair'])], [expected_a, expected_r])
        assert abs(direction*(raw['group_z'][key]-raw['group_z'][n, m, 0, 0])-expected_a-expected_r) < 1e-9
    for row in tables['interventions']:
        seed, n, mask = [int(row[k]) for k in ('seed', 'case', 'mask')]; m = (3, 15).index(mask); method = NAMES.index(row['method']); reg = reg_index.index(row['regime']); ctx = ctx_index.index(row['context'])
        raw = raw_all[seed]; values = values_all[seed]; key = (n, m, method, reg, ctx)
        for name in ('F', 'd', 'point_net', 'raw_F_change', 'd_change'): assert abs(float(row[name])-values[name][key]) < 1e-9
        for name in ('baseline_F', 'baseline_F_change'): assert abs(float(row[name])-values[name][n, m, method]) < 1e-9
        for name in ('direct', 'response', 'point_remainder'): assert abs(float(row[name])-raw[name][key]) < 1e-9
        assert int(row['first_flips']) == raw['flips'][key][0] and int(row['second_flips']) == raw['flips'][key][1]
    for row in tables['baseline_parts']:
        seed, n, mask = [int(row[k]) for k in ('seed', 'case', 'mask')]; m = (3, 15).index(mask)
        close([float(row[k]) for k in ('linear', 'first_relu', 'second_relu')], part_all[seed][n, m])
    for row in tables['selected_units']:
        seed, n, mask = [int(row[k]) for k in ('seed', 'case', 'mask')]; m = (3, 15).index(mask); sel = selected_all[seed]
        fam = ('source', 'middle', 'gate1', 'gate2').index(row['family']); variant = ('selected', 'random0', 'random1').index(row['variant']); rank = int(row['rank'])-1
        ids = sel['source_lists'][n, m] if fam == 0 else sel['middle_lists'][n, m] if fam == 1 else sel['gate_lists'][n, m, fam-2]
        coordinate = int(ids[variant, rank]); assert coordinate == int(row['id'])
        if fam < 2: score = sel['source_scores' if fam == 0 else 'middle_scores'][n, m, coordinate]
        else:
            pos = int(np.flatnonzero(sel[f'gate{fam-1}_ids'][n, m] == coordinate)[0]); score = sel[f'gate{fam-1}_scores'][n, m, pos]
            close([int(row[k]) for k in ('channel', 'y', 'x')], np.unravel_index(coordinate, (32, 32, 32)))
        assert abs(float(row['baseline_single_repair'])-score) < 1e-9
    # 按模型/组重新计算统计，分母始终取该模型/组全部原案例的d。
    for row in tables['teacher_thresholds']:
        seed = int(row['seed']); mask = np.ones((72, 2), bool)
        if row['group'] != 'pooled': mask[:, 1-(3, 15).index(int(row['group']))] = False
        if row['cohort'] == 'focus539': mask &= focus_all[seed]
        method, reg = NAMES.index(row['method']), reg_index.index(row['regime']); flags = raw_all[seed]['threshold'][mask][:, method, reg]
        assert int(row['n']) == len(flags)
        for name, j in (('same_base', 0), ('group_cross', 1), ('joint', 3), ('all_cross', 4)): assert int(row[name]) == int(flags[:, j].sum())
        close([float(row['group_cross_rate']), float(row['joint_rate'])], [flags[:, 1].mean(), flags[:, 3].mean()])
        assert int(row['within_C_joint']) == int(within_all[seed][mask][:, method, reg].sum())
    for row in tables['teacher_parts']:
        seed = int(row['seed']); subset = np.ones((72, 2), bool)
        if row['group'] != 'pooled': subset[:, 1-(3, 15).index(int(row['group']))] = False
        if row['cohort'] == 'focus539': subset &= focus_all[seed]
        x = part_all[seed][subset]; total = x.sum(1); den = np.sqrt(np.mean(total**2))
        assert int(row['n']) == len(x); assert abs(float(row['drift_rms'])-den) < 1e-9
        for j, name in enumerate(('linear', 'first', 'second')):
            close([float(row[name+'_rms']), float(row[name+'_ratio']), float(row[name+'_projection'])],
                  [np.sqrt(np.mean(x[:, j]**2)), np.sqrt(np.mean(x[:, j]**2))/den, np.mean(x[:, j]*total)/np.mean(total**2)])
    for row in tables['teacher_outputs']:
        seed = int(row['seed']); group = np.ones((72, 2), bool)
        if row['group'] != 'pooled': group[:, 1-(3, 15).index(int(row['group']))] = False
        subset = group & focus_all[seed] if row['cohort'] == 'focus539' else group
        method, reg, ctx = NAMES.index(row['method']), reg_index.index(row['regime']), ctx_index.index(row['context']); v = values_all[seed]
        delta = v['raw_F_change'][subset][:, method, reg, ctx]; net = v['point_net'][subset][:, method, reg, ctx]
        den = np.sqrt(np.mean(v['d'][:, :, 0, 0, 1][group]**2))
        close([float(row[k]) for k in ('original_full_d_rms', 'raw_F_change_rms', 'raw_F_change_mean', 'raw_F_change_ratio', 'point_net_rms', 'point_net_ratio')],
              [den, np.sqrt(np.mean(delta**2)), delta.mean(), np.sqrt(np.mean(delta**2))/den, np.sqrt(np.mean(net**2)), np.sqrt(np.mean(net**2))/den])
    for method, name in enumerate(NAMES):
        flags = np.concatenate([raw_all[s]['threshold'][focus_all[s]][:, method] for s in teacher.SEEDS]); a, c = flags[:, 0, 1], flags[:, 1, 1]
        claim = claims['methods'][name]
        for key, expected in dict(A_cross=a.sum(), coherent_cross=c.sum(), A_joint=flags[:, 0, 3].sum(), coherent_joint=flags[:, 1, 3].sum(),
                                  both_cross=(a & c).sum(), A_only_cross=(a & ~c).sum(), coherent_only_cross=(~a & c).sum(), neither_cross=(~a & ~c).sum()).items(): assert claim[key] == int(expected)
        expected_categories = [int((~flags[:, 1, 0]).sum()), int((flags[:, 1, 0] & ~flags[:, 1, 1]).sum()),
                               int((flags[:, 1, 1] & ~flags[:, 1, 3]).sum()), int(flags[:, 1, 3].sum())]
        assert claim['coherent_categories'] == expected_categories and sum(expected_categories) == 539
        assert claim['A_within_C_joint'] == sum(int(within_all[s][focus_all[s]][:, method, 0].sum()) for s in teacher.SEEDS)
        assert claim['coherent_within_C_joint'] == sum(int(within_all[s][focus_all[s]][:, method, 1].sum()) for s in teacher.SEEDS)
    for row in tables['rescue_output_contrasts']:
        seed = int(row['seed']); group = np.ones((72, 2), bool)
        if row['group'] != 'pooled': group[:, 1-(3, 15).index(int(row['group']))] = False
        method, reg, ctx = NAMES.index(row['method']), reg_index.index(row['regime']), ctx_index.index(row['context'])
        raw = raw_all[seed]; subset = group & focus_all[seed] & raw['threshold'][:, :, method, reg, 1]
        assert int(row['n']) == int(subset.sum())
        if not subset.any():
            assert row['point_contrast_rms'] == row['raw_F_change_rms'] == ''; continue
        close(np.sign(raw['base_z'][:, :, method][subset]), np.sign(raw['base_z'][:, :, 0][subset]))
        values = values_all[seed]; delta = values['raw_F_change'][subset][:, method, reg, ctx]
        contrast = values['point_net'][subset][:, method, reg, ctx]-values['point_net'][subset][:, 0, 0, ctx]
        den = np.sqrt(np.mean(values['d'][:, :, 0, 0, 1][group]**2))
        close([float(row[k]) for k in ('raw_F_change_rms', 'point_contrast_rms', 'point_contrast_ratio', 'remaining_controlled_effect_rms')],
              [np.sqrt(np.mean(delta**2)), np.sqrt(np.mean(contrast**2)), np.sqrt(np.mean(contrast**2))/den, np.sqrt(np.mean((delta-contrast)**2))])
    for path in (REPORT, RUN/'README.md', BASE/'CURRENT_RESEARCH.md'):
        text = path.read_text(encoding='utf-8'); width = None
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if not link.startswith(('http://', 'https://', '#')): assert (path.parent/link.strip('<>').split('#')[0]).resolve().exists(), (path, link)
        for line in text.splitlines():
            if line.startswith('|'):
                n = len(line.split('|'))
                if width is None: width = n
                assert n == width, (path, line)
            else: width = None
    for n in CODE: ast.parse((BASE/'scripts'/n).read_text(encoding='utf-8'))
    old_count = 0
    if deep:
        for name, record in protocol['preservation'].items():
            path = BASE/name; assert teacher.sha(path) == record['sha256'] == path.with_suffix('.sha256').read_text().strip()
            rows = teacher.read(path)['files']; assert len(rows) == record['files']
            for r in rows:
                path = BASE/r['path']; assert path.stat().st_size == r['bytes'] and teacher.sha(path) == r['sha256'], r['path']; old_count += 1
        assert old_count == 1467; teacher.verify_old()
    return dict(status='PASS', files=len(records), bytes=sum(r['bytes'] for r in records), tables=len(tables),
                table_rows=sum(len(r) for r in tables.values()), old_entries=old_count,
                independent_maximum_error=max(audit['maximum_errors'].values()), manifest_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--deep-inputs', action='store_true'); parser.add_argument('--before-manifest', action='store_true')
    args = parser.parse_args(); print(json.dumps(verify(args.deep_inputs, args.before_manifest)))

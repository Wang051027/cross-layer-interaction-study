"""分别计算 A/R 四格在每个模型内的 RMS 和投影，再对六个模型等权汇总。"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_ar_factorial import BASE, RUN, SHAPE, load, verify_protocol, verify_folder

CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
HOLDS = ('selected1', 'random0_1', 'random1_1', 'selected32', 'random0_32', 'random1_32', 'all_half_joint', 'all_first')
PATTERNS = ('either_alone_fails', 'each_alone_fails_combined_rescues', 'A_alone_fails', 'A_fails_R_rescues',
            'R_alone_fails', 'R_fails_A_rescues', 'only_combined_fails', 'all_four_joint')


def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))


def ratio(a, b): return None if abs(b) < 1e-15 else float(a/b)


def differences(values):
    """轴末四格；三项依次为A单独、R单独、A×R非加和项。"""
    v = np.asarray(values)
    return np.stack((v[..., 2]-v[..., 0], v[..., 1]-v[..., 0], v[..., 3]-v[..., 2]-v[..., 1]+v[..., 0]), -1)


def table(name, rows):
    path = RUN/f'tables/{name}.csv'; assert not path.exists(); path.parent.mkdir(exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def measures(raw, weights):
    """统一读出轴及共同参考校正，减少正文口径漂移。"""
    task = np.einsum('nmqvk,nk->nmqv', raw['natural'], weights)
    baseline = np.einsum('nmik,nk->nmi', raw['baseline'], weights)
    base4 = baseline[:, :, [0, 0, 1, 1]]
    d = task-base4[:, :, None]
    net = np.einsum('nmqvgk,nk->nmqvg', raw['natural'][:, :, :, :, None]-raw['held'], weights)
    common_hold = np.einsum('nmqvgk,nk->nmqvg', raw['common_held'], weights)
    common_base = np.einsum('nmigk,nk->nmig', raw['common_baseline_held'], weights)
    base_bias = (baseline[:, :, :, None]-common_base)[:, :, [0, 0, 1, 1]]
    common_net = task[:, :, :, :, None]-common_hold-base_bias[:, :, None]
    probe = np.einsum('nmqvk,nk->nmqv', raw['probe']-raw['natural'], weights)
    assert np.max(abs(net-raw['direct']-raw['response'])) < 1e-9
    assert np.max(abs(net[:, :, :, :2]-common_net[:, :, :, :2])) < 1e-9
    targets = dict(d=d, point_net=net[..., 0], selected32_net=net[..., 3],
                   common_point_net=common_net[..., 0], common_selected32_net=common_net[..., 3], probe_delta=probe)
    return d, net, common_net, probe, task, baseline, targets


def report():
    assert not (RUN/'analysis.json').exists(); verify_protocol()
    audit = teacher.read(RUN/'independent_audit.json'); assert audit['status'] == 'PASS'
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (r, donor) in enumerate(panel['pairs']):
        weights[n, panel['labels'][donor]] = 1.; weights[n, panel['labels'][r]] = -1.
    names = ('threshold_cells', 'failures', 'channel_drift', 'transmission', 'holds', 'factorial_cases',
             'teacher_joint', 'teacher_factorial', 'teacher_holds', 'teacher_probes', 'teacher_output_channels', 'example_channels')
    tables = {name: [] for name in names}; index = np.arange(144).reshape(72, 2)
    for seed in teacher.SEEDS:
        folder = RUN/f'evaluation/seed_{seed}'; verify_folder(folder); raw = load(folder/'data.npz'); sel = load(folder/'selection.npz')
        d, net, common, probe, task, baseline, targets = measures(raw, weights)
        for case in range(72):
            for m, mask in enumerate((3, 15)):
                meta = dict(seed=seed, case=case, mask=mask)
                direction = int(sel['half_direction'][case, m]); threshold = raw['threshold'][case, m]
                C = sel['channels'][case, m]; C = C[C >= 0]; J = threshold[:, 3].astype(int)
                pattern = int(4*J[2]+2*J[1]+J[3])
                old = 'baseline_changed' if not threshold[3, 0] else 'group_not_cross' if not threshold[3, 1] else 'single_suffices' if not threshold[3, 2] else 'joint_retained'
                az = raw['base_z'][case, m]; D = raw['incoming'][case, m]
                shift_a = direction*(az[1]-az[0]); shift_r = direction*(D[1, C]-D[0, C]).sum()
                margin = direction*raw['group_z'][case, m]
                assert abs((margin[3]-margin[0])-shift_a-shift_r) < 1e-9
                tables['failures'].append(dict(**meta, prior_category=old, pattern=pattern, pattern_name=PATTERNS[pattern],
                    joint00=J[0], joint01=J[1], joint10=J[2], joint11=J[3], half_k=int(sel['half_k'][case, m]),
                    shift_A=float(shift_a), shift_R=float(shift_r), margin00=float(margin[0]), margin11=float(margin[3]),
                    group_contribution0=float(direction*D[0, C].sum()), group_contribution1=float(direction*D[1, C].sum()),
                    reversed_group_channels=int((direction*D[1, C] < -1e-8).sum()),
                    incoming_cosine=ratio(float(D[0]@D[1]), float(np.linalg.norm(D[0])*np.linalg.norm(D[1])))))
                for cell, (i, j) in enumerate(CELLS):
                    t = threshold[cell]
                    tables['threshold_cells'].append(dict(**meta, cell=f'{i}{j}', base_z=float(az[i]),
                        group_z=float(raw['group_z'][case, m, cell]), all_z=float(raw['all_z'][case, m, cell]),
                        single_max_oriented=float(max(direction*(az[i]+D[j]))),
                        same_base=bool(t[0]), group_cross=bool(t[1]), all_singles_safe=bool(t[2]), joint=bool(t[3]),
                        minimum_direction=int(t[4]), before_margin=float(t[5]), after_margin=float(t[6]), all_endpoint_flip=bool(t[7])))
                for c in range(32):
                    tables['channel_drift'].append(dict(**meta, channel=c, frozen_member=bool(c in C),
                        incoming0=float(D[0, c]), incoming1=float(D[1, c]),
                        oriented_change=float(direction*(D[1, c]-D[0, c]))))
                for q, context in enumerate(('frozen_C', 'all32')):
                    for cell, (i, j) in enumerate(CELLS):
                        parts = raw['parts'][case, m, q, cell].sum(1)
                        tables['transmission'].append(dict(**meta, context=context, cell=f'{i}{j}',
                            baseline=float(baseline[case, m, i]), natural=float(task[case, m, q, cell]), d=float(d[case, m, q, cell]),
                            linear=float(parts[0]), first=float(parts[1]), second=float(parts[2]),
                            point_remainder=float(raw['point_remainder'][case, m, q, cell]),
                            secant_gain=float(raw['point_gain'][case, m, q, cell]) if raw['gain_valid'][case, m, q, cell] else None,
                            probe_delta=float(probe[case, m, q, cell]),
                            first_flips=int(raw['flips'][case, m, q, cell, 0]), second_flips=int(raw['flips'][case, m, q, cell, 1])))
                        for g, method in enumerate(HOLDS):
                            ch = raw['channel_net'][case, m, q, cell, g]
                            units = float(abs(ch).sum())
                            tables['holds'].append(dict(**meta, context=context, cell=f'{i}{j}', method=method,
                                d=float(d[case, m, q, cell]), net=float(net[case, m, q, cell, g]),
                                direct=float(raw['direct'][case, m, q, cell, g]), response=float(raw['response'][case, m, q, cell, g]),
                                common_corrected_net=float(common[case, m, q, cell, g]),
                                second_closed=int(raw['second_flips'][case, m, q, cell, g, 0]),
                                second_opened=int(raw['second_flips'][case, m, q, cell, g, 1]),
                                output_channel_cancellation=ratio(units-abs(float(net[case, m, q, cell, g])), units)))
                    for name, values in targets.items():
                        vals = values[case, m, q]; terms = differences(vals)
                        tables['factorial_cases'].append(dict(**meta, context=context, target=name,
                            y00=float(vals[0]), y01=float(vals[1]), y10=float(vals[2]), y11=float(vals[3]),
                            A=float(terms[0]), R=float(terms[1]), AR=float(terms[2]), total=float(vals[3]-vals[0])))
                if seed == teacher.SEEDS[0] and case == 0:
                    for q, context in enumerate(('frozen_C', 'all32')):
                        for cell, (i, j) in enumerate(CELLS):
                            for c in range(32):
                                tables['example_channels'].append(dict(**meta, context=context, cell=f'{i}{j}', channel=c,
                                    d=float(raw['d_channel'][case, m, q, cell, c]),
                                    point_net=float(raw['channel_net'][case, m, q, cell, 0, c]),
                                    selected32_net=float(raw['channel_net'][case, m, q, cell, 3, c])))
        # 先在模型内计算，保留三种源状态汇总口径。
        for group, ids in (('3', index[:, 0]), ('15', index[:, 1]), ('pooled', index.flatten())):
            ids = ids.reshape(-1); ref_d11 = d[:, :, 1, 3].flatten()[ids]; ref_rms = rms(ref_d11)
            JJ = raw['threshold'].reshape(144, 4, 8)[ids, :, 3]
            row = dict(seed=seed, group=group, states=len(ids))
            for cell, (i, j) in enumerate(CELLS): row[f'joint{i}{j}'] = int(JJ[:, cell].sum())
            tables['teacher_joint'].append(row)
            for q, context in enumerate(('frozen_C', 'all32')):
                for name, values in targets.items():
                    v = values[:, :, q].reshape(144, 4)[ids]; change = v[:, 3]-v[:, 0]; terms = differences(v)
                    den = float(np.mean(change**2)); unit = float(abs(terms).sum())
                    row = dict(seed=seed, group=group, context=context, target=name, change_rms=rms(change),
                               full_all32_d11_rms=ref_rms, cancellation=ratio(unit-float(abs(change).sum()), unit))
                    for c, key in enumerate(('A', 'R', 'AR')):
                        row[key+'_rms'] = rms(terms[:, c]); row[key+'_ratio_change'] = ratio(rms(terms[:, c]), rms(change))
                        row[key+'_ratio_d11'] = ratio(rms(terms[:, c]), ref_rms)
                        row[key+'_projection'] = ratio(float(np.mean(change*terms[:, c])), den)
                    tables['teacher_factorial'].append(row)
                for cell, (i, j) in enumerate(CELLS):
                    own_d = d[:, :, q, cell].flatten()[ids]
                    rem = raw['point_remainder'][:, :, q, cell].flatten()[ids]
                    rem0 = raw['point_remainder'][:, :, q, 0].flatten()[ids]
                    probe_val = probe[:, :, q, cell].flatten()[ids]; probe0 = probe[:, :, q, 0].flatten()[ids]
                    tables['teacher_probes'].append(dict(seed=seed, group=group, context=context, cell=f'{i}{j}',
                        remainder_rms=rms(rem), remainder_ratio00=ratio(rms(rem), rms(rem0)),
                        probe_rms=rms(probe_val), probe_ratio00=ratio(rms(probe_val), rms(probe0)), probe_mean=float(probe_val.mean()),
                        point_net_rms=rms(net[:, :, q, cell, 0].flatten()[ids]),
                        remainder_nonzero=int((rem > 1e-8).sum())))
                    for g, method in enumerate(HOLDS):
                        n = net[:, :, q, cell, g].flatten()[ids]; cc = common[:, :, q, cell, g].flatten()[ids]
                        direct = raw['direct'][:, :, q, cell, g].flatten()[ids]; response = raw['response'][:, :, q, cell, g].flatten()[ids]
                        units = float(abs(direct).sum()+abs(response).sum())
                        tables['teacher_holds'].append(dict(seed=seed, group=group, context=context, cell=f'{i}{j}', method=method,
                            effect_rms=rms(n), common_effect_rms=rms(cc), effect_ratio_d11=ratio(rms(n), ref_rms),
                            common_ratio_d11=ratio(rms(cc), ref_rms), direct_ratio_d11=ratio(rms(direct), ref_rms),
                            response_ratio_d11=ratio(rms(response), ref_rms),
                            projection_current_d=ratio(float(np.mean(n*own_d)), float(np.mean(own_d**2))),
                            cancellation=ratio(units-float(abs(n).sum()), units),
                            no_second_flip_but_effect=int(((abs(n) > 1e-8) & (raw['second_flips'][:, :, q, cell, g].sum(-1).flatten()[ids] == 0)).sum())))
        for q, context in enumerate(('frozen_C', 'all32')):
            for cell, (i, j) in enumerate(CELLS):
                for c in range(32):
                    dv = raw['d_channel'][:, :, q, cell, c]
                    p = raw['channel_net'][:, :, q, cell, 0, c]; p32 = raw['channel_net'][:, :, q, cell, 3, c]
                    tables['teacher_output_channels'].append(dict(seed=seed, context=context, cell=f'{i}{j}', channel=c,
                        d_rms=rms(dv), d_mean=float(dv.mean()), point_rms=rms(p), point_mean=float(p.mean()),
                        selected32_rms=rms(p32), selected32_mean=float(p32.mean())))
    # 互斥四格失效模式；总体与上一轮539个特定失败分开给出。
    patterns = []
    for group in ('3', '15', 'pooled'):
        for cohort in ('all', 'failed', 'baseline_changed', 'group_not_cross', 'single_suffices', 'joint_retained'):
            rows = [r for r in tables['failures'] if (group == 'pooled' or str(r['mask']) == group) and
                    (cohort == 'all' or (cohort == 'failed' and not r['joint11']) or r['prior_category'] == cohort)]
            for p in range(8): patterns.append(dict(group=group, cohort=cohort, pattern=p, pattern_name=PATTERNS[p],
                                                     count=sum(r['pattern'] == p for r in rows), denominator=len(rows)))
    tables['cohort_patterns'] = patterns
    for name, rows in tables.items(): table(name, rows)
    notes(tables)
    figures = make_figures(tables)
    teacher.dump(RUN/'analysis.json', dict(status='PASS', utc=teacher.now(), table_rows={k: len(v) for k, v in tables.items()},
                 teacher_joint=tables['teacher_joint'], teacher_factorial=tables['teacher_factorial'],
                 teacher_probes=tables['teacher_probes'], cohort_patterns=patterns, figures=figures,
                 independent_maximum_error=max(audit['maximum_errors'].values())))
    print('PASS', {k: len(v) for k, v in tables.items()}, flush=True)


def make_figures(tables):
    """独立静态科学图；固定尺度、图例和分母，方便与CSV逐项核对。"""
    os.environ['MPLCONFIGDIR'] = str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    colors = ['#156082', '#E97132', '#196B24', '#A02B93', '#6E6E6E', '#BC9A00']; paths = []
    def save(fig, name):
        for ext in ('png', 'svg'):
            p = BASE/f'reports/figures/dsprites_ar_factorial_20261017_{name}.{ext}'
            fig.savefig(p, dpi=180, bbox_inches='tight'); paths.append(str(p.relative_to(BASE)).replace('\\', '/'))
        plt.close(fig)
    order = ('00', '10', '01', '11'); labels = ['A0 R0', 'A1 R0\nA only', 'A0 R1\nR only', 'A1 R1\nBoth']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, group in zip(axes, ('3', '15')):
        for seed, color in zip(teacher.SEEDS, colors):
            r = next(r for r in tables['teacher_joint'] if r['seed'] == seed and r['group'] == group)
            ax.plot(range(4), [100*r['joint'+cell]/r['states'] for cell in order], 'o-', color=color, label=str(seed)[-3:])
        ax.set_xticks(range(4), labels); ax.set_ylim(0, 104); ax.set_ylabel('Frozen channel set remains joint (%)')
        ax.set_title('Source mask '+group); ax.grid(axis='y', alpha=.18)
    axes[1].legend(title='Teacher seed suffix', ncol=3, fontsize=8)
    fig.suptitle('Separate baseline and residual changes | same frozen coordinates and channels')
    save(fig, 'joint')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout='constrained')
    for ax, cohort, title in zip(axes, ('failed', 'group_not_cross'), ('All 787 lost joint mechanisms', '539 unchanged-baseline threshold failures')):
        rr = [r for r in tables['cohort_patterns'] if r['group'] == 'pooled' and r['cohort'] == cohort]
        values = [next(r['count'] for r in rr if r['pattern'] == p) for p in (2, 4, 0, 6)]
        bars = ax.bar(range(4), values, color=['#156082', '#E97132', '#6E6E6E', '#A02B93'])
        ax.bar_label(bars, padding=3); ax.set_xticks(range(4), ['A alone', 'R alone', 'Either alone', 'Only together'], rotation=15)
        ax.set_ylabel('Existing case-states'); ax.set_title(title)
    axes[0].set_ylim(0, 820)
    fig.suptitle('Which intervention is sufficient to lose the joint condition?\nThreshold outcomes for the four combinations of A and R')
    save(fig, 'failure')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey='row', layout='constrained')
    for col, context in enumerate(('frozen_C', 'all32')):
        for row, suffix, ylabel in ((0, '_ratio_change', 'Term RMS / RMS(d11-d00) (%)'), (1, '_projection', 'Signed projection onto d11-d00 (%)')):
            ax = axes[row, col]
            for seed, color in zip(teacher.SEEDS, colors):
                r = next(r for r in tables['teacher_factorial'] if r['seed'] == seed and r['group'] == 'pooled' and r['context'] == context and r['target'] == 'd')
                ax.plot(range(3), [100*r[k+suffix] for k in ('A', 'R', 'AR')], 'o-', color=color, label=str(seed)[-3:])
            ax.axhline(0, color='#333', linewidth=.6); ax.set_xticks(range(3), ['A only', 'R only', 'A × R'])
            ax.set_ylabel(ylabel); ax.set_title(context); ax.grid(axis='y', alpha=.18)
            if row == 0: ax.set_ylim(bottom=0)
    compared = [r for r in tables['teacher_factorial'] if r['group'] == 'pooled' and r['target'] == 'd']
    upper = max(100*r[k+'_ratio_change'] for r in compared for k in ('A', 'R', 'AR'))
    axes[0, 0].set_ylim(0, upper*1.10)
    projections = [100*r[k+'_projection'] for r in compared for k in ('A', 'R', 'AR')]
    padding = (max(projections)-min(projections))*.08
    axes[1, 0].set_ylim(min(0, min(projections))-padding, max(0, max(projections))+padding)
    axes[0, 1].legend(title='Teacher seed suffix', ncol=3, fontsize=8)
    fig.suptitle('Output change from A, R and their interaction\nSigned projections onto the total change sum to 100%')
    save(fig, 'output')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, key, ylabel in zip(axes, ('remainder_ratio00', 'probe_ratio00'), ('Point nonlinear remainder RMS / value at 00', '+1 activation probe RMS / value at 00')):
        for seed, color in zip(teacher.SEEDS, colors):
            rows = [r for r in tables['teacher_probes'] if r['seed'] == seed and r['group'] == 'pooled' and r['context'] == 'all32']
            ax.plot(range(4), [next(r[key] for r in rows if r['cell'] == cell) for cell in order], 'o-', color=color, label=str(seed)[-3:])
        ax.set_xticks(range(4), labels); ax.set_ylim(bottom=0); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=.18)
    axes[1].legend(title='Teacher seed suffix', ncol=3, fontsize=8)
    fig.suptitle('Switch amplitude and downstream response are separate controls\nProbe response to a +1 activation increment')
    save(fig, 'probe')
    return paths


def notes(tables):
    """正文中的特定失败分组、条件效应和例子均保存直接可复核的计算。"""
    failed = [r for r in tables['failures'] if r['prior_category'] == 'group_not_cross']
    keys = {(int(r['seed']), int(r['case']), int(r['mask'])) for r in failed}
    key = lambda r: (int(r['seed']), int(r['case']), int(r['mask']))
    truth = lambda value: value in (True, 'True', 1, '1')
    cell_counts = {}; focus_counts = {}
    for cell in ('00', '10', '01', '11'):
        for scope, destination in (('all', cell_counts), ('focus539', focus_counts)):
            rows = [r for r in tables['threshold_cells'] if r['cell'] == cell and (scope == 'all' or key(r) in keys)]
            counts = [0, 0, 0, 0]
            for r in rows:
                which = 0 if not truth(r['same_base']) else 1 if not truth(r['group_cross']) else 2 if not truth(r['all_singles_safe']) else 3
                counts[which] += 1
            destination[cell] = counts
    stats = {name: dict(median=float(np.median([float(r[name]) for r in failed])),
                       p10=float(np.quantile([float(r[name]) for r in failed], .1)),
                       p90=float(np.quantile([float(r[name]) for r in failed], .9)))
             for name in ('shift_A', 'shift_R', 'margin00', 'margin11', 'incoming_cosine')}
    conditional = []
    for seed in teacher.SEEDS:
        rows = [r for r in tables['factorial_cases'] if int(r['seed']) == seed and r['context'] == 'all32' and r['target'] == 'd']
        a = np.array([float(r['A']) for r in rows]); r = np.array([float(r['R']) for r in rows]); interaction = np.array([float(r['AR']) for r in rows])
        conditional.append(dict(seed=seed, A_at_R0=rms(a), A_at_R1=rms(a+interaction),
                                R_at_A0=rms(r), R_at_A1=rms(r+interaction),
                                R_AR_cosine=ratio(float(r@interaction), float(np.linalg.norm(r)*np.linalg.norm(interaction)))))
    focus_full = [r for r in tables['threshold_cells'] if r['cell'] == '11' and key(r) in keys]
    focus_trans = [r for r in tables['transmission'] if r['cell'] == '11' and r['context'] == 'frozen_C' and key(r) in keys]
    result = dict(status='PASS', categories='baseline_changed,group_not_cross,single_suffices,joint_retained',
        cell_counts=cell_counts, focus539_counts=focus_counts, focus539_quantiles=stats,
        focus539_A_adverse=sum(float(r['shift_A']) < 0 for r in failed),
        focus539_R_adverse=sum(float(r['shift_R']) < 0 for r in failed),
        focus539_has_reversed_group_channel=sum(int(r['reversed_group_channels']) > 0 for r in failed),
        focus539_outside_channels_restore_point=sum(truth(r['all_endpoint_flip']) for r in focus_full),
        focus539_C_output_nonzero=sum(abs(float(r['d'])) > 1e-8 for r in focus_trans),
        focus539_C_output_abs_median=float(np.median([abs(float(r['d'])) for r in focus_trans])),
        conditional_output=conditional)
    examples = []
    # 预定案例0/mask15和按固定编号顺序遇到的第一个539类失败；均为说明，非新检验。
    first = min(failed, key=key)
    for seed, case, mask in ((teacher.SEEDS[0], 0, 15), key(first)):
        raw = load(RUN/f'evaluation/seed_{seed}/data.npz'); sel = load(RUN/f'evaluation/seed_{seed}/selection.npz')
        m = 0 if mask == 3 else 1; flat = int(sel['lists'][case, m, 0, 0])
        channels = sel['channels'][case, m]; channels = channels[channels >= 0]
        examples.append(dict(seed=seed, case=case, mask=mask, coordinate=list(map(int, np.unravel_index(flat, (32, 32, 32)))),
            channels=channels.tolist(), base_z=raw['base_z'][case, m].tolist(), group_z=raw['group_z'][case, m].tolist(),
            incoming_C=raw['incoming'][case, m][:, channels].tolist(),
            point_output_channels11=raw['channel_net'][case, m, :, 3, 0].tolist()))
    result['examples'] = examples
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (recipient, donor) in enumerate(panel['pairs']):
        weights[n, panel['labels'][donor]] = 1.; weights[n, panel['labels'][recipient]] = -1.
    raw_output = []
    for seed in teacher.SEEDS:
        raw = load(RUN/f'evaluation/seed_{seed}/data.npz')
        task = np.einsum('nmqvk,nk->nmqv', raw['natural'], weights)
        base = np.einsum('nmik,nk->nmi', raw['baseline'], weights)
        delta_base = base[:, :, 1]-base[:, :, 0]
        row = dict(seed=seed, baseline_change_rms=rms(delta_base))
        for q, context in enumerate(('frozen_C', 'all32')):
            delta_output = task[:, :, q, 3]-task[:, :, q, 0]
            delta_residual = delta_output-delta_base
            row[context+'_raw_output_change_rms'] = rms(delta_output)
            row[context+'_residual_output_change_rms'] = rms(delta_residual)
        raw_output.append(row)
    result['raw_output_and_baseline'] = raw_output
    teacher.dump(RUN/'claim_checks.json', result)


def finish_presentation(figures=True):
    """封存前从现有CSV补齐正文计算并统一比较尺度，不再运行网络。"""
    assert not (RUN/'delivery_manifest.json').exists(); analysis = teacher.read(RUN/'analysis.json'); tables = {}
    for name in analysis['table_rows']:
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f: tables[name] = list(csv.DictReader(f))
    notes(tables)
    if not figures: return
    for rows in tables.values():
        for row in rows:
            for field in ('seed', 'states', 'count', 'pattern', 'joint00', 'joint01', 'joint10', 'joint11'):
                if field in row: row[field] = int(row[field])
            for field in ('A_ratio_change', 'R_ratio_change', 'AR_ratio_change', 'A_projection', 'R_projection', 'AR_projection', 'remainder_ratio00', 'probe_ratio00'):
                if field in row: row[field] = float(row[field])
    assert make_figures(tables) == analysis['figures']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--presentation-only', action='store_true')
    parser.add_argument('--notes-only', action='store_true')
    args = parser.parse_args()
    if args.notes_only: finish_presentation(figures=False)
    elif args.presentation_only: finish_presentation()
    else: report()

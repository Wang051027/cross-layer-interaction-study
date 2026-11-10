"""把冻结实验整理为逐案例证据表、按教师汇总和可独立保存的科学图。

比例分母为本轮两个源端点d=F(H)−F(A)的RMS，不是旧24差分的RMS。
先在每个教师内汇总，再对六教师等权平均，不给共享图片配对做p值。
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_joint_relu import BASE, RUN, SHAPE, load, verify_seal, verify_folder
from joint_relu_mechanism import MASKS, HOLD_NAMES


def rms(value): return float(np.sqrt(np.mean(np.asarray(value)**2)))


def divide(a, b): return None if abs(b) < 1e-15 else float(a/b)


def write_table(name, rows):
    path = RUN/f'tables/{name}.csv'; assert not path.exists(); path.parent.mkdir(exist_ok=True)
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_effect(target, net, direct, response, class_changed):
    den = rms(target); energy = float(np.mean(target**2))
    units = float(np.abs(direct).sum()+np.abs(response).sum())
    eligible = abs(target) >= .1
    return dict(d_rms=den, net_rms=rms(net), net_ratio=divide(rms(net), den),
                remaining_ratio=divide(rms(target-net), den), direct_ratio=divide(rms(direct), den),
                response_ratio=divide(rms(response), den), projection=divide(float(np.mean(target*net)), energy),
                cancellation=divide(units-float(np.abs(net).sum()), units),
                median_abs_net=float(np.median(abs(net))), p90_abs_net=float(np.quantile(abs(net), .9)),
                max_abs_net=float(np.max(abs(net))), count=len(target), count_abs_ge_001=int((abs(net) >= .01).sum()),
                count_abs_ge_01=int((abs(net) >= .1).sum()), relative_eligible=int(eligible.sum()),
                count_relative_ge_10pct=int((eligible & (abs(net) >= .1*abs(target))).sum()),
                argmax_changed=int(class_changed.sum()))


def report():
    assert not (RUN/'analysis.json').exists(); verify_seal()
    audit = teacher.read(RUN/'independent_audit.json'); assert audit['status'] == 'PASS'
    panel = load(SHAPE/'panel/data.npz')
    weights = np.zeros((72, 3))
    for i, (r, s) in enumerate(panel['pairs']): weights[i, panel['labels'][s]] = 1.; weights[i, panel['labels'][r]] = -1.
    tables = {k: [] for k in ('states', 'hold_effects', 'teacher_effects', 'teacher_stability',
                              'minimum_histogram', 'frozen_gate_lists', 'incoming_channel_terms', 'patches')}
    for seed in teacher.SEEDS:
        selection = load(RUN/f'discovery/seed_{seed}/selection.npz')
        half = load(RUN/f'evaluation/seed_{seed}/alpha_0.5/data.npz')
        full = load(RUN/f'evaluation/seed_{seed}/alpha_1.0/data.npz')
        for case in range(72):
            for m, mask in enumerate(MASKS):
                for method, ids in enumerate(selection['lists'][case, m]):
                    for rank, flat in enumerate(ids):
                        if flat < 0: continue
                        c, y, x = np.unravel_index(flat, (32, 32, 32))
                        tables['frozen_gate_lists'].append(dict(seed=seed, case=case, mask=mask,
                            method=('selected', 'random0', 'random1')[method], rank=rank+1,
                            channel=int(c), y=int(y), x=int(x), half_k=int(half['list_k'][case, m, method, rank]),
                            full_k=int(full['list_k'][case, m, method, rank])))
        for alpha, raw in ((.5, half), (1., full)):
            verify_folder(RUN/f'evaluation/seed_{seed}/alpha_{alpha}')
            target = np.einsum('nmk,nk->nm', raw['natural']-raw['baseline'], weights)
            net = np.einsum('nmvk,nk->nmv', raw['natural'][:, :, None]-raw['held'], weights)
            patch = np.einsum('nmpk,nk->nmp', raw['patch_logits']-raw['baseline'][:, :, None], weights)
            patch_net = np.einsum('nmpk,nk->nmp', raw['patch_logits']-raw['patch_held'], weights)
            changed = raw['held'].argmax(3) != raw['natural'].argmax(2)[:, :, None]
            assert np.max(abs(net-raw['direct']-raw['response'])) < 1e-9
            for case, (recipient, donor) in enumerate(panel['pairs']):
                for m, mask in enumerate(MASKS):
                    point = int(selection['lists'][case, m, 0, 0])
                    coordinate = (-1, -1, -1) if point < 0 else tuple(map(int, np.unravel_index(point, (32, 32, 32))))
                    counts = raw['counts'][case, m]; meta = raw['point_meta'][case, m]
                    tables['states'].append(dict(seed=seed, alpha=alpha, case=case, mask=mask,
                        recipient=int(recipient), donor=int(donor), recipient_class=int(panel['labels'][recipient]),
                        donor_class=int(panel['labels'][donor]), candidate_budget=int(selection['budget'][case, m]),
                        gate_channel=coordinate[0], gate_y=coordinate[1], gate_x=coordinate[2],
                        d=float(target[case, m]), point_net=float(net[case, m, 0]),
                        point_direct=float(raw['direct'][case, m, 0]), point_response=float(raw['response'][case, m, 0]),
                        all_flips=int(counts[1]), robust_flips=int(counts[2]), ambiguous_minimum=int(counts[3]),
                        single_sufficient=int(counts[4]), joint_minimum=int(counts[5]),
                        joint_open=int(counts[6]), joint_close=int(counts[7]),
                        za=float(meta[0]), zh=float(meta[1]), half_direction=int(meta[2]),
                        half_k=int(selection['half_k'][case, m]), current_k=int(meta[3]),
                        frozen_channels_flip=bool(meta[4]), same_direction_endpoint_joint=bool(meta[5]),
                        frozen_channels_joint=bool(meta[6]), current_best_before_margin=float(meta[7]),
                        current_best_after_margin=float(meta[8]),
                        half_channels=','.join(map(str, selection['channels'][case, m][selection['channels'][case, m] >= 0])),
                        patch_jump=float(patch[case, m, 2]-patch[case, m, 1]),
                        point_net_jump=float(patch_net[case, m, 2]-patch_net[case, m, 1]),
                        point_argmax_changed=bool(changed[case, m, 0])))
                    for j, method in enumerate(HOLD_NAMES):
                        tables['hold_effects'].append(dict(seed=seed, alpha=alpha, case=case, mask=mask,
                            method=method, d=float(target[case, m]), net=float(net[case, m, j]),
                            remaining=float(target[case, m]-net[case, m, j]),
                            direct=float(raw['direct'][case, m, j]), response=float(raw['response'][case, m, j]),
                            argmax_changed=bool(changed[case, m, j])))
                    for c in range(32):
                        tables['incoming_channel_terms'].append(dict(seed=seed, alpha=alpha, case=case, mask=mask,
                            middle_channel=c, preactivation_change=float(raw['point_change'][case, m, c]),
                            actual_solo_preactivation=float(raw['solo_z'][case, m, c]),
                            half_frozen_member=bool(c in selection['channels'][case, m])))
                    for j, label in enumerate(('A', 'frozen_kminus1', 'frozen_k', 'H')):
                        tables['patches'].append(dict(seed=seed, alpha=alpha, case=case, mask=mask,
                            patch=label, preactivation=float(raw['patch_z'][case, m, j]),
                            task_delta=float(patch[case, m, j]), point_net=float(patch_net[case, m, j]),
                            logit_square=float(raw['patch_logits'][case, m, j, 0]),
                            logit_ellipse=float(raw['patch_logits'][case, m, j, 1]),
                            logit_heart=float(raw['patch_logits'][case, m, j, 2]),
                            held_square=float(raw['patch_held'][case, m, j, 0]),
                            held_ellipse=float(raw['patch_held'][case, m, j, 1]),
                            held_heart=float(raw['patch_held'][case, m, j, 2])))
            flat_index = np.arange(144).reshape(72, 2)
            for group, index in (('3', flat_index[:, 0]), ('15', flat_index[:, 1]), ('pooled', flat_index.flatten())):
                index = index.reshape(-1); dd = target.flatten()[index]
                for j, method in enumerate(HOLD_NAMES):
                    tables['teacher_effects'].append(dict(seed=seed, alpha=alpha, group=group, method=method,
                        **summarize_effect(dd, net.reshape(144, 8)[index, j], raw['direct'].reshape(144, 8)[index, j],
                                           raw['response'].reshape(144, 8)[index, j], changed.reshape(144, 8)[index, j])))
                cc = raw['counts'].reshape(144, 8)[index]; ps = raw['pool_stats'].reshape(144, 3)[index]
                mm = raw['point_meta'].reshape(144, 11)[index]; budget = selection['budget'].flatten()[index]
                valid = budget > 0; nn = int(valid.sum())
                jump = (patch[:, :, 2]-patch[:, :, 1]).flatten()[index]
                point_jump = (patch_net[:, :, 2]-patch_net[:, :, 1]).flatten()[index]
                tables['teacher_stability'].append(dict(seed=seed, alpha=alpha, group=group, states=len(index),
                    available=nn, robust_flips=int(cc[:, 2].sum()), minimum_ambiguous=int(cc[:, 3].sum()),
                    joint_count=int(cc[:, 5].sum()), single_sufficient=int(cc[:, 4].sum()),
                    joint_fraction=divide(float(cc[:, 5].sum()), float(cc[:, 4:6].sum())),
                    frozen_pool_count=int(ps[:, 0].sum()), frozen_pool_joint_count=int(ps[:, 2].sum()),
                    frozen_pool_joint_rate=divide(float(ps[:, 2].sum()), float(ps[:, 0].sum())),
                    selected_endpoint_joint=int(mm[:, 5].sum()), selected_endpoint_joint_rate=divide(float(mm[:, 5].sum()), nn),
                    frozen_channels_flip=int(mm[:, 4].sum()), frozen_channels_joint=int(mm[:, 6].sum()),
                    frozen_channels_joint_rate=divide(float(mm[:, 6].sum()), nn),
                    same_minimum=int(((mm[:, 3] == selection['half_k'].flatten()[index]) & valid).sum()),
                    single_now_sufficient=int(((mm[:, 3] == 1) & valid).sum()),
                    no_robust_endpoint_flip=int(((mm[:, 3] == 0) & valid).sum()),
                    patch_jump_rms=rms(jump), point_net_jump_rms=rms(point_jump),
                    point_jump_to_total_ratio=divide(rms(point_jump), rms(jump))))
            for m, mask in enumerate(MASKS):
                hist = raw['histogram'][:, m].sum(0)
                for k, count in enumerate(hist, 1):
                    tables['minimum_histogram'].append(dict(seed=seed, alpha=alpha, mask=mask, k=k,
                        count=int(count), total=int(hist.sum()), fraction=divide(float(count), float(hist.sum()))))
    # 六个教师等权；计数另用sum给出，绝不将均值计数冒充实际事件总数。
    effects = tables['teacher_effects']; stability = tables['teacher_stability']
    overview = []; overview_stability = []
    for alpha in (.5, 1.):
        for group in ('3', '15', 'pooled'):
            rows = [r for r in stability if r['alpha'] == alpha and r['group'] == group]
            out = dict(alpha=alpha, group=group)
            for key in rows[0]:
                if key in ('seed', 'alpha', 'group'): continue
                vals = [r[key] for r in rows]
                out[key+'_mean'] = None if any(v is None for v in vals) else float(np.mean(vals))
                if isinstance(vals[0], int): out[key+'_sum'] = int(sum(vals))
            overview_stability.append(out)
            for method in HOLD_NAMES:
                rows = [r for r in effects if r['alpha'] == alpha and r['group'] == group and r['method'] == method]
                out = dict(alpha=alpha, group=group, method=method)
                for key in rows[0]:
                    if key in ('seed', 'alpha', 'group', 'method'): continue
                    vals = [r[key] for r in rows]
                    out[key] = None if any(v is None for v in vals) else float(np.mean(vals))
                    if isinstance(vals[0], int): out[key+'_sum'] = int(sum(vals))
                overview.append(out)
    tables['overview_effects'] = overview; tables['overview_stability'] = overview_stability
    for name, rows in tables.items(): write_table(name, rows)
    claims(tables)
    figures = make_figures(tables)
    result = dict(status='PASS', utc=teacher.now(), denominator='endpointdRMS withinteacher/group,then6teacher equalmean',
                  table_rows={name: len(rows) for name, rows in tables.items()},
                  overview_effects=overview, overview_stability=overview_stability,
                  figures=figures, independent_maximum_error=max(audit['maximum_errors'].values()))
    teacher.dump(RUN/'analysis.json', result)
    print('PASS', result['table_rows'], flush=True)


def make_figures(tables):
    os.environ['MPLCONFIGDIR'] = str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 180})
    folder = BASE/'reports/figures'; folder.mkdir(exist_ok=True); paths = []
    colors = ['#156082', '#E97132', '#196B24', '#A02B93', '#6E6E6E', '#BC9A00']
    def save(fig, suffix):
        for ext in ('png', 'svg'):
            path = folder/f'dsprites_joint_relu_20261017_{suffix}.{ext}'
            fig.savefig(path, bbox_inches='tight'); paths.append(str(path.relative_to(BASE)).replace('\\', '/'))
        plt.close(fig)
    # 各模型的冻结选点与冻结通道组复现率，统一0—100%轴。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    for ax, group in zip(axes, ('3', '15')):
        for seed, color in zip(teacher.SEEDS, colors):
            r = next(r for r in tables['teacher_stability'] if r['seed'] == seed and r['alpha'] == 1. and r['group'] == group)
            ax.plot([0, 1, 2], 100*np.array([r['frozen_pool_joint_rate'], r['selected_endpoint_joint_rate'], r['frozen_channels_joint_rate']]),
                    'o-', color=color, label=str(seed)[-3:], alpha=.85)
        ax.set_xticks([0, 1, 2], ['All half-joint\ncoordinates', 'Selected point\nendpoint joint', 'Frozen channels\njoint trigger'])
        ax.set_ylim(0, 104); ax.set_ylabel('Retained joint mechanism (%)'); ax.set_title(f'Source mask {group} | 72 paired cases per teacher')
        ax.grid(axis='y', alpha=.18)
    axes[1].legend(title='Teacher seed suffix', ncol=3, fontsize=8, loc='lower center')
    fig.suptitle('Full-strength replication of half-strength joint thresholds\nChannel rankings fixed at half strength')
    save(fig, 'stability')
    # 开关净输出幅度与有符号投影，防止把RMS幅度说成解释比例。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    methods = ('selected1', 'random0_1', 'random1_1', 'selected32', 'random0_32', 'random1_32', 'all_half_joint', 'all_first')
    labels = ['Sel.1', 'Rnd.1a', 'Rnd.1b', 'Sel.≤32', 'Rnd.≤32a', 'Rnd.≤32b', 'All half\njoint', 'All first']
    for ax, key, ylabel in zip(axes, ('net_ratio', 'projection'), ('RMS(net gate effect) / RMS(d) (%)', 'Signed projection onto d (%)')):
        for seed, color in zip(teacher.SEEDS, colors):
            y = [100*next(r[key] for r in tables['teacher_effects'] if r['seed'] == seed and r['alpha'] == 1. and r['group'] == 'pooled' and r['method'] == method) for method in methods]
            ax.plot(range(8), y, 'o-', color=color, alpha=.75, markersize=4, label=str(seed)[-3:])
        ax.axhline(0, color='#444', linewidth=.6); ax.set_xticks(range(8), labels, rotation=25, ha='right')
        ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=.18)
        if key == 'net_ratio': ax.set_ylim(bottom=0)
    axes[0].legend(title='Teacher seed suffix', ncol=3, fontsize=8, loc='upper left')
    fig.suptitle('Actual output effects of frozen gate lists | pooled fixed source masks\nAmplitude and direction are different measurements')
    save(fig, 'output')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for ax, mask in zip(axes, MASKS):
        for alpha, offset, color in ((.5, -.2, '#156082'), (1., .2, '#E97132')):
            rows = [r for r in tables['minimum_histogram'] if r['alpha'] == alpha and r['mask'] == mask]
            hist = np.array([sum(r['count'] for r in rows if r['k'] == k) for k in range(1, 33)])
            bins = np.r_[hist[:9], hist[9:].sum()]; ax.bar(np.arange(10)+offset, 100*bins/hist.sum(), width=.4, color=color, label=f'alpha={alpha}')
        ax.set_xticks(range(10), [*map(str, range(1, 10)), '10+']); ax.set_ylim(0, 50)
        ax.set_xlabel('Exact minimum whole-channel count'); ax.set_ylabel('Share of robust decidable flips (%)')
        ax.set_title(f'Source mask {mask} | descriptive census'); ax.legend(); ax.grid(axis='y', alpha=.18)
    fig.suptitle('First-ReLU threshold census across all six teachers\nCounts are pooled over ReLU locations within each model')
    save(fig, 'minimum')
    # 具体单点输出量级：每个教师一条经验分布，不掩盖零效应。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for ax, method in zip(axes, ('selected1', 'selected32')):
        for seed, color in zip(teacher.SEEDS, colors):
            values = np.sort([abs(r['net']) for r in tables['hold_effects'] if r['seed'] == seed and r['alpha'] == 1. and r['method'] == method])
            ax.step(values, np.arange(1, len(values)+1)/len(values)*100, where='post', color=color, label=str(seed)[-3:])
        ax.set_xscale('symlog', linthresh=.001); ax.set_xlim(0, 4); ax.set_ylim(0, 101)
        ax.set_xlabel('Absolute net effect on donor-minus-recipient logit'); ax.set_ylabel('Case-state cumulative proportion (%)')
        ax.set_title('Selected point' if method == 'selected1' else 'Selected ≤32 points'); ax.grid(alpha=.18)
    axes[1].legend(title='Teacher seed suffix', fontsize=8)
    fig.suptitle('A joint threshold can have a small output effect\n144 source states per model on the 36-image panel')
    save(fig, 'effect_distribution')
    return paths


def claims(tables):
    """正文中的全864状态分位数和事后示例也保留可复核的计算记录。"""
    states = [r for r in tables['states'] if float(r['alpha']) == 1.]
    result = dict(scope='fullstrength864dependent case-states;pooled quantiles differfrommean of6teacher quantiles', subgroups={})
    for name, rows in (('all', states), ('stable_endpoint', [r for r in states if r['same_direction_endpoint_joint'] in (True, 'True')]),
                       ('stable_frozen_channels', [r for r in states if r['frozen_channels_joint'] in (True, 'True')])):
        values = np.array([abs(float(r['point_net'])) for r in rows])
        result['subgroups'][name] = dict(states=len(rows), median_abs_net=float(np.median(values)),
                                           p90_abs_net=float(np.quantile(values, .9)), max_abs_net=float(values.max()))
    result['budget_below32'] = sum(int(r['candidate_budget']) < 32 for r in states)
    result['minimum_budget'] = min(int(r['candidate_budget']) for r in states)
    failures = dict(baseline_changed_or_near_zero=0, same_baseline_but_frozen_set_not_cross=0,
                    frozen_set_cross_but_single_can_cross=0, frozen_set_joint=0)
    for r in states:
        if -int(r['half_direction'])*float(r['za']) <= 1e-8:
            failures['baseline_changed_or_near_zero'] += 1
        elif r['frozen_channels_flip'] not in (True, 'True'):
            failures['same_baseline_but_frozen_set_not_cross'] += 1
        elif r['frozen_channels_joint'] not in (True, 'True'):
            failures['frozen_set_cross_but_single_can_cross'] += 1
        else: failures['frozen_set_joint'] += 1
    assert sum(failures.values()) == 864
    result['frozen_failure_categories'] = failures
    hist = [r for r in tables['minimum_histogram'] if float(r['alpha']) == 1.]
    result['full_census_histogram'] = {str(k): sum(int(r['count']) for r in hist if int(r['k']) == k) for k in range(1, 33)}
    # 按已约定案例保留一例，另有影响最大的稳定点作为明确标记的事后描述。
    result['predefined_example'] = next(r for r in states if int(r['seed']) == 202609291 and int(r['case']) == 0 and int(r['mask']) == 15)
    stable = [r for r in states if r['frozen_channels_joint'] in (True, 'True')]
    result['posthoc_largest_stable_example'] = max(stable, key=lambda r: abs(float(r['point_net'])))
    teacher.dump(RUN/'claim_checks.json', result)


def presentation():
    """允许在最终交付封存前，仅从现有CSV修正图例/刻度和保存正文计算。"""
    assert not (RUN/'delivery_manifest.json').exists()
    analysis = teacher.read(RUN/'analysis.json'); tables = {}
    for name in analysis['table_rows']:
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
        for row in rows:
            for key in ('alpha', 'net', 'net_ratio', 'projection', 'fraction', 'frozen_pool_joint_rate',
                        'selected_endpoint_joint_rate', 'frozen_channels_joint_rate'):
                if key in row: row[key] = float(row[key])
            for key in ('seed', 'mask', 'k', 'count', 'total'):
                if key in row: row[key] = int(float(row[key]))
        tables[name] = rows
    claims(tables); assert make_figures(tables) == analysis['figures']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--presentation-only', action='store_true')
    args = parser.parse_args()
    if args.presentation_only: presentation()
    else: report()

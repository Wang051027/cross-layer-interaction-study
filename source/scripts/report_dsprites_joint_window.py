"""原组精确必要性恢复的证据表、共同参照筛查、任务输出和中文科学报告。"""
from __future__ import annotations
import csv
import json
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_joint_window import BASE, RUN, AR, SHAPE, load, verify_protocol, verify_selection, verify_folder
from joint_window_mechanism import STRATEGIES, FAMILIES, VARIANTS, paths, objectives, GUARD

REPORT = BASE/'reports/dSprites_保留联合必要性的上游恢复_2026-10-18.md'
FIELDS = ('moving_C', 'fixed_C', 'actual_H', 'actual_no_C')
DROP_ENDPOINTS = (0, 25, 28, 31, 34)


def rms(values):
    """空的成功子集不作零效应；统计前在模型内区分成功与失败。"""
    return float(np.sqrt(np.mean(np.asarray(values)**2))) if len(values) else None


def write_table(name, rows):
    """逐行来源可追溯；按模型汇总案例。"""
    path = RUN/f'tables/{name}.csv'; assert not path.exists(); path.parent.mkdir(exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def measures(raw, weights):
    """原始任务读出、共同j控制及A/R相对修正明确分开。"""
    task = np.einsum('nmeqk,nk->nmeq', raw['natural'], weights)
    held = np.einsum('nmeqk,nk->nmeq', raw['held'], weights)
    baseline = np.einsum('nmek,nk->nme', raw['baseline'], weights)
    net = task-held; deltaF = task[:, :, :, 2]-task[:, :, :1, 2]
    deltaE = net[:, :, :, 2]-net[:, :, :1, 2]
    assert np.max(abs(net-raw['direct']-raw['response'])) < 1e-9
    assert np.max(abs(deltaF-raw['channel_change'].sum(-1))) < 1e-9
    assert np.max(abs(deltaF-((held[:, :, :, 2]-held[:, :, :1, 2])+deltaE))) < 1e-9
    return dict(task=task, held=held, baseline=baseline, net=net, deltaF=deltaF, deltaE=deltaE)


def report():
    """数据封存及独立审计通过后才出正文；不会根据输出再改搜索或筛名单。"""
    assert not (RUN/'analysis.json').exists(); verify_protocol(); verify_selection()
    audit = teacher.read(RUN/'independent_audit.json'); assert audit['status'] == 'PASS'
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (r, d) in enumerate(panel['pairs']): weights[n, panel['labels'][d]] = 1.; weights[n, panel['labels'][r]] = -1.
    names = ('solutions', 'transmission', 'intervention_units', 'member_drops', 'teacher_summary',
             'budget_search', 'teacher_output_channels', 'positive_controls')
    tables = {name: [] for name in names}; teacher_rows = {}; examples = {}
    for seed in teacher.SEEDS:
        verify_folder(RUN/f'search/seed_{seed}'); verify_folder(RUN/f'evaluation/seed_{seed}')
        raw = load(RUN/f'evaluation/seed_{seed}/data.npz'); search = load(RUN/f'search/seed_{seed}/data.npz')
        units = load(RUN/f'search/seed_{seed}/units.npz'); frozen = load(RUN/f'search/seed_{seed}/frozen.npz')
        prior = load(AR/f'evaluation/seed_{seed}/data.npz'); v = measures(raw, weights)
        focus = prior['threshold'][:, :, 3, 0].astype(bool) & ~prior['threshold'][:, :, 3, 1].astype(bool)
        geometric = (raw['margins'] >= GUARD).all(-1); rows = []; path_valid = np.zeros((72, 2, 48), bool)
        oldF = np.einsum('nmqvk,nk->nmqv', prior['natural'], weights)[:, :, 1, 3]
        oldA = np.einsum('nmik,nk->nmi', prior['baseline'], weights)[:, :, 1]
        denominator = oldF-oldA
        for case in range(72):
            for m, mask in enumerate((3, 15)):
                meta = dict(seed=seed, case=case, mask=mask); chosen = {k: a[case, m] for k, a in units.items()}
                pp = paths(chosen); C = frozen['channels'][case, m]; C = C[C >= 0]
                path_valid[case, m] = [p['valid'] for p in pp]
                j = int(frozen['lists'][case, m, 0, 0]); direction = int(frozen['half_direction'][case, m])
                for i in range(36):
                    strategy, family, variant = i//12, i % 12//3, i % 3; e = i+1
                    pindex, amount, eligible = search['choices'][case, m, i]; p = pp[int(pindex)] if pindex >= 0 else None
                    own, fixed, actual = map(bool, geometric[case, m, e]); physical = fixed and actual; aligned = own and physical
                    passed = bool(eligible) and (own if strategy == 0 else physical if strategy == 1 else aligned)
                    noH = raw['shifts'][case, m, e, 0] <= 1e-10
                    Fdelta = float(v['deltaF'][case, m, e]); logitdelta = float(abs(raw['natural'][case, m, e, 2]-raw['natural'][case, m, 0, 2]).max())
                    if noH:
                        assert logitdelta < 1e-8
                        assert np.max(abs(raw['margins'][case, m, e, 1:]-raw['margins'][case, m, 0, 1:])) < 1e-8
                    status = 'not_met' if not passed else 'decomposition_only' if noH else 'physical_with_task_effect' if physical and abs(Fdelta) > 1e-8 else 'physical_task_contrast_null' if physical else 'reference_dependent_H_changed'
                    neighbor = search['neighbor_margins'][case, m, i]
                    stable = bool((objectives(neighbor)[:, strategy] >= GUARD).all()) and passed
                    row = dict(**meta, focus539=bool(focus[case, m]), strategy=STRATEGIES[strategy], family=FAMILIES[family], variant=VARIANTS[variant], endpoint=e,
                        eligible=bool(eligible), path=int(pindex), unit_count=0 if p is None else p['k'],
                        ids=json.dumps([] if p is None else p['ids'].tolist()), amount=float(amount),
                        own_q=float(raw['margins'][case, m, e, 0].min()), fixed_q=float(raw['margins'][case, m, e, 1].min()),
                        actual_q=float(raw['margins'][case, m, e, 2].min()), own_joint=own, physical_joint=physical, aligned_joint=aligned,
                        passed=passed, stable_tested_neighbors=stable, status=status, H_max_change=float(raw['shifts'][case, m, e, 0]),
                        H_rms_change=float(raw['shifts'][case, m, e, 1]), A_max_change=float(raw['shifts'][case, m, e, 2]), R_max_change=float(raw['shifts'][case, m, e, 3]),
                        actual_F_change=Fdelta, actual_logit_max_change=logitdelta, baseline_F_change=float(v['baseline'][case, m, e]-v['baseline'][case, m, 0]),
                        actual_j_change=float(raw['point_z'][case, m, e, 2]-raw['point_z'][case, m, 0, 2]), actual_j_flip=bool(raw['actual_point_flip'][case, m, e]),
                        point_E=float(v['net'][case, m, e, 2]), point_E_change=float(v['deltaE'][case, m, e]),
                        C_output_effect=float(v['task'][case, m, e, 2]-v['task'][case, m, e, 3]),
                        C_point_control_contrast=float(v['net'][case, m, e, 2]-v['net'][case, m, e, 3]))
                    rows.append(row); tables['solutions'].append(row)
                    if variant == 0 and p is not None:
                        for rank, coordinate in enumerate(p['ids'], 1):
                            ch, yy, xx = (int(coordinate), '', '') if family < 2 else tuple(map(int, np.unravel_index(coordinate, (32, 32, 32))))
                            tables['intervention_units'].append(dict(**meta, strategy=STRATEGIES[strategy], family=FAMILIES[family], rank=rank,
                                id=int(coordinate), channel=ch, y=yy, x=xx, amount=float(amount), passed=passed))
                    # 主失败例固定；成功/仅分解例按编号取首个，属于明确标注的事后示例。
                    kind = 'first_aligned' if strategy == 2 and variant == 0 and passed and focus[case, m] else 'first_decomposition' if strategy == 0 and variant == 0 and passed and noH and focus[case, m] else None
                    if kind is not None and kind not in examples:
                        examples[kind] = dict(solution=row, C=C.tolist(), point=list(map(int, np.unravel_index(j, (32, 32, 32)))),
                            direction=direction, margins=raw['margins'][case, m, e].tolist(), point_z=raw['point_z'][case, m, e].tolist())
                if seed == 202609291 and case == 1 and mask == 3:
                    examples['planned'] = dict(meta=meta, C=C.tolist(), point=list(map(int, np.unravel_index(j, (32, 32, 32)))),
                        rows=[r for r in rows if r['case'] == case and r['mask'] == mask and r['variant'] == 'selected'])
                for e in range(39):
                    for q, context in enumerate(FIELDS):
                        tables['transmission'].append(dict(**meta, endpoint=e, context=context, F=float(v['task'][case, m, e, q]),
                            j_z=float(raw['point_z'][case, m, e, q]), point_net=float(v['net'][case, m, e, q]),
                            point_direct=float(raw['direct'][case, m, e, q]), second_response=float(raw['response'][case, m, e, q]),
                            held_F=float(v['held'][case, m, e, q])))
                for k, e in enumerate(DROP_ENDPOINTS):
                    fullF = float(v['task'][case, m, e, 2])
                    for index, channel in enumerate(C):
                        dropF = float(raw['drop_logits'][case, m, k, index]@weights[case])
                        tables['member_drops'].append(dict(**meta, endpoint=e, channel=int(channel), all_F=fullF, drop_F=dropF,
                            drop_F_change=dropF-fullF, all_j_z=float(raw['point_z'][case, m, e, 2]), drop_j_z=float(raw['drop_z'][case, m, k, index]),
                            drop_original_side=bool(direction*raw['drop_z'][case, m, k, index] <= -GUARD)))
                for e, control in ((37, 'all_source16'), (38, 'all_middle32')):
                    tables['positive_controls'].append(dict(**meta, control=control, moving_q=float(raw['margins'][case, m, e, 0].min()),
                        original_C_exactly_necessary=bool((raw['margins'][case, m, e, 0] > 1e-8).all()), actual_F_change=float(v['deltaF'][case, m, e])))
        teacher_rows[seed] = rows
        for group, masks in (('3', (3,)), ('15', (15,)), ('pooled', (3, 15))):
            den = rms(denominator[:, [0] if group == '3' else [1] if group == '15' else [0, 1]].ravel())
            for strategy in STRATEGIES:
                for family in FAMILIES:
                    for variant in VARIANTS:
                        rr = [r for r in rows if r['focus539'] and r['mask'] in masks and r['strategy'] == strategy and r['family'] == family and r['variant'] == variant]
                        good = [r for r in rr if r['passed']]
                        tables['teacher_summary'].append(dict(seed=seed, group=group, strategy=strategy, family=family, variant=variant, n=len(rr),
                            success=len(good), success_rate=len(good)/len(rr), exact_in_three=sum(r['aligned_joint'] for r in rr),
                            decomposition_only=sum(r['status'] == 'decomposition_only' for r in rr), reference_dependent=sum(r['status'] == 'reference_dependent_H_changed' for r in rr),
                            task_effectful=sum(abs(r['actual_F_change']) > 1e-8 for r in good), vector_effectful=sum(r['actual_logit_max_change'] > 1e-8 for r in good),
                            new_j_flips=sum(r['actual_j_flip'] for r in good), stable_neighbors=sum(r['stable_tested_neighbors'] for r in good),
                            success_F_rms=rms([r['actual_F_change'] for r in good]), success_E_change_rms=rms([r['point_E_change'] for r in good]),
                            E_to_F_rms=None if not good or rms([r['actual_F_change'] for r in good]) < 1e-15 else rms([r['point_E_change'] for r in good])/rms([r['actual_F_change'] for r in good]),
                            E_change_ratio=None if not good else rms([r['point_E_change'] for r in good])/den,
                            success_median_units=None if not good else float(np.median([r['unit_count'] for r in good])),
                            success_median_amount=None if not good else float(np.median([r['amount'] for r in good])), original_full_d_rms=den))
        for family in FAMILIES:
            good = [r for r in rows if r['focus539'] and r['strategy'] == 'aligned' and r['variant'] == 'selected' and r['family'] == family and r['passed']]
            for ch in range(32):
                dx = [raw['channel_change'][r['case'], (3, 15).index(r['mask']), r['endpoint'], ch] for r in good]
                tables['teacher_output_channels'].append(dict(seed=seed, family=family, channel=ch, n=len(good),
                    task_change_mean=None if not good else float(np.mean(dx)), task_change_rms=rms(dx)))
        for pindex in range(48):
            for strategy in range(3):
                scope = search['coarse_margins'][:, :, pindex]
                success = (objectives(scope)[:, :, 1:, strategy] >= GUARD).any(-1)
                tables['budget_search'].append(dict(seed=seed, strategy=STRATEGIES[strategy], family=FAMILIES[pindex//12],
                    variant=VARIANTS[pindex % 12//4], prefix_slot=pindex % 4, focus_n=int(focus.sum()),
                    eligible_cases=int((focus & path_valid[:, :, pindex]).sum()), grid_success=int((success & focus & path_valid[:, :, pindex]).sum())))
    claims = summarize(tables, examples)
    for name, rows in tables.items(): write_table(name, rows)
    teacher.dump(RUN/'claim_checks.json', claims); teacher.dump(RUN/'examples.json', examples)
    figures = make_figures(tables, claims, examples)
    write_report(tables, claims, examples, audit)
    teacher.dump(RUN/'analysis.json', dict(status='PASS', utc=teacher.now(), table_rows={k: len(v) for k, v in tables.items()},
        figures=figures, output_aggregation='teacher-equal;empty successes excluded withvalidteacher count;focus539fixed;all864rowsretained',
        illustrative_selection='planned291/1/3 retained;firstaligned andfirstdecomposition sortedseed/case/mask/family;notselectedbytaskoutput'))
    print(json.dumps(dict(status='PASS', rows=sum(len(r) for r in tables.values()), summary=claims['families'])), flush=True)


def summarize(tables, examples):
    """计数以状态/方法为单位，另列去重状态；效果均值只用有成功的教师。"""
    rows = [r for r in tables['solutions'] if r['focus539']]
    result = dict(status='PASS', denominator=539, families={})
    unique = set()
    for family in FAMILIES:
        moving = [r for r in rows if r['family'] == family and r['variant'] == 'selected' and r['strategy'] == 'moving_A' and r['passed']]
        physical = [r for r in rows if r['family'] == family and r['variant'] == 'selected' and r['strategy'] == 'fixed_actual' and r['passed']]
        aligned = [r for r in rows if r['family'] == family and r['variant'] == 'selected' and r['strategy'] == 'aligned' and r['passed']]
        unique.update((r['seed'], r['case'], r['mask']) for r in aligned)
        teacher_good = [r for r in tables['teacher_summary'] if r['group'] == 'pooled' and r['strategy'] == 'aligned' and r['variant'] == 'selected' and r['family'] == family and r['success'] > 0]
        random = [sum(r['passed'] for r in rows if r['family'] == family and r['variant'] == variant and r['strategy'] == 'aligned') for variant in ('random0', 'random1')]
        result['families'][family] = dict(moving_success=len(moving), moving_decomposition_only=sum(r['status'] == 'decomposition_only' for r in moving),
            moving_reference_dependent=sum(r['status'] == 'reference_dependent_H_changed' for r in moving),
            moving_also_physical=sum(r['physical_joint'] for r in moving), physical_success=len(physical), aligned_success=len(aligned), random_aligned=random,
            aligned_task_effectful=sum(abs(r['actual_F_change']) > 1e-8 for r in aligned), aligned_vector_effectful=sum(r['actual_logit_max_change'] > 1e-8 for r in aligned),
            aligned_new_j_flips=sum(r['actual_j_flip'] for r in aligned), aligned_stable_neighbors=sum(r['stable_tested_neighbors'] for r in aligned),
            success_teachers=len(teacher_good), F_rms_teacher_mean=None if not teacher_good else float(np.mean([r['success_F_rms'] for r in teacher_good])),
            deltaE_ratio_teacher_mean=None if not teacher_good else float(np.mean([r['E_change_ratio'] for r in teacher_good])),
            E_to_F_rms_teacher_mean=None if not teacher_good else float(np.mean([r['E_to_F_rms'] for r in teacher_good])))
    result.update(aligned_case_method_count=sum(r['aligned_success'] for r in result['families'].values()), aligned_unique_states=len(unique),
        examples={key: value['solution'] if 'solution' in value else value['meta'] for key, value in examples.items()})
    return result


def make_figures(tables, claims, examples):
    """标准科学图PNG/SVG；每图在交付前逐张视觉检查。"""
    assert not (RUN/'delivery_manifest.json').exists()
    os.environ['MPLCONFIGDIR'] = str(RUN/'.plot_cache')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10, 'svg.fonttype': 'none'})
    directory = BASE/'reports/figures/dsprites_joint_window_20261018'; directory.mkdir(exist_ok=True)
    files = []; labels = ('源通道≤4', '中间通道≤4', '第一激活≤16点', '第二激活≤16点')
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'svg'):
            p = directory/f'{name}.{ext}'; fig.savefig(p, dpi=140, bbox_inches='tight'); files.append(str(p.relative_to(BASE)).replace('\\', '/'))
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); x = np.arange(4)
    for j, (field, label, color) in enumerate((('moving_success', '移动A目标', '#3980a9'), ('physical_success', '固定参照+实际组外目标', '#df9a3a'), ('aligned_success', '三个参照同时目标', '#6c9d70'))):
        ax[0].bar(x+(j-1)*.24, [claims['families'][f][field] for f in FAMILIES], .23, label=label, color=color)
    ax[0].set_title('各目标分别搜索后的联合必要性通过数'); ax[0].set_ylabel('539个原失败状态中的计数'); ax[0].legend(fontsize=8)
    bottom = np.zeros(4)
    for field, label, color in (('moving_decomposition_only', '仅分解变化', '#bf6860'), ('moving_reference_dependent', 'H改变、共同参照未通过', '#b6c3cf'), ('moving_also_physical', '共同参照也通过', '#65936d')):
        values = np.array([claims['families'][f][field] for f in FAMILIES]); ax[1].bar(x, values, .6, bottom=bottom, label=label, color=color); bottom += values
    ax[1].set_title('移动A目标的成功者：同一干预再分类'); ax[1].set_ylabel('案例状态数'); ax[1].legend(fontsize=8)
    for a in ax: a.set_xticks(x, labels, rotation=8); a.grid(axis='y', alpha=.2); a.set_axisbelow(True)
    save(fig, '01_reference_checks')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    planned = next(r for r in examples['planned']['rows'] if r['family'] == 'source' and r['strategy'] == 'moving_A')
    for a, row, title in ((ax[0], planned, '固定失败例：291 / 案例1 / mask3'), (ax[1], examples['first_aligned']['solution'], '首个通过三参照的事后示例')):
        s = load(RUN/f"search/seed_{row['seed']}/data.npz"); m = (3, 15).index(row['mask']); i = row['endpoint']-1
        coarse_m = s['coarse_margins'][row['case'], m, row['path']]
        coarse_score = objectives(coarse_m)[:, STRATEGIES.index(row['strategy'])]
        center = np.linspace(0, 1, 33)[np.argmax(coarse_score[1:])+1]
        extra = np.linspace(max(0., center-1/32), min(1., center+1/32), 17)
        amounts = np.concatenate((np.linspace(0, 1, 33), extra)); order = np.argsort(amounts, kind='stable')
        coarse = np.concatenate((coarse_m, s['refine_margins'][row['case'], m, i])).min(-1)
        for q, (label, color) in enumerate(zip(('移动AI', '固定原A1', '实际组外值'), ('#347fad', '#df903d', '#7b629f'))):
            a.plot(amounts[order], coarse[order, q], color=color, lw=1.4, marker='.', ms=3, label=label)
        a.axhline(0, color='#444', lw=.8); a.axvline(row['amount'], color='#777', ls=':', lw=1)
        a.set_xlabel('回退强度 λ'); a.set_ylabel('三项条件中的最小安全余量'); a.set_title(title); a.grid(alpha=.2)
    ax[0].legend(fontsize=9); save(fig, '02_strength_and_reference')
    good = [r for r in tables['solutions'] if r['focus539'] and r['strategy'] == 'aligned' and r['variant'] == 'selected' and r['passed']]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, field in enumerate(('actual_F_change', 'point_E_change')):
        for family, color in zip(FAMILIES, ('#347fad', '#d98e38', '#8e69a7', '#689268')):
            ids = [i for i, row in enumerate(good) if row['family'] == family]
            ax[j].scatter(ids, [good[i][field] for i in ids], s=30, color=color, label=labels[FAMILIES.index(family)])
        ax[j].axhline(0, color='#777', lw=.7); ax[j].set_xlabel('按模型、案例、方法排序的成功解'); ax[j].set_ylabel('供体−受体 logit'); ax[j].grid(axis='y', alpha=.2)
    ax[0].set_title('实际完整输出变化 ΔF'); ax[1].set_title('共同j开关控制的变化 ΔE'); ax[0].legend(fontsize=8)
    fig.suptitle('每个点代表一个案例和一种方法，两个面板各用自己的纵轴范围', y=1.02); save(fig, '03_output_effects')
    ex = examples['first_aligned']['solution']; rr = [r for r in tables['member_drops'] if r['seed'] == ex['seed'] and r['case'] == ex['case'] and r['mask'] == ex['mask'] and r['endpoint'] == ex['endpoint']]
    direction = examples['first_aligned']['direction']; names = ['完整C']+[f"删通道{r['channel']}" for r in rr]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2)); x = np.arange(len(names))
    ax[0].bar(x, [direction*rr[0]['all_j_z']]+[direction*r['drop_j_z'] for r in rr], color=['#62966a']+['#769eb7']*len(rr)); ax[0].axhline(0, color='#444', lw=.8)
    ax[0].set_ylabel('沿原触发方向的预激活'); ax[0].set_title('实际组外通道保持：删掉任一成员都不触发')
    ax[1].bar(x, [0.]+[r['drop_F_change'] for r in rr], color='#c49450'); ax[1].axhline(0, color='#444', lw=.8)
    ax[1].set_ylabel('删除该通道对F的影响'); ax[1].set_title('必要性针对阈值；任务读出的符号另行测量')
    for a in ax: a.set_xticks(x, names, rotation=10); a.grid(axis='y', alpha=.2); a.set_axisbelow(True)
    save(fig, '04_each_member_necessary')
    return files


def write_report(tables, claims, examples, audit):
    """正文数值均来自已审计数组/证据表；参照依赖不自动称作虚假网络变化。"""
    assert not REPORT.exists()
    labels = ('源通道≤4', '中间通道≤4', '第一ReLU激活≤16点', '第二ReLU激活≤16点')
    summary = []; separation = []; effects = []; teachers = []
    for family, label in zip(FAMILIES, labels):
        c = claims['families'][family]; random = c['random_aligned']
        summary.append(f"| {label} | {c['moving_success']} | {c['physical_success']} | {c['aligned_success']} | {random[0]}；{random[1]} | {c['success_teachers']}/6 |")
        separation.append(f"| {label} | {c['moving_decomposition_only']} | {c['moving_reference_dependent']} | {c['moving_also_physical']} |")
        fmt = lambda value: '—' if value is None else f'{value:.6f}'
        effects.append(f"| {label} | {c['aligned_task_effectful']}/{c['aligned_success']} | {c['aligned_new_j_flips']} | {c['aligned_stable_neighbors']} | {fmt(c['F_rms_teacher_mean'])} | {fmt(c['deltaE_ratio_teacher_mean'])} |")
    for seed in teacher.SEEDS:
        rr = [r for r in tables['teacher_summary'] if r['seed'] == seed and r['group'] == 'pooled' and r['strategy'] == 'aligned' and r['variant'] == 'selected']
        teachers.append(f"| {str(seed)[-3:]} | {rr[0]['n']} | "+' | '.join(str(next(r['success'] for r in rr if r['family'] == f)) for f in FAMILIES)+' |')
    planned = next(r for r in examples['planned']['rows'] if r['family'] == 'source' and r['strategy'] == 'moving_A')
    positive = examples['first_aligned']; p = positive['solution']; decomposition = examples['first_decomposition']; d = decomposition['solution']
    member_rows = [r for r in tables['member_drops'] if (r['seed'], r['case'], r['mask'], r['endpoint']) == (p['seed'], p['case'], p['mask'], p['endpoint'])]
    members = [f"| {r['channel']} | {r['all_j_z']:+.6f} | {r['drop_j_z']:+.6f} | {r['drop_F_change']:+.6f} |" for r in member_rows]
    text = f'''# dSprites：保留联合必要性的上游恢复

2026-10-18。本轮仍用36幅既有图、72个共享这些图的有向配对、六个冻结同架构模型和源mask3/15。下游点j、原通道组C、方向和上游定向/两随机名单全部继承，只有干预强度和名单前缀在本轮搜索。通道、案例与空间坐标均从0计数。

**找到了少量共同参照下成立且有任务读出变化的恢复，同时确认大部分“移动A下的恢复”不能等同于这种恢复。** 三个参照同时通过的定向解共有{claims['aligned_case_method_count']}个案例×方法记录，去重后为{claims['aligned_unique_states']}个案例状态。它们是同面板搜索结果，不是未见输入验证。

## 1. 把“联合必要性”变成严格条件

令s为原触发方向，b为指定基准在j的预激活，t_c=s·D_c为C各通道沿该方向的贡献。原组**全部加入能触发，删去任一成员都不能触发**：

`s·b<−ε`，`s·zC>ε`，`max_c s·(zC−D_c)<−ε`。

当这些条件成立，每个t_c都为正，因此任何真子集都不会比最强删一成员更强；这等价于全部真子集均不能触发。本轮既独立验证该证书，也对选定端点显式枚举原C的全部真子集。它比“所有单通道各自不足”更强：三个通道可能每个单独不足，而两个通道已经足够，第三个并非必要。

也可写成阈值余量窗口，g=−s·b：

`Σt_c−min(t_c)+ε < g < Σt_c−ε`。

只让组合跨过零点不够；A推回太多，会让单通道或更小子集也能触发。搜索分数取基准原侧、完整C触发、最强删一成员仍失败这三项的最小安全余量q，要求q≥1e−6；数值审计容差1e−9。该恒等式描述第一预激活，不把整个网络和最终输出当线性函数。

这里的“必要”始终相对于声明的通道替换基准及冻结j而言，不是对分类结果的绝对必要性，也不是对任意基准成立的唯一电路。

## 2. 三个参照阻止只改分解也算成功

对同一上游干预I，先重算七源状态，构建AI、RI和实际组合HI。三个几何检验如下：

| 参照 | 全C场 | 基准/删通道的替换值 | 检验目的 |
|---|---|---|---|
| 移动AI | AI+P_C RI | 当前AI | 与前轮问题连续，但可能依赖分解 |
| 固定原A1 | A1+P_C(HI−A1) | 同一个原A1 | 只有实际C值改变才可能恢复 |
| 实际组外 | HI | C删掉时用原A1替换，其他通道保持HI | 验证实际下游背景中C每个成员是否必要 |

后两者只依赖HI和固定A1。**HI不变时，无论AI/RI怎样移动，这两种检验和实际完整输出都不会变化。** 这是区分只改分解与实际变化的依据。固定A1本身仍是构造参照，改变这个参照可能改变必要性结论；未通过共同参照不自动意味着H或输出没有真实变化。

本轮分别搜索三个目标：只要求移动AI；同时要求固定原A1和实际组外；要求全部三个参照。它们各自选解，不能把三列当成同一批解逐级筛选的漏斗。另对同一批移动AI成功解重新分类，见后表。

## 3. 上游干预、强度与结果

源通道、中间整通道和第一/第二ReLU后的局部激活，均从完整状态向对应半强度状态做凸组合：`vI=(1−λ)v1+λv0`，λ∈[0,1]。上游激活保持非负，后续原ReLU真实重算。非负只保证数值符号，不保证整张激活场能由自然输入产生。本轮不使用负的强制上游gate激活；AI、固定A1消融和下游gate诊断控制仍可能离开自然激活范围。

源/中间最多4通道，前缀1/2/3/4；两层激活最多16点，前缀1/4/8/16。实际预算不足则截断并去掉重复前缀。每条路径先查33个固定强度点，再在最佳路径的一格范围细化17点；找到可行解时优先较少单元，再取较大安全余量。全部六模型解先封存，再计算任务读出，F和任务梯度不参与选解。两随机对照具有相同候选池、最大预算、前缀和强度搜索，各解按实际选中的单元数记录预算。

主队列仍是上一轮固定的539个基准侧未变而原C不能触发状态，全部864状态也保留。下表均是“原C每个成员必要”，不是仅跨阈值。

| 定向回退 | 移动AI目标通过 | 固定A1+实际背景目标通过 | 三参照同时通过 | 两随机的三参照通过数 | 定向成功模型 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(summary)}

各方法共得到 {claims['aligned_case_method_count']} 条恢复记录，按案例和源状态去重后为 {claims['aligned_unique_states']} 个状态。通道编号沿用各模型自己的索引。

| 教师 | 原失败状态 | 源通道三参照通过 | 中间通道通过 | 第一激活通过 | 第二激活通过 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(teachers)}

![共同参照与结果分类](figures/dsprites_joint_window_20261018/01_reference_checks.png)

以下重新检查的是**同一批移动AI目标的成功解**，每行三类合计其成功数：

| 干预族 | HI不变、仅分解变化 | HI改变、共同参照未通过 | 共同参照也通过 |
|---|---:|---:|---:|
{chr(10).join(separation)}

其中“H改变、共同参照未通过”不能被说成只改分解或虚假输出变化；它只说明还没有建立本轮更严格的原C共同参照必要性。

全16源通道及全32中间通道回退是阳性控制，按构造回到半强度，因此移动AI检验在864/864状态恢复原C每个成员必要。它只核验实现，不计入少量干预的搜索成果。

## 4. 固定失败例和实际恢复例

前轮固定例仍为教师291、案例1、mask3，j=(28,13,19)、C=(12,31)。对源通道`{planned['ids']}`回退λ={planned['amount']:.8f}，移动AI的最小联合余量变为{planned['own_q']:+.6f}，联合必要性成立。但固定原A1为{planned['fixed_q']:+.6f}、实际组外为{planned['actual_q']:+.6f}，没有通过；实际H最大变化{planned['H_max_change']:.6f}、F改变{planned['actual_F_change']:+.6f} logit。它是实际网络发生变化、但联合恢复依赖参照的例子。

首个三参照成功例为**事后按模型/案例/mask/方法排序取首个几何通过者**，不按输出大小挑选：教师{p['seed']}、案例{p['case']}、mask{p['mask']}，j={tuple(positive['point'])}、C={tuple(positive['C'])}。回退{p['family']}的单元`{p['ids']}`，λ={p['amount']:.8f}；三参照q分别为{p['own_q']:+.6f}、{p['fixed_q']:+.6f}、{p['actual_q']:+.6f}。实际H最大改变{p['H_max_change']:.6f}，实际完整F改变{p['actual_F_change']:+.6f}，共同j控制变化ΔE={p['point_E_change']:+.6f}。

| 删去原C通道 | 完整C时j预激活 | 删去后j预激活 | 删去对F的改变 |
|---|---:|---:|---:|
{chr(10).join(members)}

第三类例子同样按编号取首个，不按输出挑选：教师{d['seed']}、案例{d['case']}、mask{d['mask']}，回退{d['family']}的`{d['ids']}`，λ={d['amount']:.8f}。移动AI的q={d['own_q']:+.6f}，但HI最大变化{d['H_max_change']:.3g}、实际三类logit最大变化{d['actual_logit_max_change']:.3g}。A最大变化{d['A_max_change']:.6f}，R最大变化{d['R_max_change']:.6f}；这是只改A/R分解的恢复，不能算实际层间传递恢复。

![强度和参照](figures/dsprites_joint_window_20261018/02_strength_and_reference.png)

图中连接的是已测强度点，不证明全部点间或所有其他名单也满足条件。原组C和输出读出不重选；全文示例与失败解均可从逐行表复核。

![逐成员消融](figures/dsprites_joint_window_20261018/04_each_member_necessary.png)

## 5. 最终输出与下游开关作用

F=供体类别logit−受体类别logit，三类原始logit全部保存。以同一个原A1的j开关为诊断参考，真实重算第二ReLU，得到点净量E；逐场核验`E=第一直接项+第二层响应`。比较干预前后，`ΔF=固定j后的输出效应变化+ΔE`。这是共同控制的效应差分，不是唯一中介份额，也不以第一代数项代替实际净量。

| 三参照定向解 | 任务对比F非零/成功数 | 实际j新翻转数 | ±1/256邻点均通过 | 成功教师内ΔF RMS再等权均值 | ΔE RMS/原d RMS均值 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(effects)}

各模型在有恢复记录时纳入均值。分母为该模型、该源掩码下全部原状态的全强度残差修正 RMS。表中直接列比值，F 的符号给出得分变化方向。点在干预前已触发而干预后仍触发时，还可能是C对该点的必要性改变，不应一律说成新开关翻转。

![实际输出与共同开关控制](figures/dsprites_joint_window_20261018/03_output_effects.png)

逐成员消融还保存原始F变化。每个成员对阈值必要，不要求它们对F具有同一符号；真实读出中仍可能有其他通道和后续ReLU抵消。末层32通道的输出变化也逐项保存，不能用一个恢复开关解释全部ΔF。

## 6. 核验、范围与文件

独立审计{audit['endpoint_fields']}个下游场和同数点控制，重算{audit['searched_geometry_fields']}个局部候选/细化/邻点场，枚举{audit['enumerated_proper_subset_fields']}个真子集场，并对{audit['original_C_member_drop_fields']}个实际通道删除场运行原模型。最大数值误差{max(audit['maximum_errors'].values()):.3g}。局部生产使用11×11线性缓存，审计另用13×13原conv/BN直接重算，不调用新核心搜索或几何函数；输出用原forward、梯度用独立自动微分。十一份历史交付1558条目及更早来源保持一致。

本轮是有明确预算的同面板自适应搜索，没有未见输入、跨架构、分类收益或天然唯一电路结论。第一/第二激活点名单继承旧gate控制的定位，没有为新凸组合控制重新排整池；未找到不证明其他点、强度和方向不存在恢复。固定原A1检验排除了仅改变本轮A/R的解释，但必要性仍依赖指定替换参照。共同j开关固定是计算诊断，可能产生负强制激活；不能把它当作自然前向网络。

- [代码与轴定义](../runs/dsprites_joint_window_20261018/README.md)
- [全部搜索解、真实H变化与输出分类](../runs/dsprites_joint_window_20261018/tables/solutions.csv)
- [具体通道/激活坐标](../runs/dsprites_joint_window_20261018/tables/intervention_units.csv)
- [原C逐成员删除](../runs/dsprites_joint_window_20261018/tables/member_drops.csv)
- [逐场传递和共同开关控制](../runs/dsprites_joint_window_20261018/tables/transmission.csv)
- [逐模型汇总](../runs/dsprites_joint_window_20261018/tables/teacher_summary.csv)
- [前轮上游定位与恢复](dSprites_基准移动的上游通道与开关干预_2026-10-17.md)
'''
    REPORT.write_text(text, encoding='utf-8')


if __name__ == '__main__': report()

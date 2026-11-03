"""上游基准移动与救回的逐项证据、六模型汇总和可导出科学图。

先在每个模型内算RMS/比例，再对六模型等权；72对共享36图，
不做把案例当独立样本的显著性检验。全文同时保留成功与失败。
"""
from __future__ import annotations
import csv
import json
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_upstream_rescue import BASE, RUN, PRIOR, SHAPE, load, verify_protocol, verify_selection, verify_folder
from upstream_rescue_mechanism import METHODS, FAMILIES, anchors

REPORT = BASE/'reports/dSprites_基准移动的上游通道与开关干预_2026-10-17.md'
REGIMES = ('A_only', 'coherent'); CONTEXTS = ('frozen_C', 'all32')
SELECTED = (1, 6, 11, 16)


def rms(v):
    """RMS先按模型和预定分组计算，绝不直接相加各项RMS。"""
    return float(np.sqrt(np.mean(np.asarray(v)**2)))


def table(name, rows):
    """UTF-8 BOM便于直接用本地Excel查看；所有来源与编号逐行保留。"""
    path = RUN/f'tables/{name}.csv'; assert not path.exists(); path.parent.mkdir(exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def measures(raw, weights):
    """区分原始任务读出、基准读出、相对基准修正及单点净作用。"""
    task = np.einsum('nmersk,nk->nmers', raw['natural'], weights)
    baseline = np.einsum('nmek,nk->nme', raw['baseline'], weights)
    d = task-baseline[..., None, None]
    net = np.einsum('nmersk,nk->nmers', raw['natural']-raw['held'], weights)
    task_change = task-task[:, :, :1, :1]
    base_change = baseline-baseline[:, :, :1]
    d_change = d-d[:, :, :1, :1]
    assert np.max(abs(task_change-base_change[..., None, None]-d_change)) < 1e-9
    assert np.max(abs(net-raw['direct']-raw['response'])) < 1e-9
    assert np.max(abs(d-raw['parts'].sum((-1, -2)))) < 1e-9
    return dict(task=task, baseline=baseline, d=d, net=net, task_change=task_change, base_change=base_change, d_change=d_change)


def selection_rows(meta, chosen):
    """把每个冻结名单的物理编号和单独修复分数转成可读证据。"""
    result = []
    for family, lists in enumerate((chosen['source_lists'], chosen['middle_lists'], *chosen['gate_lists'])):
        if family < 2:
            scores = chosen['source_scores' if family == 0 else 'middle_scores']; lookup = dict(enumerate(scores))
        else: lookup = dict(zip(chosen[f'gate{family-1}_ids'], chosen[f'gate{family-1}_scores']))
        for variant, ids in zip(('selected', 'random0', 'random1'), lists):
            for rank, coordinate in enumerate(ids[ids >= 0], 1):
                c, y, x = (int(coordinate), '', '') if family < 2 else tuple(map(int, np.unravel_index(coordinate, (32, 32, 32))))
                result.append(dict(**meta, family=FAMILIES[family], variant=variant, rank=rank,
                                   id=int(coordinate), channel=c, y=y, x=x, baseline_single_repair=float(lookup[coordinate])))
    return result


def report():
    """全部表和图只由已审计原数组派生，不再改数值定位或筛案例。"""
    assert not (RUN/'analysis.json').exists(); verify_protocol(); verify_selection()
    audit = teacher.read(RUN/'independent_audit.json'); assert audit['status'] == 'PASS'
    panel = load(SHAPE/'panel/data.npz'); weights = np.zeros((72, 3))
    for n, (r, d) in enumerate(panel['pairs']): weights[n, panel['labels'][d]] = 1.; weights[n, panel['labels'][r]] = -1.
    names = ('baseline_parts', 'source_channels', 'middle_channels', 'gate_channel_parts', 'selected_units',
             'thresholds', 'interventions', 'teacher_parts', 'teacher_thresholds', 'teacher_outputs',
             'rescue_output_contrasts', 'rescue_patterns', 'example_gates')
    tables = {name: [] for name in names}; all_focus = []; all_parts = []; all_thresholds = []; all_net = []; all_within = []; all_example = []
    for seed in teacher.SEEDS:
        verify_folder(RUN/f'discovery/seed_{seed}'); verify_folder(RUN/f'evaluation/seed_{seed}')
        sel = load(RUN/f'discovery/seed_{seed}/selection.npz'); fixed = load(RUN/f'discovery/seed_{seed}/frozen_prior.npz')
        raw = load(RUN/f'evaluation/seed_{seed}/data.npz'); prior = load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        values = measures(raw, weights); focus = prior['threshold'][:, :, 3, 0].astype(bool) & ~prior['threshold'][:, :, 3, 1].astype(bool)
        all_focus.append(focus); all_thresholds.append(raw['threshold']); all_net.append(values['net']); part_array = np.zeros((72, 2, 3))
        # 追加描述性口径：只要求C内部各单通道不足，区别于预定的全32严格口径。
        member_mask = (fixed['channels'][..., None] == np.arange(32)).any(-2)
        solo_margin = fixed['half_direction'][:, :, None, None, None]*(raw['base_z'][..., None, None]+raw['incoming'])
        C_solos_safe = np.where(member_mask[:, :, None, None], solo_margin, -np.inf).max(-1) < -1e-8
        within_joint = raw['threshold'][..., 1] & C_solos_safe; all_within.append(within_joint)
        for case in range(72):
            for m, mask in enumerate((3, 15)):
                meta = dict(seed=seed, case=case, mask=mask); chosen = {k: v[case, m] for k, v in sel.items()}
                direction = int(fixed['half_direction'][case, m]); C = fixed['channels'][case, m]; C = C[C >= 0]
                groups = np.array(anchors(mask))-1
                linear = chosen['source_linear'][groups].sum(0); gates = chosen['gate_parts'][:, groups].sum(1)
                parts = np.array([linear.sum(), gates[0].sum(), gates[1].sum()]); part_array[case, m] = parts
                delta = chosen['base_z'][1]-chosen['base_z'][0]; assert abs(parts.sum()-delta) < 1e-9
                tables['baseline_parts'].append(dict(**meta, focus539=bool(focus[case, m]), direction=direction,
                    half_b=float(chosen['base_z'][0]), full_b=float(chosen['base_z'][1]), delta_b=float(delta),
                    linear=float(parts[0]), first_relu=float(parts[1]), second_relu=float(parts[2]),
                    oriented_delta=float(direction*delta), oriented_linear=float(direction*parts[0]),
                    oriented_first=float(direction*parts[1]), oriented_second=float(direction*parts[2]),
                    linear_largest=bool(abs(parts[0]) >= max(abs(parts[1:]))), nonlinear_opposes_linear=bool(parts[0]*parts[1:].sum() < 0)))
                for c in range(16):
                    tables['source_channels'].append(dict(**meta, channel=c, source_group=c//4,
                        eligible=bool(chosen['source_pool'][c]), linear_drift=float(linear[c]),
                        oriented_linear=float(direction*linear[c]), rollback_single_repair=float(chosen['source_scores'][c])))
                for c in range(32):
                    tables['middle_channels'].append(dict(**meta, channel=c, frozen_member=bool(c in C),
                        rollback_single_repair=float(chosen['middle_scores'][c]),
                        oriented_drift=float(-chosen['middle_scores'][c])))
                    for stage in range(2): tables['gate_channel_parts'].append(dict(**meta, stage=stage+1, channel=c,
                        propagated_drift=float(gates[stage, c]), oriented_drift=float(direction*gates[stage, c])))
                tables['selected_units'].extend(selection_rows(meta, chosen))
                for method, name in enumerate(METHODS):
                    for regime, reg in enumerate(REGIMES):
                        flags = raw['threshold'][case, m, method, regime]
                        ba = raw['base_z'][case, m, method]; incoming = raw['incoming'][case, m, method, regime]
                        repair_a = direction*(ba-raw['base_z'][case, m, 0])
                        repair_r = direction*(incoming[C]-raw['incoming'][case, m, 0, 0, C]).sum()
                        tables['thresholds'].append(dict(**meta, focus539=bool(focus[case, m]), method=name, regime=reg,
                            b=float(ba), group_z=float(raw['group_z'][case, m, method, regime]),
                            all_z=float(raw['all_z'][case, m, method, regime]), oriented_margin=float(direction*raw['group_z'][case, m, method, regime]),
                            A_repair=float(repair_a), R_group_repair=float(repair_r),
                            same_base=bool(flags[0]), group_cross=bool(flags[1]), all_solos_safe=bool(flags[2]), joint=bool(flags[3]), all_cross=bool(flags[4])))
                        tables['thresholds'][-1].update(C_solos_safe=bool(C_solos_safe[case, m, method, regime]),
                                                       within_C_joint=bool(within_joint[case, m, method, regime]))
                        for context, ctx in enumerate(CONTEXTS):
                            key = (case, m, method, regime, context)
                            tables['interventions'].append(dict(**meta, focus539=bool(focus[case, m]), method=name, regime=reg, context=ctx,
                                baseline_F=float(values['baseline'][case, m, method]), F=float(values['task'][key]),
                                d=float(values['d'][key]), raw_F_change=float(values['task_change'][key]),
                                baseline_F_change=float(values['base_change'][case, m, method]), d_change=float(values['d_change'][key]),
                                point_net=float(values['net'][key]), direct=float(raw['direct'][key]), response=float(raw['response'][key]),
                                point_remainder=float(raw['point_remainder'][key]), first_flips=int(raw['flips'][key][0]), second_flips=int(raw['flips'][key][1])))
                if seed == 202609291 and case == 1 and mask == 3:
                    all_example = dict(meta=meta, C=C.tolist(), j=list(map(int, np.unravel_index(int(fixed['lists'][case, m, 0, 0]), (32, 32, 32)))),
                        source_selected=chosen['source_lists'][0][chosen['source_lists'][0] >= 0].tolist(),
                        middle_selected=chosen['middle_lists'][0][chosen['middle_lists'][0] >= 0].tolist(),
                        parts=parts.tolist(), b=raw['base_z'][case, m].tolist(), zC=raw['group_z'][case, m].tolist(), incoming=raw['incoming'][case, m].tolist(),
                        threshold=raw['threshold'][case, m].tolist(), values={k: v[case, m].tolist() for k, v in values.items()})
        all_parts.append(part_array)
        # 教师内先统计，再等权汇总；focus归属由上一轮决定，不能被本轮结果改写。
        for group, indices in (('3', np.tile([True, False], (72, 1))), ('15', np.tile([False, True], (72, 1))), ('pooled', np.ones((72, 2), bool))):
            denominator = rms(values['d'][:, :, 0, 0, 1][indices])
            for cohort, subset in (('all', indices), ('focus539', indices & focus)):
                x = part_array[subset]; total = x.sum(1); n = len(x); den = rms(total)
                row = dict(seed=seed, group=group, cohort=cohort, n=n, drift_rms=den,
                    cancellation=float(1-np.sum(abs(total))/np.sum(abs(x))),
                    linear_largest=int((abs(x[:, 0]) >= abs(x[:, 1:]).max(1)).sum()),
                    nonlinear_opposes_linear=int((x[:, 0]*x[:, 1:].sum(1) < 0).sum()))
                for j, label in enumerate(('linear', 'first', 'second')):
                    row[label+'_rms'] = rms(x[:, j]); row[label+'_ratio'] = rms(x[:, j])/den
                    row[label+'_projection'] = float(np.mean(x[:, j]*total)/np.mean(total**2))
                tables['teacher_parts'].append(row)
                flags = raw['threshold'][subset]
                for method, name in enumerate(METHODS):
                    if cohort == 'focus539':
                        a_cross = flags[:, method, 0, 1]; c_cross = flags[:, method, 1, 1]
                        a_joint = flags[:, method, 0, 3]; c_joint = flags[:, method, 1, 3]
                        tables['rescue_patterns'].append(dict(seed=seed, group=group, method=name, n=n,
                            both_cross=int((a_cross & c_cross).sum()), A_only_cross=int((a_cross & ~c_cross).sum()),
                            coherent_only_cross=int((~a_cross & c_cross).sum()), neither_cross=int((~a_cross & ~c_cross).sum()),
                            both_joint=int((a_joint & c_joint).sum()), A_only_joint=int((a_joint & ~c_joint).sum()),
                            coherent_only_joint=int((~a_joint & c_joint).sum()), neither_joint=int((~a_joint & ~c_joint).sum())))
                    for regime, reg in enumerate(REGIMES):
                        f = flags[:, method, regime]
                        tables['teacher_thresholds'].append(dict(seed=seed, group=group, cohort=cohort, n=n, method=name, regime=reg,
                            same_base=int(f[:, 0].sum()), group_cross=int(f[:, 1].sum()), joint=int(f[:, 3].sum()),
                            group_cross_rate=float(f[:, 1].mean()), joint_rate=float(f[:, 3].mean()), all_cross=int(f[:, 4].sum()),
                            within_C_joint=int(within_joint[subset][:, method, regime].sum()),
                            baseline_repair_mean=float(np.mean((raw['base_z'][:, :, method]-raw['base_z'][:, :, 0])[subset]*fixed['half_direction'][subset]))))
                        for context, ctx in enumerate(CONTEXTS):
                            delta = values['task_change'][subset][:, method, regime, context]; net = values['net'][subset][:, method, regime, context]
                            direct = raw['direct'][subset][:, method, regime, context]; response = raw['response'][subset][:, method, regime, context]
                            mass = np.sum(abs(direct)+abs(response)); cancellation = float(1-np.sum(abs(net))/mass) if mass > 1e-15 else 0.
                            tables['teacher_outputs'].append(dict(seed=seed, group=group, cohort=cohort, n=n, method=name, regime=reg, context=ctx,
                                original_full_d_rms=denominator, raw_F_change_rms=rms(delta), raw_F_change_mean=float(delta.mean()),
                                raw_F_change_ratio=rms(delta)/denominator, d_rms=rms(values['d'][subset][:, method, regime, context]),
                                d_change_rms=rms(values['d_change'][subset][:, method, regime, context]),
                                baseline_F_change_rms=rms(values['base_change'][subset][:, method]), point_net_rms=rms(net),
                                point_net_ratio=rms(net)/denominator, direct_rms=rms(direct), response_rms=rms(response),
                                point_cancellation=cancellation, positive_raw_F_changes=int((delta > 1e-8).sum()), negative_raw_F_changes=int((delta < -1e-8).sum())))
                    # 描述性追加汇总：只在原C真实恢复的状态比较同一个j开关控制。
                    # 这些状态AI仍在原侧，所以控制参考与原全强度A1相同，
                    # DeltaE可作为共同开关控制下的输出效应差分，不能称唯一份额。
                    for regime, reg in enumerate(REGIMES):
                        restored = indices & focus & raw['threshold'][:, :, method, regime, 1]
                        restored_n = int(restored.sum())
                        for context, ctx in enumerate(CONTEXTS):
                            row = dict(seed=seed, group=group, method=name, regime=reg, context=ctx, n=restored_n,
                                       original_full_d_rms=denominator)
                            if restored_n:
                                delta = values['task_change'][restored][:, method, regime, context]
                                point_contrast = values['net'][restored][:, method, regime, context]-values['net'][restored][:, 0, 0, context]
                                row.update(raw_F_change_rms=rms(delta), point_net_rms=rms(values['net'][restored][:, method, regime, context]),
                                    point_contrast_rms=rms(point_contrast), point_contrast_ratio=rms(point_contrast)/denominator,
                                    remaining_controlled_effect_rms=rms(delta-point_contrast),
                                    point_contrast_projection=float(np.mean(point_contrast*delta)/np.mean(delta**2)) if np.mean(delta**2) > 1e-20 else None)
                            else: row.update(raw_F_change_rms=None, point_net_rms=None, point_contrast_rms=None, point_contrast_ratio=None,
                                             remaining_controlled_effect_rms=None, point_contrast_projection=None)
                            if cohort == 'all': tables['rescue_output_contrasts'].append(row)
    gate_example = teacher.read(RUN/'planned_example_upstream.json')
    for gate in gate_example['gates']:
        for a in gate['anchors']:
            tables['example_gates'].append(dict(stage=gate['stage'], id=gate['id'], channel=gate['coordinate'][0], y=gate['coordinate'][1], x=gate['coordinate'][2], **a, **gate['target']))
    focus = np.asarray(all_focus); flags = np.asarray(all_thresholds); parts = np.asarray(all_parts); nets = np.asarray(all_net); within = np.asarray(all_within)
    assert int(focus.sum()) == 539 and len(tables['baseline_parts']) == 864
    claims = dict(status='PASS', scope=dict(images=36, directed_pairs=72, frozen_teachers=6, case_states=864, focus539=int(focus.sum())), methods={})
    for method, name in enumerate(METHODS):
        f = flags[focus][:, method]
        a, c = f[:, 0, 1], f[:, 1, 1]; aj, cj = f[:, 0, 3], f[:, 1, 3]
        tr = [r for r in tables['teacher_thresholds'] if r['group'] == 'pooled' and r['cohort'] == 'focus539' and r['method'] == name]
        outs = [r for r in tables['teacher_outputs'] if r['group'] == 'pooled' and r['cohort'] == 'focus539' and r['method'] == name and r['regime'] == 'coherent' and r['context'] == 'all32']
        claims['methods'][name] = dict(A_cross=int(a.sum()), coherent_cross=int(c.sum()), A_joint=int(aj.sum()), coherent_joint=int(cj.sum()),
            both_cross=int((a & c).sum()), A_only_cross=int((a & ~c).sum()), coherent_only_cross=int((~a & c).sum()), neither_cross=int((~a & ~c).sum()),
            A_cross_teacher_mean=float(np.mean([r['group_cross_rate'] for r in tr if r['regime'] == 'A_only'])),
            coherent_cross_teacher_mean=float(np.mean([r['group_cross_rate'] for r in tr if r['regime'] == 'coherent'])),
            raw_F_change_rms_teacher_mean=float(np.mean([r['raw_F_change_rms'] for r in outs])),
            raw_F_change_ratio_teacher_mean=float(np.mean([r['raw_F_change_ratio'] for r in outs])),
            point_net_ratio_teacher_mean=float(np.mean([r['point_net_ratio'] for r in outs])),
            point_cancellation_teacher_mean=float(np.mean([r['point_cancellation'] for r in outs])),
            A_within_C_joint=int(within[focus][:, method, 0].sum()), coherent_within_C_joint=int(within[focus][:, method, 1].sum()),
            coherent_categories=[int((~f[:, 1, 0]).sum()), int((f[:, 1, 0] & ~f[:, 1, 1]).sum()),
                                 int((f[:, 1, 1] & ~f[:, 1, 3]).sum()), int(f[:, 1, 3].sum())],
            restored_C_point_nonzero=int((c & (abs(nets[focus][:, method, 1, 0]) > 1e-8)).sum()),
            restored_C_point_zero=int((c & (abs(nets[focus][:, method, 1, 0]) <= 1e-8)).sum()))
    selected_parts = [r for r in tables['teacher_parts'] if r['group'] == 'pooled' and r['cohort'] == 'focus539']
    claims['focus_parts'] = dict(linear_largest=int((abs(parts[focus][:, 0]) >= abs(parts[focus][:, 1:]).max(1)).sum()),
        nonlinear_opposes_linear=int((parts[focus][:, 0]*parts[focus][:, 1:].sum(1) < 0).sum()),
        ratios={label: float(np.mean([r[label+'_ratio'] for r in selected_parts])) for label in ('linear', 'first', 'second')},
        projections={label: float(np.mean([r[label+'_projection'] for r in selected_parts])) for label in ('linear', 'first', 'second')})
    claims['planned_example'] = all_example
    claims['planned_example']['gate1_selected'] = [g['coordinate'] for g in gate_example['gates'] if g['stage'] == 1]
    claims['planned_example']['gate2_selected'] = [g['coordinate'] for g in gate_example['gates'] if g['stage'] == 2]
    claims['planned_example']['gate_details'] = gate_example['gates']
    claims['restored_output_contrasts'] = {}
    for method in SELECTED:
        rows = [r for r in tables['rescue_output_contrasts'] if r['group'] == 'pooled' and r['method'] == METHODS[method]
                and r['regime'] == 'coherent' and r['context'] == 'all32' and r['n'] > 0]
        claims['restored_output_contrasts'][METHODS[method]] = dict(valid_teachers=len(rows), n=sum(r['n'] for r in rows),
            raw_F_change_rms_teacher_mean=float(np.mean([r['raw_F_change_rms'] for r in rows])),
            point_contrast_ratio_teacher_mean=float(np.mean([r['point_contrast_ratio'] for r in rows])))
    # 正修复量并不保证跨阈值；控制改变基准方向时仍按失败保留。
    for name, rows in tables.items(): table(name, rows)
    teacher.dump(RUN/'claim_checks.json', claims)
    figures = make_figures(tables, claims)
    write_report(tables, claims, audit, figures)
    teacher.dump(RUN/'analysis.json', dict(status='PASS', utc=teacher.now(), table_rows={n: len(r) for n, r in tables.items()},
        figures=figures, aggregation='equal6teacher;separate3/15/pooled;focus539frozenpriorcohort;denominatororiginalfullall32dRMSforallcaseswithinteacher/group'))
    print(json.dumps(dict(status='PASS', table_rows=sum(len(r) for r in tables.values()), claims=claims['methods'])), flush=True)


def make_figures(tables, claims):
    """使用标准Matplotlib生成出版可导出PNG/SVG，后续逐张视觉检查。"""
    assert not (RUN/'delivery_manifest.json').exists(), 'sealed figures are read-only'
    os.environ['MPLCONFIGDIR'] = str(RUN/'.plot_cache')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10,
                         'svg.fonttype': 'none', 'figure.dpi': 130})
    directory = BASE/'reports/figures/dsprites_upstream_rescue_20261017'; directory.mkdir(exist_ok=True)
    paths = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'svg'):
            p = directory/f'{name}.{ext}'; fig.savefig(p, bbox_inches='tight'); paths.append(str(p.relative_to(BASE)).replace('\\', '/'))
        plt.close(fig)
    rows = [r for r in tables['teacher_parts'] if r['group'] == 'pooled' and r['cohort'] == 'focus539']
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.3)); x = np.arange(6)
    for j, (name, label, color) in enumerate(zip(('linear', 'first', 'second'), ('原开关下线性传播', '第一ReLU余项', '第二ReLU余项'), ('#28678d', '#d7982b', '#8d5a9b'))):
        ax[0].bar(x+(j-1)*.24, [100*r[name+'_ratio'] for r in rows], .23, label=label, color=color)
        ax[1].bar(x+(j-1)*.24, [100*r[name+'_projection'] for r in rows], .23, label=label, color=color)
    for a in ax:
        a.set_xticks(x, [str(s)[-3:] for s in teacher.SEEDS]); a.set_xlabel('冻结教师编号'); a.axhline(0, color='#888', lw=.6); a.grid(axis='y', alpha=.2); a.set_axisbelow(True)
    ax[0].set_ylabel('分项RMS / 基准移动RMS（%）'); ax[1].set_ylabel('对基准移动的有符号投影（%）')
    ax[0].set_title('原539个失败状态：幅度传播与开关余项'); ax[1].set_title('投影可为负，不是唯一因果份额')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, 1.09), ncol=3, frameon=False, fontsize=10)
    save(fig, '01_baseline_provenance')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); labels = ('源通道≤4', '中间通道≤4', '第一开关≤16', '第二开关≤16')
    for q, (regime, label) in enumerate((('A_only', '固定原R'), ('coherent', '联动重算R'))):
        for j, (metric, title) in enumerate((('group_cross_rate', '原C重新跨阈值'), ('joint_rate', '恢复全部32单通道均不足的严格条件'))):
            for control, color in (('selected', ('#3074a0', '#df943d')[q]), ('random', ('#9fc2da', '#e7c29a')[q])):
                values = []
                for method in SELECTED:
                    ns = [METHODS[method]] if control == 'selected' else [METHODS[method+1], METHODS[method+2]]
                    rr = [r[metric] for r in tables['teacher_thresholds'] if r['group'] == 'pooled' and r['cohort'] == 'focus539' and r['regime'] == regime and r['method'] in ns]
                    values.append(100*np.mean(rr))
                offset = (2*q+(control == 'random')-1.5)*.18
                ax[j].bar(np.arange(4)+offset, values, .17, label=label+('：定向' if control == 'selected' else '：随机均值'), color=color)
                ax[j].set_title(title); ax[j].set_xticks(np.arange(4), labels); ax[j].set_ylabel('六教师等权恢复率（%）'); ax[j].grid(axis='y', alpha=.2); ax[j].set_axisbelow(True)
    ax[0].legend(fontsize=8, ncol=2); fig.suptitle('仍为原539个失败；每个随机控制与本族定向方法预算相同', y=1.02)
    save(fig, '02_rescue_and_joint')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
    for j, metric in enumerate(('raw_F_change_ratio', 'point_net_ratio')):
        for q, color in enumerate(('#347cad', '#b5c6d3', '#d2a66b')):
            means = []
            for i, method in enumerate(SELECTED):
                rr = [r for r in tables['teacher_outputs'] if r['group'] == 'pooled' and r['cohort'] == 'focus539' and r['regime'] == 'coherent' and r['context'] == 'all32' and r['method'] == METHODS[method+q]]
                y = np.array([r[metric]*100 for r in rr]); means.append(y.mean()); ax[j].scatter(np.full(6, i+(q-1)*.2), y, color=color, s=17, alpha=.8)
            ax[j].scatter(np.arange(4)+(q-1)*.2, means, color=color, marker='_', s=200, linewidths=2, label=('定向', '随机0', '随机1')[q])
        ax[j].set_xticks(np.arange(4), labels); ax[j].set_ylabel('相对原全部残差d RMS（%）'); ax[j].grid(axis='y', alpha=.2)
    ax[0].set_title('实际上游干预对原始F的总变化'); ax[1].set_title('干预后该下游单点的净作用'); ax[0].legend(fontsize=9)
    fig.suptitle('联动重算、全部残差：两种效应不可混同；点=单个教师', y=1.02)
    save(fig, '03_final_output')
    e = claims['planned_example']; ids = (0, 4, 9, 11, 19, 21)
    text = ('原全强度', f"源通道{e['source_selected'][0]}", f"中间通道{e['middle_selected'][0]}", '第一层≤16点', '第二层1点', '两层全固定')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); z = np.array(e['zC']); vals = {k: np.array(v) for k, v in e['values'].items()}
    ax[0].plot(np.arange(6), z[list(ids), 0], 'o-', color='#28799f', label='固定原R：C处预激活')
    ax[0].plot(np.arange(6), z[list(ids), 1], 's-', color='#d38735', label='联动重算：C处预激活')
    ax[0].axhline(0, color='#333', lw=.8); ax[0].set_ylabel('原下游点j的预激活'); ax[0].set_title('预先指定例子：教师291 / 案例1 / mask3'); ax[0].legend(fontsize=9)
    ax[1].bar(np.arange(6)-.16, vals['task_change'][list(ids), 1, 1], .3, label='原始F总变化', color='#3982a9')
    ax[1].bar(np.arange(6)+.16, vals['net'][list(ids), 1, 1], .3, label='j点净作用（全部残差）', color='#c98e46')
    ax[1].set_ylabel('供体−受体 logit'); ax[1].set_title('同一阈值的恢复不决定输出变化方向'); ax[1].legend(fontsize=9)
    for a in ax: a.set_xticks(np.arange(6), text, rotation=12); a.grid(axis='y', alpha=.2); a.set_axisbelow(True)
    save(fig, '04_planned_example')
    return paths


def write_report(tables, claims, audit, figures):
    """正文中的数值从claim_checks与表直接插入，交付验证再次独立重算。"""
    assert not REPORT.exists(); ms = claims['methods']; p = claims['focus_parts']; e = claims['planned_example']
    assert all(ms[METHODS[i]]['restored_C_point_zero'] == 0 for i in SELECTED)
    family_table = []
    failure_table = []
    for method, label in zip(SELECTED, ('源层通道≤4', 'block2输出通道≤4', 'block2第一ReLU≤16', 'block2第二ReLU≤16')):
        c = ms[METHODS[method]]; r0, r1 = ms[METHODS[method+1]], ms[METHODS[method+2]]
        family_table.append(f"| {label} | {c['A_cross']}/539 | {c['coherent_cross']}/539 | {r0['coherent_cross']}/539；{r1['coherent_cross']}/539 | {c['coherent_within_C_joint']}/539 | {c['coherent_joint']}/539 |")
        failure_table.append(f"| {label} | "+' | '.join(map(str, c['coherent_categories']))+' |')
    detail = []
    for method in (0, 1, 4, 6, 9, 11, 14, 16, 19, 21):
        c = ms[METHODS[method]]; detail.append(f"| {METHODS[method]} | {c['both_cross']} | {c['A_only_cross']} | {c['coherent_only_cross']} | {c['neither_cross']} | {c['raw_F_change_rms_teacher_mean']:.6f} | {100*c['point_net_ratio_teacher_mean']:.3f}% |")
    example = []
    for method in (0, 1, 4, 6, 9, 11, 14, 16, 19, 21):
        v = e['values']; flags = e['threshold'][method]
        example.append(f"| {METHODS[method]} | {e['b'][method]:+.6f} | {e['zC'][method][0]:+.6f} | {e['zC'][method][1]:+.6f} | {int(flags[1][1])} / {int(flags[1][3])} | {v['task_change'][method][1][1]:+.6f} | {v['net'][method][1][1]:+.6f} |")
    projection = ', '.join(f"{label}{100*p['projections'][key]:.2f}%" for key, label in (('linear', '线性'), ('first', '第一ReLU'), ('second', '第二ReLU')))
    ratio_text = ', '.join(f"{label}{100*p['ratios'][key]:.2f}%" for key, label in (('linear', '线性'), ('first', '第一ReLU'), ('second', '第二ReLU')))
    per_teacher = []
    for seed in teacher.SEEDS:
        rr = [r for r in tables['teacher_thresholds'] if r['seed'] == seed and r['group'] == 'pooled' and r['cohort'] == 'focus539' and r['regime'] == 'coherent']
        counts = [next(r['group_cross'] for r in rr if r['method'] == METHODS[i]) for i in SELECTED]
        per_teacher.append(f"| {str(seed)[-3:]} | {rr[0]['n']} | "+' | '.join(map(str, counts))+' |')
    rescued_output = []
    for method in SELECTED:
        row = claims['restored_output_contrasts'][METHODS[method]]
        rescued_output.append(f"| {METHODS[method]} | {row['n']} | {row['valid_teachers']} | {row['raw_F_change_rms_teacher_mean']:.6f} | {100*row['point_contrast_ratio_teacher_mean']:.3f}% |")
    gate1 = next(g for g in e['gate_details'] if g['stage'] == 1 and g['coordinate'] == e['gate1_selected'][0])
    gate2 = next(g for g in e['gate_details'] if g['stage'] == 2 and g['coordinate'] == e['gate2_selected'][0])
    first_anchor, second_anchor = gate2['anchors']; first_g1 = gate1['anchors'][0]
    text = f'''# dSprites：基准移动的上游通道与开关干预

日期：2026-10-17。此轮将“为什么原通道组合失效”继续追到上游，并直接验证能否恢复。所有数值均来自本轮原数组与独立审计；先锁定方法，再以基准修复量定位，封存全部名单后才运行恢复评价。

主要结果：同一批539个失败状态中，定向回退最多4个源通道、最多4个中间通道，联动重算后分别恢复{ms['src_selected4']['coherent_cross']}、{ms['mid_selected4']['coherent_cross']}个原C的跨阈值；定向固定最多16个第一/第二开关则恢复{ms['g1_selected16']['coherent_cross']}、{ms['g2_selected16']['coherent_cross']}个。部分恢复存在且超过各族随机控制的汇总计数，但不能全部恢复，恢复后也不一定还需要多通道联合。基准三项中线性幅度传播在{p['linear_largest']}/539个状态里绝对值最大；把全部上游开关固定起来并不能普遍救回。

## 1. 问题和范围

仍使用36幅既有dSprites图、72个共享这些图的有向配对、6个同架构冻结模型、源mask3/15；没有加入图像、训练模型或重新选择下游点。原源层为block1输出；中间层H为block2输出；观察点j为block3第一ReLU；C是此前半强度冻结的中间通道组。物理通道编号只在各自模型内解释。

本轮主队列是上一轮已确定的539个“基准仍在原侧，而C不能跨阈值”的案例状态；全部864个状态也保留。案例共享输入，不作为539个独立统计样本。汇总先在模型内计算，再对6模型等权，原始计数则直接标明分母。

## 2. A从哪里来，如何向上追

对源集合S，k=|S|，在强度α下运行空源状态和各单组状态：

`Aα(S)=Σg∈S Hα(single_g)−(k−1)H(empty)`，`Rα(S)=Hα(S)−Aα(S)`。

A是构造的加和基准，不是网络自然的一层激活。空状态不随α变化，因此`ΔA=Σg[H1(single_g)−H0(single_g)]`。令`bα=block3第一预激活(Aα)[j]`，卷积+eval BN对A是仿射映射，故`Δb=L_jΔA`。这里的仿射性仅用于预激活与恒等式，整个网络及最终读出仍包含重算的非线性。

对每个单组锚点，以它自己的半强度开关为参考，精确分解：

`Δb = 原开关下的线性幅度传播 + 第一ReLU余项的传播 + 第二ReLU余项`。

第一项同时包含block2残差支路和shortcut。两层开关都固定到各锚点的半强度参考后，留下的基准移动严格等于第一项。因为各锚点参考不同，这不意味着跨源状态构造的R消失。

在原539个失败中，线性项绝对值在三项里最大的是**{p['linear_largest']}/539**；非线性两项合计与线性项反向的是**{p['nonlinear_opposes_linear']}/539**。六模型等权的分项RMS/Δb RMS分别为：{ratio_text}；有符号投影为：{projection}。RMS不能相加，投影依赖参考和分解，不是唯一因果百分比。证据支持先检验幅度传播，不能把A移动全部归因于开关翻转。

![基准来源](figures/dsprites_upstream_rescue_20261017/01_baseline_provenance.png)

## 3. 定位哪些通道和开关

只用`s·(bI−b1)`作分数，其中s是原触发方向。分数不读取C的R贡献、是否恢复或分类logit。源层候选为S内8/16通道，中间候选为32输出通道；第一/第二ReLU候选严格限于能到达j的5×5/3×3空间支持，且至少一个相关锚点有大于1e−8的实际余项。第一开关的单点分数重算第二ReLU，没有用局部梯度代替实际变化。通道和空间坐标全部从0计数。

每例取正修复分数的最多4个源/中间通道，或最多16个第一/第二开关；无正分数则保留为空操作。每族两套随机对照来自同一有效池，使用与该例定向方法相同的实际预算。各族干预单位不同，不能仅凭数量作跨族效率比较。所有物理编号、坐标、实际预算和单点分数见[冻结名单](../runs/dsprites_upstream_rescue_20261017/tables/selected_units.csv)。源通道、中间通道和开关分解另有逐项表；完整候选分数在各模型discovery NPZ。

回退源/中间通道时，把它在七个源状态中的值替换为对应半强度值。固定开关时，用对应半强度gate乘以当前预激活，再重算后续网络；强制开的反事实可能出现负激活，不能解释为自然ReLU输出。

## 4. 恢复必须分两种检验

每次先对空、4个单组和2个目标源状态一致地施加上游干预I，再重建`AI、RI`：

- **A_only**：固定原R1，运行`AI+P_C R1`及`AI+R1`。只检验基准修复是否足以恢复阈值。
- **coherent**：联动重算R，运行`AI+P_C RI`及`HI=AI+RI`。全部残差条件下才是实际上游干预后完整网络的输出。

“一致”指同一物理名单、同一回退规则；参考值和gate分别取各源状态对应的半强度，不是七状态共用一个固定gate。这个状态依赖的诊断算子不能被说成一条输入无关、天然固定的回路。

恢复跨阈值还要求AI留在原基准侧；预定的严格联合恢复另要求32个单通道各自都不能触发。另外从同一冻结数据追加“C内各单通道不足”的描述性口径，避免把组外某单通道足够误说成C内也不需联合。两者都不要求原C的每个成员必不可少。固定C从不重选，也不声称它仍是当前最小必要集合。

| 定向干预 | 固定R跨阈值 | 联动跨阈值 | 同预算随机0；随机1（联动） | 联动C内需联合 | 联动全32严格联合 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(family_table)}

各模型的联动跨阈值计数如下，分母是其原失败状态数。没有把模型间物理通道编号当成同一个功能；这张表用于查看恢复现象的异质性。

| 教师 | 原失败数 | 源通道≤4 | 中间通道≤4 | 第一开关≤16 | 第二开关≤16 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_teacher)}

![恢复率](figures/dsprites_upstream_rescue_20261017/02_rescue_and_joint.png)

不能恢复的状态也须区分：基准本身已换侧，与基准仍在原侧但C不足，含义不同。以下四类互斥且每行合计539；正向推回b的名单可能过量，排名分数并不保证基准保持原侧。

| 定向干预 | 基准换侧 | 同侧但C不足 | C跨阈值但32通道中有单通道足够 | 全32严格联合恢复 |
|---|---:|---:|---:|---:|
{chr(10).join(failure_table)}

| 方法 | 两种均恢复 | 仅固定R恢复 | 仅联动恢复 | 两种均失败 | 联动全部残差：F变化RMS均值 | 单点净作用/原d RMS均值 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail)}

“仅固定R恢复”直接说明真实上游控制同时改变R，可能破坏合成A/R条件下的恢复；“仅联动恢复”说明R重算也可能补偿。两者都必须保留，不能把基准修复等同于整个原组合机制恢复。

阈值判据可以写成严格恒等式：`mI=m1+δA+δR_C`，其中`m1=s·(b1+ΣC D1c)`、`δA=s·(bI−b1)`、`δR_C=s·ΣC(DIc−D1c)`。除基准仍在原侧外，还需要δA与δR_C合计超过原缺口；只按δA定位不能保证恢复。每例的两个有符号修复量均保存在阈值表。

## 5. 恢复如何影响最终输出

`F=供体类别logit−受体类别logit`；分别保存`F(AI)`、`F(X)`及`d=F(X)−F(AI)`。逐案例核验`ΔF=ΔF(A)+Δd`，基准变化不能从输出中悄悄扣掉。报告的F变化是干预对固定任务读出的作用，不等于分类改善。

对冻结j还实际固定其第一gate到当前AI的参考，第二ReLU重新运行。单点净量E严格等于“第一余项直接项+第二层响应”，可由第二层抵消，不能以第一代数项单独声称输出贡献。表中的单点幅度使用所有既有案例中原完整强度全部残差d RMS作共同分母；先模型内计算再等权。它不是上游干预总效应，也不是原C全部输出作用。

![输出总变化和单点作用](figures/dsprites_upstream_rescue_20261017/03_final_output.png)

为直接回答“恢复增加了多少输出作用”，另外对**联动后确实恢复的子集**做描述性汇总。这些案例的AI在j处仍与原A1同侧，因此前后固定的是同一个j开关参考，可以比较`ΔE=E_I−E_原`，逐例严格满足`ΔF=固定j后的输出效应+ΔE`。这里只改变一个点的计算控制；它不提供全部层间传递的唯一中介比例。分母仍为各教师/组全部原案例的d RMS，均值只用该族有恢复案例的模型，另列模型数。

| 方法 | 恢复案例数 | 有恢复模型数 | 恢复子集的完整网络ΔF RMS均值 | 同一j控制的ΔE RMS/原d RMS均值 |
|---|---:|---:|---:|---:|
{chr(10).join(rescued_output)}

上述子集以本轮恢复结果定义，属于事后描述，不是额外未见输入验证。逐模型和C/all32两种范围均见[恢复子集输出对照](../runs/dsprites_upstream_rescue_20261017/tables/rescue_output_contrasts.csv)。

这四类定向方法各自的联动恢复案例，在只加入C残差时，j的实际净作用均非零（阈值1e−8）。但全部残差场中，该点可能已在干预前被组外通道触发；因此恢复带来的新增作用要看同一参考下的ΔE，不能直接把恢复后的E当成新增效果。

## 6. 预先指定案例：能否具体复原

教师291、案例1、mask3，观察点`j={tuple(e['j'])}`，原中间组`C={tuple(e['C'])}`。这个例子是按上一轮539失败的编号顺序固定的，没有按本轮成功挑选。源定向名单为`{e['source_selected']}`；中间定向名单为`{e['middle_selected']}`。各ReLU物理坐标、锚点前后预激活、开关状态和带符号传播项见[例子开关表](../runs/dsprites_upstream_rescue_20261017/tables/example_gates.csv)。该例Δb的线性、第一及第二余项依次为`{e['parts']}`。

这里可直接看到两种不同情况：

- **有效回退**：只把源通道0回退到对应半强度，b由{e['b'][0]:+.6f}移动到{e['b'][4]:+.6f}，原C在联动重算后由{e['zC'][0][1]:+.6f}到{e['zC'][4][1]:+.6f}，确实重新跨过零点。完整网络F改变{e['values']['task_change'][4][1][1]:+.6f} logit。但C内通道12单独加入后预激活已为{e['b'][4]+e['incoming'][4][1][12]:+.6f}，联合必要性未恢复。
- **只改锚点的无效修复**：第二ReLU坐标`{tuple(e['gate2_selected'][0])}`在两个单组锚点里分别从{first_anchor['half_z']:+.6f}到{first_anchor['full_z']:+.6f}、从{second_anchor['half_z']:+.6f}到{second_anchor['full_z']:+.6f}，都是开→关；在实际组合状态却从{gate2['target']['target_half_z']:+.6f}到{gate2['target']['target_full_z']:+.6f}，始终关。把它固定为锚点的半强度开关后，b推回{e['b'][19]-e['b'][0]:.6f}，固定R时C的z变成{e['zC'][19][0]:+.6f}；实际组合H未改变，所以`ΔR=−ΔA`，且该输出通道31属于C，C内的这次变化恰好抵消，联动z仍是{e['zC'][19][1]:+.6f}，完整网络ΔF为0。负的强制激活属于计算控制，不是自然开关恢复。

第一ReLU的定向名单以`{tuple(e['gate1_selected'][0])}`开头：该点在单组G0由{first_g1['half_z']:+.6f}到{first_g1['full_z']:+.6f}，在组合状态却始终关。单独控制这个点也没有改变完整网络F；同时控制最多16个基准有利点时，C处联动z变为{e['zC'][11][1]:+.6f}，完整F改变{e['values']['task_change'][11][1][1]:+.6f}。完整坐标和各锚点状态保留在表里。对照地，该例第一ReLU余项合计是正值，抵抗负向线性移动；第二余项合计为负，推动不利移动，不能以“某层开关多”替代带符号传递分析。

最多4通道或16开关是上限，不保证越多越好。该例源4通道、中间4通道和第二层16开关把b本身推到正侧，即使C端点为正，也不满足“原基准侧上由C触发”的定义，已按失败保留。

特别要区分两种抵消：上述第二开关例子的`ΔR=−ΔA`来自`R=H−A`且H未变的重建代数，不能包装成两个独立物理回路互相抵消。下游点的“第一余项直接项+第二层响应”则由实际固定该点、重算后续ReLU得到。两种证据在表中分开保存。

| 方法 | AI的b | 固定R时C的z | 联动时C的z | 联动跨阈值/严格联合 | 联动全部残差ΔF | 同条件j净作用 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(example)}

![预定例子](figures/dsprites_upstream_rescue_20261017/04_planned_example.png)

## 7. 可复现性与边界

独立核验{audit['endpoint_fields']}个下游场、同数单点干预、全部864案例状态的上游分解与全部候选分数，数值最大误差为{max(audit['maximum_errors'].values()):.3g}。上游用原block2.forward注入核验，候选第一开关用原卷积在裁剪支持上核验，传播梯度另用自动微分；不会仅用同一函数重写汇总。十份旧交付1467文件条目及更早历史来源保持一致。

这是一轮同面板机制探索和内部反事实，不是未见输入验证、跨架构验证或自然语义因果识别。定位名单用已观察到的基准变化生成，不能当作事前跨输入预测能力。通道/开关回退为一种诊断控制，当前证据不把任何单点叫作永久电路，也不把输出改变叫作任务收益。

恒等式和独立数值核验保证此轮分解与干预计算一致，并不证明这些经验规律对任意模型和输入都成立。

- [本轮代码、轴定义与运行顺序](../runs/dsprites_upstream_rescue_20261017/README.md)
- [基准三项与原539归属](../runs/dsprites_upstream_rescue_20261017/tables/baseline_parts.csv)
- [源通道](../runs/dsprites_upstream_rescue_20261017/tables/source_channels.csv)、[中间通道](../runs/dsprites_upstream_rescue_20261017/tables/middle_channels.csv)、[开关通道分解](../runs/dsprites_upstream_rescue_20261017/tables/gate_channel_parts.csv)
- [恢复阈值与A/R修复量](../runs/dsprites_upstream_rescue_20261017/tables/thresholds.csv)、[读出与抵消](../runs/dsprites_upstream_rescue_20261017/tables/interventions.csv)
- [逐教师恢复](../runs/dsprites_upstream_rescue_20261017/tables/teacher_thresholds.csv)、[逐教师输出](../runs/dsprites_upstream_rescue_20261017/tables/teacher_outputs.csv)
- [前轮A/R分离](dSprites_基准与组合残差的分离控制_2026-10-17.md)
'''
    REPORT.write_text(text, encoding='utf-8')


if __name__ == '__main__': report()

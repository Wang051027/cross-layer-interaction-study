"""只读核验下游定位交付：原始数组、全部表格、统计分母和封存文件。

本脚本不重跑模型；完整模型干预由独立审计负责。这里独立展开67种
名单、逐行对照数值证据，并验证输出余量、残差支路和抵消的恒等式。
封存前只写核验记录；封存后完全只读，不修改已交付研究结果。
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_downstream_routes import BASE, RUN, PRIOR, SHAPE, verify_protocol, verify_selection, verify_folder, preservation, cohort
from run_dsprites_upstream_rescue import load

CODE = ('downstream_route_mechanism.py', 'run_dsprites_downstream_routes.py',
        'test_dsprites_downstream_routes.py', 'audit_dsprites_downstream_routes.py',
        'analyze_dsprites_downstream_routes.py', 'report_dsprites_downstream_routes.py',
        'verify_dsprites_downstream_routes_delivery.py', 'finalize_dsprites_downstream_routes.py')
COUNTS = dict(pairs=74, blocking=9916, point_candidates=23542, channel_localization=4736,
              residual_points=37888, residual_channels=9472, teacher_summary=3216)
MAXERR = 0.0


def close(actual, expected):
    """形状必须相同，避免广播掩盖坐标、轴顺序或缺值错误。"""
    global MAXERR
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.size:
        error = float(abs(a.astype(float)-b.astype(float)).max())
        MAXERR = max(MAXERR, error)
        assert error < 1e-9, error


def check(row, **expected):
    """CSV的缺失值、布尔值、标识符、数值分开验证。"""
    for key, value in expected.items():
        if value is None:
            assert row[key] == '', (key, row[key])
        elif isinstance(value, (bool, np.bool_)):
            assert row[key] == str(bool(value)), (key, row[key], value)
        elif isinstance(value, str):
            assert row[key] == value, (key, row[key], value)
        else:
            close(float(row[key]), value)


def rms(values):
    """空模型不计为零；真实零观测保留。"""
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def instructions(s, layer, point):
    """独立展开已封存名单，不调用生产代码的commands函数。"""
    result = []
    names = ('absolute', 'random0', 'random1', 'switch_only', 'stable_only', 'supporting', 'opposing')
    for i, name in enumerate(names):
        ids = s['orders'][layer, i]; ids = ids[ids >= 0]
        for budget in (1, 4, 16, 64, 256, 1024):
            used = ids[:budget]
            result.append(dict(kind='point', order=name, budget=budget, ids=used, units=len(used)))
    for i, name in enumerate(names[:3]):
        ids = s['channel_orders'][layer, i]; ids = ids[ids >= 0]
        for budget in (1, 2, 4, 8, 16, 32):
            selected = ids[:budget]
            used = np.asarray([1024*int(c)+p for c in selected for p in range(1024)], dtype=int)
            result.append(dict(kind='channel', order=name, budget=budget, ids=used, units=len(selected)))
    active = abs(s['delta_activation'][layer]) > 1e-10
    switched = s['switched'][layer]
    signed = s['point_effect'][layer]*int(s['direction'])
    pools = ([], [point] if layer == 0 else [], np.flatnonzero(active),
             np.flatnonzero(active & switched), np.flatnonzero(active & ~switched),
             np.flatnonzero(active & (signed > 1e-10)), np.flatnonzero(active & (signed < -1e-10)))
    specials = ('empty', 'original_j', 'all_active', 'all_switch', 'all_stable', 'all_supporting', 'all_opposing')
    for name, ids in zip(specials, pools):
        used = np.asarray(ids, dtype=int)
        result.append(dict(kind='special', order=name, budget=len(used), ids=used, units=len(used)))
    assert len(result) == 67
    return result


def indexed(rows, names):
    """检查主键唯一，返回可逐条核验的索引。"""
    result = {tuple(int(r[n]) for n in names): r for r in rows}
    assert len(result) == len(rows)
    return result


def verify(before_manifest=False, deep=False):
    """验证证据链；deep额外重读十三份旧交付的全部条目。"""
    global MAXERR
    MAXERR = 0.0
    manifest_records = []
    if not before_manifest:
        path = RUN/'delivery_manifest.json'
        assert teacher.sha(path) == path.with_suffix('.sha256').read_text().strip()
        manifest_records = teacher.read(path)['files']
        assert len({r['path'] for r in manifest_records}) == len(manifest_records)
        for r in manifest_records:
            path = (BASE/r['path']).resolve()
            assert path.is_relative_to(BASE.resolve())
            assert path.stat().st_size == r['bytes'] and teacher.sha(path) == r['sha256'], r['path']
    protocol = verify_protocol(); freeze = verify_selection()
    assert protocol['cohort'] == cohort()
    assert datetime.fromisoformat(protocol['utc']) < datetime.fromisoformat(freeze['utc'])
    for name in CODE:
        ast.parse((BASE/'scripts'/name).read_text(encoding='utf-8'))
    evidence = {n: teacher.read(RUN/(n+'.json')) for n in
                ('analysis', 'claim_checks', 'report_checks', 'independent_audit', 'visual_review')}
    assert all(v['status'] == 'PASS' for v in evidence.values())
    audit = evidence['independent_audit']; claims = evidence['claim_checks']; report = evidence['report_checks']
    assert [audit[k] for k in ('records', 'all_first_point_scores', 'all_last_point_scores',
                              'sampled_original_single_points', 'original_channel_blocks',
                              'first_joint_fields', 'last_joint_fields', 'old_delivery_entries')] == [37, 2424832, 2424832, 5488, 2368, 19832, 9916, 1711]
    assert max(audit['maximum_errors'].values()) < 1e-9
    assert evidence['analysis']['table_rows'] == COUNTS
    tables = {}
    for name, count in COUNTS.items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f); rows = list(reader)
            assert len(rows) == count and len(reader.fieldnames) == len(set(reader.fieldnames))
            assert all(None not in r for r in rows)
            tables[name] = rows
    pair_index = indexed(tables['pairs'], ('record', 'context'))
    block_index = indexed(tables['blocking'], ('record', 'context', 'layer', 'command'))
    point_index = indexed(tables['point_candidates'], ('record', 'context', 'layer', 'point'))
    channel_index = indexed(tables['channel_localization'], ('record', 'context', 'layer', 'channel'))
    residual_point_index = indexed(tables['residual_points'], ('record', 'context', 'layer', 'command', 'rank'))
    residual_channel_index = indexed(tables['residual_channels'], ('record', 'context', 'layer', 'command', 'channel'))
    panel = load(SHAPE/'panel/data.npz'); previous = {}; top_stable = [0, 0]
    expected_point_keys = set(); tested = defaultdict(int)
    # 原始数组轴检查：第一层额外有残差支路off/on轴，末层没有该轴。
    shapes = dict(native_logits=(2,2,3), route_logits=(2,2,2,3), route_channels=(2,2,2,32),
                  first_logits=(2,67,2,2,3), first_channels=(2,67,2,2,32), first_pn=(2,67,2,2,2),
                  first_second_flips=(2,67,2,2), last_logits=(2,67,2,3), last_channels=(2,67,2,32),
                  last_pn=(2,67,2,2), residual_top_ids=(2,2,2,128), residual_top_values=(2,2,2,128))
    selection_shapes = dict(point_effect=(2,2,32768), channel_effect=(2,2,32), delta_activation=(2,2,32768),
                            switched=(2,2,32768), orders=(2,2,7,1024), order_counts=(2,2,7),
                            channel_orders=(2,2,3,32), channel_counts=(2,2,3), direction=(2,))
    for rec in protocol['cohort']:
        number = rec['record']; seed = rec['seed']
        selection_folder = RUN/f'discovery/record_{number:03d}'; evaluation_folder = RUN/f'evaluation/record_{number:03d}'
        for folder in (selection_folder, evaluation_folder): verify_folder(folder)
        assert datetime.fromisoformat(teacher.read(selection_folder/'completion.json')['utc']) < datetime.fromisoformat(freeze['utc'])
        assert datetime.fromisoformat(teacher.read(evaluation_folder/'completion.json')['utc']) > datetime.fromisoformat(freeze['utc'])
        selection = load(selection_folder/'data.npz'); raw = load(evaluation_folder/'data.npz')
        for arrays, schema in ((selection, selection_shapes), (raw, shapes)):
            assert set(arrays) == set(schema)
            for name, shape in schema.items():
                assert arrays[name].shape == shape and np.isfinite(arrays[name]).all(), name
        assert selection['switched'].dtype == bool
        if seed not in previous: previous[seed] = load(PRIOR/f'evaluation/seed_{seed}/data.npz')
        slot = (tuple(range(25,37))+(1,4,7,10)).index(rec['endpoint'])
        old = previous[seed]['logits'][rec['case'], (3,15).index(rec['mask']), slot, :, 0]
        recipient, donor = panel['pairs'][rec['case']]; weight = np.zeros(3)
        weight[panel['labels'][donor]] = 1.; weight[panel['labels'][recipient]] = -1.
        meta = {k: rec[k] for k in ('record','seed','case','mask','endpoint','family')}
        meta.update(primary='primary26' in rec['roles'], roles=';'.join(rec['roles']))
        for ctx, endpoints in enumerate(((0,1),(2,3))):
            s = {k:v[ctx] for k,v in selection.items()}; v = {k:a[ctx] for k,a in raw.items()}
            close(v['native_logits'], old[list(endpoints)])
            close(v['route_logits'][0,0], v['native_logits'][0]); close(v['route_logits'][1,1], v['native_logits'][1])
            route = v['route_logits']@weight; f0 = float(route[0,0]); f1 = float(route[1,1]); total = f1-f0
            conv = float(route[1,0]-f0); skip = float(route[0,1]-f0)
            joint = float(route[1,1]-route[1,0]-route[0,1]+f0)
            close(total, conv+skip+joint); close(v['route_channels'].sum(-1), route-f0)
            check(pair_index[number,ctx], **meta, context=ctx, C=json.dumps(rec['C']), point=rec['point'],
                  F_off=f0, F_on=f1, C_effect=total, convolution_only=conv, skip_only=skip,
                  branch_interaction=joint, convolution_given_skip=f1-route[0,1], skip_given_convolution=f1-route[1,0],
                  branch_opposition=conv*skip<0 and abs(conv)>1e-8 and abs(skip)>1e-8)
            for layer in range(2):
                cmds = instructions(s,layer,rec['point']); scores = s['point_effect'][layer]
                delta = s['delta_activation'][layer]; active = abs(delta)>1e-10; sw = s['switched'][layer]
                if ctx == 1 and meta['primary']: top_stable[layer] += int(not sw[s['orders'][layer,0,0]])
                candidate = {rec['point']} if layer == 0 else set()
                for order in (0,3,4,5,6): candidate.update(int(p) for p in s['orders'][layer,order,:64] if p>=0)
                for point in candidate:
                    key = (number,ctx,layer,point); expected_point_keys.add(key)
                    c,y,x = np.unravel_index(point,(32,32,32)); ranks = np.flatnonzero(s['orders'][layer,0]==point)
                    check(point_index[key], **meta, context=ctx, layer=layer, point=point, channel=c, y=y, x=x,
                          single_removed=scores[point], activation_change=delta[point], switch=sw[point], active=active[point],
                          absolute_rank=int(ranks[0])+1 if len(ranks) else None, original_j=layer==0 and point==rec['point'])
                for ch in range(32):
                    a = active.reshape(32,1024)[ch]; flip = sw.reshape(32,1024)[ch]; sc = scores.reshape(32,1024)[ch]
                    check(channel_index[number,ctx,layer,ch], **meta, context=ctx, layer=layer, channel=ch,
                          whole_channel_removed=s['channel_effect'][layer,ch], sum_single_points=sc.sum(),
                          active_points=a.sum(), switch_points=(a&flip).sum(), stable_points=(a&~flip).sum(),
                          positive_single_sum=sc[sc>0].sum(), negative_single_sum=sc[sc<0].sum())
                for i, cmd in enumerate(cmds):
                    ids = cmd['ids']; score_sum = float(scores[ids].sum())
                    if layer == 0:
                        field = v['first_logits'][i]@weight
                        remaining = float(field[1,0]-f0); keep = float(field[1,1]-f0)
                        skipoff = float(field[0,0]-f0); keepoff = float(field[0,1]-f0)
                        complement = float(f1-field[1,0]-field[1,1]+route[0,1])
                        complement0 = float(route[1,0]-field[0,0]-field[0,1]+f0)
                        pieces = v['first_channels'][i,1,0]; pn = v['first_pn'][i,1,0]; logits = v['first_logits'][i,1,0]
                        close(v['first_channels'][i].sum(-1),field-f0); close(v['first_pn'][i].sum(-1),field-f0)
                    else:
                        field = v['last_logits'][i]@weight
                        remaining = float(field[0]-f0); keep = float(field[1]-f0)
                        skipoff = keepoff = complement0 = None
                        complement = float(f1-field[0]-field[1]+f0)
                        pieces = v['last_channels'][i,0]; pn = v['last_pn'][i,0]; logits = v['last_logits'][i,0]
                        close(total-remaining,score_sum); close(complement,0.)
                        close(v['last_channels'][i].sum(-1),field-f0); close(v['last_pn'][i].sum(-1),field-f0)
                    assert pn[0]>=-1e-12 and pn[1]<=1e-12
                    den = float(pn[0]-pn[1]); canceled = None if den<=1e-10 else 1-abs(float(pn.sum()))/den
                    check(block_index[number,ctx,layer,i], **meta, context=ctx, layer=layer, command=i,
                          kind=cmd['kind'], order=cmd['order'], budget=cmd['budget'], spent_units=cmd['units'], spatial_points=len(ids),
                          active_points=active[ids].sum(), switch_points=sw[ids].sum(), C_effect=total, removed=total-remaining,
                          remaining=remaining, keep_with_native_skip=keep, remaining_without_skip=skipoff, keep_without_skip=keepoff,
                          skip_only=skip if layer==0 else None, remaining_convolution_given_skip=remaining-skip if layer==0 else None,
                          sum_single_removed=score_sum, joint_minus_singles=total-remaining-score_sum,
                          selected_complement_interaction=complement, selected_complement_without_skip=complement0,
                          residual_positive=pn[0], residual_negative=pn[1], residual_cancellation=canceled,
                          channel_positive=pieces[pieces>0].sum(), channel_negative=pieces[pieces<0].sum(),
                          block_logit0=logits[0], block_logit1=logits[1], block_logit2=logits[2])
                    tested['blocking'] += 1
                    if i in (4,44):
                        for ch,value in enumerate(pieces):
                            check(residual_channel_index[number,ctx,layer,i,ch], **meta, context=ctx,layer=layer,command=i,channel=ch,residual_contribution=value)
                        ri = (4,44).index(i); rids = v['residual_top_ids'][layer,ri]; rvalues = v['residual_top_values'][layer,ri]
                        assert len(set(rids))==128 and np.all(abs(rvalues[:-1])>=abs(rvalues[1:])-1e-12)
                        for rank,(point,value) in enumerate(zip(rids,rvalues),1):
                            c,y,x = np.unravel_index(point,(32,32,32))
                            check(residual_point_index[number,ctx,layer,i,rank], **meta, context=ctx, layer=layer, command=i,
                                  rank=rank, point=point, channel=c, y=y, x=x, residual_contribution=value)
            # 全量阻断、空阻断和只保留全部变化的边界控制。
            close(v['first_logits'][62,:,0],v['route_logits'][0]); close(v['first_logits'][62,:,1],v['route_logits'][1])
            close(v['first_logits'][60,:,0],v['route_logits'][1]); close(v['first_logits'][60,:,1],v['route_logits'][0])
            close(v['last_logits'][62,0],v['native_logits'][0]); close(v['last_logits'][62,1],v['native_logits'][1])
            close(v['last_logits'][60,0],v['native_logits'][1]); close(v['last_logits'][60,1],v['native_logits'][0])
    assert set(point_index)==expected_point_keys and tested['blocking']==9916
    # 聚合使用已逐行核验的全部记录，不能把空教师或缺失skip轴记为零。
    grouped = defaultdict(list)
    for r in tables['blocking']:
        key = (int(r['seed']),int(r['context']),int(r['layer']),int(r['command']),'primary26' if r['primary']=='True' else 'controls')
        grouped[key].append(r)
    summary_keys = set()
    for row in tables['teacher_summary']:
        key = tuple(int(row[n]) for n in ('seed','context','layer','command'))+(row['cohort'],)
        assert key not in summary_keys; summary_keys.add(key)
        group = grouped[key]; denom = rms([float(r['C_effect']) for r in group])
        check(row,count=len(group),C_effect_rms=denom)
        for metric in ('remaining','removed','joint_minus_singles','keep_with_native_skip','remaining_without_skip','keep_without_skip'):
            value = rms([float(r[metric]) for r in group if r[metric]!=''])
            ratio = value/denom if value is not None and denom is not None and denom>1e-8 else None
            check(row,**{metric+'_rms':value,metric+'_rms_ratio':ratio})
    assert len(summary_keys)==3216
    for rec in claims['teacher_equal']:
        ctx,layer,command = [rec[k] for k in ('context','layer','command')]
        summary = [r for r in tables['teacher_summary'] if int(r['context'])==ctx and int(r['layer'])==layer and int(r['command'])==command and r['cohort']=='primary26' and int(r['count'])>0]
        actual = [r for r in tables['blocking'] if int(r['context'])==ctx and int(r['layer'])==layer and int(r['command'])==command and r['primary']=='True']
        assert rec['teachers']==len(summary)==5 and rec['records']==sum(int(r['count']) for r in summary)==26
        assert rec['kind']==actual[0]['kind'] and rec['order']==actual[0]['order']
        assert rec['budget']==(None if rec['kind']=='special' else int(actual[0]['budget']))
        assert rec['spent_min']==min(int(r['spent_units']) for r in actual) and rec['spent_max']==max(int(r['spent_units']) for r in actual)
        for name in summary[0]:
            if name.endswith('_rms') or name.endswith('_ratio'):
                values = [float(r[name]) for r in summary if r[name]!='']
                if values: close(rec[name],np.mean(values))
                else: assert rec[name] is None
    primary = [r for r in tables['pairs'] if r['primary']=='True' and r['context']=='1']
    assert claims['records']==37 and claims['primary_records']==len(primary)==26
    assert claims['primary_states']==len({(r['seed'],r['case'],r['mask']) for r in primary})==15
    assert claims['primary_teachers']==len({r['seed'] for r in primary})==5
    assert claims['new_context_branch_opposition']==sum(r['branch_opposition']=='True' for r in primary)==2
    assert claims['first_all_block_nonzero_skip']==sum(abs(float(r['skip_only']))>1e-8 for r in primary)==26
    assert claims['first_point256_joint_nonlinear']==sum(abs(float(block_index[int(r['record']),1,0,4]['joint_minus_singles']))>1e-8 for r in primary)==26
    decomp = [r for r in tables['pairs'] if 'decomposition_control' in r['roles']]
    assert len(decomp)==8
    close(claims['decomp_max_abs_effect'],max(abs(float(r['C_effect'])) for r in decomp)); close(claims['decomp_max_abs_effect'],0.)
    assert report['new_context_top_point_stable_counts']==top_stable==[19,26]
    assert report['primary26']==26 and report['example_record']==5
    # 固定例JSON只是完整表格的视图；逐项防止手填数字漂移。
    example = teacher.read(RUN/'examples.json'); assert example['record']==protocol['cohort'][5]
    lookup = dict(pairs=(pair_index,('record','context')),blocks=(block_index,('record','context','layer','command')),
                  points=(point_index,('record','context','layer','point')),channels=(channel_index,('record','context','layer','channel')),
                  residual_points=(residual_point_index,('record','context','layer','command','rank')))
    for name,(index,names) in lookup.items():
        for row in example[name]: check(index[tuple(row[n] for n in names)],**row)
    visual = evidence['visual_review']; assert visual['inspected_pngs']==4 and len(report['figures'])==8
    assert set(visual['files'])==set(report['figures'])
    for name,digest in visual['files'].items(): assert teacher.sha(BASE/name)==digest
    old = preservation() if deep else protocol['preservation']
    assert old==protocol['preservation'] and len(old)==13 and sum(r['files'] for r in old.values())==1711
    result = dict(status='PASS',utc=teacher.now(),records=37,table_rows=COUNTS,total_rows=sum(COUNTS.values()),
                  maximum_table_error=MAXERR,top_point_stable_counts=top_stable,old_delivery_entries=1711,
                  deep_inputs=deep,manifest_files=len(manifest_records),readonly=not before_manifest)
    if before_manifest:
        assert not (RUN/'delivery_manifest.json').exists()
        teacher.dump(RUN/'table_verification.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-manifest',action='store_true'); parser.add_argument('--deep-inputs',action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.before_manifest,args.deep_inputs)),flush=True)

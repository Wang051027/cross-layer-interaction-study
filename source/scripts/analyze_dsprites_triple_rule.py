"""按冻结判据汇总留出预测，保留逐案例误差、阻断方向和失败案例。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import dsprites_teacher as teacher
from dsprites_triple_rule import METHODS, PATCHES, TRIPLES
from run_dsprites_triple_rule import (DEFAULT,STRENGTHS,stage_path,panel_arrays,
                                      verify_protocol,verify_seal,atomic_json)

TASKS = ('middle_third_all','output_third','block_removed','keep_recovered')
PRIMARY_METHOD = ('rule','rule','chosen4','chosen4')


def csv_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as out:
        writer = csv.DictWriter(out,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rms(array):
    return float(np.sqrt(np.mean(np.square(array))))


def score(seed, strength, task, method, truth, estimate, signs=False):
    scale = rms(truth)
    error = rms(estimate-truth)
    eligible = abs(truth)>1e-9 if signs else np.zeros(0,dtype=bool)
    hits = int((np.sign(estimate[eligible])==np.sign(truth[eligible])).sum()) if signs else 0
    return dict(seed=seed,strength=strength,target=task,method=method,count=truth.size,
                truth_rms=scale,error_rms=error,ratio=error/scale if scale>1e-12 else None,
                sign_eligible=int(eligible.sum()),sign_hits=hits)


def read_npz(path):
    with np.load(path,allow_pickle=False) as source:
        return {k:source[k] for k in source.files}


def evaluate_teacher(run,panel,seed,strength,rows,patch_rows,group_rows,scores):
    pred = read_npz(stage_path(run,'predictions',seed,strength))
    data = read_npz(stage_path(run,'measurements',seed,strength))
    fit = teacher.read(run/'fit'/f'{seed}.json')
    middle = fit['middle']
    cases = pred['case']
    assert np.array_equal(cases,data['case'])
    pairs = panel['pairs'][cases]
    labels = panel['labels'][pairs]
    weight = np.eye(3)[labels[:,1]]-np.eye(3)[labels[:,0]]
    # 自然三阶差分也从原八个角直接计算，检查与预测文件中的Q相减一致。
    cube = np.array([sorted((i for i in range(16) if i&mask==i),key=lambda i:(i.bit_count(),i))
                     for mask in TRIPLES])
    contrast = np.einsum('nsc,nc->ns',data['natural_logits'],weight)
    stencil = np.array([-1,1,1,1,-1,-1,-1,1])
    truth = contrast[:,cube] @ stencil
    total = contrast[:,TRIPLES]-contrast[:,[0]]
    quad = np.einsum('ntc,nc->nt',pred['qlogits'],weight)
    np.testing.assert_allclose(truth,contrast[:,TRIPLES]-quad,atol=1e-9,rtol=0)
    estimated = np.einsum('ntmc,nc->ntm',pred['output_corrections'],weight)
    truth_logits = data['natural_logits'][:,TRIPLES]-pred['qlogits']
    true_h = data['true_middle']
    delta_h = true_h-pred['h0'][:,None]
    gamma_h = true_h-pred['qh']
    gamma_norm = np.square(gamma_h).sum((2,3,4))
    site_errors = {}
    for method_id,method in enumerate(METHODS):
        scalars = estimated[:,:,method_id]
        scores.append(score(seed,strength,'output_third',method,truth,scalars,True))
        scores.append(score(seed,strength,'output_total',method,total,total-truth+scalars))
        scores.append(score(seed,strength,'all_logits_third',method,truth_logits,pred['output_corrections'][:,:,method_id]))
        if method=='class_mean':
            continue
        correction = np.zeros_like(gamma_h)
        if method!='quadratic':
            correction[:,:,middle] = pred['middle_corrections'][:,:,method_id-1]
        scores.append(score(seed,strength,'middle_third_all',method,gamma_h,correction))
        scores.append(score(seed,strength,'middle_third_selected',method,gamma_h[:,:,middle],correction[:,:,middle]))
        scores.append(score(seed,strength,'middle_total',method,delta_h,delta_h-gamma_h+correction))
        site_errors[method] = np.square(gamma_h-correction).sum((2,3,4))
    all_patch = np.einsum('ntpoic,i,nc->ntpo',data['patch_logits'],stencil,weight)
    removed = truth[:,:,None] - all_patch[:,:,:,0]
    kept = all_patch[:,:,:,1]
    channel_estimate = np.einsum('ntjc,nc->ntj',pred['channel_logits'],weight)
    ids = pred['patch_ids']
    predicted_patch = np.zeros(removed.shape)
    for pos,ch in enumerate(middle):
        present = (ids==ch).any(-1)
        predicted_patch += present*channel_estimate[:,:,pos,None]
    np.testing.assert_allclose(predicted_patch[:,:,0],estimated[:,:,1],rtol=0,atol=1e-10)
    for pi,patch in enumerate(PATCHES):
        scores.append(score(seed,strength,'block_removed',patch,removed[:,:,pi],predicted_patch[:,:,pi],True))
        scores.append(score(seed,strength,'keep_recovered',patch,kept[:,:,pi],predicted_patch[:,:,pi],True))
    strong = abs(estimated[:,:,1]) > fit['strong_threshold']
    for name,subset in (('strong',strong),('weak',~strong)):
        actual = removed[:,:,0][subset]
        forecast = predicted_patch[:,:,0][subset]
        group_rows.append(dict(seed=seed,strength=strength,subset=name,count=actual.size,
                               development_threshold=fit['strong_threshold'],
                               removed_rms=rms(actual) if actual.size else None,
                               forecast_rms=rms(forecast) if actual.size else None,
                               error_rms=rms(forecast-actual) if actual.size else None))
    for case_id,case in enumerate(cases):
        r,d = pairs[case_id]
        size,angle = panel['factors'][r,2:4]
        for t,mask in enumerate(TRIPLES):
            common = dict(seed=seed,strength=strength,case=int(case),size=int(size),angle=int(angle),
                          recipient=int(r),donor=int(d),mask=mask)
            row = dict(**common, F_empty=contrast[case_id,0],F_quadratic=quad[case_id,t],
                       F_true=contrast[case_id,mask],third_true=truth[case_id,t],
                       predicted_stratum='strong' if strong[case_id,t] else 'weak',
                       middle_third_squared_norm=gamma_norm[case_id,t])
            for m,method in enumerate(METHODS):
                row['third_'+method] = estimated[case_id,t,m]
                if method in site_errors:
                    row['middle_error_sq_'+method] = site_errors[method][case_id,t]
            rows.append(row)
            for p,name in enumerate(PATCHES):
                chosen = ids[case_id,t,p]
                patch_rows.append(dict(**common,patch=name,channels=','.join(map(str,chosen[chosen>=0])),
                                       predicted=predicted_patch[case_id,t,p],removed=removed[case_id,t,p],
                                       actual_after_block=all_patch[case_id,t,p,0],keep=kept[case_id,t,p]))


def analyze(run):
    verify_protocol(run)
    for name,prev in (('fit_freeze','protocol'),('prediction_freeze','fit_freeze'),
                      ('measurement_freeze','prediction_freeze')):
        verify_seal(run,name,prev)
    panel = panel_arrays()
    rows,patches,groups,scores = [],[],[],[]
    for seed in teacher.SEEDS:
        for strength in STRENGTHS:
            evaluate_teacher(run,panel,seed,strength,rows,patches,groups,scores)
    assert len(rows)==864 and len(patches)==4320 and len(groups)==24
    assert len({(r['seed'],r['strength'],r['case'],r['mask']) for r in rows})==len(rows)
    for name,table in (('cases',rows),('patches',patches),('teacher_scores',scores),('strata',groups)):
        csv_table(run/'tables'/f'{name}.csv',table)
    aggregate = []
    for strength,target,method in sorted({(r['strength'],r['target'],r['method']) for r in scores}):
        group = [r for r in scores if (r['strength'],r['target'],r['method'])==(strength,target,method)]
        assert len(group)==6
        ratios = [r['ratio'] for r in group if r['ratio'] is not None]
        hit = sum(r['sign_hits'] for r in group)
        count = sum(r['sign_eligible'] for r in group)
        aggregate.append(dict(strength=strength,target=target,method=method,teachers=len(ratios),
                              ratio=float(np.mean(ratios)) if ratios else None,
                              improved=sum(v < 1 for v in ratios),
                              teacher_mean_truth_rms=float(np.mean([r['truth_rms'] for r in group])),
                              teacher_mean_error_rms=float(np.mean([r['error_rms'] for r in group])),
                              sign_hits=hit,sign_eligible=count))
    def result(strength,target,method):
        return next(r for r in aggregate if (r['strength'],r['target'],r['method'])==(strength,target,method))
    decisions = {}
    for task,method in zip(TASKS,PRIMARY_METHOD):
        r=result(1.,task,method)
        decisions[task]=r['teachers']==6 and r['ratio']<=0.95 and r['improved']>=4
    sample = rows[0]
    selected = next(r for r in patches if r['seed']==sample['seed'] and r['strength']==sample['strength']
                    and r['case']==sample['case'] and r['mask']==sample['mask'] and r['patch']=='chosen4')
    report = dict(status='PASS',utc=teacher.now(),decisions=decisions,all_links_pass=all(decisions.values()),
                  table_rows=dict(cases=len(rows),patches=len(patches),teacher_scores=len(scores),strata=len(groups)),
                  aggregates=aggregate,first_example={k:float(v) for k,v in sample.items() if isinstance(v,(int,float,np.number))},
                  example_block={k:float(selected[k]) for k in ('predicted','removed','keep')})
    atomic_json(run/'analysis.json',report)
    csv_table(run/'tables/teacher_equal.csv',aggregate)
    print('Primary:', [(task,round(result(1.,task,m)['ratio'],5),result(1.,task,m)['improved']) for task,m in zip(TASKS,PRIMARY_METHOD)],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    args=parser.parse_args()
    assert not (args.run_dir/'delivery_manifest.json').exists()
    analyze(args.run_dir)

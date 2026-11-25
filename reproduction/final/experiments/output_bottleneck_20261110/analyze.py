"""从已封存逐例结果计算模型等权指标；各模型分别归一化，避免把配对当独立样本。"""
import argparse
from pathlib import Path
import numpy as np
from run_experiment import case_key, js, put, read_csv, sha, write_csv

BUDGETS=(0,4,8,16,32)
PARTS=('error_est_C','error_omit_O','error_tail_C','error_tail_O','error_mixed')
PAIRS={'coverage_fixed':('fixed_4','fixed_32'),'readout_at4':('fixed_4','live_4'),
       'readout_at32':('fixed_32','live_32'),'coverage_live':('live_4','live_32')}


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def divide(a,b):
    if b<=1e-12:
        raise ValueError('未预期的近零分母；必须单独报告')
    return float(a/b)


def main(run):
    predseal=js(run/'prediction_seal.json'); mes=js(run/'measurement_seal.json')
    assert predseal['predictions']==sha(run/'predictions.csv') and predseal['protocol']==sha(run/'protocol.json')
    assert mes['prediction_seal']==sha(run/'prediction_seal.json')
    assert mes['measurements']==sha(run/'measurements.csv') and mes['audit']==sha(run/'audit.json')
    pred={case_key(r):r for r in read_csv(run/'predictions.csv')}
    truth={case_key(r):r for r in read_csv(run/'measurements.csv')}
    assert len(truth)==864 and set(truth)==set(pred)
    allrows={k:{**pred[k],**r} for k,r in truth.items()}
    seeds=sorted({k[0] for k in allrows}); assert len(seeds)==6
    teacher=[];summary={}; contrasts={}
    for alpha in (1.0,0.5):
        bymodel=[]
        for seed in seeds:
            rows=[allrows[k] for k in sorted(allrows) if k[:2]==(seed,alpha)]
            assert len(rows)==72
            def a(col):return np.array([float(r[col]) for r in rows])
            y=a('y');b=a('removed');k=a('keep');g=a('gap');base=a('slope_empty');dynamic=a('slope_outside')
            def error(pred,target=y): return divide(rms(pred-target),rms(target))
            e=y-a('fixed_4')
            data={'output_rms':rms(y),'old_error_rms':rms(e),'gap_rms':rms(g),
                  'removed_rms':rms(b),'keep_rms':rms(k),'class_mean':error(a('class_mean')),
                  'opposite_sign_count':int(np.sum(b*k<0)),
                  'slope_empty_removed_error':error(base,b),'slope_outside_removed_error':error(dynamic,b),
                  'slope_empty_keep_error':error(base,k),'slope_gap_error':error(dynamic-base,g),
                  'slope_shift_rms':rms(dynamic-base),
                  'flip_first':float(a('flip1_fraction').mean()),'flip_second':float(a('flip2_fraction').mean())}
            for z in BUDGETS:
                for typ in ('fixed','live','clip','oracle_fixed','oracle_live'):
                    name=f'{typ}_{z}';data[name]=error(a(name))
                data[f'middle_{z}']=float(np.sqrt(a(f'middle_error_sq_{z}').sum()/a('true_norm_sq').sum()))
                data[f'negative_{z}']=float(a(f'negative_{z}').mean())
            for name in PARTS:
                v=a(name)
                data[name+'_rms']=rms(v)
                data[name+'_proj']=divide(float(np.mean(e*v)),float(np.mean(e*e)))
            assert abs(sum(data[name+'_proj'] for name in PARTS)-1)<1e-9
            for key in ('first','second','both'):
                h=a('gap_lock_'+key)
                data['gap_left_'+key]=divide(rms(h),rms(g))
                data['gap_changed_'+key]=divide(rms(g-h),rms(g))
                data['gap_removed_proj_'+key]=divide(float(np.mean(g*(g-h))),float(np.mean(g*g)))
            bymodel.append(data)
            teacher.append(dict(seed=seed,strength=alpha,n=72,**data))
        summary[str(alpha)]={name:dict(mean=float(np.mean([v[name] for v in bymodel])),
            min=float(min(v[name] for v in bymodel)),max=float(max(v[name] for v in bymodel)),
            below_one=int(sum(v[name]<1 for v in bymodel))) for name in bymodel[0]}
        comps={}
        for k,(old,new) in PAIRS.items():
            deltas=[r[old]-r[new] for r in bymodel]
            delta=float(np.mean(deltas)); wins=sum(v>0 for v in deltas)
            comps[k]=dict(before=old,after=new,improvement=delta,improved_models=wins,
                          material=bool(delta>=.05 and wins>=4),per_model=deltas)
        left=[v['gap_left_both'] for v in bymodel]
        comps['gate_transplant']=dict(remaining_ratio=float(np.mean(left)),models_with_reduced_gap=sum(v<1 for v in left),
            passes=bool(np.mean(left)<=.95 and sum(v<1 for v in left)>=4))
        contrasts[str(alpha)]=comps
    info=dict(aggregation='RMS within each teacher, then equal weight over six. Full strength primary; half secondary.',
        per_strength_records=432,summary=summary,contrasts=contrasts,
        first_planned_case=allrows[(seeds[0],1.0,min(js(run/'protocol.json')['split']['test']),7)])
    put(run/'summary.json',info)
    write_csv(run/'teacher_scores.csv',teacher)
    print('Scored 864 records in 12 model-strength cells.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    main(ap.parse_args().run)

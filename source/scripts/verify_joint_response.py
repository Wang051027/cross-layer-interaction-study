"""Check saved local coefficients, aggregate scores and projections with an explicit polynomial and parity basis."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv, gzip, hashlib, itertools, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/joint_response_20260925'
TOL=1e-9
errors={}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
# 按绝对容差比较两个实现的结果，同时要求缺失值和有限值的位置一致。
def same(name, actual, expected):
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    assert np.array_equal(np.isnan(a),np.isnan(b)),name
    valid=np.isfinite(a)&np.isfinite(b)
    assert np.array_equal(np.isfinite(a),np.isfinite(b)),name
    error=float(np.max(np.abs(a[valid]-b[valid]))) if np.any(valid) else 0.
    errors[name]=max(errors.get(name,0.),error)
    assert error<TOL,(name,error)
# 把 JSON 的 null 转回 NaN，便于用统一数值规则复核。
def val(value): return np.nan if value is None else value
# 从记录中提取指定字段为数组，用于独立重算汇总。
def values(rows,key): return np.array([val(r[key]) for r in rows])

# 逐设置检查保存系数，以直接矩阵法复算抽定输入，并独立重算全部教师/组汇总。
def run():
    protocol=read(RUN/'protocol.json');summary=read(RUN/'summary.json')
    assert sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for rel,digest in protocol['input_manifest'].items(): assert sha(BASE/rel)==digest,rel
    for rel,digest in protocol['sources'].items():
        assert sha(BASE/rel)==digest and sha(RUN/'source'/rel)==digest,rel
    raw=gzip.decompress((BASE/'data/fashion_mnist/raw/t10k-labels-idx1-ubyte.gz').read_bytes())
    test_labels=np.frombuffer(raw,dtype=np.uint8,offset=8)
    train_labels=np.load(BASE/'data/fashion_mnist/prepared/official_train_labels.npy',allow_pickle=False)
    records=[];contexts_checked=0;query_vectors_checked=0
    for case in protocol['cases']:
        seed,pos,q=case['seed'],case['position'],case['q']
        old=BASE/'runs/fashion_switch_20260914'/case['phase']/f'seed_{seed}'/f'{pos}_q{q}'
        new=RUN/'cases'/f'seed_{seed}'/f'{pos}_q{q}'
        done=read(new/'completion.json')
        assert sha(new/'coefficients.npz')==done['coefficients_sha256']
        assert sha(new/'per_input.json')==done['rows_sha256']
        rs=read(new/'per_input.json');records.extend(rs)
        lookup={(r['alpha'],r['input_id']):r for r in rs}
        pairs=list(itertools.combinations(range(q),2))
        masks=np.arange(1<<q,dtype=np.int64)
        bits=np.array([[(int(m)>>i)&1 for i in range(q)] for m in masks],dtype=int)
        orders=bits.sum(axis=1);held=orders>2;low_ids=masks[orders<=2]
        # Direct definition of Walsh characters, independent of the fast transform.
        # 直接按奇偶性构造 Walsh 基矩阵，避免再次使用被核验的快速变换实现。
        basis=1.-2.*((bits@bits[low_ids].T)%2)
        with np.load(old/'references.npz',allow_pickle=False) as ref, np.load(new/'coefficients.npz',allow_pickle=False) as coef:
            ids=ref['ids'];labels=(train_labels if q==12 else test_labels)[ids]
            assert np.array_equal(coef['ids'],ids) and np.array_equal(coef['labels'],labels)
            assert np.array_equal(coef['low2_mask_ids'],low_ids)
            chosen=list(range(len(ids))) if q==12 else [0,len(ids)//2,len(ids)-1]
            for alpha in (1.,.5):
                y=ref[str(alpha)];prefix=f'a{alpha}_'
                same('baseline',coef[prefix+'baseline'],y[:,0])
                same('first_coefficients',coef[prefix+'first'],y[:,1<<np.arange(q)]-y[:,0,None])
                direct_second=np.stack([y[:,(1<<i)|(1<<j)]-y[:,1<<i]-y[:,1<<j]+y[:,0] for i,j in pairs],axis=1)
                same('second_coefficients',coef[prefix+'second'],direct_second)
                query_vectors_checked+=len(ids)*len(low_ids)
                for index in chosen:
                    target_weights=np.full(10,-1/9);target_weights[int(labels[index])]=1.
                    target=y[index]@target_weights
                    one=np.broadcast_to(coef[prefix+'baseline'][index],(1<<q,10)).copy()
                    for i in range(q): one+=bits[:,i,None]*coef[prefix+'first'][index,i]
                    two=one.copy()
                    for p,(i,j) in enumerate(pairs): two+=(bits[:,i]*bits[:,j])[:,None]*coef[prefix+'second'][index,p]
                    row=lookup[alpha,int(ids[index])]
                    for name,pred in [('single',one),('pair',two)]:
                        for tn,prediction,truth in [('vector',pred,y[index]),('scalar',pred@target_weights,target)]:
                            delta=(prediction-truth)[held];truth=truth[held]
                            rms=np.sqrt(np.mean(delta**2));effect=np.sqrt(np.mean(truth**2))
                            stats={'rmse':rms,'effect_rms':effect,'nrmse':rms/effect if effect>TOL else np.nan,
                                   'mae':np.abs(delta).mean(),'p95_abs':np.quantile(np.abs(delta),.95),'max_abs':np.abs(delta).max()}
                            for metric,x in stats.items(): same('explicit_prediction_'+metric,x,val(row[f'{name}_{tn}_{metric}']))
                    # 矩阵乘法直接求正交系数，与快速变换保存值交叉比较。
                    direct_coeff=basis.T@y[index]/len(masks)
                    same('direct_parity_coefficients',direct_coeff,coef[prefix+'walsh_low2'][index])
                    for degree in (1,2):
                        take=orders[low_ids]<=degree
                        # 重建最佳低阶函数，由原始响应计算残差。
                        projected=basis[:,take]@direct_coeff[take]
                        residual=y[index]-projected
                        for tn,err,truth in [('vector',residual,y[index]),('scalar',residual@target_weights,target)]:
                            rmse=np.sqrt(np.mean(err**2));effect=np.sqrt(np.mean(truth**2))
                            same('direct_projection_rmse',rmse,row[f'{tn}_oracle_degree{degree}_rmse'])
                            same('direct_projection_nrmse',rmse/effect if effect>TOL else np.nan,val(row[f'{tn}_oracle_degree{degree}_nrmse']))
                            if degree==2:
                                variance=np.mean((truth-truth.mean(axis=0))**2)
                                same('direct_high_order_variance',rmse**2/variance if variance>TOL*TOL else np.nan,val(row[f'{tn}_high_order_variance_fraction']))
                    contexts_checked+=1
    assert len(records)==6240 and len(summary['teachers'])==72 and len(summary['groups'])==12
    for record in summary['teachers']:
        rs=[r for r in records if all(r[k]==record[k] for k in ('seed','position','q','alpha'))]
        assert len(rs)==record['inputs']
        expected={
            'candidate_states_mean':values(rs,'candidate_states').mean(),
            'no_candidate_inputs':(values(rs,'candidate_states')==0).sum(),
            'crossing_states_mean':values(rs,'strict_crossing_states').mean(),
            'reconstruction_max':values(rs,'reconstruction_max_abs').max(),
            'pair_better_than_single_inputs':(values(rs,'pair_scalar_rmse')<values(rs,'single_scalar_rmse')-TOL).sum()}
        for tn in ('scalar','vector'):
            for method in ('single','pair'):
                key=f'{method}_{tn}';n=values(rs,key+'_nrmse');valid=n[np.isfinite(n)]
                effect=values(rs,key+'_effect_rms');rms=values(rs,key+'_rmse')
                expected[key+'_pooled_nrmse']=np.linalg.norm(rms)/np.linalg.norm(effect) if np.mean(effect**2)>TOL*TOL else np.nan
                expected[key+'_input_median']=np.median(valid) if len(valid) else np.nan
                expected[key+'_input_p95']=np.quantile(valid,.95) if len(valid) else np.nan
                expected[key+'_fraction_over10']=np.mean(valid>.1) if len(valid) else np.nan
                expected[key+'_zero_effect_inputs']=len(n)-len(valid)
                expected[key+'_max_abs']=values(rs,key+'_max_abs').max()
            effect=values(rs,tn+'_full_effect_rms')
            for degree in (1,2):
                rms=values(rs,f'{tn}_oracle_degree{degree}_rmse')
                expected[f'{tn}_oracle_degree{degree}_pooled_nrmse']=np.linalg.norm(rms)/np.linalg.norm(effect) if np.mean(effect**2)>TOL*TOL else np.nan
            variation=values(rs,tn+'_full_variation_rms')
            expected[tn+'_high_order_variance_fraction']=np.sum(rms**2)/np.sum(variation**2) if np.mean(variation**2)>TOL*TOL else np.nan
        for prefix,x in [('candidate','candidate_states'),('bound','target_affine_remainder_bound')]:
            for name,y in [('pair_rmse','pair_scalar_rmse'),('highorder_rmse','scalar_oracle_degree2_rmse')]:
                a,b=values(rs,x),values(rs,y)
                expected[f'{prefix}_vs_{name}_rho']=spearmanr(a,b).statistic if len(rs)>=10 and len(np.unique(a))>1 and len(np.unique(b))>1 else np.nan
        for key,x in expected.items(): same('teacher_aggregation_'+key,x,val(record[key]))
    excludes={'seed','position','q','alpha','inputs','scored_masks','phase'}
    for group in summary['groups']:
        ts=[t for t in summary['teachers'] if all(t[k]==group[k] for k in ('position','q','alpha'))]
        assert len(ts)==6
        for key in ts[0]:
            if key in excludes:continue
            a=values(ts,key);a=a[np.isfinite(a)]
            same('group_means',a.mean() if len(a) else np.nan,val(group[key+'_mean']))
            same('group_sd',a.std(ddof=1) if len(a)>1 else np.nan,val(group[key+'_sd']))
        assert sum(t['pair_scalar_pooled_nrmse']<t['single_scalar_pooled_nrmse'] for t in ts)==group['pair_better_than_single_teachers']
    for filename,json_rows in [('per_input_metrics.csv',records),('teacher_summary.csv',summary['teachers']),('group_summary.csv',summary['groups'])]:
        with (RUN/filename).open(encoding='utf-8-sig',newline='') as stream: csv_rows=list(csv.DictReader(stream))
        assert len(csv_rows)==len(json_rows)
        for a,b in zip(csv_rows,json_rows):
            assert set(a)==set(b)
            for key,item in b.items():
                if item is None: assert a[key]==''
                elif isinstance(item,str): assert a[key]==item
                else:same('csv_roundtrip',float(a[key]),item)
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),
        independent_implementation='Explicit mask polynomial and direct parity matrix; analysis modules not imported',
        input_files_hash_checked=len(protocol['input_manifest']),saved_cases_hash_checked=len(protocol['cases']),
        origin_query_vectors_checked=query_vectors_checked,selected_input_strength_games_checked=contexts_checked,
        selection='All four inputs for q12; indices 0, n//2, n-1 for q4/q8; both strengths in all 36 cases',
        teacher_rows_checked=72,group_rows_checked=12,csv_metric_rows_checked=6240,
        tolerances_absolute=TOL,maximum_difference=max(errors.values()),differences=errors,
        implementation_sha256=sha(__file__),summary_sha256=sha(RUN/'summary.json'))
    output=RUN/'independent_verification.json'
    if output.exists():raise FileExistsError('Preserve existing verification')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='differences'},ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()

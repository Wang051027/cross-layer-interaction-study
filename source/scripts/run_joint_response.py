# 中文阅读说明｜用途：复用已保存完整响应，分析低阶查询的外推能力与真正高阶残差。
# 数据流：遍历六教师、两位置、三规模、两强度；q=12 用四张既有验证输入，与主分析样本不同。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Protocol-locked descriptive reanalysis of saved coalition responses."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, csv, gzip, hashlib, json, platform, shutil, time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import scipy
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from joint_response import (cube, origin_models, walsh_analysis, fwht, contrast,
                            per_input_metrics, independent_state_response)
from test_joint_response import smoke

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/joint_response_20260925'
OLD=BASE/'runs/fashion_switch_20260914'
SEEDS=list(range(202609071,202609077))
POSITIONS=('center','upper_left')
SIZES=(4,8,12)
ALPHAS=(1.,.5)
PROTOCOL_MD='reports/联合干预响应_分析协议_2026-09-25.md'
CODE=['scripts/joint_response.py','scripts/test_joint_response.py','scripts/run_joint_response.py']
TOL=1e-9


# 用途：生成当前 UTC 时间字符串，供协议、日志或交付记录标记时间。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：datetime.now(timezone.utc).isoformat()。
def now(): return datetime.now(timezone.utc).isoformat()
# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：h.hexdigest()。
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(Path(path).read_text(encoding='utf-8'))。
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
# 用途：递归把 NumPy 数值转为可序列化对象，并将不可定义的非有限数记为 null。
# 参数：value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return） / [clean(v) for v in value] / int(value)；另有分支返回，见函数体。
def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)): return float(value) if np.isfinite(value) else None
    return value
# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path,value): Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
# 用途：按记录字段写 CSV；不同实验阶段各自规定缺失数值的写法。
# 参数：path、rows；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：None。
def csv_write(path,rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows(clean(rows))
# 用途：根据教师、位置和规模定位历史响应目录，q=12 指向补充全枚举记录。
# 参数：seed、position、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：OLD / ('supplementary' if q == 12 else 'test') / f'seed_{seed}' / f'{position}_q{q}'。
def source_case(seed,position,q):
    return OLD/('supplementary' if q==12 else 'test')/f'seed_{seed}'/f'{position}_q{q}'


# 用途：建立新的实验目录并保存数据、模型配置及源码的可核对协议。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def initialize():
    control=smoke()
    if RUN.exists(): raise FileExistsError('Preserve the existing run; do not overwrite it.')
    # 源输入既与本轮协议核对，也与之前交付清单核对，避免替换历史答案。
    old_manifest=read(BASE/'audit/switch_manifest_20260917.json')
    known={r['path']:r['sha256'] for r in old_manifest['entries']}
    old_protocol=read(OLD/'protocol.json')
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert sha(OLD/'protocol.json')==(OLD/'protocol.sha256').read_text().strip()
    files={}; cases=[]
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        for pos in POSITIONS:
            for q in SIZES:
                folder=source_case(seed,pos,q)
                for filename in ('references.npz','screening_contexts.npz','reference_commitment.json','prediction_commitment.json'):
                    path=folder/filename; rel=path.relative_to(BASE).as_posix()
                    digest=sha(path)
                    assert digest==known[rel],f'Historical manifest mismatch: {rel}'
                    files[rel]=digest
                assert sha(folder/'references.npz')==read(folder/'reference_commitment.json')['sha256']
                assert sha(folder/'screening_contexts.npz')==read(folder/'prediction_commitment.json')['screening_sha256']
                cases.append(dict(seed=seed,position=pos,q=q,phase='supplementary' if q==12 else 'test',
                                  inputs=4 if q==12 else 128,coalitions=1<<q))
    for rel in ('runs/fashion_scale_20260914/design_q4.npz','runs/fashion_scale_20260914/design_q8.npz',
                'data/fashion_mnist/prepared/official_train_labels.npy','data/fashion_mnist/raw/t10k-labels-idx1-ubyte.gz'):
        digest=sha(BASE/rel)
        if rel in known: assert digest==known[rel]
        if rel in old_protocol['data_hashes']: assert digest==old_protocol['data_hashes'][rel]
        files[rel]=digest
    # 先固定输入与代码，再产生新指标；历史数据已用过，所以不是独立预注册。
    RUN.mkdir()
    (RUN/'source').mkdir()
    for rel in CODE+[PROTOCOL_MD]:
        target=RUN/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/rel,target)
    protocol=dict(utc=now(),version='joint-response-descriptive-v1',
        purpose='Low-order query extrapolation versus best full-cube low-degree approximation',
        scope='Saved last-residual-branch local scalar activation games; no new network forward, no new training',
        prior_knowledge='All source responses were generated and inspected in earlier research; this analysis is post hoc, not an independent preregistration',
        input_manifest=files,input_manifest_matches_historical_delivery=True,
        sources={rel:sha(BASE/rel) for rel in CODE+[PROTOCOL_MD]},cases=cases,
        strengths=ALPHAS,mask_convention='1 scales source coordinate by (1-alpha); empty mask is unchanged network',
        target='true class logit change minus mean of other nine class logit changes, fixed across masks',
        prediction_models=['empty-and-singletons additive','empty-singletons-pairs multilinear quadratic'],
        prediction_scored_masks='popcount>2: 5/219/4017; no fitting on these values',
        descriptive_models='Uniform-cube Walsh degree<=1/2 projection using all labels; not held-out predictors',
        high_order_definition='degree>=3 in a multilinear Boolean game; report total-energy and centred-variance denominators separately',
        primary_slice=dict(q=8,alpha=1.,positions=POSITIONS),numeric_tolerance=TOL,effect_rms_floor=TOL,
        diagnostic_nrmse_line=.1,diagnostic_line_is_not_universal_application_standard=True,
        statistics='Teacher-first descriptive means/SD, fractions and correlations; no new confirmatory p-values',
        calibration_model_access=False,new_teacher_training_runs=0,new_network_forward_runs=0,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=1),
        references=['https://proceedings.iclr.cc/paper_files/paper/2025/hash/59b9582cd35f555ea8415030073e7b22-Abstract-Conference.html',
                    'https://proceedings.iclr.cc/paper_files/paper/2026/hash/3b36297debce91932600be70800bc094-Abstract-Conference.html'])
    dump(RUN/'protocol.json',protocol)
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n',encoding='ascii')
    dump(RUN/'synthetic_controls.json',control)
    print(json.dumps(dict(status='locked',cases=len(cases),input_files=len(files),protocol_sha256=sha(RUN/'protocol.json'))),flush=True)


# 用途：检查已保存的协议、输入、源码或冻结记录；验证失败就停止。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：p。
def verify():
    p=read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for rel,expected in p['sources'].items():
        assert sha(BASE/rel)==expected and sha(RUN/'source'/rel)==expected,rel
    for rel,expected in p['input_manifest'].items(): assert sha(BASE/rel)==expected,rel
    return p


# 用途：按保存的输入索引取真实标签；q=12 来自既有验证池，其余来自官方测试标签。
# 参数：ids、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：values[ids].astype(int)。
def labels_for(ids,q):
    if q==12:
        values=np.load(BASE/'data/fashion_mnist/prepared/official_train_labels.npy',allow_pickle=False)
    else:
        raw=gzip.decompress((BASE/'data/fashion_mnist/raw/t10k-labels-idx1-ubyte.gz').read_bytes())
        assert int.from_bytes(raw[:4],'big')==2049 and int.from_bytes(raw[4:8],'big')==10000
        values=np.frombuffer(raw,dtype=np.uint8,offset=8)
    return values[ids].astype(int)


# 用途：处理一个教师、位置与规模：载入历史响应、拟合低阶预测、作全表诊断并保存逐输入结果。
# 参数：seed、pos、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(read(out / 'per_input.json'), done) / (clean(rows), done)。
def analyze_case(seed,pos,q):
    folder=source_case(seed,pos,q)
    out=RUN/'cases'/f'seed_{seed}'/f'{pos}_q{q}'
    if out.exists():
        done=read(out/'completion.json')
        assert sha(out/'per_input.json')==done['rows_sha256']
        assert sha(out/'coefficients.npz')==done['coefficients_sha256']
        return read(out/'per_input.json'),done
    with np.load(folder/'references.npz',allow_pickle=False) as f:
        values={a:f[str(a)].copy() for a in ALPHAS};ids=f['ids'].copy()
    with np.load(folder/'screening_contexts.npz',allow_pickle=False) as f:
        state={k:f[k].copy() for k in ('h','z','W','U','ids','nodes')}
    assert np.array_equal(ids,state['ids']) and len(ids)==(4 if q==12 else 128)
    if q!=12: assert np.array_equal(ids,np.arange(896,1024))
    bits=cube(q);orders=bits.sum(axis=1).astype(int);held=orders>2
    if q!=12:
        with np.load(BASE/f'runs/fashion_scale_20260914/design_q{q}.npz',allow_pickle=False) as f:
            assert np.array_equal(bits,f['evaluation'])
    labels=labels_for(ids,q)
    saved=dict(ids=ids,labels=labels,nodes=state['nodes'],low2_mask_ids=np.flatnonzero(orders<=2))
    rows=[];checks=[]
    for alpha in ALPHAS:
        y=values[alpha];assert y.shape==(len(ids),1<<q,10)
        assert np.isfinite(y).all() and np.max(np.abs(y[:,0]))<TOL
        # 两种预测只使用空/单/双响应；后面的 target 全表只用于评分和诊断。
        single,pair,fit=origin_models(y,bits)
        # Fit identities check only the genuinely observed low-order masks.
        assert np.max(np.abs(pair[:,orders<=2]-y[:,orders<=2]))<TOL
        # 全表 Walsh 投影查看所有答案，仅衡量低阶函数空间的理论残差。
        coeff,energy,inverse_error,parseval_error=walsh_analysis(y,orders)
        target=contrast(y,labels)
        tc=contrast(coeff,labels)
        te=np.stack([np.square(tc[:,orders==k]).sum(axis=1) for k in range(q+1)],axis=1)
        np_ref,nonlinear,srows=independent_state_response(state['h'],state['z'],state['W'],state['U'],bits,alpha,labels)
        reference_error=np.max(np.abs(np_ref-y),axis=(1,2))
        # 检查二阶及以上成分是否确实全部来自单 ReLU 段的非线性修正。
        nonlinear_coeff=fwht(nonlinear)/len(bits)
        nonlinear_error=np.max(np.abs((coeff-nonlinear_coeff)[:,orders>1]),axis=(1,2))
        assert max(float(reference_error.max()),float(nonlinear_error.max()),inverse_error,parseval_error)<TOL
        for key,array in fit.items(): saved[f'a{alpha}_{key}']=array
        saved[f'a{alpha}_walsh_low2']=coeff[:,orders<=2]
        saved[f'a{alpha}_vector_order_energy']=energy
        saved[f'a{alpha}_scalar_order_energy']=te
        scores={}
        for method,pred in [('single',single),('pair',pair)]:
            scores[method,'scalar']=per_input_metrics(contrast(pred,labels),target,held,TOL)
            scores[method,'vector']=per_input_metrics(pred,y,held,TOL)
        for i,input_id in enumerate(ids):
            row=dict(seed=seed,position=pos,q=q,alpha=alpha,input_id=int(input_id),label=int(labels[i]),
                phase='supplementary_validation' if q==12 else 'reused_test',
                total_masks=len(bits),scored_masks=int(held.sum()),single_queries=1+q,
                pair_queries=int((orders<=2).sum()),**srows[i],
                reconstruction_max_abs=float(reference_error[i]),nonlinear_ge2_identity_error=float(nonlinear_error[i]))
            for (method,target_name),stat in scores.items():
                for name,arr in stat.items():
                    row[f'{method}_{target_name}_{name}']=float(arr[i])
            for target_name,en in [('scalar',te[i]),('vector',energy[i].sum(axis=1)/10)]:
                total=float(en.sum());variation=float(en[1:].sum())
                tail1=float(en[2:].sum());tail2=float(en[3:].sum())
                row[f'{target_name}_full_effect_rms']=np.sqrt(total)
                row[f'{target_name}_full_variation_rms']=np.sqrt(variation)
                row[f'{target_name}_oracle_degree1_rmse']=np.sqrt(tail1)
                row[f'{target_name}_oracle_degree2_rmse']=np.sqrt(tail2)
                row[f'{target_name}_oracle_degree1_nrmse']=np.sqrt(tail1/total) if total>TOL*TOL else np.nan
                row[f'{target_name}_oracle_degree2_nrmse']=np.sqrt(tail2/total) if total>TOL*TOL else np.nan
                row[f'{target_name}_high_order_variance_fraction']=tail2/variation if variation>TOL*TOL else np.nan
            rows.append(row)
        checks.append(dict(alpha=alpha,reference_max=float(reference_error.max()),
            nonlinear_identity_max=float(nonlinear_error.max()),inverse_max=inverse_error,parseval_max=parseval_error))
    out.mkdir(parents=True)
    # 保存具名数值数组，便于后来按同一索引回读和独立复算。
    np.savez_compressed(out/'coefficients.npz',**saved)
    dump(out/'per_input.json',rows)
    done=dict(utc=now(),seed=seed,position=pos,q=q,inputs=len(ids),strengths=2,rows=len(rows),
        vectors=len(ids)*len(bits)*2,checks=checks,rows_sha256=sha(out/'per_input.json'),coefficients_sha256=sha(out/'coefficients.npz'))
    dump(out/'completion.json',done)
    print(json.dumps(dict(status='case_complete',seed=seed,position=pos,q=q,inputs=len(ids))),flush=True)
    return clean(rows),done


# 用途：从记录列表取出指定数值字段，把缺失值转换为 NaN 以便显式排除。
# 参数：rows、key；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)。
def array(rows,key): return np.array([np.nan if r[key] is None else r[key] for r in rows],dtype=float)
# 用途：先排除不可定义的值，再对有效数值计算指定统计量。
# 参数：v、fn；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：float(fn(v)) if len(v) else np.nan。
def finite_stat(v,fn=np.mean):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    return float(fn(v)) if len(v) else np.nan
# 用途：合并输入的误差平方和与效应平方和，再开方得到总体 NRMSE。
# 参数：rows、err、scale；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：float(np.sqrt(n / d)) if d > len(rows) * TOL * TOL else np.nan。
def pooled(rows,err,scale):
    n=np.square(array(rows,err)).sum();d=np.square(array(rows,scale)).sum()
    return float(np.sqrt(n/d)) if d>len(rows)*TOL*TOL else np.nan
# 用途：计算有效输入间的 Spearman 等级相关；样本太少或变量恒定时返回不可用。
# 参数：rows、a、b；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：np.nan / float(spearmanr(av[ok], bv[ok]).statistic)。
def rho(rows,a,b):
    av,bv=array(rows,a),array(rows,b);ok=np.isfinite(av)&np.isfinite(bv)
    if ok.sum()<10 or len(np.unique(av[ok]))<2 or len(np.unique(bv[ok]))<2:return np.nan
    return float(spearmanr(av[ok],bv[ok]).statistic)


# 用途：按设置汇总已有误差与成本；模型种子与输入的层级在函数体中明确处理。
# 参数：rows、completed；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：clean(summary)。
def summarize(rows,completed):
    groups=defaultdict(list)
    for r in rows:groups[(r['seed'],r['position'],r['q'],r['alpha'])].append(r)
    seed_rows=[]
    # 先对每个教师的输入汇总，再跨教师汇总，避免组合数造成伪重复。
    for (seed,pos,q,alpha),rs in sorted(groups.items()):
        rec=dict(seed=seed,position=pos,q=q,alpha=alpha,inputs=len(rs),scored_masks=rs[0]['scored_masks'],
            phase=rs[0]['phase'],candidate_states_mean=float(array(rs,'candidate_states').mean()),
            no_candidate_inputs=int((array(rs,'candidate_states')==0).sum()),
            crossing_states_mean=float(array(rs,'strict_crossing_states').mean()),
            reconstruction_max=float(array(rs,'reconstruction_max_abs').max()))
        for target_name in ('scalar','vector'):
            for method in ('single','pair'):
                key=f'{method}_{target_name}'
                nr=array(rs,key+'_nrmse');valid=np.isfinite(nr)
                rec[key+'_pooled_nrmse']=pooled(rs,key+'_rmse',key+'_effect_rms')
                rec[key+'_input_median']=finite_stat(nr,np.median)
                rec[key+'_input_p95']=finite_stat(nr,lambda x:np.quantile(x,.95))
                rec[key+'_fraction_over10']=float((nr[valid]>.1).mean()) if valid.any() else np.nan
                rec[key+'_zero_effect_inputs']=int((~valid).sum())
                rec[key+'_max_abs']=float(array(rs,key+'_max_abs').max())
            for degree in (1,2):
                rec[f'{target_name}_oracle_degree{degree}_pooled_nrmse']=pooled(rs,f'{target_name}_oracle_degree{degree}_rmse',f'{target_name}_full_effect_rms')
            # 高阶方差比例使用去均值变化量作分母，与总效应 NRMSE 区分。
            var=np.square(array(rs,f'{target_name}_full_variation_rms')).sum()
            tail=np.square(array(rs,f'{target_name}_oracle_degree2_rmse')).sum()
            rec[f'{target_name}_high_order_variance_fraction']=float(tail/var) if var>len(rs)*TOL*TOL else np.nan
        rec['pair_better_than_single_inputs']=int((array(rs,'pair_scalar_rmse')<array(rs,'single_scalar_rmse')-TOL).sum())
        rec['candidate_vs_pair_rmse_rho']=rho(rs,'candidate_states','pair_scalar_rmse')
        rec['bound_vs_pair_rmse_rho']=rho(rs,'target_affine_remainder_bound','pair_scalar_rmse')
        rec['candidate_vs_highorder_rmse_rho']=rho(rs,'candidate_states','scalar_oracle_degree2_rmse')
        rec['bound_vs_highorder_rmse_rho']=rho(rs,'target_affine_remainder_bound','scalar_oracle_degree2_rmse')
        seed_rows.append(rec)
    pooled_groups=defaultdict(list)
    for r in seed_rows:pooled_groups[(r['position'],r['q'],r['alpha'])].append(r)
    group_rows=[]
    metric_keys=[k for k in seed_rows[0] if k not in ('seed','position','q','alpha','inputs','scored_masks','phase')]
    for (pos,q,alpha),rs in sorted(pooled_groups.items()):
        assert len(rs)==6
        out=dict(position=pos,q=q,alpha=alpha,teachers=6,inputs_per_teacher=rs[0]['inputs'],phase=rs[0]['phase'])
        for key in metric_keys:
            v=array(rs,key);finite=v[np.isfinite(v)]
            out[key+'_mean']=float(finite.mean()) if len(finite) else np.nan
            out[key+'_sd']=float(finite.std(ddof=1)) if len(finite)>1 else np.nan
        out['pair_better_than_single_teachers']=sum(r['pair_scalar_pooled_nrmse']<r['single_scalar_pooled_nrmse'] for r in rs)
        group_rows.append(out)
    csv_write(RUN/'per_input_metrics.csv',rows)
    csv_write(RUN/'teacher_summary.csv',seed_rows)
    csv_write(RUN/'group_summary.csv',group_rows)
    summary=dict(utc=now(),status='complete',cases=len(completed),teacher_position_size_strength_settings=len(seed_rows),
        input_game_strength_records=len(rows),reference_output_vectors_reused=sum(c['vectors'] for c in completed),
        new_teacher_training_runs=0,new_network_forward_runs=0,calibration_model_access=False,
        prior_outputs_are_reused_not_new_independent_validation=True,
        maximum_reconstruction_difference=max(v['reference_max'] for c in completed for v in c['checks']),
        maximum_nonlinear_identity_difference=max(v['nonlinear_identity_max'] for c in completed for v in c['checks']),
        maximum_inverse_error=max(v['inverse_max'] for c in completed for v in c['checks']),
        maximum_parseval_error=max(v['parseval_max'] for c in completed for v in c['checks']),
        groups=group_rows,teachers=seed_rows)
    dump(RUN/'summary.json',summary)
    return clean(summary)


# 用途：先核验冻结输入与源码，再遍历全部 36 个设置，汇总并写完成记录；不调用新的模型。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def run():
    protocol=verify()
    if (RUN/'complete.json').exists(): raise FileExistsError('Completed analysis is immutable.')
    start=time.perf_counter()
    rows=[];done=[]
    with threadpool_limits(limits=1):
        for case in protocol['cases']:
            rr,dd=analyze_case(case['seed'],case['position'],case['q']);rows.extend(rr);done.append(dd)
        summary=summarize(rows,done)
    verify()
    dump(RUN/'complete.json',dict(utc=now(),status='complete',elapsed_seconds=time.perf_counter()-start,
        protocol_sha256=sha(RUN/'protocol.json'),summary_sha256=sha(RUN/'summary.json'),
        metrics_sha256=sha(RUN/'per_input_metrics.csv'),cases=len(done)))
    print(json.dumps({k:summary[k] for k in ('status','cases','input_game_strength_records','reference_output_vectors_reused','maximum_reconstruction_difference')},ensure_ascii=False),flush=True)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=='__main__':
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['init','run','verify']);args=parser.parse_args()
    if args.stage=='init':initialize()
    elif args.stage=='run':run()
    else:verify();print('All source, protocol and analysis-source hashes match.')

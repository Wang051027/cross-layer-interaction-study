from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'vendor'))
from confirmatory_core import DigitsCNN,SyntheticCNN
from local_modules import LocalModule,source_activation,bit_design,low_order
from context_module import fit_module,predict,RuntimeModule

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def event(kind,**kw):
 with (ROOT/'logs/events.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'event':kind,**kw})+'\n')
# 按模型、方法和预算记录预测误差。
def metrics(p,y,held):
 e=p[:,held]-y[:,held];v=y[:,held];den=float((v*v).sum());ss=float((e*e).sum())
 return {'sse':ss,'effect_sse':den,'nrmse':float(np.sqrt(ss/max(den,1e-30))),'mae':float(abs(e).mean()),'maxerr':float(abs(e).max()),'n_inputs':len(p),'n_coalitions':int(held.sum())}
# 预热后重复执行可调用对象，返回平均或中位延时用于成本比较。
def timer(fn,n=7):
 for _ in range(2):fn()
 ts=[]
 for _ in range(n):
  t=time.perf_counter();fn();ts.append(time.perf_counter()-t)
 return float(np.median(ts))
# 对给定图像取得中间源激活，并生成局部模块需要的上下文列表。
def contexts(mod,x):
 with torch.no_grad():return [mod.prepare(source_activation(mod.model,torch.as_tensor(v[None],dtype=torch.float64),mod.layer)) for v in x]
# 对原包一个数据集、种子和层拟合上下文方法，保存全组合参考、预测、评分和计时。
def run(ds,seed,layer):
 started=time.perf_counter();tag=f'{ds}_{seed}_L{layer}'
 m=(DigitsCNN() if ds=='digits' else SyntheticCNN()).double().eval();m.load_state_dict(torch.load(ROOT/'models'/f'{ds}_teacher_seed_{seed}.pt',weights_only=True))
 ar=np.load(ROOT/'data'/f'{ds}_evaluation.npz');nd=json.loads((ROOT/'protocol'/f'nodes_{ds}_{seed}.json').read_text())[str(layer)]
 # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
 mod=LocalModule(m,nd['shape'],nd['nodes'][:8],layer);bits=bit_design(8);low,_=low_order(8);held=bits.sum(1)>2
 tr=contexts(mod,ar['x_train']);va=contexts(mod,ar['x_validation'])
 t=time.perf_counter();fit=fit_module(mod,tr,bits[low],seed+layer*101);fitsec=time.perf_counter()-t
 J=len(fit.gate);budgets=sorted(set([0,1,2,4,8,16,32,J,J//4,J//2]));budgets=[x for x in budgets if x<=J]
 fp=ROOT/'models'/f'background_fit_{tag}.npz';np.savez_compressed(fp,**fit.__dict__,nodes=mod.nodes,layer=layer)
 event('background_selector_fitted',tag=tag,fit_sha256=sha(fp),fit_seconds=fitsec,train_contexts=len(tr),construction_coalitions=len(low))
 vtrue=np.stack([mod.full_evaluate(c,bits).numpy() for c in va]);val=[]
 for k in budgets:
  pv=np.stack([predict(mod,fit,c,bits,'greedy',k) for c in va]);mm=metrics(pv,vtrue,held);val.append({'dataset':ds,'seed':seed,'layer':layer,'k':k,'J':J,**mm})
 candidates=[r for r in val if r['nrmse']<=.10];chosen=min(r['k'] for r in candidates) if candidates else J
 event('validation_choice_locked',tag=tag,selected_k=chosen,tolerance=.10,primary_fixed_k=J//4)
 te=contexts(mod,ar['x_test']);prim={}
 for name,k in [('validation_selected',chosen),('fixed_quarter',J//4),('random_quarter',J//4)]:
  method='random' if name.startswith('random') else 'greedy';prim[name]=np.stack([predict(mod,fit,c,bits,method,k) for c in te])
 pp=ROOT/'results'/f'committed_predictions_{tag}.npz';np.savez_compressed(pp,**prim,ids=ar['ids_test'])
 event('development_predictions_committed',tag=tag,sha256=sha(pp),note='historically explored test split; NOT external confirmation')
 truth={};rows=[];per=[];timings=[]
 specs=[('local_only',0),('boundary_only',0),('mean_head',0),('pool_ridge',0),('exact',J)]+[(method,k) for method in ['greedy','random','weight','pca'] for k in budgets]
 for alpha in [1.,.5]:
  target=np.stack([mod.full_evaluate(c,bits,alpha).numpy() for c in te]);truth[str(alpha)]=target
  for method,k in specs:
   pv=np.stack([predict(mod,fit,c,bits,method,k,alpha) for c in te]);mm=metrics(pv,target,held)
   background=0 if method=='local_only' else int(4*len(mod.pool_ids)+(k if method in ['greedy','random','weight','pca'] else J if method=='exact' else 0))
   rows.append({'dataset':ds,'seed':seed,'layer':layer,'alpha':alpha,'method':method,'k':k,'J':J,'retained_head_fraction':k/J,'source_scalars':8,'boundary_scalars':int(4*len(mod.pool_ids)),'transmitted_background_scalars':background,'acquired_head_scalars':J if method=='pca' or method=='exact' else k if method in ['greedy','random','weight'] else 0,'validation_selected_k':chosen,'fit_seconds':fitsec,**mm})
   s=((pv-target)[:,held]**2).sum((1,2));d=(target[:,held]**2).sum((1,2))
   for i in range(len(te)):per.append({'dataset':ds,'seed':seed,'layer':layer,'alpha':alpha,'method':method,'k':k,'input_id':str(ar['ids_test'][i]),'sse':float(s[i]),'effect_sse':float(d[i])})
  if alpha==1.:
   for name,pv in prim.items():rows.append({'dataset':ds,'seed':seed,'layer':layer,'alpha':alpha,'method':name,'k':chosen if name=='validation_selected' else J//4,'J':J,'validation_selected_k':chosen,**metrics(pv,target,held)})
 np.savez_compressed(ROOT/'results'/f'reference_{tag}.npz',ids=ar['ids_test'],full=truth['1.0'],half=truth['0.5'])
 verify=[]
 for k in sorted(set([0,J//4,chosen,J])):
  rt=RuntimeModule(mod,fit,k);fullrt=RuntimeModule(mod,fit,J)
  for n in [37,256]:
   bb=bits[low] if n==37 else bits
   for i,c in enumerate(te[:8]):
    source=c['source'];rctx=rt.prepare(source);got=rt.evaluate(rctx,bb);want=predict(mod,fit,c,bb,'greedy',k);verify.append(float(abs(got-want).max()))
    cpu=timer(lambda:rt.evaluate(rctx,bb));prep=timer(lambda:rt.prepare(source));total=timer(lambda:rt.evaluate(rt.prepare(source),bb))
    baseline=timer(lambda:mod.evaluate(mod.prepare(source),bb));fullnp=timer(lambda:fullrt.evaluate(fullrt.prepare(source),bb))
    timings.append({'dataset':ds,'seed':seed,'layer':layer,'k':k,'J':J,'queries':n,'input_index':i,'predict_s':cpu,'prepare_s':prep,'total_s':total,'torch_exact_s':baseline,'numpy_exact_s':fullnp,'strong_exact_s':min(baseline,fullnp),'artifact_numeric_bytes':rt.bytes,'selected_head_context_scalars':k,'boundary_scalars':int(4*len(mod.pool_ids))})
 pd.DataFrame(rows).to_csv(ROOT/'tables'/f'rows_{tag}.csv',index=False);pd.DataFrame(val).to_csv(ROOT/'tables'/f'validation_{tag}.csv',index=False)
 pd.DataFrame(per).to_csv(ROOT/'results'/f'per_input_{tag}.csv.gz',index=False,compression='gzip');pd.DataFrame(timings).to_csv(ROOT/'tables'/f'timing_{tag}.csv',index=False)
 completion={'tag':tag,'runtime_predictor_max_difference':max(verify),'reference_exact_max_error':max(r['maxerr'] for r in rows if r['method']=='exact'),'runtime_seconds':time.perf_counter()-started,'row_count':len(rows),'selected_k':chosen,'full_head':J}
 dump(ROOT/'logs'/f'complete_{tag}.json',completion);event('development_finished',**completion);print(json.dumps(completion),flush=True)
# 原 v0.7 在既有教师上开发局部上下文方法、保存预测和成本结果。
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['digits','synthetic','all'],default='all');ap.add_argument('--seed',type=int);ap.add_argument('--layer',type=int);a=ap.parse_args()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);threadpool_limits(1)
 cfg=json.loads((ROOT/'protocol/development_lock.json').read_text())
 for p,h in cfg['source_sha256'].items():
  if sha(ROOT/p)!=h:raise RuntimeError('Frozen code changed: '+p)
 for ds in (['synthetic','digits'] if a.dataset=='all' else [a.dataset]):
  for seed in ([a.seed] if a.seed else range(26090511,26090517)):
   for layer in ([a.layer] if a.layer else [1,2]):run(ds,seed,layer)
if __name__=='__main__':main()

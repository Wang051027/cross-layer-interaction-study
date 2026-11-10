from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'vendor'),str(R/'code')]
from confirmatory_core import DigitsCNN,SyntheticCNN
from local_modules import LocalModule,source_activation,bit_design
from context_module import Fit,predict,RuntimeModule
from run_development import metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# 原 v0.7 的保存数组回读评分与局部运行时一致性检查。
def main():
 torch.set_num_threads(1);threadpool_limits(1)
 completed=list((R/'logs').glob('complete_*.json'));assert len(completed)==24,f'{len(completed)} /24 runs'
 rows=pd.concat([pd.read_csv(p) for p in sorted((R/'tables').glob('rows_*.csv'))],ignore_index=True)
 vals=pd.concat([pd.read_csv(p) for p in sorted((R/'tables').glob('validation_*.csv'))],ignore_index=True)
 tim=pd.concat([pd.read_csv(p) for p in sorted((R/'tables').glob('timing_*.csv'))],ignore_index=True)
 rows.to_csv(R/'tables/all_results.csv',index=False);vals.to_csv(R/'tables/all_validation.csv',index=False);tim.to_csv(R/'tables/all_timing.csv',index=False)
 group=['dataset','layer','alpha','method','k','J'];summ=rows.groupby(group,dropna=False).agg(nrmse_mean=('nrmse','mean'),nrmse_sd=('nrmse','std'),mae_mean=('mae','mean'),max_error=('maxerr','max'),n_seeds=('seed','nunique')).reset_index();summ.to_csv(R/'tables/background_summary.csv',index=False)
 # Fixed quarter comparisons.
 quarter=rows[(rows.alpha==1)&(rows.k==rows.J/4)&rows.method.isin(['greedy','random','weight','pca'])]
 quarter.groupby(['dataset','layer','method']).agg(nrmse_mean=('nrmse','mean'),nrmse_sd=('nrmse','std')).reset_index().to_csv(R/'tables/quarter_budget_summary.csv',index=False)
 selection=rows[(rows.alpha==1)&(rows.method=='validation_selected')].copy()
 selection['head_reduction']=1-selection.k/selection.J;selection['meets_test_10pct']=selection.nrmse<=.1
 selection.groupby(['dataset','layer']).agg(k_mean=('k','mean'),k_min=('k','min'),k_max=('k','max'),J=('J','first'),nrmse_mean=('nrmse','mean'),nrmse_sd=('nrmse','std'),head_reduction=('head_reduction','mean'),seeds_meet_10pct=('meets_test_10pct','sum')).reset_index().to_csv(R/'tables/validation_selection_summary.csv',index=False)
 ti=tim.groupby(['dataset','seed','layer','k','J','queries']).mean(numeric_only=True).reset_index()
 ti['speedup_ratio']=ti.strong_exact_s/ti.total_s
 ti.to_csv(R/'tables/seed_timing.csv',index=False)
 ti.groupby(['dataset','layer','k','J','queries']).agg(total_ms=('total_s',lambda x:1000*x.mean()),strong_exact_ms=('strong_exact_s',lambda x:1000*x.mean()),speedup_mean=('speedup_ratio','mean'),speedup_sd=('speedup_ratio','std'),numeric_bytes=('artifact_numeric_bytes','mean'),boundary_scalars=('boundary_scalars','mean')).reset_index().to_csv(R/'tables/timing_summary.csv',index=False)
 # Reread only validation for selected budget reconstruction; independently rescore raw predictions.
 held=bit_design(8).sum(1)>2;score_error=[];pred_error=[];full_error=[];omitted_invariance=[];commit_ok=[]
 for path in sorted((R/'results').glob('committed_predictions_*.npz')):
  tag=path.stem.removeprefix('committed_predictions_');ds,seed,L=tag.split('_');seed=int(seed);layer=int(L[1:])
  pp=np.load(path);refp=R/'results'/f'reference_{tag}.npz';rr=np.load(refp);commit_ok.append(path.stat().st_mtime_ns<=refp.stat().st_mtime_ns)
  for name in ['validation_selected','fixed_quarter','random_quarter']:
   mm=metrics(pp[name],rr['full'],held);old=rows[(rows.dataset==ds)&(rows.seed==seed)&(rows.layer==layer)&(rows.alpha==1)&(rows.method==name)].iloc[0]
   score_error.append(abs(mm['nrmse']-old.nrmse))
  fp=np.load(R/'models'/f'background_fit_{tag}.npz');fit=Fit(**{k:fp[k] for k in Fit.__annotations__})
  m=(DigitsCNN() if ds=='digits' else SyntheticCNN()).double().eval();m.load_state_dict(torch.load(R/'models'/f'{ds}_teacher_seed_{seed}.pt',weights_only=True))
  ar=np.load(R/'data'/f'{ds}_evaluation.npz');x=torch.as_tensor(ar['x_test'][:1],dtype=torch.float64)
  with torch.no_grad():source=source_activation(m,x,layer)
  mod=LocalModule(m,tuple(source.shape[1:]),fp['nodes'],layer);ctx=mod.prepare(source);bits=bit_design(8)
  pr=predict(mod,fit,ctx,bits,'greedy',len(fit.gate)//4);pred_error.append(float(abs(pr-pp['fixed_quarter'][0]).max()))
  full_error.append(float(abs(mod.full_evaluate(ctx,bits).numpy()-rr['full'][0]).max()))
  corrupted=dict(ctx);corrupted['z3']=ctx['z3'].clone();drop=np.ones(len(fit.gate),bool);drop[fit.order[:len(fit.gate)//4]]=False;corrupted['z3'][drop]=10000.
  omitted_invariance.append(float(abs(pr-predict(mod,fit,corrupted,bits,'greedy',len(fit.gate)//4)).max()))
 # Repeat exact-hash dataset audit, preserving all historical arrays.
 audit={}
 for ds in ['digits','synthetic']:
  ar=np.load(R/'data'/f'{ds}_evaluation.npz');sets={};au={}
  for sp in ['train','validation','test']:
   x=ar['x_'+sp];y=ar['y_'+sp];hh=[hashlib.sha256(v.tobytes()).hexdigest() for v in x];sets[sp]=set(hh)
   au[sp]={'n':len(x),'unique':len(sets[sp]),'finite':bool(np.isfinite(x).all()),'shape':list(x.shape),'counts':{str(c):int((y==c).sum()) for c in np.unique(y)}}
  au['cross_exact_overlap']={a+'__'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};audit[ds]=au
 completion=[json.loads(p.read_text()) for p in completed]
 report={'completed_model_layer_runs':24,'teachers':12,'new_teachers':0,'external_dataset_runs':0,'test_full_tail_output_evaluations':24*128*256*2,'validation_full_tail_output_evaluations':24*64*256,'training_compiled_construction_values':24*192*37,'result_rows':len(rows),'primary_raw_predictions_rescored':len(score_error),'score_max_difference':max(score_error),'serialized_model_replay_max_difference':max(pred_error),'independent_full_tail_replay_max_difference':max(full_error),'unretained_z3_poison_test_max_difference':max(omitted_invariance),'primary_predictions_precede_references':bool(all(commit_ok)),'maximum_full_exact_error':max(x['reference_exact_max_error'] for x in completion),'maximum_runtime_vs_array_predictor_error':max(x['runtime_predictor_max_difference'] for x in completion),'total_experiment_seconds':sum(x['runtime_seconds'] for x in completion),'environment':{'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':1,'dtype':'float64'},'scope':'development reanalysis; not external or independent validation'}
 (R/'results/verification.json').write_text(json.dumps(report,indent=2));(R/'data/development_quality_audit.json').write_text(json.dumps(audit,indent=2))
 assert report['score_max_difference']<1e-12 and max(pred_error)<1e-12 and max(full_error)<1e-12 and max(omitted_invariance)==0 and all(commit_ok)
 print(json.dumps(report,indent=2));print(pd.read_csv(R/'tables/validation_selection_summary.csv').to_string(index=False));print(pd.read_csv(R/'tables/quarter_budget_summary.csv').to_string(index=False))
if __name__=='__main__':main()

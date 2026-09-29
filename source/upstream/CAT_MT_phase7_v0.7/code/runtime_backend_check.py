# 中文阅读说明｜用途：原 v0.7 的 NumPy/PyTorch 后端一致性与运行时间对照。
# 数据流：用相同局部状态比较数学结果和计时，不能把后端差异当算法收益。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Post-primary backend control: same selected-background formula in Torch.
No change to selected masks, statistical estimators, or response predictions.
All timings are descriptive and were added after the primary NumPy timing results.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,json,hashlib
from datetime import datetime,timezone
import torch,numpy as np,pandas as pd
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'vendor'),str(R/'code')]
from confirmatory_core import DigitsCNN,SyntheticCNN
from local_modules import LocalModule,source_activation,bit_design,low_order
from context_module import Fit,RuntimeModule,predict
# 用途：用 PyTorch 实现原局部选择运行时，用于与 NumPy 后端比较。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class TorchRuntime:
 # 用途：建立对象所需的网络层、固定映射或配置。用 PyTorch 实现原局部选择运行时，用于与 NumPy 后端比较。
 # 参数：mod、fit、k；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：初始化对象字段，没有显式返回值。
 def __init__(self,mod,fit,k):
  self.mod=mod;self.idx=torch.tensor(fit.order[:k]);self.k=k;self.J=len(fit.gate)
  self.A=mod.A;self.W=mod.W[self.idx].contiguous();self.U=mod.U[:,self.idx].contiguous()
  mask=torch.ones(self.J,dtype=torch.bool);mask[self.idx]=False
  self.linear=((mod.U[:,mask]*torch.tensor(fit.gate)[mask])@mod.W[mask]).contiguous()
  self.headW=mod.model.fc1.weight.detach()[self.idx].contiguous();self.headb=mod.model.fc1.bias.detach()[self.idx].contiguous()
 # 用途：在 PyTorch 内生成边界及选中头部状态，供后端一致性比较。
 # 参数：source；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：(source.flatten()[self.mod.nodes], pre.flatten()[self.mod.window_ids], z)。
 @torch.no_grad()
 def prepare(self,source):
  pre=self.mod.model.conv2(source) if self.mod.layer==1 else source
  aa=F.relu(pre) if self.mod.layer==1 else pre
  pooled=F.max_pool2d(aa,2).flatten();z=F.linear(pooled,self.headW,self.headb)
  return (source.flatten()[self.mod.nodes],pre.flatten()[self.mod.window_ids],z)
 # 用途：在相同查询掩码上计算选择方法的输出差，对齐 NumPy 版本。
 # 参数：ctx、bits；具体形状/来源结合调用处及下方关键步骤阅读。
 # 返回：out。
 @torch.no_grad()
 def evaluate(self,ctx,bits):
  h,w,z=ctx;zz=w[None]+(-bits*h[None]@self.A).reshape(len(bits),-1,4)
  aa=F.relu(zz) if self.mod.layer==1 else zz;a0=F.relu(w) if self.mod.layer==1 else w
  dp=aa.amax(2)-a0.amax(1)[None]
  out=None if self.k==self.J else dp@self.linear.T
  if self.k:
   term=(F.relu(z+dp@self.W.T)-F.relu(z))@self.U.T
   out=term if out is None else out+term
  return out

# 用途：预热后对同一可调用对象重复计时，减少首次运行开销干扰。
# 参数：fn；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：float(np.median(ts))。
def bench(fn):
 for _ in range(3):fn()
 ts=[]
 for _ in range(15):
  t=time.perf_counter();fn();ts.append(time.perf_counter()-t)
 return float(np.median(ts))
# 用途：原 v0.7 的 NumPy/PyTorch 后端一致性与运行时间对照。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
 torch.set_num_threads(1);threadpool_limits(1);rows=[];error=0
 for ds in ['synthetic','digits']:
  ar=np.load(R/'data'/f'{ds}_evaluation.npz')
  for seed in range(26090511,26090517):
   m=(DigitsCNN() if ds=='digits' else SyntheticCNN()).double().eval();m.load_state_dict(torch.load(R/'models'/f'{ds}_teacher_seed_{seed}.pt',weights_only=True))
   for layer in [1,2]:
    fp=np.load(R/'models'/f'background_fit_{ds}_{seed}_L{layer}.npz');fit=Fit(**{k:fp[k] for k in Fit.__annotations__})
    with torch.no_grad():hh=source_activation(m,torch.tensor(ar['x_test'][:4],dtype=torch.float64),layer)
    mod=LocalModule(m,tuple(hh.shape[1:]),fp['nodes'],layer);J=len(fit.gate)
    full=TorchRuntime(mod,fit,J);nf=RuntimeModule(mod,fit,J)
    selected=int(json.loads((R/'logs'/f'complete_{ds}_{seed}_L{layer}.json').read_text())['selected_k'])
    for k in sorted(set([0,J//4,selected,J])):
     part=TorchRuntime(mod,fit,k)
     for n in [37,256]:
      b=bit_design(8);b=b[low_order(8)[0]] if n==37 else b;bt=torch.tensor(b)
      for i,ss in enumerate(hh):
       s=ss[None];ctx=mod.prepare(s);got=part.evaluate(part.prepare(s),bt).numpy();want=predict(mod,fit,ctx,b,'greedy',k);error=max(error,float(abs(got-want).max()))
       tpart=bench(lambda:part.evaluate(part.prepare(s),bt));tfull=bench(lambda:full.evaluate(full.prepare(s),bt));tnumpy=bench(lambda:nf.evaluate(nf.prepare(s),b));torig=bench(lambda:mod.evaluate(mod.prepare(s),bt))
       rows.append(dict(dataset=ds,seed=seed,layer=layer,k=k,J=J,queries=n,input_index=i,torch_partial_s=tpart,torch_full_s=tfull,numpy_full_s=tnumpy,original_compiled_s=torig,strong_exact_s=min(tfull,tnumpy,torig)))
 pd.DataFrame(rows).to_csv(R/'tables/backend_timing_raw.csv',index=False)
 out=pd.DataFrame(rows).groupby(['dataset','seed','layer','k','J','queries']).mean(numeric_only=True).reset_index();out['speedup']=out.strong_exact_s/out.torch_partial_s
 out.to_csv(R/'tables/backend_timing_seeds.csv',index=False)
 summary=out.groupby(['dataset','layer','k','J','queries']).agg(partial_ms=('torch_partial_s',lambda x:1000*x.mean()),exact_ms=('strong_exact_s',lambda x:1000*x.mean()),paired_speedup_mean=('speedup','mean'),paired_speedup_sd=('speedup','std')).reset_index()
 summary.to_csv(R/'tables/backend_timing_summary.csv',index=False)
 # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
 (R/'results/backend_verification.json').write_text(json.dumps({'max_prediction_difference':error,'timing_rows':len(rows),'status':'post-primary implementation diagnostic; no accuracy retuning'},indent=2));assert error<1e-12
 print(summary[(summary.k==summary.J/4)&(summary.queries==256)].to_string(index=False));print('maxerr',error)
# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=='__main__':main()

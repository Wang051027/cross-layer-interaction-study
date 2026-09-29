# 中文阅读说明｜用途：把残差块原始 ReLU 区域内的线性响应融合成小矩阵，再单独测量成本。
# 数据流：使用冻结残差模型和既有输入；输出融合计算的一致性、缓存和计时结果。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Post-test, explicitly separate audit of an algebraically fused affine control.

No model, selector, test prediction or pre-test timing is modified. The original
region formula is collapsed to its input-dependent 10-by-8 linear response map.
"""
from pathlib import Path
from collections import defaultdict
import shutil
import time
from residual_context import *
from run_residual_controls import verify, training_data, prefix
from prepare_external import load_idx

OUT = RUN / "affine_runtime_audit"


# 用途：固定原始 ReLU 符号后，将 U、门控和 W 合并为源坐标到输出的线性映射。
# 结构：成员字段和方法共同保存本阶段需要的状态。
class FusedAffine:
    # 用途：建立对象所需的网络层、固定映射或配置。固定原始 ReLU 符号后，将 U、门控和 W 合并为源坐标到输出的线性映射。
    # 参数：local；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：初始化对象字段，没有显式返回值。
    def __init__(self,local):self.local=local

    # 用途：在当前输入的 ReLU 原始符号固定时，把 U·diag(z>0)·W 合并为响应矩阵。
    # 参数：context；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：(h, (self.local.U * (z > 0)[None] @ self.local.W).contiguous())。
    @torch.no_grad()
    def cached(self,context):
        h,z=context
        return h,((self.local.U*(z>0)[None])@self.local.W).contiguous()

    # 用途：取得原始 h/z 后构造融合矩阵；这一工作属于准备成本。
    # 参数：source、residual；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：self.cached(self.local.states(source, residual, 'exact_crop'))。
    @torch.no_grad()
    def prepare(self,source,residual):
        return self.cached(self.local.states(source,residual,"exact_crop"))

    # 用途：将源激活衰减乘以融合矩阵；跨越 ReLU 边界时只是近似。
    # 参数：context、bits、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：-alpha * bits * h[None] @ matrix.T。
    @torch.no_grad()
    def evaluate(self,context,bits,alpha=1.):
        h,matrix=context
        return (-alpha*bits*h[None])@matrix.T


# 用途：把残差块原始 ReLU 区域内的线性响应融合成小矩阵，再单独测量成本。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    configure_cpu();verify(RUN,True)
    assert read_json(RUN / "analysis/verification.json")["status"]=="passed"
    OUT.mkdir(exist_ok=False)
    own=Path(__file__).resolve()
    plan=dict(utc=now(),scope="post-test implementation/cost and gate-switch audit; not a newly preregistered accuracy comparison",
        trigger="The frozen original-region control had low aggregate test error; its per-input gates allow further algebraic fusion",
        formula="M(x)=U diag(z(x)>0) W; prediction=delta_h M(x)^T",
        equivalence="compare to all twelve already saved original-region test prediction arrays; no replacement or fitting",
        cached_state="8 source scalars plus 80 response coefficients after prepare; computation of current gates and coefficients charged in from_image",
        timing=dict(inputs="same four fixed validation contexts",queries=[37,256],threads=1,dtype="CPU float64",
                    warmup=3,rounds=9,inner=3,random_order=True,training_concurrent=False,
                    methods="all original 32 timing methods plus fused affine",comparison="fastest measured exact engine within this new matched timing session"),
        diagnostics="per-input fraction of affected states and coalitions whose last-ReLU gate changes; descriptive only",
        freeze_sha256=sha(RUN / "freeze.json"),source_sha256=sha(own),calibration_model_access=False)
    dump(OUT / "plan.json",plan);shutil.copy2(own,OUT / "source_snapshot.py")
    images,_,val_ids=training_data();vx=torch.tensor(normalized(images[val_ids[:4]]),dtype=torch.float64)
    official=load_idx(BASE / "data/fashion_mnist/raw","t10k-images-idx3-ubyte.gz")
    ids=np.load(RUN / "intervention_ids.npy",allow_pickle=False)
    tx=torch.tensor(normalized(official[ids]),dtype=torch.float64)
    bits=torch.tensor(bit_design(8),dtype=torch.float64);held=bit_design(8).sum(1)>2
    raw=[];diagnostics=[];checks=[];maximum=0.;fresh_difference=0.
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        model=load_teacher(RUN / f"seed_{seed}");fit,nodes=load_fit(RUN / f"seed_{seed}/fit.npz")
        ts,tr=prefix(model,tx);local=ResidualLocal(model,tuple(ts.shape[1:]),nodes);affine=FusedAffine(local)
        with np.load(RUN / f"test/seed_{seed}/predictions.npz",allow_pickle=False) as pp,np.load(RUN / f"test/seed_{seed}/references.npz",allow_pickle=False) as rr:
            for alpha in STRENGTHS:
                prior=pp[key("original_region",1.,alpha)];ref=rr[str(alpha)];pred=[]
                for i,(s,r) in enumerate(zip(ts,tr)):
                    context=local.states(s[None],r[None]);pred.append(affine.evaluate(affine.cached(context),bits,alpha).numpy())
                    h,z=context;d=(-alpha*bits*h[None])@local.W.T
                    switches=((z[None]+d)>0)!=(z>0)[None]
                    selected=switches[held]
                    den=float(np.square(ref[i,held]).sum());num=float(np.square((prior[i]-ref[i])[held]).sum())
                    diagnostics.append(dict(seed=seed,input_id=int(ids[i]),alpha=alpha,affected_states=local.J,
                        mean_fraction_states_switched=float(selected.double().mean()),
                        fraction_coalitions_any_switch=float(selected.any(1).double().mean()),
                        original_region_nrmse=float(np.sqrt(num/den)) if den>1e-24 else 0.,
                        usable_effect=bool(den>1e-24),effect_squared_sum=den))
                pred=np.stack(pred);difference=float(np.abs(pred-prior).max());maximum=max(maximum,difference)
                assert difference<1e-9
                metric=score(pred,ref,held)
                checks.append(dict(seed=seed,alpha=alpha,max_prediction_difference=difference,nrmse=metric["nrmse"],
                                   original_prediction_sha256=sha(RUN / f"test/seed_{seed}/predictions.npz")))
        vs,vr=prefix(model,vx)
        runtimes={f"{m}_f{f}":ContextRuntime(local,fit,m,k) for m,f,k in specs(local) if m!="exact"}
        runtimes.update({e:ContextRuntime(local,fit,"exact",local.J,e) for e in EXACT_ENGINES})
        runtimes["affine_fused"]=affine
        rng=np.random.default_rng(seed+2026091307)
        for i in range(4):
            common=local.states(vs[i:i+1],vr[i:i+1]);cached={name:rt.cached(common) for name,rt in runtimes.items()}
            for q in (37,256):
                bt=bits[low_order(8)[0]] if q==37 else bits
                for workload in ("from_image","cached"):
                    functions={}
                    for name,rt in runtimes.items():
                        if workload=="cached":functions[name]=lambda name=name,rt=rt:rt.evaluate(cached[name],bt)
                        else:
                            # 用途：构造当前运行时的一次实际执行，供计时循环调用。
                            # 参数：rt；具体形状/来源结合调用处及下方关键步骤阅读。
                            # 返回：rt.evaluate(rt.prepare(s, r), bt)。
                            def fn(rt=rt):
                                s,r=model.prefix(vx[i:i+1]);return rt.evaluate(rt.prepare(s,r),bt)
                            functions[name]=fn
                    with torch.no_grad():
                        for name,fn in functions.items():
                            fresh_difference=max(fresh_difference,float((fn()-runtimes[name].evaluate(cached[name],bt)).abs().max()))
                            for _ in range(3):fn()
                        names=list(functions)
                        for round_ in range(9):
                            for ix in rng.permutation(len(names)):
                                name=names[ix];start=time.perf_counter_ns()
                                for _ in range(3):functions[name]()
                                raw.append(dict(seed=seed,input_id=int(val_ids[i]),queries=q,workload=workload,
                                                method=name,round=round_,seconds=(time.perf_counter_ns()-start)/3e9))
        event(RUN,"affine_runtime_audit_seed_complete",seed=seed,max_equivalence_difference=maximum)
    assert fresh_difference<1e-9
    csv_write(OUT / "equivalence.csv",checks);csv_write(OUT / "switch_diagnostics.csv",diagnostics);csv_write(OUT / "raw.csv",raw)
    groups=defaultdict(list)
    for r in raw:groups[(r["seed"],r["input_id"],r["queries"],r["workload"],r["method"])].append(r["seconds"])
    medians=[dict(zip(("seed","input_id","queries","workload","method"),k),seconds=float(np.median(v))) for k,v in groups.items()]
    csv_write(OUT / "input_medians.csv",medians)
    groups=defaultdict(list)
    for r in medians:groups[(r["seed"],r["queries"],r["workload"],r["method"])].append(r["seconds"])
    summary=[dict(zip(("seed","queries","workload","method"),k),seconds=float(np.mean(v))) for k,v in groups.items()]
    csv_write(OUT / "summary.csv",summary)
    fastest=[]
    for seed in SEEDS:
        for q in (37,256):
            for workload in ("from_image","cached"):
                pool=[r for r in summary if r["seed"]==seed and r["queries"]==q and r["workload"]==workload and r["method"] in EXACT_ENGINES]
                fastest.append(min(pool,key=lambda r:r["seconds"]))
    csv_write(OUT / "fastest_exact.csv",fastest)
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert sha(own)==plan["source_sha256"]==sha(OUT / "source_snapshot.py")
    assert sha(RUN / "freeze.json")==plan["freeze_sha256"]
    verify(RUN,True)
    dump(OUT / "completion.json",dict(utc=now(),status="completed",plan_sha256=sha(OUT / "plan.json"),
        arrays_compared=len(checks),raw_timing_rows=len(raw),max_prediction_difference=maximum,
        max_fresh_cached_difference=fresh_difference,calibration_model_access=False,
        existing_test_outputs_unchanged=True,new_confirmatory_p_values=0,
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    event(RUN,"affine_runtime_audit_complete",max_difference=maximum,rows=len(raw))


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

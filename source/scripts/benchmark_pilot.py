"""CPU cost diagnostic on four pilot validation images, after other jobs finish."""
from __future__ import annotations
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from research_runtime import (BASE, Fit, LocalModule, HeadRuntime, make_teacher, source_activation,
                              bit_design, low_order, np, torch)
from threadpoolctl import threadpool_limits


def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


# 对首个 Fashion-MNIST 教师做小规模成本测量，区分获取状态和查询开销。
def main():
    torch.set_num_threads(1);torch.set_grad_enabled(False);threadpool_limits(1)
    run=BASE / "runs/fashion_pilot_20260910"
    assert json.loads((run / "completion.json").read_text())["status"]=="completed"
    out=run / "timing"
    out.mkdir(exist_ok=False)
    cfg=dict(utc=datetime.now(timezone.utc).isoformat(),stage="pilot validation cost diagnostic",
             methods=["greedy","pca_fused","pca_unfused","exact"],k=16,
             device="CPU",dtype="float64",threads=1,queries=[37,256],layers=[1,2],
             workloads=["from_image","from_source","cached"],inputs=4,rounds=15,inner=3,warmup=3,
             method_order="randomized every round, RNG seed 20260910",
             code_sha256={name:hashlib.sha256((BASE / "scripts" / name).read_bytes()).hexdigest()
                          for name in ("benchmark_pilot.py","research_runtime.py")})
    (out / "protocol.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    data=BASE / "data/fashion_mnist/prepared"
    x=np.load(data / "official_train_images.npy",mmap_mode="r")
    ix=np.load(data / "context_indices.npz")["validation"][:4]
    images=torch.tensor(x[ix][:,None].astype(np.float32)/np.float32(255),dtype=torch.float64)
    m=make_teacher("fashion").double().eval()
    m.load_state_dict(torch.load(run / "teacher_seed_202609071.pt",weights_only=True,map_location="cpu"))
    raw=[];costs=[];rng=np.random.default_rng(20260910);max_equivalence=0.
    for layer in (1,2):
        fp=np.load(run / f"background_fit_L{layer}.npz")
        fit=Fit(**{key:fp[key] for key in Fit.__annotations__})
        sources=source_activation(m,images,layer)
        mod=LocalModule(m,tuple(sources.shape[1:]),fp["nodes"],layer)
        runtimes={method:HeadRuntime(mod,fit,method,64 if method=="exact" else 16) for method in cfg["methods"]}
        for method,rt in runtimes.items():costs.append(dict(layer=layer,method=method,**rt.cost_counts()))
        for i in range(4):
            image,source=images[i:i+1],sources[i:i+1]
            ctx=mod.prepare(source);common=ctx["h"],ctx["window"],ctx["z3"]
            for q in (37,256):
                bits=bit_design(8);bits=bits[low_order(8)[0]] if q==37 else bits
                bits=torch.tensor(bits,dtype=torch.float64)
                for method,rt in runtimes.items():
                    a=rt.evaluate(rt.prepare(source),bits)
                    b=rt.evaluate(rt.prepare_cached(common),bits)
                    max_equivalence=max(max_equivalence,float((a-b).abs().max()))
                for workload in cfg["workloads"]:
                    if workload=="from_image":
                        fs={name:(lambda rt=rt:rt.evaluate(rt.prepare(source_activation(m,image,layer)),bits)) for name,rt in runtimes.items()}
                    elif workload=="from_source":
                        fs={name:(lambda rt=rt:rt.evaluate(rt.prepare(source),bits)) for name,rt in runtimes.items()}
                    else:
                        fs={name:(lambda rt=rt:rt.evaluate(rt.prepare_cached(common),bits)) for name,rt in runtimes.items()}
                    for fn in fs.values():
                        for _ in range(cfg["warmup"]):fn()
                    for round_index in range(cfg["rounds"]):
                        for method in rng.permutation(cfg["methods"]):
                            fn=fs[method];start=time.perf_counter_ns()
                            for _ in range(cfg["inner"]):fn()
                            elapsed=(time.perf_counter_ns()-start)/cfg["inner"]/1e9
                            raw.append(dict(layer=layer,input_index=i,queries=q,workload=workload,
                                            method=method,round=round_index,seconds=elapsed))
        print(json.dumps(dict(finished_layer=layer,maximum_prepare_equivalence_error=max_equivalence)),flush=True)
    by_input=defaultdict(list)
    for r in raw:by_input[tuple(r[k] for k in ("layer","queries","workload","method","input_index"))].append(r["seconds"])
    grouped=defaultdict(list)
    for key,values in by_input.items():grouped[key[:-1]].append(float(np.median(values)))
    summary=[]
    for key,values in grouped.items():
        summary.append(dict(zip(("layer","queries","workload","method"),key),
                            ms_mean=1000*float(np.mean(values)),ms_input_sd=1000*float(np.std(values,ddof=1))))
    write_csv(out / "timing_raw.csv",raw);write_csv(out / "timing_summary.csv",summary);write_csv(out / "cost_counts.csv",costs)
    assert max_equivalence<1e-9
    (out / "completion.json").write_text(json.dumps(dict(status="completed",raw_rows=len(raw),
        prepare_equivalence_max_error=max_equivalence,scope="one teacher, four validation images; descriptive timing"),indent=2),encoding="utf-8")


if __name__=="__main__":main()

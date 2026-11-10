"""Descriptive synthesis and shared-validation consistency audit; no selection."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import json
import statistics as st
import numpy as np

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/"runs/fashion_scale_20260914"
PRIOR=BASE/"runs/fashion_residual_20260913"
SEEDS=list(range(202609071,202609077))


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


# 按教师、规模、位置和强度汇总规模对照的已保存结果。
def main():
    assert read(RUN/"analysis/verification.json")["status"]=="passed"
    summary=rows(RUN/"analysis/summary.csv")
    bounds=rows(RUN/"analysis/bound_diagnostics.csv")
    pairing=rows(RUN/"analysis/paired_differences.csv")
    timing=rows(RUN/"timing/summary.csv")
    exact=rows(RUN/"timing/fastest_exact.csv")
    bound_seeds=[]
    for seed in SEEDS:
        for position in ("center","upper_left"):
            for q in (4,8,12):
                for alpha in (1.,.5):
                    group=[r for r in bounds if int(r["seed"])==seed and r["position"]==position
                           and int(r["q"])==q and float(r["alpha"])==alpha]
                    assert len(group)==128
                    usable=[r for r in group if r["usable_effect"]=="True"]
                    accepted=[r for r in group if r["accepted"]=="True"]
                    au=[r for r in usable if r["accepted"]=="True"]
                    accurate=[r for r in usable if float(r["affine_nrmse"])<=.1]
                    bound_seeds.append(dict(seed=seed,position=position,q=q,alpha=alpha,inputs=128,
                        usable=len(usable),zero_effect=128-len(usable),accepted_all=len(accepted),
                        accepted_usable=len(au),accurate_affine_usable=len(accurate),
                        rejected_but_accurate=sum(r["accepted"]!="True" for r in accurate),
                        acceptance_all=len(accepted)/128,
                        acceptance_usable=len(au)/len(usable) if usable else None,
                        hindsight_affine_usable_rate=len(accurate)/len(usable) if usable else None,
                        accepted_max_nrmse=max([float(r["affine_nrmse"]) for r in au],default=None),
                        switching_states_mean=st.mean(float(r["fraction_states_switched"]) for r in group),
                        switching_queries_mean=st.mean(float(r["fraction_queries_any_switch"]) for r in group),
                        source_l2_mean=st.mean(float(r["source_l2"]) for r in group)))
    groups=defaultdict(list)
    for r in bound_seeds:groups[(r["position"],r["q"],r["alpha"])].append(r)
    bound_summary=[]
    for (position,q,alpha),rs in groups.items():
        item=dict(position=position,q=q,alpha=alpha,seeds=len(rs))
        for name in ("acceptance_all","acceptance_usable","hindsight_affine_usable_rate",
                     "switching_states_mean","switching_queries_mean","source_l2_mean"):
            values=[r[name] for r in rs if r[name] is not None]
            item[name+"_mean"]=st.mean(values) if values else None
            item[name+"_sd"]=st.stdev(values) if len(values)>1 else None
        for name in ("usable","zero_effect","accepted_all","accepted_usable","accurate_affine_usable","rejected_but_accurate"):
            item[name+"_sum"]=sum(r[name] for r in rs)
        bound_summary.append(item)
    timing_values=[]
    for position in ("center","upper_left"):
        for q in (4,8,12):
            for method in sorted({r["method"] for r in timing}):
                for query_set in ("low_order","evaluation"):
                    for workload in ("from_image","cached"):
                        ratios=[];times=[]
                        for seed in SEEDS:
                            same=lambda r:int(r["seed"])==seed and r["position"]==position and int(r["q"])==q and r["query_set"]==query_set and r["workload"]==workload
                            value=next(float(r["seconds"]) for r in timing if same(r) and r["method"]==method)
                            reference=next(float(r["seconds"]) for r in exact if same(r))
                            ratios.append(reference/value);times.append(value*1000)
                        timing_values.append(dict(position=position,q=q,method=method,query_set=query_set,
                            workload=workload,speed_mean=st.mean(ratios),speed_sd=st.stdev(ratios),
                            faster_seeds=sum(v>1 for v in ratios),milliseconds_mean=st.mean(times)))
    groups=defaultdict(list)
    for r in pairing:
        groups[(r["position"],int(r["q"]),float(r["fraction"]),float(r["alpha"]),r["control"])].append(float(r["greedy_minus_control"]))
    contrasts=[dict(position=k[0],q=k[1],fraction=k[2],alpha=k[3],control=k[4],
        mean_difference=st.mean(v),greedy_better=sum(x < -1e-12 for x in v),
        tied=sum(abs(x)<=1e-12 for x in v),greedy_worse=sum(x > 1e-12 for x in v)) for k,v in groups.items()]
    # Same teachers, source rule, fitting data and validation images must reproduce
    # the previous q8-center predictions; this is not independent new evidence.
    arrays=0;maximum=0.;reference_difference=0.
    for seed in SEEDS:
        old=PRIOR/f"seed_{seed}/validation"
        new=RUN/f"seed_{seed}/center_q8/validation"
        with np.load(old/"predictions.npz",allow_pickle=False) as op,np.load(new/"predictions.npz",allow_pickle=False) as npred:
            assert np.array_equal(op["ids"],npred["ids"])
            for name in npred.files:
                if name=="ids" or name.startswith("bounded_affine"):
                    continue
                prior_name=name.replace("affine_fused","original_region")
                maximum=max(maximum,float(np.abs(npred[name]-op[prior_name]).max()))
                arrays+=1
        with np.load(old/"references.npz",allow_pickle=False) as old_ref,np.load(new/"references.npz",allow_pickle=False) as new_ref:
            for alpha in ("1.0","0.5"):
                reference_difference=max(reference_difference,float(np.abs(old_ref[alpha]-new_ref[alpha]).max()))
    assert arrays==108 and maximum<1e-9 and reference_difference<1e-9
    out=dict(utc=datetime.now(timezone.utc).isoformat(),status="passed",
        purpose="descriptive presentation calculations; no test-dependent model or threshold selection",
        shared_validation_replay=dict(arrays=arrays,max_prediction_difference=maximum,
            max_reference_difference=reference_difference,independent_new_evidence=False),
        summaries=summary,bound_seeds=bound_seeds,bound_summary=bound_summary,
        timing=timing_values,paired=contrasts,
        calibration_model_access=False,new_confirmatory_p_values=0)
    (BASE/"audit/scale_interpretation_20260914.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    primary=[r for r in summary if r["alpha"]=="1.0" and ((r["fraction"]=="0.25" and r["method"] in ("greedy","gate_weight","pca_fused")) or r["method"] in ("affine_fused","bounded_affine"))]
    print(json.dumps(dict(shared_replay=out["shared_validation_replay"],primary=primary,
        bounds=[r for r in bound_summary if r["alpha"]==1.],
        timing=[r for r in timing_values if r["query_set"]=="evaluation" and r["method"] in ("affine_fused_f1.0","bounded_affine_f1.0")
                and r["workload"]=="from_image"]),ensure_ascii=True))


if __name__=="__main__":main()

# 中文阅读说明｜用途：汇总原六教师主实验，重算配对统计，并复核保存预测的评分。
# 数据流：输入 test/ 和 timing/ 结果；输出按教师汇总的统计，原主检验仅限固定的一项对比。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Prespecified seed-level analysis and saved-array audit, without model fitting."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import scipy
from scipy import stats
from external_experiment import (BASE, SEEDS, METHODS, read_json, dump, csv_read, csv_write,
    sha, bit_design, score, prediction_key, np)


# 用途：对原主对比的六个教师误差做配对统计，穷举 64 种符号翻转并用 SciPy 交叉检查。
# 参数：greedy、weight；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def primary_statistics(greedy, weight):
    # 配对单位是六个教师；各教师的组合和重复输入不能当成额外独立样本。
    differences=np.asarray(greedy,dtype=float)-np.asarray(weight,dtype=float)
    assert len(differences)==6 and np.isfinite(differences).all()
    mean=float(differences.mean())
    sd=float(differences.std(ddof=1))
    assignments=2*((np.arange(64)[:,None]>>np.arange(6))&1)-1
    null=(assignments*differences).mean(1)
    exact_p=float(np.mean(np.abs(null)>=abs(mean)-1e-14))
    interval=[mean-float(stats.t.ppf(.975,5))*sd/np.sqrt(6),mean+float(stats.t.ppf(.975,5))*sd/np.sqrt(6)]
    independent=stats.permutation_test((np.asarray(greedy),np.asarray(weight)),lambda a,b:np.mean(a-b),
        permutation_type="samples",n_resamples=np.inf,vectorized=False,alternative="two-sided")
    assert abs(float(independent.pvalue)-exact_p)<1e-12
    if sd:
        ci=stats.ttest_rel(greedy,weight).confidence_interval(.95)
        assert np.max(np.abs(np.asarray(interval)-np.array([ci.low,ci.high])))<1e-12
    return dict(seeds=SEEDS,differences=differences.tolist(),mean_difference=mean,sd_difference=sd,
        t95_interval=interval,sign_flip_two_sided_p=exact_p,sign_assignments=64,
        scipy_p_check=float(independent.pvalue),null_statistics=null.tolist(),
        direction="negative favors greedy",unit="seed, conditional on fixed test inputs",
        assumptions="sign symmetry/exchangeability for sign-flip; normal approximation for t interval; n=6",
        inference_scope="single original primary contrast, layer1 alpha1 k16")


# 用途：把 CSV 的标签和数值字段恢复到整数/浮点类型，避免字符串比较。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：rows。
def numeric_rows(path):
    rows=csv_read(path)
    for r in rows:
        for key in ("seed","layer","k"):
            if key in r:r[key]=int(r[key])
        for key in ("alpha","nrmse","sse","effect_sse","seconds","input_sd_seconds","per_input_nrmse_p95"):
            if key in r:r[key]=float(r[key])
    return rows


# 用途：按教师、层、方法、预算和强度找到唯一对应记录。
# 参数：rows、seed、layer、method、k、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：next((r for r in rows if r['seed'] == seed and r['layer'] == layer and (r['method'] == …。
def matching(rows,seed,layer,method,k,alpha=1.):
    return next(r for r in rows if r["seed"]==seed and r["layer"]==layer and r["method"]==method and r["k"]==k and r["alpha"]==alpha)


# 用途：回读已保存数组重新评分并汇总，核对结果与既有记录的一致性。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def analyze(run):
    from run_external import verify_protocol, verify_freeze
    verify_protocol(run)
    lock=verify_freeze(run)
    completed=read_json(run / "test/completion.json")
    assert completed["status"]=="completed" and completed["seeds"]==SEEDS
    out=run / "analysis"
    out.mkdir(exist_ok=False)
    rows=numeric_rows(run / "test/all_results.csv")
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    greedy=[matching(rows,s,1,"greedy",16)["nrmse"] for s in SEEDS]
    weight=[matching(rows,s,1,"weight",16)["nrmse"] for s in SEEDS]
    primary=primary_statistics(greedy,weight)
    dump(out / "primary_statistics.json",primary)
    selected=[]
    for seed in SEEDS:
        for layer in (1,2):
            for method in METHODS:
                k=int(lock["validation_choices"][str(seed)][str(layer)][method])
                for alpha in (1.,.5):
                    r=matching(rows,seed,layer,method,k,alpha)
                    selected.append(dict(seed=seed,layer=layer,alpha=alpha,method=method,k=k,
                        nrmse=r["nrmse"],meets_10pct=r["nrmse"]<=.1,per_input_nrmse_p95=r["per_input_nrmse_p95"]))
    csv_write(out / "validation_selected_test.csv",selected)
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r["layer"],r["alpha"],r["method"],r["k"])].append(r)
    summaries=[]
    for key,group in sorted(grouped.items()):
        values=np.array([r["nrmse"] for r in group])
        summaries.append(dict(zip(("layer","alpha","method","k"),key),seeds=len(group),
            nrmse_mean=float(values.mean()),nrmse_sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
            nrmse_min=float(values.min()),nrmse_max=float(values.max()),seeds_meet_10pct=int((values<=.1).sum())))
    csv_write(out / "test_summary.csv",summaries)
    selected_summary=[]
    for layer in (1,2):
        for method in METHODS:
            for alpha in (1.,.5):
                group=[r for r in selected if r["layer"]==layer and r["method"]==method and r["alpha"]==alpha]
                values=np.array([r["nrmse"] for r in group])
                selected_summary.append(dict(layer=layer,method=method,alpha=alpha,nrmse_mean=float(values.mean()),
                    nrmse_sd=float(values.std(ddof=1)),seeds_meet_10pct=int((values<=.1).sum()),
                    k_values=";".join(str(r["k"]) for r in group),
                    max_seed_input_p95=float(max(r["per_input_nrmse_p95"] for r in group))))
    csv_write(out / "selected_summary.csv",selected_summary)
    timing=[]
    for seed in SEEDS:timing.extend(numeric_rows(run / f"seed_{seed}/timing/seed_summary.csv"))
    # 用途：按指定设置查找已经测量的延时，不在统计阶段重新计时。
    # 参数：seed、layer、method、k、q、workload；具体形状/来源结合调用处及下方关键步骤阅读。
    # 返回：next((r['seconds'] for r in timing if r['seed'] == seed and r['layer'] == layer and (r[…。
    def latency(seed,layer,method,k,q,workload):
        # Full-budget methods literally use the exact runtime: share its cost observation.
        if k==64:method="exact"
        return next(r["seconds"] for r in timing if r["seed"]==seed and r["layer"]==layer
                    and r["method"]==method and r["k"]==k and int(r["queries"])==q and r["workload"]==workload)
    cost_results=[]
    for seed in SEEDS:
        dev=read_json(run / f"seed_{seed}/development_complete.json")
        for layer in (1,2):
            details=next(r for r in dev["layers"] if r["layer"]==layer)
            for method in METHODS:
                for rule in ("fixed16","fixed32","validation_selected"):
                    k=int(rule[5:]) if rule.startswith("fixed") else int(lock["validation_choices"][str(seed)][str(layer)][method])
                    accuracy=matching(rows,seed,layer,method,k)["nrmse"]
                    for q in (37,256):
                        for workload in ("from_image","from_source","cached"):
                            seconds=latency(seed,layer,method,k,q,workload)
                            exact=latency(seed,layer,"exact",64,q,workload)
                            saving=exact-seconds
                            # Observed joint development includes all controls, validation and I/O.
                            # It is an explicitly named nonminimal experimental pipeline cost.
                            amortization=int(np.ceil(details["elapsed_seconds"]/saving)) if saving>1e-12 and k<64 else None
                            cost_results.append(dict(seed=seed,layer=layer,method=method,rule=rule,k=k,queries=q,
                                workload=workload,nrmse=accuracy,meets_test_10pct=accuracy<=.1,
                                seconds=seconds,exact_seconds=exact,speedup=exact/seconds,
                                background_scalars=details["boundary_scalars"]+k,
                                exact_background_scalars=details["boundary_scalars"]+64,
                                background_reduction=(64-k)/(details["boundary_scalars"]+64),
                                joint_development_seconds=details["elapsed_seconds"],
                                joint_pipeline_amortization_input_bundles=amortization))
    csv_write(out / "cost_and_accuracy_seeds.csv",cost_results)
    cost_group=defaultdict(list)
    for r in cost_results:cost_group[(r["layer"],r["method"],r["rule"],r["queries"],r["workload"])].append(r)
    cost_summary=[]
    for key,group in sorted(cost_group.items()):
        sp=np.array([r["speedup"] for r in group])
        cost_summary.append(dict(zip(("layer","method","rule","queries","workload"),key),
            milliseconds_mean=float(np.mean([r["seconds"] for r in group])*1000),
            exact_milliseconds_mean=float(np.mean([r["exact_seconds"] for r in group])*1000),
            paired_speedup_mean=float(sp.mean()),paired_speedup_sd=float(sp.std(ddof=1)),
            seeds_meet_10pct=sum(r["meets_test_10pct"] for r in group),
            seeds_meet_error_and_faster=sum(r["meets_test_10pct"] and r["speedup"]>1 for r in group),
            background_reduction_mean=float(np.mean([r["background_reduction"] for r in group]))))
    csv_write(out / "cost_summary.csv",cost_summary)
    # Independently read saved arrays; do not use the data loader or train anything.
    held=bit_design(8).sum(1)>2
    max_error=0.;rescored=0;commit_order=[]
    for seed in SEEDS:
        for layer in (1,2):
            case=run / f"test/seed_{seed}_L{layer}"
            complete=read_json(case / "completion.json")
            # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
            assert sha(case / "predictions.npz")==complete["prediction_sha256"]
            assert sha(case / "references.npz")==complete["reference_sha256"]
            commitment=read_json(case / "prediction_commitment.json")
            reference_commitment=read_json(case / "reference_commitment.json")
            assert reference_commitment["sha256"]==complete["reference_sha256"]
            commit_order.append(commitment["utc"]<reference_commitment["utc"]<=complete["utc"])
            with np.load(case / "predictions.npz",allow_pickle=False) as pp, np.load(case / "references.npz",allow_pickle=False) as rr:
                assert np.array_equal(pp["ids"],rr["ids"]) and np.array_equal(pp["ids"],np.arange(256))
                for row in [r for r in rows if r["seed"]==seed and r["layer"]==layer]:
                    pred=pp[prediction_key(row["method"],row["k"],row["alpha"])];target=rr[str(row["alpha"])]
                    value=np.sqrt(np.square((pred-target)[:,held]).sum()/max(np.square(target[:,held]).sum(),1e-30))
                    max_error=max(max_error,abs(float(value)-row["nrmse"]));rescored+=1
    assert max_error<1e-12 and all(commit_order)
    verify_freeze(run)
    checks=dict(status="passed",rescored_arrays=rescored,max_nrmse_difference=max_error,
        all_prediction_commits_before_reference_commits=all(commit_order),all_frozen_artifacts_unchanged=True,
        primary_scipy_check=True,scipy_version=scipy.__version__,calibration_model_access=False)
    dump(out / "verification.json",checks)
    print(primary)
    print(checks)


# 用途：汇总原六教师主实验，重算配对统计，并复核保存预测的评分。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=BASE / "runs/fashion_external_20260911")
    args=parser.parse_args()
    analyze(args.run_dir)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

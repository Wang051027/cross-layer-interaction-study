# 中文阅读说明｜用途：在同一 CPU 后端和边界计算下，比较不同背景处理方法的运行成本。
# 数据流：输入冻结局部模块；输出准备、查询、缓存等场景的计时与成本统计。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Descriptive matched-backend diagnostic, separate from historical timings."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone

from research_runtime import (BASE, UPSTREAM, Fit, LocalModule, bit_design, low_order,
                              make_teacher, source_activation, HeadRuntime, np, torch)
from threadpoolctl import threadpool_limits


# 用途：用统一字段顺序写出多行 CSV 记录。
# 参数：path、rows；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 用途：在同一 CPU 后端和边界计算下，比较不同背景处理方法的运行成本。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    threadpool_limits(1)
    output = BASE / "runs" / "pca_cost_audit_20260910_v2"
    output.mkdir(parents=True, exist_ok=False)
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  purpose="post-hoc implementation diagnostic on historically explored development data",
                  methods=["greedy", "pca_fused", "pca_unfused", "exact"],
                  k="J/4", teachers=12, layers=[1,2], queries=[37,256],
                  inputs_per_teacher_layer=4, input_selection="first four historical held-out inputs",
                  rounds=15, inner_repetitions=3, warmup=3,
                  method_order="random permutation in every round, seed 20260910",
                  device="CPU", dtype="float64", threads=1,
                  workloads={
                      "from_source":"source prefix already computed; prepare boundary/head and answer Q queries",
                      "from_image":"raw image; source prefix plus prepare plus Q queries",
                      "cached":"h, boundary windows and full z already cached; select/encode and answer Q queries; preceding full teacher pass excluded"},
                  no_dynamic_fallback=True,
                  code_sha256={name:hashlib.sha256((BASE / "scripts" / name).read_bytes()).hexdigest()
                               for name in ("benchmark_heads.py", "research_runtime.py")},
                  exact_cached_implementation="reuse full cached z directly, without redundant gather/copy",
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    (output / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rng = np.random.default_rng(20260910)
    raw, costs = [], []
    started = time.perf_counter()
    for dataset in ("synthetic", "digits"):
        data = np.load(UPSTREAM / "data" / f"{dataset}_evaluation.npz", allow_pickle=False)
        for seed in range(26090511,26090517):
            model = make_teacher(dataset).double().eval()
            model.load_state_dict(torch.load(UPSTREAM / "models" / f"{dataset}_teacher_seed_{seed}.pt",
                                            weights_only=True, map_location="cpu"))
            for layer in (1,2):
                tag = f"{dataset}_{seed}_L{layer}"
                saved = np.load(UPSTREAM / "models" / f"background_fit_{tag}.npz", allow_pickle=False)
                fit = Fit(**{key:saved[key] for key in Fit.__annotations__})
                images = torch.tensor(data["x_test"][:4], dtype=torch.float64)
                sources = source_activation(model, images, layer)
                mod = LocalModule(model, tuple(sources.shape[1:]), saved["nodes"], layer)
                runtimes = {method:HeadRuntime(mod, fit, method, len(fit.gate) if method=="exact" else len(fit.gate)//4)
                            for method in config["methods"]}
                for method, runtime in runtimes.items():
                    costs.append(dict(dataset=dataset, seed=seed, layer=layer, method=method,
                                      **runtime.cost_counts()))
                for i in range(4):
                    image, source = images[i:i+1], sources[i:i+1]
                    context = mod.prepare(source)
                    common = context["h"], context["window"], context["z3"]
                    for q in (37,256):
                        bits = bit_design(8)
                        if q == 37:
                            bits = bits[low_order(8)[0]]
                        bits = torch.tensor(bits, dtype=torch.float64)
                        for workload in config["workloads"]:
                            if workload == "from_source":
                                functions = {m:(lambda rt=rt:rt.evaluate(rt.prepare(source),bits)) for m,rt in runtimes.items()}
                            elif workload == "from_image":
                                functions = {m:(lambda rt=rt:rt.evaluate(rt.prepare(source_activation(model,image,layer)),bits)) for m,rt in runtimes.items()}
                            else:
                                functions = {m:(lambda rt=rt:rt.evaluate(rt.prepare_cached(common),bits)) for m,rt in runtimes.items()}
                            for fn in functions.values():
                                for _ in range(config["warmup"]):
                                    fn()
                            for round_index in range(config["rounds"]):
                                for method in rng.permutation(config["methods"]):
                                    fn = functions[method]
                                    start = time.perf_counter_ns()
                                    for _ in range(config["inner_repetitions"]):
                                        fn()
                                    elapsed = (time.perf_counter_ns()-start) / config["inner_repetitions"] / 1e9
                                    raw.append(dict(dataset=dataset,seed=seed,layer=layer,input_index=i,
                                                    queries=q,workload=workload,method=method,
                                                    round=round_index,seconds=elapsed))
                print(json.dumps(dict(finished=tag,elapsed_seconds=time.perf_counter()-started)), flush=True)
    write_csv(output / "timing_raw.csv", raw)
    write_csv(output / "cost_counts.csv", costs)
    by_input = defaultdict(list)
    for r in raw:
        key = tuple(r[k] for k in ("dataset","seed","layer","queries","workload","method","input_index"))
        by_input[key].append(r["seconds"])
    by_seed = defaultdict(list)
    for key, values in by_input.items():
        by_seed[key[:-1]].append(float(np.median(values)))
    seed_rows = [dict(zip(("dataset","seed","layer","queries","workload","method"),key),
                      seconds=float(np.mean(values))) for key,values in by_seed.items()]
    write_csv(output / "timing_seeds.csv", seed_rows)
    grouped = defaultdict(list)
    for row in seed_rows:
        exact = next(r["seconds"] for r in seed_rows if r["dataset"]==row["dataset"] and r["seed"]==row["seed"]
                     and r["layer"]==row["layer"] and r["queries"]==row["queries"]
                     and r["workload"]==row["workload"] and r["method"]=="exact")
        grouped[tuple(row[k] for k in ("dataset","layer","queries","workload","method"))].append((row["seconds"],exact/row["seconds"]))
    summary=[]
    for key, values in grouped.items():
        values=np.array(values)
        summary.append(dict(zip(("dataset","layer","queries","workload","method"),key),
                            ms_mean=float(values[:,0].mean()*1000),ms_seed_sd=float(values[:,0].std(ddof=1)*1000),
                            paired_speedup_mean=float(values[:,1].mean()),paired_speedup_sd=float(values[:,1].std(ddof=1))))
    write_csv(output / "timing_summary.csv",summary)
    (output / "completion.json").write_text(json.dumps(dict(status="completed",raw_rows=len(raw),
        elapsed_seconds=time.perf_counter()-started, statistical_claim="descriptive on this machine, not a speed guarantee"),indent=2),encoding="utf-8")


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

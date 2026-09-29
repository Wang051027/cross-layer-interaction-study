# 中文阅读说明｜用途：复算用户提供的 v0.7 模型与组合响应，检查导入后是否保持原结果。
# 数据流：输入原包模型、数据和拟合参数；输出数组一致性与指标复核记录。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Re-score/replay v0.7 without modifying its imported archive contents."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from research_runtime import (BASE, UPSTREAM, Fit, LocalModule, bit_design, low_order,
                              make_teacher, source_activation, predict, fit_module,
                              HeadRuntime, score, np, torch)
from threadpoolctl import threadpool_limits


# 用途：用统一字段顺序写出多行 CSV 记录。
# 参数：path、rows；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 用途：复算用户提供的 v0.7 模型与组合响应，检查导入后是否保持原结果。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(1)
    output = BASE / "runs" / "upstream_replay_20260910_v2"
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    bits = bit_design(8)
    # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
    held = bits.sum(1) > 2
    low = low_order(8)[0]
    rows, audit = [], []
    for dataset in ("synthetic", "digits"):
        data = np.load(UPSTREAM / "data" / f"{dataset}_evaluation.npz", allow_pickle=False)
        for seed in range(26090511, 26090517):
            model = make_teacher(dataset).double().eval()
            model.load_state_dict(torch.load(UPSTREAM / "models" / f"{dataset}_teacher_seed_{seed}.pt",
                                            map_location="cpu", weights_only=True), strict=True)
            for layer in (1, 2):
                tag = f"{dataset}_{seed}_L{layer}"
                fp = np.load(UPSTREAM / "models" / f"background_fit_{tag}.npz", allow_pickle=False)
                fit = Fit(**{key: fp[key] for key in Fit.__annotations__})
                with torch.no_grad():
                    sources = source_activation(model, torch.tensor(data["x_test"], dtype=torch.float64), layer)
                    train_sources = source_activation(model, torch.tensor(data["x_train"], dtype=torch.float64), layer)
                mod = LocalModule(model, tuple(sources.shape[1:]), fp["nodes"], layer)
                contexts = [mod.prepare(s[None]) for s in sources]
                train_contexts = [mod.prepare(s[None]) for s in train_sources]
                refit = fit_module(mod, train_contexts, bits[low], seed + layer * 101)
                refit_order_matches = bool(np.array_equal(refit.order, fit.order))
                # SVD sign is not identifiable: compare projectors, not signed vectors.
                quarter = len(fit.gate) // 4
                V, Vnew = fit.pca_components[:quarter], refit.pca_components[:quarter]
                fit_error = max(float(np.abs(refit.gate-fit.gate).max()),
                                float(np.abs(V.T @ V - Vnew.T @ Vnew).max()))
                committed = np.load(UPSTREAM / "results" / f"committed_predictions_{tag}.npz", allow_pickle=False)
                reference = np.load(UPSTREAM / "results" / f"reference_{tag}.npz", allow_pickle=False)
                with (UPSTREAM / "tables" / f"rows_{tag}.csv").open(newline="") as f:
                    old = list(csv.DictReader(f))
                selected_k = json.loads((UPSTREAM / "logs" / f"complete_{tag}.json").read_text())["selected_k"]
                err = {key: 0.0 for key in ("rescore", "prediction", "refit_primary", "full_tail", "exact_local",
                                             "pca_fusion", "pca_runtime", "omitted_poison")}
                for name, method, k in (("fixed_quarter", "greedy", quarter),
                                        ("random_quarter", "random", quarter),
                                        ("validation_selected", "greedy", selected_k)):
                    original_score = next(r for r in old if r["method"] == name and float(r["alpha"]) == 1)
                    reread = score(committed[name], reference["full"], held)
                    err["rescore"] = max(err["rescore"], abs(reread["nrmse"] - float(original_score["nrmse"])))
                    replay = np.stack([predict(mod, fit, ctx, bits, method, k) for ctx in contexts])
                    err["prediction"] = max(err["prediction"], float(np.abs(replay - committed[name]).max()))
                    if method == "greedy":
                        new_prediction = np.stack([predict(mod, refit, ctx, bits, method, k) for ctx in contexts])
                        err["refit_primary"] = max(err["refit_primary"], float(np.abs(new_prediction-replay).max()))
                fused = HeadRuntime(mod, fit, "pca_fused", quarter)
                unfused = HeadRuntime(mod, fit, "pca_unfused", quarter)
                bt = torch.tensor(bits, dtype=torch.float64)
                for alpha, key in ((1., "full"), (.5, "half")):
                    # All held-out contexts, every coalition, actual original tail execution.
                    actual = np.stack([mod.full_evaluate(ctx, bt, alpha).numpy() for ctx in contexts])
                    err["full_tail"] = max(err["full_tail"], float(np.abs(actual-reference[key]).max()))
                    exact = np.stack([mod.evaluate(ctx, bt, alpha).numpy() for ctx in contexts])
                    err["exact_local"] = max(err["exact_local"], float(np.abs(exact-actual).max()))
                    pca = np.stack([fused.evaluate(fused.prepare(s[None]), bt, alpha).numpy() for s in sources])
                    pca_unfused = np.stack([unfused.evaluate(unfused.prepare(s[None]), bt, alpha).numpy() for s in sources])
                    pca_numpy = np.stack([predict(mod, fit, ctx, bits, "pca", quarter, alpha) for ctx in contexts])
                    err["pca_fusion"] = max(err["pca_fusion"], float(np.abs(pca-pca_unfused).max()))
                    err["pca_runtime"] = max(err["pca_runtime"], float(np.abs(pca-pca_numpy).max()))
                    for method in ("greedy", "random", "weight", "pca_fused"):
                        prediction = pca if method == "pca_fused" else np.stack([
                            predict(mod, fit, ctx, bits, method, quarter, alpha) for ctx in contexts])
                        rows.append(dict(dataset=dataset, seed=seed, layer=layer, alpha=alpha,
                                         method=method, k=quarter, J=len(fit.gate),
                                         **score(prediction, actual, held)))
                corrupted = dict(contexts[0])
                corrupted["z3"] = contexts[0]["z3"].clone()
                dropped = np.ones(len(fit.gate), bool)
                dropped[fit.order[:quarter]] = False
                corrupted["z3"][dropped] = 10000.
                err["omitted_poison"] = float(np.abs(
                    predict(mod, fit, corrupted, bits, "greedy", quarter)
                    - predict(mod, fit, contexts[0], bits, "greedy", quarter)).max())
                different = np.flatnonzero(fit.order != refit.order)
                different_gain_max = max(float(np.abs(fit.train_gain[different]).max(initial=0)),
                                         float(np.abs(refit.train_gain[different]).max(initial=0)))
                check = dict(tag=tag, inputs=len(contexts), refit_order_matches=refit_order_matches,
                             refit_different_order_positions=different.tolist(),
                             refit_different_order_gain_max=different_gain_max,
                             refit_gate_and_projector_max_error=fit_error, **err)
                audit.append(check)
                # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
                (output / "progress.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
                write_csv(output / "quarter_results.csv", rows)
                print(json.dumps(check), flush=True)
                assert max(err.values()) < 1e-10, check
                assert fit_error < 1e-9, check
                assert different_gain_max < 1e-10 * max(1., float(np.abs(fit.train_gain).max())), check
    write_csv(output / "quarter_results.csv", rows)
    summary = []
    for dataset in ("digits", "synthetic"):
        for layer in (1, 2):
            for method in ("greedy", "random", "weight", "pca_fused"):
                subset = [r for r in rows if r["dataset"] == dataset and r["layer"] == layer
                          and r["method"] == method and r["alpha"] == 1.]
                values = [r["nrmse"] for r in subset]
                summary.append(dict(dataset=dataset, layer=layer, method=method,
                                    nrmse_mean=float(np.mean(values)), nrmse_sd=float(np.std(values, ddof=1))))
    write_csv(output / "quarter_summary.csv", summary)
    # Extraction timestamps do not establish when upstream predictions were committed.
    # We verify payload hashes, not historical chronological independence.
    unchanged = all(hashlib.sha256((UPSTREAM / line.split(None, 1)[1].strip()).read_bytes()).hexdigest()
                    == line.split()[0] for line in (UPSTREAM / "MANIFEST.sha256").read_text().splitlines())
    result = dict(status="passed", utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-begin,
                  model_layer_runs=len(audit), teachers=12, cached_primary_arrays_rescored=72,
                  full_tail_intervened_outputs=24 * 128 * 256 * 2,
                  upstream_manifest_still_matches=unchanged,
                  maximum_errors={key:max(row[key] for row in audit) for key in err},
                  environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                   device="CPU", dtype="float64", threads=1),
                  timing_replicated=False, new_external_teachers=0,
                  chronology="ZIP hashes cannot independently attest upstream pre-reference commit timing",
                  checks=audit)
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert unchanged
    print(json.dumps({key:value for key,value in result.items() if key!="checks"}, indent=2), flush=True)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

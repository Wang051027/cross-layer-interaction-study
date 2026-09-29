# 中文阅读说明｜用途：集中定义原六教师实验的数据处理、训练、局部预测与公平计时函数。
# 数据流：其他调度脚本调用本文件；模型权重、选择顺序与预算必须由开发阶段确定。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Shared, frozen implementation for the six-seed external experiment.

All model/method choices use training or validation data. Numerical estimators
come from the audited v0.7 code. Full-budget timing uses the exact implementation
for all methods, since all J states then give the same mathematical response.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from research_runtime import (BASE, UPSTREAM, Fit, HeadRuntime, LocalModule, make_teacher,
    fit_module, predict, source_activation, bit_design, low_order, score, np, torch, F)
from threadpoolctl import threadpool_limits

PILOT = BASE / "runs/fashion_pilot_20260910"
DATA = BASE / "data/fashion_mnist/prepared"
SEEDS = list(range(202609071, 202609077))
GRID = [0, 1, 2, 4, 8, 16, 32, 64]
METHODS = ["greedy", "weight", "pca_fused"]


# 用途：生成当前 UTC 时间字符串，供协议、日志或交付记录标记时间。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：datetime.now(timezone.utc).isoformat()。
def now():
    return datetime.now(timezone.utc).isoformat()


# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.sha256(Path(path).read_bytes()).hexdigest()。
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 用途：读取 UTF-8 JSON，返回其中的协议、配置或结果对象。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(Path(path).read_text(encoding='utf-8'))。
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path, value):
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


# 用途：按记录字段写 CSV；不同实验阶段各自规定缺失数值的写法。
# 参数：path、rows；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 用途：把 CSV 的每行读为字典；需要数值类型的字段由调用方转换。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(f))。
def csv_read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# 用途：追加带时间的执行事件，记录阶段进度及已完成的结果。
# 参数：run、kind、**values；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def event(run, kind, **values):
    record = {"utc":now(), "event":kind, **values}
    with (run / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


# 用途：固定 CPU 线程和确定性设置，使不同方法在统一后端条件下运行。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def configure_cpu():
    torch.set_num_threads(1)
    threadpool_limits(1)
    torch.use_deterministic_algorithms(True)


# 用途：把图像像素转为浮点并除以 255，同时补上单通道维度。
# 参数：images；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：images[:, None].astype(np.float32) / np.float32(255)。
def normalized(images):
    return images[:, None].astype(np.float32) / np.float32(255)


# 用途：在 eval/no_grad 模式批量计算验证交叉熵与准确率，用于选择检查点。
# 参数：model、x、y；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(float(loss.item() / len(x)), float(correct.item() / len(x)))。
def teacher_validation(model, x, y):
    model.eval()
    loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    with torch.no_grad():
        for start in range(0, len(x), 512):
            z = model(x[start:start+512])
            target = y[start:start+512]
            # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
            loss += F.cross_entropy(z, target, reduction="sum")
            correct += (z.argmax(1) == target).sum()
    return float(loss.item()/len(x)), float(correct.item()/len(x))


# 用途：按配置训练当前教师，用验证集选择检查点，保存训练曲线与摘要。
# 参数：run、seed、x、y、indices、config；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：None。
def train_teacher(run, seed, x, y, indices, config):
    directory = run / f"seed_{seed}"
    if (directory / "teacher_summary.json").exists():
        summary = read_json(directory / "teacher_summary.json")
        # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
        assert sha(directory / "teacher.pt") == summary["checkpoint_sha256"]
        return
    if (directory / "teacher.pt").exists():
        raise RuntimeError("Incomplete teacher run exists; preserve and inspect it before retry")
    directory.mkdir(exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = make_teacher("fashion").cuda()
    xt = torch.tensor(normalized(x[indices["train"]]), device="cuda", dtype=torch.float32)
    yt = torch.tensor(y[indices["train"]], device="cuda", dtype=torch.long)
    xv = torch.tensor(normalized(x[indices["validation"]]), device="cuda", dtype=torch.float32)
    yv = torch.tensor(y[indices["validation"]], device="cuda", dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    # 等待 GPU 完成已提交任务，避免计时只统计异步调度开销。
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, config["max_epochs"]+1):
        model.train()
        order = torch.randperm(len(xt), generator=generator).cuda()
        total = torch.zeros((), device="cuda")
        with torch.enable_grad():
            for begin in range(0, len(xt), config["batch"]):
                ix = order[begin:begin+config["batch"]]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xt[ix]), yt[ix])
                # 这里才会计算训练梯度并更新模型；后续干预阶段固定模型权重。
                loss.backward()
                optimizer.step()
                total += loss.detach()*len(ix)
        val_loss, val_accuracy = teacher_validation(model, xv, yv)
        row = dict(epoch=epoch, train_cross_entropy=float(total.item()/len(xt)),
                   validation_cross_entropy=val_loss, validation_accuracy=val_accuracy)
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            # 把当前选中的模型参数保存为检查点，后续干预从该固定权重载入。
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, directory / "teacher.pt")
        else:
            stale += 1
        dump(directory / "training_history.json", history)
        event(run, "teacher_epoch", seed=seed, **row, best_epoch=best_epoch)
        if stale >= config["early_stopping_patience"]:
            break
    torch.cuda.synchronize()
    summary = dict(seed=seed, best_epoch=best_epoch, best_validation_cross_entropy=best_loss,
        best_validation_accuracy=history[best_epoch-1]["validation_accuracy"], epochs=len(history),
        seconds=time.perf_counter()-start, checkpoint_sha256=sha(directory / "teacher.pt"),
        reused_pilot=False, test_used_for_selection=False)
    dump(directory / "teacher_summary.json", summary)
    event(run, "teacher_finished", **summary)
    del model, optimizer, xt, yt, xv, yv
    torch.cuda.empty_cache()


# 用途：载入冻结的 CNN 权重并进入评价模式，供干预与计时复用。
# 参数：directory；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：model。
def load_model(directory):
    model = make_teacher("fashion").double().eval()
    model.load_state_dict(torch.load(directory / "teacher.pt", weights_only=True, map_location="cpu"), strict=True)
    return model


# 用途：回读保存的背景统计、排序或 PCA 参数，恢复拟合结果对象。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(Fit(**{name: saved[name] for name in Fit.__annotations__}), saved['nodes'])。
def load_fit(path):
    saved = np.load(path, allow_pickle=False)
    return Fit(**{name:saved[name] for name in Fit.__annotations__}), saved["nodes"]


# 用途：在固定空间窗口内按训练样本的平均激活选择源坐标，保持并列时的索引顺序。
# 参数：sources；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：central[np.argsort(-means[central], kind='stable')[:8]]。
def choose_nodes(sources):
    _, channels, height, width = sources.shape
    ids = np.arange(channels*height*width).reshape(channels, height, width)
    central = ids[:,height//2-1:height//2+2,width//2-1:width//2+2].ravel()
    means = sources.mean(0).numpy().ravel()
    return central[np.argsort(-means[central], kind="stable")[:8]]


# 用途：列出事先固定的方法与预算组合，供一致的验证和测试比较。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：[(method, k) for method in ('greedy', 'weight', 'random', 'pca_fused') for k in (16, 32…。
def fixed_specs():
    return [(method,k) for method in ("greedy","weight","random","pca_fused") for k in (16,32)] + [
        ("boundary_only",0),("mean_head",0),("pool_ridge",0),("exact",64)]


# 用途：列出用于验证阶段选择预算的候选配置。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：sorted(set(fixed_specs() + [(method, k) for method in METHODS for k in GRID]))。
def validation_specs():
    return sorted(set(fixed_specs() + [(method,k) for method in METHODS for k in GRID]))


# 用途：把开发阶段已经做出的选择转换为固定测试配置。
# 参数：choices；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：sorted(set(fixed_specs() + [(method, int(choices[method])) for method in METHODS]))。
def test_specs(choices):
    return sorted(set(fixed_specs() + [(method,int(choices[method])) for method in METHODS]))


# 用途：把方法、预算和强度组合成稳定的数组名称，避免不同设置混淆。
# 参数：method、k、alpha；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：f'{method}_k{k}_a{alpha}'。
def prediction_key(method, k, alpha):
    return f"{method}_k{k}_a{alpha}"


# 用途：用同一批原始上下文评估所有候选方法，同时核对重新获取与缓存路径一致。
# 参数：mod、fit、sources、specs、alphas；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(outputs, contexts, dict(fresh_cached_max_error=max_path_error, legacy_numpy_max_error=…。
def prepare_predictions(mod, fit, sources, specs, alphas=(1., .5)):
    """Use cached unperturbed states for accuracy; verify the fresh acquisition path."""
    bits = bit_design(8)
    bt = torch.tensor(bits, dtype=torch.float64)
    contexts = [mod.prepare(s[None]) for s in sources]
    common = [(c["h"], c["window"], c["z3"]) for c in contexts]
    outputs = {}
    max_path_error, max_numpy_error = 0., 0.
    for method,k in specs:
        runtime = HeadRuntime(mod, fit, method, k)
        prepared = [runtime.prepare_cached(c) for c in common]
        for alpha in alphas:
            value = np.stack([runtime.evaluate(c, bt, alpha).numpy() for c in prepared])
            outputs[prediction_key(method,k,alpha)] = value
            for i in range(min(3,len(sources))):
                fresh = runtime.evaluate(runtime.prepare(sources[i:i+1]),bt,alpha).numpy()
                max_path_error = max(max_path_error,float(np.abs(fresh-value[i]).max()))
                legacy_method = "pca" if method=="pca_fused" else method
                legacy = predict(mod,fit,contexts[i],bits,legacy_method,k,alpha)
                max_numpy_error = max(max_numpy_error,float(np.abs(legacy-value[i]).max()))
            assert np.isfinite(value).all()
            assert np.max(np.abs(value[:,0])) < 1e-10
    assert max(max_path_error,max_numpy_error)<1e-9
    return outputs, contexts, dict(fresh_cached_max_error=max_path_error,legacy_numpy_max_error=max_numpy_error)


# 用途：在固定留出组合上给保存预测评分，并记录输入和设置标签。
# 参数：predictions、references、specs、seed、layer、ids；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(rows, per_input)。
def score_predictions(predictions, references, specs, seed, layer, ids):
    held = bit_design(8).sum(1)>2
    rows, per_input = [], []
    for alpha in (1., .5):
        target = references[str(alpha)]
        denominator = np.square(target[:,held]).sum((1,2))
        positive = denominator>1e-24
        for method,k in specs:
            prediction = predictions[prediction_key(method,k,alpha)]
            metric = score(prediction,target,held)
            squared_error = np.square((prediction-target)[:,held]).sum((1,2))
            relative = np.sqrt(squared_error[positive]/denominator[positive])
            rows.append(dict(seed=seed,layer=layer,alpha=alpha,method=method,k=k,**metric,
                per_input_nrmse_p50=float(np.median(relative)) if len(relative) else 0.,
                per_input_nrmse_p95=float(np.quantile(relative,.95)) if len(relative) else 0.,
                near_zero_effect_inputs=int((~positive).sum()), n_inputs=len(ids), n_coalitions=219))
            for i, input_id in enumerate(ids):
                per_input.append(dict(seed=seed,layer=layer,alpha=alpha,method=method,k=k,
                    input_id=int(input_id),sse=float(squared_error[i]),effect_sse=float(denominator[i])))
    return rows,per_input


# 用途：对一个教师完成训练/验证上下文拟合、预算选择及开发记录保存。
# 参数：run、seed、x、context_ids；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：None。
def develop_seed(run, seed, x, context_ids):
    directory = run / f"seed_{seed}"
    if (directory / "development_complete.json").exists():
        return
    model = load_model(directory)
    tx = torch.tensor(normalized(x[context_ids["train"]]),dtype=torch.float64)
    vx = torch.tensor(normalized(x[context_ids["validation"]]),dtype=torch.float64)
    all_rows, summaries, all_choices = [], [], {}
    specs = validation_specs()
    for layer in (1,2):
        start = time.perf_counter()
        with torch.no_grad():
            train_sources = source_activation(model,tx,layer)
            val_sources = source_activation(model,vx,layer)
        nodes = choose_nodes(train_sources)
        mod = LocalModule(model,tuple(train_sources.shape[1:]),nodes,layer)
        training_contexts = [mod.prepare(s[None]) for s in train_sources]
        context_seconds = time.perf_counter()-start
        fit_start = time.perf_counter()
        refit = fit_module(mod,training_contexts,bit_design(8)[low_order(8)[0]],seed+layer*101)
        fit_seconds = time.perf_counter()-fit_start
        first_refit_error = 0.
        if seed==SEEDS[0]:
            fit, saved_nodes = load_fit(PILOT / f"background_fit_L{layer}.npz")
            assert np.array_equal(nodes,saved_nodes)
            for k in (16,32):
                for c in training_contexts[:3]:
                    first_refit_error = max(first_refit_error,float(np.abs(
                        predict(mod,fit,c,bit_design(8),"greedy",k) -
                        predict(mod,refit,c,bit_design(8),"greedy",k)).max()))
            assert first_refit_error<1e-9
        else:
            fit = refit
        fit_path = directory / f"fit_L{layer}.npz"
        # 保存具名数值数组，便于后来按同一索引回读和独立复算。
        np.savez_compressed(fit_path,**fit.__dict__,nodes=nodes,layer=layer)
        predictions,contexts,path_check = prepare_predictions(mod,fit,val_sources,specs)
        pred_path = directory / f"validation_predictions_L{layer}.npz"
        np.savez_compressed(pred_path,**predictions,ids=context_ids["validation"])
        event(run,"validation_predictions_committed",seed=seed,layer=layer,sha256=sha(pred_path))
        if seed==SEEDS[0]:
            with np.load(PILOT / f"validation_references_L{layer}.npz",allow_pickle=False) as saved:
                assert np.array_equal(saved["ids"],context_ids["validation"])
                references = {str(a):saved[str(a)].copy() for a in (1.,.5)}
        else:
            references = {str(a):np.stack([mod.full_evaluate(c,bit_design(8),a).numpy() for c in contexts]) for a in (1.,.5)}
        np.savez_compressed(directory / f"validation_references_L{layer}.npz",**references,ids=context_ids["validation"])
        rows, per_input = score_predictions(predictions,references,specs,seed,layer,context_ids["validation"])
        all_rows.extend(rows)
        choices = {}
        for method in METHODS:
            eligible = [r["k"] for r in rows if r["method"]==method and r["alpha"]==1. and r["nrmse"]<=.10]
            choices[method] = min(eligible) if eligible else 64
        all_choices[str(layer)] = choices
        exact_error = max(r["maxerr"] for r in rows if r["method"]=="exact")
        assert exact_error<1e-9
        summary = dict(seed=seed,layer=layer,choices=choices,boundary_scalars=4*len(mod.pool_ids),
            context_preparation_seconds=context_seconds,fit_all_controls_seconds=fit_seconds,
            full_exact_max_error=exact_error,first_teacher_refit_prediction_error=first_refit_error,
            fit_sha256=sha(fit_path),**path_check,elapsed_seconds=time.perf_counter()-start)
        summaries.append(summary)
        csv_write(directory / f"validation_per_input_L{layer}.csv",per_input)
        csv_write(directory / "validation_results.csv",all_rows)
        event(run,"validation_layer_finished",**summary)
        del predictions,references,training_contexts,contexts
    dump(directory / "validation_choices.json",all_choices)
    dump(directory / "development_complete.json",dict(utc=now(),seed=seed,layers=summaries,
        choices=all_choices,official_test_model_access=False))


# 用途：创建当前方法与预算的计时执行器，完整预算使用精确路径。
# 参数：mod、fit、method、k；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：HeadRuntime(mod, fit, 'exact', 64) / HeadRuntime(mod, fit, method, k)。
def timing_runtime(mod, fit, method, k):
    if k==64 and method in METHODS + ["random"]:
        return HeadRuntime(mod,fit,"exact",64)
    return HeadRuntime(mod,fit,method,k)


# 用途：对一个教师的固定验证输入做配对计时，比较查询量和状态获取场景。
# 参数：run、seed、x、context_ids；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：None。
def benchmark_seed(run, seed, x, context_ids):
    directory = run / f"seed_{seed}"
    out = directory / "timing"
    if (out / "completion.json").exists():
        return
    out.mkdir(exist_ok=False)
    model = load_model(directory)
    images = torch.tensor(normalized(x[context_ids["validation"][:4]]),dtype=torch.float64)
    choices = read_json(directory / "validation_choices.json")
    raw, costs, compile_rows = [], [], []
    rng = np.random.default_rng(20260911+seed)
    max_error = 0.
    for layer in (1,2):
        fit,nodes = load_fit(directory / f"fit_L{layer}.npz")
        with torch.no_grad():
            sources = source_activation(model,images,layer)
        mod = LocalModule(model,tuple(sources.shape[1:]),nodes,layer)
        specs = sorted(set([(method,k) for method in METHODS for k in (16,32)] +
                           [(method,int(choices[str(layer)][method])) for method in METHODS]+[("exact",64)]))
        runtimes = {}
        for method,k in specs:
            start = time.perf_counter()
            runtime = timing_runtime(mod,fit,method,k)
            compile_rows.append(dict(seed=seed,layer=layer,method=method,k=k,seconds=time.perf_counter()-start))
            runtimes[(method,k)] = runtime
            costs.append(dict(seed=seed,layer=layer,method=method,k=k,
                implementation=runtime.method,**runtime.cost_counts()))
        for i in range(4):
            image,source = images[i:i+1],sources[i:i+1]
            context = mod.prepare(source)
            common = context["h"],context["window"],context["z3"]
            for q in (37,256):
                bits = bit_design(8)
                if q==37:
                    bits=bits[low_order(8)[0]]
                bt = torch.tensor(bits,dtype=torch.float64)
                for (method,k),runtime in runtimes.items():
                    current = runtime.evaluate(runtime.prepare(source),bt).numpy()
                    wanted = predict(mod,fit,context,bits,"pca" if method=="pca_fused" else method,k)
                    max_error = max(max_error,float(np.abs(current-wanted).max()))
                for workload in ("from_image","from_source","cached"):
                    if workload=="from_image":
                        fs = {key:(lambda rt=rt:rt.evaluate(rt.prepare(source_activation(model,image,layer)),bt)) for key,rt in runtimes.items()}
                    elif workload=="from_source":
                        fs = {key:(lambda rt=rt:rt.evaluate(rt.prepare(source),bt)) for key,rt in runtimes.items()}
                    else:
                        fs = {key:(lambda rt=rt:rt.evaluate(rt.prepare_cached(common),bt)) for key,rt in runtimes.items()}
                    with torch.no_grad():
                        for fn in fs.values():
                            for _ in range(3):fn()
                        keys=list(fs)
                        for repeat in range(15):
                            for ix in rng.permutation(len(keys)):
                                method,k=keys[ix]
                                fn=fs[(method,k)]
                                start=time.perf_counter_ns()
                                for _ in range(3):fn()
                                elapsed=(time.perf_counter_ns()-start)/3/1e9
                                raw.append(dict(seed=seed,layer=layer,input_index=i,queries=q,
                                    workload=workload,method=method,k=k,round=repeat,seconds=elapsed))
        event(run,"timing_layer_finished",seed=seed,layer=layer,max_formula_difference=max_error)
    assert max_error<1e-9
    by_input=defaultdict(list)
    for r in raw:
        key=tuple(r[name] for name in ("seed","layer","queries","workload","method","k","input_index"))
        by_input[key].append(r["seconds"])
    grouped=defaultdict(list)
    for key,values in by_input.items():grouped[key[:-1]].append(float(np.median(values)))
    summary=[dict(zip(("seed","layer","queries","workload","method","k"),key),
        seconds=float(np.mean(values)),input_sd_seconds=float(np.std(values,ddof=1))) for key,values in grouped.items()]
    csv_write(out / "raw.csv",raw)
    csv_write(out / "seed_summary.csv",summary)
    csv_write(out / "cost_counts.csv",costs)
    csv_write(out / "runtime_construction.csv",compile_rows)
    dump(out / "completion.json",dict(utc=now(),seed=seed,rows=len(raw),max_formula_difference=max_error,
        selection_scope="validation only",training_concurrent=False))

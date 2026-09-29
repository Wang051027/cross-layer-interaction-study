# 中文阅读说明｜用途：验证稳定状态合并后的精确响应，并计入状态筛选和缓存的完整成本。
# 数据流：复用六个教师；新批次 q=4/8 完整枚举，q=12 全枚举另限四张验证输入。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Frozen exact-switch-screening validation, timing, and new intervention batch."""
import argparse
import hashlib
import shutil
import time
from types import SimpleNamespace
from collections import defaultdict
from switch_context import *
from run_scale_controls import file_sha, check_hashes, independent_reference, verify as verify_scale
from run_residual_controls import training_data, prefix
from prepare_external import load_idx

RUN = SWITCH_RUN
SOURCE_NAMES = ["scripts/switch_context.py", "scripts/run_switch_controls.py"]


# 用途：生成教师种子、空间位置和源坐标数对应的子目录名。
# 参数：seed、position、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：f'seed_{seed}/{position}_q{q}'。
def name_of(seed, position, q):
    return f"seed_{seed}/{position}_q{q}"


# 用途：读取本设置保存的组合掩码；全枚举与低阶/抽样集合由参数区分。
# 参数：run、q、query_set；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：bit_design(q) / saved['train' if query_set == 'low_order' else 'evaluation'].copy()。
def design(run, q, query_set="evaluation"):
    if query_set == "exhaustive":
        return bit_design(q)
    with np.load(SCALE_RUN/f"design_q{q}.npz", allow_pickle=False) as saved:
        return saved["train" if query_set == "low_order" else "evaluation"].copy()


# 用途：从前阶段保存的布局中读取源坐标，保持各阶段干预对象一致。
# 参数：run、seed、position、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：saved[f'{seed}_{position}_q{q}'].copy()。
def get_nodes(run, seed, position, q):
    with np.load(run/"layouts.npz", allow_pickle=False) as saved:
        return saved[f"{seed}_{position}_q{q}"].copy()


# 用途：创建该阶段的新运行目录，固定设计、输入索引和依赖文件哈希。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def init(run):
    verify_scale(SCALE_RUN, True)
    run.mkdir(exist_ok=False)
    prior = read_json(SCALE_RUN/"protocol.json")
    imported = dict(prior["imported_sources"])
    imported.update(read_json(SCALE_RUN/"implementation.json")["sources"])
    old_entries = {r["path"]: r["sha256"] for r in read_json(BASE/"audit/scale_manifest_20260914.json")["entries"]
                   if r["path"].startswith("runs/fashion_scale_20260914/")}
    check_hashes(BASE, old_entries)
    layouts = {}
    fits = {}
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        for position in POSITIONS:
            for q in SIZES:
                path = SCALE_RUN/name_of(seed, position, q)/"fit.npz"
                _, nodes = load_fit(path)
                layouts[f"{seed}_{position}_q{q}"] = nodes
                fits[path.relative_to(BASE).as_posix()] = file_sha(path)
    # 保存具名数值数组，便于后来按同一索引回读和独立复算。
    np.savez_compressed(run/"layouts.npz", **layouts)
    np.save(run/"intervention_ids.npy", np.arange(896, 1024))
    protocol = dict(utc=now(), version="exact-switch-screening-v1",
        purpose="Test exact correction only at states that can cross zero over all source subsets, including complete preparation cost",
        scope="Same final residual branch, six fixed teachers, two positions, three source sizes; no deeper-layer or new-dataset claim",
        prior_knowledge="Four intervention batches 0..895 and the whole classification pool have been accessed. This is a development-informed descriptive follow-up",
        seeds=SEEDS, sizes=SIZES, positions=list(POSITIONS), strengths=STRENGTHS,
        source_selection="Reuse all36 frozen source layouts from the scale study; no fitting or new model training",
        formula="Stable-on states form M=U diag(stable_on) W; add exact ReLU differences only on candidate states",
        screening=dict(domain="Every independent binary source mask, alpha in [0,1]; one cache covers both evaluated strengths",
            low="sum_i min(-W_ji*h_i,0)", high="sum_i max(-W_ji*h_i,0)",
            guard="64*float64_eps*(abs(z)+high-low+1)",
            stable_on="z+low>guard", stable_off="z+high<-guard", candidates="all remaining states, including boundary cases",
            guarantee="Algebraically exact in real arithmetic; ordinary float64 implementation checked numerically, not a formal outward-rounded certificate"),
        evaluation=dict(validation_contexts=128, validation_reference="Previously saved original unfused scale references, explicitly reused development evidence",
            new_test_contexts=128, new_test_ids="official896..1023", q4="all16 masks", q8="all256 masks",
            q12="Same fixed335 masks:79 low-order and256 sampled high-order, not exhaustive on the128 new inputs",
            supplementary="All4096 masks on first4 existing validation inputs, all6 teachers and2 positions; original unfused CUDA reference",
            smoke="Synthetic boundary/empty/all/zero-source cases plus original CPU/CUDA smoke on first2 validation inputs of first teacher",
            metric="Max absolute error on ALL queried outputs, NRMSE on order>2; zero-effect denominator threshold1e-24",
            numeric_tolerance=1e-9, no_new_confirmatory_p_values=True),
        timing=dict(inputs=4, input_source="first4 fixed validation contexts", threads=1, dtype="CPU float64",
            methods=["switch_exact", "affine_fused", "bounded_affine"]+EXACT_ENGINES,
            workloads=["from_image", "from_context", "cached"],
            from_image="prefix, complete background acquisition, interval screening, stable matrix, gathers, and queries",
            from_context="start with complete h,z; charge method preparation and queries",
            cached="queries after method-specific preparation; shared weights stay retained",
            query_sets="low-order and evaluation for all q; additionally4096 exhaustive masks for q12",
            warmup=2, rounds=7, inner=3, random_order=True, alpha=1., training_concurrent=False,
            comparator="fastest of three exact acquisition engines per seed/layout/query-set/workload, measured in same session",
            memory="actual cache tensor bytes and retained shared tensors; not peak process memory or total model compression"),
        prediction_before_new_reference=True, new_teacher_training_runs=0, new_classification_runs=0,
        calibration_model_access=False, imported_sources=imported, data_hashes=prior["data_hashes"],
        teachers_sha256=prior["teachers_sha256"], prior_fit_hashes=fits,
        prior_scale_run_hashes=old_entries, prior_scale_manifest_sha256=file_sha(BASE/"audit/scale_manifest_20260914.json"),
        prior_scale_freeze_sha256=file_sha(SCALE_RUN/"freeze.json"),
        layouts_sha256=file_sha(run/"layouts.npz"), intervention_ids_sha256=file_sha(run/"intervention_ids.npy"))
    dump(run/"protocol.json", protocol)
    (run/"protocol.sha256").write_text(file_sha(run/"protocol.json")+"\n", encoding="ascii")
    event(run, "switch_protocol_fixed", sha256=file_sha(run/"protocol.json"))


# 用途：检查已保存的协议、输入、源码或冻结记录；验证失败就停止。
# 参数：run、frozen；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：protocol。
def verify(run, frozen=False):
    protocol = read_json(run/"protocol.json")
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert file_sha(run/"protocol.json") == (run/"protocol.sha256").read_text(encoding="ascii").strip()
    assert file_sha(SCALE_RUN/"freeze.json") == protocol["prior_scale_freeze_sha256"]
    assert file_sha(BASE/"audit/scale_manifest_20260914.json") == protocol["prior_scale_manifest_sha256"]
    check_hashes(BASE, protocol["imported_sources"])
    check_hashes(BASE, protocol["data_hashes"])
    check_hashes(BASE, protocol["prior_fit_hashes"])
    for seed, expected in protocol["teachers_sha256"].items():
        assert file_sha(PRIOR_RUN/f"seed_{seed}/teacher.pt") == expected
    assert file_sha(run/"layouts.npz") == protocol["layouts_sha256"]
    assert file_sha(run/"intervention_ids.npy") == protocol["intervention_ids_sha256"]
    if (run/"implementation.json").exists():
        impl = read_json(run/"implementation.json")
        assert file_sha(run/"implementation.json") == (run/"implementation.sha256").read_text(encoding="ascii").strip()
        assert impl["protocol_sha256"] == file_sha(run/"protocol.json")
        check_hashes(BASE, impl["sources"])
        check_hashes(run/"source", impl["sources"])
    if frozen:
        assert file_sha(run/"freeze.json") == (run/"freeze.sha256").read_text(encoding="ascii").strip()
        check_hashes(run, read_json(run/"freeze.json")["artifacts"])
    return protocol


# 用途：用规模较小且有可比较答案的情形检查实现，数值不一致时立即停止。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
@torch.no_grad()
def smoke(run):
    verify(run)
    assert not (run/"smoke.json").exists() and not (run/"implementation.json").exists()
    errors = defaultdict(float)
    generator = torch.Generator().manual_seed(2026091417)
    counts = defaultdict(int)
    # These tests probe the algebra and the strict boundary handling independently
    # of any trained network or chosen source window.
    for q in (1, 4, 8, 12):
        W = torch.randn(32, q, generator=generator, dtype=torch.float64)
        U = torch.randn(10, 32, generator=generator, dtype=torch.float64)
        for case in ("random", "stable_on", "stable_off", "all_candidates", "endpoints", "zero_source"):
            h = torch.rand(q, generator=generator, dtype=torch.float64)
            a = -W*h[None]
            low, high = a.clamp(max=0).sum(1), a.clamp(min=0).sum(1)
            if case == "stable_on": z = 100.+high-low
            elif case == "stable_off": z = -100.-high+low
            elif case == "all_candidates": z = -(low+high)/2
            elif case == "endpoints":
                z = torch.where(torch.arange(32)%2 == 0, -low, -high)
                z[::4] = torch.nextafter(z[::4], torch.full_like(z[::4], float("inf")))
                z[1::4] = torch.nextafter(z[1::4], torch.full_like(z[1::4], -float("inf")))
            else: z = torch.randn(32, generator=generator, dtype=torch.float64)
            if case == "zero_source": h.zero_()
            local = SimpleNamespace(W=W, U=U)
            rt = SwitchExact(local)
            context = (h, z)
            cc = rt.cached(context)
            _, candidate = rt.screen(context)
            bits = torch.tensor(bit_design(q), dtype=torch.float64)
            if case in ("stable_on", "stable_off"): assert not bool(candidate.any())
            if case == "all_candidates": assert bool(candidate.all())
            for alpha in (0., .25, .5, 1.):
                t = (-alpha*bits*h[None])@W.T
                gold = (F.relu(z[None]+t)-F.relu(z)[None])@U.T
                errors["synthetic_exact"] = max(errors["synthetic_exact"], float((rt.evaluate(cc,bits,alpha)-gold).abs().max()))
                errors["synthetic_omitted_remainder"] = max(errors["synthetic_omitted_remainder"], omitted_state_error(local,context,bits,alpha,candidate))
                counts["synthetic_output_vectors"] += len(bits)
            counts["synthetic_cases"] += 1
    images, _, val_ids = training_data()
    model = load_teacher(PRIOR_RUN/f"seed_{SEEDS[0]}")
    gpu = load_teacher(PRIOR_RUN/f"seed_{SEEDS[0]}", device="cuda")
    sources, residuals = prefix(model, torch.tensor(normalized(images[val_ids[:2]]), dtype=torch.float64))
    for position in POSITIONS:
        for q in SIZES:
            nodes = get_nodes(run,SEEDS[0],position,q)
            local = ResidualLocal(model,tuple(sources.shape[1:]),nodes)
            rt = SwitchExact(local)
            bits = torch.tensor(design(run,q),dtype=torch.float64)
            refs = independent_reference(gpu,sources,residuals,nodes,bits)
            for i in range(2):
                s,r = sources[i:i+1],residuals[i:i+1]
                common = local.states(s,r)
                cache = rt.cached(common)
                for alpha in STRENGTHS:
                    original_cpu = local.full(s,r,bits,alpha)
                    errors["cpu_cuda_original"] = max(errors["cpu_cuda_original"],float(np.abs(original_cpu.numpy()-refs[str(alpha)][i]).max()))
                    value = rt.evaluate(cache,bits,alpha)
                    errors["network_switch_exact"] = max(errors["network_switch_exact"],float((value-original_cpu).abs().max()))
                    for engine in EXACT_ENGINES:
                        fresh_rt = SwitchExact(local,engine)
                        fresh = fresh_rt.evaluate(fresh_rt.prepare(s,r),bits,alpha)
                        errors["acquisition_engines"] = max(errors["acquisition_engines"],float((fresh-value).abs().max()))
    assert max(errors.values()) < 1e-9, dict(errors)
    dump(run/"smoke.json",dict(utc=now(),status="passed",errors=dict(errors),**counts,
        network_layouts=6,validation_inputs=2,calibration_model_access=False))
    event(run,"switch_smoke_passed",errors=dict(errors))


# 用途：在小规模检查通过后保存实现副本和哈希，固定后续运行使用的代码。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def lock(run):
    verify(run)
    assert read_json(run/"smoke.json")["status"] == "passed"
    assert not (run/"implementation.json").exists()
    record = dict(utc=now(),protocol_sha256=file_sha(run/"protocol.json"),sources={n:file_sha(BASE/n) for n in SOURCE_NAMES})
    for name in SOURCE_NAMES:
        destination = run/"source"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,destination)
    dump(run/"implementation.json",record)
    (run/"implementation.sha256").write_text(file_sha(run/"implementation.json")+"\n",encoding="ascii")
    event(run,"switch_implementation_locked",sha256=file_sha(run/"implementation.json"))


# 用途：保存稳定状态筛选预测和内部状态，再取得或复用原网络参考进行核对。
# 参数：run、seed、position、q、model、gpu、sources、residuals、ids、phase、exhaustive；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：done。
@torch.no_grad()
def run_case(run, seed, position, q, model, gpu, sources, residuals, ids, phase, exhaustive=False):
    out = run/("supplementary" if exhaustive else phase)/name_of(seed,position,q)
    assert not out.exists()
    out.mkdir(parents=True)
    started = time.perf_counter()
    nodes = get_nodes(run,seed,position,q)
    local = ResidualLocal(model,tuple(sources.shape[1:]),nodes)
    bits_np = design(run,q,"exhaustive" if exhaustive else "evaluation")
    bits = torch.tensor(bits_np,dtype=torch.float64)
    contexts = [local.states(s[None],r[None]) for s,r in zip(sources,residuals)]
    runtimes = runtime_set(local)
    if exhaustive: runtimes = {k:v for k,v in runtimes.items() if k in ("switch_exact","exact")}
    predictions = {}
    diagnostics = []
    candidates = []
    stable = []
    switch_rt = runtimes["switch_exact"]
    max_omitted = 0.
    fresh_difference = 0.
    for i, context in enumerate(contexts):
        stable_on, candidate = switch_rt.screen(context)
        candidates.append(candidate.numpy())
        stable.append(stable_on.numpy())
        cache = switch_rt.cached(context)
        diagnostics.append(dict(seed=seed,position=position,q=q,input_id=int(ids[i]),affected_states=local.J,
            candidates=int(candidate.sum()),stable_on=int(stable_on.sum()),stable_off=int((~candidate & ~stable_on).sum()),
            candidate_fraction=float(candidate.double().mean()),switch_cache_bytes=tensor_bytes(cache),
            exact_cache_bytes=tensor_bytes(context),source_l2=float(context[0].norm())))
        for alpha in STRENGTHS:
            max_omitted = max(max_omitted,omitted_state_error(local,context,bits,alpha,candidate))
    for method, rt in runtimes.items():
        cached = [rt.cached(c) for c in contexts]
        for alpha in STRENGTHS:
            values = torch.stack([rt.evaluate(c,bits,alpha) for c in cached]).numpy()
            predictions[f"{method}_a{alpha}"] = values
            for i in range(min(2,len(ids))):
                fresh = rt.evaluate(rt.prepare(sources[i:i+1],residuals[i:i+1]),bits,alpha).numpy()
                fresh_difference = max(fresh_difference,float(np.abs(fresh-values[i]).max()))
    np.savez_compressed(out/"predictions.npz",**predictions,ids=ids)
    np.savez_compressed(out/"screening_contexts.npz",h=torch.stack([c[0] for c in contexts]).numpy(),
        z=torch.stack([c[1] for c in contexts]).numpy(),candidates=np.stack(candidates),stable_on=np.stack(stable),
        W=local.W.numpy(),U=local.U.numpy(),nodes=nodes,ids=ids)
    csv_write(out/"screening.csv",diagnostics)
    dump(out/"prediction_commitment.json",dict(utc=now(),sha256=file_sha(out/"predictions.npz"),
        screening_sha256=file_sha(out/"screening_contexts.npz")))
    if phase == "validation" and not exhaustive:
        ref_path = SCALE_RUN/name_of(seed,position,q)/"validation/references.npz"
        with np.load(ref_path,allow_pickle=False) as saved:
            assert np.array_equal(saved["ids"],ids)
            refs = {str(alpha):saved[str(alpha)].copy() for alpha in STRENGTHS}
        ref_kind = "reused_original_unfused_scale_validation_reference"
    else:
        refs = independent_reference(gpu,sources,residuals,nodes,bits)
        ref_path = out/"references.npz"
        np.savez_compressed(ref_path,**refs,ids=ids)
        ref_kind = "new_original_unfused_CUDA_float64_reference"
    dump(out/"reference_commitment.json",dict(utc=now(),path=ref_path.relative_to(BASE).as_posix(),
        sha256=file_sha(ref_path),kind=ref_kind))
    results = []
    for method in runtimes:
        for alpha in STRENGTHS:
            pred, ref = predictions[f"{method}_a{alpha}"],refs[str(alpha)]
            labels = dict(seed=seed,position=position,q=q,method=method,alpha=alpha)
            result = scale_metrics(pred,ref,bits_np,**labels)
            result["all_queries_max_abs"] = float(np.abs(pred-ref).max())
            results.append(result)
    exact_max = max(r["all_queries_max_abs"] for r in results if r["method"] in ("switch_exact","exact"))
    assert max(exact_max,max_omitted,fresh_difference) < 1e-9
    csv_write(out/"results.csv",results)
    done = dict(utc=now(),status="completed",phase=phase,exhaustive=exhaustive,seed=seed,position=position,q=q,
        inputs=len(ids),queried_coalitions=len(bits),arrays=len(results),reference_kind=ref_kind,
        original_reference_vectors=len(ids)*len(bits)*2,new_reference_vectors=0 if ref_kind.startswith("reused") else len(ids)*len(bits)*2,
        max_exact_error=exact_max,max_omitted_remainder=max_omitted,fresh_cached_difference=fresh_difference,
        seconds=time.perf_counter()-started)
    dump(out/"completion.json",done)
    event(run,"switch_case_complete",**done)
    return done


# 用途：对固定的开发输入拟合或检查方法，保存测试前需要的参数和结果。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def develop(run):
    verify(run)
    assert (run/"implementation.json").exists() and not (run/"test_opened.json").exists()
    assert not (run/"development_complete.json").exists()
    images,_,ids = training_data()
    x = torch.tensor(normalized(images[ids]),dtype=torch.float64)
    completed = []
    for seed in SEEDS:
        model = load_teacher(PRIOR_RUN/f"seed_{seed}")
        gpu = load_teacher(PRIOR_RUN/f"seed_{seed}",device="cuda")
        sources,residuals = prefix(model,x)
        for position in POSITIONS:
            for q in SIZES:
                completed.append(run_case(run,seed,position,q,model,gpu,sources,residuals,ids,"validation"))
            completed.append(run_case(run,seed,position,12,model,gpu,sources[:4],residuals[:4],ids[:4],"validation",True))
        del gpu
        torch.cuda.empty_cache()
    verify(run)
    dump(run/"development_complete.json",dict(utc=now(),cases=36,supplementary_cases=12,
        arrays=sum(r["arrays"] for r in completed),max_exact_error=max(r["max_exact_error"] for r in completed),
        reused_reference_vectors=sum(r["original_reference_vectors"]-r["new_reference_vectors"] for r in completed),
        new_exhaustive_reference_vectors=sum(r["new_reference_vectors"] for r in completed),
        new_official_interventions_accessed=False,calibration_model_access=False))
    event(run,"switch_development_complete",cases=36,supplementary_cases=12)


# 用途：在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
@torch.no_grad()
def timing(run):
    verify(run)
    assert (run/"development_complete.json").exists() and not (run/"test_opened.json").exists()
    out = run/"timing"
    out.mkdir(exist_ok=False)
    images,_,val_ids = training_data()
    x = torch.tensor(normalized(images[val_ids[:4]]),dtype=torch.float64)
    raw,costs = [],[]
    maximum = 0.
    for seed in SEEDS:
        model = load_teacher(PRIOR_RUN/f"seed_{seed}")
        sources,residuals = prefix(model,x)
        for position in POSITIONS:
            for q in SIZES:
                local = ResidualLocal(model,tuple(sources.shape[1:]),get_nodes(run,seed,position,q))
                runtimes = runtime_set(local,True)
                query_sets = ["low_order","evaluation"] + (["exhaustive"] if q == 12 else [])
                rng = np.random.default_rng(seed+q*100+(0 if position=="center" else 1)+2026091417)
                for i in range(4):
                    common = local.states(sources[i:i+1],residuals[i:i+1])
                    cached = {name:rt.cached(common) for name,rt in runtimes.items()}
                    for name,rt in runtimes.items():
                        retained = [v for owner in (local,rt) for v in vars(owner).values() if isinstance(v,torch.Tensor)]
                        costs.append(dict(seed=seed,position=position,q=q,input_id=int(val_ids[i]),method=name,
                            affected_states=local.J,cached_tensor_bytes=tensor_bytes(cached[name]),
                            retained_non_teacher_tensor_bytes=tensor_bytes(retained),
                            teacher_state_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values())))
                    for query_set in query_sets:
                        bt = torch.tensor(design(run,q,query_set),dtype=torch.float64)
                        for workload in ("from_image","from_context","cached"):
                            functions = {}
                            for name,rt in runtimes.items():
                                if workload == "cached":
                                    functions[name] = lambda name=name,rt=rt:rt.evaluate(cached[name],bt)
                                elif workload == "from_context":
                                    functions[name] = lambda rt=rt:rt.evaluate(rt.cached(common),bt)
                                else:
                                    # 用途：构造当前运行时的一次实际执行，供计时循环调用。
                                    # 参数：rt；具体形状/来源结合调用处及下方关键步骤阅读。
                                    # 返回：rt.evaluate(rt.prepare(s, r), bt)。
                                    def fn(rt=rt):
                                        s,r = model.prefix(x[i:i+1])
                                        return rt.evaluate(rt.prepare(s,r),bt)
                                    functions[name] = fn
                            gold = local.exact(common,bt)
                            for name,fn in functions.items():
                                value = fn()
                                if name in EXACT_ENGINES or name == "switch_exact":
                                    maximum = max(maximum,float((value-gold).abs().max()))
                                for _ in range(2):fn()
                            names = list(functions)
                            for round_ in range(7):
                                for j in rng.permutation(len(names)):
                                    name = names[j]
                                    started = time.perf_counter_ns()
                                    for _ in range(3):functions[name]()
                                    raw.append(dict(seed=seed,position=position,q=q,input_id=int(val_ids[i]),
                                        query_set=query_set,queries=len(bt),workload=workload,method=name,
                                        round=round_,seconds=(time.perf_counter_ns()-started)/3e9))
                event(run,"switch_timing_layout_complete",seed=seed,position=position,q=q,rows=len(raw))
    assert maximum < 1e-9 and len(raw) == 42336
    csv_write(out/"raw.csv",raw)
    csv_write(out/"cost_counts.csv",costs)
    keys = ("seed","position","q","input_id","query_set","queries","workload","method")
    groups = defaultdict(list)
    for row in raw:groups[tuple(row[k] for k in keys)].append(row["seconds"])
    medians = [dict(zip(keys,key),seconds=float(np.median(values))) for key,values in groups.items()]
    csv_write(out/"input_medians.csv",medians)
    keys = tuple(k for k in keys if k != "input_id")
    groups = defaultdict(list)
    for row in medians:groups[tuple(row[k] for k in keys)].append(row["seconds"])
    summary = [dict(zip(keys,key),seconds=float(np.mean(values))) for key,values in groups.items()]
    csv_write(out/"summary.csv",summary)
    dump(out/"completion.json",dict(utc=now(),rows=len(raw),cases=36,max_exact_path_difference=maximum,
        alpha=1.,training_concurrent=False,threads=torch.get_num_threads()))
    event(run,"switch_timing_complete",rows=len(raw))


# 用途：把开发阶段选定的模型、方法、预算与相关文件写入测试前冻结记录。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def freeze(run):
    verify(run)
    assert not (run/"freeze.json").exists() and not (run/"test_opened.json").exists()
    assert (run/"development_complete.json").exists() and (run/"timing/completion.json").exists()
    files = [p for p in run.rglob("*") if p.is_file() and p.name != "events.jsonl"]
    record = dict(utc=now(),protocol_sha256=file_sha(run/"protocol.json"),implementation_sha256=file_sha(run/"implementation.json"),
        artifacts={p.relative_to(run).as_posix():file_sha(p) for p in sorted(files)},
        new_official_interventions_accessed=False,prior_official_classification_access=True,calibration_model_access=False)
    dump(run/"freeze.json",record)
    (run/"freeze.sha256").write_text(file_sha(run/"freeze.json")+"\n",encoding="ascii")
    event(run,"switch_frozen",artifacts=len(files),sha256=file_sha(run/"freeze.json"))


# 用途：核验冻结状态后处理预留测试输入，保存预测、原网络参考及评分。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def test(run):
    verify(run,True)
    assert not (run/"test_opened.json").exists()
    dump(run/"test_opened.json",dict(utc=now(),freeze_sha256=file_sha(run/"freeze.json")))
    ids = np.load(run/"intervention_ids.npy",allow_pickle=False)
    assert np.array_equal(ids,np.arange(896,1024))
    images = load_idx(BASE/"data/fashion_mnist/raw","t10k-images-idx3-ubyte.gz")
    x = torch.tensor(normalized(images[ids]),dtype=torch.float64)
    completed = []
    for seed in SEEDS:
        model = load_teacher(PRIOR_RUN/f"seed_{seed}")
        gpu = load_teacher(PRIOR_RUN/f"seed_{seed}",device="cuda")
        sources,residuals = prefix(model,x)
        for position in POSITIONS:
            for q in SIZES:
                completed.append(run_case(run,seed,position,q,model,gpu,sources,residuals,ids,"test"))
        del gpu
        torch.cuda.empty_cache()
    verify(run,True)
    dump(run/"test/completion.json",dict(utc=now(),status="completed",cases=len(completed),
        full_tail_output_vectors=sum(r["new_reference_vectors"] for r in completed),
        arrays=sum(r["arrays"] for r in completed),max_exact_error=max(r["max_exact_error"] for r in completed),
        max_omitted_remainder=max(r["max_omitted_remainder"] for r in completed),
        new_teacher_training_runs=0,new_classification_runs=0,calibration_model_access=False))
    event(run,"switch_test_complete",cases=len(completed))


# 用途：回读已保存数组重新评分并汇总，核对结果与既有记录的一致性。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def analyze(run):
    verify(run,True)
    assert read_json(run/"test/completion.json")["status"] == "completed"
    out = run/"analysis"
    out.mkdir(exist_ok=False)
    counts = defaultdict(int)
    rows,screening = [],[]
    max_score_diff,max_omitted,max_split = 0.,0.,0.
    for phase in ("validation","supplementary","test"):
        for seed in SEEDS:
            for position in POSITIONS:
                for q in ([12] if phase == "supplementary" else SIZES):
                    case = run/phase/name_of(seed,position,q)
                    pc,rc = read_json(case/"prediction_commitment.json"),read_json(case/"reference_commitment.json")
                    assert pc["utc"] < rc["utc"]
                    assert file_sha(case/"predictions.npz") == pc["sha256"]
                    assert file_sha(case/"screening_contexts.npz") == pc["screening_sha256"]
                    assert file_sha(BASE/rc["path"]) == rc["sha256"]
                    bt = design(run,q,"exhaustive" if phase == "supplementary" else "evaluation")
                    with np.load(case/"predictions.npz",allow_pickle=False) as pred,np.load(BASE/rc["path"],allow_pickle=False) as refs:
                        assert np.array_equal(pred["ids"],refs["ids"])
                        for row in csv_read(case/"results.csv"):
                            p,r = pred[f"{row['method']}_a{row['alpha']}"],refs[row["alpha"]]
                            recomputed = scale_metrics(p,r,bt)
                            recomputed["all_queries_max_abs"] = float(np.abs(p-r).max())
                            for key in ("nrmse","sse","effect_sse","mae","maxerr","input_p95","all_queries_max_abs"):
                                max_score_diff = max(max_score_diff,abs(float(row[key])-recomputed[key]))
                            counts[phase] += 1
                            rows.append(dict(phase=phase,**row))
                        # Separate NumPy replay of interval endpoints and folded
                        # prediction; do not call the implementation under test.
                        with np.load(case/"screening_contexts.npz",allow_pickle=False) as state:
                            W,U = state["W"],state["U"]
                            for i,(h,z) in enumerate(zip(state["h"],state["z"])):
                                c = -W*h[None]
                                low,high = np.minimum(c,0).sum(1),np.maximum(c,0).sum(1)
                                guard = 64*np.finfo(np.float64).eps*(np.abs(z)+high-low+1)
                                on,off = z+low>guard,z+high < -guard
                                candidate = ~(on|off)
                                assert np.array_equal(candidate,state["candidates"][i])
                                assert np.array_equal(on,state["stable_on"][i])
                                for alpha in STRENGTHS:
                                    delta = -alpha*bt*h[None]
                                    t = delta@W.T
                                    tangent_remainder = np.maximum(z+t,0)-np.maximum(z,0)-t*(z>0)
                                    if (~candidate).any(): max_omitted = max(max_omitted,float(np.abs(tangent_remainder[:,~candidate]).max()))
                                    split = delta@((U*on[None])@W).T
                                    split += (np.maximum(z[candidate]+t[:,candidate],0)-np.maximum(z[candidate],0))@U[:,candidate].T
                                    max_split = max(max_split,float(np.abs(split-pred[f"switch_exact_a{alpha}"][i]).max()))
                    screening.extend(dict(phase=phase,**r) for r in csv_read(case/"screening.csv"))
    assert dict(counts) == dict(validation=288,supplementary=48,test=288)
    assert max_score_diff < 1e-12 and max(max_omitted,max_split) < 1e-9
    csv_write(out/"results.csv",rows)
    csv_write(out/"screening.csv",screening)
    # Recompute timing aggregation from raw rounds before forming paired ratios.
    raw = csv_read(run/"timing/raw.csv")
    keys = ("seed","position","q","input_id","query_set","queries","workload","method")
    group = defaultdict(list)
    for row in raw:group[tuple(row[k] for k in keys)].append(float(row["seconds"]))
    assert len(raw) == 42336 and all(len(v)==7 for v in group.values())
    k2 = tuple(k for k in keys if k != "input_id")
    medians = defaultdict(list)
    for key,values in group.items():
        row = dict(zip(keys,key))
        medians[tuple(row[k] for k in k2)].append(float(np.median(values)))
    summary = {key:float(np.mean(values)) for key,values in medians.items()}
    for row in csv_read(run/"timing/summary.csv"):
        assert abs(summary[tuple(row[k] for k in k2)]-float(row["seconds"])) < 1e-15
    settings = defaultdict(dict)
    k3 = tuple(k for k in k2 if k != "method")
    for key,seconds in summary.items():
        row = dict(zip(k2,key))
        settings[tuple(row[k] for k in k3)][row["method"]] = seconds
    ratios = []
    for key,methods in settings.items():
        fastest = min(EXACT_ENGINES,key=lambda e:methods[e])
        for method in ("switch_exact","affine_fused","bounded_affine"):
            ratios.append(dict(zip(k3,key),method=method,seconds=methods[method],
                fastest_exact_engine=fastest,exact_seconds=methods[fastest],speedup=methods[fastest]/methods[method]))
    csv_write(out/"paired_timings.csv",ratios)
    verify(run,True)
    dump(out/"verification.json",dict(utc=now(),status="passed",arrays_rescored=dict(counts),
        max_metric_difference=max_score_diff,max_numpy_split_difference=max_split,max_omitted_remainder=max_omitted,
        timing_rows=len(raw),all_timing_aggregations_reproduced=True,calibration_model_access=False,new_confirmatory_p_values=0))
    event(run,"switch_analysis_complete",arrays=dict(counts),max_numpy_split_difference=max_split)


# 用途：验证稳定状态合并后的精确响应，并计入状态筛选和缓存的完整成本。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=("init","smoke","lock","develop","timing","freeze","test","analyze"))
    args = parser.parse_args()
    configure_cpu()
    try:
        globals()[args.stage](RUN)
    except Exception as exc:
        if RUN.exists():event(RUN,"stage_failed",stage=args.stage,error=repr(exc),source_sha256=file_sha(Path(__file__)))
        raise


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

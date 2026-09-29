# 中文阅读说明｜用途：检查 4/8/12 源坐标和两个空间位置，比较近似方法及有上界的精确回退。
# 数据流：普通 q=12 评价只含固定子集；完整 4096 组合是后续 switch 补充阶段的单独记录。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Locked size/position follow-up; no prior numerical artifact is overwritten."""
import argparse
import hashlib
import json
import shutil
import time
from collections import defaultdict
from scale_context import *
from run_residual_controls import training_data, prefix, verify as verify_prior
from prepare_external import load_idx

RUN = NEW_RUN


# 用途：以 SHA-256 标识文件内容，供历史承诺和当前文件逐项比较。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.file_digest(stream, 'sha256').hexdigest()。
def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 用途：逐个验证清单中的文件路径与摘要，防止读取范围外或已被改写的文件。
# 参数：root、values；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def check_hashes(root, values):
    for name, expected in values.items():
        target = (root/name).resolve()
        assert target.is_relative_to(root.resolve())
        # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
        assert file_sha(target) == expected, name


# 用途：创建该阶段的新运行目录，固定设计、输入索引和依赖文件哈希。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def init(run):
    verify_prior(PRIOR_RUN, True)
    run.mkdir(exist_ok=False)
    prior = read_json(PRIOR_RUN/"protocol.json")
    sources = dict(prior["sources"])
    sources.update(read_json(PRIOR_RUN/"implementation.json")["sources"])
    sources["scripts/audit_residual_affine_runtime.py"] = file_sha(BASE/"scripts/audit_residual_affine_runtime.py")
    protocol = dict(utc=now(), version="residual-size-position-v1",
        purpose="Map approximation failure across source-set size and spatial position; evaluate a fixed analytic-bound fallback",
        scope="Same final residual branch, two spatial windows; no earlier-layer or external-dataset generalization claim",
        prior_knowledge="All prior three intervention batches 0..767 and full official classification pool have been inspected; development-informed descriptive follow-up",
        teachers="Reuse all six frozen residual teachers; no retraining, exclusion or new classification evaluation",
        seeds=SEEDS, sizes=SIZES, positions={k:list(v) for k,v in POSITIONS.items()},
        source_selection="Nested top-q training-mean activations within each channel-by-3x3 window; stable index-order ties",
        budgets=BUDGETS, strengths=STRENGTHS,
        methods=["greedy", "gate_weight", "pca_fused", "mean_gate", "affine_fused", "bounded_affine", "exact"],
        method_training=dict(contexts=512, coalitions="all masks with cardinality <=2; alpha1 only", omitted="training mean gate",
                             greedy="same audited residual-component Gram greedy", pca="same fused preactivation encoding"),
        evaluation=dict(validation_contexts=128, test_contexts=128, test_ids="official 768..895",
            q4="all16 masks, score5 masks of order>2",
            q8="all256 masks, score219 masks of order>2",
            q12="all79 low-order plus256 uniformly sampled unique high-order masks, fixed before development; not exhaustive",
            order_stratification="report each available cardinality from the same saved arrays",
            zero_effect_threshold=1e-24, statistic="six-seed means/SD and paired differences; no new confirmatory p-values"),
        fallback=dict(threshold=THRESHOLD, training="none; fixed analytical rule before development",
            upper="For each state, compute interval over all source subsets and bound its nonnegative ReLU tangent remainder; map remainder box through readout U",
            criterion="sqrt(N)*beta <= 0.1/1.1*norm(affine_predictions_on_scored_queries); if no high-order queries, use all supplied queries",
            on_reject="exact nonlinear local evaluation using the already acquired complete background",
            cached="retain full background for fallback plus source values, response matrix and bounds for both fixed strengths",
            precision="exact-real-arithmetic bound evaluated in ordinary float64; not outward-rounded formal verification",
            check="independent reference norm error must be <= computed upper +1e-9; report any numerical relative-threshold exceptions"),
        timing=dict(inputs=4, queries="low-order design and complete evaluation design for each q",
            workloads=["from_image", "cached"], dtype="CPU float64", threads=1,
            warmup=2, rounds=7, inner=3, random_order=True, alpha=1.,
            exact_engines=EXACT_ENGINES, comparison="fastest measured exact engine per seed,position,q,query-set,workload",
            training_concurrent=False),
        reference="original unfused teacher tail, CUDA float64; predictions committed before references; CPU/CUDA smoke cross-check",
        calibration_model_access=False, no_new_confirmatory_p_values=True,
        prior_freeze_sha256=file_sha(PRIOR_RUN/"freeze.json"),
        prior_manifest_sha256=file_sha(BASE/"audit/residual_manifest_20260913.json"),
        teachers_sha256={str(s):file_sha(PRIOR_RUN/f"seed_{s}/teacher.pt") for s in SEEDS},
        imported_sources=sources, data_hashes=prior["data_hashes"],
        references=["https://arxiv.org/abs/1810.12715"])
    np.save(run/"intervention_ids.npy", np.arange(768,896))
    designs = {}
    for q in SIZES:
        train, evaluation, indices = query_design(q)
        # 保存具名数值数组，便于后来按同一索引回读和独立复算。
        np.savez_compressed(run/f"design_q{q}.npz", train=train, evaluation=evaluation, full_mask_indices=indices)
        designs[str(q)] = dict(sha256=file_sha(run/f"design_q{q}.npz"),
            train=len(train), evaluation=len(evaluation), scored=int((evaluation.sum(1)>2).sum()),
            scored_order_counts={str(int(o)):int((evaluation.sum(1)==o).sum()) for o in np.unique(evaluation.sum(1)) if o>2})
    protocol["designs"] = designs
    protocol["intervention_ids_sha256"] = file_sha(run/"intervention_ids.npy")
    dump(run/"protocol.json", protocol)
    (run/"protocol.sha256").write_text(file_sha(run/"protocol.json")+"\n", encoding="ascii")
    event(run, "scale_protocol_fixed", protocol_sha256=file_sha(run/"protocol.json"))


# 用途：检查已保存的协议、输入、源码或冻结记录；验证失败就停止。
# 参数：run、frozen；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：protocol。
def verify(run, frozen=False):
    protocol = read_json(run/"protocol.json")
    assert file_sha(run/"protocol.json") == (run/"protocol.sha256").read_text(encoding="ascii").strip()
    assert file_sha(PRIOR_RUN/"freeze.json") == protocol["prior_freeze_sha256"]
    assert file_sha(BASE/"audit/residual_manifest_20260913.json") == protocol["prior_manifest_sha256"]
    check_hashes(BASE, protocol["imported_sources"])
    check_hashes(BASE, protocol["data_hashes"])
    for s, expected in protocol["teachers_sha256"].items():
        assert file_sha(PRIOR_RUN/f"seed_{s}/teacher.pt") == expected
    for q, record in protocol["designs"].items():
        assert file_sha(run/f"design_q{q}.npz") == record["sha256"]
    assert file_sha(run/"intervention_ids.npy") == protocol["intervention_ids_sha256"]
    if (run/"implementation.json").exists():
        impl = read_json(run/"implementation.json")
        assert impl["protocol_sha256"] == file_sha(run/"protocol.json")
        assert file_sha(run/"implementation.json") == (run/"implementation.sha256").read_text(encoding="ascii").strip()
        check_hashes(BASE, impl["sources"])
        check_hashes(run/"source", impl["sources"])
    if frozen:
        freeze = read_json(run/"freeze.json")
        assert file_sha(run/"freeze.json") == (run/"freeze.sha256").read_text(encoding="ascii").strip()
        check_hashes(run, freeze["artifacts"])
    return protocol


# 用途：读取本设置保存的组合掩码；全枚举与低阶/抽样集合由参数区分。
# 参数：run、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(f['train'].copy(), f['evaluation'].copy())。
def design(run, q):
    with np.load(run/f"design_q{q}.npz", allow_pickle=False) as f:
        return f["train"].copy(), f["evaluation"].copy()


# 用途：用原始未融合教师尾部逐组合生成独立参考。
# 输入：sources/residuals 为原始中间状态；nodes 为源索引；bits 为本设置查询矩阵。
# 输出：返回按强度索引的 [图像,组合,10] 数组；参考采用 CUDA float64。
@torch.no_grad()
def independent_reference(model, sources, residuals, nodes, bits):
    values = {}
    bt = bits.cuda()
    for alpha in STRENGTHS:
        outputs = []
        for s, r in zip(sources, residuals):
            s, r = s[None].cuda(), r[None].cuda()
            baseline = model.tail(s, r)
            groups = []
            for begin in range(0, len(bt), 128):
                chunk = bt[begin:begin+128]
                h = s.expand(len(chunk), -1, -1, -1).clone()
                h.flatten(1)[:, nodes] *= (1-alpha*chunk)
                groups.append((model.tail(h, r)-baseline).cpu().numpy())
            outputs.append(np.concatenate(groups))
        values[str(alpha)] = np.stack(outputs)
    return values


# 用途：用规模较小且有可比较答案的情形检查实现，数值不一致时立即停止。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def smoke(run):
    verify(run)
    assert not (run/"smoke.json").exists() and not (run/"implementation.json").exists()
    images, train_ids, val_ids = training_data()
    model = load_teacher(PRIOR_RUN/f"seed_{SEEDS[0]}")
    gpu = load_teacher(PRIOR_RUN/f"seed_{SEEDS[0]}", device="cuda")
    ts, tr = prefix(model, torch.tensor(normalized(images[train_ids[:32]]), dtype=torch.float64))
    vs, vr = prefix(model, torch.tensor(normalized(images[val_ids[:4]]), dtype=torch.float64))
    errors = defaultdict(float)
    for position in POSITIONS:
        for q in SIZES:
            nodes = choose_source_nodes(ts, position, q)
            local = ResidualLocal(model, tuple(ts.shape[1:]), nodes)
            contexts = [local.states(s[None], r[None]) for s,r in zip(ts,tr)]
            train_bits, eval_bits = design(run,q)
            fit, _ = fit_scale(local, contexts, train_bits, SEEDS[0])
            bits = torch.tensor(eval_bits, dtype=torch.float64)
            refs = independent_reference(gpu, vs, vr, nodes, bits)
            for i in range(4):
                s,r = vs[i:i+1], vr[i:i+1]
                context = local.states(s,r)
                for engine in EXACT_ENGINES:
                    errors["acquisition"] = max(errors["acquisition"], float((local.states(s,r,engine)[1]-context[1]).abs().max()))
                errors["affine_background"] = max(errors["affine_background"], float((local.affine@local.vector(s,r)+local.row_bias-context[1]).abs().max()))
                bound_rt = BoundedAffine(local)
                for alpha in STRENGTHS:
                    direct = local.full(s,r,bits,alpha).numpy()
                    errors["cpu_cuda_reference"] = max(errors["cpu_cuda_reference"], float(np.abs(direct-refs[str(alpha)][i]).max()))
                    errors["local_exact"] = max(errors["local_exact"], float(np.abs(local.exact(context,bits,alpha).numpy()-direct).max()))
                    affine = FusedAffine(local).evaluate(FusedAffine(local).cached(context), bits, alpha).numpy()
                    pred, info = bound_rt.decision(bound_rt.cached(context), bits, alpha)
                    # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
                    held = eval_bits.sum(1)>2
                    norm_error = np.linalg.norm((affine-direct)[held])
                    errors["bound_excess"] = max(errors["bound_excess"], max(0., float(norm_error-info["error_norm_upper"])))
                    if info["accepted"]:
                        errors["accepted_absolute_threshold_excess"] = max(errors["accepted_absolute_threshold_excess"],
                            max(0., float(np.linalg.norm((pred.numpy()-direct)[held])-.1*np.linalg.norm(direct[held]))))
                for method in ("greedy","gate_weight","pca_fused","mean_gate","exact"):
                    k = 0 if method=="mean_gate" else (local.J if method=="exact" else 8)
                    rt = ScaleRuntime(local,fit,method,k)
                    cc = rt.cached(context)
                    for alpha in STRENGTHS:
                        many = rt.evaluate_many([cc],bits,alpha)
                        errors["batch_single"] = max(errors["batch_single"],float((many[0]-rt.evaluate(cc,bits,alpha)).abs().max()))
                        errors["fresh_cached"] = max(errors["fresh_cached"],float((rt.evaluate(rt.prepare(s,r),bits,alpha)-many[0]).abs().max()))
            # q12 also checks the all-subset bound against all4096 masks on one
            # existing validation input, outside the primary sampled evaluation.
            if q==12:
                all_bits = torch.tensor(bit_design(q),dtype=torch.float64)
                context=local.states(vs[:1],vr[:1])
                bound_rt=BoundedAffine(local)
                for alpha in STRENGTHS:
                    p, info=bound_rt.decision(bound_rt.cached(context),all_bits,alpha)
                    affine=FusedAffine(local).evaluate(FusedAffine(local).cached(context),all_bits,alpha)
                    exact=local.exact(context,all_bits,alpha)
                    held=all_bits.sum(1)>2
                    excess=float((affine-exact)[held].norm())-info["error_norm_upper"]
                    errors["exhaustive_q12_bound_excess"]=max(errors["exhaustive_q12_bound_excess"],max(0.,excess))
    assert max(errors.values()) < 1e-9, dict(errors)
    dump(run/"smoke.json",dict(utc=now(),status="passed",errors=dict(errors),
        training_contexts=32,validation_contexts=4,teacher_seed=SEEDS[0],
        layouts=6, pca_smoke_budget=8, q12_exhaustive_bound_inputs_per_position=1,
        calibration_model_access=False,new_official_interventions_accessed=False))
    event(run,"scale_smoke_passed",max_error=max(errors.values()))


# 用途：在小规模检查通过后保存实现副本和哈希，固定后续运行使用的代码。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def lock(run):
    verify(run)
    assert read_json(run/"smoke.json")["status"]=="passed"
    assert not (run/"implementation.json").exists()
    names=["scripts/scale_context.py","scripts/run_scale_controls.py"]
    record=dict(utc=now(),protocol_sha256=file_sha(run/"protocol.json"),
                sources={name:file_sha(BASE/name) for name in names})
    for name in names:
        dest=run/"source"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,dest)
    dump(run/"implementation.json",record)
    (run/"implementation.sha256").write_text(file_sha(run/"implementation.json")+"\n",encoding="ascii")
    event(run,"scale_implementation_locked",sha256=file_sha(run/"implementation.json"))


# 用途：在一个教师/位置/源规模设置中运行固定方法，记录误差、上界判定和参考承诺。
# 参数：run、seed、position、q、model、gpu、sources、residuals、ids、fit、nodes、out、phase；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：results。
def run_case(run, seed, position, q, model, gpu, sources, residuals, ids, fit, nodes, out, phase):
    assert not out.exists(), f"Preserve partial case before retry: {out}"
    out.mkdir(parents=True)
    started=time.perf_counter()
    _, eval_bits=design(run,q)
    bits=torch.tensor(eval_bits,dtype=torch.float64)
    held=eval_bits.sum(1)>2
    local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
    contexts=[local.states(s[None],r[None]) for s,r in zip(sources,residuals)]
    predictions={}; decision_info={}; runtime_error=0.
    for method,fraction,k in scale_specs(local):
        rt=make_runtime(local,fit,method,k)
        cc=[rt.cached(c) for c in contexts]
        for alpha in STRENGTHS:
            if method=="bounded_affine":
                pairs=[rt.decision(c,bits,alpha) for c in cc]
                values=torch.stack([v[0] for v in pairs]).numpy()
                decision_info[str(alpha)]=[v[1] for v in pairs]
            else:
                values=evaluate_many(rt,cc,bits,alpha)
            predictions[key(method,fraction,alpha)]=values
            for i in range(2):
                fresh=rt.evaluate(rt.prepare(sources[i:i+1],residuals[i:i+1]),bits,alpha).numpy()
                runtime_error=max(runtime_error,float(np.abs(values[i]-fresh).max()))
    assert runtime_error<1e-9
    np.savez_compressed(out/"predictions.npz",**predictions,ids=ids)
    dump(out/"decisions_before_reference.json",decision_info)
    dump(out/"prediction_commitment.json",dict(utc=now(),sha256=file_sha(out/"predictions.npz"),
        decisions_sha256=file_sha(out/"decisions_before_reference.json")))
    refs=independent_reference(gpu,sources,residuals,nodes,bits)
    np.savez_compressed(out/"references.npz",**refs,ids=ids)
    dump(out/"reference_commitment.json",dict(utc=now(),sha256=file_sha(out/"references.npz")))
    results=[]; inputs=[]; orders=[]; diagnostics=[]; bound_excess=0.; threshold_exceptions=0
    for method,fraction,k in scale_specs(local):
        for alpha in STRENGTHS:
            pred=predictions[key(method,fraction,alpha)]
            ref=refs[str(alpha)]
            labels=dict(seed=seed,position=position,q=q,method=method,fraction=fraction,k=k,alpha=alpha)
            results.append(scale_metrics(pred,ref,eval_bits,**labels))
            den=np.square(ref[:,held]).sum((1,2))
            num=np.square((pred-ref)[:,held]).sum((1,2))
            for i in range(len(ids)):
                usable=bool(den[i]>1e-24)
                inputs.append(dict(**labels,input_id=int(ids[i]),usable_effect=usable,
                    effect_squared_sum=float(den[i]),error_squared_sum=float(num[i]),
                    nrmse=float(np.sqrt(num[i]/den[i])) if usable else "",
                    over_10pct=bool(num[i]>.01*den[i]) if usable else ""))
            for order in np.unique(eval_bits.sum(1)[held]).astype(int):
                mask=eval_bits.sum(1)==order
                scores=score(pred,ref,mask)
                orders.append(dict(**labels,order=int(order),coalitions=int(mask.sum()),**scores))
    for alpha in STRENGTHS:
        affine=predictions[key("affine_fused",1.,alpha)]
        ref=refs[str(alpha)]
        for i,c in enumerate(contexts):
            info=decision_info[str(alpha)][i]
            num=float(np.square((affine[i]-ref[i])[held]).sum())
            den=float(np.square(ref[i,held]).sum())
            error_norm=math.sqrt(num)
            excess=max(0.,error_norm-info["error_norm_upper"])
            bound_excess=max(bound_excess,excess)
            relative=math.sqrt(num/den) if den>1e-24 else None
            exception=bool(info["accepted"] and relative is not None and relative>.1+1e-9)
            threshold_exceptions+=int(exception)
            h,z=c
            t=(-alpha*bits*h[None])@local.W.T
            switches=((z[None]+t)>0)!=(z>0)[None]
            diagnostics.append(dict(seed=seed,position=position,q=q,alpha=alpha,input_id=int(ids[i]),
                **info,source_l2=float(h.norm()),source_nonzero=int((h!=0).sum()),
                usable_effect=bool(den>1e-24),effect_squared_sum=den,affine_error_squared_sum=num,
                affine_nrmse=relative if relative is not None else "",
                bound_excess=excess,relative_threshold_exception=exception,
                fraction_states_switched=float(switches[held].double().mean()),
                fraction_queries_any_switch=float(switches[held].any(1).double().mean())))
    exact_error=max(r["maxerr"] for r in results if r["method"]=="exact")
    assert exact_error<1e-9 and bound_excess<1e-9
    csv_write(out/"results.csv",results)
    csv_write(out/"input_results.csv",inputs)
    csv_write(out/"order_results.csv",orders)
    csv_write(out/"bound_diagnostics.csv",diagnostics)
    done=dict(utc=now(),phase=phase,seed=seed,position=position,q=q,inputs=len(ids),
        queried_coalitions=len(eval_bits),scored_coalitions=int(held.sum()),
        arrays=len(results),affected_states=local.J,exact_max_error=exact_error,
        fresh_cached_max_error=runtime_error,bound_excess=bound_excess,
        accepted_relative_threshold_exceptions=threshold_exceptions,
        full_tail_output_vectors=len(ids)*len(eval_bits)*2,seconds=time.perf_counter()-started,
        reference_device="CUDA float64",runtime_device="CPU float64")
    dump(out/"completion.json",done)
    event(run,"scale_case_complete",**done)
    return results


# 用途：对固定的开发输入拟合或检查方法，保存测试前需要的参数和结果。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def develop(run):
    verify(run)
    assert (run/"implementation.json").exists() and not (run/"test_opened.json").exists()
    assert not (run/"development_complete.json").exists()
    images,train_ids,val_ids=training_data()
    train_x=torch.tensor(normalized(images[train_ids]),dtype=torch.float64)
    val_x=torch.tensor(normalized(images[val_ids]),dtype=torch.float64)
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        model=load_teacher(PRIOR_RUN/f"seed_{seed}")
        gpu=load_teacher(PRIOR_RUN/f"seed_{seed}",device="cuda")
        ts,tr=prefix(model,train_x)
        vs,vr=prefix(model,val_x)
        for position in POSITIONS:
            for q in SIZES:
                folder=run/f"seed_{seed}"/f"{position}_q{q}"
                folder.mkdir(parents=True,exist_ok=True)
                if not (folder/"fit_complete.json").exists():
                    assert not (folder/"fit.npz").exists()
                    nodes=choose_source_nodes(ts,position,q)
                    local=ResidualLocal(model,tuple(ts.shape[1:]),nodes)
                    contexts=[local.states(s[None],r[None]) for s,r in zip(ts,tr)]
                    train_bits,_=design(run,q)
                    started=time.perf_counter()
                    fit,diagnostic=fit_scale(local,contexts,train_bits,seed)
                    np.savez_compressed(folder/"fit.npz",**vars(fit),nodes=nodes)
                    dump(folder/"fit_complete.json",dict(utc=now(),seed=seed,position=position,q=q,
                        fit_sha256=file_sha(folder/"fit.npz"),seconds=time.perf_counter()-started,**diagnostic))
                    event(run,"scale_fit_complete",seed=seed,position=position,q=q,J=local.J)
                fit,nodes=load_fit(folder/"fit.npz")
                assert file_sha(folder/"fit.npz")==read_json(folder/"fit_complete.json")["fit_sha256"]
                if not (folder/"validation/completion.json").exists():
                    run_case(run,seed,position,q,model,gpu,vs,vr,val_ids,fit,nodes,folder/"validation","validation")
        del gpu
        torch.cuda.empty_cache()
    verify(run)
    dump(run/"development_complete.json",dict(utc=now(),cases=36,seeds=SEEDS,
        calibration_model_access=False,new_official_interventions_accessed=False))
    event(run,"scale_development_complete",cases=36)


# 用途：在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def timing(run):
    verify(run)
    assert (run/"development_complete.json").exists() and not (run/"test_opened.json").exists()
    out=run/"timing"
    out.mkdir(exist_ok=False)
    images,_,val_ids=training_data()
    x=torch.tensor(normalized(images[val_ids[:4]]),dtype=torch.float64)
    raw=[];costs=[];maximum=0.
    for seed in SEEDS:
        model=load_teacher(PRIOR_RUN/f"seed_{seed}")
        sources,residuals=prefix(model,x)
        for position in POSITIONS:
            for q in SIZES:
                fit,nodes=load_fit(run/f"seed_{seed}/{position}_q{q}/fit.npz")
                local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
                runtimes={f"{m}_f{f}":make_runtime(local,fit,m,k)
                          for m,f,k in scale_specs(local) if m!="exact"}
                runtimes.update({e:make_runtime(local,fit,"exact",local.J,e) for e in EXACT_ENGINES})
                train_bits,eval_bits=design(run,q)
                query_sets={"low_order":torch.tensor(train_bits,dtype=torch.float64),
                            "evaluation":torch.tensor(eval_bits,dtype=torch.float64)}
                rng=np.random.default_rng(seed+q*100+(0 if position=="center" else 1)+20260914)
                for i in range(4):
                    common=local.states(sources[i:i+1],residuals[i:i+1])
                    cached={name:rt.cached(common) for name,rt in runtimes.items()}
                    if i==0:
                        # 用途：递归遍历缓存中的张量，供外层按底层存储去重并统计字节数。
                        # 参数：value；具体形状/来源结合调用处及下方关键步骤阅读。
                        # 返回：逐次生成 value / 逐次生成 tensors(item)。
                        def tensors(value):
                            if isinstance(value,torch.Tensor):
                                yield value
                            elif isinstance(value,(tuple,list)):
                                for item in value:yield from tensors(item)
                            elif isinstance(value,dict):
                                for item in value.values():yield from tensors(item)
                        for name,rt in runtimes.items():
                            actual={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors(cached[name])}
                            retained=[v for owner in (local,rt) for v in vars(owner).values() if isinstance(v,torch.Tensor)]
                            storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in retained}
                            costs.append(dict(seed=seed,position=position,q=q,method=name,
                                affected_states=local.J,cached_tensor_bytes=sum(actual.values()),
                                retained_non_teacher_tensor_bytes=sum(storage.values()),
                                teacher_state_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                                scope="actual cached tensor storage plus retained local/estimator tensors; excludes Python/NumPy overhead and query temporaries, not peak process memory"))
                    for query_set,bt in query_sets.items():
                        for workload in ("from_image","cached"):
                            functions={}
                            for name,rt in runtimes.items():
                                if workload=="cached":
                                    functions[name]=lambda name=name,rt=rt:rt.evaluate(cached[name],bt)
                                else:
                                    # 用途：构造当前运行时的一次实际执行，供计时循环调用。
                                    # 参数：rt；具体形状/来源结合调用处及下方关键步骤阅读。
                                    # 返回：rt.evaluate(rt.prepare(s, r), bt)。
                                    def fn(rt=rt):
                                        s,r=model.prefix(x[i:i+1])
                                        return rt.evaluate(rt.prepare(s,r),bt)
                                    functions[name]=fn
                            with torch.no_grad():
                                for name,fn in functions.items():
                                    maximum=max(maximum,float((fn()-runtimes[name].evaluate(cached[name],bt)).abs().max()))
                                    for _ in range(2):fn()
                                names=list(functions)
                                for round_ in range(7):
                                    for j in rng.permutation(len(names)):
                                        name=names[j];start=time.perf_counter_ns()
                                        for _ in range(3):functions[name]()
                                        raw.append(dict(seed=seed,position=position,q=q,input_id=int(val_ids[i]),
                                            query_set=query_set,queries=len(bt),workload=workload,method=name,
                                            round=round_,seconds=(time.perf_counter_ns()-start)/3e9))
                event(run,"scale_timing_layout_complete",seed=seed,position=position,q=q,rows=len(raw))
    assert maximum<1e-9
    csv_write(out/"raw.csv",raw)
    csv_write(out/"cost_counts.csv",costs)
    group_keys=("seed","position","q","input_id","query_set","queries","workload","method")
    groups=defaultdict(list)
    for r in raw:groups[tuple(r[k] for k in group_keys)].append(r["seconds"])
    medians=[dict(zip(group_keys,k),seconds=float(np.median(v))) for k,v in groups.items()]
    csv_write(out/"input_medians.csv",medians)
    group_keys=tuple(k for k in group_keys if k!="input_id")
    groups=defaultdict(list)
    for r in medians:groups[tuple(r[k] for k in group_keys)].append(r["seconds"])
    summary=[dict(zip(group_keys,k),seconds=float(np.mean(v))) for k,v in groups.items()]
    csv_write(out/"summary.csv",summary)
    exact_groups=defaultdict(list)
    for r in summary:
        if r["method"] in EXACT_ENGINES:
            exact_groups[tuple(r[k] for k in group_keys if k!="method")].append(r)
    fastest=[min(v,key=lambda r:r["seconds"]) for v in exact_groups.values()]
    csv_write(out/"fastest_exact.csv",fastest)
    assert len(raw)==48384
    dump(out/"completion.json",dict(utc=now(),rows=len(raw),cases=36,
        max_path_difference=maximum,training_concurrent=False,alpha=1.))
    event(run,"scale_timing_complete",rows=len(raw))


# 用途：把开发阶段选定的模型、方法、预算与相关文件写入测试前冻结记录。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def freeze(run):
    verify(run)
    assert not (run/"freeze.json").exists() and not (run/"test_opened.json").exists()
    assert (run/"development_complete.json").exists() and (run/"timing/completion.json").exists()
    files=[p for p in run.rglob("*") if p.is_file() and p.name!="events.jsonl"]
    record=dict(utc=now(),protocol_sha256=file_sha(run/"protocol.json"),
        implementation_sha256=file_sha(run/"implementation.json"),
        artifacts={p.relative_to(run).as_posix():file_sha(p) for p in sorted(files)},
        new_official_interventions_accessed=False,prior_official_classification_access=True,
        calibration_model_access=False)
    dump(run/"freeze.json",record)
    (run/"freeze.sha256").write_text(file_sha(run/"freeze.json")+"\n",encoding="ascii")
    event(run,"scale_frozen",artifacts=len(files),sha256=file_sha(run/"freeze.json"))


# 用途：核验冻结状态后处理预留测试输入，保存预测、原网络参考及评分。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def test(run):
    verify(run,True)
    assert not (run/"test/completion.json").exists()
    out=run/"test"
    out.mkdir(exist_ok=True)
    if not (run/"test_opened.json").exists():
        dump(run/"test_opened.json",dict(utc=now(),freeze_sha256=file_sha(run/"freeze.json")))
    ids=np.load(run/"intervention_ids.npy",allow_pickle=False)
    assert np.array_equal(ids,np.arange(768,896))
    images=load_idx(BASE/"data/fashion_mnist/raw","t10k-images-idx3-ubyte.gz")
    x=torch.tensor(normalized(images[ids]),dtype=torch.float64)
    rows=[];completed=[]
    for seed in SEEDS:
        model=load_teacher(PRIOR_RUN/f"seed_{seed}")
        gpu=load_teacher(PRIOR_RUN/f"seed_{seed}",device="cuda")
        sources,residuals=prefix(model,x)
        for position in POSITIONS:
            for q in SIZES:
                fit,nodes=load_fit(run/f"seed_{seed}/{position}_q{q}/fit.npz")
                case=out/f"seed_{seed}"/f"{position}_q{q}"
                if (case/"completion.json").exists():
                    pc=read_json(case/"prediction_commitment.json")
                    rc=read_json(case/"reference_commitment.json")
                    assert file_sha(case/"predictions.npz")==pc["sha256"]
                    assert file_sha(case/"references.npz")==rc["sha256"]
                    rows.extend(csv_read(case/"results.csv"))
                else:
                    rows.extend(run_case(run,seed,position,q,model,gpu,sources,residuals,ids,fit,nodes,case,"official_test"))
                completed.append(read_json(case/"completion.json"))
        del gpu
        torch.cuda.empty_cache()
    csv_write(out/"all_results.csv",rows)
    verify(run,True)
    dump(out/"completion.json",dict(utc=now(),status="completed",cases=len(completed),
        full_tail_output_vectors=sum(r["full_tail_output_vectors"] for r in completed),
        max_exact_error=max(r["exact_max_error"] for r in completed),
        max_bound_excess=max(r["bound_excess"] for r in completed),
        accepted_relative_threshold_exceptions=sum(r["accepted_relative_threshold_exceptions"] for r in completed),
        new_teacher_training_runs=0,new_classification_runs=0,calibration_model_access=False))
    event(run,"scale_test_complete",cases=len(completed))


# 用途：回读已保存数组重新评分并汇总，核对结果与既有记录的一致性。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def analyze(run):
    verify(run,True)
    assert read_json(run/"test/completion.json")["status"]=="completed"
    out=run/"analysis"
    out.mkdir(exist_ok=False)
    counts=defaultdict(int);maximum=0.;tail_difference=0.
    bounds=[];by_order=[];input_rows=[]
    for phase in ("validation","official_test"):
        for seed in SEEDS:
            for position in POSITIONS:
                for q in SIZES:
                    case=(run if phase=="validation" else run/"test")/f"seed_{seed}"/f"{position}_q{q}"
                    if phase=="validation":case=case/"validation"
                    pc=read_json(case/"prediction_commitment.json");rc=read_json(case/"reference_commitment.json")
                    assert pc["utc"]<rc["utc"]<=read_json(case/"completion.json")["utc"]
                    if phase=="official_test":assert read_json(run/"test_opened.json")["utc"]<pc["utc"]
                    assert file_sha(case/"predictions.npz")==pc["sha256"]
                    assert file_sha(case/"references.npz")==rc["sha256"]
                    assert file_sha(case/"decisions_before_reference.json")==pc["decisions_sha256"]
                    _,bits=design(run,q);held=bits.sum(1)>2
                    with np.load(case/"predictions.npz",allow_pickle=False) as pp,np.load(case/"references.npz",allow_pickle=False) as rr:
                        assert np.array_equal(pp["ids"],rr["ids"])
                        if phase=="official_test":assert np.array_equal(pp["ids"],np.arange(768,896))
                        for r in csv_read(case/"results.csv"):
                            p=pp[key(r["method"],float(r["fraction"]),float(r["alpha"]))]
                            ref=rr[str(float(r["alpha"]))]
                            den=np.square(ref[:,held]).sum((1,2))
                            num=np.square((p-ref)[:,held]).sum((1,2))
                            value=float(np.sqrt(num.sum()/max(den.sum(),1e-30)))
                            maximum=max(maximum,abs(value-float(r["nrmse"])))
                            usable=den>1e-24
                            per=np.sqrt(num[usable]/den[usable])
                            p95=float(np.quantile(per,.95)) if len(per) else 0.
                            tail_difference=max(tail_difference,abs(p95-float(r["input_p95"])))
                            counts[phase]+=1
                    if phase=="official_test":
                        bounds.extend(csv_read(case/"bound_diagnostics.csv"))
                        by_order.extend(csv_read(case/"order_results.csv"))
                        input_rows.extend(csv_read(case/"input_results.csv"))
    assert max(maximum,tail_difference)<1e-12
    raw=csv_read(run/"test/all_results.csv")
    csv_write(out/"seed_nrmse.csv",raw)
    group_keys=("position","q","method","fraction","alpha")
    groups=defaultdict(list)
    for r in raw:groups[tuple(r[k] for k in group_keys)].append(r)
    summary=[]
    for k,rs in sorted(groups.items()):
        assert len(rs)==6
        vals=np.array([float(r["nrmse"]) for r in rs])
        summary.append(dict(zip(group_keys,k),seeds=6,nrmse_mean=float(vals.mean()),nrmse_sd=float(vals.std(ddof=1)),
            seeds_meet_10pct=int((vals<=.1).sum()),
            input_p95_mean=float(np.mean([float(r["input_p95"]) for r in rs])),
            input_fraction_over_10pct_mean=float(np.mean([float(r["input_fraction_over_10pct"]) for r in rs])),
            zero_effect_inputs_mean=float(np.mean([float(r["zero_effect_inputs"]) for r in rs])),
            effect_rms_mean=float(np.mean([float(r["effect_rms"]) for r in rs]))))
    csv_write(out/"summary.csv",summary)
    csv_write(out/"bound_diagnostics.csv",bounds)
    csv_write(out/"order_results.csv",by_order)
    csv_write(out/"input_results.csv",input_rows)
    paired=[]
    for seed in SEEDS:
        for position in POSITIONS:
            for q in SIZES:
                for fraction in BUDGETS:
                    for alpha in STRENGTHS:
                        values={r["method"]:float(r["nrmse"]) for r in raw if int(r["seed"])==seed and r["position"]==position
                                and int(r["q"])==q and float(r["fraction"])==fraction and float(r["alpha"])==alpha}
                        for control in ("gate_weight","pca_fused"):
                            paired.append(dict(seed=seed,position=position,q=q,fraction=fraction,alpha=alpha,
                                control=control,greedy_minus_control=values["greedy"]-values[control]))
    csv_write(out/"paired_differences.csv",paired)
    verify(run,True)
    dump(out/"verification.json",dict(utc=now(),status="passed",arrays_rescored=dict(counts),
        max_nrmse_difference=maximum,max_input_p95_difference=tail_difference,
        source_and_freeze_hashes_unchanged=True,calibration_model_access=False,new_confirmatory_p_values=0))
    event(run,"scale_analysis_complete",arrays=dict(counts),max_nrmse_difference=maximum)


# 用途：检查 4/8/12 源坐标和两个空间位置，比较近似方法及有上界的精确回退。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=("init","smoke","lock","develop","timing","freeze","test","analyze"))
    args=parser.parse_args()
    configure_cpu()
    try:
        globals()[args.stage](RUN)
    except Exception as exc:
        if RUN.exists():
            event(RUN,"stage_failed",stage=args.stage,error=repr(exc),
                  runner_sha256=file_sha(Path(__file__)),runtime_sha256=file_sha(BASE/"scripts/scale_context.py"))
        raise


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":
    main()

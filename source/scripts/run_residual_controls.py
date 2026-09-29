# 中文阅读说明｜用途：调度残差块内部干预的开发、计时、冻结、测试与回读复核。
# 数据流：预测先保存承诺，再由原始未融合网络独立产生参考；测试不会重新拟合方法。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Frozen residual-block context experiment, including independent references."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import shutil
import time
from residual_context import *
from prepare_external import load_idx


# 用途：读取保存的图像与训练/验证上下文索引，保持已有划分。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：(images, train_ids, val_ids)。
def training_data():
    images=np.load(DATA / "official_train_images.npy",mmap_mode="r",allow_pickle=False)
    with np.load(DATA / "context_indices.npz",allow_pickle=False) as saved:
        train_ids=saved["train"];val_ids=saved["validation"]
    return images,train_ids,val_ids


# 用途：分批执行教师前缀，拼接全部输入的源激活与残差旁路。
# 参数：model、x；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))。
@torch.no_grad()
def prefix(model,x):
    pairs=[model.prefix(x[i:i+64]) for i in range(0,len(x),64)]
    return torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs])


# 用途：复制源特征图、按掩码衰减指定坐标，再运行原教师尾部生成各强度参考。
# 参数：model_gpu、sources、residuals、bits；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：values。
def references(model_gpu,sources,residuals,bits):
    bt=bits.cuda();values={}
    with torch.no_grad():
        for alpha in STRENGTHS:
            result=[]
            for source,residual in zip(sources,residuals):
                s=source[None].cuda();r=residual[None].cuda()
                h=s.expand(len(bt),-1,-1,-1).clone()
                h.flatten(1)[:,references.nodes]*=(1-alpha*bt)
                result.append((model_gpu.tail(h,r)-model_gpu.tail(s,r)).cpu().numpy())
            values[str(alpha)]=np.stack(result)
    return values


# 用途：用规模较小且有可比较答案的情形检查实现，数值不一致时立即停止。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def smoke(run):
    verify_protocol(run)
    images,train_ids,val_ids=training_data()
    model=load_teacher(run / f"seed_{SEEDS[0]}")
    sources,residuals=prefix(model,torch.tensor(normalized(images[train_ids[:32]]),dtype=torch.float64))
    nodes=choose_nodes(sources);local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
    contexts=[local.states(s[None],r[None]) for s,r in zip(sources,residuals)]
    fit,_=fit_context(local,contexts,SEEDS[0])
    vs,vr=prefix(model,torch.tensor(normalized(images[val_ids[:4]]),dtype=torch.float64))
    vc=[local.states(s[None],r[None]) for s,r in zip(vs,vr)]
    bits=torch.tensor(bit_design(8),dtype=torch.float64)
    gpu=load_teacher(run / f"seed_{SEEDS[0]}",device="cuda")
    references.nodes=nodes;refs=references(gpu,vs,vr,bits)
    errors=defaultdict(float)
    for i,(s,r,c) in enumerate(zip(vs,vr,vc)):
        for engine in EXACT_ENGINES:
            errors["acquisition_engines"]=max(errors["acquisition_engines"],float((local.states(s[None],r[None],engine)[1]-c[1]).abs().max()))
        affine=local.affine@local.vector(s[None],r[None])+local.row_bias
        errors["affine_map"]=max(errors["affine_map"],float((affine-c[1]).abs().max()))
        grad_source=s[None].detach().clone().requires_grad_(True);logits=model.tail(grad_source,r[None])
        jac=torch.stack([torch.autograd.grad(logits[0,j],grad_source,retain_graph=True)[0].flatten()[nodes] for j in range(10)],1).detach()
        for alpha in STRENGTHS:
            direct=local.full(s[None],r[None],bits,alpha)
            errors["exact_cpu_tail"]=max(errors["exact_cpu_tail"],float((local.exact(c,bits,alpha)-direct).abs().max()))
            errors["reference_cuda_cpu"]=max(errors["reference_cuda_cpu"],float(np.abs(refs[str(alpha)][i]-direct.numpy()).max()))
            old_region=(((-alpha*bits*c[0][None])@local.W.T)*(c[1]>0)[None])@local.U.T
            errors["autograd_original_region"]=max(errors["autograd_original_region"],float((old_region-(-alpha*bits*c[0][None])@jac).abs().max()))
    for method in METHODS+["mean_gate","exact","original_region"]:
        k=0 if method=="mean_gate" else local.J if method in ("exact","original_region") else 8
        rt=ContextRuntime(local,fit,method,k);cached=[rt.cached(c) for c in vc]
        for alpha in STRENGTHS:
            batch=rt.evaluate_many(cached,bits,alpha)
            for i,(s,r,c) in enumerate(zip(vs,vr,cached)):
                one=rt.evaluate(c,bits,alpha);fresh=rt.evaluate(rt.prepare(s[None],r[None]),bits,alpha)
                errors["batch_single"]=max(errors["batch_single"],float((batch[i]-one).abs().max()))
                errors["fresh_cached"]=max(errors["fresh_cached"],float((fresh-one).abs().max()))
    assert max(errors.values())<1e-9,dict(errors)
    dump(run / "smoke.json",dict(utc=now(),status="passed",training_contexts=32,validation_contexts=4,
        source_rule="smoke only; final nodes and fit use all512 training contexts",errors=dict(errors),official_test_model_access=False))
    event(run,"residual_smoke_passed",**dict(errors))


# 用途：在小规模检查通过后保存实现副本和哈希，固定后续运行使用的代码。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def lock(run):
    verify_protocol(run)
    assert read_json(run / "smoke.json")["status"]=="passed"
    assert not (run / "implementation.json").exists()
    names=["scripts/residual_context.py","scripts/run_residual_controls.py",
           "upstream/CAT_MT_phase7_v0.7/code/prepare_external.py","upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py"]
    record=dict(utc=now(),protocol_sha256=sha(run / "protocol.json"),sources={n:sha(BASE/n) for n in names})
    for name in names:
        dest=run / "source" / name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,dest)
    dump(run / "implementation.json",record)
    (run / "implementation.sha256").write_text(sha(run / "implementation.json")+"\n",encoding="ascii")
    event(run,"residual_implementation_locked",sha256=sha(run / "implementation.json"))


# 用途：检查已保存的协议、输入、源码或冻结记录；验证失败就停止。
# 参数：run、frozen；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def verify(run,frozen=False):
    verify_protocol(run)
    impl=read_json(run / "implementation.json")
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert sha(run / "implementation.json")== (run / "implementation.sha256").read_text(encoding="ascii").strip()
    assert impl["protocol_sha256"]==sha(run / "protocol.json")
    for name,value in impl["sources"].items():
        assert sha(BASE/name)==value and sha(run / "source" / name)==value,name
    if frozen:
        frozen_record=read_json(run / "freeze.json")
        assert sha(run / "freeze.json")== (run / "freeze.sha256").read_text(encoding="ascii").strip()
        assert frozen_record["implementation_sha256"]==sha(run / "implementation.json")
        for name,value in frozen_record["artifacts"].items():assert sha(run/name)==value,name


# 用途：处理一位残差教师的指定输入：构造局部状态、保存预测，再生成原尾部参考和指标。
# 参数：run、seed、ids、x、out、phase；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：rows。
def case(run,seed,ids,x,out,phase):
    assert not (out / "completion.json").exists()
    out.mkdir(exist_ok=True)
    assert not (out / "predictions.npz").exists(),"Preserve partial predictions and inspect before rerunning"
    directory=run / f"seed_{seed}"
    model=load_teacher(directory);fit,nodes=load_fit(directory / "fit.npz")
    sources,residuals=prefix(model,x);local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
    contexts=[local.states(s[None],r[None]) for s,r in zip(sources,residuals)]
    bits=torch.tensor(bit_design(8),dtype=torch.float64)
    predictions={};runtime_check=0.;start=time.perf_counter()
    for method,fraction,k in specs(local):
        rt=ContextRuntime(local,fit,method,k);cc=[rt.cached(c) for c in contexts]
        for alpha in STRENGTHS:
            value=rt.evaluate_many(cc,bits,alpha).numpy()
            predictions[key(method,fraction,alpha)]=value
            for i in range(min(2,len(x))):
                fresh=rt.evaluate(rt.prepare(sources[i:i+1],residuals[i:i+1]),bits,alpha).numpy()
                runtime_check=max(runtime_check,float(np.abs(value[i]-fresh).max()))
    assert runtime_check<1e-9
    # 保存具名数值数组，便于后来按同一索引回读和独立复算。
    np.savez_compressed(out / "predictions.npz",**predictions,ids=ids)
    dump(out / "prediction_commitment.json",dict(utc=now(),sha256=sha(out / "predictions.npz")))
    gpu=load_teacher(directory,device="cuda");references.nodes=nodes
    refs=references(gpu,sources,residuals,bits)
    np.savez_compressed(out / "references.npz",**refs,ids=ids)
    dump(out / "reference_commitment.json",dict(utc=now(),sha256=sha(out / "references.npz")))
    rows=[metrics(predictions[key(m,f,a)],refs[str(a)],seed,m,f,k,a) for m,f,k in specs(local) for a in STRENGTHS]
    exact_error=max(r["maxerr"] for r in rows if r["method"]=="exact")
    assert exact_error<1e-9
    csv_write(out / "results.csv",rows)
    record=dict(utc=now(),seed=seed,phase=phase,inputs=len(ids),affected_states=local.J,
        exact_max_error=exact_error,fresh_cached_max_error=runtime_check,
        arrays=len(rows),seconds=time.perf_counter()-start,reference_device="CUDA float64",runtime_device="CPU float64")
    dump(out / "completion.json",record)
    event(run,"residual_case_complete",**record)
    del gpu;torch.cuda.empty_cache()
    return rows


# 用途：对固定的开发输入拟合或检查方法，保存测试前需要的参数和结果。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def develop(run):
    verify(run);assert not (run / "test_opened.json").exists()
    assert (run / "training_complete.json").exists()
    images,train_ids,val_ids=training_data()
    train_x=torch.tensor(normalized(images[train_ids]),dtype=torch.float64)
    val_x=torch.tensor(normalized(images[val_ids]),dtype=torch.float64)
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        directory=run / f"seed_{seed}"
        if not (directory / "fit_complete.json").exists():
            assert not (directory / "fit.npz").exists()
            model=load_teacher(directory);sources,residuals=prefix(model,train_x)
            nodes=choose_nodes(sources);local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
            contexts=[local.states(s[None],r[None]) for s,r in zip(sources,residuals)]
            start=time.perf_counter();fit,diagnostic=fit_context(local,contexts,seed)
            np.savez_compressed(directory / "fit.npz",**vars(fit),nodes=nodes)
            dump(directory / "fit_complete.json",dict(utc=now(),seed=seed,affected_states=local.J,
                fit_seconds=time.perf_counter()-start,fit_sha256=sha(directory / "fit.npz"),**diagnostic))
            event(run,"residual_fit_complete",seed=seed,J=local.J,training_active=diagnostic["training_active_units"])
        if not (directory / "validation/completion.json").exists():
            case(run,seed,val_ids,val_x,directory / "validation","validation")
    verify(run)
    dump(run / "development_complete.json",dict(utc=now(),seeds=SEEDS,official_test_model_access=False,calibration_model_access=False))


# 用途：在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def timing(run):
    verify(run);assert not (run / "test_opened.json").exists()
    assert (run / "development_complete.json").exists() and (run / "training_complete.json").exists()
    out=run / "timing";out.mkdir(exist_ok=False)
    images,_,val_ids=training_data();x=torch.tensor(normalized(images[val_ids[:4]]),dtype=torch.float64)
    bits=torch.tensor(bit_design(8),dtype=torch.float64)
    raw=[];costs=[];maximum=0.
    for seed in SEEDS:
        model=load_teacher(run / f"seed_{seed}");fit,nodes=load_fit(run / f"seed_{seed}/fit.npz")
        sources,residuals=prefix(model,x);local=ResidualLocal(model,tuple(sources.shape[1:]),nodes)
        runtimes={f"{m}_f{f}":ContextRuntime(local,fit,m,k) for m,f,k in specs(local) if m!="exact"}
        runtimes.update({engine:ContextRuntime(local,fit,"exact",local.J,engine) for engine in EXACT_ENGINES})
        for name,rt in runtimes.items():costs.append(dict(seed=seed,method=name,**rt.costs()))
        rng=np.random.default_rng(seed+20260913)
        for i in range(4):
            common=local.states(sources[i:i+1],residuals[i:i+1])
            cached={name:rt.cached(common) for name,rt in runtimes.items()}
            for q in (37,256):
                bt=bits[low_order(8)[0]] if q==37 else bits
                for workload in ("from_image","cached"):
                    functions={}
                    for name,rt in runtimes.items():
                        if workload=="cached":functions[name]=lambda rt=rt,name=name:rt.evaluate(cached[name],bt)
                        else:
                            # 用途：构造当前运行时的一次实际执行，供计时循环调用。
                            # 参数：rt；具体形状/来源结合调用处及下方关键步骤阅读。
                            # 返回：rt.evaluate(rt.prepare(s, r), bt)。
                            def fn(rt=rt):
                                s,r=model.prefix(x[i:i+1]);return rt.evaluate(rt.prepare(s,r),bt)
                            functions[name]=fn
                    with torch.no_grad():
                        for name,fn in functions.items():
                            baseline=runtimes[name].evaluate(cached[name],bt)
                            maximum=max(maximum,float((fn()-baseline).abs().max()))
                            for _ in range(3):fn()
                        names=list(functions)
                        for round_ in range(9):
                            for index in rng.permutation(len(names)):
                                name=names[index];start=time.perf_counter_ns()
                                for _ in range(3):functions[name]()
                                raw.append(dict(seed=seed,input_index=int(val_ids[i]),queries=q,workload=workload,
                                    method=name,round=round_,seconds=(time.perf_counter_ns()-start)/3e9))
        event(run,"residual_timing_seed_complete",seed=seed,rows=len(raw))
    assert maximum<1e-9
    csv_write(out / "raw.csv",raw);csv_write(out / "cost_counts.csv",costs)
    grouped=defaultdict(list)
    for r in raw:grouped[(r["seed"],r["input_index"],r["queries"],r["workload"],r["method"])].append(r["seconds"])
    medians=[dict(zip(("seed","input_index","queries","workload","method"),k),seconds=float(np.median(v))) for k,v in grouped.items()]
    csv_write(out / "input_medians.csv",medians)
    average=defaultdict(list)
    for r in medians:average[(r["seed"],r["queries"],r["workload"],r["method"])].append(r["seconds"])
    summary=[dict(zip(("seed","queries","workload","method"),k),seconds=float(np.mean(v))) for k,v in average.items()]
    csv_write(out / "summary.csv",summary)
    fastest=[]
    for seed in SEEDS:
        for q in (37,256):
            for workload in ("from_image","cached"):
                pool=[r for r in summary if r["seed"]==seed and r["queries"]==q and r["workload"]==workload and r["method"] in EXACT_ENGINES]
                fastest.append(min(pool,key=lambda r:r["seconds"]))
    csv_write(out / "fastest_exact.csv",fastest)
    dump(out / "completion.json",dict(utc=now(),rows=len(raw),max_path_difference=maximum,training_concurrent=False))


# 用途：把开发阶段选定的模型、方法、预算与相关文件写入测试前冻结记录。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def freeze(run):
    verify(run);assert not (run / "test_opened.json").exists() and not (run / "freeze.json").exists()
    assert (run / "development_complete.json").exists() and (run / "timing/completion.json").exists()
    files=[run/"protocol.json",run/"implementation.json",run/"smoke.json",run/"intervention_ids.npy",
           run/"training_complete.json",run/"development_complete.json"]
    for seed in SEEDS:files.extend(p for p in (run / f"seed_{seed}").rglob("*") if p.is_file())
    files.extend(p for p in (run / "timing").iterdir() if p.is_file())
    record=dict(utc=now(),protocol_sha256=sha(run / "protocol.json"),implementation_sha256=sha(run / "implementation.json"),
        artifacts={p.relative_to(run).as_posix():sha(p) for p in sorted(files)},
        official_test_model_access=False,calibration_model_access=False)
    dump(run / "freeze.json",record)
    (run / "freeze.sha256").write_text(sha(run / "freeze.json")+"\n",encoding="ascii")
    event(run,"residual_frozen",artifacts=len(files),sha256=sha(run / "freeze.json"))


# 用途：核验冻结状态后处理预留测试输入，保存预测、原网络参考及评分。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def test(run):
    verify(run,True);out=run / "test";out.mkdir(exist_ok=True)
    assert not (out / "completion.json").exists()
    if not (run / "test_opened.json").exists():
        dump(run / "test_opened.json",dict(utc=now(),freeze_sha256=sha(run / "freeze.json")))
        event(run,"residual_test_opened",ids="512..767")
    images=load_idx(BASE / "data/fashion_mnist/raw","t10k-images-idx3-ubyte.gz")
    labels=load_idx(BASE / "data/fashion_mnist/raw","t10k-labels-idx1-ubyte.gz")
    ids=np.load(run / "intervention_ids.npy",allow_pickle=False)
    assert np.array_equal(ids,np.arange(512,768))
    x=torch.tensor(normalized(images[ids]),dtype=torch.float64)
    results=[];classes=[]
    for seed in SEEDS:
        cp=out / f"classification_{seed}.json"
        if not cp.exists():
            model=load_teacher(run / f"seed_{seed}",double=False,device="cuda");logits=[]
            with torch.no_grad():
                for i in range(0,len(images),512):
                    logits.append(model(torch.tensor(normalized(images[i:i+512]),device="cuda")).cpu().numpy())
            z=np.concatenate(logits)
            np.savez_compressed(out / f"classification_{seed}.npz",logits=z,labels=labels)
            dump(cp,dict(seed=seed,accuracy=float((z.argmax(1)==labels).mean()),n=10000,
                cross_entropy=float(F.cross_entropy(torch.tensor(z),torch.tensor(labels,dtype=torch.long))),test_used_for_selection=False))
            del model;torch.cuda.empty_cache()
        classes.append(read_json(cp))
        folder=out / f"seed_{seed}"
        if (folder / "completion.json").exists():results.extend(csv_read(folder / "results.csv"))
        else:results.extend(case(run,seed,ids,x,folder,"official_test"))
    csv_write(out / "all_results.csv",results);csv_write(out / "classification.csv",classes)
    verify(run,True)
    dump(out / "completion.json",dict(utc=now(),status="completed",seeds=SEEDS,cases=6,
        full_tail_output_vectors=6*256*256*2,calibration_model_access=False,
        max_exact_error=max(read_json(out / f"seed_{s}/completion.json")["exact_max_error"] for s in SEEDS)))
    event(run,"residual_test_complete",cases=6)


# 用途：回读已保存数组重新评分并汇总，核对结果与既有记录的一致性。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def analyze(run):
    verify(run,True);assert read_json(run / "test/completion.json")["status"]=="completed"
    out=run / "analysis";out.mkdir(exist_ok=False)
    raw=csv_read(run / "test/all_results.csv")
    rows=[];randoms=defaultdict(list)
    for r in raw:
        one={k:r[k] for k in ("seed","method","fraction","k","alpha","nrmse")}
        for k in ("seed","k"):one[k]=int(one[k])
        for k in ("fraction","alpha","nrmse"):one[k]=float(one[k])
        if one["method"].startswith("active_random_"):
            randoms[(one["seed"],one["fraction"],one["k"],one["alpha"])].append(one["nrmse"])
        else:rows.append(one)
    for (seed,fraction,k,alpha),values in randoms.items():
        assert len(values)==4
        rows.append(dict(seed=seed,method="active_random_mean",fraction=fraction,k=k,alpha=alpha,nrmse=float(np.mean(values))))
    csv_write(out / "seed_nrmse.csv",rows)
    groups=defaultdict(list)
    for r in rows:groups[(r["method"],r["fraction"],r["alpha"])].append(r["nrmse"])
    summary=[dict(zip(("method","fraction","alpha"),k),seeds=len(v),nrmse_mean=float(np.mean(v)),
        nrmse_sd=float(np.std(v,ddof=1)),seeds_meet_10pct=int((np.array(v)<=.1).sum())) for k,v in sorted(groups.items())]
    csv_write(out / "summary.csv",summary)
    differences=[]
    for seed in SEEDS:
        for fraction in FRACTIONS:
            for alpha in STRENGTHS:
                selected={r["method"]:r["nrmse"] for r in rows if r["seed"]==seed and r["fraction"]==fraction and r["alpha"]==alpha}
                for control in ("weight","gate_weight","activity","pca_fused","active_random_mean"):
                    differences.append(dict(seed=seed,fraction=fraction,alpha=alpha,control=control,greedy_minus_control=selected["greedy"]-selected[control]))
    csv_write(out / "paired_differences.csv",differences)
    counts=defaultdict(int);maximum=0.;held=bit_design(8).sum(1)>2
    for phase in ("validation","official_test"):
        for seed in SEEDS:
            folder=run / f"seed_{seed}/validation" if phase=="validation" else run / f"test/seed_{seed}"
            pc=read_json(folder / "prediction_commitment.json");rc=read_json(folder / "reference_commitment.json")
            assert pc["utc"]<rc["utc"]<=read_json(folder / "completion.json")["utc"]
            assert sha(folder / "predictions.npz")==pc["sha256"] and sha(folder / "references.npz")==rc["sha256"]
            if phase=="official_test":assert read_json(run / "test_opened.json")["utc"]<pc["utc"]
            with np.load(folder / "predictions.npz",allow_pickle=False) as pp,np.load(folder / "references.npz",allow_pickle=False) as rr:
                assert np.array_equal(pp["ids"],rr["ids"])
                if phase=="official_test":assert np.array_equal(pp["ids"],np.arange(512,768))
                for r in csv_read(folder / "results.csv"):
                    p=pp[key(r["method"],float(r["fraction"]),float(r["alpha"]))];ref=rr[str(float(r["alpha"]))]
                    value=float(np.sqrt(np.square((p-ref)[:,held]).sum()/max(np.square(ref[:,held]).sum(),1e-30)))
                    maximum=max(maximum,abs(value-float(r["nrmse"])));counts[phase]+=1
    assert maximum<1e-12
    for seed in SEEDS:
        summary=read_json(run / f"seed_{seed}/teacher_summary.json")
        history=read_json(run / f"seed_{seed}/training_history.json")
        assert min(history,key=lambda r:r["validation_cross_entropy"])["epoch"]==summary["best_epoch"]
        assert sha(run / f"seed_{seed}/teacher.pt")==summary["checkpoint_sha256"]
    verify(run,True)
    dump(out / "verification.json",dict(utc=now(),status="passed",arrays_rescored=dict(counts),max_nrmse_difference=maximum,
        source_and_freeze_hashes_unchanged=True,all_lowest_validation_ce_checkpoints=True,
        calibration_model_access=False,new_confirmatory_p_values=0))
    event(run,"residual_analysis_complete",arrays_rescored=dict(counts),max_difference=maximum)


# 用途：调度残差块内部干预的开发、计时、冻结、测试与回读复核。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,
        choices=("smoke","lock","develop","timing","freeze","test","analyze"))
    args=parser.parse_args();configure_cpu();globals()[args.stage](RUN)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

"""Compare head activity and GAP/linear readouts in the Fashion-MNIST structural controls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import platform
import random
import shutil
import time
from pathlib import Path
from structure_runtime import *
from external_experiment import DATA, event, csv_read, teacher_validation
from run_external import verify_freeze as verify_prior_freeze, verify_protocol as verify_prior_protocol
from prepare_external import load_idx

DEFAULT = BASE / "runs/fashion_structure_20260911"
SOURCES = ["scripts/structure_runtime.py", "scripts/run_structure_controls.py",
           "scripts/research_runtime.py", "scripts/external_experiment.py", "scripts/run_external.py",
           "upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py",
           "upstream/CAT_MT_phase7_v0.7/code/context_module.py",
           "upstream/CAT_MT_phase7_v0.7/code/prepare_external.py",
           "upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py"]


# 读取保存的图像与训练/验证上下文索引，保持已有划分。
def training_data():
    return (np.load(DATA / "official_train_images.npy", mmap_mode="r", allow_pickle=False),
            np.load(DATA / "official_train_labels.npy", mmap_mode="r", allow_pickle=False),
            np.load(DATA / "split_indices.npz", allow_pickle=False),
            np.load(DATA / "context_indices.npz", allow_pickle=False))


def initialize(run):
    verify_prior_protocol(PRIOR)
    verify_prior_freeze(PRIOR)
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "intervention_ids.npy", np.arange(256, 512))
    prior_protocol = read_json(PRIOR / "execution_protocol.json")
    protocol = dict(utc=now(), version="activity-gap-controls-v1",
        status="development-informed follow-up with locked additional intervention batch",
        prior_knowledge="all prior six CNN classification outputs and interventions at official IDs 0..255 were evaluated; head inactivity on 512 training contexts was inspected",
        scope="same dataset and same six original CNN teachers; NOT a new external dataset or untouched overall test pool",
        data=dict(training=50000, validation=5000, calibration=5000, calibration_model_access=False,
            method_training_contexts=512, validation_contexts=128, intervention_test_ids="official indices 256..511",
            previous_intervention_ids="0..255", disjoint_intervention_batches=True,
            classification="new GAP teachers on all 10000 official test images, only after freeze"),
        seeds=SEEDS, old_teachers="reuse unchanged; no seed excluded", gap_training=prior_protocol["teacher"],
        gap_architecture="Conv(1,8,3,pad1)-ReLU-Conv(8,16,3,pad1)-ReLU-MaxPool2-GAP-Linear(16,10)",
        gap_training_details="same splits, AdamW, lr 0.001, decay 0.0001, batch 128, max 15, patience 4, lowest validation CE, CUDA float32; no augmentation, no tuning or discarding weak teachers",
        gap_intervention_layers={"1":"after ReLU1", "2":"after ReLU2 before MaxPool", "3":"after MaxPool, only GAP plus linear readout remains"},
        cnn_methods=ALL_METHODS, budgets=BUDGETS,
        definitions=dict(active_weight="gate>0 first, then unchanged local weight-norm order",
            activity="descending training activation frequency g, stable index ties",
            gate_weight="descending g*(1-g)*norm(local input weights)*norm(output weights), stable index ties",
            active_random="four fixed independent permutations of units with g>0 followed by units with g=0; average replicate scores within teacher, never select the best",
            all_controls="same original mean-gate treatment of omitted units, same boundary and weights"),
        gap_methods=["exact", "mean_boundary", "original_region"],
        gap_definitions=dict(exact="full affected pooling-window context followed by linear readout; zero head context",
            mean_boundary="replace input-specific windows with training mean windows",
            original_region="keep original ReLU gates and pooling winners; not assumed valid for finite changes"),
        strengths=STRENGTHS, coalitions=256, scored_coalitions=219,
        statistics="descriptive six-seed means and SD; no new confirmatory p-values; four random replicate errors averaged per seed before seed summary",
        predefined_strata={"all":SEEDS,"more_than_32_training_active_head_units":SEEDS[:3],"at_most_32_training_active_head_units":SEEDS[3:]},
        strata_note="defined from previously observed training activity, never from this batch; retain all seeds",
        interpretation="sample inactive does not imply globally dead; GAP replaces head and changes trained function; do not compare raw NRMSE across architectures as method superiority",
        timing=dict(dtype="CPU float64",threads=1,inputs="first four fixed validation contexts",queries=256,
            workloads=["from_image","cached"],warmup=3,rounds=15,inner=3,
            ordering="random permutation per round",training_concurrent=False,
            gap_reference="literal independently rerun tail, descriptive only, not claimed to be the fastest exact competitor"),
        sources={name:sha(BASE/name) for name in SOURCES},
        data_hashes=prior_protocol["data_sha256"],
        prior_freeze_sha256=sha(PRIOR / "frozen_before_test.json"),
        intervention_ids_sha256=sha(run / "intervention_ids.npy"),
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
    for name in SOURCES:
        path=run / "source" / name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,path)
    dump(run / "protocol.json",protocol)
    (run / "protocol.sha256").write_text(sha(run / "protocol.json")+"\n",encoding="ascii")
    event(run,"structure_protocol_locked",sha256=sha(run / "protocol.json"))


def verify(run, frozen=False):
    protocol=read_json(run / "protocol.json")
    assert sha(run / "protocol.json")==(run / "protocol.sha256").read_text(encoding="ascii").strip()
    for name,value in protocol["sources"].items():
        assert sha(BASE/name)==value and sha(run / "source" / name)==value, name
    for name,value in protocol["data_hashes"].items():
        assert sha(BASE/name)==value, name
    assert sha(run / "intervention_ids.npy")==protocol["intervention_ids_sha256"]
    assert sha(PRIOR / "frozen_before_test.json")==protocol["prior_freeze_sha256"]
    verify_prior_freeze(PRIOR)
    if frozen:
        lock=read_json(run / "freeze.json")
        assert sha(run / "freeze.json")== (run / "freeze.sha256").read_text(encoding="ascii").strip()
        assert lock["protocol_sha256"]==sha(run / "protocol.json")
        for name,value in lock["artifacts"].items():
            assert sha(run/name)==value, name
    return protocol


# 只用训练和验证划分训练 GAP 结构对照教师，并保存选中的检查点。
def train_gap(run, seed, images, labels, indices):
    directory=run / f"gap_seed_{seed}"
    if (directory / "teacher_summary.json").exists():
        assert sha(directory / "teacher.pt")==read_json(directory / "teacher_summary.json")["checkpoint_sha256"]
        return
    assert not (directory / "teacher.pt").exists(), "Preserve and inspect incomplete training before retry"
    directory.mkdir(exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=GapCNN().cuda()
    xt=torch.tensor(normalized(images[indices["train"]]),device="cuda",dtype=torch.float32)
    yt=torch.tensor(labels[indices["train"]],device="cuda",dtype=torch.long)
    xv=torch.tensor(normalized(images[indices["validation"]]),device="cuda",dtype=torch.float32)
    yv=torch.tensor(labels[indices["validation"]],device="cuda",dtype=torch.long)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    history=[];best=float("inf");best_epoch=0;stale=0
    # 等待 GPU 完成已提交任务，避免计时只统计异步调度开销。
    torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(1,16):
        model.train();order=torch.randperm(len(xt),generator=generator).cuda()
        total=torch.zeros((),device="cuda")
        for offset in range(0,len(xt),128):
            ix=order[offset:offset+128]
            optimizer.zero_grad(set_to_none=True)
            # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
            loss=F.cross_entropy(model(xt[ix]),yt[ix]);loss.backward();optimizer.step()
            total+=loss.detach()*len(ix)
        ce,acc=teacher_validation(model,xv,yv)
        row=dict(epoch=epoch,train_cross_entropy=float(total.item()/len(xt)),validation_cross_entropy=ce,validation_accuracy=acc)
        history.append(row)
        if ce<best:
            best,best_epoch,stale=ce,epoch,0
            torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},directory / "teacher.pt")
        else:stale+=1
        dump(directory / "training_history.json",history)
        event(run,"gap_epoch",seed=seed,**row,best_epoch=best_epoch)
        if stale>=4:break
    torch.cuda.synchronize()
    dump(directory / "teacher_summary.json",dict(seed=seed,best_epoch=best_epoch,
        best_validation_cross_entropy=best,best_validation_accuracy=history[best_epoch-1]["validation_accuracy"],
        epochs=len(history),seconds=time.perf_counter()-start,checkpoint_sha256=sha(directory / "teacher.pt"),
        test_used_for_selection=False))
    del model,optimizer,xt,yt,xv,yv
    torch.cuda.empty_cache()


# 仅利用该阶段规定的训练/验证输入完成模型或方法开发并保存选择。
def development(run):
    assert not (run / "test_opened.json").exists()
    images,labels,indices,contexts=training_data()
    train_x=torch.tensor(normalized(images[contexts["train"]]),dtype=torch.float64)
    val_x=torch.tensor(normalized(images[contexts["validation"][:4]]),dtype=torch.float64)
    bits=torch.tensor(bit_design(8),dtype=torch.float64)
    for seed in SEEDS:
        cnn_dir=run / f"cnn_seed_{seed}";cnn_dir.mkdir(exist_ok=True)
        if not (cnn_dir / "completion.json").exists():
            model=load_model(PRIOR / f"seed_{seed}")
            diagnostics=[]
            for layer in (1,2):
                fit,nodes=load_fit(PRIOR / f"seed_{seed}/fit_L{layer}.npz")
                with torch.no_grad():sources=source_activation(model,val_x,layer)
                mod=LocalModule(model,tuple(sources.shape[1:]),nodes,layer)
                orders=activity_orders(mod,fit,seed,layer)
                np.savez_compressed(cnn_dir / f"orders_L{layer}.npz",**orders)
                error=0.
                for method in ALL_METHODS:
                    for k in BUDGETS:
                        rt=head_runtime(mod,fit,orders,method,k)
                        for source in sources[:2]:
                            c=mod.prepare(source[None]);common=c["h"],c["window"],c["z3"]
                            fresh=rt.evaluate(rt.prepare(source[None]),bits)
                            cached=rt.evaluate(rt.prepare_cached(common),bits)
                            error=max(error,float((fresh-cached).abs().max()))
                assert error<1e-9
                diagnostics.append(dict(layer=layer,positive_training_head_units=int((fit.gate>0).sum()),
                    active_weight_equals_weight_at16=bool(np.array_equal(orders["active_weight"][:16],fit.weight_order[:16])),
                    active_weight_equals_weight_at32=bool(np.array_equal(orders["active_weight"][:32],fit.weight_order[:32])),
                    fresh_cached_error=error))
            dump(cnn_dir / "completion.json",dict(utc=now(),seed=seed,diagnostics=diagnostics))
        train_gap(run,seed,images,labels,indices)
        gap_dir=run / f"gap_seed_{seed}"
        if not (gap_dir / "development_complete.json").exists():
            gap=load_gap(gap_dir);checks=[]
            for layer in (1,2,3):
                with torch.no_grad():
                    sources=gap.source(train_x,layer);val_sources=gap.source(val_x,layer)
                nodes=choose_nodes(sources)
                local=GapLocal(gap,tuple(sources.shape[1:]),nodes,layer)
                windows=torch.stack([local.prepare(s[None])[1] for s in sources])
                mean=windows.mean(0)
                np.savez_compressed(gap_dir / f"context_L{layer}.npz",nodes=nodes,mean_window=mean.numpy(),layer=layer)
                error=0.
                for s in val_sources:
                    for alpha in STRENGTHS:
                        error=max(error,float((local.evaluate(local.prepare(s[None]),bits,alpha)-local.full_evaluate(s[None],bits,alpha)).abs().max()))
                assert error<1e-9
                checks.append(dict(layer=layer,exact_error=error,boundary_scalars=int(mean.numel()),head_context_scalars=0))
            dump(gap_dir / "development_complete.json",dict(utc=now(),seed=seed,checks=checks))
        event(run,"structure_development_seed_complete",seed=seed)


# 回读固定活动选择排序，保证计时与测试使用同一配置。
def read_orders(run,seed,layer):
    with np.load(run / f"cnn_seed_{seed}/orders_L{layer}.npz",allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


# 生成 CNN 活动筛选对照中预先规定的方法与预算列表。
def cnn_specs():
    return [(m,k) for m in ALL_METHODS for k in BUDGETS]+[("exact",64)]


# 在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
def timing(run):
    assert not (run / "test_opened.json").exists()
    if (run / "timing/completion.json").exists():return
    out=run / "timing";out.mkdir(exist_ok=False)
    images,_,_,contexts=training_data()
    x=torch.tensor(normalized(images[contexts["validation"][:4]]),dtype=torch.float64)
    bits=torch.tensor(bit_design(8),dtype=torch.float64)
    raw=[];costs=[]
    for seed in SEEDS:
        rng=np.random.default_rng(2026091101+seed)
        model=load_model(PRIOR / f"seed_{seed}")
        for layer in (1,2):
            fit,nodes=load_fit(PRIOR / f"seed_{seed}/fit_L{layer}.npz")
            with torch.no_grad():sources=source_activation(model,x,layer)
            mod=LocalModule(model,tuple(sources.shape[1:]),nodes,layer)
            orders=read_orders(run,seed,layer)
            runtimes={f"{m}_k{k}":head_runtime(mod,fit,orders,m,k) for m,k in cnn_specs()}
            for name,rt in runtimes.items():
                costs.append(dict(architecture="cnn",seed=seed,layer=layer,method=name,**rt.cost_counts()))
            for i in range(4):
                image=x[i:i+1];source=sources[i:i+1];c=mod.prepare(source);common=c["h"],c["window"],c["z3"]
                for scope in ("from_image","cached"):
                    if scope=="from_image":
                        fs={name:(lambda rt=rt:rt.evaluate(rt.prepare(source_activation(model,image,layer)),bits)) for name,rt in runtimes.items()}
                    else:
                        fs={name:(lambda rt=rt:rt.evaluate(rt.prepare_cached(common),bits)) for name,rt in runtimes.items()}
                    raw.extend(dict(architecture="cnn",seed=seed,layer=layer,input_index=i,workload=scope,queries=256,**r) for r in timed_functions(fs,rng))
        gap=load_gap(run / f"gap_seed_{seed}")
        for layer in (1,2,3):
            with np.load(run / f"gap_seed_{seed}/context_L{layer}.npz",allow_pickle=False) as saved:
                nodes=saved["nodes"]
            with torch.no_grad():sources=gap.source(x,layer)
            local=GapLocal(gap,tuple(sources.shape[1:]),nodes,layer)
            for i in range(4):
                image=x[i:i+1];source=sources[i:i+1];context=local.prepare(source)
                for scope in ("from_image","cached"):
                    if scope=="from_image":
                        fs={"exact":lambda:local.evaluate(local.prepare(gap.source(image,layer)),bits),
                            "literal_full_tail":lambda:local.full_evaluate(gap.source(image,layer),bits)}
                    else:
                        fs={"exact":lambda:local.evaluate(context,bits),
                            "literal_full_tail":lambda:local.full_evaluate(source,bits)}
                    raw.extend(dict(architecture="gap",seed=seed,layer=layer,input_index=i,workload=scope,queries=256,**r) for r in timed_functions(fs,rng))
        event(run,"structure_timing_seed_complete",seed=seed)
    csv_write(out / "raw.csv",raw);csv_write(out / "cost_counts.csv",costs)
    by_input=defaultdict(list)
    for r in raw:
        key=tuple(r[k] for k in ("architecture","seed","layer","workload","method","input_index"))
        by_input[key].append(r["seconds"])
    grouped=defaultdict(list)
    for key,values in by_input.items():grouped[key[:-1]].append(float(np.median(values)))
    rows=[dict(zip(("architecture","seed","layer","workload","method"),key),seconds=float(np.mean(v))) for key,v in grouped.items()]
    csv_write(out / "summary.csv",rows)
    dump(out / "completion.json",dict(utc=now(),rows=len(raw),training_concurrent=False))


# 把开发阶段选定的模型、方法、预算与相关文件写入测试前冻结记录。
def freeze(run):
    assert not (run / "test_opened.json").exists() and not (run / "freeze.json").exists()
    assert (run / "timing/completion.json").exists()
    for seed in SEEDS:
        assert (run / f"gap_seed_{seed}/development_complete.json").exists()
        assert (run / f"cnn_seed_{seed}/completion.json").exists()
    paths=[]
    for folder in list(run.glob("cnn_seed_*"))+list(run.glob("gap_seed_*"))+[run / "timing"]:
        paths.extend(p for p in folder.rglob("*") if p.is_file())
    dump(run / "freeze.json",dict(utc=now(),protocol_sha256=sha(run / "protocol.json"),
        artifacts={p.relative_to(run).as_posix():sha(p) for p in paths},calibration_model_access=False))
    (run / "freeze.sha256").write_text(sha(run / "freeze.json")+"\n",encoding="ascii")
    event(run,"structure_frozen_before_test",sha256=sha(run / "freeze.json"),artifacts=len(paths))


# 按教师、层和架构生成全部对照方法的预测及原始参考。
def case_predictions(run,seed,layer,architecture,x):
    bits=torch.tensor(bit_design(8),dtype=torch.float64);pred={}
    if architecture=="cnn":
        model=load_model(PRIOR / f"seed_{seed}")
        fit,nodes=load_fit(PRIOR / f"seed_{seed}/fit_L{layer}.npz")
        with torch.no_grad():sources=source_activation(model,x,layer)
        local=LocalModule(model,tuple(sources.shape[1:]),nodes,layer)
        contexts=[local.prepare(s[None]) for s in sources]
        common=[(c["h"],c["window"],c["z3"]) for c in contexts]
        orders=read_orders(run,seed,layer);specs=cnn_specs()
        for method,k in specs:
            rt=head_runtime(local,fit,orders,method,k)
            cc=[rt.prepare_cached(c) for c in common]
            for alpha in STRENGTHS:
                pred[prediction_key(method,k,alpha)]=np.stack([rt.evaluate(c,bits,alpha).numpy() for c in cc])
        reference=lambda a:np.stack([local.full_evaluate(c,bits,a).numpy() for c in contexts])
        state=dict(boundary_scalars=4*len(local.pool_ids),head_context_scalars=64)
    else:
        model=load_gap(run / f"gap_seed_{seed}")
        with np.load(run / f"gap_seed_{seed}/context_L{layer}.npz",allow_pickle=False) as saved:
            nodes=saved["nodes"];mean=torch.tensor(saved["mean_window"],dtype=torch.float64)
        with torch.no_grad():sources=model.source(x,layer)
        local=GapLocal(model,tuple(sources.shape[1:]),nodes,layer)
        contexts=[local.prepare(s[None]) for s in sources]
        specs=[(method,0) for method in ("exact","mean_boundary","original_region")]
        for method,k in specs:
            for alpha in STRENGTHS:
                pred[prediction_key(method,k,alpha)]=np.stack([local.evaluate(c,bits,alpha,method,mean).numpy() for c in contexts])
        reference=lambda a:np.stack([local.full_evaluate(s[None],bits,a).numpy() for s in sources])
        state=dict(boundary_scalars=int(mean.numel()),head_context_scalars=0)
    assert all(np.isfinite(v).all() and np.abs(v[:,0]).max()<1e-10 for v in pred.values())
    return pred,reference,specs,state


# 核验冻结状态后处理预留测试输入，保存预测、原网络参考及评分。
def test(run):
    verify(run,True)
    out=run / "test";out.mkdir(exist_ok=True)
    assert not (out / "completion.json").exists(), "Completed follow-up test cannot be overwritten"
    if not (run / "test_opened.json").exists():
        dump(run / "test_opened.json",dict(utc=now(),freeze_sha256=sha(run / "freeze.json")))
        event(run,"structure_test_opened",ids="256..511")
    images=load_idx(BASE / "data/fashion_mnist/raw","t10k-images-idx3-ubyte.gz")
    labels=load_idx(BASE / "data/fashion_mnist/raw","t10k-labels-idx1-ubyte.gz")
    ids=np.load(run / "intervention_ids.npy",allow_pickle=False)
    assert np.array_equal(ids,np.arange(256,512))
    x=torch.tensor(normalized(images[ids]),dtype=torch.float64)
    rows=[];checks=[];class_rows=[]
    for seed in SEEDS:
        class_path=out / f"gap_classification_{seed}.json"
        if not class_path.exists():
            model=load_gap(run / f"gap_seed_{seed}",False);logits=[]
            with torch.no_grad():
                for i in range(0,len(images),512):
                    logits.append(model(torch.from_numpy(normalized(images[i:i+512]))).numpy())
            logits=np.concatenate(logits)
            np.savez_compressed(out / f"gap_classification_{seed}.npz",logits=logits,labels=labels)
            dump(class_path,dict(seed=seed,accuracy=float((logits.argmax(1)==labels).mean()),
                cross_entropy=float(F.cross_entropy(torch.from_numpy(logits),torch.tensor(labels,dtype=torch.long))),
                n=10000,used_for_selection=False))
        class_rows.append(read_json(class_path))
        for architecture,layers in (("cnn",(1,2)),("gap",(1,2,3))):
            for layer in layers:
                case=out / f"{architecture}_{seed}_L{layer}";case.mkdir(exist_ok=True)
                if (case / "completion.json").exists():
                    rows.extend(csv_read(case / "results.csv"));checks.append(read_json(case / "completion.json"));continue
                assert not (case / "predictions.npz").exists(), "Incomplete case preserved; inspect before resuming"
                start=time.perf_counter()
                pred,reference,specs,state=case_predictions(run,seed,layer,architecture,x)
                np.savez_compressed(case / "predictions.npz",**pred,ids=ids)
                dump(case / "prediction_commitment.json",dict(utc=now(),sha256=sha(case / "predictions.npz")))
                refs={str(a):reference(a) for a in STRENGTHS}
                np.savez_compressed(case / "references.npz",**refs,ids=ids)
                dump(case / "reference_commitment.json",dict(utc=now(),sha256=sha(case / "references.npz")))
                current=[metrics(pred[prediction_key(m,k,a)],refs[str(a)],seed,layer,m,k,a,architecture) for m,k in specs for a in STRENGTHS]
                error=max(r["maxerr"] for r in current if r["method"]=="exact")
                assert error<1e-9
                csv_write(case / "results.csv",current);rows.extend(current)
                check=dict(utc=now(),seed=seed,architecture=architecture,layer=layer,inputs=len(ids),
                    exact_max_error=error,seconds=time.perf_counter()-start,**state)
                dump(case / "completion.json",check);checks.append(check)
                event(run,"structure_test_case_complete",**check)
                del pred,refs,reference
    csv_write(out / "all_results.csv",rows);csv_write(out / "gap_classification.csv",class_rows)
    verify(run,True)
    dump(out / "completion.json",dict(utc=now(),status="completed",cases=len(checks),
        full_tail_output_vectors=6*5*256*256*2,calibration_model_access=False,
        max_exact_error=max(c["exact_max_error"] for c in checks)))
    event(run,"structure_test_completed",cases=len(checks))


def analyze(run):
    verify(run,True)
    assert read_json(run / "test/completion.json")["status"]=="completed"
    out=run / "analysis";out.mkdir(exist_ok=False)
    rows=csv_read(run / "test/all_results.csv")
    fields=("architecture","seed","layer","method","k","alpha")
    for r in rows:
        for key in ("seed","layer","k"):r[key]=int(r[key])
        for key in ("alpha","nrmse","maxerr","input_p95"):r[key]=float(r[key])
    per_seed=[r.copy() for r in rows if r["method"] not in RANDOM_NAMES]
    random_groups=defaultdict(list)
    for r in rows:
        if r["method"] in RANDOM_NAMES:
            random_groups[(r["seed"],r["layer"],r["k"],r["alpha"])].append(r)
    for (seed,layer,k,alpha),group in random_groups.items():
        assert len(group)==4
        per_seed.append(dict(architecture="cnn",seed=seed,layer=layer,method="active_random_mean",k=k,alpha=alpha,
            nrmse=float(np.mean([r["nrmse"] for r in group]))))
    csv_write(out / "seed_nrmse.csv",[{f:r[f] for f in fields+("nrmse",)} for r in per_seed])
    groups=defaultdict(list)
    for r in per_seed:
        for stratum,seeds in read_json(run / "protocol.json")["predefined_strata"].items():
            if r["seed"] in seeds:
                groups[(stratum,r["architecture"],r["layer"],r["method"],r["k"],r["alpha"])].append(r["nrmse"])
    summary=[]
    for key,values in sorted(groups.items()):
        summary.append(dict(zip(("stratum","architecture","layer","method","k","alpha"),key),seeds=len(values),
            nrmse_mean=float(np.mean(values)),nrmse_sd=float(np.std(values,ddof=1)),
            seeds_meet_10pct=int((np.array(values)<=.1).sum())))
    csv_write(out / "summary.csv",summary)
    differences=[]
    for seed in SEEDS:
        for layer in (1,2):
            for k in BUDGETS:
                for alpha in STRENGTHS:
                    greedy=next(r["nrmse"] for r in per_seed if r["architecture"]=="cnn" and r["seed"]==seed and r["layer"]==layer and r["k"]==k and r["alpha"]==alpha and r["method"]=="greedy")
                    for method in ("weight","active_weight","activity","gate_weight","active_random_mean","pca_fused"):
                        control=next(r["nrmse"] for r in per_seed if r["architecture"]=="cnn" and r["seed"]==seed and r["layer"]==layer and r["k"]==k and r["alpha"]==alpha and r["method"]==method)
                        differences.append(dict(seed=seed,layer=layer,k=k,alpha=alpha,control=method,greedy_minus_control=greedy-control))
    csv_write(out / "paired_differences.csv",differences)
    rescored=0;max_difference=0.;held=bit_design(8).sum(1)>2
    opened=read_json(run / "test_opened.json")
    for case in sorted((run / "test").glob("*_L*")):
        commit=read_json(case / "prediction_commitment.json");refcommit=read_json(case / "reference_commitment.json")
        complete=read_json(case / "completion.json")
        assert opened["utc"]<commit["utc"]<refcommit["utc"]<=complete["utc"]
        assert sha(case / "predictions.npz")==commit["sha256"] and sha(case / "references.npz")==refcommit["sha256"]
        with np.load(case / "predictions.npz",allow_pickle=False) as pp,np.load(case / "references.npz",allow_pickle=False) as rr:
            assert np.array_equal(pp["ids"],np.arange(256,512)) and np.array_equal(pp["ids"],rr["ids"])
            for r in csv_read(case / "results.csv"):
                p=pp[prediction_key(r["method"],int(r["k"]),float(r["alpha"]))];ref=rr[str(float(r["alpha"]))]
                value=float(np.sqrt(np.square((p-ref)[:,held]).sum()/max(np.square(ref[:,held]).sum(),1e-30)))
                max_difference=max(max_difference,abs(value-float(r["nrmse"])));rescored+=1
    assert max_difference<1e-12
    for seed in SEEDS:
        history=read_json(run / f"gap_seed_{seed}/training_history.json")
        teacher=read_json(run / f"gap_seed_{seed}/teacher_summary.json")
        assert min(history,key=lambda r:r["validation_cross_entropy"])["epoch"]==teacher["best_epoch"]
    verify(run,True)
    dump(out / "verification.json",dict(utc=now(),status="passed",arrays_rescored=rescored,
        max_nrmse_difference=max_difference,freeze_unchanged=True,calibration_model_access=False,
        all_gap_checkpoints_lowest_validation_ce=True,intervention_ids_disjoint_from_prior=True,
        classification_pool_previously_accessed_for_original_cnn=True,new_confirmatory_p_values=0))
    event(run,"structure_analysis_complete",arrays_rescored=rescored,max_difference=max_difference)


def smoke():
    images,_,_,contexts=training_data()
    x=torch.tensor(normalized(images[contexts["validation"][:2]]),dtype=torch.float64)
    torch.manual_seed(1907);gap=GapCNN().double().eval()
    bits=torch.tensor(bit_design(8),dtype=torch.float64);errors=[]
    for layer in (1,2,3):
        with torch.no_grad():sources=gap.source(x,layer)
        nodes=choose_nodes(sources);local=GapLocal(gap,tuple(sources.shape[1:]),nodes,layer)
        for source in sources:
            for a in STRENGTHS:
                error=float((local.evaluate(local.prepare(source[None]),bits,a)-local.full_evaluate(source[None],bits,a)).abs().max())
                assert error<1e-9;errors.append(error)
    # Algebraic linear-tail counterexample/identity uses arbitrary pooled values.
    source=torch.rand((1,16,14,14),dtype=torch.float64)
    local=GapLocal(gap,(16,14,14),np.arange(8),3)
    linear=local.evaluate(local.prepare(source),bits)
    singleton=linear[[1<<i for i in range(8)]]
    assert float((linear-bits@singleton).abs().max())<1e-12
    print(dict(smoke="passed",max_exact_error=max(errors),linear_tail_additive=True))


# 用活动选择基线和 GAP 网络检查早期结果是否只来自结构冗余。
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=("smoke","init","development","timing","freeze","test","analyze"))
    parser.add_argument("--run-dir",type=Path,default=DEFAULT)
    args=parser.parse_args();run=args.run_dir.resolve()
    assert run.is_relative_to(BASE.resolve())
    configure_cpu()
    if args.stage=="smoke":smoke();return
    if args.stage=="init":initialize(run);return
    verify(run)
    globals()[args.stage](run)


if __name__=="__main__":main()

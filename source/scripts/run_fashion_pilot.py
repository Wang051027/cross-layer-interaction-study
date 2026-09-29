# 中文阅读说明｜用途：执行 Fashion-MNIST 首个教师的训练、验证干预和初步方法比较。
# 数据流：读取固定划分与原数值模块；产生首轮模型、预测及报告所需表格。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""First fixed teacher, train/validation feasibility only; no official test access.

This implements the previously frozen architecture/training settings. Execution
details omitted in that protocol are recorded before training in pilot_protocol.json.
Primary external six-seed test inference is deliberately not represented by a pilot.
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone

from research_runtime import (BASE, UPSTREAM, Fit, LocalModule, HeadRuntime, make_teacher,
                              fit_module, predict, source_activation, bit_design, low_order,
                              score, np, torch, F)
from threadpoolctl import threadpool_limits


# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.sha256(path.read_bytes()).hexdigest()。
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path, value):
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


# 用途：追加带时间的执行事件，记录阶段进度及已完成的结果。
# 参数：output、kind、**kwargs；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def event(output, kind, **kwargs):
    value = dict(utc=datetime.now(timezone.utc).isoformat(),event=kind,**kwargs)
    with (output / "events.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(value)+"\n")
    print(json.dumps(value),flush=True)


# 用途：用统一字段顺序写出多行 CSV 记录。
# 参数：path、rows；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def write_csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


# 用途：在 eval/no_grad 模式批量计算验证交叉熵与准确率，用于选择检查点。
# 参数：model、x、y；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(float(loss.item() / len(x)), float(correct.item() / len(x)))。
def teacher_validation(model, x, y):
    model.eval()
    loss = torch.zeros((),device=x.device)
    correct = torch.zeros((),device=x.device)
    with torch.no_grad():
        for start in range(0,len(x),512):
            z=model(x[start:start+512]); target=y[start:start+512]
            # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
            loss += F.cross_entropy(z,target,reduction="sum")
            correct += (z.argmax(1)==target).sum()
    return float(loss.item()/len(x)), float(correct.item()/len(x))


# 用途：训练首个 Fashion-MNIST 教师并按验证表现保存检查点。
# 参数：output、x、y、indices、protocol；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def train(output, x, y, indices, protocol):
    config=protocol["teacher"]
    seed=config["seeds"][0]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda")
    model=make_teacher("fashion").to(device)
    # Data remain in memory; CPU generator chooses the reproducible epoch order.
    xt=torch.tensor(x[indices["train"]][:,None].astype(np.float32)/np.float32(255),device=device,dtype=torch.float32)
    yt=torch.tensor(y[indices["train"]],device=device,dtype=torch.long)
    xv=torch.tensor(x[indices["validation"]][:,None].astype(np.float32)/np.float32(255),device=device,dtype=torch.float32)
    yv=torch.tensor(y[indices["validation"]],device=device,dtype=torch.long)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    generator=torch.Generator().manual_seed(seed)
    best_loss=float("inf"); best_epoch=0; stale=0; history=[]
    # 等待 GPU 完成已提交任务，避免计时只统计异步调度开销。
    torch.cuda.synchronize();start=time.perf_counter()
    for epoch in range(1,config["max_epochs"]+1):
        model.train();order=torch.randperm(len(xt),generator=generator).to(device)
        total=torch.zeros((),device=device)
        for begin in range(0,len(xt),config["batch"]):
            ix=order[begin:begin+config["batch"]]
            optimizer.zero_grad(set_to_none=True)
            # 这里才会计算训练梯度并更新模型；后续干预阶段固定模型权重。
            loss=F.cross_entropy(model(xt[ix]),yt[ix]);loss.backward();optimizer.step()
            total += loss.detach()*len(ix)
        val_loss,val_accuracy=teacher_validation(model,xv,yv)
        row=dict(epoch=epoch,train_cross_entropy=float(total.item()/len(xt)),
                 validation_cross_entropy=val_loss,validation_accuracy=val_accuracy)
        history.append(row)
        if val_loss < best_loss:
            best_loss=val_loss;best_epoch=epoch;stale=0
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            # 把当前选中的模型参数保存为检查点，后续干预从该固定权重载入。
            torch.save(state,output / "teacher_seed_202609071.pt")
        else:
            stale+=1
        dump(output / "training_history.json",history)
        event(output,"teacher_epoch",**row,best_epoch=best_epoch,elapsed_seconds=time.perf_counter()-start)
        if stale >= config["early_stopping_patience"]:
            break
    torch.cuda.synchronize()
    result=dict(seed=seed,best_epoch=best_epoch,best_validation_cross_entropy=best_loss,
                best_validation_accuracy=history[best_epoch-1]["validation_accuracy"],
                epochs=len(history),seconds=time.perf_counter()-start,
                checkpoint_sha256=sha(output / "teacher_seed_202609071.pt"),
                official_test_evaluated=False)
    dump(output / "teacher_summary.json",result)
    event(output,"teacher_finished",**result)
    del xt,yt,xv,yv,model,optimizer
    torch.cuda.empty_cache()


# 用途：在固定空间窗口内按训练样本的平均激活选择源坐标，保持并列时的索引顺序。
# 参数：sources、q；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：central[np.argsort(-means[central], kind='stable')[:q]]。
def choose_nodes(sources,q):
    _, channels, height, width=sources.shape
    ids=np.arange(channels*height*width).reshape(channels,height,width)
    central=ids[:,height//2-1:height//2+2,width//2-1:width//2+2].ravel()
    means=sources.mean(0).numpy().ravel()
    return central[np.argsort(-means[central],kind="stable")[:q]]


# 用途：用固定训练/验证上下文拟合局部方法，保存预测、原尾部参考与评分。
# 参数：output、x、contexts、protocol；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def evaluate(output,x,contexts,protocol):
    model=make_teacher("fashion").double().eval()
    model.load_state_dict(torch.load(output / "teacher_seed_202609071.pt",weights_only=True,map_location="cpu"))
    # 评分只包含三个及以上源坐标同时被干预的组合，避免混入二阶拟合点。
    bits=bit_design(8);bt=torch.tensor(bits,dtype=torch.float64);held=bits.sum(1)>2
    low=low_order(8)[0]
    # Reproduce float32 / 255 teacher input values, then promote for intervention checks.
    train_x=torch.tensor(x[contexts["train"]][:,None].astype(np.float32)/np.float32(255),dtype=torch.float64)
    val_x=torch.tensor(x[contexts["validation"]][:,None].astype(np.float32)/np.float32(255),dtype=torch.float64)
    rows=[];checks=[]
    for layer in (1,2):
        started=time.perf_counter()
        with torch.no_grad():
            train_sources=source_activation(model,train_x,layer)
            val_sources=source_activation(model,val_x,layer)
        nodes=choose_nodes(train_sources,8)
        mod=LocalModule(model,tuple(train_sources.shape[1:]),nodes,layer)
        tc=[mod.prepare(s[None]) for s in train_sources]
        vc=[mod.prepare(s[None]) for s in val_sources]
        fit_start=time.perf_counter()
        fit=fit_module(mod,tc,bits[low],202609071+layer*101)
        fit_seconds=time.perf_counter()-fit_start
        # 保存具名数值数组，便于后来按同一索引回读和独立复算。
        np.savez_compressed(output / f"background_fit_L{layer}.npz",**fit.__dict__,nodes=nodes,layer=layer)
        event(output,"selector_frozen",layer=layer,nodes=nodes.tolist(),fit_seconds=fit_seconds,
              training_contexts=len(tc),fit_sha256=sha(output / f"background_fit_L{layer}.npz"))
        specs=[("greedy",k) for k in (0,1,2,4,8,16,32,64)]
        specs += [(method,16 if method in ("random","weight","pca_fused") else 64 if method=="exact" else 0)
                  for method in ("random","weight","pca_fused","boundary_only","mean_head","pool_ridge","exact")]
        predictions={}
        runtimes={}
        for method,k in specs:
            rt=HeadRuntime(mod,fit,method,k);runtimes[(method,k)]=rt
            prepared=[rt.prepare(s[None]) for s in val_sources]
            for alpha in (1.,.5):
                predictions[(method,k,alpha)]=np.stack([rt.evaluate(c,bt,alpha).numpy() for c in prepared])
        np.savez_compressed(output / f"validation_predictions_L{layer}.npz",
                            **{f"{method}_k{k}_a{alpha}":v for (method,k,alpha),v in predictions.items()},
                            ids=contexts["validation"])
        event(output,"validation_predictions_saved",layer=layer,
              sha256=sha(output / f"validation_predictions_L{layer}.npz"),scope="pilot validation, not confirmatory test")
        full_truth={};max_exact=0.;max_pca=0.;max_fusion=0.
        for alpha in (1.,.5):
            reference=np.stack([mod.full_evaluate(c,bt,alpha).numpy() for c in vc]);full_truth[str(alpha)]=reference
            for method,k in specs:
                prediction=predictions[(method,k,alpha)]
                metric=score(prediction,reference,held)
                err_input=np.square((prediction-reference)[:,held]).sum((1,2))
                effect_input=np.square(reference[:,held]).sum((1,2))
                positive=effect_input>1e-24
                relative=np.sqrt(err_input[positive]/effect_input[positive])
                rows.append(dict(layer=layer,alpha=alpha,method=method,k=k,**metric,
                                 per_input_nrmse_p50=float(np.median(relative)),
                                 per_input_nrmse_p95=float(np.quantile(relative,.95)),
                                 near_zero_effect_inputs=int((~positive).sum()),
                                 target_rms=float(np.sqrt(np.square(reference[:,held]).mean())),
                                 context_scalars=8+4*len(mod.pool_ids)+(64 if method=="exact" else k),
                                 background_scalars=4*len(mod.pool_ids)+(64 if method=="exact" else k)))
                if method=="exact":max_exact=max(max_exact,metric["maxerr"])
                if method=="pca_fused":
                    orig=np.stack([predict(mod,fit,c,bits,"pca",k,alpha) for c in vc])
                    max_pca=max(max_pca,float(np.abs(prediction-orig).max()))
                    unfused=HeadRuntime(mod,fit,"pca_unfused",k)
                    direct=np.stack([unfused.evaluate(unfused.prepare(s[None]),bt,alpha).numpy() for s in val_sources])
                    max_fusion=max(max_fusion,float(np.abs(prediction-direct).max()))
        np.savez_compressed(output / f"validation_references_L{layer}.npz",**full_truth,ids=contexts["validation"])
        eligible=[r["k"] for r in rows if r["layer"]==layer and r["alpha"]==1. and r["method"]=="greedy" and r["nrmse"]<=.10]
        check=dict(layer=layer,inputs=len(vc),all_coalitions=256,scored_high_order_coalitions=219,
                   full_exact_max_error=max_exact,pca_fusion_max_error=max_fusion,pca_vs_numpy_max_error=max_pca,
                   selected_validation_k=min(eligible) if eligible else 64,
                   validation_active_source_mean=float(torch.stack([c["h"]!=0 for c in vc]).sum(1).double().mean()),
                   boundary_scalars=4*len(mod.pool_ids),fit_seconds=fit_seconds,seconds=time.perf_counter()-started)
        checks.append(check);event(output,"layer_finished",**check)
        write_csv(output / "validation_results.csv",rows);dump(output / "verification.json",checks)
        assert max(max_exact,max_pca,max_fusion)<1e-9
    dump(output / "completion.json",dict(status="completed",utc=datetime.now(timezone.utc).isoformat(),
         teachers=1,teacher_seeds=[202609071],teacher_validation_inputs=5000,selector_training_inputs=512,
         explainer_validation_inputs=128,validation_full_tail_outputs=2*2*128*256,
         official_test_inputs_evaluated=0,calibration_inputs_evaluated=0,
         scope="first external task train/validation feasibility; not six-seed external test result"))


# 用途：执行 Fashion-MNIST 首个教师的训练、验证干预和初步方法比较。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("all","evaluate"),default="all");args=ap.parse_args()
    output=BASE / "runs/fashion_pilot_20260910"
    data=BASE / "data/fashion_mnist/prepared"
    torch.set_num_threads(1);threadpool_limits(1)
    x=np.load(data / "official_train_images.npy",mmap_mode="r",allow_pickle=False)
    y=np.load(data / "official_train_labels.npy",mmap_mode="r",allow_pickle=False)
    indices=np.load(data / "split_indices.npz",allow_pickle=False)
    contexts=np.load(data / "context_indices.npz",allow_pickle=False)
    upstream_protocol=json.loads((UPSTREAM / "protocol/external_protocol.json").read_text())
    if args.stage=="all":
        output.mkdir(parents=True,exist_ok=False)
        protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            stage="train/validation feasibility, first locked teacher seed; retain all outcomes",
            original_protocol_sha256=sha(UPSTREAM / "protocol/external_protocol.json"),
            teacher=upstream_protocol["teacher"],explainer=upstream_protocol["explainer"],
            original_primary_contrast_retained=upstream_protocol["metrics"]["primary"],
            implementation_details=dict(training_device="CUDA",training_dtype="float32",analysis_device="CPU",
                analysis_dtype="float64",input_normalization="float32 / 255 then promotion to float64 for analysis",
                augmentation="none",lr_schedule="constant",batch_order="CPU torch.Generator seeded once, advanced every epoch",
                early_stopping="strictly lowest validation cross entropy, no min_delta; patience 4",
                coordinates="descending mean over the fixed 512 training contexts in central 3x3 across channels; stable flat-index tie break",
                context_selection="fixed context_indices.npz",partial_background_fit="unchanged upstream fit_module",
                adaptive_validation_budgets=[0,1,2,4,8,16,32,64]),
            amendment="Add algebraically equivalent fused preactivation PCA; retain old PCA predictions. Correct its acquisition accounting to k linear codes; include full decoder and response head cost.",
            access_policy="Only official-training-pool train/validation images enter a model. Calibration and official test are reserved; no official test labels or outputs are loaded by this entrypoint.",
            code_sha256={name:sha(BASE / "scripts" / name) for name in ("run_fashion_pilot.py","research_runtime.py","prepare_external_split.py")},
            split_sha256=sha(data / "split_indices.npz"),context_sha256=sha(data / "context_indices.npz"),
            environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        dump(output / "pilot_protocol.json",protocol)
        (output / "pilot_protocol.sha256").write_text(sha(output / "pilot_protocol.json")+"\n",encoding="ascii")
        event(output,"protocol_locked",sha256=sha(output / "pilot_protocol.json"))
        if not torch.cuda.is_available():raise RuntimeError("Declared CUDA training backend unavailable")
        train(output,x,y,indices,protocol)
    else:
        protocol=json.loads((output / "pilot_protocol.json").read_text())
        if (output / "completion.json").exists():raise RuntimeError("Completed run must not be overwritten")
        event(output,"resume_evaluation",checkpoint_sha256=sha(output / "teacher_seed_202609071.pt"))
    evaluate(output,x,contexts,protocol)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":
    main()

# 中文阅读说明｜用途：调度原六教师外部数据实验：初始化、开发、计时、冻结与测试。
# 数据流：命令行 --stage 选择阶段；依赖前序模型和协议，不能在已完成目录随意重跑。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Stage-gated six-seed Fashion-MNIST experiment; never tune on official test."""
from __future__ import annotations
import argparse
import platform
import shutil
import time
from pathlib import Path
from external_experiment import (BASE, UPSTREAM, PILOT, DATA, SEEDS, GRID, METHODS,
    now, sha, dump, read_json, csv_write, csv_read, event, configure_cpu, normalized,
    train_teacher, develop_seed, benchmark_seed, load_model, load_fit, test_specs,
    prepare_predictions, score_predictions, make_teacher, LocalModule, source_activation,
    bit_design, np, torch, F)
from prepare_external import load_idx

DEFAULT_RUN = BASE / "runs/fashion_external_20260911"
SOURCES = ["scripts/external_experiment.py", "scripts/run_external.py", "scripts/external_analysis.py",
    "scripts/research_runtime.py", "upstream/CAT_MT_phase7_v0.7/vendor/local_modules.py",
    "upstream/CAT_MT_phase7_v0.7/code/context_module.py",
    "upstream/CAT_MT_phase7_v0.7/code/prepare_external.py",
    "upstream/CAT_MT_phase7_v0.7/code/acquire_fashion.py"]


# 用途：建立新的实验目录并保存数据、模型配置及源码的可核对协议。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def initialize(run):
    run.mkdir(parents=True, exist_ok=False)
    original = read_json(UPSTREAM / "protocol/external_protocol.json")
    assert original["teacher"]["seeds"] == SEEDS
    # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
    assert sha(UPSTREAM / "protocol/external_protocol.json") == "d88eb0fc980282b53fda7807f246f2442b4ac7a05c98faaf38a53f32e850b1a8"
    old = read_json(PILOT / "pilot_protocol.json")
    assert all(sha(BASE / "scripts" / name)==value for name,value in old["code_sha256"].items())
    assert sha(DATA / "split_indices.npz")==old["split_sha256"]
    assert sha(DATA / "context_indices.npz")==old["context_sha256"]
    prepared = [DATA / name for name in ("split_indices.npz","context_indices.npz","official_train_images.npy","official_train_labels.npy")]
    raw = list((BASE / "data/fashion_mnist/raw").glob("*.gz"))
    assert len(raw)==4
    protocol = dict(version="external-execution-20260911-v1",utc=now(),
        purpose="complete the original six-seed external feasibility experiment; no general superiority claim",
        prior_knowledge="historical development results and first-seed train/validation pilot were inspected; official test model outputs have not been accessed",
        original_protocol_sha256=sha(UPSTREAM / "protocol/external_protocol.json"),
        teacher=original["teacher"],explainer=original["explainer"],
        first_teacher="reuse unchanged pilot checkpoint and fitted parameters, no seed discarded",
        original_primary=original["metrics"]["primary"],
        primary_inference=dict(unit="six seed-level paired NRMSE differences conditional on the fixed 256 test images",
            statistic="mean(greedy k16 NRMSE minus weight k16 NRMSE), layer 1 alpha 1",
            sign_flip="enumerate all 64 independent sign assignments; two-sided absolute mean statistic with 1e-14 tie tolerance",
            confidence_interval="95% Student t interval for mean paired difference, df=5",
            assumptions="sign exchangeability/symmetry under the paired null; approximate normality for t interval; six seeds provide low power",
            alpha=.05,secondary_inference="descriptive; no additional confirmatory p-values"),
        secondary=dict(fixed_budgets=[16,32],adaptive_grid=GRID,
            adaptive_methods=METHODS,adaptive_rule="smallest k with validation NRMSE <= 0.10 at alpha 1, separately per method and layer; fallback to 64",
            reused_for_half_strength=True,no_inputwise_adaptation=True,
            controls=["random k16/k32","mean head","boundary only","pooled ridge","full exact"],
            pca="unchanged PCA of preactivations; fused linear encoder; full decoder and nonlinear response retained"),
        test=dict(intervention_ids="context_indices.npz:test_reserved, exactly official indices 0..255",
            inputs_per_teacher_layer=256,coalitions=256,scored_coalitions=219,strengths=[1.,.5],
            classification="all 10000 official test images, CPU float32, descriptive after all choices frozen",
            prediction_commit="save each frozen model/layer prediction array before independently executing its intervened reference outputs",
            no_refitting=True,calibration="reserved and not evaluated"),
        implementation_details=old["implementation_details"],
        timing=dict(device="CPU",dtype="float64",threads=1,inputs="first four fixed validation contexts",queries=[37,256],
            workloads=["from_image","from_source","cached"],warmup=3,rounds=15,inner_repetitions=3,
            method_order="random permutation every round; seed 20260911 + teacher seed",
            full_budget="use mathematically equivalent exact runtime at k=64 for all methods; verify against unfactored estimator",
            full_budget_summary="identical exact implementations share the exact timing observation, so k64 speedup is 1 by construction",
            concurrency="no teacher training or other experiment runs during timing",
            offline="record joint development/validation wall time and runtime construction separately; any amortization calculation names this nonminimal joint pipeline cost"),
        source_sha256={name:sha(BASE/name) for name in SOURCES},
        data_sha256={p.relative_to(BASE).as_posix():sha(p) for p in prepared+raw},
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    for name in SOURCES:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,destination)
    dump(run / "execution_protocol.json",protocol)
    (run / "execution_protocol.sha256").write_text(sha(run / "execution_protocol.json")+"\n",encoding="ascii")
    first = run / f"seed_{SEEDS[0]}"
    first.mkdir()
    shutil.copy2(PILOT / "teacher_seed_202609071.pt",first / "teacher.pt")
    shutil.copy2(PILOT / "training_history.json",first / "training_history.json")
    summary = read_json(PILOT / "teacher_summary.json")
    assert sha(first / "teacher.pt")==summary["checkpoint_sha256"]
    summary.update(reused_pilot=True,source_checkpoint=str(PILOT / "teacher_seed_202609071.pt"),test_used_for_selection=False)
    dump(first / "teacher_summary.json",summary)
    event(run,"execution_protocol_locked",sha256=sha(run / "execution_protocol.json"),first_teacher_sha256=summary["checkpoint_sha256"])


# 用途：核对协议、源码及依赖文件的原始哈希，确认执行条件未被更换。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：protocol。
def verify_protocol(run):
    protocol = read_json(run / "execution_protocol.json")
    assert sha(run / "execution_protocol.json")==(run / "execution_protocol.sha256").read_text().strip()
    for name,value in protocol["source_sha256"].items():
        assert sha(BASE/name)==value, f"Frozen source changed: {name}"
        assert sha(run / "source" / name)==value
    for name,value in protocol["data_sha256"].items():
        assert sha(BASE/name)==value, f"Frozen data changed: {name}"
    return protocol


# 用途：读取固定的训练数据及划分索引，供开发阶段使用。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：(np.load(DATA / 'official_train_images.npy', mmap_mode='r', allow_pickle=False), np.loa…。
def train_data():
    return (np.load(DATA / "official_train_images.npy",mmap_mode="r",allow_pickle=False),
            np.load(DATA / "official_train_labels.npy",mmap_mode="r",allow_pickle=False),
            np.load(DATA / "split_indices.npz",allow_pickle=False),
            np.load(DATA / "context_indices.npz",allow_pickle=False))


# 用途：仅利用该阶段规定的训练/验证输入完成模型或方法开发并保存选择。
# 参数：run、protocol、first_only；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def development(run, protocol, first_only=False):
    assert not (run / "test_opened.json").exists(), "No development after test is opened"
    x,y,indices,contexts = train_data()
    for seed in (SEEDS[:1] if first_only else SEEDS):
        train_teacher(run,seed,x,y,indices,protocol["teacher"])
        develop_seed(run,seed,x,contexts)
        if first_only:
            benchmark_seed(run,seed,x,contexts)
    event(run,"development_stage_complete",seeds=SEEDS[:1] if first_only else SEEDS,official_test_model_access=False)


# 用途：在统一设备、线程、查询量和输入下测量各方法，区分准备与查询开销。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def timing(run):
    assert not (run / "test_opened.json").exists()
    x,_,_,contexts = train_data()
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        assert (run / f"seed_{seed}/development_complete.json").exists()
        benchmark_seed(run,seed,x,contexts)
    event(run,"all_timing_completed",seeds=SEEDS,official_test_model_access=False)


# 用途：把开发阶段选定的模型、方法、预算与相关文件写入测试前冻结记录。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def freeze(run):
    assert not (run / "test_opened.json").exists()
    if (run / "frozen_before_test.json").exists():
        raise RuntimeError("Test freeze already exists; do not overwrite")
    records,choices = {},{}
    for seed in SEEDS:
        directory = run / f"seed_{seed}"
        assert (directory / "development_complete.json").exists()
        assert (directory / "timing/completion.json").exists()
        choices[str(seed)] = read_json(directory / "validation_choices.json")
        for name in ("teacher.pt","teacher_summary.json","fit_L1.npz","fit_L2.npz","validation_choices.json",
                     "validation_results.csv","development_complete.json","timing/seed_summary.csv","timing/cost_counts.csv","timing/completion.json"):
            records[(directory/name).relative_to(run).as_posix()] = sha(directory/name)
    lock = dict(utc=now(),seeds=SEEDS,execution_protocol_sha256=sha(run / "execution_protocol.json"),
        artifacts=records,validation_choices=choices,official_test_model_access=False,
        method_selection_finished=True,primary_unchanged=True)
    dump(run / "frozen_before_test.json",lock)
    (run / "frozen_before_test.sha256").write_text(sha(run / "frozen_before_test.json")+"\n",encoding="ascii")
    event(run,"all_artifacts_frozen_before_test",sha256=sha(run / "frozen_before_test.json"),choices=choices)


# 用途：核对最终冻结清单，确保模型、拟合参数和实验选择没有在测试后改变。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：lock。
def verify_freeze(run):
    assert sha(run / "frozen_before_test.json")==(run / "frozen_before_test.sha256").read_text().strip()
    lock = read_json(run / "frozen_before_test.json")
    assert lock["seeds"]==SEEDS
    assert lock["execution_protocol_sha256"]==sha(run / "execution_protocol.json")
    for name,value in lock["artifacts"].items():
        assert sha(run/name)==value, f"Post-freeze artifact changed: {name}"
    return lock


# 用途：在冻结教师上批量计算官方测试分类表现，仅记录评价结果。
# 参数：directory、images、labels；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：(logits, dict(accuracy=float((logits.argmax(1) == labels).mean()), cross_entropy=float(…。
def classify(directory, images, labels):
    model = make_teacher("fashion").eval()
    model.load_state_dict(torch.load(directory / "teacher.pt",weights_only=True,map_location="cpu"))
    outputs = []
    with torch.no_grad():
        for start in range(0,len(images),512):
            outputs.append(model(torch.from_numpy(normalized(images[start:start+512]))).numpy())
    logits = np.concatenate(outputs)
    # 分类交叉熵是教师学习/评价的任务损失，区别于后续干预响应的 NRMSE。
    loss = F.cross_entropy(torch.from_numpy(logits),torch.as_tensor(labels.astype(np.int64))).item()
    return logits,dict(accuracy=float((logits.argmax(1)==labels).mean()),cross_entropy=float(loss),n=len(labels),backend="CPU float32")


# 用途：核验冻结状态后处理预留测试输入，保存预测、原网络参考及评分。
# 参数：run；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def test(run):
    lock = verify_freeze(run)
    out = run / "test"
    if (out / "completion.json").exists():
        raise RuntimeError("Official test already completed; use verification/analysis, not another test run")
    out.mkdir(exist_ok=True)
    if not (run / "test_opened.json").exists():
        dump(run / "test_opened.json",dict(utc=now(),frozen_before_test_sha256=sha(run / "frozen_before_test.json"),reason="all six teachers, fits, choices and costs frozen"))
        event(run,"official_test_opened",freeze_sha256=sha(run / "frozen_before_test.json"))
    raw = BASE / "data/fashion_mnist/raw"
    images = load_idx(raw,"t10k-images-idx3-ubyte.gz")
    labels = load_idx(raw,"t10k-labels-idx1-ubyte.gz")
    assert images.shape==(10000,28,28) and labels.shape==(10000,)
    contexts = np.load(DATA / "context_indices.npz",allow_pickle=False)
    ids = contexts["test_reserved"]
    assert np.array_equal(ids,np.arange(256))
    x = torch.tensor(normalized(images[ids]),dtype=torch.float64)
    all_rows, class_rows, checks = [],[],[]
    for seed in SEEDS:
        directory = run / f"seed_{seed}"
        choices = lock["validation_choices"][str(seed)]
        if (out / f"classification_{seed}.json").exists():
            classification = read_json(out / f"classification_{seed}.json")
        else:
            logits,classification = classify(directory,images,labels)
            # 保存具名数值数组，便于后来按同一索引回读和独立复算。
            np.savez_compressed(out / f"classification_{seed}.npz",logits=logits,labels=labels)
            classification.update(seed=seed,used_for_selection=False)
            dump(out / f"classification_{seed}.json",classification)
        class_rows.append(classification)
        model = load_model(directory)
        for layer in (1,2):
            case = out / f"seed_{seed}_L{layer}"
            case.mkdir(exist_ok=True)
            if (case / "completion.json").exists():
                all_rows.extend(csv_read(case / "results.csv"))
                checks.append(read_json(case / "completion.json"))
                continue
            start = time.perf_counter()
            fit,nodes = load_fit(directory / f"fit_L{layer}.npz")
            with torch.no_grad():
                sources = source_activation(model,x,layer)
            mod = LocalModule(model,tuple(sources.shape[1:]),nodes,layer)
            specs = test_specs(choices[str(layer)])
            pred_path = case / "predictions.npz"
            if pred_path.exists():
                commitment = read_json(case / "prediction_commitment.json")
                assert sha(pred_path)==commitment["sha256"]
                with np.load(pred_path,allow_pickle=False) as saved:
                    assert np.array_equal(saved["ids"],ids)
                    predictions={k:saved[k].copy() for k in saved.files if k!="ids"}
                prepared = [mod.prepare(s[None]) for s in sources]
                path_check = commitment["path_verification"]
            else:
                predictions,prepared,path_check = prepare_predictions(mod,fit,sources,specs)
                np.savez_compressed(pred_path,**predictions,ids=ids)
                dump(case / "prediction_commitment.json",dict(utc=now(),sha256=sha(pred_path),path_verification=path_check))
                event(run,"test_predictions_committed",seed=seed,layer=layer,sha256=sha(pred_path))
            reference_path = case / "references.npz"
            if reference_path.exists():
                assert sha(reference_path)==read_json(case / "reference_commitment.json")["sha256"]
                with np.load(reference_path,allow_pickle=False) as saved:
                    assert np.array_equal(saved["ids"],ids)
                    references={str(a):saved[str(a)].copy() for a in (1.,.5)}
            else:
                references={str(a):np.stack([mod.full_evaluate(c,bit_design(8),a).numpy() for c in prepared]) for a in (1.,.5)}
                np.savez_compressed(reference_path,**references,ids=ids)
                dump(case / "reference_commitment.json",dict(utc=now(),sha256=sha(reference_path)))
            rows,per_input = score_predictions(predictions,references,specs,seed,layer,ids)
            exact_error=max(r["maxerr"] for r in rows if r["method"]=="exact")
            assert exact_error<1e-9
            csv_write(case / "results.csv",rows)
            csv_write(case / "per_input.csv",per_input)
            check=dict(utc=now(),seed=seed,layer=layer,inputs=256,coalitions=256,strengths=2,
                prediction_sha256=sha(pred_path),reference_sha256=sha(reference_path),full_exact_max_error=exact_error,
                active_source_mean=float(torch.stack([c["h"]!=0 for c in prepared]).sum(1).double().mean()),
                boundary_scalars=4*len(mod.pool_ids),**path_check,seconds=time.perf_counter()-start)
            dump(case / "completion.json",check)
            checks.append(check)
            all_rows.extend(rows)
            event(run,"test_layer_finished",**check)
            del predictions,references,prepared
    csv_write(out / "all_results.csv",all_rows)
    csv_write(out / "teacher_classification.csv",class_rows)
    verify_freeze(run)
    dump(out / "completion.json",dict(utc=now(),status="completed",seeds=SEEDS,model_layer_cases=12,
        classification_inputs_per_teacher=10000,intervention_inputs_per_teacher=256,
        full_tail_intervened_outputs=6*2*2*256*256,calibration_model_access=False,
        max_exact_error=max(c["full_exact_max_error"] for c in checks),post_test_refitting=False))
    event(run,"official_test_completed",seeds=SEEDS,full_tail_intervened_outputs=6*2*2*256*256)


# 用途：调度原六教师外部数据实验：初始化、开发、计时、冻结与测试。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：None。
def main():
    # 解析运行阶段/路径等命令行参数；具体允许的阶段由 choices 或分派逻辑限定。
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("init","pilot32","development","timing","freeze","test"),required=True)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    assert run.is_relative_to(BASE.resolve())
    configure_cpu()
    if args.stage=="init":
        initialize(run)
        return
    protocol=verify_protocol(run)
    if args.stage=="pilot32":development(run,protocol,True)
    elif args.stage=="development":development(run,protocol)
    elif args.stage=="timing":timing(run)
    elif args.stage=="freeze":freeze(run)
    elif args.stage=="test":test(run)


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__=="__main__":main()

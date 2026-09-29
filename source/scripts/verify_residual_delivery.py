# 中文阅读说明｜用途：核验残差阶段的时间顺序、文件哈希、计时配对与结果完整性。
# 数据流：输入冻结协议和完成记录；输出交付审计，属于验证工具。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Preserve prior evidence and verify the completed residual study delivery.

This does not retrain, rerun interventions, replace saved predictions, or package
archives. The numerical array rescore is the completed frozen analysis; this
audit rechecks its input hashes and adds classification and delivery checks.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from urllib.parse import unquote
import ast
import csv
import hashlib
import json
import math
import re
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/fashion_residual_20260913"
UPSTREAM = BASE / "upstream/CAT_MT_phase7_v0.7"
SEEDS = list(range(202609071, 202609077))


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.file_digest(stream, 'sha256').hexdigest()。
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path, value):
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# 用途：把相对路径解析到指定根目录，并拒绝超出该目录的路径。
# 参数：root、name；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：path。
def checked(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), name
    return path


# 用途：对数组样本或清单文件计算/核对哈希；本文件的输入形式决定具体用法。
# 参数：root、entries；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：len(entries)。
def hashes(root, entries):
    for name, expected in entries.items():
        # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
        assert sha(checked(root, name)) == expected, name
    return len(entries)


# 用途：读取带标题的 CSV，并按列名返回记录。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：list(csv.DictReader(stream))。
def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


# 用途：把记录中的 UTC 字符串转为可比较的时间，核验阶段先后顺序。
# 参数：record；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：datetime.fromisoformat(record['utc'])。
def utc(record):
    return datetime.fromisoformat(record["utc"])


# 用途：核对计时结果是否覆盖预期输入、方法和重复轮次，保持公平配对。
# 参数：folder、input_key、methods、expected；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：字典（字段见 return）。
def timing_check(folder, input_key, methods, expected):
    rows = csv_rows(folder / "raw.csv")
    assert len(rows) == expected
    assert len({r["method"] for r in rows}) == methods
    assert {int(r["seed"]) for r in rows} == set(SEEDS)
    groups = defaultdict(list)
    input_ids = set()
    for r in rows:
        assert float(r["seconds"]) > 0 and math.isfinite(float(r["seconds"]))
        assert r["queries"] in {"37", "256"} and r["workload"] in {"from_image", "cached"}
        input_ids.add(int(r[input_key]))
        key = tuple(r[k] for k in ("seed", input_key, "queries", "workload", "method"))
        groups[key].append(int(r["round"]))
    assert len(input_ids) == 4
    assert all(sorted(v) == list(range(9)) for v in groups.values())
    return dict(rows=len(rows), methods=methods, validation_input_ids=sorted(input_ids))


# 用途：核验残差阶段的时间顺序、文件哈希、计时配对与结果完整性。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    archive_hash = sha(BASE / "incoming/CAT_MT_background_selection_v0.7.zip")
    assert archive_hash == "8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    upstream_count = 0
    for line in (UPSTREAM / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        assert sha(checked(UPSTREAM, name)) == expected, name
        upstream_count += 1
    assert upstream_count == 271

    prior_checks = []
    for run_name, manifest, protocol_name, freeze_name in [
        ("fashion_external_20260911", "followup_manifest_20260911.json",
         "execution_protocol.json", "frozen_before_test.json"),
        ("fashion_structure_20260911", "structure_manifest_20260911.json",
         "protocol.json", "freeze.json"),
    ]:
        root = BASE / "runs" / run_name
        prefix = f"runs/{run_name}/"
        entries = [e for e in read(BASE / "audit" / manifest)["entries"] if e["path"].startswith(prefix)]
        for e in entries:
            assert sha(checked(BASE, e["path"])) == e["sha256"], e["path"]
        old_protocol, old_freeze = read(root / protocol_name), read(root / freeze_name)
        hashes(BASE, old_protocol.get("sources", old_protocol.get("source_sha256", {})))
        hashes(root, old_freeze["artifacts"])
        prior_checks.append(dict(run=prefix, files_unchanged=len(entries),
                                 frozen_artifacts=len(old_freeze["artifacts"])))

    expected_records = {
        "protocol": "ff9bafa1cba65b4553c3545dbd780fb21b52574093c4ca234b3e5b2952d59502",
        "implementation": "fbb9fe2545b500dc78a905e440dfc6d5c80bff0a8a66a04ffc3a47f8e24118db",
        "freeze": "243e5ba1424f9be89e133b99bb1a70cb684169dfbe137a13985100141e73e03a",
    }
    for stem, expected in expected_records.items():
        assert sha(RUN / f"{stem}.json") == expected
        assert (RUN / f"{stem}.sha256").read_text(encoding="ascii").strip() == expected
    protocol, impl, freeze = (read(RUN / f"{s}.json") for s in expected_records)
    opened, done = read(RUN / "test_opened.json"), read(RUN / "test/completion.json")
    trained, developed = read(RUN / "training_complete.json"), read(RUN / "development_complete.json")
    assert impl["protocol_sha256"] == freeze["protocol_sha256"] == expected_records["protocol"]
    assert freeze["implementation_sha256"] == expected_records["implementation"]
    assert opened["freeze_sha256"] == expected_records["freeze"]
    assert protocol["prior_freeze_sha256"] == sha(BASE / "runs/fashion_structure_20260911/freeze.json")
    assert protocol["intervention_ids_sha256"] == sha(RUN / "intervention_ids.npy")
    for record in (protocol, impl):
        hashes(BASE, record["sources"])
        hashes(RUN / "source", record["sources"])
    hashes(BASE, protocol["data_hashes"])
    assert hashes(RUN, freeze["artifacts"]) == 78
    assert utc(protocol) < utc(trained) < utc(impl) < utc(developed) < utc(freeze) < utc(opened) < utc(done)
    assert not freeze["official_test_model_access"] and not freeze["calibration_model_access"]
    assert protocol["seeds"] == trained["seeds"] == developed["seeds"] == done["seeds"] == SEEDS
    assert done["status"] == "completed" and done["cases"] == 6
    assert done["full_tail_output_vectors"] == 786432 and done["max_exact_error"] < 1e-9
    assert not done["calibration_model_access"]
    assert trained["task_competent"] and trained["mean_validation_accuracy"] >= .90
    assert trained["minimum_validation_accuracy"] >= .88
    assert np.array_equal(np.load(RUN / "intervention_ids.npy", allow_pickle=False), np.arange(512, 768))

    teachers, classification, case_counts = [], [], Counter()
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        folder = RUN / f"seed_{seed}"
        teacher, history, fit = (read(folder / s) for s in
                                ("teacher_summary.json", "training_history.json", "fit_complete.json"))
        best = min(history, key=lambda r: r["validation_cross_entropy"])
        assert best["epoch"] == teacher["best_epoch"]
        assert best["validation_cross_entropy"] == teacher["best_validation_cross_entropy"]
        assert best["validation_accuracy"] == teacher["best_validation_accuracy"]
        assert len(history) == teacher["epochs"] == 20 and not teacher["test_used_for_selection"]
        assert sha(folder / "teacher.pt") == teacher["checkpoint_sha256"]
        assert sha(folder / "fit.npz") == fit["fit_sha256"]
        assert fit["training_active_units"] == fit["changing_training_gates"] == fit["affected_states"]
        assert fit["always_open_training_units"] == 0
        teachers.append(dict(seed=seed, best_epoch=best["epoch"], states=fit["affected_states"],
                             checkpoint_verified=True, all_gates_observed_both_states=True))
        for phase, case in (("validation", folder / "validation"),
                            ("official_test", RUN / f"test/seed_{seed}")):
            prediction, reference, completed = (read(case / s) for s in
                ("prediction_commitment.json", "reference_commitment.json", "completion.json"))
            assert utc(prediction) < utc(reference) <= utc(completed)
            if phase == "validation":
                assert utc(fit) < utc(prediction) < utc(completed) < utc(freeze)
            else:
                assert utc(opened) < utc(prediction) < utc(completed) <= utc(done)
            assert sha(case / "predictions.npz") == prediction["sha256"]
            assert sha(case / "references.npz") == reference["sha256"]
            with np.load(case / "predictions.npz", allow_pickle=False) as pp, \
                 np.load(case / "references.npz", allow_pickle=False) as rr:
                assert np.array_equal(pp["ids"], rr["ids"])
                assert len(pp.files) - 1 == completed["arrays"] == 60
                if phase == "official_test":
                    assert np.array_equal(pp["ids"], np.arange(512, 768))
            assert completed["phase"] == phase
            assert completed["inputs"] == (128 if phase == "validation" else 256)
            assert completed["exact_max_error"] < 1e-9
            assert completed["reference_device"] == "CUDA float64"
            assert completed["runtime_device"] == "CPU float64"
            case_counts[phase] += completed["arrays"]
        c = read(RUN / f"test/classification_{seed}.json")
        with np.load(RUN / f"test/classification_{seed}.npz", allow_pickle=False) as a:
            assert a["logits"].shape == (10000, 10) and a["labels"].shape == (10000,)
            correct = int((a["logits"].argmax(1) == a["labels"]).sum())
            assert correct / 10000 == c["accuracy"]
        assert not c["test_used_for_selection"]
        classification.append(dict(seed=seed, correct=correct, n=10000))

    analysis = read(RUN / "analysis/verification.json")
    assert analysis["status"] == "passed" and analysis["arrays_rescored"] == dict(case_counts)
    assert dict(case_counts) == {"validation": 360, "official_test": 360}
    assert analysis["max_nrmse_difference"] == 0 and analysis["source_and_freeze_hashes_unchanged"]
    assert not analysis["calibration_model_access"] and analysis["new_confirmatory_p_values"] == 0
    smoke = read(RUN / "smoke.json")
    assert smoke["status"] == "passed" and max(smoke["errors"].values()) < 1e-9
    assert utc(smoke) < utc(impl) and not smoke["official_test_model_access"]
    timing = read(RUN / "timing/completion.json")
    assert not timing["training_concurrent"] and utc(timing) < utc(freeze)
    pre_timing = timing_check(RUN / "timing", "input_index", 32, 27648)
    assert pre_timing["rows"] == timing["rows"]

    affine = RUN / "affine_runtime_audit"
    plan, affine_done = read(affine / "plan.json"), read(affine / "completion.json")
    assert utc(done) < utc(plan) < utc(affine_done)
    assert plan["freeze_sha256"] == expected_records["freeze"]
    assert plan["source_sha256"] == sha(affine / "source_snapshot.py") == sha(BASE / "scripts/audit_residual_affine_runtime.py")
    assert affine_done["plan_sha256"] == sha(affine / "plan.json")
    assert hashes(affine, affine_done["artifacts"]) == 8
    assert affine_done["status"] == "completed" and affine_done["arrays_compared"] == 12
    assert affine_done["max_prediction_difference"] < 1e-12
    assert affine_done["existing_test_outputs_unchanged"] and not affine_done["calibration_model_access"]
    assert affine_done["new_confirmatory_p_values"] == 0
    for r in csv_rows(affine / "equivalence.csv"):
        assert r["original_prediction_sha256"] == sha(RUN / f"test/seed_{r['seed']}/predictions.npz")
    post_timing = timing_check(affine, "input_id", 33, 28512)
    assert post_timing["rows"] == affine_done["raw_timing_rows"]
    assert pre_timing["validation_input_ids"] == post_timing["validation_input_ids"]
    assert not plan["timing"]["training_concurrent"]

    docs = [BASE / s for s in ("README.md", "RUNBOOK.md", "data/fashion_mnist/DATA_CARD.md",
                              "reports/残差块内部干预与背景选择_2026-09-13.md",
                              "reports/从原始设想到当前实验_2026-09-11.md")]
    link_count = 0
    for doc in docs:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (doc.parent / unquote(target.strip("<>"))).resolve().exists(), (doc, target)
            link_count += 1
    sources = list((BASE / "scripts").glob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    state = read(BASE / "RESEARCH_STATUS.json")
    assert state["current_run"] == state["latest_followup_run"] == RUN.relative_to(BASE).as_posix()
    assert state["total_fashion_teachers"] == state["official_test_classification_teacher_runs"] == 18
    assert state["completed_fashion_intervention_cases"] == 48 and state["external_official_test_batches"] == 3
    assert state["residual_intervention_ids"] == [512, 767] and not state["calibration_model_access"]
    assert state["structure_followup_summary"]["model_layer_cases"] == 30
    assert not list(BASE.glob("*.zip")) and not list(BASE.parents[1].glob("*.zip"))
    figures = [BASE / f"reports/residual_context_results.{extension}" for extension in ("png", "pdf")]
    assert all(p.is_file() and p.stat().st_size > 10000 for p in figures)

    checks_path = BASE / "audit/final_residual_delivery_20260913.json"
    checks = dict(utc=datetime.now(timezone.utc).isoformat(), status="passed",
        original_archive_sha256=archive_hash, upstream_manifest_files=upstream_count,
        prior_runs=prior_checks, record_sha256=expected_records, frozen_artifacts=78,
        source_and_data_hashes_match=True, protocol_freeze_prediction_reference_order_valid=True,
        teachers=teachers, classification_recomputed_from_saved_logits=classification,
        model_layer_cases=6, full_tail_vectors=done["full_tail_output_vectors"],
        existing_saved_array_rescore_verified=analysis, smoke_cross_checks=smoke["errors"],
        pre_test_timing=pre_timing, post_test_affine_timing=post_timing,
        affine_audit_is_post_test=True, affine_prediction_max_difference=affine_done["max_prediction_difference"],
        local_links_checked=link_count, broken_links=[], python_sources_parse=len(sources),
        calibration_model_access=False, new_confirmatory_p_values=0,
        deleted_delivery_archives_not_recreated=True,
        figure_review="PNG visually inspected after final rendering; four panels and labels readable. PDF exported from the same Matplotlib figure.",
        notes="Historical presentation snapshots are preserved but mutable reports may evolve. Prior numerical runs and all frozen evidence were hash-verified.")
    dump(checks_path, checks)
    paths = {p for p in RUN.rglob("*") if p.is_file()}
    paths.update(sources + docs + figures + [checks_path, BASE / "RESEARCH_STATUS.json"])
    entries = [dict(path=p.relative_to(BASE).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(paths)]
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(),
        scope="Completed residual study, separate post-test affine implementation audit, and current delivery documentation; no ZIP generated",
        files=len(entries), bytes=sum(e["bytes"] for e in entries), entries=entries)
    dump(BASE / "audit/residual_manifest_20260913.json", snapshot)
    print(json.dumps(dict(status="passed", upstream_files=upstream_count, prior_runs=prior_checks,
        frozen_artifacts=78, arrays=case_counts, timing_rows=[pre_timing["rows"], post_timing["rows"]],
        links=link_count, snapshot_files=len(entries), bytes=snapshot["bytes"])))


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

# 中文阅读说明｜用途：核验结构对照的冻结顺序、模型和交付文件。
# 数据流：依赖已有原主实验和结构实验记录；输出审计摘要。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Read-only preservation checks and a separate snapshot for structure controls."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
import re
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/fashion_structure_20260911"
PRIOR = BASE / "runs/fashion_external_20260911"
UPSTREAM = BASE / "upstream/CAT_MT_phase7_v0.7"
SEEDS = list(range(202609071, 202609077))
VISUAL = Path("visualizations/local-diamond.html")


# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.file_digest(stream, 'sha256').hexdigest()。
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(path.read_text(encoding='utf-8'))。
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


# 用途：核验结构对照的冻结顺序、模型和交付文件。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    archive_hash = sha(BASE / "incoming/CAT_MT_background_selection_v0.7.zip")
    assert archive_hash == "8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    upstream_count = 0
    for line in (UPSTREAM / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
        assert sha(checked(UPSTREAM, name)) == expected, name
        upstream_count += 1
    assert upstream_count == 271
    old_snapshot = read(BASE / "audit/followup_manifest_20260911.json")
    old_count = 0
    for entry in old_snapshot["entries"]:
        if entry["path"].startswith("runs/fashion_external_20260911/"):
            assert sha(checked(BASE, entry["path"])) == entry["sha256"], entry["path"]
            old_count += 1
    old_protocol = read(PRIOR / "execution_protocol.json")
    old_freeze = read(PRIOR / "frozen_before_test.json")
    for name, expected in old_protocol["source_sha256"].items():
        assert sha(checked(BASE, name)) == expected, name
    for name, expected in old_freeze["artifacts"].items():
        assert sha(checked(PRIOR, name)) == expected, name

    protocol = read(RUN / "protocol.json")
    freeze = read(RUN / "freeze.json")
    opened = read(RUN / "test_opened.json")
    done = read(RUN / "test/completion.json")
    for stem in ("protocol", "freeze"):
        assert sha(RUN / f"{stem}.json") == (RUN / f"{stem}.sha256").read_text(encoding="ascii").strip()
    assert freeze["protocol_sha256"] == sha(RUN / "protocol.json")
    assert opened["freeze_sha256"] == sha(RUN / "freeze.json")
    assert protocol["prior_freeze_sha256"] == sha(PRIOR / "frozen_before_test.json")
    assert protocol["intervention_ids_sha256"] == sha(RUN / "intervention_ids.npy")
    for name, expected in protocol["sources"].items():
        assert sha(checked(BASE, name)) == expected, name
        assert sha(checked(RUN / "source", name)) == expected, name
    for name, expected in protocol["data_hashes"].items():
        assert sha(checked(BASE, name)) == expected, name
    for name, expected in freeze["artifacts"].items():
        assert sha(checked(RUN, name)) == expected, name
    assert protocol["utc"] < freeze["utc"] < opened["utc"] < done["utc"]
    assert done["status"] == "completed" and done["cases"] == 30
    assert done["full_tail_output_vectors"] == 3932160
    assert done["max_exact_error"] < 1e-9 and not done["calibration_model_access"]
    teacher_checks = []
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        folder = RUN / f"gap_seed_{seed}"
        teacher = read(folder / "teacher_summary.json")
        history = read(folder / "training_history.json")
        best = min(history, key=lambda r: r["validation_cross_entropy"])
        assert best["epoch"] == teacher["best_epoch"]
        assert best["validation_cross_entropy"] == teacher["best_validation_cross_entropy"]
        assert best["validation_accuracy"] == teacher["best_validation_accuracy"]
        assert len(history) == teacher["epochs"] <= 15
        assert not teacher["test_used_for_selection"]
        assert teacher["checkpoint_sha256"] == sha(folder / "teacher.pt")
        assert read(folder / "development_complete.json")["utc"] < freeze["utc"]
        teacher_checks.append(dict(seed=seed, best_epoch=best["epoch"], checkpoint_verified=True))
    cases = sorted((RUN / "test").glob("*_L*"))
    assert len(cases) == 30
    for case in cases:
        prediction = read(case / "prediction_commitment.json")
        reference = read(case / "reference_commitment.json")
        completed = read(case / "completion.json")
        assert opened["utc"] < prediction["utc"] < reference["utc"] <= completed["utc"]
        assert sha(case / "predictions.npz") == prediction["sha256"]
        assert sha(case / "references.npz") == reference["sha256"]
    analysis = read(RUN / "analysis/verification.json")
    assert analysis["status"] == "passed" and analysis["arrays_rescored"] == 660
    assert analysis["max_nrmse_difference"] == 0
    assert not analysis["calibration_model_access"] and analysis["new_confirmatory_p_values"] == 0
    interpretation = read(BASE / "audit/structure_interpretation_20260911.json")
    assert interpretation["status"] == "passed"
    assert interpretation["original_region_autograd"]["validation_cases"] == 24
    assert interpretation["original_region_autograd"]["max_difference"] < 1e-10
    assert not interpretation["calibration_model_access"] and not interpretation["confirmed_test_outputs_replaced"]
    timing = read(RUN / "timing/completion.json")
    with (RUN / "timing/raw.csv").open(encoding="utf-8", newline="") as stream:
        timing_rows = sum(1 for _ in csv.DictReader(stream))
    assert timing_rows == timing["rows"] == 37440
    assert not timing["training_concurrent"] and timing["utc"] < freeze["utc"]
    docs = [BASE / "README.md", BASE / "RUNBOOK.md", BASE / "data/fashion_mnist/DATA_CARD.md",
            BASE / "reports/激活基线与线性读出对照_2026-09-11.md",
            BASE / "reports/从原始设想到当前实验_2026-09-11.md"]
    links = 0
    for doc in docs:
        content = doc.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (doc.parent / unquote(target.strip("<>"))).resolve().exists(), (doc, target)
            links += 1
    narrative = docs[-1].read_text(encoding="utf-8")
    assert "<!-- END_STRUCTURE_RESULTS -->" in narrative
    sources = list((BASE / "scripts").glob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    state = read(BASE / "RESEARCH_STATUS.json")
    assert state["latest_followup_run"] == "runs/fashion_structure_20260911"
    assert state["additional_gap_teachers"] == 6 and state["total_fashion_teachers"] == 12
    assert state["followup_intervention_ids"] == [256, 511]
    assert not state["calibration_model_access"]
    fragment = VISUAL.read_text(encoding="utf-8")
    assert "local-diamond-explainer" in fragment and "<script>" in fragment
    assert "教学网络" in fragment and "数值不是研究实验结果" in fragment
    assert "<!doctype" not in fragment.lower()
    checks_path = BASE / "audit/final_structure_delivery_20260911.json"
    checks = dict(utc=datetime.now(timezone.utc).isoformat(), status="passed",
        upstream_manifest_files=upstream_count, original_archive_sha256=archive_hash,
        prior_run_files_unchanged=old_count, prior_frozen_artifacts=len(old_freeze["artifacts"]),
        frozen_artifacts=len(freeze["artifacts"]), source_and_data_hashes_match=True,
        protocol_freeze_prediction_reference_order_valid=True, gap_teachers=teacher_checks,
        model_layer_cases=len(cases), full_tail_vectors=done["full_tail_output_vectors"],
        saved_array_audit=analysis, timing_raw_rows=timing_rows,
        local_links_checked=links, broken_links=[], python_sources_parse=len(sources),
        calibration_model_access=False, new_confirmatory_p_values=0,
        visualization=dict(path=str(VISUAL),sha256=sha(VISUAL),kind="educational, not experiment data"),
        notes="Numerical audit and artifact preservation. Scientific figure visually inspected separately; no browser rendering claimed for inline teaching fragment.")
    dump(checks_path, checks)
    paths = set(p for p in RUN.rglob("*") if p.is_file())
    paths.update(sources); paths.update(docs)
    paths.update([BASE / "RESEARCH_STATUS.json", checks_path, BASE / "audit/origin_trace_20260911.json",
                  BASE / "audit/structure_interpretation_20260911.json",
                  BASE / "reports/structure_controls.png", BASE / "reports/structure_controls.pdf"])
    entries = [dict(path=p.relative_to(BASE).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(paths)]
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(),
        scope="Activity and GAP controls, current narrative and project status; prior run and original archive separately preserved",
        files=len(entries), bytes=sum(r["bytes"] for r in entries), entries=entries)
    dump(BASE / "audit/structure_manifest_20260911.json", snapshot)
    print(json.dumps(dict(status="passed", upstream_files=upstream_count, prior_files=old_count,
        frozen_artifacts=len(freeze["artifacts"]), arrays=analysis["arrays_rescored"],
        timing_rows=timing_rows, links=links, snapshot_files=len(entries), bytes=snapshot["bytes"])))


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

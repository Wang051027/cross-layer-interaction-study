# 中文阅读说明｜用途：检查原六教师实验交付是否与其协议、检查点和文件清单一致。
# 数据流：验证源码语法、结果与引用路径；输出审计清单，不重新训练模型。
# 阅读副本仅增加 # 注释，保留原语句、公式、参数和文档字符串。
# 原研究目录的冻结源码未改；需要哈希核验的历史流程仍使用原目录。
"""Read-only experiment checks, then a separate September 11 delivery snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
import re
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/fashion_external_20260911"
UPSTREAM = BASE / "upstream/CAT_MT_phase7_v0.7"
SEEDS = list(range(202609071, 202609077))


# 用途：对文件字节计算 SHA-256，用于发现数据、源码或检查点是否改变。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：hashlib.file_digest(stream, 'sha256').hexdigest()。
def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# 用途：读取本函数指定的记录文件，返回解析后的对象；JSON/CSV 形式见函数体。
# 参数：path；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：json.loads(Path(path).read_text(encoding='utf-8'))。
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 用途：把结构化记录序列化为 JSON 并写到指定路径。
# 参数：path、value；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def dump(path, value):
    # 把结构化配置或结果落盘，供后续阶段核验和报告读取。
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# 用途：将清单中的相对路径解析为绝对路径，并拒绝越过指定根目录的路径。
# 参数：root、name；具体形状/来源结合调用处及下方关键步骤阅读。
# 返回：path。
def checked_path(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), (root, name)
    return path


# 用途：检查原六教师实验交付是否与其协议、检查点和文件清单一致。
# 参数：无显式入参；使用本模块已配置的环境或文件。
# 返回：结果通过文件、打印或对象/运行环境状态更新体现；没有显式 return。
def main():
    upstream_count = 0
    for line in (UPSTREAM / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(None, 1)
        # 核对内容承诺；注释版的文件字节会变化，原冻结实验应使用原研究目录。
        assert sha(checked_path(UPSTREAM, name)) == expected, name
        upstream_count += 1
    assert upstream_count == 271
    archive_hash = sha(BASE / "incoming/CAT_MT_background_selection_v0.7.zip")
    assert archive_hash == "8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    protocol = read(RUN / "execution_protocol.json")
    freeze = read(RUN / "frozen_before_test.json")
    opened = read(RUN / "test_opened.json")
    completion = read(RUN / "test/completion.json")
    for stem in ("execution_protocol", "frozen_before_test"):
        assert sha(RUN / f"{stem}.json") == (RUN / f"{stem}.sha256").read_text(encoding="utf-8").strip()
    assert freeze["execution_protocol_sha256"] == sha(RUN / "execution_protocol.json")
    assert opened["frozen_before_test_sha256"] == sha(RUN / "frozen_before_test.json")
    for name, expected in protocol["source_sha256"].items():
        assert sha(checked_path(BASE, name)) == expected, name
        assert sha(checked_path(RUN / "source", name)) == expected, name
    for name, expected in protocol["data_sha256"].items():
        assert sha(checked_path(BASE, name)) == expected, name
    for name, expected in freeze["artifacts"].items():
        assert sha(checked_path(RUN, name)) == expected, name
    assert protocol["utc"] < freeze["utc"] < opened["utc"] < completion["utc"]
    assert completion["status"] == "completed" and completion["seeds"] == SEEDS
    assert not completion["post_test_refitting"] and not completion["calibration_model_access"]
    assert completion["model_layer_cases"] == 12
    assert completion["full_tail_intervened_outputs"] == 1572864
    teacher_checks = []
    timing_rows = 0
    # 逐个教师处理同一套设置；跨教师统计与一张图像内部的组合数量要分开。
    for seed in SEEDS:
        directory = RUN / f"seed_{seed}"
        summary = read(directory / "teacher_summary.json")
        history = read(directory / "training_history.json")
        best = min(history, key=lambda row: row["validation_cross_entropy"])
        assert best["epoch"] == summary["best_epoch"]
        assert best["validation_cross_entropy"] == summary["best_validation_cross_entropy"]
        assert best["validation_accuracy"] == summary["best_validation_accuracy"]
        assert summary["epochs"] == len(history) <= 15
        assert sha(directory / "teacher.pt") == summary["checkpoint_sha256"]
        assert not summary["test_used_for_selection"]
        timing = read(directory / "timing/completion.json")
        with (directory / "timing/raw.csv").open(encoding="utf-8", newline="") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        assert count == timing["rows"]
        assert timing["utc"] < freeze["utc"] and not timing["training_concurrent"]
        assert timing["max_formula_difference"] < 1e-9
        timing_rows += count
        for layer in (1, 2):
            case = RUN / f"test/seed_{seed}_L{layer}"
            predicted = read(case / "prediction_commitment.json")
            referenced = read(case / "reference_commitment.json")
            done = read(case / "completion.json")
            assert opened["utc"] < predicted["utc"] < referenced["utc"] <= done["utc"]
            assert sha(case / "predictions.npz") == predicted["sha256"] == done["prediction_sha256"]
            assert sha(case / "references.npz") == referenced["sha256"] == done["reference_sha256"]
        teacher_checks.append(dict(seed=seed, best_epoch=best["epoch"], checkpoint_hash_matches=True,
                                   lowest_validation_cross_entropy_checkpoint=True))
    assert sha(RUN / "seed_202609071/teacher.pt") == sha(BASE / "runs/fashion_pilot_20260910/teacher_seed_202609071.pt")
    analysis = read(RUN / "analysis/verification.json")
    assert analysis["status"] == "passed" and analysis["rescored_arrays"] == 304
    assert analysis["max_nrmse_difference"] == 0
    source_count = 0
    for source in (BASE / "scripts").glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        source_count += 1
    docs = [BASE / "README.md", BASE / "RUNBOOK.md", BASE / "data/fashion_mnist/DATA_CARD.md",
            BASE / "reports/六种子外部测试与推进建议_2026-09-11.md"]
    links_checked = 0
    for doc in docs:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "#")):
                continue
            target = (doc.parent / unquote(link.strip("<>"))).resolve()
            assert target.exists(), (doc, link)
            links_checked += 1
    state = read(BASE / "RESEARCH_STATUS.json")
    assert state["external_teacher_seeds"] == SEEDS and state["external_official_test_runs"] == 6
    assert state["official_test_model_access"] and not state["calibration_model_access"]
    checks_path = BASE / "audit/final_delivery_checks_20260911.json"
    checks = dict(utc=datetime.now(timezone.utc).isoformat(), status="passed",
        upstream_manifest_files=upstream_count, upstream_file_count=sum(p.is_file() for p in UPSTREAM.rglob("*")),
        upstream_archive_sha256=archive_hash, original_archive_and_manifest_unchanged=True,
        source_snapshot_and_live_hashes_match=True, data_hashes_match=True,
        frozen_artifacts_checked=len(freeze["artifacts"]), protocol_freeze_test_order_valid=True,
        teachers=teacher_checks, first_teacher_identical_to_pilot=True,
        timing_raw_rows=timing_rows, saved_array_audit=analysis,
        local_links_checked=links_checked, broken_local_links=[], python_sources_parse=source_count,
        official_test_model_runs=6, calibration_model_access=False,
        notes="Source and numerical result audit; figure separately inspected visually. Historical records retain their original phase status.")
    dump(checks_path, checks)
    manifest_paths = set(path for path in RUN.rglob("*") if path.is_file())
    manifest_paths.update((BASE / "scripts").glob("*.py"))
    manifest_paths.update(docs)
    manifest_paths.update((BASE / "RESEARCH_STATUS.json", checks_path,
        BASE / "audit/head_activity_20260911.json", BASE / "audit/head_activity_20260911.csv",
        BASE / "reports/six_seed_external_results.png", BASE / "reports/six_seed_external_results.pdf"))
    entries = [dict(path=path.relative_to(BASE).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
               for path in sorted(manifest_paths)]
    manifest = dict(utc=datetime.now(timezone.utc).isoformat(),
        scope="September 11 six-seed run and current reporting/source snapshot; original archive separately verified",
        files=len(entries), bytes=sum(row["bytes"] for row in entries), entries=entries)
    dump(BASE / "audit/followup_manifest_20260911.json", manifest)
    print(json.dumps(dict(status="passed", upstream_files=upstream_count,
        frozen_artifacts=len(freeze["artifacts"]), timing_rows=timing_rows, links=links_checked,
        delivery_files=manifest["files"], delivery_bytes=manifest["bytes"])))


# 只有直接运行本文件时才触发这个入口；作为模块导入时不会执行此分支。
if __name__ == "__main__":
    main()

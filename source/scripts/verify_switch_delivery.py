"""Preservation, provenance, and delivery audit for the exact-switch study."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from urllib.parse import unquote
import ast
import csv
import hashlib
import json
import re
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE/"runs/fashion_switch_20260914"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


# 核对输入/源码等清单指向的文件与预期哈希一致。
def check(root, values):
    for name, expected in values.items():
        path = (root/name).resolve()
        assert path.is_relative_to(root.resolve())
        assert sha(path) == expected, name


def rows(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


# 核验状态筛选实验及 q=12 补充全枚举的文件、数值和统计完整性。
def main():
    archive = BASE/"incoming/CAT_MT_background_selection_v0.7.zip"
    assert sha(archive) == "8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    upstream = BASE/"upstream/CAT_MT_phase7_v0.7"
    original = {}
    for line in (upstream/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        value,name = line.split(None,1)
        original[name] = value
    check(upstream,original)
    assert len(original) == 271
    prior_runs = []
    for name,snapshot in (("fashion_external_20260911","followup_manifest_20260911.json"),
                          ("fashion_structure_20260911","structure_manifest_20260911.json"),
                          ("fashion_residual_20260913","residual_manifest_20260913.json"),
                          ("fashion_scale_20260914","scale_manifest_20260914.json")):
        entries = {e["path"]:e["sha256"] for e in read(BASE/"audit"/snapshot)["entries"]
                   if e["path"].startswith(f"runs/{name}/")}
        check(BASE,entries)
        prior_runs.append(dict(run=name,files_unchanged=len(entries)))
    expected = dict(protocol="8a9ae6fbe3d516b5024df583d74040b55a5dc1ad707668fd56c4391fee62d3b9",
                    implementation="6218efaf3509fc406fb7d4a88eb2f318060f1c04204ea5bee02307dcc7a8895a")
    for stem in ("protocol","implementation","freeze"):
        actual = sha(RUN/f"{stem}.json")
        assert actual == (RUN/f"{stem}.sha256").read_text(encoding="ascii").strip()
        if stem in expected:assert actual == expected[stem]
        expected[stem] = actual
    protocol,impl,freeze = (read(RUN/f"{stem}.json") for stem in ("protocol","implementation","freeze"))
    development = read(RUN/"development_complete.json")
    timing = read(RUN/"timing/completion.json")
    interruption = read(RUN/"timing_interruption.json")
    opened,completed = read(RUN/"test_opened.json"),read(RUN/"test/completion.json")
    assert protocol["utc"] < impl["utc"] < development["utc"] < interruption["utc"] < timing["utc"] < freeze["utc"] < opened["utc"] < completed["utc"]
    assert opened["freeze_sha256"] == expected["freeze"]
    assert freeze["protocol_sha256"] == impl["protocol_sha256"] == expected["protocol"]
    assert freeze["implementation_sha256"] == expected["implementation"]
    assert interruption["saved_timing_rows"] == 0 and not interruption["test_accessed"]
    assert not list((RUN/"timing_interrupted_20260914").iterdir())
    check(BASE,protocol["imported_sources"])
    check(BASE,protocol["data_hashes"])
    check(BASE,protocol["prior_fit_hashes"])
    check(BASE,protocol["prior_scale_run_hashes"])
    check(BASE,impl["sources"])
    check(RUN/"source",impl["sources"])
    check(RUN,freeze["artifacts"])
    assert sha(RUN/"layouts.npz") == protocol["layouts_sha256"]
    assert sha(RUN/"intervention_ids.npy") == protocol["intervention_ids_sha256"]
    assert np.array_equal(np.load(RUN/"intervention_ids.npy",allow_pickle=False),np.arange(896,1024))
    counts = Counter()
    vectors = Counter()
    candidate_min,candidate_max = 100000,0
    for phase in ("validation","supplementary","test"):
        for seed in range(202609071,202609077):
            assert sha(BASE/f"runs/fashion_residual_20260913/seed_{seed}/teacher.pt") == protocol["teachers_sha256"][str(seed)]
            for position in ("center","upper_left"):
                for q in ([12] if phase == "supplementary" else (4,8,12)):
                    case = RUN/phase/f"seed_{seed}/{position}_q{q}"
                    pc,rc,done = (read(case/f"{s}.json") for s in ("prediction_commitment","reference_commitment","completion"))
                    assert pc["utc"] < rc["utc"] <= done["utc"]
                    assert sha(case/"predictions.npz") == pc["sha256"]
                    assert sha(case/"screening_contexts.npz") == pc["screening_sha256"]
                    assert sha(BASE/rc["path"]) == rc["sha256"]
                    assert done["max_exact_error"] < 1e-9 and done["max_omitted_remainder"] < 1e-9
                    assert done["fresh_cached_difference"] < 1e-9
                    if phase == "test":
                        assert opened["utc"] < pc["utc"] < done["utc"] <= completed["utc"]
                        assert done["inputs"] == 128 and done["arrays"] == 8
                    else:
                        assert impl["utc"] < pc["utc"] < done["utc"] < development["utc"] < freeze["utc"]
                    if phase == "supplementary":
                        assert done["inputs"] == 4 and done["queried_coalitions"] == 4096 and done["arrays"] == 4
                    else:
                        assert done["queried_coalitions"] == {4:16,8:256,12:335}[q]
                    counts[phase] += done["arrays"]
                    vectors[phase] += done["original_reference_vectors"]
                    for row in rows(case/"screening.csv"):
                        J,K = int(row["affected_states"]),int(row["candidates"])
                        assert K+int(row["stable_on"])+int(row["stable_off"]) == J
                        assert int(row["switch_cache_bytes"]) == 8*(11*q+K*(q+11))
                        assert int(row["exact_cache_bytes"]) == 8*(q+J)
                        candidate_min,candidate_max = min(candidate_min,K),max(candidate_max,K)
    assert dict(counts) == dict(validation=288,supplementary=48,test=288)
    assert dict(vectors) == dict(validation=1864704,supplementary=393216,test=1864704)
    assert completed["status"] == "completed" and completed["cases"] == 36
    assert completed["full_tail_output_vectors"] == vectors["test"]
    verification = read(RUN/"analysis/verification.json")
    assert verification["status"] == "passed" and verification["arrays_rescored"] == dict(counts)
    assert verification["max_metric_difference"] < 1e-12
    assert verification["max_numpy_split_difference"] < 1e-9 and verification["max_omitted_remainder"] < 1e-9
    assert verification["all_timing_aggregations_reproduced"]
    raw = rows(RUN/"timing/raw.csv")
    assert len(raw) == timing["rows"] == verification["timing_rows"] == 42336
    groups = defaultdict(list)
    for row in raw:
        key = tuple(row[k] for k in ("seed","position","q","input_id","query_set","workload","method"))
        groups[key].append(int(row["round"]))
        assert float(row["seconds"]) > 0.
    assert all(sorted(v) == list(range(7)) for v in groups.values())
    assert timing["threads"] == 1 and not timing["training_concurrent"]
    assert read(BASE/"audit/switch_interpretation_20260917.json")["status"] == "passed"
    state = read(BASE/"RESEARCH_STATUS.json")
    assert state["current_run"] == "runs/fashion_switch_20260914" and state["date"] == "2026-09-17"
    assert state["total_fashion_teachers"] == state["official_test_classification_teacher_runs"] == 18
    assert state["external_official_test_batches"] == 5 and state["completed_fashion_intervention_settings"] == 120
    assert not state["calibration_model_access"] and state["switch_arrays_rescored"] == 624
    docs = [BASE/n for n in ("README.md","RUNBOOK.md","data/fashion_mnist/DATA_CARD.md",
        "reports/可翻转状态筛选与精确组合计算_2026-09-17.md",
        "reports/稳定状态合并与精确修正_数学说明_2026-09-17.md",
        "reports/从原始设想到当前实验_2026-09-11.md")]
    audit_path = BASE/"audit/final_switch_delivery_20260917.json"
    links = 0
    for doc in docs:
        content = doc.read_text(encoding="utf-8")
        assert not any(ord(char)<32 and char not in "\n\t\r" for char in content),doc
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",content):
            if target.startswith(("https://","http://","#")):continue
            path = (doc.parent/unquote(target.strip("<>"))).resolve()
            assert path.exists() or path == audit_path.resolve(),(doc,target)
            links += 1
    sources = list((BASE/"scripts").glob("*.py"))
    for source in sources:ast.parse(source.read_text(encoding="utf-8-sig"),filename=str(source))
    figures = [BASE/f"reports/switch_exact_results.{ext}" for ext in ("png","pdf")]
    assert all(p.is_file() and p.stat().st_size>10000 for p in figures)
    assert not list(BASE.glob("*.zip")) and not list(BASE.parents[1].glob("*.zip"))
    result = dict(utc=datetime.now(timezone.utc).isoformat(),status="passed",prior_runs=prior_runs,
        original_archive_sha256=sha(archive),upstream_manifest_files=len(original),
        frozen_records=expected,frozen_artifacts=len(freeze["artifacts"]),arrays_rescored=dict(counts),
        reference_vectors=dict(vectors),new_test_prediction_before_reference=True,
        validation_references_reused=True,timing_interruption_retained=True,timing_rows=len(raw),
        numerical_audit=verification,candidate_count_range=[candidate_min,candidate_max],cache_storage_formula_checked=True,
        calibration_model_access=False,new_confirmatory_p_values=0,archives_not_recreated=True,
        local_links_checked=links,python_sources_parsed=len(sources),
        figure_review="Final PNG visually inspected; PDF exported from the same Matplotlib figure")
    audit_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    paths = {p for p in RUN.rglob("*") if p.is_file()}
    paths.update(sources+docs+figures+[BASE/"RESEARCH_STATUS.json",audit_path,BASE/"audit/switch_interpretation_20260917.json"])
    entries = [dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(),scope="Exact switch screening completed September17, immutable numerical provenance begun September14; previous four runs preserved",
        files=len(entries),bytes=sum(e["bytes"] for e in entries),entries=entries)
    (BASE/"audit/switch_manifest_20260917.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="passed",prior_runs=prior_runs,arrays=dict(counts),timing_rows=len(raw),
        snapshot_files=len(entries),bytes=snapshot["bytes"],links=links)))


if __name__ == "__main__":
    main()

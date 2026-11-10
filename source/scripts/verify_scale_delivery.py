"""Final preservation and delivery checks for the size/position study."""
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

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/"runs/fashion_scale_20260914"
SEEDS=list(range(202609071,202609077))


def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


# 按根目录核对每个保存文件的摘要，并防止路径越界。
def check(root,values):
    for name,expected in values.items():
        path=(root/name).resolve()
        assert path.is_relative_to(root.resolve())
        assert sha(path)==expected,name


# 核验规模/位置阶段的协议、数组、方法标签及交付文件。
def main():
    archive=BASE/"incoming/CAT_MT_background_selection_v0.7.zip"
    assert sha(archive)=="8fe0094d2260e84e98905e53f2fdec211fa0644b1e0ecfa194ab2633ad6f0aae"
    upstream=BASE/"upstream/CAT_MT_phase7_v0.7"
    manifest={}
    for line in (upstream/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        value,name=line.split(None,1);manifest[name]=value
    check(upstream,manifest)
    assert len(manifest)==271
    prior=[]
    for name,snapshot in [
        ("fashion_external_20260911","followup_manifest_20260911.json"),
        ("fashion_structure_20260911","structure_manifest_20260911.json"),
        ("fashion_residual_20260913","residual_manifest_20260913.json")]:
        entries={r["path"]:r["sha256"] for r in read(BASE/"audit"/snapshot)["entries"]
                 if r["path"].startswith(f"runs/{name}/")}
        check(BASE,entries)
        prior.append(dict(run=name,files_unchanged=len(entries)))
    expected={
        "protocol":"b24c68896892313172e9bf239943d2d6bb7660cae865ace6ec124a8b6b904f2b",
        "implementation":"f8c948a7e6f23af679fe83c7252594aa2e00dfaa95e152f7ef6a8922b5d178d6",
        "freeze":"94f011f9278d365ef6bc5a4741e2e69be3a288aaee1bdfdf8c2f5bad0705ec47"}
    for stem,value in expected.items():
        assert sha(RUN/f"{stem}.json")==value
        assert (RUN/f"{stem}.sha256").read_text(encoding="ascii").strip()==value
    protocol,impl,freeze=(read(RUN/f"{s}.json") for s in expected)
    opened=read(RUN/"test_opened.json");completed=read(RUN/"test/completion.json")
    assert protocol["utc"]<impl["utc"]<freeze["utc"]<opened["utc"]<completed["utc"]
    assert impl["protocol_sha256"]==freeze["protocol_sha256"]==expected["protocol"]
    assert freeze["implementation_sha256"]==expected["implementation"]
    assert opened["freeze_sha256"]==expected["freeze"]
    check(BASE,protocol["imported_sources"]);check(BASE,protocol["data_hashes"])
    check(BASE,impl["sources"]);check(RUN/"source",impl["sources"])
    check(RUN,freeze["artifacts"])
    assert len(freeze["artifacts"])==450
    for seed in SEEDS:
        assert sha(BASE/f"runs/fashion_residual_20260913/seed_{seed}/teacher.pt")==protocol["teachers_sha256"][str(seed)]
    for q in (4,8,12):
        path=RUN/f"design_q{q}.npz"
        assert sha(path)==protocol["designs"][str(q)]["sha256"]
        with np.load(path,allow_pickle=False) as d:
            assert len(np.unique(d["full_mask_indices"]))==len(d["evaluation"])
            assert d["train"].shape==(1+q+q*(q-1)//2,q)
            assert len(d["evaluation"])=={4:16,8:256,12:335}[q]
            assert int((d["evaluation"].sum(1)>2).sum())=={4:5,8:219,12:256}[q]
    counts=Counter();validation_vectors=0;test_vectors=0
    for seed in SEEDS:
        for position in ("center","upper_left"):
            old_nodes=None
            for q in (4,8,12):
                folder=RUN/f"seed_{seed}/{position}_q{q}"
                fit=read(folder/"fit_complete.json")
                assert sha(folder/"fit.npz")==fit["fit_sha256"]
                with np.load(folder/"fit.npz",allow_pickle=False) as a:
                    nodes=a["nodes"]
                    assert len(nodes)==q
                    if old_nodes is not None:assert np.array_equal(nodes[:len(old_nodes)],old_nodes)
                    old_nodes=nodes.copy()
                for phase,case in (("validation",folder/"validation"),
                                   ("official_test",RUN/f"test/seed_{seed}/{position}_q{q}")):
                    pc=read(case/"prediction_commitment.json");rc=read(case/"reference_commitment.json")
                    done=read(case/"completion.json")
                    assert pc["utc"]<rc["utc"]<=done["utc"]
                    if phase=="validation":
                        assert fit["utc"]<pc["utc"]<done["utc"]<freeze["utc"]
                        validation_vectors+=done["full_tail_output_vectors"]
                    else:
                        assert opened["utc"]<pc["utc"]<done["utc"]<=completed["utc"]
                        test_vectors+=done["full_tail_output_vectors"]
                    assert sha(case/"predictions.npz")==pc["sha256"]
                    assert sha(case/"references.npz")==rc["sha256"]
                    assert sha(case/"decisions_before_reference.json")==pc["decisions_sha256"]
                    assert done["arrays"]==20 and done["inputs"]==128 and done["phase"]==phase
                    assert done["exact_max_error"]<1e-9 and done["bound_excess"]<1e-9
                    assert done["accepted_relative_threshold_exceptions"]==0
                    counts[phase]+=done["arrays"]
    assert dict(counts)=={"validation":720,"official_test":720}
    assert completed["status"]=="completed" and completed["cases"]==36
    assert test_vectors==completed["full_tail_output_vectors"]==validation_vectors==1864704
    assert completed["new_teacher_training_runs"]==completed["new_classification_runs"]==0
    audit=read(RUN/"analysis/verification.json")
    assert audit["status"]=="passed" and audit["arrays_rescored"]==dict(counts)
    assert audit["max_nrmse_difference"]<1e-12 and audit["max_input_p95_difference"]<1e-12
    assert not audit["calibration_model_access"] and audit["new_confirmatory_p_values"]==0
    interpretation=read(BASE/"audit/scale_interpretation_20260914.json")
    assert interpretation["status"]=="passed" and interpretation["shared_validation_replay"]["arrays"]==108
    assert interpretation["shared_validation_replay"]["max_prediction_difference"]<1e-9
    with (RUN/"timing/raw.csv").open(encoding="utf-8",newline="") as f:timing=list(csv.DictReader(f))
    assert len(timing)==48384
    rounds=defaultdict(list)
    for r in timing:
        key=tuple(r[k] for k in ("seed","position","q","input_id","query_set","workload","method"))
        rounds[key].append(int(r["round"]))
        assert float(r["seconds"])>0
    assert all(sorted(v)==list(range(7)) for v in rounds.values())
    timing_done=read(RUN/"timing/completion.json")
    assert timing_done["rows"]==len(timing) and timing_done["utc"]<freeze["utc"]
    assert not timing_done["training_concurrent"]
    state=read(BASE/"RESEARCH_STATUS.json")
    assert state["current_run"]=="runs/fashion_scale_20260914"
    assert state["total_fashion_teachers"]==state["official_test_classification_teacher_runs"]==18
    assert state["scale_position_cases"]==36 and state["external_official_test_batches"]==4
    assert not state["calibration_model_access"]
    docs=[BASE/n for n in ("README.md","RUNBOOK.md","data/fashion_mnist/DATA_CARD.md",
        "reports/规模位置变化与精确回退_2026-09-14.md",
        "reports/局部线性回退规则_数学说明_2026-09-14.md",
        "reports/从原始设想到当前实验_2026-09-11.md")]
    links=0
    for doc in docs:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",doc.read_text(encoding="utf-8")):
            if target.startswith(("http://","https://","#")):continue
            assert (doc.parent/unquote(target.strip("<>"))).resolve().exists(),(doc,target)
            links+=1
    sources=list((BASE/"scripts").glob("*.py"))
    for p in sources:ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))
    figures=[BASE/f"reports/scale_position_results.{ext}" for ext in ("png","pdf")]
    assert all(p.is_file() and p.stat().st_size>10000 for p in figures)
    assert not list(BASE.glob("*.zip")) and not list(BASE.parents[1].glob("*.zip"))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),status="passed",
        original_archive_sha256=sha(archive),upstream_manifest_files=len(manifest),prior_runs=prior,
        frozen_records=expected,frozen_artifacts=len(freeze["artifacts"]),
        prediction_before_reference=True,arrays_rescored=dict(counts),
        array_verification=audit,shared_validation_replay=interpretation["shared_validation_replay"],
        validation_reference_vectors=validation_vectors,test_reference_vectors=test_vectors,
        timing_rows=len(timing),bound_relative_threshold_exceptions=completed["accepted_relative_threshold_exceptions"],
        calibration_model_access=False,new_confirmatory_p_values=0,
        local_links_checked=links,python_sources_parse=len(sources),
        archives_not_recreated=True,figure_review="Final PNG visually inspected; PDF exported from the same Matplotlib figure")
    audit_path=BASE/"audit/final_scale_delivery_20260914.json"
    audit_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    paths={p for p in RUN.rglob("*") if p.is_file()}
    paths.update(sources+docs+figures+[BASE/"RESEARCH_STATUS.json",audit_path,BASE/"audit/scale_interpretation_20260914.json"])
    entries=[dict(path=p.relative_to(BASE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    snapshot=dict(utc=datetime.now(timezone.utc).isoformat(),scope="Size/position study and analytic-bound fallback; all previous numerical runs preserved",
        files=len(entries),bytes=sum(e["bytes"] for e in entries),entries=entries)
    (BASE/"audit/scale_manifest_20260914.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="passed",prior_runs=prior,arrays=dict(counts),timing_rows=len(timing),
                         snapshot_files=len(entries),bytes=snapshot["bytes"],links=links)))


if __name__=="__main__":main()

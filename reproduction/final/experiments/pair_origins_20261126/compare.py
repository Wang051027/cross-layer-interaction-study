"""比较两套权重前向结果、冻结名单、分类阈值和逐例开关事件。"""
import argparse
from pathlib import Path

import numpy as np

from mechanism import HERE, js, put, sha, verify
from verify_replay import compare_table


TABLES = ("development_channels", "development_conditions", "predictions", "conditions", "switches",
          "arm_scores", "joint_effects", "upstream_four", "factor_counts", "rule_scores", "site_scores",
          "case_effects", "gate_balance", "gate_balance_cases", "local_cases", "local_coordinates", "local_summary",
          "coverage_cases", "coverage_scores")


def check_folder(path):
    selection = js(path/"selection_seal.json")
    prediction = js(path/"prediction_seal.json")
    measurement = js(path/"measurement_seal.json")
    verify(path, selection["files"])
    verify(path, {"selection_seal.json":prediction["selection_seal"], **prediction["files"]})
    verify(path, {"prediction_seal.json":measurement["prediction_seal"], **measurement["files"]})
    local = js(path/"local_checks.json")
    coverage = js(path/"coverage_checks.json")
    scored = js(path/"score_checks.json")
    verify(path, {"measurement_seal.json":local["measurement"], **local["files"]})
    verify(path, {"measurement_seal.json":coverage["previous"], **coverage["files"]})
    ranking = js(path/"coverage_order.json")
    if (coverage["status"]!="PASS" or ranking["source"]!=sha(HERE/"coverage.py")
        or ranking["design"]!=sha(HERE/"COVERAGE.md")):
        raise ValueError("Coverage protocol or source differs")
    if (local["status"]!="PASS" or scored["status"]!="PASS" or scored["measurement"]!=local["measurement"]
        or sha(HERE/"LOCAL_DESIGN.md")!=local["design"] or sha(HERE/"local_trace.py")!=local["source"]):
        raise ValueError("Gate replay or score seal differs")
    return dict(device=js(path/"protocol.json")["device"],measurement=local["measurement"])


def main(reference, replay, out):
    if reference.resolve()==replay.resolve():
        raise ValueError("Independent directories required")
    devices=[check_folder(p) for p in (reference,replay)]
    if (js(reference/"selection.json")!=js(replay/"selection.json")
        or js(reference/"coverage_order.json")["orders"]!=js(replay/"coverage_order.json")["orders"]):
        raise ValueError("Channel selection, coordinates or sign thresholds differ")
    with np.load(reference/"responses.npz",allow_pickle=False) as a,np.load(replay/"responses.npz",allow_pickle=False) as b:
        for k in ("keys","pairs","arms","columns"):
            np.testing.assert_array_equal(a[k],b[k])
        x,y=a["values"],b["values"]
        np.testing.assert_allclose(x,y,rtol=0,atol=1e-8)
        raw=float(np.max(np.abs(x-y)))
    tables={name:compare_table(name,reference,replay) for name in TABLES}
    worst=max(raw,max(v["max_abs"] for v in tables.values()))
    put(out,dict(status="PASS",devices=devices,selection_identical=True,raw_max_abs=raw,
                 tables=tables,maximum=worst,tolerance=1e-8))
    print(f"All channel pairs, gate coordinates, thresholds and {len(TABLES)} tables matched. Max {worst:.3g}.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("replay",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    main(args.reference,args.replay,args.out)

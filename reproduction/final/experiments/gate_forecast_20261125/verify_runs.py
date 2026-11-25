"""比较两次实验的逐例预测、实测、位置名单和八状态差分。"""
import argparse
from pathlib import Path

import numpy as np
from analyze import keyed
from run import js, put, read_csv, sha, verify


def compare_table(first, second):
    a,b=read_csv(first),read_csv(second)
    if not a or len(a)!=len(b) or list(a[0])!=list(b[0]):
        raise ValueError(f"Table layout changed: {first.name}")
    maximum=0.
    for name in a[0]:
        x,y=[r[name] for r in a],[r[name] for r in b]
        if x==y:
            continue
        x,y=np.array(x,dtype=float),np.array(y,dtype=float)
        gap=float(np.max(np.abs(x-y)))
        if not np.isfinite(gap) or gap>1e-8:
            raise ValueError((first.name,name,gap))
        maximum=max(maximum,gap)
    return dict(rows=len(a),max_abs_difference=maximum,identical_file=sha(first)==sha(second))


def confirm_seals(path):
    p,m=js(path/"prediction_seal.json"),js(path/"measurement_seal.json")
    verify(path,{"protocol.json":p["protocol"],**p["files"],**m["files"],"prediction_seal.json":m["predictions"]})
    corner=path/"eight_corners"
    p,m=js(corner/"prediction_seal.json"),js(corner/"measurement_seal.json")
    verify(corner,{**p["files"],**m["files"],"prediction_seal.json":m["prediction_seal"]})
    verify(path,{"measurement_seal.json":m["primary_measurement_seal"]})
    return js(path/"summary.json")["by_strength"]


def main(args):
    primary,replay=args.primary,args.replay
    a,b=confirm_seals(primary),confirm_seals(replay)
    for strength in ("1.0","0.5"):
        assert a[strength]["decisions"]==b[strength]["decisions"]
    tables=["predictions.csv","measurements.csv","intervention_predictions.csv","interventions.csv",
            "teacher_scores.csv","gate_scores.csv","intervention_scores.csv","selected_accuracy.csv","layer_interaction.csv",
            "eight_corners/low_corner_effects.csv","eight_corners/predictions.csv","eight_corners/measurements.csv",
            "eight_corners/teacher_scores.csv"]
    differences={f:compare_table(primary/f,replay/f) for f in tables}
    arrays={}
    for f in ("forecast_sites.npz","observed_gates.npz"):
        with np.load(primary/f,allow_pickle=False) as old,np.load(replay/f,allow_pickle=False) as new:
            assert set(old.files)==set(new.files)
            for key in old.files:
                np.testing.assert_array_equal(old[key],new[key])
                arrays[f+"/"+key]=list(old[key].shape)
    old_scores=keyed(args.nov10/"predictions.csv")
    old_truth=keyed(args.nov10/"measurements.csv")
    ours=keyed(primary/"predictions.csv")
    current_truth=keyed(primary/"measurements.csv")
    assert set(ours)==set(old_scores)==set(old_truth)==set(current_truth)
    columns=("class_mean","fixed_0","fixed_4","fixed_32","live_0","live_4","live_32")
    old_gaps={name:max(abs(float(p[name])-float(old_scores[k][name])) for k,p in ours.items()) for name in columns}
    old_gaps["y"]=max(abs(float(p["y"])-float(old_truth[k]["y"])) for k,p in current_truth.items())
    assert max(old_gaps.values())<1e-8
    original=js(args.nov10/"independent_audit.json")
    repeated=js(args.nov10_replay/"independent_audit.json")
    independent=js(primary/"independent_predictions.json")
    assert original["status"]==repeated["status"]==independent["status"]=="PASS"
    previous={f:compare_table(args.nov10/f,args.nov10_replay/f)
              for f in ("predictions.csv","measurements.csv","teacher_scores.csv","gate_identity.csv")}
    result=dict(status="PASS",primary_device=js(primary/"audit.json")["device"],
        replay_device=js(replay/"audit.json")["device"],primary_measurement=js(primary/"audit.json")["measurement"],
        replay_measurement=js(replay/"audit.json")["measurement"],primary_seal=sha(primary/"measurement_seal.json"),
        replay_seal=sha(replay/"measurement_seal.json"),complete_cases=864,endpoint_interventions=7776,
        seven_corner_interventions=54432,eight_state_contrasts=7776,matching_arrays=arrays,tables=differences,
        compared_nov10_columns=old_gaps,nov10_replay=previous,nov10_cpu_independent_max=max(repeated["maxima"].values()),
        independent_forecasts=independent,maximum_abs_difference=max(d["max_abs_difference"] for d in differences.values()))
    put(args.out,result)
    print("PASS",result["maximum_abs_difference"],"maximum table difference; exact gates and position IDs.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",required=True,type=Path)
    parser.add_argument("--replay",required=True,type=Path)
    parser.add_argument("--nov10",required=True,type=Path)
    parser.add_argument("--nov10-replay",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    main(parser.parse_args())

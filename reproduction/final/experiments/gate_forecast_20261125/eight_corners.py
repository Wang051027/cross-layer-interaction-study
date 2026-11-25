"""在同一份三组位置名单上测量八角阻断后的三阶变化。"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

from analyze import ARMS, LAYERS, agreement, avg, error_ratio, keyed, rms
from run import HERE, case_key, checksums, js, location, put, read_csv, sha, stamp, verify, write_csv


def predict(args):
    run = args.run.resolve()
    out = run / "eight_corners"
    protocol = js(run / "protocol.json")
    seal = js(run / "prediction_seal.json")
    verify(run, {"protocol.json":seal["protocol"], **seal["files"]})
    frozen, _, source = location(args.assets.resolve())
    verify(frozen, {k.removeprefix("runs/"):v for k,v in protocol["inputs"].items()})
    verify(source, protocol["shared_code"])
    verify(HERE, protocol["design_and_code"])
    sys.path.insert(0, str(source))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import dsprites_teacher as teacher
    import torch
    import gates

    teacher.configure()
    torch.set_grad_enabled(False)
    out.mkdir(exist_ok=False)
    put(out / "protocol.json", dict(utc=stamp(), primary_prediction_seal=sha(run/"prediction_seal.json"),
        primary_measurement_seal_known=sha(run/"measurement_seal.json"),
        source=checksums(HERE, ("EIGHT_CORNERS.md","eight_corners.py")), engine=args.engine, device=args.device))
    with np.load(run/"forecast_sites.npz", allow_pickle=False) as archive:
        positions, position_keys = archive["sites"], archive["keys"]
    with np.load(frozen / "dsprites_shape_donor_20261010/panel/data.npz", allow_pickle=False) as archive:
        panel = {k:archive[k] for k in ("images","labels","pairs")}
    endpoint_predictions = keyed(run / "intervention_predictions.csv", True)
    coefficients = np.array([1, -1, -1, -1, 1, 1, 1])
    cases = protocol["held_cases"]
    corner_rows, forecasts = [], []
    row_index = 0
    for seed in protocol["seeds"]:
        net = teacher.DSpritesCNN()
        net.load_state_dict(torch.load(frozen/f"dsprites_20260929/seed_{seed}/teacher.pt", weights_only=True, map_location="cpu"))
        net = net.double().to(args.device).eval().requires_grad_(False)
        bank = gates.rule.source_bank(net, panel, cases, args.device)
        for strength in (1.0, 0.5):
            for case in cases:
                ri,di = map(int,panel["pairs"][case])
                classes = tuple(map(int,panel["labels"][[ri,di]]))
                masks = gates.rule.LOW
                inputs = gates.rule.mixed_sources(bank[ri],bank[di],masks,strength)
                middle = net.block2(inputs)
                natural = gates.endpoint(net,middle,classes)
                empty = gates.Pair(*(field[:1] for field in natural.gates))
                for triple in gates.rule.TRIPLES:
                    assert tuple(position_keys[row_index]) == (seed,int(strength*100),case,triple)
                    low = gates.rule.corners(triple)[:7]
                    ids = [masks.index(m) for m in low]
                    state = gates.Endpoint(middle[ids], gates.Pair(*(z[ids] for z in natural.z)),
                                           gates.Pair(*(g[ids] for g in natural.gates)), natural.score[ids])
                    lists = np.repeat(positions[row_index:row_index+1], len(low), axis=0)
                    engine = gates.reference_patches if args.engine=="indexed" else gates.patch_effects
                    changes = engine(net, state, empty, classes, lists).cpu().numpy()
                    assert np.max(np.abs(changes[0])) < 1e-9
                    q_changes = np.einsum("s,sal->al",coefficients,changes)
                    for a,arm in enumerate(ARMS):
                        for l,layer in enumerate(LAYERS):
                            fields = dict(seed=seed,strength=strength,case=case,mask=triple,arm=arm,layer=layer)
                            key = (seed,strength,case,triple,arm,layer)
                            for i, subset in enumerate(low):
                                corner_rows.append(dict(**fields,subset=subset,low_change=changes[i,a,l]))
                            q = float(q_changes[a,l])
                            p = float(endpoint_predictions[key]["predicted_change"])
                            forecasts.append(dict(**fields,q_low_change=q,predicted_third_change=p-q))
                    row_index += 1
        print("seven matched low states",seed,flush=True)
    assert row_index == 864 and len(forecasts)==7776 and len(corner_rows)==54432
    write_csv(out/"low_corner_effects.csv",corner_rows)
    write_csv(out/"predictions.csv",forecasts)
    put(out/"prediction_seal.json",dict(utc=stamp(),files=checksums(out,
        ("protocol.json","low_corner_effects.csv","predictions.csv"))))


def score_experiment(run):
    out = run/"eight_corners"
    primary = js(run/"measurement_seal.json")
    verify(run,primary["files"])
    verify(out,js(out/"prediction_seal.json")["files"])
    verify(HERE,js(out/"protocol.json")["source"])
    pred = keyed(out/"predictions.csv",True)
    patch = keyed(run/"interventions.csv",True)
    original = keyed(run/"intervention_predictions.csv",True)
    truth = keyed(run/"measurements.csv")
    assert set(pred)==set(patch)==set(original) and len(pred)==7776
    # 用子集个数决定符号，再从原始七个干预值独立重建差分。
    low_sum = {}
    for row in read_csv(out/"low_corner_effects.csv"):
        key = case_key(row)+(row["arm"],row["layer"])
        subset = int(row["subset"])
        assert subset & key[3] == subset and subset != key[3]
        values = low_sum.setdefault(key, {})
        assert subset not in values
        values[subset] = float(row["low_change"])
    assert set(low_sum)==set(pred)
    measured = []
    max_error = 0.0
    for key, row in pred.items():
        q = float(row["q_low_change"])
        assert len(low_sum[key])==7
        check = sum((-1)**int(s).bit_count()*v for s,v in low_sum[key].items())
        observed = float(patch[key]["actual_change"])
        p = float(row["predicted_third_change"])
        delta = observed-q
        unchanged_error = float(original[key]["predicted_change"])-observed
        max_error=max(max_error,abs(q-check),abs((p-delta)-unchanged_error))
        y = float(truth[key[:4]]["y"])
        measured.append(dict(**row,actual_endpoint_change=observed,actual_third_change=delta,
                             natural_third=y,patched_third=y+delta))
    assert max_error < 1e-9
    write_csv(out/"measurements.csv",measured)
    teacher = []
    for alpha in (1.,0.5):
        for seed in sorted({k[0] for k in pred}):
            for arm in ARMS:
                for layer in LAYERS:
                    rows=[r for r in measured if float(r["strength"])==alpha and int(r["seed"])==seed
                          and (r["arm"],r["layer"])==(arm,layer)]
                    assert len(rows)==72
                    def array(col):
                        return np.array([float(r[col]) for r in rows])
                    actual, forecast, y = (array(k) for k in ("actual_third_change","predicted_third_change","natural_third"))
                    endpoint = array("actual_endpoint_change")
                    hits, total = agreement(forecast,actual)
                    teacher.append(dict(seed=seed,strength=alpha,arm=arm,layer=layer,actual_rms=rms(actual),
                        error_ratio=error_ratio(forecast,actual),correct_sign=hits,sign_total=total,
                        remaining_ratio=rms(y+actual)/rms(y),
                        removal_projection=float(-np.dot(actual,y)/np.dot(y,y)),attenuated_cases=int(np.sum(abs(y+actual)<abs(y))),
                        opposite_to_endpoint=int(np.sum(endpoint*actual<0))))
    write_csv(out/"teacher_scores.csv",teacher)
    summary = {}
    for alpha in (1.,0.5):
        result = {}
        for mode in LAYERS:
            by_arm={}
            for arm in ARMS:
                group=[r for r in teacher if r["strength"]==alpha and (r["arm"],r["layer"])==(arm,mode)]
                by_arm[arm] = {col:avg(group,col) for col in ("actual_rms","error_ratio","remaining_ratio","removal_projection")}
                by_arm[arm].update({col:sum(r[col] for r in group) for col in
                                  ("correct_sign","sign_total","opposite_to_endpoint","attenuated_cases")})
                by_arm[arm]["attenuated_models"]=sum(r["remaining_ratio"]<1 for r in group)
            result[mode]=by_arm
        summary[str(alpha)]=result
    put(out/"summary.json",dict(by_strength=summary,cases=864,low_arm_states=len(pred)*7,
                                 interventions=len(pred),algebra_maximum=max_error))
    put(out/"measurement_seal.json",dict(utc=stamp(),prediction_seal=sha(out/"prediction_seal.json"),
        primary_measurement_seal=sha(run/"measurement_seal.json"),
        files=checksums(out,("measurements.csv","teacher_scores.csv","summary.json"))))
    print("Eight-corner signed effects:",len(measured),"maximum identity discrepancy:",max_error)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("predict","score"))
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--assets",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--engine",choices=("batched","indexed"),default="batched")
    args=parser.parse_args()
    if args.phase=="predict":
        if args.assets is None:
            parser.error("--assets is required for predict")
        predict(args)
    else:
        score_experiment(args.run.resolve())

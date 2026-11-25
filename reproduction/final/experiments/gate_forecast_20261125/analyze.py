"""从封存的逐例开关、得分和阻断记录生成模型级统计。"""
import argparse
from pathlib import Path

import numpy as np
from run import KEYS, case_key, js, put, read_csv, sha, verify, write_csv


BIT_COUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
ARMS = ("target", "random1", "random2")
LAYERS = ("first", "second", "both")
EPS = 1e-9


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def error_ratio(predicted, actual):
    return rms(predicted - actual) / rms(actual)


def agreement(predicted, actual):
    valid = np.abs(actual) > EPS
    hits = int(np.sum(np.sign(predicted[valid]) == np.sign(actual[valid])))
    return hits, int(valid.sum())


def keyed(path, patch=False):
    rows = read_csv(path)
    data = {}
    for row in rows:
        key = case_key(row)
        if patch:
            key += (row["arm"], row["layer"])
        data[key] = row
    if len(rows) != len(data):
        raise ValueError(f"Duplicate measurement in {path}")
    return data


def gate_counts(guess, true, empty):
    for label, predicted, observed in (
        ("flip", guess ^ empty, true ^ empty),
        ("open", guess & ~empty, true & ~empty),
        ("close", ~guess & empty, ~true & empty),
    ):
        tp = int(BIT_COUNT[predicted & observed].sum())
        total_predicted = int(BIT_COUNT[predicted].sum())
        total_observed = int(BIT_COUNT[observed].sum())
        yield dict(direction=label, true_positive=tp, predicted_count=total_predicted, actual_count=total_observed,
                   precision=tp / total_predicted if total_predicted else None,
                   recall=tp / total_observed if total_observed else None,
                   f1=2 * tp / (total_predicted + total_observed) if total_predicted + total_observed else None)


def selected_accuracy(indices, guess, true, empty):
    predicted, observed, baseline = [np.unpackbits(field, axis=-1, bitorder="little")
                                     for field in (guess, true, empty)]
    counts = {arm:[0, 0] for arm in ARMS}
    budget = indices.shape[-1]
    for i, row in enumerate(indices):
        flip = predicted[i] != baseline[i]
        open_count = None
        for arm, points in zip(ARMS, row):
            ids = points[points >= 0]
            if (len(np.unique(ids)) != len(ids) or len(ids) != min(budget, int(flip.sum()))
                    or not flip[ids].all()):
                raise ValueError("Incorrect candidate pool or position budget")
            opened = int(predicted[i, ids].sum())
            if open_count is None:
                open_count = opened
            elif opened != open_count:
                raise ValueError("Unmatched opening/closing counts")
            counts[arm][0] += int(np.sum(predicted[i, ids] == observed[i, ids]))
            counts[arm][1] += len(ids)
    return counts


def avg(rows, col):
    return float(np.mean([r[col] for r in rows]))


def contrast(old, new):
    delta = np.asarray(old) - np.asarray(new)
    wins = int(np.sum(delta > 0))
    return dict(mean_improvement=float(delta.mean()), improved_models=wins,
                material=bool(delta.mean() >= 0.05 and wins >= 4))


def main(path):
    prediction_seal = js(path / "prediction_seal.json")
    measurement_seal = js(path / "measurement_seal.json")
    verify(path, {**prediction_seal["files"], **measurement_seal["files"],
                  "protocol.json":prediction_seal["protocol"], "prediction_seal.json":measurement_seal["predictions"]})
    pred, actual = keyed(path / "predictions.csv"), keyed(path / "measurements.csv")
    proposed = keyed(path / "intervention_predictions.csv", True)
    measured = keyed(path / "interventions.csv", True)
    assert set(proposed) == set(measured) and len(proposed) == 7776
    with np.load(path / "forecast_sites.npz", allow_pickle=False) as archive:
        keys = [(int(s), float(a) / 100, int(c), int(m)) for s, a, c, m in archive["keys"]]
        guessed, voted, empty, selections = [archive[k] for k in ("predicted", "majority", "empty", "sites")]
        packed_keys = archive["keys"]
    with np.load(path / "observed_gates.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["keys"], packed_keys)
        truth = archive["true"]
    assert guessed.shape == voted.shape == empty.shape == truth.shape == (864, 2, 4096)
    assert selections.shape == (864, 3, 2, 64)
    assert len(keys) == len(pred) == 864 and set(keys) == set(pred) == set(actual)
    output_methods = [k for k in pred[keys[0]] if k not in {*KEYS, "recipient_image", "donor_image", "q_output"}]
    seeds = sorted({key[0] for key in keys})
    output_models, gate_models, patch_models, selection_models, joint_models = [], [], [], [], []
    summary = {}
    for alpha in (1.0, 0.5):
        by_strength = []
        for seed in seeds:
            indices = [i for i, key in enumerate(keys) if key[:2] == (seed, alpha)]
            sample = [keys[i] for i in indices]
            assert len(sample) == 72
            def values(table, column):
                return np.array([float(table[key][column]) for key in sample])
            y = values(actual, "y")
            scores = {name:error_ratio(values(pred, name), y) for name in output_methods}
            signs, count = agreement(values(pred, "predicted_4"), y)
            item = dict(seed=seed, strength=alpha, records=len(sample), output_rms=rms(y),
                        output_sign_correct=signs, output_sign_total=count, **scores)
            output_models.append(item)
            by_strength.append(item)
            for layer in (0, 1):
                correct, base = truth[indices, layer], empty[indices, layer]
                for name, field in (("predicted", guessed), ("majority", voted)):
                    stats = gate_counts(field[indices, layer], correct, base)
                    for row in stats:
                        gate_models.append(dict(seed=seed, strength=alpha, layer=LAYERS[layer], method=name, **row))
                counts = selected_accuracy(selections[indices, :, layer], guessed[indices, layer], correct, base)
                for arm, (hit, total) in counts.items():
                    selection_models.append(dict(seed=seed, strength=alpha, layer=LAYERS[layer], arm=arm,
                                                 hits=hit, total=total, precision=hit / total))
            for arm in ARMS:
                observed_by_layer = []
                predicted_by_layer = []
                for layer in LAYERS:
                    case_ids = [key + (arm, layer) for key in sample]
                    p = np.array([float(proposed[k]["predicted_change"]) for k in case_ids])
                    a = np.array([float(measured[k]["actual_change"]) for k in case_ids])
                    hits, total = agreement(p, a)
                    patch_models.append(dict(seed=seed, strength=alpha, arm=arm, layer=layer, actual_rms=rms(a),
                                             predicted_rms=rms(p), signed_error_rms=rms(p-a), error_ratio=error_ratio(p,a),
                                             correct_sign=hits, sign_total=total, near_zero=len(sample)-total))
                    observed_by_layer.append(a)
                    predicted_by_layer.append(p)
                first, second, both = observed_by_layer
                pf, ps, pb = predicted_by_layer
                remainder = both - first - second
                joint_models.append(dict(seed=seed, strength=alpha, arm=arm, opposite_layers=int(np.sum(first*second < 0)),
                                         net_rms=rms(both), nonadditive_rms=rms(remainder),
                                         predicted_nonadditive_rms=rms(pb-pf-ps), records=len(sample)))

        mean_scores = {name:avg(by_strength, name) for name in output_methods}
        comparisons = {name:contrast([row[name] for row in by_strength], [row["predicted_4"] for row in by_strength])
                       for name in ("fixed_4", "class_mean", "empty_masks_4", "majority_4", "live_4")}
        gate_summary = {}
        for layer in LAYERS[:2]:
            rows = [r for r in gate_models if r["strength"] == alpha and r["layer"] == layer and r["direction"] == "flip"]
            methods = {m:[r for r in rows if r["method"] == m] for m in ("predicted", "majority")}
            delta = [a["f1"]-b["f1"] for a,b in zip(methods["predicted"], methods["majority"])]
            gate_summary[layer] = {m:{col:avg(v,col) for col in ("precision","recall","f1")} for m,v in methods.items()}
            gate_summary[layer].update(mean_f1_gain=float(np.mean(delta)), improved_models=sum(v>0 for v in delta),
                                      passes=bool(np.mean(delta)>0 and sum(v>0 for v in delta)>=4))
        patch_summary = {}
        for layer in LAYERS:
            rows = [r for r in patch_models if r["strength"] == alpha and r["layer"] == layer]
            groups = {arm:[r for r in rows if r["arm"] == arm] for arm in ARMS}
            out = {}
            for arm, group in groups.items():
                out[arm] = dict(actual_rms=avg(group,"actual_rms"), error_ratio=avg(group,"error_ratio"),
                                below_one=sum(r["error_ratio"]<1 for r in group),
                                correct_sign=sum(r["correct_sign"] for r in group),
                                sign_total=sum(r["sign_total"] for r in group))
            delta = [t["actual_rms"] - (a["actual_rms"]+b["actual_rms"])/2
                     for t,a,b in zip(groups["target"],groups["random1"],groups["random2"])]
            out.update(localization_gain=float(np.mean(delta)), larger_models=sum(v>0 for v in delta),
                       localization_pass=bool(np.mean(delta)>0 and sum(v>0 for v in delta)>=4),
                       effect_pass=bool(out["target"]["error_ratio"]<=0.95 and out["target"]["below_one"]>=4))
            patch_summary[layer] = out
        selected = {layer:{arm:avg([r for r in selection_models if r["strength"]==alpha and
                          r["layer"]==layer and r["arm"]==arm],"precision") for arm in ARMS} for layer in LAYERS[:2]}
        joins = [r for r in joint_models if r["strength"]==alpha and r["arm"]=="target"]
        summary[str(alpha)] = dict(records=432, mean_output_error=mean_scores, comparisons=comparisons,
            output_sign_correct=sum(r["output_sign_correct"] for r in by_strength),
            output_sign_total=sum(r["output_sign_total"] for r in by_strength), gate_scores=gate_summary,
            selected_precision=selected, interventions=patch_summary,
            joint_nonadditive_rms=avg(joins,"nonadditive_rms"), opposite_layers=sum(r["opposite_layers"] for r in joins),
            decisions=dict(gate_localization=all(gate_summary[k]["passes"] for k in LAYERS[:2]),
                output=all(comparisons[k]["material"] for k in ("fixed_4","class_mean")),
                intervention=patch_summary["both"]["localization_pass"] and patch_summary["both"]["effect_pass"]))
    for name, rows in (("teacher_scores",output_models),("gate_scores",gate_models),("intervention_scores",patch_models),
                       ("selected_accuracy",selection_models),("layer_interaction",joint_models)):
        write_csv(path / (name+".csv"), rows)
    put(path / "summary.json", dict(measurement_seal=sha(path/"measurement_seal.json"), by_strength=summary))
    print("Scored", len(keys), "cases and", len(measured), "interventions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)

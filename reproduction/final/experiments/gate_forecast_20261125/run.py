"""三组开关实验的预测封存与真实干预。"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "output_bottleneck_20261110"))
from run_experiment import FIT, PROTOCOL, case_key, js, put, read_csv, sha, stamp, write_csv

PREDICTION_FILES = ("predictions.csv", "intervention_predictions.csv", "forecast_sites.npz")
MEASUREMENT_FILES = ("measurements.csv", "interventions.csv", "observed_gates.npz", "audit.json")
KEYS = ("seed", "strength", "case", "mask")


def checksums(root, files):
    return {name: sha(root / name) for name in files}


def location(assets):
    public = assets / "reproduction/held_condition"
    if public.is_dir():
        return public / "runs", public / "reference", assets / "source/scripts"
    return assets / "runs", assets / "runs/dsprites_rule_discrimination_20261031", assets / "scripts"


def verify(root, hashes):
    for name, expected in hashes.items():
        if sha(root / name) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")


def start_seal(out, protocol):
    seal = js(out / "prediction_seal.json")
    verify(out, {"protocol.json": seal["protocol"], **seal["files"]})
    if set(seal["files"]) != set(PREDICTION_FILES):
        raise ValueError("Incomplete forecast seal")
    for key in ("design_and_code", "shared_code", "inputs", "fits", "held_cases", "seeds", "prior_protocol", "prior_fit_seal"):
        if js(out / "protocol.json")[key] != protocol[key]:
            raise ValueError(f"Protocol changed: {key}")
    if (out / "measurement_seal.json").exists():
        raise FileExistsError("Measurement already complete")


def patch_rows(keys, values, name, arms, layers):
    return [dict(**key, arm=arm, layer=layer, **{name:float(values[i, a, j])})
            for i, key in enumerate(keys) for a, arm in enumerate(arms) for j, layer in enumerate(layers)]


def main(args):
    frozen, reference, source = location(args.assets.resolve())
    sys.path.insert(0, str(source))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import dsprites_teacher as teacher
    import torch
    import gates

    rule = gates.rule
    teacher.configure()
    torch.set_grad_enabled(False)
    out = args.out.resolve()
    prior_path = reference / "protocol.json"
    fit_path = reference / "fit_freeze.json"
    verify(reference, {"protocol.json": PROTOCOL, "fit_freeze.json": FIT})
    prior, sealed = js(prior_path), js(fit_path)
    if sealed["previous"] != PROTOCOL:
        raise ValueError("Unmatched fit seal")
    verify(frozen, {name.removeprefix("runs/"): digest for name, digest in prior["inputs"].items()})
    verify(reference, sealed["files"])
    with np.load(frozen / "dsprites_shape_donor_20261010/panel/data.npz", allow_pickle=False) as data:
        panel = {name:data[name] for name in ("images", "pairs", "labels", "factors", "official_ids")}
    split = rule.split_panel(panel)
    assert panel["images"].shape == (36, 64, 64)
    assert all(cases == prior["split"][part]["cases"] for part, cases in split.items())
    cases = split["test"]
    seeds = prior["seeds"]
    assert seeds == list(range(202609291, 202609297)) and len(cases) == 18
    code = {name:sha(HERE / name) for name in ("gates.py", "run.py", "DESIGN.md")}
    code["../output_bottleneck_20261110/run_experiment.py"] = sha(HERE.parent / "output_bottleneck_20261110/run_experiment.py")
    shared = [Path(name).name for name in prior["source"] if Path(name).name != "run_dsprites_triple_rule.py"]
    shared.extend(("external_experiment.py", "research_runtime.py"))
    protocol = dict(utc=stamp(), prior_protocol=PROTOCOL, prior_fit_seal=FIT, inputs=prior["inputs"],
                    fits=sealed["files"], held_cases=cases, seeds=seeds, design_and_code=code,
                    shared_code=checksums(source, shared), strengths=[1.0, 0.5], primary_budget=4,
                    position_budget_per_layer=gates.POINTS, rule="sign of order-two preactivation at each layer",
                    packed_layout="record, layer, bytes; little bit order; flattened (channel,y,x) in C order",
                    axes=dict(arms=gates.ARMS, layers=gates.LAYERS), software=dict(python=sys.version.split()[0],
                    torch=torch.__version__, numpy=np.__version__, device=args.device))
    predicting = args.phase == "predict"
    keys_array = []
    output_rows, intervention_rows = [], []
    pieces = {name:[] for name in (("predicted", "majority", "empty", "sites") if predicting else ("true",))}
    maxima = {}
    if predicting:
        out.mkdir(parents=True, exist_ok=False)
        put(out / "protocol.json", protocol)
    else:
        start_seal(out, protocol)
        predictions = {case_key(row):row for row in read_csv(out / "predictions.csv")}
        with np.load(out / "forecast_sites.npz", allow_pickle=False) as data:
            positions, frozen_keys = data["sites"], data["keys"]

    def compare(name, a, b):
        error = float((a - b).abs().max())
        if not np.isfinite(error) or error > 1e-8:
            raise ArithmeticError((name, error))
        maxima[name] = max(error, maxima.get(name, 0.0))

    for seed in seeds:
        net = teacher.DSpritesCNN()
        filename = frozen / f"dsprites_20260929/seed_{seed}/teacher.pt"
        net.load_state_dict(torch.load(filename, weights_only=True, map_location="cpu"))
        net = net.double().to(args.device).eval().requires_grad_(False)
        bank = rule.source_bank(net, panel, cases, args.device)
        fit = js(reference / f"fit/{seed}.json")
        for strength in (1.0, 0.5):
            for case in cases:
                ri, di = map(int, panel["pairs"][case])
                classes = tuple(map(int, panel["labels"][[ri, di]]))
                keys = [dict(seed=seed, strength=strength, case=case, mask=t) for t in rule.TRIPLES]
                previous_count = len(keys_array)
                # 强度在二进制表中编码为 100 和 50，所有文字表使用 1 和 0.5。
                ids = [(seed, int(strength * 100), case, t) for t in rule.TRIPLES]
                keys_array.extend(ids)
                if predicting:
                    results, primary, base, vote, second_q = gates.forecast(net, bank[ri], bank[di], classes, strength, fit)
                    chosen = gates.point_lists(net, primary, base, classes, seed, strength, case)
                    changes = gates.patch_effects(net, primary, base, classes, chosen, second_q)
                    pieces["predicted"].append(gates.packed(primary.gates))
                    pieces["majority"].append(gates.packed(vote))
                    pieces["empty"].append(np.repeat(gates.packed(base), len(keys), axis=0))
                    pieces["sites"].append(chosen)
                    effect_column = "predicted_change"
                else:
                    part = slice(previous_count, previous_count + len(keys))
                    if not np.array_equal(frozen_keys[part], ids):
                        raise ValueError("Unmatched intervention list")
                    all_masks = tuple(range(15))
                    mixed = rule.mixed_sources(bank[ri], bank[di], all_masks, strength)
                    middle = net.block2(mixed)
                    natural = gates.endpoint(net, middle, classes)
                    native = rule.tail_from_middle(net, middle)
                    native_score = native[:, classes[1]] - native[:, classes[0]]
                    compare("native_endpoint", native_score, natural.score)
                    true_endpoint = gates.endpoint(net, middle[list(rule.TRIPLES)], classes)
                    low_output = rule.order_two(native_score[list(rule.LOW)])
                    sealed_low = low_output.new_tensor([float(predictions[case_key(k)]["q_output"]) for k in keys])
                    compare("sealed_low_output", low_output, sealed_low)
                    results = dict(y=true_endpoint.score - low_output, endpoint_score=true_endpoint.score)
                    pieces["true"].append(gates.packed(true_endpoint.gates))
                    base = gates.Pair(*(field[:1] for field in natural.gates))
                    chosen = positions[part]
                    if args.measurement == "indexed":
                        changes = gates.reference_patches(net, true_endpoint, base, classes, chosen)
                    else:
                        changes = gates.patch_effects(net, true_endpoint, base, classes, chosen)
                    effect_column = "actual_change"
                arrays = {k:v.cpu().numpy() for k, v in results.items()}
                for i, key in enumerate(keys):
                    output_rows.append(dict(**key, recipient_image=ri, donor_image=di,
                                            **{col:float(v[i]) for col, v in arrays.items()}))
                intervention_rows.extend(patch_rows(keys, changes.cpu().numpy(), effect_column, gates.ARMS, gates.LAYERS))
        print(args.phase, seed, flush=True)

    assert len(output_rows) == len({case_key(row) for row in output_rows}) == 864
    assert len(intervention_rows) == 7776
    arrays = {name:np.concatenate(items) for name, items in pieces.items()}
    arrays["keys"] = np.asarray(keys_array, dtype=np.int32)
    if predicting:
        write_csv(out / "predictions.csv", output_rows)
        write_csv(out / "intervention_predictions.csv", intervention_rows)
        np.savez_compressed(out / "forecast_sites.npz", **arrays)
        put(out / "prediction_seal.json", dict(utc=stamp(), protocol=sha(out / "protocol.json"),
            cases=len(output_rows), interventions=len(intervention_rows), files=checksums(out, PREDICTION_FILES)))
    else:
        write_csv(out / "measurements.csv", output_rows)
        write_csv(out / "interventions.csv", intervention_rows)
        np.savez_compressed(out / "observed_gates.npz", **arrays)
        put(out / "audit.json", dict(status="PASS", device=args.device, measurement=args.measurement,
                                     numerical_maxima=maxima, cases=len(output_rows), interventions=len(intervention_rows)))
        put(out / "measurement_seal.json", dict(utc=stamp(), predictions=sha(out / "prediction_seal.json"),
                                                files=checksums(out, MEASUREMENT_FILES)))
    print(args.phase, "sealed", len(output_rows), len(intervention_rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "measure"))
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--measurement", choices=("batched", "indexed"), default="batched")
    main(parser.parse_args())

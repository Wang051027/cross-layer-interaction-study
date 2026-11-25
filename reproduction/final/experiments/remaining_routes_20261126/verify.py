"""逐项比较独立复跑的开发排名、留出干预、空间位置和通道连接。"""
import argparse
from pathlib import Path

import numpy as np

from experiment import js, put, read_csv, sha, verify


TABLES = ("development", "cases", "group_interventions", "spatial_extremes", "teachers",
          "components", "budget_scores", "channel_scores", "group_routes", "link_scores")
IDENTIFIERS = {"seed", "strength", "case", "mask", "rank", "channel", "y", "x", "direction",
               "gate_group", "arm", "budget", "layer", "stage", "term", "development_rank",
               "first_channel", "last_channel", "last_in_frozen_four"}
TOLERANCE = 1e-8


def compare_table(left, right, name):
    a, b = (read_csv(folder / (name + ".csv")) for folder in (left, right))
    if len(a) != len(b) or a[0].keys() != b[0].keys():
        raise ValueError(f"Table layout changed: {name}")
    largest = 0.0
    for i, (old, new) in enumerate(zip(a, b)):
        for field, expected in old.items():
            actual = new[field]
            if actual == expected:
                continue
            if field in IDENTIFIERS or expected == "" or actual == "":
                raise ValueError((name, i, field, expected, actual))
            error = abs(float(expected) - float(actual))
            if not np.isfinite(error) or error > TOLERANCE:
                raise ArithmeticError((name, i, field, error))
            largest = max(largest, error)
    return {"rows": len(a), "max_abs": largest}


def compare_values(expected, actual):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError("Summary fields differ")
        return max((compare_values(expected[k], actual[k]) for k in expected), default=0.)
    if isinstance(expected, (bool, int)):
        if expected != actual:
            raise ValueError((expected, actual))
        return 0.
    error = abs(expected - actual)
    if not np.isfinite(error) or error > TOLERANCE:
        raise ArithmeticError((expected, actual))
    return error


def check_run(folder):
    selected = js(folder / "selection_seal.json")
    measured = js(folder / "measurement_seal.json")
    verify(folder, selected["files"])
    verify(folder, {"selection_seal.json": measured["selection_seal"], **measured["files"]})
    result = js(folder / "checks.json")
    summary = js(folder / "summary.json")
    if result["status"] != "PASS" or summary["measurement_seal"] != sha(folder / "measurement_seal.json"):
        raise ValueError("Unverified measurements")
    return {"device": result["device"], "engine": result["engine"],
            "measurement_seal": summary["measurement_seal"]}


def main(reference, replay, out):
    if reference.resolve() == replay.resolve():
        raise ValueError("A separate replay directory is required")
    ref, copy = check_run(reference), check_run(replay)
    if js(reference / "rankings.json") != js(replay / "rankings.json"):
        raise ValueError("Development or random rankings differ")
    tables = {name: compare_table(reference, replay, name) for name in TABLES}
    with np.load(reference / "channels.npz", allow_pickle=False) as a, np.load(replay / "channels.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files) or not np.array_equal(a["keys"], b["keys"]):
            raise ValueError("Channel archive keys differ")
        arrays = {}
        for key in a.files:
            if key == "keys":
                continue
            if a[key].shape != b[key].shape:
                raise ValueError(key)
            np.testing.assert_allclose(a[key], b[key], atol=TOLERANCE, rtol=0)
            arrays[key] = float(np.max(np.abs(a[key] - b[key])))
    summaries = [js(path / "summary.json")["strengths"] for path in (reference, replay)]
    summary_gap = compare_values(*summaries)
    maximum = max(summary_gap, max(arrays.values()), max(item["max_abs"] for item in tables.values()))
    put(out, dict(status="PASS", reference=ref, replay=copy, tolerance=TOLERANCE, max_abs=maximum,
                  identical_rankings=True, tables=tables, arrays=arrays, summary_max_abs=summary_gap))
    print(f"Replay matched across {len(tables)} tables; maximum absolute difference {maximum:.3g}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.replay, args.out)

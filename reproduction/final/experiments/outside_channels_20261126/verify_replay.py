"""逐例对照独立复跑的名单、八状态差分、三组端点及汇总。"""
import argparse
from pathlib import Path

import numpy as np

from interventions import js, put, sha, verify
from experiment import read_csv


TABLES = ("development_channels", "development_pairs", "models", "channel_scores", "set_scores",
          "pair_scores", "candidate_cases", "cumulative")
IDENTIFIERS = {"seed", "strength", "rank", "dev_rank", "case", "mask", "channel", "first", "second",
               "channels", "label", "candidate"}
ATOL = 1e-8


def compare_table(name, first, second):
    a, b = [read_csv(folder / (name + ".csv")) for folder in (first, second)]
    if len(a) != len(b) or a[0].keys() != b[0].keys():
        raise ValueError(name)
    worst = 0.
    for n, (one, two) in enumerate(zip(a, b)):
        for k, expected in one.items():
            if expected == two[k]:
                continue
            if k in IDENTIFIERS:
                raise ValueError((name, n, k, expected, two[k]))
            gap = abs(float(expected)-float(two[k]))
            if not np.isfinite(gap) or gap > ATOL:
                raise ArithmeticError((name, n, k, expected, two[k]))
            worst = max(worst, gap)
    return dict(rows=len(a), max_abs=worst)


def compare_tree(a, b):
    if a is None and b is None:
        return 0.
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Summary schema changed")
        return max(compare_tree(a[k], b[k]) for k in a)
    if isinstance(a, (int, bool)) and a != b:
        raise ValueError((a, b))
    delta = abs(float(a)-float(b))
    if not np.isfinite(delta) or delta > ATOL:
        raise ArithmeticError((a, b))
    return delta


def verify_folder(folder):
    measurement = js(folder / "measurement_seal.json")
    verify(folder, {"selection_seal.json": measurement["selection_seal"], **measurement["files"]})
    verify(folder, js(folder / "selection_seal.json")["files"])
    seal = sha(folder / "measurement_seal.json")
    checks = js(folder / "checks.json")
    reconstruction = js(folder / "reconstruction.json")
    if (js(folder / "summary.json")["measurement_seal"] != seal or checks["status"] != "PASS"
        or reconstruction["measurements"] != seal or reconstruction["status"] != "PASS"):
        raise ValueError("Unverified measurements")
    return dict(device=checks["device"], full_map_check=checks["full_map_check"], seal=seal)


def main(reference, replay, out):
    if reference.resolve() == replay.resolve():
        raise ValueError("Select two separate runs")
    first, second = verify_folder(reference), verify_folder(replay)
    selection = js(reference / "selection.json")
    if selection != js(replay / "selection.json"):
        raise ValueError("Independently fitted channel ranks or pair choices differ")
    raw = {}
    for seed in selection:
        with np.load(reference/f"model_{seed}.npz", allow_pickle=False) as a, np.load(replay/f"model_{seed}.npz", allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(seed)
            for k in ("keys", "group_names", "outside", "channel_pairs"):
                np.testing.assert_array_equal(a[k], b[k])
            for k in ("singles", "pairs", "groups"):
                x, y = a[k], b[k]
                np.testing.assert_allclose(x, y, rtol=0, atol=ATOL)
                raw[k] = max(raw.get(k, 0.), float(np.max(np.abs(x - y))))
    tables = {name: compare_table(name, reference, replay) for name in TABLES}
    summary_error = compare_tree(js(reference/"summary.json")["strengths"], js(replay/"summary.json")["strengths"])
    largest = max(max(raw.values()), max(item["max_abs"] for item in tables.values()), summary_error)
    put(out, dict(status="PASS", tolerance=ATOL, reference=first, replay=second, identical_selection=True,
                  raw_max_abs=raw, tables=tables, summary_max_abs=summary_error, maximum=largest))
    print(f"Six independently fitted selections matched. Maximum absolute difference {largest:.3g}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.replay, args.out)

"""封存开发名单，逐例重算末层组外通道的真实分类头输出。"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from interventions import (HERE, MEASUREMENT, SELECTION, Inputs, block_channels, checksums, choose_model,
                           final_maps, groups, js, make_masks, previous, put, routed_terms, sha, stamp,
                           verify, write_csv)


def select(inputs, destination, source):
    destination.mkdir(parents=True, exist_ok=False)
    old = js(source / "rankings.json")
    protocol = dict(utc=stamp(), device=inputs.device, previous_selection=SELECTION, previous_measurement=MEASUREMENT,
                    inputs=inputs.spec, code=checksums(HERE, ("DESIGN.md", "interventions.py", "run.py")))
    put(destination / "protocol.json", protocol)
    selection, channel_rows, pair_rows = {}, [], []
    for seed in inputs.seeds:
        upstream = old[str(seed)]["first"]["target"][:4]
        last = old[str(seed)]["last"]["target"][:4]
        values = [routed_terms(state, final_maps(state, upstream))
                  for _, state, _, _ in inputs.samples(seed, "development")]
        if len(values) != 72:
            raise ValueError("Unexpected number of development cases")
        rule, channels, pairs = choose_model(seed, upstream, last, np.stack(values))
        selection[str(seed)] = rule
        channel_rows.extend(channels)
        pair_rows.extend(pairs)
    put(destination / "selection.json", selection)
    write_csv(destination / "development_channels.csv", channel_rows)
    write_csv(destination / "development_pairs.csv", pair_rows)
    files = ("protocol.json", "selection.json", "development_channels.csv", "development_pairs.csv")
    put(destination / "selection_seal.json", dict(utc=stamp(), files=checksums(destination, files)))
    print("Selection sealed for six models.", flush=True)


def measure(inputs, destination, source, spatial):
    if (destination / "measurement_seal.json").exists():
        raise FileExistsError("Measurement already sealed")
    verify(destination, js(destination / "selection_seal.json")["files"])
    protocol = js(destination / "protocol.json")
    if inputs.spec != protocol["inputs"] or protocol["code"] != checksums(HERE, protocol["code"]):
        raise ValueError("Inputs or calculation code changed after selection")
    selection = js(destination / "selection.json")
    verify(source, js(source / "measurement_seal.json")["files"])
    with np.load(source / "channels.npz", allow_pickle=False) as old:
        lookup = {tuple(key): i for i, key in enumerate(old["keys"])}
        old_terminal = old["terminal"]
        old_transfer = old["first4_to_last"]

    errors = {}
    def compare(name, result, expected):
        gap = float(np.max(np.abs(np.asarray(result) - np.asarray(expected))))
        if not np.isfinite(gap) or gap > 1e-8:
            raise ArithmeticError((name, gap))
        errors[name] = max(errors.get(name, 0.), gap)

    files, seen = [], set()
    for seed in inputs.seeds:
        spec = selection[str(seed)]
        outside = spec["outside"]
        named = groups(spec)
        singles = [[j] for j in outside]
        channel_pairs = list(combinations(outside, 2))
        pair_indices = np.array(list(combinations(range(28), 2)))
        masks = make_masks([*named.values(), *singles, *channel_pairs], inputs.device)
        group_count = len(named)
        direct_count = group_count + 28 if spatial else 0
        group_names = list(named)
        indices = {name: group_names.index(name) for name in ("base", "old4", "outside28", "all32")}
        keys, group_values, single_values, pair_values = [], [], [], []
        for key, state, _, _ in inputs.samples(seed, "test"):
            record = (seed, round(100*key["strength"]), key["case"], key["mask"])
            if record in seen:
                raise ValueError(record)
            seen.add(record)
            maps = final_maps(state, spec["upstream"])
            actual, spatial_error = block_channels(state, maps, masks, direct_count)
            group = actual[:group_count]
            single = actual[group_count:group_count + 28]
            pair = actual[group_count + 28:]
            base = group[indices["base"]]
            # 对三阶和端点两个读数、两个上游背景，分别验算末层加性。
            single_change = base[None] - single
            expected_pairs = single_change[pair_indices[:, 0]] + single_change[pair_indices[:, 1]]
            compare("pair_minus_singles", base[None] - pair, expected_pairs)
            compare("all_outside_minus_singles", base - group[indices["outside28"]], single_change.sum(0))
            example = lookup[record]
            route = single_change[:, 0, 0] - single_change[:, 1, 0]
            compare("archived_individual_routing", route, old_transfer[example, outside])
            compare("archived_terminal", single_change[:, 0, 0], old_terminal[example, outside])
            compare("archived_output", base[0, 0], old_terminal[example].sum())
            compare("archived_upstream", base[0, 0] - base[1, 0], old_transfer[example].sum())
            old_fixed = group[indices["old4"]]
            compare("outside_by_old_four_block", old_fixed[0, 0] - old_fixed[1, 0], route.sum())
            compare("all32_third_zero", group[indices["all32"], :, 0], 0.)
            if spatial:
                compare("before_vs_after_pooling", spatial_error, 0.)
            keys.append(record)
            group_values.append(group)
            single_values.append(single)
            pair_values.append(pair)
        if len(keys) != 144:
            raise ValueError(seed)
        file = f"model_{seed}.npz"
        np.savez_compressed(destination / file, keys=np.asarray(keys), groups=np.stack(group_values),
                            singles=np.stack(single_values), pairs=np.stack(pair_values),
                            group_names=np.array(group_names), outside=np.array(outside), channel_pairs=np.array(channel_pairs))
        files.append(file)
        print("Measured", seed, flush=True)
    if seen != set(lookup):
        raise ValueError("Mismatch in held conditions")
    put(destination / "checks.json", dict(status="PASS", cases=len(seen), singles=len(seen)*28,
        pairs=len(seen)*378, named_sets=len(seen)*len(named), device=inputs.device, full_map_check=spatial, max_abs=errors))
    put(destination / "measurement_seal.json", dict(utc=stamp(), selection_seal=sha(destination / "selection_seal.json"),
                                                      files=checksums(destination, [*files, "checks.json"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "measure", "all"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda","cpu"), default="cpu")
    parser.add_argument("--pooled-only", action="store_true", help="用池化后阻断进行独立复算")
    args = parser.parse_args()
    inputs = Inputs(args.assets.resolve(), args.device)
    folder = previous(inputs)
    if args.phase in ("all", "select"):
        select(inputs, args.out.resolve(), folder)
    if args.phase in ("all", "measure"):
        measure(inputs, args.out.resolve(), folder, not args.pooled_only)

"""按模型汇总八状态阻断，分别评价上游传递和最终得分。"""
import argparse
from pathlib import Path

import numpy as np

from interventions import BUDGETS, EPS, groups, js, put, sha, verify, write_csv


def rms(x, axis=0):
    return np.sqrt(np.mean(x**2, axis=axis))


def changed(before, after):
    """跨过零点的记录也按绝对幅度比较。"""
    difference = np.abs(after) - np.abs(before)
    return int(np.sum(difference < -EPS)), int(np.sum(difference > EPS))


def metric_rows(base, outcomes, outside, labels):
    # 数组轴为案例、阻断配置、上游背景、三阶差分或三组全选端点。
    y, yu = base[:, 0, 0], base[:, 1, 0]
    f = y - yu
    rows = []
    for col, label in enumerate(labels):
        result = outcomes[:, col]
        after_y = result[:, 0, 0]
        after_f = after_y - result[:, 1, 0]
        output, transmitted = y - after_y, f - after_f
        y_down, y_up = changed(y, after_y)
        f_down, f_up = changed(f, after_f)
        endpoint_change = base[:, 0, 1] - result[:, 0, 1]
        rows.append(dict(label=label, routed_rms=rms(transmitted), projection_F=transmitted@f/(f@f),
            share_outside=transmitted@f/(outside@f), projection_O=transmitted@outside/(outside@outside),
            output_change_rms=rms(output), projection_Y=output@y/(y@y), output_remaining_ratio=rms(after_y)/rms(y),
            upstream_remaining_ratio=rms(after_f)/rms(f), output_smaller=y_down, output_larger=y_up,
            upstream_smaller=f_down, upstream_larger=f_up, endpoint_change_rms=rms(endpoint_change)))
    return rows


def pair_metrics(seed, strength, base, single, joint, pairs, outside, selected, keys):
    y = base[:, 0, 0]
    f = y - base[:, 1, 0]
    i, j = np.array([(outside.index(a), outside.index(b)) for a, b in pairs]).T
    change = y[:, None] - single[:, :, 0, 0]
    path = f[:, None] - (single[:, :, 0, 0] - single[:, :, 1, 0])
    pair_change = y[:, None] - joint[:, :, 0, 0]
    pair_path = f[:, None] - (joint[:, :, 0, 0] - joint[:, :, 1, 0])
    rows = [dict(seed=seed, strength=strength, first=a, second=b) for a, b in pairs]
    for prefix, values, combined in (("route", path, pair_path), ("output", change, pair_change)):
        a, b = values[:, i], values[:, j]
        both = (np.abs(a) > EPS) & (np.abs(b) > EPS)
        same, opposite = both & (a*b > 0), both & (a*b < 0)
        overlap = np.minimum(np.abs(a), np.abs(b))
        magnitude = (np.abs(a) + np.abs(b)).sum(axis=0)
        loss = np.divide(magnitude - np.abs(combined).sum(axis=0), magnitude,
                         out=np.zeros_like(magnitude), where=magnitude > EPS)
        data = dict(same=same.sum(0), opposite=opposite.sum(0), same_overlap=(same*overlap).mean(0),
                    opposite_overlap=(opposite*overlap).mean(0), cancelled_fraction=loss, pair_rms=rms(combined))
        for k, row in enumerate(rows):
            row.update({prefix+"_"+name: series[k] for name, series in data.items()})
    examples = []
    for k, (a, b) in enumerate(pairs):
        after = joint[:, k, 0, 0]
        rows[k]["output_remaining_ratio"] = rms(after)/rms(y)
        rows[k]["output_smaller"], rows[k]["output_larger"] = changed(y, after)
        for name in ("reinforce_pair", "cancel_pair"):
            if [a, b] != selected[name]:
                continue
            for n, (_, _, case, mask) in enumerate(keys):
                examples.append(dict(seed=seed, strength=strength, case=case, mask=mask, candidate=name,
                    first=a, second=b, y=y[n], y_first=single[n, i[k], 0, 0], y_second=single[n, j[k], 0, 0],
                    y_both=after[n], upstream_effect=f[n], route_first=path[n, i[k]], route_second=path[n, j[k]],
                    route_both=pair_path[n, k], output_first=change[n, i[k]], output_second=change[n, j[k]],
                    output_both=pair_change[n, k], endpoint=base[n, 0, 1], endpoint_both=joint[n, k, 0, 1]))
    return rows, examples


def summarize(run):
    sealed = js(run / "measurement_seal.json")
    verify(run, {"selection_seal.json": sealed["selection_seal"], **sealed["files"]})
    verify(run, js(run / "selection_seal.json")["files"])
    selected = js(run / "selection.json")
    teachers, channels, group_rows, pairs_table, examples, curves = [], [], [], [], [], []
    errors = {"paired_route": 0., "named_route": 0., "named_readouts": 0.}
    for seed_key, frozen in selected.items():
        seed = int(seed_key)
        with np.load(run / f"model_{seed}.npz", allow_pickle=False) as file:
            keys = file["keys"]
            named, outside = file["group_names"].tolist(), file["outside"].tolist()
            groups_data, singles_data, pairs_data = file["groups"], file["singles"], file["pairs"]
            pairs = [tuple(v) for v in file["channel_pairs"].tolist()]
        if named != list(groups(frozen)) or outside != frozen["outside"] or len(pairs) != 378:
            raise ValueError("Blocking sets do not match the sealed selection")
        for strength in (1., .5):
            ids = np.flatnonzero(keys[:, 1] == round(100*strength))
            group, single, joint = groups_data[ids], singles_data[ids], pairs_data[ids]
            base = group[:, named.index("base")]
            y = base[:, 0, 0]
            f = y - base[:, 1, 0]
            old4 = group[:, named.index("old4")]
            outer = old4[:, 0, 0] - old4[:, 1, 0]
            t = f[:, None] - (single[:, :, 0, 0] - single[:, :, 1, 0])
            np.testing.assert_allclose(t.sum(1), outer, rtol=0, atol=1e-8)
            common = dict(seed=seed, strength=strength)
            individual = metric_rows(base, single, outer, outside)
            for row in individual:
                ch = row.pop("label")
                channels.append(dict(**common, channel=ch, dev_rank=frozen["order"].index(ch)+1, **row))
            channel_change = base[:, None] - single
            pair_indices = np.array([(outside.index(a), outside.index(b)) for a, b in pairs])
            paired_route = f[:, None] - (joint[:, :, 0, 0] - joint[:, :, 1, 0])
            pair_gap = paired_route - t[:, pair_indices[:, 0]] - t[:, pair_indices[:, 1]]
            errors["paired_route"] = max(errors["paired_route"], float(np.abs(pair_gap).max()))
            for idx, row in enumerate(metric_rows(base, group, outer, named)):
                members = groups(frozen)[row["label"]]
                includes_old = bool(set(members) & set(frozen["old_last"]))
                picked = [outside.index(ch) for ch in members if ch in outside]
                expected = channel_change[:, picked].sum(axis=1)
                if includes_old:
                    if not set(frozen["old_last"]).issubset(members):
                        raise ValueError("Expected the entire original four-channel group")
                    expected += base - old4
                    row["share_outside"] = None
                gap = base - group[:, idx] - expected
                errors["named_readouts"] = max(errors["named_readouts"], float(np.abs(gap).max()))
                errors["named_route"] = max(errors["named_route"], float(np.abs(gap[:, 0, 0]-gap[:, 1, 0]).max()))
                group_rows.append(dict(**common, **row))
            order = [outside.index(j) for j in frozen["order"]]
            cumulative = np.cumsum([individual[k]["share_outside"] for k in order])
            for k, share in enumerate(cumulative, 1):
                curves.append(dict(**common, channels=k, cumulative_share=share, random_expectation=k/28))
            np.testing.assert_allclose(cumulative[-1], 1., rtol=0, atol=1e-8)
            teachers.append(dict(**common, n=len(ids), output_rms=rms(y), upstream_rms=rms(f), outside_rms=rms(outer),
                outside_projection_F=outer@f/(f@f), k50=int(np.flatnonzero(cumulative>=.5)[0]+1),
                k80=int(np.flatnonzero(cumulative>=.8)[0]+1), negative_dev=len(frozen["negative"]),
                negative_held=sum(row["projection_F"] < -EPS for row in individual),
                outside_channel_cancellation=1-np.abs(outer).sum()/np.abs(t).sum(),
                cases_opposing=int(np.sum((t.min(1)<-EPS)&(t.max(1)>EPS)))))
            new_pairs, new_examples = pair_metrics(seed, strength, base, single, joint, pairs, outside, frozen, keys[ids])
            pairs_table.extend(new_pairs)
            examples.extend(new_examples)
    for name, data in (("models",teachers),("channel_scores",channels),("set_scores",group_rows),("pair_scores",pairs_table),
                       ("candidate_cases",examples),("cumulative",curves)):
        write_csv(run / (name+".csv"), data)
    result = {}
    for strength in (1., .5):
        items = [r for r in teachers if r["strength"] == strength]
        sets = [r for r in group_rows if r["strength"] == strength]
        avg = lambda rows, field: float(np.mean([r[field] for r in rows]))
        by_group = {}
        for label in dict.fromkeys(r["label"] for r in sets):
            records = [r for r in sets if r["label"] == label]
            counts = ("output_smaller","output_larger","upstream_smaller","upstream_larger")
            by_group[label] = {key:(avg(records, key) if records[0][key] is not None else None)
                               for key in records[0] if key not in ("label","seed","strength",*counts)}
            by_group[label].update({key:sum(r[key] for r in records) for key in counts})
        primary = [r for r in sets if r["label"]=="top4"]
        wins = sum(r["share_outside"]>4/28 for r in primary)
        lift = avg(primary,"share_outside")-4/28
        candidate_stats = {}
        for label in ("reinforce_pair", "cancel_pair"):
            rows = [r for r in pairs_table if r["strength"]==strength and
                    [r["first"],r["second"]]==selected[str(r["seed"])][label]]
            candidate_stats[label] = {f:sum(int(r[f]) for r in rows) for f in
                ("route_same","route_opposite","output_same","output_opposite")}
            for f in ("route_same_overlap","route_opposite_overlap","route_cancelled_fraction","route_pair_rms",
                      "output_same_overlap","output_opposite_overlap","output_cancelled_fraction","output_pair_rms"):
                candidate_stats[label][f] = avg(rows, f)
        result[str(strength)] = dict(records=sum(r["n"] for r in items), projection_outside=avg(items,"outside_projection_F"),
            cancellation=avg(items,"outside_channel_cancellation"), cases_with_both_signs=sum(r["cases_opposing"] for r in items),
            sets=by_group, candidates=candidate_stats, primary=dict(exact_random_mean=4/28, wins=int(wins),
                                                                   lift=lift, passes=bool(wins>=4 and lift>=.05)))
    if not all(np.isfinite(v) and v < 1e-8 for v in errors.values()):
        raise ArithmeticError(errors)
    put(run / "reconstruction.json", dict(status="PASS", max_abs=errors, measurements=sha(run/"measurement_seal.json")))
    put(run / "summary.json", dict(measurement_seal=sha(run/"measurement_seal.json"), strengths=result))
    print(f"Scored {len(channels)} channel, {len(pairs_table)} channel-pair and {len(group_rows)} group records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run)

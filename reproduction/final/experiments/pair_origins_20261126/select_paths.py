"""用模型 291 的开发条件确定上游通道、开关坐标和一个可检验的方向阈值。"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from mechanism import (HERE, PAIRS, SEED, PREVIOUS_MEASURE, Inputs, balance, candidate_readouts, checksums, features,
                       label, meta, middle_block, put, selected_tail, signs, site_scores,
                       source, stamp, write_csv)
from paths import third


def random_sets(channels, salt):
    pool = sorted(set(range(32))-set(channels))
    rng = np.random.default_rng([20261126, 291, salt])
    draws = rng.choice(pool, 4, replace=False).tolist()
    return [sorted(draws[:2]), sorted(draws[2:])]


def screen(inputs, selection, stage, order, log):
    candidates = {}
    for pair in PAIRS:
        spec = selection[label(pair)]
        ids = list(range(32)) if order == 1 else spec[stage+"_pool"]
        candidates[pair] = list(combinations(sorted(ids),order))
    results = {p:[] for p in PAIRS}
    for _, s, _, _ in inputs.samples(SEED, "development"):
        if stage=="H":
            sets = sorted(set(candidates[PAIRS[0]]) | set(candidates[PAIRS[1]]))
            values = candidate_readouts(s, sets, [j for p in PAIRS for j in p])
        for i, pair in enumerate(PAIRS):
            base = s.phi[list(pair)].cpu().numpy()
            if stage=="H":
                new = values[[sets.index(c) for c in candidates[pair]], 2*i:2*i+2]
                b0, b1 = balance(base), balance(new)
                flip = signs(b0)*signs(b1) == -1
                score = np.column_stack((np.where(flip, np.minimum(abs(b0),abs(b1)), 0.), abs(b0-b1)))
            else:
                altered = middle_block(s, selection[label(pair)]["H"])
                _, _, post = selected_tail(s,altered.act,altered.skip,list(pair))
                old = (third(post).mean((-2,-1))*s.weights[list(pair)]).cpu().numpy()
                new = candidate_readouts(s, candidates[pair], list(pair), altered=altered)
                gain = ((old-base)**2).sum() - ((new-base)**2).sum(1)
                score = np.column_stack((gain,np.zeros_like(gain)))
            results[pair].append(score)
    for i, pair in enumerate(PAIRS):
        if len(results[pair]) != 72:
            raise ValueError("Development split changed")
        spec, sets = selection[label(pair)], candidates[pair]
        scores = np.mean(results[pair],axis=0)
        ranks = sorted(range(len(sets)), key=lambda j:(-round(scores[j,0],10),-round(scores[j,1],10),sets[j]))
        for rank, j in enumerate(ranks,1):
            log.append(dict(pair=label(pair),stage=stage,count=order,rank=rank,channels="+".join(map(str,sets[j])),
                            primary=scores[j,0],secondary=scores[j,1]))
        if order==1:
            spec[stage+"_pool"] = [sets[j][0] for j in ranks[:8]]
        else:
            spec[stage] = list(sets[ranks[0]])
            spec[stage+"_random"] = random_sets(spec[stage], (1 if stage=="H" else 7)+i)
    print("Development",stage,order,flush=True)


def fixed_sites(total, counts, channels, salt):
    values = total.reshape(-1)
    frequency = counts.reshape(-1)
    ids = np.arange(len(values))
    order = np.lexsort((ids,-np.round(values/72,12)))
    chosen = list(map(int,order[:8]))
    area = total.shape[1]*total.shape[2]
    controls = []
    for replica in (1,2):
        rng = np.random.default_rng([20261126,291,salt,replica])
        tie = rng.permutation(len(values))
        selected = []
        for idx in chosen:
            valid = (ids//area == idx//area) & ~np.isin(ids, chosen+selected)
            pool = ids[valid]
            rank = np.lexsort((tie[pool],np.abs(frequency[pool]-frequency[idx])))
            selected.append(int(pool[rank[0]]))
        controls.append(selected)
    def unpack(seq):
        result=[]
        for idx in seq:
            c,y,x = np.unravel_index(idx,total.shape)
            result.append([channels[c],int(y),int(x)])
        return result
    return dict(target=unpack(chosen), random1=unpack(controls[0]),random2=unpack(controls[1]),
                score=[round(float(values[i]/72),12) for i in chosen], counts=[int(frequency[i]) for i in chosen])


def balanced(y, predicted):
    return np.mean([np.mean(predicted[y==k]==k) for k in (-1,1) if np.any(y==k)])


def majority(y):
    return 1 if np.sum(y>0)>=np.sum(y<0) else -1


def fit_threshold(matrix, y):
    valid = y!=0
    y = y[valid]
    best = dict(feature="constant",cut=0.,left=majority(y),right=majority(y),dev_balanced=round(float(balanced(y,np.full(len(y),majority(y)))),10))
    for name in matrix[0]:
        x = np.array([row[name] for row in matrix])[valid]
        unique = np.unique(x)
        for value in (unique[:-1]+unique[1:])/2:
            cut = round(float(value),12)
            left = x<=cut
            if min(left.sum(),(~left).sum())<8:
                continue
            for direction in (1,-1):
                pred = np.where(left,direction,-direction)
                metric = round(float(balanced(y,pred)),10)
                if metric>best["dev_balanced"]:
                    best = dict(feature=name,cut=cut,left=direction,right=-direction,dev_balanced=metric)
    return best


def main(root, out, device):
    inputs = Inputs(root,device)
    source(inputs)
    out.mkdir(parents=True, exist_ok=False)
    put(out/"protocol.json",dict(utc=stamp(),seed=SEED,prior=PREVIOUS_MEASURE,inputs=inputs.spec,device=device,
        code=checksums(HERE,["DESIGN.md","mechanism.py","select_paths.py","measure.py"])))
    selection = {label(p):{} for p in PAIRS}
    log = []
    for stage in ("H","K"):
        for n in (1,2):
            screen(inputs,selection,stage,n,log)
    write_csv(out/"development_channels.csv",log)

    accumulated = {}
    for _, s, _, _ in inputs.samples(SEED,"development"):
        for pair in PAIRS:
            spec = selection[label(pair)]
            altered = middle_block(s,spec["H"])
            values = site_scores(s,altered,pair,spec["K"])
            for stage,(scores,counts) in zip(("first","last"),values):
                key = pair,stage
                if key not in accumulated:
                    accumulated[key] = [scores.copy(),counts.copy()]
                else:
                    accumulated[key][0] += scores
                    accumulated[key][1] += counts
    for i,pair in enumerate(PAIRS):
        spec = selection[label(pair)]
        spec["sites"] = {stage:fixed_sites(*accumulated[pair,stage],channels,20+2*i+j)
                         for j,(stage,channels) in enumerate((("first",spec["K"]),("last",list(pair))))}
    dev = []
    feature_sets = {p:[] for p in PAIRS}
    for key,s,_,_ in inputs.samples(SEED,"development"):
        metadata = meta(inputs,key)
        for pair in PAIRS:
            spec = selection[label(pair)]
            x = features(s,pair,spec)
            a,b = s.phi[list(pair)].cpu().numpy()
            dev.append(dict(**metadata,pair=label(pair),a=a,b=b,relation=int(signs(balance(np.array([a,b]))))))
            feature_sets[pair].append(x)
    for pair in PAIRS:
        spec = selection[label(pair)]
        subset = [r for r in dev if r["pair"]==label(pair)]
        y = np.array([r["relation"] for r in subset])
        spec["threshold"] = fit_threshold(feature_sets[pair], y)
        spec["majority"] = majority(y)
        spec["class_mask"] = {f"{c}_{m}":majority(np.array([r["relation"] for r in subset if (r["recipient"],r["mask"])==(c,m)]))
                              for c in (0,1,2) for m in (7,11,13,14)}
    write_csv(out/"development_conditions.csv",dev)
    put(out/"selection.json",selection)
    put(out/"selection_seal.json",dict(utc=stamp(),files=checksums(out,["protocol.json","development_channels.csv",
        "development_conditions.csv","selection.json"])))
    print("Channel pairs, gate coordinates and thresholds sealed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    main(args.assets.resolve(),args.out.resolve(),args.device)

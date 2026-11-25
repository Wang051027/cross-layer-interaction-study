"""把支路、开关和通道的逐例结果按模型等权汇总。"""
import argparse
from pathlib import Path

import numpy as np

from experiment import ARMS, case_key, js, put, read_csv, sha, verify, write_csv


PARTS = ("conv_only","skip_only","branch_mixed","gate_affine","gate_first","gate_last","gate_mixed",
         "first_only_free","last_only_free","site_locked","site_always_on","site_switching","site_always_off")
STAGES = ("first","last","both")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def projection(piece, target):
    return float(np.dot(piece,target)/np.dot(target,target))


def array(rows, key):
    return np.array([float(row[key]) for row in rows])


def mean(rows, column):
    return float(np.mean([row[column] for row in rows]))


def opposite(a, b):
    valid = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
    return int(np.sum(valid & (a * b < 0)))


def index_rows(rows, extra=()):
    data={case_key(r)+tuple(r[k] for k in extra):r for r in rows}
    if len(data)!=len(rows):
        raise ValueError("Duplicate case")
    return data


def main(run):
    sealed=js(run/"measurement_seal.json")
    verify(run,{"selection_seal.json":sealed["selection_seal"],**sealed["files"]})
    verify(run,js(run/"selection_seal.json")["files"])
    rankings=js(run/"rankings.json")
    rows=index_rows(read_csv(run/"cases.csv"))
    interventions=index_rows(read_csv(run/"group_interventions.csv"),("arm","budget"))
    with np.load(run/"channels.npz",allow_pickle=False) as a:
        keys=[(int(seed),int(strength)/100,int(case),int(mask)) for seed,strength,case,mask in a["keys"]]
        terminal,first,through,single_links=[a[k] for k in ("terminal","first_removed","first4_to_last","first4_individual_to_last")]
    assert len(keys)==864 and set(keys)==set(rows) and len(interventions)==8640
    assert terminal.shape==first.shape==through.shape==(864,32) and single_links.shape==(864,4,32)
    teachers, parts, budgets, routes, channels, links = [], [], [], [], [], []
    for strength in (1.,0.5):
        for seed in sorted({key[0] for key in keys}):
            ids=[i for i,key in enumerate(keys) if key[:2]==(seed,strength)]
            cohort=[keys[i] for i in ids]
            cell=[rows[k] for k in cohort]
            assert len(cell)==72
            base=dict(seed=seed,strength=strength)
            y=array(cell,"remaining")
            natural=array(cell,"natural")
            original, residual=rms(natural),rms(y)
            teachers.append(dict(**base,n=72,natural_rms=original,remaining_rms=residual,
                remaining_vs_natural=residual/original,
                position_cancellation=1-np.abs(y).sum()/array(cell,"site_abs").sum(),
                channel_cancellation=1-np.abs(y).sum()/array(cell,"channel_abs").sum(),
                branch_opposite=opposite(array(cell,"conv_only"),array(cell,"skip_only")),
                first_last_gate_opposite=opposite(array(cell,"gate_first"),array(cell,"gate_last")),
                site_positive_mean=float(array(cell,"site_positive").mean()),site_negative_mean=float(array(cell,"site_negative").mean())))
            for part in PARTS:
                vals=array(cell,part)
                item=dict(**base,term=part,rms=rms(vals),ratio=rms(vals)/residual,
                          signed_projection=projection(vals,y),opposite=opposite(vals,y))
                if part.startswith("site_"):
                    item["mean_sites"]=float(array(cell,"count_"+part[5:]).mean())
                parts.append(item)
            for fields in (PARTS[:3], PARTS[3:7], PARTS[-4:]):
                np.testing.assert_allclose(sum(array(cell,k) for k in fields),y,atol=1e-9,rtol=0)
            order=rankings[str(seed)]
            v=order["last"]["target"][:4]
            u=order["first"]["target"][:4]
            for ch in range(32):
                for layer, removed, ranking in (("first",first[ids,ch],order["first"]["target"]),
                                                ("last",terminal[ids,ch],order["last"]["target"])):
                    channels.append(dict(**base,layer=layer,channel=ch,development_rank=ranking.index(ch)+1,
                        removed_rms=rms(removed),signed_projection=projection(removed,y),remaining_ratio=rms(y-removed)/residual))
            for a,src in enumerate(u):
                for dst in range(32):
                    effect=single_links[ids,a,dst]
                    links.append(dict(**base,first_channel=src,last_channel=dst,last_in_frozen_four=int(dst in v),
                        effect_rms=rms(effect),projection_on_first_effect=projection(effect,first[ids,src]),
                        projection_on_total=projection(effect,y)))
            for arm,budget in [(arm,n) for arm in ARMS for n in (1,4,8)]+[("target",32)]:
                observed=[interventions[k+(arm,str(budget))] for k in cohort]
                for stage in STAGES:
                    left=array(observed,stage+"_remaining")
                    removed=y-left
                    budgets.append(dict(**base,arm=arm,budget=budget,stage=stage,remaining_ratio=rms(left)/residual,
                        remaining_vs_natural=rms(left)/original,removed_rms=rms(removed),removed_projection=projection(removed,y),
                        smaller_cases=int(np.sum(abs(left)<abs(y)))))
                if arm=="target" and budget==4:
                    removed=array(observed,"first_removed")
                    overlap=array(observed,"overlap")
                    outside=removed-overlap
                    check=through[ids][:,v].sum(1)
                    np.testing.assert_allclose(check,overlap,atol=1e-9,rtol=0)
                    np.testing.assert_allclose(through[ids].sum(1),removed,atol=1e-9,rtol=0)
                    routes.append(dict(**base,first_removed_rms=rms(removed),routed_to_last4_rms=rms(overlap),
                        routed_to_other28_rms=rms(outside),projection_on_first_removed=projection(overlap,removed),
                        opposite_to_other28=opposite(overlap,outside),first_group_nonadditive_rms=rms(array(observed,"first_nonadditive"))))
    # 同一表内各行均保留相同字段；位置信息只适用于末层分组。
    for row in parts:
        row.setdefault("mean_sites","")
    for name,data in (("teachers",teachers),("components",parts),("channel_scores",channels),
                      ("budget_scores",budgets),("group_routes",routes),("link_scores",links)):
        write_csv(run/(name+".csv"),data)
    overview={}
    for strength in (1.,.5):
        t=[r for r in teachers if r["strength"]==strength]
        p=[r for r in parts if r["strength"]==strength]
        b=[r for r in budgets if r["strength"]==strength]
        u=[r for r in routes if r["strength"]==strength]
        comps={}
        for part in PARTS:
            item=[r for r in p if r["term"]==part]
            comps[part]={k:mean(item,k) for k in ("rms","ratio","signed_projection")}
            comps[part]["opposite"]=sum(r["opposite"] for r in item)
            if part.startswith("site_"):
                comps[part]["mean_sites"]=mean(item,"mean_sites")
        contrasts={}
        for stage in STAGES:
            for budget in (1,4,8):
                sets={arm:[r for r in b if (r["arm"],r["stage"],r["budget"])==(arm,stage,budget)] for arm in ARMS}
                target=sets["target"]
                delta=np.array([(one["remaining_ratio"]+two["remaining_ratio"])/2-tar["remaining_ratio"]
                                for tar,one,two in zip(target,sets["random1"],sets["random2"])])
                contrasts[f"{stage}_{budget}"]=dict(remaining=mean(target,"remaining_ratio"),
                    remaining_vs_natural=mean(target,"remaining_vs_natural"),
                    random1=mean(sets["random1"],"remaining_ratio"),random2=mean(sets["random2"],"remaining_ratio"),
                    improvement=float(delta.mean()),wins=int(np.sum(delta>0)),passes=bool(delta.mean()>=.05 and np.sum(delta>0)>=4),
                    smaller_models=sum(r["remaining_ratio"]<1 for r in target))
        overview[str(strength)]=dict(records=432,model_means={k:mean(t,k) for k in
            ("natural_rms","remaining_rms","remaining_vs_natural","position_cancellation","channel_cancellation",
             "site_positive_mean","site_negative_mean")},branch_opposite=sum(r["branch_opposite"] for r in t),
            components=comps,channel_interventions=contrasts,
            routing={k:mean(u,k) for k in ("first_removed_rms","routed_to_last4_rms","routed_to_other28_rms",
                                            "projection_on_first_removed","first_group_nonadditive_rms")},
            routed_signs_opposed=sum(r["opposite_to_other28"] for r in u))
    put(run/"summary.json",dict(measurement_seal=sha(run/"measurement_seal.json"),strengths=overview))
    print("Scored 864 eight-state residuals and 8640 sets of channel interventions.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)

"""计算符号转变、成对路径恢复、两条通道之外的输出以及 U 的子集分摊。"""
import argparse
from math import factorial
from pathlib import Path

import numpy as np

from mechanism import EPS, PAIRS, U, balance, js, label, put, read_csv, sha, signs, verify, write_csv


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def transitions(before, after):
    x,y = signs(balance(before)), signs(balance(after))
    robust = (np.minimum(np.abs(before).min(1),np.abs(after).min(1)) >= .001)
    return dict(same_to_opp=int(np.sum((x==1)&(y==-1))),opp_to_same=int(np.sum((x==-1)&(y==1))),
                zero=int(np.sum(y==0)),same=int(np.sum(y==1)),opposite=int(np.sum(y==-1)),
                robust_flips=int(np.sum((x*y==-1)&robust)))


def predict_counts(y,p):
    pos,neg=y==1,y==-1
    tp,tn=int(np.sum(pos&(p==1))),int(np.sum(neg&(p==-1)))
    recalls=[count/n.sum() for count,n in ((tp,pos),(tn,neg)) if n.any()]
    return dict(same_total=int(pos.sum()),opposite_total=int(neg.sum()),same_correct=tp,opposite_correct=tn,
                predicted_zero=int(np.sum(p==0)),accuracy=(tp+tn)/int((pos|neg).sum()),balanced=float(np.mean(recalls)))


def local_summary(run):
    seal=js(run/"local_checks.json")
    verify(run,seal["files"])
    data=read_csv(run/"local_cases.csv")
    groups={}
    for row in data:
        group=(row["pair"],row["strength"],row["arm"])
        groups.setdefault(group,[]).append(row)
    results=[]
    for (tag,strength,name),rows in groups.items():
        sizes=[int(r["sites"]) for r in rows]
        results.append(dict(pair=tag,strength=strength,arm=name,n=len(rows),solved=sum(int(r["solved"]) for r in rows),
            minimum=min(sizes),median=float(np.median(sizes)),maximum=max(sizes),
            h_flips_restored=sum(r["h_relation"]!=r["goal_relation"] and r["relation"]==r["goal_relation"] for r in rows),
            output_from_h_rms=rms(np.array([float(r["dy_from_h"]) for r in rows])),
            endpoint_from_h_rms=rms(np.array([float(r["dend_from_h"]) for r in rows]))))
    write_csv(run/"local_summary.csv",results)


def main(run):
    measured=js(run/"measurement_seal.json")
    verify(run,{"prediction_seal.json":measured["prediction_seal"],**measured["files"]})
    forecast=js(run/"prediction_seal.json")
    verify(run,{"selection_seal.json":forecast["selection_seal"],**forecast["files"]})
    verify(run,js(run/"selection_seal.json")["files"])
    with np.load(run/"responses.npz",allow_pickle=False) as a:
        keys=a["keys"]
        arms=a["arms"].tolist()
        values=a["values"]
    def key(r):
        return float(r["strength"]),int(r["case"]),int(r["mask"]),r["pair"]
    metadata={key(r):r for r in read_csv(run/"conditions.csv")}
    predictions={key(r):r for r in read_csv(run/"predictions.csv")}
    switches=read_csv(run/"switches.csv")
    selected=js(run/"selection.json")
    sets,interactions,subsets,conditions,rules,sites,case_rows=[],[],[],[],[],[],[]
    gate_balance, gate_cases = [], []
    checks={}
    def check(name,x):
        error=float(np.max(np.abs(x)))
        if not np.isfinite(error) or error>1e-8:
            raise ArithmeticError((name,error))
        checks[name]=max(checks.get(name,0.), error)
    for strength in (1.,.5):
        ids=np.flatnonzero(keys[:,1]==round(strength*100))
        for p,pair in enumerate(PAIRS):
            tag=label(pair)
            info=dict(pair=tag,strength=strength)
            registry=[(strength,int(k[2]),int(k[3]),tag) for k in keys[ids]]
            meta=[metadata[k] for k in registry]
            spec=selected[tag]
            records=values[ids,p]
            table={name:records[:,i] for i,name in enumerate(arms)}
            base,h=table["base"],table["h_pair"]
            a0,ah=base[:,2:4],h[:,2:4]
            goal=a0-ah
            goal_norm=float(np.square(goal).sum())
            h_flipped=signs(balance(a0))*signs(balance(ah))==-1
            truth=signs(balance(a0))
            # 在相同预激活上固定末次开关。两次读出之差是实测的门控项。
            for context,active,fixed in (("natural",base,table["natural_last_all"]),
                                         ("H",h,table["h_last_all"])):
                carry=fixed[:,2:4]
                gate=active[:,2:4]-carry
                changes=signs(carry)*signs(active[:,2:4])==-1
                relation_changed=signs(balance(carry))*signs(balance(active[:,2:4]))==-1
                gate_balance.append(dict(**info,background=context,n=len(ids),a_overturned=int(changes[:,0].sum()),
                    b_overturned=int(changes[:,1].sum()),both_overturned=int(changes.all(1).sum()),
                    relation_changed=int(relation_changed.sum()),carry_a_rms=rms(carry[:,0]),carry_b_rms=rms(carry[:,1]),
                    gate_a_rms=rms(gate[:,0]),gate_b_rms=rms(gate[:,1]),gate_output_rms=rms(active[:,0]-fixed[:,0])))
                for n,(_,case,mask,_) in enumerate(registry):
                    gate_cases.append(dict(**info,case=case,mask=mask,background=context,carry_a=carry[n,0],
                        gate_a=gate[n,0],carry_b=carry[n,1],gate_b=gate[n,1],
                        relation_fixed=int(signs(balance(carry[n]))),relation_actual=int(signs(balance(active[n,2:4])))))
            for name,field in table.items():
                a=field[:,2:4]
                moved=field-base
                from_h=field-h
                recovered=from_h[:,2:4]
                yy=moved[:,0]
                pp=moved[:,2:4].sum(1)
                outside=moved[:,8]
                check("pair_and_other30",yy-pp-outside)
                result=dict(**info,arm=name,n=len(ids),**transitions(a0,a),
                    a_change_rms=rms(moved[:,2]),b_change_rms=rms(moved[:,3]),pair_change_rms=rms(pp),
                    total_change_rms=rms(yy),other30_change_rms=rms(outside),endpoint_change_rms=rms(moved[:,1]),
                    y_from_h_rms=rms(from_h[:,0]),endpoint_from_h_rms=rms(from_h[:,1]),
                    pair_from_h_rms=rms(recovered.sum(1)),other30_from_h_rms=rms(from_h[:,8]),
                    recovery_projection=float(np.sum(recovered*goal)/goal_norm),
                    recovery_error_ratio=rms(a-a0)/rms(goal),h_signs_restored=int(np.sum(h_flipped&(signs(balance(a))==truth))),
                    extra_flips=int(np.sum((~h_flipped)&(signs(balance(a))*truth==-1))),
                    via_u_flips=int(np.sum(signs(balance(base[:,6:8]))*signs(balance(field[:,6:8]))==-1)),
                    via_u_same=int(np.sum(signs(balance(field[:,6:8]))==1)),
                    via_u_opposite=int(np.sum(signs(balance(field[:,6:8]))==-1)))
                sets.append(result)
                if name in ("base","h1","h2","h_pair","block_k_pair","rescue_k_pair","h_first_all","h_last_all",
                            "h_both_all","h_first_target","h_last_target","h_both_target","natural_both_all"):
                    for n,(_,case,mask,_) in enumerate(registry):
                        case_rows.append(dict(**info,case=case,mask=mask,arm=name,y=field[n,0],endpoint=field[n,1],
                            a=a[n,0],b=a[n,1],t_a=field[n,6],t_b=field[n,7],other30=field[n,8],
                            relation=int(signs(balance(a[n]))),dy=yy[n],d_pair=pp[n],d_other30=outside[n]))
            for name,start,one,two,both in (
                ("block_H","base","h1","h2","h_pair"),("block_K","base","block_k1","block_k2","block_k_pair"),
                ("restore_K","h_pair","rescue_k1","rescue_k2","rescue_k_pair"),
                ("replay_gates","h_pair","h_first_all","h_last_all","h_both_all"),
                ("replay_sites","h_pair","h_first_target","h_last_target","h_both_target")):
                extra=table[both]-table[one]-table[two]+table[start]
                interactions.append(dict(**info,operation=name,delta_a_rms=rms(extra[:,2]),delta_b_rms=rms(extra[:,3]),
                    delta_pair_rms=rms(extra[:,2:4].sum(1)),delta_output_rms=rms(extra[:,0]),
                    delta_endpoint_rms=rms(extra[:,1])))
            # 四个旧通道的 Shapley 值分摊其实际联合干预，包括它们之间的非加和项。
            allocation=[]
            for i,ch in enumerate(U):
                phi=np.zeros_like(base)
                for mask in range(16):
                    if mask & (1<<i):
                        continue
                    size=mask.bit_count()
                    weight=factorial(size)*factorial(3-size)/24
                    phi+=weight*(table[f"u_{mask}"]-table[f"u_{mask|(1<<i)}"])
                allocation.append(phi)
                total=base-table["u_15"]
                row=dict(**info,channel=ch)
                for j,keyname in ((0,"output"),(2,"a"),(3,"b")):
                    row[keyname+"_projection"]=float(phi[:,j]@total[:,j]/(total[:,j]@total[:,j]))
                    row[keyname+"_rms"]=rms(phi[:,j])
                subsets.append(row)
            check("shapley_efficiency",np.sum(allocation,axis=0)-(base-table["u_15"]))
            for pred in ("majority","class_mask","threshold_relation","sparse_relation"):
                guess=np.array([int(predictions[k][pred]) for k in registry])
                rules.append(dict(**info,method=pred,**predict_counts(truth,guess)))
            for factor in ("all","recipient","mask","size_angle","recipient_mask"):
                if factor=="all":
                    vals=["all"]*len(meta)
                elif factor in ("recipient","mask"):
                    vals=[r[factor] for r in meta]
                elif factor=="size_angle":
                    vals=[r["size"]+"_"+r["angle"] for r in meta]
                else:
                    vals=[r["recipient"]+"_"+r["mask"] for r in meta]
                for value in dict.fromkeys(vals):
                    ix=np.flatnonzero(np.array(vals)==value)
                    x,y=a0[ix],ah[ix]
                    weight=np.array([[float(meta[i]["weight_a"]),float(meta[i]["weight_b"])] for i in ix])
                    conditions.append(dict(**info,factor=factor,value=value,n=len(ix),natural_same=int(np.sum(truth[ix]==1)),
                        natural_opposite=int(np.sum(truth[ix]==-1)),routed_same=int(np.sum(signs(balance(base[ix,6:8]))==1)),
                        same_activation_third=int(np.sum(signs(balance(x/weight))==1)),
                        same_weight_sign=int(np.sum(weight.prod(1)>0)),h_flips=int(h_flipped[ix].sum()),
                        h_y_change_rms=rms(h[ix,0]-base[ix,0]),first_open=sum(int(meta[i]["first_opened"]) for i in ix),
                        first_close=sum(int(meta[i]["first_closed"]) for i in ix),last_open=sum(int(meta[i]["last_opened"]) for i in ix),
                        last_close=sum(int(meta[i]["last_closed"]) for i in ix)))
            for stage in ("first","last"):
                for rank,coord in enumerate(spec["sites"][stage]["target"],1):
                    subset=[r for r in switches if r["pair"]==tag and float(r["strength"])==strength and r["stage"]==stage and int(r["rank"])==rank]
                    if len(subset)!=len(ids):
                        raise ValueError("Missing gate records")
                    arm=table[f"{stage}_site_{rank}"]
                    signs_after=signs(balance(arm[:,2:4]))
                    sites.append(dict(**info,stage=stage,rank=rank,channel=coord[0],y=coord[1],x=coord[2],
                        h_opened=sum(int(r["opened"]).bit_count() for r in subset),
                        h_closed=sum(int(r["closed"]).bit_count() for r in subset),
                        cases_changed=sum(r["original"]!=r["altered"] for r in subset),
                        a_change_rms=rms(arm[:,2]-h[:,2]),b_change_rms=rms(arm[:,3]-h[:,3]),
                        y_change_rms=rms(arm[:,0]-h[:,0]),endpoint_change_rms=rms(arm[:,1]-h[:,1]),
                        flips_from_h=int(np.sum(signs_after*signs(balance(ah))==-1)),
                        h_signs_restored=int(np.sum(h_flipped&(signs_after==truth)))))
    for name,data in (("arm_scores",sets),("joint_effects",interactions),("upstream_four",subsets),
                      ("factor_counts",conditions),("rule_scores",rules),("site_scores",sites),("case_effects",case_rows),
                      ("gate_balance",gate_balance),("gate_balance_cases",gate_cases)):
        write_csv(run/(name+".csv"),data)
    put(run/"score_checks.json",dict(status="PASS",measurement=sha(run/"measurement_seal.json"),max_abs=checks))
    local_summary(run)
    print("Scored",len(sets),"arm summaries and",len(sites),"fixed gate records.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    main(parser.parse_args().run)

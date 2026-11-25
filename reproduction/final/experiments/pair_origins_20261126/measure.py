"""在留出条件先给出方向预判，再重放通道、开关及真实分类头。"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mechanism import (HERE, COLUMNS, PAIRS, SEED, U, Inputs, Tail, balance, coordinate_mask, features, js, label,
                       meta, middle_block, natural, put, read_csv, replay, restored, selected_tail, sha, signs,
                       source, stamp, verify, checksums, write_csv)
from paths import third


def predict(inputs, selection, out):
    rows=[]
    for key,s,_,_ in inputs.samples(SEED,"test"):
        metadata = meta(inputs,key)
        for pair in PAIRS:
            spec = selection[label(pair)]
            x = features(s,pair,spec)
            rule = spec["threshold"]
            value = x.get(rule["feature"],0.)
            k = spec["K"]
            # 预测时其余第一处激活和捷径均取空组，只把 K 送入预选读出。
            sparse = restored(s.first[:1].expand_as(s.first),s.first,k)
            _,_,field = selected_tail(s,sparse,s.skip[:1].expand_as(s.skip),list(pair))
            a = (third(field).mean((-2,-1))*s.weights[list(pair)]).cpu().numpy()
            rows.append(dict(**metadata,pair=label(pair),threshold_value=value,
                threshold_relation=rule["left"] if value<=rule["cut"] else rule["right"],
                majority=spec["majority"],class_mask=spec["class_mask"][f"{metadata['recipient']}_{key['mask']}"],
                sparse_relation=int(signs(balance(a))),sparse_a=a[0],sparse_b=a[1],
                drive_a=x["drive_a"],drive_b=x["drive_b"],weight_product=x["readout_product"],
                selected_opened=x["selected_opened"],selected_closed=x["selected_closed"]))
    write_csv(out/"predictions.csv",rows)
    put(out/"prediction_seal.json",dict(utc=stamp(),selection_seal=sha(out/"selection_seal.json"),
                                          files=checksums(out,["predictions.csv"])))
    print("Predictions sealed for 144 conditions and both pairs.",flush=True)


def changes(before, after, ids):
    a, b = before[:,ids], after[:,ids]
    return int((~a & b).sum()), int((a & ~b).sum())


def gate_records(info, stage, before, after, site_list):
    rows=[]
    for rank,(ch,y,x) in enumerate(site_list,1):
        def pattern(field):
            values=field[:,ch,y,x].cpu().numpy().astype(np.uint8)
            return int(np.packbits(values,bitorder="little")[0])
        original,intervened=pattern(before),pattern(after)
        rows.append(dict(**info,stage=stage,rank=rank,channel=ch,y=y,x=x,original=original,
                         altered=intervened,opened=intervened & ~original,closed=original & ~intervened))
    return rows


def inspect_case(s, pair, spec, base, upstream_subsets):
    ref=natural(s)
    h,k=spec["H"],spec["K"]
    dest=list(pair)
    h_state=middle_block(s,h)
    perturbed=Tail(s,h_state.act,h_state.skip)
    rows={"base":base.read(pair),"h_pair":perturbed.read(pair)}
    def whole(name,act,skip):
        rows[name]=Tail(s,act,skip).read(pair)
    for name,channels in (("h1",[h[0]]),("h2",[h[1]]),("h_random1",spec["H_random"][0]),("h_random2",spec["H_random"][1])):
        changed=middle_block(s,channels)
        whole(name,changed.act,changed.skip)
    only=middle_block(s,h,keep=True)
    whole("h_only",only.act,only.skip)
    for name,channels in (("k1",[k[0]]),("k2",[k[1]]),("k_pair",k)):
        whole("block_"+name,restored(ref.act,ref.act[:1],channels),ref.skip)
        whole("rescue_"+name,restored(h_state.act,ref.act,channels),h_state.skip)
    for i,channels in enumerate(spec["K_random"],1):
        whole(f"rescue_random{i}",restored(h_state.act,ref.act,channels),h_state.skip)
    whole("k_only",restored(ref.act[:1].expand_as(ref.act),ref.act,k),ref.skip[:1].expand_as(ref.skip))

    def gate_mask(stage,arm):
        if arm=="all":
            ids = k if stage=="first" else dest
            return coordinate_mask(s.g0_first,channels=ids)
        return coordinate_mask(s.g0_first,coords=spec["sites"][stage][arm])

    for background,first,tail,first_gate,last_gate,arms in (
            ("h",h_state,perturbed,ref.gates,s.last_gates,("all","target","random1","random2")),
            ("natural",ref,base,s.g0_first[None],s.g0_last[None],("all","target"))):
        for arm in arms:
            m1,m2=gate_mask("first",arm),gate_mask("last",arm)
            first_restored=replay(first.z,first.act,m1,first_gate)
            current=Tail(s,first_restored,first.skip)
            rows[f"{background}_first_{arm}"]=current.read(pair)
            rows[f"{background}_last_{arm}"]=tail.read(pair,m2,last_gate)
            rows[f"{background}_both_{arm}"]=current.read(pair,m2,last_gate)

    for index,coord in enumerate(spec["sites"]["first"]["target"],1):
        region=coordinate_mask(s.g0_first,coords=[coord])
        updated=replay(h_state.z,h_state.act,region,ref.gates)
        whole(f"first_site_{index}",updated,h_state.skip)
    for index,coord in enumerate(spec["sites"]["last"]["target"],1):
        region=coordinate_mask(s.g0_last,coords=[coord])
        rows[f"last_site_{index}"]=perturbed.read(pair,region,s.last_gates)
    for mask,values in enumerate(upstream_subsets):
        rows[f"u_{mask}"]=values[label(pair)]

    before_first, after_first = ref.gates, h_state.gates
    before_last, after_last = s.last_gates,perturbed.gates[:,0]
    count=(*changes(before_first,after_first,k),*changes(before_last,after_last,dest))
    fields=(("first",before_first,after_first),("last",before_last,after_last))
    return rows,count,fields


def measure(inputs, out, old, selected):
    if (out/"measurement_seal.json").exists():
        raise FileExistsError("Measurement already sealed")
    seal=js(out/"prediction_seal.json")
    verify(out,{"selection_seal.json":seal["selection_seal"],**seal["files"]})
    old_seal=js(old/"measurement_seal.json")
    old_file=f"model_{SEED}.npz"
    verify(old,{old_file:old_seal["files"][old_file]})
    with np.load(old/old_file,allow_pickle=False) as data:
        archive={k:data[k] for k in data.files}
    lookup={tuple(key):i for i,key in enumerate(archive["keys"])}
    pred=read_csv(out/"predictions.csv")
    stored={(float(r["strength"]),int(r["case"]),int(r["mask"]),r["pair"]):r for r in pred}
    records,coords,metadata,keys=[],[],[],[]
    names=None
    errors={}
    def check(name,a,b):
        gap=float(np.abs(np.asarray(a)-np.asarray(b)).max())
        if not np.isfinite(gap) or gap>1e-8:
            raise ArithmeticError((name,gap))
        errors[name]=max(errors.get(name,0.),gap)
    for key,s,_,_ in inputs.samples(SEED,"test"):
        info=meta(inputs,key)
        record=(SEED,int(100*info["strength"]),info["case"],info["mask"])
        idx=lookup[record]
        base=Tail(s,s.first,s.skip)
        by_subset=[]
        for mask in range(16):
            channels=[ch for bit,ch in enumerate(U) if (mask>>bit)&1]
            tail=Tail(s,restored(s.first,s.first[:1],channels),s.skip) if channels else base
            by_subset.append({label(pair):tail.read(pair) for pair in PAIRS})
        pair_values=[]
        for pair in PAIRS:
            spec=selected[label(pair)]
            rows,counts,fields=inspect_case(s,pair,spec,base,by_subset)
            if names is None:
                names=list(rows)
            if names!=list(rows):
                raise ValueError("Intervention order changed")
            values=np.stack(list(rows.values()))
            check("head_other30",values[:,-1],0.)
            check("full_U_route",rows["u_15"][6:8],0.)
            actual=rows["base"]
            pos=archive["group_names"].tolist().index("base")
            old_base=archive["groups"][idx,pos]
            ix=[archive["outside"].tolist().index(j) for j in pair]
            singles=archive["singles"][idx,ix]
            contrasts=old_base[None,:,0]-singles[:,:,0]
            check("old_head",actual[[0,1,4,5]],old_base.ravel())
            check("old_A",actual[2:4],contrasts[:,0])
            check("old_T",actual[6:8],contrasts[:,0]-contrasts[:,1])
            col=archive["channel_pairs"].tolist().index(list(pair))
            check("old_joint_cut",actual[8:10],archive["pairs"][idx,col,0])
            prediction=stored[info["strength"],info["case"],info["mask"],label(pair)]
            check("sparse_forecast",rows["k_only"][2:4],[float(prediction["sparse_a"]),float(prediction["sparse_b"])])
            metadata.append(dict(**info,pair=label(pair),first_opened=counts[0],first_closed=counts[1],
                                 last_opened=counts[2],last_closed=counts[3],weight_a=float(s.weights[pair[0]]),
                                 weight_b=float(s.weights[pair[1]])))
            for stage,before,after in fields:
                coords.extend(gate_records(dict(**info,pair=label(pair)),stage,before,after,spec["sites"][stage]["target"]))
            pair_values.append(values[:,:-1])
        keys.append(record)
        records.append(pair_values)
    if len(keys)!=144 or set(keys)!=set(lookup):
        raise ValueError("Held input registry changed")
    np.savez_compressed(out/"responses.npz",keys=np.asarray(keys),pairs=np.array(PAIRS),arms=np.array(names),
                        columns=np.array(COLUMNS[:-1]),values=np.array(records))
    write_csv(out/"conditions.csv",metadata)
    write_csv(out/"switches.csv",coords)
    put(out/"checks.json",dict(status="PASS",device=inputs.device,cases=144,pairs=2,arms=len(names),max_abs=errors))
    files=("responses.npz","conditions.csv","switches.csv","checks.json")
    put(out/"measurement_seal.json",dict(utc=stamp(),prediction_seal=sha(out/"prediction_seal.json"),files=checksums(out,files)))
    print("Measured",len(names),"interventions per target pair.",flush=True)


def main(root,out,device):
    inputs=Inputs(root,device)
    previous=source(inputs)
    verify(out,js(out/"selection_seal.json")["files"])
    protocol=js(out/"protocol.json")
    if protocol["inputs"]!=inputs.spec or protocol["code"]!=checksums(HERE,protocol["code"]):
        raise ValueError("A frozen input or source changed")
    selected=js(out/"selection.json")
    if not (out/"prediction_seal.json").exists():
        predict(inputs,selected,out)
    measure(inputs,out,previous,selected)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    main(args.assets.resolve(),args.out.resolve(),args.device)

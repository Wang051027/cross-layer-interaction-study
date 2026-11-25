"""按上游信号重建固定末层开关的两条读数，逐预算与真实阻断核对。"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from score import predict_counts, rms
from mechanism import (HERE, SEED, PAIRS, Tail, Inputs, balance, bn_kernel, coordinate_mask, js, label, meta, put,
                       restored, sha, signs, source, stamp, verify, write_csv, checksums)
from paths import third


def upstream_terms(s,pair):
    dest=list(pair)
    kernel=bn_kernel(s,dest,list(range(32)))
    area=s.g0_last.shape[-1]*s.g0_last.shape[-2]
    gates=s.g0_last[dest].to(s.z1.dtype)*s.weights[dest,None,None]/area
    delta=third(s.first)
    pieces=[]
    for j in range(2):
        slopes=F.conv_transpose2d(gates[None,j:j+1],kernel[j:j+1],padding=1)[0]
        pieces.append((delta*slopes).sum((-2,-1)))
    skip=(third(s.skip[:,dest])*gates).sum((-2,-1))
    return torch.stack(pieces,-1).cpu().numpy(),skip.cpu().numpy()


def select(inputs,out):
    pools={p:[] for p in PAIRS}
    for _,s,_,_ in inputs.samples(SEED,"development"):
        for p in PAIRS:
            pools[p].append(upstream_terms(s,p)[0])
    selection={}
    for pair,rows in pools.items():
        x=np.stack(rows)
        residual=x.sum(1)
        chosen=[]
        for _ in range(32):
            order=sorted(set(range(32))-set(chosen))
            errors=((residual[:,None]-x[:,order])**2).mean(axis=(0,2))
            idx=min(range(len(order)),key=lambda i:(round(float(errors[i]),12),order[i]))
            chosen.append(order[idx])
            residual-=x[:,order[idx]]
        selection[label(pair)]=chosen
    put(out/"coverage_order.json",dict(orders=selection,design=sha(HERE/"COVERAGE.md"),source=sha(Path(__file__)),
        previous=sha(out/"selection_seal.json"),utc=stamp()))
    return selection


def main(root,out,device):
    inputs=Inputs(root,device)
    source(inputs)
    final=js(out/"measurement_seal.json")
    verify(out, final["files"])
    with np.load(out/"responses.npz",allow_pickle=False) as archive:
        idx={tuple(k):i for i,k in enumerate(archive["keys"])}
        origin=archive["values"][:,:,archive["arms"].tolist().index("base")]
    inherited=js(out/"selection.json")
    order=select(inputs,out)
    records=[]
    errors={"fixed_readout":0.,"full_free_forward":0.}
    def check(key,a,b):
        d=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        if not np.isfinite(d) or d>1e-8:
            raise ValueError((key,d))
        errors[key]=max(errors[key],d)
    for key,s,_,_ in inputs.samples(SEED,"test"):
        metadata=meta(inputs,key)
        row=idx[SEED,round(100*key["strength"]),key["case"],key["mask"]]
        for i,pair in enumerate(PAIRS):
            tag=label(pair)
            terms,skip=upstream_terms(s,pair)
            sets={f"top{k}":order[tag][:k] for k in (0,2,4,8,16,32)}
            sets["relay"]=inherited[tag]["K"]
            for name,channels in sets.items():
                expected=skip+terms[channels].sum(0)
                keep=restored(s.first[:1].expand_as(s.first),s.first,channels)
                tail=Tail(s,keep,s.skip)
                fixed=tail.read(pair,coordinate_mask(s.g0_last,channels=list(pair)),s.g0_last[None])
                free=tail.read(pair)
                check("fixed_readout",expected,fixed[2:4])
                if name=="top32":
                    check("full_free_forward",free[:-1],origin[row,i])
                records.append(dict(**metadata,pair=tag,budget=name,target=int(signs(balance(origin[row,i,2:4]))),
                    origin_y=origin[row,i,0],fixed_a=expected[0],fixed_b=expected[1],fixed=int(signs(balance(expected))),
                    fixed_y=fixed[0],fixed_endpoint=fixed[1],free_a=free[2],free_b=free[3],
                    free=int(signs(balance(free[2:4]))),free_y=free[0],free_endpoint=free[1]))
    write_csv(out/"coverage_cases.csv",records)
    result=[]
    for pair in PAIRS:
        for strength in (1.,.5):
            for name in sets:
                cell=[r for r in records if (r["pair"],r["strength"],r["budget"])==(label(pair),strength,name)]
                target=np.array([r["target"] for r in cell])
                for mode in ("fixed","free"):
                    pred=np.array([r[mode] for r in cell])
                    moved=np.array([r[mode+"_y"]-r["origin_y"] for r in cell])
                    result.append(dict(pair=label(pair),strength=strength,budget=name,mode=mode,
                        **predict_counts(target,pred),output_change_rms=rms(moved)))
    write_csv(out/"coverage_scores.csv",result)
    put(out/"coverage_checks.json",dict(status="PASS",max_abs=errors,previous=sha(out/"measurement_seal.json"),
        files=checksums(out,["coverage_order.json","coverage_cases.csv","coverage_scores.csv"])))
    print("Checked all seven upstream budgets and both terminal gate settings.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    main(args.assets.resolve(),args.run.resolve(),args.device)

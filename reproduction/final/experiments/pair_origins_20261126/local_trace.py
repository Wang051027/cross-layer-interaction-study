"""逐例定位 H 阻断引发的开闭差异，并在同一例中实测空间位置回放。"""
import argparse
from pathlib import Path

import numpy as np

from mechanism import (HERE, PAIRS, SEED, Inputs, Tail, balance, coordinate_mask, js, label, meta, middle_block,
                       natural, put, replay, sha, signs, site_scores, source, verify, write_csv, checksums)
from paths import third


def positions(flat, channels, shape):
    return [(int(channels[c]),int(y),int(x)) for c,y,x in zip(*np.unravel_index(flat,shape))]


def local_top(scores, counts, channels, rng):
    ids=np.flatnonzero(counts.ravel()>0)
    order=np.lexsort((ids,-np.round(scores.ravel()[ids],12)))
    chosen=ids[order[:8]]
    coordinates=positions(chosen,channels,scores.shape)
    control=[]
    area=scores.shape[1]*scores.shape[2]
    for i,_ in enumerate(channels):
        n=int(np.sum(chosen//area==i))
        eligible=ids[ids//area==i]
        draw=rng.choice(eligible,n,replace=False)
        control.extend(positions(draw,channels,scores.shape))
    return coordinates,control


def describe(info,kind,coords,stage,ref_z,new_z,ref_g,new_g,contributions=None):
    rows=[]
    for rank,(ch,y,x) in enumerate(coords,1):
        def at(field):
            return field[:,ch,y,x].detach().cpu().numpy()
        original,altered=at(ref_g),at(new_g)
        z0,zh=at(ref_z),at(new_z)
        peak=int(np.argmax(np.round(np.abs(zh)*(original!=altered),10)))
        rows.append(dict(**info,kind=kind,stage=stage,rank=rank,channel=ch,y=y,x=x,
            original_bits=int(np.packbits(original,bitorder="little")[0]),
            h_bits=int(np.packbits(altered,bitorder="little")[0]),peak_corner=peak,
            original_z=z0[peak],h_z=zh[peak],
            signed_delta=("" if contributions is None else float(contributions[ch,y,x]))))
    return rows


def main(root,run,device):
    inputs=Inputs(root,device)
    source(inputs)
    sealed=js(run/"measurement_seal.json")
    verify(run,sealed["files"])
    spec=js(run/"selection.json")
    with np.load(run/"responses.npz",allow_pickle=False) as data:
        lookup={tuple(key):i for i,key in enumerate(data["keys"])}
        arms=data["arms"].tolist()
        truth=data["values"]
    rows,coords_table=[],[]
    errors={}
    def check(name,a,b):
        gap=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        if not np.isfinite(gap) or gap>1e-8:
            raise ArithmeticError((name,gap))
        errors[name]=max(errors.get(name,0.),gap)
    for key,s,_,_ in inputs.samples(SEED,"test"):
        inf=meta(inputs,key)
        idx=lookup[SEED,round(100*inf["strength"]),inf["case"],inf["mask"]]
        ref=natural(s)
        for i,pair in enumerate(PAIRS):
            tag=label(pair)
            info=dict(**inf,pair=tag)
            rule=spec[tag]
            h=middle_block(s,rule["H"])
            tail=Tail(s,h.act,h.skip)
            base=truth[idx,i,arms.index("base")]
            result=tail.read(pair)
            check("frozen_H",result[:-1],truth[idx,i,arms.index("h_pair")])
            goal=base[2:4]
            start=result[2:4]
            rng=np.random.default_rng([291,inf["mask"],inf["case"],round(100*inf["strength"]),i,20261126])
            def evaluate(stage,coords):
                region=coordinate_mask(s.g0_first,coords=coords)
                if stage=="first":
                    new=replay(h.z,h.act,region,ref.gates)
                    return Tail(s,new,h.skip).read(pair)
                return tail.read(pair,region,s.last_gates)
            def store(name,n,v,eligible=0,solved=0):
                rows.append(dict(**info,arm=name,sites=n,eligible=eligible,solved=solved,y=v[0],endpoint=v[1],
                    a=v[2],b=v[3],relation=int(signs(balance(v[2:4]))),goal_relation=int(signs(balance(goal))),
                    h_relation=int(signs(balance(start))),dy_from_h=v[0]-result[0],dend_from_h=v[1]-result[1],
                    da_from_h=v[2]-start[0],db_from_h=v[3]-start[1]))
            z=tail.z[:,0]
            g=tail.gates[:,0]
            spatial=third(z*(s.last_gates.to(z.dtype)-g.to(z.dtype)))*s.weights[:,None,None]/(z.shape[-2]*z.shape[-1])
            field=spatial.cpu().numpy()
            ranking=site_scores(s,h,pair,rule["K"])
            for stage,channels,(scores,counts),oldz,newz,oldg,newg in zip(
                    ("first","last"),(rule["K"],pair),ranking,(s.z1,s.z2),(h.z,z),(ref.gates,s.last_gates),(h.gates,g)):
                top,random=local_top(scores,counts,channels,rng)
                for name,coords in (("top8",top),("random8",random)):
                    current=evaluate(stage,coords)
                    store(stage+"_"+name,len(coords),current)
                    if stage=="last":
                        expected=np.array([sum(field[c,y,x] for c,y,x in coords if c==target) for target in pair])
                        check("terminal_sites",current[2:4]-start,expected)
                coords_table.extend(describe(info,"top8",top,stage,oldz,newz,oldg,newg,
                                             field if stage=="last" else None))
            robust=(np.min(np.abs(np.r_[goal,start]))>=.001 and signs(balance(goal))*signs(balance(start))==-1)
            if not robust:
                continue
            changed=np.flatnonzero(np.sign(goal)!=np.sign(start))
            if len(changed)!=1:
                raise ValueError("Expected one changed channel")
            j=int(changed[0])
            ch=pair[j]
            direction=np.sign(goal[j])
            gain=(direction*field[ch]).ravel()
            valid=np.flatnonzero(gain>0)
            order=valid[np.lexsort((valid,-np.round(gain[valid],13)))]
            running=direction*start[j]+np.cumsum(gain[order])
            enough=np.flatnonzero(running>=.001)
            solved=bool(len(enough))
            n=int(enough[0])+1 if solved else len(order)
            chosen=positions(order[:n],[ch],(1,*field[ch].shape))
            v=evaluate("last",chosen)
            store("minimal" if solved else "all_helpful",n,v,1,int(solved))
            check("minimal_set",v[2+j]-start[j],sum(field[c,y,x] for c,y,x in chosen))
            if solved:
                short=evaluate("last",chosen[:-1])
                store("one_fewer",n-1,short,1)
                if v[2+j]*direction<.001-1e-10 or short[2+j]*direction>=.001+1e-10:
                    raise ValueError("Threshold boundary check failed")
                pool=np.flatnonzero((s.last_gates[:,ch]!=g[:,ch]).any(0).cpu().numpy().ravel())
                random=positions(rng.choice(pool,n,replace=False),[ch],(1,*field[ch].shape))
                control=evaluate("last",random)
                store("minimal_random",n,control,1,int(control[2+j]*direction>=.001))
            coords_table.extend(describe(info,"minimal" if solved else "all_helpful",chosen,"last",s.z2,z,s.last_gates,g,field))
    write_csv(run/"local_cases.csv",rows)
    write_csv(run/"local_coordinates.csv",coords_table)
    put(run/"local_checks.json",dict(status="PASS",max_abs=errors,measurement=sha(run/"measurement_seal.json"),
        design=sha(HERE/"LOCAL_DESIGN.md"),source=sha(Path(__file__)),files=checksums(run,["local_cases.csv","local_coordinates.csv"])))
    print("Verified",len(rows),"case-specific replays.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    main(args.assets.resolve(),args.run.resolve(),args.device)

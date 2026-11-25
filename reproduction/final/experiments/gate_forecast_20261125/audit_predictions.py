"""按子集奇偶、完整卷积与自动微分独立复核开关、选点和有符号预测。"""
import argparse
from pathlib import Path
import sys

import numpy as np
from analyze import ARMS, LAYERS, keyed
from run import js, location, put, sha, verify


def main(assets, run):
    frozen, reference, src = location(assets)
    sys.path.insert(0, str(src))
    import dsprites_teacher as teacher
    import torch

    teacher.configure()
    torch.set_grad_enabled(False)
    protocol = js(run/"protocol.json")
    seal = js(run/"prediction_seal.json")
    verify(run, {**seal["files"],"protocol.json":seal["protocol"]})
    verify(reference,protocol["fits"])
    verify(frozen,{k.removeprefix("runs/"):v for k,v in protocol["inputs"].items()})
    outputs = keyed(run/"predictions.csv")
    patches = keyed(run/"intervention_predictions.csv",True)
    with np.load(run/"forecast_sites.npz",allow_pickle=False) as archive:
        bitfields = {name:archive[name] for name in ("empty","predicted","majority")}
        ranking, key_array = archive["sites"], archive["keys"]
    with np.load(frozen/"dsprites_shape_donor_20261010/panel/data.npz",allow_pickle=False) as archive:
        pixels,pairs,labels = [archive[k] for k in ("images","pairs","labels")]
    triples = (7,11,13,14)
    masks = sorted(m for m in range(16) if m.bit_count()<=2)
    rows = [[i for i,s in enumerate(masks) if s&t==s] for t in triples]
    coeff = torch.tensor([[(-1.)**s.bit_count() if s&t==s else 0. for s in masks] for t in triples],dtype=torch.double)
    def q(field):
        return (coeff@field.flatten(start_dim=1)).reshape(len(triples),*field.shape[1:])
    def flags(first,second):
        arrays=np.stack([f.reshape(4,-1).numpy() for f in (first,second)],axis=1)
        return np.packbits(arrays,axis=-1,bitorder="little")
    maxima={}
    def check(name, array, expected):
        a=np.asarray(array)
        b=np.asarray(expected)
        error=float(np.max(np.abs(a-b)))
        if not np.isfinite(error) or error>1e-8:
            raise ValueError((name,error))
        maxima[name]=max(maxima.get(name,0.),error)
    row_number=0
    for seed in protocol["seeds"]:
        model=teacher.DSpritesCNN()
        model.load_state_dict(torch.load(frozen/f"dsprites_20260929/seed_{seed}/teacher.pt",map_location="cpu",weights_only=True))
        model=model.double().eval().requires_grad_(False)
        first_block,tail=model.block2,model.block3
        bank=model.block1(model.stem(torch.tensor(pixels[:,None],dtype=torch.double)))
        fit=js(reference/f"fit/{seed}.json")
        bits=torch.tensor([[bool(m&(1<<(ch//4))) for ch in range(16)] for m in masks],dtype=torch.double)[:,:,None,None]
        for strength in (1.,.5):
            interpolation=bits*strength
            for case in protocol["held_cases"]:
                ri,di=pairs[case]
                r,d=labels[[ri,di]]
                def read(h):
                    logits=model.readout(h.mean((-2,-1)))
                    return logits[...,d]-logits[...,r]
                def v2(h):
                    return first_block.bn2(first_block.conv2(h))
                mixed=bank[ri]*(1-interpolation)+bank[di]*interpolation
                u=first_block.bn1(first_block.conv1(mixed))
                a=u.relu()
                v=v2(a)+first_block.shortcut(mixed)
                h=v.relu()
                low1=tail.bn1(tail.conv1(h))
                low2=tail.bn2(tail.conv2(low1.relu()))+tail.shortcut(h)
                q1,q2=q(low1),q(low2)
                g1,g2=q1>0,q2>0
                e1,e2=low1[0]>0,low2[0]>0
                majority=[torch.stack([(field[ids]>0).sum(0)>3 for ids in rows]) for field in (low1,low2)]
                span=slice(row_number,row_number+4)
                np.testing.assert_array_equal(key_array[span],[(seed,int(strength*100),case,t) for t in triples])
                for name, fields in (("empty",(e1.expand_as(g1),e2.expand_as(g2))),
                                     ("predicted",(g1,g2)),("majority",majority)):
                    np.testing.assert_array_equal(flags(*fields),bitfields[name][span])
                # 用完整卷积相减取得 block2 的传播修正，再单独替换四个中间通道。
                qa=q(a)
                moved=qa.clone()
                for i,t in enumerate(triples):
                    singles=[masks.index(m) for m in (1,2,4,8) if t&m]
                    moved[i,fit["first"]]=(u[singles].sum(0)-2*u[0]).relu()[fit["first"]]
                revised=(q(v)+(v2(moved)-v2(qa))).relu()
                middle=q(h)
                middle[:,fit["middle"]]=revised[:,fit["middle"]]
                z1=tail.bn1(tail.conv1(middle))
                activated=z1*g1
                with torch.enable_grad():
                    leaf=activated.detach().requires_grad_(True)
                    raw=tail.bn2(tail.conv2(leaf))+tail.shortcut(middle)
                    gradient=torch.autograd.grad(read(raw*g2).sum(),leaf)[0].detach()
                z2=raw.detach()
                true_forecast=read(z2*g2)
                low_readout=read(low2.relu()).unsqueeze(-1)
                expected=[outputs[seed,strength,case,t] for t in triples]
                check("forecast_output",true_forecast-q(low_readout).squeeze(1),[float(row["predicted_4"]) for row in expected])
                weight=(model.readout.weight[d]-model.readout.weight[r])[:,None,None]/(middle.shape[-1]*middle.shape[-2])
                for i,t in enumerate(triples):
                    lists=ranking[row_number+i]
                    for layer,(sg,guess,empty) in enumerate(((gradient[i]*z1[i],g1[i],e1),(weight*z2[i],g2[i],e2))):
                        candidates=np.flatnonzero((guess!=empty).numpy())
                        vals=np.round(abs(sg).flatten().numpy()[candidates],11)
                        expected_list=candidates[np.lexsort((candidates,-vals))[:64]]
                        np.testing.assert_array_equal(lists[0,layer][lists[0,layer]>=0],expected_list)
                    for group,arm in zip(lists,ARMS):
                        g_first=g1[i:i+1].expand(3,-1,-1,-1).clone()
                        ids1,ids2=(ls[ls>=0].astype(np.int64) for ls in group)
                        for variant in (0,2):
                            g_first[variant].view(-1)[ids1]=e1.flatten()[ids1]
                        counter=tail.bn2(tail.conv2(z1[i]*g_first))+tail.shortcut(middle[i:i+1])
                        g_second=q2[i]+(counter-z2[i])>0
                        # 第二层单独阻断时，第一处保持原预测。
                        g_second[1]=g2[i]
                        for variant in (1,2):
                            g_second[variant].view(-1)[ids2]=e2.flatten()[ids2]
                        diff=read(counter*g_second)-true_forecast[i]
                        check("forecast_patch",diff,[float(patches[seed,strength,case,t,arm,layer]["predicted_change"])
                                                     for layer in LAYERS])
                row_number+=4
        print("independent forecast",seed,flush=True)
    assert row_number==864
    put(run/"independent_predictions.json",dict(status="PASS",device="cpu",cases=row_number,patched_forecasts=7776,
        matching_top_lists=row_number*2,matching_two_layer_mask_sets=["predicted","majority","empty"],
        numerical_maxima=maxima,prediction_seal=sha(run/"prediction_seal.json"),audit_source=sha(Path(__file__))))
    print("Independent forecast check:",maxima)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    main(args.assets.resolve(),args.run.resolve())

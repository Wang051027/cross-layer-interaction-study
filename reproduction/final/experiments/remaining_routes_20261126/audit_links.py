"""在原网络中逐个替换卷积通道对的消息，核对其实际输出作用。"""
import argparse
from pathlib import Path

import numpy as np
from experiment import Inputs, case_key, js, put, read_csv, sha, verify


def main(assets, run):
    inputs=Inputs(assets,"cpu")
    torch=inputs.torch
    seal=js(run/"measurement_seal.json")
    verify(run,{"selection_seal.json":seal["selection_seal"],**seal["files"]})
    verify(run,js(run/"selection_seal.json")["files"])
    with np.load(run/"channels.npz",allow_pickle=False) as saved:
        ids=saved["keys"]
        expected=saved["first4_individual_to_last"]
    with np.load(inputs.previous/"forecast_sites.npz",allow_pickle=False) as old:
        np.testing.assert_array_equal(ids,old["keys"])
        sites=old["sites"][:,0]
    targets={case_key(r):r for r in read_csv(run/"cases.csv")}
    orders=js(run/"rankings.json")
    panel=inputs.panel
    row=0
    maxima={}
    def compare(name, v, ref):
        gap=float(np.max(np.abs(np.asarray(v)-np.asarray(ref))))
        if not np.isfinite(gap) or gap>1e-8:
            raise ValueError((name,gap))
        maxima[name]=max(gap,maxima.get(name,0.))
    def enforce(z,flat,empty):
        # 只改写原来锁定的 64 个坐标，其余位置保留自然 ReLU。
        native=z.relu().contiguous()
        coord=np.unravel_index(flat,tuple(z.shape[-3:]))
        native[:,coord[0],coord[1],coord[2]]=z[:,coord[0],coord[1],coord[2]]*empty[coord]
        return native
    masks=np.arange(15)
    bits=torch.tensor([[(int(mask)>>(ch//4))&1 for ch in range(16)] for mask in masks],dtype=torch.double)[:,:,None,None]
    for seed in inputs.seeds:
        net=inputs.model(seed)
        block=net.block3
        first_ids=orders[str(seed)]["first"]["target"][:4]
        bank=net.block1(net.stem(torch.tensor(panel["images"][:,None],dtype=torch.double)))
        for strength in (1.,.5):
            for case in inputs.split["test"]:
                ri,di=panel["pairs"][case]
                r,d=panel["labels"][[ri,di]]
                h=net.block2((1-strength*bits)*bank[ri]+strength*bits*bank[di])
                z1=block.bn1(block.conv1(h))
                skip=block.shortcut(h)
                empty1=z1[0]>0
                empty2=(block.bn2(block.conv2(block.mid(h[:1])))+skip[:1])[0]>0
                for t in (7,11,13,14):
                    assert tuple(ids[row])==(seed,int(100*strength),case,t)
                    corners=[int(m) for m in masks if int(m)&t==int(m)]
                    # 三阶差分由子集大小独立计算，顺序按掩码从小到大。
                    signs=torch.tensor([(-1.)**(3-m.bit_count()) for m in corners],dtype=torch.double)
                    first_points,last_points=sites[row]
                    a=enforce(z1[corners],first_points,empty1)
                    c_raw=block.conv2(a)
                    residual=skip[corners]
                    def finish(raw,bypass):
                        return enforce(block.bn2(raw)+bypass,last_points,empty2)
                    def response(act):
                        logits=net.readout(act.mean((-2,-1)))
                        return (logits[:,d]-logits[:,r])@signs
                    z=finish(c_raw,residual)
                    before=response(z)
                    truth=targets[seed,strength,case,t]
                    compare("remaining",before,float(truth["remaining"]))
                    compare("conv_only",response(finish(c_raw,residual[:1])),float(truth["conv_only"]))
                    compare("skip_only",response(finish(c_raw[:1].expand_as(c_raw),residual)),float(truth["skip_only"]))
                    pooled=z.mean((-2,-1))
                    destinations=torch.arange(pooled.shape[1])
                    for k,source in enumerate(first_ids):
                        fixed=a.clone()
                        fixed[:,source]=a[:1,source]
                        other_raw=block.conv2(fixed)
                        other=finish(other_raw,residual).mean((-2,-1))
                        # 单个末层通道读取被替换的消息，其他通道读取原支路。
                        # 分类头在同一个批次上实际计算 32 个单边干预输出。
                        hybrid=pooled[:,None,:].expand(-1,len(destinations),-1).clone()
                        hybrid[:,destinations,destinations]=other
                        scores=net.readout(hybrid)
                        isolated=before-signs@(scores[:,:,d]-scores[:,:,r])
                        compare("isolated_links",isolated,expected[row,k])
                        # 再于 BN 之前直接替换一条通道消息，核对池化时汇合的等价性。
                        dst=orders[str(seed)]["last"]["target"][k]
                        one=c_raw.clone()
                        one[:,dst]=other_raw[:,dst]
                        compare("isolated_before_bn",before-response(finish(one,residual)),isolated[dst])
                    row+=1
        print("isolated links",seed,flush=True)
    assert row==864
    put(run/"isolated_links.json",dict(status="PASS",device="cpu",cases=row,individual_links=row*4*32,
        pre_bn_checks=row*4,maximum=max(maxima.values()),comparisons=maxima,
        measurements=sha(run/"measurement_seal.json"),audit_source=sha(Path(__file__))))
    print("Isolated link check passed:",maxima)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--run",required=True,type=Path)
    args=parser.parse_args()
    main(args.assets.resolve(),args.run.resolve())

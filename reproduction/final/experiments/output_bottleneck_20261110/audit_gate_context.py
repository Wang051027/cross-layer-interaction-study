"""独立复核：整卷积重算预测，自动微分验证匹配门控后的残差来自哪条方向斜率。"""
import argparse
import sys
from pathlib import Path
import numpy as np
from run_experiment import case_key, js, put, read_csv, sha, write_csv


def main(assets,run,device):
    pub=(assets/'reproduction/held_condition').is_dir()
    code=assets/('source/scripts' if pub else 'scripts')
    frozen=assets/('reproduction/held_condition/runs' if pub else 'runs')
    refs=assets/('reproduction/held_condition/reference' if pub else 'runs/dsprites_rule_discrimination_20261031')
    sys.path.insert(0,str(code))
    import dsprites_teacher as teacher
    import torch
    teacher.configure();torch.set_grad_enabled(False)
    protocol=js(run/'protocol.json')
    for path,expected in protocol['inputs'].items():assert sha(frozen/path.removeprefix('runs/'))==expected
    seal=js(run/'measurement_seal.json');pseal=js(run/'prediction_seal.json')
    assert sha(run/'measurements.csv')==seal['measurements'] and sha(run/'predictions.csv')==pseal['predictions']
    prev={case_key(r):r for r in read_csv(run/'predictions.csv')}
    measured={case_key(r):r for r in read_csv(run/'measurements.csv')}
    with np.load(frozen/'dsprites_shape_donor_20261010/panel/data.npz',allow_pickle=False) as source:
        pix,pairs,labels=[source[k] for k in ('images','pairs','labels')]
    ids=sorted(set(pairs[protocol['split']['test']].flatten().tolist()))
    assert len(ids)==18
    mask=(7,11,13,14)
    # 八角差分由包含关系与奇偶性生成，不调用原算子。
    coeff=np.array([[(-1.)**(3-int(s).bit_count()) if s&t==s else 0. for s in range(15)] for t in mask])
    lowids=[s for s in range(15) if s not in mask]
    def tensor(a):return torch.as_tensor(a,dtype=torch.float64,device=device)
    kernel=tensor(coeff);lowmat=-kernel[:,lowids]
    def op(w,x):return (w @ x.reshape(len(x),-1)).reshape(len(w),*x.shape[1:])
    maximum={};rows=[]
    def check(name,calc,archive):
        n=float((calc-tensor(archive)).abs().max())
        if n>1e-8:raise ValueError((name,n))
        maximum[name]=max(n,maximum.get(name,0))
    for seed in range(202609291,202609297):
        net=teacher.DSpritesCNN()
        net.load_state_dict(torch.load(frozen/f'dsprites_20260929/seed_{seed}/teacher.pt',map_location='cpu',weights_only=True))
        net=net.double().to(device).eval().requires_grad_(False)
        bank=dict(zip(ids,net.block1(net.stem(tensor(pix[ids,None])))))
        filename=f'fit/{seed}.json';assert sha(refs/filename)==protocol['fit_files'][filename]
        fit=js(refs/filename);order=protocol['channel_orders'][str(seed)]
        for strength in (1.,.5):
            weights=tensor([[strength*int((s>>(ch//4))&1) for ch in range(16)] for s in range(15)])[:,:,None,None]
            for case in protocol['split']['test']:
                ri,di=pairs[case];r,d=labels[[ri,di]]
                inputs=(1.-weights)*bank[ri]+weights*bank[di]
                def F(h):
                    v=net.readout(net.block3(h).mean((2,3)))
                    return v[:,d]-v[:,r]
                def derivative(h):
                    with torch.enable_grad():
                        leaf=h.detach().requires_grad_(True)
                        return torch.autograd.grad(F(leaf).sum(),leaf)[0].detach()
                bs=net.block2; lo=inputs[lowids]
                z=bs.bn1(bs.conv1(lo)); u=z.relu()
                v=bs.bn2(bs.conv2(u))+bs.shortcut(lo);hlow=v.relu()
                qh=op(lowmat,hlow); qf=op(lowmat,F(hlow));qu=op(lowmat,u)
                proposed=qu.clone()
                for j,t in enumerate(mask):
                    singleton=[lowids.index(g) for g in (1,2,4,8) if g&t]
                    # 源替换相加，第一预激活的三组值可由空/单组精确取得。
                    new=(z[singleton].sum(0)-2*z[0]).relu()
                    proposed[j,fit['first']]=new[fit['first']]
                # 不截取卷积核、不手乘 BN 斜率；用两次完整卷积之差取得传播量。
                def second(q):return bs.bn2(bs.conv2(q))
                allcorr=(op(lowmat,v)+second(proposed)-second(qu)).relu()-qh
                J0=derivative(hlow[:1])
                keys=[(seed,strength,case,t) for t in mask]
                for budget in (0,4,8,16,32):
                    pred=qh.clone();ii=order[:budget]
                    if ii:pred[:,ii]+=allcorr[:,ii]
                    for name,y in [(f'fixed_{budget}',((pred-qh)*J0).sum((1,2,3))),
                                   (f'live_{budget}',F(pred)-qf),(f'clip_{budget}',F(pred.clamp_min(0))-qf)]:
                        check(name,y,[float(prev[k][name]) for k in keys])
                # 诊断此处使用真实场；先前的预测核对已经完成。
                h=bs(inputs);h0=h[:1];sel=fit['middle']
                hc=h0.expand_as(h).clone();hc[:,sel]=h[:,sel]
                o=h-hc;c=hc-h0
                jc=derivative(hc);jo=derivative(h0+o)
                remain=op(kernel,((jc-J0)*o).sum((1,2,3)))
                shift=op(kernel,((jo-J0)*c).sum((1,2,3)))
                check('matched_lock_derivative',remain,[float(measured[k]['gap_lock_both']) for k in keys])
                check('outside_slope_autograd',shift,[float(measured[k]['slope_shift']) for k in keys])
                for k,b in zip(keys,remain.tolist()):
                    gap=float(measured[k]['gap']);rows.append(dict(seed=seed,strength=strength,case=case,mask=k[-1],
                        measured_gap=gap,carried_by_C_changed_slope=b,outside_gate_remainder=gap-b))
    assert len(rows)==864
    write_csv(run/'gate_identity.csv',rows)
    stats={}
    for strength in (1.,.5):
        out=[]
        for seed in range(202609291,202609297):
            cell=[r for r in rows if r['seed']==seed and r['strength']==strength]
            a,b=np.array([[r['outside_gate_remainder'],r['carried_by_C_changed_slope']] for r in cell]).T
            out.append(dict(seed=seed,opposed_records=int(np.sum(a*b<0)),cosine=float(a@b/np.sqrt((a@a)*(b@b)))))
        stats[str(strength)]=dict(by_model=out,opposed_records=sum(r['opposed_records'] for r in out),
                                 mean_cosine=float(np.mean([r['cosine'] for r in out])))
    put(run/'independent_audit.json',dict(status='PASS',records=864,device=device,
        compared_forecasts_per_record=15,source=sha(Path(__file__)),maxima=maximum,
        gate_identity_sha256=sha(run/'gate_identity.csv'),cancellation=stats))
    print('Independent forecasts, matched-gate and tangent identities passed:',max(maximum.values()))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--device',choices=('cuda','cpu'),default='cpu')
    a=ap.parse_args();main(a.assets.resolve(),a.run.resolve(),a.device)

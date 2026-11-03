"""数值预检：原forward、四状态边界、抵消恒等式和旧端点复现。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import coalition_context_mechanism as core
from run_dsprites_coalition_context import BASE,PRIOR,SHAPE
from run_dsprites_upstream_rescue import load,sources_for,two_traces
from joint_window_mechanism import intervention_specs


def test():
    """只用已有面板case0；空/全集/同值交换是实现测试，不新增研究样本。"""
    device=teacher.configure(); panel=load(SHAPE/'panel/data.npz'); errors=[]
    with torch.no_grad():
        for seed in teacher.SEEDS:
            model=teacher.load_teacher(seed,device); source=sources_for(model,panel,device)
            recipient,donor=panel['pairs'][0]; half,full=two_traces(model,source,recipient,donor)
            search=load(PRIOR/f'search/seed_{seed}/data.npz'); units=load(PRIOR/f'search/seed_{seed}/units.npz'); fixed=load(PRIOR/f'search/seed_{seed}/frozen.npz'); prior=load(PRIOR/f'evaluation/seed_{seed}/data.npz')
            for m,mask in enumerate((3,15)):
                specs=intervention_specs({k:v[0,m] for k,v in units.items()},{k:v[0,m] for k,v in search.items()})
                before,after=core.endpoints(model,half,full,mask,[specs[e] for e in core.ENDPOINTS])
                C=fixed['channels'][0,m]; C=C[C>=0]; point=int(fixed['lists'][0,m,0,0]); weight=np.zeros(3)
                weight[panel['labels'][donor]]=1.;weight[panel['labels'][recipient]]=-1.
                out=core.measure(model,before,after,C,point,weight)
                fields=core.four_fields(before,after,C)
                logits=model.readout(model.block3(fields.flatten(0,1)).mean((2,3))).reshape(16,4,3).cpu().numpy()
                errors.append(float(abs(logits-out['logits'][:,:,0]).max()))
                errors.append(float(abs(logits[:,3]-prior['natural'][0,m,list(core.ENDPOINTS),2]).max()))
                z=out['point_z']; errors.append(float(abs(z[:,3]-z[:,1]-z[:,2]+z[:,0]).max()))
                for boundaryC in ([],list(range(32))):
                    q=core.four_fields(before,after,boundaryC)
                    a,b=(0,3) if not boundaryC else (3,0)
                    assert torch.equal(q[:,1],q[:,a]) and torch.equal(q[:,2],q[:,b])
                same=core.four_fields(before,before[None],C)
                assert torch.equal(same[:,0],same[:,1]) and torch.equal(same[:,0],same[:,2]) and torch.equal(same[:,0],same[:,3])
            print(json.dumps(dict(seed=seed,status='PASS',maximum_error=max(errors))),flush=True)
    assert max(errors)<1e-9
    p=BASE/'reports/dsprites_coalition_context_preflight_20261025.json'; assert not p.exists()
    result=dict(status='PASS',utc=teacher.now(),models=6,case=0,masks=[3,15],endpoints=16,maximum_error=max(errors),checks=['originalforward','priorendpoints','pointaffineinteractionzero','emptyC','allC','sameH'])
    teacher.dump(p,result); return result


if __name__=='__main__': print(json.dumps(test()))

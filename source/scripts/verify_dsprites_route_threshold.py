"""固定例子的事后联合阈值验证，独立于半强度名单的主评价。

预定第一教师/案例0/源mask15、半强度选出的第一单点不变；全强度
测32个中间通道的第一预激活变化，并证明该点最少需要多少整通道
残差同时加入才越阈值。只证明当前A和当前R下的局部条件。
"""
from __future__ import annotations
import numpy as np
import torch
import dsprites_teacher as teacher
from run_dsprites_residual_routes import RUN,BASE,load,verify_seal
from test_dsprites_residual_routes import original_holds


@torch.no_grad()
def verify():
    assert not (RUN/'threshold_check.json').exists();verify_seal();device=teacher.configure()
    model=teacher.load_teacher(teacher.SEEDS[0],device)
    for p in model.parameters():p.requires_grad_(False)
    prior=BASE/'runs/dsprites_layer_transfer_20261013/evaluation/seed_202609291/alpha_1.0/example_fields.npz';values=load(prior)
    a=torch.as_tensor(values['additive_middle'][15:16],device=device);h=torch.as_tensor(values['natural_middle'][15:16],device=device);r=h-a
    chosen=load(RUN/'discovery/seed_202609291/selection.npz');coordinate=int(chosen['gates'][0,0,5,0]);ch,yy,xx=np.unravel_index(coordinate,(32,32,32))
    b=model.block3;za=b.bn1(b.conv1(a));ga=za>0;va=b.bn2(b.conv2(torch.relu(za)))+a;gb=va>0
    base=float(za[0,ch,yy,xx]);natural=float(b.bn1(b.conv1(h))[0,ch,yy,xx]);assert base>0 and natural<0
    solo=[]
    for c in range(32):
        value=a.clone();value[:,c]=h[:,c];solo.append(float(b.bn1(b.conv1(value))[0,ch,yy,xx])-base)
    solo=np.array(solo);assert abs(solo.sum()-(natural-base))<1e-9
    order=np.lexsort((np.arange(32),solo));lower=base+np.cumsum(solo[order]);k=int(np.flatnonzero(lower<0)[0])+1
    assert np.all(lower[:k-1]>1e-9) and lower[k-1]<-1e-9
    # 每个不超过k−1的子集之和，不会比k−1个最负变化之和更小。
    ids=order[:k];groups=[('empty',np.array([],dtype=int))]+[(f'single_{c}',np.array([c])) for c in ids]
    groups +=[(f'leading_{k-1}',ids[:k-1]),(f'leading_{k}',ids),('all32',np.arange(32))]
    w=torch.tensor([-1.,1.,0.],device=device,dtype=a.dtype);baseline=float(model.readout(b(a).mean((2,3)))@w);rows=[]
    for name,channels in groups:
        value=a.clone();value[:,channels]=h[:,channels]
        z=float(b.bn1(b.conv1(value))[0,ch,yy,xx]);predicted=base+solo[channels].sum();assert abs(z-predicted)<1e-9
        f=float(model.readout(b(value).mean((2,3)))@w);held=float(original_holds(model,value,ga,gb,np.array([coordinate]),None)@w)
        rows.append(dict(group=name,channels=list(map(int,channels)),preactivation=z,open=bool(z>0),predicted=float(predicted),task_output=f,
            task_delta=f-baseline,point_held_output=held,point_net_removed=f-held))
    assert all(x['open'] for x in rows if x['group'].startswith('single_'))
    assert next(x for x in rows if x['group']==f'leading_{k-1}')['open']
    assert not next(x for x in rows if x['group']==f'leading_{k}')['open']
    result=dict(status='PASS',utc=teacher.now(),scope='post hoc mechanism probe of predefined teacher202609291/case0/mask15 andhalf-selected first point;not held-out confirmation',
        source_file_sha256=teacher.sha(prior),coordinate=[int(ch),int(yy),int(xx)],baseline=base,natural=natural,
        single_preactivation_changes=solo.tolist(),minimum_channels=k,chosen_channels=ids.tolist(),best_k_minus_one_bound=float(lower[k-2]),
        kth_bound=float(lower[k-1]),rows=rows,
        proof='affine first preactivation in32 binary residual-channel presences;sum ofk-1 most negative changes is global lower bound for any <=k-1 subset;actualk-subset closes point',
        limitation='only this A/R/gate,whole spatial channel patches,no universal/minimal-output-circuit claim')
    teacher.dump(RUN/'threshold_check.json',result);print({key:result[key] for key in ('status','coordinate','minimum_channels','chosen_channels','best_k_minus_one_bound','kth_bound')})


if __name__=='__main__':verify()

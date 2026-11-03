"""三阶前置核验：手算多项式、阈值抵消和旧开发图原网络正向。

不读取本轮定位面板或结果。原block3.forward采用where门控覆盖，
不复用新尾部修补函数；自动微分核对第一层传播权重。
"""
from __future__ import annotations

import dsprites_teacher as teacher
import numpy as np
import torch
from torch.nn import functional as F
import triple_gate_mechanism as triple
import channel_gate_mechanism as previous
from test_dsprites_channel_gates import forward_override
from shape_donor_mechanism import matched_pairs


def known_answers():
    """纯三阶和12，四阶以上−2；第三通道能改变二阶阈值效应。"""
    d=triple.design();bits=d['bits'];x=3+bits@np.arange(1,9)
    x=x+4*bits[:,0]*bits[:,1]+7*bits[:,0]*bits[:,1]*bits[:,2]+5*bits[:,4]*bits[:,5]*bits[:,6]
    x=x-2*np.prod(bits[:,:4],1)+10*bits[:,6]*bits[:,7]
    p2=previous.differences(x,d['pair_indices']).sum();p3=triple.contrast(x,d['triple_indices']).sum()
    joint=(d['joint']@x)[1];assert p2==14 and p3==12 and joint==24 and joint-p2-p3==-2
    simple=((np.arange(8)[:,None]>>np.arange(3))&1).sum(1)
    first=np.maximum(simple-.5,0);second=.7*np.maximum(simple-2.5,0)
    assert np.isclose(triple.contrast(first),-.5) and np.isclose(triple.contrast(second),.35)
    assert np.isclose(triple.contrast(first+second),-.15)
    return dict(status='PASS',pair_sum=14,triple_sum=12,joint8=24,orders4plus=-2,threshold_signed_parts=[-.5,.35],threshold_net=-.15)


@torch.no_grad()
def original_partial(model,x,base,stage,lists):
    """在原forward中改开关掩码，八种名单组成64状态批次。"""
    b=model.block3;z1=b.bn1(b.conv1(base));g1=z1>0;g2=(b.bn2(b.conv2(F.relu(z1)))+base)>0
    support=torch.zeros((len(lists),32768),dtype=torch.bool,device=x.device)
    for j,ids in enumerate(lists):support[j,ids]=True
    support=support.repeat_interleave(8,0).reshape(-1,32,32,32)
    def mid(v):
        z=b.bn1(b.conv1(v));g=z>0
        if stage==0:g=torch.where(support,g1,g)
        return torch.where(g,z,torch.zeros_like(z))
    def finish(h,residual):
        z=b.bn2(b.conv2(h))+residual;g=z>0
        if stage==1:g=torch.where(support,g2,g)
        return torch.where(g,z,torch.zeros_like(z))
    try:
        b.mid=mid;b.finish=finish
        return model.readout(b(x.repeat(len(lists),1,1,1)).mean((2,3))).reshape(len(lists),8,3).cpu().numpy()
    finally:delattr(b,'mid');delattr(b,'finish')


def check(seed,device):
    """每位教师验证全95状态、选中三元组的两背景、所有干预名单。"""
    model=teacher.load_teacher(seed,device)
    for p in model.parameters():p.requires_grad_(False)
    factors=np.load(teacher.DATA/'development_factors.npy',allow_pickle=False);labels=np.load(teacher.DATA/'development_labels.npy',allow_pickle=False)
    r,donor=matched_pairs(factors,labels)[0];images=np.load(teacher.DATA/'development_images.npy',allow_pickle=False)
    with torch.no_grad():
        image=torch.as_tensor(images[[r,donor],None],dtype=torch.float64,device=device)
        source=model.block1(model.stem(image));base=model.block2(source[:1]);reference=model.block2(.5*source[:1]+.5*source[1:])
    d=triple.design();x=previous.patch_states(base,reference,np.arange(8),d['bits']);t=previous.trace_modes(model,x)
    errors={};weight=np.array([-1.,1.,0.])
    for mode in range(4):
        ref=forward_override(model,x,base,stage=0 if mode==1 else 1 if mode==2 else None,both=mode==3)
        errors['global_forward']=max(errors.get('global_forward',0.),float((ref-t['logits'][:,mode]).abs().max()))
    summary=triple.global_summary(t,d);score=summary['triple_logits'][:,0]@weight;selected=np.argsort(-abs(score),kind='stable')[:3]
    rng=np.random.default_rng(seed+123)
    for p in selected:
        for context in range(2):
            local=triple.local_trace(model,base,reference,np.arange(8),d['triples'][p],context,d,t)
            units=triple.unit_contributions(model,local,weight);effect=triple.contrast(local['logits'].cpu().numpy())
            parts=np.stack((effect[2]-effect[3],effect[0]-effect[2]))@weight
            errors['unit_sum']=max(errors.get('unit_sum',0.),float(abs(units.sum((1,2,3)).cpu().numpy()-parts).max()))
            errors['affine_zero']=max(errors.get('affine_zero',0.),float(abs(effect[3]).max()))
            leaf=local['h1'][:1].detach().clone().requires_grad_(True)
            z=model.block3.bn2(model.block3.conv2(leaf))+local['x'][:1]
            output=model.readout((local['g2']*z).mean((2,3)));gradient=torch.autograd.grad((output*torch.as_tensor(weight,device=device)).sum(),leaf)[0]
            errors['autograd_units']=max(errors.get('autograd_units',0.),float((triple.contrast(local['r1'])*gradient[0]-units[0]).abs().max()))
            selection=triple.select_sites(local,units,float(effect[0]@weight),rng);actual=triple.partial_holds(model,local,selection)
            for stage in range(2):
                ref=original_partial(model,local['x'],local['x'][:1],stage,triple.variant_ids(selection,stage))
                errors['actual_holds']=max(errors.get('actual_holds',0.),float(abs(ref-actual[stage]).max()))
            details=triple.details(local,units,selection)
            last=triple.contrast(np.moveaxis(actual[1],1,0))@weight
            errors['last_exact_removal']=max(errors.get('last_exact_removal',0.),float(abs(last-(effect[0]@weight-details['variant_contribution_sums'][1])).max()))
    assert max(errors.values())<1e-9,errors
    return dict(seed=seed,status='PASS',errors=errors)


if __name__=='__main__':
    import json
    device=teacher.configure();print(json.dumps(dict(status='PASS',known=known_answers(),teachers=[check(s,device) for s in teacher.SEEDS]),ensure_ascii=False))

"""新通道和门控计算的前置检查，只访问旧开发图，不访问新定位结果。

手算容斥多项式、原block3.forward门控覆盖、autograd伴随和精确单元
贡献全部对照。输出JSON供协议记录，不写历史实验文件。
"""
from __future__ import annotations

import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
import channel_gate_mechanism as gates
from shape_donor_mechanism import matched_pairs


def known_answers():
    """八变量函数含可手算的二、三、四阶项，检查各阶不会重算或遗漏。"""
    d=gates.design();bits=d['bits'];x=3+bits@np.arange(1,9)
    x=x+4*bits[:,0]*bits[:,1]+7*bits[:,0]*bits[:,1]*bits[:,2]
    x=x-2*np.prod(bits[:,:4],axis=1)+10*bits[:,6]*bits[:,7]
    p=gates.differences(x,d['stencils']);assert p.sum()==14 and p[0]==4 and p[-1]==10
    m=gates.mobius4(x[d['p4_indices']]);assert m[3]==4 and m[7]==7 and m[15]==-2
    assert np.array_equal(d['joint']@x,[9,19]) and (d['joint']@x)[1]-p.sum()==5
    return dict(status='PASS',pair_sum=14,joint4=9,joint8=19,higher_than_pair8=5)


@torch.no_grad()
def forward_override(model,x,base,stage=None,ids=None,both=False):
    """覆盖原ResidualBlock的mid/finish调用，以where写法独立验证门控。

    ids为None代表整层；first修改后原finish仍重算下游。退出时恢复类方法。
    """
    b=model.block3;z1=b.bn1(b.conv1(base));g1=z1>0
    z2=b.bn2(b.conv2(F.relu(z1)))+base;g2=z2>0
    def first(v):
        z=b.bn1(b.conv1(v));g=z>0
        if stage==0 or both:
            if ids is None:g=g1.expand_as(g)
            else:g=g.clone();g.flatten(1)[:,ids]=g1.flatten()[ids]
        return torch.where(g,z,torch.zeros_like(z))
    def finish(h,residual):
        z=b.bn2(b.conv2(h))+residual;g=z>0
        if stage==1 or both:
            if ids is None:g=g2.expand_as(g)
            else:g=g.clone();g.flatten(1)[:,ids]=g2.flatten()[ids]
        return torch.where(g,z,torch.zeros_like(z))
    try:
        b.mid=first;b.finish=finish
        return model.readout(b(x).mean((2,3)))
    finally:delattr(b,'mid');delattr(b,'finish')


def check(seed,device):
    """全部六教师都测试43状态、三门控对照和选择后的两预算修补。"""
    model=teacher.load_teacher(seed,device)
    for p in model.parameters():p.requires_grad_(False)
    factors=np.load(teacher.DATA/'development_factors.npy',allow_pickle=False)
    labels=np.load(teacher.DATA/'development_labels.npy',allow_pickle=False)
    r,donor=matched_pairs(factors,labels)[0]
    pixels=np.load(teacher.DATA/'development_images.npy',allow_pickle=False)
    images=torch.as_tensor(pixels[[r,donor],None],dtype=torch.float64,device=device)
    with torch.no_grad():
        sources=model.block1(model.stem(images));base=model.block2(sources[:1]);reference=model.block2(.5*sources[:1]+.5*sources[1:])
    d=gates.design();x=gates.patch_states(base,reference,np.arange(8),d['bits'])
    traced=gates.trace_modes(model,x);weight=np.array([-1.,1.,0.]);errors={}
    for j,mode in enumerate(gates.MODES):
        independent=forward_override(model,x,base,stage=0 if j==1 else 1 if j==2 else None,both=j==3)
        errors['forward_'+mode]=float((independent-traced['logits'][:,j]).abs().max())
    units=gates.pair_unit_contributions(model,traced,weight,d['stencils']);s=gates.summaries(traced,d,weight)
    errors['unit_sum']=float(np.abs(units.sum((2,3,4)).cpu().numpy()-s['pair_stageparts']@weight).max())
    errors['frozen_pair']=float(np.abs(s['pair_logits'][:,3]).max())
    errors['frozen_joint']=float(np.abs(s['joint_logits'][:,3]).max())
    # 自动微分独立核验第一层坐标的传播权重，不复用卷积转置公式。
    leaf=traced['h1'][:1].detach().clone().requires_grad_(True)
    z=model.block3.bn2(model.block3.conv2(leaf))+x[:1]
    output=model.readout((traced['g2']*z).mean((2,3)))
    p=torch.autograd.grad((output*torch.as_tensor(weight,device=device)).sum(),leaf)[0]
    first=gates.differences(traced['r1'],d['stencils'])*p
    errors['adjoint_autograd']=float((first-units[:,0]).abs().max())
    selection=gates.select_pairs_and_gates(traced,units,s['pair_logits'][:,0]@weight,d['stencils'],seed)
    partial=gates.partial_holds(model,traced,d['stencils'],selection)
    for rank,pair in enumerate(selection['selected_pairs']):
        indices=d['stencils'][pair][[3,1,2,0]]
        for stage in range(2):
            for bi,budget in enumerate(gates.GATE_BUDGETS):
                ids=selection['selected_gates'][rank,stage];ids=ids[ids>=0][:budget]
                independent=forward_override(model,x[indices],base,stage=stage,ids=ids).cpu().numpy()
                errors['partial_forward']=max(errors.get('partial_forward',0.),float(np.abs(independent-partial[rank,stage,bi,0]).max()))
                if stage==1:
                    actual=(partial[rank,stage,bi,0,-1]-partial[rank,stage,bi,0,1]-partial[rank,stage,bi,0,2]+partial[rank,stage,bi,0,0])@weight
                    predicted=s['pair_logits'][pair,0]@weight-float(units[pair,1].flatten()[ids].sum())
                    errors['last_gate_exact_removal']=max(errors.get('last_gate_exact_removal',0.),abs(actual-predicted))
    assert max(errors.values())<1e-9,errors
    return dict(seed=seed,status='PASS',errors=errors)


if __name__=='__main__':
    device=teacher.configure()
    print(json.dumps(dict(status='PASS',known_answers=known_answers(),teachers=[check(s,device) for s in teacher.SEEDS]),ensure_ascii=False))

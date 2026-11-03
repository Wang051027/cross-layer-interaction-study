"""56个空背景三元组、两层ReLU分解及冻结坐标的反事实计算。

三阶系数是八项容斥差分。补全其余5通道的背景对比可含四阶以上项，
不能称为纯三阶系数。所有函数不读写文件、不训练、不修改旧模块。
"""
from __future__ import annotations

import itertools
import numpy as np
import torch
from torch.nn import functional as F
import channel_gate_mechanism as previous

SIGNS=np.array([-1.,1.,1.,-1.,1.,-1.,-1.,1.])
TOP=3
VARIANTS=('selected32','random0_32','random1_32','selected128','random0_128','random1_128','aligned_subset','opposed_subset')
SELECTION_KEYS=('selected_gates','random_gates','aligned_gates','opposed_gates','eligible_counts','alignment_sign')


def design():
    """95状态包含所有0至3成员子集、旧前4全体和全8全体。"""
    masks=np.array(sorted({s for s in range(256) if s.bit_count()<=3}|{15,255}))
    lookup={int(s):i for i,s in enumerate(masks)};triples=np.array(list(itertools.combinations(range(8),3)))
    triple_indices=np.array([[lookup[sum((1<<int(c)) for j,c in enumerate(t) if s&(1<<j))] for s in range(8)] for t in triples])
    pairs=np.array(list(itertools.combinations(range(8),2)))
    pair_indices=np.array([[lookup[(1<<a)|(1<<b)],lookup[1<<a],lookup[1<<b],lookup[0]] for a,b in pairs])
    joint=np.zeros((2,95))
    for row,k in enumerate((4,8)):
        joint[row,lookup[(1<<k)-1]]=1;joint[row,lookup[0]]=k-1
        for j in range(k):joint[row,lookup[1<<j]]=-1
    assert len(masks)==95 and len(triples)==56
    return dict(masks=masks,bits=((masks[:,None]>>np.arange(8))&1),triples=triples,triple_indices=triple_indices,
        pairs=pairs,pair_indices=pair_indices,joint=joint,p4_indices=np.array([lookup[s] for s in range(16)]),signs=SIGNS)


def contrast(value,indices=None):
    """沿状态轴按0,A,B,AB,C,AC,BC,ABC顺序累加八项。"""
    if indices is not None:value=value[indices]
    axis=1 if indices is not None else 0
    result=value.take(0,axis=axis)*0 if isinstance(value,np.ndarray) else (value[:,0] if axis else value[0])*0
    for j,sign in enumerate(SIGNS):
        piece=np.take(value,j,axis=axis) if isinstance(value,np.ndarray) else (value[:,j] if axis else value[j])
        result=result+sign*piece
    return result


def global_summary(t,d):
    """分别保存二阶、纯三阶和四阶以上，带符号和重建联合余项。"""
    logits=t['logits'].cpu().numpy();triples=contrast(logits,d['triple_indices'])
    pairs=previous.differences(logits,d['pair_indices']);joint=np.einsum('km,msc->ksc',d['joint'],logits)
    p2=pairs.sum(0);p3=triples.sum(0);r3=joint[1]-p2;r4=r3-p3
    return dict(logits=logits,triple_logits=triples,pair_logits=pairs,joint_logits=joint,
        triple_stageparts=np.stack((triples[:,2]-triples[:,3],triples[:,0]-triples[:,2]),1),
        order_totals=np.stack((p2,p3,r4)),higher3=r3,
        order_stageparts=np.stack((np.stack((p2,p3,r4))[:,2]-np.stack((p2,p3,r4))[:,3],
            np.stack((p2,p3,r4))[:,0]-np.stack((p2,p3,r4))[:,2]),1),
        mobius4=np.stack([previous.mobius4(logits[d['p4_indices'],m]) for m in range(4)],1))


def local_trace(model,base,reference,channels,triple,context,d,global_trace):
    """空背景复用全局8子集；补全背景独立计算当地基准开关。"""
    if context==0:
        key=tuple(map(int,triple));p=next(i for i,t in enumerate(d['triples']) if tuple(t)==key)
        ids=d['triple_indices'][p]
        return {k:v if k in ('g1','g2') else v[ids] for k,v in global_trace.items()}
    bits=np.ones((8,8),dtype=np.int64)
    for j,c in enumerate(triple):bits[:,c]=(np.arange(8)>>j)&1
    x=previous.patch_states(base,reference,channels,bits)
    return previous.trace_modes(model,x)


@torch.no_grad()
def unit_contributions(model,t,weight):
    """坐标项p*Δ³r1、q*Δ³r2；可正可负，第一层项不是净干预效应。"""
    b=model.block3;w=torch.as_tensor(weight,dtype=t['x'].dtype,device=t['x'].device)
    q=(w@model.readout.weight)[:,None,None]/1024
    scale=b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps)
    p=F.conv_transpose2d(t['g2']*q[None]*scale[None,:,None,None],b.conv2.weight,
        stride=b.conv2.stride,padding=b.conv2.padding,dilation=b.conv2.dilation,groups=b.conv2.groups)
    return torch.stack((contrast(t['r1'])*p[0],contrast(t['r2'])*q))


def select_sites(t,unit,effect,rng):
    """半强度取绝对贡献前128，随机池相同；按半强度符号拆分该128名单。"""
    output={k:[] for k in SELECTION_KEYS if k!='alignment_sign'}
    alignment=1 if effect>=0 else -1
    pad=lambda ids:np.pad(ids,(0,128-len(ids)),constant_values=-1).astype(np.int64)
    for stage,key in enumerate(('z1','z2')):
        eligible=np.flatnonzero(torch.any((t[key]>0)!=(t[key][:1]>0),0).flatten().cpu().numpy())
        score=unit[stage].flatten().cpu().numpy();ids=eligible[np.argsort(-np.abs(score[eligible]),kind='stable')[:128]]
        output['selected_gates'].append(pad(ids));output['eligible_counts'].append(len(eligible))
        output['random_gates'].append([pad(rng.permutation(eligible)[:128]) for _ in range(2)])
        output['aligned_gates'].append(pad(ids[score[ids]*alignment>0]))
        output['opposed_gates'].append(pad(ids[score[ids]*alignment<0]))
    return {**{k:np.asarray(v) for k,v in output.items()},'alignment_sign':np.asarray(alignment)}


def variant_ids(selection,stage):
    """符号组是同一128名单的子集，数量可能不同，不作等预算优劣比较。"""
    valid=lambda p:p[p>=0]
    chosen=valid(selection['selected_gates'][stage]);random=[valid(p) for p in selection['random_gates'][stage]]
    return [chosen[:32],random[0][:32],random[1][:32],chosen[:128],random[0][:128],random[1][:128],
        valid(selection['aligned_gates'][stage]),valid(selection['opposed_gates'][stage])]


def details(t,unit,selection):
    """记录冻结坐标的当前八状态模式，并保留所有坐标正负和，量化抵消。"""
    values=np.zeros((2,128));pre=np.zeros((2,8,128));sums=np.zeros((2,2));variant_sums=np.zeros((2,8))
    counts=np.zeros((2,8),dtype=np.int64)
    for stage,key in enumerate(('z1','z2')):
        score=unit[stage].flatten().cpu().numpy();ids=selection['selected_gates'][stage];ids=ids[ids>=0]
        values[stage,:len(ids)]=score[ids];pre[stage,:,:len(ids)]=t[key].flatten(1).cpu().numpy()[:,ids]
        sums[stage]=[score[score>0].sum(),score[score<0].sum()]
        for j,v in enumerate(variant_ids(selection,stage)):variant_sums[stage,j]=score[v].sum();counts[stage,j]=len(v)
    return dict(unit_contributions=values,gate_preactivations=pre,unit_signed_sums=sums,
        variant_contribution_sums=variant_sums,variant_counts=counts)


@torch.no_grad()
def partial_holds(model,t,selection):
    """每层8种名单各重算八子集。第一层改动后第二层自然重新开关。"""
    results=[];b=model.block3
    for stage in range(2):
        h=t['h1' if stage==0 else 'h2'];z=t['z1' if stage==0 else 'z2'];gate=t['g1' if stage==0 else 'g2'].flatten()
        patched=[]
        for ids in variant_ids(selection,stage):
            value=h.clone();value.flatten(1)[:,ids]=gate[ids]*z.flatten(1)[:,ids];patched.append(value)
        combined=torch.cat(patched)
        if stage==0:combined=F.relu(b.bn2(b.conv2(combined))+t['x'].repeat(8,1,1,1))
        results.append(model.readout(combined.mean((2,3))).reshape(8,8,3).cpu().numpy())
    return np.asarray(results)

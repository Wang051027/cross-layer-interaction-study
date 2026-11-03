"""定位block2通道组合在block3两次ReLU中产生的非加和效应。

前四通道枚举全部16子集，八通道枚举空集、单点、28对及全体。
两者并集43状态。门控代数分解保留符号，门控干预另运行真实卷积。
本模块只计算，不读文件、不训练、不更新上一轮名单。
"""
from __future__ import annotations

import itertools
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

GATE_BUDGETS=(32,128)
TOP_PAIRS=3
RANDOM_REPEATS=2
MODES=('natural','hold_first','hold_second','hold_both')


def design():
    """用固定二进制顺序构造43子集、28空背景配对与两预算联合差分。"""
    masks=np.asarray(sorted({s for s in range(256) if s.bit_count()<=2}|set(range(16))|{255}))
    lookup={int(s):i for i,s in enumerate(masks)}
    bits=((masks[:,None]>>np.arange(8))&1).astype(np.int64)
    pairs=np.asarray(list(itertools.combinations(range(8),2)))
    stencils=np.asarray([[lookup[(1<<a)|(1<<b)],lookup[1<<a],lookup[1<<b],lookup[0]] for a,b in pairs])
    joint=np.zeros((2,len(masks)))
    for j,k in enumerate((4,8)):
        joint[j,lookup[(1<<k)-1]]=1.;joint[j,lookup[0]]=k-1
        for a in range(k):joint[j,lookup[1<<a]]=-1.
    assert len(masks)==43
    return dict(masks=masks,bits=bits,pairs=pairs,stencils=stencils,joint=joint,p4_indices=np.array([lookup[s] for s in range(16)]))


def differences(x,stencils):
    """按组合轴第0维进行四项差分，支持logit或每个ReLU坐标。"""
    a,b,c,d=stencils.T;return x[a]-x[b]-x[c]+x[d]


def mobius4(x):
    """四通道子集的容斥系数；这是代数展开，不是新归因定理。"""
    result=np.asarray(x).copy()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):result[mask]-=result[mask^(1<<bit)]
    return result


@torch.no_grad()
def patch_states(base,reference,channels,bits):
    """固定八通道名单内部替换，每个通道仍为完整32×32激活图。"""
    x=base.expand(len(bits),-1,-1,-1).clone()
    for j,channel in enumerate(channels):
        rows=torch.as_tensor(np.flatnonzero(bits[:,j]),device=x.device)
        x[rows,int(channel)]=reference[0,int(channel)]
    return x


@torch.no_grad()
def trace_modes(model,x):
    """自然、固定第一开关、固定第二开关、固定两者四种真实尾部计算。

    固定值取当前无通道修补状态。固定第一层后第二层自然重算，因此
    第一层干预能影响第二层；不能把其输出变化当作唯一的第一层贡献。
    """
    b=model.block3;assert isinstance(b.shortcut,nn.Identity)
    z1=b.bn1(b.conv1(x));h1=F.relu(z1);g1=z1[:1]>0
    z2=b.bn2(b.conv2(h1))+x;h2=F.relu(z2);g2=z2[:1]>0
    fixed_h1=g1*z1;fixed_z2=b.bn2(b.conv2(fixed_h1))+x
    output=lambda h:model.readout(h.mean((2,3)))
    logits=torch.stack((output(h2),output(F.relu(fixed_z2)),output(g2*z2),output(g2*fixed_z2)),1)
    return dict(x=x,z1=z1,z2=z2,h1=h1,h2=h2,g1=g1,g2=g2,logits=logits,
        r1=h1-g1*z1,r2=h2-g2*z2)


@torch.no_grad()
def pair_unit_contributions(model,traced,weight,stencils):
    """每对的两层坐标贡献：p*ΔΔr1和q*ΔΔr2，总和精确重建ΔΔ输出。

    q为标签对比读出的空间平均权重，p为第二卷积BN的伴随映射作用于
    无修补末层开关*q。第一、第二份相加精确，但依赖选定门控基线。
    """
    b=model.block3;w=torch.as_tensor(weight,dtype=traced['x'].dtype,device=traced['x'].device)
    q=(w@model.readout.weight)[:,None,None]/(traced['x'].shape[-2]*traced['x'].shape[-1])
    scale=b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps)
    p=F.conv_transpose2d(traced['g2']*q[None]*scale[None,:,None,None],b.conv2.weight,
        stride=b.conv2.stride,padding=b.conv2.padding,dilation=b.conv2.dilation,groups=b.conv2.groups)
    first=differences(traced['r1'],stencils)*p
    second=differences(traced['r2'],stencils)*q
    return torch.stack((first,second),1)


def summaries(traced,design_values,weight):
    """保存全部配对、四通道各阶、八通道三阶以上总和及四种门控结果。"""
    logits=traced['logits'].cpu().numpy();d=design_values
    pair=differences(logits,d['stencils']);joint=np.einsum('km,msc->ksc',d['joint'],logits)
    coefficients=np.stack([mobius4(logits[d['p4_indices'],j]) for j in range(4)],axis=1)
    parts=np.stack((pair[:,2]-pair[:,3],pair[:,0]-pair[:,2]),axis=1)
    jparts=np.stack((joint[:,2]-joint[:,3],joint[:,0]-joint[:,2]),axis=1)
    high=joint[1]-pair.sum(0)
    return dict(logits=logits,pair_logits=pair,joint_logits=joint,mobius4=coefficients,
        pair_stageparts=parts,joint_stageparts=jparts,higher_than_pair8=high,
        stage_flip_counts=np.stack([((traced[k]>0)!=(traced[k][:1]>0)).flatten(1).sum(-1).cpu().numpy() for k in ('z1','z2')],-1))


def select_pairs_and_gates(traced,units,pair_effects,stencils,random_seed):
    """只用半强度：按配对对比绝对值选3对，每层取最大绝对坐标贡献。

    对照从半强度实际发生开关变化的坐标抽取，相同预算，两固定随机
    重复。名单封存后不按全强度输出调整。坐标按C,H,W扁平化。
    """
    order=np.argsort(-np.abs(pair_effects),kind='stable')[:TOP_PAIRS]
    chosen=np.full((TOP_PAIRS,2,128),-1,dtype=np.int64)
    random=np.full((TOP_PAIRS,2,RANDOM_REPEATS,128),-1,dtype=np.int64)
    eligible_counts=np.zeros((TOP_PAIRS,2),dtype=np.int64)
    contributions=np.zeros((TOP_PAIRS,2,128));preactivation=np.zeros((TOP_PAIRS,2,4,128))
    rng=np.random.default_rng(random_seed)
    for rank,pair in enumerate(order):
        indices=stencils[pair][[3,1,2,0]]
        for stage,key in enumerate(('z1','z2')):
            z=traced[key][indices];eligible=torch.any((z>0)!=(traced[key][:1]>0),dim=0).flatten().cpu().numpy()
            ids=np.flatnonzero(eligible);eligible_counts[rank,stage]=len(ids)
            score=units[pair,stage].flatten().cpu().numpy()
            top=ids[np.argsort(-np.abs(score[ids]),kind='stable')[:128]]
            chosen[rank,stage,:len(top)]=top;contributions[rank,stage,:len(top)]=score[top]
            preactivation[rank,stage,:,:len(top)]=z.flatten(1).cpu().numpy()[:,top]
            for rep in range(RANDOM_REPEATS):
                sampled=rng.permutation(ids)[:128];random[rank,stage,rep,:len(sampled)]=sampled
    return dict(selected_pairs=order,selected_gates=chosen,random_gates=random,
        eligible_counts=eligible_counts,selection_contributions=contributions,selection_preactivations=preactivation)


@torch.no_grad()
def partial_holds(model,traced,stencils,selection):
    """冻结半强度选定坐标，在四状态上实际替换开关并重新计算输出。

    返回[配对3,层2,预算2,名单3,状态4,类别3]；状态顺序空、a、b、ab。
    第一层修改后实际重算卷积及第二ReLU；第二层修改保留自然上游。
    """
    result=[];b=model.block3
    for rank,pair in enumerate(selection['selected_pairs']):
        indices=stencils[pair][[3,1,2,0]];stage_values=[]
        for stage in range(2):
            h=traced['h1' if stage==0 else 'h2'][indices]
            z=traced['z1' if stage==0 else 'z2'][indices]
            gate=traced['g1' if stage==0 else 'g2'].flatten();patched=[]
            for budget in GATE_BUDGETS:
                candidates=[selection['selected_gates'][rank,stage],*selection['random_gates'][rank,stage]]
                for padded in candidates:
                    ids=padded[padded>=0][:budget];state=h.clone()
                    state.flatten(1)[:,ids]=gate[ids]*z.flatten(1)[:,ids];patched.append(state)
            combined=torch.cat(patched)
            if stage==0:combined=F.relu(b.bn2(b.conv2(combined))+traced['x'][indices].repeat(len(patched),1,1,1))
            output=model.readout(combined.mean((2,3))).reshape(2,3,4,3)
            stage_values.append(output.cpu().numpy())
        result.append(stage_values)
    return np.asarray(result)


def selected_details(traced,units,stencils,selection):
    """记录封存坐标在当前强度的符号模式与贡献，不能据此换名单。"""
    values=np.zeros((3,2,128));zvalues=np.zeros((3,2,4,128))
    for rank,pair in enumerate(selection['selected_pairs']):
        indices=stencils[pair][[3,1,2,0]]
        for stage,key in enumerate(('z1','z2')):
            ids=selection['selected_gates'][rank,stage];ids=ids[ids>=0]
            values[rank,stage,:len(ids)]=units[pair,stage].flatten().cpu().numpy()[ids]
            zvalues[rank,stage,:,:len(ids)]=traced[key][indices].flatten(1).cpu().numpy()[:,ids]
    return dict(unit_contributions=values,gate_preactivations=zvalues)

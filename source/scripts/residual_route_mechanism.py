"""中间组合残差R=H−H≤1的通道、两次ReLU与抵消机制。

基准为每个源状态自己的H≤1，不是统一空状态。精确端点分解
deltaF=J_A R + 第一ReLU余项 + 第二ReLU余项；实际固定第一开关
会重算第二开关，净作用不一定等于第一项的代数贡献。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from audit_dsprites_gradient import affine_adjoint
from layer_transfer_mechanism import differences,tail

CHANNEL_SPECS=('selected4','selected8','positive8','negative8','random0_4','random1_4','random0_8','random1_8','all32')
GATE_SPECS=('selected128','random0_128','random1_128','positive128','negative128','selected1','positive1','negative1')
EPS=1e-10


@torch.no_grad()
def trace(model,a,h,weight):
    """显式两层原运算，同时得到状态特异基准、实际四模式和精确余项。"""
    b=model.block3
    za=b.bn1(b.conv1(a));ua=F.relu(za);va=b.bn2(b.conv2(ua))+a;ea=F.relu(va)
    zh=b.bn1(b.conv1(h));uh=F.relu(zh);vh=b.bn2(b.conv2(uh))+h;eh=F.relu(vh)
    ga=za>0;gb=va>0
    fixed_u=ga*zh;vf=b.bn2(b.conv2(fixed_u))+h
    read=lambda v:model.readout(v.mean((2,3)))
    modes=torch.stack((read(eh),read(F.relu(vf)),read(gb*vh),read(gb*vf)),1)
    w=torch.as_tensor(weight,dtype=a.dtype,device=a.device)
    q=(w@model.readout.weight)[None,:,None,None].expand_as(a)/1024
    p=affine_adjoint(b.conv2,b.bn2,gb*q,ua.shape)
    gradient=affine_adjoint(b.conv1,b.bn1,ga*p,a.shape)+gb*q
    first=(uh-ga*zh)*p;second=(eh-gb*vh)*q
    linear=gradient*(h-a)
    updated_second=(F.relu(vf)-gb*vf)*q
    result=dict(a=a,h=h,za=za,va=va,zh=zh,vh=vh,uh=uh,eh=eh,ga=ga,gb=gb,q=q,p=p,
        modes=modes,baseline=read(ea),gradient=gradient,linear=linear,first=first,second=second,updated_second=updated_second)
    delta=(modes[:,0]-result['baseline'])@w
    assert float(abs(delta-linear.sum((1,2,3))-first.sum((1,2,3))-second.sum((1,2,3))).max())<1e-9
    assert float(abs((modes[:,3]-result['baseline'])@w-linear.sum((1,2,3))).max())<1e-9
    return result


def aligned_scores(target,contributions):
    """贡献轴[24,候选]投影到24条件实际修正；这是选择分数，不是因果份额。"""
    denominator=float(np.sum(target**2))
    return np.zeros(contributions.shape[1]) if denominator<1e-20 else target@contributions/denominator


def ranked(scores,eligible,budget,kind='absolute'):
    """零附近分数归零，按物理编号稳定处理平分；空名单用−1补齐。"""
    scores=np.where(abs(scores)<EPS,0,scores)
    ids=np.asarray(eligible,dtype=np.int64)
    if kind=='positive':ids=ids[scores[ids]>0];key=-scores[ids]
    elif kind=='negative':ids=ids[scores[ids]<0];key=scores[ids]
    else:key=-abs(scores[ids])
    chosen=ids[np.lexsort((ids,key))[:budget]]
    return np.pad(chosen,(0,budget-len(chosen)),constant_values=-1)


@torch.no_grad()
def single_restores(model,a,h):
    """32个物理通道逐一加入组合残差，实际运行原尾部。"""
    fields=[]
    for channel in range(32):
        value=a.clone();value[:,channel]=h[:,channel];fields.append(value)
    return tail(model,torch.cat(fields)).reshape(32,16,3)


@torch.no_grad()
def select(traced,single,weight,stencils,seed):
    """只用半强度确定通道及每层坐标；随机来自相同候选池和实际预算。"""
    w=np.asarray(weight);baseline=traced['baseline'].cpu().numpy()@w
    delta=traced['modes'][:,0].cpu().numpy()@w-baseline
    target=differences(delta,stencils)
    single_delta=single.cpu().numpy()@w-baseline[None]
    channel_score=aligned_scores(target,differences(single_delta.T,stencils))
    rng=np.random.default_rng(seed);random_ch=np.array([rng.permutation(32) for _ in range(2)])
    chosen=ranked(channel_score,np.arange(32),8);pos=ranked(channel_score,np.arange(32),8,'positive');neg=ranked(channel_score,np.arange(32),8,'negative')
    channels=np.stack([np.pad(v,(0,32-len(v)),constant_values=-1) for v in
        (chosen[:4],chosen,pos,neg,random_ch[0,:4],random_ch[1,:4],random_ch[0,:8],random_ch[1,:8],np.arange(32))])
    gates=[];gate_scores=[];counts=[]
    for name in ('first','second'):
        values=traced[name].flatten(1).cpu().numpy()
        # 按实际ReLU非线性余项幅度筛候选，去除纯舍入量级。
        remainder=(traced['uh']-traced['ga']*traced['zh']) if name=='first' else (traced['eh']-traced['gb']*traced['vh'])
        eligible=np.flatnonzero(np.max(abs(remainder.flatten(1).cpu().numpy()),axis=0)>EPS)
        score=aligned_scores(target,differences(values,stencils));counts.append(len(eligible))
        absolute=ranked(score,eligible,128);positive=ranked(score,eligible,128,'positive');negative=ranked(score,eligible,128,'negative')
        size=min(128,len(eligible));random=[np.pad(rng.permutation(eligible)[:size],(0,128-size),constant_values=-1) for _ in range(2)]
        rows=[absolute,*random,positive,negative,np.pad(absolute[:1],(0,127),constant_values=-1),np.pad(positive[:1],(0,127),constant_values=-1),np.pad(negative[:1],(0,127),constant_values=-1)]
        gates.append(rows);gate_scores.append(score)
    return dict(channels=channels,gates=np.asarray(gates),channel_score=channel_score,gate_pool_count=np.array(counts),
        selected_gate_score=np.array([[score[np.maximum(row,0)]*(row>=0) for row in rows] for score,rows in zip(gate_scores,gates)]))


@torch.no_grad()
def interventions(model,t,selection):
    """实际中间通道替换、部分开关固定和第一层后第二层重算。"""
    a,h=t['a'],t['h'];channel_fields=[]
    for padded in selection['channels']:
        ids=padded[padded>=0];erase=h.clone();erase[:,ids]=a[:,ids]
        restore=a.clone();restore[:,ids]=h[:,ids];channel_fields.extend((erase,restore))
    channel_out=tail(model,torch.cat(channel_fields)).reshape(9,2,16,3)
    b=model.block3;outputs=[];direct=[];responses=[]
    for stage in range(2):
        fields=[];summed=[]
        for padded in selection['gates'][stage]:
            ids=padded[padded>=0]
            value=t['uh' if stage==0 else 'eh'].clone()
            z=t['zh' if stage==0 else 'vh'];gate=t['ga' if stage==0 else 'gb']
            value.flatten(1)[:,ids]=gate.flatten(1)[:,ids]*z.flatten(1)[:,ids]
            fields.append(value);summed.append(t['first' if stage==0 else 'second'].flatten(1)[:,ids].sum(1))
        joined=torch.cat(fields)
        if stage==0:
            z2=b.bn2(b.conv2(joined))+h.repeat(len(fields),1,1,1)
            remainder=(F.relu(z2)-t['gb'].repeat(len(fields),1,1,1)*z2)*t['q'].repeat(len(fields),1,1,1)
            responses=((t['second'].sum((1,2,3))[None]-remainder.sum((1,2,3)).reshape(8,16))).cpu().numpy()
            joined=F.relu(z2)
        out=model.readout(joined.mean((2,3))).reshape(8,16,3)
        outputs.append(out.cpu().numpy());direct.append(torch.stack(summed).cpu().numpy())
    return dict(channel_logits=channel_out.cpu().numpy(),gate_logits=np.array(outputs),gate_direct=np.array(direct),first_second_response=responses)


@torch.no_grad()
def compute(model,a,h,weight,stencils,selection=None,seed=None,example=False):
    """完整端点分解；坐标不保存全体巨型张量，只留统计与预定例子。"""
    t=trace(model,a,h,weight);single=single_restores(model,a,h)
    if selection is None:selection=select(t,single,weight,stencils,seed)
    patch=interventions(model,t,selection)
    signed=[];gate_channel=[]
    for key in ('first','second'):
        inter=differences(t[key],stencils)
        flat=inter.flatten(1);signed.append(torch.stack((flat.clamp_min(0).sum(1),flat.clamp_max(0).sum(1)),1).cpu().numpy())
        gate_channel.append(t[key].sum((2,3)).cpu().numpy())
    linear=t['linear'].sum((2,3)).cpu().numpy()
    w=np.asarray(weight);nat=t['modes'][:,0].cpu().numpy()@w
    gate=t['modes'].cpu().numpy()@w
    # 单层干预的净移除=选中代数项+另一层响应，逐源状态核验。
    assert np.max(abs(nat[None]-patch['gate_logits'][0]@w-patch['gate_direct'][0]-patch['first_second_response']))<1e-9
    assert np.max(abs(nat[None]-patch['gate_logits'][1]@w-patch['gate_direct'][1]))<1e-9
    out=dict(modes=t['modes'].cpu().numpy(),baseline=t['baseline'].cpu().numpy(),single_logits=single.cpu().numpy(),
        linear_channel=linear,gate_channel=np.array(gate_channel),gate_signed_mass=np.array(signed).transpose(1,0,2),
        full_first_response=(t['second']-t['updated_second']).sum((1,2,3)).cpu().numpy(),
        gate_flip_counts=np.stack([((t[k]>0)!=g).flatten(1).sum(1).cpu().numpy() for k,g in (('zh',t['ga']),('vh',t['gb']))],1),
        **patch)
    assert np.max(abs((gate[:,0]-gate[:,1])-out['gate_channel'][0].sum(1)-out['full_first_response']))<1e-9
    assert np.max(abs(out['channel_logits'][-1,0]-out['baseline']))==0
    assert np.max(abs(out['channel_logits'][-1,1]-out['modes'][:,0]))==0
    fields={} if not example else {key:t[key].cpu().numpy() for key in ('za','va','zh','vh','gradient','linear','first','second','updated_second')}
    return out,selection,fields

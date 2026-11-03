"""同条件形状状态替换、通道中介修补与沿路径梯度。

源单位是四组通道，每组4×64×64标量；中介单位是整个32×32通道。
所有输出仍由冻结残差网络计算。只存梯度与实际变化方向的内积及范数
诊断，避免把几千张全层Jacobian写成数GB的冗余数组。
"""
from __future__ import annotations

import numpy as np
import torch
import dsprites_mechanism as mechanism


def pair_weights(recipient, donor):
    """任务目标是对照形状logit减原形状logit，不使用softmax概率。"""
    assert int(recipient)!=int(donor)
    weight=np.zeros(3);weight[int(donor)]=1;weight[int(recipient)]=-1
    return weight


def matched_pairs(factors, labels):
    """按大小、角度、x、y严格配对；每个条件生成六个有向形状转换。"""
    contexts={}
    for i,(row,label) in enumerate(zip(factors,labels)):
        key=tuple(map(int,row[2:]));contexts.setdefault(key,{})
        assert int(label) not in contexts[key],'Duplicate class within a condition'
        contexts[key][int(label)]=i
    pairs=[]
    for key,rows in sorted(contexts.items()):
        assert set(rows)=={0,1,2},key
        for recipient in range(3):
            for donor in range(3):
                if donor!=recipient:pairs.append([rows[recipient],rows[donor]])
    return np.asarray(pairs,dtype=np.int64)


def replace_groups(recipient,donor,bits,alpha):
    """四组按通道索引0..3、4..7、8..11、12..15固定，覆盖完整源切口。

    source=(1−α*mask)*recipient+α*mask*donor；全强度全组替换精确等于
    donor，不用“recipient+(donor-recipient)”累积端点舍入误差。
    """
    masks=bits.repeat_interleave(4,dim=1)[:,:,None,None]
    return (1-alpha*masks)*recipient+alpha*masks*donor


def state_effects(model,states,source_delta,recipient_label,donor_label):
    """返回logit、状态，以及源组和中介通道的方向梯度。

    中介基线项为Σ_xy J0,cxy*(hS−h0)cxy，状态项改用JS。二者差含
    真实方向，不能仅凭梯度范数断言任务效应。梯度变化指标使用固定的
    donor-recipient目标，分别在整个源张量和整个中介张量上求范数。
    """
    leaf=states.detach().clone().requires_grad_(True)
    middle=model.block2(leaf);logits=mechanism.tail_from_middle(model,middle)
    middle_delta=(middle-middle[:1]).detach()
    source_gradients=[];middle_gradients=[]
    src_effect=[];base_effect=[];state_effect=[]
    for category in range(3):
        sg,mg=torch.autograd.grad(logits[:,category].sum(),(leaf,middle),retain_graph=category<2)
        source_gradients.append(sg.detach());middle_gradients.append(mg.detach())
        src_effect.append((sg*source_delta).reshape(len(states),4,4,*states.shape[-2:]).sum((2,3,4)).detach())
        base_effect.append((mg[:1]*middle_delta).sum((2,3)).detach())
        state_effect.append((mg*middle_delta).sum((2,3)).detach())
    source_g=source_gradients[int(donor_label)]-source_gradients[int(recipient_label)]
    middle_g=middle_gradients[int(donor_label)]-middle_gradients[int(recipient_label)]
    drift=[];baseline_norm=[]
    for grad in (source_g,middle_g):
        g=grad.flatten(1);base=torch.linalg.vector_norm(g[0]);norm=torch.linalg.vector_norm(g,dim=-1)
        change=torch.linalg.vector_norm(g-g[:1],dim=-1)/base.clamp_min(1e-30)
        cosine=(g*g[:1]).sum(-1)/(norm*base).clamp_min(1e-30)
        drift.append(torch.stack((change,cosine,norm),-1));baseline_norm.append(float(base))
    return dict(natural=logits.detach(),middle=middle.detach(),
        source_effect=torch.stack(src_effect,-1),baseline_middle_effect=torch.stack(base_effect,-1),
        state_middle_effect=torch.stack(state_effect,-1),gradient_drift=torch.stack(drift,1),
        baseline_gradient_norm=np.asarray(baseline_norm))


@torch.no_grad()
def singleton_channels(model,middle):
    """仅查询最终状态的32通道单独恢复与移植，共64个尾部输出。

    middle需包含无干预和当前全源组状态，返回[32通道,操作2,类别3]。
    不查询其余14组合的单点修补，也不把本参照称为最优联合集合。
    """
    assert len(middle)==2 and middle.shape[1]==32
    restored=middle[1:2].expand(32,-1,-1,-1).clone()
    transplanted=middle[:1].expand(32,-1,-1,-1).clone()
    channel=torch.arange(32,device=middle.device)
    restored[channel,channel]=middle[0,channel]
    transplanted[channel,channel]=middle[1,channel]
    output=mechanism.tail_from_middle(model,torch.cat((restored,transplanted)))
    return output.reshape(2,32,3).transpose(0,1)


@torch.no_grad()
def patch_channels(model,middle,selections):
    """真实联合修补，返回[方案,恢复或移植,组合16,类别3]。

    通道索引直接切第二轴，不与旧扁平标量坐标混用。每4套集合一批，
    运行完整block3及读出；不能把单点输出相加代替这一步。
    """
    baseline=middle[:1].expand_as(middle);results=[]
    for start in range(0,len(selections),4):
        group=selections[start:start+4];states=[]
        for channels in group:
            restored=middle.clone();restored[:,channels]=baseline[:,channels]
            transplanted=baseline.clone();transplanted[:,channels]=middle[:,channels]
            states.extend((restored,transplanted))
        combined=torch.cat(states)
        output=torch.cat([mechanism.tail_from_middle(model,x) for x in combined.split(64)])
        results.append(output.reshape(len(group),2,len(middle),3))
    return torch.cat(results)


def endpoint_scores(natural,singleton,weight):
    """单点任务收益：E²−[R²+(T−E)²]/2；任务目标固定为有向形状对比。"""
    origin=natural[0]@weight;effect=(natural[-1]@weight)-origin
    response=singleton@weight-origin
    return effect**2-.5*(response[:,0]**2+(response[:,1]-effect)**2)


def predicted_singletons(natural,baseline_effect,state_effect):
    """基线或状态梯度只改变恢复预测；移植从原状态出发均用基线导数。"""
    baseline=natural[0];final=natural[-1]
    transplant=baseline[None]+baseline_effect[-1]
    return np.stack((np.stack((final[None]-baseline_effect[-1],transplant),1),
                     np.stack((final[None]-state_effect[-1],transplant),1)))


def rank_channels(score):
    """按分数稳定排序，原计算值全部保存；不按新图调并列阈值。"""
    return np.argsort(-score,kind='stable')


def integrate_source_path(model,recipient,donor,recipient_label,donor_label,counts=(8,32)):
    """全源状态直线路径的中点积分；样本数事先固定，不按误差加点。

    分组方向导数的积分和应重建任务对比的端点变化。输出保存每个
    采样点的原logit及四组方向导数，供独立伴随映射核验。
    """
    positions=np.concatenate([(np.arange(n)+.5)/n for n in counts])
    alpha=torch.as_tensor(positions,dtype=recipient.dtype,device=recipient.device)[:,None,None,None]
    leaf=((1-alpha)*recipient+alpha*donor).detach().requires_grad_(True)
    logits=mechanism.tail_from_middle(model,model.block2(leaf))
    target=logits[:,int(donor_label)]-logits[:,int(recipient_label)]
    grad=torch.autograd.grad(target.sum(),leaf)[0]
    directional=(grad*(donor-recipient)).reshape(len(positions),4,4,*leaf.shape[-2:]).sum((2,3,4))
    groups=[];offset=0
    for n in counts:groups.append(directional[offset:offset+n].mean(0));offset+=n
    return dict(positions=positions,path_logits=logits.detach().cpu().numpy(),
        path_group_derivative=directional.detach().cpu().numpy(),integrated_groups=torch.stack(groups).detach().cpu().numpy())


def known_answers():
    """检查配对、四组覆盖、任务收益和仿射修补的可手算答案。"""
    r=torch.arange(16*3*3,dtype=torch.float64).reshape(1,16,3,3)
    d=-r-2;bits,_,_=mechanism.design();bt=torch.as_tensor(bits,dtype=r.dtype)
    altered=replace_groups(r,d,bt,1.)
    assert torch.equal(altered[0],r[0]) and torch.equal(altered[-1],d[0])
    assert float(mechanism.pair_difference(altered,mechanism.design()[1]).abs().max())==0
    weight=np.array([-1.,1.,0.]);natural=np.array([[2.,0.,0.],[0.,2.,0.]])
    singleton=np.array([[[2.,0.,0.],[0.,2.,0.]],[[0.,2.,0.],[2.,0.,0.]]])
    np.testing.assert_array_equal(endpoint_scores(natural,singleton,weight),[16.,0.])
    effects=np.zeros((2,32,3));effects[1,0]=natural[1]-natural[0]
    predicted=predicted_singletons(natural,effects,effects)
    for method in range(2):np.testing.assert_allclose(predicted[method,0],singleton[0],atol=0,rtol=0)
    return dict(status='PASS',group_source_interaction=0.,perfect_channel_utility=16.,unused_channel_utility=0.)

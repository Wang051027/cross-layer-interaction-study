"""源层四组组合到中间层再到输出的传递计算。

源组仍为block1的四个连续4通道组。全16状态在block2产生的组合项
由容斥分离；擦除组合项保留原空状态和单组状态，完整block3再运行。
本模块不读写数据、不训练、不选择新的输入或通道名单。
"""
from __future__ import annotations

import itertools
import numpy as np
import torch
from torch.nn import functional as F

MODES=('natural','upstream_fixed','downstream_fixed','both_fixed')
SPECS=(('selected',4),('selected',8),('random0',4),('random1',4),('random0',8),('random1',8),('all',32))
ANCHORS=(0,1,2,4,8)


def design():
    """二进制顺序及容斥/重建矩阵，24差分包括全部其余两组背景。"""
    masks=np.arange(16);bits=(masks[:,None]>>np.arange(4))&1
    order=np.array([int(s).bit_count() for s in masks]);mobius=np.zeros((16,16));zeta=np.zeros((16,16))
    for s in range(16):
        for t in range(16):
            if t&s==t:mobius[s,t]=(-1)**(int(s).bit_count()-int(t).bit_count());zeta[s,t]=1
    stencils=[];metadata=[]
    for a,b in itertools.combinations(range(4),2):
        for background in range(16):
            if background&((1<<a)|(1<<b)):continue
            stencils.append([background|(1<<a)|(1<<b),background|(1<<a),background|(1<<b),background])
            metadata.append([a,b,background])
    joint=np.zeros(16);joint[15]=1;joint[0]=3;joint[list(ANCHORS[1:])]=-1
    return dict(masks=masks,bits=bits,order=order,mobius=mobius,zeta=zeta,stencils=np.array(stencils),metadata=np.array(metadata),joint=joint)


def differences(value,stencils):
    """第一轴为16状态；四项差分与稠密矩阵审计分别实现。"""
    a,b,c,d=stencils.T
    return value[a]-value[b]-value[c]+value[d]


def coefficients(value):
    """二进制蝶形容斥，支持Torch全层张量或NumPy读出。"""
    result=value.clone() if isinstance(value,torch.Tensor) else value.copy()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):result[mask]=result[mask]-result[mask^(1<<bit)]
    return result


@torch.no_grad()
def lower_fields(middle,d):
    """重建不超过1/2/3阶的中间场；精确保留空及单组原值。

    删去高阶项后可以出现负值，这是内部计算反事实。另提供非负
    截断补充，但截断会重新引入组合项，不能称为纯高阶擦除。
    """
    co=coefficients(middle);zeta=torch.as_tensor(d['zeta'],dtype=middle.dtype,device=middle.device)
    fields=[]
    for degree in (1,2,3):
        kept=co.clone();kept[torch.as_tensor(d['order']>degree,device=middle.device)]=0
        value=(zeta@kept.flatten(1)).reshape_as(middle)
        value[list(ANCHORS)]=middle[list(ANCHORS)];fields.append(value)
    return co,torch.stack(fields)


@torch.no_grad()
def trace_modes(model,source):
    """四个真实门控网络：两模块自然/固定的2×2组合。

    基准是无源替换状态。固定block3仍可传递自然block2的高阶项；
    只有block2及block3均固定时，源到输出整体才成为仿射映射。
    """
    blocks=(model.block2,model.block3);natural=[];gates=[];value=source
    for b in blocks:
        z1=b.bn1(b.conv1(value));h1=F.relu(z1)
        z2=b.bn2(b.conv2(h1))+b.shortcut(value);value=F.relu(z2)
        natural.extend((z1,z2));gates.extend((z1[:1]>0,z2[:1]>0))
    outputs=[];middles=[]
    for up,down in ((False,False),(True,False),(False,True),(True,True)):
        value=source
        for stage,(b,fixed) in enumerate(zip(blocks,(up,down))):
            z1=b.bn1(b.conv1(value));h1=gates[2*stage]*z1 if fixed else F.relu(z1)
            z2=b.bn2(b.conv2(h1))+b.shortcut(value);value=gates[2*stage+1]*z2 if fixed else F.relu(z2)
            if stage==0:middles.append(value)
        outputs.append(model.readout(value.mean((2,3))))
    return dict(logits=torch.stack(outputs,1),middle=middles[0],upstream_fixed_middle=middles[1],z=natural)


@torch.enable_grad()
def state_gradients(model,source,weight,direction):
    """原网络自动微分：保存源方向导数、状态梯度漂移及中间层梯度。

    参数冻结，梯度是任务logit对激活的导数，不是训练损失梯度。
    批次行相互独立；零基准范数时不解释相对漂移或余弦。
    """
    leaf=source.detach().clone().requires_grad_(True);middle=model.block2(leaf)
    logits=model.readout(model.block3(middle).mean((2,3)))
    w=torch.as_tensor(weight,dtype=source.dtype,device=source.device)
    sg,mg=torch.autograd.grad((logits*w).sum(),(leaf,middle))
    drift=[];base_norm=[]
    for grad in (sg,mg):
        value=grad.flatten(1);norm=torch.linalg.vector_norm(value,dim=1);base=norm[0]
        relative=torch.linalg.vector_norm(value-value[:1],dim=1)/base.clamp_min(1e-30)
        cosine=(value*value[:1]).sum(1)/(norm*base).clamp_min(1e-30)
        if float(base)<1e-20:relative*=0;cosine*=0
        drift.append(torch.stack((relative,cosine,norm),1));base_norm.append(float(base))
    derivative=(sg*direction).reshape(16,4,4,64,64).sum((2,3,4))
    return dict(logits=logits.detach(),source_gradient=sg.detach(),middle_gradient=mg.detach(),
        directional_source=derivative.detach(),gradient_drift=torch.stack(drift,1).detach(),baseline_gradient_norm=np.array(base_norm))


@torch.no_grad()
def tail(model,middle):
    """保持原block3.forward，批次64以限制显存。"""
    return torch.cat([model.readout(model.block3(x).mean((2,3))) for x in middle.split(64)])


@torch.no_grad()
def calculate(model,source,weight,direction,selected,random,d,example=False):
    """计算层间传递、分阶擦除、既有名单擦除/恢复以及等范数排列对照。"""
    trace=trace_modes(model,source);middle=trace['middle'];co,lower=lower_fields(middle,d)
    grad=state_gradients(model,source,weight,direction);mg=grad['middle_gradient']
    checks={}
    def check(name,value):
        error=float(torch.max(torch.abs(value))) if isinstance(value,torch.Tensor) else float(np.max(abs(value)))
        assert error<1e-9,(name,error);checks[name]=error
    check('original_gradient_forward',grad['logits']-trace['logits'][:,0])
    interaction=differences(middle,d['stencils'])
    context_gradient=mg[torch.as_tensor(d['stencils'][:,3],device=source.device)]
    baseline_projection=(interaction*mg[:1]).sum((2,3))
    context_projection=(interaction*context_gradient).sum((2,3))
    co_projection=(co*mg[:1]).sum((2,3))
    lower_logits=tail(model,lower.reshape(-1,32,32,32)).reshape(3,16,3)
    additive=lower[0];residue=middle-additive
    selections=[];patched=[];negative=[]
    for method,budget in SPECS:
        ids=np.arange(32) if method=='all' else selected[:budget] if method=='selected' else random[int(method[-1]),:budget]
        selections.append(np.pad(ids,(0,32-len(ids)),constant_values=-1))
        erased=middle.clone();erased[:,ids]=additive[:,ids]
        restored=additive.clone();restored[:,ids]=middle[:,ids]
        patched.extend((erased,restored));negative.append([(x<0).flatten(1).double().mean(1) for x in (erased,restored)])
    patched_logits=tail(model,torch.cat(patched)).reshape(len(SPECS),2,16,3)
    shuffled=torch.stack([additive+residue[:,order] for order in random])
    permuted_logits=tail(model,shuffled.reshape(-1,32,32,32)).reshape(2,16,3)
    rectified_logits=tail(model,F.relu(additive))
    check('anchor_preservation',lower[:,list(ANCHORS)]-middle[list(ANCHORS)][None])
    check('middle_additive_high_orders',coefficients(additive)[torch.as_tensor(d['order']>=2,device=source.device)])
    check('upstream_fixed_high_orders',coefficients(trace['upstream_fixed_middle'])[torch.as_tensor(d['order']>=2,device=source.device)])
    check('both_fixed_high_orders',coefficients(trace['logits'][:,3])[torch.as_tensor(d['order']>=2,device=source.device)])
    check('all_erasure',patched_logits[-1,0]-lower_logits[0]);check('all_restore',patched_logits[-1,1]-trace['logits'][:,0])
    w=torch.as_tensor(weight,dtype=source.dtype,device=source.device)
    down_co=coefficients(trace['logits'][:,2])@w
    check('affine_transmission',down_co[1:]-co_projection.sum(1)[1:])
    check('permutation_norm',((shuffled-additive).flatten(2).norm(dim=2)-residue.flatten(1).norm(dim=1)[None])/residue.flatten(1).norm(dim=1).clamp_min(1)[None])
    result=dict(logits=trace['logits'].cpu().numpy(),lower_logits=lower_logits.cpu().numpy(),patched_logits=patched_logits.cpu().numpy(),
        permuted_logits=permuted_logits.cpu().numpy(),rectified_logits=rectified_logits.cpu().numpy(),selections=np.asarray(selections),
        source_mobius_channel_norm=coefficients(source).flatten(2).norm(dim=2).cpu().numpy(),
        middle_mobius_channel_norm=co.flatten(2).norm(dim=2).cpu().numpy(),
        conditional_middle_channel_norm=interaction.flatten(2).norm(dim=2).cpu().numpy(),
        baseline_mobius_channel_projection=co_projection.cpu().numpy(),
        conditional_baseline_channel_projection=baseline_projection.cpu().numpy(),conditional_context_channel_projection=context_projection.cpu().numpy(),
        gradient_drift=grad['gradient_drift'].cpu().numpy(),baseline_gradient_norm=grad['baseline_gradient_norm'],
        directional_source=grad['directional_source'].cpu().numpy(),
        gate_flip_counts=np.stack([((z>0)!=(z[:1]>0)).flatten(1).sum(1).cpu().numpy() for z in trace['z']],1),
        lower_negative_fraction=(lower<0).flatten(2).double().mean(2).cpu().numpy(),
        patched_negative_fraction=torch.stack([torch.stack(v) for v in negative]).cpu().numpy(),
        permuted_negative_fraction=(shuffled<0).flatten(2).double().mean(2).cpu().numpy())
    fields={} if not example else dict(source=source.cpu().numpy(),natural_middle=middle.cpu().numpy(),middle_coefficients=co.cpu().numpy(),
        additive_middle=additive.cpu().numpy(),source_gradient=grad['source_gradient'].cpu().numpy(),middle_gradient=mg.cpu().numpy())
    return result,checks,fields

"""在验证图上比较 block3 反向卷积、逐样本自动微分与小扰动差商，核对雅可比矩阵的批次轴。"""
from __future__ import annotations

import json
import time
import dsprites_teacher as teacher
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_mechanism as mechanism
import gradient_mechanism as gradient


@torch.no_grad()
def manual_middle_gradient(model, middle, category):
    """沿读出、池化和输出门控反传，合并两次转置卷积与恒等旁路的梯度。BatchNorm 使用评价模式下的仿射参数。"""
    block=model.block3
    z1=block.bn1(block.conv1(middle));h1=F.relu(z1)
    z2=block.bn2(block.conv2(h1))+middle
    channels,height,width=middle.shape[1:]
    terminal=model.readout.weight[category][None,:,None,None].expand(len(middle),channels,height,width)/(height*width)
    outgoing=terminal*(z2>0)
    def adjoint(conv,bn,value):
        """卷积加BN的伴随线性映射，不包含偏置。"""
        scale=bn.weight/torch.sqrt(bn.running_var+bn.eps)
        return F.conv_transpose2d(value*scale[None,:,None,None],conv.weight,padding=conv.padding)
    return adjoint(block.conv1,block.bn1,adjoint(block.conv2,block.bn2,outgoing)*(z1>0))+outgoing


def check(seed,device):
    """每个教师只用旧验证集第一图，所有类别和两类激活导数都核验。"""
    images,labels=teacher.data('validation',device=device)
    model=teacher.load_teacher(seed,device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    with np.load(teacher.RUN/f'development/seed_{seed}/discovery.npz',allow_pickle=False) as old:
        nodes=old['source_nodes'].copy();pool=old['pool'].copy()
    bits,stencil,_=mechanism.design();bt=torch.as_tensor(bits,dtype=torch.float64,device=device)
    source=mechanism.source_activation(model,images[:1]);altered=mechanism.changed_sources(source,nodes,bt,.5)
    started=time.perf_counter();v=gradient.state_gradients(model,altered,pool,nodes)
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    expected=mechanism.trace(model,altered)
    errors=dict(forward=float((v['logits']-expected['output']).abs().max()))
    errors['manual_middle']=max(float((manual_middle_gradient(model,v['middle'],c).flatten(1)[:,pool]-v['middle_jacobian'][:,c]).abs().max()) for c in range(3))
    # 对组合13单独求梯度，验证批次求和并未将其他样本的导数混入。
    solo=gradient.state_gradients(model,altered[13:14],pool,nodes)
    errors['single_batch']=max(float((solo[k][0]-v[k][13]).abs().max()) for k in ('middle_jacobian','source_jacobian'))
    # 沿干预强度方向求差商，此方向与实际研究的源衰减完全一致。
    eps=1e-6
    plus=mechanism.trace(model,mechanism.changed_sources(source,nodes,bt,.5+eps))['output'].cpu().numpy()
    minus=mechanism.trace(model,mechanism.changed_sources(source,nodes,bt,.5-eps))['output'].cpu().numpy()
    numeric=(plus-minus)/(2*eps)
    analytic=np.einsum('scj,sj->sc',v['source_jacobian'].cpu().numpy(),-source.flatten()[nodes].cpu().numpy()[None]*bits)
    errors['strength_difference']=float(np.max(np.abs(numeric-analytic)))
    frozen=gradient.frozen_gate_output(model,altered,expected)
    errors['frozen_gate_interaction']=float(mechanism.pair_difference(frozen,stencil).abs().max())
    errors['frozen_gate_baseline']=float((frozen[0]-v['logits'][0]).abs().max())
    assert max(value for key,value in errors.items() if key!='strength_difference')<1e-10,errors
    assert errors['strength_difference']<2e-7,errors
    return dict(seed=seed,status='PASS',errors=errors,gradient16_seconds=elapsed)


if __name__=='__main__':
    device=teacher.configure()
    print(json.dumps(dict(known_answers=gradient.known_answers(),teachers=[check(s,device) for s in teacher.SEEDS]),ensure_ascii=False),flush=True)

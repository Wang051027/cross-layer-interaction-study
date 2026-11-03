"""形状替换的前置数值检验，只读取旧开发图，不读取新评价输出。

验证新通道单位的真实forward钩子、手推梯度内积和路径积分的仿射答案。
这里检验代码正确性，不根据测试结果改变新面板或选择规则。
"""
from __future__ import annotations

import json
import dsprites_teacher as teacher
import numpy as np
import torch
from torch import nn
import dsprites_mechanism as mechanism
import shape_donor_mechanism as donor
from audit_dsprites_gradient import independent_states_and_gradients


@torch.no_grad()
def hook_reference(model,recipient_image,donor_image,bits,alpha,mode=None,channels=None):
    """在原model.forward两个出口独立改状态，不使用新的替换或修补函数。"""
    saved=[]
    handle=model.block2.register_forward_hook(lambda module,args,output:saved.append(output.detach().clone()))
    model(recipient_image);handle.remove();baseline=saved[0]
    donor_source=model.block1(model.stem(donor_image));handles=[]
    masks=torch.as_tensor(np.repeat(bits,4,axis=1),dtype=recipient_image.dtype,device=recipient_image.device)[:,:,None,None]
    def source_hook(module,args,output):return output*(1-alpha*masks)+donor_source*(alpha*masks)
    def middle_hook(module,args,output):
        reference=baseline.expand_as(output)
        if mode=='restore':patched=output.clone();patched[:,channels]=reference[:,channels]
        else:patched=reference.clone();patched[:,channels]=output[:,channels]
        return patched
    try:
        handles.append(model.block1.register_forward_hook(source_hook))
        if mode is not None:handles.append(model.block2.register_forward_hook(middle_hook))
        return model(recipient_image.expand(len(bits),-1,-1,-1)).detach()
    finally:
        for handle in handles:handle.remove()


def affine_integration(device):
    """人工仿射尾部的积分在8和32点都应精确等于端点差。"""
    model=nn.Module();model.block2=nn.Identity();model.block3=nn.Identity();model.readout=nn.Linear(16,3).double().to(device)
    with torch.no_grad():
        model.readout.weight.copy_(torch.arange(48,dtype=torch.float64,device=device).reshape(3,16)/31)
        model.readout.bias.zero_()
    r=torch.arange(16,dtype=torch.float64,device=device).reshape(1,16,1,1)/7;d=-r+.2
    result=donor.integrate_source_path(model,r,d,0,1)
    expected=float((model.readout((d-r).flatten(1)))[0,1]-(model.readout((d-r).flatten(1)))[0,0])
    error=float(np.max(np.abs(result['integrated_groups'].sum(-1)-expected)))
    assert error<1e-12,error
    return dict(status='PASS',maximum_error=error)


def check(seed,device):
    """六教师同用旧开发集第一组匹配条件，独立检查两个状态和所有类别。"""
    pixels,labels=teacher.data('development',device=device)
    factors=np.load(teacher.DATA/'development_factors.npy',allow_pickle=False)
    pairs=donor.matched_pairs(factors,labels.cpu().numpy());r,d=pairs[0]
    model=teacher.load_teacher(seed,device)
    for p in model.parameters():p.requires_grad_(False)
    bits,stencil,_=mechanism.design();bt=torch.as_tensor(bits,dtype=torch.float64,device=device)
    sr=mechanism.source_activation(model,pixels[r:r+1]);sd=mechanism.source_activation(model,pixels[d:d+1])
    states=donor.replace_groups(sr,sd,bt,.5)
    v=donor.state_effects(model,states,sd-sr,int(labels[r]),int(labels[d]))
    output,middle,mg,sg,_=independent_states_and_gradients(model,states)
    errors=dict(manual_forward=float((v['natural']-output).abs().max()),
                hook_forward=float((v['natural']-hook_reference(model,pixels[r:r+1],pixels[d:d+1],bits,.5)).abs().max()))
    mg=mg.reshape(16,3,32,32,32);sg=sg.reshape(16,3,16,64,64);delta=middle-middle[:1]
    base=(mg[:1]*delta[:,None]).sum((3,4)).permute(0,2,1)
    state=(mg*delta[:,None]).sum((3,4)).permute(0,2,1)
    src=(sg*(sd-sr)[:,None]).reshape(16,3,4,4,64,64).sum((3,4,5)).permute(0,2,1)
    for key,manual in [('source_effect',src),('baseline_middle_effect',base),('state_middle_effect',state)]:
        errors[key]=float((v[key]-manual).abs().max())
    selected=[np.array([0,7,12,31]),np.arange(32)]
    patched=donor.patch_channels(model,middle,selected)
    for j,channels in enumerate(selected):
        for op,mode in enumerate(('restore','transplant')):
            reference=hook_reference(model,pixels[r:r+1],pixels[d:d+1],bits,.5,mode,channels)
            errors[f'hook_{mode}_{len(channels)}']=float((patched[j,op]-reference).abs().max())
    singleton=donor.singleton_channels(model,middle[[0,15]])
    for op,mode in enumerate(('restore','transplant')):
        reference=hook_reference(model,pixels[r:r+1],pixels[d:d+1],bits,.5,mode,[7])[-1]
        errors[f'singleton_{mode}']=float((singleton[7,op]-reference).abs().max())
    complete=donor.replace_groups(sr,sd,bt,1.)
    errors['full_donor']=float((mechanism.trace(model,complete)['output'][-1]-model(pixels[d:d+1])[0]).abs().max())
    assert max(errors.values())<1e-10,errors
    return dict(seed=seed,status='PASS',errors=errors)


if __name__=='__main__':
    device=teacher.configure()
    print(json.dumps(dict(known_answers=donor.known_answers(),affine_integral=affine_integration(device),
        teachers=[check(s,device) for s in teacher.SEEDS]),ensure_ascii=False),flush=True)

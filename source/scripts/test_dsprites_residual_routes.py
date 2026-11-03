"""新通道与开关路由的前置检查，使用旧面板预定案例。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import residual_route_mechanism as core
from residual_route_path import integrate
from layer_transfer_mechanism import design
from test_dsprites_layer_transfer import independent_lower
from shape_donor_mechanism import replace_groups,pair_weights


@torch.no_grad()
def original_holds(model,values,base1,base2,ids1=None,ids2=None):
    """在原mid/finish注入where开关，再调用原block3.forward；最终恢复方法。"""
    b=model.block3;mid=b.mid;finish=b.finish
    def held_mid(x):
        natural=mid(x)
        if ids1 is not None:
            natural=natural.clone();z=b.bn1(b.conv1(x));natural.flatten(1)[:,ids1]=torch.where(base1.flatten(1)[:,ids1],z.flatten(1)[:,ids1],0)
        return natural
    def held_finish(u,residual):
        natural=finish(u,residual)
        if ids2 is not None:
            natural=natural.clone();z=b.bn2(b.conv2(u))+residual;natural.flatten(1)[:,ids2]=torch.where(base2.flatten(1)[:,ids2],z.flatten(1)[:,ids2],0)
        return natural
    b.mid=held_mid;b.finish=held_finish
    try:return model.readout(b(values).mean((2,3)))
    finally:delattr(b,'mid');delattr(b,'finish')


def known():
    """一个关闭第一开关的直接项被末层响应完全抵消的已知答案。"""
    a=.4;h=.1;z=lambda x:x-.3;v=lambda x:2*max(z(x),0)+x-.5
    ga=z(a)>0;gb=v(a)>0;delta=max(v(h),0)-max(v(a),0)
    linear=(2*ga+1)*gb*(h-a);first=2*gb*(max(z(h),0)-ga*z(h));second=max(v(h),0)-gb*v(h)
    assert np.max(abs(np.array([delta,linear,first,second])-[-.1,-.9,.4,.4]))<1e-12
    held=max(2*ga*z(h)+h-.5,0);assert abs(held-max(v(h),0))<1e-12
    return dict(delta=delta,linear=linear,first=first,second=second,actual_first_fix_change=0.)


def preflight():
    path=teacher.BASE/'reports/dsprites_residual_routes_preflight_20261013.json';assert not path.exists()
    device=teacher.configure();d=design();panel=np.load(teacher.BASE/'runs/dsprites_shape_donor_20261010/panel/data.npz');rows=[]
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        errors={}
        def check(key,a,b):
            aa=np.asarray(a);bb=np.asarray(b);assert aa.shape==bb.shape;(error,)= (float(np.max(abs(aa-bb))),)
            assert error<1e-9,(key,error);errors[key]=max(errors.get(key,0.),error)
        with torch.no_grad():source=model.block1(model.stem(torch.as_tensor(panel['images'][:2,None],device=device,dtype=torch.float64)))
        for alpha in (.5,1.):
            src=replace_groups(source[:1],source[1:],torch.as_tensor(d['bits'],device=device),alpha)
            with torch.no_grad():h=model.block2(src)
            _,low=independent_lower(h.cpu().numpy());a=torch.as_tensor(low[0],device=device);weight=pair_weights(0,1)
            out,selection,fields=core.compute(model,a,h,weight,d['stencils'],seed=seed,example=True)
            t=core.trace(model,a,h,weight);allids=np.arange(32768)
            for mode,(i1,i2) in enumerate(((None,None),(allids,None),(None,allids),(allids,allids))):
                check('original_modes',out['modes'][:,mode],original_holds(model,h,t['ga'],t['gb'],i1,i2).cpu().numpy())
            leaf=a.clone().requires_grad_(True);f=model.readout(model.block3(leaf).mean((2,3)))@torch.as_tensor(weight,device=device,dtype=torch.float64)
            check('baseline_autograd',fields['gradient'],torch.autograd.grad(f.sum(),leaf)[0].cpu().numpy())
            for stage in range(2):
                ids=selection['gates'][stage,0];ids=ids[ids>=0]
                ref=original_holds(model,h,t['ga'],t['gb'],ids if stage==0 else None,ids if stage==1 else None)
                check('partial_original',out['gate_logits'][stage,0],ref.cpu().numpy())
            check('anchors_unchanged',out['modes'][[0,1,2,4,8],0],out['baseline'][[0,1,2,4,8]])
            if seed==teacher.SEEDS[0]:
                for mask in (3,15):
                    result=integrate(model,a[mask:mask+1],h[mask:mask+1],weight);errors['path_integral']=max(errors.get('path_integral',0.),float(result['error']))
        rows.append(dict(seed=seed,status='PASS',maximum_errors=errors));print(json.dumps(rows[-1]),flush=True)
    result=dict(status='PASS',utc=teacher.now(),known=known(),teachers=rows);teacher.dump(path,result)


if __name__=='__main__':preflight()

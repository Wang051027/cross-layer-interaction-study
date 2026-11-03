"""层间传递前置检查：只读取旧开发集第一形状配对。

原模块forward与where门控独立核对生产实现；手算例子区分上游产生、
下游产生和相互抵消。审计可复用参考forward，不调用新数值核心。
"""
from __future__ import annotations

import dsprites_teacher as teacher
import numpy as np
import torch
import layer_transfer_mechanism as core
from shape_donor_mechanism import matched_pairs,replace_groups,pair_weights
from audit_dsprites_gradient import independent_states_and_gradients


def dense_design():
    """独立容斥和重建矩阵，不使用生产蝶形变换。"""
    m=np.zeros((16,16));z=np.zeros((16,16))
    members=[{j for j in range(4) if s&(1<<j)} for s in range(16)]
    for i,a in enumerate(members):
        for j,b in enumerate(members):
            if b<=a:m[i,j]=(-1)**(len(a)-len(b));z[i,j]=1
    return m,z,np.array([len(s) for s in members])


@torch.no_grad()
def original_modes(model,source):
    """在原block2/3.forward中覆盖门控，仍执行实际卷积、BN和残差。"""
    b2,b3=model.block2,model.block3;value=source[:1];gates=[]
    for b in (b2,b3):
        z1=b.bn1(b.conv1(value));h1=torch.where(z1>0,z1,torch.zeros_like(z1))
        z2=b.bn2(b.conv2(h1))+b.shortcut(value);value=torch.where(z2>0,z2,torch.zeros_like(z2))
        gates.extend((z1>0,z2>0))
    outputs=[];middle=[]
    for up,down in ((False,False),(True,False),(False,True),(True,True)):
        try:
            for stage,(b,fixed) in enumerate(zip((b2,b3),(up,down))):
                def mid(x,block=b,frozen=fixed,gate=gates[2*stage]):
                    z=block.bn1(block.conv1(x));return torch.where(gate if frozen else z>0,z,torch.zeros_like(z))
                def finish(h,residual,block=b,frozen=fixed,gate=gates[2*stage+1]):
                    z=block.bn2(block.conv2(h))+residual;return torch.where(gate if frozen else z>0,z,torch.zeros_like(z))
                b.mid=mid;b.finish=finish
            h=b2(source);middle.append(h.detach());outputs.append(model.readout(b3(h).mean((2,3))))
        finally:
            for b in (b2,b3):delattr(b,'mid');delattr(b,'finish')
    return torch.stack(outputs,1),middle[0],middle[1]


@torch.no_grad()
def original_image(model,image,source):
    """用原model.forward的源出口hook核对整条图像到输出路线。"""
    handle=model.block1.register_forward_hook(lambda module,args,output:source)
    try:return model(image.expand(len(source),-1,-1,-1))
    finally:handle.remove()


def independent_lower(middle):
    """NumPy稠密容斥重建三种阶次截断，原空/单组值精确保留。"""
    m,z,degree=dense_design();values=np.asarray(middle);co=np.tensordot(m,values,axes=(1,0));fields=[]
    for k in (1,2,3):
        selected=co.copy();selected[degree>k]=0
        h=np.tensordot(z,selected,axes=(1,0));h[[0,1,2,4,8]]=values[[0,1,2,4,8]];fields.append(h)
    return co,np.stack(fields)


def known_answers():
    """下游线性仍能传递上游非加和，擦除交互也可能让输出交互变强。"""
    d=core.design();a,b=d['bits'][:,:2].T
    pair=np.array([3,1,2,0]);contrast=lambda x:float(x[pair]@np.array([1.,-1.,-1.,1.]))
    upstream=np.maximum(a+b-.5,0);assert contrast(upstream)==.5 and contrast(2*upstream)==1
    linear=a+b;assert contrast(linear)==0 and contrast(np.maximum(linear-.5,0))==.5
    curved=linear-.7*a*b;natural=np.maximum(curved-.5,0);erased=np.maximum(linear-.5,0)
    assert np.isclose(contrast(natural),-.2) and np.isclose(contrast(erased),.5)
    values=torch.as_tensor(np.stack((1+linear,curved,2*a*b+3*a*b*d['bits'][:,2]),1)[:,:,None,None],dtype=torch.float64)
    co,lower=core.lower_fields(values,d);m,z,_=dense_design()
    assert np.max(abs(co.numpy()-np.tensordot(m,values.numpy(),axes=(1,0))))<1e-14
    assert np.max(abs(lower[0].numpy()[[0,1,2,4,8]]-values.numpy()[[0,1,2,4,8]]))==0
    return dict(status='PASS',upstream_pair=.5,linear_tail_pair=1,linear_middle_pair=0,downstream_generated_pair=.5,
        nonlinear_pair_before=-.2,nonlinear_pair_after_middle_erasure=.5)


def check(seed,device):
    """每教师两强度，验证主计算、实际部分擦除、梯度及结构对照。"""
    model=teacher.load_teacher(seed,device)
    for p in model.parameters():p.requires_grad_(False)
    factors=np.load(teacher.DATA/'development_factors.npy',allow_pickle=False);labels=np.load(teacher.DATA/'development_labels.npy',allow_pickle=False)
    r,d=matched_pairs(factors,labels)[0];pixels=np.load(teacher.DATA/'development_images.npy',mmap_mode='r',allow_pickle=False)
    image=torch.as_tensor(np.asarray(pixels[[r,d]])[:,None],dtype=torch.float64,device=device)
    with torch.no_grad():source=model.block1(model.stem(image))
    design=core.design();bits=torch.as_tensor(design['bits'],dtype=torch.float64,device=device)
    weight=pair_weights(labels[r],labels[d]);random=np.stack([np.random.default_rng(seed+v).permutation(32) for v in range(2)])
    errors={}
    def compare(name,a,b):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
        error=float(np.max(abs(a-b)));assert error<1e-9,(name,error);errors[name]=max(errors.get(name,0.),error)
    for alpha in (1.,.5):
        states=replace_groups(source[:1],source[1:],bits,alpha)
        out,checks,_=core.calculate(model,states,weight,alpha*(source[1:]-source[:1]),np.arange(8),random,design)
        errors.update({k:max(errors.get(k,0.),v) for k,v in checks.items()})
        actual,middle,_=original_modes(model,states);compare('original_modes',out['logits'],actual.cpu().numpy())
        compare('whole_model_hook',out['logits'][:,0],original_image(model,image[:1],states).cpu().numpy())
        co,lower=independent_lower(middle.cpu().numpy());compare('lower_fields',out['lower_logits'],core.tail(model,torch.as_tensor(lower.reshape(-1,32,32,32),device=device)).reshape(3,16,3).cpu().numpy())
        fields=[]
        for row in out['selections']:
            ids=row[row>=0];erased=middle.cpu().numpy().copy();erased[:,ids]=lower[0,:,ids].transpose(1,0,2,3)
            restored=lower[0].copy();restored[:,ids]=middle.cpu().numpy()[:,ids];fields.extend((erased,restored))
        reference=core.tail(model,torch.as_tensor(np.concatenate(fields),device=device)).reshape(7,2,16,3).cpu().numpy()
        compare('independent_patch',out['patched_logits'],reference)
        logits,h,mg,sg,_=independent_states_and_gradients(model,states)
        gradients=core.state_gradients(model,states,weight,alpha*(source[1:]-source[:1]))
        compare('source_adjoint',gradients['source_gradient'].cpu().numpy(),np.einsum('nkp,k->np',sg.cpu().numpy(),weight).reshape(16,16,64,64))
        compare('middle_adjoint',gradients['middle_gradient'].cpu().numpy(),np.einsum('nkp,k->np',mg.cpu().numpy(),weight).reshape(16,32,32,32))
    return dict(seed=seed,status='PASS',maximum_errors=errors)


if __name__=='__main__':
    import json
    device=teacher.configure();result=dict(status='PASS',known=known_answers(),teachers=[check(s,device) for s in teacher.SEEDS])
    path=teacher.BASE/'reports/dsprites_layer_transfer_preflight_20261013.json';assert not path.exists();teacher.dump(path,result)
    print(json.dumps(result,ensure_ascii=False))

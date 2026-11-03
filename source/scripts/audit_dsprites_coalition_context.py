"""独立审计：不调用四状态数值核心，从原模型forward与实际激活重建。

全部冻结端点、四状态、两种非负点控制和逐成员恢复均核验；
分层余项用原BN/卷积的仿射差分验证，不复用生产的卷积缩放实现。
"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
from run_dsprites_coalition_context import BASE,RUN,PRIOR,SHAPE,verify_protocol,verify_folder,preservation,event
from run_dsprites_upstream_rescue import load

ERRORS = {}
ENDPOINTS = tuple(range(25,37))+(1,4,7,10)


def check(name,a,b):
    """检查形状和逐元素数值；布尔/编号也必须完全一致。"""
    if torch.is_tensor(a): a=a.detach().cpu().numpy()
    if torch.is_tensor(b): b=b.detach().cpu().numpy()
    a,b=np.asarray(a),np.asarray(b); assert a.shape==b.shape,(name,a.shape,b.shape)
    error=float(abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.
    assert error<1e-9,(name,error); ERRORS[name]=max(error,ERRORS.get(name,0.))


@torch.no_grad()
def native(block,x,point=None,value=None,patch_ids=None,patch_half=None,amount=0.):
    """原block.forward执行，hook只读中间值；控制只替换mid的指定激活。"""
    result={}; oldmid,oldfinish=block.mid,block.finish
    def first(module,args,out): result['z']=out
    def second(module,args,out): result['branch']=out
    hooks=[block.bn1.register_forward_hook(first),block.bn2.register_forward_hook(second)]
    def mid(arg):
        out=oldmid(arg)
        if point is not None:
            out=out.clone(); out.flatten(1)[:,point]=value
        if patch_ids is not None:
            out=out.clone(); current=out.flatten(1)[:,patch_ids]
            out.flatten(1)[:,patch_ids]=(1-amount)*current+amount*patch_half.flatten(1)[:,patch_ids]
        result['u']=out; return out
    def finish(arg,skip):
        out=oldfinish(arg,skip); result['v']=result['branch']+skip; return out
    try:
        block.mid,block.finish=mid,finish; result['h']=block(x)
    finally:
        block.mid,block.finish=oldmid,oldfinish
        for hook in hooks: hook.remove()
    return result


def decode(units,choice):
    """从旧路径编号独立解码家族、随机名单和前缀。"""
    path,amount,eligible=choice
    if int(path)<0: return -1,np.array([],dtype=int),0.
    p=int(path); family=p//12; variant=p%12//4; prefix=p%4
    group=(units['source_lists'],units['middle_lists'],units['gate_lists'][0],units['gate_lists'][1])[family]
    ids=group[variant]; ids=ids[ids>=0]
    count=(1,2,3,4)[prefix] if family<2 else (1,4,8,16)[prefix]
    return family,ids[:count],float(amount)


@torch.no_grad()
def actual_fields(model,source,recipient,donor,mask,units,choices):
    """直接构造当前mask的半/全强度源，逐个原forward干预，不重建七锚点。"""
    keep=torch.zeros_like(source[recipient:recipient+1])
    for g in range(4):
        if mask&(1<<g): keep[:,4*g:4*g+4]=1.
    recipient_x=source[recipient:recipient+1]; change=(source[donor:donor+1]-recipient_x)*keep
    x0=recipient_x+.5*change; x1=recipient_x+change
    half=native(model.block2,x0); full=native(model.block2,x1); values=[]
    for endpoint in ENDPOINTS:
        family,ids,amount=decode(units,choices[endpoint-1])
        if family<0 or amount==0: h=full['h'].clone()
        elif family==0:
            x=x1.clone(); x[:,ids]=(1-amount)*x1[:,ids]+amount*x0[:,ids]; h=model.block2(x)
        elif family==2:
            h=native(model.block2,x1,patch_ids=ids,patch_half=half['u'],amount=amount)['h']
        else:
            h=full['h'].clone()
            if family==1: h[:,ids]=(1-amount)*full['h'][:,ids]+amount*half['h'][:,ids]
            else: h.flatten(1)[:,ids]=(1-amount)*full['h'].flatten(1)[:,ids]+amount*half['h'].flatten(1)[:,ids]
        values.append(h[0])
    return full['h'][0],torch.stack(values)


def differences(v):
    """输入endpoint,state,...；显式六差分，避免与生产矩阵实现共用。"""
    a=v[:,1]-v[:,0]; c=v[:,2]-v[:,0]; j=v[:,3]-v[:,1]-v[:,2]+v[:,0]
    return torch.stack((a,c,j,v[:,3]-v[:,0],v[:,3]-v[:,2],v[:,3]-v[:,1]),1)


@torch.no_grad()
def audit_case(model,before,after,C,point,weight,raw):
    """核验所有空间、通道、点控制及余项统计，返回实际forward字段数。"""
    four=[]
    for h in after:
        only=before.clone(); only[C]=h[C]
        outside=h.clone(); outside[C]=before[C]
        four.append(torch.stack((before,only,outside,h)))
    fields=torch.stack(four); x=fields.reshape(-1,32,32,32); b=model.block3
    nat=native(b,x); n=len(after); w=torch.as_tensor(weight,device=x.device,dtype=x.dtype)
    q=(model.readout.weight.T@w).reshape(1,1,32,1,1)/1024
    shape=lambda v:v.reshape(n,4,32,32,32)
    z,u,v,e=[shape(nat[k]) for k in ('z','u','v','h')]
    zj=nat['z'].flatten(1)[:,point].reshape(n,4); uj=nat['u'].flatten(1)[:,point].reshape(n,4)
    check('point_z',zj,raw['point_z']);check('point_u',uj,raw['point_u'])
    check('point_affine_interaction',zj[:,3]-zj[:,1]-zj[:,2]+zj[:,0],np.zeros(n))
    branches=[];logs=[];chs=[];spatial=[];flips=[]
    for mode in range(3):
        record=nat if mode==0 else native(b,x,point=point,value=uj[:,0 if mode==1 else 3,None].repeat(1,4).reshape(-1))
        logits=model.readout(record['h'].mean((2,3))).reshape(n,4,3); logs.append(logits)
        maps=differences(shape(record['h']))*q
        chs.append(maps.sum((-1,-2))); spatial.append(torch.stack((torch.where(maps>0,maps,0.).sum((2,3,4)),torch.where(maps<0,maps,0.).sum((2,3,4))),-1))
        if mode: flips.append(((record['v']>0)!=(nat['v']>0)).sum((1,2,3)).reshape(n,4))
    check('logits',torch.stack(logs,2),raw['logits']);check('channel_terms',torch.stack(chs,1),raw['channel_terms'])
    check('spatial_pn',torch.stack(spatial,1),raw['spatial_pn']);check('patch_second_flips',torch.stack(flips,-1),raw['patch_second_flips'])
    check('natural_flips',torch.stack((((z>0)!=(z[:,:1]>0)).sum((2,3,4)),((v>0)!=(v[:,:1]>0)).sum((2,3,4))),-1),raw['natural_flips'])
    # 仿射差分来自原BN/卷积；与生产scale*conv实现独立。
    l2=lambda a:shape(b.bn2(b.conv2(a.reshape(-1,32,32,32)))-b.bn2(b.conv2(torch.zeros_like(a.reshape(-1,32,32,32)))))
    dz=z-z[:,:1]; dh=fields-fields[:,:1]; du=u-u[:,:1]; g1=z[:,:1]>0; g2=v[:,:1]>0
    parts=(g2*(l2(g1*dz)+dh),g2*l2(du-g1*dz),e-e[:,:1]-g2*(v-v[:,:1]))
    for part in parts: branches.append((differences(part)*q).sum((2,3,4)))
    check('branch_terms',torch.stack(branches,-1),raw['branch_terms'])
    outsideC=[c for c in range(32) if c not in C]
    check('shifts',torch.stack((abs(after[:,C]-before[C]).flatten(1).max(1).values,abs(after[:,outsideC]-before[outsideC]).flatten(1).max(1).values),-1),raw['shifts'])
    drops=[]
    for slot in (0,3,6,9):
        for state in (1,3):
            for c in C:
                h=fields[slot,state].clone();h[c]=before[c];drops.append(h)
    ds=torch.stack(drops); rec=native(b,ds); k=len(C)
    check('member_logits',model.readout(rec['h'].mean((2,3))).reshape(4,2,k,3),raw['drop_logits'][:,:,:k])
    check('member_z',rec['z'].flatten(1)[:,point].reshape(4,2,k),raw['drop_z'][:,:,:k])
    check('member_padding_logits',raw['drop_logits'][:,:,k:],np.zeros_like(raw['drop_logits'][:,:,k:]))
    check('member_padding_z',raw['drop_z'][:,:,k:],np.zeros_like(raw['drop_z'][:,:,k:]))
    return len(x),2*len(x),len(ds)


def audit():
    """全量审计通过才写PASS；不覆盖完成审计。"""
    assert not (RUN/'independent_audit.json').exists();verify_protocol();device=teacher.configure();panel=load(SHAPE/'panel/data.npz')
    counters=np.zeros(3,dtype=int)
    with torch.no_grad():
        for seed in teacher.SEEDS:
            verify_folder(RUN/f'evaluation/seed_{seed}'); raw=load(RUN/f'evaluation/seed_{seed}/data.npz')
            search=load(PRIOR/f'search/seed_{seed}/data.npz'); units=load(PRIOR/f'search/seed_{seed}/units.npz'); fixed=load(PRIOR/f'search/seed_{seed}/frozen.npz')
            model=teacher.load_teacher(seed,device)
            # 逐图原prefix，避免复用生产的sources_for批次函数。
            source=torch.cat([model.block1(model.stem(torch.as_tensor(im[None,None],dtype=torch.float64,device=device))) for im in panel['images']])
            for case,(r,d) in enumerate(panel['pairs']):
                weight=np.zeros(3);weight[panel['labels'][d]]=1.;weight[panel['labels'][r]]=-1.
                for m,mask in enumerate((3,15)):
                    us={k:v[case,m] for k,v in units.items()}; before,after=actual_fields(model,source,r,d,mask,us,search['choices'][case,m])
                    C=fixed['channels'][case,m];C=C[C>=0];point=int(fixed['lists'][case,m,0,0])
                    counters+=audit_case(model,before,after,C,point,weight,{k:v[case,m] for k,v in raw.items()})
                if (case+1)%24==0:event('audit_progress',seed=seed,cases=case+1)
    preserved=preservation()
    result=dict(status='PASS',utc=teacher.now(),native_fields=int(counters[0]),point_patches=int(counters[1]),member_reversions=int(counters[2]),maximum_errors=ERRORS,old_delivery_entries=sum(p['files'] for p in preserved.values()),independence='no productioncore,independent source masks/path decode/field assembly,originalforward/hook pointpatch,originalaffinedifferences')
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':audit()

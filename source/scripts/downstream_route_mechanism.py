"""定位原C的下游传递：实际单点阻断、联合阻断、保留恢复与残差支路。

每个组外背景分别比较C-off与C-on，替换值始终取对应C-off的实测
非负激活。第一ReLU阻断后原第二ReLU重算；末层ReLU后的联合阻断
因GAP+线性头而可加，这个恒等式不当作新的联合机制发现。
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

EPS=1e-10
POINT_BUDGETS=(1,4,16,64,256,1024)
CHANNEL_BUDGETS=(1,2,4,8,16,32)
ORDERS=('absolute','random0','random1','switch_only','stable_only','supporting','opposing')
SPECIALS=('empty','original_j','all_active','all_switch','all_stable','all_supporting','all_opposing')


@torch.no_grad()
def trace(model,x):
    """原block3两层预激活/激活与读出，用于同背景下的C配对。"""
    b=model.block3;z=b.bn1(b.conv1(x));u=F.relu(z);v=b.bn2(b.conv2(u))+x;e=F.relu(v)
    return dict(x=x,z=z,u=u,v=v,e=e,logits=model.readout(e.mean((2,3))))


def rank(values,eligible):
    """绝对单阻断效应排序，物理编号打破并列；允许选到负向抵消路径。"""
    ids=np.flatnonzero(eligible)
    return ids[np.lexsort((ids,-abs(values[ids])))]


@torch.no_grad()
def first_point_effects(model,t0,t1,weight):
    """全部32768第一ReLU点的精确单点阻断输出差。

一个点只改变后续卷积的3×3邻域。使用实际v1逐项重算第二ReLU，
不是梯度近似；卷积核反向索引与图像边界在预检中用原forward核验。
"""
    b=model.block3;du=(t0['u']-t1['u'])[0].reshape(32,1024)
    base=F.unfold(t1['v'],3,padding=1).reshape(32,9,1024)
    valid=F.unfold(torch.ones_like(t1['v'][:,:1]),3,padding=1).reshape(1,9,1024)
    scale=b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps)
    kernel=(b.conv2.weight*scale[:,None,None,None]).flip((-1,-2)).reshape(32,32,9)
    w=torch.as_tensor(weight,device=du.device,dtype=du.dtype);q=(w@model.readout.weight)/1024
    values=[]
    for channel in range(32):
        moved=base+kernel[:,channel,:,None]*du[channel][None,None]
        values.append(((F.relu(base)-F.relu(moved))*valid*q[:,None,None]).sum((0,1)))
    return torch.stack(values).reshape(-1)


@torch.no_grad()
def locate(model,t0,t1,weight,rng_seed):
    """按当前配对的真实单阻断效应定位；这是同面板输出指导的探索。"""
    w=torch.as_tensor(weight,device=t0['x'].device,dtype=t0['x'].dtype)
    q=(w@model.readout.weight)[:,None,None]/1024
    delta=torch.stack((t1['u'][0]-t0['u'][0],t1['e'][0]-t0['e'][0])).reshape(2,-1)
    switched=torch.stack(((t1['z'][0]>0)!=(t0['z'][0]>0),(t1['v'][0]>0)!=(t0['v'][0]>0))).reshape(2,-1)
    scores=torch.stack((first_point_effects(model,t0,t1,weight),((t1['e']-t0['e'])[0]*q).flatten()))
    # 在第一层阻断整张通道图后运行 finish。
    batch=t1['u'].expand(32,-1,-1,-1).clone()
    for c in range(32):batch[c,c]=t0['u'][0,c]
    blocked=model.readout(model.block3.finish(batch,t1['x'].expand(32,-1,-1,-1)).mean((2,3)))
    channel_scores=torch.stack(((t1['logits']-blocked)@w,scores[1].reshape(32,1024).sum(-1)))
    score=scores.cpu().numpy();change=delta.cpu().numpy();switch=switched.cpu().numpy();chscore=channel_scores.cpu().numpy()
    orders=np.full((2,7,1024),-1,dtype=np.int32);counts=np.zeros((2,7),dtype=np.int32)
    channel_orders=np.full((2,3,32),-1,dtype=np.int32);channel_counts=np.zeros((2,3),dtype=np.int32)
    total=float((t1['logits']-t0['logits'])@w);direction=1 if total>=0 else -1
    for layer in range(2):
        active=abs(change[layer])>EPS;support=active&(direction*score[layer]>EPS);oppose=active&(direction*score[layer]<-EPS)
        rng=np.random.default_rng(rng_seed+layer*10007)
        lists=[rank(score[layer],active),rng.permutation(np.flatnonzero(active)),rng.permutation(np.flatnonzero(active)),
               rank(score[layer],active&switch[layer]),rank(score[layer],active&~switch[layer]),rank(score[layer],support),rank(score[layer],oppose)]
        for index,ids in enumerate(lists):counts[layer,index]=len(ids);orders[layer,index,:min(1024,len(ids))]=ids[:1024]
        chactive=active.reshape(32,1024).any(-1)
        clists=[rank(chscore[layer],chactive),rng.permutation(np.flatnonzero(chactive)),rng.permutation(np.flatnonzero(chactive))]
        for index,ids in enumerate(clists):channel_counts[layer,index]=len(ids);channel_orders[layer,index,:len(ids)]=ids
    return dict(point_effect=score,channel_effect=chscore,delta_activation=change,switched=switch,
                orders=orders,order_counts=counts,channel_orders=channel_orders,channel_counts=channel_counts,direction=np.array(direction))


def commands(selection,layer,point):
    """固定预算的单点/联合名单及完整阳性控制；相同实际名单保留明确标签。"""
    out=[]
    for order,name in enumerate(ORDERS):
        ids=selection['orders'][layer,order];ids=ids[ids>=0]
        for budget in POINT_BUDGETS:out.append(dict(kind='point',order=name,budget=budget,ids=ids[:budget]))
    for order,name in enumerate(ORDERS[:3]):
        ids=selection['channel_orders'][layer,order];ids=ids[ids>=0]
        for budget in CHANNEL_BUDGETS:
            used=ids[:budget];physical=(used[:,None]*1024+np.arange(1024)[None,:]).reshape(-1)
            out.append(dict(kind='channel',order=name,budget=budget,ids=physical,channels=used))
    active=abs(selection['delta_activation'][layer])>EPS;sw=selection['switched'][layer];score=selection['point_effect'][layer]*int(selection['direction'])
    sets=[np.array([],dtype=int),np.array([point],dtype=int) if layer==0 else np.array([],dtype=int),np.flatnonzero(active),np.flatnonzero(active&sw),np.flatnonzero(active&~sw),np.flatnonzero(active&(score>EPS)),np.flatnonzero(active&(score<-EPS))]
    for name,ids in zip(SPECIALS,sets):out.append(dict(kind='special',order=name,budget=len(ids),ids=ids))
    assert len(out)==67
    return out


@torch.no_grad()
def evaluate(model,t0,t1,weight,selection,point):
    """同一名单在残差支路前/后两背景下做阻断与保留恢复。

block: u1的选定点换回u0；keep: u0仅保留选定点的u1变化。
残差支路取x0/x1，形成路径四状态，并精确保留未阻断输出的正负分量。
"""
    device,dtype=t0['x'].device,t0['x'].dtype;w=torch.as_tensor(weight,device=device,dtype=dtype)
    q=(w@model.readout.weight)[None,:,None,None]/1024
    route=[];route_e=[]
    for ui in (t0['u'],t1['u']):
        er=torch.cat([model.block3.finish(ui,x) for x in (t0['x'],t1['x'])])
        route.append(model.readout(er.mean((2,3))));route_e.append(er)
    route_logits=torch.stack(route);route_e=torch.stack(route_e) # branch,skip,...
    outputs={};residual_ids=np.zeros((2,2,128),np.int32);residual_values=np.zeros((2,2,128))
    selected_indices=(4,44) # absolute point256; absolute channel4
    for layer in range(2):
        cmds=commands(selection,layer,point);v0=t0['u'] if layer==0 else t0['e'];v1=t1['u'] if layer==0 else t1['e']
        batch=[]
        for cmd in cmds:
            ids=cmd['ids'];block=v1.clone();keep=v0.clone()
            block.flatten(1)[:,ids]=v0.flatten(1)[:,ids];keep.flatten(1)[:,ids]=v1.flatten(1)[:,ids]
            batch.extend((block[0],keep[0]))
        batch=torch.stack(batch)
        logits=[];channels=[];pn=[];second_flips=[];maps_keep=[]
        skips=(t0['x'],t1['x']) if layer==0 else (None,)
        for skip in skips:
            if layer==0:
                v=model.block3.bn2(model.block3.conv2(batch))+skip;e=F.relu(v)
                second_flips.append(((v>0)!=(t0['v']>0)).sum((1,2,3)).reshape(67,2))
            else:e=batch
            logits.append(model.readout(e.mean((2,3))).reshape(67,2,3))
            maps=((e-t0['e'])*q).reshape(67,2,32,32,32)
            channels.append(maps.sum((-1,-2)));pn.append(torch.stack((maps.clamp_min(0).sum((2,3,4)),maps.clamp_max(0).sum((2,3,4))),-1))
            maps_keep.append(maps)
        if layer==0:
            outputs['first_logits']=torch.stack(logits,1);outputs['first_channels']=torch.stack(channels,1);outputs['first_pn']=torch.stack(pn,1);outputs['first_second_flips']=torch.stack(second_flips,1)
        else:outputs['last_logits']=logits[0];outputs['last_channels']=channels[0];outputs['last_pn']=pn[0]
        # 原生残差支路下，两个预定预算联合阻断后最大的128个末层余量位置。
        maps=maps_keep[-1]
        for j,index in enumerate(selected_indices):
            value=maps[index,0].flatten().cpu().numpy();ids=np.lexsort((np.arange(32768),-abs(value)))[:128]
            residual_ids[layer,j]=ids;residual_values[layer,j]=value[ids]
    outputs['route_logits']=route_logits
    outputs['route_channels']=((route_e-t0['e'])*q).sum((-1,-2))
    outputs['native_logits']=torch.cat((t0['logits'],t1['logits']))
    # 所有第一层变化阻断后只剩skip路径；最后层全部阻断后效应为0。
    all_index=62
    assert torch.max(abs(outputs['first_logits'][all_index,:,0]-route_logits[0]))<1e-9
    assert torch.max(abs(outputs['first_logits'][all_index,:,1]-route_logits[1]))<1e-9
    assert torch.max(abs(outputs['last_logits'][all_index,0]-t0['logits'][0]))<1e-9
    result={k:v.cpu().numpy() for k,v in outputs.items()}
    result.update(residual_top_ids=residual_ids,residual_top_values=residual_values)
    return result

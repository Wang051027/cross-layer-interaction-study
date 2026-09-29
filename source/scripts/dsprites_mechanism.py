"""将旧跨层计算迁移到任意图片尺寸/类别数；不修改旧冻结源码。

直接复用组合差分、原网络逐层计算和ReLU代数分解。新增部分处理动态尺寸、
三类margin、标签无关输出向量、全局/按类/逐图选择，以及严格平移坐标。
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from cross_layer_mechanism import (design,pair_difference,source_activation,changed_sources,
    tail_from_middle,trace,decomposition,original_forward)

TARGETS=('margin','vector')
BUDGETS=(16,32)
METHODS=tuple(f'{target}_{scope}_{score}' for target in TARGETS
              for score in ('gradient','finite') for scope in ('global','class','input'))+(
    'margin_input_full_gradient','vector_input_full_gradient',
    'margin_global_single','margin_class_single','margin_input_single',
    'random_0','random_1','random_2','random_3')


def project(logits,label,mode):
    """始终保留末尾目标轴：margin为1维，vector为去类别均值的C维。

    vector不使用标签。它去掉所有logit共同平移的部分，保留类别对比信息。
    """
    c=logits.shape[-1]
    assert c>=2 and mode in TARGETS
    if mode=='margin':return ((c*logits[...,int(label)]-logits.sum(-1))/(c-1))[...,None]
    return logits-logits.mean(-1,keepdims=True)


def contrasts(values,stencil):
    """组合轴为倒数第二维，目标轴在末尾；适配单个或多个输出方向。"""
    return (values[...,stencil[:,0],:]-values[...,stencil[:,1],:]
            -values[...,stencil[:,2],:]+values[...,stencil[:,3],:])


@torch.no_grad()
def patch(model,middle,selections,selection_chunk=4):
    """对任意维度的完整中介层做恢复和移植，分组构造以限制GPU内存。

    返回[方案数,操作2,组合数,类别数]；操作0为恢复、1为移植。
    """
    outputs=[];baseline=middle[:1].expand_as(middle)
    for start in range(0,len(selections),selection_chunk):
        group=selections[start:start+selection_chunk];states=[]
        for nodes in group:
            restored=middle.clone();restored.flatten(1)[:,nodes]=baseline.flatten(1)[:,nodes]
            transplanted=baseline.clone();transplanted.flatten(1)[:,nodes]=middle.flatten(1)[:,nodes]
            states.extend((restored,transplanted))
        combined=torch.cat(states)
        out=torch.cat([tail_from_middle(model,x) for x in combined.split(64)])
        outputs.append(out.reshape(len(group),2,len(middle),out.shape[-1]))
    return torch.cat(outputs)


@torch.no_grad()
def reachable(model,nodes,source):
    """从真实源张量的形状构造拓扑支持集，消除原16×28×28常量。"""
    support=torch.zeros_like(source[:1]);support.flatten()[nodes]=1
    def conv_support(conv,x):
        """只传播非零连接的可达性，不按当前图的门控筛选。"""
        return (F.conv2d(x,(conv.weight!=0).to(x.dtype),None,conv.stride,conv.padding,conv.dilation,conv.groups)>0).to(x.dtype)
    block=model.block2
    main=conv_support(block.conv2,conv_support(block.conv1,support))
    skip=support if isinstance(block.shortcut,nn.Identity) else conv_support(block.shortcut[0],support)
    return torch.nonzero((main+skip).flatten()>0).flatten().cpu().numpy()


@torch.no_grad()
def select_source(model,images,device):
    """沿用旧中心3×3训练平均激活前四规则，空间大小由实际输出确定。"""
    total=None
    for batch in images.split(16):
        state=source_activation(model,batch.to(device))
        part=state.sum(0);total=part if total is None else total+part
    channels,height,width=total.shape
    candidates=np.arange(channels*height*width).reshape(channels,height,width)[:,height//2-1:height//2+2,width//2-1:width//2+2].ravel()
    means=(total/len(images)).flatten().cpu().numpy()
    return candidates[np.argsort(-means[candidates],kind='stable')[:4]]


def features(model,middle,label,stencil):
    """返回四类逐坐标分数：margin交互、margin单独效应、vector交互、激活。

    用全部类别的基线梯度构造标签无关向量评分；margin梯度由同一批梯度
    精确线性组合得到。这里仍是提名方法，不能代替有限修补的实际效果。
    """
    baseline=middle[:1].detach().clone().requires_grad_(True)
    logits=tail_from_middle(model,baseline)[0];c=len(logits)
    gradients=torch.stack([torch.autograd.grad(logits[j],baseline,retain_graph=j<c-1)[0][0] for j in range(c)])
    with torch.no_grad():
        centered=gradients-gradients.mean(0,keepdim=True)
        grad=centered[int(label)]*c/(c-1)
        inter=pair_difference(middle,stencil)
        single=middle[[1,2,4,8]]-middle[:1]
        margin=(inter*grad).abs().mean(0)
        main=(single*grad).abs().mean(0)
        vector=(inter[:,None]*centered[None]).square().mean((0,1)).sqrt()
        return np.stack([v.flatten().cpu().numpy() for v in (margin,main,vector,middle[0].abs())])


def finite_scores(singleton,natural,label,stencil):
    """返回[2目标,64候选]的实际单点联合误差改善，保持符号、不作小分母除法。"""
    result=[]
    for mode in TARGETS:
        original=contrasts(project(natural,label,mode),stencil)
        response=contrasts(project(singleton,label,mode),stencil)
        score=np.mean(original**2)-.5*(np.mean(response[:,0]**2,axis=(-2,-1))
              +np.mean((response[:,1]-original)**2,axis=(-2,-1)))
        result.append(score)
    return np.stack(result)


def rank(scores,candidates):
    """候选是支持集内部索引；按给定顺序稳定处理完全相等的分数。"""
    candidates=np.asarray(candidates,dtype=int)
    return candidates[np.argsort(-np.asarray(scores)[candidates],kind='stable')]


def make_shortlist(mean_features):
    """共同64候选：margin交互前24、单独效应前16、vector交互前24并集后补齐。"""
    pool=np.arange(mean_features.shape[1])
    orders=[rank(mean_features[j],pool) for j in (0,1,2)]
    selected=set(orders[0][:24])|set(orders[1][:16])|set(orders[2][:24])
    for node in orders[0]:
        if len(selected)==64:break
        selected.add(int(node))
    assert len(selected)==64
    return np.array(sorted(selected),dtype=int)


def fixed_orders(feature,finite,labels,shortlist,seed):
    """开发集拟合全局和三种形状各自的固定排序，三类都使用同一候选名单。"""
    result={}
    for ti,mode in enumerate(TARGETS):
        fi=0 if mode=='margin' else 2
        for scope in ('global','class'):
            masks=[np.ones(len(labels),dtype=bool)] if scope=='global' else [labels==c for c in range(3)]
            gradient=[];actual=[]
            for mask in masks:
                gradient.append(rank(feature[mask,fi].mean(0),shortlist).tolist())
                actual.append(shortlist[np.argsort(-finite[mask,ti].mean(0),kind='stable')].tolist())
            result[f'{mode}_{scope}_gradient']=gradient[0] if scope=='global' else gradient
            result[f'{mode}_{scope}_finite']=actual[0] if scope=='global' else actual
    result['margin_global_single']=rank(feature[:,1].mean(0),shortlist).tolist()
    result['margin_class_single']=[rank(feature[labels==c,1].mean(0),shortlist).tolist() for c in range(3)]
    for repeat in range(4):
        result[f'random_{repeat}']=np.random.default_rng(seed+repeat+2026092900).permutation(shortlist).tolist()
    return result


def specifications(pool_size):
    """固定两种预算及方法顺序；整名单和整可达池只是较大集合参照。"""
    return [dict(method=m,budget=k) for k in BUDGETS for m in METHODS]+[
        dict(method='all_shortlist',budget=64),dict(method='all_reachable',budget=int(pool_size))]


def select_input(feature,finite,label,shortlist,pool,fixed):
    """只用该图半强度信息选点，返回实际扁平神经元坐标。"""
    orders={name:(value[int(label)] if '_class_' in name else value) for name,value in fixed.items()}
    all_indices=np.arange(len(pool))
    for ti,mode in enumerate(TARGETS):
        fi=0 if mode=='margin' else 2
        orders[f'{mode}_input_gradient']=rank(feature[fi],shortlist)
        orders[f'{mode}_input_finite']=shortlist[np.argsort(-finite[ti],kind='stable')]
        orders[f'{mode}_input_full_gradient']=rank(feature[fi],all_indices)
    orders['margin_input_single']=rank(feature[1],shortlist)
    selections=[]
    for spec in specifications(len(pool)):
        method,k=spec['method'],spec['budget']
        index=shortlist if method=='all_shortlist' else all_indices if method=='all_reachable' else orders[method][:k]
        selected=np.asarray(pool[np.asarray(index,dtype=int)],dtype=int)
        assert len(selected)==k and len(np.unique(selected))==k
        selections.append(selected)
    return selections


def shifted_nodes(nodes,shape,dx,dy=0):
    """按真实通道/行/列平移坐标，禁止越界或通道串位。"""
    c,y,x=np.unravel_index(np.asarray(nodes,dtype=int),shape)
    assert np.all((x+dx>=0)&(x+dx<shape[2])&(y+dy>=0)&(y+dy<shape[1]))
    return np.ravel_multi_index((c,y+dy,x+dx),shape)


def encode(selections,width):
    """不同预算名单以-1填充，避免保存Python对象数组。"""
    result=np.full((len(selections),width),-1,dtype=np.int64)
    for i,nodes in enumerate(selections):result[i,:len(nodes)]=nodes
    return result


def metric_arrays(natural,patched,labels,stencil,mode):
    """返回逐输入的自然、恢复和移植误差均方量，供按模型/类别聚合。"""
    ni=np.stack([contrasts(project(x,int(y),mode),stencil) for x,y in zip(natural,labels)])
    pi=np.stack([contrasts(project(x,int(y),mode),stencil) for x,y in zip(patched,labels)])
    return np.mean(ni**2,axis=(-2,-1)),np.mean(pi[:,:,0]**2,axis=(-2,-1)),np.mean((pi[:,:,1]-ni[:,None])**2,axis=(-2,-1))

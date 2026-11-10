"""用稠密对比矩阵重算交互和单点收益。用四层卷积的伴随运算重建中间层和源层的雅可比矩阵。"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import dsprites_teacher as teacher
import numpy as np
import torch
from torch.nn import functional as F

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_gradient_20261009'
OLD=teacher.RUN
PRIOR=BASE/'runs/dsprites_factor_transfer_20261007'
METHODS=('class_fixed','input_gradient','fixed_linear','state_linear','input_finite',
         'random_0','random_1','random_2','random_3','all_shortlist','all_reachable')
STRENGTHS=(1.,.25,.5,.75)
ERRORS={}
COUNT={}
ROUND_OFF_RANKS=[]


def compare(key,actual,expected,tolerance=1e-10):
    """先检查形状再比较，防止广播把维度错误隐藏成看似小的误差。"""
    actual=np.asarray(actual);expected=np.asarray(expected)
    assert actual.shape==expected.shape,(key,actual.shape,expected.shape)
    error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
    assert np.isfinite(error) and error<tolerance,(key,error)
    ERRORS[key]=max(ERRORS.get(key,0.),error)
    COUNT[key]=COUNT.get(key,0)+actual.size


def contrast_matrix():
    """独立枚举6对×4背景；四项符号为+、-、-、+。"""
    matrix=[];stencil=[]
    for a,b in itertools.combinations(range(4),2):
        for background in range(16):
            if background & ((1<<a)|(1<<b)):continue
            ids=[background|(1<<a)|(1<<b),background|(1<<a),background|(1<<b),background]
            row=np.zeros(16);row[ids]=[1.,-1.,-1.,1.]
            matrix.append(row);stencil.append(ids)
    return np.asarray(matrix),np.asarray(stencil)


def interaction(raw,labels,target,matrix):
    """组合轴倒数第二，稠密矩阵乘法与主程序的四项索引实现独立。"""
    values=np.einsum('km,n...mc->n...kc',matrix,raw)
    if target=='vector':return values-values.mean(-1,keepdims=True)
    weights=np.full((len(labels),3),-.5);weights[np.arange(len(labels)),labels]=1.
    return np.einsum('n...kc,nc->n...k',values,weights)[...,None]


@torch.no_grad()
def affine_adjoint(conv,bn,gradient,input_shape):
    """评价BN与卷积的伴随映射，输出padding由原输入形状反推。

    这一计算同时覆盖block2降采样主分支和1×1旁路，不漏掉残差贡献。
    """
    scale=bn.weight/torch.sqrt(bn.running_var+bn.eps)
    padding=[]
    for axis in range(2):
        standard=(gradient.shape[2+axis]-1)*conv.stride[axis]-2*conv.padding[axis]+conv.dilation[axis]*(conv.kernel_size[axis]-1)+1
        padding.append(input_shape[2+axis]-standard)
    return F.conv_transpose2d(gradient*scale[None,:,None,None],conv.weight,
        stride=conv.stride,padding=conv.padding,output_padding=tuple(padding),
        groups=conv.groups,dilation=conv.dilation)


@torch.no_grad()
def independent_states_and_gradients(model,source):
    """逐层前向后，用卷积伴随分别沿主支路与残差支路反传各类别的梯度。ReLU 在零点取导数 0。"""
    b2,b3=model.block2,model.block3
    z1=b2.bn1(b2.conv1(source));h1=F.relu(z1)
    z2=b2.bn2(b2.conv2(h1))+b2.shortcut(source);middle=F.relu(z2)
    z3=b3.bn1(b3.conv1(middle));h3=F.relu(z3)
    z4=b3.bn2(b3.conv2(h3))+middle;end=F.relu(z4)
    logits=model.readout(end.mean((2,3)))
    middle_gradients=[];source_gradients=[]
    for category in range(3):
        terminal=model.readout.weight[category][None,:,None,None].expand_as(end)/(end.shape[2]*end.shape[3])
        g4=terminal*(z4>0)
        g3=affine_adjoint(b3.conv2,b3.bn2,g4,h3.shape)*(z3>0)
        gm=affine_adjoint(b3.conv1,b3.bn1,g3,middle.shape)+g4
        g2=gm*(z2>0)
        g1=affine_adjoint(b2.conv2,b2.bn2,g2,h1.shape)*(z1>0)
        gs=affine_adjoint(b2.conv1,b2.bn1,g1,source.shape)
        gs+=affine_adjoint(b2.shortcut[0],b2.shortcut[1],g2,source.shape)
        middle_gradients.append(gm.flatten(1));source_gradients.append(gs.flatten(1))
    return logits,middle,torch.stack(middle_gradients,1),torch.stack(source_gradients,1),(z1,z2,z3,z4)


def load(path):
    """关闭NPZ文件句柄后返回纯数值数组，禁止反序列化对象。"""
    with np.load(path,allow_pickle=False) as values:return {k:values[k].copy() for k in values.files}


def verify_preservation(protocol):
    """核对每份清单及其所列文件的哈希。"""
    counts={}
    for name,record in protocol['preservation'].items():
        assert teacher.sha(BASE/name)==record['sha256']
        p=teacher.read(BASE/name)
        for item in p['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        assert len(p['files'])==record['files'];counts[name]=len(p['files'])
    teacher.verify_old()
    return counts


def audit():
    """所有原始梯度逐点核验；报告的统计单位仍是六个共享面板教师。"""
    assert not (RUN/'independent_audit.json').exists(),'Preserve existing audit'
    device=teacher.configure();protocol=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
    for name,value in protocol['sources'].items():
        assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
    frozen=teacher.read(RUN/'selection_freeze.json')
    assert teacher.sha(RUN/'selection_freeze.json')==(RUN/'selection_freeze.sha256').read_text().strip()
    for name,value in frozen['files'].items():assert teacher.sha(RUN/name)==value,name
    panel=load(RUN/'panel/data.npz');labels=panel['labels'];matrix,stencil=contrast_matrix()
    # 单独从全部先前图像重建像素集合，并对照官方数据与因素索引。
    import hashlib
    digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    prior_pixels=set()
    for split in ('train','validation','evaluation','translated'):
        prior_pixels.update(digest(x) for x in np.load(teacher.DATA/f'{split}_images.npy',mmap_mode='r',allow_pickle=False))
    prior_pixels.update(digest(x) for x in load(PRIOR/'panel/data.npz')['images'])
    hashes=[digest(x) for x in panel['images']]
    assert len(set(hashes))==36 and not set(hashes)&prior_pixels
    full=BASE/'data/dsprites'
    for file,key in [('imgs.npy','images'),('latents_classes.npy','factors'),('latents_values.npy','physical')]:
        official=np.load(full/file,mmap_mode='r',allow_pickle=False)
        compare('official_'+key,panel[key],official[panel['official_ids']])
    assert np.array_equal(panel['labels'],panel['factors'][:,1])
    assert np.array_equal(np.unique(panel['factors'][:,3]),[20,37])
    assert np.all(panel['factors'][:,4:]==15)
    assert len(set(map(tuple,panel['factors'][:,1:4])))==36
    rows=teacher.read(RUN/'summary.json')['rows']
    lookup={(r['seed'],r['alpha'],r['target'],r['method']):r for r in rows}
    bits=((np.arange(16)[:,None]>>np.arange(4))&1)
    for seed in teacher.SEEDS:
        selection_folder=RUN/f'selection/seed_{seed}'
        selection=load(selection_folder/'data.npz');pool=selection['pool'];shortlist=selection['shortlist'];nodes=selection['source_nodes']
        old=load(OLD/f'development/seed_{seed}/discovery.npz')
        for key in ('pool','shortlist','source_nodes'):compare('old_'+key,selection[key],old[key])
        compare('stencil',selection['stencil'],stencil)
        assert protocol['utc']<teacher.read(selection_folder/'completion.json')['utc']<frozen['utc']
        # 独立预测单点恢复/移植，再按两种输出目标重算收益。
        predictions=[]
        for i,label in enumerate(labels):
            delta=selection['delta'][i][:,shortlist]
            # 连续切片后take避免NumPy混合高级索引改变轴顺序。
            jac=np.take(selection['middle_jacobian'][i],shortlist,axis=-1)
            effect0=np.einsum('cj,sj->jsc',jac[0],delta)
            effectS=np.einsum('scj,sj->jsc',jac,delta)
            natural=selection['natural'][i]
            predictions.append(np.stack([np.stack([natural[None]-effect0,natural[None,:1]+effect0],1),
                                         np.stack([natural[None]-effectS,natural[None,:1]+effect0],1)]))
        predictions=np.asarray(predictions)
        predicted_scores=[];actual_scores=[]
        for target in ('margin','vector'):
            ni=interaction(selection['natural'],labels,target,matrix)
            pi=interaction(predictions,labels,target,matrix)
            si=interaction(selection['singleton_logits'],labels,target,matrix)
            n2=np.mean(ni**2,axis=(-2,-1))
            predicted_scores.append(n2[:,None,None]-.5*(np.mean(pi[:,:,:,0]**2,axis=(-2,-1))+
                np.mean((pi[:,:,:,1]-ni[:,None,None])**2,axis=(-2,-1))))
            actual_scores.append(n2[:,None]-.5*(np.mean(si[:,:,0]**2,axis=(-2,-1))+
                np.mean((si[:,:,1]-ni[:,None])**2,axis=(-2,-1))))
        predicted_scores=np.stack(predicted_scores,axis=2);actual_scores=np.stack(actual_scores,axis=1)
        compare('predicted_scores',selection['predicted_scores'],predicted_scores)
        compare('finite_scores',selection['finite_scores'],actual_scores)
        fixed=teacher.read(OLD/f'development/seed_{seed}/fixed_orders.json')
        for i,label in enumerate(labels):
            j0=selection['middle_jacobian'][i,0];mg=(3*j0[int(label)]-j0.sum(0))/2
            middle_i=matrix@selection['delta'][i]
            feature=np.mean(np.abs(middle_i*mg),axis=0)
            orders=[np.asarray(fixed['margin_class_finite'][int(label)]),shortlist[np.argsort(-feature[shortlist],kind='stable')],
                    shortlist[np.argsort(-selection['predicted_scores'][i,0,0],kind='stable')],
                    shortlist[np.argsort(-selection['predicted_scores'][i,1,0],kind='stable')],
                    shortlist[np.argsort(-selection['finite_scores'][i,0],kind='stable')]]
            orders +=[np.asarray(fixed[f'random_{r}']) for r in range(4)]
            expected=[pool[order[:16]] for order in orders]+[pool[shortlist],pool]
            for j,coords in enumerate(expected):
                compare('selection_coordinates',selection['selections'][i,j][selection['selections'][i,j]>=0],coords)
            # 四项消差可在理论零分数附近产生约 1e-18 的误差。名单按保存的分数排序，另行核对截断处的差值。
            for j,independent,recorded in [(2,predicted_scores[i,0,0],selection['predicted_scores'][i,0,0]),
                    (3,predicted_scores[i,1,0],selection['predicted_scores'][i,1,0]),
                    (4,actual_scores[i,0],selection['finite_scores'][i,0])]:
                chosen=np.argsort(-recorded,kind='stable')[:16]
                omitted=np.setdiff1d(np.arange(64),chosen)
                score_error=float(np.max(np.abs(independent-recorded)))
                arithmetic_scale=max(float(np.max(np.abs(independent))),float(np.max(np.abs(recorded))),1e-300)
                tolerance=2*score_error+32*np.finfo(np.float64).eps*arithmetic_scale
                gap=float(np.max(independent[omitted])-np.min(independent[chosen]))
                assert gap<=tolerance,(seed,i,METHODS[j],gap,tolerance)
                dense_order=np.argsort(-independent,kind='stable')[:16]
                if not np.array_equal(dense_order,chosen):
                    ROUND_OFF_RANKS.append(dict(seed=seed,image=i,method=METHODS[j],
                        same_selected_set=bool(set(dense_order)==set(chosen)),
                        independent_score_error=score_error,ranking_gap=gap,derived_tolerance=tolerance))
        model=teacher.load_teacher(seed,device)
        for parameter in model.parameters():parameter.requires_grad_(False)
        for alpha in STRENGTHS:
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz')
            assert frozen['utc']<teacher.read(folder/'completion.json')['utc']
            for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name
            compare('stencil',raw['stencil'],stencil)
            if alpha==.5:
                for key in ('natural','delta','middle_jacobian','source_jacobian'):
                    compare('half_strength_selection_'+key,raw[key],selection[key])
            for target in ('margin','vector'):
                natural=interaction(raw['natural'],labels,target,matrix)
                patched=interaction(raw['patched'],labels,target,matrix)
                n=np.mean(natural**2,axis=(-2,-1));r=np.mean(patched[:,:,0]**2,axis=(-2,-1))
                t=np.mean((patched[:,:,1]-natural[:,None])**2,axis=(-2,-1))
                metrics=load(folder/f'metrics_{target}.npz')
                compare('natural_mse',metrics['natural_mse'],n);compare('restore_mse',metrics['restore_mse'],r)
                compare('transplant_error_mse',metrics['transplant_error_mse'],t)
                for j,method in enumerate(METHODS):
                    row=lookup[seed,alpha,target,method]
                    compare('summary_ratios',np.array([row['restore_ratio'],row['transplant_error_ratio']]),
                            np.sqrt([r[:,j].mean()/n.mean(),t[:,j].mean()/n.mean()]))
            # 全中介池修补分别与原输出和自然移植输出逐值比较。
            compare('all_pool_restore',raw['patched'][:,-1,0],np.broadcast_to(raw['natural'][:,:1],raw['natural'].shape))
            compare('all_pool_transplant',raw['patched'][:,-1,1],raw['natural'])
            compare('frozen_gate_zero',np.einsum('km,nmc->nkc',matrix,raw['frozen_gate_logits']),np.zeros((36,24,3)))
            compare('frozen_gate_baseline',raw['frozen_gate_logits'][:,0],raw['natural'][:,0])
            linear=np.einsum('ncj,nmj->nmc',raw['middle_jacobian'][:,0],raw['delta'])
            li=np.einsum('km,nmc->nkc',matrix,linear)
            residual=raw['natural']-raw['natural'][:,:1]-linear
            compare('linear_decomposition',raw['linear_middle_interaction'],li)
            compare('nonlinear_decomposition',raw['nonlinear_tail_interaction'],np.einsum('km,nmc->nkc',matrix,residual))
            compare('decomposition_sum',raw['linear_middle_interaction']+raw['nonlinear_tail_interaction'],
                    np.einsum('km,nmc->nkc',matrix,raw['natural']))
            # 对全部36图、16组合及三类梯度做独立手推反向验证。
            for i,image in enumerate(panel['images']):
                x=torch.as_tensor(image[None,None],dtype=torch.float64,device=device)
                with torch.no_grad():source=model.block1(model.stem(x))
                altered=source.expand(16,-1,-1,-1).clone()
                altered.flatten(1)[:,nodes]*=1-alpha*torch.as_tensor(bits,dtype=torch.float64,device=device)
                logits,middle,mg,sg,z=independent_states_and_gradients(model,altered)
                mg=mg[:,:,pool].cpu().numpy();sg=sg[:,:,nodes].cpu().numpy()
                compare('natural_forward',raw['natural'][i],logits.cpu().numpy())
                compare('middle_delta',raw['delta'][i],(middle-middle[:1]).flatten(1)[:,pool].cpu().numpy())
                compare('manual_middle_jacobian',raw['middle_jacobian'][i],mg)
                compare('manual_source_jacobian',raw['source_jacobian'][i],sg)
                direction=-source.flatten()[nodes].cpu().numpy()[None]*bits
                compare('source_directional_derivative',raw['directional_derivative'][i],np.einsum('scj,sj->sc',sg,direction))
                flips=np.stack([((a>0)!=(a[:1]>0)).flatten(1).sum(1).cpu().numpy() for a in z],-1)
                compare('gate_flips',raw['gate_flips'][i],flips)
                for key,jac in [('middle',mg),('source',sg)]:
                    g=(3*jac[:,int(labels[i])]-jac.sum(1))/2
                    bn=np.linalg.norm(g[0]);nn=np.linalg.norm(g[1:],axis=-1)
                    drift=np.stack([np.linalg.norm(g[1:]-g[0],axis=-1)/max(bn,1e-30),
                        np.sum(g[1:]*g[0],axis=-1)/np.maximum(nn*bn,1e-30),nn],axis=-1)
                    compare(key+'_gradient_drift',raw[key+'_drift'][i],drift)
        print(json.dumps(dict(event='independent_teacher_audited',seed=seed),ensure_ascii=False),flush=True)
    preservation=verify_preservation(protocol)
    result=dict(status='PASS',utc=teacher.now(),teachers=6,images_per_teacher=36,strengths=4,
        maximum_errors=ERRORS,checked_scalar_counts=COUNT,old_delivery_files=preservation,
        gradient_validation='all states and classes, manual convolution adjoints with both residual branches',
        timing='all selections complete before joint seal, all evaluations after seal',pixel_overlap=0,
        floating_rank_notes=ROUND_OFF_RANKS,
        ranking_validation='Exact stable ranking from frozen raw scores; independent score arrays verified; independent rank gaps bounded by twice their measured round-off discrepancy')
    teacher.dump(RUN/'independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':audit()

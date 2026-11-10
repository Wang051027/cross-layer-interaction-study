"""用卷积伴随重算方向梯度，逐通道重建配对修补，并用稠密差分矩阵核算交互。"""
from __future__ import annotations

import hashlib
import itertools
import json
import dsprites_teacher as teacher
import numpy as np
import torch
from audit_dsprites_gradient import independent_states_and_gradients,contrast_matrix
from test_dsprites_shape_donor import hook_reference

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_shape_donor_20261010'
MAXIMUM={};COUNTS={};RANK_NOTES=[]


def load(path):
    """关闭句柄后返回数值数组，拒绝pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(key,a,b,tolerance=1e-9):
    """形状及有限性先检查，记录实际误差与核验标量数。"""
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.isfinite(error) and error<tolerance,(key,error)
    MAXIMUM[key]=max(MAXIMUM.get(key,0.),error);COUNTS[key]=COUNTS.get(key,0)+a.size


def pairs_from_factors(factors,labels):
    """独立枚举生成因素条件，禁止按模型结果配对或剔除。"""
    result=[]
    for key in sorted(set(map(tuple,factors[:,2:]))):
        ids=np.flatnonzero(np.all(factors[:,2:]==np.asarray(key),axis=1))
        assert len(ids)==3 and set(labels[ids])=={0,1,2}
        for r in range(3):
            for d in range(3):
                if r!=d:result.append([int(ids[labels[ids]==r][0]),int(ids[labels[ids]==d][0])])
    return np.asarray(result,dtype=np.int64)


def weights_for(labels,pairs):
    """有向类别目标由标签重建，不依赖原文件保存的投影向量。"""
    result=np.zeros((len(pairs),3));result[np.arange(len(pairs)),labels[pairs[:,0]]]=-1
    result[np.arange(len(pairs)),labels[pairs[:,1]]]=1;return result


def utility(natural,singletons,weights):
    """每配对的任务端点单点收益，先作独立类别投影再算平方量。"""
    origin=np.einsum('nc,nc->n',natural[:,0],weights)
    effect=np.einsum('nc,nc->n',natural[:,-1]-natural[:,0],weights)
    patched=np.einsum('nkoc,nc->nko',singletons,weights)-origin[:,None,None]
    return effect[:,None]**2-.5*(patched[:,:,0]**2+(patched[:,:,1]-effect[:,None])**2)


@torch.no_grad()
def independent_replacement(sr,sd,bits,alpha):
    """只赋值被替换组，其余通道保留，写法与主程序的广播混合独立。"""
    out=sr.expand(len(bits),-1,-1,-1).clone()
    for group in range(4):
        rows=np.flatnonzero(bits[:,group]);sl=slice(4*group,4*group+4)
        out[rows,sl]=(1-alpha)*sr[0,sl]+alpha*sd[0,sl]
    return out


@torch.no_grad()
def independent_singletons(model,middle):
    """逐通道构造恢复及移植，再以真实block3计算64个输出。"""
    restored=[];transplanted=[]
    for channel in range(32):
        r=middle[-1:].clone();r[:,channel]=middle[:1,channel]
        t=middle[:1].clone();t[:,channel]=middle[-1:,channel]
        restored.append(r);transplanted.append(t)
    states=torch.cat(restored+transplanted)
    logits=model.readout(model.block3(states).mean((2,3)))
    return logits.reshape(2,32,3).permute(1,0,2).cpu().numpy()


@torch.no_grad()
def independent_joint_patches(model,middle,selections):
    """以补集赋值另造全部联合修补，逐方案核对，不把单点响应相加。

    恢复从基线出发保留未选通道，移植从当前状态出发保留未选通道的
    基线值；与主程序的选中通道赋值互为独立构造。
    """
    result=[]
    for padded in selections:
        channels=padded[padded>=0];other=np.setdiff1d(np.arange(32),channels)
        restored=middle[:1].expand_as(middle).clone();restored[:,other]=middle[:,other]
        transplanted=middle.clone();transplanted[:,other]=middle[:1,other]
        states=torch.cat((restored,transplanted))
        logits=model.readout(model.block3(states).mean((2,3)))
        result.append(logits.reshape(2,len(middle),3).cpu().numpy())
    return np.asarray(result)


def reduced_gradients(middle,mg,sg,sr,sd,recipient,donor):
    """用NumPy点积另算通道和源组方向梯度，避开主程序Torch分组求和。"""
    n=len(middle);m=mg.cpu().numpy().reshape(n,3,32,1024)
    delta=(middle-middle[:1]).cpu().numpy().reshape(n,32,1024)
    base=np.einsum('ckp,nkp->nkc',m[0],delta)
    state=np.einsum('nckp,nkp->nkc',m,delta)
    s=sg.cpu().numpy().reshape(n,3,-1);source_delta=(sd-sr).cpu().numpy().ravel()
    source=np.stack([np.einsum('ncp,p->nc',s[:,:,g*16384:(g+1)*16384],source_delta[g*16384:(g+1)*16384]) for g in range(4)],axis=1)
    drift=[];norms=[]
    for jac in (s,m.reshape(n,3,-1)):
        g=jac[:,donor]-jac[:,recipient];norm=np.linalg.norm(g,axis=-1);baseline=np.linalg.norm(g[0]);norms.append(baseline)
        drift.append(np.stack((np.linalg.norm(g-g[0],axis=-1)/max(baseline,1e-30),
            np.sum(g*g[0],axis=-1)/np.maximum(norm*baseline,1e-30),norm),-1))
    return dict(source_effect=source,baseline_middle_effect=base,state_middle_effect=state,
                gradient_drift=np.stack(drift,axis=1),baseline_gradient_norm=np.asarray(norms))


def verify_ranking(seed,image,name,recorded,independent,selected):
    """精确验证冻结原分数排序；独立分数近并列用实际误差推导界检查。"""
    order=np.argsort(-recorded,kind='stable');compare('selected_'+name,selected,order[:len(selected)])
    rejected=np.setdiff1d(np.arange(32),selected);error=float(np.max(np.abs(recorded-independent)))
    scale=max(float(np.max(np.abs(recorded))),float(np.max(np.abs(independent))),1e-300)
    tolerance=2*error+32*np.finfo(np.float64).eps*scale
    gap=float(np.max(independent[rejected])-np.min(independent[selected])) if len(rejected) else 0.
    assert gap<=tolerance,(seed,image,name,gap,tolerance)
    if set(selected)!=set(np.argsort(-independent,kind='stable')[:len(selected)]):
        RANK_NOTES.append(dict(seed=seed,image=image,method=name,budget=len(selected),score_error=error,ranking_gap=gap,tolerance=tolerance))


def audit():
    """核验全部配对、单点输出、方向梯度和指标，并抽查原forward钩子。"""
    assert not (RUN/'independent_audit.json').exists(),'Preserve existing audit'
    device=teacher.configure();p=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    for name,value in p['sources'].items():assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
    freezes={name:teacher.read(RUN/f'{name}.json') for name in ('fit_freeze','selection_freeze')}
    for name,f in freezes.items():
        assert teacher.sha(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
        for path,value in f['files'].items():assert teacher.sha(RUN/path)==value,path
    panel=load(RUN/'panel/data.npz');labels=panel['labels'];pairs=pairs_from_factors(panel['factors'],labels)
    compare('matched_pairs',panel['pairs'],pairs);assert len(pairs)==72
    weights=weights_for(labels,pairs);matrix,stencil=contrast_matrix();bits=((np.arange(16)[:,None]>>np.arange(4))&1)
    digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    previous=set()
    for split in ('train','validation','evaluation','translated'):
        previous.update(digest(x) for x in np.load(teacher.DATA/f'{split}_images.npy',mmap_mode='r',allow_pickle=False))
    for name in ('dsprites_factor_transfer_20261007','dsprites_gradient_20261009'):
        previous.update(digest(x) for x in load(BASE/f'runs/{name}/panel/data.npz')['images'])
    hashes=[digest(x) for x in panel['images']];assert len(set(hashes))==36 and not set(hashes)&previous
    for file,key in [('imgs.npy','images'),('latents_classes.npy','factors'),('latents_values.npy','physical')]:
        official=np.load(BASE/'data/dsprites'/file,mmap_mode='r',allow_pickle=False)
        compare('official_'+key,panel[key],official[panel['official_ids']])
    compare('pixel_areas',panel['pixel_areas'],panel['images'].sum((1,2)))
    assert np.all(panel['factors'][:,4:]==15) and set(panel['factors'][:,3])=={17,22}
    specs=teacher.read(RUN/'specs.json');assert specs==p['specs'] and len(specs)==19
    lookup={(r['seed'],r['alpha'],r['target'],r['method'],r['budget']):r for r in teacher.read(RUN/'summary.json')['rows']}
    dev_pixels=np.load(teacher.DATA/'development_images.npy',allow_pickle=False)
    dev_labels=np.load(teacher.DATA/'development_labels.npy',allow_pickle=False)
    dev_factors=np.load(teacher.DATA/'development_factors.npy',allow_pickle=False)
    dev_pairs=pairs_from_factors(dev_factors,dev_labels);assert len(dev_pairs)==288
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for parameter in model.parameters():parameter.requires_grad_(False)
        fit=load(RUN/f'fitted/seed_{seed}/data.npz');fixed=teacher.read(RUN/f'fitted/seed_{seed}/orders.json')
        compare('development_pairs',fit['pairs'],dev_pairs)
        dw=weights_for(dev_labels,dev_pairs);actual=utility(fit['natural'],fit['singleton_logits'],dw)
        compare('fit_utilities',fit['scores'],actual)
        compare('fit_global_order',fixed['global_fixed'],np.argsort(-fit['scores'].mean(0),kind='stable'))
        for r in range(3):
            for d in range(3):
                if r==d:continue
                mask=(dev_labels[dev_pairs[:,0]]==r)&(dev_labels[dev_pairs[:,1]]==d)
                compare('fit_pair_order',fixed['pair_fixed'][f'{r}>{d}'],np.argsort(-fit['scores'][mask].mean(0),kind='stable'))
        for repeat in range(4):compare('random_order',fixed[f'random_{repeat}'],np.random.default_rng(seed+2026101000+repeat).permutation(32))
        assert p['utc']<teacher.read(RUN/f'fitted/seed_{seed}/completion.json')['utc']<freezes['fit_freeze']['utc']
        # 开发单点全部重算，另构造64输出；不只核对由原始响应算出的分数。
        with torch.no_grad():
            dev_sources=torch.cat([model.block1(model.stem(torch.as_tensor(dev_pixels[start:start+16,None],dtype=torch.float64,device=device))) for start in range(0,144,16)])
        for i,(r,d) in enumerate(dev_pairs):
            states=independent_replacement(dev_sources[r:r+1],dev_sources[d:d+1],np.array([[0]*4,[1]*4]),.5)
            with torch.no_grad():middle=model.block2(states);out=model.readout(model.block3(middle).mean((2,3)))
            compare('fit_natural_outputs',fit['natural'][i],out.cpu().numpy())
            compare('fit_singleton_outputs',fit['singleton_logits'][i],independent_singletons(model,middle))
        selection=load(RUN/f'selection/seed_{seed}/data.npz')
        assert freezes['fit_freeze']['utc']<teacher.read(RUN/f'selection/seed_{seed}/completion.json')['utc']<freezes['selection_freeze']['utc']
        compare('selection_pairs',selection['pairs'],pairs)
        predicted=[]
        for i in range(72):
            base=selection['baseline_middle_effect'][i,-1];state=selection['state_middle_effect'][i,-1]
            n=selection['natural'][i];trans=n[0]+base
            predicted.append(np.stack((np.stack((n[-1]-base,trans),1),np.stack((n[-1]-state,trans),1))))
        predicted=np.asarray(predicted)
        ps=np.stack([utility(selection['natural'],predicted[:,j],weights) for j in range(2)],axis=1)
        fs=utility(selection['natural'],selection['singleton_logits'],weights)
        compare('prediction_scores',selection['predicted_scores'],ps);compare('finite_scores',selection['finite_scores'],fs)
        for i,(r,d) in enumerate(pairs):
            for j,spec in enumerate(specs):
                name=spec['method'];selected=selection['selections'][i,j];selected=selected[selected>=0]
                assert len(selected)==spec['budget'] and len(set(selected))==len(selected)
                if name=='all_channels':compare('all_channels',selected,np.arange(32))
                elif name in ('fixed_linear','state_linear','input_finite'):
                    if name=='input_finite':recorded=selection['finite_scores'][i];independent=fs[i]
                    else:
                        k=0 if name=='fixed_linear' else 1;recorded=selection['predicted_scores'][i,k];independent=ps[i,k]
                    verify_ranking(seed,i,name,recorded,independent,selected)
                else:
                    order=fixed['pair_fixed'][f'{labels[r]}>{labels[d]}'] if name=='pair_fixed' else fixed[name]
                    compare('fixed_selection',selected,np.asarray(order)[:spec['budget']])
        with torch.no_grad():
            images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
            sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
            native=torch.cat([model(x) for x in images.split(16)]).cpu().numpy()
        compare('selection_native',selection['native_logits'],native)
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz')
            assert freezes['selection_freeze']['utc']<teacher.read(folder/'completion.json')['utc']
            for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name
            compare('raw_weights',raw['weights'],weights);compare('raw_stencil',raw['stencil'],stencil)
            compare('full_middle_restore',raw['patched'][:,-1,0],np.broadcast_to(raw['natural'][:,:1],(72,16,3)))
            compare('full_middle_transplant',raw['patched'][:,-1,1],raw['natural'])
            compare('frozen_interaction',np.einsum('ks,nsc->nkc',matrix,raw['frozen_gate_logits']),np.zeros((72,24,3)))
            if alpha==1.:compare('donor_endpoint',raw['natural'][:,-1],native[pairs[:,1]])
            for target in ('task_endpoint','interaction'):
                if target=='task_endpoint':
                    n=np.einsum('nc,nc->n',raw['natural'][:,-1]-raw['natural'][:,0],weights)
                    response=np.einsum('nmoc,nc->nmo',raw['patched'][:,:,:,-1]-raw['natural'][:,None,None,0],weights)
                else:
                    ni=np.einsum('ks,nsc->nkc',matrix,raw['natural']);pi=np.einsum('ks,nmosc->nmokc',matrix,raw['patched'])
                    n=np.einsum('nkc,nc->nk',ni,weights);response=np.einsum('nmokc,nc->nmok',pi,weights)
                if target=='task_endpoint':nm=n*n;rm=response[:,:,0]**2;tm=(response[:,:,1]-n[:,None])**2
                else:nm=np.mean(n*n,-1);rm=np.mean(response[:,:,0]**2,-1);tm=np.mean((response[:,:,1]-n[:,None])**2,-1)
                stored=load(folder/f'metrics_{target}.npz')
                for key,expected in [('natural_mse',nm),('restore_mse',rm),('transplant_error_mse',tm)]:compare(key,stored[key],expected)
                for j,spec in enumerate(specs):
                    row=lookup[seed,alpha,target,spec['method'],spec['budget']]
                    compare('summary_ratios',np.array([row['restore_ratio'],row['transplant_error_ratio']]),np.sqrt([rm[:,j].mean()/nm.mean(),tm[:,j].mean()/nm.mean()]))
            path=load(folder/'path.npz') if alpha==1. else None
            for i,(r,d) in enumerate(pairs):
                sr,sd=sources[r:r+1],sources[d:d+1];states=independent_replacement(sr,sd,bits,alpha)
                output,middle,mg,sg,z=independent_states_and_gradients(model,states)
                compare('natural_forward',raw['natural'][i],output.cpu().numpy())
                compare('joint_patch_outputs',raw['patched'][i],independent_joint_patches(model,middle,selection['selections'][i]))
                reduced=reduced_gradients(middle,mg,sg,sr,sd,int(labels[r]),int(labels[d]))
                for key,expected in reduced.items():compare(key,raw[key][i],expected)
                flips=np.stack([((a>0)!=(a[:1]>0)).flatten(1).sum(1).cpu().numpy() for a in z],-1)
                compare('gate_flips',raw['gate_flips'][i],flips)
                if alpha==.5:
                    for key in ('natural','source_effect','baseline_middle_effect','state_middle_effect','gradient_drift'):
                        compare('selection_endpoint_'+key,selection[key][i],raw[key][i,[0,15]])
                    compare('selection_singleton_outputs',selection['singleton_logits'][i],independent_singletons(model,middle[[0,15]]))
                if i==0:
                    # 新图上抽查真实forward的源替换与一个联合修补，不只看切口端点。
                    compare('hook_natural',raw['natural'][i],hook_reference(model,images[r:r+1],images[d:d+1],bits,alpha).cpu().numpy())
                    channels=selection['selections'][i,3,:4]
                    for op,mode in enumerate(('restore','transplant')):
                        compare('hook_'+mode,raw['patched'][i,3,op],hook_reference(model,images[r:r+1],images[d:d+1],bits,alpha,mode,channels).cpu().numpy())
                if path is not None:
                    positions=np.concatenate([(np.arange(n)+.5)/n for n in (8,32)])
                    compare('path_positions',path['positions'][i],positions)
                    a=torch.as_tensor(positions,dtype=torch.float64,device=device)[:,None,None,None]
                    states=(1-a)*sr+a*sd
                    path_output,path_middle,pm,pg,_=independent_states_and_gradients(model,states)
                    compare('path_outputs',path['path_logits'][i],path_output.cpu().numpy())
                    s=pg.cpu().numpy();target=s[:,int(labels[d])]-s[:,int(labels[r])];delta=(sd-sr).cpu().numpy().ravel()
                    derivative=np.stack([np.einsum('np,p->n',target[:,g*16384:(g+1)*16384],delta[g*16384:(g+1)*16384]) for g in range(4)],-1)
                    compare('path_derivatives',path['path_group_derivative'][i],derivative)
                    compare('path_integrals',path['integrated_groups'][i],np.stack((derivative[:8].mean(0),derivative[8:].mean(0))))
            print(json.dumps(dict(event='independent_frame_audited',seed=seed,alpha=alpha)),flush=True)
    preservation={}
    for name,record in p['preservation'].items():
        assert teacher.sha(BASE/name)==record['sha256']
        old=teacher.read(BASE/name)
        for item in old['files']:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        preservation[name]=len(old['files'])
    teacher.verify_old()
    result=dict(status='PASS',utc=teacher.now(),teachers=6,new_images=36,directed_pairs=72,
        maximum_errors=MAXIMUM,checked_scalar_counts=COUNTS,old_delivery_files=preservation,floating_rank_notes=RANK_NOTES,
        source_gradients='manual four-layer adjoints, all states and classes, same frozen implementation as Oct9 audit',
        singleton_validation='all development and all new half-strength singleton outputs independently reconstructed',
        joint_validation='all19 sets, all16 states, both operations independently reconstructed using complement assignments',
        path_validation='all8+32 sampling points independently differentiated, group sums and integration rechecked',
        timing='development fit freeze before new selection, selection seal before full-strength evaluation',pixel_overlap=0)
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':audit()

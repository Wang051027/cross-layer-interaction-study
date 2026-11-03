"""独立核验中间组合残差路由：原尾部、稠密差分、autograd及手推路径梯度。"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
from test_dsprites_residual_routes import original_holds
from test_dsprites_layer_transfer import independent_lower
from audit_dsprites_gradient import contrast_matrix,affine_adjoint

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_residual_routes_20261013';PRIOR=BASE/'runs/dsprites_layer_transfer_20261013'
ERRORS={};COUNTS={}


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(name,a,b,tol=1e-9):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
    e=float(np.max(abs(a-b))) if a.size else 0.;assert np.isfinite(e) and e<tol,(name,e)
    ERRORS[name]=max(ERRORS.get(name,0.),e);COUNTS[name]=COUNTS.get(name,0)+a.size


@torch.no_grad()
def tail(model,values):
    return torch.cat([model.readout(model.block3(x).mean((2,3))) for x in values.split(64)])


@torch.enable_grad()
def reference(model,a,h,w):
    """基准梯度用原forward的autograd，代替核心手写伴随；余项分别构造。"""
    b=model.block3;leaf=a.detach().clone().requires_grad_(True);weight=torch.as_tensor(w,device=a.device,dtype=a.dtype)
    base=model.readout(b(leaf).mean((2,3)));gradient=torch.autograd.grad((base@weight).sum(),leaf)[0].detach()
    with torch.no_grad():
        za=b.bn1(b.conv1(a));ua=b.mid(a);va=b.bn2(b.conv2(ua))+a;ga=za>0;gb=va>0
        zh=b.bn1(b.conv1(h));uh=b.mid(h);vh=b.bn2(b.conv2(uh))+h;eh=b.finish(uh,h)
    middle=uh.detach().clone().requires_grad_(True)
    z2=b.bn2(b.conv2(middle))+h
    forced=torch.where(gb,z2,torch.zeros_like(z2));fixed_out=model.readout(forced.mean((2,3)))
    p=torch.autograd.grad((fixed_out@weight).sum(),middle)[0].detach()
    with torch.no_grad():
        q=(weight@model.readout.weight)[None,:,None,None].expand_as(h)/1024
        r1=uh-torch.where(ga,zh,0);r2=eh-torch.where(gb,vh,0)
        first=p*r1;second=q*r2;linear=gradient*(h-a)
        vf=b.bn2(b.conv2(torch.where(ga,zh,0)))+h;updated=q*(F.relu(vf)-torch.where(gb,vf,0))
    return dict(base=base.detach(),ga=ga,gb=gb,zh=zh,vh=vh,uh=uh,eh=eh,za=za,va=va,first=first,second=second,linear=linear,gradient=gradient,updated_second=updated,r1=r1,r2=r2)


def choose(scores,ids,budget,kind):
    """Python排序与核心lexsort独立；近零分数采用协议规则。"""
    score=np.where(abs(scores)<1e-10,0,scores);ids=list(map(int,ids))
    if kind=='positive':ids=[i for i in ids if score[i]>0];key=lambda i:(-score[i],i)
    elif kind=='negative':ids=[i for i in ids if score[i]<0];key=lambda i:(score[i],i)
    else:key=lambda i:(-abs(score[i]),i)
    result=sorted(ids,key=key)[:budget];return np.array(result+[-1]*(budget-len(result)),dtype=np.int64)


@torch.no_grad()
def audit_path(model,a,h,w,raw):
    """独立伴随梯度重算每段通道导数；每段端点核对无遗漏内根。"""
    b=model.block3;r=h-a;bounds=raw['bounds'];mid=(bounds[:-1]+bounds[1:])/2;weight=torch.as_tensor(w,device=a.device,dtype=a.dtype)
    compare('path_midpoint',raw['midpoint'],mid);compare('path_width',raw['width'],np.diff(bounds))
    channels=[];events=[];previous=None
    for start in range(0,len(mid),64):
        ts=mid[start:start+64];x=a+torch.as_tensor(ts,device=a.device)[:,None,None,None]*r
        z1=b.bn1(b.conv1(x));u=F.relu(z1);z2=b.bn2(b.conv2(u))+x
        q=(weight@model.readout.weight)[None,:,None,None].expand_as(x)/1024
        v=q*(z2>0);p=affine_adjoint(b.conv2,b.bn2,v,u.shape)
        g=affine_adjoint(b.conv1,b.bn1,p*(z1>0),x.shape)+v
        channels.extend((g*r).sum((2,3)).cpu().numpy())
        flags=np.stack(((z1>0).flatten(1).cpu().numpy(),(z2>0).flatten(1).cpu().numpy()),1)
        for row,current in enumerate(flags):
            index=start+row
            if previous is not None:
                ss,cc=np.where(previous!=current)
                events.extend([[bounds[index],s,c,int(previous[s,c]),int(current[s,c])] for s,c in zip(ss,cc)])
            previous=current.copy()
    compare('path_manual_channel_gradient',raw['channel_derivative'],channels)
    integrated=(np.array(channels)*np.diff(bounds)[:,None]).sum(0);compare('path_integrated_channels',raw['integrated_channel'],integrated)
    end=(tail(model,torch.cat((a,h)))@weight).cpu().numpy();compare('path_endpoints',raw['endpoints'],end)
    compare('path_net_integral',integrated.sum(),end[1]-end[0]);compare('path_events',raw['events'],np.array(events).reshape(-1,5))
    # 端点预激活在每段内仿射，若两端越过零且幅度>1e-9，则有漏根。
    previous=None
    for offset in range(0,len(bounds),64):
        ts=bounds[offset:offset+64];x=a+torch.as_tensor(ts,device=a.device)[:,None,None,None]*r
        z1=b.bn1(b.conv1(x));z2=b.bn2(b.conv2(F.relu(z1)))+x
        values=np.stack((z1.flatten(1).cpu().numpy(),z2.flatten(1).cpu().numpy()),1)
        for current in values:
            if previous is not None:assert not np.any((previous*current<0)&(abs(previous)>1e-9)&(abs(current)>1e-9)),'missing path root'
            previous=current.copy()


def audit():
    assert not (RUN/'independent_audit.json').exists();device=teacher.configure()
    from run_dsprites_residual_routes import verify_seal,verify_folder
    freeze=verify_seal();protocol=teacher.read(RUN/'protocol.json');matrix,stencils=contrast_matrix()
    panel=load(BASE/'runs/dsprites_shape_donor_20261010/panel/data.npz');w=np.zeros((72,3));w[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1;w[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==456 and len(summary['decomposition'])==12
    summary_lookup={(x['seed'],x['alpha'],x['family'],x['method'],x['operation']):x for x in summary['rows']}
    response_maps=[];response_meta=[];response_coords=[];channel_edges=[]
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        with torch.no_grad():sources=torch.cat([model.block1(model.stem(x)) for x in torch.as_tensor(panel['images'][:,None],device=device,dtype=torch.float64).split(16)])
        selection=load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (.5,1.):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz');savedsel=load(folder/'selection.npz')
            assert teacher.read(folder/'completion.json')['utc']>freeze['utc']
            for key in selection:compare('frozen_'+key,savedsel[key],selection[key])
            old=load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz');compare('previous_natural',raw['modes'][:,:,0],old['logits'][:,:,0]);compare('previous_additive',raw['baseline'],old['lower_logits'][:,0])
            for case,(rr,dd) in enumerate(panel['pairs']):
                with torch.no_grad():
                    src=sources[rr:rr+1].expand(16,-1,-1,-1).clone()
                    for state in range(16):
                        for group in range(4):
                            if state&(1<<group):src[state,group*4:(group+1)*4]=(1-alpha)*sources[rr,group*4:(group+1)*4]+alpha*sources[dd,group*4:(group+1)*4]
                    h=model.block2(src);_,low=independent_lower(h.cpu().numpy());a=torch.as_tensor(low[0],device=device)
                ref=reference(model,a,h,w[case]);allids=np.arange(32768)
                for mode,(i1,i2) in enumerate(((None,None),(allids,None),(None,allids),(allids,allids))):compare('original_four_modes',raw['modes'][case,:,mode],original_holds(model,h,ref['ga'],ref['gb'],i1,i2).cpu().numpy())
                compare('baseline_original',raw['baseline'][case],ref['base'].cpu().numpy())
                compare('linear_autograd',raw['linear_channel'][case],ref['linear'].sum((2,3)).cpu().numpy())
                gate_channels=np.array([ref[k].sum((2,3)).cpu().numpy() for k in ('first','second')]);compare('gate_channels',raw['gate_channel'][case],gate_channels)
                compare('full_first_second_response',raw['full_first_response'][case],(ref['second']-ref['updated_second']).sum((1,2,3)).cpu().numpy())
                signed=[]
                for key in ('first','second'):
                    values=np.tensordot(matrix,ref[key].cpu().numpy(),axes=(1,0)).reshape(24,-1)
                    signed.append(np.stack((np.maximum(values,0).sum(1),np.minimum(values,0).sum(1)),1))
                compare('gate_coordinate_signed_mass',raw['gate_signed_mass'][case],np.array(signed).transpose(1,0,2))
                compare('gate_actual_flip_counts',raw['gate_flip_counts'][case],np.stack([((ref[k]>0)!=ref[g]).flatten(1).sum(1).cpu().numpy() for k,g in (('zh','ga'),('vh','gb'))],1))
                channel_fields=[]
                for c in range(32):
                    value=a.clone();value[:,c]=h[:,c];channel_fields.append(value)
                single=tail(model,torch.cat(channel_fields)).reshape(32,16,3).cpu().numpy();compare('all32_original_single_restore',raw['single_logits'][case],single)
                fields=[]
                for padded in selection['channels'][case]:
                    ids=padded[padded>=0];erase=h.clone();erase[:,ids]=a[:,ids];restore=a.clone();restore[:,ids]=h[:,ids];fields.extend((erase,restore))
                compare('all_original_channel_interventions',raw['channel_logits'][case],tail(model,torch.cat(fields)).reshape(9,2,16,3).cpu().numpy())
                for stage in range(2):
                    for index,padded in enumerate(selection['gates'][case,stage]):
                        ids=padded[padded>=0]
                        out=original_holds(model,h,ref['ga'],ref['gb'],ids if stage==0 else None,ids if stage==1 else None)
                        compare('all_original_gate_interventions',raw['gate_logits'][case,stage,index],out.cpu().numpy())
                        direct=ref['first' if stage==0 else 'second'].flatten(1)[:,ids].sum(1).cpu().numpy();compare('selected_gate_direct',raw['gate_direct'][case,stage,index],direct)
                        net=(raw['modes'][case,:,0]@w[case])-out.cpu().numpy()@w[case]
                        if stage==0:
                            compare('actual_first_net_identity',net,direct+raw['first_second_response'][case,index])
                            if seed==teacher.SEEDS[0] and case==0 and index in (0,5,6,7):
                                # 在预定例子中独立测下一开关的完整响应场；坐标排名仅描述。
                                with torch.no_grad():
                                    held=ref['uh'].clone();held.flatten(1)[:,ids]=torch.where(ref['ga'].flatten(1)[:,ids],ref['zh'].flatten(1)[:,ids],0)
                                    znew=model.block3.bn2(model.block3.conv2(held))+h
                                    q=(torch.as_tensor(w[case],device=device)@model.readout.weight)[None,:,None,None]/1024
                                    response=ref['second']-q*(F.relu(znew)-torch.where(ref['gb'],znew,0))
                                compare('example_measured_second_response',response.sum((1,2,3)).cpu().numpy(),raw['first_second_response'][case,index])
                                response_maps.append(response.cpu().numpy());response_meta.append([alpha,index])
                                for mask in (3,15):
                                    flat=response[mask].flatten().cpu().numpy();order=np.argsort(-abs(flat),kind='stable')[:8]
                                    for rank,coordinate in enumerate(order):
                                        ch,yy,xx=np.unravel_index(coordinate,(32,32,32))
                                        response_coords.append(dict(alpha=alpha,operation=index,mask=mask,rank=rank,channel=int(ch),y=int(yy),x=int(xx),response=float(flat[coordinate]),
                                            baseline=float(ref['va'][mask,ch,yy,xx]),natural=float(ref['vh'][mask,ch,yy,xx]),held=float(znew[mask,ch,yy,xx]),
                                            natural_open=bool(ref['vh'][mask,ch,yy,xx]>0),held_open=bool(znew[mask,ch,yy,xx]>0)))
                        else:compare('actual_second_net_identity',net,direct)
                if alpha==.5:
                    target=matrix@((raw['modes'][case,:,0]-raw['baseline'][case])@w[case])
                    cscore=target@(matrix@(single@w[case]-ref['base'].cpu().numpy()@w[case]).T)/np.sum(target**2)
                    compare('half_channel_score',selection['channel_score'][case],cscore)
                    rng=np.random.default_rng(seed+20261013000+case*97);rand=[rng.permutation(32) for _ in range(2)]
                    ch=choose(cscore,range(32),8,'absolute');pos=choose(cscore,range(32),8,'positive');neg=choose(cscore,range(32),8,'negative')
                    lists=[ch[:4],ch,pos,neg,rand[0][:4],rand[1][:4],rand[0][:8],rand[1][:8],np.arange(32)]
                    compare('half_channel_lists',selection['channels'][case],np.array([np.pad(v,(0,32-len(v)),constant_values=-1) for v in lists]))
                    for stage,key in enumerate(('first','second')):
                        values=ref[key].flatten(1).cpu().numpy();scores=target@(matrix@values)/np.sum(target**2)
                        ids=np.flatnonzero(np.max(abs(ref['r1' if stage==0 else 'r2'].flatten(1).cpu().numpy()),axis=0)>1e-10)
                        absolute=choose(scores,ids,128,'absolute');positive=choose(scores,ids,128,'positive');negative=choose(scores,ids,128,'negative');n=min(128,len(ids))
                        random=[np.pad(rng.permutation(ids)[:n],(0,128-n),constant_values=-1) for _ in range(2)]
                        lists=[absolute,*random,positive,negative,*[np.pad(v[:1],(0,127),constant_values=-1) for v in (absolute,positive,negative)]]
                        compare('half_gate_lists',selection['gates'][case,stage],lists)
                        compare('half_gate_pool_count',selection['gate_pool_count'][case,stage],len(ids))
                        compare('half_selected_gate_score',selection['selected_gate_score'][case,stage],np.array([scores[np.maximum(v,0)]*(v>=0) for v in lists]))
                if seed==teacher.SEEDS[0] and case==0:
                    example=load(folder/'example_fields.npz')
                    for key in example:compare('example_'+key,example[key],ref[key].cpu().numpy())
                    for mask in (3,15):audit_path(model,a[mask:mask+1],h[mask:mask+1],w[case],load(folder/f'path_mask{mask}.npz'))
                    # 逐中间通道测原卷积到固定开关坐标的预激活，连接通道→阈值。
                    positions=[]
                    for stage in (0,1):
                        for kind,index in (('selected1',5),('positive1',6),('negative1',7)):
                            coordinate=int(selection['gates'][case,stage,index,0])
                            if coordinate>=0:positions.append((stage,kind,coordinate))
                    # 加入预定干预的第二层响应最大点，只作解释，不用于再干预。
                    for mask in (3,15):
                        candidates=[x for x in response_coords if x['alpha']==alpha and x['operation']==0 and x['mask']==mask and x['rank']==0]
                        for x in candidates:positions.append((1,f'selected128_response_mask{mask}',x['channel']*1024+x['y']*32+x['x']))
                    selectedids=selection['channels'][case,1];selectedids=selectedids[selectedids>=0]
                    group=a.clone();group[:,selectedids]=h[:,selectedids]
                    measured=[]
                    with torch.no_grad():
                        for values in (*channel_fields,group):
                            z1=model.block3.bn1(model.block3.conv1(values));z2=model.block3.bn2(model.block3.conv2(F.relu(z1)))+values
                            measured.append([z1.flatten(1).cpu().numpy(),z2.flatten(1).cpu().numpy()])
                    for stage,kind,coordinate in positions:
                        ch,yy,xx=np.unravel_index(coordinate,(32,32,32));keya,keyh=('za','zh') if stage==0 else ('va','vh')
                        for mask in (3,15):
                            zbase=float(ref[keya][mask,ch,yy,xx]);znatural=float(ref[keyh][mask,ch,yy,xx]);zgroup=float(measured[-1][stage][mask,coordinate])
                            allchanges=np.array([m[stage][mask,coordinate]-zbase for m in measured[:32]])
                            if stage==0:
                                compare('channel_to_first_gate_additivity',allchanges.sum(),znatural-zbase)
                                compare('selected_group_first_gate_additivity',allchanges[selectedids].sum(),zgroup-zbase)
                            for c in range(32):channel_edges.append(dict(alpha=alpha,mask=mask,stage=stage+1,kind=kind,gate_channel=int(ch),gate_y=int(yy),gate_x=int(xx),middle_channel=c,
                                selected8=bool(c in selectedids),baseline=zbase,natural=znatural,solo=float(measured[c][stage][mask,coordinate]),group8=zgroup,
                                solo_change=float(allchanges[c]),natural_change=znatural-zbase,group_change=zgroup-zbase))
            base=np.einsum('nsk,nk->ns',raw['baseline'],w);modes=np.einsum('nsmk,nk->nsm',raw['modes'],w)-base[:,:,None]
            cp=lambda x:np.einsum('js,ns...->nj...',matrix,x)
            target=cp(modes[:,:,0]);channels=np.einsum('nvosk,nk->nvos',raw['channel_logits'],w)-base[:,None,None,:]
            gate=np.einsum('navsk,nk->navs',raw['gate_logits'],w)-base[:,None,None,:]
            conditions=[('full',name,'hold',cp(modes[:,:,j])) for j,name in enumerate(('natural','hold_first_all','hold_second_all','hold_both_all'))]
            chnames=('selected4','selected8','positive8','negative8','random0_4','random1_4','random0_8','random1_8','all32')
            gnames=('selected128','random0_128','random1_128','positive128','negative128','selected1','positive1','negative1')
            conditions +=[('channel',name,op,cp(channels[:,v,j])) for v,name in enumerate(chnames) for op,j in (('erase',0),('restore',1))]
            conditions +=[(f'gate{stage+1}',name,'hold',cp(gate[:,stage,v])) for stage in (0,1) for v,name in enumerate(gnames)]
            den=np.mean(target**2)
            for family,name,op,value in conditions:
                row=summary_lookup[seed,alpha,family,name,op]
                for key,result in [('target_rms',np.sqrt(den)),('remaining_ratio',np.sqrt(np.mean(value**2)/den)),('error_ratio',np.sqrt(np.mean((value-target)**2)/den)),('signed_removal',np.mean(target*(target-value))/den)]:compare('summary_'+key,row[key],result)
            linear=cp(raw['linear_channel']);gg=cp(raw['gate_channel'].transpose(0,2,1,3));pieces=np.concatenate((linear.sum(2)[:,:,None],gg.sum(3)),2)
            compare('conditional_decomposition',pieces.sum(2),target)
            row=next(x for x in summary['decomposition'] if x['seed']==seed and x['alpha']==alpha)
            compare('decomposition_rms',row['component_rms_ratios'],np.sqrt(np.mean(pieces**2,axis=(0,1))/den))
            compare('decomposition_projection',row['component_projection'],np.mean(pieces*target[:,:,None],axis=(0,1))/den)
            masses=abs(linear).sum()+raw['gate_signed_mass'][...,0].sum()-raw['gate_signed_mass'][...,1].sum()
            compare('coordinate_cancellation',row['coordinate_cancellation'],1-abs(target).sum()/masses);compare('layer_cancellation',row['layer_cancellation'],1-abs(target).sum()/abs(pieces).sum())
            print(json.dumps(dict(event='frame_audited',seed=seed,alpha=alpha,utc=teacher.now())),flush=True)
    old_counts={}
    for name,record in protocol['preservation'].items():
        path=BASE/name;assert teacher.sha(path)==record['sha256'];items=teacher.read(path)['files'];assert len(items)==record['files']
        for item in items:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        old_counts[name]=len(items)
    assert sum(old_counts.values())==1151;teacher.verify_old()
    np.savez_compressed(RUN/'response_examples.npz',response=np.array(response_maps),metadata=np.array(response_meta))
    teacher.dump(RUN/'response_coordinates.json',dict(scope='descriptive top8 at fixed example masks3/15;not used to select intervention',rows=response_coords))
    teacher.dump(RUN/'channel_gate_edges.json',dict(scope='fixed single-gate lists plus descriptive response coordinate;actual32single/group preactivations at predefined example;not new interventions',rows=channel_edges))
    teacher.dump(RUN/'independent_audit.json',dict(status='PASS',utc=teacher.now(),maximum_errors=ERRORS,checked_scalar_counts=COUNTS,old_delivery_files=old_counts,
        example_response_files={name:teacher.sha(RUN/name) for name in ('response_examples.npz','response_coordinates.json','channel_gate_edges.json')},
        method='original full andpartial gates;all32singles andall channel interventions;dense contrasts;original autograd baseline;independent manual path integral and roots;frozenhalf selection;all456 summaries',scope='same36images andstate-specific internal baseline;not unique causal attribution'))
    print(json.dumps(dict(status='PASS',maximum_error=max(ERRORS.values()),old_entries=1151)))


if __name__=='__main__':audit()

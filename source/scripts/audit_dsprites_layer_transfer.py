"""独立核验层间传递，不调用新核心的数值函数。

源字段逐组构造；原forward及where控制重放；中间场用稠密容斥矩阵
重建；梯度以独立卷积伴随和残差反向核对。旧六轮交付保持只读。
"""
from __future__ import annotations

import json
import dsprites_teacher as teacher
import numpy as np
import torch
from torch.nn import functional as F
from test_dsprites_layer_transfer import dense_design,original_modes,original_image,independent_lower
from audit_dsprites_gradient import affine_adjoint

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_layer_transfer_20261013'
SHAPE=BASE/'runs/dsprites_shape_donor_20261010';TRIPLE=BASE/'runs/dsprites_triple_gates_20261011'
ERRORS={};COUNTS={};NEGATIVE_ERRORS={}
ZERO_BAND=1e-9


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(name,a,b,tolerance=1e-9):
    """先核对形状，记录实际误差及核对标量数。"""
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
    error=float(np.max(abs(a-b))) if a.size else 0.
    assert np.isfinite(error) and error<tolerance,(name,error)
    ERRORS[name]=max(ERRORS.get(name,0.),error);COUNTS[name]=COUNTS.get(name,0)+a.size


def negative_fraction(name,saved,fields,axes):
    """零附近的符号依赖求和顺序；检查边界并另存有幅度的负值比例。

    主计算的严格<0计数保持原样。稠密矩阵和蝶形容斥在约1e-15
    处可能给出不同符号，因此不把这类计数误差当作场值不一致。
    1e-9与场值审计容差一致；有效负值定义为H<-1e-9。
    """
    fields=np.asarray(fields);saved=np.asarray(saved)
    robust=np.mean(fields < -ZERO_BAND,axis=axes)
    possible=np.mean(fields < ZERO_BAND,axis=axes)
    assert saved.shape==robust.shape,(name,saved.shape,robust.shape)
    assert np.all(saved >= robust-1e-12) and np.all(saved <= possible+1e-12),name
    exact=np.mean(fields<0,axis=axes)
    NEGATIVE_ERRORS[name]=max(NEGATIVE_ERRORS.get(name,0.),float(np.max(abs(saved-exact))))
    return robust


def independent_contrasts():
    """按集合成员生成24个对比向量，避免复用主差分索引。"""
    import itertools
    members=[{j for j in range(4) if s&(1<<j)} for s in range(16)];lookup={frozenset(s):i for i,s in enumerate(members)}
    rows=[];metadata=[]
    for a,b in itertools.combinations(range(4),2):
        for background,g in enumerate(members):
            if a in g or b in g:continue
            row=np.zeros(16)
            for included,sign in ((g|{a,b},1),(g|{a},-1),(g|{b},-1),(g,1)):row[lookup[frozenset(included)]]=sign
            rows.append(row);metadata.append([a,b,background])
    return np.array(rows),np.array(metadata)


@torch.no_grad()
def original_tail(model,values):
    """用原block3.forward重新计算实际内部反事实，批次不超过64。"""
    x=torch.as_tensor(values,dtype=torch.float64,device=next(model.parameters()).device)
    return torch.cat([model.readout(model.block3(v).mean((2,3))) for v in x.split(64)]).cpu().numpy()


@torch.no_grad()
def reverse_task(model,source,weight):
    """手推读出→block3→block2两主支和两旁路，不用autograd。"""
    b2,b3=model.block2,model.block3
    z1=b2.bn1(b2.conv1(source));h1=torch.where(z1>0,z1,torch.zeros_like(z1))
    z2=b2.bn2(b2.conv2(h1))+b2.shortcut(source);h=torch.where(z2>0,z2,torch.zeros_like(z2))
    z3=b3.bn1(b3.conv1(h));h3=torch.where(z3>0,z3,torch.zeros_like(z3))
    z4=b3.bn2(b3.conv2(h3))+h;end=torch.where(z4>0,z4,torch.zeros_like(z4))
    w=torch.as_tensor(weight,dtype=torch.float64,device=source.device)
    q=(w@model.readout.weight)[None,:,None,None].expand_as(end)/1024
    g4=q*(z4>0);g3=affine_adjoint(b3.conv2,b3.bn2,g4,h3.shape)*(z3>0)
    mg=affine_adjoint(b3.conv1,b3.bn1,g3,h.shape)+g4
    g2=mg*(z2>0);g1=affine_adjoint(b2.conv2,b2.bn2,g2,h1.shape)*(z1>0)
    sg=affine_adjoint(b2.conv1,b2.bn1,g1,source.shape)+affine_adjoint(b2.shortcut[0],b2.shortcut[1],g2,source.shape)
    return h,mg,sg,(z1,z2,z3,z4)


def audit():
    """所有教师、强度和案例都核验；全部成功后才落盘PASS。"""
    assert not (RUN/'independent_audit.json').exists();device=teacher.configure()
    p=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in p['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    assert teacher.sha(RUN/'channel_orders.npz')==p['channel_orders_sha256']
    d=load(RUN/'design.npz');m,z,degree=dense_design();matrix,metadata=independent_contrasts()
    compare('mobius_design',d['mobius'],m);compare('zeta_design',d['zeta'],z);compare('order_design',d['order'],degree);compare('pair_metadata',d['metadata'],metadata)
    direct=np.zeros((24,16));a,b,c,e=d['stencils'].T
    direct[np.arange(24),a]=1;direct[np.arange(24),b]=-1;direct[np.arange(24),c]=-1;direct[np.arange(24),e]=1
    compare('contrast_design',direct,matrix)
    joint=np.zeros(16);joint[15]=1;joint[0]=3;joint[[1,2,4,8]]=-1;compare('joint_design',d['joint'],joint)
    panel=load(SHAPE/'panel/data.npz');orders=load(RUN/'channel_orders.npz');weights=np.zeros((72,3))
    weights[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1;weights[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==576 and len(summary['decomposition'])==24
    rows={(r['seed'],r['alpha'],r['metric'],r['method'],r['operation']):r for r in summary['rows']}
    specs=(('selected',4),('selected',8),('random0',4),('random1',4),('random0',8),('random1',8),('all',32))
    negative_records={k:[] for k in ('seed','alpha','lower','patched','permuted')}
    for j,seed in enumerate(teacher.SEEDS):
        inherited=load(TRIPLE/f'discovery/seed_{seed}/data.npz')['channels'];compare('inherited_channels',orders['selected'][j],inherited)
        expected=np.array([np.random.default_rng(seed+2026101300+v).permutation(32) for v in range(2)]);compare('fixed_random_orders',orders['random'][j],expected)
        model=teacher.load_teacher(seed,device)
        for parameter in model.parameters():parameter.requires_grad_(False)
        images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
        with torch.no_grad():sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz');old=load(SHAPE/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
            completion=teacher.read(folder/'completion.json');assert completion['utc']>p['utc']
            for name,value in completion['files'].items():assert teacher.sha(folder/name)==value,name
            compare('previous_natural',raw['logits'][:,:,0],old['natural']);compare('previous_all_gate_fixed',raw['logits'][:,:,3],old['frozen_gate_logits'])
            compare('previous_source_directions',raw['directional_source'],alpha*np.einsum('nsgk,nk->nsg',old['source_effect'],weights))
            compare('previous_gradient_drift',raw['gradient_drift'],old['gradient_drift'])
            compare('inherited_saved_channels',raw['channels'],inherited);compare('saved_random',raw['random_orders'],expected)
            frame_negatives={k:[] for k in ('lower','patched','permuted')}
            for i,(recipient,donor) in enumerate(panel['pairs']):
                # 从接收体出发逐组赋值，与主程序的完整mask广播分别实现。
                src=sources[recipient:recipient+1].expand(16,-1,-1,-1).clone()
                for state in range(16):
                    for group in range(4):
                        if state&(1<<group):
                            ids=slice(4*group,4*group+4)
                            src[state,ids]=(1-alpha)*sources[recipient,ids]+alpha*sources[donor,ids]
                modes,middle,fixed_middle=original_modes(model,src)
                compare('original_four_modes',raw['logits'][i],modes.cpu().numpy())
                compare('whole_model_source_hook',raw['logits'][i,:,0],original_image(model,images[recipient:recipient+1],src).cpu().numpy())
                value=middle.cpu().numpy();co,lower=independent_lower(value)
                source_co=np.tensordot(m,src.cpu().numpy(),axes=(1,0));norm=np.linalg.norm(source_co.reshape(16,16,-1),axis=2)
                compare('source_coefficient_norm',raw['source_mobius_channel_norm'][i],norm)
                compare('source_has_no_high_orders',norm[degree>=2],np.zeros((11,16)))
                compare('middle_coefficient_norm',raw['middle_mobius_channel_norm'][i],np.linalg.norm(co.reshape(16,32,-1),axis=2))
                compare('additive_middle_zero_high_orders',np.tensordot(m,lower[0],axes=(1,0))[degree>=2],np.zeros((11,32,32,32)))
                compare('upstream_affine_middle_zero_high_orders',np.tensordot(m,fixed_middle.cpu().numpy(),axes=(1,0))[degree>=2],np.zeros((11,32,32,32)))
                fields=[];selected_rows=[];negative=[]
                for method,budget in specs:
                    ids=np.arange(32) if method=='all' else inherited[i,:budget] if method=='selected' else expected[int(method[-1]),:budget]
                    selected_rows.append(np.pad(ids,(0,32-len(ids)),constant_values=-1))
                    erased=value.copy();erased[:,ids]=lower[0][:,ids]
                    restored=lower[0].copy();restored[:,ids]=value[:,ids];fields.extend((erased,restored))
                    negative.append([np.mean(v<0,axis=(1,2,3)) for v in (erased,restored)])
                residue=value-lower[0];permuted=np.array([lower[0]+residue[:,order] for order in expected])
                all_fields=np.concatenate((lower.reshape(-1,32,32,32),np.concatenate(fields),permuted.reshape(-1,32,32,32),np.maximum(lower[0],0)))
                out=original_tail(model,all_fields)
                compare('original_lower_orders',raw['lower_logits'][i],out[:48].reshape(3,16,3))
                compare('original_channel_erasure_restore',raw['patched_logits'][i],out[48:272].reshape(7,2,16,3))
                compare('original_equal_norm_permutation',raw['permuted_logits'][i],out[272:304].reshape(2,16,3))
                compare('original_rectified_additive',raw['rectified_logits'][i],out[304:])
                compare('fixed_channel_lists',raw['selections'][i],selected_rows)
                frame_negatives['lower'].append(negative_fraction('lower_negative_fraction',raw['lower_negative_fraction'][i],lower,(2,3,4)))
                frame_negatives['patched'].append(negative_fraction('patch_negative_fraction',raw['patched_negative_fraction'][i],np.array(fields).reshape(7,2,16,32,32,32),(3,4,5)))
                frame_negatives['permuted'].append(negative_fraction('permutation_negative_fraction',raw['permuted_negative_fraction'][i],permuted,(2,3,4)))
                compare('permutation_norm',np.linalg.norm((permuted-lower[0]).reshape(2,16,-1),axis=2),np.broadcast_to(np.linalg.norm(residue.reshape(16,-1),axis=1),(2,16)),1e-8)
                h,mg,sg,zz=reverse_task(model,src,weights[i]);compare('manual_original_middle',h.cpu().numpy(),value)
                g=mg.cpu().numpy();direction=alpha*(sources[donor:donor+1]-sources[recipient:recipient+1])
                directional=(sg*direction).reshape(16,4,4,64,64).sum((2,3,4)).cpu().numpy()
                compare('source_chain_direction',raw['directional_source'][i],directional)
                inter=np.tensordot(matrix,value,axes=(1,0));compare('conditional_middle_norm',raw['conditional_middle_channel_norm'][i],np.linalg.norm(inter.reshape(24,32,-1),axis=2))
                compare('baseline_channel_transmission',raw['conditional_baseline_channel_projection'][i],np.sum(inter*g[:1],axis=(2,3)))
                compare('local_context_channel_transmission',raw['conditional_context_channel_projection'][i],np.sum(inter*g[d['stencils'][:,3]],axis=(2,3)))
                compare('mobius_channel_transmission',raw['baseline_mobius_channel_projection'][i],np.sum(co*g[:1],axis=(2,3)))
                compare('actual_gate_changes',raw['gate_flip_counts'][i],np.stack([np.sum((zv>0).cpu().numpy()!=(zv[:1]>0).cpu().numpy(),axis=(1,2,3)) for zv in zz],1))
                drift=[];base=[]
                for gradient in (sg.cpu().numpy(),g):
                    flat=gradient.reshape(16,-1);n=np.linalg.norm(flat,axis=1);bn=n[0]
                    rel=np.linalg.norm(flat-flat[:1],axis=1)/max(bn,1e-30);cos=(flat*flat[:1]).sum(1)/np.maximum(n*bn,1e-30)
                    if bn<1e-20:rel*=0;cos*=0
                    drift.append(np.stack((rel,cos,n),1));base.append(bn)
                compare('manual_gradient_drift',raw['gradient_drift'][i],np.stack(drift,1));compare('baseline_gradient_norm',raw['baseline_gradient_norm'][i],base)
                fixed=raw['logits'][i,:,2]@weights[i]
                compare('upstream_interaction_survives_affine_tail',np.tensordot(m,fixed,axes=(1,0))[1:],raw['baseline_mobius_channel_projection'][i].sum(1)[1:])
                compare('both_modules_affine_zero',np.tensordot(m,raw['logits'][i,:,3],axes=(1,0))[degree>=2],np.zeros((11,3)))
                if seed==teacher.SEEDS[0] and i==0:
                    example=load(folder/'example_fields.npz')
                    for name,reference in [('source',src.cpu().numpy()),('natural_middle',value),('additive_middle',lower[0]),('middle_coefficients',co),('source_gradient',sg.cpu().numpy()),('middle_gradient',g)]:compare('example_'+name,example[name],reference)
            # 汇总在本轮中已保存；用稠密矩阵另核对分母、RMS及带符号量。
            modes=np.einsum('nsmk,nk->nsm',raw['logits'],weights)
            lower=np.einsum('nqsk,nk->nqs',raw['lower_logits'],weights)
            patch=np.einsum('nvosk,nk->nvos',raw['patched_logits'],weights)
            perm=np.einsum('nvsk,nk->nvs',raw['permuted_logits'],weights);rect=np.einsum('nsk,nk->ns',raw['rectified_logits'],weights)
            for metric,cm in [('conditional_pairs',matrix),('joint_all4',joint[None])]:
                change=lambda x:np.einsum('js,ns->nj',cm,x)
                natural=change(modes[:,:,0]);add=change(lower[:,0]);target=natural-add
                conditions=[(name,'forward',natural,change(modes[:,:,index])) for index,name in enumerate(('natural','upstream_fixed','downstream_fixed','both_fixed'))]
                conditions +=[(f'middle_le{k}','erase',natural,change(lower[:,k-1])) for k in (1,2,3)]
                for index,(method,budget) in enumerate(specs):
                    conditions.extend([(f'{method}{budget}','erase',natural,change(patch[:,index,0])),(f'{method}{budget}','restore',target,change(patch[:,index,1])-add)])
                conditions +=[(f'permuted{v}','forward',natural,change(perm[:,v])) for v in range(2)]
                conditions.append(('rectified_additive','erase',natural,change(rect)))
                for method,op,original,response in conditions:
                    row=rows[seed,alpha,metric,method,op];den=np.mean(original**2)
                    compare('summary_natural_rms',row['natural_rms'],np.sqrt(den))
                    if den>=1e-20:
                        compare('summary_remaining',row['remaining_ratio'],np.sqrt(np.mean(response**2)/den))
                        compare('summary_changed',row['changed_ratio'],np.sqrt(np.mean((response-original)**2)/den))
                        compare('summary_signed_removal',row['signed_removal'],np.mean(original*(original-response))/den)
                        if op=='restore':compare('summary_restore_error',row['restore_error_ratio'],np.sqrt(np.mean((response-original)**2)/den))
            natural=np.einsum('js,ns->nj',matrix,modes[:,:,0]);den=np.mean(natural**2)
            for baseline,key in [('global','conditional_baseline_channel_projection'),('context','conditional_context_channel_projection')]:
                transmitted=raw[key].sum(2);remainder=natural-transmitted
                row=next(r for r in summary['decomposition'] if r['seed']==seed and r['alpha']==alpha and r['baseline']==baseline)
                if den>=1e-20:
                    compare('summary_transmitted',row['transmitted_rms_ratio'],np.sqrt(np.mean(transmitted**2)/den))
                    compare('summary_tail_remainder',row['downstream_remainder_rms_ratio'],np.sqrt(np.mean(remainder**2)/den))
                    compare('summary_transmitted_projection',row['transmitted_projection'],np.mean(natural*transmitted)/den)
                    compare('summary_tail_projection',row['downstream_projection'],np.mean(natural*remainder)/den)
            negative_records['seed'].append(seed);negative_records['alpha'].append(alpha)
            for name in frame_negatives:negative_records[name].append(np.array(frame_negatives[name]))
            print(json.dumps(dict(event='frame_audited',seed=seed,alpha=alpha,utc=teacher.now())),flush=True)
    old_counts={}
    for name,value in p['preservation'].items():
        path=BASE/name;assert teacher.sha(path)==value['sha256'];records=teacher.read(path)['files'];assert len(records)==value['files']
        for item in records:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        old_counts[name]=len(records)
    teacher.verify_old()
    np.savez_compressed(RUN/'negative_fraction_audit.npz',**{k:np.array(v) for k,v in negative_records.items()},zero_band=ZERO_BAND)
    result=dict(status='PASS',utc=teacher.now(),maximum_errors=ERRORS,checked_scalar_counts=COUNTS,old_delivery_files=old_counts,
        negative_fraction=dict(zero_band=ZERO_BAND,strict_sign_fraction_discrepancies=NEGATIVE_ERRORS,
            rule='strict<0 sign fractions bounded by independently reconstructed +/-1e-9 band;report robust H<-1e-9 fractions',
            file='negative_fraction_audit.npz',sha256=teacher.sha(RUN/'negative_fraction_audit.npz')),
        method='all16 source states;original4gate-modes andmodel hook;allmiddle-order/channel counterfactuals;dense inclusion-exclusion;manual adjoint source/middle chain;allsummary rows',
        scope='same seen36images/72directedpairs;no new input selection or unique causal attribution')
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(dict(status='PASS',maximum_error=max(ERRORS.values()),old_delivery_file_entries=sum(old_counts.values()))))


if __name__=='__main__':audit()

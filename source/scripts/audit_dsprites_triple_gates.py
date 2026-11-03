"""独立核验全部56纯三阶、条件三阶、符号抵消与真实门控反事实。

不用新核心模块的数值函数。子集用成员集合生成，差分采用独立容斥
矩阵；原block3.forward以where改开关，伴随与autograd共同核对坐标。
"""
from __future__ import annotations

import dsprites_teacher as teacher
import itertools
import json
import numpy as np
import torch
from test_dsprites_channel_gates import forward_override
from test_dsprites_triple_gates import original_partial
from audit_dsprites_gradient import affine_adjoint

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_triple_gates_20261011'
PRIOR=BASE/'runs/dsprites_channel_gates_20261011';SHAPE=BASE/'runs/dsprites_shape_donor_20261010'
MAXIMUM={};COUNTS={};RANK_NOTES=[]
NAMES=('selected32','random0_32','random1_32','selected128','random0_128','random1_128','aligned_subset','opposed_subset')


def load(path):
    """纯数值读取，不修改历史文件。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(key,a,b,tol=1e-9):
    """先核对形状，保存实际误差与标量核验数量。"""
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.isfinite(error) and error<tol,(key,error)
    MAXIMUM[key]=max(MAXIMUM.get(key,0.),error);COUNTS[key]=COUNTS.get(key,0)+a.size


def independent_design():
    """成员集合容斥，避免复用主程序八项累计和二进制蝶形变换。"""
    members=[{j for j in range(8) if s&(1<<j)} for s in range(256)]
    states=[g for g in members if len(g)<=3 or g==set(range(4)) or len(g)==8]
    masks=np.array([sum(1<<j for j in g) for g in states]);lookup={frozenset(g):i for i,g in enumerate(states)}
    triples=list(itertools.combinations(range(8),3));pairs=list(itertools.combinations(range(8),2))
    tm=np.zeros((56,95));pm=np.zeros((28,95));jm=np.zeros((2,95))
    for matrix,sets in ((tm,triples),(pm,pairs)):
        for row,group in enumerate(sets):
            g=set(group)
            for subset in states:
                if subset<=g:matrix[row,lookup[frozenset(subset)]]=(-1)**(len(g)-len(subset))
    for row,k in enumerate((4,8)):
        jm[row,lookup[frozenset(range(k))]]=1;jm[row,lookup[frozenset()]]=k-1
        for c in range(k):jm[row,lookup[frozenset([c])]]=-1
    signs=np.array([(-1)**(3-s.bit_count()) for s in range(8)])
    inc=np.zeros((16,16))
    for s in range(16):
        for t in range(16):
            if t&s==t:inc[s,t]=(-1)**(s.bit_count()-t.bit_count())
    return masks,states,triples,tm,pm,jm,signs,inc


@torch.no_grad()
def trace(model,x):
    """沿原模块序列用where记录自然状态，基准是当地背景的第0行。"""
    b=model.block3;z1=b.bn1(b.conv1(x));h1=torch.where(z1>0,z1,torch.zeros_like(z1))
    z2=b.bn2(b.conv2(h1))+b.shortcut(x);h2=torch.where(z2>0,z2,torch.zeros_like(z2))
    return dict(x=x,z1=z1,z2=z2,h1=h1,h2=h2,g1=z1[:1]>0,g2=z2[:1]>0,
        r1=h1-(z1[:1]>0)*z1,r2=h2-(z2[:1]>0)*z2)


def units(model,t,weight,signs):
    """用旧伴随及自动微分求传播权重，八项稠密矩阵作用于坐标余项。"""
    w=torch.as_tensor(weight,dtype=torch.float64,device=t['x'].device)
    q=(w@model.readout.weight)[:,None,None]/1024
    p=affine_adjoint(model.block3.conv2,model.block3.bn2,q[None]*t['g2'],t['h1'][:1].shape)
    leaf=t['h1'][:1].detach().clone().requires_grad_(True)
    z=model.block3.bn2(model.block3.conv2(leaf))+t['x'][:1]
    output=model.readout((t['g2']*z).mean((2,3)))
    grad=torch.autograd.grad((output*w).sum(),leaf)[0]
    compare('adjoint_autograd',p.cpu().numpy(),grad.cpu().numpy(),1e-12)
    a=np.einsum('m,mchw->chw',signs,t['r1'].cpu().numpy())*p[0].cpu().numpy()
    b=np.einsum('m,mchw->chw',signs,t['r2'].cpu().numpy())*q.cpu().numpy()
    return np.stack((a,b))


def all_modes(model,x):
    """调用原block3.forward，四模式核验不依赖新trace或尾部修补。"""
    values=[]
    for m in range(4):values.append(forward_override(model,x,x[:1],stage=0 if m==1 else 1 if m==2 else None,both=m==3).cpu().numpy())
    return np.stack(values,1)


def inputs_from_members(base,ref,channels,members):
    """从参考状态开始，将所有未包含的32通道恢复基准，独立于主patch。"""
    x=ref.expand(len(members),-1,-1,-1).clone()
    for row,g in enumerate(members):
        included={int(channels[j]) for j in g};other=np.array(sorted(set(range(32))-included))
        x[row,other]=base[0,other]
    return x


def audit():
    """每个teacher/alpha/case/operation都核验；只有全部通过才保存PASS。"""
    assert not (RUN/'independent_audit.json').exists();device=teacher.configure()
    protocol=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
    freeze=teacher.read(RUN/'selection_freeze.json');assert teacher.sha(RUN/'selection_freeze.json')==(RUN/'selection_freeze.sha256').read_text().strip()
    assert freeze['protocol_sha256']==teacher.sha(RUN/'protocol.json')
    for name,value in freeze['files'].items():assert teacher.sha(RUN/name)==value,name
    masks,members,triples,tm,pm,jm,signs,inc=independent_design();d=load(RUN/'design.npz')
    compare('design_masks',d['masks'],masks);compare('design_triples',d['triples'],triples)
    compare('design_bits',d['bits'],((masks[:,None]>>np.arange(8))&1));compare('design_signs',d['signs'],signs)
    panel=load(SHAPE/'panel/data.npz');labels=panel['labels'];weights=np.zeros((72,3))
    weights[np.arange(72),labels[panel['pairs'][:,1]]]=1;weights[np.arange(72),labels[panel['pairs'][:,0]]]=-1
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==1536 and len(summary['hierarchy'])==72
    rows={(r['seed'],r['alpha'],r['operation'],r['ranks'],r['context'],r['stage'],r['variant']):r for r in summary['rows']}
    old_design=load(PRIOR/'design.npz');old_ids=[int(np.flatnonzero(masks==s)[0]) for s in old_design['masks']]
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz');prior_selection=load(PRIOR/f'discovery/seed_{seed}/data.npz')
        compare('inherited_channels',frozen['channels'],prior_selection['channels'])
        assert protocol['utc']<teacher.read(RUN/f'discovery/seed_{seed}/completion.json')['utc']<freeze['utc']
        with torch.no_grad():
            images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
            sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
            native=torch.cat([model.block2(x) for x in sources.split(16)])
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz');old=load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
            assert freeze['utc']<teacher.read(folder/'completion.json')['utc']
            for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name
            compare('old43_modes',raw['logits'][:,:,old_ids],old['logits']);compare('old_pair_effects',raw['pair_logits'],old['pair_logits'])
            compare('old_joint_effects',raw['joint_logits'],old['joint_logits']);compare('old_four_coefficients',raw['mobius4'],old['mobius4'])
            compare('old_orders3plus',raw['higher3'],old['higher_than_pair8'])
            for key in ('selected_triples','selected_gates','random_gates','aligned_gates','opposed_gates','eligible_counts','alignment_sign'):
                compare('unchanged_selection_'+key,raw[key],frozen[key])
            if alpha==.5:
                for key in frozen:
                    if key in raw:compare('half_repeated_'+key,raw[key],frozen[key])
            natural=np.einsum('norcmk,nk->norcm',raw['context_effects'],weights)[...,0]
            partial=np.einsum('norcsvtk,nk->norcsvt',raw['partial_logits'],weights);held=np.einsum('norcsvt,t->norcsv',partial,signs)
            for op in range(2):
                for ranks in ((0,),(0,1,2)):
                    for ctx in range(2):
                        original=natural[:,op,list(ranks),ctx];den=np.mean(original**2)
                        for stage in range(2):
                            for variant,name in enumerate(NAMES):
                                v=held[:,op,list(ranks),ctx,stage,variant];row=rows[seed,alpha,('restore','transplant')[op],'first' if len(ranks)==1 else 'three',('empty','complement')[ctx],stage+1,name]
                                compare('summary_ratio',row['remaining_ratio'],np.sqrt(np.mean(v**2)/den))
                                compare('summary_projection',row['signed_remaining'],np.mean(original*v)/den)
            for i,(r,donor) in enumerate(panel['pairs']):
                with torch.no_grad():endpoint=model.block2((1-alpha)*sources[r:r+1]+alpha*sources[donor:donor+1])
                for op in range(2):
                    base,ref=(endpoint,native[r:r+1]) if op==0 else (native[r:r+1],endpoint)
                    x=inputs_from_members(base,ref,frozen['channels'][i],members);actual=all_modes(model,x)
                    compare('original_global_modes',raw['logits'][i,op],actual)
                    t3=np.einsum('pm,msc->psc',tm,actual);p2=np.einsum('pm,msc->psc',pm,actual);joint=np.einsum('km,msc->ksc',jm,actual)
                    compare('all56_triples',raw['triple_logits'][i,op],t3);compare('all28_pairs',raw['pair_logits'][i,op],p2)
                    compare('joint95_matrix',raw['joint_logits'][i,op],joint)
                    totals=np.stack((p2.sum(0),t3.sum(0),joint[1]-p2.sum(0)-t3.sum(0)))
                    compare('hierarchy_totals',raw['order_totals'][i,op],totals)
                    compare('hierarchy_stage_parts',raw['order_stageparts'][i,op],np.stack((totals[:,2]-totals[:,3],totals[:,0]-totals[:,2]),1))
                    compare('all_triple_stage_parts',raw['triple_stageparts'][i,op],np.stack((t3[:,2]-t3[:,3],t3[:,0]-t3[:,2]),1))
                    compare('affine_triples_zero',t3[:,3],np.zeros((56,3)))
                    compare('mobius4_inclusion',raw['mobius4'][i,op],np.einsum('st,tmc->smc',inc,actual[[int(np.flatnonzero(masks==s)[0]) for s in range(16)]]))
                    if alpha==.5:
                        expected=np.argsort(-abs(t3[:,0]@weights[i]),kind='stable')[:3]
                        compare('half_triple_ranking',frozen['selected_triples'][i,op],expected)
                        rng=np.random.default_rng(seed+2026101120+100*i+op)
                    for rank,p in enumerate(frozen['selected_triples'][i,op]):
                        tri=set(triples[p]);local_sets=[{c for j,c in enumerate(triples[p]) if s&(1<<j)} for s in range(8)]
                        for ctx in range(2):
                            groups=[g|(set(range(8))-tri) if ctx else g for g in local_sets]
                            xx=inputs_from_members(base,ref,frozen['channels'][i],groups);t=trace(model,xx);output=all_modes(model,xx)
                            compare('original_context_modes',raw['context_logits'][i,op,rank,ctx],output)
                            effect=np.einsum('m,mkc->kc',signs,output);parts=np.stack((effect[2]-effect[3],effect[0]-effect[2]))
                            compare('context_effects',raw['context_effects'][i,op,rank,ctx],effect)
                            compare('context_stage_parts',raw['context_stageparts'][i,op,rank,ctx],parts)
                            unit=units(model,t,weights[i],signs);compare('unit_map_sums',unit.sum((1,2,3)),parts@weights[i])
                            for stage,key in enumerate(('z1','z2')):
                                chosen=frozen['selected_gates'][i,op,rank,ctx,stage];ids=chosen[chosen>=0]
                                score=unit[stage].ravel();z=t[key].flatten(1).cpu().numpy()
                                values=np.zeros(128);values[:len(ids)]=score[ids];zz=np.zeros((8,128));zz[:,:len(ids)]=z[:,ids]
                                compare('coordinate_values',raw['unit_contributions'][i,op,rank,ctx,stage],values)
                                compare('coordinate_preactivations',raw['gate_preactivations'][i,op,rank,ctx,stage],zz)
                                compare('positive_negative_sums',raw['unit_signed_sums'][i,op,rank,ctx,stage],[score[score>0].sum(),score[score<0].sum()])
                                if alpha==.5:
                                    eligible=np.flatnonzero(np.any((z>0)!=(z[:1]>0),0))
                                    compare('eligible_count',frozen['eligible_counts'][i,op,rank,ctx,stage],len(eligible))
                                    expected=eligible[np.argsort(-abs(score[eligible]),kind='stable')[:128]]
                                    if not np.array_equal(ids,expected):
                                        excluded=np.setdiff1d(eligible,ids);gap=float(np.max(abs(score[excluded]))-np.min(abs(score[ids]))) if len(excluded) else 0.
                                        bound=2*float(np.max(abs(frozen['unit_contributions'][i,op,rank,ctx,stage,:len(ids)]-score[ids])))+64*np.finfo(float).eps*max(abs(score).max(),1e-300)
                                        assert gap<=bound,(seed,i,op,rank,ctx,stage,gap,bound)
                                        RANK_NOTES.append(dict(seed=seed,case=i,op=op,rank=rank,context=ctx,stage=stage,gap=gap,bound=bound))
                                    for repeat in range(2):
                                        expected=rng.permutation(eligible)[:128];padded=np.pad(expected,(0,128-len(expected)),constant_values=-1)
                                        compare('random_fixed_lists',frozen['random_gates'][i,op,rank,ctx,stage,repeat],padded)
                                # 符号组按已记录的半强度分数定义；该分数已独立核验，避免零边界换组。
                                sign=1 if float(frozen['context_effects'][i,op,rank,ctx,0]@weights[i])>=0 else -1
                                compare('half_alignment',frozen['alignment_sign'][i,op,rank,ctx],sign)
                                half=frozen['unit_contributions'][i,op,rank,ctx,stage,:len(ids)]
                                for k,member in [('aligned_gates',ids[half*sign>0]),('opposed_gates',ids[half*sign<0])]:
                                    pad=np.pad(member,(0,128-len(member)),constant_values=-1)
                                    compare('frozen_sign_subsets',frozen[k][i,op,rank,ctx,stage],pad)
                                valid=lambda a:a[a>=0]
                                random=[valid(p) for p in frozen['random_gates'][i,op,rank,ctx,stage]]
                                lists=[ids[:32],random[0][:32],random[1][:32],ids,random[0],random[1],valid(frozen['aligned_gates'][i,op,rank,ctx,stage]),valid(frozen['opposed_gates'][i,op,rank,ctx,stage])]
                                predicted=np.array([score[p].sum() for p in lists]);compare('variant_coordinate_sums',raw['variant_contribution_sums'][i,op,rank,ctx,stage],predicted)
                                compare('variant_counts',raw['variant_counts'][i,op,rank,ctx,stage],[len(p) for p in lists])
                                real=original_partial(model,xx,xx[:1],stage,lists);compare('original_partial_holds',raw['partial_logits'][i,op,rank,ctx,stage],real)
                                if stage==1:
                                    value=np.einsum('m,vmk,k->v',signs,real,weights[i]);compare('last_exact_removal',value,effect[0]@weights[i]-predicted)
                                if seed==teacher.SEEDS[0] and i==0 and rank==0:
                                    example=load(folder/'example.npz')['single_logits']
                                    for j,unit_id in enumerate(chosen[:8]):
                                        one=np.array([unit_id]) if unit_id>=0 else np.array([],dtype=np.int64)
                                        real=forward_override(model,xx,xx[:1],stage=stage,ids=one).cpu().numpy()
                                        compare('single_site_actual',example[op,ctx,stage,j],real)
                                        if stage==1:compare('single_last_removal',effect[0]@weights[i]-np.einsum('m,mk,k->',signs,real,weights[i]),score[unit_id] if unit_id>=0 else 0.)
            # 单独从审计重建的阶次公式核对汇总，不把三阶以上都叫三阶。
            joint=np.einsum('nobmk,nk->nobm',raw['joint_logits'],weights)[:,:,1,0]
            orders=np.einsum('nobmk,nk->nobm',raw['order_totals'],weights)[...,0]
            for op in range(2):
                for j,name in enumerate(('pairs','pure_triples','orders4plus')):
                    row=next(r for r in summary['hierarchy'] if r['seed']==seed and r['alpha']==alpha and r['operation']==('restore','transplant')[op] and r['order']==name)
                    compare('hierarchy_ratio',row['rms_ratio'],np.sqrt(np.mean(orders[:,op,j]**2)/np.mean(joint[:,op]**2)))
                    compare('hierarchy_projection',row['signed_projection'],np.mean(orders[:,op,j]*joint[:,op])/np.mean(joint[:,op]**2))
            print(json.dumps(dict(event='frame_audited',seed=seed,alpha=alpha)),flush=True)
    old_counts={}
    for name,value in protocol['preservation'].items():
        assert teacher.sha(BASE/name)==value['sha256'];old=teacher.read(BASE/name)['files'];assert len(old)==value['files']
        for item in old:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        old_counts[name]=len(old)
    teacher.verify_old()
    result=dict(status='PASS',utc=teacher.now(),maximum_errors=MAXIMUM,checked_scalar_counts=COUNTS,
        floating_coordinate_rank_notes=RANK_NOTES,old_delivery_files=old_counts,
        verification='all95 four-mode original forwards,all56 inclusion-exclusion triples,both contexts of frozen3,all8 variants,signed coordinate sums,actual single sites;autograd adjoints',
        limitation='same seen panel;conditional contrasts can contain order4+;no semantic channel alignment')
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':audit()

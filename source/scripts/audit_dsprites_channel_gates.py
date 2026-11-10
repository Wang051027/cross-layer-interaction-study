"""用稠密容斥矩阵核对通道配对系数。用 block3 前向运算和自动微分核对指定 ReLU 位置的干预。"""
from __future__ import annotations

import itertools
import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
from test_dsprites_channel_gates import forward_override
from audit_dsprites_gradient import affine_adjoint

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_channel_gates_20261011'
PRIOR=BASE/'runs/dsprites_shape_donor_20261010'
MAXIMUM={};COUNTS={};RANK_NOTES=[]


def load(path):
    """数值文件只读。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def compare(key,a,b,tolerance=1e-9):
    """先检查形状，记录实际核验误差和数值计数。"""
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(key,a.shape,b.shape)
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.isfinite(error) and error<tolerance,(key,error)
    MAXIMUM[key]=max(MAXIMUM.get(key,0.),error);COUNTS[key]=COUNTS.get(key,0)+a.size


def independent_design():
    """由成员集合枚举，稠密矩阵和容斥矩阵独立于主程序蝶形运算。"""
    members=[]
    for s in range(256):
        group={j for j in range(8) if s&(1<<j)}
        if len(group)<=2 or group.issubset(set(range(4))) or len(group)==8:members.append(group)
    masks=np.array([sum(1<<j for j in group) for group in members]);lookup={int(s):i for i,s in enumerate(masks)}
    pairs=list(itertools.combinations(range(8),2));matrix=np.zeros((28,43));joint=np.zeros((2,43))
    for row,(a,b) in enumerate(pairs):
        for s,sign in [((1<<a)|(1<<b),1),(1<<a,-1),(1<<b,-1),(0,1)]:matrix[row,lookup[s]]=sign
    for row,k in enumerate((4,8)):
        joint[row,lookup[(1<<k)-1]]=1;joint[row,lookup[0]]=k-1
        for j in range(k):joint[row,lookup[1<<j]]=-1
    inclusion=np.zeros((16,16))
    for s in range(16):
        for t in range(16):
            if t&s==t:inclusion[s,t]=(-1)**(s.bit_count()-t.bit_count())
    return masks,pairs,lookup,matrix,joint,inclusion


@torch.no_grad()
def reference_trace(model,x):
    """在原block3模块序列上另写where正向，记录两ReLU前后状态。"""
    b=model.block3;z1=b.bn1(b.conv1(x));h1=torch.where(z1>0,z1,torch.zeros_like(z1))
    z2=b.bn2(b.conv2(h1))+b.shortcut(x);h2=torch.where(z2>0,z2,torch.zeros_like(z2))
    g1=z1[:1]>0;g2=z2[:1]>0
    return dict(x=x,z1=z1,z2=z2,h1=h1,h2=h2,g1=g1,g2=g2,r1=h1-g1*z1,r2=h2-g2*z2)


def coordinate_maps(model,t,weight,matrix):
    """旧伴随算法和自动微分分别得到权重，独立差分矩阵得到28对坐标项。"""
    w=torch.as_tensor(weight,dtype=torch.float64,device=t['x'].device)
    q=(w@model.readout.weight)[:,None,None]/1024
    p=affine_adjoint(model.block3.conv2,model.block3.bn2,q[None]*t['g2'],t['h1'][:1].shape)
    leaf=t['h1'][:1].detach().clone().requires_grad_(True)
    z=model.block3.bn2(model.block3.conv2(leaf))+t['x'][:1]
    logits=model.readout((t['g2']*z).mean((2,3)))
    grad=torch.autograd.grad((logits*w).sum(),leaf)[0]
    compare('coordinate_adjoint_autograd',p.cpu().numpy(),grad.cpu().numpy(),1e-12)
    d1=np.einsum('pm,mchw->pchw',matrix,t['r1'].cpu().numpy())
    d2=np.einsum('pm,mchw->pchw',matrix,t['r2'].cpu().numpy())
    first=d1*p[0].cpu().numpy();second=d2*q.cpu().numpy()
    return np.stack((first,second),1)


@torch.no_grad()
def independent_partial(model,x,base,stage,id_lists):
    """将六份通道名单组成批次，运行 block3.forward。"""
    b=model.block3;inputs=x.repeat(len(id_lists),1,1,1)
    z0=b.bn1(b.conv1(base));g10=z0>0;g20=(b.bn2(b.conv2(F.relu(z0)))+base)>0
    support=torch.zeros((len(id_lists),32768),dtype=torch.bool,device=x.device)
    for i,ids in enumerate(id_lists):support[i,ids]=True
    support=support.repeat_interleave(4,dim=0).reshape_as(inputs)
    def first(v):
        z=b.bn1(b.conv1(v));g=z>0
        if stage==0:g=torch.where(support,g10,g)
        return torch.where(g,z,torch.zeros_like(z))
    def finish(h,residual):
        z=b.bn2(b.conv2(h))+residual;g=z>0
        if stage==1:g=torch.where(support,g20,g)
        return torch.where(g,z,torch.zeros_like(z))
    try:
        b.mid=first;b.finish=finish
        return model.readout(b(inputs).mean((2,3))).reshape(2,3,4,3).cpu().numpy()
    finally:delattr(b,'mid');delattr(b,'finish')


def audit():
    """核验所有配对和所有门控方案，原数字保存后才给PASS。"""
    assert not (RUN/'independent_audit.json').exists(),'Preserve completed audit'
    device=teacher.configure();protocol=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
    for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
    freeze=teacher.read(RUN/'gate_freeze.json')
    assert teacher.sha(RUN/'gate_freeze.json')==(RUN/'gate_freeze.sha256').read_text().strip()
    for name,value in freeze['files'].items():assert teacher.sha(RUN/name)==value,name
    masks,pairs,lookup,matrix,joint,inclusion=independent_design();d=load(RUN/'design.npz')
    compare('design_masks',d['masks'],masks);compare('design_pairs',d['pairs'],pairs);compare('design_joint',d['joint'],joint)
    compare('design_bits',d['bits'],((masks[:,None]>>np.arange(8))&1))
    panel=load(PRIOR/'panel/data.npz');labels=panel['labels']
    weights=np.zeros((72,3));weights[np.arange(72),labels[panel['pairs'][:,1]]]=1;weights[np.arange(72),labels[panel['pairs'][:,0]]]=-1
    summaries=teacher.read(RUN/'summary.json')['rows'];assert len(summaries)==576
    row_lookup={(r['seed'],r['alpha'],r['operation'],r['ranks'],r['stage'],r['budget'],r['method']):r for r in summaries}
    for seed in teacher.SEEDS:
        model=teacher.load_teacher(seed,device)
        for p in model.parameters():p.requires_grad_(False)
        frozen=load(RUN/f'discovery/seed_{seed}/data.npz')
        oldselection=load(PRIOR/f'selection/seed_{seed}/data.npz')
        compare('inherited_eight_channels',frozen['channels'],oldselection['selections'][:,12,:8])
        compare('inherited_four_channels',frozen['channels'][:,:4],oldselection['selections'][:,3,:4])
        assert protocol['utc']<teacher.read(RUN/f'discovery/seed_{seed}/completion.json')['utc']<freeze['utc']
        with torch.no_grad():
            images=torch.as_tensor(panel['images'][:,None],dtype=torch.float64,device=device)
            sources=torch.cat([model.block1(model.stem(x)) for x in images.split(16)])
            native_middle=torch.cat([model.block2(x) for x in sources.split(16)])
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz')
            assert freeze['utc']<teacher.read(folder/'completion.json')['utc']
            for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name
            old=load(PRIOR/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
            compare('same_channels_evaluation',raw['channels'],frozen['channels'])
            if alpha==.5:
                for key in ('logits','pair_logits','joint_logits','mobius4','pair_stageparts','joint_stageparts','higher_than_pair8','stage_flip_counts'):
                    compare('discovery_repeated_'+key,raw[key],frozen[key])
            pair_phi=np.einsum('nopmc,nc->nopm',raw['pair_logits'],weights)
            part_phi=np.einsum('norlbmsc,nc->norlbms',raw['partial_logits'],weights)
            pair_partial=part_phi[...,-1]+part_phi[...,0]-part_phi[...,1]-part_phi[...,2]
            for op in range(2):
                for ranks in ((0,),(0,1,2)):
                    truth=np.array([pair_phi[i,op,frozen['selected_pairs'][i,op,list(ranks)],0] for i in range(72)])
                    denominator=np.mean(truth**2)
                    for stage in range(2):
                        for bi,budget in enumerate((32,128)):
                            for method,name in enumerate(('selected','random_0','random_1')):
                                value=pair_partial[:,op,list(ranks),stage,bi,method]
                                row=row_lookup[seed,alpha,('restore','transplant')[op],'first' if len(ranks)==1 else 'three',stage+1,budget,name]
                                compare('summary_remaining',row['remaining_ratio'],np.sqrt(np.mean(value**2)/denominator))
            for i,(r,donor) in enumerate(panel['pairs']):
                with torch.no_grad():endpoint=model.block2((1-alpha)*sources[r:r+1]+alpha*sources[donor:donor+1])
                for op in range(2):
                    base,ref=(endpoint,native_middle[r:r+1]) if op==0 else (native_middle[r:r+1],endpoint)
                    # 对未选通道赋值而非主程序的选中通道赋值，独立生成43状态。
                    x=ref.expand(43,-1,-1,-1).clone()
                    for state,s in enumerate(masks):
                        included={int(frozen['channels'][i,j]) for j in range(8) if s&(1<<j)}
                        other=np.array(sorted(set(range(32))-included));x[state,other]=base[0,other]
                    trace=reference_trace(model,x);allmodes=[]
                    for mode in range(4):
                        independent=forward_override(model,x,base,stage=0 if mode==1 else 1 if mode==2 else None,both=mode==3)
                        allmodes.append(independent.cpu().numpy())
                    actual=np.stack(allmodes,1);compare('original_forward_all_modes',raw['logits'][i,op],actual)
                    pi=np.einsum('pm,msc->psc',matrix,actual);ji=np.einsum('km,msc->ksc',joint,actual)
                    compare('pair_dense_contrasts',raw['pair_logits'][i,op],pi)
                    compare('joint_dense_contrasts',raw['joint_logits'][i,op],ji)
                    compare('mobius4_inclusion',raw['mobius4'][i,op],np.einsum('st,tmc->smc',inclusion,actual[[lookup[j] for j in range(16)]]))
                    parts=np.stack((pi[:,2]-pi[:,3],pi[:,0]-pi[:,2]),1)
                    compare('pair_stage_parts',raw['pair_stageparts'][i,op],parts)
                    compare('joint_stage_parts',raw['joint_stageparts'][i,op],np.stack((ji[:,2]-ji[:,3],ji[:,0]-ji[:,2]),1))
                    compare('higher_than_pair8',raw['higher_than_pair8'][i,op],ji[1]-pi.sum(0))
                    compare('affine_pair_zero',pi[:,3],np.zeros((28,3)))
                    compare('affine_joint_zero',ji[:,3],np.zeros((2,3)))
                    for budget,index in ((4,3),(8,12)):
                        compare('previous_joint_patch',actual[lookup[(1<<budget)-1],0],old['patched'][i,index,op,-1])
                    unit=coordinate_maps(model,trace,weights[i],matrix)
                    compare('unit_maps_sum',unit.sum((2,3,4)),parts@weights[i])
                    flips=np.stack([((trace[k]>0)!=(trace[k][:1]>0)).flatten(1).sum(-1).cpu().numpy() for k in ('z1','z2')],-1)
                    compare('stage_flip_counts',raw['stage_flip_counts'][i,op],flips)
                    if alpha==.5:
                        selected_pairs=np.argsort(-np.abs(pi[:,0]@weights[i]),kind='stable')[:3]
                        compare('half_pair_ranking',frozen['selected_pairs'][i,op],selected_pairs)
                        rng=np.random.default_rng(seed+2026101100+100*i+op)
                    for rank,pair in enumerate(frozen['selected_pairs'][i,op]):
                        a,b=pairs[pair];indices=[lookup[0],lookup[1<<a],lookup[1<<b],lookup[(1<<a)|(1<<b)]]
                        for stage,key in enumerate(('z1','z2')):
                            z=trace[key][indices].flatten(1).cpu().numpy()
                            padded=frozen['selected_gates'][i,op,rank,stage];ids=padded[padded>=0]
                            score=unit[pair,stage].ravel();values=np.zeros(128);values[:len(ids)]=score[ids]
                            zz=np.zeros((4,128));zz[:,:len(ids)]=z[:,ids]
                            compare('recorded_unit_values',raw['unit_contributions'][i,op,rank,stage],values)
                            compare('recorded_gate_preactivations',raw['gate_preactivations'][i,op,rank,stage],zz)
                            if alpha==.5:
                                compare('discovery_unit_values',frozen['selection_contributions'][i,op,rank,stage],values)
                                compare('discovery_gate_preactivations',frozen['selection_preactivations'][i,op,rank,stage],zz)
                                eligible=np.flatnonzero(np.any((z>0)!=(trace[key][:1].flatten(1).cpu().numpy()>0),axis=0))
                                compare('eligible_counts',frozen['eligible_counts'][i,op,rank,stage],len(eligible))
                                expected=eligible[np.argsort(-np.abs(score[eligible]),kind='stable')[:128]]
                                if not np.array_equal(ids,expected):
                                    excluded=np.setdiff1d(eligible,ids)
                                    gap=float(np.max(np.abs(score[excluded]))-np.min(np.abs(score[ids]))) if len(excluded) else 0.
                                    bound=2*float(np.max(np.abs(frozen['selection_contributions'][i,op,rank,stage,:len(ids)]-score[ids])))+64*np.finfo(float).eps*max(np.abs(score).max(),1e-300)
                                    assert gap<=bound,(seed,i,op,rank,stage,gap,bound)
                                    RANK_NOTES.append(dict(seed=seed,pair_case=i,operation=op,rank=rank,stage=stage,gap=gap,bound=bound))
                                for repeat in range(2):
                                    random=rng.permutation(eligible)[:128];pad=np.full(128,-1);pad[:len(random)]=random
                                    compare('seeded_random_units',frozen['random_gates'][i,op,rank,stage,repeat],pad)
                            candidates=[padded,*frozen['random_gates'][i,op,rank,stage]]
                            lists=[p[p>=0][:budget] for budget in (32,128) for p in candidates]
                            independent=independent_partial(model,x[indices],base,stage,lists)
                            compare('original_partial_holds',raw['partial_logits'][i,op,rank,stage],independent)
                            if stage==1:
                                for bi,budget in enumerate((32,128)):
                                    held=independent[bi,0];effect=(held[3]-held[1]-held[2]+held[0])@weights[i]
                                    compare('last_gate_exact_removal',effect,pi[pair,0]@weights[i]-score[ids[:budget]].sum())
            print(json.dumps(dict(event='frame_audited',seed=seed,alpha=alpha)),flush=True)
    old_counts={}
    for name,value in protocol['preservation'].items():
        assert teacher.sha(BASE/name)==value['sha256']
        records=teacher.read(BASE/name)['files'];assert len(records)==value['files']
        for item in records:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path']
        old_counts[name]=len(records)
    teacher.verify_old()
    result=dict(status='PASS',utc=teacher.now(),teachers=6,images=36,paired_cases=72,
        maximum_errors=MAXIMUM,checked_scalar_counts=COUNTS,floating_unit_rank_notes=RANK_NOTES,
        old_delivery_files=old_counts,verification='original block3 forward overrides, all43 states/four modes/all partial interventions; independent dense inclusion-exclusion; autograd coordinate adjoints',
        timing='half discovery sealed before full evaluation; no full-strength reselection')
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':audit()

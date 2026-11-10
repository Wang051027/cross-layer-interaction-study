"""汇总八通道内的纯三阶、条件三阶和四阶以上余项，比较两种强度的开关与符号。"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_triple_gates_20261011'
SHAPE=BASE/'runs/dsprites_shape_donor_20261010'
REPORT=BASE/'reports/dSprites_三阶通道与ReLU抵消机制_2026-10-11.md'
OPS=('restore','transplant');OP_CN=('恢复','移植');CTX=('empty','complement')
CTX_CN=('其余通道未替换','其余5通道已替换')
SIGNS=np.array([-1.,1.,1.,-1.,1.,-1.,-1.,1.])
VARIANTS=('selected32','random0_32','random1_32','selected128','random0_128','random1_128','aligned_subset','opposed_subset')


def load(path):
    """禁止pickle，及时关闭压缩文件的句柄。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def rms(value):
    """幅度先在一个教师内部计算；跨教师再取均值。"""
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def write_csv(name,rows):
    """保留案例和条件，UTF8 BOM方便中文Excel阅读。"""
    assert rows
    folder=RUN/'tables';folder.mkdir(exist_ok=True)
    with (folder/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def cancellation(total,absolute_parts):
    """加权抵消指数：1−Σ|净和|/Σ各项绝对值；零分母定义0。"""
    den=float(np.sum(absolute_parts));num=float(np.sum(np.abs(total)))
    assert num<=den+1e-8,(num,den)
    return 0. if den<1e-20 else float(np.clip(1-num/den,0,1))


def pattern(values):
    """按空/A/B/AB/C/AC/BC/ABC八状态记录开关，0视为关闭。"""
    return ''.join('1' if z>0 else '0' for z in values)


def choose_rows(rows,**conditions):
    return [r for r in rows if all(r[k]==v for k,v in conditions.items())]


def validate_second_response():
    """重放预定例子，独立记录实际第二层余项，核对推得的下游响应。

    同一第一教师/案例0/半强度首位、两背景两操作两强度，复用8组
    固定及8个预定单点，不新增坐标或重新挑选三元组。使用原block3
    forward记录第二层预激活，检查仅针对报告新增的代数推导。
    """
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    path=RUN/'net_response_check.json';assert not path.exists()
    import torch
    from run_dsprites_triple_gates import middle_states,verify_seal
    verify_seal();device=teacher.configure();model=teacher.load_teacher(teacher.SEEDS[0],device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    panel=load(SHAPE/'panel/data.npz');frozen=load(RUN/f'discovery/seed_{teacher.SEEDS[0]}/data.npz');d=load(RUN/'design.npz')
    recipient,donor=map(int,panel['pairs'][0]);weight=np.zeros(3)
    weight[panel['labels'][donor]]=1;weight[panel['labels'][recipient]]=-1
    q=(torch.as_tensor(weight,device=device)@model.readout.weight)[:,None,None]/1024
    sign=torch.as_tensor(SIGNS,device=device)[None,:,None,None,None]
    maximum={};conditions=0;response_rows=[];response_coordinates=[];maps_all=[];natural_codes=[];held_codes=[]
    def check(name,a,b):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
        error=float(abs(a-b).max());assert np.isfinite(error) and error<1e-9,(name,error)
        maximum[name]=max(maximum.get(name,0.),error)
    @torch.no_grad()
    def captured(x,lists):
        """原forward中只改第一层开关，finish记录自然重算的z2。"""
        block=model.block3;base_gate=block.bn1(block.conv1(x[:1]))>0
        support=torch.zeros((len(lists),32768),dtype=torch.bool,device=device)
        for j,ids in enumerate(lists):support[j,ids]=True
        support=support.repeat_interleave(8,0).reshape(-1,32,32,32);record={}
        def mid(value):
            z=block.bn1(block.conv1(value));g=torch.where(support,base_gate,z>0)
            return torch.where(g,z,torch.zeros_like(z))
        def finish(h,residual):
            z=block.bn2(block.conv2(h))+residual;record['z2']=z
            return torch.where(z>0,z,torch.zeros_like(z))
        try:
            block.mid=mid;block.finish=finish
            output=model.readout(block(x.repeat(len(lists),1,1,1)).mean((2,3)))
            return output.reshape(len(lists),8,3).cpu().numpy(),record['z2'].reshape(len(lists),8,32,32,32)
        finally:delattr(block,'mid');delattr(block,'finish')
    for alpha in (1.,.5):
        raw=load(RUN/f'evaluation/seed_{teacher.SEEDS[0]}/alpha_{alpha}/data.npz')
        example=load(RUN/f'evaluation/seed_{teacher.SEEDS[0]}/alpha_{alpha}/example.npz')['single_logits']
        with torch.no_grad():baseline,endpoint=middle_states(model,panel,device,alpha)
        for op in range(2):
            base,reference=(endpoint[:1],baseline[recipient:recipient+1]) if op==0 else (baseline[recipient:recipient+1],endpoint[:1])
            relative=d['triples'][frozen['selected_triples'][0,op,0]]
            for ctx in range(2):
                # 成员集合构造patch，与主程序的bits乘法实现分开。
                x=reference.expand(8,-1,-1,-1).clone()
                for state in range(8):
                    included=set(range(8))-set(map(int,relative)) if ctx else set()
                    included|={int(c) for j,c in enumerate(relative) if state&(1<<j)}
                    physical={int(frozen['channels'][0,c]) for c in included}
                    other=np.array(sorted(set(range(32))-physical));x[state,other]=base[0,other]
                none=np.array([],dtype=np.int64);natural,z0=captured(x,[none])
                check('natural_original_output',natural[0],raw['context_logits'][0,op,0,ctx,:,0])
                g2=z0[0,:1]>0
                def second_part(z):
                    residue=torch.where(z>0,z,torch.zeros_like(z))-g2*z
                    return (residue*sign*q[None,None]).sum((1,2,3,4)).cpu().numpy()
                old_second=second_part(z0)[0]
                check('natural_second_part',np.asarray(old_second),np.asarray(raw['context_stageparts'][0,op,0,ctx,1]@weight))
                valid=lambda v:v[v>=0]
                chosen=valid(frozen['selected_gates'][0,op,0,ctx,0]);random=[valid(v) for v in frozen['random_gates'][0,op,0,ctx,0]]
                lists=[chosen[:32],random[0][:32],random[1][:32],chosen,random[0],random[1],valid(frozen['aligned_gates'][0,op,0,ctx,0]),valid(frozen['opposed_gates'][0,op,0,ctx,0])]
                lists +=[np.array([v],dtype=np.int64) if v>=0 else none for v in frozen['selected_gates'][0,op,0,ctx,0,:8]]
                output,z=captured(x,lists)
                expected=np.concatenate((raw['partial_logits'][0,op,0,ctx,0],example[op,ctx,0]))
                check('saved_original_holds',output,expected)
                check('unchanged_baseline_z2',z[:,0].cpu().numpy(),z0[:,0].expand(16,-1,-1,-1).cpu().numpy())
                natural_theta=float(np.einsum('t,tk,k->',SIGNS,natural[0],weight))
                held=np.einsum('t,vtk,k->v',SIGNS,output,weight)
                direct=np.concatenate((raw['variant_contribution_sums'][0,op,0,ctx,0],raw['unit_contributions'][0,op,0,ctx,0,:8]))
                predicted=natural_theta-held-direct;actual=old_second-second_part(z)
                check('inferred_vs_recorded_second_response',predicted,actual)
                old_residue=torch.where(z0>0,z0,torch.zeros_like(z0))-g2*z0
                new_residue=torch.where(z>0,z,torch.zeros_like(z))-g2*z
                maps=(((old_residue-new_residue)*sign).sum(1)*q).cpu().numpy()
                check('response_coordinate_sum',maps.sum((1,2,3)),actual)
                powers=(1<<np.arange(8)).astype(np.uint16)
                nat=np.sum((z0[0]>0).cpu().numpy()*powers[:,None,None,None],axis=0).astype(np.uint8)
                after=np.sum((z>0).cpu().numpy()*powers[None,:,None,None,None],axis=1).astype(np.uint8)
                popcount=np.array([int(v).bit_count() for v in range(256)])
                for j,name in enumerate((*VARIANTS,*[f'single_{v+1}' for v in range(8)])):
                    row=dict(map_index=conditions+j,seed=teacher.SEEDS[0],case=0,alpha=alpha,operation=OPS[op],context=CTX[ctx],intervention=name,
                        source_gate_id=int(frozen['selected_gates'][0,op,0,ctx,0,j-8]) if j>=8 else -1,
                        natural_theta=natural_theta,held_theta=float(held[j]),first_removed=float(direct[j]),second_response=float(actual[j]),
                        second_flip_sites=int(np.sum(after[j]!=nat)),second_flip_events=int(popcount[after[j]^nat].sum()))
                    response_rows.append(row);score=maps[j].ravel()
                    # 全强度后的描述性响应坐标排名，没有按这些点新增干预。
                    for label,ids in [('positive',np.flatnonzero(score>0)),('negative',np.flatnonzero(score<0))]:
                        ids=ids[np.argsort(-abs(score[ids]),kind='stable')[:4]]
                        for rank,v in enumerate(ids):
                            channel,y,xx=map(int,np.unravel_index(int(v),(32,32,32)))
                            before_code=int(nat.ravel()[v]);after_code=int(after[j].ravel()[v])
                            response_coordinates.append(dict(map_index=conditions+j,alpha=alpha,operation=OPS[op],context=CTX[ctx],intervention=name,
                                response_sign=label,response_rank=rank+1,second_gate_channel=channel,second_gate_y=y,second_gate_x=xx,
                                natural_pattern=''.join('1' if before_code&(1<<s) else '0' for s in range(8)),
                                held_pattern=''.join('1' if after_code&(1<<s) else '0' for s in range(8)),
                                signed_response=float(score[v]),bit_flip_events=(before_code^after_code).bit_count()))
                maps_all.extend(maps);natural_codes.extend([nat]*len(lists));held_codes.extend(after);conditions+=len(lists)
    write_csv('net_response_cases.csv',response_rows);write_csv('net_response_coordinates.csv',response_coordinates)
    np.savez_compressed(RUN/'net_response_maps.npz',signed_response=np.asarray(maps_all),natural_pattern_code=np.asarray(natural_codes),held_pattern_code=np.asarray(held_codes))
    result=dict(status='PASS',utc=teacher.now(),conditions=conditions,maximum_errors=maximum,
        scope='first teacher case0 half-top1,all2contexts/2operations/2strengths;8groups+8predefined first-stage single sites',
        method='original block3.forward with where gate override;capture actual second preactivations;baseline g2 unchanged;readout-weighted remainder difference',
        selection='reuse frozen lists;response coordinates ranked descriptively after full outputs;no new intervention candidates',
        coordinate_rows=len(response_coordinates),case_rows=len(response_rows),maps_shape=[128,32,32,32],
        files={name:teacher.sha(RUN/name) for name in ('net_response_maps.npz','tables/net_response_cases.csv','tables/net_response_coordinates.csv')})
    assert conditions==128;teacher.dump(path,result)
    return result


def analyze():
    """先要求独立数值核验PASS，随后只读取全量数据形成描述性证据。"""
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert teacher.read(RUN/'net_response_check.json')['status']=='PASS'
    assert not (RUN/'analysis.json').exists(),'Preserve completed analysis'
    panel=load(SHAPE/'panel/data.npz');d=load(RUN/'design.npz');summary=teacher.read(RUN/'summary.json')
    weights=np.zeros((72,3));weights[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1
    weights[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    primary=[];signed_groups=[]
    for op in OPS:
        for stage in (1,2):
            rows=choose_rows(summary['rows'],alpha=1.,operation=op,ranks='first',context='empty',stage=stage)
            selected=np.array([next(r['remaining_ratio'] for r in rows if r['seed']==s and r['variant']=='selected128') for s in teacher.SEEDS])
            random=np.array([np.mean([r['remaining_ratio'] for r in rows if r['seed']==s and r['variant'] in ('random0_128','random1_128')]) for s in teacher.SEEDS])
            primary.append(dict(operation=op,stage=stage,selected_mean=float(selected.mean()),random_mean=float(random.mean()),
                gain_pp=float(100*(random-selected).mean()),better_than_random_teachers=int((selected<random).sum()),
                below_natural_teachers=int((selected<1).sum()),
                per_teacher=[dict(seed=s,selected=float(selected[j]),random_mean=float(random[j])) for j,s in enumerate(teacher.SEEDS)]))
            for name in ('aligned_subset','opposed_subset'):
                part=choose_rows(rows,variant=name)
                signed_groups.append(dict(operation=op,stage=stage,variant=name,
                    signed_removal=float(np.mean([1-r['signed_remaining'] for r in part])),
                    remaining_ratio=float(np.mean([r['remaining_ratio'] for r in part])),
                    mean_gate_count=float(np.mean([r['mean_gate_count'] for r in part])),
                    per_teacher=[dict(seed=r['seed'],signed_removal=1-r['signed_remaining'],remaining_ratio=r['remaining_ratio'],mean_gate_count=r['mean_gate_count']) for r in part]))
    orders=[];cancel_rows=[];layers=[];contexts=[];stability=[];gate_rows=[]
    physical_rows=[];top_physical=[];triple_rows=[];chosen_rows=[];coordinates=[];examples=[];single_sites=[]
    first_net_rows=[];first_net_aggregates=[]
    for seed in teacher.SEEDS:
        half=load(RUN/f'discovery/seed_{seed}/data.npz')
        full=load(RUN/f'evaluation/seed_{seed}/alpha_1.0/data.npz')
        ht=np.einsum('notmk,nk->notm',half['triple_logits'],weights)
        ft=np.einsum('notmk,nk->notm',full['triple_logits'],weights)
        fp=np.einsum('notsk,nk->nots',full['triple_stageparts'],weights)
        hc=np.einsum('norcmk,nk->norcm',half['context_effects'],weights)
        fc=np.einsum('norcmk,nk->norcm',full['context_effects'],weights)
        cp=np.einsum('norcsk,nk->norcs',full['context_stageparts'],weights)
        partial=np.einsum('norcsvtk,nk->norcsvt',full['partial_logits'],weights)
        held=np.einsum('norcsvt,t->norcsv',partial,SIGNS)
        # 对全部 32,768 个坐标的带符号贡献求和以重建三阶值。
        signed=full['unit_signed_sums'];pos=signed[...,0].sum(-1);neg=signed[...,1].sum(-1)
        assert np.max(abs(pos+neg-fc[...,0]))<1e-9
        example_single={alpha:load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/example.npz')['single_logits'] for alpha in (1.,.5)} if seed==teacher.SEEDS[0] else {}
        for op in range(2):
            group=defaultdict(list);top1=defaultdict(int);top3=defaultdict(int)
            correlations=[];overlap=[];sign_retention=[]
            for i in range(72):
                now=np.argsort(-abs(ft[i,op,:,0]),kind='stable')[:3]
                correlations.append(float(spearmanr(abs(ht[i,op,:,0]),abs(ft[i,op,:,0])).statistic))
                overlap.append(len(set(now)&set(half['selected_triples'][i,op]))/3)
                selected=half['selected_triples'][i,op]
                sign_retention.extend((ht[i,op,selected,0]*ft[i,op,selected,0]>0).tolist())
                for t,relative in enumerate(d['triples']):
                    ordered=tuple(int(half['channels'][i,c]) for c in relative);key=tuple(sorted(ordered))
                    h=float(ht[i,op,t,0]);v=float(ft[i,op,t,0]);group[key].append((h,v))
                    top1[key]+=int(t==selected[0]);top3[key]+=int(t in selected)
                    triple_rows.append(dict(seed=seed,case=i,operation=OPS[op],triple_index=t,
                        channel_a=ordered[0],channel_b=ordered[1],channel_c=ordered[2],
                        half_theta=h,full_theta=v,full_first_part=float(fp[i,op,t,0]),full_second_part=float(fp[i,op,t,1]),
                        half_top3_rank=int(np.flatnonzero(selected==t)[0]+1) if t in selected else 0))
                for rank,t in enumerate(selected):
                    ordered=tuple(int(half['channels'][i,c]) for c in d['triples'][t])
                    for ctx in range(2):
                        value=float(fc[i,op,rank,ctx,0]);parts=cp[i,op,rank,ctx]
                        z=full['context_logits'][i,op,rank,ctx,:,0]@weights[i]
                        off=float(z[3]-z[1]-z[2]+z[0]);on=float(z[7]-z[5]-z[6]+z[4])
                        assert abs(on-off-value)<1e-9
                        cr=dict(seed=seed,case=i,operation=OPS[op],half_rank=rank+1,context=CTX[ctx],
                            channel_a=ordered[0],channel_b=ordered[1],channel_c=ordered[2],half_theta=float(hc[i,op,rank,ctx,0]),full_theta=value,
                            pair_ab_without_c=off,pair_ab_with_c=on,first_part=float(parts[0]),second_part=float(parts[1]),
                            first_positive=float(signed[i,op,rank,ctx,0,0]),first_negative=float(signed[i,op,rank,ctx,0,1]),
                            second_positive=float(signed[i,op,rank,ctx,1,0]),second_negative=float(signed[i,op,rank,ctx,1,1]))
                        chosen_rows.append(cr)
                        # 基准状态固定后不变，p也不变；净变化扣除第一层项即第二层余项的响应。
                        # 这是从已核验输出推得的读出加权差值，没有测量额外的第二层坐标名单。
                        for variant,name in enumerate(VARIANTS):
                            direct=float(full['variant_contribution_sums'][i,op,rank,ctx,0,variant])
                            net=value-float(held[i,op,rank,ctx,0,variant])
                            first_net_rows.append(dict(seed=seed,case=i,operation=OPS[op],half_rank=rank+1,context=CTX[ctx],variant=name,
                                natural_theta=value,held_theta=float(held[i,op,rank,ctx,0,variant]),net_removed=net,
                                first_removed=direct,inferred_second_response=net-direct))
                        if seed==teacher.SEEDS[0] and i==0:
                            examples.append(dict(**cr,held=[list(map(float,held[i,op,rank,ctx,s])) for s in range(2)],
                                gate_counts=full['variant_counts'][i,op,rank,ctx].tolist(),local_values=z.tolist()))
                        for stage in range(2):
                            for j,unit_id in enumerate(half['selected_gates'][i,op,rank,ctx,stage,:8]):
                                if unit_id<0:continue
                                channel,y,x=map(int,np.unravel_index(int(unit_id),(32,32,32)))
                                hp=half['gate_preactivations'][i,op,rank,ctx,stage,:,j]
                                zp=full['gate_preactivations'][i,op,rank,ctx,stage,:,j]
                                row=dict(seed=seed,case=i,operation=OPS[op],half_rank=rank+1,context=CTX[ctx],stage=stage+1,coordinate_rank=j+1,
                                    channel_a=ordered[0],channel_b=ordered[1],channel_c=ordered[2],gate_channel=channel,gate_y=y,gate_x=x,
                                    half_pattern=pattern(hp),full_pattern=pattern(zp),
                                    half_contribution=float(half['unit_contributions'][i,op,rank,ctx,stage,j]),
                                    full_contribution=float(full['unit_contributions'][i,op,rank,ctx,stage,j]))
                                coordinates.append(row)
                                if seed==teacher.SEEDS[0] and i==0 and rank==0:
                                    actual={a:float(np.einsum('t,tk,k->',SIGNS,v[op,ctx,stage,j],weights[i])) for a,v in example_single.items()}
                                    one=dict(**row,half_theta=float(hc[i,op,rank,ctx,0]),full_theta=value,
                                        half_held=actual[.5],full_held=actual[1.],half_removed=float(hc[i,op,rank,ctx,0]-actual[.5]),full_removed=value-actual[1.])
                                    one['inferred_second_response']=one['full_removed']-one['full_contribution'] if stage==0 else 0.
                                    if stage==1:assert abs(one['full_removed']-one['full_contribution'])<1e-9
                                    single_sites.append(one)
            rows=[]
            for key,observations in group.items():
                values=np.array(observations)
                rows.append(dict(seed=seed,operation=OPS[op],channel_1=key[0],channel_2=key[1],channel_3=key[2],
                    observed_cases=len(values),half_energy=float(np.sum(values[:,0]**2)),half_rms=rms(values[:,0]),full_rms=rms(values[:,1]),
                    half_top1_cases=top1[key],half_top3_cases=top3[key]))
            rows.sort(key=lambda r:(-r['half_energy'],r['channel_1'],r['channel_2'],r['channel_3']))
            physical_rows.extend(rows);top_physical.extend([dict(rank=j+1,**r) for j,r in enumerate(rows[:3])])
            stability.append(dict(seed=seed,operation=OPS[op],abs_rank_spearman=float(np.mean(correlations)),
                top3_overlap=float(np.mean(overlap)),selected_sign_retention=float(np.mean(sign_retention))))
            for ctx in range(2):
                natural=fc[:,op,0,ctx,0];parts=cp[:,op,0,ctx]
                layer_abs=np.abs(parts).sum(-1);unit_abs=pos[:,op,0,ctx]-neg[:,op,0,ctx]
                cancel_rows.append(dict(seed=seed,operation=OPS[op],context=CTX[ctx],
                    layer_cancellation=cancellation(natural,layer_abs),unit_cancellation=cancellation(natural,unit_abs),
                    positive_sum=float(pos[:,op,0,ctx].sum()),negative_sum=float(neg[:,op,0,ctx].sum()),
                    absolute_net_sum=float(abs(natural).sum()),absolute_unit_sum=float(unit_abs.sum())))
                den=float(np.mean(natural**2))
                for variant,name in enumerate(VARIANTS):
                    direct=full['variant_contribution_sums'][:,op,0,ctx,0,variant]
                    net=natural-held[:,op,0,ctx,0,variant];downstream=net-direct
                    # 净投影与独立审计过的汇总交叉核对；两个项分别可增益或抵消。
                    reference=next(r for r in summary['rows'] if r['seed']==seed and r['alpha']==1. and r['operation']==OPS[op] and r['ranks']=='first' and r['context']==CTX[ctx] and r['stage']==1 and r['variant']==name)
                    projection=float(np.mean(natural*net)/den);assert abs(projection-(1-reference['signed_remaining']))<1e-10
                    first_net_aggregates.append(dict(seed=seed,operation=OPS[op],context=CTX[ctx],variant=name,
                        net_projection=projection,first_projection=float(np.mean(natural*direct)/den),
                        inferred_second_projection=float(np.mean(natural*downstream)/den),
                        net_rms_ratio=rms(net)/rms(natural),first_rms_ratio=rms(direct)/rms(natural),inferred_second_rms_ratio=rms(downstream)/rms(natural)))
                layers.append(dict(seed=seed,operation=OPS[op],context=CTX[ctx],natural_rms=rms(natural),
                    hold_first_ratio=rms(fc[:,op,0,ctx,1])/rms(natural),hold_second_ratio=rms(fc[:,op,0,ctx,2])/rms(natural),
                    hold_both_ratio=rms(fc[:,op,0,ctx,3])/rms(natural),
                    first_signed_projection=float(np.mean(natural*parts[:,0])/den),second_signed_projection=float(np.mean(natural*parts[:,1])/den)))
                for stage in range(2):
                    ids=half['selected_gates'][:,op,0,ctx,stage];valid=ids>=0
                    z=full['gate_preactivations'][:,op,0,ctx,stage]
                    switched=np.any((z>0)!=(z[:,0:1]>0),axis=1)&valid
                    a=half['unit_contributions'][:,op,0,ctx,stage];b=full['unit_contributions'][:,op,0,ctx,stage]
                    nonzero=valid&(a!=0)
                    gate_rows.append(dict(seed=seed,operation=OPS[op],context=CTX[ctx],stage=stage+1,
                        actual_budget=float(np.mean(valid.sum(1))),full_switch_retention=float(switched.sum()/valid.sum()),
                        full_contribution_sign_retention=float(((a*b>0)&nonzero).sum()/nonzero.sum()) if nonzero.any() else 0.,
                        aligned_expected_direction_fraction=float(np.mean((natural*(natural-held[:,op,0,ctx,stage,6]))>1e-12)),
                        opposed_expected_direction_fraction=float(np.mean((natural*(natural-held[:,op,0,ctx,stage,7]))< -1e-12))))
            empty=fc[:,op,0,0,0];complement=fc[:,op,0,1,0]
            contexts.append(dict(seed=seed,operation=OPS[op],complement_to_empty_rms=rms(complement)/rms(empty),
                difference_to_empty_rms=rms(complement-empty)/rms(empty),same_sign_fraction=float(np.mean(empty*complement>0))))
            for alpha in (1.,.5):
                raw=full if alpha==1. else load(RUN/f'evaluation/seed_{seed}/alpha_{alpha}/data.npz')
                total=np.einsum('nobmk,nk->nobm',raw['joint_logits'],weights)[:,op,1,0]
                parts=np.einsum('nobmk,nk->nobm',raw['order_totals'],weights)[:,op,:,0]
                thirds=np.einsum('notmk,nk->notm',raw['triple_logits'],weights)[:,op,:,0]
                assert np.max(abs(parts.sum(1)-total))<1e-9
                residual=parts[:,1]+parts[:,2]
                orders.append(dict(seed=seed,alpha=alpha,operation=OPS[op],
                    hierarchy_cancellation=cancellation(total,abs(parts).sum(1)),
                    third_and_higher_cancellation=cancellation(residual,abs(parts[:,1:]).sum(1)),
                    within_pure_triples_cancellation=cancellation(parts[:,1],abs(thirds).sum(1)),
                    remaining4_to_old3plus_rms=rms(parts[:,2])/rms(residual),pure3_to_old3plus_rms=rms(parts[:,1])/rms(residual),
                    triple_vs_orders4_cosine=float(np.sum(parts[:,1]*parts[:,2])/np.sqrt(np.sum(parts[:,1]**2)*np.sum(parts[:,2]**2)))))
    assert len(triple_rows)==48384 and len(chosen_rows)==5184 and len(first_net_rows)==41472
    write_csv('summary.csv',summary['rows']);write_csv('hierarchy.csv',summary['hierarchy'])
    for name,rows in [('all56_triples.csv',triple_rows),('physical_triples.csv',physical_rows),('chosen_triples.csv',chosen_rows),
        ('gate_coordinates_top8.csv',coordinates),('single_sites.csv',single_sites),('cancellation.csv',cancel_rows),
        ('order_cancellation.csv',orders),('layer_holds.csv',layers),('context_dependence.csv',contexts),('ranking_stability.csv',stability),('gate_stability.csv',gate_rows),
        ('first_stage_net.csv',first_net_rows),('first_stage_net_summary.csv',first_net_aggregates)]:write_csv(name,rows)
    result=dict(status='PASS',utc=teacher.now(),primary=primary,signed_groups=signed_groups,
        hierarchy=summary['hierarchy'],order_cancellation=orders,cancellation=cancel_rows,layer_holds=layers,
        context_dependence=contexts,ranking_stability=stability,gate_stability=gate_rows,
        first_stage_net_summary=first_net_aggregates,top_physical_triples=top_physical,example=examples,example_single_sites=single_sites,
        table_rows=dict(all56_triples=len(triple_rows),physical_triples=len(physical_rows),chosen_triples=len(chosen_rows),gate_coordinates_top8=len(coordinates),single_sites=len(single_sites),summary=len(summary['rows']),hierarchy=len(summary['hierarchy']),first_stage_net=len(first_net_rows),first_stage_net_summary=len(first_net_aggregates)))
    teacher.dump(RUN/'analysis.json',result)
    return result


def figures(a):
    """论文图采用标准Matplotlib，输出独立PNG/SVG便于复用。"""
    cache=RUN/'.plot_cache';cache.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    folder=BASE/'reports/figures';folder.mkdir(exist_ok=True)
    def save(fig,suffix):
        for ext in ('png','svg'):fig.savefig(folder/f'dsprites_triple_gates_20261011{suffix}.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row in a['primary']:
        ax=axes[OPS.index(row['operation']),row['stage']-1];v=row['per_teacher'];x=np.arange(6)
        ax.bar(x-.17,[r['selected']*100 for r in v],.32,label='Frozen top 128',color='#2563a6')
        ax.bar(x+.17,[r['random_mean']*100 for r in v],.32,label='Two random lists (mean)',color='#a9b8c6')
        ax.axhline(100,color='#a94b48',ls='--',lw=1);ax.set_xticks(x,[str(r['seed'])[-3:] for r in v])
        ax.set_title(f"{row['operation'].capitalize()} / ReLU {row['stage']}")
        ax.set_ylabel('Remaining triple RMS (%)');ax.set_xlabel('Frozen teacher seed suffix')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Primary: full strength, empty background, half-selected first triple')
    save(fig,'')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');colors=('#597cab','#dda245','#a65866')
    for op in range(2):
        for j,key in enumerate(('signed_projection','rms_ratio')):
            ax=axes[op,j];x=np.arange(6)
            for part,name in enumerate(('pairs','pure_triples','orders4plus')):
                values=[next(r[key] for r in a['hierarchy'] if r['seed']==s and r['alpha']==1. and r['operation']==OPS[op] and r['order']==name) for s in teacher.SEEDS]
                ax.bar(x+(part-1)*.24,values,.22,color=colors[part],label=('Sum of pairs','Sum of pure triples','Orders 4-8 remainder')[part])
            ax.axhline(0,color='black',lw=.7);ax.axhline(1,color='gray',lw=.7,ls='--')
            ax.set_xticks(x,[str(s)[-3:] for s in teacher.SEEDS]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.set_title(OPS[op].capitalize());ax.set_xlabel('Frozen teacher seed suffix')
            ax.set_ylabel('Signed projection onto joint effect' if j==0 else 'RMS / joint effect RMS')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Eight-channel joint effect = pair sum + pure triple sum + orders 4-8')
    save(fig,'_hierarchy')
    raw=load(RUN/f'evaluation/seed_{teacher.SEEDS[0]}/alpha_1.0/data.npz')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for op in range(2):
        ex=next(r for r in a['example'] if r['operation']==OPS[op] and r['half_rank']==1 and r['context']=='empty')
        for stage in range(2):
            ax=axes[stage,op];ids=raw['selected_gates'][0,op,0,0,stage,:8];valid=ids>=0
            z=raw['gate_preactivations'][0,op,0,0,stage,:,:8][:,valid].T;limit=max(float(abs(z).max()),1e-12)
            im=ax.imshow(z,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            labels=[str(tuple(map(int,np.unravel_index(int(v),(32,32,32))))) for v in ids[valid]]
            ax.set_yticks(np.arange(len(labels)),labels);ax.set_xticks(np.arange(8),['empty','A','B','AB','C','AC','BC','ABC'],fontsize=8)
            ax.set_ylabel('Gate (channel, y, x)')
            ax.set_title(f"{OPS[op].capitalize()} / ReLU {stage+1}\nA={ex['channel_a']}, B={ex['channel_b']}, C={ex['channel_c']}; triple={ex['full_theta']:+.5f}",fontsize=10)
            for y in range(len(labels)):
                for x in range(8):ax.text(x,y,'1' if z[y,x]>0 else '0',ha='center',va='center',color='white' if abs(z[y,x])>limit*.5 else 'black',fontsize=8)
            fig.colorbar(im,ax=ax,label='Preactivation',shrink=.85)
    fig.suptitle('Fixed case: teacher 291, square to ellipse; first 8 half-selected sites, full strength, empty background')
    save(fig,'_example')


def table(headers,rows):
    """中文报告的小表保持列数可读，详细条件留在CSV。"""
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def mean(rows,key,**conditions):
    selected=choose_rows(rows,**conditions);assert selected
    return float(np.mean([r[key] for r in selected]))


def report(a):
    """具体机制、实际反事实与证据边界一起写，避免只汇报均值。"""
    audit=teacher.read(RUN/'independent_audit.json');net_check=teacher.read(RUN/'net_response_check.json')
    with (RUN/'tables/net_response_cases.csv').open(encoding='utf-8-sig',newline='') as f:net_cases=list(csv.DictReader(f))
    with (RUN/'tables/net_response_coordinates.csv').open(encoding='utf-8-sig',newline='') as f:net_coordinates=list(csv.DictReader(f))
    net_example=[r for r in net_cases if float(r['alpha'])==1. and r['operation']=='transplant' and r['context']=='empty' and r['intervention'] in ('selected128','aligned_subset','opposed_subset','single_1','single_3','single_8')]
    net_labels={'selected128':'全部选中128','aligned_subset':'半强度同向','opposed_subset':'半强度反向','single_1':'单点1','single_3':'单点3','single_8':'单点8'}
    primary=a['primary'];full_orders=choose_rows(a['order_cancellation'],alpha=1.)
    example=next(r for r in a['example'] if r['operation']=='restore' and r['half_rank']==1 and r['context']=='empty')
    triplet=tuple(example[k] for k in ('channel_a','channel_b','channel_c'))
    failures=[(r['operation'],r['stage'],v) for r in primary for v in r['per_teacher'] if v['selected']>=v['random_mean']]
    failure_text='主比较中未优于随机均值的条件也保留：'+ '；'.join(
        f'{OP_CN[OPS.index(op)]}、ReLU{stage}、教师{v["seed"]}，选中剩余{100*v["selected"]:.2f}%而随机均值{100*v["random_mean"]:.2f}%'
        for op,stage,v in failures)+'。因此平均改善不表示所有教师都成功。' if failures else '四个主条件在六教师中均优于随机均值；这仍限定于当前冻结名单和已见面板。'
    lines=['# dSprites三阶通道组合与ReLU抵消机制','',
        '2026年10月11日。本轮接续通道配对定位，在六个冻结CNN的既有36图、72个有向形状配对上枚举8通道名单内全部56个三元组，精确拆分上一轮的三阶及以上余项，并对半强度定位的具体开关实施全强度反事实。使用六个模型的冻结权重。','',
        '本轮回答的是：第三个通道怎样改变两个通道的联合效应；三阶系数在哪些ReLU坐标产生正、负贡献；这些贡献和四阶以上项怎样抵消。三元组、开关坐标、符号分组和预定单点例子均在全强度计算前封存。','',
        '直接结论是：纯三阶和与四阶以上余项都不能忽略，而且二者存在抵消；定位的第一层开关固定平均比同池随机对照更能削弱三阶效应，但强度或背景变化会使某些三元组变号。具体开关的正向与反向作用已由预定单点验证，不能把小净效应解释成每个通道或开关都没有作用。','',
        '## 三阶到底检验什么','',
        '任务值F是供体形状logit减接收体形状logit。A、B、C代表block2输出中三个完整空间通道的替换，不是三个block3开关。恢复以源干预后的中介为基准、用接收体通道替换；移植方向相反。强度α=0.5用于定位，α=1.0用于主验证。','',
        r'$$\theta_{ABC}=F(ABC)-F(AB)-F(AC)-F(BC)+F(A)+F(B)+F(C)-F(\varnothing).$$','',
        '它也等于“C已替换时AB的配对效应”减去“C未替换时AB的配对效应”。因此检测的是第三通道改变配对关系，而不只是三通道替换后输出改变。八个状态固定为：空、A、B、AB、C、AC、BC、ABC。三阶不是三层或三个开关同时作用：单个阈值ReLU也能产生非零三阶差分。','',
        '前置检查含一个可手算的阈值例子：A/B/C∈{0,1}时，ReLU(A+B+C−0.5)的八项差分为−0.5；0.7·ReLU(A+B+C−2.5)为+0.35；两项相加只剩−0.15。它说明一个阈值就能产生三阶项，两个较大阈值贡献可以留下较小净值。此例用于协议前程序检查，不是另一次模型训练，也不替代真实CNN的实际固定验证。','',
        '每个输入继承此前冻结的8通道有序名单，并保持前4通道前缀。半强度按空背景三阶效应绝对值选择前3个三元组；每个三元组再在“其余5通道未替换”和“其余5通道已替换”两个背景验证。补全背景中的三阶差分包含该三元组的四阶以上严格超集贡献。只复测同样三个三元组，没有扫描全部条件背景后再挑最大值。','',
        r'$$\theta_{ABC}(K)=\sum_{U\subseteq K}\theta_{ABC\cup U}(\varnothing).$$','',
        '这里K是另外5通道，来自有限集合容斥恒等式。补全与空背景的差值合并31个包含ABC的四阶以上系数，既不是完整四阶以上合并余项，也不能唯一指定某个四元组。','',
        '## 上一轮余项怎样被拆开','',
        r'$$C_8=F(P_8)-F(\varnothing)-\sum_{i\in P_8}[F(i)-F(\varnothing)]=P_2+T_3+R_{4..8}.$$','',
        'P₂是全部28个空背景配对系数的带符号和，T₃是全部56个纯三阶系数的带符号和，R₄..₈是四阶至八阶的合并余项。上一轮的“三阶及以上”是T₃+R₄..₈，不等于纯三阶。本轮95个全局子集足以精确计算这三个量，但没有逐一计算四阶以上的163个系数。','',
        table(['操作','成分','相对C₈的RMS比','带符号投影'],[[OP_CN[op],label,
            f'{100*mean(a["hierarchy"],"rms_ratio",alpha=1.,operation=OPS[op],order=name):.2f}%',
            f'{mean(a["hierarchy"],"signed_projection",alpha=1.,operation=OPS[op],order=name):+.3f}']
            for op in range(2) for name,label in [('pairs','二阶和P₂'),('pure_triples','纯三阶和T₃'),('orders4plus','四至八阶R₄..₈')]]),'',
        '带符号投影为E[C₈·成分]/E[C₈²]，三个投影精确相加为1。RMS比可以超过100%，不能把它们当成互斥解释比例。下表直接检查T₃与R₄..₈怎样组合成此前的余项，以及56个三阶系数内部是否相消。','',
        table(['操作','T₃/旧三阶以上RMS','R₄..₈/旧三阶以上RMS','T₃与R₄..₈余弦','两成分抵消指数','56系数内部抵消'],[[OP_CN[op],
            f'{100*mean(full_orders,"pure3_to_old3plus_rms",operation=OPS[op]):.2f}%',
            f'{100*mean(full_orders,"remaining4_to_old3plus_rms",operation=OPS[op]):.2f}%',
            f'{mean(full_orders,"triple_vs_orders4_cosine",operation=OPS[op]):+.3f}',
            f'{100*mean(full_orders,"third_and_higher_cancellation",operation=OPS[op]):.2f}%',
            f'{100*mean(full_orders,"within_pure_triples_cancellation",operation=OPS[op]):.2f}%'] for op in range(2)]),'',
        '抵消指数使用1−Σ|净和|/Σ各项绝对值，先对一个教师的72例加权，再跨六教师取均值；0表示同向、接近1表示正负贡献大幅抵消。它是代数结构量，不是因果贡献率。','',
        '![八通道阶次分解](figures/dsprites_triple_gates_20261011_hierarchy.png)','',
        '## 冻结开关是否真的改变三阶效应','',
        '每层有32768个标量开关。半强度在有开关变化的坐标中，按带符号坐标贡献的绝对值选最多128个；随机对照使用同一候选池、相同实际数量、两个固定随机排列。全强度使用原名单，固定值为当前背景的基准开关。主比较只使用空背景、半强度第一三元组、预算128；先算每教师剩余RMS比，再跨教师平均。','',
        table(['操作','固定层','选中128剩余RMS','随机均值','与随机差距','优于随机教师'],[[OP_CN[OPS.index(r['operation'])],f'ReLU{r["stage"]}',f'{100*r["selected_mean"]:.2f}%',f'{100*r["random_mean"]:.2f}%',f'{r["gain_pp"]:+.2f}个百分点',f'{r["better_than_random_teachers"]}/6'] for r in primary]),'',
        failure_text,'',
        '低于100%表示三阶效应幅度减弱，超过100%表示幅度增强；这不是分类正确率。预算32、前3三元组、补全背景、半强度重复均保留在[全条件汇总](../runs/dsprites_triple_gates_20261011/tables/summary.csv)，不以补充结果替代主比较。','',
        '![冻结开关主比较](figures/dsprites_triple_gates_20261011.png)','',
        '把同一128名单按半强度贡献相对半强度θ的方向，拆成同向和反向两组。组内坐标数量不同，因此这些结果用于检验抵消。带符号净削弱投影是E[θ(θ−θ固定)]/E[θ²]：正值表示移除沿θ方向的贡献，负值表示固定之后θ方向上的效应反而增强。','',
        table(['操作','层','半强度符号组','平均坐标数','全强度净削弱投影','剩余RMS'],[[OP_CN[OPS.index(r['operation'])],r['stage'],'同向' if r['variant']=='aligned_subset' else '反向',f'{r["mean_gate_count"]:.1f}',f'{r["signed_removal"]:+.3f}',f'{100*r["remaining_ratio"]:.2f}%'] for r in a['signed_groups']]),'',
        '分组成员保持半强度定义，某个坐标在全强度可能不再变化或贡献符号改变。第一层干预会改变第二层预激活和门控余项，并可能改变开关位，所以实际净作用不保证遵循半强度代数符号。每例的方向检验、开关保留比例和符号稳定性见[gate_stability.csv](../runs/dsprites_triple_gates_20261011/tables/gate_stability.csv)。','',
        '## 两层贡献与坐标内部抵消','',
        '以当前背景的基准开关g₁⁰、g₂⁰为参照，r₁=ReLU(z₁)−g₁⁰z₁，r₂=ReLU(z₂)−g₂⁰z₂；q是含1/1024空间平均的任务读出权重，p=A₂ᵀ(g₂⁰q)。八项差分把仿射部分消掉，得到精确恒等式：','',
        r'$$\theta_{ABC}=\sum_u p_u\Delta_A\Delta_B\Delta_C r_{1u}+\sum_v q_v\Delta_A\Delta_B\Delta_C r_{2v}=L_1+L_2.$$','',
        '第一项在固定第二层开关的传播下定义；第二项使用自然上游产生的第二层预激活。分量由所选基准定义。固定第一层之后第二层会重新计算，所以第一层坐标代数贡献不等于实际净变化。第二层没有后续非线性，固定某些第二层坐标的净变化精确等于这些坐标贡献的和。','',
        r'$$\nabla_H F=A_1^\top D_1 A_2^\top D_2q+D_2q.$$','',
        '这延续前轮的梯度问题：在当前残差块内，D₁、D₂是两层自然开关的对角矩阵；梯度式在非阈值点成立，恰好为0时采用程序的关闭约定。开关固定时尾部梯度不随H改变，映射仿射。不同通道子集跨过阈值后，D改变并改变传播路径；三阶差分检验第三通道改变AB配对关系。开关变化是产生非加和效应的必要计算来源，但多处正负贡献可抵消，所以有变化并不保证大的净三阶值。','',
        '对第一层实际固定，基准状态的输出保持不变，g₂⁰和p也保持不变。因此净变化可再拆成“直接移除的第一层项”和“第二层余项重新计算的响应”：δθ=Σ被固定点pΔ³r₁+qΔ³(r₂原−r₂固定后)。后一个量从已核验的净输出差减去精确第一层项推得，属于读出加权的合并响应。全体案例使用这个推得的合并量；预定例子另记录了第二层逐坐标响应。以下主条件投影同样相对于原θ。','',
        '另外使用原block3.forward重放预定例子的两操作、两背景、两强度，直接记录第一层固定后第二层预激活。8组固定和8个预定单点合计128条件；实测第二层余项差与推得响应的最大误差为'+f'{net_check["maximum_errors"]["inferred_vs_recorded_second_response"]:.2e}'+'。这个复核针对新增的分析推导，名单全部复用原协议。','',
        table(['操作','第一层名单','实际净削弱投影','直接第一层投影','推得第二层响应投影'],[[OP_CN[op],label,
            f'{mean(a["first_stage_net_summary"],"net_projection",operation=OPS[op],context="empty",variant=name):+.3f}',
            f'{mean(a["first_stage_net_summary"],"first_projection",operation=OPS[op],context="empty",variant=name):+.3f}',
            f'{mean(a["first_stage_net_summary"],"inferred_second_projection",operation=OPS[op],context="empty",variant=name):+.3f}']
            for op in range(2) for name,label in [('selected128','全部选中128'),('aligned_subset','半强度同向'),('opposed_subset','半强度反向')]]),'',
        table(['操作','L₁带符号投影','L₂带符号投影','两层之间抵消','所有坐标之间抵消','整层R1固定剩余','整层R2固定剩余'],[[OP_CN[op],
            f'{mean(a["layer_holds"],"first_signed_projection",operation=OPS[op],context="empty"):+.3f}',
            f'{mean(a["layer_holds"],"second_signed_projection",operation=OPS[op],context="empty"):+.3f}',
            f'{100*mean(a["cancellation"],"layer_cancellation",operation=OPS[op],context="empty"):.2f}%',
            f'{100*mean(a["cancellation"],"unit_cancellation",operation=OPS[op],context="empty"):.2f}%',
            f'{100*mean(a["layer_holds"],"hold_first_ratio",operation=OPS[op],context="empty"):.2f}%',
            f'{100*mean(a["layer_holds"],"hold_second_ratio",operation=OPS[op],context="empty"):.2f}%'] for op in range(2)]),'',
        '坐标抵消统计包含每层全部32768坐标，不限于选中128。两层同时固定后的最大三阶误差为'+f'{audit["maximum_errors"]["affine_triples_zero"]:.2e}'+'，对应同时固定两处开关后的仿射映射。强制基准开关可能让负预激活传向后层，属于内部计算反事实。','',
        '## 预定输入的具体三元组和开关','',
        '例子在协议中预定：教师202609291、案例0，最小大小、角度索引17（约157°）、方形到椭圆。下表列出恢复和移植各自半强度前3三元组及两个背景，没有按全强度结果换例子。物理通道顺序保留冻结名单中的A/B/C顺序。','',
        table(['操作','半强度名次','背景','A/B/C物理号','半强度θ','全强度θ','L₁','L₂'],[
            [OP_CN[OPS.index(r['operation'])],r['half_rank'],'空' if r['context']=='empty' else '另5通道',
             f'({r["channel_a"]},{r["channel_b"]},{r["channel_c"]})',f'{r["half_theta"]:+.5f}',f'{r["full_theta"]:+.5f}',f'{r["first_part"]:+.5f}',f'{r["second_part"]:+.5f}'] for r in a['example']]),'',
        '空背景恢复的第一三元组A/B/C='+str(triplet)+'。第三通道对配对关系的改变如下，两个配对值相减就是θ；补全背景的参照包含其余5通道。','',
        table(['操作','背景','AB效应：C未替换','AB效应：C已替换','差值θ'],[[OP_CN[OPS.index(r['operation'])],r['context'],f'{r["pair_ab_without_c"]:+.6f}',f'{r["pair_ab_with_c"]:+.6f}',f'{r["full_theta"]:+.6f}'] for r in a['example'] if r['half_rank']==1]),'',
        '以下仅取预定恢复、空背景第一三元组，按层列出全部坐标正负和。净θ很小并不意味着每一坐标作用都小：同一层内的正、负和以及两层净值可能相互抵消。','',
        table(['层','全部坐标正贡献之和','全部坐标负贡献之和','层净贡献'],[
            [1,f'{example["first_positive"]:+.6f}',f'{example["first_negative"]:+.6f}',f'{example["first_part"]:+.6f}'],
            [2,f'{example["second_positive"]:+.6f}',f'{example["second_negative"]:+.6f}',f'{example["second_part"]:+.6f}']]),'',
        table(['固定层','固定名单','实际坐标数','固定后θ','净移除θ−θ固定'],[[stage+1,('全部选中128','半强度同向子集','半强度反向子集')[j],example['gate_counts'][stage][variant],f'{example["held"][stage][variant]:+.6f}',f'{example["full_theta"]-example["held"][stage][variant]:+.6f}'] for stage in range(2) for j,variant in enumerate((3,6,7))]),'',
        '协议预先规定各层半强度前8坐标逐点固定，不是看到全强度结果后挑点。下面完整列出恢复、空背景的16个单点。开关串按空/A/B/AB/C/AC/BC/ABC排序；(C,Y,X)属于block3，不能与block2输入通道按编号对应。第一层“坐标贡献”和“实际净移除”可能不同，甚至异号；第二层两列数值一致。','',
        table(['层','开关(C,Y,X)','半强度串','全强度串','全强度坐标贡献','单点固定后θ','实际净移除'],[
            [r['stage'],f'({r["gate_channel"]},{r["gate_y"]},{r["gate_x"]})',r['half_pattern'],r['full_pattern'],f'{r["full_contribution"]:+.6f}',f'{r["full_held"]:+.6f}',f'{r["full_removed"]:+.6f}']
            for r in a['example_single_sites'] if r['operation']=='restore' and r['context']=='empty']),'',
        '移植空背景的第一三元组另列前3点，以及预定名单第8点作为代数与净作用的对照。仍使用原半强度次序，没有按全强度净变化重排。','',
        table(['层','半强度名次','开关(C,Y,X)','全强度串','坐标贡献','单点固定后θ','实际净移除','推得第二层响应'],[
            [r['stage'],r['coordinate_rank'],f'({r["gate_channel"]},{r["gate_y"]},{r["gate_x"]})',r['full_pattern'],f'{r["full_contribution"]:+.6f}',f'{r["full_held"]:+.6f}',f'{r["full_removed"]:+.6f}',f'{r["inferred_second_response"]:+.6f}']
            for r in a['example_single_sites'] if r['operation']=='transplant' and r['context']=='empty' and r['stage']==1 and r['coordinate_rank'] in (1,2,3,8)]),'',
        '00010111表示空背景和三个单通道都关闭，而任意二通道及三通道组合打开；01111111表示只有空背景关闭，其余组合打开。开关串本身不决定贡献正负，还必须乘传播权重并计算八项差分。固定第一层某点后，即便其三阶代数项接近0，第二层余项重算仍可能造成非零净变化；最后一行保留了这个检验。','',
        '![八状态预激活与开关](figures/dsprites_triple_gates_20261011_example.png)','',
        '下面直接追踪预定移植三元组(20,5,12)在空背景中固定第一层后的第二层响应。“改变坐标数”指至少一个局部状态的开关位改变的第二层坐标数；它不等于贡献非零的坐标数。第一层名单均已预定，下面的第二层响应坐标排序属于结果后的描述性定位，没有用来新增坐标干预。','',
        table(['第一层固定','直接第一层移除','实测第二层响应','固定后θ','第二层改变坐标数'],[[net_labels[r['intervention']],f'{float(r["first_removed"]):+.6f}',f'{float(r["second_response"]):+.6f}',f'{float(r["held_theta"]):+.6f}',r['second_flip_sites']] for r in net_example]),'',
        '固定第一层选中128点后，该三元组仍有θ=0.399853：原第一层项0.314344减去直接移除0.055192，剩0.259151；原第二层项0.465739减去响应0.325038，剩0.140702。两项相加构成剩余三阶效应。反向组则移除负贡献，使两层合并效应增强。','',
        '下表显示选中128组、单点1和单点8的第二层正、负响应各前2坐标。响应为qᵥΔ³(r₂原−r₂固定后)，只反映该第一层反事实下的第二层变化，不是新执行的第二层单点消融。开关位不变时，也可能因已有跨状态门控与预激活幅度改变而产生响应。','',
        table(['第一层固定','第二层坐标(C,Y,X)','原开关串','固定后串','读出加权响应'],[[net_labels[r['intervention']],f'({r["second_gate_channel"]},{r["second_gate_y"]},{r["second_gate_x"]})',r['natural_pattern'],r['held_pattern'],f'{float(r["signed_response"]):+.7f}']
            for r in net_coordinates if float(r['alpha'])==1. and r['operation']=='transplant' and r['context']=='empty' and r['intervention'] in ('selected128','single_1','single_8') and int(r['response_rank'])<=2]),'',
        '例如单点8对应第一层(16,17,16)，直接三阶贡献接近0，但固定后15个第二层坐标有开关位变化，第二层总响应+0.004187。其最大正响应点(9,16,16)的开关串仍是00000001，说明单看翻转次数不能解释全部效应。全部128条件及正负响应各前4坐标见[net_response_cases.csv](../runs/dsprites_triple_gates_20261011/tables/net_response_cases.csv)、[net_response_coordinates.csv](../runs/dsprites_triple_gates_20261011/tables/net_response_coordinates.csv)；全部第二层响应图和开关编码保存在net_response_maps.npz。','',
        '其余移植及补全背景的预定逐点结果全部保留在[single_sites.csv](../runs/dsprites_triple_gates_20261011/tables/single_sites.csv)。六教师所有输入前3三元组、两背景、两层前8坐标的半/全强度开关串见[gate_coordinates_top8.csv](../runs/dsprites_triple_gates_20261011/tables/gate_coordinates_top8.csv)，全部128坐标预激活在各评价data.npz中。','',
        '## 背景依赖与强度稳定性','',
        table(['操作','补全背景/空背景RMS','背景变化/空背景RMS','两背景同号比例','56系数绝对排序Spearman','半/全前3重合'],[[OP_CN[op],
            f'{mean(a["context_dependence"],"complement_to_empty_rms",operation=OPS[op]):.3f}',
            f'{mean(a["context_dependence"],"difference_to_empty_rms",operation=OPS[op]):.3f}',
            f'{100*mean(a["context_dependence"],"same_sign_fraction",operation=OPS[op]):.2f}%',
            f'{mean(a["ranking_stability"],"abs_rank_spearman",operation=OPS[op]):.3f}',
            f'{100*mean(a["ranking_stability"],"top3_overlap",operation=OPS[op]):.2f}%'] for op in range(2)]),'',
        '上表背景对比使用半强度第一三元组，两个背景的三元组完全一致。背景变化包含四阶以上超集作用，不能据此把所有变化归因于这三个通道。完整四阶以上合并余项仍待逐组合定位。强度排序变化则说明半强度名单并非每次都在全强度最强；本轮坚持原名单，所以主结果可以暴露定位失效。','',
        '具体到预定移植第一三元组(20,5,12)，全强度空背景θ='+f'{next(r["full_theta"] for r in a["example"] if r["operation"]=="transplant" and r["half_rank"]==1 and r["context"]=="empty"):+.6f}'+
        '，另外5通道已替换时θ='+f'{next(r["full_theta"] for r in a["example"] if r["operation"]=="transplant" and r["half_rank"]==1 and r["context"]=="complement"):+.6f}'+
        '。两个背景改变了效应方向，说明该三元组的作用需要连同背景描述。','',
        '物理三元组按其确实出现于输入8通道名单的案例统计，缺失的组合没有填0。按半强度平方和排序并附覆盖次数，见[physical_triples.csv](../runs/dsprites_triple_gates_20261011/tables/physical_triples.csv)；各教师前三组合在analysis.json中。排序仅描述已有数据，不再用于本轮干预选择。不同教师的同号通道不作语义对齐。','',
        '为方便定位，下表列出每教师、每操作的半强度平方和第一物理组合。平方和同时受覆盖数量和幅度影响，不能把不同覆盖的行直接当作同预算消融方法比较；每例实际干预仍由该例半强度前三名单决定。','',
        table(['教师','操作','物理三元组（排序显示）','覆盖案例','半强度RMS','全强度RMS','半强度首位次数'],[
            [r['seed'],OP_CN[OPS.index(r['operation'])],f'({r["channel_1"]},{r["channel_2"]},{r["channel_3"]})',r['observed_cases'],f'{r["half_rms"]:.5f}',f'{r["full_rms"]:.5f}',r['half_top1_cases']]
            for r in a['top_physical_triples'] if r['rank']==1]),'',
        '## 证据边界和下一步','',
        '本轮给出了可检验的链条：通道子集改变第一层预激活与开关，经过卷积和基准第二层传播产生一部分三阶贡献；自然第二层的门控余项产生另一部分；正负坐标与更高阶项抵消后形成观察到的净余项。第一层固定又改变第二层预激活，既可能改变开关位，也可能在开关串不变时改变已有门控的贡献。精确代数支持这一分解，实际固定实验检验部分坐标的作用，两类证据应分开解释。','',
        '主结果、符号子集和单点反事实都限定于六个教师、继承8通道名单、具体输入与修补方向。没有证明一个跨模型普适电路，也没有证明开关固定改善分类。36图已用于多轮观察，只有约157°、203°两个角度且形状面积不同，尚不能断言纯形状语义或旋转泛化。72 个有向配对来自同一组 36 张图。','',
        '后续应优先做两个检验：对尚未定位的四阶以上合并余项，预先选定候选四元组并测完整16状态；对目前的三元组与开关模式，在未用于定位的因素组合及面积/边缘对照上执行固定名单干预。第一层若出现代数与净作用异号，应同时记录干预后的第二层开关变化，定位下游抵消，而不是用代数排名替代实际反事实。','',
        '## 可复核材料','',
        f'独立核验通过：全部95状态、四种全层模式、全部56纯三阶、冻结前3三元组的两背景、八种部分固定名单及预定逐点干预均与原block3.forward核对；坐标传播权重同时用独立伴随和autograd核对。最大数值误差{max(audit["maximum_errors"].values()):.2e}，浮点坐标排名边界记录{len(audit["floating_coordinate_rank_notes"])}条。此前五轮共943条交付文件SHA256均保持一致。','',
        '- [协议、数组维度、代码说明与只读复核入口](../runs/dsprites_triple_gates_20261011/README.md)',
        '- [全部56三元组的半/全强度系数和两层贡献](../runs/dsprites_triple_gates_20261011/tables/all56_triples.csv)',
        '- [已选三元组、条件配对、两层正负贡献](../runs/dsprites_triple_gates_20261011/tables/chosen_triples.csv)',
        '- [层与坐标抵消](../runs/dsprites_triple_gates_20261011/tables/cancellation.csv)',
        '- [第一层固定净作用与第二层合并响应](../runs/dsprites_triple_gates_20261011/tables/first_stage_net.csv)',
        '- [数值核心代码](../scripts/triple_gate_mechanism.py)',
        '- [独立审计代码](../scripts/audit_dsprites_triple_gates.py)',
        '- [报告分析代码](../scripts/report_dsprites_triple_gates.py)','',
        f'协议SHA256：`{teacher.sha(RUN/"protocol.json")}`。本轮精确容斥与门控恒等式是当前计算图中的结构验证，尚未提出新的普遍归因定理。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-docs',action='store_true')
    args=parser.parse_args()
    assert not (RUN/'delivery_manifest.json').exists(),'Delivery frozen; use read-only delivery verifier'
    # 最终封存前修订文字或图形，只读既有分析，不覆盖数值数据和表格。
    if not args.refresh_docs and not (RUN/'net_response_check.json').exists():validate_second_response()
    a=teacher.read(RUN/'analysis.json') if args.refresh_docs else analyze()
    assert a['status']=='PASS' and teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    figures(a);report(a)
    print(json.dumps(dict(status='PASS',report=str(REPORT),primary=a['primary'],table_rows=a['table_rows']),ensure_ascii=False))

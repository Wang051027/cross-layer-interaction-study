"""从已独立核验的层间传递数组提取证据表和科研图。

不训练、不新增输入、不选择新的通道或例子。六教师先分别计算指标，
再等权平均；共享图像的配对不当成独立样本做显著性检验。
"""
from __future__ import annotations

import csv
import os
import dsprites_teacher as teacher
import numpy as np

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_layer_transfer_20261013'
FIG=BASE/'reports/figures'
PREFIX='dsprites_layer_transfer_20261013'


def load(path):
    """关闭NPZ句柄并拒绝pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def table(name,rows):
    """UTF-8 BOM供本地Excel阅读；数值列保留完整精度。"""
    assert rows and not (RUN/f'tables/{name}.csv').exists(),name
    path=RUN/f'tables/{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def ratio(a,b):
    """近零分母保持缺失值，避免把零基准写成确定比例。"""
    return None if abs(b)<1e-10 else float(a/b)


def make_figures(s,examples,gradients,repair=False):
    """全部图由固定案例或全部六教师生成，保存PNG和SVG。"""
    assert not (RUN/'delivery_manifest.json').exists()
    os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    FIG.mkdir(parents=True,exist_ok=True)
    files=[]
    def save(fig,suffix):
        for ext in ('png','svg'):
            path=FIG/f'{PREFIX}{suffix}.{ext}';assert repair or not path.exists()
            fig.savefig(path,dpi=180,bbox_inches='tight');files.append(str(path.relative_to(BASE)).replace('\\','/'))
        plt.close(fig)
    def values(alpha,method,operation='erase',key='remaining_ratio'):
        return np.array([next(x[key] for x in s['rows'] if x['seed']==seed and x['alpha']==alpha and x['metric']=='conditional_pairs' and x['method']==method and x['operation']==operation) for seed in teacher.SEEDS])
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
    methods=[('middle_le1','Erase middle orders 2–4','erase'),('rectified_additive','Erase + rectify','erase'),('upstream_fixed','Fix block2 gates','forward'),('downstream_fixed','Fix block3 gates','forward'),('both_fixed','Fix all four gates','forward')]
    for ax,alpha in zip(axes,(1.,.5)):
        for index,(method,label,op) in enumerate(methods):
            y=values(alpha,method,op)*100
            ax.scatter(index+np.linspace(-.15,.15,6),y,color='tab:blue',alpha=.6,s=22)
            ax.plot([index-.24,index+.24],[y.mean()]*2,color='black',linewidth=2)
        ax.axhline(100,color='gray',linestyle='--',linewidth=1)
        ax.set_xticks(range(5),[x[1] for x in methods],rotation=23,ha='right')
        ax.set_ylabel('Remaining conditional-pair RMS (%)');ax.set_title(f'Alpha = {alpha:g} · same 36 images / 72 directed pairs')
        ax.set_ylim(-7,195);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Middle interaction erasure and layer gate controls\nDots: six frozen teachers; black line: their equal-weight mean',fontsize=12)
    save(fig,'')

    # 预先固定的G0/G1、空背景、第一教师第0案例；不按效应大小挑选。
    example=load(RUN/'evaluation/seed_202609291/alpha_1.0/example_fields.npz')
    pair=example['middle_coefficients'][3]
    norm=np.sqrt(np.mean(pair**2,axis=0));projection=np.sum(pair*example['middle_gradient'][0],axis=0)
    fig=plt.figure(figsize=(12.2,8),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1,1.1])
    ax=fig.add_subplot(grid[0,0]);im=ax.imshow(norm,cmap='viridis');fig.colorbar(im,ax=ax,label='RMS across 32 channels')
    ax.set_title('G0/G1 middle interaction · empty background');ax.set_xlabel('Middle x');ax.set_ylabel('Middle y')
    ax=fig.add_subplot(grid[0,1]);lim=float(np.max(abs(projection)))
    im=ax.imshow(projection,cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=ax,label='Task-weighted contribution')
    ax.set_title('Same field projected through baseline gradient');ax.set_xlabel('Middle x');ax.set_ylabel('Middle y')
    ax=fig.add_subplot(grid[1,:]);rows=[x for x in examples if x['alpha']==1.]
    x=np.arange(11);width=.25
    for offset,key,label,color in [(-1,'natural','Natural','tab:blue'),(0,'downstream_fixed','Fixed block3 gates','tab:orange'),(1,'middle_le1','Erased middle orders 2–4','tab:green')]:
        ax.bar(x+offset*width,[r[key] for r in rows],width,label=label,color=color,alpha=.85)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,[r['groups'] for r in rows],rotation=24,ha='right',fontsize=9);ax.set_ylabel('Pure source-subset coefficient of task logit')
    ax.set_title('All 11 order 2–4 coefficients in the same predefined case');ax.legend(ncol=3,fontsize=9);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Teacher 202609291 · case 0 · square → ellipse · alpha 1\nSpatial magnitude and output direction are different quantities',fontsize=12)
    save(fig,'_example')

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,alpha in zip(axes,(1.,.5)):
        for index,cut in enumerate(('source','middle')):
            selected=[r for r in gradients if r['alpha']==alpha and r['cut']==cut]
            y=[r['full_state_relative_change'] for r in selected]
            ax.scatter(index+np.linspace(-.13,.13,6),y,color=('tab:blue','tab:orange')[index],s=35)
            ax.plot([index-.2,index+.2],[np.mean(y)]*2,color='black',linewidth=2)
        ax.set_xticks([0,1],['Source: block1 output','Middle: block2 output']);ax.set_ylabel('Relative task-gradient change, full vs empty state')
        ax.set_title(f'Alpha = {alpha:g} · equal weight for six teachers');ax.set_ylim(0,1.15);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Gradients describe the active propagation paths\nFinite interaction is verified separately by actual forward interventions',fontsize=12)
    save(fig,'_gradients')
    return files


def report():
    """一次性提取；完成交付后拒绝覆盖任何结果。"""
    assert not (RUN/'delivery_manifest.json').exists() and not (RUN/'analysis.json').exists()
    from run_dsprites_layer_transfer import verify_protocol,verify_folder
    verify_protocol();audit=teacher.read(RUN/'independent_audit.json');assert audit['status']=='PASS'
    assert teacher.sha(RUN/audit['negative_fraction']['file'])==audit['negative_fraction']['sha256']
    s=teacher.read(RUN/'summary.json');assert len(s['rows'])==576 and len(s['decomposition'])==24
    d=load(RUN/'design.npz');panel=load(BASE/'runs/dsprites_shape_donor_20261010/panel/data.npz')
    w=np.zeros((72,3));w[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1;w[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    neg=load(RUN/'negative_fraction_audit.npz');anchors=np.array([0,1,2,4,8]);nonanchors=np.array([k for k in range(16) if k not in anchors])
    records={k:[] for k in ('coefficients','transmission','gradients','gradient_states','examples','example_states','negative','permutations','channels')}
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);raw=load(folder/'data.npz')
            modes=np.einsum('nsmk,nk->nsm',raw['logits'],w);lower=np.einsum('nqsk,nk->nqs',raw['lower_logits'],w)
            co=np.einsum('vs,nsm->nvm',d['mobius'],modes);lowerco=np.einsum('vs,nqs->nqv',d['mobius'],lower)
            cp=lambda f:f[:,d['stencils'][:,0]]-f[:,d['stencils'][:,1]]-f[:,d['stencils'][:,2]]+f[:,d['stencils'][:,3]]
            nat=cp(modes[:,:,0]);add=cp(lower[:,0]);delta=nat-add
            for case in range(72):
                for state in range(16):
                    row=dict(seed=seed,alpha=alpha,case=case,mask=state,order=int(d['order'][state]),
                        source_norm=float(np.linalg.norm(raw['source_mobius_channel_norm'][case,state])),
                        middle_norm=float(np.linalg.norm(raw['middle_mobius_channel_norm'][case,state])),
                        natural=float(co[case,state,0]),upstream_fixed=float(co[case,state,1]),downstream_fixed=float(co[case,state,2]),both_fixed=float(co[case,state,3]),
                        middle_le1=float(lowerco[case,0,state]),middle_le2=float(lowerco[case,1,state]),middle_le3=float(lowerco[case,2,state]),
                        baseline_transmitted=float(raw['baseline_mobius_channel_projection'][case,state].sum()))
                    records['coefficients'].append(row)
                    if seed==teacher.SEEDS[0] and case==0:
                        if d['order'][state]>=2:records['examples'].append({**row,'groups':'+'.join(f'G{g}' for g in range(4) if state&(1<<g))})
                        records['example_states'].append(dict(alpha=alpha,mask=state,groups='+'.join(f'G{g}' for g in range(4) if state&(1<<g)) or 'empty',natural=float(modes[case,state,0]),
                            upstream_fixed=float(modes[case,state,1]),downstream_fixed=float(modes[case,state,2]),both_fixed=float(modes[case,state,3]),middle_le1=float(lower[case,0,state]),
                            source_relative_gradient=float(raw['gradient_drift'][case,state,0,0]),middle_relative_gradient=float(raw['gradient_drift'][case,state,1,0]),
                            block2_mid_flips=int(raw['gate_flip_counts'][case,state,0]),block2_end_flips=int(raw['gate_flip_counts'][case,state,1]),
                            block3_mid_flips=int(raw['gate_flip_counts'][case,state,2]),block3_end_flips=int(raw['gate_flip_counts'][case,state,3])))
                for v,(g1,g2,bg) in enumerate(d['metadata']):
                    glob=float(raw['conditional_baseline_channel_projection'][case,v].sum());local=float(raw['conditional_context_channel_projection'][case,v].sum())
                    records['transmission'].append(dict(seed=seed,alpha=alpha,case=case,g1=int(g1),g2=int(g2),background_mask=int(bg),natural=float(nat[case,v]),middle_le1=float(add[case,v]),
                        mediated_change=float(delta[case,v]),middle_interaction_norm=float(np.linalg.norm(raw['conditional_middle_channel_norm'][case,v])),
                        global_transmitted=glob,global_remainder=float(nat[case,v]-glob),context_transmitted=local,context_remainder=float(nat[case,v]-local)))
            for index,cut in enumerate(('source','middle')):
                valid=raw['baseline_gradient_norm'][:,index]>=1e-20;full=raw['gradient_drift'][valid,15,index]
                records['gradients'].append(dict(seed=seed,alpha=alpha,cut=cut,valid_cases=int(valid.sum()),excluded_zero_base=int((~valid).sum()),
                    full_state_relative_change=None if not valid.any() else float(full[:,0].mean()),full_state_cosine=None if not valid.any() else float(full[:,1].mean()),full_state_norm=None if not valid.any() else float(full[:,2].mean())))
                for case in range(72):
                    for state in range(16):
                        g=raw['gradient_drift'][case,state,index]
                        records['gradient_states'].append(dict(seed=seed,alpha=alpha,case=case,mask=state,cut=cut,valid_base=bool(valid[case]),baseline_norm=float(raw['baseline_gradient_norm'][case,index]),
                            relative_change=float(g[0]) if valid[case] else None,cosine=float(g[1]) if valid[case] else None,norm=float(g[2])))
            ni=np.where((neg['seed']==seed)&(neg['alpha']==alpha))[0];assert len(ni)==1
            for degree in (1,2,3):
                values=neg['lower'][ni[0],:,degree-1][:,nonanchors]
                records['negative'].append(dict(seed=seed,alpha=alpha,degree=degree,zero_band=float(neg['zero_band']),mean_nonanchor_fraction=float(values.mean()),maximum_nonanchor_fraction=float(values.max()),
                    raw_strict_zero_fraction=float(raw['lower_negative_fraction'][:,degree-1][:,nonanchors].mean())))
            perm=np.einsum('nvsk,nk->nvs',raw['permuted_logits'],w)
            for v in (0,1):
                difference=cp(perm[:,v])-nat
                records['permutations'].append(dict(seed=seed,alpha=alpha,repeat=v,equal_norm=True,mediated_target_rms=rms(delta),replacement_error_rms=rms(difference),replacement_error_ratio=ratio(rms(difference),rms(delta))))
            for budget in (4,8):
                for op,key in (('erase','remaining_ratio'),('restore','restore_error_ratio')):
                    row=lambda method:next(x for x in s['rows'] if x['seed']==seed and x['alpha']==alpha and x['metric']=='conditional_pairs' and x['method']==method and x['operation']==op)
                    selected=row(f'selected{budget}')[key];random=float(np.mean([row(f'random{v}{budget}')[key] for v in (0,1)]))
                    records['channels'].append(dict(seed=seed,alpha=alpha,budget=budget,operation=op,metric=key,selected=selected,random_mean=random,selected_smaller=bool(selected<random)))
    overview=[]
    for alpha in (1.,.5):
        for metric in ('conditional_pairs','joint_all4'):
            frame=[r for r in s['rows'] if r['alpha']==alpha and r['metric']==metric]
            for method,op in dict.fromkeys((r['method'],r['operation']) for r in frame):
                group=[r for r in frame if r['method']==method and r['operation']==op];assert len(group)==6
                keys=('natural_rms','remaining_ratio','changed_ratio','signed_removal')+(('restore_error_ratio',) if op=='restore' else ())
                overview.append(dict(alpha=alpha,metric=metric,method=method,operation=op,teachers=6,**{k:None if any(r[k] is None for r in group) else float(np.mean([r[k] for r in group])) for k in keys}))
    # CSV需要统一列集合；restore列对forward/erase置空，明确语义不同。
    for row in overview:row.setdefault('restore_error_ratio',None)
    summary_rows=[{**r,'restore_error_ratio':r.get('restore_error_ratio')} for r in s['rows']]
    counts={name:table(name,rows) for name,rows in [('summary',summary_rows),('condition_overview',overview),('jacobian_decomposition',s['decomposition'])]}
    counts.update({name:table(name,rows) for name,rows in records.items()})
    primary=[r for r in s['rows'] if r['metric']=='conditional_pairs' and r['method']=='middle_le1']
    counts['primary_per_teacher']=table('primary_per_teacher',primary)
    assert counts['coefficients']==13824 and counts['transmission']==20736 and counts['gradient_states']==27648 and counts['examples']==22
    figures=make_figures(s,records['examples'],records['gradients'])
    result=dict(status='PASS',utc=teacher.now(),primary=primary,condition_overview=overview,gradient_summary=records['gradients'],channel_controls=records['channels'],
        negative_fraction=records['negative'],permutation_controls=records['permutations'],examples=records['examples'],table_rows=counts,figures=figures,
        averaging='RMS over72cases x24 contrasts within each teacher,then equal-weight mean of6 ratios;no p-values or independent-pair assumption',
        negatives='H<-1e-9 on11 nonanchor states;raw strict<0 preserved but near-zero signs depend on floating summation order',
        main_readout='donor-minus-recipient class logit;restore ratios target natural-minus-additive output contrast,not natural total',
        scope='same36images/72pairs;no additional inputs,training,newselection or unique causal share')
    teacher.dump(RUN/'analysis.json',result)
    print({'status':'PASS','table_rows':counts,'figures':len(figures)})


if __name__=='__main__':report()

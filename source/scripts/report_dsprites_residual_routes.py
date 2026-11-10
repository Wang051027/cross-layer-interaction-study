"""将固定通道名单的传递、阈值与抵消结果写入 CSV 和图表。"""
from __future__ import annotations
import csv
import os
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_residual_routes import RUN,BASE,SHAPE,load,verify_seal


def table(name,rows):
    assert rows and not (RUN/f'tables/{name}.csv').exists()
    path=RUN/f'tables/{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)


def report():
    assert not (RUN/'delivery_manifest.json').exists() and not (RUN/'analysis.json').exists();verify_seal()
    audit=teacher.read(RUN/'independent_audit.json');assert audit['status']=='PASS'
    s=teacher.read(RUN/'summary.json');d=load(RUN/'design.npz');panel=load(SHAPE/'panel/data.npz')
    w=np.zeros((72,3));w[np.arange(72),panel['labels'][panel['pairs'][:,1]]]=1;w[np.arange(72),panel['labels'][panel['pairs'][:,0]]]=-1
    cp=lambda x:x[:,d['stencils'][:,0]]-x[:,d['stencils'][:,1]]-x[:,d['stencils'][:,2]]+x[:,d['stencils'][:,3]]
    records={k:[] for k in ('primary','states','cancellation','gate_lists','example_channels','example_single_gates','example_patterns','path_segments','path_channels','path_events')}
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';raw=load(folder/'data.npz');sel=load(folder/'selection.npz')
            base=np.einsum('nsk,nk->ns',raw['baseline'],w);q=np.einsum('nsmk,nk->nsm',raw['modes'],w)-base[:,:,None]
            target=cp(q[:,:,0]);linear=cp(raw['linear_channel']);g1=cp(raw['gate_channel'][:,0]);g2=cp(raw['gate_channel'][:,1])
            row=lambda family,method,op:next(x for x in s['rows'] if x['seed']==seed and x['alpha']==alpha and x['family']==family and x['method']==method and x['operation']==op)
            for family,method,op,key in [('channel','selected8','restore','error_ratio'),('gate1','selected128','hold','remaining_ratio'),('gate2','selected128','hold','remaining_ratio')]:
                selected=row(family,method,op)[key];names=('random0_8','random1_8') if family=='channel' else ('random0_128','random1_128')
                random=float(np.mean([row(family,name,op)[key] for name in names]));records['primary'].append(dict(seed=seed,alpha=alpha,family=family,metric=key,selected=selected,random_mean=random,selected_smaller=bool(selected<random)))
            parts=next(x for x in s['decomposition'] if x['seed']==seed and x['alpha']==alpha)
            mass=raw['gate_signed_mass'];within=[]
            for stage,piece in enumerate((g1,g2)):
                denominator=mass[:,:,stage,0].sum()-mass[:,:,stage,1].sum();within.append(None if denominator<1e-20 else float(1-abs(piece.sum(2)).sum()/denominator))
            records['cancellation'].append(dict(seed=seed,alpha=alpha,**{f'{name}_rms':parts['component_rms_ratios'][j] for j,name in enumerate(('linear','first','second'))},
                **{f'{name}_projection':parts['component_projection'][j] for j,name in enumerate(('linear','first','second'))},coordinate_cancellation=parts['coordinate_cancellation'],layer_cancellation=parts['layer_cancellation'],first_within_cancellation=within[0],second_within_cancellation=within[1]))
            for case in range(72):
                for mask in range(16):records['states'].append(dict(seed=seed,alpha=alpha,case=case,mask=mask,delta=float(q[case,mask,0]),linear=float(raw['linear_channel'][case,mask].sum()),
                    first=float(raw['gate_channel'][case,0,mask].sum()),second=float(raw['gate_channel'][case,1,mask].sum()),hold_first_delta=float(q[case,mask,1]),hold_second_delta=float(q[case,mask,2]),
                    hold_both_delta=float(q[case,mask,3]),full_first_second_response=float(raw['full_first_response'][case,mask])))
                for stage in (0,1):
                    for rank,coordinate in enumerate(sel['gates'][case,stage,0,:8]):
                        if coordinate<0:continue
                        ch,yy,xx=np.unravel_index(coordinate,(32,32,32));records['gate_lists'].append(dict(seed=seed,alpha=alpha,case=case,stage=stage+1,rank=rank,channel=int(ch),y=int(yy),x=int(xx),half_score=float(sel['selected_gate_score'][case,stage,0,rank]),half_pool_count=int(sel['gate_pool_count'][case,stage])))
            if seed==teacher.SEEDS[0]:
                fields=load(folder/'example_fields.npz');single=raw['single_logits'][0]@w[0]-base[0][None]
                score=target[0]@cp(single.T[None])[0]/np.sum(target[0]**2)
                for c in range(32):records['example_channels'].append(dict(alpha=alpha,channel=c,selected8=bool(c in sel['channels'][0,1]),half_score=float(sel['channel_score'][0,c]),current_score=float(score[c]),
                    single_restore_mask3=float(single[c,3]),single_restore_mask15=float(single[c,15]),linear_mask3=float(raw['linear_channel'][0,3,c]),linear_mask15=float(raw['linear_channel'][0,15,c])))
                for stage in (0,1):
                    for index,name in ((5,'selected1'),(6,'positive1'),(7,'negative1')):
                        coordinate=int(sel['gates'][0,stage,index,0]);ch,yy,xx=np.unravel_index(max(coordinate,0),(32,32,32))
                        for mask in (3,15):
                            value=float(raw['gate_logits'][0,stage,index,mask]@w[0]-base[0,mask]);direct=float(raw['gate_direct'][0,stage,index,mask]);response=float(raw['first_second_response'][0,index,mask]) if stage==0 else 0.
                            records['example_single_gates'].append(dict(alpha=alpha,stage=stage+1,kind=name,mask=mask,coordinate=coordinate,channel=int(ch) if coordinate>=0 else None,y=int(yy) if coordinate>=0 else None,x=int(xx) if coordinate>=0 else None,
                                natural_delta=float(q[0,mask,0]),held_delta=value,net_removed=float(q[0,mask,0]-value),direct_removed=direct,second_response=response))
                    for kind,index in (('selected128',0),('positive1',6),('negative1',7)):
                        ids=sel['gates'][0,stage,index,:8];keya,keyh=('za','zh') if stage==0 else ('va','vh')
                        for rank,coordinate in enumerate(ids):
                            if coordinate<0:continue
                            ch,yy,xx=np.unravel_index(coordinate,(32,32,32));before=fields[keya][:,ch,yy,xx];after=fields[keyh][:,ch,yy,xx]
                            records['example_patterns'].append(dict(alpha=alpha,stage=stage+1,kind=kind,rank=rank,channel=int(ch),y=int(yy),x=int(xx),
                                baseline_pattern='states:'+''.join(map(str,(before>0).astype(int))),natural_pattern='states:'+''.join(map(str,(after>0).astype(int))),
                                baseline_mask3=float(before[3]),natural_mask3=float(after[3]),baseline_mask15=float(before[15]),natural_mask15=float(after[15])))
                for mask in (3,15):
                    path=load(folder/f'path_mask{mask}.npz');derivative=path['channel_derivative'].sum(1)
                    for j,t in enumerate(path['midpoint']):records['path_segments'].append(dict(alpha=alpha,mask=mask,index=j,left=float(path['bounds'][j]),right=float(path['bounds'][j+1]),
                        midpoint=float(t),output=float(path['midpoint_output'][j]),derivative=float(derivative[j])))
                    for c,value in enumerate(path['integrated_channel']):records['path_channels'].append(dict(alpha=alpha,mask=mask,channel=c,integrated_contribution=float(value),
                        actual_delta=float(np.diff(path['endpoints'])[0]),endpoint_linear=float(raw['linear_channel'][0,mask,c])))
                    for t,stage,coord,before,after in path['events']:
                        ch,yy,xx=np.unravel_index(int(coord),(32,32,32));records['path_events'].append(dict(alpha=alpha,mask=mask,t=float(t),stage=int(stage)+1,channel=int(ch),y=int(yy),x=int(xx),before=int(before),after=int(after)))
    overview=[]
    for alpha in (1.,.5):
        frames=[r for r in s['rows'] if r['alpha']==alpha]
        for family,method,op in dict.fromkeys((r['family'],r['method'],r['operation']) for r in frames):
            rows=[r for r in frames if r['family']==family and r['method']==method and r['operation']==op];assert len(rows)==6
            overview.append(dict(alpha=alpha,family=family,method=method,operation=op,**{key:float(np.mean([r[key] for r in rows])) for key in ('target_rms','remaining_ratio','error_ratio','signed_removal')}))
    counts={name:table(name,rows) for name,rows in records.items()}
    counts['summary']=table('summary',s['rows']);counts['overview']=table('overview',overview)
    response=teacher.read(RUN/'response_coordinates.json')['rows'];counts['response_coordinates']=table('response_coordinates',response)
    counts['channel_gate_edges']=table('channel_gate_edges',teacher.read(RUN/'channel_gate_edges.json')['rows'])
    threshold=teacher.read(RUN/'threshold_check.json');assert threshold['status']=='PASS'
    counts['threshold_probe']=table('threshold_probe',[{**r,'channels':','.join(map(str,r['channels']))} for r in threshold['rows']])
    figures=make_figures(records)
    result=dict(status='PASS',utc=teacher.now(),overview=overview,primary=records['primary'],cancellation=records['cancellation'],example_single_gates=records['example_single_gates'],
        example_channels=records['example_channels'],threshold=threshold,table_rows=counts,figures=figures,
        interpretation='delta=F(H)-F(H<=1) statewise;source contrasts thereafter;state-specific gate reference;not unique causal share',scope='same36images;half selection;full fixed lists;no new inputs')
    teacher.dump(RUN/'analysis.json',result);print(dict(status='PASS',tables=counts,figures=len(figures)))


def make_figures(records,repair=False):
    assert not (RUN/'delivery_manifest.json').exists();os.environ['MPLCONFIGDIR']=str(RUN/'.plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    out=[];folder=BASE/'reports/figures';folder.mkdir(exist_ok=True)
    def save(fig,suffix):
        for ext in ('png','svg'):
            path=folder/f'dsprites_residual_routes_20261013{suffix}.{ext}';assert repair or not path.exists();fig.savefig(path,dpi=180,bbox_inches='tight');out.append(str(path.relative_to(BASE)).replace('\\','/'))
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.7),layout='constrained')
    for ax,family,title in zip(axes,('channel','gate1','gate2'),('Restore middle channels (8)','Fix first ReLU (128)','Fix second ReLU (128)')):
        rows=[r for r in records['primary'] if r['alpha']==1 and r['family']==family]
        for r in rows:ax.plot([0,1],[r['selected']*100,r['random_mean']*100],marker='o',alpha=.7)
        ax.set_xticks([0,1],['Selected at alpha 0.5','Random mean'],rotation=15,ha='right');ax.set_title(title)
        ax.set_ylabel('Delta reconstruction error (%)' if family=='channel' else 'Remaining delta-contrast RMS (%)');ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,110);ax.axhline(100,color='gray',linestyle='--',linewidth=.8)
    fig.suptitle('Full-strength evaluation on the same panel\nSix frozen teachers; lists and random candidate pools sealed at half strength',fontsize=12);save(fig,'')
    fig,axes=plt.subplots(2,1,figsize=(12.3,7),layout='constrained')
    for ax,mask in zip(axes,(3,15)):
        ch=[r for r in records['example_channels'] if r['alpha']==1];col='linear_mask3' if mask==3 else 'linear_mask15'
        ax.bar(range(32),[r[col] for r in ch],color='tab:blue',alpha=.75);ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(32));ax.set_xlabel('Physical middle channel');ax.set_ylabel('Baseline-gradient contribution')
        row=next(r for r in records['states'] if r['seed']==202609291 and r['alpha']==1 and r['case']==0 and r['mask']==mask)
        ax.set_title(f"Mask {mask}: linear {row['linear']:+.4f}, first-gate {row['first']:+.4f}, second-gate {row['second']:+.4f}, actual delta {row['delta']:+.4f}")
    fig.suptitle('Predefined case: channel direction and signed layer cancellation\nTeacher 202609291, case 0, alpha 1; mask 3 = G0/G1, mask 15 = all four groups',fontsize=12);save(fig,'_channels')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for column,mask in enumerate((3,15)):
        rows=[r for r in records['path_segments'] if r['alpha']==1 and r['mask']==mask]
        bounds=[rows[0]['left']]+[r['right'] for r in rows];g=[r['derivative'] for r in rows]
        y=[rows[0]['output']-g[0]*(rows[0]['right']-rows[0]['left'])/2]+[r['output']+r['derivative']*(r['right']-r['left'])/2 for r in rows]
        axes[0,column].plot(bounds,y,color='tab:blue');axes[0,column].set_title(f'Mask {mask} · actual task output');axes[0,column].set_ylabel('Donor − recipient logit')
        axes[1,column].stairs(g,bounds,color='tab:orange',baseline=None);axes[1,column].set_ylabel('Directional derivative along residual');axes[1,column].axhline(0,color='gray',linewidth=.7)
        for ax in axes[:,column]:ax.set_xlabel('t in H<=1 + t(H − H<=1)');ax.grid(alpha=.15)
    fig.suptitle('Predefined example: complete ReLU intervals and gradient integration\nCurve direction can change when downstream gates change',fontsize=12);save(fig,'_path')
    probe=teacher.read(RUN/'threshold_check.json');ids=probe['chosen_channels'];changes=np.array(probe['single_preactivation_changes'])[ids]
    fig,ax=plt.subplots(figsize=(10.5,4.5),layout='constrained');x=np.arange(len(ids)+1)
    cumulative=probe['baseline']+np.r_[0,np.cumsum(changes)]
    ax.plot(x,cumulative,marker='o',label='Cumulative preactivation (affine sum)',color='tab:blue')
    ax.scatter(x[1:],probe['baseline']+changes,label='Each channel alone (actual forward)',color='tab:orange',zorder=3)
    ax.scatter(x[-2:],cumulative[-2:],marker='D',color='black',s=65,label='Actual group forward: 5 / 6 channels',zorder=4)
    ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(x,['Empty']+[f'+ ch {c}' for c in ids]);ax.set_ylabel('First ReLU preactivation at (4, 15, 13)');ax.grid(axis='y',alpha=.2)
    ax.set_title('Post hoc threshold probe in the predefined case\nNo <=5 residual-channel subset can close this gate; the selected six do');ax.legend(fontsize=9)
    save(fig,'_threshold')
    return out


if __name__=='__main__':report()

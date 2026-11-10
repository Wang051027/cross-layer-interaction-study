"""按模型、形状、大小和角度汇总 dSprites 评价响应，绘制因素分层结果。"""
import os,json,csv
from pathlib import Path
import numpy as np
from dsprites_teacher import BASE,RUN,DATA,SEEDS,read,dump,sha,now

os.environ['MPLCONFIGDIR']=str(RUN/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def csv_write(path,rows):
    """UTF-8 BOM 方便本机 Excel 阅读，保留明确列名和空的未定义值。"""
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(values):
    """以六个模型为描述性重复单位，报告样本标准差及完整逐模型数值。"""
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all()
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),minimum=float(values.min()),
                maximum=float(values.max()),values=values.tolist())


def image_figure():
    """展示真实评价面板与派生平移，避免只用数据集名字代替任务说明。"""
    images=np.load(DATA/'evaluation_images.npy');shifted=np.load(DATA/'translated_images.npy')
    factors=np.load(DATA/'evaluation_factors.npy');labels=np.load(DATA/'evaluation_labels.npy')
    fig,axes=plt.subplots(3,7,figsize=(12,5.6),layout='constrained')
    names=['Square','Ellipse','Heart']
    for c in range(3):
        for s in range(6):
            i=np.flatnonzero((labels==c)&(factors[:,2]==s)&(factors[:,3]==1))[0]
            axes[c,s].imshow(images[i],cmap='gray',vmin=0,vmax=1)
            if c==0:axes[c,s].set_title(f'Scale index {s}',fontsize=10)
        i=np.flatnonzero((labels==c)&(factors[:,2]==3)&(factors[:,3]==1))[0]
        axes[c,6].imshow(shifted[i],cmap='gray',vmin=0,vmax=1)
        if c==0:axes[c,6].set_title('Scale 3, right 4 px',fontsize=10)
        axes[c,0].set_ylabel(names[c],fontsize=11)
        for ax in axes[c]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('dSprites: native 64 x 64, matched shape / scale / orientation / position',fontsize=13)
    path=BASE/'reports/dsprites_panel_20260929.png';fig.savefig(path,dpi=180);plt.close(fig)
    return str(path)


def main():
    """完整结果到齐后生成后处理，数值审计必须先通过。"""
    assert read(RUN/'independent_audit.json')['status']=='PASS'
    summary=read(RUN/'summary.json');rows=summary['methods'];factors=np.load(DATA/'evaluation_factors.npy')
    base=[r for r in rows if r['frame']=='base' and r['alpha']==1 and r['label']==-1]
    def values(mode,method,key,budget=16,alpha=1.,label=-1):
        """按固定种子顺序返回六个模型的指标。"""
        found=[r for r in rows if r['frame']=='base' and r['alpha']==alpha and r['label']==label
               and r['target']==mode and r['method']==method and (r['budget']==budget or budget is None)]
        assert len(found)==6,(mode,method,key,budget,alpha,label,len(found))
        return np.asarray([next(r[key] for r in found if r['seed']==seed) for seed in SEEDS])
    groups=[];comparisons=[];factor_rows=[];stage_rows=[];overlap=[]
    core=lambda mode:[f'{mode}_{scope}_{score}' for score in ('gradient','finite') for scope in ('global','class','input')]
    for mode in ('margin','vector'):
        methods=core(mode)+[f'{mode}_input_full_gradient','margin_global_single','margin_class_single','margin_input_single','all_shortlist','all_reachable']
        for method in methods:
            budget=None if method.startswith('all_') else 16
            # 按模型分别统计可达池大小。
            budget_label='whole structural pool (varies by model)' if method=='all_reachable' else 64 if method=='all_shortlist' else 16
            groups.append(dict(target=mode,method=method,budget=budget_label,
                **{key:stats(values(mode,method,key,budget)) for key in ('restore_ratio','transplant_error_ratio','joint_loss')}))
        random={key:stats(np.mean([values(mode,f'random_{j}',key) for j in range(4)],axis=0))
                for key in ('restore_ratio','transplant_error_ratio','joint_loss')}
        groups.append(dict(target=mode,method='random_mean_of_4',budget=16,**random))
        pairs=[(f'{mode}_class_{score}',f'{mode}_global_{score}') for score in ('gradient','finite')]
        pairs +=[(f'{mode}_input_{score}',f'{mode}_class_{score}') for score in ('gradient','finite')]
        pairs +=[(f'{mode}_input_finite',f'{mode}_input_gradient'),(f'{mode}_input_finite',f'{mode}_input_full_gradient')]
        for a,b in pairs:
            differences={key:values(mode,a,key)-values(mode,b,key) for key in ('restore_ratio','transplant_error_ratio','joint_loss')}
            comparisons.append(dict(target=mode,candidate=a,reference=b,budget=16,
                **{key:stats(value) for key,value in differences.items()},
                wins={key:int((value<0).sum()) for key,value in differences.items()},
                both_wins=int(((differences['restore_ratio']<0)&(differences['transplant_error_ratio']<0)).sum())))
    for seed in SEEDS:
        folder=RUN/'evaluation'/f'seed_{seed}'/'base_alpha_1.0';specs=read(folder/'specs.json')
        with np.load(folder/'responses.npz') as z:
            raw=z['natural'];part=z['contributions'];stencil=z['stencil'];labels=z['labels']
            stage=z['stage_statistics']
        for mode in ('margin','vector'):
            with np.load(folder/f'metrics_{mode}.npz') as z:n,r,t=(z[k] for k in ('natural_mse','restore_mse','transplant_error_mse'))
            def project(x):
                """后处理的线性投影只用于描述分量，主数值已独立审计。"""
                if mode=='vector':return x-x.mean(-1,keepdims=True)
                weights=np.full((len(labels),3),-.5);weights[np.arange(len(labels)),labels]=1
                return np.einsum('n...c,nc->n...',x,weights)[...,None]
            contributions=project(part);denominator=np.sqrt(n.mean())
            effect=project(raw[:,1:]-raw[:,:1]);effect_rms=float(np.sqrt((effect**2).mean()))
            stage_rows.append(dict(seed=seed,target=mode,interaction_rms=float(denominator),effect_rms=effect_rms,
                interaction_over_effect=float(denominator/effect_rms),
                prediction_changed_fraction=float((raw[:,1:].argmax(-1)!=raw[:,0].argmax(-1)[:,None]).mean()),
                projected_stage_ratio=(np.sqrt((contributions**2).mean((0,2,3)))/denominator).tolist(),
                mean_gate_flips=stage[:,:,3].mean(0).tolist(),
                caveat='Stage RMS ratios need not sum to one; signed contributions can cancel.'))
            for c in range(3):
                for axis,column in [('scale',2),('orientation',3)]:
                    for level in np.unique(factors[:,column]):
                        mask=(labels==c)&(factors[:,column]==level);denom=float(np.sqrt(n[mask].mean()))
                        for method in core(mode):
                            j=next(j for j,s in enumerate(specs) if s['method']==method and s['budget']==16)
                            restore=float(np.sqrt(r[mask,j].mean()));error=float(np.sqrt(t[mask,j].mean()))
                            factor_rows.append(dict(seed=seed,target=mode,label=c,axis=axis,level=int(level),
                                inputs=int(mask.sum()),method=method,interaction_rms=denom,restore_rms=restore,
                                transplant_error_rms=error,restore_ratio=restore/denom if denom>1e-9 else None,
                                transplant_error_ratio=error/denom if denom>1e-9 else None))
        with np.load(RUN/'input_selection'/f'seed_{seed}'/'selection.npz') as z:selected=z['selections']
        for mode in ('margin','vector'):
            for score in ('gradient','finite'):
                j=next(j for j,s in enumerate(specs) if s==dict(method=f'{mode}_input_{score}',budget=16))
                k=next(j for j,s in enumerate(specs) if s==dict(method=f'{mode}_class_{score}',budget=16))
                jac=[]
                for i in range(144):
                    count=len(set(selected[i,j,:16])&set(selected[i,k,:16]));jac.append(count/(32-count))
                overlap.append(dict(seed=seed,target=mode,score=score,input_vs_class_jaccard=float(np.mean(jac))))
    result=dict(utc=now(),posthoc_descriptive=True,aggregation='within teacher first; six teacher mean and sample SD',
        primary=groups,paired_comparisons=comparisons,stage_diagnostics=stage_rows,selection_overlap=overlap,
        source_sha256=sha(Path(__file__)),all_factor_cells_kept=True)
    dump(RUN/'descriptive_analysis.json',result)
    csv_write(RUN/'method_summary.csv',rows);csv_write(RUN/'factor_strata.csv',factor_rows)
    # 配对点给出六个模型各自的得分。
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    labels_plot=['Global\ngradient','Class\ngradient','Input\ngradient','Global\nfinite','Class\nfinite','Input\nfinite']
    colors=['#6F8090','#4C86AE','#236F93','#B9A18B','#C98753','#A65322']
    for row,mode in enumerate(('margin','vector')):
        for col,key in enumerate(('restore_ratio','transplant_error_ratio')):
            ax=axes[row,col]
            for x,method in enumerate(core(mode)):
                v=values(mode,method,key)*100
                ax.scatter(x+np.linspace(-.12,.12,6),v,s=21,color=colors[x],alpha=.8)
                ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt='_',color='black',capsize=4,markersize=15)
            ax.axhline(100,color='#AAAAAA',ls=':',lw=1);ax.set_xticks(range(6),labels_plot,fontsize=8)
            ax.set_ylabel('Relative RMS (%) — lower is better');ax.grid(axis='y',alpha=.18)
            ax.set_title(f'{mode}: '+('remaining after restoration' if col==0 else 'transplant reconstruction error'))
        ax=axes[row,2]
        if row==0:
            for i,target in enumerate(('margin','vector')):
                v=[r['interaction_rms'] for r in stage_rows if r['target']==target]
                ax.scatter(np.arange(6)+i*.16,v,label=target,s=34)
            ax.set_xticks(np.arange(6)+.08,[str(s)[-3:] for s in SEEDS]);ax.set_yscale('log')
            ax.set_title('Natural interaction magnitude');ax.set_ylabel('RMS in logit units');ax.set_xlabel('Model seed suffix');ax.legend()
        else:
            for i,frame in enumerate(('translation_absolute','translation_aligned')):
                v=[next(r['natural_relative_difference'] for r in summary['translation']
                        if r['seed']==seed and r['frame']==frame and r['target']=='margin') for seed in SEEDS]
                ax.scatter(i+np.linspace(-.12,.12,6),np.maximum(v,1e-16),s=34)
            ax.set_xticks([0,1],['Absolute coordinates','Aligned coordinates']);ax.set_yscale('log')
            ax.set_title('Right 4 px: paired interaction change');ax.set_ylabel('RMS difference / original RMS')
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('dSprites transfer: 6 models, 144 factor records, k=16, evaluation alpha=1',fontsize=15)
    fig.savefig(BASE/'reports/dsprites_results_20260929.png',dpi=190);plt.close(fig)
    image_figure()
    print(json.dumps(dict(status='PASS',primary_rows=len(groups),comparisons=len(comparisons),factor_rows=len(factor_rows)),ensure_ascii=False))


if __name__=='__main__':main()

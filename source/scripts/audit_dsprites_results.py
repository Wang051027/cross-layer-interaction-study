"""从 dSprites 的逐例输出重算中介评分。用差分矩阵和类别投影矩阵核对交互量。"""
import itertools
import json
from pathlib import Path
import numpy as np
from dsprites_teacher import BASE,RUN,DATA,SEEDS,sha,read,dump,now,verify_old
from run_dsprites_mechanism import verify_all


def arrays(path):
    """一次性读入压缩数组，避免在内循环重复解压。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def matrix():
    """直接从四个二元干预构造 24×16 差分矩阵及元数据。"""
    rows=[];meta=[];stencils=[]
    for a,b in itertools.combinations(range(4),2):
        for t in range(16):
            if t&(1<<a) or t&(1<<b):continue
            ids=[t+(1<<a)+(1<<b),t+(1<<a),t+(1<<b),t]
            row=np.zeros(16);row[ids]=[1,-1,-1,1]
            rows.append(row);meta.append([a,b,t]);stencils.append(ids)
    return np.array(rows),np.array(meta),np.array(stencils)


def interactions(raw,labels,mode,contrast):
    """先对原始各类 logit 作矩阵差分，再投影，与主实现的运算顺序不同。"""
    values=np.einsum('sm,n...mc->n...sc',contrast,raw)
    classes=raw.shape[-1]
    if mode=='vector':return values@(np.eye(classes)-np.ones((classes,classes))/classes)
    weights=np.full((len(labels),classes),-1/(classes-1))
    weights[np.arange(len(labels)),labels]=1
    return np.einsum('n...c,nc->n...',values,weights)[...,None]


def check(name,actual,expected,maxima,tolerance=1e-10):
    """记录各项数值核验的最大绝对差。"""
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    maxima[name]=max(maxima.get(name,0.),error)
    assert error<tolerance,(name,error,tolerance)


def order(scores,indices):
    """独立重建稳定排序，索引相等的分数按冻结候选顺序处理。"""
    return sorted(map(int,indices),key=lambda k:-float(scores[k]))


def fixed_lists(a,seed):
    """从保存的开发数据重建全局、按类与随机名单。"""
    labels=a['labels'];short=a['shortlist'];feature=a['features'];finite=a['finite_scores']
    result={}
    for ti,mode in enumerate(('margin','vector')):
        fi=0 if mode=='margin' else 2
        for scope in ('global','class'):
            groups=[np.ones(len(labels),bool)] if scope=='global' else [labels==c for c in range(3)]
            g=[order(feature[ix,fi].mean(0),short) for ix in groups]
            f=[[int(short[k]) for k in order(finite[ix,ti].mean(0),range(64))] for ix in groups]
            result[f'{mode}_{scope}_gradient']=g[0] if scope=='global' else g
            result[f'{mode}_{scope}_finite']=f[0] if scope=='global' else f
    result['margin_global_single']=order(feature[:,1].mean(0),short)
    result['margin_class_single']=[order(feature[labels==c,1].mean(0),short) for c in range(3)]
    for r in range(4):result[f'random_{r}']=np.random.default_rng(seed+r+2026092900).permutation(short).tolist()
    return result


def audit():
    """核查每个教师/设置，生成独立审计记录；失败立即停止。"""
    verify_all()
    maxima={};contrast,meta,stencil=matrix();summary=read(RUN/'summary.json')
    # 派生平移图与其官方原图共用因素标签。
    base=np.load(DATA/'evaluation_images.npy');moved=np.load(DATA/'translated_images.npy')
    assert np.array_equal(base[:,:,:-4],moved[:,:,4:])
    assert not moved[:,:,:4].any() and np.array_equal(base.sum((1,2)),moved.sum((1,2)))
    factor=np.load(DATA/'evaluation_factors.npy')
    assert len(np.unique(factor[:,2:],axis=0))==48
    cases=0;selection_rows=0
    for seed in SEEDS:
        dev=arrays(RUN/'development'/f'seed_{seed}'/'discovery.npz')
        selected=arrays(RUN/'input_selection'/f'seed_{seed}'/'selection.npz')
        stored=read(RUN/'development'/f'seed_{seed}'/'fixed_orders.json')
        assert stored==fixed_lists(dev,seed)
        assert np.array_equal(np.bincount(dev['labels']),[48,48,48])
        means=dev['features'].mean(0)
        shortlist=set(order(means[0],range(means.shape[1]))[:24])
        shortlist.update(order(means[1],range(means.shape[1]))[:16])
        shortlist.update(order(means[2],range(means.shape[1]))[:24])
        for node in order(means[0],range(means.shape[1])):
            if len(shortlist)==64:break
            shortlist.add(node)
        assert sorted(shortlist)==dev['shortlist'].tolist()==selected['shortlist'].tolist()
        for stage,a in [('development',dev),('input_selection',selected)]:
            assert np.array_equal(a['stencil'],stencil) and np.array_equal(a['pair_metadata'],meta)
            assert np.isfinite(a['features']).all() and a['features'].shape[1]==4
            for ti,mode in enumerate(('margin','vector')):
                natural=interactions(a['natural'],a['labels'],mode,contrast)
                single=interactions(a['singleton_logits'],a['labels'],mode,contrast)
                expected=np.mean(natural**2,(-2,-1))[:,None]-.5*(
                    np.mean(single[:,:,0]**2,(-2,-1))+np.mean((single[:,:,1]-natural[:,None])**2,(-2,-1)))
                check('finite_'+stage+'_'+mode,a['finite_scores'][:,ti],expected,maxima)
        # 核对 144×44 份名单的有效坐标，跳过尾部填充。
        specs=read(RUN/'input_selection'/f'seed_{seed}'/'specs.json')
        short=selected['shortlist'];pool=selected['pool']
        for i,label in enumerate(selected['labels']):
            orders={key:(value[int(label)] if '_class_' in key else value) for key,value in stored.items()}
            for ti,mode in enumerate(('margin','vector')):
                fi=0 if mode=='margin' else 2
                orders[f'{mode}_input_gradient']=order(selected['features'][i,fi],short)
                orders[f'{mode}_input_full_gradient']=order(selected['features'][i,fi],range(len(pool)))
                orders[f'{mode}_input_finite']=[int(short[k]) for k in order(selected['finite_scores'][i,ti],range(64))]
            orders['margin_input_single']=order(selected['features'][i,1],short)
            for j,spec in enumerate(specs):
                key,k=spec['method'],spec['budget']
                indices=short if key=='all_shortlist' else np.arange(len(pool)) if key=='all_reachable' else orders[key][:k]
                assert np.array_equal(selected['selections'][i,j,:k],pool[indices])
                assert np.all(selected['selections'][i,j,k:]==-1)
                selection_rows+=1
        for frame,alpha in [('base',1.),('base',.5),('translation_absolute',1.),('translation_aligned',1.)]:
            folder=RUN/'evaluation'/f'seed_{seed}'/f'{frame}_alpha_{alpha}'
            raw=arrays(folder/'responses.npz');specs=read(folder/'specs.json')
            natural=np.einsum('sm,nmc->nsc',contrast,raw['natural'])
            check('sum_layer_contributions',raw['contributions'].sum(1),natural,maxima,1e-9)
            assert raw['stage_statistics'].shape==(144,4,5)
            for mode in ('margin','vector'):
                ni=interactions(raw['natural'],raw['labels'],mode,contrast)
                pi=interactions(raw['patched'],raw['labels'],mode,contrast)
                n=np.mean(ni**2,(-2,-1));r=np.mean(pi[:,:,0]**2,(-2,-1));t=np.mean((pi[:,:,1]-ni[:,None])**2,(-2,-1))
                saved=arrays(folder/f'metrics_{mode}.npz')
                for key,value in [('natural_mse',n),('restore_mse',r),('transplant_error_mse',t)]:
                    check('metrics_'+key+'_'+mode,saved[key],value,maxima)
                # 独立重算六教师分组汇总中的绝对量；比率从同一绝对量构造。
                rows=[row for row in summary['methods'] if row['seed']==seed and row['frame']==frame
                      and row['alpha']==alpha and row['target']==mode]
                assert len(rows)==len(specs)*4
                for row in rows:
                    label=row['label'];mask=np.ones(144,bool) if label==-1 else raw['labels']==label
                    j=next(j for j,spec in enumerate(specs) if spec['method']==row['method'] and spec['budget']==row['budget'])
                    denominator=float(np.sqrt(n[mask].mean()))
                    restore=float(np.sqrt(r[mask,j].mean()));transplant=float(np.sqrt(t[mask,j].mean()))
                    for key,value in [('interaction_rms',denominator),('restore_rms',restore),('transplant_error_rms',transplant)]:
                        check('summary_'+key,row[key],value,maxima)
                    if denominator>1e-9:
                        for key,value in [('restore_ratio',restore/denominator),('transplant_error_ratio',transplant/denominator)]:
                            check('summary_'+key,row[key],value,maxima,1e-7)
                    else:assert row['restore_ratio'] is None and row['transplant_error_ratio'] is None
            cases+=1
        print(json.dumps(dict(seed=seed,status='PASS'),ensure_ascii=False),flush=True)
    assert cases==24 and selection_rows==6*144*44
    result=dict(status='PASS',utc=now(),source_sha256=sha(Path(__file__)),cases=cases,selection_rows=selection_rows,
        max_abs_errors=maxima,legacy_manifest_hashes=verify_old(),
        checks=['dense 24x16 contrasts','independent class projection','all singleton scores','global/class/input ranks',
                'all recorded selections','all per-input metrics','teacher/class summary values','additive layer decomposition',
                'strict pixel translation without clipping','freeze ordering and original artifact hashes'],
        limitation='Audit confirms computations and preservation, not scientific novelty or uniqueness of a causal explanation.')
    dump(RUN/'independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':audit()

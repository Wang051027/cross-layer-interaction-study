"""dSprites迁移评估的阶段入口，先锁定规则和半强度名单，再评价。

主面板固定位置；平移面板复用主面板名单，分别采用绝对坐标与同步平移坐标。
旧模型用于CPU数值一致性核验，新机制结果全部来自独立训练的dSprites模型。
"""
import argparse,shutil,time
import numpy as np
import torch
from dsprites_teacher import (BASE,RUN,DATA,SEEDS,read,dump,sha,now,event,configure,
    verify_protocol,verify_old,data,load_teacher)
import dsprites_mechanism as m
from test_dsprites_mechanism import known_answers,fashion_portability,check_model

CODE=['scripts/dsprites_mechanism.py','scripts/test_dsprites_mechanism.py','scripts/run_dsprites_mechanism.py']
TRANSLATION_METHODS=('margin_global_gradient','margin_class_gradient','margin_input_gradient',
                     'margin_input_finite','random_0','all_reachable')


def clean(value):
    """将数组、标量与近零分母变为标准JSON可审阅值。"""
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save(path,value):
    """明确将无定义比率写null，避免误写成零或无穷大。"""
    dump(path,clean(value))


def initialize():
    """模型完整强度评价之前锁定所有选择和比较规则，允许训练仍在进行。"""
    verify_protocol()
    assert not (RUN/'mechanism_protocol.json').exists()
    for name in CODE:
        destination=RUN/'mechanism_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,destination)
    protocol=dict(utc=now(),training_protocol_sha256=sha(RUN/'protocol.json'),
        sources={name:sha(BASE/name) for name in CODE},targets=m.TARGETS,budgets=m.BUDGETS,
        methods=m.METHODS,selection_alpha=.5,primary_alpha=1.,additional_alpha=.5,
        primary_budget=16,shortlist='training margin interaction top24 + margin single top16 + vector interaction top24; fill margin to64',
        source='four highest mean development activations in central3x3 of block1 output; no test retuning',
        scopes=['global','class','input'],class_development_count=48,
        translation_methods=TRANSLATION_METHODS,translation_budget=16,translation_alpha=1.,
        translation_rule='reuse base selections; absolute coordinates or source dx4/middle dx2 aligned coordinates',
        frames=['base_alpha1','base_alpha0.5','translation_absolute_alpha1','translation_aligned_alpha1'],
        evaluation_contexts=48,images=144,targets_not_independent=True,
        note='Prior protocol text list of fixed sizes refers to old6272 middle scalars; this implementation derives dimensions from tensors',
        tolerance=1e-9,statistics='descriptive teacher-first; class/factor stratified; keep inactive records and absolute magnitudes')
    save(RUN/'mechanism_protocol.json',protocol)
    (RUN/'mechanism_protocol.sha256').write_text(sha(RUN/'mechanism_protocol.json')+'\n',encoding='ascii')
    save(RUN/'known_answer_controls.json',known_answers())
    save(RUN/'fashion_portability.json',fashion_portability())
    event('mechanism_protocol_locked',sha256=sha(RUN/'mechanism_protocol.json'))


def verify_mechanism(require_models=True):
    """每个阶段核验源码、数据与旧记录；需要模型时核验全部六个检查点。"""
    verify_protocol()
    assert sha(RUN/'mechanism_protocol.json')==(RUN/'mechanism_protocol.sha256').read_text().strip()
    p=read(RUN/'mechanism_protocol.json')
    for name,expected in p['sources'].items():
        assert sha(BASE/name)==expected and sha(RUN/'mechanism_source'/name)==expected,name
    if require_models:
        assert sha(RUN/'model_freeze.json')==(RUN/'model_freeze.sha256').read_text().strip()
        for path,expected in read(RUN/'model_freeze.json')['checkpoints'].items():assert sha(RUN/path)==expected,path


def complete(folder,files,seconds):
    """原始数组写完才提交完成标志，保留阶段耗时和每个文件哈希。"""
    save(folder/'completion.json',dict(utc=now(),seconds=seconds,files={name:sha(folder/name) for name in files}))


def verify_complete(folder):
    """已完成阶段只读复用；任何缺损停止而不覆盖。"""
    saved=read(folder/'completion.json')
    for name,expected in saved['files'].items():assert sha(folder/name)==expected,(folder,name)
    return saved


def traced_image(model,image,nodes,bits,alpha,device):
    """复用旧完整残差前向，所有分支自然传播源干预。"""
    source=m.source_activation(model,image[None].to(device))
    return m.trace(model,m.changed_sources(source,nodes,bits,alpha))


def develop(device):
    """开发面板估计公共候选、全局和按类别排序；不读取评价图的模型输出。"""
    verify_mechanism();assert not (RUN/'fixed_freeze.json').exists()
    images,label_tensor=data('development');labels=label_tensor.numpy()
    validation,val_labels=data('validation')
    bits,stencil,metadata=m.design();bits=torch.as_tensor(bits,dtype=torch.float64,device=device)
    for seed in SEEDS:
        folder=RUN/'development'/f'seed_{seed}'
        if folder.exists():verify_complete(folder);continue
        folder.mkdir(parents=True);started=time.perf_counter()
        model=load_teacher(seed,device)
        nodes=m.select_source(model,images,device)
        source=m.source_activation(model,images[:1].to(device));pool=m.reachable(model,nodes,source)
        checks=check_model(model,validation[0],int(val_labels[0]),nodes,device)
        feature=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            tr=traced_image(model,image,nodes,bits,.5,device)
            feature.append(m.features(model,tr['h'][1],int(label),stencil)[:,pool])
        feature=np.stack(feature);shortlist=m.make_shortlist(feature.mean(0))
        natural=[];singletons=[];finite=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            tr=traced_image(model,image,nodes,bits,.5,device)
            raw=m.patch(model,tr['h'][1],[[int(x)] for x in pool[shortlist]]).cpu().numpy()
            output=tr['output'].cpu().numpy()
            natural.append(output);singletons.append(raw);finite.append(m.finite_scores(raw,output,int(label),stencil))
            if (i+1)%24==0:event('development_progress',seed=seed,inputs=i+1,total=len(images))
        finite=np.stack(finite)
        orders=m.fixed_orders(feature,finite,labels,shortlist,seed)
        np.savez_compressed(folder/'discovery.npz',features=feature,finite_scores=finite,natural=np.stack(natural),
            singleton_logits=np.stack(singletons),source_nodes=nodes,pool=pool,shortlist=shortlist,labels=labels,
            stencil=stencil,pair_metadata=metadata)
        save(folder/'fixed_orders.json',orders);save(folder/'checks.json',checks)
        complete(folder,['discovery.npz','fixed_orders.json','checks.json'],time.perf_counter()-started)
        event('development_complete',seed=seed,source_nodes=nodes.tolist(),pool_size=len(pool),seconds=time.perf_counter()-started)
        del model


def freeze(name,stage):
    """六教师统一冻结开发或逐图名单，严禁选完一个就边评边改。"""
    verify_mechanism();assert not (RUN/f'{name}.json').exists()
    paths={}
    for seed in SEEDS:
        folder=RUN/stage/f'seed_{seed}';completion=verify_complete(folder)
        for name_file in [*completion['files'],'completion.json']:
            paths[(folder/name_file).relative_to(RUN).as_posix()]=sha(folder/name_file)
    save(RUN/f'{name}.json',dict(utc=now(),mechanism_sha256=sha(RUN/'mechanism_protocol.json'),files=paths))
    (RUN/f'{name}.sha256').write_text(sha(RUN/f'{name}.json')+'\n',encoding='ascii')
    event(name,files=len(paths))


def verify_freeze(name):
    """校验冻结时的算法绑定和名单原字节。"""
    verify_mechanism()
    assert sha(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
    record=read(RUN/f'{name}.json')
    assert record['mechanism_sha256']==sha(RUN/'mechanism_protocol.json')
    for path,expected in record['files'].items():assert sha(RUN/path)==expected,path


def select(device):
    """仅在原始中心评价图的半强度下生成逐图中介；平移实验复用这些名单。"""
    verify_freeze('fixed_freeze');assert not (RUN/'adaptive_freeze.json').exists()
    assert not (RUN/'evaluation_opened.json').exists()
    if not (RUN/'selection_opened.json').exists():save(RUN/'selection_opened.json',dict(utc=now(),alpha=.5))
    images,label_tensor=data('evaluation');labels=label_tensor.numpy()
    bits,stencil,metadata=m.design();bits=torch.as_tensor(bits,dtype=torch.float64,device=device)
    for seed in SEEDS:
        folder=RUN/'input_selection'/f'seed_{seed}'
        if folder.exists():verify_complete(folder);continue
        folder.mkdir(parents=True);started=time.perf_counter()
        dev=RUN/'development'/f'seed_{seed}'
        with np.load(dev/'discovery.npz') as a:
            nodes,pool,shortlist=a['source_nodes'].copy(),a['pool'].copy(),a['shortlist'].copy()
        fixed=read(dev/'fixed_orders.json');model=load_teacher(seed,device)
        natural=[];singletons=[];feature=[];finite=[];selections=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            tr=traced_image(model,image,nodes,bits,.5,device)
            raw=m.patch(model,tr['h'][1],[[int(x)] for x in pool[shortlist]]).cpu().numpy()
            output=tr['output'].cpu().numpy()
            scores=m.finite_scores(raw,output,int(label),stencil)
            f=m.features(model,tr['h'][1],int(label),stencil)[:,pool]
            chosen=m.select_input(f,scores,int(label),shortlist,pool,fixed)
            selections.append(m.encode(chosen,len(pool)))
            natural.append(output);singletons.append(raw);finite.append(scores);feature.append(f)
            if (i+1)%24==0:event('selection_progress',seed=seed,inputs=i+1,total=len(images))
        np.savez_compressed(folder/'selection.npz',natural=np.stack(natural),singleton_logits=np.stack(singletons),
            finite_scores=np.stack(finite),features=np.stack(feature),selections=np.stack(selections),
            labels=labels,source_nodes=nodes,pool=pool,shortlist=shortlist,stencil=stencil,pair_metadata=metadata)
        save(folder/'specs.json',m.specifications(len(pool)))
        complete(folder,['selection.npz','specs.json'],time.perf_counter()-started)
        event('selection_complete',seed=seed,seconds=time.perf_counter()-started)
        del model


def evaluate(device):
    """全强度与平移评价；每种设置保存原始输出、逐层投影和数值检查。"""
    verify_freeze('fixed_freeze');verify_freeze('adaptive_freeze')
    if not (RUN/'evaluation_opened.json').exists():save(RUN/'evaluation_opened.json',dict(utc=now()))
    images,label_tensor=data('evaluation');labels=label_tensor.numpy()
    moved,_=data('translated')
    bits,stencil,metadata=m.design();bits=torch.as_tensor(bits,dtype=torch.float64,device=device)
    contexts=[('base',1.),('base',.5),('translation_absolute',1.),('translation_aligned',1.)]
    for seed in SEEDS:
        chosen_folder=RUN/'input_selection'/f'seed_{seed}'
        with np.load(chosen_folder/'selection.npz') as a:
            nodes=a['source_nodes'].copy();selected=a['selections'].copy();half=a['natural'].copy()
        specs=read(chosen_folder/'specs.json');model=load_teacher(seed,device)
        for frame,alpha in contexts:
            folder=RUN/'evaluation'/f'seed_{seed}'/f'{frame}_alpha_{alpha}'
            if folder.exists():verify_complete(folder);continue
            folder.mkdir(parents=True);started=time.perf_counter()
            indices=list(range(len(specs))) if frame=='base' else [j for j,s in enumerate(specs)
                if s['method'] in TRANSLATION_METHODS and (s['budget']==16 or s['method']=='all_reachable')]
            current_specs=[specs[j] for j in indices]
            xs=images if frame=='base' else moved
            source_nodes=m.shifted_nodes(nodes,(16,64,64),4) if frame=='translation_aligned' else nodes
            natural=[];patched=[];parts=[];stage=[];maxima={}
            for i,image in enumerate(xs):
                tr=traced_image(model,image,source_nodes,bits,alpha,device)
                selections=[selected[i,j,:specs[j]['budget']] for j in indices]
                if frame=='translation_aligned':selections=[m.shifted_nodes(n,(32,32,32),2) for n in selections]
                output=m.patch(model,tr['h'][1],selections).cpu().numpy();original=tr['output'].cpu().numpy()
                contribution,stages,errors=m.decomposition(model,tr,stencil)
                # 平移绝对坐标仍用原可达池；对齐坐标则同步移动整个支持集。
                errors['all_restore']=float(np.max(np.abs(output[-1,0]-original[:1])))
                errors['all_transplant']=float(np.max(np.abs(output[-1,1]-original)))
                if frame=='base' and alpha==.5:errors['half_replay']=float(np.max(np.abs(original-half[i])))
                assert max(errors.values())<1e-9,errors
                for key,value in errors.items():maxima[key]=max(maxima.get(key,0.),float(value))
                natural.append(original);patched.append(output);parts.append(contribution.cpu().numpy());stage.append(stages)
                if (i+1)%24==0:event('evaluation_progress',seed=seed,frame=frame,alpha=alpha,inputs=i+1,total=len(xs))
            np.savez_compressed(folder/'responses.npz',natural=np.stack(natural),patched=np.stack(patched),
                contributions=np.stack(parts),stage_statistics=np.stack(stage),labels=labels,stencil=stencil,pair_metadata=metadata,
                source_nodes=source_nodes,parent_spec_indices=np.asarray(indices))
            save(folder/'specs.json',current_specs);save(folder/'checks.json',dict(status='PASS',errors=maxima))
            complete(folder,['responses.npz','specs.json','checks.json'],time.perf_counter()-started)
            event('evaluation_case_complete',seed=seed,frame=frame,alpha=alpha,seconds=time.perf_counter()-started,max_error=max(maxima.values()))
        del model


def summarize():
    """按模型、目标、类别汇总；平移比较使用配对绝对交互差而不猜测机制。"""
    verify_freeze('adaptive_freeze');assert not (RUN/'summary.json').exists()
    rows=[];baseline=[];translation=[]
    factors=np.load(DATA/'evaluation_factors.npy')
    for seed in SEEDS:
        base_arrays={}
        for frame,alpha in [('base',1.),('base',.5),('translation_absolute',1.),('translation_aligned',1.)]:
            folder=RUN/'evaluation'/f'seed_{seed}'/f'{frame}_alpha_{alpha}'
            verify_complete(folder);specs=read(folder/'specs.json')
            with np.load(folder/'responses.npz') as a:
                natural,patched,labels,stencil=(a[k].copy() for k in ('natural','patched','labels','stencil'))
            if frame=='base' and alpha==1.:
                base_arrays=dict(natural=natural,patched=patched,specs=specs)
            for mode in m.TARGETS:
                n,r,t=m.metric_arrays(natural,patched,labels,stencil,mode)
                # 用紧凑数值数组保留全部逐输入量；无需上百万行重复列名的CSV。
                metric_path=folder/f'metrics_{mode}.npz'
                np.savez_compressed(metric_path,natural_mse=n,restore_mse=r,transplant_error_mse=t,labels=labels,factors=factors)
                for label in (-1,0,1,2):
                    mask=np.ones(len(labels),dtype=bool) if label==-1 else labels==label
                    denominator=float(np.sqrt(n[mask].mean()))
                    baseline.append(dict(seed=seed,frame=frame,alpha=alpha,target=mode,label=label,
                        interaction_rms=denominator,active_inputs=int((np.sqrt(n[mask])>1e-9).sum()),inputs=int(mask.sum()),
                        accuracy=float((natural[mask,0].argmax(1)==labels[mask]).mean())))
                    for j,spec in enumerate(specs):
                        restore=float(np.sqrt(r[mask,j].mean()));error=float(np.sqrt(t[mask,j].mean()))
                        rr=restore/denominator if denominator>1e-9 else np.nan
                        tt=error/denominator if denominator>1e-9 else np.nan
                        rows.append(dict(seed=seed,frame=frame,alpha=alpha,target=mode,label=label,**spec,
                            interaction_rms=denominator,restore_rms=restore,transplant_error_rms=error,
                            restore_ratio=rr,transplant_error_ratio=tt,joint_loss=.5*(rr*rr+tt*tt)))
                if frame!='base':
                    original=np.stack([m.contrasts(m.project(x,int(y),mode),stencil) for x,y in zip(base_arrays['natural'],labels)])
                    changed=np.stack([m.contrasts(m.project(x,int(y),mode),stencil) for x,y in zip(natural,labels)])
                    difference=float(np.sqrt(np.mean((changed-original)**2)))
                    denominator=float(np.sqrt(np.mean(original**2)))
                    for j,spec in enumerate(specs):
                        k=next(k for k,s in enumerate(base_arrays['specs']) if s==spec)
                        base_patch=np.stack([m.contrasts(m.project(x,int(y),mode),stencil) for x,y in zip(base_arrays['patched'][:,k],labels)])
                        moved_patch=np.stack([m.contrasts(m.project(x,int(y),mode),stencil) for x,y in zip(patched[:,j],labels)])
                        translation.append(dict(seed=seed,frame=frame,target=mode,**spec,natural_interaction_rms=denominator,
                            natural_pair_difference_rms=difference,natural_relative_difference=difference/denominator if denominator>1e-9 else np.nan,
                            patch_pair_difference_rms=float(np.sqrt(np.mean((moved_patch-base_patch)**2)))))
        event('summary_teacher_complete',seed=seed)
    save(RUN/'summary.json',dict(status='PASS',utc=now(),methods=rows,baseline=baseline,translation=translation,
        classification_gate=read(RUN/'model_freeze.json')['quality_gate'],teachers=6,original_panel_images=144,
        derived_translation_images=144,factor_contexts=48,old_statistics_not_transferred=True))
    event('summary_complete',method_rows=len(rows),baseline_rows=len(baseline),translation_rows=len(translation))


def verify_all():
    """完成后的只读哈希与阶段次序检查；独立数值回读由另一个脚本实现。"""
    verify_freeze('fixed_freeze');verify_freeze('adaptive_freeze')
    assert read(RUN/'fixed_freeze.json')['utc']<read(RUN/'selection_opened.json')['utc']
    assert read(RUN/'adaptive_freeze.json')['utc']<read(RUN/'evaluation_opened.json')['utc']
    for seed in SEEDS:
        for stage in ('development','input_selection'):verify_complete(RUN/stage/f'seed_{seed}')
        for frame,alpha in [('base',1.),('base',.5),('translation_absolute',1.),('translation_aligned',1.)]:
            verify_complete(RUN/'evaluation'/f'seed_{seed}'/f'{frame}_alpha_{alpha}')
    assert read(RUN/'summary.json')['status']=='PASS'
    print(clean(dict(status='PASS',old_manifests=verify_old(),models=6,cases=24)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','develop','freeze','select','seal','evaluate','summarize','verify'])
    args=parser.parse_args();device=configure()
    if args.stage=='init':initialize()
    elif args.stage=='develop':develop(device)
    elif args.stage=='freeze':
        assert not (RUN/'selection_opened.json').exists();freeze('fixed_freeze','development')
    elif args.stage=='select':select(device)
    elif args.stage=='seal':
        assert not (RUN/'evaluation_opened.json').exists();verify_freeze('fixed_freeze');freeze('adaptive_freeze','input_selection')
    elif args.stage=='evaluate':evaluate(device)
    elif args.stage=='summarize':summarize()
    else:verify_all()

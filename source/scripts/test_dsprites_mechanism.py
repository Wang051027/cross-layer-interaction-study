"""迁移核验：旧十类计算的一致性、三类真实模型钩子及动态维度已知答案。"""
import json
import numpy as np
import torch
import cross_layer_mechanism as old
import adaptive_mediator as old_select
import dsprites_mechanism as new


def known_answers():
    """核查投影定义、标签无关性、完美修补评分与坐标平移不串通道。"""
    bits,stencil,_=old.design()
    for classes in (3,10):
        logits=np.zeros((16,classes));logits[:,0]=np.maximum(1.5-bits[:,0]-bits[:,1],0)
        assert np.array_equal(new.project(logits,0,'margin')[:,0],logits[:,0])
        assert np.array_equal(new.project(logits,0,'vector'),new.project(logits,classes-1,'vector'))
        baseline=np.broadcast_to(logits[:1],logits.shape)
        singles=np.stack([np.stack([baseline,logits]),np.stack([logits,baseline])])
        scores=new.finite_scores(singles,logits,0,stencil)
        # 无效修补的理论评分为零；中心化投影可引入约 1e-19 的舍入误差。
        assert np.all(scores[:,0]>0) and np.all(np.abs(scores[:,1])<1e-15)
    nodes=np.array([2*64*64+31*64+31,9*64*64+32*64+33])
    shifted=new.shifted_nodes(nodes,(16,64,64),4)
    assert np.array_equal(shifted,nodes+4)
    assert np.array_equal(new.shifted_nodes(shifted,(16,64,64),-4),nodes)
    return dict(status='PASS',controls=['3/10-class margin','label-free centered vector',
        'perfect/ineffective finite mediation','coordinate translation preserves channel'])


def check_model(model,image,label,nodes,device):
    """动态3类/64×64版本逐项与原model.forward钩子比较，不复用修补实现。"""
    bits,stencil,_=old.design();bits=torch.as_tensor(bits,dtype=torch.float64,device=device)
    image=image[None].to(device)
    maxima={}
    def record(name,value):
        """记录各核验项最大绝对差，完整保留而不只输出PASS。"""
        maxima[name]=max(maxima.get(name,0.),float(value))
    source=old.source_activation(model,image)
    for alpha in (.5,1.):
        traced=old.trace(model,old.changed_sources(source,nodes,bits,alpha))
        original=old.original_forward(model,image,nodes,bits,alpha)
        record('trace_hook',(traced['output']-original).abs().max())
        _,_,errors=old.decomposition(model,traced,stencil)
        for key,value in errors.items():record(key,value)
        pool=new.reachable(model,nodes,source)
        subset=pool[:16]
        patched=new.patch(model,traced['h'][1],[[],pool,subset])
        record('empty_restore',(patched[0,0]-original).abs().max())
        record('empty_transplant',(patched[0,1]-original[:1]).abs().max())
        record('all_restore',(patched[1,0]-original[:1]).abs().max())
        record('all_transplant',(patched[1,1]-original).abs().max())
        for operation,mode in enumerate(('restore','transplant')):
            reference=old.original_forward(model,image,nodes,bits,alpha,mode,subset)
            record('patch_hook_'+mode,(reference-patched[2,operation]).abs().max())
        feature=new.features(model,traced['h'][1],int(label),stencil)
        assert feature.shape==(4,traced['h'][1][0].numel()) and np.isfinite(feature).all()
    assert max(maxima.values())<1e-9,maxima
    return dict(status='PASS',errors=maxima)


def fashion_portability():
    """只取原Fashion验证图，在同一模型上比较新旧10类公式，不跑新的测试。"""
    from run_cross_layer_mechanism import BASE,OLD,DATA,data
    from residual_teacher import load_teacher
    torch.set_num_threads(1)
    with np.load(BASE/'runs/cross_layer_20260928/input_ids.npz') as a:ids=a['validation'][:1]
    images,labels=data('train',ids)
    model=load_teacher(OLD/'seed_202609071',device='cpu')
    setup=json.loads((BASE/'runs/cross_layer_20260928/development/seed_202609071/selections.json').read_text(encoding='utf-8'))
    nodes=setup['nodes'];bits,stencil,_=old.design();bits=torch.as_tensor(bits,dtype=torch.float64)
    source=old.source_activation(model,images)
    traced=old.trace(model,old.changed_sources(source,nodes,bits,.5))
    pool=old.reachable_middle(model,nodes,'cpu')
    assert np.array_equal(pool,new.reachable(model,nodes,source))
    selections=[[],pool[:16],pool]
    a=old.patch_outputs(model,traced['h'][1],selections).numpy()
    b=new.patch(model,traced['h'][1],selections).numpy()
    old_feature=old_select.input_features(model,traced['h'][1],int(labels[0]),stencil)
    new_feature=new.features(model,traced['h'][1],int(labels[0]),stencil)
    raw=traced['output'].numpy()
    checks=dict(patch=float(np.max(np.abs(a-b))),features=float(np.max(np.abs(old_feature[:2]-new_feature[:2]))),
        target=float(np.max(np.abs(old_select.scalar_output(raw,int(labels[0]))-new.project(raw,int(labels[0]),'margin')[:,0]))),
        finite=float(np.max(np.abs(old_select.finite_scores(a,raw,int(labels[0]),stencil)-new.finite_scores(b,raw,int(labels[0]),stencil)[0]))))
    assert max(checks.values())<1e-9,checks
    return dict(status='PASS',errors=checks,context='original Fashion validation input only; CPU float64')


if __name__=='__main__':
    print(json.dumps(known_answers(),ensure_ascii=False),flush=True)
    print(json.dumps(fashion_portability(),ensure_ascii=False),flush=True)

"""第二轮的独立数组核验器：不调用执行模块的差分、目标或指标函数。

用24×16差分矩阵和10维目标向量重新计算；名单由已核验评分独立重建。
该脚本是分析阶段工具，不冒称评价前冻结的数值执行代码。
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RUN = BASE/'runs/adaptive_mediator_20260928'
SEEDS = range(202609071, 202609077)


def read(path):
    """读取UTF-8记录，独立于实验入口的JSON辅助函数。"""
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    """核验原字节，不修改原始响应或冻结记录。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_contrasts():
    """从位集合定义独立建立条件成对差分，不使用保存的模板生成数值。"""
    matrix = np.zeros((24,16)); metadata = []
    for i,j in itertools.combinations(range(4),2):
        for background in range(16):
            if background & ((1<<i)|(1<<j)):
                continue
            row=len(metadata)
            matrix[row,background|(1<<i)|(1<<j)]=1
            matrix[row,background|(1<<i)]=-1
            matrix[row,background|(1<<j)]=-1
            matrix[row,background]=1
            metadata.append([i,j,background])
    return matrix,np.array(metadata)


def projected_interactions(logits, labels, matrix):
    """先进行十维输出差分，再以目标向量投影；运算顺序与执行代码不同。"""
    weight=np.full((len(labels),10),-1/9.)
    weight[np.arange(len(labels)),labels]=1.
    vector=np.einsum('sm,n...mc->n...sc',matrix,logits)
    return np.einsum('n...sc,nc->n...s',vector,weight)


def independent_order(values, candidates):
    """明确以编号作并列次序；主名单和可达池原本均按编号排列。"""
    return sorted(map(int,candidates),key=lambda node:(-float(values[node]),node))


def main():
    """检查冻结、有限评分、所有选择与评价原始量；结果只写入本轮新记录。"""
    matrix,metadata=independent_contrasts()
    maxima={}; comparisons=0; selection_count=0; score_count=0
    def compare(name, actual, expected, tolerance=1e-9):
        """固定绝对容差；不在看到不一致后按结果改变阈值。"""
        nonlocal comparisons
        actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,dtype=float)
        assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(),name
        error=float(np.max(np.abs(actual-expected))) if actual.size else 0.
        maxima[name]=max(maxima.get(name,0.),error);comparisons+=actual.size
        assert error<tolerance,(name,error)

    # 原始两个交付、两次冻结和源文件哈希均为明确核验对象。
    protocol=read(RUN/'protocol.json')
    assert digest(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    assert digest(RUN/'input_ids.npz')==protocol['ids_sha256']
    for name,expected in protocol['sources'].items():
        assert digest(BASE/name)==expected and digest(RUN/'source'/name)==expected
    frozen_files=0
    for name in ('fixed_freeze','adaptive_freeze'):
        assert digest(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
        f=read(RUN/f'{name}.json')
        for relative,expected in f['files'].items():
            assert digest(RUN/relative)==expected,relative
            frozen_files+=1
    assert read(RUN/'fixed_freeze.json')['utc']<read(RUN/'selection_opened.json')['utc']
    assert read(RUN/'adaptive_freeze.json')['utc']<read(RUN/'evaluation_opened.json')['utc']
    old_files=0
    for path,key in [(BASE/'runs/cross_layer_20260928/delivery_manifest.json','files'),
                     (BASE/'audit/joint_response_manifest_20260925.json','entries')]:
        for entry in read(path)[key]:
            assert digest(BASE/entry['path'])==entry['sha256'],entry['path']
            old_files+=1
    summary=read(RUN/'summary.json')
    table={(r['seed'],r['alpha'],r['method'],r['budget']):r for r in summary['methods']}
    with (RUN/'per_input.csv').open(encoding='utf-8-sig',newline='') as stream:
        per_input={(int(r['seed']),float(r['alpha']),int(r['input_id']),r['method'],int(r['budget'])):r
                   for r in csv.DictReader(stream)}
    assert len(table)==456 and len(per_input)==29184
    for seed in SEEDS:
        dev=RUN/'development'/f'seed_{seed}'
        chosen=RUN/'input_selection'/f'seed_{seed}'
        fixed=read(dev/'fixed_orders.json'); specs=read(chosen/'specs.json')
        legacy=read(BASE/f'runs/cross_layer_20260928/development/seed_{seed}/selections.json')['selections']
        with np.load(dev/'discovery.npz',allow_pickle=False) as a:
            mean_feature=a['features'].mean(0); short=a['shortlist'].copy(); pool=a['pool'].copy()
            mean_score=a['finite_scores'].mean(0)
            inter_order=independent_order(mean_feature[0],pool)
            single_order=independent_order(mean_feature[1],pool)
            shortlist=set(inter_order[:32])|set(single_order[:32])
            for node in inter_order:
                if len(shortlist)==64:break
                shortlist.add(node)
            assert np.array_equal(short,np.array(sorted(shortlist)))
            assert fixed['fixed_gradient']==independent_order(mean_feature[0],short)
            assert fixed['fixed_single']==independent_order(mean_feature[1],short)
            assert fixed['fixed_finite']==[int(short[i]) for i in sorted(range(64),key=lambda j:(-mean_score[j],int(short[j])))]
        for stage, filename in [(dev,'discovery.npz'),(chosen,'selection.npz')]:
            with np.load(stage/filename,allow_pickle=False) as archive:
                # 压缩NPZ的索引每次都会解压；先读入一次，避免逐坐标重复解压。
                a={name:archive[name] for name in archive.files}
                compare('metadata',a['pair_metadata'],metadata)
                ni=projected_interactions(a['natural'],a['labels'],matrix)
                si=projected_interactions(a['singleton_logits'],a['labels'],matrix)
                scores=(np.einsum('ns,ns->n',ni,ni)[:,None]
                        -.5*(np.einsum('njs,njs->nj',si[:,:,0],si[:,:,0])
                        +np.einsum('njs,njs->nj',si[:,:,1]-ni[:,None],si[:,:,1]-ni[:,None])))/24
                compare('finite_scores',a['finite_scores'],scores);score_count+=scores.size
                if filename=='selection.npz':
                    for i,feature in enumerate(a['features']):
                        input_orders=dict(fixed)
                        input_orders['input_gradient']=independent_order(feature[0],short)
                        input_orders['input_single']=independent_order(feature[1],short)
                        input_orders['input_gradient_full']=independent_order(feature[0],pool)
                        input_orders['input_finite']=[int(short[j]) for j in sorted(range(64),key=lambda j:(-a['finite_scores'][i,j],int(short[j])))]
                        for j,spec in enumerate(specs):
                            method,k=spec['method'],spec['budget']
                            if method=='all_shortlist':expected=short
                            elif method=='all_reachable':expected=pool
                            elif method=='legacy_fixed':expected=next(s['nodes'] for s in legacy if s['method']=='interaction' and s['budget']==k)
                            else:expected=input_orders[method][:k]
                            assert np.array_equal(a['selections'][i,j,:k],expected),(seed,i,method,k)
                            assert np.all(a['selections'][i,j,k:]==-1)
                            selection_count+=1
        for alpha in (1.,.5):
            folder=RUN/'evaluation'/f'seed_{seed}'/f'alpha_{alpha}'
            for name,expected in read(folder/'completion.json')['files'].items():
                assert digest(folder/name)==expected
            with np.load(folder/'responses.npz',allow_pickle=False) as a:
                ni=projected_interactions(a['natural'],a['labels'],matrix)
                pi=projected_interactions(a['patched'],a['labels'],matrix)
                dn=np.linalg.norm(ni,axis=-1)/np.sqrt(24.)
                rr=np.linalg.norm(pi[:,:,0],axis=-1)/np.sqrt(24.)
                tt=np.linalg.norm(pi[:,:,1]-ni[:,None],axis=-1)/np.sqrt(24.)
                for j,spec in enumerate(specs):
                    row=table[(seed,alpha,spec['method'],spec['budget'])]
                    den=float(np.linalg.norm(dn)/np.sqrt(len(dn)))
                    res=float(np.linalg.norm(rr[:,j])/np.sqrt(len(dn)))
                    err=float(np.linalg.norm(tt[:,j])/np.sqrt(len(dn)))
                    compare('teacher_primitives',[row['interaction_rms'],row['restored_rms'],row['transplant_error_rms']], [den,res,err])
                    compare('teacher_ratios',[row['restore_ratio'],row['transplant_error_ratio'],row['joint_loss']], [res/den,err/den,.5*((res/den)**2+(err/den)**2)])
                    for i,input_id in enumerate(a['ids']):
                        r=per_input[(seed,alpha,int(input_id),spec['method'],spec['budget'])]
                        values=[float(r[k]) for k in ('interaction_rms','restored_rms','transplant_error_rms')]
                        compare('input_primitives',values,[dn[i],rr[i,j],tt[i,j]])
                        # 比率核验用已经独立核验的保存分子分母，不把小分母舍入放大误当原始量错误。
                        if values[0]>1e-9:
                            compare('saved_input_division',[float(r['restore_ratio']),float(r['transplant_error_ratio'])],
                                    [values[1]/values[0],values[2]/values[0]])
                        else:
                            assert r['restore_ratio']==r['transplant_error_ratio']==''
    audit=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scalar_comparisons=comparisons,
        finite_scores_verified=score_count,selected_sets_verified=selection_count,method_rows=len(table),
        input_rows=len(per_input),frozen_files=frozen_files,old_files_preserved=old_files,
        max_absolute_differences=maxima,ratio_check='teacher ratios independently recomputed; per-input ratios divide independently verified stored primitives',
        tolerance=1e-9,source_sha256=digest(Path(__file__)))
    output=RUN/'independent_verification.json'
    assert not output.exists(),'Keep completed audit; use read-only integrity checks'
    output.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

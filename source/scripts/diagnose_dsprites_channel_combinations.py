"""事后只读诊断通道联合非加和性，不新增模型查询或重新选择名单。

使用半强度已有32通道精确单点响应和19方案真实联合响应，分开记录
高阶联合余项、端点匹配误差与分类结果。该诊断不属于原主比较，
不能用它改写冻结协议、调整预算或据此宣称更好的名单。
"""
from __future__ import annotations

import csv
import json
import dsprites_teacher as teacher
import numpy as np

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_shape_donor_20261010'


def load(path):
    """纯数值原件只读。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def verify_inputs():
    """要求原独立审计完成，原始阶段哈希仍一致。"""
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    for name in ('fit_freeze','selection_freeze'):
        assert teacher.sha(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
        for path,value in teacher.read(RUN/f'{name}.json')['files'].items():assert teacher.sha(RUN/path)==value,path
    for seed in teacher.SEEDS:
        folder=RUN/f'evaluation/seed_{seed}/alpha_0.5'
        for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name


def diagnose():
    """总高阶余项为联合变化减精确单点变化之和，不识别单独某一对。"""
    assert not (RUN/'combination_diagnostic.json').exists(),'Preserve completed diagnostic'
    verify_inputs();specs=teacher.read(RUN/'specs.json');rows=[];maximum=0.
    # 手算非线性集合例：单点1、2，联合5，联合余项为2；仿射例余项为0。
    assert 5-0-((1-0)+(2-0))==2 and 3-0-((1-0)+(2-0))==0
    for seed in teacher.SEEDS:
        selected=load(RUN/f'selection/seed_{seed}/data.npz')
        raw=load(RUN/f'evaluation/seed_{seed}/alpha_0.5/data.npz');weights=raw['weights']
        origin=np.einsum('nc,nc->n',raw['natural'][:,0],weights)
        endpoint=np.einsum('nc,nc->n',raw['natural'][:,-1],weights);effect=endpoint-origin
        effect_rms=float(np.sqrt(np.mean(effect**2)));assert effect_rms>0
        single=np.einsum('nkoc,nc->nko',selected['singleton_logits'],weights)
        joint=np.einsum('nmoc,nc->nmo',raw['patched'][:,:,:,-1],weights)
        r_step=single[:,:,0]-endpoint[:,None];t_step=single[:,:,1]-origin[:,None]
        for j,spec in enumerate(specs):
            chosen=[x[x>=0] for x in selected['selections'][:,j]]
            additive=np.asarray([[endpoint[i]+r_step[i,c].sum(),origin[i]+t_step[i,c].sum()] for i,c in enumerate(chosen)])
            residue=joint[:,j]-additive
            # 独立计算：先在三类logit空间相加，再作类别投影，核对换序运算。
            direct=[]
            for i,channels in enumerate(chosen):
                r_ind=raw['natural'][i,-1]+sum(selected['singleton_logits'][i,c,0]-raw['natural'][i,-1] for c in channels)
                t_ind=raw['natural'][i,0]+sum(selected['singleton_logits'][i,c,1]-raw['natural'][i,0] for c in channels)
                direct.append(np.stack((r_ind,t_ind)))
            independent=np.einsum('noc,nc->no',np.asarray(direct),weights)
            error=float(np.max(np.abs(independent-additive)));maximum=max(maximum,error);assert error<1e-10
            # 空集和单点不含二阶及以上余项，这里报告的是实际多个通道联合余项。
            response=joint[:,j]-origin[:,None];classes=raw['patched'][:,j,:,-1].argmax(-1)
            rows.append(dict(seed=seed,alpha=.5,**spec,effect_rms=effect_rms,
                restore_nonadditive_ratio=float(np.linalg.norm(residue[:,0])/np.linalg.norm(effect)),
                transplant_nonadditive_ratio=float(np.linalg.norm(residue[:,1])/np.linalg.norm(effect)),
                restore_ratio=float(np.linalg.norm(response[:,0])/np.linalg.norm(effect)),
                transplant_error_ratio=float(np.linalg.norm(response[:,1]-effect)/np.linalg.norm(effect)),
                restore_overshoot_fraction=float(np.mean(response[:,0]*effect<0)),
                transplant_overshoot_fraction=float(np.mean((response[:,1]-effect)*effect>0)),
                recipient_return_rate=float(np.mean(classes[:,0]==raw['recipient_labels'])),
                donor_reproduction_rate=float(np.mean(classes[:,1]==raw['donor_labels']))))
    result=dict(status='PASS',utc=teacher.now(),kind='posthoc frozen-response diagnostic',alpha=.5,
        new_model_queries=0,new_selections=0,independent_projection_maximum_error=maximum,
        restore_formula='F_R(P)-F_all-sum_j_in_P(F_R(j)-F_all)',
        transplant_formula='F_T(P)-F_0-sum_j_in_P(F_T(j)-F_0)',
        meaning='total orders2+ nonadditivity at this baseline, not unique pairwise interaction or Shapley values',rows=rows)
    teacher.dump(RUN/'combination_diagnostic.json',result)
    with (RUN/'tables/channel_combinations.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='PASS',rows=len(rows),new_queries=0,maximum_error=maximum),ensure_ascii=False))


if __name__=='__main__':diagnose()

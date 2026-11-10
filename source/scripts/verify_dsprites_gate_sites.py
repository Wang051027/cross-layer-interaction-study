"""对第一个模型和首个配对做逐坐标干预。按半强度排名选取每层前八个位置，再固定对应开关并测量输出。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import numpy as np
import torch
import dsprites_teacher as teacher
from test_dsprites_channel_gates import forward_override

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_channel_gates_20261011'
PRIOR=BASE/'runs/dsprites_shape_donor_20261010';FOLDER=RUN/'single_sites'
SELF=BASE/'scripts/verify_dsprites_gate_sites.py';SEED=teacher.SEEDS[0]


def load(path):
    """数值数据只读，拒绝pickle。"""
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def init():
    """记录接续实验的输入面板和已有定位结果。"""
    assert not FOLDER.exists();FOLDER.mkdir()
    files=[RUN/'protocol.json',RUN/'gate_freeze.json',RUN/'design.npz',
        RUN/f'discovery/seed_{SEED}/data.npz',RUN/f'evaluation/seed_{SEED}/alpha_1.0/data.npz',
        PRIOR/'panel/data.npz',teacher.RUN/f'seed_{SEED}/teacher.pt',
        BASE/'scripts/test_dsprites_channel_gates.py',BASE/'scripts/dsprites_teacher.py',
        BASE/'scripts/residual_teacher.py',BASE/'scripts/channel_gate_mechanism.py']
    teacher.dump(FOLDER/'protocol.json',dict(utc=teacher.now(),version='fixed-example-single-sites-v1',
        status='exploratory supplement after main results inspected',
        rule='seed202609291,case0,full strength,half-rank0 pair,first8 frozen gate coordinates,each alone andall8;both operations and stages',
        metric='signed natural pair effect minus actual pair effect after baseline gate holding',
        source_sha256=teacher.sha(SELF),inputs={str(p.relative_to(BASE)).replace('\\','/'):teacher.sha(p) for p in files}))
    shutil.copy2(SELF,FOLDER/'source.py')
    (FOLDER/'protocol.sha256').write_text(teacher.sha(FOLDER/'protocol.json')+'\n',encoding='ascii')
    print(json.dumps(dict(status='SEALED',protocol_sha256=teacher.sha(FOLDER/'protocol.json'))))


def evaluate():
    """原block3真实forward逐点重算；第二层净变化还核对精确坐标贡献。"""
    assert not (FOLDER/'result.json').exists();p=teacher.read(FOLDER/'protocol.json')
    assert teacher.sha(FOLDER/'protocol.json')==(FOLDER/'protocol.sha256').read_text().strip()
    assert teacher.sha(SELF)==teacher.sha(FOLDER/'source.py')==p['source_sha256']
    for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
    device=teacher.configure();model=teacher.load_teacher(SEED,device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    panel=load(PRIOR/'panel/data.npz');frozen=load(RUN/f'discovery/seed_{SEED}/data.npz')
    full=load(RUN/f'evaluation/seed_{SEED}/alpha_1.0/data.npz');d=load(RUN/'design.npz')
    r,donor=panel['pairs'][0];w=np.zeros(3);w[panel['labels'][donor]]=1;w[panel['labels'][r]]=-1
    with torch.no_grad():
        images=torch.as_tensor(panel['images'][[r,donor],None],dtype=torch.float64,device=device)
        source=model.block1(model.stem(images));recipient=model.block2(source[:1]);endpoint=model.block2(source[1:])
    rows=[];errors=[]
    def effect(values):return float((values[3]-values[1]-values[2]+values[0])@w)
    for op in range(2):
        pair=int(frozen['selected_pairs'][0,op,0]);channels=frozen['channels'][0,d['pairs'][pair]]
        base,reference=(endpoint,recipient) if op==0 else (recipient,endpoint)
        states=base.repeat(4,1,1,1)
        for state,members in enumerate(((),(0,),(1,),(0,1))):
            for j in members:states[state,int(channels[j])]=reference[0,int(channels[j])]
        natural=forward_override(model,states,base).cpu().numpy();truth=effect(natural)
        saved=float(full['pair_logits'][0,op,pair,0]@w);errors.append(abs(truth-saved))
        for stage in range(2):
            selected=frozen['selected_gates'][0,op,0,stage];ids=selected[selected>=0][:8]
            for rank,unit_list in enumerate([np.array([unit]) for unit in ids]+[ids]):
                actual=forward_override(model,states,base,stage=stage,ids=unit_list).cpu().numpy()
                after=effect(actual);algebra=float(full['unit_contributions'][0,op,0,stage,rank]) if rank<len(ids) else float(full['unit_contributions'][0,op,0,stage,:len(ids)].sum())
                if stage==1:errors.append(abs((truth-after)-algebra))
                rows.append(dict(seed=SEED,case=0,operation=('restore','transplant')[op],stage=stage+1,
                    state_a_channel=int(channels[0]),state_b_channel=int(channels[1]),
                    frozen_unit_rank=rank+1 if rank<len(ids) else 'group8',
                    coordinates=';'.join(','.join(map(str,np.unravel_index(int(unit),(32,32,32)))) for unit in unit_list),
                    natural_effect=truth,held_effect=after,net_effect_removed=truth-after,
                    algebra_contribution=algebra,net_minus_algebra=(truth-after)-algebra))
            for bi,budget in enumerate((32,128)):
                actual=forward_override(model,states,base,stage=stage,ids=selected[selected>=0][:budget]).cpu().numpy()
                errors.append(float(np.max(np.abs(actual-full['partial_logits'][0,op,0,stage,bi,0]))))
    assert max(errors)<1e-9,errors
    with (FOLDER/'single_sites.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS',utc=teacher.now(),protocol_sha256=teacher.sha(FOLDER/'protocol.json'),
        protocol_status=p['status'],rows=rows,maximum_reference_error=max(errors),
        identities='natural and32/128gate outputs match main experiment; second gate net change equals selected algebra contribution',
        boundary='first-gate net change can differ from its algebra part due to recalculated second gate; no new model/input generalization')
    teacher.dump(FOLDER/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('init','evaluate'))
    args=parser.parse_args();init() if args.stage=='init' else evaluate()

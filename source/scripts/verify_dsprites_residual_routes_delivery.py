"""只读核验新路由交付、固定时序、数值恒等式和七轮旧研究。"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import re
import numpy as np
import dsprites_teacher as teacher
from run_dsprites_residual_routes import RUN,BASE,load,verify_seal,verify_folder


def verify(deep=False):
    manifest=RUN/'delivery_manifest.json';assert teacher.sha(manifest)==manifest.with_suffix('.sha256').read_text().strip();records=teacher.read(manifest)['files']
    assert len(set(r['path'] for r in records))==len(records)
    for r in records:
        p=(BASE/r['path']).resolve();assert p.is_relative_to(BASE.resolve()) and p.stat().st_size==r['bytes'] and teacher.sha(p)==r['sha256'],r['path']
    freeze=verify_seal();a=teacher.read(RUN/'analysis.json');audit=teacher.read(RUN/'independent_audit.json');s=teacher.read(RUN/'summary.json')
    assert a['status']==audit['status']=='PASS' and len(s['rows'])==456 and len(s['decomposition'])==12 and len(a['primary'])==36
    for name,value in audit['example_response_files'].items():assert teacher.sha(RUN/name)==value
    threshold=teacher.read(RUN/'threshold_check.json');assert threshold['status']=='PASS' and a['threshold']==threshold
    changes=np.array(threshold['single_preactivation_changes']);order=np.lexsort((np.arange(32),changes));k=threshold['minimum_channels'];cum=threshold['baseline']+np.cumsum(changes[order])
    assert np.all(cum[:k-1]>1e-9) and cum[k-1]<-1e-9 and order[:k].tolist()==threshold['chosen_channels']
    # 事后例子的单通道阈值值，另外与独立审计原forward边表核对。
    edges=teacher.read(RUN/'channel_gate_edges.json')['rows'];coordinate=threshold['coordinate']
    for c in range(32):
        row=next(r for r in edges if r['alpha']==1 and r['mask']==15 and r['stage']==1 and r['kind']=='selected1' and [r['gate_channel'],r['gate_y'],r['gate_x']]==coordinate and r['middle_channel']==c)
        assert abs(row['solo_change']-changes[c])<1e-9
    sixth=teacher.read(RUN/'threshold_sixth_order.json');assert sixth['status']=='PASS' and sixth['rows']==64
    subset_file=RUN/'tables/six_channel_subsets.csv';assert teacher.sha(subset_file)==sixth['table_sha256'] and teacher.sha(RUN/'threshold_check.json')==sixth['threshold_check_sha256']
    with subset_file.open(encoding='utf-8-sig',newline='') as f:subsets=list(csv.DictReader(f))
    assert len(subsets)==64 and [int(r['mask']) for r in subsets]==list(range(64))
    signs=np.array([(-1)**(6-m.bit_count()) for m in range(64)])
    for column,key in [('activation','activation_sixth'),('natural_output','natural_output_sixth'),('point_held_output','point_held_output_sixth')]:
        values=np.array([float(r[column]) for r in subsets]);assert abs(signs@values-sixth[key])<1e-9
    net=np.array([float(r['point_net_removed']) for r in subsets]);assert np.max(abs(net[:63]))==0 and abs(net[-1]-sixth['point_net_sixth'])<1e-9
    for row in a['overview']:
        frame=[x for x in s['rows'] if all(x[k]==row[k] for k in ('alpha','family','method','operation'))];assert len(frame)==6
        for key in ('target_rms','remaining_ratio','error_ratio','signed_removal'):assert abs(row[key]-np.mean([x[key] for x in frame]))<1e-10
    for name,count in a['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);header=next(reader);rows=list(reader);assert len(set(header))==len(header) and len(rows)==count and all(len(r)==len(header) for r in rows),name
    for seed in teacher.SEEDS:
        discovery=load(RUN/f'discovery/seed_{seed}/selection.npz')
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';verify_folder(folder);assert teacher.read(folder/'completion.json')['utc']>freeze['utc']
            sel=load(folder/'selection.npz');assert all(np.array_equal(sel[k],discovery[k]) for k in sel)
            raw=load(folder/'data.npz');assert raw['modes'].shape==(72,16,4,3) and raw['single_logits'].shape==(72,32,16,3)
            assert raw['gate_logits'].shape==(72,2,8,16,3) and raw['channel_logits'].shape==(72,9,2,16,3)
            assert all(np.isfinite(x).all() for x in raw.values())
            assert np.array_equal(raw['channel_logits'][:,-1,0],raw['baseline']) and np.array_equal(raw['channel_logits'][:,-1,1],raw['modes'][:,:,0])
            if seed==teacher.SEEDS[0]:
                for mask in (3,15):
                    p=load(folder/f'path_mask{mask}.npz');assert np.all(p['width']>0) and abs(p['width'].sum()-1)<1e-12
                    assert abs(p['integrated_channel'].sum()-np.diff(p['endpoints'])[0])<1e-9
    report=BASE/'reports/dSprites_中间组合效应的通道与ReLU传递_2026-10-13.md'
    for p in (report,RUN/'README.md',BASE/'CURRENT_RESEARCH.md'):
        text=p.read_text(encoding='utf-8');inside=False;width=0
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('http://','https://','#')):assert (p.parent/target.strip('<>').split('#')[0]).resolve().exists(),(p,target)
        for line in text.splitlines():
            if line.startswith('|'):
                current=len(line.split('|'))
                if not inside:width=current;inside=True
                assert current==width,(p,line)
            else:inside=False
    for p in (RUN/'source/scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    count=0
    if deep:
        protocol=teacher.read(RUN/'protocol.json')
        for name,row in protocol['preservation'].items():
            p=BASE/name;assert teacher.sha(p)==row['sha256']==p.with_suffix('.sha256').read_text().strip();old=teacher.read(p)['files'];assert len(old)==row['files']
            for r in old:
                file=BASE/r['path'];assert file.stat().st_size==r['bytes'] and teacher.sha(file)==r['sha256'],r['path'];count+=1
        assert count==1151;teacher.verify_old()
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),old_entries=count,deep_inputs=deep,
        tables=len(a['table_rows'])+1,table_rows=sum(a['table_rows'].values())+64,manifest_sha256=teacher.sha(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deep-inputs',action='store_true');args=parser.parse_args();print(json.dumps(verify(args.deep_inputs)))

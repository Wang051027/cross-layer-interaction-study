"""核对层间传递实验的文件、汇总表与历史输入。"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import re
import numpy as np
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_layer_transfer_20261013'


def close(a,b,tolerance=1e-10):
    """缺失比例与零基准分开核验。"""
    if a is None or b is None:assert a is b,(a,b)
    else:assert abs(float(a)-float(b))<tolerance,(a,b)


def verify(deep=False):
    """核验完成交付；deep另外检查六轮1059条历史文件及更早输入。"""
    manifest=RUN/'delivery_manifest.json';assert teacher.sha(manifest)==manifest.with_suffix('.sha256').read_text().strip()
    records=teacher.read(manifest)['files'];assert len({r['path'] for r in records})==len(records)
    for item in records:
        path=(BASE/item['path']).resolve();assert path.is_relative_to(BASE.resolve()),item['path']
        assert path.stat().st_size==item['bytes'] and teacher.sha(path)==item['sha256'],item['path']
    protocol=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    assert teacher.sha(RUN/'channel_orders.npz')==protocol['channel_orders_sha256']
    for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    audit=teacher.read(RUN/'independent_audit.json');a=teacher.read(RUN/'analysis.json');s=teacher.read(RUN/'summary.json')
    assert audit['status']==a['status']=='PASS'
    assert teacher.sha(RUN/audit['negative_fraction']['file'])==audit['negative_fraction']['sha256']
    assert len(s['rows'])==576 and len(s['decomposition'])==24 and len(a['examples'])==22 and len(a['primary'])==12
    for r in a['condition_overview']:
        group=[x for x in s['rows'] if all(x[k]==r[k] for k in ('alpha','metric','method','operation'))];assert len(group)==6
        for key in ('natural_rms','remaining_ratio','changed_ratio','signed_removal','restore_error_ratio'):
            vals=[x.get(key) for x in group];expected=None if any(v is None for v in vals) else float(np.mean(vals));close(r[key],expected)
    for r in a['primary']:assert r in s['rows'] and r['method']=='middle_le1' and r['metric']=='conditional_pairs'
    for name,count in a['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);header=next(reader);assert len(set(header))==len(header)
            rows=list(reader);assert len(rows)==count,(name,len(rows),count)
            assert all(len(r)==len(header) for r in rows),name
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}';completion=teacher.read(folder/'completion.json');assert completion['utc']>protocol['utc']
            for name,value in completion['files'].items():assert teacher.sha(folder/name)==value,name
            with np.load(folder/'data.npz',allow_pickle=False) as z:
                assert z['logits'].shape==(72,16,4,3) and z['patched_logits'].shape==(72,7,2,16,3)
                assert z['directional_source'].shape==(72,16,4) and z['gradient_drift'].shape==(72,16,2,3)
                for name in z.files:assert np.isfinite(z[name]).all(),(seed,alpha,name)
                assert np.max(abs(z['lower_logits'][:,0]-z['patched_logits'][:,6,0]))==0
                assert np.max(abs(z['logits'][:,:,0]-z['patched_logits'][:,6,1]))==0
    # 原位置文档的相对链接必须能找到文件；快照中的链接仍以原项目阅读。
    report=BASE/'reports/dSprites_源层组合与层间传递验证_2026-10-13.md'
    for path in (report,RUN/'README.md',BASE/'CURRENT_RESEARCH.md'):
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http://','https://','#')):continue
            target=target.strip('<>').split('#')[0]
            assert (path.parent/target).resolve().exists(),(path,target)
        in_table=False;width=0
        for line in text.splitlines():
            if line.startswith('|'):
                count=len(line.split('|'))
                if not in_table:width=count;in_table=True
                assert count==width,(path,line)
            else:in_table=False
    for path in RUN.glob('source/scripts/*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    old_count=0
    if deep:
        for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in protocol['preservation'].items():
            path=BASE/name;assert teacher.sha(path)==value['sha256']==path.with_suffix('.sha256').read_text().strip(),name
            old=teacher.read(path)['files'];assert len(old)==value['files']
            for item in old:
                p=BASE/item['path'];assert p.stat().st_size==item['bytes'] and teacher.sha(p)==item['sha256'],item['path'];old_count+=1
        assert old_count==1059;teacher.verify_old()
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),deep_inputs=deep,old_delivery_file_entries=old_count,
        manifest_sha256=teacher.sha(manifest),protocol_sha256=teacher.sha(RUN/'protocol.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deep-inputs',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.deep_inputs),ensure_ascii=False))

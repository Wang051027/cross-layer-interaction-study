"""只读核验三阶交付、协议、名单和表格；深核验包含全部历史输入。"""
from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_triple_gates_20261011'


def verify(deep=False):
    """逐项校验文件大小和 SHA-256。"""
    manifest=RUN/'delivery_manifest.json';assert teacher.sha(manifest)==manifest.with_suffix('.sha256').read_text().strip()
    records=teacher.read(manifest)['files'];assert isinstance(records,list)
    assert len({r['path'] for r in records})==len(records)
    for item in records:
        path=(BASE/item['path']).resolve();assert path.is_relative_to(BASE.resolve()),item['path']
        assert path.stat().st_size==item['bytes'],item['path'];assert teacher.sha(path)==item['sha256'],item['path']
    protocol=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    freeze=teacher.read(RUN/'selection_freeze.json');assert teacher.sha(RUN/'selection_freeze.json')==(RUN/'selection_freeze.sha256').read_text().strip()
    assert freeze['protocol_sha256']==teacher.sha(RUN/'protocol.json')
    for name,value in freeze['files'].items():assert teacher.sha(RUN/name)==value,name
    for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==teacher.sha(RUN/'source'/name)==value,name
    audit=teacher.read(RUN/'independent_audit.json');a=teacher.read(RUN/'analysis.json')
    assert audit['status']==a['status']=='PASS'
    check=teacher.read(RUN/'net_response_check.json');assert check['status']=='PASS' and check['conditions']==128
    for name,value in check['files'].items():assert teacher.sha(RUN/name)==value,name
    with (RUN/'tables/net_response_cases.csv').open(encoding='utf-8-sig',newline='') as f:net_cases=list(csv.DictReader(f))
    assert len(net_cases)==check['case_rows']==128
    assert [int(r['map_index']) for r in net_cases]==list(range(128))
    with (RUN/'tables/net_response_coordinates.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);next(reader);assert sum(1 for _ in reader)==check['coordinate_rows']
    with np.load(RUN/'net_response_maps.npz',allow_pickle=False) as z:
        assert list(z['signed_response'].shape)==check['maps_shape']==[128,32,32,32]
        assert np.isfinite(z['signed_response']).all()
        assert np.array_equal(z['natural_pattern_code']&1,z['held_pattern_code']&1),'Baseline gate bits must stay unchanged'
        expected=np.array([float(r['second_response']) for r in net_cases])
        assert np.max(abs(z['signed_response'].sum((1,2,3))-expected))<1e-9
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==1536 and len(summary['hierarchy'])==72
    assert len(a['primary'])==4 and len(a['example_single_sites'])==64
    for name,count in a['table_rows'].items():
        with (RUN/f'tables/{name}.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);next(reader);assert sum(1 for _ in reader)==count,(name,count)
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):
            folder=RUN/f'evaluation/seed_{seed}/alpha_{alpha}'
            for name,value in teacher.read(folder/'completion.json')['files'].items():assert teacher.sha(folder/name)==value,name
    old_count=0
    if deep:
        for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in protocol['preservation'].items():
            path=BASE/name;assert teacher.sha(path)==value['sha256'],name
            assert teacher.sha(path)==path.with_suffix('.sha256').read_text().strip(),name
            old=teacher.read(path)['files'];assert len(old)==value['files']
            for item in old:
                p=BASE/item['path'];assert p.stat().st_size==item['bytes'] and teacher.sha(p)==item['sha256'],item['path'];old_count+=1
        teacher.verify_old()
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),deep_inputs=deep,
        old_delivery_file_entries=old_count,manifest_sha256=teacher.sha(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deep-inputs',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.deep_inputs),ensure_ascii=False))

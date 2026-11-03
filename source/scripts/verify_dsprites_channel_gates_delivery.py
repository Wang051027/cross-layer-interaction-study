"""只读核验交付、协议、选择名单；深核验覆盖全部冻结输入和旧交付。"""
from __future__ import annotations

import argparse
import json
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_channel_gates_20261011'


def verify(deep=False):
    """先验证清单自身，再验证大小和SHA256，不修改事件或任何数据。"""
    manifest=RUN/'delivery_manifest.json';assert teacher.sha(manifest)==(RUN/'delivery_manifest.sha256').read_text().strip()
    records=teacher.read(manifest)['files'];assert isinstance(records,list)
    for item in records:
        path=(BASE/item['path']).resolve();assert path.is_relative_to(BASE.resolve()),item['path']
        assert path.stat().st_size==item['bytes'],item['path']
        assert teacher.sha(path)==item['sha256'],item['path']
    protocol=teacher.read(RUN/'protocol.json');assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    freeze=teacher.read(RUN/'gate_freeze.json');assert teacher.sha(RUN/'gate_freeze.json')==(RUN/'gate_freeze.sha256').read_text().strip()
    assert freeze['protocol_sha256']==teacher.sha(RUN/'protocol.json')
    for name,value in freeze['files'].items():assert teacher.sha(RUN/name)==value,name
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert teacher.read(RUN/'analysis.json')['status']=='PASS'
    assert teacher.read(RUN/'four_channel_terms.json')['status']=='PASS'
    site_protocol=RUN/'single_sites/protocol.json';site=teacher.read(site_protocol)
    assert teacher.sha(site_protocol)==(RUN/'single_sites/protocol.sha256').read_text().strip()
    assert teacher.read(RUN/'single_sites/result.json')['status']=='PASS'
    assert teacher.read(RUN/'single_sites/result.json')['protocol_sha256']==teacher.sha(site_protocol)
    old_count=0
    if deep:
        assert teacher.sha(BASE/'scripts/verify_dsprites_gate_sites.py')==teacher.sha(RUN/'single_sites/source.py')==site['source_sha256']
        for name,value in site['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
        for name,old in protocol['preservation'].items():
            assert teacher.sha(BASE/name)==old['sha256'],name
            entries=teacher.read(BASE/name)['files'];assert len(entries)==old['files']
            for item in entries:assert teacher.sha(BASE/item['path'])==item['sha256'],item['path'];old_count+=1
        teacher.verify_old()
    return dict(status='PASS',files=len(records),deep_inputs=deep,old_delivery_file_entries=old_count,manifest_sha256=teacher.sha(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deep-inputs',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.deep_inputs),ensure_ascii=False))

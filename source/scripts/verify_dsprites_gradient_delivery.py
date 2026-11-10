"""核对梯度实验的清单、历史输入与冻结记录。深度核验还检查原始数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import dsprites_teacher as teacher

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_gradient_20261009'


def verify(deep=False):
    """manifest以项目根目录为相对基准，先确认路径没有逃出项目。"""
    manifest=RUN/'delivery_manifest.json'
    assert teacher.sha(manifest)==(RUN/'delivery_manifest.sha256').read_text().strip()
    records=teacher.read(manifest)['files']
    assert isinstance(records,list)
    for item in records:
        path=(BASE/item['path']).resolve()
        assert path.is_relative_to(BASE.resolve()),item['path']
        assert path.stat().st_size==item['bytes'],item['path']
        assert teacher.sha(path)==item['sha256'],item['path']
    protocol=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    freeze=teacher.read(RUN/'selection_freeze.json')
    assert teacher.sha(RUN/'selection_freeze.json')==(RUN/'selection_freeze.sha256').read_text().strip()
    for name,value in freeze['files'].items():assert teacher.sha(RUN/name)==value,name
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    old_files=0
    if deep:
        for name,value in protocol['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in protocol['sources'].items():assert teacher.sha(BASE/name)==value,name
        for name,previous in protocol['preservation'].items():
            assert teacher.sha(BASE/name)==previous['sha256'],name
            for item in teacher.read(BASE/name)['files']:
                assert teacher.sha(BASE/item['path'])==item['sha256'],item['path'];old_files+=1
        teacher.verify_old()
    return dict(status='PASS',files=len(records),deep_inputs=deep,old_delivery_file_entries=old_files,
                manifest_sha256=teacher.sha(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep-inputs',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.deep_inputs),ensure_ascii=False))

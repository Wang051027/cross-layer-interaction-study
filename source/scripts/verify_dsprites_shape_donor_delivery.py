"""只读核验形状配对实验，可选核对全部旧交付和冻结输入。

轻核验检查新交付、协议和两次封存；深核验还需要完整旧数据和教师。
不会调用写状态的实验阶段，不将清单里的整数计数当作文件列表。
"""
from __future__ import annotations

import argparse
import json
import dsprites_teacher as teacher

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_shape_donor_20261010'


def verify(deep=False):
    """逐文件确认大小、SHA256和路径边界，旧证据仅只读访问。"""
    manifest=RUN/'delivery_manifest.json'
    assert teacher.sha(manifest)==(RUN/'delivery_manifest.sha256').read_text().strip()
    records=teacher.read(manifest)['files'];assert isinstance(records,list)
    for item in records:
        path=(BASE/item['path']).resolve();assert path.is_relative_to(BASE.resolve()),item['path']
        assert path.stat().st_size==item['bytes'],item['path']
        assert teacher.sha(path)==item['sha256'],item['path']
    p=teacher.read(RUN/'protocol.json')
    assert teacher.sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    for name in ('fit_freeze','selection_freeze'):
        assert teacher.sha(RUN/f'{name}.json')==(RUN/f'{name}.sha256').read_text().strip()
        frozen=teacher.read(RUN/f'{name}.json')
        assert frozen['protocol_sha256']==teacher.sha(RUN/'protocol.json')
        for path,value in frozen['files'].items():assert teacher.sha(RUN/path)==value,path
    assert teacher.read(RUN/'independent_audit.json')['status']=='PASS'
    assert teacher.read(RUN/'combination_diagnostic.json')['status']=='PASS'
    old_count=0
    if deep:
        for name,value in p['inputs'].items():assert teacher.sha(BASE/name)==value,name
        for name,value in p['sources'].items():
            assert teacher.sha(BASE/name)==value and teacher.sha(RUN/'source'/name)==value,name
        for name,previous in p['preservation'].items():
            assert teacher.sha(BASE/name)==previous['sha256'],name
            old=teacher.read(BASE/name)['files'];assert len(old)==previous['files']
            for item in old:
                assert teacher.sha(BASE/item['path'])==item['sha256'],item['path'];old_count+=1
        teacher.verify_old()
    return dict(status='PASS',files=len(records),deep_inputs=deep,old_delivery_file_entries=old_count,
                manifest_sha256=teacher.sha(manifest))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deep-inputs',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.deep_inputs),ensure_ascii=False))

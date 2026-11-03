"""审计和报告完成后封存通道与ReLU研究，本轮交付不覆盖历史证据。"""
from __future__ import annotations

import json
import shutil
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_channel_gates_20261011'
REPORT=BASE/'reports/dSprites_通道组合与ReLU开关定位_2026-10-11.md'
SCRIPTS=('channel_gate_mechanism.py','run_dsprites_channel_gates.py','test_dsprites_channel_gates.py',
    'audit_dsprites_channel_gates.py','report_dsprites_channel_gates.py',
    'verify_dsprites_gate_sites.py',
    'finalize_dsprites_channel_gates.py','verify_dsprites_channel_gates_delivery.py')


def finalize():
    """先检查数值封存和旧文件，再复制当日文档；清单不包含自身。"""
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve existing delivery'
    from run_dsprites_channel_gates import verify_seal
    verify_seal();audit=teacher.read(RUN/'independent_audit.json');assert audit['status']=='PASS'
    assert teacher.read(RUN/'analysis.json')['status']=='PASS'
    assert teacher.read(RUN/'single_sites/result.json')['status']=='PASS'
    assert teacher.read(RUN/'four_channel_terms.json')['status']=='PASS'
    assert len(teacher.read(RUN/'analysis.json')['primary'])==4
    assert len(teacher.read(RUN/'summary.json')['rows'])==576
    files=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in SCRIPTS]
    figures=[BASE/f'reports/figures/dsprites_channel_gates_20261011{kind}.{ext}' for kind in ('','_example') for ext in ('png','svg')]
    files +=[(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md'),(REPORT,RUN/'source/reports'/REPORT.name)]
    files +=[(p,RUN/'source/reports/figures'/p.name) for p in figures]
    for origin,destination in files:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(origin)==teacher.sha(destination),str(origin)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/n for n in SCRIPTS]+[REPORT]+figures+[BASE/'reports/dsprites_channel_gates_preflight_20261011.json']
    records=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),version='channel-gates-delivery-v1',
        files=records,mutable_progress='CURRENT_RESEARCH.md excluded; dated snapshot included',
        preserved_old_deliveries=audit['old_delivery_files']))
    sha=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(sha+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=sha)


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False))

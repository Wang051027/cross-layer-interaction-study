"""生成形状配对实验的代码、统计表、报告及文件哈希清单。"""
from __future__ import annotations

import json
import shutil
import dsprites_teacher as teacher

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_shape_donor_20261010'
REPORT=BASE/'reports/dSprites_形状配对与通道中介_2026-10-10.md'
SCRIPTS=('shape_donor_mechanism.py','run_dsprites_shape_donor.py','test_dsprites_shape_donor.py',
    'audit_dsprites_shape_donor.py','report_dsprites_shape_donor.py',
    'diagnose_dsprites_channel_combinations.py',
    'finalize_dsprites_shape_donor.py','verify_dsprites_shape_donor_delivery.py')


def finalize():
    """拒绝覆盖已有交付，全部路径限定在项目根目录内。"""
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve existing delivery'
    audit=teacher.read(RUN/'independent_audit.json');assert audit['status']=='PASS'
    assert len(teacher.read(RUN/'summary.json')['rows'])==456
    assert len(teacher.read(RUN/'analysis.json')['primary'])==6
    assert teacher.read(RUN/'combination_diagnostic.json')['status']=='PASS'
    files=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in SCRIPTS]
    figures=[BASE/f'reports/figures/dsprites_shape_donor_20261010{kind}.{x}'
             for kind in ('','_path') for x in ('png','svg')]
    files +=[(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md'),
             (REPORT,RUN/'source/reports'/REPORT.name)]
    files +=[(p,RUN/'source/reports/figures'/p.name) for p in figures]
    for origin,destination in files:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(origin)==teacher.sha(destination),str(origin)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/n for n in SCRIPTS]+[REPORT]+figures
    records=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),
        version='dsprites-shape-donor-delivery-v1',files=records,
        mutable_progress='CURRENT_RESEARCH.md excluded; dated source copy included',
        preserved_old_deliveries=audit['old_delivery_files']))
    sha=teacher.sha(RUN/'delivery_manifest.json')
    (RUN/'delivery_manifest.sha256').write_text(sha+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=sha)


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False))

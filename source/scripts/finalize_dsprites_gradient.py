"""保存本轮交付快照并生成哈希清单；数值记录与旧交付不修改。

先完成独立审计及报告，最后封存。mutable进度入口只保存当时副本，
防止今后更新进度时破坏本轮证据。已有交付清单拒绝覆盖。
"""
from __future__ import annotations

import json
import shutil
import dsprites_teacher as teacher

BASE=teacher.BASE
RUN=BASE/'runs/dsprites_gradient_20261009'
REPORT=BASE/'reports/dSprites_梯度变化与交互传递_2026-10-09.md'
SCRIPTS=('gradient_mechanism.py','run_dsprites_gradient.py','test_dsprites_gradient.py',
         'audit_dsprites_gradient.py','report_dsprites_gradient.py',
         'finalize_dsprites_gradient.py','verify_dsprites_gradient_delivery.py')


def finalize():
    """仅封存新目录及新报告；源快照若已存在，必须与原件完全相同。"""
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve existing delivery'
    audit=teacher.read(RUN/'independent_audit.json')
    assert audit['status']=='PASS'
    assert len(teacher.read(RUN/'summary.json')['rows'])==528
    teacher.read(RUN/'analysis.json')
    for name in SCRIPTS:
        origin=BASE/'scripts'/name;destination=RUN/'source/scripts'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(destination)==teacher.sha(origin),name
        else:shutil.copy2(origin,destination)
    copy_paths=[(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md'),
                (REPORT,RUN/'source/reports'/REPORT.name)]
    figures=[BASE/f'reports/figures/dsprites_gradient_20261009.{extension}' for extension in ('png','svg')]
    copy_paths +=[(p,RUN/'source/reports/figures'/p.name) for p in figures]
    for origin,destination in copy_paths:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(destination)==teacher.sha(origin),str(origin)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/name for name in SCRIPTS]+[REPORT]+figures
    records=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),
                            bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),
        version='dsprites-gradient-delivery-v1',files=records,
        mutable_progress='CURRENT_RESEARCH.md excluded; its dated copy included',
        preserved_old_deliveries=audit['old_delivery_files']))
    manifest_sha=teacher.sha(RUN/'delivery_manifest.json')
    (RUN/'delivery_manifest.sha256').write_text(manifest_sha+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=manifest_sha)


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False))

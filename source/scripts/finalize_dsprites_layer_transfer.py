"""汇总层间传递实验的输出文件和哈希。"""
from __future__ import annotations

import json
import shutil
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_layer_transfer_20261013'
REPORT=BASE/'reports/dSprites_源层组合与层间传递验证_2026-10-13.md'
SCRIPTS=('layer_transfer_mechanism.py','run_dsprites_layer_transfer.py','test_dsprites_layer_transfer.py',
    'audit_dsprites_layer_transfer.py','report_dsprites_layer_transfer.py',
    'finalize_dsprites_layer_transfer.py','verify_dsprites_layer_transfer_delivery.py')


def finalize():
    """已有快照先查一致；封存后不允许刷新源码、数组或报告。"""
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve completed delivery'
    from run_dsprites_layer_transfer import verify_protocol,verify_folder
    protocol=verify_protocol();audit=teacher.read(RUN/'independent_audit.json');analysis=teacher.read(RUN/'analysis.json')
    assert audit['status']==analysis['status']=='PASS'
    assert teacher.sha(RUN/audit['negative_fraction']['file'])==audit['negative_fraction']['sha256']
    assert len(analysis['primary'])==12 and len(analysis['examples'])==22
    assert analysis['table_rows']['coefficients']==13824 and analysis['table_rows']['transmission']==20736
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==576 and len(summary['decomposition'])==24
    assert (RUN/'README.md').is_file() and REPORT.is_file()
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):verify_folder(RUN/f'evaluation/seed_{seed}/alpha_{alpha}')
    figures=[BASE/name for name in analysis['figures']];assert len(figures)==6
    files=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in SCRIPTS]
    files +=[(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md'),(REPORT,RUN/'source/reports'/REPORT.name)]
    files +=[(p,RUN/'source/reports/figures'/p.name) for p in figures]
    for origin,destination in files:
        assert origin.is_file();destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(origin)==teacher.sha(destination),str(origin)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/n for n in SCRIPTS]+[REPORT]+figures+[BASE/'reports/dsprites_layer_transfer_preflight_20261013.json']
    records=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        assert path.name not in ('delivery_manifest.json','delivery_manifest.sha256')
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),version='layer-transfer-delivery-v1',files=records,
        protocol_sha256=teacher.sha(RUN/'protocol.json'),mutable_progress='CURRENT_RESEARCH.md excluded;dated snapshot included',
        preserved_old_deliveries=protocol['preservation']))
    sha=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(sha+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=sha)


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False))

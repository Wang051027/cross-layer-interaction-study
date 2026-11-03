"""全部独立核验和报告完成后封存三阶研究，保留前五轮交付。"""
from __future__ import annotations

import json
import shutil
import dsprites_teacher as teacher

BASE=teacher.BASE;RUN=BASE/'runs/dsprites_triple_gates_20261011'
REPORT=BASE/'reports/dSprites_三阶通道与ReLU抵消机制_2026-10-11.md'
SCRIPTS=('triple_gate_mechanism.py','run_dsprites_triple_gates.py','test_dsprites_triple_gates.py',
    'audit_dsprites_triple_gates.py','report_dsprites_triple_gates.py',
    'finalize_dsprites_triple_gates.py','verify_dsprites_triple_gates_delivery.py')


def finalize():
    """先验哈希与结果状态；只新增快照，不覆盖已封存的数值源码。"""
    assert not (RUN/'delivery_manifest.json').exists(),'Preserve completed delivery'
    from run_dsprites_triple_gates import verify_seal,verify_folder
    verify_seal();audit=teacher.read(RUN/'independent_audit.json');a=teacher.read(RUN/'analysis.json')
    assert audit['status']==a['status']=='PASS'
    check=teacher.read(RUN/'net_response_check.json');assert check['status']=='PASS' and check['conditions']==128
    for name,value in check['files'].items():assert teacher.sha(RUN/name)==value,name
    assert len(a['primary'])==4 and len(a['example_single_sites'])==64
    summary=teacher.read(RUN/'summary.json');assert len(summary['rows'])==1536 and len(summary['hierarchy'])==72
    assert a['table_rows']['all56_triples']==48384 and a['table_rows']['chosen_triples']==5184
    assert (RUN/'README.md').is_file() and REPORT.is_file()
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):verify_folder(RUN/f'evaluation/seed_{seed}/alpha_{alpha}')
    files=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in SCRIPTS]
    figures=[BASE/f'reports/figures/dsprites_triple_gates_20261011{suffix}.{ext}' for suffix in ('','_hierarchy','_example') for ext in ('png','svg')]
    files +=[(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md'),(REPORT,RUN/'source/reports'/REPORT.name)]
    files +=[(p,RUN/'source/reports/figures'/p.name) for p in figures]
    for origin,destination in files:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(origin)==teacher.sha(destination),str(origin)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/n for n in SCRIPTS]+[REPORT]+figures+[BASE/'reports/dsprites_triple_gates_preflight_20261011.json']
    records=[]
    for path in sorted(set(paths)):
        assert path.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(path.relative_to(BASE)).replace('\\','/'),bytes=path.stat().st_size,sha256=teacher.sha(path)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),version='triple-gates-delivery-v1',files=records,
        mutable_progress='CURRENT_RESEARCH.md excluded; dated snapshot included',preserved_old_deliveries=audit['old_delivery_files']))
    sha=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(sha+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(r['bytes'] for r in records),manifest_sha256=sha)


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False))

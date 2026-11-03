"""封存中间组合效应路由研究，保留全部七轮旧交付。"""
from __future__ import annotations
import json
import shutil
import dsprites_teacher as teacher
from run_dsprites_residual_routes import RUN,BASE,verify_seal,verify_folder

REPORT=BASE/'reports/dSprites_中间组合效应的通道与ReLU传递_2026-10-13.md'
CODE=('residual_route_mechanism.py','residual_route_path.py','run_dsprites_residual_routes.py','test_dsprites_residual_routes.py',
    'audit_dsprites_residual_routes.py','report_dsprites_residual_routes.py','finalize_dsprites_residual_routes.py','verify_dsprites_residual_routes_delivery.py','verify_dsprites_route_threshold.py','verify_dsprites_route_sixth_order.py')


def finalize():
    assert not (RUN/'delivery_manifest.json').exists();verify_seal();a=teacher.read(RUN/'analysis.json');audit=teacher.read(RUN/'independent_audit.json')
    assert a['status']==audit['status']=='PASS' and len(a['primary'])==36
    assert teacher.read(RUN/'threshold_check.json')['status']=='PASS'
    assert teacher.read(RUN/'threshold_sixth_order.json')['status']=='PASS'
    assert REPORT.exists() and (RUN/'README.md').exists() and len(a['figures'])==8
    for name,value in audit['example_response_files'].items():assert teacher.sha(RUN/name)==value
    for seed in teacher.SEEDS:
        for alpha in (1.,.5):verify_folder(RUN/f'evaluation/seed_{seed}/alpha_{alpha}')
    copies=[(BASE/'scripts'/n,RUN/'source/scripts'/n) for n in CODE]
    copies +=[(REPORT,RUN/'source/reports'/REPORT.name),(BASE/'CURRENT_RESEARCH.md',RUN/'source/CURRENT_RESEARCH.md')]
    copies +=[(BASE/name,RUN/'source/reports/figures'/name.split('/')[-1]) for name in a['figures']]
    for origin,destination in copies:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert teacher.sha(origin)==teacher.sha(destination)
        else:shutil.copy2(origin,destination)
    paths=[p for p in RUN.rglob('*') if p.is_file() and '.plot_cache' not in p.parts]
    paths +=[BASE/'scripts'/n for n in CODE]+[REPORT]+[BASE/name for name in a['figures']]+[BASE/'reports/dsprites_residual_routes_preflight_20261013.json']
    records=[]
    for p in sorted(set(paths)):
        assert p.resolve().is_relative_to(BASE.resolve())
        records.append(dict(path=str(p.relative_to(BASE)).replace('\\','/'),bytes=p.stat().st_size,sha256=teacher.sha(p)))
    teacher.dump(RUN/'delivery_manifest.json',dict(status='PASS',utc=teacher.now(),version='residual-routes-delivery-v1',files=records,
        mutable_progress='live CURRENT_RESEARCH.md excluded;dated snapshot included',old_delivery_files=audit['old_delivery_files']))
    digest=teacher.sha(RUN/'delivery_manifest.json');(RUN/'delivery_manifest.sha256').write_text(digest+'\n',encoding='ascii')
    return dict(status='PASS',files=len(records),bytes=sum(p['bytes'] for p in records),manifest_sha256=digest)


if __name__=='__main__':print(json.dumps(finalize()))

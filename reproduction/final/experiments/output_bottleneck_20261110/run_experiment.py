"""先封存全部预测，再测量真实输出与下游门控；只读已发布的模型、图像和旧规则。"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = '1a3b5af11efccc92f35d390200268484b9d6a44937453a8923e2adec486e7318'
FIT = 'da47c095308994d1f368db4998b540e05093980cc20b26b1adf464e67b6ef34c'
DELIVERY = '667dde7502fd589b9675f4cb8793950b56953a7b5df4cfa53629dccd0fe9abd5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def js(path):
    return json.loads(path.read_bytes())


def put(path,obj):
    tmp=path.with_suffix('.part')
    tmp.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    tmp.replace(path)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    tmp=path.with_suffix('.part')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def case_key(row):
    return int(row['seed']),float(row['strength']),int(row['case']),int(row['mask'])


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True,help='公开仓库根目录或原研究目录')
    ap.add_argument('--out',type=Path,required=True,help='新运行目录，必须不存在')
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=ap.parse_args()
    assets=args.assets.resolve()
    public=(assets/'reproduction/held_condition').is_dir()
    runs=assets/'reproduction/held_condition/runs' if public else assets/'runs'
    ref=assets/'reproduction/held_condition/reference' if public else runs/'dsprites_rule_discrimination_20261031'
    src=assets/'source/scripts' if public else assets/'scripts'
    sys.path.insert(0,str(src))
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import dsprites_teacher as teacher
    import torch
    import mechanism as mech
    rule=mech.rule
    teacher.configure()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    torch.set_grad_enabled(False)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    assert sha(ref/'protocol.json')==PROTOCOL and sha(ref/'fit_freeze.json')==FIT
    protocol=js(ref/'protocol.json');sealed=js(ref/'fit_freeze.json')
    assert sealed['previous']==PROTOCOL
    seeds=protocol['seeds']; strengths=(1.0,0.5)
    assert seeds==list(range(202609291,202609297))
    for name,digest in protocol['inputs'].items():
        assert sha(runs/name.removeprefix('runs/'))==digest
    fits={};orders={}
    for seed in seeds:
        name=f'fit/{seed}.json'
        assert sha(ref/name)==sealed['files'][name]
        fits[seed]=js(ref/name)
        orders[seed]=np.lexsort((np.arange(32),-np.array(fits[seed]['gain']))).tolist()
        assert orders[seed][:4]==fits[seed]['middle']
    pan=runs/'dsprites_shape_donor_20261010/panel/data.npz'
    with np.load(pan,allow_pickle=False) as z:
        panel={k:z[k] for k in ('images','pairs','labels','factors','official_ids')}
    assert panel['images'].shape==(36,64,64)
    split=rule.split_panel(panel)
    assert all(split[k]==protocol['split'][k]['cases'] for k in split)
    test=split['test']; assert len(test)==18
    code={f:sha(HERE/f) for f in ('run_experiment.py','mechanism.py','DESIGN.md')}
    shared={f:sha(src/f) for f in ('dsprites_triple_rule.py','dsprites_teacher.py','residual_teacher.py',
                                    'cross_layer_mechanism.py','shape_donor_mechanism.py','dsprites_mechanism.py')}
    put(out/'protocol.json',dict(utc=stamp(),prior_protocol_sha256=PROTOCOL,prior_fit_seal_sha256=FIT,
        inputs=protocol['inputs'],fit_files=sealed['files'],split=split,channel_orders=orders,
        budgets=mech.BUDGETS,strengths=strengths,study_files=code,shared_code=shared,
        software=dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,
                      device=args.device,device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU')))
    def load(seed):
        model=teacher.DSpritesCNN()
        weights=runs/f'dsprites_20260929/seed_{seed}/teacher.pt'
        model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        return model.double().to(args.device).eval().requires_grad_(False)
    def to_records(seed,strength,case,v):
        ri,di=panel['pairs'][case]
        arrays={k:mech.values(t) for k,t in v.items()}
        return [dict(seed=seed,strength=strength,case=case,mask=mask,
                     recipient_image=int(ri),donor_image=int(di),
                     **{k:float(a[i]) for k,a in arrays.items()}) for i,mask in enumerate(rule.TRIPLES)]
    def loop(measuring,expected=None,checks=None):
        rows=[]
        for seed in seeds:
            model=load(seed); bank=rule.source_bank(model,panel,test,args.device)
            for strength in strengths:
                for case in test:
                    ri,di=panel['pairs'][case];r,d=map(int,panel['labels'][[ri,di]])
                    inp=(model,bank[ri],bank[di],r,d,strength,fits[seed],orders[seed])
                    pred,ctx=mech.forecast(*inp)
                    pred_rows=to_records(seed,strength,case,pred)
                    if not measuring:
                        rows.extend(pred_rows);continue
                    for row in pred_rows:
                        stored=expected[case_key(row)]
                        for k in pred:
                            assert abs(float(stored[k])-row[k])<1e-9,(k,case)
                    measured,local=mech.measure(*inp,ctx,pred)
                    rows.extend(to_records(seed,strength,case,measured))
                    for k,v in local.items():
                        assert v<1e-8,(k,seed,case,v)
                        checks[k]=max(v,checks.get(k,0.0))
            print('measured' if measuring else 'predicted',seed,flush=True)
        assert len(rows)==len({case_key(v) for v in rows})==864
        return rows
    preds=loop(False)
    write_csv(out/'predictions.csv',preds)
    put(out/'prediction_seal.json',dict(utc=stamp(),protocol=sha(out/'protocol.json'),
        predictions=sha(out/'predictions.csv'),records=len(preds)))
    print('All 864 forecasts sealed; beginning measurements.',flush=True)
    seal=js(out/'prediction_seal.json')
    assert seal['predictions']==sha(out/'predictions.csv') and seal['protocol']==sha(out/'protocol.json')
    indexed={case_key(r):r for r in read_csv(out/'predictions.csv')}
    checks={}; measured=loop(True,indexed,checks)
    # 此时才打开上一轮答案。逐例重合比只比较汇总四舍五入后的数字更严格。
    assert sha(ref/'delivery_manifest.json')==DELIVERY
    oldhash={e['path']:e['sha256'] for e in js(ref/'delivery_manifest.json')['files']}
    for fname in ('cases.csv','patches.csv'):
        assert sha(ref/'tables'/fname)==oldhash['runs/dsprites_rule_discrimination_20261031/tables/'+fname]
    old={case_key(v):v for v in read_csv(ref/'tables/cases.csv')}
    patches={case_key(v):v for v in read_csv(ref/'tables/patches.csv') if v['patch']=='chosen4'}
    assert set(old)==set(patches)==set(indexed)
    for row in measured:
        key=case_key(row);p=indexed[key]
        comparisons={'old_y':(row['y'],old[key]['third_true']),
            'old_middle_norm':(row['true_norm_sq'],old[key]['middle_third_squared_norm']),
            'old_middle_error':(row['middle_error_sq_4'],old[key]['middle_error_sq_rule']),
            'old_rule':(p['fixed_4'],old[key]['third_rule']),
            'old_class_mean':(p['class_mean'],old[key]['third_class_mean']),
            'old_removed':(row['removed'],patches[key]['removed']),'old_kept':(row['keep'],patches[key]['keep'])}
        for name,(av,bv) in comparisons.items():
            av,bv=float(av),float(bv)
            assert np.isclose(av,bv,atol=1e-9,rtol=1e-9),(name,key,av,bv)
            checks[name]=max(checks.get(name,0),abs(av-bv))
    write_csv(out/'measurements.csv',measured)
    put(out/'audit.json',dict(status='PASS',records=len(measured),prior_checks_per_record=7,
        maxima=checks,prior_delivery_sha256=DELIVERY,freeze_before_measurement=True))
    put(out/'measurement_seal.json',dict(utc=stamp(),prediction_seal=sha(out/'prediction_seal.json'),
        measurements=sha(out/'measurements.csv'),audit=sha(out/'audit.json')))
    print('Measured and audited',len(measured),'records.',flush=True)


if __name__=='__main__':
    main()

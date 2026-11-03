"""新下游阻断的数值预检：原finish、边界核方向、联合控制与末层可加性。"""
from __future__ import annotations
import json
import numpy as np
import torch
import dsprites_teacher as teacher
import downstream_route_mechanism as core
from run_dsprites_downstream_routes import BASE,WINDOW,SHAPE,cohort,four_for_record
from run_dsprites_upstream_rescue import load,sources_for


def test():
    """六模型各首个冻结案例；额外边界激活替换仅作实现测试，不计研究样本。"""
    device=teacher.configure();panel=load(SHAPE/'panel/data.npz');records=cohort();errors=[];checked=0
    with torch.no_grad():
        for seed in teacher.SEEDS:
            model=teacher.load_teacher(seed,device);source=sources_for(model,panel,device)
            search=load(WINDOW/f'search/seed_{seed}/data.npz');units=load(WINDOW/f'search/seed_{seed}/units.npz')
            record=next(r for r in records if r['seed']==seed);fields,weight=four_for_record(model,source,panel,record,search,units)
            t0=core.trace(model,fields[2:3]);t1=core.trace(model,fields[3:4]);w=torch.as_tensor(weight,device=device,dtype=torch.float64)
            loc=core.locate(model,t0,t1,weight,42);out=core.evaluate(model,t0,t1,weight,loc,record['point'])
            selected=loc['orders'][0,0,:32];selected=selected[selected>=0]
            ids=np.unique(np.r_[selected,[record['point'],0,31,992,1023,31744,32767]])
            block=t1['u'].expand(len(ids),-1,-1,-1).clone()
            for i,p in enumerate(ids):block.flatten(1)[i,p]=t0['u'].flatten()[p]
            logits=model.readout(model.block3.finish(block,t1['x'].expand(len(ids),-1,-1,-1)).mean((2,3)))
            actual=((t1['logits']-logits)@w).cpu().numpy();errors.append(float(abs(actual-loc['point_effect'][0,ids]).max()));checked+=len(ids)
            # 正脉冲在边界/内部测试核翻转和零填充，即使原C变化在那里恰为0也能核验。
            boundary=np.array([0,31,992,1023,1024+16*32+16,32767]);reference={**t0,'u':t1['u'].clone()}
            reference['u'].flatten()[:] = t1['u'].flatten()
            reference['u'].flatten()[boundary] += .25
            predicted=core.first_point_effects(model,reference,t1,weight).cpu().numpy()
            block=t1['u'].expand(len(boundary),-1,-1,-1).clone()
            for i,p in enumerate(boundary):block.flatten(1)[i,p]+=.25
            actual=((t1['logits']-model.readout(model.block3.finish(block,t1['x'].expand(len(boundary),-1,-1,-1)).mean((2,3))))@w).cpu().numpy()
            errors.append(float(abs(actual-predicted[boundary]).max()));checked+=len(boundary)
            scores=loc['point_effect'][1];native=float((t1['logits']-t0['logits'])@w)
            for i,cmd in enumerate(core.commands(loc,1,record['point'])):
                blocked=float(out['last_logits'][i,0]@weight);errors.append(abs(float(t1['logits']@w)-blocked-scores[cmd['ids']].sum()))
            errors.append(abs(native-loc['channel_effect'][1].sum()))
            assert max(errors)<1e-9;print(json.dumps(dict(seed=seed,status='PASS',maximum_error=max(errors))),flush=True)
    path=BASE/'reports/dsprites_downstream_routes_preflight_20261025.json';assert not path.exists()
    result=dict(status='PASS',utc=teacher.now(),models=6,cohort_records=len(records),fullforward_singlepoint_checks=checked,maximum_error=max(errors),checks=['localexactscore vsoriginalfinish','nonzero boundaryimpulses numericalonly','allfirstblock skipidentity','alllastblockzero','lastjointadditivity','keepcomplement controls'])
    teacher.dump(path,result);return result


if __name__=='__main__':print(json.dumps(test()),flush=True)

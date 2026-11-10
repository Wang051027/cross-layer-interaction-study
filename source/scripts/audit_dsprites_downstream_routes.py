"""按九种空间位移累加单点作用，用原模型前向运算复核前 16 个位置、目标开关、边界点及联合阻断。"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.nn import functional as F
import dsprites_teacher as teacher
from run_dsprites_downstream_routes import BASE,RUN,WINDOW,PRIOR,SHAPE,verify_protocol,verify_selection,verify_folder,preservation,event
from run_dsprites_upstream_rescue import load
from audit_dsprites_coalition_context import actual_fields,native,ENDPOINTS

ERRORS={}
NAMES=('absolute','random0','random1','switch_only','stable_only','supporting','opposing')


def check(name,a,b,tol=1e-9):
    """检查形状与误差，保留最大误差而不只比总和。"""
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape)
    error=float(abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.
    assert error<tol,(name,error);ERRORS[name]=max(ERRORS.get(name,0.),error)


@torch.no_grad()
def shifted_scores(model,t0,t1,weight):
    """按输出相对源位置的九种位移累加贡献。"""
    b=model.block3;du=(t0['u']-t1['u'])[0];v=t1['v'][0]
    q=(torch.as_tensor(weight,device=v.device,dtype=v.dtype)@model.readout.weight)/1024
    scale=b.bn2.weight/torch.sqrt(b.bn2.running_var+b.bn2.eps)
    result=torch.zeros_like(du)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            ys=slice(max(0,-dy),min(32,32-dy));xs=slice(max(0,-dx),min(32,32-dx))
            yo=slice(max(0,dy),min(32,32+dy));xo=slice(max(0,dx),min(32,32+dx))
            base=v[:,yo,xo][:,None];coef=(scale[:,None]*b.conv2.weight[:,:,1-dy,1-dx])[:,:,None,None]
            shifted=base+coef*du[:,ys,xs][None]
            loss=(F.relu(base)-F.relu(shifted))*q[:,None,None,None]
            result[:,ys,xs]+=loss.sum(0)
    return result.flatten()


def own_commands(selection,layer,point):
    """独立恢复67个名单，核对数量与坐标。"""
    cmds=[]
    for order in range(7):
        available=selection['orders'][layer,order];available=available[available>=0]
        for count in (1,4,16,64,256,1024):cmds.append(available[:count])
    for order in range(3):
        available=selection['channel_orders'][layer,order];available=available[available>=0]
        for count in (1,2,4,8,16,32):cmds.append(np.asarray([1024*int(c)+p for c in available[:count] for p in range(1024)],dtype=int))
    active=abs(selection['delta_activation'][layer])>1e-10;switch=selection['switched'][layer];s=selection['point_effect'][layer]*int(selection['direction'])
    cmds += [np.array([],int),np.array([point],int) if layer==0 else np.array([],int),np.flatnonzero(active),np.flatnonzero(active&switch),np.flatnonzero(active&~switch),np.flatnonzero(active&(s>1e-10)),np.flatnonzero(active&(s<-1e-10))]
    assert len(cmds)==67;return cmds


def verify_orders(sel,seed):
    """规则核验使用冻结分数排序；分数本身另由独立数值算法核验。"""
    sort=lambda values,ids:np.asarray(sorted(ids,key=lambda i:(-abs(values[i]),i)),dtype=int)
    for layer in range(2):
        active=abs(sel['delta_activation'][layer])>1e-10;ids=np.flatnonzero(active);values=sel['point_effect'][layer];sw=sel['switched'][layer];direction=int(sel['direction'])
        rng=np.random.default_rng(seed+layer*10007)
        arrays=[sort(values,ids),rng.permutation(ids),rng.permutation(ids),sort(values,np.flatnonzero(active&sw)),sort(values,np.flatnonzero(active&~sw)),sort(values,np.flatnonzero(active&(direction*values>1e-10))),sort(values,np.flatnonzero(active&(direction*values<-1e-10)))]
        for i,arr in enumerate(arrays):
            expected=np.full(1024,-1,dtype=int);expected[:min(len(arr),1024)]=arr[:1024];check('point_orders',expected,sel['orders'][layer,i]);assert len(arr)==sel['order_counts'][layer,i]
        chids=np.flatnonzero(active.reshape(32,1024).any(1));arrays=[sort(sel['channel_effect'][layer],chids),rng.permutation(chids),rng.permutation(chids)]
        for i,arr in enumerate(arrays):
            expected=np.full(32,-1,dtype=int);expected[:len(arr)]=arr;check('channel_orders',expected,sel['channel_orders'][layer,i]);assert len(arr)==sel['channel_counts'][layer,i]


@torch.no_grad()
def through_original(model,x,u,skip):
    """通过原block.forward注入完整mid/skip值，原finish及BN2照常执行。"""
    block=model.block3;oldmid,oldfinish=block.mid,block.finish;record={}
    def capture(module,args,out):record['branch']=out
    hook=block.bn2.register_forward_hook(capture)
    def mid(value):return u
    def finish(value,ignored):
        out=oldfinish(value,skip);record['v']=record['branch']+skip;return out
    try:block.mid,block.finish=mid,finish;record['e']=block(x)
    finally:block.mid,block.finish=oldmid,oldfinish;hook.remove()
    record['logits']=model.readout(record['e'].mean((2,3)))
    return record


@torch.no_grad()
def audit_pair(model,x0,x1,t0,t1,weight,sel,raw,point,rngseed):
    """复核全地图、预算、原forward单点样本和所有联合阻断/恢复场。"""
    w=torch.as_tensor(weight,device=x0.device,dtype=x0.dtype);q=(model.readout.weight.T@w)[None,:,None,None]/1024
    f0=model.readout(t0['h'].mean((2,3)));f1=model.readout(t1['h'].mean((2,3)))
    expected_direction=1 if float((f1-f0)@w)>=0 else -1;assert int(sel['direction'])==expected_direction
    change=torch.stack((t1['u']-t0['u'],t1['h']-t0['h'])).reshape(2,-1)
    check('delta_activation',change,sel['delta_activation']);check('switched',torch.stack(((t1['z']>0)!=(t0['z']>0),(t1['v']>0)!=(t0['v']>0))).reshape(2,-1),sel['switched'])
    check('first_all_scores',shifted_scores(model,t0,t1,weight),sel['point_effect'][0])
    lastmap=((t1['h']-t0['h'])*q).flatten();check('last_all_scores',lastmap,sel['point_effect'][1])
    verify_orders(sel,rngseed)
    # 对每个名单前16点、原j和边界进行原forward单点抽查；不称全图原forward枚举。
    sample=np.unique(np.r_[sel['orders'][0,:,:16].ravel(),[point,0,31,992,1023,31744,32767]])
    sample=sample[sample>=0];us=t1['u'].expand(len(sample),-1,-1,-1).clone()
    for i,p in enumerate(sample):us.flatten(1)[i,p]=t0['u'].flatten()[p]
    single=through_original(model,x1.expand(len(sample),-1,-1,-1),us,x1.expand(len(sample),-1,-1,-1))
    check('sample_original_single_scores',((f1-single['logits'])@w),sel['point_effect'][0,sample])
    ch=t1['u'].expand(32,-1,-1,-1).clone()
    for c in range(32):ch[c,c]=t0['u'][0,c]
    channel=through_original(model,x1.expand(32,-1,-1,-1),ch,x1.expand(32,-1,-1,-1))
    check('first_channel_scores',(f1-channel['logits'])@w,sel['channel_effect'][0]);check('last_channel_scores',lastmap.reshape(32,1024).sum(1),sel['channel_effect'][1])
    routes=[];route_maps=[]
    for us in (t0['u'],t1['u']):
        rr=through_original(model,torch.cat((x0,x1)),us.expand(2,-1,-1,-1),torch.cat((x0,x1)))
        routes.append(rr['logits']);route_maps.append(((rr['e']-t0['h'])*q).sum((2,3)))
    check('route_logits',torch.stack(routes),raw['route_logits']);check('route_channels',torch.stack(route_maps),raw['route_channels']);check('native_logits',torch.cat((f0,f1)),raw['native_logits'])
    for layer in range(2):
        cmds=own_commands(sel,layer,point);u0=t0['u'] if layer==0 else t0['h'];u1=t1['u'] if layer==0 else t1['h'];values=[]
        for ids in cmds:
            blocked=u1.clone();kept=u0.clone();blocked.flatten(1)[:,ids]=u0.flatten(1)[:,ids];kept.flatten(1)[:,ids]=u1.flatten(1)[:,ids];values.extend((blocked[0],kept[0]))
        values=torch.stack(values);logits=[];maps=[];pn=[];flips=[];spatial=[]
        for skip in ((x0,x1) if layer==0 else (None,)):
            if layer==0:
                rec=through_original(model,x1.expand(134,-1,-1,-1),values,skip.expand(134,-1,-1,-1));e=rec['e'];logits.append(rec['logits'].reshape(67,2,3))
                flips.append(((rec['v']>0)!=(t0['v']>0)).sum((1,2,3)).reshape(67,2))
            else:e=values;logits.append(model.readout(e.mean((2,3))).reshape(67,2,3))
            signed=((e-t0['h'])*q).reshape(67,2,32,32,32);maps.append(signed.sum((3,4)))
            pn.append(torch.stack((torch.where(signed>0,signed,0.).sum((2,3,4)),torch.where(signed<0,signed,0.).sum((2,3,4))),-1));spatial.append(signed)
        if layer==0:
            check('first_logits',torch.stack(logits,1),raw['first_logits']);check('first_channels',torch.stack(maps,1),raw['first_channels']);check('first_pn',torch.stack(pn,1),raw['first_pn']);check('first_second_flips',torch.stack(flips,1),raw['first_second_flips'])
        else:
            check('last_logits',logits[0],raw['last_logits']);check('last_channels',maps[0],raw['last_channels']);check('last_pn',pn[0],raw['last_pn'])
        for j,index in enumerate((4,44)):
            vector=spatial[-1][index,0].flatten().cpu().numpy();ids=raw['residual_top_ids'][layer,j]
            assert len(set(ids.tolist()))==128
            check('residual_top_values',vector[ids],raw['residual_top_values'][layer,j])
            assert np.min(abs(vector[ids]))>=np.partition(abs(vector),-128)[-128]-1e-9
    return len(sample),32,268,134


def audit():
    """全部37记录都审核，完成后再保全十三份历史交付。"""
    assert not (RUN/'independent_audit.json').exists();p=verify_protocol();verify_selection();panel=load(SHAPE/'panel/data.npz');device=teacher.configure();counts=np.zeros(4,dtype=int)
    with torch.no_grad():
        for seed in teacher.SEEDS:
            model=teacher.load_teacher(seed,device)
            source=torch.cat([model.block1(model.stem(torch.as_tensor(im[None,None],device=device,dtype=torch.float64))) for im in panel['images']])
            search=load(WINDOW/f'search/seed_{seed}/data.npz');units=load(WINDOW/f'search/seed_{seed}/units.npz')
            for record in [r for r in p['cohort'] if r['seed']==seed]:
                case=record['case'];m=(3,15).index(record['mask']);recipient,donor=panel['pairs'][case]
                before,afters=actual_fields(model,source,recipient,donor,record['mask'],{k:v[case,m] for k,v in units.items()},search['choices'][case,m])
                after=afters[ENDPOINTS.index(record['endpoint'])];C=record['C'];only=before.clone();only[C]=after[C];outside=after.clone();outside[C]=before[C]
                fields=torch.stack((before,only,outside,after));full=native(model.block3,fields)
                weight=np.zeros(3);weight[panel['labels'][donor]]=1.;weight[panel['labels'][recipient]]=-1.
                sf=RUN/f"discovery/record_{record['record']:03d}";ef=RUN/f"evaluation/record_{record['record']:03d}";verify_folder(sf);verify_folder(ef);sel=load(sf/'data.npz');raw=load(ef/'data.npz')
                for ctx,(off,on) in enumerate(((0,1),(2,3))):
                    t0={k:v[off:off+1] for k,v in full.items()};t1={k:v[on:on+1] for k,v in full.items()}
                    rngseed=seed+2026102500+case*97+record['mask']*13+record['endpoint']*1009+ctx*11
                    counts+=audit_pair(model,fields[off:off+1],fields[on:on+1],t0,t1,weight,{k:v[ctx] for k,v in sel.items()},{k:v[ctx] for k,v in raw.items()},record['point'],rngseed)
                event('audited',record=record['record'],seed=seed)
    preserved=preservation();result=dict(status='PASS',utc=teacher.now(),records=37,all_first_point_scores=37*2*32768,all_last_point_scores=37*2*32768,
        sampled_original_single_points=int(counts[0]),original_channel_blocks=int(counts[1]),first_joint_fields=int(counts[2]),last_joint_fields=int(counts[3]),maximum_errors=ERRORS,old_delivery_entries=sum(r['files'] for r in preserved.values()),
        independence='all scoremaps via different shifted-output algorithm; fulloriginalforward selectedtop16/boundary/j sample; alljointpatches via originalforward+mid/skip hooks; frozen scores independentlyranked; no production numericalcore')
    teacher.dump(RUN/'independent_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':audit()

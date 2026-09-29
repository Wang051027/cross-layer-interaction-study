"""dSprites受控迁移：保留原残差架构，原生64×64输入，重新训练3类输出。

训练只读取train/validation；六个种子都保留，模型冻结后才做机制评价。
旧Fashion检查点、源码和交付记录不修改。
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,random,shutil,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from residual_teacher import SmallResidualCNN

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/dsprites_20260929'
DATA=BASE/'data/dsprites/controlled_v1'
SEEDS=list(range(202609291,202609297))
CODE=['scripts/prepare_dsprites.py','scripts/dsprites_teacher.py','reports/dSprites_迁移评估协议_2026-09-29.md']


def sha(path):
    """流式哈希，用于新旧来源的完整性承诺。"""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    """读取UTF-8记录。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    """只写当前阶段指定的新文件，禁止JSON中出现隐式NaN。"""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def now():
    """统一记录UTC，便于核对训练、选点、评价的先后。"""
    return datetime.now(timezone.utc).isoformat()


def event(kind,**details):
    """实时记录进展；日志不进入选择算法。"""
    value=dict(utc=now(),event=kind,**details)
    with (RUN/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def configure():
    """固定随机数以外的数值执行选项；训练float32，干预检查float64。"""
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class DSpritesCNN(SmallResidualCNN):
    """原网络前缀完全同结构，输出层改为3类；GAP自然支持64×64输入。"""
    def __init__(self):
        super().__init__()
        self.readout=nn.Linear(32,3)


def data(split,dtype=torch.float64,device='cpu'):
    """官方dSprites像素已是0/1，不能套用Fashion的除以255预处理。"""
    pixels=np.load(DATA/f'{split}_images.npy',allow_pickle=False)
    assert pixels.shape[1:]==(64,64) and set(np.unique(pixels)).issubset({0,1})
    labels=np.load(DATA/f'{split}_labels.npy',allow_pickle=False)
    return torch.as_tensor(pixels[:,None],dtype=dtype,device=device),torch.as_tensor(labels,dtype=torch.long,device=device)


def verify_old():
    """保全三份历史交付清单，记录其哈希，避免无意改变旧结论的证据。"""
    sources=[('runs/cross_layer_20260928/delivery_manifest.json','files'),
             ('runs/adaptive_mediator_20260928/delivery_manifest.json','files'),
             ('audit/joint_response_manifest_20260925.json','entries')]
    result={}
    for manifest,key in sources:
        for entry in read(BASE/manifest)[key]:assert sha(BASE/entry['path'])==entry['sha256'],entry['path']
        result[manifest]=sha(BASE/manifest)
    return result


def initialize(device):
    """冻结训练设置、划分、旧来源和迁移协议，拒绝覆盖已有运行。"""
    assert not RUN.exists(),'Preserve existing run'
    old=verify_old();split=read(DATA/'manifest.json')
    for filename,expected in split['files'].items():assert sha(DATA/filename)==expected
    RUN.mkdir()
    for name in CODE:
        destination=RUN/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/name,destination)
    sources={name:sha(BASE/name) for name in CODE}
    dependencies=['scripts/residual_teacher.py','scripts/external_experiment.py','scripts/cross_layer_mechanism.py',
                  'scripts/adaptive_mediator.py','scripts/research_runtime.py']
    protocol=dict(version='dsprites-controlled-transfer-v1',utc=now(),seeds=SEEDS,
        dataset='official pinned dSprites; controlled central-position-domain subset',
        model='same SmallResidualCNN blocks at native64; readout32->3; fresh weights',
        teacher=dict(epochs=20,batch=256,optimizer='AdamW',lr=.002,min_lr=.00002,weight_decay=.0001,
                     schedule='cosine across20epochs',augmentation='none',selection='lowest validation cross entropy; keep earliest tie'),
        quality_gate=dict(mean_validation_accuracy=.95,minimum_validation_accuracy=.90,
                          failure='report all models; do not claim task-competent mechanism if gate fails'),
        split_manifest_sha256=sha(DATA/'manifest.json'),data_files=split['files'],sources=sources,
        dependencies={name:sha(BASE/name) for name in dependencies},old_manifests=old,
        calibration_access=False,environment=dict(torch=torch.__version__,numpy=np.__version__,device=device,
            gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
    dump(RUN/'protocol.json',protocol)
    (RUN/'protocol.sha256').write_text(sha(RUN/'protocol.json')+'\n',encoding='ascii')
    event('training_protocol_locked',sha256=sha(RUN/'protocol.json'))


def verify_protocol():
    """轻量核验约114MB的准备数据和源码；不反复加载3GB全数据数组。"""
    p=read(RUN/'protocol.json')
    assert sha(RUN/'protocol.json')==(RUN/'protocol.sha256').read_text().strip()
    assert sha(DATA/'manifest.json')==p['split_manifest_sha256']
    for name,expected in {**p['sources'],**p['dependencies']}.items():assert sha(BASE/name)==expected,name
    for name,expected in p['sources'].items():assert sha(RUN/'source'/name)==expected,name
    for name,expected in p['data_files'].items():assert sha(DATA/name)==expected,name
    for name,expected in p['old_manifests'].items():assert sha(BASE/name)==expected,name
    return p


@torch.no_grad()
def validation_metrics(model,images,labels,batch=256):
    """选择检查点只看验证损失，同时记录每类准确率，不接触评价面板。"""
    model.eval();total=0.;correct=0;per_class=np.zeros((3,2),dtype=int)
    for start in range(0,len(images),batch):
        x,y=images[start:start+batch],labels[start:start+batch]
        logits=model(x);prediction=logits.argmax(1)
        total+=float(F.cross_entropy(logits,y,reduction='sum'));correct+=int((prediction==y).sum())
        for c in range(3):
            mask=y==c;per_class[c,0]+=int(((prediction==y)&mask).sum());per_class[c,1]+=int(mask.sum())
    return dict(cross_entropy=total/len(images),accuracy=correct/len(images),
                per_class_accuracy=(per_class[:,0]/per_class[:,1]).tolist())


def train(device):
    """六个固定种子各训练20轮，按验证损失选权重；训练中断的原件不覆盖。"""
    cfg=verify_protocol()['teacher']
    assert not (RUN/'model_freeze.json').exists()
    images,labels=data('train',torch.float32,device)
    val_images,val_labels=data('validation',torch.float32,device)
    for seed in SEEDS:
        folder=RUN/f'seed_{seed}'
        if (folder/'teacher_summary.json').exists():
            assert sha(folder/'teacher.pt')==read(folder/'teacher_summary.json')['checkpoint_sha256'];continue
        assert not folder.exists(),'Preserve incomplete training'
        folder.mkdir();start=time.perf_counter()
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device=='cuda':torch.cuda.manual_seed_all(seed)
        model=DSpritesCNN().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(seed)
        history=[];best=float('inf');best_epoch=None
        for epoch in range(1,cfg['epochs']+1):
            lr=cfg['min_lr']+.5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*(epoch-1)/(cfg['epochs']-1)))
            for group in optimizer.param_groups:group['lr']=lr
            model.train();order=torch.randperm(len(images),generator=generator).to(device)
            loss_sum=torch.zeros((),device=device)
            for start_batch in range(0,len(images),cfg['batch']):
                ix=order[start_batch:start_batch+cfg['batch']]
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(images[ix]),labels[ix]);loss.backward();optimizer.step()
                loss_sum+=loss.detach()*len(ix)
            metrics=validation_metrics(model,val_images,val_labels)
            row=dict(epoch=epoch,lr=lr,train_ce=float(loss_sum/len(images)),**metrics)
            history.append(row)
            if metrics['cross_entropy']<best:
                best=metrics['cross_entropy'];best_epoch=epoch
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'teacher.pt')
            dump(folder/'history.json',history)
            event('training_epoch',seed=seed,**row)
        selected=history[best_epoch-1]
        summary=dict(seed=seed,best_epoch=best_epoch,validation=selected,parameters=sum(p.numel() for p in model.parameters()),
                     checkpoint_sha256=sha(folder/'teacher.pt'),seconds=time.perf_counter()-start,test_used_for_selection=False)
        dump(folder/'teacher_summary.json',summary);event('teacher_complete',**summary)
        del model,optimizer
        if device=='cuda':torch.cuda.empty_cache()
    summaries=[read(RUN/f'seed_{seed}'/'teacher_summary.json') for seed in SEEDS]
    accuracies=[s['validation']['accuracy'] for s in summaries]
    gate=float(np.mean(accuracies))>=.95 and min(accuracies)>=.90
    freeze=dict(utc=now(),protocol_sha256=sha(RUN/'protocol.json'),quality_gate=bool(gate),
                mean_validation_accuracy=float(np.mean(accuracies)),minimum_validation_accuracy=min(accuracies),
                checkpoints={f'seed_{s}/teacher.pt':sha(RUN/f'seed_{s}'/'teacher.pt') for s in SEEDS})
    dump(RUN/'model_freeze.json',freeze)
    (RUN/'model_freeze.sha256').write_text(sha(RUN/'model_freeze.json')+'\n',encoding='ascii')
    event('all_models_frozen',**freeze)


def load_teacher(seed,device='cpu',double=True):
    """只加载新dSprites权重；拒绝沿用旧Fashion分类器权重冒充迁移完成。"""
    model=DSpritesCNN()
    model.load_state_dict(torch.load(RUN/f'seed_{seed}'/'teacher.pt',map_location='cpu',weights_only=True))
    return (model.double() if double else model).to(device).eval()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['init','train','verify'],required=True)
    args=parser.parse_args();device=configure()
    if args.stage=='init':initialize(device)
    elif args.stage=='train':train(device)
    else:verify_protocol();print(json.dumps({'status':'PASS','old_manifests':verify_old()}))

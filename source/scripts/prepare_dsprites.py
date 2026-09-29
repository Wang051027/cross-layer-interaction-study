"""从官方固定提交获取dSprites，安全解压数字数组并建立受控形状分类划分。

不反序列化NPZ中的object元数据；只读取公开的数值图像和潜因子数组。
所有图像保留原生64×64，三种形状标签由latents_classes直接取得。
"""
from pathlib import Path
import argparse, hashlib, json, shutil, urllib.request, zipfile
from datetime import datetime, timezone
import numpy as np

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'data/dsprites'
COMMIT='fa310c66517cfc1939d77fe17c725154efc97127'
FILE='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
ROOT=f'https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/{COMMIT}/'


def sha(path):
    """流式计算哈希，大图像数组不会整体载入内存。"""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    """保存来源、划分和统计；拒绝非标准浮点值。"""
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def fetch():
    """只下载三个已命名的官方文件，完成后保留提交号和原始文件哈希。"""
    DATA.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'upstream.json'
    if manifest.exists():
        info=json.loads(manifest.read_text(encoding='utf-8'))
        for name,digest in info['files'].items():assert sha(DATA/name)==digest
        print(json.dumps({'status':'PASS','download':'already verified'}),flush=True);return
    hashes={}
    for name in (FILE,'README.md','LICENSE'):
        dest=DATA/name
        assert not dest.exists(),f'Preserve incomplete download for inspection: {dest}'
        req=urllib.request.Request(ROOT+name,headers={'User-Agent':'dsprites-research'})
        with urllib.request.urlopen(req,timeout=60) as response,dest.open('wb') as output:
            shutil.copyfileobj(response,output,1024*1024)
        hashes[name]=sha(dest)
        print(json.dumps({'downloaded':name,'bytes':dest.stat().st_size,'sha256':hashes[name]}),flush=True)
    dump(manifest,dict(commit=COMMIT,url=ROOT,utc=datetime.now(timezone.utc).isoformat(),files=hashes))


def extract():
    """限定成员名复制，不调用extractall，不加载metadata.npy中的pickle。"""
    manifest=DATA/'arrays.json'
    if manifest.exists():
        print(json.dumps({'status':'PASS','arrays':'already prepared'}),flush=True);return
    entries={}
    with zipfile.ZipFile(DATA/FILE) as archive:
        for member in ('imgs.npy','latents_classes.npy','latents_values.npy'):
            dest=DATA/member
            assert not dest.exists(),f'Preserve incomplete array: {dest}'
            with archive.open(member) as source,dest.open('wb') as target:
                shutil.copyfileobj(source,target,8*1024*1024)
            a=np.load(dest,mmap_mode='r',allow_pickle=False)
            entries[member]=dict(sha256=sha(dest),shape=list(a.shape),dtype=str(a.dtype),bytes=dest.stat().st_size)
            print(json.dumps({'array':member,**entries[member]}),flush=True)
    assert entries['imgs.npy']['shape']==[737280,64,64]
    assert entries['latents_classes.npy']['shape']==[737280,6]
    dump(manifest,entries)


def ids(shape,scale,angle,x,y):
    """公开笛卡尔积顺序：shape、scale、orientation、x、y；color只有一个值。"""
    return (((np.asarray(shape)*6+scale)*40+angle)*32+x)*32+y


def group_ids(scale,angle,x,y):
    """一个因素情境含三个形状，划分时整组处理，防止形状配对被拆散。"""
    return np.asarray(ids(np.arange(3),scale,angle,x,y),dtype=np.int64)


def translated(image,dx,dy):
    """零填充整数平移；dx是列方向、dy是行方向，不使用插值。"""
    output=np.zeros_like(image)
    ys=slice(max(0,-dy),min(64,64-dy));xs=slice(max(0,-dx),min(64,64-dx))
    output[slice(max(0,dy),min(64,64+dy)),slice(max(0,dx),min(64,64+dx))]=image[ys,xs]
    return output


def digest_image(image):
    """像素完全一致的图像即视为重复，不仅核对不同的潜变量编号。"""
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def prepare():
    """固定匹配因素面板，并在分组划分之外额外排除跨集合像素重复。

    主面板位置(15,15)，6大小×8角度×3形状=144图。
    平移面板从官方中心图构造严格向右4像素的副本，供步幅2对齐控制。
    官方位置潜变量的变化可能伴随栅格化差异，不能直接冒充严格像素平移。
    训练只使用位置索引8..23的受控子域；不声称覆盖完整dSprites分布。
    """
    out=DATA/'controlled_v1'
    assert not out.exists(),'Preserve prepared split'
    image=np.load(DATA/'imgs.npy',mmap_mode='r',allow_pickle=False)
    factors=np.load(DATA/'latents_classes.npy',mmap_mode='r',allow_pickle=False)
    assert np.array_equal(factors.min(0),[0,0,0,0,0,0])
    assert np.array_equal(factors.max(0),[0,2,5,39,31,31])
    dev_contexts=[(s,a,15,15) for s in range(6) for a in (3,8,13,18,23,28,33,38)]
    eval_contexts=[(s,a,15,15) for s in range(6) for a in (1,6,11,16,21,26,31,36)]
    dev=np.concatenate([group_ids(*c) for c in dev_contexts])
    evaluation=np.concatenate([group_ids(*c) for c in eval_contexts])
    moved=evaluation.copy()
    for ix in np.concatenate([dev,evaluation,moved]):
        _,s,scale,a,x,y=factors[ix]
        assert int(ids(s,scale,a,x,y))==int(ix)
    dev_hashes={digest_image(image[i]) for i in dev}
    moved_images=np.stack([translated(image[i],4,0) for i in evaluation])
    assert np.array_equal(moved_images.sum((1,2)),image[evaluation].sum((1,2))),'Translation clipped foreground'
    evaluation_hashes={digest_image(image[i]) for i in evaluation}|{digest_image(v) for v in moved_images}
    assert not dev_hashes & evaluation_hashes,'Development/evaluation pixel overlap'
    shifts=np.tile([4,0],(len(evaluation),1))
    forbidden_contexts=set(dev_contexts+eval_contexts)
    candidates=[(s,a,x,y) for s in range(6) for a in range(40) for x in range(8,24) for y in range(8,24)
                if (s,a,x,y) not in forbidden_contexts]
    rng=np.random.default_rng(2026092901);rng.shuffle(candidates)
    # 开发面板属于训练集；验证集独立，所有三形状始终作为一组。
    train=list(map(int,dev));validation=[]
    train_hashes=set(dev_hashes);val_hashes=set();rejected=0
    for context in candidates:
        group=group_ids(*context)
        hashes={digest_image(image[i]) for i in group}
        if hashes&evaluation_hashes:rejected+=1;continue
        if len(train)<24576:
            train.extend(map(int,group));train_hashes.update(hashes)
        elif len(validation)<3072:
            if hashes&train_hashes:rejected+=1;continue
            validation.extend(map(int,group));val_hashes.update(hashes)
        else:break
    assert len(train)==24576 and len(validation)==3072
    assert not train_hashes&val_hashes and not (train_hashes|val_hashes)&evaluation_hashes
    out.mkdir()
    np.savez_compressed(out/'indices.npz',train=np.array(train),validation=np.array(validation),development=dev,
        evaluation=evaluation,translated=moved,shifts=np.array(shifts))
    for name,indices in [('train',train),('validation',validation),('development',dev),('evaluation',evaluation),('translated',moved)]:
        selected=moved_images if name=='translated' else np.asarray(image[indices],dtype=np.uint8)
        np.save(out/f'{name}_images.npy',selected,allow_pickle=False)
        np.save(out/f'{name}_labels.npy',np.asarray(factors[indices,1],dtype=np.int64),allow_pickle=False)
        np.save(out/f'{name}_factors.npy',np.asarray(factors[indices],dtype=np.int64),allow_pickle=False)
    manifest=dict(utc=datetime.now(timezone.utc).isoformat(),task='3-class shape classification at native64x64',
        class_names=['square','ellipse','heart'],split='grouped nuisance contexts; exact pixel overlap purged between train/validation/evaluation',
        training=24576,validation=3072,development=144,evaluation=144,translated=144,
        development_is_training_subset=True,position_domain=[8,23],evaluation_position=[15,15],
        translation='derived exact +4 column pixels with zero fill; no clipping; translated IDs/factors refer to source originals, not an official new latent row',
        shifts_unique=np.unique(shifts,axis=0).tolist(),rejected_candidate_groups=rejected,
        unique_pixels=dict(train=len(train_hashes),validation=len(val_hashes),evaluation_and_translation=len(evaluation_hashes)),
        source_commit=COMMIT,source_archive_sha256=sha(DATA/FILE),
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    dump(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['fetch','extract','prepare'],required=True)
    stage=parser.parse_args().stage
    {'fetch':fetch,'extract':extract,'prepare':prepare}[stage]()

"""独立检查 dSprites 来源行、标签、因素分组与跨划分像素重合。

只读官方数值数组和当前划分；派生平移单独核验，不把它当成官方新行。
"""
import hashlib,itertools
from pathlib import Path
import numpy as np
from dsprites_teacher import BASE,RUN,DATA,read,dump,sha,now


def audit():
    """逐行对照来源，按内容与潜因子两条独立途径排查划分错误。"""
    root=DATA.parent;manifest=read(DATA/'manifest.json');upstream=read(root/'upstream.json')
    for name,expected in upstream['files'].items():assert sha(root/name)==expected,name
    for name,expected in manifest['files'].items():assert sha(DATA/name)==expected,name
    official=np.load(root/'imgs.npy',mmap_mode='r',allow_pickle=False)
    classes=np.load(root/'latents_classes.npy',mmap_mode='r',allow_pickle=False)
    with np.load(DATA/'indices.npz') as archive:ids={k:archive[k] for k in archive.files}
    hashes={};contexts={};rows={}
    for split in ('train','validation','development','evaluation','translated'):
        images=np.load(DATA/f'{split}_images.npy');factors=np.load(DATA/f'{split}_factors.npy')
        labels=np.load(DATA/f'{split}_labels.npy');index=ids[split]
        assert np.array_equal(classes[index],factors) and np.array_equal(factors[:,1],labels)
        if split!='translated':
            for start in range(0,len(index),512):assert np.array_equal(images[start:start+512],official[index[start:start+512]])
        else:
            parent=np.load(DATA/'evaluation_images.npy')
            assert np.array_equal(index,ids['evaluation'])
            assert np.array_equal(images[:,:,4:],parent[:,:,:-4]) and not images[:,:,:4].any()
            assert np.array_equal(images.sum((1,2)),parent.sum((1,2)))
        hashes[split]={hashlib.sha256(image.tobytes()).digest() for image in images}
        contexts[split]={tuple(row[2:].tolist()) for row in factors}
        assert len(contexts[split])*3==len(index)
        rows[split]=dict(rows=len(index),unique_pixels=len(hashes[split]),contexts=len(contexts[split]),
                         class_counts=np.bincount(labels,minlength=3).tolist())
    assert set(ids['development']).issubset(set(ids['train']))
    overlap={}
    for a,b in [('train','validation'),('train','evaluation'),('validation','evaluation'),
                ('train','translated'),('validation','translated')]:
        n=len(hashes[a]&hashes[b]);assert n==0,(a,b,n)
        overlap[a+'__'+b]=n
    for a,b in itertools.combinations(('train','validation','evaluation'),2):
        assert not contexts[a]&contexts[b],(a,b)
    assert rows['train']['unique_pixels']==manifest['unique_pixels']['train']
    assert len(hashes['evaluation']|hashes['translated'])==manifest['unique_pixels']['evaluation_and_translation']
    output=dict(status='PASS',utc=now(),source_sha256=sha(Path(__file__)),rows=rows,pixel_overlap=overlap,
        official_rows_and_labels_verified=True,group_disjoint_train_validation_evaluation=True,
        development_is_training_subset=True,translation_verified=True,
        combined_evaluation_translation_unique_pixels=len(hashes['evaluation']|hashes['translated']),
        caveat='Excluding exact duplicate pixels does not remove correlation between nearby factors or symmetric shapes.')
    dump(RUN/'independent_data_audit.json',output)
    print(output)


if __name__=='__main__':audit()

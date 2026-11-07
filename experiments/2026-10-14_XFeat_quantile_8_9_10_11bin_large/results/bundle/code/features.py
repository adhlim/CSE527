"""Frozen XFeat extraction and dataset helpers. No LUT or bit-selection training."""

from pathlib import Path

import subprocess,sys,zipfile

import numpy as np

SEED=20261011

XFEAT_COMMIT='e92685f57f8318b18725c5c8c0bd28c7fe188d9a'

def project(k,H):
    p=np.column_stack([k,np.ones(len(k))])@H.T
    return (p[:,:2]/p[:,2:]).astype(np.float32)

def load_xfeat(root,top_k=1000):
    import torch
    root=Path(root);repo=root/'xfeat_repo'
    if not repo.exists():
        subprocess.run(['git','clone','https://github.com/verlab/accelerated_features.git',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'checkout',XFEAT_COMMIT],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!=XFEAT_COMMIT:raise RuntimeError('XFeat checkout differs from pinned experiment commit.')
    sys.path.insert(0,str(repo.resolve()))
    from modules.xfeat import XFeat
    torch.set_num_threads(2);torch.manual_seed(SEED)
    model=XFeat(top_k=top_k)
    for p in model.parameters():p.requires_grad_(False)
    return model,commit

def extract(model,image):
    import torch,cv2
    t=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)[None].float().to(model.dev)/255.
    with torch.inference_mode():f=model.detectAndCompute(t,top_k=1000)[0]
    d=f['descriptors'].cpu().numpy().astype(np.float32)
    d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    return f['keypoints'].cpu().numpy().astype(np.float32),np.ascontiguousarray(d)

def make_pairs(images,model,warps,seed):
    import cv2
    rng=np.random.default_rng(seed);pairs=[]
    for name,img in images:
        h,w=img.shape[:2];k1,d1=extract(model,img)
        for r in range(warps):
            aff=cv2.getRotationMatrix2D((w/2,h/2),float(rng.uniform(-20,20)),float(rng.uniform(.88,1.08)))
            aff[:,2]+=rng.uniform(-.06,.06,2)*[w,h]
            H=np.vstack([aff,[rng.uniform(-7e-5,7e-5),rng.uniform(-7e-5,7e-5),1]]).astype(np.float32)
            im=cv2.warpPerspective(img,H,(w,h),borderMode=cv2.BORDER_CONSTANT)
            im=np.clip(im.astype(np.float32)*rng.uniform(.82,1.12)+rng.uniform(-12,18),0,255).astype(np.uint8)
            k2,d2=extract(model,im)
            if min(len(d1),len(d2))<2:continue
            pairs.append(dict(name=f'{name}/warp{r}',d1=d1,d2=d2,k1=k1,k2=k2,H=H))
        print('extract',name,len(d1),flush=True)
    return pairs

def resize_decode(raw,max_side=640):
    import cv2
    im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise ValueError('Unreadable image')
    h,w=im.shape[:2];s=min(1,max_side/max(h,w));return cv2.resize(im,(max(64,int(w*s)//32*32),max(64,int(h*s)//32*32)))

def read_coco(zip_path,count=24):
    with zipfile.ZipFile(zip_path) as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith('.jpg'))
        if len(names)<count:raise ValueError('Not enough COCO images')
        chosen=np.random.default_rng(SEED).choice(len(names),count,False)
        return [(Path(names[i]).name,resize_decode(z.read(names[i]))) for i in chosen]

def save_pair_cache(path,pairs):
    payload={}
    for i,p in enumerate(pairs):
        for k,v in p.items():payload[f'{i}_{k}']=np.array(v)
    payload['count']=np.array(len(pairs));np.savez_compressed(path,**payload)

def read_pair_cache(path):
    with np.load(path,allow_pickle=False) as z:
        return [{k:(str(z[f'{i}_{k}']) if k=='name' else z[f'{i}_{k}']) for k in ['name','d1','d2','k1','k2','H']} for i in range(int(z['count']))]

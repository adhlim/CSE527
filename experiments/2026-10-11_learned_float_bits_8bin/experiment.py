"""Learn fixed float-bit positions and ordered 8-bin LUTs; compare exact L1 implementations.
Training is discrete greedy selection + supervised coordinate descent, not neural training.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, platform, subprocess, sys, time, urllib.request, zipfile
import numpy as np

SEED=20261011
XFEAT_COMMIT='e92685f57f8318b18725c5c8c0bd28c7fe188d9a'

def patterns(x, positions):
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    c=np.zeros(u.shape,np.uint16)
    for r,p in enumerate(positions): c |= (((u >> int(p)) & 1).astype(np.uint16) << r)
    return c

def bins_from_edges(x, edges):
    return (x[:,:,None] >= edges[None,:,:]).sum(-1).astype(np.uint8)

def initial_lut(x, positions, targets):
    c=patterns(x,positions); size=1<<len(positions)
    lut=np.zeros((64,size),np.uint8); counts=np.zeros((64,size),np.int32)
    for d in range(64):
        n=np.bincount(c[:,d],minlength=size)
        s=np.bincount(c[:,d],weights=targets[:,d],minlength=size)
        # Unseen patterns deterministically map to the training dimension's mean bin.
        means=np.full(size,float(targets[:,d].mean()))
        np.divide(s,n,out=means,where=n>0)
        lut[d]=np.clip(np.rint(means),0,7).astype(np.uint8); counts[d]=n
    return lut,counts

def encode_np(x,positions,lut): return lut[np.arange(64)[None,:],patterns(x,positions)]

def triplet_metrics(a,p,n):
    dp=np.abs(a.astype(np.int16)-p.astype(np.int16)).sum(1)
    dn=np.abs(a.astype(np.int16)-n.astype(np.int16)).sum(1)
    return dp,dn,float((dp<dn).mean()),float(np.maximum(0,8+dp-dn).mean()/64)

def train_codec(triplets, calibration, out, nbits=8, search_samples=2000, sweeps=2):
    rng=np.random.default_rng(SEED); order=rng.permutation(len(triplets[0]))
    if len(order)<100: raise ValueError('Need at least 100 geometrically labelled triplets.')
    cut=max(1,int(len(order)*.2)); inner=order[:cut]; fit=order[cut:]
    fit=fit[:search_samples]; inner=inner[:search_samples]
    a,p,n=[np.ascontiguousarray(x[fit],np.float32) for x in triplets]
    va,vp,vn=[np.ascontiguousarray(x[inner],np.float32) for x in triplets]
    # All quantile teachers, bit selection and LUT updates use training scenes only.
    edges=np.quantile(calibration,np.arange(1,8)/8,axis=0).T.astype(np.float32)
    x=np.concatenate([a,p,n]); target=bins_from_edges(x,edges)
    vtarget=bins_from_edges(np.concatenate([va,vp,vn]),edges)
    selected=[]; history=[]; started=time.perf_counter()
    for step in range(nbits):
        candidates=[]
        for bit in range(32):
            if bit in selected: continue
            positions=selected+[bit]; lut,_=initial_lut(x,positions,target)
            codes=encode_np(np.concatenate([va,vp,vn]),positions,lut)
            ca,cp,cn=np.split(codes,3)
            _,_,rank,loss=triplet_metrics(ca,cp,cn)
            distill=float(np.square(codes.astype(np.float32)-vtarget).mean())
            score=loss+.03*distill
            candidates.append((score,bit,rank,loss,distill))
        score,bit,rank,loss,distill=min(candidates)
        selected.append(bit)
        row=dict(step=step+1,bit=int(bit),score=score,inner_rank=rank,margin_loss=loss,distill_mse=distill)
        history.append(row); print('bit selection',row,flush=True)
        (out/'selection_history.json').write_text(json.dumps(history,indent=2))
    # Refit LUT on all sampled training triplets, then supervise its ordered labels.
    take=order[:min(len(order),8000)]
    a,p,n=[np.ascontiguousarray(v[take],np.float32) for v in triplets]
    x=np.concatenate([a,p,n]); target=bins_from_edges(x,edges)
    lut,counts=initial_lut(x,selected,target)
    ca,cp,cn=[patterns(v,selected) for v in (a,p,n)]
    ya,yp,yn=[lut[np.arange(64)[None,:],c].astype(np.int16) for c in (ca,cp,cn)]
    ta,tp,tn=np.split(target,3); dp,dn,_,_=triplet_metrics(ya,yp,yn)
    refinement=[]; regularizer=.03
    for sweep in range(sweeps):
        changes=0
        for d in range(64):
            for c in rng.permutation(np.flatnonzero(counts[d])):
                ids=np.flatnonzero((ca[:,d]==c)|(cp[:,d]==c)|(cn[:,d]==c))
                old=int(lut[d,c]); aa=ya[ids,d]; pp=yp[ids,d]; nn=yn[ids,d]
                oldp=np.abs(aa-pp); oldn=np.abs(aa-nn)
                maska=ca[ids,d]==c; maskp=cp[ids,d]==c; maskn=cn[ids,d]==c
                best=(float('inf'),old,None,None,None,None,None)
                for val in range(8):
                    na=np.where(maska,val,aa); np_=np.where(maskp,val,pp); nneg=np.where(maskn,val,nn)
                    ndp=dp[ids]-oldp+np.abs(na-np_); ndn=dn[ids]-oldn+np.abs(na-nneg)
                    # This is the affected part of the global triplet loss + teacher regularization.
                    reg=((na-ta[ids,d])**2+(np_-tp[ids,d])**2+(nneg-tn[ids,d])**2).sum()
                    score=float(np.maximum(0,8+ndp-ndn).sum()+regularizer*reg/3)
                    if score<best[0]-1e-9 or (abs(score-best[0])<1e-9 and val==old):
                        best=(score,val,ndp,ndn,na,np_,nneg)
                if best[1]!=old:
                    changes+=1;lut[d,c]=best[1];dp[ids]=best[2];dn[ids]=best[3]
                    ya[ids,d]=best[4];yp[ids,d]=best[5];yn[ids,d]=best[6]
        row=dict(sweep=sweep+1,changed_entries=changes,train_rank=float((dp<dn).mean()),
                 margin_loss=float(np.maximum(0,8+dp-dn).mean()/64))
        refinement.append(row);print('LUT training',row,flush=True)
    result=dict(positions=selected,nbits=nbits,levels=8,selection_history=history,refinement=refinement,
                seconds=time.perf_counter()-started,training_triplets=len(triplets[0]),lut_training_triplets=len(a),
                method='greedy bit selection with internal training-only holdout; supervised integer LUT coordinate descent',
                lut_scope='per descriptor dimension, shared across all images and keypoints',
                teacher='training-scene per-dimension quantile bins; regularization only',
                unseen_pattern_policy='training dimension mean bin; never fitted on evaluation data')
    np.savez(out/'learned_codec.npz',positions=np.array(selected,np.int32),lut=lut,counts=counts,edges=edges)
    (out/'training.json').write_text(json.dumps(result,indent=2))
    return selected,lut,edges,result

class Engine:
    def __init__(self,root):
        root=Path(root); lib=root/'bit_codec.so'
        subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(root/'bit_codec.cpp'),'-o',str(lib)],check=True)
        self.lib=ctypes.CDLL(str(lib.resolve())); self.flags=['-O3','-march=native'];self.root=root
        P=ctypes.c_void_p; I=ctypes.c_int
        for name,args in {'encode':[P,I,P,I,P,P],'pack3':[P,I,P],'unpack3':[P,I,P],
                          'packthermo':[P,I,P],'absdist':[P,P,I,I,P],'thermodist':[P,P,I,I,P],
                          'absmnn':[P,P,I,I,P,P],'thermomnn':[P,P,I,I,P,P]}.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=None
    @staticmethod
    def ptr(a):return a.ctypes.data_as(ctypes.c_void_p)
    def encode(self,x,pos,lut):
        x=np.ascontiguousarray(x,np.float32);pos=np.ascontiguousarray(pos,np.int32);lut=np.ascontiguousarray(lut,np.uint8)
        out=np.empty(x.shape,np.uint8);self.lib.encode(self.ptr(x),len(x),self.ptr(pos),len(pos),self.ptr(lut),self.ptr(out));return out
    def pack3(self,x):
        x=np.ascontiguousarray(x,np.uint8);out=np.empty((len(x),24),np.uint8)
        self.lib.pack3(self.ptr(x),len(x),self.ptr(out));return out
    def unpack3(self,x):
        x=np.ascontiguousarray(x,np.uint8);out=np.empty((len(x),64),np.uint8)
        self.lib.unpack3(self.ptr(x),len(x),self.ptr(out));return out
    def thermo(self,x):
        x=np.ascontiguousarray(x,np.uint8);out=np.empty((len(x),7),np.uint64)
        self.lib.packthermo(self.ptr(x),len(x),self.ptr(out));return out
    def distance(self,a,b,thermo=False):
        out=np.empty((len(a),len(b)),np.uint16)
        getattr(self.lib,'thermodist' if thermo else 'absdist')(self.ptr(a),self.ptr(b),len(a),len(b),self.ptr(out));return out
    def mnn(self,a,b,thermo=False):
        if len(a)==0 or len(b)==0:return np.empty(0,int),np.empty(0,int)
        row=np.empty(len(a),np.int32);col=np.empty(len(b),np.int32)
        getattr(self.lib,'thermomnn' if thermo else 'absmnn')(self.ptr(a),self.ptr(b),len(a),len(b),self.ptr(row),self.ptr(col))
        ids=np.flatnonzero(col[row]==np.arange(len(a)));return ids,row[ids]

def self_test(engine):
    rng=np.random.default_rng(SEED)
    a=rng.integers(0,8,(35,64),dtype=np.uint8);b=rng.integers(0,8,(31,64),dtype=np.uint8)
    # Force ties to verify deterministic first-index MNN rules.
    a[1]=a[0];b[1]=b[0]
    assert np.array_equal(a,engine.unpack3(engine.pack3(a)))
    expected=np.abs(a[:,None,:].astype(np.int16)-b[None,:,:].astype(np.int16)).sum(-1)
    assert np.array_equal(engine.distance(a,b),expected)
    at,bt=engine.thermo(a),engine.thermo(b)
    assert np.array_equal(engine.distance(at,bt,True),expected)
    row=expected.argmin(1);col=expected.argmin(0);ids=np.flatnonzero(col[row]==np.arange(len(a)))
    for aa,bb,t in [(a,b,False),(at,bt,True)]:
        i,j=engine.mnn(aa,bb,t);assert np.array_equal(i,ids) and np.array_equal(j,row[ids])
    # Exhaustively test all bin pairs and the sign bit extraction.
    q=np.repeat(np.arange(8,dtype=np.uint8)[:,None],64,axis=1)
    assert np.array_equal(engine.distance(engine.thermo(q),engine.thermo(q),True),64*np.abs(np.arange(8)[:,None]-np.arange(8)[None,:]))
    x=rng.normal(size=(20,64)).astype(np.float32);pos=[31,25,24,23,22,21,20,19]
    lut=rng.integers(0,8,(64,256),dtype=np.uint8)
    assert np.array_equal(engine.encode(x,pos,lut),encode_np(x,pos,lut))
    print('PASS: bit extraction, 3-bit roundtrip, all-bin ABS/popcount equality, full matrices, MNN including ties',flush=True)

def project(k,H):
    p=np.column_stack([k,np.ones(len(k))])@H.T
    return (p[:,:2]/p[:,2:]).astype(np.float32)

def geometric_pairs(pair):
    from scipy.spatial.distance import cdist
    d=cdist(project(pair['k1'],pair['H']),pair['k2']);back=cdist(project(pair['k2'],np.linalg.inv(pair['H'])),pair['k1'])
    j=d.argmin(1);i=np.arange(len(j));good=(d[i,j]<3)&(back[j,i]<3)&(back.argmin(1)[j]==i)
    return i[good],j[good],d

def build_triplets(pairs):
    result=[[],[],[]];cal=[];rng=np.random.default_rng(SEED)
    for pair in pairs:
        i,j,geo=geometric_pairs(pair)
        if not len(i):continue
        if len(i)>300:
            sub=rng.choice(len(i),300,False);i,j=i[sub],j[sub]
        # Hard negatives: most cosine-similar target at least 10 px from projected anchor.
        sim=pair['d1'][i]@pair['d2'].T;sim[geo[i]<=10]=-np.inf
        keep=np.isfinite(sim.max(1));i,j,sim=i[keep],j[keep],sim[keep]
        if not len(i):continue
        neg=sim.argmax(1)
        for dest,x in zip(result,[pair['d1'][i],pair['d2'][j],pair['d2'][neg]]):dest.append(x)
        cal.extend([pair['d1'],pair['d2']])
    if not result[0]:raise RuntimeError('No valid geometric triplets; inspect images and extraction.')
    return tuple(np.concatenate(x) for x in result),np.concatenate(cal)

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

def read_local_images(folder,count=24):
    files=sorted(p for p in Path(folder).glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'])
    if len(files)<count:raise ValueError(f'Need {count} distinct images; found {len(files)}')
    return [(p.name,resize_decode(p.read_bytes())) for p in files[:count]]

def save_pair_cache(path,pairs):
    payload={}
    for i,p in enumerate(pairs):
        for k,v in p.items():payload[f'{i}_{k}']=np.array(v)
    payload['count']=np.array(len(pairs));np.savez_compressed(path,**payload)

def read_pair_cache(path):
    with np.load(path,allow_pickle=False) as z:
        return [{k:(str(z[f'{i}_{k}']) if k=='name' else z[f'{i}_{k}']) for k in ['name','d1','d2','k1','k2','H']} for i in range(int(z['count']))]

def benchmark(fn,repeats=9):
    for _ in range(3):fn()
    vals=[]
    for _ in range(repeats):
        s=time.perf_counter();fn();vals.append((time.perf_counter()-s)*1000)
    return float(np.median(vals))

def evaluate(pairs,model,engine,pos,lut,edges,out,label='heldout'):
    import torch
    records=[]
    for p in pairs:
        a,b=[engine.encode(p[k],pos,lut) for k in ['d1','d2']]
        aa,bb=engine.pack3(a),engine.pack3(b);at,bt=engine.thermo(a),engine.thermo(b)
        mi,mj=engine.mnn(a,b);ti,tj=engine.mnn(at,bt,True)
        assert np.array_equal(mi,ti) and np.array_equal(mj,tj)
        assert np.array_equal(engine.distance(a[:32],b[:32]),engine.distance(at[:32],bt[:32],True))
        fa,fb=torch.from_numpy(p['d1']),torch.from_numpy(p['d2'])
        def official():
            with torch.inference_mode():return model.match(fa,fb,min_cossim=-1)
        fi,fj=official();fi,fj=fi.numpy(),fj.numpy()
        qa,qb=[bins_from_edges(p[k],edges) for k in ['d1','d2']];qi,qj=engine.mnn(qa,qb)
        for method,i,j in [('official_float_cosine',fi,fj),('fixed_quantile_abs',qi,qj),('learned_bits_abs',mi,mj),('learned_bits_thermometer',ti,tj)]:
            err=np.linalg.norm(project(p['k1'][i],p['H'])-p['k2'][j],axis=1)
            gt1,gt2,_=geometric_pairs(p);gt=set(zip(gt1.tolist(),gt2.tolist()));hits=len(gt&set(zip(i.tolist(),j.tolist())))
            records.append(dict(dataset=label,pair=p['name'],method=method,n=len(a),m=len(b),mutual=len(i),correct=int((err<3).sum()),
                precision=float((err<3).mean()*100) if len(err) else 0.,median_px=float(np.median(err)) if len(err) else None,
                geometric_pair_recall=float(hits/len(gt)) if gt else None))
        def encode_abs():
            a,b=[engine.encode(p[k],pos,lut) for k in ['d1','d2']];return engine.mnn(a,b)
        def encode_packed3():
            a,b=[engine.pack3(engine.encode(p[k],pos,lut)) for k in ['d1','d2']]
            return engine.mnn(engine.unpack3(a),engine.unpack3(b))
        def encode_thermo():
            a,b=[engine.thermo(engine.encode(p[k],pos,lut)) for k in ['d1','d2']];return engine.mnn(a,b,True)
        timings=dict(pair=p['name'],n=len(a),m=len(b),official_float_mnn_ms=benchmark(official),
            bin_encode_both_ms=benchmark(lambda:(engine.encode(p['d1'],pos,lut),engine.encode(p['d2'],pos,lut))),
            abs_preencoded_mnn_ms=benchmark(lambda:engine.mnn(a,b)),
            abs_packed3_preencoded_mnn_including_unpack_ms=benchmark(lambda:engine.mnn(engine.unpack3(aa),engine.unpack3(bb))),
            thermometer_preencoded_mnn_ms=benchmark(lambda:engine.mnn(at,bt,True)),
            abs_encode_mnn_ms=benchmark(encode_abs),abs_encode_pack_unpack_mnn_ms=benchmark(encode_packed3),
            thermometer_encode_pack_mnn_ms=benchmark(encode_thermo))
        for r in records[-4:]:r['timings']=timings
        print('eval',label,p['name'],[(r['method'],round(r['precision'],2),r['mutual']) for r in records[-4:]],flush=True)
    (out/f'{label}_results.json').write_text(json.dumps(records,indent=2));return records

def hpatches_pairs(zip_path,model,max_sequences=8):
    import cv2
    pairs=[]
    with zipfile.ZipFile(zip_path) as z:
        names=set(z.namelist());refs=sorted(n for n in names if n.endswith('/1.ppm'))[:max_sequences]
        for ref in refs:
            directory=ref.rsplit('/',1)[0];im1=cv2.imdecode(np.frombuffer(z.read(ref),np.uint8),cv2.IMREAD_COLOR)
            k1,d1=extract(model,im1)
            for index in range(2,7):
                imname=f'{directory}/{index}.ppm';hname=f'{directory}/H_1_{index}'
                if imname not in names or hname not in names:continue
                im2=cv2.imdecode(np.frombuffer(z.read(imname),np.uint8),cv2.IMREAD_COLOR);k2,d2=extract(model,im2)
                H=np.loadtxt(z.open(hname)).astype(np.float32)
                if min(len(d1),len(d2))>=2:pairs.append(dict(name=f'{directory}/1-{index}',d1=d1,d2=d2,k1=k1,k2=k2,H=H))
    if not pairs:raise ValueError('HPatches archive missing expected sequence images/homographies')
    return pairs

def run(root,coco_zip=None,image_dir=None,count=24,warps=3,search_samples=2000,sweeps=2,hpatches=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);out=root/'run';out.mkdir(exist_ok=True)
    engine=Engine(root);self_test(engine);model,commit=load_xfeat(root)
    images=sorted(read_coco(coco_zip,count) if coco_zip else read_local_images(image_dir,count),key=lambda v:v[0])
    order=np.random.default_rng(SEED).permutation(count);nt=max(2,int(count*.67));nv=max(1,int(count*.16))
    groups=dict(train=order[:nt],validation=order[nt:nt+nv],test=order[nt+nv:])
    manifest=dict(seed=SEED,source='COCO val2017',groups={k:[images[i][0] for i in ids] for k,ids in groups.items()},
                  xfeat_commit=commit,top_k=1000,max_side=640,warps_per_image=warps,
                  note='image-disjoint synthetic homographies; COCO val2017 is our training source, not a COCO official benchmark')
    manifest['cache_fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()+b''.join(hashlib.sha256(im.tobytes()).digest() for _,im in images)).hexdigest()
    manifest_path=out/'data_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('cache_fingerprint')!=manifest['cache_fingerprint']:
        raise RuntimeError('Existing pair cache belongs to a different dataset/config. Use a new experiment root or remove only its run/*_pairs.npz and data_manifest.json.')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    pairs={}
    extraction=time.perf_counter()
    for offset,(name,ids) in enumerate(groups.items()):
        cache=out/f'{name}_pairs.npz'
        if cache.exists():pairs[name]=read_pair_cache(cache)
        else:
            pairs[name]=make_pairs([images[i] for i in ids],model,warps,SEED+offset);save_pair_cache(cache,pairs[name])
    extraction=time.perf_counter()-extraction
    triplets,cal=build_triplets(pairs['train'])
    np.savez_compressed(out/'training_triplets.npz',anchor=triplets[0],positive=triplets[1],negative=triplets[2])
    pos,lut,edges,training=train_codec(triplets,cal,out,search_samples=search_samples,sweeps=sweeps)
    assert np.array_equal(engine.encode(cal[:100],pos,lut),encode_np(cal[:100],pos,lut))
    records=[]
    for name in ['validation','test']:records+=evaluate(pairs[name],model,engine,pos,lut,edges,out,name)
    hp_status='not run'
    if hpatches:
        records+=evaluate(hpatches_pairs(hpatches,model),model,engine,pos,lut,edges,out,'hpatches');hp_status='run, never used for training'
    report=dict(training=training,manifest=manifest,extraction_seconds=extraction,hpatches=hp_status,
        environment=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,cpp_flags=engine.flags,cpp_threads=1,torch_threads=2),
        storage_bytes=dict(float32=256,selected_bit_intermediate=64,bin_uint8_working=64,bin_packed3=24,thermometer_packed=56,
                           shared_lut=lut.nbytes,selected_positions=np.array(pos,np.int32).nbytes),
        accuracy_equivalence='ABS and thermometer must match exactly, including deterministic tie-breaking',
        timing_scope='CPU encode/pack/matching; XFeat extraction excluded and reported separately; no claimed SIMD4 optimization',
        records=records)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    import csv
    with (out/'accuracy.csv').open('w',newline='') as f:
        rows=[{k:v for k,v in r.items() if k!='timings'} for r in records];w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    with (out/'timing.csv').open('w',newline='') as f:
        rows=[dict(dataset=r['dataset'],**r['timings']) for r in records if r['method']=='learned_bits_abs']
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print('Saved:',out,flush=True);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).parent));ap.add_argument('--coco-zip');ap.add_argument('--image-dir')
    ap.add_argument('--count',type=int,default=24);ap.add_argument('--warps',type=int,default=3);ap.add_argument('--search-samples',type=int,default=2000)
    ap.add_argument('--sweeps',type=int,default=2);ap.add_argument('--hpatches');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test(Engine(args.root))
    else:run(args.root,args.coco_zip,args.image_dir,args.count,args.warps,args.search_samples,args.sweeps,args.hpatches)

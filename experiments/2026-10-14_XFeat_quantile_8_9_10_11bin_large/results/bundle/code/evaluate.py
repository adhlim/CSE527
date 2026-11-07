from pathlib import Path
import json,time,hashlib,platform,sys,subprocess
import numpy as np
import pandas as pd
import torch
import features as f
from optimized import ThermometerVariant
from binary import BinaryVariant,verify_binary

BINS=(8,9,10,11,16)
SEED=20261014
REPEATS=9
WARMUP=3

def median_ms(fn):
 for _ in range(WARMUP):fn()
 values=[]
 for _ in range(REPEATS):
  s=time.perf_counter();fn();values.append(1000*(time.perf_counter()-s))
 return float(np.median(values))

def verify(v):
 rng=np.random.default_rng(SEED+v.bins)
 for n,m in [(0,3),(3,0),(1,1),(13,19),(8,3)]:
  a=rng.normal(size=(n,64)).astype(np.float32);b=rng.normal(size=(m,64)).astype(np.float32)
  for tied in [False,True]:
   if tied:a[:]=0;b[:]=0
   pa,pb=v.encode(a),v.encode(b)
   assert np.array_equal(pa,v.encode(a,False));assert np.array_equal(pb,v.encode(b,False))
   ca,cb=v.codes(a),v.codes(b)
   d=np.abs(ca[:,None,:].astype(np.int16)-cb[None,:,:].astype(np.int16)).sum(-1)
   assert np.array_equal(d,v.matrix(pa,pb))
   ii,jj=v.match(pa,v.soa(pb))
   if n and m:
    row=d.argmin(1);col=d.argmin(0);ids=np.flatnonzero(col[row]==np.arange(n))
    assert np.array_equal(ii,ids);assert np.array_equal(jj,row[ids])
   else:assert len(ii)==len(jj)==0
 for a in [v.et,np.nextafter(v.et,-np.inf),np.nextafter(v.et,np.inf)]:
  assert np.array_equal(v.encode(a),v.encode(a,False))
 print('kernel verified',v.bins,flush=True)

def prepare_coco(zip_path,model,root,count=256,calibration_count=192):
 root=Path(root);root.mkdir(exist_ok=True,parents=True)
 path=root/'manifest.json'
 if path.exists():
  man=json.loads(path.read_text())
  if man['count']!=count or man['calibration_count']!=calibration_count or man['seed']!=SEED:raise ValueError('Use a new root for changed data settings')
 if (root/'calibration.npy').exists() and (root/'test_pairs.npz').exists():
  return np.load(root/'calibration.npy'),f.read_pair_cache(root/'test_pairs.npz'),json.loads(path.read_text())
 images=sorted(f.read_coco(zip_path,count),key=lambda x:x[0])
 ids=np.random.default_rng(SEED).permutation(count);train=ids[:calibration_count];test=ids[calibration_count:]
 man=dict(source='COCO val2017',count=count,calibration_count=calibration_count,test_source_count=len(test),seed=SEED,image_selection_seed=f.SEED,calibration_files=[images[i][0] for i in train],test_files=[images[i][0] for i in test],top_k=1000,max_side=640,test_warps_per_image=3,calibration_augmentation=False)
 assert not set(man['calibration_files'])&set(man['test_files'])
 path.write_text(json.dumps(man,indent=2));desc=[];start=time.perf_counter()
 for num,i in enumerate(train):
  _,d=f.extract(model,images[i][1]);desc.append(d)
  if num%24==0:print('COCO calibration extraction',num+1,len(train),flush=True)
 calibration=np.ascontiguousarray(np.concatenate(desc),np.float32)
 np.save(root/'calibration.npy',calibration)
 pairs=f.make_pairs([images[i] for i in test],model,3,SEED+1)
 f.save_pair_cache(root/'test_pairs.npz',pairs)
 man['test_pairs']=len(pairs);man['calibration_descriptors']=len(calibration);man['extraction_seconds']=time.perf_counter()-start
 man['calibration_sha256']=hashlib.sha256(calibration.tobytes()).hexdigest()
 path.write_text(json.dumps(man,indent=2));return calibration,pairs,man

def fit_variants(calibration,root):
 root=Path(root);times={};edges={}
 for bins in BINS:
  s=time.perf_counter()
  edges[bins]=np.ascontiguousarray(np.quantile(calibration,np.arange(1,bins)/bins,axis=0).T,np.float32)
  times[bins]=time.perf_counter()-s
 old_path=root/'fixed_quantile_edges.npz'
 if old_path.exists():
  with np.load(old_path) as previous:
   for k in (8,9,10,11):
    if f'bins_{k}' in previous:assert np.array_equal(previous[f'bins_{k}'],edges[k]),'Existing quantization changed'
 np.savez(old_path,**{f'bins_{k}':v for k,v in edges.items()})
 variants={k:ThermometerVariant(k,edges[k]) for k in BINS}
 for v in variants.values():verify(v)
 for k in (8,16):
  binary=BinaryVariant(k,edges[k]);verify_binary(binary,variants[k]);variants[f'binary_{k}']=binary
 return variants,times

def evaluate(pairs,variants,model,out,label):
 out=Path(out);out.mkdir(exist_ok=True,parents=True);rows=[]
 for ix,p in enumerate(pairs):
  a=np.ascontiguousarray(p['d1'],np.float32);b=np.ascontiguousarray(p['d2'],np.float32)
  methods=[('official_float_cosine',None)]+[(f'quantile_{k}bin_thermometer_AVX2',variants[k]) for k in BINS]+[(f'quantile_{k}bin_binary_{3 if k==8 else 4}bit_ABS',variants[f'binary_{k}']) for k in (8,16)]
  matches={}
  # Rotate benchmark order to avoid always placing the same method first.
  shift=ix%len(methods);methods=methods[shift:]+methods[:shift]
  for name,v in methods:
   if v is None:
    ta,tb=torch.from_numpy(a),torch.from_numpy(b)
    def full():
     with torch.inference_mode():return model.match(ta,tb,min_cossim=-1)
    i,j=full();i=np.asarray(i);j=np.asarray(j);enc_ms=0.;match_ms=median_ms(full);total_ms=median_ms(full);bins=0;byte=256;working=256;representation='float32'
   elif getattr(v,'kind',None)=='binary':
    pa,pb=v.encode(a),v.encode(b);i,j=v.match(pa,pb);bins=v.bins;byte=v.bytes_per_kp;working=v.working_bytes_per_kp;representation='ordinary_binary'
    def full():return v.match(v.encode(a),v.encode(b))
    enc_ms=median_ms(lambda:(v.encode(a),v.encode(b)))
    match_ms=median_ms(lambda:v.match(pa,pb));total_ms=median_ms(full)
   else:
    pa,pb=v.encode(a),v.encode(b);bs=v.soa(pb)
    i,j=v.match(pa,bs);bins=v.bins;byte=8*(bins-1);working=byte;representation='thermometer'
    def full():return v.match(v.encode(a),v.soa(v.encode(b)))
    enc_ms=median_ms(lambda:(v.encode(a),v.soa(v.encode(b))))
    match_ms=median_ms(lambda:v.match(pa,bs));total_ms=median_ms(full)
    # For every pair check scalar bit-distance reference MNN against SIMD4.
    n,m=len(pa),len(pb)
    ab=np.empty(n,np.int32);ba=np.empty(m,np.int32)
    if n and m:
     import ctypes
     P=ctypes.c_void_p
     v.lib.fused(pa.ctypes.data_as(P),pb.ctypes.data_as(P),n,m,0,2,ab.ctypes.data_as(P),ba.ctypes.data_as(P))
     ids=np.flatnonzero(ba[ab]==np.arange(n));assert np.array_equal(i,ids);assert np.array_equal(j,ab[ids])
   matches[name]=(i,j)
   err=np.linalg.norm(f.project(p['k1'][i],p['H'])-p['k2'][j],axis=1)
   category='illumination' if p['name'].split('/')[0].startswith('i_') else 'viewpoint' if p['name'].split('/')[0].startswith('v_') else 'synthetic'
   rows.append(dict(dataset=label,category=category,pair=p['name'],method=name,bins=bins,n=len(a),m=len(b),mutual=len(i),correct_1px=int((err<1).sum()),correct_3px=int((err<3).sum()),correct_5px=int((err<5).sum()),encode_ms=enc_ms,preencoded_matching_ms=match_ms,total_ms=total_ms,descriptor_bytes=byte,working_bytes=working,representation=representation))
  for k in (8,16):
   thermo=matches[f'quantile_{k}bin_thermometer_AVX2'];binary=matches[f'quantile_{k}bin_binary_{3 if k==8 else 4}bit_ABS']
   for x,y in zip(thermo,binary):assert np.array_equal(x,y),'Encoding representation changed MNN'
  if ix%20==0:print('evaluate',label,ix+1,len(pairs),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'per_pair.csv',index=False)
 def summarize(g):
  result=dict(pairs=len(g),mutual=int(g.mutual.sum()),correct_3px=int(g.correct_3px.sum()))
  for px in [1,3,5]:result[f'precision_{px}px']=100*g[f'correct_{px}px'].sum()/g.mutual.sum() if g.mutual.sum() else np.nan
  for k in ['encode_ms','preencoded_matching_ms','total_ms']:result[k]=float(g[k].median())
  result['descriptor_bytes']=int(g.descriptor_bytes.iloc[0]);result['working_bytes']=int(g.working_bytes.iloc[0]);result['representation']=g.representation.iloc[0];result['mean_correct_3px_per_pair']=float(g.correct_3px.mean());return result
 summary=pd.DataFrame([dict(method=name,**summarize(g)) for name,g in df.groupby('method',sort=True)])
 order=['official_float_cosine']+[f'quantile_{k}bin_thermometer_AVX2' for k in BINS]+[f'quantile_{k}bin_binary_{3 if k==8 else 4}bit_ABS' for k in (8,16)]
 summary=summary.set_index('method').loc[order].reset_index()
 summary['speedup_vs_float']=summary.iloc[0].total_ms/summary.total_ms
 summary.to_csv(out/'summary.csv',index=False)
 by_type=pd.DataFrame([dict(category=c,method=n,**summarize(g)) for (c,n),g in df.groupby(['category','method'])])
 by_type.to_csv(out/'by_type.csv',index=False)
 print(label,summary.to_string(index=False),flush=True);return summary,df

def run(root,coco_zip,hpatches_zip=None,hpatches_cache=None,count=256,calibration_count=192):
 root=Path(root);root.mkdir(exist_ok=True,parents=True)
 # Reuse a verified local XFeat checkout if supplied by the caller.
 model,commit=f.load_xfeat(root)
 cal,coco,man=prepare_coco(coco_zip,model,root/'coco',count,calibration_count)
 variants,fit_seconds=fit_variants(cal,root)
 # Do not use HPatches in boundary fitting or parameter selection.
 if hpatches_cache and Path(hpatches_cache).exists():
  hp=f.read_pair_cache(hpatches_cache)
 else:
  from hpatches_source import hpatches_all
  hp=hpatches_all(hpatches_zip,model,root/'hpatches_cache')
 assert len(hp)==580,'Expected all 116 HPatches sequences, reference versus 2..6'
 summaries={};all_records=[]
 for label,pairs in [('COCO_synthetic',coco),('HPatches',hp)]:
  summary,records=evaluate(pairs,variants,model,root/label,label);summaries[label]=summary;all_records.append(records)
 env=dict(date='2026-10-14',seed=SEED,xfeat_commit=commit,torch_version=torch.__version__,numpy=np.__version__,platform=platform.platform(),torch_threads=2,cpp_threads=2,repeats=REPEATS,warmup=WARMUP,bins=list(BINS),fit_seconds=fit_seconds,calibration=man,scope='CPU descriptor encoding + transpose + exact MNN; XFeat extraction excluded; timing every evaluated pair',metric='pooled precision (<1/3/5 px homography error), not official HPatches MMA',hpatches_count=580,lut=False,projection=False,quantile_fitted_once=True,binary_comparison='Only bin notation changes; same per-dimension fixed quantiles and integer ABS; no XOR Hamming on ordinary binary',binary3_unpack='Once per matching call for both descriptor sets; included in preencoded and total time',binary4_direct='Packed nibble ABS without unpacked descriptor arrays',equivalence_verified_pairs=772)
 (root/'experiment_manifest.json').write_text(json.dumps(env,indent=2))
 return summaries

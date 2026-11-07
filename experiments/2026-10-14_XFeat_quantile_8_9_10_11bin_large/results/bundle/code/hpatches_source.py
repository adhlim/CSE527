from pathlib import Path
import json,time,zipfile,io,urllib.request
import numpy as np
import cv2
import features as e


URL='https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip'
def hpatches_all(zip_path,model,cache_dir):
 cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
 if (cache_dir/'hpatches_pairs.npz').exists():return e.read_pair_cache(cache_dir/'hpatches_pairs.npz')
 pairs=[];manifest=[];start=time.perf_counter()
 def decode(raw):
  orig=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
  if orig is None:raise ValueError('HPatches image decode failed')
  img=e.resize_decode(raw,640)
  scale=np.diag([img.shape[1]/orig.shape[1],img.shape[0]/orig.shape[0],1.]).astype(np.float64)
  return img,scale
 with zipfile.ZipFile(zip_path) as z:
  names=set(z.namelist());folders=sorted(n.rsplit('/',1)[0] for n in names if n.endswith('/1.ppm'))
  for folder in folders:
   scene=folder.split('/')[-1]
   if not scene.startswith(('i_','v_')):continue
   im,S1=decode(z.read(folder+'/1.ppm'));k1,d1=e.extract(model,im)
   for idx in range(2,7):
    raw=folder+f'/{idx}.ppm';hname=folder+f'/H_1_{idx}'
    if raw not in names or hname not in names:raise ValueError(f'Missing image/homography: {scene}/{idx}')
    im2,S2=decode(z.read(raw));k2,d2=e.extract(model,im2)
    H0=np.loadtxt(io.BytesIO(z.read(hname)));H=S2@H0@np.linalg.inv(S1)
    pairs.append(dict(name=f'{scene}/1-{idx}',d1=d1,d2=d2,k1=k1,k2=k2,H=H))
   manifest.append(scene);print('extract',len(manifest),scene,flush=True)
 e.save_pair_cache(cache_dir/'hpatches_pairs.npz',pairs)
 (cache_dir/'hpatches_manifest.json').write_text(json.dumps(dict(scenes=manifest,n_scenes=len(manifest),n_pairs=len(pairs),illumination_scenes=sum(x.startswith('i_') for x in manifest),viewpoint_scenes=sum(x.startswith('v_') for x in manifest),max_side=640,top_k=1000,resize_homography='S2 @ H @ inv(S1)',source=URL,extraction_seconds=time.perf_counter()-start,protocol='all archive sequences, 1 versus 2..6; no excluded sequences; precision <3px, not official MMA benchmark',no_hpatches_training=True),indent=2))
 return pairs

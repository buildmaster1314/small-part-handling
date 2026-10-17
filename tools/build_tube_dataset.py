"""Build train/validation or isolated test data from physical rollout NPZs."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--test',action='store_true');p.add_argument('--statistics-from',type=Path);a=p.parse_args()
if a.test and not a.statistics_from:raise ValueError('Test data must use training normalization statistics')
a.output.parent.mkdir(parents=True,exist_ok=True);records=[];train=[];seen=set()
with h5py.File(a.output,'w') as f:
 f.attrs.update(schema='lewm_tube_episode_adapter_v2',frame_dt=.1,history_size=3)
 for path in sorted(a.source.glob('*.npz')):
  meta=json.loads(path.with_suffix('.json').read_text());split=meta['split']
  if (split=='test')!=a.test:continue
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest in seen:raise ValueError('Duplicate episode')
  seen.add(digest)
  with np.load(path) as episode:
   actions=episode['actions'].astype('float32').reshape(-1,300);pixels=episode['frames']
   if len(pixels)!=len(actions)+1 or len(pixels)<4 or not np.isfinite(actions).all():raise ValueError(path)
   g=f.create_group(split+'/'+path.stem);g.create_dataset('pixels',data=pixels,compression='lzf');g.create_dataset('actions',data=actions,compression='lzf');g.attrs['scenario_json']=json.dumps(meta);g.attrs['source_sha256']=digest
   if split=='train':train.append(actions)
   records.append(dict(name=path.stem,split=split,frames=len(pixels),success=meta['success'],sha256=digest))
 if a.test:
  with h5py.File(a.statistics_from,'r') as source:
   for key in ('action_mean','action_std'):f.create_dataset(key,data=source[key][:])
 else:
  if not train or 'validation' not in f:raise ValueError('Both training and validation episodes required')
  action=np.concatenate(train);f.create_dataset('action_mean',data=action.mean(0));f.create_dataset('action_std',data=np.maximum(action.std(0,ddof=1),1e-6))
 if not records:raise ValueError('No episodes')
a.output.with_suffix('.manifest.json').write_text(json.dumps(dict(episodes=records,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),test=a.test),indent=2));print(a.output,len(records))

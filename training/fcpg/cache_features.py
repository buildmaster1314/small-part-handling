"""Freeze visual LeWM and cache aligned RGB features, actions and encoder states."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lewm_official'))
from runtime import build,preprocess
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,action='append',required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--backbone-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
m,_=build(a.upstream,a.backbone_source);m.model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));m.eval();episodes={};records=[];commands=[]
for directory in a.source:
 for path in sorted(directory.glob('*.npz')):
  meta=json.loads(path.with_suffix('.json').read_text())
  if meta.get('frames_are_placeholders',False):raise ValueError(f'Scoring-only replay is not visual training data: {path}')
  data=np.load(path);pixels=data['frames'];actions=data['actions'].astype('float32');states=data['qpos'][:,:6].astype('float32')
  assert len(pixels)==len(actions)+1==len(states)
  features=[]
  with torch.no_grad():
   for start in range(0,len(pixels),16):
    pp=torch.from_numpy(pixels[start:start+16]).permute(0,3,1,2).unsqueeze(1);features.append(m.model.encode({'pixels':preprocess(pp)})['emb'][:,0].cpu())
  key=path.stem;episodes[key]={'embedding':torch.cat(features),'actions':torch.from_numpy(actions.reshape(-1,300)),'states':torch.from_numpy(states),'split':meta['split'],'parent':meta.get('parent_episode',path.name)}
  if meta['split']=='train':commands.append(actions.reshape(-1,6))
  records.append({'name':key,'parent':episodes[key]['parent'],'split':meta['split'],'frames':len(pixels),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});print(key,len(pixels),flush=True)
command=np.concatenate(commands);mean=command.mean(0);std=np.maximum(command.std(0,ddof=1),.1)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
# Test is stored independently; it cannot be loaded by the training entry point.
for name,splits in [('trainval',{'train','validation'}),('test',{'test'})]:
 selected={k:v for k,v in episodes.items() if v['split'] in splits}
 torch.save({'episodes':selected,'state_mean':torch.from_numpy(mean),'state_std':torch.from_numpy(std),'base_action_mean':ck['action_mean'],'base_action_std':ck['action_std'],'base_weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest()},a.output/(name+'.pt'))
(a.output/'manifest.json').write_text(json.dumps({'episodes':records,'split_unit':'whole parent episode; all exploration descendants retain parent split','frozen_visual_weights':str(a.weights),'base_weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'state_normalization':'train commands, six shared coordinates; std floor .1 rad','base_action_normalization':'unchanged original 300D LeWM statistics; passed through explicit action_embedding'},indent=2))

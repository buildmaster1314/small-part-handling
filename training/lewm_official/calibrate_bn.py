"""Train-only, deterministic BatchNorm-statistics calibration for inference.

Weights and official model/loss stay unchanged. Original checkpoints are kept.
This optional adaptation is selected on validation data, never test data.
"""
from pathlib import Path
import argparse,json
import numpy as np,torch,h5py
from runtime import build,preprocess
from train import load_dataset

p=argparse.ArgumentParser()
for name in ['upstream','backbone-source','data','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--weights',type=Path);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
torch.set_num_threads(a.threads)
h,cfg=build(a.upstream,a.backbone_source);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);h.load_state_dict(ck['harness'])
if a.weights:h.model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True))
h.eval();m=h.model;a.output.mkdir(parents=True,exist_ok=True)
episodes,indices,mean,std=load_dataset(a.data)
@torch.no_grad()
def evaluate():
 ix=indices['validation'];chosen=[ix[i] for i in np.linspace(0,len(ix)-1,min(128,len(ix))).round().astype(int)]
 total=np.zeros(4)
 for j in range(0,len(chosen),8):
  take=chosen[j:j+8];pixels=preprocess(torch.stack([episodes[k][0][i:i+4] for k,i in take]));actions=torch.stack([episodes[k][1][i:i+4] for k,i in take])
  out=m.encode({'pixels':pixels,'action':actions});emb=out['emb'];pred=m.predict(emb[:,:3],out['act_emb'][:,:3]);target=emb[:,1:]
  total+=np.asarray([float((pred-target).square().mean()),float((emb[:,:3]-target).square().mean()),float(emb.flatten(0,1).std(0).mean()),len(take)])*np.asarray([len(take),len(take),len(take),1])
 pred,persist,spread=total[:3]/total[3]
 return dict(prediction_mse=pred,persistence_mse=persist,embedding_std=spread,ratio=pred/max(persist,1e-12))
def set_stats(bn,total,total2,count):
 bn.running_mean.copy_((total/count).float());bn.running_var.copy_(((total2-total.square()/count)/(count-1)).clamp_min(1e-8).float());bn.num_batches_tracked.fill_(1)
before=evaluate();raw={};total=torch.zeros(2048,dtype=torch.float64);total2=total.clone();count=0
with torch.no_grad():
 for key,(pixels,actions) in episodes.items():
  if not key.startswith('train/'):continue
  vec=[]
  for j in range(0,len(pixels),16):
   x=preprocess(pixels[j:j+16].unsqueeze(1)).squeeze(1)
   vec.append(m.encoder(x,interpolate_pos_encoding=True).last_hidden_state[:,0])
  raw[key]=torch.cat(vec);values=m.projector.net[0](raw[key]).double()
  weight=torch.from_numpy(np.convolve(np.ones(len(pixels)-3),np.ones(4))).double().unsqueeze(1)
  total+=(values*weight).sum(0);total2+=(values.square()*weight).sum(0);count+=int(weight.sum())
 set_stats(m.projector.net[1],total,total2,count)
 embedded={key:m.projector(values) for key,values in raw.items()}
 total.zero_();total2.zero_();count=0
 for j in range(0,len(indices['train']),32):
  take=indices['train'][j:j+32];emb=torch.stack([embedded[k][i:i+3] for k,i in take]);actions=torch.stack([episodes[k][1][i:i+3] for k,i in take])
  hidden=m.predictor(emb,m.action_encoder(actions));values=m.pred_proj.net[0](hidden.flatten(0,1)).double()
  total+=values.sum(0);total2+=values.square().sum(0);count+=len(values)
 set_stats(m.pred_proj.net[1],total,total2,count)
after=evaluate();torch.save(m.state_dict(),a.output/'calibrated_weights.pt')
report={'checkpoint_step':ck['step'],'before':before,'after':after,'train_only':True,'optimized_parameters_changed':False,'buffers_changed':['projector.net.1.running_mean','projector.net.1.running_var','pred_proj.net.1.running_mean','pred_proj.net.1.running_var'],'selection_eligible':bool(after['ratio']<before['ratio'] and after['embedding_std']>.01),'method':'Exact training-window-weighted encoder projector statistics, then predictor projector statistics; eval mode, no dropout or optimizer update.'}
(a.output/'calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

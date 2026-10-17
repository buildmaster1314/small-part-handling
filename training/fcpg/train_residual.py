from pathlib import Path
import argparse,json,time,hashlib,math
import numpy as np,torch
from residual_runtime import build_residual,predict

p=argparse.ArgumentParser()
for name in ['fcpg-root','upstream','backbone-source','base-weights','cache','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--state-weight',type=float,default=1.);p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch',type=int,default=32);p.add_argument('--threads',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(a.threads);torch.manual_seed(5120);rng=np.random.default_rng(5120)
cache=torch.load(a.cache,map_location='cpu',weights_only=False);episodes=cache['episodes']
assert all(e['split'] in ('train','validation') for e in episodes.values()),'Test data must not enter trainer'
assert hashlib.sha256(a.base_weights.read_bytes()).hexdigest()==cache['base_weights_sha256']
model=build_residual(a.fcpg_root,a.upstream,a.backbone_source,a.base_weights)
trainable=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(trainable,lr=5e-5,weight_decay=1e-3)
indices={split:[(k,i) for k,e in episodes.items() if e['split']==split for i in range(len(e['embedding'])-3)] for split in ['train','validation']}
validation=[indices['validation'][i] for i in np.linspace(0,len(indices['validation'])-1,128).round().astype(int)]
source=a.fcpg_root/'fcpg/so101_state_residual_lewm.py'
meta={'method':'unmodified FCPG TemporalStateResidualLeWM (SO-101 v5 branch)','repository':'https://github.com/gw769/feedback-corrected-grounding','commit':'70763d0017d8ad3d65e7a46e58bd0614bf848c92','module_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'base_weights_sha256':cache['base_weights_sha256'],'cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'trainable_parameters':sum(p.numel() for p in trainable),'base_frozen':True,'steps':a.steps,'batch_size':a.batch,'seed':5120,'validation_selection':'fixed endpoint, no best-checkpoint selection','history':3,'action_dim':300,'normalization_adapter':'shared 6D state/command zscore; original 300D action embedding provided explicitly','objective':'visual residual MSE + smooth L1 state residual + 0.02 SIGReg on frozen embeddings; regularizer is gradient-constant','adaptations':['local CPU cache/optimizer loop','our task-specific frozen LeWM base was trained with SIGReg .09, not the repository public-data base .02','all six state values are joint encoders, never object pose','two cameras concatenated as in the existing base model'],'train_windows':len(indices['train']),'validation_windows':len(indices['validation']),'test_used':False,'state_loss_weight':a.state_weight}
(a.output/'provenance.json').write_text(json.dumps(meta,indent=2))
def batch(entries):
 return tuple(torch.stack([episodes[k][key][i:i+length] for k,i in entries]) for key,length in [('embedding',4),('actions',3),('states',4)])
def validate(entries):
 model.eval();totals={k:[] for k in ['visual_prediction','visual_base','visual_copy','state_prediction','state_command','state_copy','state_shuffled_action','visual_shuffled_action','state_shuffled_input','action_rank_win']}
 with torch.no_grad():
  for j in range(0,len(entries),a.batch):
   emb,act,state=batch(entries[j:j+a.batch]);base,vis,cmd,future=predict(model,emb[:,:3],act,state[:,:3],cache)
   _,wv,_,ws=predict(model,emb[:,:3],act.roll(1,0),state[:,:3],cache)
   _,_,_,wrongstate=predict(model,emb[:,:3],act,state[:,:3].roll(1,0),cache)
   vp=(vis[:,-1]-emb[:,-1]).square().mean(-1);sp=(future[:,-1]-state[:,-1]).square().mean(-1);sw=(ws[:,-1]-state[:,-1]).square().mean(-1)
   vals=[vp,(base[:,-1]-emb[:,-1]).square().mean(-1),(emb[:,-2]-emb[:,-1]).square().mean(-1),sp,(cmd[:,-1]-state[:,-1]).square().mean(-1),(state[:,-2]-state[:,-1]).square().mean(-1),sw,(wv[:,-1]-emb[:,-1]).square().mean(-1),(wrongstate[:,-1]-state[:,-1]).square().mean(-1),(sp<sw).float()]
   for key,value in zip(totals,vals):totals[key].extend(value.tolist())
 result={key:float(np.mean(value)) for key,value in totals.items()};result['state_vs_hold_ratio']=result['state_prediction']/max(result['state_copy'],1e-12);result['state_vs_command_ratio']=result['state_prediction']/max(result['state_command'],1e-12);result['visual_vs_hold_ratio']=result['visual_prediction']/max(result['visual_copy'],1e-12)
 return result
started=time.time();history=[]
from module import SIGReg
sigreg=SIGReg(knots=17,num_proj=1024)
for step in range(1,a.steps+1):
 model.train();model.base.eval();choices=rng.integers(0,len(indices['train']),size=a.batch);emb,act,state=batch([indices['train'][i] for i in choices])
 base,vis,cmd,future=predict(model,emb[:,:3],act,state[:,:3],cache)
 vl=(vis-emb[:,1:]).square().mean();sl=torch.nn.functional.smooth_l1_loss((future-cache['state_mean'])/cache['state_std'],(state[:,1:]-cache['state_mean'])/cache['state_std'])
 with torch.no_grad():reg=sigreg(emb.transpose(0,1))
 loss=vl+a.state_weight*sl+.02*reg;optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(trainable,1.)
 if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite optimization')
 scale=min(1,step/100)*(.5*(1+math.cos(math.pi*max(0,step-100)/max(1,a.steps-100))))
 for group in optimizer.param_groups:group['lr']=5e-5*scale
 optimizer.step()
 if step==1 or step%200==0 or step==a.steps:
  val=validate(validation);record={'step':step,'loss':float(loss.detach()),'visual_loss':float(vl.detach()),'state_huber':float(sl.detach()),'sigreg_constant':float(reg),'validation':val,'elapsed_seconds':time.time()-started};history.append(record)
  branch={k:v for k,v in model.state_dict().items() if not k.startswith('base.')}
  payload={'branch':branch,'step':step,'optimizer':optimizer.state_dict(),'statistics':{k:cache[k] for k in ['state_mean','state_std','base_action_mean','base_action_std','base_weights_sha256']},'provenance':meta,'history':history,'rng_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state()}
  torch.save(payload,a.output/'checkpoint.tmp');(a.output/'checkpoint.tmp').replace(a.output/'training_checkpoint.pt')
  (a.output/'training_summary.json').write_text(json.dumps({'status':'complete' if step==a.steps else 'training','step':step,'history':history},indent=2));print(json.dumps(record),flush=True)
full=validate(indices['validation']);(a.output/'full_validation.json').write_text(json.dumps(full,indent=2));print('FULL_VALIDATION',json.dumps(full),flush=True)

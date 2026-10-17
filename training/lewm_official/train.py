"""CPU-capable runner for the pinned original LeWM architecture and objective."""
from pathlib import Path
import argparse,hashlib,json,math,os,random,time
import h5py,numpy as np,torch
from omegaconf import OmegaConf
from runtime import build,preprocess


def atomic_json(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def atomic_save(data,path):
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(data,tmp);tmp.replace(path)

def load_dataset(path, splits=("train","validation")):
 episodes={};indices={split:[] for split in splits}
 with h5py.File(path,'r') as f:
  mean=torch.from_numpy(f['action_mean'][:]);std=torch.from_numpy(f['action_std'][:])
  for split in indices:
   for name,g in f[split].items():
    key=split+'/'+name
    pixels=torch.from_numpy(g['pixels'][:]).permute(0,3,1,2)
    action=torch.from_numpy(g['actions'][:]);action=(action-mean)/std
    # The fourth action is unused by the official 3-frame, 1-step objective.
    action=torch.cat([action,torch.zeros_like(action[:1])])
    episodes[key]=(pixels,action)
    indices[split].extend((key,i) for i in range(len(pixels)-3))
 return episodes,indices,mean,std

def main():
 p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--backbone-source',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',type=Path);p.add_argument('--init-weights',type=Path);p.add_argument('--eval-every',type=int,default=100);args=p.parse_args()
 if args.steps<1 or args.batch_size<2:raise ValueError('steps >= 1 and batch size >= 2 required')
 torch.set_num_threads(args.threads);torch.manual_seed(3072);np.random.seed(3072);random.seed(3072)
 episodes,indices,mean,std=load_dataset(args.data);model,cfg=build(args.upstream,args.backbone_source,mean.numel())
 optimizer=torch.optim.AdamW(model.model.parameters(),lr=5e-5,weight_decay=1e-3)
 args.output.mkdir(parents=True,exist_ok=True);started=time.time();rng=np.random.default_rng(3072);history=[];start=0;best=float('inf');best_ratio=float('inf')
 source_files=[args.upstream/'jepa.py',args.upstream/'module.py',args.upstream/'train.py',args.upstream/'config/train/model/lewm.yaml',args.backbone_source]
 hashes={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_files}
 manifest={'upstream_repository':'https://github.com/lucas-maes/le-wm','upstream_commit':'8edfeb336732b5f3ce7b8b210d0ba370a09e2cac','backbone_repository':'https://github.com/galilai-group/stable-pretraining','backbone_commit':'ab836bf699a2be712dfcf980a5eb70d36391e876','source_hashes':hashes,'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'parameter_count':sum(p.numel() for p in model.model.parameters()),'device':'cpu','precision':'float32','batch_size':args.batch_size,'steps_requested':args.steps,'history_size':3,'num_preds':1,'image_size':[224,224],'action_dim':int(mean.numel()),'sigreg_weight':.09,'sigreg_knots':17,'sigreg_projections':1024,'train_windows':len(indices['train']),'validation_windows':len(indices['validation']),'adaptations':['Local HDF5 data loader and CPU optimizer loop, not upstream Lightning launcher','Whole-episode train/validation split','Two-camera mosaic resized to 224x224 and normalized with ImageNet statistics','300-dimensional action contains all 50 physics commands between image frames',f'CPU float32, batch {args.batch_size}, {args.steps} updates, step-based warmup/cosine schedule'],'limitations':'Expanded parameter/slot suite; all material parameters remain uncalibrated. Prediction metrics alone do not establish control success.'}
 manifest['initial_weights']=str(args.init_weights);manifest['evaluation_windows']=128;manifest['eval_every']=args.eval_every
 if args.init_weights:
  if args.resume:raise ValueError('Choose initialization or resume')
  model.model.load_state_dict(torch.load(args.init_weights,map_location='cpu',weights_only=True),strict=True)
  manifest['initial_weights_sha256']=hashlib.sha256(args.init_weights.read_bytes()).hexdigest()
 atomic_json(args.output/'provenance.json',manifest);OmegaConf.save(cfg,args.output/'model_config.yaml')
 if args.resume:
  ck=torch.load(args.resume,map_location='cpu',weights_only=False)
  if ck['data_sha256']!=manifest['data_sha256']:raise ValueError('Resume dataset differs')
  model.load_state_dict(ck['harness']);optimizer.load_state_dict(ck['optimizer']);start=ck['step'];history=ck['history'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);best=ck['best_validation'];best_ratio=ck.get('best_prediction_ratio',float('inf'))
 def batch(entries):
  pix=[];acts=[]
  for key,i in entries:
   pp,aa=episodes[key];pix.append(pp[i:i+4]);acts.append(aa[i:i+4])
  return {'pixels':preprocess(torch.stack(pix)),'action':torch.stack(acts)}
 def validate(full=False):
  model.eval();choices=indices['validation']
  if not full:choices=[choices[i] for i in np.linspace(0,len(choices)-1,min(128,len(choices))).round().astype(int)]
  totals={k:0. for k in ['loss','pred_loss','sigreg_loss','persistence_loss','zero_action_loss','embedding_std']};count=0
  with torch.random.fork_rng(),torch.no_grad():
   torch.manual_seed(9001)
   for i in range(0,len(choices),args.batch_size):
    entries=choices[i:i+args.batch_size];out=model(batch(entries),'val');count+=len(entries)
    emb=out['emb'];out['persistence_loss']=(emb[:,:3]-emb[:,1:]).square().mean()
    zero=model.model.action_encoder(torch.zeros_like(out['action'][:,:3]));out['zero_action_loss']=(model.model.predict(emb[:,:3],zero)-emb[:,1:]).square().mean()
    out['embedding_std']=emb.flatten(0,1).std(dim=0).mean()
    for k in totals:totals[k]+=float(out[k])*len(entries)
  result={k:v/count for k,v in totals.items()};result['prediction_to_persistence']=result['pred_loss']/max(result['persistence_loss'],1e-12);return result
 def checkpoint(step):
  atomic_save({'harness':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'history':history,'best_validation':best,'best_prediction_ratio':best_ratio,'action_mean':mean,'action_std':std,'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'data_sha256':manifest['data_sha256'],'model_config':OmegaConf.to_container(cfg,resolve=True),'provenance':manifest},args.output/'training_checkpoint.pt')
 print(json.dumps({'event':'start',**{k:manifest[k] for k in ['parameter_count','batch_size','train_windows','validation_windows','upstream_commit']}}),flush=True)
 for step in range(start+1,args.steps+1):
  model.train();ids=rng.integers(0,len(indices['train']),size=args.batch_size)
  warmup=min(20,max(1,args.steps//10));scale=min(1.,step/warmup)
  if step>warmup:scale=.5*(1+math.cos(math.pi*(step-warmup)/max(1,args.steps-warmup)))
  for group in optimizer.param_groups:group['lr']=5e-5*scale
  out=model(batch([indices['train'][i] for i in ids]));optimizer.zero_grad(set_to_none=True);out['loss'].backward()
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(out['loss']) or not torch.isfinite(grad):raise RuntimeError('Non-finite loss or gradient')
  optimizer.step()
  if step==1 or step%args.eval_every==0 or step==args.steps:
   val=validate(full=False)
   rec={'step':step,'train_loss':float(out['loss'].detach()),'prediction_loss':float(out['pred_loss'].detach()),'sigreg_loss':float(out['sigreg_loss'].detach()),'validation':val,'gradient_norm_before_clip':float(grad),'elapsed_seconds':time.time()-started};history.append(rec)
   if val['loss']<best:
    best=val['loss'];atomic_save(model.model.state_dict(),args.output/'best_weights.pt')
   if val['prediction_to_persistence']<best_ratio and val['embedding_std']>.01:
    best_ratio=val['prediction_to_persistence'];atomic_save(model.model.state_dict(),args.output/'best_dynamics_weights.pt')
   checkpoint(step)
   atomic_json(args.output/'training_summary.json',{'status':'complete' if step==args.steps else 'training','step':step,'steps_requested':args.steps,'history':history,'limitations':manifest['limitations']})
   print(json.dumps(rec),flush=True)
 atomic_json(args.output/'full_validation.json',validate(full=True))
 atomic_save(model.model.state_dict(),args.output/'lewm_tube_official_weights.pt')
 atomic_save(model.model,args.output/'lewm_tube_official_object.ckpt')
 print('COMPLETE',args.output,flush=True)

if __name__=='__main__':main()

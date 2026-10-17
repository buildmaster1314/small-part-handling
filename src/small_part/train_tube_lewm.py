"""Start local two-camera LeWM-style training from verified physical trajectories.

This trains a latent predictor, not a validated autonomous control policy.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from small_part.lewm_tube import LeWM

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=400);a=p.parse_args()
 report=json.loads(a.report.read_text())
 if not report['success']:raise RuntimeError('Refusing success-demonstration training on a failed rollout')
 torch.set_num_threads(4);torch.manual_seed(7);rng=np.random.default_rng(7)
 pack=np.load(a.data);frames=torch.from_numpy(pack['frames'].copy()).permute(0,3,1,2).float()/255
 actions=torch.from_numpy(pack['actions'].astype(np.float32)).flatten(1)
 # Split whole time blocks, not adjacent random transitions. A single
 # episode still cannot establish generalization to unseen trials.
 n=len(actions);valid=np.arange(n)[(np.arange(n)//25)%5==4];train=np.setdiff1d(np.arange(n),np.r_[valid,np.maximum(valid-1,0),np.minimum(valid+1,n-1)])
 mean=actions[train].mean(0);std=actions[train].std(0).clamp_min(.05);actions=(actions-mean)/std
 model=LeWM(action_dim=actions.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
 a.output.parent.mkdir(parents=True,exist_ok=True);start=time.time();history=[]
 for step in range(1,a.steps+1):
  ids=torch.tensor(rng.choice(train,32));model.train();loss=model.loss(frames[ids],actions[ids],frames[ids+1]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  if step==1 or step%25==0 or step==a.steps:
   model.eval()
   with torch.no_grad():
    ids=torch.tensor(valid[:64]);val=model.loss(frames[ids],actions[ids],frames[ids+1]).item()
   rec={'step':step,'training_loss':loss.item(),'validation_loss':val,'elapsed_seconds':time.time()-start};history.append(rec);print(json.dumps(rec),flush=True)
   torch.save({'model':model.state_dict(),'action_dim':model.action_dim,'action_mean':mean,'action_std':std,'step':step,'optimizer':opt.state_dict(),'camera_names':['overhead','wrist'],'source':str(a.data),'description':'Local LeWM-style latent predictor; no trained control policy','frame_dt':float(pack['frame_dt'])},a.output)
   a.output.with_suffix('.json').write_text(json.dumps({'status':'complete' if step==a.steps else 'training','device':'cpu','train_transitions':len(train),'validation_transitions':len(valid),'episodes':1,'history':history,'limitation':'Single physical demonstration. Blocked within-episode validation only; no unseen-episode or learned-policy success claim.'},indent=2))
if __name__=='__main__':main()

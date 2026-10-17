"""Reload the trained model and evaluate against a persistence baseline."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from runtime import build, preprocess
from train import load_dataset

p=argparse.ArgumentParser()
for name in ['upstream','backbone-source','data','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--split',choices=['validation','test'],default='validation')
p.add_argument('--weights',default='lewm_tube_official_weights.pt')
p.add_argument('--report',default='checkpoint_verification.json')
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(3072)
m,cfg=build(a.upstream,a.backbone_source)
# Match actual component names without assuming the internal naming of upstream.
initial={prefix:next(p.detach().clone() for name,p in m.model.named_parameters() if name.startswith(prefix)) for prefix in ['encoder.','action_encoder.','predictor.']}
state=torch.load(a.output/a.weights,map_location='cpu',weights_only=True)
m.model.load_state_dict(state,strict=True);m.eval()
changes={prefix:float((next(p for name,p in m.model.named_parameters() if name.startswith(prefix))-value).detach().abs().max()) for prefix,value in initial.items()}
episodes,indices,mean,std=load_dataset(a.data,splits=(a.split,))
chosen=[indices[a.split][i] for i in np.linspace(0,len(indices[a.split])-1,64).round().astype(int)]
totals={'prediction_mse':0.,'persistence_mse':0.,'zero_normalized_action_mse':0.,'action_sensitivity_mse':0.}
with torch.no_grad():
 for start in range(0,len(chosen),8):
  batch=chosen[start:start+8]
  pp=torch.stack([episodes[k][0][i:i+4] for k,i in batch]);aa=torch.stack([episodes[k][1][i:i+4] for k,i in batch])
  out=m.model.encode({'pixels':preprocess(pp),'action':aa});emb=out['emb'];pred=m.model.predict(emb[:,:3],out['act_emb'][:,:3]);target=emb[:,1:]
  alt=m.model.predict(emb[:,:3],m.model.action_encoder(torch.zeros_like(aa[:,:3])))
  vals=[(pred-target).square().mean(),(emb[:,:3]-target).square().mean(),(alt-target).square().mean(),(alt-pred).square().mean()]
  for key,value in zip(totals,vals):totals[key]+=float(value)*len(batch)/len(chosen)
 # Use upstream planning interface for a shape/finite check, not a control claim.
 rollout=m.model.rollout({'pixels':preprocess(pp[:1,:3]).unsqueeze(1)},aa[:1].unsqueeze(1))['predicted_emb']
report={'strict_load':True,'parameter_changes_max_abs':changes,'all_components_updated':all(x>0 for x in changes.values()),'split':a.split,'heldout_windows':len(chosen),'metrics':totals,'rollout_shape':list(rollout.shape),'rollout_finite':bool(torch.isfinite(rollout).all()),'note':'Persistence baseline is essential at 10 Hz. Lower training loss does not establish learned control or useful long-horizon dynamics.'}
(a.output/a.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

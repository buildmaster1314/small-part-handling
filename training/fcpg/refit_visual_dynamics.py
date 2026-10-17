"""Refit the unchanged official LeWM predictor on frozen task embeddings.

This is an explicit local adaptation, before freezing LeWM for FCPG. Encoder
and projector stay bit-identical. Train/validation only; fixed 2500-step budget.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from residual_runtime import build

p=argparse.ArgumentParser()
for name in ('upstream','backbone-source','base-weights','cache','output'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--steps',type=int,default=2500)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(6631);rng=np.random.default_rng(6631)
cache=torch.load(a.cache,map_location='cpu',weights_only=False);episodes=cache['episodes']
assert all(e['split'] in ('train','validation') for e in episodes.values())
harness,_=build(a.upstream,a.backbone_source);model=harness.model
initial=torch.load(a.base_weights,map_location='cpu',weights_only=True);model.load_state_dict(initial);model.eval()
for name,param in model.named_parameters():param.requires_grad_(not name.startswith(('encoder.','projector.')))
parameters=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=.001)
indices={split:[(k,i) for k,e in episodes.items() if e['split']==split for i in range(len(e['embedding'])-7)] for split in ('train','validation')}
fixed=[indices['validation'][i] for i in np.linspace(0,len(indices['validation'])-1,128).round().astype(int)]
def batch(entries):
    emb=torch.stack([episodes[k]['embedding'][i:i+8] for k,i in entries])
    action=torch.stack([episodes[k]['actions'][i:i+7] for k,i in entries])
    return emb,(action-cache['base_action_mean'])/cache['base_action_std']
def rollout(emb,actions):
    history=emb[:,:3];predictions=[]
    for j in range(5):
        pred=model.predict(history,model.action_encoder(actions[:,j:j+3]))[:,-1:]
        predictions.append(pred);history=torch.cat((history[:,1:],pred),1)
    return torch.cat(predictions,1)
def validate():
    errors=[];copies=[]
    with torch.no_grad():
        for j in range(0,len(fixed),32):
            em,ac=batch(fixed[j:j+32]);pred=rollout(em,ac)
            errors.append((pred-em[:,3:]).square().mean(-1));copies.append((em[:,2:3]-em[:,3:]).square().mean(-1))
    err=torch.cat(errors).mean(0);hold=torch.cat(copies).mean(0)
    return dict(prediction_mse=err.tolist(),hold_mse=hold.tolist(),ratio=(err/hold).tolist())
started=time.time();history=[]
for step in range(1,a.steps+1):
    em,ac=batch([indices['train'][i] for i in rng.integers(0,len(indices['train']),32)])
    predicted=rollout(em,ac);loss=(predicted-em[:,3:]).square().mean()
    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
    if step==1 or step%500==0 or step==a.steps:
        report=dict(step=step,loss=float(loss.detach()),validation=validate(),elapsed_seconds=time.time()-started)
        history.append(report);print(json.dumps(report),flush=True)
        torch.save(model.state_dict(),a.output/'weights.tmp');(a.output/'weights.tmp').replace(a.output/'weights.pt')
        (a.output/'summary.json').write_text(json.dumps(dict(status='complete' if step==a.steps else 'training',history=history),indent=2))
for name,value in model.state_dict().items():
    if name.startswith(('encoder.','projector.')):assert torch.equal(value,initial[name]),name
newhash=hashlib.sha256((a.output/'weights.pt').read_bytes()).hexdigest()
# Cached embeddings remain exact because the full encoding path is unchanged.
cache['base_weights_sha256']=newhash;torch.save(cache,a.output/'trainval.pt')
(a.output/'provenance.json').write_text(json.dumps(dict(method='official LeWM predictor/action encoder/prediction projector refit; frozen image encoder and projector',local_adaptation=True,steps=a.steps,history_frames=3,recursive_training_horizon=5,train_windows=len(indices['train']),validation_windows=len(indices['validation']),test_used=False,encoder_and_projector_bit_identical=True,base_input_sha256=hashlib.sha256(a.base_weights.read_bytes()).hexdigest(),base_output_sha256=newhash,optimizer='AdamW lr=5e-5 weight_decay=.001; gradient norm <=1',model_eval_mode='fixed BN statistics, gradients enabled in prediction path',seed=6631),indent=2))

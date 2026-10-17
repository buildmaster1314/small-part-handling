"""Causal shadow replay of a new physical execution, with no action selection.

The artifact is an off-policy diagnostic, never evidence that FCPG controlled
the robot. Only RGB frames, six encoder readings and executed joint commands
enter the model. Object poses in the trace are excluded.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from residual_runtime import build_residual,load_checkpoint,predict,preprocess

p=argparse.ArgumentParser()
for name in ('fcpg-root','upstream','backbone-source','base-weights','checkpoint','trace','run-result','output'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4)
run=json.loads(a.run_result.read_text());assert not run.get('frames_are_placeholders',False)
data=np.load(a.trace);pixels=data['frames'];actions=torch.from_numpy(data['actions'].reshape(-1,300).astype('float32'));states=torch.from_numpy(data['qpos'][:,:6].astype('float32'))
assert len(pixels)==len(states)==len(actions)+1
model=build_residual(a.fcpg_root,a.upstream,a.backbone_source,a.base_weights);ck=load_checkpoint(model,a.checkpoint,a.base_weights);stats=ck['statistics']
features=[];records=[]
with torch.no_grad():
    for start in range(0,len(pixels),16):
        x=torch.from_numpy(pixels[start:start+16]).permute(0,3,1,2)[:,None]
        features.append(model.base.encode({'pixels':preprocess(x)})['emb'][:,0])
    emb=torch.cat(features)
    for index in range(2,len(actions)):
        base,visual,command,future=predict(model,emb[index-2:index+1][None],actions[index-2:index+1][None],states[index-2:index+1][None],stats)
        target=states[index+1];visual_target=emb[index+1]
        records.append(dict(frame=index,time=index*.1,joint_prediction=future[0,-1].tolist(),joint_target=target.tolist(),state_mse=float((future[0,-1]-target).square().mean()),state_command_mse=float((command[0,-1]-target).square().mean()),state_hold_mse=float((states[index]-target).square().mean()),visual_mse=float((visual[0,-1]-visual_target).square().mean()),visual_base_mse=float((base[0,-1]-visual_target).square().mean()),visual_hold_mse=float((emb[index]-visual_target).square().mean())))
metrics={k:float(np.mean([r[k] for r in records])) for k in ('state_mse','state_command_mse','state_hold_mse','visual_mse','visual_base_mse','visual_hold_mse')}
metrics['joint_vs_command']=metrics['state_mse']/max(metrics['state_command_mse'],1e-12);metrics['visual_vs_hold']=metrics['visual_mse']/max(metrics['visual_hold_mse'],1e-12)
result=dict(mode='causal off-policy shadow audit; model did not choose commands',object_pose_input=False,future_state_input=False,frame_count=len(pixels),trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),execution_success=run['success'],metrics=metrics,records=records)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

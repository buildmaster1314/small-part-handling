from pathlib import Path
from dataclasses import asdict
import json,argparse,time
import numpy as np,mujoco
from PIL import Image
from small_part.tube_scene import load_tube_model,PhysicsConfig,camera_signature
from small_part.tube_cycle_physical import run_tube_cycle

SCENARIOS=[
 ('train',1,1,1.,.2,1.5,2.4,-.015,[0,0],[0,0]),
 ('train',0,0,1.,.2,1.5,2.4,-.015,[.0006,-.0004],[.01,0]),
 ('train',2,2,1.,.2,1.5,2.4,-.015,[-.0006,.0005],[0,-.01]),
 ('train',1,0,.8,.1,1.5,2.4,-.01,[.0008,.0004],[0,0]),
 ('train',0,1,1.2,.4,1.5,2.4,-.02,[-.0007,.0004],[.01,.01]),
 ('train',2,1,.9,.2,1.,1.5,-.015,[.0004,-.0008],[0,0]),
 ('train',1,2,1.1,.4,1.,1.,-.025,[-.0004,-.0005],[-.01,0]),
 ('train',0,2,1.,.1,1.5,2.4,-.005,[.0006,.0006],[0,.01]),
 ('validation',1,1,.85,.3,1.5,2.4,-.012,[.0007,-.0007],[.005,0]),
 ('validation',2,0,1.15,.3,1.5,2.4,-.018,[-.0008,.0006],[0,.005]),
 ('validation',0,1,.95,.5,1.,1.5,-.02,[.0005,.0008],[.005,-.005]),
 ('validation',1,2,1.05,.3,1.,1.,-.01,[-.0006,-.0007],[0,0]),
 ('test',3,3,1.,.2,1.5,2.4,-.015,[0,0],[0,0]),
 ('test',1,1,1.,.2,1.5,2.4,-.015,[.001,-.0006],[0,0]),
 ('test',2,1,1.25,.6,1.5,2.4,-.015,[-.0008,-.0008],[.01,-.01]),
 ('test',0,0,.75,.3,.8,1.,-.005,[.0008,.0008],[0,.01]),
]
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--output',type=Path,default=Path('outputs/tube_suite/scenarios'));a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=True)
for idx,values in enumerate(SCENARIOS):
 if idx%a.workers!=a.worker:continue
 split,source,target,speed,mu,jaw,tube,grip,jitter,tilt=values
 name=f'{idx:02d}_{split}'
 if (out/(name+'.json')).exists():continue
 config=PhysicsConfig(wall_friction=mu,jaw_friction=jaw,tube_friction=tube)
 m,d=load_tube_model(config);r=mujoco.Renderer(m,120,160)
 frames=[];blocks=[];states=[];stages=[];commands=[];count=0;t=time.time();contacts=0
 def frame(stage):
  imgs=[]
  for cam in ('overhead','wrist'):
   r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;imgs.append(r.render().copy())
  frames.append(np.asarray(Image.fromarray(np.concatenate(imgs,axis=1)).resize((192,96))))
  states.append(d.qpos.copy());stages.append(stage)
 def reset(m,d):frame('reset')
 def snap(stage,m,d):
  global count,contacts
  count+=1;commands.append(d.ctrl.copy())
  contacts+=sum(any(f>=2 for f in c.flex) for c in d.contact)
  if count%50==0:
   frame(stage);blocks.append(np.asarray(commands));commands.clear()
 try:
  result=run_tube_cycle(m,d,source_index=source,target_index=target,speed=speed,grip=grip,xy_jitter=jitter,initial_tilt=tilt,abort_on_failed_lift=False,on_reset=reset,on_step=snap)
 except Exception as e:
  result={'success':False,'error':str(e),'sim_time':float(d.time)}
 finally:r.close()
 result.update(split=split,scenario_id=idx,physics_config=asdict(config),initial_xy_jitter=jitter,initial_tilt=tilt,source_index=source,target_index=target,speed=speed,grip=grip,cameras=camera_signature(m),frame_count=len(frames),wall_seconds=time.time()-t,housing_contact_steps=contacts)
 np.savez_compressed(out/(name+'.npz'),frames=np.asarray(frames),actions=np.asarray(blocks),qpos=np.asarray(states),stages=np.asarray(stages),frame_dt=.1,physics_dt=.002)
 (out/(name+'.json')).write_text(json.dumps(result,indent=2));print('EPISODE',name,result['success'],len(frames),flush=True)

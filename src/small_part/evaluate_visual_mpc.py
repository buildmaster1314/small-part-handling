"""Evaluate LeWM with pixels/proprioception, score with a separate oracle."""
from pathlib import Path
from dataclasses import asdict
import argparse,sys,json,subprocess,os,time,hashlib
import numpy as np,mujoco,torch
from PIL import Image,ImageDraw
from small_part.control import set_home,joint_index,actuator_index
from small_part.tube_scene import load_tube_model,PhysicsConfig,camera_signature,hole_xy,SOURCE_ROW,TARGET_ROW,TUBE_CENTER_Z
from small_part.tube_cycle_physical import JOINTS
from small_part.tube_evaluation import TubeEvaluator
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/lewm_official'))
from visual_mpc import VisualMPC,Observation

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--weights',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['model','reference','blind'],default='model');p.add_argument('--jitter',type=float,nargs=2,default=[0,0]);p.add_argument('--radius',type=float,default=.003);p.add_argument('--video',action='store_true');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
 torch.set_num_threads(a.threads);a.output.mkdir(parents=True,exist_ok=True)
 meta=json.loads(a.reference.with_suffix('.json').read_text());physics=PhysicsConfig(**meta['physics_config']);m,d=load_tube_model(physics)
 indices=np.array([joint_index(m,n) for n in JOINTS]);acts=np.array([actuator_index(m,n) for n in JOINTS]);limits=m.actuator_ctrlrange[acts].copy()
 mujoco.mj_resetData(m,d);set_home(m,d);source=hole_xy(SOURCE_ROW,meta['source_index']);target=hole_xy(TARGET_ROW,meta['target_index'])
 tube=joint_index(m,'tube_free');d.qpos[tube:tube+7]=[*list(source+np.asarray(a.jitter)),TUBE_CENTER_Z,1,0,0,0];d.qpos[indices[5]]=.5;d.ctrl[acts[5]]=.5;mujoco.mj_forward(m,d)
 evaluator=TubeEvaluator(m,target)
 policy=VisualMPC(ROOT/'third_party/le_wm',ROOT/'third_party/stable_pretraining_reference/utils.py',a.checkpoint,a.reference,limits,weights=a.weights,mode=a.mode,radius=a.radius)
 r=mujoco.Renderer(m,240 if a.video else 120,320 if a.video else 160)
 process=None
 if a.video:
  ffmpeg=os.environ.get('FFMPEG_BINARY')
  if not ffmpeg:raise RuntimeError('Set FFMPEG_BINARY')
  process=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x240','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(a.output/'attempt.mp4')],stdin=subprocess.PIPE)
 observations=[]
 def observe(previous):
  imgs=[]
  for cam in ('overhead','wrist'):
   r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;imgs.append(r.render().copy())
  mosaic=np.concatenate(imgs,axis=1)
  if process:
   video=Image.fromarray(mosaic);draw=ImageDraw.Draw(video);draw.rectangle((0,0,640,15),fill='black');draw.text((4,2),f'LeWM reference-guided MPC | {a.mode} | t={d.time:.1f}s',fill='white');process.stdin.write(np.asarray(video).tobytes())
  pixels=np.array(Image.fromarray(mosaic).resize((192,96)))
  observations.append(pixels.copy())
  return Observation(pixels=pixels,joints=d.qpos[indices].copy(),previous_action=None if previous is None else previous.copy())
 start=time.time();executed=[];states=[d.qpos.copy()];failure=None
 try:
  previous=None
  for i in range(len(policy.reference_actions)):
   action=policy.act(observe(previous))
   if action.shape!=(50,6) or not np.isfinite(action).all():raise RuntimeError('Invalid policy action')
   for row in action:
    d.ctrl[acts]=row;mujoco.mj_step(m,d);evaluator.update(m,d)
   previous=action;executed.append(action);states.append(d.qpos.copy())
   if i%50==0:print('FRAME',i,'sim',d.time,flush=True)
  observe(previous)
 except Exception as e:failure=str(e)
 finally:
  r.close()
  if process:
   process.stdin.close()
   if process.wait():failure='Video encoder failed'
 result=evaluator.result(m,d);result.update(mode=a.mode,policy='demonstration-guided LeWM MPC' if a.mode=='model' else a.mode,learned_without_reference=False,oracle_input=False,policy_input_fields=list(Observation.__dataclass_fields__),reference=str(a.reference),checkpoint=str(a.checkpoint),weights=str(a.weights),initial_xy_jitter=a.jitter,physics_config=asdict(physics),cameras=camera_signature(m),error=failure,wall_seconds=time.time()-start,planning_calls=len(policy.logs),non_nominal_plans=sum(x['candidate']!=0 for x in policy.logs),trust_radius_rad=a.radius,source_xy=source.tolist(),target_xy=target.tolist(),source_index=meta['source_index'],target_index=meta['target_index'],frame_count=len(observations),weights_sha256=hashlib.sha256((a.weights or a.checkpoint).read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),physics={'equalities':int(m.neq),'mocap':int(m.nmocap),'object_pose_writes_after_reset':False,'actuator_torque_limits':m.actuator_forcerange.tolist()})
 if failure:result['success']=False
 (a.output/'result.json').write_text(json.dumps(result,indent=2));(a.output/'planning.json').write_text(json.dumps(policy.logs,indent=2));np.savez_compressed(a.output/'executed.npz',frames=np.asarray(observations),actions=np.asarray(executed),qpos=np.asarray(states),frame_dt=.1,physics_dt=.002)
 print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()

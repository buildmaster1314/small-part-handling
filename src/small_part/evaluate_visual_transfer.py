"""Run a reference-free visual controller; keep privileged scoring in the harness."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,os,subprocess,hashlib,shutil
import numpy as np,mujoco
from PIL import Image,ImageDraw
from small_part.tube_scene import load_tube_model,PhysicsConfig,SOURCE_ROW,TARGET_ROW,hole_xy,TUBE_CENTER_Z,camera_signature
from small_part.control import set_home
from small_part.robot_kinematics import RobotKinematics,JOINTS
from small_part.visual_transfer import VisualTransfer,VisualObservation
from small_part.tube_evaluation import TubeEvaluator

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-index',type=int,default=1);p.add_argument('--target-index',type=int);p.add_argument('--jitter',type=float,nargs=2,default=[0,0]);p.add_argument('--rack-shift',type=float,nargs=2,default=[0,0]);p.add_argument('--wall-friction',type=float,default=.2);p.add_argument('--tube-friction',type=float,default=2.4);p.add_argument('--jaw-friction',type=float,default=1.5);p.add_argument('--grip',type=float,default=-.015);p.add_argument('--mode',choices=['live','blind','replay'],default='live');p.add_argument('--blind-after',type=float,default=0.);p.add_argument('--replay',type=Path);p.add_argument('--video',action='store_true');p.add_argument('--max-frames',type=int,default=800);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'result.json').exists():raise ValueError('Use a new run directory; completed attempts must be retained')
 if a.mode=='replay' and a.replay is None:raise ValueError('Replay baseline requires an explicit trace')
 source_snapshot=a.output/'source';source_snapshot.mkdir(exist_ok=True)
 for source_file in Path(__file__).parent.glob('*.py'):
  if not source_file.name.startswith('_probe'):shutil.copy2(source_file,source_snapshot/source_file.name)
 physics=PhysicsConfig(wall_friction=a.wall_friction,tube_friction=a.tube_friction,jaw_friction=a.jaw_friction);m,d=load_tube_model(physics)
 shift=np.asarray(a.rack_shift)
 for name in ['source_rack','target_rack']:m.body_pos[m.body(name).id,:2]+=shift
 mujoco.mj_resetData(m,d);set_home(m,d);kin=RobotKinematics(m);qadr=kin.qadr;act=np.array([m.actuator(n).id for n in JOINTS]);ti=m.jnt_qposadr[m.joint('tube_free').id]
 source=hole_xy(SOURCE_ROW,a.source_index)+shift;d.qpos[ti:ti+7]=[*list(source+np.asarray(a.jitter)),TUBE_CENTER_Z,1,0,0,0];d.qpos[qadr[5]]=.5;d.ctrl[act[5]]=.5;mujoco.mj_forward(m,d)
 m.vis.global_.offwidth=1600;m.vis.global_.offheight=1200;renderer=mujoco.Renderer(m,1200,1600)
 policy=VisualTransfer(kin,target_index=a.target_index,grip=a.grip);expected_index=a.target_index if a.target_index is not None else a.source_index
 target=hole_xy(TARGET_ROW,expected_index)+shift;evaluator=TubeEvaluator(m,target)
 replay=np.load(a.replay)['actions'] if a.mode=='replay' else None
 process=None
 if a.video:
  ff=os.environ['FFMPEG_BINARY'];process=subprocess.Popen([ff,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(a.output/'attempt.mp4')],stdin=subprocess.PIPE)
 frames=[];states=[d.qpos.copy()];actions=[];oracle_errors=[];started=time.time();exception=None
 def observe():
  if a.mode=='replay' and not a.video:
   # Replay commands need no image processing. Mark these frames explicitly so
   # a scoring-only run cannot silently become a visual training episode.
   frames.append(np.zeros((96,192,3),dtype=np.uint8))
   return None
  images=[]
  for name in ('overhead','wrist'):
   renderer.update_scene(d,camera=name);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;images.append(renderer.render().copy())
  mosaic=np.concatenate([np.asarray(Image.fromarray(im).resize((640,480))) for im in images],axis=1)
  frames.append(np.asarray(Image.fromarray(mosaic).resize((192,96))))
  if process:
   video=Image.fromarray(mosaic);draw=ImageDraw.Draw(video);draw.rectangle((0,0,1280,20),fill='black');draw.text((5,3),f'RGB geometric control (not learned LeWM) | {a.mode} | {policy.stage} | t={d.time:.1f}s',fill='white');process.stdin.write(np.asarray(video).tobytes())
  if a.mode=='blind' and d.time>=a.blind_after:images=[np.zeros_like(im) for im in images]
  return VisualObservation(tuple(images),d.qpos[qadr].copy())
 try:
  for i in range(a.max_frames):
   obs=observe()
   if replay is not None:
    if i>=len(replay):break
    block=replay[i]
   else:
    command=policy.act(obs)
    if policy.last_center is not None:
     # Evaluation only: never returned to the policy.
     oracle_errors.append({'time':float(d.time),'position_error_m':(policy.last_center-d.qpos[ti:ti+3]).tolist(),'stage':policy.stage})
    if policy.done:break
    first=d.ctrl[act].copy();block=np.array([first+(command-first)*(k+1)/50 for k in range(50)])
   for row in block:
    d.ctrl[act]=row;mujoco.mj_step(m,d);evaluator.update(m,d)
   actions.append(block);states.append(d.qpos.copy())
   if i%20==0:
    progress=dict(frame=i,simulation_time=float(d.time),stage=policy.stage,vision=policy.logs[-1] if policy.logs else None)
    (a.output/'progress.json').write_text(json.dumps(progress,indent=2))
    print('FRAME',i,'stage',policy.stage,'estimate',None if policy.last_center is None else policy.last_center.round(4).tolist(),flush=True)
  else:exception='frame_budget_exhausted'
  if len(frames)==len(actions):observe()
 except Exception as e:exception=repr(e)
 finally:
  renderer.close()
  if process:
   process.stdin.close()
   if process.wait():exception='video_encoder_failed'
 result=evaluator.result(m,d);result['physical_success']=result['success'];result['strict_success']=bool(result['success'] and result['final_within_two_degrees'])
 result.update(controller='reference-free RGB geometric feedback, not learned LeWM',policy_finished=policy.done,policy_failure=policy.failure,events=policy.events,retries=policy.retry_count,selected_hole=policy.target_index,detected_target=None if policy.target is None else policy.target.tolist(),target_xy=target.tolist(),source_index=a.source_index,target_index=expected_index,initial_xy_jitter=a.jitter,rack_shift=a.rack_shift,mode=a.mode,physics_config=asdict(physics),cameras=camera_signature(m),reference_actions_input=(a.mode=='replay'),reference_goal_images_input=False,reference_time_progress_input=False,policy_inputs=['RGB images','joint encoders'],liquid_visual_only=True,geometry_prior='known tube and rack dimensions; robot/camera calibration; initially seated height',equalities=int(m.neq),mocap=int(m.nmocap),object_pose_writes_after_reset=False,frame_count=len(frames),error=exception,wall_seconds=time.time()-started)
 if exception or policy.failure:result['success']=False;result['strict_success']=False
 if a.mode=='replay':
  result['controller']='recorded joint command replay; no image feedback'
  result['policy_inputs']=['recorded joint command blocks']
 result.update(blind_after_seconds=a.blind_after,frames_are_placeholders=bool(a.mode=='replay' and not a.video),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source_snapshot.glob('*.py')})
 (a.output/'result.json').write_text(json.dumps(result,indent=2));(a.output/'vision.json').write_text(json.dumps(policy.logs,indent=2));(a.output/'localization_audit.json').write_text(json.dumps(oracle_errors,indent=2));np.savez_compressed(a.output/'executed.npz',frames=np.asarray(frames),actions=np.asarray(actions),qpos=np.asarray(states),frame_dt=.1,physics_dt=.002)
 print('RESULT',json.dumps({k:result[k] for k in ('success','strict_success','policy_failure','worst_tilt','final_tilt','target_error_mm','frame_count','error','wall_seconds')}),flush=True)
if __name__=='__main__':main()

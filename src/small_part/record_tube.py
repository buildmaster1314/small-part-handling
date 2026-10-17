from pathlib import Path
import json,subprocess,os,shutil,argparse,tempfile
import numpy as np,mujoco
from PIL import Image
from small_part.tube_scene import load_tube_model, PhysicsConfig
from dataclasses import asdict
from small_part.tube_cycle_physical import run_tube_cycle
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs');args=parser.parse_args()
destination=args.output;destination.mkdir(parents=True,exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='.tube-recording-',dir=destination))
ffmpeg=os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg')
if not ffmpeg:
 candidates=list((Path.home()/'.cache/uv/archive-v0').glob('*/imageio_ffmpeg/binaries/ffmpeg-linux*'))
 if not candidates:raise RuntimeError('Set FFMPEG_BINARY to an ffmpeg executable')
 ffmpeg=str(candidates[0])
m,d=load_tube_model()
cam={n:{'pos':m.cam_pos[m.camera(n).id].tolist(),'quat':m.cam_quat[m.camera(n).id].tolist(),'fovy':float(m.cam_fovy[m.camera(n).id]),'body':int(m.cam_bodyid[m.camera(n).id])} for n in ['overhead','wrist']}
(out/'cameras.json').write_text(json.dumps(cam,indent=2))
r=mujoco.Renderer(m,height=240,width=320)
p=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','640x240','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(out/'tube_place.mp4')],stdin=subprocess.PIPE)
frames=[];actions=[];block=[];stages=[];states=[];count=0
# Training frame k is the state at the end of action block k.
def image_pair():
 imgs=[]
 for cam in ['overhead','wrist']:
  r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;imgs.append(r.render().copy())
 return np.concatenate(imgs,axis=1)
def snap(stage,model,data):
 global count
 block.append(data.ctrl.copy());count+=1
 if count%50:return
 im=image_pair();p.stdin.write(im.tobytes());frames.append(np.array(Image.fromarray(im).resize((192,96))))
 actions.append(np.asarray(block).copy());block.clear();stages.append(stage);states.append(data.qpos.copy())
 if count%1000==0:print('recorded',count,'physics steps',flush=True)
try:result=run_tube_cycle(m,d,on_step=snap)
finally:r.close();p.stdin.close();code=p.wait()
if code:raise RuntimeError('Video encoding failed')
# Each image is at the end of its action block. Block k+1 maps image k to k+1.
np.savez_compressed(out/'tube_demonstration.npz',frames=np.asarray(frames),actions=np.asarray(actions[1:]),qpos=np.asarray(states),stages=np.asarray(stages),physics_dt=m.opt.timestep,frame_dt=.1)
result['friction']={'jaw':m.flex_friction[0].tolist(),'tube':m.geom_friction[m.geom('tube_wall').id].tolist(),'shelf_sliding':[PhysicsConfig().shelf_friction]*2,'wall_sliding':[PhysicsConfig().wall_friction]*2};result['video_frames']=len(frames);result['physics_config']=asdict(PhysicsConfig());result['demonstration_type']='privileged-state teacher; not learned control'
(out/'tube_physical_validation.json').write_text(json.dumps(result,indent=2))
print('FINAL',json.dumps(result),flush=True)
if not result['success']:raise SystemExit(1)

subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(out/'tube_place.mp4'),'-filter_complex','fps=10,split[a][b];[a]palettegen[p];[b][p]paletteuse','-loop','0',str(out/'tube_place.gif')],check=True)
for artifact in out.iterdir():artifact.replace(destination/artifact.name)
out.rmdir()
print('Saved verified recording to',destination)

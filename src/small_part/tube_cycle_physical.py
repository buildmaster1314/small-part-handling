"""Native-jaw friction grasp, Cartesian feedback through joint actuators only.

IK has its own MjData. After reset the live object's pose and velocity are never
written; no welds, mocap, applied object forces, or collision toggling are used.
The controller reads simulator state (not a learned visual policy).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from small_part.control import set_home, joint_index, actuator_index
from small_part.tube_scene import load_tube_model, tube_position, tube_tilt, hole_xy, SOURCE_ROW, TARGET_ROW, TUBE_CENTER_Z, MAX_TILT

JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
OFFSET=np.array([.004,0,-.085])

def run_tube_cycle(model, data, *, source_index=1, target_index=1, on_step=None, xy_jitter=(0.,0.), grip=-.015, speed=1., initial_tilt=(0.,0.), abort_on_failed_lift=True, on_reset=None, release_height=.1615):
    if not .5 <= speed <= 1.5: raise ValueError("speed outside [0.5,1.5]")
    m,d=model,data
    arm=[joint_index(m,n) for n in JOINTS];acts=[actuator_index(m,n) for n in JOINTS]
    source=hole_xy(SOURCE_ROW,source_index);target=hole_xy(TARGET_ROW,target_index)
    probe=mujoco.MjData(m)
    low=m.jnt_range[:5,0]+.001;high=m.jnt_range[:5,1]-.001
    mujoco.mj_resetData(m,d);set_home(m,d)
    ti=joint_index(m,'tube_free')
    d.qpos[ti:ti+7]=[*(source+xy_jitter),TUBE_CENTER_Z,1,0,0,0]
    from scipy.spatial.transform import Rotation
    xyzw=Rotation.from_euler("xy",initial_tilt).as_quat();d.qpos[ti+3:ti+7]=xyzw[[3,0,1,2]]
    d.qpos[arm[5]]=.5;d.ctrl[acts[5]]=.5;mujoco.mj_forward(m,d)
    if on_reset is not None: on_reset(m,d)
    worst=0.;max_z=TUBE_CENTER_Z;stages=[];stage='approach';samples=[]
    def step():
        nonlocal worst,max_z
        mujoco.mj_step(m,d)
        worst=max(worst,tube_tilt(d));max_z=max(max_z,float(tube_position(d)[2]))
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):
            raise RuntimeError('MuJoCo warning or non-finite state')
        if on_step is not None:on_step(stage,m,d)
    def go(q,seconds):
        start=d.ctrl[acts].copy();n=max(1,int(seconds/speed/m.opt.timestep))
        for i in range(1,n+1):
            t=i/n;t=t*t*(3-2*t);d.ctrl[acts]=start+(q-start)*t;step()
    def ik(point,seed,axis=None):
        def residual(q):
            probe.qpos[arm[:5]]=q;mujoco.mj_kinematics(m,probe)
            b=probe.body('gripper');r=b.xmat.reshape(3,3)
            orientation=r[:2,1] if axis is None else r[:,1]-axis
            return np.r_[(b.xpos+r@OFFSET-point)*10,orientation]
        return least_squares(residual,np.clip(seed,low+1e-7,high-1e-7),bounds=(low,high),max_nfev=100).x
    def move(point,jaw,seconds):
        q=d.ctrl[acts[:5]].copy();probe.qpos[arm[:5]]=q;mujoco.mj_kinematics(m,probe)
        b=probe.body('gripper');start=b.xpos+b.xmat.reshape(3,3)@OFFSET
        for t in np.linspace(0,1,41)[1:]:
            w=t*t*(3-2*t);q=ik(start+(np.array(point)-start)*w,q)
            go(np.r_[q,jaw],seconds/40)
    def servo(destination,seconds):
        start=tube_position(d);destination=np.array(destination);n=int(seconds/.04)
        for i in range(1,n+1):
            t=i/n;w=t*t*(3-2*t);want=start+(destination-start)*w
            b=d.body('gripper');r=b.xmat.reshape(3,3).copy();tip=b.xpos+r@OFFSET
            axis=d.body('tube').xmat.reshape(3,3)[:,2].copy()
            correction=np.cross(axis,[0,0,1])
            desired=r[:,1]+np.cross(correction,r[:,1])*.3;desired/=np.linalg.norm(desired)
            aim=tip+np.clip(want-tube_position(d),-.002,.002)
            go(np.r_[ik(aim,d.qpos[arm[:5]],desired),grip],.04)
    def note(name):
        record={'stage':name,'time':float(d.time),'position':tube_position(d).tolist(),'tilt':tube_tilt(d)}
        stages.append(record);print(json.dumps(record),flush=True)
    move([*source,.285],.5,2);note('hover')
    stage='descend';move([*source,.202],.5,3);note('down')
    stage='grasp';go(np.r_[d.ctrl[acts[:5]],grip],2);note('grip')
    stage='lift';servo([*source,.26],6);note('lift')
    if abort_on_failed_lift and tube_position(d)[2] < .245:raise RuntimeError('Tube did not clear source rack')
    stage='carry';servo([*target,.26],8);note('carry')
    stage='insert';servo([*target,release_height],8);note('lower')
    stage='release';go(np.r_[d.ctrl[acts[:5]],.5],2);note('release')
    stage='retreat';move([*target,.285],.5,3)
    stage='settle'
    for _ in range(1000):
        step();samples.append(tube_position(d))
    note('end')
    pos=tube_position(d);error=pos[:2]-target
    jid=m.joint('tube_free').id;v=d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid]+6]
    drift=float(np.max(np.linalg.norm(np.asarray(samples[-500:])-pos,axis=1)))
    seated=bool(np.linalg.norm(error)<.002 and abs(pos[2]-TUBE_CENTER_Z)<.002 and tube_tilt(d)<MAX_TILT and drift<.001)
    result={'success':bool(seated and worst<MAX_TILT),'seated':seated,'upright':bool(worst<MAX_TILT),'worst_tilt':float(worst),'final_tilt':tube_tilt(d),'position':pos.tolist(),'target_error_mm':(error*1000).tolist(),'max_center_height':max_z,'settle_drift_m':drift,'final_linear_speed_m_s':float(np.linalg.norm(v[:3])),'stages':stages,'source_index':source_index,'target_index':target_index,'speed':speed,'grip':grip,'initial_tilt':list(initial_tilt),'controller':'simulator-state Cartesian feedback, joint actuators only','physics':{'pads':False,'object_weld':False,'object_pose_writes_after_reset':False,'applied_object_force':False,'rack_collision_enabled':True,'actuator_torque_limit':3.35},'camera_names':['overhead','wrist'],'source_xy':source.tolist(),'target_xy':target.tolist(),'rack_separation_m':float(np.linalg.norm(target-source))}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,default=Path('outputs/tube_physical_validation.json'));a=p.parse_args()
    m,d=load_tube_model();result=run_tube_cycle(m,d)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if not result['success']:raise SystemExit(1)
if __name__=='__main__':main()

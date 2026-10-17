"""Robot-only forward/inverse kinematics from encoder readings and calibration."""
import numpy as np,mujoco
from scipy.optimize import least_squares
from small_part.vision_geometry import Camera
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
OFFSET=np.array([.004,0,-.085])
class RobotKinematics:
    def __init__(self,model,width=1600,height=1200):
        self.model=model;self.probe=mujoco.MjData(model)
        self.qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS]);self.jids=np.array([model.joint(n).id for n in JOINTS]);self.width=width;self.height=height
        self.low=model.jnt_range[self.jids[:5],0]+.001;self.high=model.jnt_range[self.jids[:5],1]-.001
    def pose(self,joints):
        self.probe.qpos[self.qadr]=joints;mujoco.mj_kinematics(self.model,self.probe);mujoco.mj_camlight(self.model,self.probe)
        b=self.probe.body('gripper');R=b.xmat.reshape(3,3).copy();tip=b.xpos+R@OFFSET
        cams=[]
        for name in ('overhead','wrist'):
            c=self.model.camera(name).id;cams.append(Camera(self.probe.cam_xpos[c].copy(),self.probe.cam_xmat[c].reshape(3,3).copy(),float(self.model.cam_fovy[c]),self.width,self.height))
        return tip.copy(),R,cams
    def solve(self,point,seed,axis=None):
        def residual(q):
            joints=np.r_[q,seed[5]];tip,R,_=self.pose(joints)
            orient=R[:2,1] if axis is None else R[:,1]-axis
            return np.r_[(tip-point)*10,orient]
        return least_squares(residual,np.clip(seed[:5],self.low+1e-7,self.high-1e-7),bounds=(self.low,self.high),max_nfev=45).x

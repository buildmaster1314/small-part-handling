"""Privileged ground-truth scoring kept outside the policy input interface."""
from collections import deque
import numpy as np
import mujoco
from small_part.tube_scene import tube_position,tube_tilt,TUBE_CENTER_Z,MAX_TILT
from small_part.tube_cycle_physical import OFFSET

class TubeEvaluator:
    def __init__(self,model,target):
        self.target=np.asarray(target);self.worst_tilt=0.;self.max_height=TUBE_CENTER_Z
        self.tail=deque(maxlen=round(1/model.opt.timestep));self.warnings=False
        self.housing_contacts=0
    def update(self,model,data):
        self.worst_tilt=max(self.worst_tilt,tube_tilt(data));pos=tube_position(data)
        self.max_height=max(self.max_height,float(pos[2]));self.tail.append(pos)
        self.warnings=self.warnings or bool(np.any(data.warning.number)) or not np.isfinite(data.qpos).all()
        self.housing_contacts+=sum(any(f>=2 for f in c.flex) for c in data.contact)
    def result(self,model,data):
        pos=tube_position(data);error=pos[:2]-self.target
        drift=float(np.max(np.linalg.norm(np.asarray(self.tail)-pos,axis=1))) if self.tail else float('inf')
        body=data.body('gripper');tip=body.xpos+body.xmat.reshape(3,3)@OFFSET
        jaw=float(data.qpos[model.jnt_qposadr[model.joint('gripper').id]])
        released=jaw>.35 and np.linalg.norm(tip-pos)>.06
        seated=np.linalg.norm(error)<.002 and abs(pos[2]-TUBE_CENTER_Z)<.002 and drift<.001
        upright=self.worst_tilt<MAX_TILT;lifted=self.max_height>.245
        return dict(success=bool(seated and upright and lifted and released and not self.warnings),seated=bool(seated),upright=bool(upright),lifted=bool(lifted),released_and_retreated=bool(released),worst_tilt=self.worst_tilt,final_tilt=tube_tilt(data),final_within_two_degrees=bool(tube_tilt(data)<np.deg2rad(2)),target_error_mm=(error*1000).tolist(),position=pos.tolist(),settle_drift_m=drift,max_center_height=self.max_height,warnings=self.warnings,housing_contact_samples=self.housing_contacts)

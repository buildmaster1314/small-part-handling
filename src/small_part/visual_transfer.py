"""Event-driven visual transfer. No demonstration, time-indexed path or LeWM prior.

A geometric visual controller is an explicit baseline, NOT a learned LeWM
policy. Known dimensions and robot/camera calibration are permitted priors.
"""
from dataclasses import dataclass
import numpy as np
from small_part.vision_geometry import TubeTracker,orange_holes

@dataclass(frozen=True)
class VisualObservation:
    images: tuple
    joints: np.ndarray

class VisualTransfer:
    def __init__(self,kinematics,target_index=None,dt=.1,max_retries=1,grip=-.015):
        self.kin=kinematics;self.target_index=target_index;self.dt=dt;self.max_retries=max_retries;self.grip=grip
        self.tracker=TubeTracker();self.stage='acquire';self.elapsed=0.;self.total=0.;self.retry_count=0;self.stable=0;self.lost=0
        self.source=None;self.target=None;self.last_center=None;self.last_axis=None;self.last_tip=None;self.logs=[];self.events=[];self.done=False;self.failure=None;self.command=None
    def transition(self,stage):
        self.events.append({'time':self.total,'from':self.stage,'to':stage});self.stage=stage;self.elapsed=0.;self.stable=0
    def stop(self,reason):self.failure=reason;self.done=True;self.transition('stopped')
    def _reached(self,condition,count=3):
        self.stable=self.stable+1 if condition else 0
        return self.stable>=count
    def act(self,obs):
        if type(obs) is not VisualObservation or obs.joints.shape!=(6,):raise TypeError('Explicit RGB/encoder interface required')
        q=obs.joints.copy();tip,R,cams=self.kin.pose(q)
        if self.command is None:self.command=q.copy()
        self.total+=self.dt;self.elapsed+=self.dt
        if self.done:return self.command.copy()
        try:
            if self.target is None:
                holes,diagnostics=orange_holes(obs.images[0],cams[0]);self.hole_diagnostics=diagnostics;self.holes=holes
            predicted=None if self.last_center is None else self.last_center.copy()
            if predicted is not None and self.stage in ('lift','carry','insert'):
                predicted+=tip-self.last_tip
            p,n,info=self.tracker.estimate(obs.images,cams,predicted_center=predicted,predicted_axis=self.last_axis,supported=self.stage in ('acquire','hover','descend','grasp','retry_open','retry_retreat','settle'))
            if self.last_center is not None and np.linalg.norm(p-self.last_center)>.025:raise ValueError('Visual pose discontinuity')
            self.last_center=p;self.last_axis=n;self.last_tip=tip;self.lost=0
        except ValueError as e:
            self.lost+=1;self.logs.append({'time':self.total,'stage':self.stage,'vision_error':str(e),'holding':True})
            if self.lost>=10:self.stop('visual_localization_lost')
            return self.command.copy()
        if self.target is None:
            self.source=p[:2].copy()
            if self.target_index is None:self.target_index=int(np.argmin(np.linalg.norm(self.holes-p[:2],axis=1)))
            if self.target_index not in range(len(self.holes)):raise ValueError('Requested visual hole index unavailable')
            self.target=self.holes[self.target_index].copy();self.transition('hover')
        tilt=float(np.arccos(np.clip(n[2],-1,1)))
        self.logs.append({'time':self.total,'stage':self.stage,'center':p.tolist(),'axis':n.tolist(),'estimated_tilt':tilt,'target':self.target.tolist(),'tip':tip.tolist(),**info})
        if tilt>.16:self.stop('visual_excessive_tilt');return self.command.copy()
        if self.total>80:self.stop('task_timeout');return self.command.copy()
        jaw=self.grip;goal=None;orientation=None
        if self.stage in ('hover','descend'):
            self.source=.8*self.source+.2*p[:2]
            goal=np.r_[self.source,.285 if self.stage=='hover' else .198];jaw=.5
            if self._reached(np.linalg.norm(goal-tip)<.0015):self.transition('descend' if self.stage=='hover' else 'grasp')
        elif self.stage=='grasp':
            jaw=self.grip
            if self.elapsed>.6:
                correction=np.cross(n,[0,0,1]);orientation=R[:,1]+np.cross(correction,R[:,1])*.12;orientation/=np.linalg.norm(orientation)
                goal=tip+np.r_[np.clip(self.source-p[:2],-.0003,.0003),0.]
            if self.elapsed>3.:
                self.grasp_height=p[2];self.transition('lift')
        elif self.stage in ('lift','carry','insert'):
            destination=np.r_[self.source if self.stage=='lift' else self.target,.26 if self.stage!='insert' else .1615]
            error=destination-p
            delta=np.clip(error,-.0015,.0015)
            goal=tip+delta
            correction=np.cross(n,[0,0,1]);orientation=R[:,1]+np.cross(correction,R[:,1])*(.20 if len(info['views'])==2 else .03);orientation/=np.linalg.norm(orientation)
            if self.stage=='lift':
                if self._reached(np.linalg.norm(error)<.0025):self.transition('carry')
                elif self.elapsed>9 and p[2]<.185:
                    if self.retry_count<self.max_retries and tilt<.10 and np.linalg.norm(p[:2]-self.source)<.012:
                        self.retry_count+=1;self.grip=max(-.04,self.grip-.01);self.transition('retry_open')
                    else:self.stop('failed_lift')
            elif self.stage=='carry':
                if p[2]<.22:self.stop('visual_slip_during_carry')
                elif self._reached(np.linalg.norm(error)<.002):self.transition('insert')
            else:
                if self._reached(np.linalg.norm(error[:2])<.0012 and abs(error[2])<.0013,count=5):self.transition('release')
                elif self.elapsed>18:self.stop('insertion_not_confirmed')
        elif self.stage=='release':
            jaw=.5
            if self.elapsed>2:self.transition('retreat')
        elif self.stage=='retreat':
            jaw=.5;goal=np.r_[self.target,.285]
            if self._reached(np.linalg.norm(goal-tip)<.002):self.transition('settle')
        elif self.stage=='settle':
            jaw=.5
            if self.elapsed>2:
                self.done=True;self.transition('complete')
        elif self.stage=='retry_open':
            jaw=.5
            if self.elapsed>2:self.transition('retry_retreat')
        elif self.stage=='retry_retreat':
            jaw=.5;goal=np.r_[p[:2],.25]
            if self._reached(np.linalg.norm(goal-tip)<.002):self.source=p[:2].copy();self.transition('descend')
        if goal is not None:
            if self.stage in ('hover','descend','retreat','retry_retreat'):
                delta=goal-tip;length=np.linalg.norm(delta);goal=tip+delta*min(1,.0025/max(length,1e-12))
            arm=self.kin.solve(goal,q,orientation)
        else:arm=self.command[:5]
        # Per-tick slew bound, no recorded nominal action sequence.
        command=np.r_[arm,jaw];command=self.command+np.clip(command-self.command,-.035,.035)
        self.command=command
        return command.copy()

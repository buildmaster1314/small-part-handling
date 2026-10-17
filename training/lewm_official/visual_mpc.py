"""Image-only LeWM candidate scoring with an explicit demonstration warm start.

This module deliberately does not import MuJoCo. No live object pose, contact,
segmentation ID, reward or evaluator state is accepted by the policy interface.
It is demonstration-guided MPC, not a learned skill without a reference.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from runtime import build, preprocess

@dataclass(frozen=True)
class Observation:
    pixels: np.ndarray
    joints: np.ndarray
    previous_action: np.ndarray | None

class VisualMPC:
    def __init__(self, upstream, backbone, checkpoint, reference, limits, *, weights=None,
                 mode='model', horizon=5, candidates=12, radius=.003, seed=42):
        if mode not in ('model','reference','blind'):raise ValueError(mode)
        self.mode=mode;self.horizon=horizon;self.candidates=candidates;self.radius=radius
        self.limits=np.asarray(limits);self.rng=np.random.default_rng(seed)
        with np.load(reference) as f:
            self.reference_actions=f['actions'].astype('float32')
            self.reference_frames=f['frames'].copy()
        if len(self.reference_frames)!=len(self.reference_actions)+1 or len(self.reference_actions)<3:raise ValueError('Reference alignment')
        if self.limits.shape!=(6,2) or not np.isfinite(self.reference_actions).all():raise ValueError('Invalid actions or limits')
        if horizon<1 or candidates<2 or radius<0:raise ValueError('Planning configuration')
        self.model,_=build(upstream,backbone)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if weights:self.model.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        else:self.model.load_state_dict(ck['harness'])
        self.model.eval();self.mean=ck['action_mean'];self.std=ck['action_std']
        self.images=[];self.past_actions=[];self.i=0;self.pending=[];self.logs=[]
        with torch.no_grad():
            goal=[]
            for j in range(0,len(self.reference_frames),16):
                x=torch.from_numpy(self.reference_frames[j:j+16]).permute(0,3,1,2).unsqueeze(1)
                goal.append(self.model.model.encode({'pixels':preprocess(x)})['emb'][:,0])
            self.goals=torch.cat(goal)
    def _normalize(self,commands):
        return (torch.as_tensor(commands,dtype=torch.float32).flatten(-2)-self.mean)/self.std
    @torch.no_grad()
    def act(self,obs:Observation):
        if type(obs) is not Observation:raise TypeError('Only the explicit observation interface is accepted')
        if obs.joints.shape!=(6,) or obs.pixels.shape!=(96,192,3) or not np.isfinite(obs.joints).all():raise ValueError('Observation shape')
        if self.i and (obs.previous_action is None or obs.previous_action.shape!=(50,6)):raise ValueError('Missing executed action')
        self.images.append(obs.pixels.copy());self.images=self.images[-3:]
        if obs.previous_action is not None:self.past_actions.append(obs.previous_action.copy());self.past_actions=self.past_actions[-2:]
        if not self.pending:
            self.pending=list(self._plan())
        action=self.pending.pop(0).copy();self.i+=1
        return action
    def _plan(self):
        H=min(self.horizon,len(self.reference_actions)-self.i)
        if H<=0:return [self.reference_actions[-1].copy()]
        nominal=self.reference_actions[self.i:self.i+H].copy()
        # Radius is a joint-command trust region, explicitly reported in results.
        offsets=self.rng.uniform(-self.radius,self.radius,(self.candidates,H,1,6)).astype('float32')
        offsets[0]=0;offsets[:,:,:,5]*=.5
        # Smooth offsets across blocks to avoid instantaneous command jumps.
        offsets=np.cumsum(offsets,axis=1)/np.arange(1,H+1,dtype='float32')[None,:,None,None]
        candidate=nominal[None]+offsets
        candidate=np.clip(candidate,self.limits[:,0],self.limits[:,1])
        pix=self.images if self.mode!='blind' else list(self.reference_frames[max(0,self.i-2):self.i+1])
        pix=[pix[0]]*(3-len(pix))+pix
        x=preprocess(torch.from_numpy(np.stack(pix)).permute(0,3,1,2).unsqueeze(0))
        encoded=self.model.model.encode({'pixels':x})['emb'].expand(self.candidates,-1,-1).clone()
        history=self.past_actions
        if len(history)<2:history=[np.repeat(self.reference_actions[0,:1],50,axis=0)]*(2-len(history))+history
        act=self._normalize(np.stack(history)).unsqueeze(0).expand(self.candidates,-1,-1).clone()
        future=self._normalize(candidate);cost=torch.zeros(self.candidates)
        for j in range(H):
            act=torch.cat([act,future[:,j:j+1]],dim=1)
            predicted=self.model.model.predict(encoded[:,-3:],self.model.model.action_encoder(act[:,-3:]))[:,-1:]
            encoded=torch.cat([encoded,predicted],dim=1)
            goal=self.goals[min(self.i+j+1,len(self.goals)-1)]
            cost+=(predicted[:,0]-goal).square().mean(-1)/H
        if not torch.isfinite(cost).all():raise RuntimeError('Non-finite model planning cost')
        selected=0 if self.mode=='reference' else int(cost.argmin())
        self.logs.append({'frame':self.i,'candidate':selected,'nominal_cost':float(cost[0]),'selected_cost':float(cost[selected]),'cost_spread':float(cost.max()-cost.min()),'max_command_delta':float(np.abs(candidate[selected]-nominal).max())})
        return candidate[selected]

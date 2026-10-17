"""Fixed-window, recursive held-out audit. Future measured states are targets only."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import torch
from residual_runtime import build_residual, load_checkpoint, predict


def main():
    p=argparse.ArgumentParser()
    for name in ('fcpg-root','upstream','backbone-source','base-weights','checkpoint','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),required=True)
    p.add_argument('--per-episode',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(713)
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    model=build_residual(a.fcpg_root,a.upstream,a.backbone_source,a.base_weights)
    checkpoint=load_checkpoint(model,a.checkpoint,a.base_weights);stats=checkpoint['statistics']
    assert stats['base_weights_sha256']==data['base_weights_sha256']
    for key in ('state_mean','state_std','base_action_mean','base_action_std'):
        assert torch.equal(stats[key],data[key]),key
    episodes={k:e for k,e in data['episodes'].items() if e['split']==a.split}
    records=[]
    with torch.no_grad():
        for key,e in episodes.items():
            starts=np.unique(np.linspace(0,len(e['embedding'])-13,a.per_episode).round().astype(int))
            em=torch.stack([e['embedding'][i:i+13] for i in starts])
            ac=torch.stack([e['actions'][i:i+12] for i in starts])
            st=torch.stack([e['states'][i:i+13] for i in starts])
            # Cyclic shifts within the same episode keep the ablation distribution
            # local. All variants share identical starting windows and targets.
            for variant in ('residual','base','shuffled_action','shuffled_state','shuffled_image'):
                eh=em[:,:3].clone();sh=st[:,:3].clone()
                if variant=='shuffled_state':sh=sh.roll(1,0)
                if variant=='shuffled_image':eh=eh.roll(1,0)
                actions=ac.roll(1,0) if variant=='shuffled_action' else ac
                for step in range(10):
                    base,visual,command,future=predict(model,eh,actions[:,step:step+3],sh,stats)
                    vp=base[:,-1] if variant=='base' else visual[:,-1]
                    sp=command[:,-1] if variant=='base' else future[:,-1]
                    if step+1 in (1,5,10):
                        ve=(vp-em[:,step+3]).square().mean(-1)
                        se=(sp-st[:,step+3]).square().mean(-1)
                        copyv=(em[:,2]-em[:,step+3]).square().mean(-1)
                        copys=(st[:,2]-st[:,step+3]).square().mean(-1)
                        cmds=(ac[:,step+2,-6:]-st[:,step+3]).square().mean(-1)
                        for j,start in enumerate(starts):
                            records.append(dict(episode=key,parent=e['parent'],start=int(start),variant=variant,horizon=step+1,
                                visual_mse=float(ve[j]),state_mse_rad2=float(se[j]),state_joint_mse_rad2=(sp[j]-st[j,step+3]).square().tolist(),visual_hold=float(copyv[j]),state_hold=float(copys[j]),state_command=float(cmds[j])))
                    eh=torch.cat((eh[:,1:],vp[:,None]),1)
                    sh=torch.cat((sh[:,1:],sp[:,None]),1)
    summary={}
    for h in (1,5,10):
        groups={v:[r for r in records if r['horizon']==h and r['variant']==v] for v in ('residual','base','shuffled_action','shuffled_state','shuffled_image')}
        means={v:{k:float(np.mean([r[k] for r in rows])) for k in ('visual_mse','state_mse_rad2','visual_hold','state_hold','state_command')} for v,rows in groups.items()}
        main=means['residual'];main['visual_vs_hold']=main['visual_mse']/max(main['visual_hold'],1e-12)
        main['state_vs_hold']=main['state_mse_rad2']/max(main['state_hold'],1e-12)
        main['state_vs_command']=main['state_mse_rad2']/max(main['state_command'],1e-12)
        main['state_rmse_deg']=float(np.rad2deg(np.sqrt(main['state_mse_rad2'])))
        main['per_joint_rmse_deg']=np.rad2deg(np.sqrt(np.mean([r['state_joint_mse_rad2'] for r in groups['residual']],axis=0))).tolist()
        paired=list(zip(groups['residual'],groups['shuffled_action']))
        main['state_action_rank_win']=float(np.mean([x['state_mse_rad2']<y['state_mse_rad2'] for x,y in paired]))
        main['visual_action_rank_win']=float(np.mean([x['visual_mse']<y['visual_mse'] for x,y in paired]))
        summary[str(h)]=means
    # Split-specific gate; passing validation alone never authorizes a claim of
    # held-out success. Joint forecasting and visual forecasting are independent.
    state_gate=all(summary[str(h)]['residual']['state_vs_hold']<1 and summary[str(h)]['residual']['state_vs_command']<1 and summary[str(h)]['residual']['state_action_rank_win']>.6 for h in (1,5))
    visual_gate=all(summary[str(h)]['residual']['visual_vs_hold']<1 and summary[str(h)]['residual']['visual_action_rank_win']>.6 for h in (1,5))
    report=dict(split=a.split,checkpoint_step=checkpoint['step'],checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),episodes=len(episodes),independent_parent_episodes=len({e['parent'] for e in episodes.values()}),windows_per_horizon=len(groups['residual']),joint_order=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'],history_frames=3,horizons_seconds=[.1,.5,1.],future_measured_state_input=False,selection='16 evenly spaced windows per whole episode; fixed before test evaluation',summary=summary,joint_forecast_gate=state_gate,visual_planning_gate=visual_gate,control_success_proven=False,records=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()

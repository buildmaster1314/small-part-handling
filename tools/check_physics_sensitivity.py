from pathlib import Path
from dataclasses import asdict
import json,argparse
from small_part.tube_scene import PhysicsConfig,load_tube_model
from small_part.tube_cycle_physical import run_tube_cycle
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/tube_suite/sensitivity'));a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=True)
for mu in [.01,.2,.5]:
 for noslip in [0,20]:
  path=out/f'mu{mu}_noslip{noslip}.json'
  if path.exists():continue
  config=PhysicsConfig(wall_friction=mu,noslip_iterations=noslip)
  m,d=load_tube_model(config)
  try:r=run_tube_cycle(m,d,abort_on_failed_lift=False)
  except Exception as e:r={'success':False,'error':str(e)}
  r['physics_config']=asdict(config);path.write_text(json.dumps(r,indent=2));print('CASE',mu,noslip,r['success'],r.get('worst_tilt'),flush=True)

"""Stratify fixed audit windows; do not change training or select by test data."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=json.loads(a.input.read_text());groups={}
for horizon in (1,5,10):
    groups[str(horizon)]={}
    for name in ('full_transfer','exploration'):
        rows=[r for r in data['records'] if r['horizon']==horizon and r['episode'].startswith('explore_')==(name=='exploration')]
        results={}
        for variant in ('residual','base','shuffled_action','shuffled_state','shuffled_image'):
            rr=[r for r in rows if r['variant']==variant]
            metrics={key:float(np.mean([r[key] for r in rr])) for key in ('visual_mse','visual_hold','state_mse_rad2','state_hold','state_command')}
            metrics['visual_vs_hold']=metrics['visual_mse']/max(metrics['visual_hold'],1e-12)
            metrics['state_vs_command']=metrics['state_mse_rad2']/max(metrics['state_command'],1e-12)
            metrics['state_vs_hold']=metrics['state_mse_rad2']/max(metrics['state_hold'],1e-12)
            results[variant]=metrics
        groups[str(horizon)][name]=results
report=dict(source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),split=data['split'],independent_parents=len({r['parent'] for r in data['records']}),note='Exploration clips are not complete transfers. Ratios below compare identical windows and normalization, not historical evaluations with different window selection.',groups=groups)
a.output.write_text(json.dumps(report,indent=2))

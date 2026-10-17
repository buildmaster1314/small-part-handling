"""Regression checks for the upstream model adapter and checkpoint boundary."""
from pathlib import Path
import argparse,json
import torch
from residual_runtime import build_residual,load_checkpoint,predict

p=argparse.ArgumentParser()
for name in ('fcpg-root','upstream','backbone-source','base-weights','checkpoint','cache','output'):
    p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4)
cache=torch.load(a.cache,map_location='cpu',weights_only=False)
model=build_residual(a.fcpg_root,a.upstream,a.backbone_source,a.base_weights)
ck=load_checkpoint(model,a.checkpoint,a.base_weights);stats=ck['statistics']
e=next(e for e in cache['episodes'].values() if e['split']=='validation')
emb=e['embedding'][:3][None];act=e['actions'][:3][None];state=e['states'][:3][None]
with torch.no_grad():
    expected=predict(model,emb,act,state,stats)
    assert torch.allclose(expected[2],act[:,:,-6:],atol=2e-7),'Shared six-coordinate normalization broken'
    assert not any(p.requires_grad for p in model.base.parameters())
    for altered in range(3):
        inputs=[emb.clone(),act.clone(),state.clone()]
        inputs[altered][:,1:]+=.7
        outputs=predict(model,*inputs,stats)
        for x,y in zip(expected,outputs):
            assert torch.allclose(x[:,:1],y[:,:1],atol=2e-6),'Future context leaked into earlier prediction'
    assert ((expected[1]-expected[0]).abs()<=.250001).all()
    assert (((expected[3]-expected[2])/stats['state_std']).abs()<=1.000001).all()
    again=build_residual(a.fcpg_root,a.upstream,a.backbone_source,a.base_weights)
    load_checkpoint(again,a.checkpoint,a.base_weights)
    for x,y in zip(expected,predict(again,emb,act,state,stats)):assert torch.equal(x,y)
result=dict(shared_state_command_normalization=True,original_base_action_normalization_preserved=True,base_frozen=True,causal_for_visual_action_and_state_context=True,bounded_residuals=True,checkpoint_reload_bit_exact=True,step=ck['step'])
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

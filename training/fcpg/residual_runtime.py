"""Adapter for the unmodified FCPG SO-101 temporal State-Residual implementation."""
from pathlib import Path
import sys,torch
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[1]/'lewm_official'))
from runtime import build,preprocess

def build_residual(fcpg_root,upstream,backbone,base_weights):
    sys.path.insert(0,str(fcpg_root))
    from fcpg.so101_state_residual_lewm import TemporalStateResidualLeWM
    harness,_=build(upstream,backbone)
    harness.model.load_state_dict(torch.load(base_weights,map_location='cpu',weights_only=True),strict=True)
    for p in harness.model.parameters():p.requires_grad_(False)
    model=TemporalStateResidualLeWM(harness.model)
    model.base.eval()
    return model

def predict(model,embedding,raw_actions,raw_states,stats):
    # The inherited LeWM uses per-position 300D action z-scores. The residual
    # branch must use the SAME six-coordinate scale for measured state and final
    # command; supply the original encoded action through the upstream API.
    base_actions=(raw_actions-stats['base_action_mean'])/stats['base_action_std']
    action_embedding=model.base.action_encoder(base_actions)
    common_actions=((raw_actions.reshape(*raw_actions.shape[:2],50,6)-stats['state_mean'])/stats['state_std']).flatten(-2)
    state=(raw_states-stats['state_mean'])/stats['state_std']
    base,visual,command,future=model.predict_context(embedding,common_actions,state,action_embedding=action_embedding)
    return base,visual,command*stats['state_std']+stats['state_mean'],future*stats['state_std']+stats['state_mean']

def load_checkpoint(model,path,base_weights):
    import hashlib
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    expected=checkpoint['statistics']['base_weights_sha256']
    if hashlib.sha256(Path(base_weights).read_bytes()).hexdigest()!=expected:
        raise ValueError('Residual checkpoint belongs to a different frozen base')
    missing,unexpected=model.load_state_dict(checkpoint['branch'],strict=False)
    if unexpected or any(not key.startswith('base.') for key in missing):
        raise ValueError(f'Incomplete residual checkpoint: {missing}, {unexpected}')
    model.eval()
    return checkpoint

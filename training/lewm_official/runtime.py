"""Load the unmodified upstream LeWM model and upstream loss function.

Only data, orchestration and CPU runtime are local. Function AST loading avoids
importing the optional Lightning/W&B/environment stack; function bodies are
compiled directly from the pinned upstream sources, not reimplemented.
"""
from pathlib import Path
import ast,hashlib,importlib.util,logging,sys
import torch
from torch import nn
from transformers import ViTConfig,ViTModel
from omegaconf import OmegaConf

def source_function(path,name,namespace):
    tree=ast.parse(Path(path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if node.decorator_list:raise ValueError('Expected undecorated upstream function')
    unit=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(unit)
    exec(compile(unit,str(path),'exec'),namespace)
    return namespace[name]

def build(upstream,backbone_source,action_dim=300):
    upstream=Path(upstream);sys.path.insert(0,str(upstream))
    from jepa import JEPA
    from module import SIGReg,ARPredictor,Embedder,MLP
    vit=source_function(backbone_source,'vit_hf',{'nn':nn,'ViTConfig':ViTConfig,'ViTModel':ViTModel,'_TRANSFORMERS_AVAILABLE':True,'logging':logging.getLogger('official_vit')})
    cfg=OmegaConf.create({'img_size':224,'embed_dim':192,'history_size':3,'num_preds':1,'loss':{'sigreg':{'weight':.09,'kwargs':{'knots':17,'num_proj':1024}}},'model':OmegaConf.load(upstream/'config/train/model/lewm.yaml')})
    cfg.model.action_encoder.input_dim=action_dim
    args=OmegaConf.to_container(cfg.model,resolve=True)
    def keywords(k):return {name:value for name,value in args[k].items() if name!='_target_'}
    enc=vit(**keywords('encoder'))
    def mlp(k):
        kw=keywords(k);kw['norm_fn']=nn.BatchNorm1d;return MLP(**kw)
    model=JEPA(encoder=enc,predictor=ARPredictor(**keywords('predictor')),action_encoder=Embedder(**keywords('action_encoder')),projector=mlp('projector'),pred_proj=mlp('pred_proj'))
    objective=source_function(upstream/'train.py','lejepa_forward',{'torch':torch})
    class Harness(nn.Module):
        def __init__(self):
            super().__init__();self.model=model;self.sigreg=SIGReg(**cfg.loss.sigreg.kwargs)
        def log_dict(self,*args,**kwargs):pass
        def forward(self,batch,stage='train'):return objective(self,batch,stage,cfg)
    return Harness(),cfg

def preprocess(pixels):
    # Two fixed camera images occupy the left/right halves of the recorded mosaic.
    # Explicit dataset adaptation: resize the mosaic to the 224x224 ViT input.
    x=pixels.float()/255
    shape=x.shape;x=x.reshape(-1,*shape[-3:])
    x=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return ((x-mean)/std).reshape(*shape[:-3],3,224,224)

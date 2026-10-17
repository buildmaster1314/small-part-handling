#!/usr/bin/env python3
"""Render a four-panel human audit of one read-only SO-101 shadow session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fcpg.so101_kinematics import SO101_ANCHOR_NAMES, SO101MuJoCoKinematics
from fcpg.so101_shadow_session import load_shadow_session
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from scripts.train_so101_lewm import IMAGENET_MEAN, IMAGENET_STD
from scripts.visualize_so101_lewm import (
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    predict_state_residual_batch,
)


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--so101-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def encode_frames(
    model: torch.nn.Module, frames: np.ndarray, batch_size: int
) -> np.ndarray:
    device = next(model.parameters()).device
    parts: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        values = np.asarray(frames[start : start + batch_size]).copy()
        pixels = torch.from_numpy(values).permute(0, 3, 1, 2).float().div_(255.0)
        pixels = ((pixels - IMAGENET_MEAN) / IMAGENET_STD).to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            latent = model.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
        parts.append(latent.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def action_chunk(
    actions: np.ndarray,
    start: int,
    frameskip: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(actions[start : start + frameskip], dtype=np.float32)
    if raw.shape != (frameskip, 6):
        raise ValueError(f"incomplete action chunk at session frame {start}")
    return ((raw - center) / scale).reshape(frameskip * 6)


def html_document(report: dict) -> str:
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 shadow inference</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #111416; color: #eef2f3; }}
header {{ padding: 14px 18px; border-bottom: 1px solid #30383c; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
h1 {{ font-size: 18px; margin: 0; }}
.meta {{ color:#aebbc0; font-size:13px; }}
.controls {{ margin-left:auto; display:flex; align-items:center; gap:10px; }}
button {{ width:36px; height:32px; border:1px solid #59666b; background:#22292c; color:white; border-radius:4px; }}
input[type=range] {{ width:min(440px,42vw); }}
main {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#30383c; min-height:calc(100vh - 64px); }}
.panel {{ background:#171b1d; min-height:360px; position:relative; display:flex; flex-direction:column; }}
.label {{ padding:9px 12px; font-size:13px; color:#c7d0d3; border-bottom:1px solid #30383c; }}
.image-wrap {{ flex:1; display:grid; place-items:center; min-height:0; }}
img {{ width:100%; height:100%; max-height:430px; object-fit:contain; image-rendering:auto; }}
canvas {{ width:100%; height:100%; min-height:310px; }}
.readout {{ position:absolute; right:10px; bottom:9px; padding:5px 7px; background:#0d1113d9; font:12px ui-monospace,monospace; }}
@media(max-width:800px) {{ main {{ grid-template-columns:1fr; }} .panel {{ min-height:300px; }} }}
</style>
</head>
<body>
<header><h1>SO-101 Shadow Inference</h1><div class="meta" id="meta"></div>
<div class="controls"><button id="play" title="播放/暂停">▶</button><input id="slider" type="range" min="0" value="0"><span id="counter"></span></div></header>
<main>
<section class="panel"><div class="label">最近观测（模型输入）</div><div class="image-wrap"><img id="input"></div></section>
<section class="panel"><div class="label">真实未来（仅用于人工对照）</div><div class="image-wrap"><img id="target"></div></section>
<section class="panel"><div class="label">未来六关节：LeWM 预测 vs 真实</div><canvas id="joints"></canvas><div class="readout" id="error"></div></section>
<section class="panel"><div class="label">三锚点 FK：LeWM 预测 vs 真实</div><canvas id="anchors"></canvas></section>
</main>
<script>
const report={payload}; const rows=report.rows; let i=0,timer=null;
const slider=document.getElementById('slider'); slider.max=Math.max(0,rows.length-1);
function canvasSize(c){{const d=devicePixelRatio||1;c.width=Math.floor(c.clientWidth*d);c.height=Math.floor(c.clientHeight*d);return [c.getContext('2d'),d];}}
function axes(ctx,w,h,title){{ctx.fillStyle='#171b1d';ctx.fillRect(0,0,w,h);ctx.fillStyle='#aebbc0';ctx.font='13px system-ui';ctx.fillText(title,14,22);}}
function drawJoints(r){{const c=document.getElementById('joints'),[x]=canvasSize(c),w=c.width,h=c.height;axes(x,w,h,'蓝色真实 / 红色预测');const names=['pan','lift','elbow','wrist','roll','grip'];const left=90,right=w-24,top=50,row=(h-top-24)/6;for(let k=0;k<6;k++){{const y=top+k*row;x.fillStyle='#b8c3c7';x.fillText(names[k],14,y+5);x.strokeStyle='#3d474b';x.beginPath();x.moveTo(left,y);x.lineTo(right,y);x.stroke();for(const [v,color] of [[r.actual_state[k],'#25a7b8'],[r.predicted_state[k],'#ef5b5b']]){{const p=left+(Math.max(-110,Math.min(110,v))+110)/220*(right-left);x.fillStyle=color;x.beginPath();x.arc(p,y,6,0,Math.PI*2);x.fill();}}}}}}
function drawAnchors(r){{const c=document.getElementById('anchors'),[x]=canvasSize(c),w=c.width,h=c.height;axes(x,w,h,'XY (left) / XZ (right), circles=actual, crosses=predicted');const colors=['#f5c451','#43c59e','#e9724c'];function pane(cx,cy,pw,ph,a,b){{const all=[...r.actual_anchors,...r.predicted_anchors];const va=all.map(v=>v[a]),vb=all.map(v=>v[b]);let amin=Math.min(...va),amax=Math.max(...va),bmin=Math.min(...vb),bmax=Math.max(...vb);const pad=.02;amin-=pad;amax+=pad;bmin-=pad;bmax+=pad;const pt=v=>[cx+(v[a]-amin)/(amax-amin)*pw,cy+ph-(v[b]-bmin)/(bmax-bmin)*ph];x.strokeStyle='#384247';x.strokeRect(cx,cy,pw,ph);r.actual_anchors.forEach((v,k)=>{{const p=pt(v);x.fillStyle=colors[k];x.beginPath();x.arc(p[0],p[1],6,0,Math.PI*2);x.fill();}});r.predicted_anchors.forEach((v,k)=>{{const p=pt(v);x.strokeStyle=colors[k];x.lineWidth=3;x.beginPath();x.moveTo(p[0]-6,p[1]-6);x.lineTo(p[0]+6,p[1]+6);x.moveTo(p[0]+6,p[1]-6);x.lineTo(p[0]-6,p[1]+6);x.stroke();}});}}pane(36,48,w/2-56,h-76,0,1);pane(w/2+20,48,w/2-56,h-76,0,2);}}
function show(n){{i=(n+rows.length)%rows.length;slider.value=i;const r=rows[i];document.getElementById('input').src=r.input_image;document.getElementById('target').src=r.target_image;document.getElementById('counter').textContent=`${{i+1}} / ${{rows.length}}`;document.getElementById('error').textContent=`joint RMSE ${{r.joint_rmse.toFixed(3)}}`;drawJoints(r);drawAnchors(r);}}
slider.oninput=()=>show(+slider.value);document.getElementById('play').onclick=(event)=>{{if(timer){{clearInterval(timer);timer=null;event.currentTarget.textContent='▶';}}else{{timer=setInterval(()=>show(i+1),500);event.currentTarget.textContent='Ⅱ';}}}};
document.getElementById('meta').textContent=`${{report.source}} | step ${{report.checkpoint_step}} | ${{report.frameskip}} raw steps ahead | FCPG waits for real camera calibration`;
window.onresize=()=>show(i);show(0);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda:1"):
        raise ValueError("CUDA1 is forbidden for this project")
    if args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("stride and batch-size must be positive")
    session = load_shadow_session(args.session)
    if session.actions is None:
        raise ValueError("LeWM shadow report requires recorded leader or policy actions")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"report output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "frames"
    image_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    checkpoint, embedded, _, frameskip, history, _ = checkpoint_contract(
        args.checkpoint.expanduser().resolve(), None
    )
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise ValueError("shadow report requires a State-Residual LeWM checkpoint")
    center, scale, normalization = checkpoint_action_normalizer(checkpoint, embedded)
    latents = encode_frames(model, session.frames, args.batch_size)
    destinations = np.arange(frameskip * history, session.frame_count, args.stride)
    if len(destinations) == 0:
        raise ValueError(
            f"session needs more than {frameskip * history} frames for this checkpoint"
        )
    records = []
    contexts = []
    chunks = []
    context_states = []
    for destination in destinations:
        indexes = np.arange(
            destination - frameskip * history, destination, frameskip, dtype=np.int64
        )
        contexts.append(latents[indexes])
        chunks.append(
            np.stack(
                [
                    action_chunk(session.actions, int(index), frameskip, center, scale)
                    for index in indexes
                ]
            )
        )
        context_states.append((np.asarray(session.states[indexes]) - center) / scale)
    predicted_parts = []
    for start in range(0, len(destinations), args.batch_size):
        _, prediction = predict_state_residual_batch(
            model,
            np.stack(contexts[start : start + args.batch_size]),
            np.stack(chunks[start : start + args.batch_size]),
            np.stack(context_states[start : start + args.batch_size]),
        )
        predicted_parts.append(prediction * scale + center)
    predicted = np.concatenate(predicted_parts, axis=0)

    kinematics = SO101MuJoCoKinematics(args.so101_model)
    actual = np.asarray(session.states[destinations], dtype=np.float32)
    actual_anchors = kinematics.forward_anchors(actual)
    predicted_anchors = kinematics.forward_anchors(predicted)
    for row_index, destination in enumerate(destinations):
        input_index = int(destination - frameskip)
        input_name = f"frames/input_{row_index:05d}.jpg"
        target_name = f"frames/target_{row_index:05d}.jpg"
        Image.fromarray(np.asarray(session.frames[input_index])).save(
            output / input_name, quality=88
        )
        Image.fromarray(np.asarray(session.frames[destination])).save(
            output / target_name, quality=88
        )
        records.append(
            {
                "destination": int(destination),
                "input_image": input_name,
                "target_image": target_name,
                "actual_state": actual[row_index].tolist(),
                "predicted_state": predicted[row_index].tolist(),
                "joint_rmse": float(np.sqrt(np.mean((predicted[row_index] - actual[row_index]) ** 2))),
                "actual_anchors": actual_anchors[row_index].tolist(),
                "predicted_anchors": predicted_anchors[row_index].tolist(),
            }
        )
    report = {
        "format": "so101-shadow-report-v1",
        "source": session.manifest["source"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "frameskip": frameskip,
        "history": history,
        "action_normalization": normalization,
        "anchor_names": list(SO101_ANCHOR_NAMES),
        "mean_joint_rmse": float(np.mean([row["joint_rmse"] for row in records])),
        "rows": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "index.html").write_text(html_document(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

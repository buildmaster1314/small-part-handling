#!/usr/bin/env python3
"""Render an interactive old-vs-adapted FCPG multiview audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpg.transformer_grounding import load_fcpg_vit_adapter  # noqa: E402
from scripts.train_fcpg_a4_multiview_adaptation import (  # noqa: E402
    _fixed_nominals,
)


DEFAULT_DATA = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "joint_orbit12_gripper_phaseaware_a4c_v4_48train_24dev"
)
DEFAULT_OLD = ROOT / (
    "outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/"
    "fcpg_transformer_residual_best.pt"
)
DEFAULT_ADAPTED = ROOT / (
    "outputs/transformer_residual/a4_multiview_projection_seed40601_2k/"
    "fcpg_a4_multiview_best.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/lewm_action_prefix_a4/"
    "fcpg_a4_multiview_adaptation_seed40601_step800_visual"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--old-checkpoint", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--adapted-checkpoint", type=Path, default=DEFAULT_ADAPTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=40601)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _cuda0(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("the multiview renderer permits only CUDA device 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


@torch.inference_mode()
def _predict(
    checkpoint: Path,
    arrays: dict[str, np.ndarray],
    nominal: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model, _ = load_fcpg_vit_adapter(checkpoint, device=device)
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, nominal.shape[0], batch_size):
        stop = min(start + batch_size, nominal.shape[0])
        rgb = torch.from_numpy(
            np.array(arrays["rgb"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        rgb.div_(255.0)
        q_nominal = torch.from_numpy(
            np.array(nominal[start:stop], dtype=np.float32, copy=True)
        ).to(device)
        xyz = torch.from_numpy(
            np.array(arrays["xyz"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        proprio = torch.from_numpy(
            np.array(arrays["proprio"][start:stop], dtype=np.float32, copy=True)
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(rgb, q_nominal, xyz, proprio)
        chunks.append(output.q_corrected_px.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _epe(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - truth, axis=-1).mean(axis=-1)


def _round_points(value: np.ndarray) -> list[list[float]]:
    return np.round(np.asarray(value, dtype=np.float32), 3).tolist()


def _html(samples_json: str, summary_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A4 multiview FCPG audit</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #1d252b; background: #eef1f2; }}
header {{ padding: 18px 24px 14px; background: #ffffff; border-bottom: 1px solid #ccd3d6; }}
h1 {{ margin: 0 0 5px; font-size: 22px; letter-spacing: 0; }}
.sub {{ color: #52616a; font-size: 13px; line-height: 1.45; max-width: 1100px; }}
.controls {{ display: flex; gap: 14px; align-items: end; flex-wrap: wrap; padding: 13px 24px; background: #f8f9f9; border-bottom: 1px solid #ccd3d6; }}
label {{ display: grid; gap: 4px; color: #52616a; font-size: 12px; }}
select, input, button {{ height: 34px; border: 1px solid #aeb8bc; border-radius: 4px; background: white; color: #1d252b; padding: 0 9px; }}
button {{ cursor: pointer; min-width: 74px; }}
input[type=range] {{ width: min(420px, 70vw); padding: 0; }}
.status {{ margin-left: auto; min-width: 260px; color: #35444c; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }}
main {{ padding: 16px 24px 28px; max-width: 1500px; margin: 0 auto; }}
.summary {{ display: grid; grid-template-columns: repeat(5, minmax(145px, 1fr)); gap: 8px; margin-bottom: 12px; }}
.metric {{ background: white; border: 1px solid #ccd3d6; border-radius: 6px; padding: 9px 11px; }}
.metric b {{ display: block; font-size: 17px; margin-top: 3px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
.panel {{ background: white; border: 1px solid #c7cfd2; border-radius: 6px; overflow: hidden; }}
.panel-head {{ padding: 9px 11px; border-bottom: 1px solid #d9dfe1; display: flex; justify-content: space-between; gap: 8px; }}
.panel-head strong {{ font-size: 14px; }}
.panel-head span {{ color: #52616a; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }}
canvas {{ display: block; width: 100%; aspect-ratio: 1 / 1; image-rendering: auto; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 11px; color: #46555d; font-size: 12px; }}
.swatch {{ display: inline-block; width: 11px; height: 11px; margin-right: 5px; vertical-align: -1px; }}
.truth {{ background: #16a36f; border-radius: 50%; }} .nominal {{ background: #e29b27; }} .corrected {{ background: #2c6fbb; transform: rotate(45deg); }}
.note {{ margin-top: 10px; color: #66757d; font-size: 12px; line-height: 1.45; }}
@media (max-width: 850px) {{ .grid {{ grid-template-columns: 1fr; }} .summary {{ grid-template-columns: repeat(2, 1fr); }} .status {{ margin-left: 0; }} }}
</style>
</head>
<body>
<header><h1>A4 multiview FCPG visual audit</h1><div class="sub">Same RGB and physical snapshot in all four panels. Clean uses the exact nominal projection. Drift changes only the nominal camera extrinsic; the image and ground-truth anchors stay fixed. This is a snapshot audit ordered by task phase, not a continuous policy rollout.</div></header>
<div class="controls">
  <label>Task<select id="task"></select></label>
  <label>Camera clock<select id="clock"></select></label>
  <button id="play" type="button">Play</button>
  <label>Speed<select id="speed"><option value="1000">1 fps</option><option value="500">2 fps</option><option value="250" selected>4 fps</option><option value="125">8 fps</option></select></label>
  <label>Snapshot<input id="frame" type="range" min="0" max="0" value="0"></label>
  <div class="status" id="status"></div>
</div>
<main>
  <div class="summary" id="summary"></div>
  <div class="grid">
    <section class="panel"><div class="panel-head"><strong>Clean / old FCPG</strong><span id="m0"></span></div><canvas id="c0" width="224" height="224"></canvas></section>
    <section class="panel"><div class="panel-head"><strong>Clean / adapted FCPG</strong><span id="m1"></span></div><canvas id="c1" width="224" height="224"></canvas></section>
    <section class="panel"><div class="panel-head"><strong>Drift / old FCPG</strong><span id="m2"></span></div><canvas id="c2" width="224" height="224"></canvas></section>
    <section class="panel"><div class="panel-head"><strong>Drift / adapted FCPG</strong><span id="m3"></span></div><canvas id="c3" width="224" height="224"></canvas></section>
  </div>
  <div class="legend"><span><i class="swatch truth"></i>true anchors</span><span><i class="swatch nominal"></i>drifted nominal</span><span><i class="swatch corrected"></i>FCPG corrected</span></div>
  <div class="note">Anchor order is left fingertip, right fingertip, and tool center. Development views are odd clock hours; training views are even clock hours. The adapted model is selected at step 800 using odd-view development error. Large clean corrections, especially clock11, remain a deployment risk.</div>
</main>
<script>
const SAMPLES = {samples_json};
const SUMMARY = {summary_json};
const taskSelect = document.getElementById('task');
const clockSelect = document.getElementById('clock');
const frame = document.getElementById('frame');
const play = document.getElementById('play');
const speed = document.getElementById('speed');
let current = [], timer = null;
const tasks = [...new Set(SAMPLES.map(x => x.task))];
for (const value of tasks) taskSelect.add(new Option(value, value));
taskSelect.value = tasks.includes('towel-wipe-v0') ? 'towel-wipe-v0' : tasks[0];
function clocksForTask() {{ return [...new Set(SAMPLES.filter(x => x.task === taskSelect.value).map(x => x.clock))].sort((a,b)=>a-b); }}
function fillClocks() {{ const previous = Number(clockSelect.value || 12); clockSelect.innerHTML=''; for (const value of clocksForTask()) clockSelect.add(new Option(String(value).padStart(2,'0')+':00', value)); clockSelect.value = [...clockSelect.options].some(x=>Number(x.value)===previous) ? previous : 12; if (!clockSelect.value) clockSelect.selectedIndex=0; refreshGroup(); }}
function refreshGroup() {{ stop(); current = SAMPLES.filter(x => x.task===taskSelect.value && x.clock===Number(clockSelect.value)).sort((a,b)=>a.raw_step-b.raw_step || a.episode.localeCompare(b.episode)); frame.max=Math.max(current.length-1,0); frame.value=0; render(); }}
function stop() {{ if(timer) clearInterval(timer); timer=null; play.textContent='Play'; }}
function toggle() {{ if(timer) {{ stop(); return; }} play.textContent='Pause'; timer=setInterval(()=>{{ frame.value=(Number(frame.value)+1)%Math.max(current.length,1); render(); }}, Number(speed.value)); }}
function dist(a,b) {{ let s=0; for(let i=0;i<3;i++) s+=Math.hypot(a[i][0]-b[i][0],a[i][1]-b[i][1]); return s/3; }}
function marker(ctx,p,color,kind) {{ ctx.save(); ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=2; const x=p[0],y=p[1]; if(kind==='diamond') {{ ctx.beginPath(); ctx.moveTo(x,y-4);ctx.lineTo(x+4,y);ctx.lineTo(x,y+4);ctx.lineTo(x-4,y);ctx.closePath();ctx.fill(); }} else if(kind==='x') {{ ctx.beginPath();ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x+4,y-4);ctx.lineTo(x-4,y+4);ctx.stroke(); }} else {{ ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill(); }} ctx.restore(); }}
function anchors(ctx, points, color, kind) {{ ctx.save();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);ctx.lineTo(points[2][0],points[2][1]);ctx.lineTo(points[1][0],points[1][1]);ctx.stroke();ctx.restore(); for(const p of points) marker(ctx,p,color,kind); }}
function panel(canvasId,img,s,mode) {{ const ctx=document.getElementById(canvasId).getContext('2d');ctx.clearRect(0,0,224,224);ctx.drawImage(img,0,0,224,224);anchors(ctx,s.truth,'#16a36f','circle'); if(mode.startsWith('drift')) anchors(ctx,s.drift_nominal,'#e29b27','x'); const pred=mode.endsWith('old')?(mode.startsWith('drift')?s.drift_old:s.clean_old):(mode.startsWith('drift')?s.drift_adapted:s.clean_adapted);anchors(ctx,pred,'#2c6fbb','diamond'); }}
function metric(label,value) {{ return `<div class="metric">${{label}}<b>${{value}}</b></div>`; }}
function render() {{ if(!current.length) return; const i=Math.min(Number(frame.value),current.length-1),s=current[i]; const img=new Image(); img.onload=()=>{{ panel('c0',img,s,'clean-old');panel('c1',img,s,'clean-adapted');panel('c2',img,s,'drift-old');panel('c3',img,s,'drift-adapted'); }}; img.src=s.image; const values=[dist(s.clean_old,s.truth),dist(s.clean_adapted,s.truth),dist(s.drift_old,s.truth),dist(s.drift_adapted,s.truth)]; ['m0','m1','m2','m3'].forEach((id,j)=>document.getElementById(id).textContent=`EPE ${{values[j].toFixed(2)}} px`); const driftBefore=dist(s.drift_nominal,s.truth); document.getElementById('summary').innerHTML=metric('Drift nominal',driftBefore.toFixed(2)+' px')+metric('Old drift corrected',values[2].toFixed(2)+' px')+metric('Adapted drift corrected',values[3].toFixed(2)+' px')+metric('Adapted clean penalty',values[1].toFixed(2)+' px')+metric('View split',s.split); document.getElementById('status').textContent=`${{i+1}}/${{current.length}} | step ${{s.raw_step}} | ${{s.phase}} | ${{s.episode}}`; }}
taskSelect.addEventListener('change',fillClocks);clockSelect.addEventListener('change',refreshGroup);frame.addEventListener('input',render);play.addEventListener('click',toggle);speed.addEventListener('change',()=>{{if(timer){{stop();toggle();}}}});fillClocks();
</script>
</body>
</html>"""


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = _cuda0(args.device)
    torch.cuda.set_device(device)
    data = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rgb": np.load(data / "current_rgb.npy", mmap_mode="r"),
        "truth": np.load(data / "current_q_true.npy", mmap_mode="r"),
        "xyz": np.load(data / "current_keypoints_xyz.npy", mmap_mode="r"),
        "proprio": np.load(data / "current_proprio.npy", mmap_mode="r"),
        "intrinsic": np.load(data / "camera_intrinsic.npy", mmap_mode="r"),
        "extrinsic": np.load(
            data / "camera_extrinsic_cam_from_world.npy", mmap_mode="r"
        ),
    }
    count = int(arrays["rgb"].shape[0])
    with np.load(data / "index.npz", allow_pickle=False) as index:
        metadata = {
            name: np.array(index[name], copy=True)
            for name in (
                "task_name",
                "camera_clock_hour",
                "split",
                "source_episode_id",
                "proposal_raw_step",
                "proposal_phase_name",
            )
        }
    drift_args = SimpleNamespace(
        translation_cm_max=4.0,
        rotation_deg_max=5.0,
        minimum_drift_epe_px=1.0,
        maximum_drift_axis_px=19.0,
    )
    drift_nominal = np.empty_like(np.asarray(arrays["truth"]), dtype=np.float32)
    for split_name, offset in (("train", 2), ("development", 1)):
        indices = np.flatnonzero(metadata["split"].astype(str) == split_name)
        values, _ = _fixed_nominals(
            indices,
            arrays,
            seed=int(args.seed) + offset,
            args=drift_args,
        )
        drift_nominal[indices] = values
    truth = np.asarray(arrays["truth"], dtype=np.float32)
    old_checkpoint = args.old_checkpoint.expanduser().resolve()
    adapted_checkpoint = args.adapted_checkpoint.expanduser().resolve()
    clean_old = _predict(
        old_checkpoint, arrays, truth,
        batch_size=int(args.batch_size), device=device,
    )
    clean_adapted = _predict(
        adapted_checkpoint, arrays, truth,
        batch_size=int(args.batch_size), device=device,
    )
    drift_old = _predict(
        old_checkpoint, arrays, drift_nominal,
        batch_size=int(args.batch_size), device=device,
    )
    drift_adapted = _predict(
        adapted_checkpoint, arrays, drift_nominal,
        batch_size=int(args.batch_size), device=device,
    )

    samples: list[dict[str, Any]] = []
    for row in range(count):
        image_name = f"frame_{row:04d}.jpg"
        image_path = assets / image_name
        if not image_path.exists():
            image = np.asarray(arrays["rgb"][row]).transpose(1, 2, 0)
            Image.fromarray(image, mode="RGB").save(
                image_path, format="JPEG", quality=86, optimize=True
            )
        samples.append(
            {
                "row": row,
                "image": f"assets/{image_name}",
                "task": str(metadata["task_name"][row]),
                "clock": int(metadata["camera_clock_hour"][row]),
                "split": str(metadata["split"][row]),
                "episode": str(metadata["source_episode_id"][row]),
                "raw_step": int(metadata["proposal_raw_step"][row]),
                "phase": str(metadata["proposal_phase_name"][row]),
                "truth": _round_points(truth[row]),
                "drift_nominal": _round_points(drift_nominal[row]),
                "clean_old": _round_points(clean_old[row]),
                "clean_adapted": _round_points(clean_adapted[row]),
                "drift_old": _round_points(drift_old[row]),
                "drift_adapted": _round_points(drift_adapted[row]),
            }
        )
    summary = {
        "protocol": "fcpg_a4_multiview_adaptation_visual_audit_v1",
        "samples": count,
        "old_checkpoint": str(old_checkpoint),
        "adapted_checkpoint": str(adapted_checkpoint),
        "clean_old_epe_px": float(_epe(clean_old, truth).mean()),
        "clean_adapted_epe_px": float(_epe(clean_adapted, truth).mean()),
        "drift_nominal_epe_px": float(_epe(drift_nominal, truth).mean()),
        "drift_old_epe_px": float(_epe(drift_old, truth).mean()),
        "drift_adapted_epe_px": float(_epe(drift_adapted, truth).mean()),
    }
    manifest = {"summary": summary, "samples": samples}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        _html(
            json.dumps(samples, separators=(",", ":"), allow_nan=False),
            json.dumps(summary, separators=(",", ":"), allow_nan=False),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

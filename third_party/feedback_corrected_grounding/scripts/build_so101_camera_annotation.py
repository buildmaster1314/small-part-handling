#!/usr/bin/env python3
"""Build a browser annotation page for real SO-101 camera projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fcpg.so101_kinematics import SO101_ANCHOR_NAMES, SO101MuJoCoKinematics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist-cache", type=Path, required=True)
    parser.add_argument("--column-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=6)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def _load_cache(path: Path) -> dict[str, object]:
    manifest = json.loads((path / "manifest.json").read_text())
    return {
        "manifest": manifest,
        "frames": np.load(path / "frames.npy", mmap_mode="r"),
        "states": np.load(path / "states.npy", mmap_mode="r"),
        "episodes": np.load(path / "episodes.npy", mmap_mode="r"),
        "frame_indices": np.load(path / "frame_indices.npy", mmap_mode="r"),
    }


def _select_rows(episodes: np.ndarray, samples_per_episode: int) -> list[int]:
    if samples_per_episode <= 0:
        raise ValueError("--samples-per-episode must be positive")
    selected: list[int] = []
    for episode in sorted(int(value) for value in np.unique(episodes)):
        rows = np.flatnonzero(episodes == episode)
        quantiles = np.linspace(0.05, 0.95, samples_per_episode)
        offsets = np.unique(np.rint(quantiles * (len(rows) - 1)).astype(np.int64))
        selected.extend(int(rows[offset]) for offset in offsets)
    return selected


def _write_image(path: Path, frame: np.ndarray) -> None:
    Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
        path, format="JPEG", quality=94, optimize=True
    )


def _html(config: dict[str, object]) -> str:
    payload = json.dumps(config, ensure_ascii=True, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 双相机 FCPG 投影标注</title>
<style>
:root{--bg:#eef2f0;--ink:#17211d;--muted:#65716b;--line:#cbd5cf;--accent:#087e8b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:18px 24px}
h1{margin:0 0 5px;font-size:22px}p{margin:4px 0;color:var(--muted)}
main{max-width:1180px;margin:auto;padding:18px 22px 40px}
.bar{display:flex;gap:10px;align-items:center;background:#fff;border:1px solid var(--line);padding:10px;flex-wrap:wrap}
button{border:1px solid #87958c;background:#fff;padding:8px 12px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
select{padding:7px;border:1px solid #87958c;background:#fff}.status{margin-left:auto;color:var(--muted)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.card{background:#fff;border:1px solid var(--line);padding:10px}.card h2{font-size:15px;margin:0 0 8px}
.canvas-wrap{position:relative;width:100%;background:#202522}canvas{display:block;width:100%;height:auto;cursor:crosshair}
.legend{display:flex;gap:14px;margin-top:8px;color:var(--muted);font-size:13px;flex-wrap:wrap}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px;vertical-align:-1px}
.instructions{background:#fff;border:1px solid var(--line);padding:12px;margin-top:14px}
code{font-family:ui-monospace,monospace}.warn{color:#8b302c;font-weight:650}
@media(max-width:800px){.grid{grid-template-columns:1fr}.status{margin-left:0;width:100%}}
</style></head><body>
<header><h1>SO-101 双相机 FCPG 投影标注</h1>
<p>只标注当前 RGB 中三个机器人锚点：中心、固定指尖、活动指尖。不会连接机械臂，也不会修改 LeRobot。</p>
<p class="warn">请按每个视角的顺序点击：蓝色中心 → 黄色固定指尖 → 粉色活动指尖。坐标按 224×224 缓存图记录。</p></header>
<main><section class="bar">
<button id="prev">上一帧</button><button id="next">下一帧</button>
<button id="undoWrist">撤销 wrist</button><button id="undoColumn">撤销 column</button>
<button id="clear">清空当前帧</button><button id="clearAll">清空全部</button>
<label>帧 <select id="sample"></select></label>
<button id="export" class="primary">导出 annotations.json</button><span id="status" class="status"></span>
</section>
<section class="grid"><article class="card"><h2>Wrist camera</h2><div class="canvas-wrap"><canvas id="wrist"></canvas></div></article>
<article class="card"><h2>Column camera</h2><div class="canvas-wrap"><canvas id="column"></canvas></div></article></section>
<section class="legend"><span><i class="dot" style="background:#00a5d8"></i>中心</span><span><i class="dot" style="background:#e4a11b"></i>固定指尖</span><span><i class="dot" style="background:#dc4f9b"></i>活动指尖</span></section>
<section class="instructions"><b>当前帧信息</b><div id="info"></div>
<p>推荐标注 24 帧。每个 episode 的多个阶段用于拟合，程序会按 episode 留出帧检查误差。若某个点被机械臂遮挡，先跳过该帧，不能凭猜测点击。</p>
<p>点击导出后，把下载的 <code>annotations.json</code> 放到此 HTML 所在目录，下一步即可拟合两路投影并生成 FCPG 输入。</p></section></main>
<script>
const CONFIG=__CONFIG__;
const state=JSON.parse(localStorage.getItem('so101-camera-annotations-v1')||'{}');
let index=0;const colors=['#00a5d8','#e4a11b','#dc4f9b'];
function key(view){return `${CONFIG.samples[index].id}:${view}`;}
function points(view){return state[key(view)]||[];}
function save(){localStorage.setItem('so101-camera-annotations-v1',JSON.stringify(state));}
function setupCanvas(id,view){const canvas=document.getElementById(id),sample=CONFIG.samples[index];
const image=new Image();image.onload=()=>{canvas.width=CONFIG.width;canvas.height=CONFIG.height;const ctx=canvas.getContext('2d');
ctx.drawImage(image,0,0,canvas.width,canvas.height);for(const [i,p] of points(view).entries()){ctx.fillStyle=colors[i];ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#17211d';ctx.font='bold 12px sans-serif';ctx.fillText(String(i+1),p[0]+7,p[1]-7);}};
image.src=sample.images[view];canvas.onclick=(event)=>{const rect=canvas.getBoundingClientRect();const x=(event.clientX-rect.left)*canvas.width/rect.width;const y=(event.clientY-rect.top)*canvas.height/rect.height;const old=points(view);if(old.length>=3)return;state[key(view)]=[...old,[Number(x.toFixed(3)),Number(y.toFixed(3))]];save();draw();};}
function draw(){const s=CONFIG.samples[index];setupCanvas('wrist','wrist');setupCanvas('column','column');document.getElementById('sample').value=String(index);document.getElementById('status').textContent=`${index+1} / ${CONFIG.samples.length} | 已标注 ${Object.keys(state).length} 组视角`;document.getElementById('info').textContent=`episode ${s.episode} | 原始帧 ${s.frame_index} | 行 ${s.row} | wrist ${points('wrist').length}/3 | column ${points('column').length}/3`;}
function move(delta){index=(index+delta+CONFIG.samples.length)%CONFIG.samples.length;draw();}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('sample').onchange=e=>{index=Number(e.target.value);draw();};
function undo(view){const old=points(view);if(old.length){state[key(view)]=old.slice(0,-1);if(!state[key(view)].length)delete state[key(view)];save();draw();}}
document.getElementById('undoWrist').onclick=()=>undo('wrist');document.getElementById('undoColumn').onclick=()=>undo('column');
document.getElementById('clear').onclick=()=>{delete state[key('wrist')];delete state[key('column')];save();draw();};
document.getElementById('clearAll').onclick=()=>{if(confirm('清空全部标注？')){for(const k of Object.keys(state))delete state[k];save();draw();}};
document.getElementById('export').onclick=()=>{const annotations={version:'so101-camera-annotations-v1',width:CONFIG.width,height:CONFIG.height,samples:CONFIG.samples,points:state};const blob=new Blob([JSON.stringify(annotations,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
for(const s of CONFIG.samples){const o=document.createElement('option');o.value=String(CONFIG.samples.indexOf(s));o.textContent=`${s.id} | ep ${s.episode} | frame ${s.frame_index}`;document.getElementById('sample').appendChild(o);}draw();
</script></body></html>'''
    return template.replace("__CONFIG__", payload)


def main() -> None:
    args = parse_args()
    wrist = _load_cache(args.wrist_cache.resolve())
    column = _load_cache(args.column_cache.resolve())
    wrist_manifest = wrist["manifest"]
    column_manifest = column["manifest"]
    if wrist_manifest["dataset_id"] != column_manifest["dataset_id"]:
        raise ValueError("wrist and column caches must belong to the same dataset")
    for key in ("episodes", "frame_indices", "states"):
        if not np.array_equal(wrist[key], column[key]):
            raise ValueError(f"paired caches disagree on {key}")
    if wrist["frames"].shape != column["frames"].shape:
        raise ValueError("paired camera frames have different shapes")

    rows = _select_rows(np.asarray(wrist["episodes"]), args.samples_per_episode)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("no annotation rows selected")
    kinematics = SO101MuJoCoKinematics(args.model)
    states = np.asarray(wrist["states"])[rows].astype(np.float64)
    anchors_base, anchors_gripper = kinematics.forward_anchor_frames(states)
    output = args.output.resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for sample_index, row in enumerate(rows):
        episode = int(wrist["episodes"][row])
        frame_index = int(wrist["frame_indices"][row])
        sample_id = f"sample_{sample_index:03d}"
        wrist_name = f"{sample_id}_wrist.jpg"
        column_name = f"{sample_id}_column.jpg"
        _write_image(images / wrist_name, wrist["frames"][row])
        _write_image(images / column_name, column["frames"][row])
        samples.append(
            {
                "id": sample_id,
                "row": int(row),
                "episode": episode,
                "frame_index": frame_index,
                "state": states[sample_index].tolist(),
                "anchors_base_xyz": anchors_base[sample_index].tolist(),
                "anchors_gripper_xyz": anchors_gripper[sample_index].tolist(),
                "images": {"wrist": f"images/{wrist_name}", "column": f"images/{column_name}"},
            }
        )
    config = {
        "version": "so101-camera-annotation-config-v1",
        "dataset_id": str(wrist_manifest["dataset_id"]),
        "width": int(wrist["frames"].shape[2]),
        "height": int(wrist["frames"].shape[1]),
        "anchor_names": list(SO101_ANCHOR_NAMES),
        "wrist_cache": str(args.wrist_cache.resolve()),
        "column_cache": str(args.column_cache.resolve()),
        "model": str(args.model.resolve()),
        "samples": samples,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "index.html").write_text(_html(config), encoding="utf-8")
    print(f"annotation_html={output / 'index.html'}")
    print(f"samples={len(samples)}")


if __name__ == "__main__":
    main()

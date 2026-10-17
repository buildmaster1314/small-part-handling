#!/usr/bin/env python3
"""Build one inspectable HTML page for a matched Towel grounding comparison."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


MODES = ("nominal", "fcpg", "oracle")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for mode in MODES:
        parser.add_argument(f"--{mode}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_result(path: Path, expected_mode: str) -> tuple[dict[str, Any], Path]:
    source = path.expanduser().resolve()
    metrics_path = source / "metrics.json" if source.is_dir() else source
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    task = payload.get("tasks", {}).get("towel-wipe-v0")
    if not isinstance(task, dict):
        raise ValueError(f"{metrics_path} has no towel-wipe-v0 result")
    metadata = payload.get("metadata", {})
    mode = str(metadata.get("grounding_mode", task.get("grounding_mode", "")))
    if mode != expected_mode:
        raise ValueError(f"{metrics_path} grounding {mode!r} != {expected_mode!r}")
    return {"root": payload, "task": task, "metadata": metadata}, metrics_path.parent


def _matched_identity(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = ("environment_seed", "raw_frame_count", "stable_success_step", "task_instance")
    reference = {key: results["nominal"]["task"].get(key) for key in keys}
    for mode in MODES[1:]:
        candidate = {key: results[mode]["task"].get(key) for key in keys}
        if candidate != reference:
            raise ValueError(f"{mode} physical replay differs from nominal")
    checkpoints = {str(results[mode]["root"].get("checkpoint")) for mode in MODES}
    if len(checkpoints) != 1:
        raise ValueError("comparison inputs use different LeWM checkpoints")
    return {**reference, "checkpoint": checkpoints.pop()}


def _render(
    results: dict[str, dict[str, Any]],
    roots: dict[str, Path],
    output_dir: Path,
    identity: dict[str, Any],
) -> str:
    rows = []
    sections = []
    for mode in MODES:
        task = results[mode]["task"]
        horizon = task["horizon_metrics"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{task['current_grounding_anchor_mean_epe_px']:.4f}</td>"
            f"<td>{horizon['H1']['anchor_mean_epe_px']:.4f}</td>"
            f"<td>{horizon['H4']['anchor_mean_epe_px']:.4f}</td>"
            f"<td>{horizon['H8']['anchor_mean_epe_px']:.4f}</td>"
            f"<td>{task['stitched_anchor_mean_epe_px']:.4f}</td>"
            "</tr>"
        )
        video = os.path.relpath(
            roots[mode] / str(task["video"]), start=output_dir
        )
        sections.append(
            f"<section><h2>{html.escape(mode.title())}</h2>"
            f"<p>Current grounding EPE {task['current_grounding_anchor_mean_epe_px']:.4f} px; "
            f"stitched forecast EPE {task['stitched_anchor_mean_epe_px']:.4f} px.</p>"
            f"<video controls muted loop playsinline preload='metadata'><source src='{html.escape(video, quote=True)}' type='video/mp4'></video></section>"
        )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Towel grounding comparison</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#edf1f3;color:#17212a;font-family:Arial,sans-serif}}header{{background:#17252e;color:white;padding:24px max(20px,calc((100vw - 960px)/2))}}main{{max-width:960px;margin:auto;padding:22px 20px 60px}}.note{{background:white;border-left:5px solid #2976a8;padding:16px;line-height:1.55}}table{{width:100%;border-collapse:collapse;background:white;margin:18px 0}}th,td{{padding:10px;border-bottom:1px solid #dde3e6;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{padding:24px 0;border-top:1px solid #c8d0d5}}section p{{color:#52616b}}video{{display:block;width:100%;height:auto;background:#20272c;border:1px solid #bbc5cb}}</style></head><body><header><h1>Towel Wipe: Matched Grounding Diagnostic</h1><p>One physical replay and one frozen LeWM checkpoint; only the current grounding source changes.</p></header><main><div class='note'><b>Interpretation:</b> nominal and FCPG are deployable inputs; oracle uses current-frame q_true only as an offline ceiling. All three receive the same current RGB, proprioception, task identity, and recorded H1-H8 action prefix. Future RGB and future q_true are never model inputs.</div><table><thead><tr><th>Grounding</th><th>Current EPE</th><th>H1</th><th>H4</th><th>H8</th><th>Stitched</th></tr></thead><tbody>{''.join(rows)}</tbody></table>{''.join(sections)}<p>Environment seed {identity['environment_seed']}; raw frames {identity['raw_frame_count']}; stable success step {identity['stable_success_step']}.</p></main></body></html>"""


def main() -> int:
    args = _parser().parse_args()
    results: dict[str, dict[str, Any]] = {}
    roots: dict[str, Path] = {}
    for mode in MODES:
        results[mode], roots[mode] = _read_result(getattr(args, mode), mode)
    identity = _matched_identity(results)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    document = _render(results, roots, output, identity)
    (output / "index.html").write_text(document, encoding="utf-8")
    summary = {
        "protocol": "lewm_towel_matched_grounding_comparison_v1",
        "identity": identity,
        "modes": {
            mode: {
                "current_grounding_anchor_mean_epe_px": results[mode]["task"][
                    "current_grounding_anchor_mean_epe_px"
                ],
                "stitched_anchor_mean_epe_px": results[mode]["task"][
                    "stitched_anchor_mean_epe_px"
                ],
                "horizon_metrics": results[mode]["task"]["horizon_metrics"],
            }
            for mode in MODES
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

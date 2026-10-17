# FCPG Architecture Panorama

> Recovered investigation, 2026-07-26. Branch `research/feedback-corrected-grounding`
> @ `d571ca2` + uncommitted working tree. Scale: 207 modules in `fcpg/`, 281 scripts,
> 315 test files / ~2520 test functions.

## 0. Three sub-projects in one repo

| Layer | Where | What |
|---|---|---|
| A. Core FCPG research | `fcpg/transformer_grounding.py`, `fcpg/modules.py`, `fcpg/geometry.py`, `fcpg/structured_grounding.py` | The paper contribution: per-keypoint 2D residuals on FK-projected anchors, policy features resampled at corrected pixels |
| B. LeWM experiment fleet | 152 `fcpg/lewm_*.py` + ~200 scripts + `config/lewm/` | MetaWorld/towel/pusht research program; mostly No-Go per README |
| C. SO-101 realtime stack | 22 `fcpg/so101_*.py` (10.3k LOC) + `scripts/serve_so101_realtime_fcpg_lewm.py` | Real-hardware closed-loop MPC web service (two cameras) |

`NOTES_innovation_review.md` is the authoritative novelty critique (GeoProp, CtRNet, Kalib,
Deformable DETR) — read it first in any session about paper positioning.

## 1. Module map of `fcpg/`

### 1.1 Core research modules

- **`transformer_grounding.py:243` — `AnchorConditionedResidualTransformer`**: samples a 5×5
  local token grid within 16 px of each FK-projected anchor from a frozen-ViT dense map
  (`sample_local_feature_tokens:54`), 4-layer local + 2-layer cross-keypoint encoders, emits a
  tanh-bounded residual (cap 20 px) + reliability. `q_corrected = q_nominal + delta_px`.
  Also: `ViTDenseEncoder:95`, `LeWMAnchorResidualCorrector:390`,
  `TwoTipStructuredLeWMAnchorCorrector:540` (evaluated, not adopted: EPE 1.9012→1.9782),
  `FCPGViTAdapter:633`, `load_fcpg_vit_adapter:838`. Imported by 42 files.
- **`modules.py:275` — `GroundedTokenAdapter`**: the policy-facing bridge; docstring names the
  three configurations (GeoProp-lite / FCPG residual-only / reliability). Hosts
  `ResidualCorrectionHead:46`, `ReliabilityConfidenceHead:88`, differentiable samplers
  (`pixels_to_grid_torch:156`, `sample_feature_map_at_pixels:181`). The only module re-exported
  by `fcpg/__init__.py`.
- **`geometry.py`**: NumPy pinhole `Camera:17`, `project_points:38`, `soft_argmax_patch:127`,
  `perturb_extrinsic:148` (the synthetic-drift label generator).
- **`structured_grounding.py`**: parked Stage-II differentiable SE(3) branch.

### 1.2 Baselines & measurement infrastructure

`absolute_keypoint_baseline.py`, `fk_deform_attn_baseline.py` (GeoProp-equivalent control),
`geometric_baselines.py`, `action_chunk_policy.py`, `visual_trust*.py` (the thrice-No-Go U module),
`camera_pairing.py` + `paired_camera_*.py` (fail-closed cohort pairing — statistical backbone),
`independent_fcpg_measurement.py`, `mujoco_camera*.py`, `metaworld_*.py`.

### 1.3 The `lewm_*` fleet (152 modules)

Per-experiment pattern: `_collection.py` / `_training.py` / `_metrics.py` + matching
`scripts/train_*`/`evaluate_*`/`score_*` + `config/lewm/{train,eval,splits}`. Largest:
`lewm_counterfactual_v2.py` (2639), `lewm_runtime.py` (2250). Cross-cutting:
`lewm_grounded_model.py:974`, `lewm_provenance.py` (SHA sealing), `lewm_dev_gate.py`.

### 1.4 SO-101 realtime stack (22 modules)

- `so101_realtime_planner.py` (2540 LOC) — the brain; `PreparedPlannerConfig:100`,
  `PreparedSO101RealtimePlanner:501`. Imports four `scripts/*` modules (scripts are a de-facto
  library here).
- `so101_realtime_runtime.py` (1023) — `RealtimeMPCController:263` (thread-safe state machine,
  fault latching), `ReplaySO101IO:181`, `PlanRejectedError:165`.
- `so101_fcpg_lewm_bridge.py` (783) — anchor bridge + dual-camera fusion + FCPG-conditioned LeWM.
- `so101_kinematics.py` (573) — `SO101MuJoCoKinematics:131` (FK, `forward_anchors`, bounded IK).
- `so101_ik_candidates.py` (421) — `generate_ik_candidate_prefixes:153` (adaptive arrival),
  `track_anchor_trajectory:347`.
- `so101_goal_directed_candidates.py` (366) — damped-least-squares image-servo variants in the
  joint-limit tangent space; gripper-phase prefixes.
- `so101_realtime_mpc.py` — safety/phase layer: `PhaseTracker:85`, `validate_command_chunk:262`,
  `PlanningRoundGuard:313`.
- `so101_visual_targets.py` (453) — template registration/re-matching for pickup/drop_zone.
- `so101_multiscale_memory.py` (506), `so101_retrieval_mpc.py` (518, demo-bank retrieval +
  `candidate_safety_checks`), `so101_active_gripper.py` (456, pulse-based visual re-anchoring),
  `so101_v9.py` (checkpoint transplant), plus calibration/shadow/memory/synthetic helpers.
- NEW/untracked: `so101_grounded_action_policy.py` — see §4.

## 2. Realtime pipeline data flow

Single process: `scripts/serve_so101_realtime_fcpg_lewm.py` `main()` (~`:1630`) hosts a
`ThreadingHTTPServer` on `127.0.0.1:8765`; UI is a self-contained HTML page (`page()` ~`:1358`).

1. Camera+bus read: `HardwareSO101IO.observe()` (three-attempt USB recovery) or `ReplaySO101IO`.
2. `POST /api/plan` → `RealtimeMPCController.replan()` → `_replan_locked()`.
3. FK projection + live similarity transform (`observe_grounding`, planner `:1068-1080`).
4. **Grounding (FCPG core)**: `observe_grounding()` → `encode_fcpg_view()` (wraps the residual
   transformer); fails closed on `*_anchor_out_of_bounds` / `*_fcpg_outside_training_envelope`.
5. Target selection: `VisualTargetTracker.track()` per role/view; `detect_live_target()`.
6. Pixel → base XYZ: `column_live_pixel_to_base_xyz:317`, `bounded_column_servo_target_xyz:342`
   (40 mm planar hop cap + 50 mm descent radius + standoff).
7. Phase: `PhaseTracker` → APPROACH/GRASP/TRANSPORT/RELEASE (or TRAJECTORY in replay-only
   anchor_trajectory mode).
8. Candidates (concatenated): demo-bank retrieval + image-servo variants + calibrated Cartesian
   IK; gripper phases use gripper prefixes.
9. LeWM forward prediction: `rollout_multiscale_candidates` over the V9 checkpoint.
10. Scoring (planner `:2262`): `cartesian_goal_cost + 0.10*pixel_goal_cost +
    0.05*retrieval_cost + 0.05*model_command_cost` — terminal-anchored (final 3 horizon steps),
    no step-size or progress term.
11. Safety conjunction (planner `:2357`): hard_command & ik_safe & prediction bounds & phase &
    distance & workspace & chunk — argmin over the safe set only.
12. Plan + artifacts (`RoundArtifactWriter`: dual-camera overlays + MuJoCo GIF).
13. `POST /api/execute` (confirmed, round_id): up to 4 chunks × 5 commands; each chunk
    re-observes, re-detects, re-validates, then replans from the settled endpoint.
14. Hardware write: 5 interpolated LeRobot commands + echo assert + settle loop.
15. Only executed chunks advance model history (`observation_clock=task_event`).

Permission tiers: default `hardware-read-only`; `--gripper-calibration-enabled`;
`--execute-enabled --acknowledge-workspace-clear` = armed; `--replay-session` = replay.

## 3. Training / evaluation side

SO-101 training chain: `train_so101_synthetic_fcpg` → `train_so101_fcpg_context_bridge` →
`train_so101_lewm` → `train_so101_state_residual_lewm` → `train_so101_multiscale_memory` →
`build_so101_lewm_v9` (deterministic checkpoint composition).

Storage: `config/` curated (~60 files, names encode seed/budget); `outputs/` gitignored
(103 experiment dirs; SO-101 under `outputs/lerobot/so101/`, ~150 run dirs); `data/` gitignored
caches with `data/lerobot/so101` → symlink to `/home/james/so101-data-disk/...`; `results/`
tracked sealed Gate decisions; `docs/` 10 files (`experiment_plan.md` 168 KB is the substantive
one); `README.md` 501 KB = lab notebook / evidence ledger (key sections: 创新边界 `:371`,
Legacy 基线证据 `:1938`, Stage I 冻结规格 `:2656`, 当前交接状态 `:3387`, Test Plan `:4408`).

## 4. NEW untracked: grounded action policy (incomplete, currently negative)

`fcpg/so101_grounded_action_policy.py` (386 LOC): encoder-decoder transformer over
3 frames × 2 views × 3 anchors of FCPG features + state history + past chunks → 40 causal
action queries → per-step-capped, bounds-clamped absolute commands in 8×5×6 chunks.
Zero-initialized delta head (untrained = hold current state). Contract tests only.

Status: (1) the trainer is missing — the module is a reconstruction of a lost training script's
model class (`rollout_frozen_lewm` is `NotImplementedError`); (2) eval says training made it
worse than the hold-still baseline (hold first5 MAE 2.0222° vs step-900 2.4168°); (3) not wired
into the realtime service, not in README. Companion:
`scripts/evaluate_so101_grounded_action_history.py` + feature cache
`my_first_task_clean_v1_grounded_policy_features_v1` (2183 rows, latent 192).

## 5. Third-party & environments

- `third_party/so_arm100/` — vendored MuJoCo subset; `Simulation/SO101/scene.xml` is
  `DEFAULT_SCENE` for all FK/IK and renderers.
- `third_party/lewm/` — unmodified provenance snapshot of `lucas-maes/le-wm@8edfeb33`.
- External via PYTHONPATH: `/home/james/lewm` (+ its `.venv`: torch 2.5.1+cu121, timm),
  `/home/james/lerobot/src` (bus driver, cameras, draccus), so101-data-disk symlink.
- **Dual-environment contract** (README `:282`): the realtime service needs conda `lerobot`
  Python + LeWM venv site-packages on PYTHONPATH simultaneously; either alone fails.
  Offline tests/training: `env PYTHONPATH=.:/home/james/lewm /home/james/lewm/.venv/bin/python3`.

## 6. Test suite

315 flat files in `tests/`, no conftest/markers; 1:1 naming with modules/scripts.
`*lewm*` 217 files (~1841 tests), `*so101*` 35 files (~263), `*policy*` 22, `*fcpg*` 15.
Biggest SO-101 files: `test_so101_realtime_runtime.py`, `test_serve_so101_realtime_fcpg_lewm.py`
(parses the embedded JS), `test_so101_realtime_planner_calibration.py`. README Test Plan
(`:4408`) pins a 22-file focused list + full `pytest -q` + `py_compile` + `git diff --check`.

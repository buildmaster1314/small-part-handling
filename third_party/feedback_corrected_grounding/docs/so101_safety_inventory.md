# SO-101 Realtime Execution Chain — Safety Mechanism Inventory

> Recovered investigation, 2026-07-26 (working tree vs `d571ca2`). Line numbers refer to
> the working tree **at survey time, before the same-day safety fixes**.
>
> **Post-survey status update (2026-07-26 afternoon fixes):** G4 (TRAJECTORY fake-pass →
> honest skip + hardware rejection + serve-arg gate), G5 (`max_plan_age_s=0` defaults →
> 15 s runtime / 120 s serve, 0 forbidden when armed), G8 (`outward_tolerance` default →
> 0.0, 0.1 opt-in only at the runtime live re-check), G9 (`minimum-expected-motion`
> serve default → 0.5), and G6's receding-horizon OR-rescue (deleted) are FIXED in the
> working tree. G1 (no HTTP auth/CSRF), G2 (no collision model), G3, G7, G10-G18 remain
> open.

Chain overview: `HTTP handler` → `RealtimeMPCController` (runtime) →
`PreparedSO101RealtimePlanner.plan()` (planning) → candidate generators
(IK / image-servo / gripper / retrieval) → `so101_realtime_mpc` pure-contract
validators → `HardwareSO101IO` (LeRobot bus).

---

## Layer 1 — HTTP API (`scripts/serve_so101_realtime_fcpg_lewm.py`)

| # | Mechanism | Location | Protects against | Trigger → action | Weakened? |
|---|---|---|---|---|---|
| A1 | Workspace-clear acknowledgement gate | `:446-451` | Arming motion with people/objects in the cell | motion flags without `--acknowledge-workspace-clear` → `parser.error` | No |
| A2 | Replay/hardware permission exclusion | `:452-455` | Replay session granted motion rights | `--replay-session` + motion flag → `parser.error` | No |
| A3 | Default disarmed | `:417-418`, `:1676-1678` | Accidental armed start | without flags: `hardware-read-only` | No |
| A4 | Command-guard margin | `:398`, `:444-445`, `:1632`, `:1645-1646`, `:1683` | Planner emitting commands at the driver limit | planner+runtime get `max_command_jump − 0.5` (7.5) vs driver 8.0 | No |
| A5 | CLI threshold validation | `:422-445` | NaN/zero/negative thresholds; CUDA1 ban | `parser.error` | was partially weakened (max_plan_age_s) — fixed |
| A6 | Required-input existence check | `:456-466` | Missing checkpoints/calibration | `parser.error` | No |
| A7 | Loopback bind default | `:368` | Remote access to an armed robot | default only; `0.0.0.0` accepted silently | only network control |
| A8 | Body size + type limit | `:1555-1564` | Memory exhaustion / non-object payloads | >4096 B or non-dict → HTTP 400 | No |
| A9 | Explicit `confirmed:true` on every motion endpoint | `:1571-1584` | Unintended motion from a stray POST | `RuntimeError` | No |
| A10 | `round_id` binding on execute | `:1573` → runtime `:868`, mpc `:343-344` | Executing a plan the operator never saw | mismatch → stale-round rejection | chunk 1 only (G3) |
| A11 | Hard chunk cap at API | `:1575` (`max_chunks=4`) | Unbounded run from one confirmation | ≤ 20 commands per confirmation | No |
| A12 | Asset path-traversal guard | `:1538-1543` | Reading outside output root | 404 | No |
| A13 | Error taxonomy / status codes | `:1452-1468`, `:1604-1622` | Silent failures | full status + `request_error_code` | No |
| A14 | Client-side confirm dialogs + button disabling | `:1394-1410` | Mis-clicks | advisory; server authoritative | Advisory |
| A15 | Single-flight client guard | `:1405` | Double-submit | client-side; backstop B3 | Advisory |

**No authentication, session token, or CSRF/Origin check in `do_POST` (`:1566-1622`).** See G1.

---

## Layer 2 — Execution runtime (`fcpg/so101_realtime_runtime.py`)

### Arming / disarming

| # | Mechanism | Location | Detail |
|---|---|---|---|
| B1 | `execute_enabled` (static permission) | set `:304`; enforced `:631-632`, `:849-850` | Immutable per process; `calibration_enabled` separate (`:305-309`, `:517`) |
| B2 | `execution_armed` (one-shot arming) | True at `:364/:448/:501/:576/:718`; False at `:908` (success) and `:913` (rejection); enforced `:739-742` | After any execute batch the operator must replan |
| B3 | Operation mutex | `:343-352` | Concurrent planning/execution → HTTP 409 `operation_busy` |
| B4 | Operator stop / E-stop | `:603-613`; polled in IO `:812-813/:835-836` | Bypasses mutex deliberately; latched, no re-arm path |
| B5 | Fault latch | `_latch_fault` `:590-601` | Any unexpected exception → terminal stop until restart |
| B6 | `PlanRejectedError` vs fault distinction | `:165-167` + raise sites | Pre-motion rejections disarm but don't latch; post-motion anomalies latch |

### Per-execution gates (`_execute_one_locked`, `execute`)

| # | Mechanism | Location | Trigger → action |
|---|---|---|---|
| B7 | Human confirmation | `:847-848` | `confirmed != True` → RuntimeError |
| B8 | `max_chunks` range check | `:851-856`, cap `:266` | outside 1..4 → ValueError |
| B9 | Controller-stopped check | `:738` etc. | RuntimeError |
| B10 | Plan-executable gate | `:743-746`, `:875-883` | non-executable → rejection / loop break |
| B11 | Input-exhaustion gate (replay) | `:730-734` | rejection |
| B12 | Live-target re-detection before motion | `:750-763` | fresh-frame re-detect + kind mismatch → rejection. TRAJECTORY: now rejected on hardware, honest skip in replay (fixed) |
| B13 | `plan.guard.validate_live` | `:764-775` | round/age/state-drift/target-drift → rejection (see C9) |
| B14 | Live `validate_command_chunk` re-check | `:776-787` | re-validated against live state at 7.5; `outward_tolerance=0.1` opt-in here only |
| B15 | Command-echo integrity check | `:795-796` | driver clipping → fault latch |
| B16 | Motion-stall / no-follow watchdog | `:797-809` | expected ≥ min-expected and observed < 0.1 → fault latch (serve default restored to 0.5) |
| B17 | Model-clock discipline | `:811-814` | only executed-command observations advance history |
| B18 | Post-execution progress commit | `:815-818` | continuation desync → fault latch |
| B19 | Receipt trail | `:819-836` | every chunk logged (now incl. `target_check`) |

### Other runtime contracts

B20 observation contract (`:32-43`); B21 chunk contract (`:53-63`); B22 history completeness (`:367-375`);
B23 causal-memory offsets (`:386-398`); B24 return-home guards (`:628-681`, 10 Hz, `home_max_step=3.0`);
B25 grounding-calibration preconditions (`:515-546`); B26 target-reference validation on frozen plan frame (`:466-488`);
B27 monitor/model clock separation (`:615-626`); B28 `max_plan_age_s` validation (`:299-300`).

---

## Layer 3 — MPC validation contracts (`fcpg/so101_realtime_mpc.py`)

| # | Mechanism | Location | Detail |
|---|---|---|---|
| C1 | `PhaseThresholds` validation | `:40-55` | nonsense thresholds → ValueError |
| C2 | `PhaseEvidence` validation | `:70-82` | NaN evidence → ValueError |
| C3 | Phase-transition hysteresis | `:99-112`, confirmations=2 | anti single-frame glitch |
| C4 | Terminal phase latch | `:114-116`, `:158-161` | DONE/FAULT absorbing |
| C5 | `candidate_phase_mask` | `:192-221` | gripper-phase compatibility per phase |
| C6 | `phase_prediction_bounds_mask` | `:224-251` | LeWM forecasts outside limits masked; APPROACH checks 5/6 dims (G12) |
| C7 | `validate_command_chunk` | `:262-309` | hard bounds / first jump / step jump / outward-at-limit (strict 0.0 default, fixed) |
| C8 | `PlanningRoundGuard` construction | `:320-330` | garbage guard → ValueError |
| C9 | `PlanningRoundGuard.validate_live` | `:332+` | round/age/state-drift 2.0/target-drift 15 px; `live_target_px=None` = recorded skip (fixed) |

---

## Layer 4 — Planning (`fcpg/so101_realtime_planner.py`)

Construction: D1 config validation (`:133-159`, CUDA1 ban); D2 checkpoint contracts (`:533-598`);
D3 paired-camera validation; D4 demo-bank integrity (`:646-681`); D5 gripper observability (`:718-724`);
D6 workspace AABB = training FK ± 0.04 m (`:712-717`) — the **only** spatial model (G2);
D7/D8 memory contracts.

Grounding gates: D9 shape contract; D10 anchor out-of-bounds → blocked (`:1120-1130`);
D11 FCPG correction envelope 18 px (`:1131-1133`); D12 posterior commit gating; D13 memory fallback;
D14 live-column-calibration requirement (`:1778-1779`, unreachable in anchor_trajectory — G7);
D15 offset envelope 35 px; D16 similarity-transform bounds; D17 posterior reset.

Target/geometry gates: D18 target-visibility fail-closed (`:1914-1918`); D19 backprojection failure blocks;
D20 bounded column servo 0.040 m (`:342-407`); D21 descent standoff 0.050 m; D22 phase DONE/FAULT block;
D23 target-XYZ availability; D24 `detect_live_target` fail-closed (`:1685-1706`).

Candidate filtering: D25 `candidate_safety_checks` (7.5° first/step, bounds, prediction margin 3.0);
D26 workspace containment predicted+commanded (`:2319-2327`); D27 per-candidate chunk validation;
D28-D30 phase masks + object-phase distance 0.08; D31 `candidate_ik_safe` — IK family strictly
`converged` (OR-rescue deleted in fix); non-IK families hardcoded True by convention (G6 residual);
D32 conjunctive AND; D33 argmin over safe set only, else blocked; D34 `executable` composition;
D35 per-candidate reason reporting.

TRAJECTORY path: D36 pre-filter; D37 demonstration-continuation lock; D38 anchor-IK convergence
hard gate; D39 command re-validation; D40 commanded-workspace containment; D41 no-safe block.
(Whole path now replay-only on hardware.)

---

## Layer 5 — Candidate generation

`so101_ik_candidates.py`: E1 spec validation; E2 input validation; E3 arrival range check;
E4 honest `converged` reporting; E5 posture-regularizer fallback; E6 gripper freeze
(`terminal_state[5]=current[5]`); E7 near-limit joint freezing in anchor IK (`limit_guard=3.0`);
E8 hard solve bounds; E10 ⚠ realtime path specs target the object surface (clearance 0) by design.

`so101_kinematics.py`: E11-E14 IK input validation, bounds clipping every Newton step,
step-size limiting, best-iterate retention with honest convergence.

`so101_goal_directed_candidates.py`: E15 input validation; E16 outward-at-limit zeroing;
E17 per-step cap ≤ 0.95°; E18 hard clip to bounds; E19 Jacobian support check; E20 arm-only servo
(joints 0-3); E21 gripper-phase prefix clamping (step ≤ 1.5); E22 grasp/release targets never at
absolute limits.

`so101_visual_targets.py`: E23-E30 role/view legality, registration bounds, fail-closed
unconfigured target, score/margin confidence gate (0.56 / 0.015 — G18), continuity-constrained
local search, `update=False` during execution.

---

## Layer 6 — Hardware driver (`HardwareSO101IO`, serve `:533-1029`)

F1 calibration presence+match; F2 connect cleanup; F3 gripper calibration sanity;
F4 per-command hard bounds — raises, never clamps (`:764-767`); F5 per-command jump ≤ 8.0
(`:768-773`); F6 shape/finiteness; F7 post-write echo re-check; F8 stop-event polling per command;
F9 settle watchdog (3.0 s / tol 3.0 / step 0.25 / 3 samples); F10 camera read timeout 1 s;
F11 bounded camera recovery (3 attempts); F12 stale-handle purge; F13 camera lock;
F14 bus retries (4); F15 gripper absolute range; F16 gripper pulse step ≤ 1.5;
F17 arm-drift abort during gripper calibration (1.5); F18 settle+frame cap;
F19 close-open-close ordering; F20 localization arm-motion guard; F21 calibration artifacts;
F22 persisted calibration contract (atomic write); F23 torque retained on disconnect;
F24 capture cadence validation.

---

## Gaps and single points of failure

- **G1 — No auth/CSRF on the HTTP API (highest structural gap).** Only control is the loopback
  bind default. `/api/home` and `/api/calibrate` need only `{"confirmed": true}` — any local
  browser page can POST them cross-origin. `--host 0.0.0.0` leaves no backup at all.
- **G2 — No collision or self-collision checking anywhere.** Only spatial guard is the training-FK
  AABB; objects, table, and the robot's own body are invisible to the planner.
- **G3 — `round_id` real only for chunk 1 of 4**; chunks 2-4 self-validate (one confirmation
  authorizes 4 replan-execute cycles).
- **G4 — FIXED**: TRAJECTORY fake-pass → hardware rejection + honest recorded skip.
- **G5 — FIXED**: plan-age expiry restored (15 s runtime / 120 s serve; 0 forbidden when armed).
- **G6 — partially FIXED**: receding-horizon OR-rescue deleted (IK family strictly `converged`);
  non-IK families remain hardcoded True by HEAD convention (bounded at generation instead).
- **G7 — `require_live_column_calibration` unreachable in anchor_trajectory mode** (`:1735-1743`
  returns early). Moot on hardware now that the mode is replay-only, still true in replay.
- **G8 — FIXED**: outward tolerance strict 0.0 default; 0.1 (one encoder count) opt-in at the
  runtime live gate only.
- **G9 — FIXED**: `minimum-expected-motion` serve default back to 0.5.
- **G10 — LeRobot `max_relative_target` clamp configured but inert** (server writes straight to
  the bus); the wire-level jump limit is a single `if` (`:768-773`).
- **G11 — `_write_gripper_position` has no relative-jump check**; only caller-side step limiting.
- **G12 — APPROACH prediction bounds skip the gripper dim** (documented).
- **G13 — APPROACH gripper mask is a loose direction test** (delta ≥ −0.5).
- **G14 — `PhaseTracker` never regresses**; dropped object mid-transport leaves wrong phase until
  return-home.
- **G15 — Echo/stall/progress checks fire post-motion only.**
- **G16 — `stop()` freezes in place**; no safe-pose retract, no re-arm API (restart required).
- **G17 — `close()` bypasses the operation mutex** (currently only reachable from `main()` finally).
- **G18 — Detection thresholds permissive** (0.56 / 0.015) and continuity path admits zero-margin
  drifting locks; backup is C9's 15 px drift check.

## What a single confirmed `/api/execute` traverses

confirmed (B7) → execute_enabled (B1) → max_chunks ≤ 4 (B8) → mutex (B3) → stopped (B9) →
per chunk ×4: can_execute (B11) → armed (B2, chunk 1) → executable (B10) → fresh observe (B20) →
live target re-detect (B12; TRAJECTORY: hardware-rejected / replay honest skip) →
validate_live round/age/state/target (B13/C9) → validate_command_chunk vs live (B14/C7) →
per command ×5: stop_event (F8) → hard bounds (F4) → jump ≤ 8.0 (F5) → sync_write → echo (F7) →
settle (F9) → echo vs requested (B15) → stall watchdog (B16) → progress commit (B18) →
replan (D9-D35) → disarm (B2).

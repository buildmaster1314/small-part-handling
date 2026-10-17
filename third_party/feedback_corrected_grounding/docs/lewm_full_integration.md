# Full LeWM integration for FCPG

## Status: Archived No-Go

**ARCHIVED 2026-07-13. DO NOT RUN THE COMMANDS IN THIS DOCUMENT AS THE CURRENT PLAN.** The matched full-grounded-LeWM Gate L1 completed two `10,000`-step runs and the full 16-cell matrix, then failed its preregistered gate: primary improvement was `6.61%` rather than the required `10%`, and `reach-v3` did not beat hold-last. L2 counterfactual ranking and L3 CEM were therefore canceled. This file preserves the historical design only; the active execution contract is the Pixel-Innovation G0 section in `README.md` and its canonical configs.

The upstream repository is vendored in full under `third_party/lewm/` at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The snapshot contains the upstream
training and evaluation entry points, Hydra configs, assets, JEPA wrapper,
transformer modules, utilities, README, and MIT license. It excludes only the
upstream `.git` metadata. FCPG-specific code lives outside the snapshot.

## Two different LeWM paths

The repository now contains two deliberately separate experiments:

| Path | Training objective | Test-time behavior | Current evidence |
| --- | --- | --- | --- |
| `lewm-predictive` action policy | behavior cloning plus an optional one-step next-latent auxiliary | deterministic action chunk; no recursive latent rollout and no MPC | existing negative/ablation results remain valid only for this path |
| full grounded LeWM | next-embedding MSE plus SIGReg; no behavior-cloning loss | recursive latent rollout; CEM was conditional on L1 | two matched 10k runs and 16-cell L1 completed; No-Go, so CEM never ran |

The two paths use different objectives and inference procedures, so the action-policy ablation did not decide the full-model gate. That full-model gate was later run independently and failed its frozen L1 threshold; under the preregistered protocol it never became eligible for paired CEM rollouts or a paper contribution.

## FCPG adaptation

`fcpg/lewm_grounded_model.py` retains the essential upstream LeWM structure:

- `LeWMViTDenseEncoder` wraps the upstream ViT-Tiny/14 image encoder. It exposes
  both the CLS feature and the `16 x 16` patch grid needed for geometric sampling.
- `GroundedFrameEncoder` combines the global visual token, robot proprioception,
  and visual features sampled at FCPG-corrected keypoints. Nominal projection,
  corrected residual, metric 3D keypoint, identity, and validity remain explicit.
- Deployment validity depends only on nominal geometric validity and whether the
  corrected point is inside the image; the model does not consume `q_true`.
- The archived offline sequence dataset required `q_true/valid_true` for
  diagnostics. The planned online wrapper was canceled with this No-Go branch;
  it is not part of the current PixelObserver-G0 execution path.
- The projected frame latent uses the upstream `MLP` with `BatchNorm1d`, followed
  by the upstream action `Embedder` with smoothing width `10`, causal
  `ARPredictor`, prediction projector, and `SIGReg` implementation.
- Proprioception and raw action blocks use train-split-only z-score statistics
  stored in the checkpoint and reused by prediction and planning.
- `GroundedLeWM.forward` trains the faithful `H+1` teacher-forced objective.
  There is no behavior-cloning loss in this path.
- `rollout_embeddings` recursively feeds predicted embeddings back into the
  history window. `fcpg/lewm_planning.py` scores those rollouts against a visual
  goal and implements an environment-independent CEM core over action sequences.

The corrected grounding cache is an input, not a label for future latent
prediction. Each task uses the existing `q_corrected.npy` produced by the
four-task shared seed-0 corrector. The raw RGB frame is still encoded by the
trainable upstream visual backbone.

## Strict sequence contract

`fcpg/lewm_sequence_data.py` enforces episode-local, pre-action-aligned windows.
The configured contract is:

- history `H=3` and one training prediction, so each sample has four state frames;
- `frame_skip=5`, giving frame IDs such as `[0, 5, 10, 15]`;
- every transition concatenates five 4D MetaWorld actions into one 20D block;
- raw span `1 + (H + rollout_steps - 1) * frame_skip = 16`;
- no terminal padding, cross-episode window, non-contiguous episode, or source
  boundary is accepted;
- formal `fcpg` mode requires one correction cache for every source, while
  explicit `nominal` mode rejects correction-cache arguments;
- a correction cache must match the dataset path, sample count, keypoint count,
  shape and floating dtype; its corrector split unit, seed, fraction and source
  contract must also match the formal LeWM split;
- the formal 80/20 episode split uses seed `0`, exactly matching the shared
  corrector instead of silently introducing a world-model-only split.

The real four-task build passed with each 300-episode source split into `240/60`
train/validation episodes, producing `20,400/5,100` windows per source and
`81,600/20,400` windows overall. Train-only normalization has dimensions `4`
for proprioception and `20` for action blocks and is reused on validation.

## Archived experiment contract and artifacts

The complete four-task training specification is recorded in
`config/lewm/train/fcpg_metaworld.yaml`. It uses the real 300-episode pick,
button, reach, and push datasets and their four corresponding
`shared_4task_seed0_sig0_*` corrected caches. The upstream defaults are retained
where applicable: ViT-Tiny/14, latent width 192, history 3, predictor depth 6,
MLP/BatchNorm projectors, batch 128, action smoothing width 10, and SIGReg weight
0.09. The 80/20 split deliberately matches the shared corrector rather than the
upstream 90/10 split. A matched `SIGReg=0` control was included because high
latent rank alone is not evidence of useful action-conditioned dynamics; both
matched formal runs have completed and the resulting Gate L1 was No-Go.

`config/lewm/eval/fcpg_metaworld.yaml` records the historical online protocol
that was conditional on Gate L1. Recursive rollout and the adapted CEM core remain
as archival utilities; the MetaWorld wrapper, camera-curve runner, and CEM
ablations were canceled after the No-Go and are not pending work.

The executable entry points are `scripts/train_lewm_grounded.py` and
`scripts/evaluate_lewm_grounded.py`; both `--help` interfaces are verified. Their
output directories must remain separate from `outputs/action_chunk/lewm_stage1/`
so a full-world-model checkpoint cannot be confused with a `lewm-predictive` BC
checkpoint.

Historical reproduction only, not a current execution plan:

```bash
env PYTHONPATH=. /home/james/lewm/.venv/bin/python3 scripts/train_lewm_grounded.py \
  --config config/lewm/train/fcpg_metaworld.yaml --device cuda:0

env PYTHONPATH=. /home/james/lewm/.venv/bin/python3 scripts/train_lewm_grounded.py \
  --config config/lewm/train/fcpg_metaworld.yaml --device cuda:0 \
  --output-dir outputs/lewm_grounded/fcpg_metaworld_sig0 \
  loss.sigreg_weight=0
```

The following historical per-source diagnostic is also reproduction-only:

```bash
env PYTHONPATH=. /home/james/lewm/.venv/bin/python3 scripts/train_lewm_grounded.py \
  --config config/lewm/train/fcpg_metaworld.yaml --device cuda:0 \
  --output-dir outputs/lewm_grounded/fcpg_metaworld_sig009_persource \
  loss.per_source_sigreg=true
```

The historical runtime accepted `--resume PATH` and rejected incompatible
configuration, path, split, normalization, and upstream contracts. These notes
describe archived artifacts and do not authorize a new run.

The archived offline diagnostic command was:

```bash
env PYTHONPATH=. /home/james/lewm/.venv/bin/python3 scripts/evaluate_lewm_grounded.py \
  --checkpoint outputs/lewm_grounded/fcpg_metaworld_sig009/lewm_grounded_best.pt \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --correction-cache data/grounding_cache/lewm_stage1/shared_4task_seed0_sig0_pick \
  --data data/metaworld_projection/scripted_button_actualcam_se3_clean25_300eps_224.npz \
  --correction-cache data/grounding_cache/lewm_stage1/shared_4task_seed0_sig0_button \
  --data data/metaworld_projection/scripted_reach_actualcam_se3_clean25_300eps_224.npz \
  --correction-cache data/grounding_cache/lewm_stage1/shared_4task_seed0_sig0_reach \
  --data data/metaworld_projection/scripted_push_actualcam_se3_clean25_300eps_224.npz \
  --correction-cache data/grounding_cache/lewm_stage1/shared_4task_seed0_sig0_push \
  --rollout-steps 5 --split validation --grounding-mode checkpoint \
  --output outputs/lewm_grounded/fcpg_metaworld_sig009/diagnostics_val_h5.json
```

For a nominal-grounding diagnostic, use `--grounding-mode nominal` and omit all
`--correction-cache` arguments. The JSON report hashes the checkpoint,
`q_corrected.npy`, and cache `metadata.json` files.

## Verification boundary

At the time this archived branch was closed, its focused suite reported `24 passed`. An exact raw
ViT smoke check produced a feature map of shape `(1, 192, 16, 16)` and a global
feature of shape `(1, 192)`. The formal four-task build also verified the
`240/60` episode split, `20,400/5,100` windows per task, cache split provenance,
deployable validity, and persisted train-only normalization. These checks verify
interfaces, shapes, provenance, strict sequence construction, checkpoint/resume,
offline diagnostics, recursive rollout plumbing, and adapted CEM mechanics. They
do **not** verify optimization stability, prediction quality, online planning
quality, or downstream task success.

The archived reporting contract required the full config, upstream commit,
dataset/split manifest, checkpoint/cache hashes, seed, one-step and recursive
errors, copy and cross-episode action-shuffle controls, and paired camera-curve
rollouts. Gate L1 failed before this route became a paper contribution.

The vendored upstream `train.py` is an archival fidelity copy. With the pinned
`stable-pretraining==0.1.7`, its historical `Resize` call is API-incompatible.
This does not affect the FCPG-owned trainer and evaluator above, which are the
verified execution path for this integration.


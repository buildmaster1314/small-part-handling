# SO-101 public-data expansion

## Storage

The SO-101 data tree lives on the dedicated ext4 volume mounted at:

```text
/home/james/so101-data-disk/lerobot/so101
```

The repository path remains stable through a symlink:

```text
data/lerobot/so101 -> /home/james/so101-data-disk/lerobot/so101
```

The volume is `/dev/nvme0n1p3` (`so101-data`, about 213 GiB usable) and is
mounted by UUID through `/etc/fstab`. Training and preparation commands should
continue to use repository-relative paths.

## Selected sources

All revisions and licenses below are pinned before preprocessing.

| Local source | Hugging Face source | Revision | License | Episodes | Frames | Tasks | FPS | Selected camera |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `so101_multitask_7tasks_700epi` | `Cache-SCA/SO101-teleop_multitask_7tasks_700epi` | `0709747704b8faece1b1ba66e12e8901b3e965f9` | Apache-2.0 | 700 | 346,181 | 7 | 30 | `observation.images.top` |
| `so101_multi_task_v1` | `abdul004/so101_multi_task_v1` | `1574910d5ef258d5a585a8a3eb654d59b3c0595a` | Apache-2.0 | 125 | 50,875 | 11 | 30 | `observation.images.wrist` |
| `so101_declutter_v1` | `ShubhamK32/so101_declutter_v1` | `3e19cc6cd9f69576798956c778a30ae191f6c307` | Apache-2.0 | 51 | 22,899 | 1 | 30 | `observation.images.topview` |

The `so101_multi_task_v1` camera names are counterintuitive: sampled frames
show that `observation.images.wrist` is the fixed external view, while
`observation.images.overhead` is mounted close to the gripper. Keep the chosen
camera fixed unless a new visual audit proves otherwise. Exposure varies
substantially across its video shards, so source-balanced training should retain
brightness augmentation or an explicit quality audit.

All three selected sources contain real-robot video and are stored upright. In
the 700-episode source, however, the global top view can show two SO-101 arms
moving while the metadata provides only one six-channel state/action stream.
A full task-1 shard audit found that the recorded stream aligns strongly with
the left execution arm (state-delta/optical-flow Spearman about `0.58`, versus
about `0.05` for the right-side arm), and that action changes lead state changes
by roughly 3--4 frames. This is sufficient for an expanded-data development
pilot, but all seven tasks still require the same audit before the source can
enter a clean action-conditioned confirmation set.

Episode 308 in the 700-episode source declares 604 tabular samples but its
video segment contains 650 frames. Visual inspection shows that the extra 46
frames are a static post-success tail, not a second action segment. Preparation
therefore follows the episode timestamp metadata, caches the first 604 aligned
frames, skips the 46-frame tail, and resumes episode 309 at its true first
frame. The generated manifest records this as `discarded_video_frames: 46`.

Together these sources add 876 source-scoped episode identifiers, 419,955
frames, and 19 source-local tasks. One 224x224 RGB cache per selected camera
occupies about 63 GB (59 GiB) because the cache stores decoded `uint8` frames
rather than MP4.

## Preparation

```bash
/home/james/lewm/.venv/bin/python3 scripts/prepare_so101_lewm.py \
  --root data/lerobot/so101/raw/so101_multitask_7tasks_700epi \
  --camera observation.images.top \
  --name so101_multitask_7tasks_700epi_top \
  --output data/lerobot/so101/cache/so101_multitask_7tasks_700epi_top

/home/james/lewm/.venv/bin/python3 scripts/prepare_so101_lewm.py \
  --root data/lerobot/so101/raw/so101_multi_task_v1 \
  --camera observation.images.wrist \
  --name so101_multi_task_v1_external \
  --output data/lerobot/so101/cache/so101_multi_task_v1_external

/home/james/lewm/.venv/bin/python3 scripts/prepare_so101_lewm.py \
  --root data/lerobot/so101/raw/so101_declutter_v1 \
  --camera observation.images.topview \
  --name so101_declutter_v1_topview \
  --output data/lerobot/so101/cache/so101_declutter_v1_topview
```

`prepare_so101_lewm.py` reads both LeRobot v2 `tasks.jsonl` and v3
`tasks.parquet`, including the index-encoded task labels in
`so101_multi_task_v1`.

## First expanded-data pilot

The first bounded pilot should add the 700- and 125-episode caches to the three
existing sources, while keeping the 51-episode declutter source sealed for a
later external-source test:

```bash
env PYTHONPATH=.:third_party/lewm CUDA_VISIBLE_DEVICES=0 \
/home/james/lewm/.venv/bin/python3 scripts/train_so101_lewm.py \
  --cache data/lerobot/so101/cache/so101_fruit_front_224 \
  --cache data/lerobot/so101/cache/so101_table_cleanup_front_224 \
  --cache data/lerobot/so101/cache/so101_drawing_front_224 \
  --cache data/lerobot/so101/cache/so101_multitask_7tasks_700epi_top \
  --cache data/lerobot/so101/cache/so101_multi_task_v1_external \
  --output outputs/lerobot/so101/so101_shared_lewm_expanded5_dev_seed5120_2k \
  --steps 2000 --batch-size 128 --workers 8 --device cuda:0 \
  --seed 5120 --split-seed 7301 --val-fraction 0.10 \
  --sampling source-task-natural \
  --action-normalization physical-scale --joint-scale 100 \
  --frameskip 5 --history 3 --motion-threshold 0.01 \
  --eval-every 250 --save-every 1000 --val-batches 32 \
  --exclude-episode so101_drawing:22 \
  --exclude-episode so101_table_cleanup:23 \
  --exclude-episode so101_table_cleanup:24 \
  --exclude-episode so101_table_cleanup:37
```

This is a development pilot, not a confirmatory experiment. The sampling mode
balances sources first and tasks within each source; it does not sample in
proportion to frame count. Start from scratch: the current `--init` path expects
a flat state dictionary, while normal checkpoints store weights below `model`.
Passing `latest.pt` would therefore load no model tensors.

## Expanded five-source development results

The first `2k` base endpoint was under-trained: task-macro prediction/copy MSE
was `0.0124148/0.0114452`, with action-shuffle delta `0.0002457`. A fresh run
under the same contract reached the fixed `10k` endpoint with prediction/copy
MSE `0.0076255/0.0088392`, a `13.73%` relative copy improvement. All `30/30`
source-task groups had positive action-shuffle sensitivity, although only
`13/30` beat copy in visual-latent MSE.

The state-residual experiment freezes that `10k` base (`base_lr_scale=0`) and
trains only a `4,647,174`-parameter temporal residual. It uses a two-layer
causal Transformer (`d=512`, eight heads, FFN `1024`), physical `/100`
state/action scaling, and `SIGReg=0.02`. The fixed `10k` run is reproducible as:

```bash
env PYTHONPATH=.:third_party/lewm CUDA_VISIBLE_DEVICES=0 \
/home/james/lewm/.venv/bin/python3 scripts/train_so101_state_residual_lewm.py \
  --cache data/lerobot/so101/cache/so101_fruit_front_224 \
  --cache data/lerobot/so101/cache/so101_table_cleanup_front_224 \
  --cache data/lerobot/so101/cache/so101_drawing_front_224 \
  --cache data/lerobot/so101/cache/so101_multitask_7tasks_700epi_top \
  --cache data/lerobot/so101/cache/so101_multi_task_v1_external \
  --output outputs/lerobot/so101/so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_10k \
  --base-checkpoint outputs/lerobot/so101/so101_shared_lewm_expanded5_dev_seed5120_10k/latest.pt \
  --steps 10000 --batch-size 128 --workers 8 --device cuda:0 \
  --seed 5120 --split-seed 7301 --val-fraction 0.10 \
  --normalization physical-scale --joint-scale 100 \
  --frameskip 5 --history 3 --motion-threshold 0.01 \
  --base-lr-scale 0 --residual-architecture temporal \
  --temporal-model-dim 512 --temporal-layers 2 --temporal-heads 8 \
  --temporal-ffn-dim 1024 --temporal-max-steps 32 \
  --warmup-steps 200 --eval-every 500 --save-every 1000 \
  --val-batches 32 \
  --exclude-episode so101_drawing:22 \
  --exclude-episode so101_table_cleanup:23 \
  --exclude-episode so101_table_cleanup:24 \
  --exclude-episode so101_table_cleanup:37
```

Full evaluation uses all `48,187` validation sequences rather than the `4,096`
sequence quick-validation subset:

| Metric | Frozen base / simple baseline | Residual `2k` | Residual `10k` |
| --- | ---: | ---: | ---: |
| visual prediction MSE | base `0.0076255` | `0.0074635` | `0.0064999` |
| visual copy MSE | `0.0088392` | `0.0089908` | `0.0088392` |
| six-joint state MSE | command `0.0014361` | `0.0002180` | `0.0001671` |
| six-joint hold-current MSE | `0.0027209` | `0.0027209` | `0.0027209` |
| visual action/state shuffle delta | n/a | `0.0028816/0.0006958` | `0.0079148/0.0041736` |
| state action/input shuffle delta | n/a | `0.0893086/0.0125385` | `0.0801487/0.0161398` |

At `10k`, visual MSE is `14.76%` below the frozen base and `26.46%` below
copy; state MSE is `88.36%` below last-command and `93.86%` below
hold-current. State prediction beats both baselines for `30/30` groups. Visual
prediction beats the frozen base for `28/30` and copy for `20/30`. The new
700-episode source is positive on both visual comparisons for all `7/7` tasks;
fruit is `0/3` and table-cleanup `0/4` against copy, so short/static operations
remain an explicit limitation rather than a universal visual-dynamics claim.

Artifacts are local and git-ignored:

```text
outputs/lerobot/so101/so101_shared_lewm_expanded5_dev_seed5120_10k/evaluation.json
outputs/lerobot/so101/so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_2k/evaluation.json
outputs/lerobot/so101/so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_10k/evaluation.json
```

## Zero-training external declutter result

The frozen `10k` temporal checkpoint was evaluated once on every eligible
sequence from the previously unused `so101_declutter_v1` top-view cache. No
declutter frame was used for training, checkpoint selection, normalization, or
adaptation. The evaluation covers `51` physical episodes and `22,134`
sequences (`18,364` active-motion and `3,770` near-static):

| External metric | Prediction | Baseline | Relative change |
| --- | ---: | ---: | ---: |
| six-joint state | `0.0005074` | last-command `0.0011919` | `57.43%` lower |
| six-joint state | `0.0005074` | hold-current `0.0008964` | `43.39%` lower |
| visual latent | `0.0137118` | frozen base `0.0128738` | `6.51%` worse |
| visual latent | `0.0137118` | copy `0.0023364` | worse |

State action/input shuffle deltas are `0.0803135/0.0115030`, and active-motion
state prediction improves over last-command/hold-current by `61.48%/51.96%`.
The state dynamics therefore transfer to an unseen source, camera and task.
The visual residual does not: its positive action/state sensitivity is not
enough to overcome the external camera and scene shift.

The accompanying episode-0 audit freezes each real observation while H1-H8
checkpoint predictions play in the official SO-101 MuJoCo model. Across `640`
forecast records, EEF RMSE is `11.27 mm` (`9.01 mm` at H1 and `11.48 mm` at
H8). The public camera is uncalibrated, so the two panels are intentionally not
presented as pixel-aligned reconstruction.

```text
outputs/lerobot/so101/external_declutter_temporal5m_seed5120_10k/evaluation.json
outputs/lerobot/so101/external_declutter_temporal5m_seed5120_10k/summary.md
outputs/lerobot/so101/external_declutter_temporal5m_seed5120_10k_mujoco_ep0/index.html
```

This remains external offline prediction evidence. It does not establish
real-robot shadow accuracy, policy control improvement, or FCPG performance,
and the 700-episode two-arm action-alignment caveat remains in force.

## Scientific boundary

- These datasets are suitable for source-balanced LeWM pretraining. A true
  held-out-task evaluation requires a separate task-level split; the current
  trainer only performs within-task episode splits. More overlapping windows
  do not increase the number of independent episodes.
- Keep split assignment at the `(dataset_id, episode)` level. Do not put
  different frames or cameras from one physical episode in train and test.
- The selected sources use six SO-101 motor-position channels near the
  normalized `[-100, 100]` convention. Do not raw-mix MuJoCo radian datasets
  with them under one global normalization. Even among these real sources,
  joint and gripper ranges differ; prefer physical `/100` scaling plus source
  balancing, and audit servo calibration before making cross-source dynamics
  claims.
- Public videos do not provide the camera intrinsics/extrinsics and
  anchor-pixel ground truth required to supervise FCPG. They can strengthen
  LeWM representation/dynamics experiments, but cannot by themselves support
  a real-robot FCPG calibration-drift claim.
- The 700-episode repository card calls the data teleoperation data but does
  not fully document collection provenance. Inspect representative videos and
  treat embodiment provenance as an audit item before publication.

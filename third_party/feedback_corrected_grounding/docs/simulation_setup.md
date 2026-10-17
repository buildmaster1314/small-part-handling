# 仿真环境与基线对齐

本文档只服务第一版实验落地：先用 MetaWorld 做 FCPG 的最小机制验证，再决定是否扩到 RLBench / RoboTwin / 真机。

## 结论

第一阶段优先使用 **MetaWorld**。

原因：

- GeoProp 在 MetaWorld 上做了 50 个任务的单任务实验，每个任务使用 25 条 scripted demonstrations，和我们的方向最贴近；
- 当前机器已有可用的 Python 3.10 虚拟环境 `/home/james/lewm/.venv`，其中已经安装 `torch`、`mujoco`、`gymnasium`、`numpy`；
- MetaWorld 的安装和渲染成本低于 RLBench / RoboTwin，适合先验证 `q_nominal -> delta_q -> q_corrected` 是否成立；
- 第一版不需要完整多任务 imitation learning，先收集图像、机器人状态、真实投影点和扰动投影点即可。

## 当前 Python 环境

使用：

```bash
/home/james/lewm/.venv/bin/python3
```

已确认：

```text
Python 3.10.20
numpy 1.26.4
torch 2.5.1+cu121
timm 1.0.27
mujoco 3.3.0
gymnasium 1.2.3
metaworld 3.1.1
```

缺失：

```text
rlbench
robosuite
```

`metaworld==3.1.1` 已经安装。安装时 pip 将 `mujoco` 固定到 `3.3.0`，这和 `dm-control` 的 `mujoco>=3.8.1` 需求存在版本冲突。第一阶段先使用该环境跑 MetaWorld；如果后续需要依赖 `dm-control` 的项目，建议单独建环境。

## GeoProp 实验设置对齐

根据 GeoProp 论文，相关设置如下：

| Benchmark | GeoProp 设置 | FCPG 第一版处理 |
|---|---|---|
| MetaWorld | 50 个任务，每任务 25 条 scripted demos，corner-view RGB | 先选 3-5 个任务做机制验证，再扩到 10+ 任务 |
| RLBench | 6 个任务，每任务 100 demos | 第二阶段再考虑 |
| RoboTwin | 7 个任务，每任务 50 demos | 第二阶段再考虑 |

GeoProp 主要 baseline：

- `No-Proprio`：只用视觉，不输入机器人本体状态；
- `Vanilla`：视觉特征和本体向量直接拼接；
- `Heatmap`：把本体投影位置做成视觉热力图提示；
- `RC-PE`：robotic-centralized positional encoding；
- `GeoProp`：基于机器人本体投影位置采样局部视觉特征，并注入策略。

FCPG 第一版必须对齐的 baseline：

- `State concat`
- `GeoProp-lite single-keypoint`
- `GeoProp-lite multi-keypoint`
- `GeoProp-lite multi-keypoint + projection jitter augmentation`
- `FCPG residual only`
- `FCPG reliability only`
- `FCPG full`
- `Oracle corrected GeoProp`
- `No-image residual predictor`

## 第一版任务选择

第一轮不要直接上 MetaWorld-50。先选 4 个任务覆盖基本交互类型：

```text
reach-v3
pick-place-v3
push-v3
button-press-v3
```

选择原则：

- 关键点和任务成功强相关；
- 渲染图像中能看到夹爪/指尖；
- 不涉及过长时序和复杂接触；
- scripted policy 或随机采样状态能快速收集足够帧。

如果第一轮结果稳定，再扩到 10+ 任务。

## 数据记录格式

每一帧至少保存：

```text
rgb: uint8 image
robot_state: proprioceptive state from env
camera_intrinsic: K
camera_extrinsic_true: T_true
camera_extrinsic_nominal: T_nominal
keypoints_3d: X_i from MuJoCo body/site positions
q_true: Project(K, T_true, X_i)
q_nominal: Project(K, T_nominal, X_i)
visibility: optional, first version can start without it
task_name
episode_id
frame_id
perturbation_id
```

第一版优先使用 visually anchored keypoints：

```text
left_fingertip
right_fingertip
tool_tip or hand/site position if visually meaningful
```

`wrist_center`、`gripper_center` 这类 kinematic-only keypoints 可以先只作为 proprio embedding 或 ablation。

## 训练 / 测试划分

不要只做随机 train/test split。必须显式划分扰动分布。

### 机制验证数据

```text
train: seen perturbation magnitudes and directions
val: same family, unseen episodes
test_seen: same perturbation family, unseen episodes
test_heldout: held-out directions / larger magnitudes / mixed rotation+translation
```

### 推荐扰动

训练扰动：

```text
translation: +/- 0.5 cm, +/- 1 cm
rotation: +/- 1 deg, +/- 2 deg
```

测试扰动：

```text
translation: 0 cm, 0.5 cm, 1 cm, 2 cm, 4 cm
rotation: 0 deg, 1 deg, 2 deg, 5 deg, 10 deg
mixed translation + rotation
held-out translation direction
held-out rotation magnitude
```

关键比较：

```text
Local-image residual predictor > No-image residual predictor
FCPG full > GeoProp-lite + projection jitter augmentation
```

## 已完成的最小验证

当前已验证：

- `metaworld==3.1.1` 可在 `/home/james/lewm/.venv` 中 import；
- `reach-v3`、`pick-place-v3`、`push-v3`、`button-press-v3` 都可以创建、step、render；
- `corner` 相机可以输出 128x128 RGB；
- `leftEndEffector`、`rightEndEffector`、`endEffector` 三个 site 可作为第一版机器人关键点；
- 默认扰动下投影误差 max 小于 10px，`radius=16` 的 local patch 足够覆盖。

Stage 0 统计：

| 任务 | mean px | p90 px | max px |
|---|---:|---:|---:|
| reach-v3 | 1.68 | 3.95 | 5.85 |
| pick-place-v3 | 1.70 | 3.97 | 5.99 |
| push-v3 | 1.69 | 3.97 | 5.86 |
| button-press-v3 | 2.56 | 4.95 | 9.40 |

## 第一版 Go / No-Go

必须先通过无策略机制验证：

```text
Synthetic heatmap: E_after / E_before < 0.2
Rendered image: E_after / E_before <= 0.7
Local-image residual clearly beats no-image residual on held-out perturbations
```

这些门槛已经通过。当前接入的是 deterministic action-chunk Transformer；DDPM 不属于主线，ACT 仅作为后续跨骨架扩展。

## 安装建议

使用现有虚拟环境：

```bash
/home/james/lewm/.venv/bin/python3 -m pip install -r requirements-metaworld.txt
```

如果 PyPI 包不可用，再使用 GitHub 源安装：

```bash
/home/james/lewm/.venv/bin/python3 -m pip install "git+https://github.com/Farama-Foundation/Metaworld.git"
```

安装后运行：

```bash
/home/james/lewm/.venv/bin/python3 scripts/check_env.py
```

如果 `metaworld` 仍然缺失，先不要写 policy 代码。

最小 probe：

```bash
/home/james/lewm/.venv/bin/python3 scripts/metaworld_probe.py --task reach-v3 --camera corner --width 128 --height 128
```

当前已验证 `reach-v3` 能创建、step、render，并能读取以下 MuJoCo site 作为第一版关键点：

```text
leftEndEffector
rightEndEffector
endEffector
```

数据采集 smoke test：

```bash
/home/james/lewm/.venv/bin/python3 scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3,push-v3 \
  --camera corner \
  --width 128 \
  --height 128 \
  --episodes-per-task 1 \
  --steps-per-episode 5 \
  --warmup-steps 1 \
  --perturbations-per-frame 2 \
  --output data/metaworld_projection/smoke_reach_push.npz

/home/james/lewm/.venv/bin/python3 scripts/inspect_projection_dataset.py \
  data/metaworld_projection/smoke_reach_push.npz
```

已验证 smoke 数据：20 个样本，`rgb=(20,128,128,3)`，`obs=(20,39)`，`q_true/q_nominal=(20,3,2)`，`inside_patch_rate=1.0`。

Scripted policy 数据入口：

```bash
/home/james/lewm/.venv/bin/python3 scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3,push-v3 \
  --camera corner \
  --width 128 \
  --height 128 \
  --episodes-per-task 1 \
  --steps-per-episode 8 \
  --warmup-steps 1 \
  --perturbations-per-frame 2 \
  --action-source scripted \
  --output data/metaworld_projection/scripted_reach_preact_smoke.npz
```

已验证 MetaWorld v3 环境中有 50 个 `Sawyer*Policy` scripted policies。当前有效 smoke 数据包含 `action_source=scripted`、`success` 和 `observation_action_alignment=pre_action` 字段。短轨迹 smoke 只检查格式；用于 BC / policy integration 时需要更长 horizon 和更多 episode。旧 `scripted_smoke_*` 数据是采集器修正前的历史 plumbing 数据，不作为策略效果证据。

Stage 2 residual benchmark after fixing the local-image head to preserve patch spatial layout (`CoordConv + flatten`, no global average pooling):

| held-out direction | model | mean ratio over 2 seeds |
| --- | --- | ---: |
| x- | no-image | 1.1023 |
| x- | local-image | 0.7229 |
| y- | no-image | 0.8334 |
| y- | local-image | 0.3188 |

Detailed run table is saved at:

```text
outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv/summary.md
```

Interpretation: on this smoke benchmark, preserving local patch position makes the visual residual predictor consistently better than the no-image baseline. This is still a Stage 2 sanity check, not a final paper result. The earlier AdaptiveAvgPool local-image head was too translation-invariant for offset prediction and failed on `x-`; that failure is now documented as an implementation lesson.



Held-out magnitude result:

| split | held-out | no-image mean ratio | local-image mean ratio |
| --- | --- | ---: | ---: |
| translation_cm | 4 | 0.9976 | 0.4309 |
| yaw_deg | 10 | 0.8380 | 0.2397 |

Detailed table:

```text
outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/summary.md
```

Interpretation: when the largest translation or yaw magnitude is held out from training, the no-image baseline mostly cannot recover the projection residual, while the local-image predictor still reduces E_after/E_before substantially. This is closer to the paper claim than the direction-only split, because it tests extrapolation to stronger perturbations.


Current Stage 2 figure summaries:

| figure | split | no-image ratio | local-image ratio |
| --- | --- | ---: | ---: |
| direction | x- | 1.1023 | 0.7229 |
| direction | y- | 0.8334 | 0.3188 |
| magnitude | translation_cm=4 | 0.9976 | 0.4309 |
| magnitude | yaw_deg=10 | 0.8380 | 0.2397 |

These figures are sanity-check plots for the mechanism stage. The current repository now also includes perturbation severity curves over translation and yaw levels.


Current perturbation severity curve summary, averaged over 2 seeds:

| curve | severity | no-image ratio | local-image ratio |
| --- | ---: | ---: | ---: |
| translation_cm | 0 | 1.0303 | 0.4913 |
| translation_cm | 0.5 | 1.0739 | 0.4967 |
| translation_cm | 1 | 0.8978 | 0.3416 |
| translation_cm | 2 | 0.9459 | 0.2688 |
| translation_cm | 4 | 0.9976 | 0.4309 |
| yaw_deg | 0 | 1.0828 | 0.2429 |
| yaw_deg | 1 | 1.0686 | 0.2992 |
| yaw_deg | 2 | 0.9617 | 0.2942 |
| yaw_deg | 5 | 0.8579 | 0.2316 |
| yaw_deg | 10 | 0.8380 | 0.2397 |

This is the first paper-shaped mechanism plot: local visual residual correction consistently reduces projection error across perturbation severity levels, while no-image residual correction often stays near or above the no-correction line.

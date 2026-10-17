# 面向小型零件搬运的机械臂控制系统设计

当前主视频是 2026-10-17 用已训练的官方 LeWM 权重重新录制的示教参考引导 MPC。它使用双相机图像、关节和上一动作，并依赖一条示教的名义动作与目标图像。这不是脱离示教的自主策略，也不能把单次放入孔里说成模型已经学会搬运。FCPG 残差预测仍未通过规划验收，没有接管这次录像。

## 最新离线训练实验（2026-10-17）

新增 8 条完整教师轨迹和 128 个成组接触干预片段，按新父轨迹 4/2/2 隔离训练、验证和测试。FCPG 残差完成 1200 步多步训练。新测试的接触动作分支中，0.5 秒视觉误差/保持基线为 0.925，1 秒为 1.099；关节误差仍高于关节命令基线。学习控制仍关闭，新权重未接管机械臂，也未替换现有搬运视频。

完整数据、固定权重、同起点动作对照及限制见 [本轮结果报告](outputs/fcpg_contact_round2/结果报告.md) 和 [重现说明](outputs/fcpg_contact_round2/README.md)。新测试仅两个独立父轨迹，不代表可靠成功率。原场景的相机安装、物理参数与 RGB 几何控制器保持不变。

## 当前结果

蓝架到远端橙架距离为 268 mm。默认 1→1、整体架位移动 4 mm/−3 mm、移动 12 mm，以及 1→2 跨孔四个固定场景检查均完成搬运、松爪和退开。每项只运行一次，不能作为可靠成功率估计。12 mm 架位移动下，原样动作回放失败，实时视觉控制通过；小幅移动下二者都通过。

原始夹爪与两个相机安装位置保持不变，试管只受接触与重力作用。没有等式约束、mocap、额外夹持垫或复位后的试管位姿写入。全过程最大倾角满足 0.10 rad 判定，但松爪后约 4.2°，**未满足 2°竖直要求**。摩擦仍未经实物标定，液体仍只是视觉几何。

详细验收、失败记录、训练配置、模型选择、独立测试和限制见 `outputs/fcpg_tube/结果报告.md` 与 `outputs/fcpg_tube/使用说明.md`。旧版说明保存在 `outputs/fcpg_tube/历史README_20261016.md`，旧视频在安装备份目录中。

## 运行

在项目根目录执行，每次使用新的结果目录：

```bash
PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m small_part.evaluate_visual_transfer \
  --output outputs/new_visual_run
```

录像时增加 `--video` 并设置 `FFMPEG_BINARY`。指定橙架孔位使用 `--target-index 2`。默认根据 RGB 检测到的孔位选择最近目标。策略使用当前场景的颜色、已知尺寸和相机标定，尚不是通用识别系统。

`outputs/tube_place.mp4` 是这次 LeWM 参考引导 MPC 的双相机录像，36 秒、640×240；`tube_overhead.mp4`、`tube_wrist.mp4` 是分视角，`tube_place.gif` 是同一段。对应结果在 `outputs/lewm_video_20261017/result.json`，并已复制为 `tube_physical_validation.json` 与 `tube_visual_execution.npz`。被替换的 RGB 几何录像在 `outputs/backup_before_lewm_video_20261017_213537`。权重是 `outputs/improved_tube/training/calibration/calibrated_weights.pt`，参考轨迹是 `outputs/improved_tube/scenarios/00_train.npz`。

## FCPG 训练

采用公开的 `gw769/feedback-corrected-grounding`，固定提交 `70763d0017d8ad3d65e7a46e58bd0614bf848c92`，实际导入原始 `TemporalStateResidualLeWM`。基础 LeWM 来自 `lucas-maes/le-wm` 的 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。具体本地适配、第三方来源和 CPU 训练入口在结果包中，不声称完整复现论文实验。

原有 16 条轨迹加上 48 段短时探索，按父轨迹分为 8/4/4 个训练、验证、测试父组。完成三次各 2000 步的残差训练和一次 2500 步基础预测补训。选定权重在 `outputs/fcpg_tube/selected_fcpg`，基础权重与归一化统计一并保存。

独立测试中的视觉预测和长时关节预测仍未超过简单基线；新视觉搬运轨迹上的预测也明显变差。因此 `deployment_gate.json` 标记学习控制未启用。不能把这次成功搬运归功于 FCPG，也不能把训练损失下降当作规划成功。

## 检查与限制

```bash
MUJOCO_GL=egl .venv-lewm/bin/python -m unittest discover -s tests -v
```

真实机械臂、实物摩擦标定、流体仿真、可靠失败恢复、所有孔位的稳定搬运和学习模型带来的成功率收益仍未完成。原有教师和示教引导 MPC 代码保留作为历史实验，不是当前默认入口。

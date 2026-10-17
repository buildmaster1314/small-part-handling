# 实验计划

## 2026-07-12 P1 状态覆盖

本节仅是 Stage I FCPG P1 的权威状态；下文保留的 seed-0 数字和 reliability 方案属于历史阶段。Anchor-Persistent LeWM 的最新执行状态以 2026-07-16 状态覆盖和 README 为准。

- pick residual-only adapter、FCPG policy 和 matched GeoProp+jitter policy seed `0/1/2` 已完成；
- rotation `0/5/10 deg` 与 translation `0/2/4 cm` fresh curves 已完成；
- frozen ViT prototype refinement、带 `camera_center_v1` 退化拒绝的 online PnP、oracle 同轨迹矩阵已完成；
- reliability gating 为 No-Go，正式主线固定 `rho=1`；
- LeWM 保持独立，不参与 FCPG P1 结论；
- frozen prototype 的 rotation success AUC 与 FCPG 基本持平，下一阶段必须解决局部匹配等价性，不能把当前结果表述成普遍优于传统 refinement。

正式聚合见：

```text
outputs/action_chunk_rollout/p1/pick_seed012_rot_0_5_10_fresh20/seed_aggregate/
outputs/action_chunk_rollout/p1/pick_seed012_trans_0_2_4_fresh20/seed_aggregate/
outputs/action_chunk_rollout/p1/pick_seed012_external_rot_0_5_10_fresh20/seed_aggregate/
outputs/action_chunk_rollout/p1/pick_seed012_external_trans_0_2_4_fresh20/seed_aggregate/
```

## Counterfactual v2：泄漏隔离的 G0 collector（schema-v2 smoke 已验证）

该协议是旧 full grounded LeWM / post-hoc L2 之后的独立 external-development 设计，不改变 Gate L1、anchor-input ablation 或 PixelObserver 的 No-Go 结论。旧 action-ranking pilot 有三项结构性缺陷，因此不能作为控制证据：

- 从测试 episode 的未来成功帧和 `q_true` 构造 hindsight goal，向 scorer 暴露部署时不可得信息；
- 用 latent-to-goal distance 充当“真实排序”，但它不等同于环境真实 return/progress；
- 物理 branch point 太少且 phase 选择不充分，不能支撑任务级结论。

### G0 数据边界

collector 必须 fail closed 地分离：

```text
deployment_inputs = {
  current/history RGB, proprio, FK/keypoint geometry,
  nominal/corrected grounding, candidate actions,
  current-scene ViT-prefix context + task ID, predicted future latent
}

offline_labels = {
  actual return/progress/success, oracle-anchor actual-future visual latent,
  q_true/valid_true, goal world/pixel coordinates
}
```

未来数据可以保存为监督、离线标签和表征上限，但任何 deployable scorer/runtime 输入出现 `q_true`、future success frame、actual return 或 oracle-anchor future latent 都必须拒绝。当前所谓 `goal_visual` 只是 current-RGB frozen-ViT prefix 的固定投影，应称 scene context；它不是经过验证的显式 goal representation。

### 固定采样与物理配对

- 四任务固定为 `pick-place-v3 / button-press-v3 / reach-v3 / push-v3`；
- phase 预先固定在 step `10/25/35`，命名 early/mid/late，但不声称它们对应接触事件，也不得根据未来成功、reward 或 episode 结果移动；
- 完整 development 为每任务 `10` 个 physical episode、每 snapshot `8` 个候选；smoke 保留全部四任务、三 phase 和三 camera view，但缩为每任务 `1` 个 episode、`4` 个候选；
- train/validation 按完整 physical episode 做确定性 `7/3` 拆分，同一 episode 的不同 phase 不得跨集合；
- 候选动作 bank 在所有 camera condition 间逐字节一致；每个候选从同一 physical snapshot 只执行一次并保存 endpoint states；
- `clean / rotation5 / translation4` 只重渲染相同 history/endpoint physical states。真实 return/progress/success 必须跨 view 完全共享且不含 view axis；
- 保存并核验 physical-state、RNG、candidate-action、snapshot、corrector 与 checkpoint hashes；越界 corrected anchor 必须进入与训练一致的 `valid_grounded` mask。

LeWM 特征矩阵固定为：

```text
train grounding: nominal-trained / FCPG-trained
eval grounding:  nominal / FCPG
camera view:     clean / rotation5 / translation4
```

即保存完整 `2 x 2 x 3` 轴，而不是只挑有利 cell。FCPG corrector 与两个 `10k` LeWM checkpoint 全部冻结，G0 不执行任何上游重训。

### Gate 状态与后续门槛

1. **G0 数据与配对审计**：full external-development store 的 schema、部署/标签隔离、同 snapshot/RNG/action、单次物理分支、哈希与原子存储合同均通过；完整上游 bundle 因历史 corrector/action checkpoint 未绑定 NPZ 内容 SHA，仍为 `g0_eligible=false`。
2. **G1 当前表征可读性（Fail）**：flat 与 horizon-structured scorer 都未使 oracle-anchor actual-future visual latent 超过相同输入合同的 DirectRanker；这不是 fully observed simulator control-state oracle，不能据此外推所有 goal-conditioned scorer。
3. **G2 predicted dynamics（Fail）**：flat scorer 只在六候选 broad bank 正向；唯一一次结构救援中，predicted future 在六候选与 strict-near 上均显著弱于 DirectRanker。
4. **G3 FCPG 必要性（Mixed）**：同一 structured predicted scorer 下，六候选 regret 与 strict-near Spearman 正向，另外两个配对指标 CI 跨零，未达到双指标门槛。
5. **G4 当前接口结构救援（No-Go）**：只执行一次预先限定的 horizon-wise shared-cost GRU；Direct/Predicted/Oracle 同构同参数并共用 train-history scaler。结果未救回 G1/G2，不再在当前已暴露 development set 上加宽、加层或扫参。
6. **G5 CEM（取消）**：G1-G4 未全部通过，不启动在线动作搜索。

主指标使用环境真实 return/progress 的 Spearman 与 normalized regret；latent distance 只作诊断。当前实现路径为：

```text
config/lewm/eval/counterfactual_v2_development_seed8192.json
scripts/collect_lewm_counterfactual_v2.py
fcpg/lewm_counterfactual_v2.py
fcpg/lewm_counterfactual_v2_runtime.py
fcpg/lewm_counterfactual_v2_storage.py
```

### 2026-07-15 schema-v2 smoke 结果

已用冻结的 FCPG corrector、action policy、nominal/FCPG `10k` LeWM 和 oracle target encoder 完成四任务 CUDA smoke；没有执行训练或 CEM。原子 store 位于：

```text
outputs/lewm_counterfactual_v2/development_seed8192_smoke
result_sha256 = 17dd5ce9391a55f1c49719c6218a6bdcacdee4114508ab92386c2f2a91500b22
```

结果规模与审计：

- `4` 个 physical episode，`12` 个固定 phase sample，`4` 个匿名置换 candidate slot，共 `48` 次物理 branch；
- `history_rgb` 为 `[12,3,3,224,224,3]`，全部 `108` 帧动态范围为 `255`，三种视角均非黑帧且确实不同；
- endpoint RGB 共 `720` 帧，最小动态范围为 `255`，扰动视角相对 clean 的最小变化像素数为 `28621`，逐帧 SHA-256 已持久化；
- return/progress/success 为 `[12,4,5]`，没有 camera axis；predicted future latent 与 oracle future latent 分别保存在 deployment/offline 分区；
- schema-v2 store 共 `33` 个数组（`14 deployment + 9 offline labels + 10 audit`），保留 array-aligned JSONL 行序，强制 `sample_index == row position` 且所有数组首维等于 index row count；独立磁盘进程重读后重建的 G0 manifest 与内嵌 manifest 完全一致；
- collector 在每次三视角重渲染时核对并保存 history/endpoint physical-state 与 RNG hashes；candidate action bank 跨 view 只有一份，exact branch execution count 为 `48`；
- 每个匿名 candidate slot 的 action、真实 return/progress/success、oracle future latent、endpoint physical/RNG 与 endpoint RGB 共同进入 branch receipt；独立重读从数组重算 receipt，可拒绝任一候选轴错配；
- deployable scorer 只允许接收 role-specific、单一 model/grounding/view 的只读最小视图。G2 的 Direct/Predicted 配对机械固定同一 cell；G3 机械固定 model/view/goal/action，只切换 grounding；oracle probe 明确为 offline-only；
- 四候选在全部 sample 中唯一，最小 pairwise action L2 为 `0.07969`；return/progress/success 分别在 `11/9/5` 个 snapshot 上有排序信号。标签并列只报告 evaluability，不影响数据合同通过；
- artifact 绑定 clean implementation commit `e1c534f6480d5ad4fe23cd38861022e08036da58`（`git_dirty=false`）；`no_training_verified=true` 且所有冻结 module 的采集前后 state hashes 相等；checkpoint 文件 SHA 与实际加载的 corrector 实例均被绑定。

判定边界：`g0_audit.passed=true` 只证明 bounded smoke 的数据合同与实现通过；`g0_eligible=false`。后续 full external-development、flat scorer 与唯一一次 structured scorer 已完成，见下一节；smoke 自身仍不具备正式资格。完整回归为 `886 passed, 3 skipped, 33 warnings`，`py_compile` 与 `git diff --check` 通过。

### 2026-07-15 full development 与当前 CostAdapter 接口判定

full store 包含 `40` 个 physical episode、`120` 个 snapshot 和 `960` 个一次性执行的 candidate branch，result SHA-256 为 `fc1b45ab0d9c0cfe8c3a7b619e9c8a4735a7893fd8887240bf8371a4c9a93390`。内部 train/validation physical episode 无交集。后验目标签名审计对四个旧 NPZ 均得到零精确交集，但只能正式覆盖已绑定这些 NPZ SHA 的两个 LeWM checkpoint；corrector/action 的历史输入 receipt 缺失，因此 `formal_claim_allowed=false` 保持不变。

flat tiny scorer 封存为 `outputs/lewm_counterfactual_v2/scorer_e1_e3_seed8201/`，result SHA `6590a12ec2650f281934b14634ca6f105ee5c0ca56e7082117494737bf217c59`。E1 Oracle-vs-Direct 的六候选 Spearman 为 `-0.1499 [-0.2398,-0.0548]`；E2 Predicted-vs-Direct 在 broad bank 为正，但 strict `expert+3noise` 事后审计不稳定；E3 也未双指标通过。

为排除 flat hidden `4` 的瓶颈，只允许一次 `HorizonStructuredCostAdapter`：共享 `192->16` state projection、action-prefix GRU、五步 shared cost 求和。三路均为 `7,181` 参数，只替换 future state；共享 scaler 仅从共同 train `z_history` 拟合，缩放前后所有共享输入逐字节一致。封存输出为 `outputs/lewm_counterfactual_v2/structured_cost_e1_e3_seed8201/`，result SHA `9a8106d3b60a798b8aa43cfe5978e0ac1ff45cd96f8e382d738131b4b075c1d0`。

结构化 E2 在 drift/FCPG 下明确反向：六候选 Spearman/regret improvement 为 `-0.1017 [-0.1391,-0.0595] / -0.0385 [-0.0669,-0.0051]`；strict-near 为 `-0.0925 [-0.1380,-0.0517] / -0.0824 [-0.1344,-0.0279]`。Predicted 相对 cyclic future 错配仍显著为正，说明训练出的 scorer 使用了 future 输入，但在当前协议下验证表现弱于 DirectRanker，不能写成 future information 本质有害。

该协议的独立验证只有每任务 `3`、总计 `12` 个 physical episode。更关键的是，当前 `goal_visual` 并非显式目标表示；非预注册的 read-only ridge sanity probe 只能把 task-macro goal-pixel RMSE 从 `9.09 px` 降到 `7.46 px`，说明它弱含目标信息，但没有证明足以支撑 return ranking。oracle 也只是由真实未来图像和 `q_true` anchor 构造的 visual latent。候选 bank 还存在 candidate-family shortcut：只用 train episode 拟合的 `task + candidate-kind` 查表在 validation 上得到 broad/strict Spearman `0.6208/0.6882`，structured DirectRanker 为 `0.5792/0.5377`。查表与 Direct 的配对差异 CI 跨零，不能断言 Direct 只靠类别作弊，但当前协议确实不能隔离状态依赖后果推理。

因此只把**当前 v2 接口**判为 No-Go，G5 取消，且不得继续在本 development set 上扫 Adapter；不能据此外推 fully observed control-state oracle、真正的显式 goal-conditioned scorer、平衡候选协议或整个 LeWM/CostAdapter 路线普遍无效。Stage I FCPG 的独立结果不受影响。

未来若重测，必须创建不复用 v2 validation 的独立 v3：以 policy action 为中心构造等范数、不同方向且匿名置乱的候选；固定同一个 explicit goal，按 `StateOracle -> oracle-anchor visual future -> LeWM predicted future -> deployable GoalToken -> FCPG vs nominal` 的顺序逐级判定。scorer 使用 `stopgrad(DirectScore) + FutureCostDelta`，Delta 末层零初始化，并加入 repeated-current/cyclic-future 的参数匹配对照。`StateOracle` 未显著超过 Direct 时，说明候选、goal 或 horizon 尚未形成需要 future dynamics 的问题，后续层级一律不启动。

## Counterfactual v3：一次性 StateOracle 协议修复 Gate（2026-07-15 预注册）

本次 v3 不复用 v2 数据或 scorer，不加载 LeWM，也不重训 FCPG/policy。四任务各使用 `12` 个 fresh physical episode，固定 step `10/25/35`，按 episode SHA 做 `8/4` train/validation 拆分。冻结 policy action chunk 是中心；只在 `25 x 3` motion block 上生成四个正交方向及其 `+/-`，夹爪不变；最多按冻结 seed 尝试 `64` 组 QR basis 并取第一组可行解。经过执行器 clipping 后每个候选的 motion RMS 固定为 `0.15`，最小两两 RMS `>=0.135`、最大两两 cosine `<=0.60`，八个 public slot 匿名置乱。每个主标签候选从同一 physical snapshot 只执行一次；每任务首个 early snapshot 额外做 repeat/reversed-order 恢复审计，不进入主样本。

数据分区强制为：

```text
deployment inputs:
center_action_raw, candidate_action_raw

privileged offline diagnostic / labels:
58D current_state, H5 future_state, goal_world, physical_task_score,
actual_return, progress, success

audit only:
task/episode/env/snapshot/phase/split,
hidden axis/sign/permutation, candidate_delta_rms
```

58 维 state 保留 TCP/夹爪/物体位姿/接触/力/物体初始位置 18 维语义前缀，并追加固定 padding 的 MuJoCo `qpos16/qvel15/ctrl2/mocap_pos3/mocap_quat4`。StateOracle 仅由该 H5 future state 与显式 `goal_world` 计算固定 staged physical trajectory cost，再确定 `physical_task_score`；同一物理轨迹必须同分，不得读取 RGB、视觉 latent、`q_true`、reward/return/progress/success 标签或未来成功帧。Privileged DirectRanker 获得相同完整 current physical state、goal、task ID 与 candidate action，三 seed 各固定 `2,000` steps，因此 P1 比较的是 future physical state 的净增量，而不是缺状态或 privileged-vs-deployable 输入差异。MetaWorld actual return 只用于 P0 并行有效性检查。

固定 Gate：

1. P0 对每个任务都要求至少 `80%` validation snapshot 的 physical-score spread `>=0.1` 且 H5 actual-return spread `>=0.01`；物理 task-cost 相对 uniform control 排序 H5 actual return 的 paired bootstrap 必须通过；train-only `task x anonymous-slot` 与隐藏 `task x axis/sign` lookup 的 Spearman bootstrap CI 必须完整落入 `[-0.1,0.1]`，regret-improvement bootstrap CI 上界 `<=0.02`。失败即停止且禁止 Direct 训练。
2. Direct qualification 要求每 seed 训练 loss 至少下降 `10%`，ensemble train macro Spearman `>=0.2` 且严格超过 uniform/slot/axis-sign train priors；失败只报告 baseline invalid，不做 Oracle 比较。
3. P1 要求 StateOracle 相对合格 Direct 和 canonical-axis cyclic-future control，在 Spearman 与 normalized top1 regret 上的 paired physical-episode bootstrap CI 下界全部大于 `0`；跨两个 baseline 的共同交集中，至少相同 `3/4` 任务在双指标上均为正。
4. P1 通过只允许预注册下一层 VisualOracle；失败只说明当前 candidate/horizon/physical-cost 协议没有暴露 future dynamics 决策价值。不得启动 LeWM、GoalToken、FCPG-vs-nominal 或 CEM。

冻结配置：`config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json`，canonical-config SHA-256 `8b8a63cd2dbb5e04bdc9362233c8d36414cd996338fb5e75bd4436d53aacdc6f`。collector/gate 均要求同一个 clean git commit；输出使用独占原子目录，底层沿用已审计的 v2 atomic-store，但科学 schema 是 v3；任何阈值、数据量、动作半径、Direct 预算或停止顺序都不得在结果产生后修改。

**正式结果（2026-07-15，P0 No-Go）**：唯一一次采集在 commit `65503b8c0e28faba08aabc298be2a0fef93deb2a` 上得到 `144` 个 snapshot、`1,152` 条主物理分支和 `36` 条恢复审计分支（dataset SHA-256 `9165f9cec3ade056f116826da4e185e8163818ddb8ff05650f65cfa41edab887`）。H5 actual-return spread 为 `48/48`，physical-score/actual-return 的 Spearman 与 normalized-regret improvement 分别为 `0.6473 [0.5928,0.7034]` 和 `0.2738 [0.2242,0.3229]`，均通过；但 frozen physical-score spread 仅 `14/48`，四任务全部低于 `80%` 门槛，且 anonymous-slot 与 hidden axis/sign shortcut 审计均失败。P0 作为合取 Gate 因此停止，Direct 未训练，后续 P1/VisualOracle/LeWM/GoalToken/FCPG-vs-nominal/CEM 均未启动。Gate SHA-256 为 `3e21b7a381e8233ec01d5e88a7fa300618388fb999331da4938166143eed76eb`。该结果只否定当前 candidate/horizon/frozen-physical-cost 协议，不能外推为 LeWM、CostAdapter 或 FCPG 的一般性 No-Go；v3 不调门槛、不扩样本、不重跑。

### Counterfactual v3 actual-return post-hoc（2026-07-15，当前接口 No-Go）

正式 P0 结果保持冻结。一次独立标记为 exposed-validation/descriptive-only 的诊断改用真实 H5 terminal return 作为唯一 target，比较 axis/sign prior、DirectRanker 与零初始化的 nested privileged future-state residual；`physical_task_score` 不参与。Direct 在 validation 上为 Spearman `0.6314`、normalized regret `0.1735`，相对 axis/sign 的 `0.0665 / 0.4372` 在四任务均联合正向。

Naive residual 因 Direct 在同一 train 样本上已达到 `0.9948 / 0.000279` 而存在 stacking 饱和混杂。唯一一次修正采用固定 4-fold physical-episode cross-fitting：每任务 `8` 个 train episode 按 SHA 排序 `rank % 4`，每 fold 用 `6` 个训练、`2` 个 OOF；scaler/normalizer 只在 fold-train 拟合，fold score 经 raw-return 坐标转到统一全 train 坐标。全部 `96` 个 train snapshot OOF coverage 恰为 `1`，Residual 只在 OOF Direct base 上训练，validation 叠加到 full Direct。

OOF Direct train 为 `0.4551 / 0.2822`，nested train 被拟合到 `0.9931 / 0.000366`；但 nested validation 退化为 `0.5213 / 0.2176`。相对 full Direct，Spearman 变化 `-0.1101 [-0.1915,-0.0298]`，regret improvement `-0.0441 [-0.1130,0.0309]`，只有 `1/4` 任务双指标联合正向。当前 `FutureStateResidual + full Direct` 接口因此为 descriptive No-Go，不再在该暴露 split 上调整，也不启动 VisualOracle/LeWM/GoalToken/FCPG-vs-nominal/CEM。该结果不能外推为 future state 或世界模型的一般性失败；它只说明当前小样本 nested residual 没有跨 episode 泛化。


## 已完成候选验证：KineParity global-only（No-Go）

P1 完成后，本项目曾预注册并完成一个可证伪候选：在指尖或工具 control anchors 的视觉对应、真实投影和残差标签完全不参与共享更新求解的条件下，全臂 observer points 能否恢复当前可观测的共享相机漂移，并将更新可靠传播到 held-out control anchors。该候选的主战场是 local FCPG 失效的 10 度旋转。Gate A 通过，但 Gate B 的 pilot 与唯一一次 frozen rescue 均未超过 matched direct-control prototype，因此最终判为 No-Go，不再是当前目标。

固定边界：

- observer points 只来自底座、肩、上臂、肘、前臂、腕和夹爪连杆；
- control-anchor 视觉对应、`q_C_true` 和残差标签禁止进入共享更新；已知 FK 三维位置 `X_C` 与名义 Jacobian `J_C` 只允许用于 task-oriented 后验效用和更新传播；
- `q_C_true` 只能用于 held-out 评估和 oracle 上限，不能参与模态选择；
- oracle Gate A 固定三帧 observer history；每帧使用各自 FK/Jacobian，但共享窗口内相机扰动。Gate B 必须复用同一窗口定义；
- 几何模态先按白化奇异值与 control-anchor 后验效用筛选，再以 `alpha=0.01` 的按 effective-rank 卡方 innovation test 决定是否执行共享更新；
- reliability gating 与 LeWM 不进入该阶段。

### Gate A：oracle observer geometry

先使用 observer points 的 oracle 2D correspondences 验证白化 MAP / TSVD，不训练视觉网络、不接 policy。必须报告 held-out control-anchor mean / median / P95 pixel error、`E_global / E_nominal`、白化奇异谱、effective rank、update rejection rate、错误更新尾部和 control-anchor posterior pixel uncertainty。

预注册 Go 条件：

- 10 deg：`E_global / E_nominal <= 0.25`，且 control-anchor P95 error `<= 12 px`；
- 4 cm：`E_global / E_nominal <= 0.30`；
- clean：共享更新引入的 control-anchor mean displacement `<= 0.5 px`；
- 退化构型下，相比无约束 PnP / 6D 更新显著降低 P95 和最大错误更新；
- 数据 manifest 和单元测试能证明 control-anchor 视觉观测与真值标签完全留出。

Gate A 已通过并冻结；上述 oracle/noise stress 样本不得继续用于阈值调参。随后执行的 Gate B 已判为 No-Go，未接入 policy。

2026-07-12 Gate A 完成状态：先以无噪声 oracle pilot 验证 observer-control 公式，再冻结阈值并加入 correspondence noise、dropout、有限历史和病态星座压力测试。最终 full-9 设置为 `H=3`、`2 px` Gaussian noise、`25%` dropout、false-alarm probability `0.01`；每个 severity 使用 `5 episodes x 20 frames x 10 trials`。control-anchor 视觉对应、真实像素和残差标签均未进入求解。

最终结果：clean 下 KineParity mean/P95 为 `0.027/0.000 px`，非零更新窗口约 `1.1%`；MAP-all 为 `1.308/2.682 px`。`5 deg` 为 `20.391 -> 1.137 px`，P95 `2.299 px`；`10 deg` 为 `40.862 -> 1.871 px`，P95 `3.419 px`，ratio `0.0458`；`4 cm` 为 `6.249 -> 1.425 px`，P95 `2.751 px`，ratio `0.2280`。4 cm 上 MAP-all 的 mean/P95 为 `1.328/2.603 px`，略好于 KineParity，因此不声称模态拒绝在每种扰动上都占优。

Severe stress 使用 `4 px` noise、`50%` dropout 和 `10 deg`。Full-9 下 KineParity mean/P95/max 为 `3.184/6.461/42.283 px`，iterative PnP 为 `4.753/8.944/47.628 px`，linear-6D max 为 `994.713 px`。Distal-3 下 KineParity 为 `5.009/11.736/47.675 px` 且没有 `>64 px` catastrophe；linear-6D 的 P95 为 `121.610 px`、catastrophic fraction `17.57%`，PnP 为 `2.67%`。按预注册标准，oracle geometry Gate A 判为 **Go**。这只锁定几何机制，不证明 learned correspondence 或控制收益。

产物：

```text
outputs/kineparity/gate_a/pick_rotation_seed20260712_e5s20.json
outputs/kineparity/gate_a/pick_translation_seed20260712_e5s20.json
outputs/kineparity/gate_a/pick_rotation_chi2_history3_full9_final_seed20260719_e5s20_t10.json
outputs/kineparity/gate_a/pick_translation_chi2_history3_full9_final_seed20260720_e5s20_t10.json
outputs/kineparity/gate_a/pick_rotation10_chi2_history3_severe_all_seed20260721_e5s20_t10.json
```

### Gate B：matched visual correspondences

所有 solver 必须使用同一组 observer correspondences、可见性 mask 和时间窗口，比较 nominal、逐点 FCPG、frozen prototype、全臂 robust PnP、direct regularized 6D regression、KineParity global-only 与 oracle observer KineParity。

Go 条件：10 deg held-out control anchors 上，KineParity global-only 的 mean error 不得比 prototype 高超过 1 px，并应显著降低 P95；在低 effective-rank 构型下，其错误更新率必须低于 robust PnP 和 direct 6D regression。observer matcher 在 no-image / shuffled-image 下必须明显退化，排除记忆扰动分布。

2026-07-12 pilot 判定为 **No-Go**。`0-5 deg` 训练的 learned observer head 在独立 `10 deg` 上把 observer mean 从 `37.91` 降到 `19.40 px`，视觉特征清零/打乱后 held-out control mean 回到 `37.37/38.59 px`；`SIGReg=0.02` 为负消融。matched observer prototype + full-9 KineParity 得到 held-out control `6.61/15.24/24.69 px`（mean/P95/max），但同帧 direct control prototype 为 `5.23/10.92/25.76 px`，超过了预注册 mean margin 且 P95 更差。distal-3 下 KineParity 的长尾显著优于 MAP/6D/PnP，只支持退化机制，不足以通过 full-9 主 gate。

Frozen rescue 已按上述协议完成并判为 **No-Go**。正式 learned observer mean 为 `13.87 px`，full-9 KineParity control mean 为 `13.23 px`；matched observer prototype + KineParity 为 `6.17/15.89 px`（mean/P95），direct control prototype 为 `5.14/10.85 px`。distal-3 的退化鲁棒性仍成立，但 full-9 主 gate 失败，因此 Stage II 终止，不进入 Gate C。

### Gate C：corrected-anchor resampling and control（取消）

Gate B frozen rescue 未通过，本节不执行，仅保留预注册协议供审计。

通过 Gate B 后，在同一冻结 policy checkpoint 中执行：

```text
observer features
-> global SE(3) update
-> q_C_global
-> Sample(F, q_C_global)
-> unchanged policy
```

这是 mandatory corrected-anchor resampling，不重新运行视觉编码器，也不加入局部 residual head。正式 rollout 使用 3 个训练 seed，每个 seed、每个严重度至少 60 条 fresh paired episodes。

控制 Go 条件：

- 10 deg success 相比当前 local FCPG 至少提升 8 pp，分层 paired 95% CI 下界大于 0，目标绝对成功率至少 15%；当前 oracle 约 20% 只作为上限参考；
- 5 deg 相对当前 FCPG 非劣，预注册 margin 为 `-5 pp`；
- clean 相对 nominal / FCPG 非劣，预注册 margin 为 `-3 pp`；
- 4 cm 当前 FCPG 已接近 oracle policy 上限，只要求非劣和更低的尾部 grounding error，不强制成功率继续上涨。

只有 same-checkpoint intervention 先通过，才训练 matched KineParity policy。

### Gate D：optional bounded local refinement

只有 global-only 通过 Gate C 后，才允许测试 `q_C_local = q_C_global + r_local * tanh(local_head(...))`。固定 `r_local <= 8 px`，local head 不得参与共享 `SE(3)` 求解，并必须同时报告 global-only、local-only 和 global+local。若局部输出持续饱和或不能带来跨 seed 稳定控制增益，则删除该组件。

判定原则：KineParity 的主要价值是恢复 10 度大偏移；几何误差下降不等于控制收益；如果 global-only 无法超过 matched prototype / robust PnP 的尾部稳定性，observer-control transfer 不形成独立方法贡献。

## Wide prototype capture-range confirmatory gate（2026-07-15，完成 / No-Go）

本 Gate 与已经终止的 KineParity 无关。它只检验一个冻结的强外部基线：在同一 ViT descriptor bank 上，将 hard top-1 prototype 搜索从 `radius=32, grid=9` 扩到 `radius=64, grid=17`，能否在不损伤 clean / 5 deg 性能的前提下恢复 10 deg 捕获范围。它不使用 observer constellation、共享 `SE(3)` 求解、global propagation、oracle 候选或置信度拒绝，因此通过时也只能作为 capture-range 证据和强 baseline，不能冒充独立方法创新。

暴露的 development set 使用 `3 policy seeds x 20 paired episodes`。wide prototype 相对 native FCPG 的成功率差为：clean `+3.33 pp`、5 deg `+3.33 pp`、10 deg `+16.67 pp`；10 deg 三个 seed 分别为 `+30/+5/+15 pp`，但分层 bootstrap 95% CI 为 `[0,+35 pp]`，未达到严格下界大于零的确认门槛。该数据只授权一次全新确认，不作为最终论文统计。

确认协议在看新结果前固定如下：

- 三个独立训练 pipeline：pick policy/corrector seed `0/1/2`；
- 每个 pipeline 在 clean、固定 5 deg、固定 10 deg 各运行 `60` 个 paired episodes；
- 方法固定为 `nominal / native FCPG / hard top-1 prototype`，prototype 固定 `radius=64, grid=17, temperature=0`，不得按确认结果调搜索范围、grid 或拒绝阈值；
- fresh identity 固定为 task index `50000..50059`、environment seed `120000..120059`、runner seed `20260817`；与 prototype bank 的 source task episodes `0..99` 以及此前 rollout 范围隔离；
- 主统计为 policy seed 与跨 seed 共享 physical episode identity 的 two-way crossed bootstrap：两个因子独立重采样，同一 episode replicate 在所有 seed 上共享；点估计对三个 policy seed 等权，EPE 只作机制诊断。

合取 Go 条件：

1. 10 deg 的 `prototype - native` 成功率至少 `+8 pp`，分层 95% CI 下界严格大于 `0`，且 prototype 绝对成功率至少 `15%`；
2. 5 deg 的差值 95% CI 下界大于 `-5 pp`；
3. clean 的差值 95% CI 下界大于 `-3 pp`；
4. 10 deg 不允许任一独立 policy seed 出现负向差值。

若本 Gate 通过，只能说明 wide frozen matching 是解决大偏移的有效强基线。要把它变成 FCPG 扩展，仍需一个不读取 `q_true` 或已知扰动幅度的部署式选择机制，在新的独立数据上决定何时使用 native residual、何时使用 wide matching。

正式确认已按上述冻结协议完成。`prototype - native` 的 paired success difference 为：clean `-2.22 pp [-10.00,+6.11]`、5 deg `-3.89 pp [-12.22,+3.33]`、10 deg `+8.89 pp [+1.67,+16.67]`；区间均为 `100,000` 次 two-way crossed bootstrap 95% CI。10 deg 三个 seed 的差值分别为 `+8.33/+8.33/+10.00 pp`，方向条件通过，prototype 绝对成功率为 `12.22%`。

因此合取 Gate 判为 **No-Go**：10 deg 的 effect size、CI 与逐 seed 方向通过，但绝对成功率低于 `15%`；clean 与 5 deg 的非劣 CI 下界也分别低于 `-3/-5 pp`。该结果证明扩大 frozen descriptor 搜索范围能够恢复一部分大偏移捕获能力，同时暴露了无条件 hard top-1 wide matching 的正常工况代价。它只保留为强 baseline 和 capture-range 诊断，不进入方法贡献，也不授权在本确认集上事后拟合 native/wide selector。

正式产物：

```text
outputs/action_chunk_rollout/p1/pick_seed012_external_wide_r64g17_rot_0_5_10_confirm60_seed20260817/
outputs/action_chunk_rollout/p1/pick_seed012_external_wide_r64g17_rot_0_5_10_confirm60_seed20260817/paired_seeds_prototype_vs_native/
```

## Joint prototype control-headroom G0（2026-07-15，两任务 post-hoc development / partial）

该诊断只回答：固定 wide prototype 候选集合中，是否存在能让同一 policy 输出更接近 `q_true`-grounded oracle action 的候选接地点。它不训练 selector，也不评估 LeWM、posterior writeback 或闭环控制。

协议在 Pick/Button 上均固定为 `10 deg`、`radius=64 px`、`grid=17`、per-anchor `L=16`、joint `M=16`、`NMS=0`。Pick 为 `30 episodes x 50 frames x 3 policy seeds`，Button 为 `30 episodes x 100 frames x 3 policy seeds`，对应 `4,500/9,000` structural pairs。每个 snapshot 只替换 current grounding；主指标为同 checkpoint 下候选动作相对 `q_true`-grounded teacher action 的 checkpoint-normalized XYZ MSE（`H=8`）。`q_true`、`valid_true` 和 oracle action 仅作 offline teacher/capture labels 与 oracle candidate selection，不进入候选生成、deployable policy batch 或 selector。

| Task | capture coverage (95% CI) | native H8 | wide H8 | best-of-three H8 | geometry oracle H8 | joint oracle H8 | capture oracle H8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pick | 94.93% [93.27%,96.47%] | 0.064144 | 0.018472 | 0.010902 | 0.010282 | **0.003604** | **0.004873** |
| Button | 94.97% [92.50%,96.83%] | 0.004574 | 0.001163 | 0.000389 | 0.000624 | **0.000156** | **0.000179** |

Pick 的 wide/geometry EPE 为 `5.2291/3.6438 px`，joint/capture oracle 相对 best-of-three 降幅分别为 `66.94% [47.53%,79.42%]` 和 `56.22% [31.11%,71.01%]`。Button 的 wide/geometry EPE 为 `5.0591/3.5327 px`，joint oracle 绝对 delta 为 `-0.0002326 [-0.0004123,-0.0000166]`，相对降幅 `59.84% [2.88%,89.13%]`；capture-constrained 绝对 delta 为 `-0.0001655 [-0.0003164,+0.0000030]`，CI 略跨零。因此无约束 oracle action-output headroom 已在两任务同向复现，但 Button 的 `8 px` capture-constrained 复现尚未统计确认。

**Composite G0 = partial / incomplete。** 这仍是离线 oracle 候选上限，不是 deployable selector、LeWM、structured dynamics、posterior writeback、feature re-indexing、rollout success 或控制收益。正式产物：

```text
outputs/prototype_control_candidates/pick_rot10_seed012_r64g17_l16m16_development.json
outputs/prototype_control_candidates/button_rot10_seed012_r64g17_l16m16_development.json
sha256: pick=7e92d079eccc1fb6d0c892b50b570cf8c9e70d907299b1fef4fd9f0f084d844d
        button=a72884f98012e6c0f0c1b07da7d6003d6c576f538739363d03bce10886f57f8c
```

## Dynamic-camera v2 infrastructure（2026-07-15，implementation only）

schema `fcpg_metaworld_dynamic_camera_v2` 支持每 episode 独立采样的扰动 onset、阶跃相机移位、慢速线性漂移、4--6 帧 anchor dropout 和恢复段。Paired clean 图像在同一 pre-action physical state 上重渲染，与动态观测共享物理状态和待执行动作，每个时刻只执行一次物理分支。Collector 同时支持 measurement withholding 和 dynamic/paired-clean 配对的语义局部 visual occlusion，并保留 raw render 用于审计。`q_true` 仅可在离线收集时定位 visual occlusion，不是模型输入；deployment measurement mask 仅依赖 nominal support 和预定 dropout schedule。

collector、配对与 dropout/recovery 合同已完成。Anchor-Persistent LeWM 的独立 2k development 结果见下一节；仍无 selector、posterior writeback、rollout 或 CEM 结果。

## Anchor-Persistent LeWM v1（2026-07-16，2k dynamics Go / recovery mixed）

该候选只修复 legacy grounded LeWM 过早汇总状态的问题：frame encoder 后保留 `[B,T,K,D]` 逐锚点状态，将锚点轴折入 batch，复用官方 LeWorldModel v3 的共享 `ARPredictor`，并对每个锚点独立计算 `SIGReg=0.02`。相机变化仍由当前 FCPG 视觉测量处理；LeWM 不预测外生 camera jump。

固定训练预算为 batch `192`、seed `9216`、`2,000` optimizer steps。冻结 endpoint 为：

```text
outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/lewm_grounded_latest.pt
sha256 = a6c329d88692810e852685be1460569f6da28753b8d672468b20538811b9c092
```

schema-v2 evaluator 使用 `same_task_different_episode_only` action shuffle，并同时报告 copy、same-checkpoint nominal intervention 与逐锚点 spectrum。完整结果：

| Source | factual / copy / strict-shuffle MSE | FCPG intervention relative gain | effective rank / PR |
| --- | ---: | ---: | ---: |
| clean validation | `0.017840 / 0.236096 / 0.247548` | `+33.59%` | `14.62 / 12.42` |
| linear drift `2--4 cm` | `0.015806 / 0.178093 / 0.188799` | `+2.84%` | `12.74 / 10.38` |
| step rotation `5 deg` | `0.129905 / 0.278940 / 0.267915` | `+4.00%` | `10.85 / 7.38` |
| step translation `4 cm` | `0.092363 / 0.271630 / 0.319213` | `+0.55%` | `13.34 / 11.17` |

因此逐锚点、动作敏感 dynamics mechanism 通过，且不是表示塌缩造成的低误差。遮挡恢复另用全新 `seed=9517` 的四任务各 5 episode schema-v2 数据：5 deg camera step 后真实遮挡 anchor 1 共 5 帧，dropout/recovery masks 只用于统计。总体 factual/copy/shuffle 为 `0.189248 / 0.346862 / 0.320823`，但：

- recovery onset：`0.951940 / 1.028692 / 0.943405`；factual 未胜 shuffle；
- full recovery segment：`0.125084 / 0.122900 / 0.171586`；factual 未胜 copy。

旧 recovery 协议没有通过其合取标准，该结果保持冻结；但它不能外推为一般性的 occlusion-recovery mechanism No-Go：onset 恰逢不可预测的外生相机 step，full recovery 又被恢复后的近静态尾段主导，且该切分不是独立确认协议。准确边界为 **dynamics mechanism Go / legacy recovery Gate failed / mechanism evidence mixed-inconclusive / 10k expansion unauthorized**。不得延长训练、扫描 SIGReg、启动 selector/CEM 或将本候选列为论文第二贡献；旧 summary-pooled LeWM、PixelObserver、Counterfactual 和 KineParity 的既有 No-Go 均保持有效。

```text
clean report sha256     = b2b7bc28e5dd98df654f3e418401a0c41a377e5d71d18459f66b0e445bf02a78
dynamic report sha256   = 98b9fe284ea194109067333ce04ddcd4cb057a6ccaf2dcd1efaaccef42a23db0
occlusion report sha256 = f0d16915269385b091753a16e7827802ecf8de4f41da4b05f7f39bd8b31cdf93
occlusion data sha256   = fce447de53f0b7cb1a62df420102cf46898302d87b5a872b2829e233dd76aa2f
```

### Zero-training posterior headroom（2026-07-16，exposed post-hoc development）

冻结 `latest@2000`，用遮挡 `rgb + deployment correction` 构造 measurement，以同物理帧未遮挡 `rgb_camera_only + target-only correction` 构造 canonical target；两条 cache 均显式绑定 image key、selected-image SHA 和 deployment/target-only role。target 分支只用于离线评分和 oracle upper bound，不进入部署输入或递归历史。五个 raw 遮挡帧只支持五条 phase stream 的 masked state 和一次写回后的 next pre-measurement prior，不能命名为五个独立 raw `o0/r0`，也不能检验多步 `o1/o2`。primary 只取直接遮挡 anchor；写回只替换 dropout-active anchor。

直接遮挡 anchor 上，measurement / fixed-alpha-0.75 / oracle-convex 的 masked MSE 为 `1.02369 / 0.95071 / 0.74601`，下一 prior MSE 为 `0.95312 / 0.89616 / 0.72291`。相对 measurement，oracle 改善分别为 `0.27768 [0.22230,0.34269]` 与 `0.23021 [0.17494,0.29565]`；fixed alpha 在下一 prior 上的区间跨零。冻结 LeWM prior 单独使用更差（`1.91891/1.97449`），因为 `100/100` 个 history 均跨越外生 camera step。

判定为：存在 **measurement-dominant adaptive fusion** 的 oracle upper-bound headroom，但统一 fixed alpha 的 next-prior 改善未确认，且 `push-v3` 稳定退化；当前结果不授权论文 claim。唯一一次 fresh Gate 固定为：camera step 先稳定，随后至少 `15` 个 raw-frame 的测量破坏；primary 使用直接遮挡 anchor 的 `o1/o2` post-update posterior，恢复只取有限 `r1` 窗口。Adapter 从 measurement 出发，只允许每 anchor 共享标量 gate 与有界 prior 差分，hidden `<=32`、总参数约 `20k`，冻结 LeWM/encoder/FCPG。部署 gain 不得读取 `q_true`、预设 schedule、task ID 或 annotation validity。正式必要对照为 measurement-only、验证集选择的 fixed gain 与 `no_lewm`：`no_lewm` 只把 LeWM prior 换成递归 hold-last，同时保留真实动作输入；`action_blind` 只屏蔽 Adapter 的直接动作输入，`action_shuffle` 作为诊断。要求相对 measurement 改善 `>=10%`、episode-level CI 下界大于零、胜 fixed gain 与 `no_lewm`、各 task/cohort 退化不超过 `5%`；clean 条件使用“附加 clean MSE / perturbed-primary measurement MSE”，严格小于 `2%`，否则停止。

### Fresh posterior Adapter Gate（2026-07-16，完成 / No-Go）

唯一一次 fresh Gate 已完成。数据为四任务 `30` train episode/task、`10` perturbed-validation episode/task 和 `10` independent-clean episode/task；训练 `full / action_blind / no_lewm` 三种模式、五个 seed、每个 `2,000` steps，冻结 FCPG 与 Anchor-Persistent LeWM。

五个 full endpoint 在直接遮挡 anchor 的 `o1/o2` post-update 上均为强正：measurement MSE `0.89556`，full MSE `0.06063--0.08388`，相对改善 `88.2%--89.8%`。最佳 seed 相对 `no_lewm` 的 MSE 改善约 `81.8%`；action-shuffle 也从 factual `0.06063` 退化到 `0.14285`。这支持 learned transition prior 相对 hold-last 的遮挡期增量，并说明模型对动作错配敏感；A0.5 的容量匹配 no-action 比较未建立稳定优势，因此不支持 factual-action 因果必要性 claim。

合取 Gate 仍为 No-Go，因为五个 learned endpoint 都未通过 safety filter：`r1` 恢复 normalized excess 为 `3.93%--4.30%`，单侧上界 `8.47%--8.88%`；clean 点估计为 `1.59%--1.73%`，但单侧上界 `3.17%--3.30%`。固定 `beta=0.25` 虽通过安全检查，主改善仅 `13.0%`。因此当前 Adapter 的失败点不是遮挡期预测能力，而是视觉恢复后仍残留 prior。停止控制 Gate、参数扫描和同 split 救援；未来若重开，必须使用新的可部署视觉置信度/恢复拒绝接口和全新 episode split。

### VisualTrust 最小 Probe（2026-07-16，完成 / Development No-Go）

冻结 FCPG、ViT、LeWM 与 policy，只训练逐锚点 current-only VisualTrust head。canonical feature 仅提供离线 correctness 标签；模型只读取当前 corrected-location visual feature 与七维可部署几何描述。训练/development/clean 为 `120/40/40` 个互斥物理 episode，覆盖 square/irregular occlusion、texture distractor 与 local blur。三个训练 seed 是优化 seed，不是独立数据 seed。

三组 VisualTrust AUROC 为 `0.9649--0.9668`，unsafe recall 为 `0.9111--0.9123`；同参数 auxiliary-only AUROC 为 `0.4471--0.4506`，说明视觉输入有效。但 causal temporal-distance baseline 为 `0.9451`，相对增量仅 `0.0198--0.0218`，未达到 `+0.03`。独立 clean FRR 为 `9.58%--10.22%`，最差任务 `12.43%--14.77%`；local-blur early-reopen 为 `22.5%--23.75%`。`40/40` recovery event 均先实际 unsafe 且 window 有效，整体 detection `97.5%`、两帧在线确认中位延迟 `1`，但 texture recovery 仅 `88.9%`。

因此 current-only 绝对 VisualTrust 接口判为 Development No-Go，不启动 hard-switch posterior。该结果不否定遮挡期 LeWM prior；后续若重开必须更换信息接口（例如真正局部的匹配响应/margin 或单独预注册的时序一致性），不得在当前 development draw 上调 operating threshold。

### Measurement-vs-Prior recursive oracle（2026-07-16，fixed-schedule Diagnostic Go）

该 Gate 冻结 FCPG、Anchor-Persistent LeWM 与现有 posterior cache，不训练新 head。每种 selector 独立维护递归历史，只允许直接遮挡 anchor 在 measurement 与 prior 之间切换。canonical target 仅用于 privileged 离线决策与评分；deployment runtime 不得读取 canonical、遮挡 schedule 或 `measurement_valid`。

teacher-forced factual-prior 使用全局 direct-anchor、逐事件比值：hard 在 `o1/o2` 改善 `13.29%/10.03%`，continuous convex 改善 `13.67%/10.17%`，池化 hard 保留 continuous 收益 `97.85%`。递归结果则使用 task-macro、physical-episode bootstrap，不应与 teacher-forced 绝对幅度直接比较。递归 privileged hard/convex 的主改善为 `89.87%/91.42%`；always-prior 虽改善 `89.87%`，但 recovery/clean one-sided upper 为 `25.00%/13.16%`，证明必须具备 source rejection。fixed schedule 的 clean upper 仍为 `5.09%`，暴露当前固定时序的 shortcut。

用于判定的受约束 FSM 固定为：prior error 连续两个采样事件严格小于 measurement error 才进入 prior；任一采样事件 prior 不再严格更优（包括平局）就立即退出。当前采样事件相隔 `5` 个 raw frames，因此不能把它解释成逐帧“两帧进入、一帧退出”的检测延迟。其 `o1/o2` task-macro 改善为 `21.38% [17.71%,25.16%]`。recovery 与 clean normalized excess one-sided upper 均为 `0`，但这是 target-aware oracle 在 prior 不优时按构造选择 measurement 的结果，不是部署式恢复检测证据。因此只判 **Diagnostic Go / Formal Gate NotEvaluated**：允许采集 fresh randomized development split，不允许直接训练 comparator、启动控制或形成论文 claim。`oracle_sequence_exact` 枚举五事件窗口全部 `32` 条 hard-switch 序列，并最小化完整五事件流累计误差，只作 hindsight upper bound，不是 `o1/o2` primary 的独立最优解。

该后续阶段已经按要求随机化 corrupted anchor、onset、duration、corruption family、severity、camera trajectory 和 episode-relative onset，并加入 time-only/schedule shortcut；结果见下一节的 fresh randomized No-Go。此前预定的 comparator 形式为：

```text
A_norm = (E_measurement - E_prior) / (E_measurement + E_prior + eps)
```

小 advantage 落入 dead zone 并默认 measurement。部署输入限定为 measurement、prior、innovation、局部视觉证据、因果历史和动作；canonical 只生成离线标签。正式 diagnostic 输出位于 `outputs/lewm_posterior_preference_oracle/current_split_diagnostic_v4/`。

### Fresh randomized preference oracle（2026-07-16，Development No-Go）

该 Gate 已按固定接口完成，不训练任何 comparator。Calibration/evaluation 分别包含 `40/80` 个 camera-view episode，但 step 与 linear-drift 是同一机器人源轨迹的配对视图，因此独立源轨迹数为 `20/40`，即每任务 `5/10` 条。配对视图的 frame、robot observation、action 和三维关键点完全一致；每个 view 独立随机 corrupted anchor、onset、`10/15/20` 帧 duration、corruption family/severity 与 episode-relative onset，不使用语义 task-phase 标签。

time-only control 只在 calibration 上拟合 normalized full-episode time、frame phase 与 anchor identity，但它只在真实 corruption event stream 上被调用，direct anchor 也由离线 dropout support 路由，因此是 privileged event-aligned shortcut，不是可部署 baseline。paired clean 保持相同物理帧和 camera trajectory，仅移除局部 corruption。

主统计排除 `o0`，采用 phase -> event -> camera view -> source trajectory -> task-macro 聚合；paired views 在 calibration 和 evaluation 中都先在源轨迹内等权平均，bootstrap 按每任务 `10` 条独立源轨迹重采样。受约束 cautious-enter/fast-exit FSM 相对 measurement 改善 `8.58% [5.31%,12.02%]`；button/pick/push/reach 分别为 `9.18% / 8.69% / 14.44% / 2.01%`，四任务均正向。Event-aligned time shortcut 改善 `6.67% [2.36%,11.24%]`，FSM 相对其增量为 `+1.91% [-2.85%,6.49%]`，点估计转正但区间跨零。

无约束 privileged 上限仍很强：always-prior、oracle-hard、oracle-convex 的 task-macro active MSE 分别为 `0.2702 / 0.1401 / 0.1200`，measurement 为 `1.1133`。这说明 LeWM prior 与 source selection 存在机制 headroom，但固定的两事件进入 FSM 没有提取出超越时序先验的增量。FSM 在 paired recovery/clean 的 normalized excess one-sided upper 均为 `0`，仅证明 canonical-aware fast-exit 的递归可行性，不证明部署式恢复检测。

合取判定为 **No-Go**：未达到固定的 `10%` primary、未达到相对 time shortcut `+5%`、incremental CI 未排除零，且 push/reach 的 incremental point effect 为负。即使不把 privileged time shortcut 当竞争基线，FSM 仍因绝对改善不足而不能授权 comparator。当前 development split 不调门槛、不改 FSM 后重跑；不授权 normalized-advantage comparator、policy control 或 CEM。该结果不能外推为 Anchor-Persistent LeWM、FCPG 或 source-selection 原理的一般性失败。

正式汇总：

```text
outputs/lewm_posterior_preference_oracle/randomized_development_equal_view_calibration_seed17101/summary.{json,md}
```

### Random-Hazard predictor isolation A0（2026-07-16，Development Go）

先用 privileged `sensor-valid` mask 隔离 predictor 能力，不训练或评估 deployable selector。该 Gate 冻结 `latest@2000` Anchor-Persistent LeWM，复用 fresh randomized evaluation split 的 `40` 条独立源轨迹及其 `80` 个 paired camera views。Factual、hold-last、latent constant velocity、zero-action LeWM 和 cross-source shuffled-action LeWM 各自维护独立递归历史；主统计排除首个 hidden event `o0`，paired views 先在源轨迹内等权，bootstrap 单位为 task 内 source trajectory。

Factual LeWM 的 task-macro MSE 为 `0.270214`。相对 hold-last、latent CV、zero-action 和 shuffled-action 的改善依次为 `54.26% [43.29%,64.57%]`、`85.99% [80.14%,91.27%]`、`47.51% [37.39%,57.29%]` 和 `35.02% [28.54%,44.78%]`，四个比较均为 `4/4` 任务同向。因此历史最小条件判为 **A0 Go**：learned predictor utility 与对动作输入错配的敏感性成立；参数匹配 A0.5 完成后，不再把 frozen zero-action 或 shuffled-action 单独解释为 factual action 的必要性。

解释边界：zero-action 是冻结 factual 模型的输入消融，不是独立训练的参数匹配模型；A0.5 已补上该基线并使动作必要性判为 No-Go。latent CV 仍只是最小缺测传播基线；当前没有 FK/KF、time-only state predictor、未见 corruption、policy rollout 或 clean 非劣结果。后续 A1 只用 fresh same-state 八动作分叉隔离 `T(b,a)` 中 executed-action 的增量，不把 FK/KF、selector 或 policy rollout 混入这一因果 Gate；这些系统基线属于 A1 通过后的完整 interruption benchmark。

正式汇总：

```text
outputs/lewm_posterior_predictor_isolation/randomized_development_seed17101/summary.{json,md}
```

### Random-Hazard matched-action A0.5（2026-07-16，Development No-Go）

为替换 A0 中较弱的 frozen zero-action 输入消融，冻结原 `2k` anchor encoder，并从相同初始化训练 factual 与 no-action 两个同容量 dynamics head。两头共享数据、latent target、dropout RNG 和 optimizer schedule，各自独立裁剪梯度，固定使用 `latest@2000`。训练端点单步 validation MSE 为 factual `0.015800`、no-action `0.054065`，factual 改善 `70.78%`。

在原 A0 的 `40` 条独立源轨迹和 `80` 个 paired views 上做独立递归写回后，factual MSE 为 `0.274172`，matched no-action 为 `0.308184`；按任务内相对改善再对四任务等权得到 `7.54% [-3.88%,18.29%]`，并非直接对两个 task-macro MSE 求比值。pick/push/reach/button 分别为 `30.35% / 2.33% / 5.38% / -7.90%`。因此 `3/4` 方向条件和 `>=5%` 点估计通过，但 CI 下界未大于零，合取 Gate 判为 **No-Go**。同一 factual head 仍显著优于 hold-last `53.77%`、latent CV `85.73%` 和 shuffled-action `35.38%`，均为 `4/4` 任务同向。

该结果将结论收窄为：anchor dynamics predictor 有效，但当前单策略轨迹未证明 factual action 相对独立 no-action recurrence 的跨任务稳定必要性。不得用 frozen zero/shuffled-action 的较大差距替代 matched baseline，也不得在当前 split 继续调参或启动 selector/CEM。若后续重新检验 action necessity，必须使用 fresh physical episodes，并优先增加同状态下的动作多样性或 matched-state action intervention，而不是重复同一行为策略的数据量。

正式汇总：

```text
outputs/lewm_posterior_predictor_isolation/matched_no_action_seed17301/summary.{json,md}
```

### Action-Branch Random-Hazard A1（2026-07-16，Screen No-Go）

A1 用 fresh same-state action intervention 隔离 `T(b,a)` 中的 executed-action 增量。每个 hazard 保存同一物理 state/RNG/history，恢复后执行八条匿名 QR 动作分支，隐藏当前 measurement，并递归预测 `H=8` 个模型步。主误差只保留同一 bundle 内的 branch-relative future-state 变化；factual 和 matched no-action heads 在三个 seed 上同构、同初始化、同优化预算，各训练 `2,000` steps。冻结 encoder 的 `SIGReg=0.02` 保持不变，新 heads 不添加 SIGReg。

正式 store 为 `450` bundles，train/development/test 为 `218/113/119`。Development 的 `48` 条独立 physical episodes 上，factual 相对 no-action 与七种 wrong-action mean 分别改善 `55.75% / 73.74%`，`H6:H8` 仍改善 `52.72% / 70.37%`，三个训练 seed 均正向，raw whitened MSE 也通过 sanity check。

合取 Screen 仍为 **No-Go**。informative branch support 为 pick `30/30`、reach `28/28`、push `26/26`，但 Button 只有 `10/29=34.48%`，低于冻结的每任务 `80%`；Button 全量 factual-vs-no-action 为 `-0.73%`。十个 informative Button bundles 上的事后 `+8.63%` 只作诊断，不改变 Gate。结果只支持“充分 intervention support 上存在强 action-alignment predictive increment”，不支持四任务 action necessity、selector、policy 或完整 belief update。按协议不启动 fresh `10k`，不打开 independent test。

Button-only exposed-split 物理重放进一步定位了失败来源。该 disposable diagnostic 不训练模型，只复现原 `train+development` 的 `84` 个 Button rows，在同一 snapshot 上执行保存的八条动作，并记录 TCP、`qpos`、按钮状态、success 与原 latent dispersion。pre-contact/contact/post-success 分别有 `25/6/53` bundles；latent branch variance 中位数为 `0.022873/0.010889/0.000146`，支持率为 `96%/50%/0%`；TCP H8 最大 pairwise spread 中位数同步从 `5.51 cm` 降为 `2.01 cm/0.40 cm`。latent 与 TCP/qpos branch variance 的 Spearman 均约 `0.94`。因此当前没有“物理分叉明显但 encoder 不分叉”的证据，主要问题是随机 hazard 大量采到首次 success 后的近吸收态。

该诊断不改变 A1 No-Go，也不授权 test 或 `10k`。若建立独立 A2，必须在实验前固定以下二选一：一是任务在首次 success 时终止、只在 active-control interval 采样；二是把 Button 换成 drawer/door/随机目标运输等持续动力学任务。A2 必须使用全新 episode split，并重新验证同状态 branch support，不能在 A1 exposed split 上补采或调低门槛。

```text
outputs/lewm_action_branch_a1/action_branch_a1_formal.npz
outputs/lewm_action_branch_a1/a1d_screen_gate/summary.{json,md}
outputs/lewm_action_branch_a1/button_phase_diagnostic/{rows.npz,summary.json,summary.md}
```

### Action-Branch Continuous-Task A2（2026-07-17，2k Development-Go / confirmation support No-Go）

A2 在 fresh `24 train / 12 development / 12 reserved test` split 上用 Drawer Open 替换 Button，其他结构保持为 same-state `8` branches、H=`8`、动作 RMS `0.15`、gripper command 保留。原始 H1-H8 support 仍为 No-Go；development-only H6-H8 endpoint 中，Drawer 为 `28/30` informative bundles（bundle fraction `93.33%`），按 `12` 个 physical episode 等权的 support 为 `93.06%`，四任务均通过。

三个固定 seed `25801/25802/25803` 使用 batch `16`、CUDA FP16、冻结 anchor encoder `SIGReg=0.02`，各训练 `2,000` steps。H6-H8 development 上 factual 相对 matched no-action 与七种 wrong-action mean 分别改善 `67.69% / 82.99%`；最弱 wrong-action shift 仍改善 `82.45%`，四任务与三 seed 均同向。raw task-macro MSE 为 `0.153933 / 0.580041 / 0.945480 / 9.416028`（factual / no-action / hold-last / latent CV）。因此该阶段为强 Development-Go，但没有确认性 CI。

独立 confirmation 在 test 未打开前只收集 fresh train：四任务各 `24` 个 episode，共 `218` bundles。Pick/Reach/Push 的 H6-H8 episode-macro support 均为 `100%`，Drawer 为 `72.92%`，低于冻结门槛 `80%`。Drawer 的 step `3-9/10-16` support 均为 `100%`，step `17-23/24-31` 降至约 `42.9%/50.0%`，说明 scripted Drawer 后段仍缺乏稳定动作分叉。合取 Gate 因此为 **Confirmation support No-Go**：`10k` 训练为 `0` steps，reserved test 未采集；不得换 seed、调门槛或按结果过滤 episode。该结果否定当前 confirmation 数据协议，不否定已观察到的 development action signal，也不构成论文第二贡献。

```text
outputs/lewm_action_branch_a2/support_gate/summary.json
outputs/lewm_action_branch_a2/a2d_screen_gate/summary.{json,md}
outputs/lewm_action_branch_a2_confirm/train_support.json
```

### Action-Branch Continuous-Task A3（2026-07-17，四任务 preliminary Confirmation-Go）

A3 使用全新 `24 train / 12 development / 12 reserved test` split，并在任何训练前固定四任务 `pick-place-v3 / drawer-open-v3 / reach-v3 / push-v3`、同状态八动作分支、H6-H8 primary、每任务 train-only whitening 与 `80%` episode-macro support 门槛。八分支为 expert、reverse、hold、half-speed 和四个正交 transverse 方向；slot 在每个 snapshot 匿名置换。这里的“七”仅指 factual 之外的七种 non-identity wrong-action shifts，不是七个任务。

每个训练 seed 对应**一个四任务联合 LeWM**。Development seeds `29201/29202/29203` 各训练 `2,000` steps，固定 batch `16`、CUDA FP16、`fixed_latest` 和唯一 `SIGReg=0.02`；fail-closed Gate 得到 factual 相对 matched no-action `68.80%`、相对七种 wrong-action `85.20%`，四任务与三 seed 全正，因此只授权一次 reserved-test confirmation。

Confirmation seeds `29301/29302/29303` 从同一 frozen base fresh-start，各训练固定 `10,000` steps，不从 2k endpoint resume。test-only scorer 的正式结果为：

- factual 相对 matched no-action 改善 `74.01%`；
- factual 相对 seven-shift mean 改善 `88.18%`；
- 七个 individual shifts 均改善 `87.90%--88.55%`；
- drawer/pick/push/reach 的 factual-vs-no-action 分别为 `77.25% / 70.64% / 73.90% / 74.22%`；
- `4/4` 任务、`3/3` seed、每 seed `4/4` cross-replication 全通过；
- raw factual/no-action/hold-last/CV task-macro MSE 为 `0.202650 / 0.712049 / 1.111683 / 20.221723`。

因此 A3 判为 **Confirmation-Go**。该 Gate 只确认共享 Anchor-Persistent transition 对 factual action 的 held-out predictive increment；不授权 selector、posterior switching、policy rollout、CEM、未见任务或真机 claim。后续 A4/A4b 已按独立协议完成七任务扩展；这不改写本次冻结的四任务结果。

### Action-Branch Active-Control A4/A4b（七任务整改，Confirmation-Go）

A3 结果生成后，对 `action_branch_a3_confirmation.npz` 对应的完整 source trajectories 做了任务阶段重放。`322` 个 hazard 中有 `194` 个在起点已成功：Pick `58/86`、Drawer `37/82`、Reach `53/75`、Push `46/79`。因此 A3 数值继续作为 factual-action robot-anchor transition 机制证据，但不再支持完整 active-task dynamics。Pick 与 Reach 的 reset 距离实际分别为 `0.131--0.172 m` 和 `0.221--0.308 m`；旧可视化把 hazard snapshot 命名为 `Start`，不能据此修改原环境 reset。Push 的 H8 只覆盖 `40` 个 raw control steps，应与完整 source rollout 分开报告。

A4 固定七任务：

```text
pick-place-v3
drawer-retrieve-v0
reach-v3
push-v3
door-open-v3
window-open-v3
towel-wipe-v0
```

`drawer-retrieve-v0` 是新的本地环境，不得以 `drawer-open-v3` 冒名。MVP 使用视觉尺寸与碰撞尺寸一致的 primitive geometry：绿色外壳、红色活动抽屉/内壳、可抓取方块和绿色外壳顶面的薄平面目标区；阶段切换只依据抽屉行程、TCP/方块距离、夹爪接触、方块高度与目标距离，并要求拉开后先抬手，再平移、下探、抓取、抬升、运输到顶板、降低、释放并撤离。成功必须要求顶板接触支撑和夹爪释放，持块悬停在目标处不得成功。

`towel-wipe-v0` 是第二个本地环境，并取代标准 `sweep-v3`。毛巾是带 `freejoint` 的独立物体；机器人必须先用左右夹垫夹紧抓持块、抬起验证抓持，再将毛巾压回桌面，在左右端点间完成六个 half-cycle（即三次完整往返）后抬起。端点计数同时要求真实双侧夹持、毛巾处于表面高度且姿态近似水平，避免把未夹持或悬空横移计作擦拭。环境与专家必须先通过完整 rollout、snapshot 恢复及任务语义测试；两个本地环境还必须通过 FCPG anchor EPE/validity smoke。

注意，标准环境 smoke 只覆盖五个 MetaWorld 任务；加上本地 `drawer-retrieve-v0` 与 `towel-wipe-v0` 后才构成正式七任务面板。A4 的训练单位始终是每 seed 一个 shared seven-task LeWM，batch 固定为每任务四个 bundle（总计 `28`），不得把标准子面板误写成独立训练，也不得拆成七个 task-specific checkpoint。

`drawer-retrieve-v0` 当前提供 `48` 个确定性且互异的 block/顶面目标 instance；专家在全部 `48/48` 上成功，最慢实例不超过 `337` raw steps。代表性预览在 step `204` 首次满足“抽屉开启、方块位于顶面目标区、由顶板支撑且夹爪释放”。`towel-wipe-v0` 的代表性物理抓持轨迹在 step `258` 完成三次往返并抬起。该结果只通过环境/专家/任务语义 preflight，不替代 FCPG EPE、active-control support 或 task-state sufficiency Gate。

proposal 时刻仍在 episode 前生成，以避免按结果挑时刻；是否接收只允许使用当前及历史信息：

```text
eligible_t = proposal_t
  and complete_history_t
  and not success_t
  and not ever_success_before_t
  and active_stage_t
  and expert_xyz_norm_t >= frozen_minimum
  and recent_action_rms_t >= frozen_minimum
  and workspace_and_joint_margin_t >= frozen_minimum
  and not unexpected_collision_t
  and remaining_raw_steps_t >= branch_horizon
```

所有 proposal 都记录接受/拒绝原因。样本一旦接收，八条分支必须全部执行并保存；未来 latent variance、物理 spread、reward 和 branch success 只能用于 task-level support Gate，不能用于逐行过滤。每任务至少 `80%` physical episodes 产生 active bundle，accepted rows 的 post-success/近零动作/当前饱和比例必须为零，source expert success 作为数据集质量指标单独报告。

当前 Anchor-Persistent v1 每帧只持久化三个机器人 anchor tokens。A4 在训练前必须增加 task-state sufficiency Gate：比较 anchor-only 与 `summary/task-context + anchors` 对 object pose、goal-relative progress、接触/抓取状态和分支结果的可读性。只有能表示任务上下文的隔离 checkpoint 变体才可被称为 task dynamics；否则 A4 仍只能写成 robot-anchor transition。

训练固定为每个 seed 一个七任务共享模型，按任务和 episode 均衡采样。全部环境、active-control、physical/latent support 与 task-state Gate 通过后，才允许三个 fresh seed 各跑 `2,000` steps；仅使用 `SIGReg=0.02`。Development 至少要求 `6/7` 任务同向、`2/3` seed 复现，并胜 matched no-action、七个 wrong-action shifts、hold-last 与 CV。通过后才收 reserved test，并从相同 frozen base fresh-start 跑 `3 x 10,000`，不得从 2k resume 或把 A3 test 复用于 A4。

正式 train+development cache 已按该协议完成：`893` 个 accepted bundles、`3,024` 个 proposals，test split 为零。七任务 active/physical/latent support 均通过，逐任务通过数为 Pick `19`、Drawer `24`、Reach `21`、Push `22`、Door `24`、Window `24`、Towel `20`，因此为 **Preflight-Go**。

Development seeds `30201/30202/30203` 各训练固定 `2,000` steps。H6-H8 train-only-whitened branch-centered primary 上，factual 相对 matched no-action 与七种 wrong-action mean 分别改善 `64.120836% / 80.911444%`；两项 paired CI 下界均大于零，七任务和三个 seed 全部同向，冻结的八项检查全部通过，判为 **Development-Go**。该结果当时只授权后续 fresh confirmation；确认已由下述 A4b 独立 panel 完成。

### A4b random-hazard interruption predictor（2026-07-21，development screen Go）

该 screen 不把模糊或遮挡固定在某个 horizon。每个完整专家 episode 的 interruption onset、`2--8` 个 model-step duration 和最多三次事件只由独立随机种子生成，调度器不得读取 task state、action、reward 或 success。`H1--H8` 仅表示距离最后一次有效视觉观测已经经过的递归步数。失效前保留最后三次 frozen FCPG-grounded state；失效后 factual predictor 只读取此后真实执行过的五步 action chunks。

实验冻结 A4b 三组 fixed-latest `10k` factual/matched-no-action checkpoint，不重新训练。七任务各取三个 development physical episode，共 `21` 个 episode 和 `134` 个唯一随机 hidden decisions；每个决策由三个模型 seed 复现，形成 `402` 个模型级记录，但不增加物理样本量。task-macro anchor-state MSE 为 factual `0.076332`、hold-last `0.546122`、latent CV `4.427457`、matched no-action `0.319388`、wrong-action `0.583685`。factual 相对四个基线分别改善 `86.02% / 98.28% / 76.10% / 86.92%`，在 `7/7` 任务上胜 hold/CV，在 `6/7` 任务上胜 matched no-action/wrong-action；age `1--8` 的 pooled factual MSE 均低于 hold-last。

Towel 是明确边界：总体 factual 相对 hold-last 改善 `21.2%`，但 MSE `0.26463` 高于 matched no-action `0.16688`。阶段诊断把主要失败定位到 `wipe_left`（factual/no-action `0.45495/0.22062`），而 `lift_towel` 为 `0.11765/0.30160`。因此该结果支持现有 A4b LeWM 在随机视觉缺测期间提供六任务的强 action-conditioned propagation，并在七任务全部超过非学习传播基线；它不支持把 Towel 边界解释为 interruption 调度失败，也不授权通过延长训练或固定 hazard 时刻来救结果。

准确标签为 **predictor development screen Go**。当前路由使用离线已知 sensor validity，target 为 clean latent；尚未评估真实 corruption detector、learned selector、posterior recovery/re-anchoring、policy rollout 或 task success。下一步只能先用 oracle validity mask 做 paired policy rollout，验证 prediction headroom 是否转化为控制收益；该 Gate 通过后才训练部署式视觉可靠性 selector。

```text
outputs/lewm_action_branch_a4/random_hazard_a4b_7task_dev3_v2_seed42601/summary.json
```

### A4 q-center 非线性 geometry Phase-0（2026-07-17，Exploratory No-Go）

完整轨迹预览暴露出 latent-to-pixel 线性 readout 误差较大，因此在不打开 reserved test 的前提下增加一次有界诊断。该 Gate 使用固定任务视角和 privileged simulator `q_true` center 起点；geometry head 只读取三个 anchor 的 target/predicted latent delta、q0 与 task one-hot，不读取 horizon、time 或 action。三个 head seed 仅在物理 episode 隔离的 train target latent 上拟合，并冻结评估 development target/factual/no-action/wrong-action。TaskTimeQ0 与 DirectAction 均作参数匹配基线；DirectAction 读取三帧 anchor history、两帧历史动作和 expert future-action prefix。主误差先按 head seed、LeWM seed 及 wrong-action shift 分别计算，再做 error-space 聚合，禁止先平均坐标造成假改善。

Target latent 的 mean/P95/PCK@8 为 `5.384 / 11.700 px / 0.786`，factual 为 `7.267 / 14.384 px / 0.656`。Factual 相对 matched no-action `14.901 px` 和 wrong-action `15.526 px` 分别改善 `51.23% / 53.19%`，两项均为 `3/3` seed、`7/7` task 同向；但 TaskTimeQ0 为 `7.034 px`，参数匹配 DirectAction 为 `2.313 px`，factual 对 DirectAction 的配对 mean-EPE 差为 `+4.954 px [4.517, 5.334]`。Target/factual 绝对坐标门槛、TaskTimeQ0 必要性和 DirectAction 必要性均失败，最终为 **Exploratory No-Go**。

该结果只说明当前 A4 latent 含动作敏感的粗几何信息，不支持跨视角、部署式 q0、逐 anchor identity、FCPG/control 或确认性 claim。正式三锚点坐标采集与 geometry head 不启动。若后续重开精确坐标路线，必须改为部署可得 q-state 的直接动力学头，或在 LeWM 训练阶段加入显式坐标辅助监督，并继续把参数匹配 DirectAction 作为必要基线。

### A4 paired-camera input diagnostic（2026-07-17，pixel Go / latent No-Go）

为隔离固定视角是否掩盖 A4 输入问题，使用 schema-v2 collector 生成三组 fresh development 数据：Pick/Reach/Push 各 `2` episode、每 episode `80` 帧，同一 pre-action physical state 重渲染 dynamic 与 nominal-clean RGB。相机范围固定为 `1--3 cm / 2--8 deg`，轨迹分别为 episode-level random static、`20%--80%` linear drift 和 episode `30%--70%` random step。冻结 shared-4task FCPG 与 Anchor-Persistent `latest@2000`，不使用或打开 A4 reserved test。

静态随机、线性漂移和阶跃 post 段的 nominal-to-FCPG EPE 分别为 `18.013->3.085 / 6.538->0.598 / 21.782->3.445 px`，paired clean FCPG EPE 均约 `0.105 px`。因此 FCPG 对这些随机/运动视角仍有明确 pixel-level correction utility。对应的 nominal/FCPG/q-true-oracle 跨视角 projected-anchor latent gap 分别为 `15.707/16.461/16.560`、`10.682/10.797/10.819`、`22.033/21.035/20.650`；FCPG-to-oracle ratio 为 `0.994/0.998/1.019`。完美像素位置没有消除表征漂移，说明主要瓶颈是视觉 token 的 view dependence，而非 FCPG 捕获精度。

判定为 **FCPG pixel mechanism Go / current Anchor-Persistent cross-view canonicalization No-Go**。禁止把相机随机化直接混入 LeWM 训练并把 resulting latent dynamics 解释为纯机器人状态变化，也不据此扩训 FCPG/A4。下一候选仅允许 fresh paired-view canonicalization 或显式 camera-conditioned anchor adapter，并必须保留 raw augmentation、camera-conditioned MLP、q-true target-only oracle 与 clean identity 对照；外生不可预测 jump 仍由 FCPG 当前视觉重锚定。当前数据规模只作接口诊断，不形成论文 claim。

### A4 fresh paired-view canonicalization Gate（2026-07-17，Development No-Go）

在不使用 A4 reserved test 的条件下，另采 static-random train `8 episode/task x 3 tasks x 80 frames`，以及 physical-episode 隔离的 linear-drift 和 random-step development，各 `4 episode/task x 3 tasks x 80 frames`。冻结 shared-4task FCPG 与 Anchor-Persistent `latest@2000` encoder；三种 `70,080` 参数 Adapter 使用相同 seed、batch 和固定 `2,000` steps，只改变输入为 latent-only、部署式 residual/FK context 或 privileged relative-camera transform。输出采用零初始化 residual，训练目标为 paired dynamic-to-clean anchor consistency、clean identity 和 moment preservation。

三种 Adapter 在 static train 上分别改善 `82.61% / 86.01% / 87.47%`，但 unseen linear drift 仅为 `-2.77% / 0.48% / 6.34%`，random-step post 为 `16.01% / 12.97% / 16.46%`。部署式 residual/FK 没有稳定优于 latent-only；即使 privileged camera transform 也未达到可用上限。三组在 linear/step clean cohort 的 adapter gap 分别为 `10.15/9.25`、`9.50/8.58`、`8.36/7.70`，远高于未经 Adapter 的约 `0.08`，且 temporal-delta error 全部恶化。std ratio 为 `0.923--0.978`、有效秩接近 target，排除了简单坍塌解释，指向静态视角映射的 episode/view memorization。

因此该 post-hoc canonicalization 接口固定为 **Development No-Go**：不接入 A4、不重训 10k、不启动 selector/control，也不在当前 split 上加宽或扫 loss。该接口负结果不影响 A4b fresh-panel 的 action-conditioned prediction Confirmation-Go；两者回答的问题不同。若未来重开跨视角 canonicalization，只允许把 paired-view invariance 直接加入视觉 encoder/Anchor-Persistent 训练目标，并使用新的 physical split；这属于新实验，不得事后改写本 Gate。

### Frozen FCPG dense-feature probe and minimal bridge（2026-07-18，probe Go / bridge expansion No-Go）

为区分“采样位置正确但 Tiny descriptor 不稳定”与“整个 Anchor-Persistent 接口无效”，在原始 paired-view 数据上冻结比较 FCPG checkpoint 中完整的 ViT-B-backed dense encoder（backbone + layers `2/5/8/11` 的已训练 projection/fusion/global head）与当前 LeWM ViT-Tiny。完整 `train_static` 含 `1,920` 帧、`5,760` 个 anchor；两个未见 split `dev_linear/dev_step` 各含 `960` 帧、`2,880` 个 anchor。FCPG dense encoder 在 corrected location 的 cosine-gap 改善分别为 `35.99% / 28.76% / 41.44%`，Tiny 为 `2.79% / 1.39% / 4.51%`；两个未见 split 均支持 FCPG descriptor 对校正位置更敏感。该差异不能单独归因于 ViT-B backbone，只证明冻结完整局部描述子的 paired-view 稳定性，不证明 dynamics 或控制收益。

随后执行唯一一次 `2,000`-step 最小输入 compatibility pilot：冻结完整 FCPG dense encoder，以 trainable `1x1`/global projection 接入 `192D` Anchor-Persistent 接口，非视觉参数从 Tiny `latest@2000` warm-start，batch、seed、SIGReg 和训练数据保持一致。最终 bridge val prediction/copy/shuffle MSE 为 `0.012930 / 0.077595 / 0.080864`，Tiny@2k 为 `0.017840 / 0.236096 / 0.261499`。跨 encoder raw MSE 存在 latent 尺度混杂；copy-normalized pred error 为 `0.16663` vs `0.07556`，copy-normalized action-shuffle delta 为 `0.87551` vs `1.03203`。bridge 仍明显 action-sensitive，SIGReg `4.8946` 也排除了零尺度坍塌，但相对 representation dynamics 的预测质量没有优于 Tiny。

因此冻结判定为 **feature probe Go / current warm-start bridge 10k-expansion No-Go**。该 pilot 从 Tiny@2k 继续优化 2k，并非同总预算的严格 encoder ablation；不得把 raw error 差异因果归于 ViT-B，也不得外推所有 ViT-B/LeWM 接法均失败。当前结果不授权扩训 10k、替换 A4、启动 selector/CEM 或写成控制贡献；Stage I FCPG 与 A3 action-conditioned transition 的独立结论均不受影响。权威记录见 `docs/frozen_feature_probe_bridge_status.md`，正式 endpoint 位于 `outputs/lewm_grounded/anchor_persistent_fcpg_vitb_bridge_sig002_seed9216_2k_formal2/`。

### A4 orbit camera-frame action and capacity Gate（2026-07-20，camera action positive / capacity rescue No-Go）

多视角动画暴露出一个具体失效：`clock03` 等未见方位下，Towel 的预测轨迹仍呈训练视角方向。原 ActionPrefixDynamics 接收世界坐标动作，却直接预测相机像素运动；相机旋转后，同一世界动作对应的图像方向会改变。首个修复因此不增加离散 camera ID，而仅使用已知连续相机旋转 `R_cam<-world` 把五个 horizon 的 XYZ 动作变换到 camera frame，夹爪动作保持不变。该变换是必要的坐标对齐，不作为独立算法 novelty。

在同一七任务、同一 seed `40571`、batch `84`、`SIGReg=0.02` 和固定 `2,000` steps 下，camera-frame action 将 odd-view EPE 从原 orbit shared-10k 的 `6.441984 px` 降到 `5.087794 px`，方向 MAE 为 `23.517773 deg`，cyclic wrong-action EPE penalty 为 `18.326225 px`。为隔离额外训练效应，matched world-action control 从同一 parent 用相同配置继续 `2,000` steps，EPE/方向 MAE 为 `6.165520 px / 27.720135 deg`；camera-frame action 分别降低 `17.48% / 15.16%`。Towel `clock03` 也从 `5.883646 px / 34.453701 deg` 降到 `3.686438 px / 27.978096 deg`。因此连续动作坐标对齐确实改善像素运动解码，但 camera-action 的 action-shuffle delta 比 world control 低 `25.35%`，不能据此声称 latent dynamics 整体更强。

随后只允许一次参数量匹配的容量筛查。`0.782784M` bounded anchor Transformer Adapter 与 `0.781248M` token-local MLP Adapter 使用相同输入、相同 train-only q99 residual cap、零初始化有界输出和相同预算。两者相对 camera-action parent 的 EPE 仅改善 `0.134% / 0.108%`，方向 MAE 均略微恶化；因此 anchor attention 和额外 Adapter 容量没有显示可用增量，判为 **Adapter No-Go**。此前未形成公平固定 endpoint 的 `51.52M` expansion 不进入方法比较。

作为容量对照，对 camera-action parent 不新增参数继续训练 `2,000` steps，累计 `4,000` steps 后 EPE 为 `4.788498 px`、方向 MAE 为 `21.935537 deg`，相对 parent 分别改善 `5.88% / 6.73%`，且 cyclic wrong-action penalty 保持为 `18.638723 px`。但 Towel `clock03` 的 EPE/方向误差从 `3.686438 px / 27.978096 deg` 退化到 `4.018343 px / 29.031769 deg`，不得用总体均值掩盖。该结果只授权一次从原 shared parent fresh-start、固定 schedule 的 `10,000`-step camera-frame action endpoint；禁止挑选中间 checkpoint、扫 SIGReg 或启动新的 Adapter。

固定 `latest@10000` 已完成。总体 EPE/方向 MAE 为 `4.768442 px / 22.061935 deg`，相对 `2k` 改善 `6.28% / 6.19%`；但 latent prediction MSE 从 `0.027342` 恶化到 `0.034836`，action-shuffle delta 从 `0.006406` 降到 `0.003659`。相对累计 `4k`，EPE 只再改善 `0.42%`，方向 MAE恶化 `0.58%`。Towel `clock03` 更从 `2k` 的 `3.686438 px / 27.978096 deg` 退化到 `5.763170 px / 35.579232 deg`。因此 fixed `10k` 判为 **global mean saturation / target-view capacity rescue No-Go**，不扩训、不增加 Adapter，也不挑选中间 checkpoint 翻案。

完整 Towel `clock03` 人眼审查使用同一个 reserved-test instance 和 `371` 个真实 raw frames；左侧每前进 `5` 帧暂停，右侧仅根据当前观测和 H1-H8 action prefix 播放 `40` 帧预测。`2k/10k` 各生成 `3,050` 帧 MP4，max-lookahead EPE 为 `7.870109 / 8.113441 px`，H8 为 `8.421792 / 8.802178 px`。页面位于：

```text
outputs/lewm_action_prefix_a4/camera_action_seed40571_2k_towel_clock03_full/index.html
outputs/lewm_action_prefix_a4/camera_action_seed40571_10k_towel_clock03_full/index.html
```

### A3-to-policy known-drop interface（2026-07-18，privileged ceilings Go / strict bridge No-Go）

本阶段绕过已失败的 latent-to-pixel readout，直接在冻结策略 `GroundedTokenAdapter` 后、memory 前替换被判为无效的 `256D` grounded slots。A3 时钟固定为每个 transition `5` 个 raw actions；Pick seed-0 的 `20` episode screen 在 `5/8` raw action per decision 下成功率为 `0.50/0.55`，只支持沿用 A3 的五步时钟，不作性能 claim。

Pick seed-0、`20` episode 的 decision-synchronous known-drop native-token oracle 给出 clean / measurement / hold-last / oracle 成功率 `0.70 / 0.55 / 0.55 / 0.65`。Oracle 相对 measurement 和 hold-last 的成功率均为 `+10 pp`，95% CI `[-15,+35] pp`；相对 clean-policy action 的距离改善分别为 `0.247057 [0.214139,0.284711]` 和 `0.274303 [0.245532,0.301970]`。该 oracle 直接使用同物理时刻 native clean token，并在一个决策中同步覆盖两个历史 slot；它不使用 LeWM prediction，也不满足因果 onset/recovery 时序。因此只判 **privileged policy-interface headroom Go**，不能写成显著 rollout 改善或 deployable bridge。

独立 fresh `20`-episode privileged clock screen 比较 frozen P1 的 raw-adjacent `[t-1,t]`、repeat-current `[t,t]`、decision-spaced `[t-5,t]` 与 hold-last。Active-decision action MSE 分别为 `0.000203 / 0.001221 / 0.008271 / 0.077244`；raw-adjacent 相对 decision-spaced 和 repeat-current 的改善为 `0.008069 [0.006678,0.009458]` 与 `0.001019 [0.000697,0.001406]`。四路 success 为 `0.75/0.75/0.80/0.05`，但方法执行后轨迹分叉，只作诊断。该结果证明 P1 的两个 token slot 具有 raw-adjacent 语义，不能直接用两个五步间隔 A3 state 取代。

三种当前 learned interface 均为 No-Go。Absolute bridge 的 validation token MSE 为 main `0.005456`、参数匹配 geometry `0.005258`、mean-token `0.004637`；main 相对 geometry 为 `-3.76%`，仅 `2/3` 任务正向。Teacher-forced residual bridge 为 `0.001388`，弱于 hold-last `0.001059`、geometry `0.001176` 和 delta-blind `0.001364`，相对 hold-last `-31.13%`、`0/3` 任务正向。使用真实 current measurement embedding 的 privileged oracle-current residual 仍为 `0.001308`，相对 hold-last `-23.50%`、`0/3` 正向。这些结果只否定 shared-policy、旧 `2k` cache 上的当前转换目标，不能外推 A3 transition 或所有 bridge 一般无效。

独立的 H1-H8 privileged horizon ceiling 覆盖 validation `18,600` pairs、`30` physical episodes 与 Pick/Reach/Push。Oracle-current bridge overall token MSE 为 `0.002678`，hold-last 为 `0.003407`，geometry-hold-exact 为 `0.003403`，相对改善 `21.39% / 21.29%`；相对参数匹配 delta-blind `0.002985` 改善 `10.27%`。对 hold-last，H1 改善 `8.46%` 且 `2/3` 任务正向；H2-H8 改善 `16.77%--23.87%` 且全部 `3/3` 任务正向。由于输入是目标时刻真实 current embedding，报告明确 `uses_a3_prediction=false`、`deployable=false`、`control_claim=false`。准确判定是 **privileged representation ceiling Go**：H1-H8 映射本身存在上限，但尚未证明 A3 能提供足够准确的 causal embedding。

Runtime 因此固定为：sensor-valid 时 mask 全 false，完整保留 native `[t-1,t]`；sensor-invalid 时把单个当前 bridge estimate 重复为 `[pred_t,pred_t]`，不使用语义错误的 `[pred_{t-5},pred_t]`；recovery 立即撤销覆盖并恢复 native history。Warmup 至少三次有效观测，A3 只接收实际执行并裁剪后的 `5 x 4` action block；factual、matched no-action 和 geometry/hold controls 各自维护递归历史，learned path 无 hidden-clean 输入。

Strict collector 不是新 physical split，而是重放已暴露 A3 screen 的 train/development physical rows，并用冻结 A3 与精确 Pick-P1 两套 encoder 同时提取。Cache 共 `90` source rows、`36` physical episodes、`8` branches 和 H1-H8；train/validation 为 `62/28` source rows、`24/12` physical episodes，flatten 后为 `3,968/1,792` rows。三枚 A3 `10k` seed 是 correlated prediction replicate，不扩增独立样本。A3 history/target 最大重建误差 `3.10e-4/4.75e-4`，低于 `1e-3`；P1 policy/corrector、source action、split、checkpoint 与 cache hash 均通过严格验证。

同一数据上分别训练 factual 与 matched-no-action residual bridge，初始化 seeds `15715/15716/15717` 全部 `formal_screen_passed=false`。三组 factual validation MSE 为 `0.008357 / 0.007759 / 0.007758`，均弱于共同 hold-last `0.007044`。代表 seed `15715` 的 matched-no-action / geometry-hold-exact / target-oracle MSE 为 `0.007790 / 0.007040 / 0.007458`。Factual 相对 exact geometry-hold 为 `-18.71% [-59.07,+11.26]`，相对 matched no-action 为 `-7.28% [-18.76,+4.18]`；target-oracle 相对 factual 改善 `10.75%`，但仍比 hold-last 差 `5.89%`。代表 seed 的 validation H1-H8 全部弱于 hold-last，而 train factual `0.002361` 显著优于 train hold `0.007640`，显示明显 episode-level overfit。

最终判为 **strict exposed-development cross-backbone interface No-Go**。不启动 causal rollout、CEM 或同 split 参数救援，也不列为论文第二贡献。该结果只否定当前 A3-10k-to-P1 residual bridge；A3 action-conditioned transition、privileged H1-H8 ceiling、Stage I FCPG 与 LeWM 的一般价值不受此结论外推。

```text
outputs/lewm_known_drop_policy/oracle_preflight_pick_s0_dev20/
outputs/lewm_known_drop_policy/slot_spacing_pick_s0_fresh20/
outputs/lewm_policy_token_bridge/interface_pilot_teacher_forced_seed15711_2k/
outputs/lewm_policy_token_bridge/residual_interface_teacher_forced_seed15712_2k_v2_geometry_hold/
outputs/lewm_policy_token_bridge/residual_interface_oracle_current_seed15712_2k/
outputs/lewm_policy_token_bridge/horizon_oracle_h1_h8_seed15714_2k/
outputs/lewm_a3_p1_bridge/strict_pick_train_development_full.npz
outputs/lewm_a3_p1_bridge/formal_pick_a3_10k_p1_h1_h8/
outputs/lewm_policy_token_bridge/formal_a3_10k_pick_seed157{15,16,17}_2k/
```

### P1-native known-drop dynamics（2026-07-18，Development-No-Go）

本 Gate 直接在冻结 Pick-P1 的 `256D` grounded-token 空间学习动作条件缺测传播，以排除 `192D A3 -> 256D P1` cross-backbone bridge 的混杂。严格 cache 是 exposed train/development 数据，不打开 independent test；`24/12` 个 physical episode 隔离，训练 batch 按 episode 无放回抽样、每 episode 一个 hazard，并保留全部八条 same-state action branches。H0-H2 可见，H3-H8 隐藏；所有未来动作、标签与 wrong-action shift 的时间对齐经过 fail-closed 检查。

同容量 factual/matched-no-action heads 在 seeds `15801/15802/15803` 各训练固定 `2,000` steps。三组 factual physical-episode macro MSE 为 `0.003857 / 0.003863 / 0.003814`；相对 matched no-action 改善 `27.67% / 28.07% / 30.39%`，相对 hold-last 改善 `22.97% / 22.85% / 23.83%`，相对 exact-FK geometry-hold 改善 `22.93% / 22.81% / 23.80%`，相对 exact-FK visual-CV 改善 `90.90% / 90.89% / 91.01%`。七种 cyclic wrong-action 对照全部通过。

然而三个 seed 在首个 hidden horizon H3 均相对最强非学习基线退化 `37.02%--37.96%`；H4-H8 才稳定转正，改善范围约 `4.56%--37.79%`。Hold-last 和 exact-FK geometry-hold 的点估计虽一致正向，但以 `12` 个 development physical episode 做 paired bootstrap 时 CI 下界均跨零。冻结 Gate 要求四个 primary comparison 都达到 `10%` 且 CI 下界大于零、H3 退化不超过 `5%`，所以三个 single-seed Gate 全部失败，cross-seed decision 为 **Development-No-Go（0/3）**。

该结果保留为“native dynamics 的 H4-H8 机制信号强，但 onset 不安全”。不扩训 `10k`，不启动 policy rollout/CEM，不在当前 split 调参，也不列为论文第二贡献。若未来重开，必须用新的 causal measurement-to-prior onset 接口和 fresh physical split；当前输出为：

```text
outputs/lewm_p1_native_dynamics/dev_seed1580{1,2,3}_2k/
outputs/lewm_p1_native_dynamics/cross_seed_summary/
```

### P1-native geometry-anchored complementary observer（2026-07-18，Fresh-Go）

H3 common-mode 诊断在 exposed development 上固定了 zero-training 互补更新 `z_h=b_h+K_h(p_h-b_h)`，其中 `b_h` 为 last-valid visual 加当前 exact FK 的 geometry baseline，`p_h` 为 frozen P1-native LeWM prior，`K_H3:H8=[.25,.50,.75,.75,.75,.75]`。更新后的 posterior 每步写回各自历史；视觉有效时必须 bitwise bypass 到 native measurement。三个 `latest@2000` factual/no-action checkpoint、增益和全部 Gate 阈值均在 fresh 数据前冻结。

正式 panel 是 Pick `30` 个新 physical episode、每个 episode `8` 条 same-state branches；H0-H2 只进入 deployment cache，H3-H8 只进入独立 offline-label cache。Fresh 完整 `39D` reset observation + 三枚 FK keypoint 签名为 `30/30` 唯一，与旧 P1 源的 `30,000` 个 exact signature 零重叠。旧源没有完整 reset digest receipt，因此不声称 reset-state 全覆盖非重叠。首次 collector 在 cache 发布前因 `4D` fresh proprio 与 `39D` legacy observation 的签名域错误而 fail closed，未生成 prediction/metric；修正同域 shape contract 后原冻结 panel 原样重跑。

正式统计含 `6,480` 行，并以 physical episode 为唯一独立单位共享 bootstrap draws。Seeds `15801/15802/15803` 的 complementary MSE 为 `0.007202 / 0.007239 / 0.007309`；相对 geometry-hold 改善 `51.60% / 51.35% / 50.89%`，相对 hold-last 改善 `51.63% / 51.38% / 50.92%`，相对 matched no-action 改善 `33.90% / 35.25% / 35.24%`。H3 点改善为 `20.99% / 21.40% / 21.33%`，对应单侧 95% 下界 `19.49% / 19.98% / 19.97%`；H3-H8 `6/6` horizon 和七种 wrong-action `7/7` 均通过，cross-seed decision 为 **Fresh-Go（3/3）**。

该 Gate 只确认 known visual dropout 下的策略原生 token propagation；它没有自动 validity selector、unknown corruption 或 policy-success evidence。positive result 仅授权了一次 paired known-drop policy rollout，不授权 `10k`、VisualTrust、CEM 或论文第二贡献。该 rollout 已完成并判为 policy-success Development-No-Go，见下一节。正式 cache SHA-256 为 deployment `947aed59290032d9993f6ddde88d330e21c8687e6c6b2cf619f7fd42298f62b7`、offline label `52b670b4e41106e7399544ff85c9b770b2309133ea0bff2bd0ad2a5813c70cf6`；完整 representation 结果位于 `outputs/lewm_p1_native_complementary_fresh/evaluation/summary/summary.{json,md}`。

### P1-native paired known-drop policy rollout（2026-07-18，mechanism-positive / policy-success Development-No-Go）

冻结 Pick policy seed `0`、FCPG corrector、三个 P1-native factual/no-action checkpoint 和互补增益，在 `30` 个 fresh physical episode 上执行单进程配对面板。每个 episode 只有一次已知的六决策局部视觉缺测；`clean_native` 建立 same-state clean-policy 参照，八种方法共享 reset、相机、缺测 schedule、drop-onset physical-state digest 与逐字节相同的 pre-drop prefix。视觉有效/恢复时硬旁路到原生 FCPG；`q_true` 只用于外部 corruption renderer，hidden-clean action 只生成离线 action-distance label。

三个 factual seed 相对 exact-FK geometry-hold 的 success 变化为 `0 / +6.67 / 0 pp`；physical-episode clustered seed-macro 为 `+2.22 pp [-10.00,+14.44]`。相对 matched no-action 的 seed-macro success 为 `+6.67 pp [-6.67,+21.11]`。二者均未通过 success Gate。与此同时，相对 geometry 的 action-distance seed-macro 改善为 `44.55% [34.96%,52.71%]`，reward 增加 `146.83 [50.49,253.63]`；相对 no-action 分别为 `45.38% [35.69%,53.60%]` 与 `139.07 [41.06,259.07]`。

因此本 panel 只支持“LeWM 增量改善 known-drop 下的动作一致性和平均累计回报”，不支持 task-success、自动遮挡检测或完整闭环贡献。停止大样本 confirmation、selector 与 CEM。当前 any-step success primary 不得事后改为 final/sustained success；未来若用全新协议重开，必须预先冻结这些更稳健的 success 定义、跨 policy seed 与功效分析后的 episode 数。正式结果只引用 `outputs/lewm_p1_native_policy_rollout/development_all3_seed41901/policy_rollout_report.json`；分进程 seed-specific reports 因 clean prefix 不同不得合并。

### P1-native joint camera-drift + known-drop confirmation（2026-07-22）

独立的联合扰动 screen 先在 `12` 个 episode 上显示 action-distance、reward 与最大连续成功帧均正向。随后在读取确认结果前，以 commit `22ff7ec` 冻结 `30` 个新 physical episode、三个 LeWM seed、固定 `5 deg world-z` 相机旋转、六决策中断、方法面板和 action-distance primary；确认运行禁止通过 CLI 替换 episode、seed 或方法。

确认 primary 通过：factual 相对 exact-FK geometry 的 seed-macro clean-action distance 改善 `46.70% [37.45%,54.33%]`，相对 matched no-action 改善 `51.10% [45.10%,56.53%]`，三个 seed 单独均显著正向。但相对 geometry 的 reward `+67.80 [-0.86,+152.37]` 与最大连续成功帧 `+1.87 [-5.26,+8.83]` 均未显著；any-step success 显著下降 `12.22 pp [2.22,24.44]`。相对 no-action 的 success 也下降 `8.89 pp [1.11,18.89]`。

因此该协议最终判为 **action-fidelity Confirmation-Go / task-control Confirmation-No-Go**。动作条件 LeWM 确实让缺测期 posterior 更接近同状态 clean-policy action，但该 proxy 没有转化为任务收益。停止扩大同一 frozen-policy panel；下一条合法路线必须训练或评估 control-aligned objective/policy，不能继续将 clean-action imitation 当作任务控制贡献。正式报告位于 `outputs/lewm_p1_native_policy_rollout/joint_drift_drop_confirmation_seed52901/policy_rollout_report.json`。

### FCPG x matched-LeWM 缓慢移动目标四格（2026-07-23，development）

本实验复用严格配对 v2 的 `F0L0 / F1L0 / F0L1 / F1L1` 定义、三个 matched LeWM seed 和 `30` 个 Pick physical episode，只将环境绿色目标改为平滑移动目标。XY 振幅为 `1.0/0.6 cm`、周期为 `440` 个 raw step；每个 episode 的目标相位由独立 seed 决定。目标轨迹同时驱动 reward、observation 与可视 goal site。四格共享 action prefix；同 grounding 的 L0/L1 共享 native-token 与 sampled-visual prefix；同一 physical episode 在四格和三个 LeWM seed 间共享逐点完全相同的 `336`-point 目标轨迹。实际最大目标位移范围为 `1.27--2.00 cm`。

`F0L0 / F1L0 / F0L1 / F1L1` 的 sustained success 为 `16.67 / 26.67 / 11.11 / 26.67%`。完整系统相对 FCPG-only 为 `0 pp [-13.33,+13.33]`，相对 LeWM-only 为 `+15.56 pp [+3.33,+30.00]`。Post-drop success fraction 为 `1.829 / 4.161 / 0.860 / 6.401%`，完整系统相对 FCPG-only 增加 `+2.240 pp [+0.046,+5.155]`，交互为 `+3.210 pp [+0.124,+7.637]`。最大连续成功帧为 `5.50 / 12.27 / 2.53 / 19.02`，对应增加 `+6.756 [+0.111,+15.544]`、交互 `+9.722 [+0.367,+23.200]`。累计 reward 为 `160.67 / 145.94 / 145.92 / 206.91`，对应增加 `+60.97 [+8.34,+126.22]`、交互 `+75.72 [+2.61,+173.03]`。三个 LeWM seed 的三项连续 Full-vs-FCPG-only 点估计均为正。

一个新 identity 的 `10`-episode screen 在 moving/static 两种目标下均只有 `1/10` sustained success，说明该冻结策略在新 panel 上存在覆盖 floor，而非移动目标单独造成退化。因此正式 `30`-episode 结果仍属于已知 informative panel 上的 development mechanism evidence。判定为：**移动目标下持续控制联合机制 Go，二值 task-success 增量未通过。** 不把该结果外推为跨任务、fresh confirmation 或完整第二贡献。配置与结果位于：

```text
config/lewm/eval/fcpg_lewm_factorial_pick_matched_moving_goal_dev30_v2.json
outputs/fcpg_lewm_factorial_v2/pick_matched_moving_goal_dev30_seed990000/
```

正式 crossed summary SHA-256 为 `96cf9be98263b0eaa80642ca3f8c9c13c621b058d51637fe05f624f2590b92a8`。

### Causal expert-action residual（2026-07-22，offline imitation Go / policy-control No-Go）

冻结 Pick policy、FCPG 与 P1-native observer，以 scripted expert 的未来五步 XYZ 动作为标签，分别为 factual、matched no-action 和 exact-FK geometry 三种状态源训练同构 residual heads。每个 head 仅修改 XYZ，gripper bitwise 旁路；训练时使用 `30` 个 expert-forced episode、development 使用 `15` 个独立 episode，三个优化 seed 均固定 `2,000` steps。测试控制路径不读取 expert、hidden-clean action 或 `q_true`，黑色遮挡以 `q_nominal` 为中心，视觉有效时 residual head 被严格旁路。

离线 Gate 通过：factual residual 相对 factual base、matched no-action residual 与 geometry residual 的 XYZ MSE 分别改善 `57.72% / 30.08% / 25.57%`，三项 paired CI 下界均大于零。随后在 `30` 个 fresh Pick physical episode 上执行一次固定 sustained-success primary 的 rollout。Factual residual 相对 factual base/no-action/geometry 的 seed-macro sustained-success 差为 `-1.11 / 0 / +3.33 pp`，对应 95% CI `[-17.78,+15.56] / [0,0] / [0,+7.78]`；没有 head seed 同时胜三类基线，最终 Gate 为 **No-Go**。

该结果证明 expert-forced 短期动作拟合与闭环任务收益之间仍有明显分布和控制目标错位。停止扩训、扫参、追加 episode 与 CEM；不得把离线 MSE 改善写成 LeWM 控制贡献。正式结果位于 `outputs/lewm_expert_action_residual_pick/{matched_heads_2k,fresh_rollout_30eps}/report.json`。

### On-policy posterior DAgger residual（2026-07-22，No-Go）

为隔离旧 expert-forced residual 的 covariate shift，本协议执行两轮 decision-level DAgger。每个 decision 先保存完整模拟器、RNG 与 observation 状态，从同一状态分支执行五步 scripted expert 只生成标签；恢复后执行 learner 自己的五步 action chunk，并只将该 learner chunk 写回 observer history。Round 0 禁止 behavior head，round 1 必须加载同方法、同 seed、固定训练 endpoint 且来自上一轮的 head。三种方法使用相同初始 physical episode panel；轨迹分叉后不强行伪造逐 decision 配对。每轮每方法为 `12/6` 个 train/development episode，训练对两轮等权；最终 head 固定 seed `64301`、`2,000` steps，只修正 XYZ，gripper 和恢复帧严格旁路。

离线 factual head 相对自身 base 将 XYZ MSE `0.53737 -> 0.46386`，改善 `13.68%`，但 95% CI `[-23.15%,+28.42%]`。相对 no-action residual 点改善 `11.21%`，相对 geometry residual为 `-1.84%`；因 method-specific learner trajectory 分叉，两项使用 unpaired physical-episode bootstrap，均无稳定优势。

`12` 个 fresh Pick episode 的 sustained success 为 factual base/factual DAgger/no-action DAgger/geometry DAgger=`58.33/50.00/25.00/41.67%`。Factual DAgger 相对 factual base 为 `-8.33 pp [-33.33,+16.67]`，相对 no-action 为 `+25.00 pp [0,+50.00]`，相对 geometry 为 `+8.33 pp [-16.67,+33.33]`。所有方法 recovery exact-native bypass 为 `100%`。Primary 因未超过 factual base 判为 **No-Go**；factual-vs-no-action 只保留为 single-seed development 动作条件信号，不升级为控制贡献。停止追加 DAgger round、seed、episode、head 容量或 CEM。结果位于 `outputs/lewm_posterior_dagger_pick/on_policy_v2/{heads/final,rollout/fresh}/report.json`。

### Geometry-Anchored Control-Sufficient Dynamics（2026-07-19，P0-No-Go）

#### 研究假设与借鉴边界

当前 P1 的剩余问题定义为 representation-objective mismatch，而不是默认归因于容量、full-memory 或 block-causal interface：普通 token MSE 对所有维度等权，但冻结策略可能只对少数 anchor-token 方向高度敏感。候选先验证控制充分性，再决定是否训练；旧 P1-native、complementary observer 和 policy rollout 的 Gate 与数字全部冻结，不得事后改写。

本候选直接借鉴 *Feedback World Model Enables Precise Guidance of Diffusion Policy*（arXiv:2605.15705）的 Eq. 8--15。Eq. 8--12 给出 latent velocity、观测 residual、feedback correction、feedback state 递推和下一观测 residual：

```text
v_t(a)     = (f_theta(z_t, a) - z_t) / dt
e_t        = z_t - zbar_t
vhat_t(a)  = v_t(a) + L e_t
zbar_next  = zbar_t + dt * vhat_t(a_executed)
e_next     = z_next - zbar_next
```

Eq. 14--15 则从候选动作反事实方差得到逐维 controllability，并按均值归一化后用 `beta` 插值：

```text
w_j        = mean_i Var_{a in A_demo} [f_theta(z_i, a)]_j
wbar_j     = w_j / (mean_k(w_k) + eps)
w_beta,j   = 1 + beta * (wbar_j - 1)
L_ctrl     = sum_j w_beta,j * error_j^2
```

这些公式、observer 解释、feedback gain 与 controllability weighting 都是外部工作的已有贡献，不是本项目创新。原方法在每个环境步都取得下一观测，并在 always-observed feedback setting 下给出 Eq. 13 的有界性分析；本项目在视觉完全缺测时没有 Eq. 12 的新 residual，只能保持缺测前最后一个因果 residual。该 held-residual missing-measurement extension 明确不继承 Eq. 13 的定理。

解析 observer 合同必须在读取 offline label 前全局冻结为 `dt=1`、`L=I`（scalar `feedback_gain=1`）、`residual_decay=1`。同一参数用于所有 episode、horizon、coordinate、branch 和 model seed，不允许按 hidden-clean target 调 gain/decay。P0 的 controllability fit 同样只产生一个全局向量：在与 scored panel 的 physical episode 和 observed-state signature 均不重叠的独立 fit 源上，每个 episode 固定匿名 `branch-0` 因果历史，再将八条 future-action bank 从该同一状态进行冻结 LeWM rollout，计算 Eq. 14-style variance；fit target 不被读取。随后按 Eq. 15 固定 `beta=1`。这不是原论文的 state-independent demonstration action pool，必须标记为 project-specific local-branch transfer，而不是“复现 Eq. 14 action pool”。

FCPG 的独立边界保持不变：名义 FK 投影提供固定物理身份 anchor，当前局部 RGB 预测二维 residual，校正位置用于重新读取策略特征。视觉有效时 observer measurement 必须是该 FCPG token；视觉缺失时只能使用 last-valid measurement、当前 FK/proprio、已执行动作和历史 feedback state，不能读取 hidden-clean token、真实可见性、未来帧、`q_true` 或 corruption schedule。LeWM 只建模机器人/任务动力学，不预测外生相机 jump、遮挡 onset 或背景变化。

#### P0：zero-training Gate

P0 复用冻结 P1-native cache、三个 `latest@2000` checkpoints、same-state 八动作 branch、独立的 deployment-fields-only controllability fit 与 hidden-clean offline labels，不产生任何 optimizer step。它是 known-mask/offline exposed Gate：真实 `sensor-valid` 只负责缺测路由，健康视觉 carrier 与 clean anchor 只用于离线 exact policy scoring；因此本 Gate 不证明可部署的 corruption detection、unknown-mask switching 或 rollout 收益。每一行必须绑定 physical episode、horizon、branch、policy/FCPG/P1 hashes；统计以 physical episode 聚类，不能把同一 episode 的 branch 或 horizon 当作独立样本。

冻结 policy 的 action sensitivity 必须在完全相同的 raw-adjacent two-slot carrier `[N,2,K,D]` 上测量。RGB/视觉前缀、proprio/FK、task 和其余 carrier 输入逐字节保持不变；按照已冻结的 endpoint alignment contract，将一个 model-step endpoint 重复到两个 anchor slots，再分别前向 predicted anchors 与 clean anchors。`E_action` 是二者的 exact frozen-policy action MSE，而不是 action-distance proxy。clean anchors 在同一 replay 中记录，但 label arrays 必须在所有 deployment prediction、observer rollout 和 global weight fit 完成后才 stack、打开并用于评分，且从不进入 predictor forward。

统一输出：

1. `E_uniform = ||z_pred-z_clean||^2`；
2. `E_cf`：用独立 fit 源的 same-state local action branches 估计、对全 scored panel 冻结的单一 Eq. 14-style weight vector 所得到的 weighted error，Eq. 15 `beta=1`；
3. `E_jac = e^T J_pi^T J_pi e` 与 exact `E_action = ||pi(M_pred)-pi(M_clean)||^2`；
4. raw factual LeWM、matched no-action、七种 causal wrong-action、hold-last、exact-FK geometry、既有 complementary observer；
5. 全局固定 `unit-gain + held pre-drop residual` 的 analytic feedback observer。它不读取 hidden-clean target，也不是 target-only upper bound；但因缺测路由已知，仍只属于 exposed 诊断。

P0 合取门槛：

- `E_cf` 相对 `E_uniform` 对 exact action error 的 Spearman 提升至少 `0.10`，差值 95% CI 下界 `>0`；Spearman 必须先在同一 `(physical episode, horizon)` 的八个 branch 内计算，再做 macro average，不能跨状态池化；
- 全局固定 analytic observer 相对既有 complementary observer 的 exact action error 改善至少 `10%`，95% CI 下界 `>0`；
- H3 action error 相对 complementary observer 不得退化超过 `5%`，H3-H8 至少 `5/6` horizon 正向；
- factual exact action error 的点估计必须低于 matched no-action 和七种 wrong-action controls；这些 control 全部使用各自因果递归 history 和相同 exact policy carrier，episode-bootstrap CI 只作诊断，不新增门槛；同时不能通过缩小 latent 尺度使 copy/constant predictor 同步改善。

P0 是 exposed diagnostic。上述阈值是为本候选预先冻结的**工程决策门槛**，不是 Feedback-WM 定理、通用统计标准或其论文阈值。通过只表示 control-aware training 有可测 headroom；失败则立即停止，不得改阈值、换相关性指标、扩大模型或训练后再反向解释 P0。只有 P0 合取 Gate 为 Go，才允许启动下面参数、数据、seed 和预算匹配的 `2,000`-step 比较。

#### P0 执行结果

旧 carrier 的 H1/H2 已经包含 branch-specific action，首次 evaluator 因 same-snapshot branches 的 causal pre-drop state 不同而 fail closed；该结果只揭示协议错误，不是模型 No-Go。修复后的 common-prefix v2 先共享 `H-1/H0/H1/H2`，再于 H2 后产生八条 action branches，并只执行/评分 H3-H8。正式评分包含 `30` 个 physical episodes、`1,440` 行；三个 predictor seed 只做宏平均，episode-cluster bootstrap 不把 branch、horizon 或 model seed 当独立样本。

动作对照为强正：factual exact action error `0.05157`，matched no-action `0.11178`，相对改善 `53.87%`；七种 wrong-action 相对改善 `57.28%--59.09%`，全部方向通过。因此 fixed-identity LeWM 中确有冻结 policy 可消费的动作条件信息。两个关键迁移假设却同时失败：uniform/weighted error 对 exact action error 的 Spearman 分别为 `0.24696 / 0.23624`，增量 `-0.01071`、95% CI `[-0.03056,0.00860]`；analytic observer 相对 complementary observer 的 action-error 改善为 `-30.63%`、95% CI `[-45.07%,-17.99%]`，H3-H8 `0/6` 正向。H3 只退化 `0.91%`，但 H4-H8 的退化扩大到 `27.81%--45.59%`，符合 held residual 持续注入导致 common-state bias 累积的解释。

最终合取判定为 **P0-No-Go**。不启动下面的 matched `2k`，训练步数为 `0`；不重调 gain、decay、`beta`、相关性指标或 Gate 阈值，也不授权 rollout、`10k`、selector、CEM 或第二贡献。正式结果位于 `outputs/policy_token_sensitivity_carrier/exposed_p0_v2_common_prefix_scored_h3_h8/`、`outputs/control_sufficient_p0/exposed_p0_v2_scored_h3_h8_row_panel.json` 与 `outputs/control_sufficient_p0/exposed_p0_v2_scored_h3_h8_summary.json`。

#### P1：matched 2k（P0 未通过，未授权）

P0 通过后才冻结新 physical split、config 和 endpoint ledger。训练三个 matched variants：

```text
M0 = L_token
M1 = L_token + lambda_ctrl * L_cf
M2 = L_token + lambda_ctrl * L_cf
             + lambda_pi * ||pi(M_pred)-stopgrad(pi(M_clean))||^2
```

三组必须保持相同 P1-native architecture、data、初始化、三个 model seeds、optimizer/scheduler、batch、`2,000` steps 和 fixed-latest selection。FCPG、policy、视觉 encoder 全部冻结；策略允许梯度传到 predicted anchors，但策略参数不得更新。Controllability weights 只从 train-side same-state action interventions 估计；`lambda_ctrl/lambda_pi` 按 train loss scale 预先归一化并在查看 fresh development 前冻结。M0 是必要 matched baseline，不能拿旧 checkpoint 代替。

P1 Go 要求 M1 或 M2 相对 M0 **以及**既有 complementary observer 的 exact action error 都改善至少 `10%`、paired CI 下界 `>0`；uniform token MSE 恶化不超过 `5%`；H3 非劣、H3-H8 至少 `5/6` 正向；wrong-action `7/7` 保持，至少 `2/3` model seeds 单独通过。满足后只授权一次 fresh known-drop policy rollout，并预先冻结 final/sustained success、跨 policy seed 与 episode power；不自动授权 `10k`、unknown-corruption selector 或 CEM。

#### 禁用 claim

P0/P1 完成前以及仅有 representation/action-distance 结果时，禁止声称：提出 feedback world model、latent observer、controllability weighting、理论收敛、Bayesian posterior、uncertainty calibration、unknown-corruption detection、camera-jump prediction、online recalibration、planning/CEM、task-success improvement 或完整 closed-loop system。潜在第二贡献只能在 fresh policy-success 通过后写成“FCPG geometry-anchored measurements 与 control-sufficient fixed-identity anchor dynamics 的组合”；LeWM、SIGReg、Transformer、feedback observer 和 controllability 本身均不得列为 novelty。

Confirmation seeds `30301/30302/30303` 已从同一 frozen base fresh-start 并分别完成固定 `10,000` steps；不从 2k resume、不挑 best，SIGReg 仍唯一固定为 `0.02`。三个 latest checkpoint 均严格达到 step `10000` 并通过双模型状态、训练 seed、固定端点和缓存绑定检查。

随后执行了旧 A4 fixed-panel 协议内唯一一次 test-only collection。冻结合同要求七任务各 `12` 个 test instance 都产生至少一个合格 active-control bundle；本次 fixed panel 中至少一个任务未满足该覆盖，因此 collector 在发布 test cache 前 fail closed。没有生成 reserved-test cache、三 seed predictions 或 confirmation metrics，也没有读取结果后调门槛或更换 instance。该旧 A4 fixed panel 的状态为 **Confirmation collection No-Go / unscorable**：七任务 `2k` Development-Go 保留，三个 `10k` endpoint 保留，但不能从该 panel 形成确认 claim，且不得在本 panel 上重采翻案。

上述结论只针对旧 A4 fixed panel。随后建立的 A4b 使用 fresh `a4b-task-instance-v1` identity namespace、split manifest `a4b-fresh-preflight8-final40-seed31401-v1`、benchmark seed `31411` 和新 physical episodes；与 legacy preflight-final digest 的交集为零。它没有补采旧 panel、降低门槛或挑 checkpoint。七任务各 `12` 个 test physical episodes，共 `84` 个 physical episodes、`325` 个 accepted test bundles；三个 checkpoint 固定为 seeds `30301/30302/30303` 的 `latest@10000`。

A4b H6-H8 task-macro branch-centered whitened MSE 为 factual `0.271343`、matched no-action `0.765870`。`65.627%` 是先在每个任务内计算相对改善、再对七任务等权平均的 estimand，并非上述两个宏 MSE 的直接比值；paired task-macro MSE-difference 为 `0.494527 [0.460417,0.525799]`。`7/7` 任务、`3/3` seed 点估计同向，七种 wrong-action donor shift 均改善 `83.24%--83.85%` 且各自 CI 下界大于零。未中心化但仍按任务 whitening 的 factual/no-action/hold-last/CV MSE 为 `0.321770 / 0.869193 / 1.834268 / 45.813435`；这些不是像素或物理单位误差。冻结 Gate 全部通过，正式判为 **Confirmation-Go**。

该 Gate 只授权如下结论：共享七任务 TaskContext-Persistent LeWM 在 fresh active-control branch panel 上稳定利用 factual action 改善未来 task/anchor-state prediction。它不授权 policy rollout、task-success、CEM、缺测 switching、未见任务或端到端闭环 claim。正式结果为 `outputs/lewm_action_branch_a4b/confirmation_gate/summary.{json,md}`。

### A4/A4b full-trajectory coverage audit（2026-07-20，boundary diagnostic）

为区分 active-control transition 与完整任务轨迹，新增 reset-to-expert replay audit。审计按 physical episode 和 unique proposal 去重，固定 `H=8`、每个 model step 对应 `5` 个 raw steps。A4b 的 `84` 个 test episodes、`325` 个 accepted bundles 全部在 proposal 时处于 pre-success，故 active-control 数据没有 post-success 混入；但 H8 跨越成功转移的比例并不均衡：

| Task | H8 首次 success | H8 稳定 success |
| --- | ---: | ---: |
| pick-place-v3 | 85.7% | 66.7% |
| reach-v3 | 91.4% | 91.4% |
| push-v3 | 50.0% | 30.0% |
| door-open-v3 | 39.1% | 37.5% |
| window-open-v3 | 28.9% | 24.4% |
| drawer-retrieve-v0 | 0.0% | 0.0% |
| towel-wipe-v0 | 0.0% | 0.0% |

当前 `joint_orbit12_gripper_towel_fullcycle_24train_8dev` raw dataset 的 `210` 个 physical episodes、`224` 个去重 bundles 也显示同一边界：稳定 success H8 coverage 为 Pick `46.9%`、Reach `75.0%`、Push `6.25%`、Door `0%`、Window `0%`、Drawer `0%`、Towel `12.5%`。Towel 的 full-cycle replacement 修复了其 proposal 时段，但没有让大多数 H8 窗口覆盖完整任务；其余六个任务仍沿用 legacy phase-biased rows。

准确结论：A4b Confirmation-Go 仍只支持 fresh active-control action-conditioned transition prediction；它不支持完整任务轨迹预测、success transition coverage 或长时 closed-loop control。当前 seed `40571` 的 `2k` endpoint 同样只能作 diagnostic。后续数据必须按任务阶段/成功转移采样，或为长任务使用分阶段更长窗口，并在训练前报告 H8 coverage；不能用增加视角、重复早期 proposal 或增加训练步数替代 coverage。

正式产物：`outputs/lewm_action_branch_a4b/final/trajectory_coverage_audit.json` 与 `outputs/lewm_action_prefix_a4/joint_orbit12_gripper_towel_fullcycle_24train_8dev/trajectory_coverage_audit.json`。入口为 `scripts/audit_lewm_a4b_trajectory_coverage.py --dataset <raw-dataset-dir>`。

### A4c phase-aware data refresh（2026-07-20，protocol ready）

为修复完整任务覆盖不足，新增独立 A4c protocol；A4/A4b 的冻结数据和结论保持不变。A4c 仍使用七任务共享 LeWM、`H=8`、`5` raw steps/model step、每 episode 目标 `12` 个 proposal。不同之处是先从 reset 完整回放 expert，再按 phase/contact transition、进度分层和 success 前窗口生成冻结 schedule；schedule 在 branch 执行前写出，branch outcome 不参与 proposal 选择。这个 schedule 设计合同明确允许使用 expert replay 作为 collection-time 设计信息，但不允许用未来 branch reward、future latent 或 branch success 做 row filtering。

对七任务全部 `24 train + 12 development` episodes 的 CPU dry-run（共 `252` 个 physical episodes）结果：

| Task | mean first success | mean selected | H8 success crossing | phase transition |
| --- | ---: | ---: | ---: | ---: |
| Pick | `53.9` | `8.14` | `36/36` | `36/36` |
| Reach | `50.0` | `8.31` | `36/36` | `36/36` |
| Push | `61.4` | `10.19` | `36/36` | `36/36` |
| Door | `79.0` | `12.00` | `35/36` | `36/36` |
| Window | `87.4` | `12.00` | `36/36` | `36/36` |
| Drawer retrieve | `239.9` | `12.00` | `35/36` | `36/36` |
| Towel wipe | `298.4` | `12.00` | `36/36` | `36/36` |

这证明的是 schedule coverage，不是 LeWM 性能。Drawer 和 Towel 的 phase 分层已覆盖完整操作链；Door/Drawer 各有一个因终止边界过近而无法得到完整 H8 success-crossing 的 episode，未做伪造 padding。A4c 产物和入口为 `outputs/lewm_action_branch_a4c/phase_schedule.json`、`scripts/plan_lewm_a4c_phase_schedule.py`、`scripts/collect_lewm_action_branch_a4c.py`。accepted-cache 复审入口为 `scripts/audit_lewm_a4b_trajectory_coverage.py --environment a4 --ledger outputs/lewm_action_branch_a4c/proposal_ledger.npz --cache outputs/lewm_action_branch_a4c/state_cache.npz`。当前尚未发布 A4c state cache，也未基于旧 raw dataset 追加训练。下一步必须先用 CUDA0 收集、重新审计 accepted bundles，再运行单个 task-balanced shared model；SIGReg 只允许 `0.02`。

### A4c FCPG-to-LeWM innovation transport（2026-07-21，minimal mechanism Gate）

冻结七任务 `10k` history-conditioned world-3D LeWM 和 A4 multiview FCPG。部署合同严格保持 nominal camera calibration；实际漂移外参只用于 RGB 生成和 `q_true` 评估。FCPG 当前 measurement 与 LeWM nominal future projection 的最小接口为：

```text
innovation_t = q_fcpg_t - q_nominal_t
q_future = project(K, T_nominal, X_future_LeWM) + gain(innovation_t) * innovation_t
```

`gain` 的第一版不训练：anchor correction magnitude `<=4 px` 为 `0`，`>=8 px` 为 `1`，中间线性过渡。`4/8 px` 在 Gate 前根据 even-view FCPG development 的 clean/drift correction scale 固定。`clock03 + 5 deg world-z`、每任务一个 matched episode 的暴露机制结果为：nominal/current-only/constant/gated/current-oracle 的 task-macro stitched EPE 分别是 `20.5437/20.1819/7.4316/7.4314/7.1143 px`。Gated 相对 nominal 改善 `63.83%`，`7/7` 任务同向；clean penalty 为 `0.33%`，而 always-on constant 为 `8.77%`。

Renderer 现支持 `--test-instance-rank 0..11`。不改模型、阈值或 checkpoint 的 `rank=1` fresh sanity 已覆盖 `clock01/clock05`：在 `5 deg world-z` 下 gated 相对 nominal 的 task-macro stitched EPE 改善为 `65.94%/69.70%`，两者均 `7/7` 任务同向；clean penalty 为 `0.58%/1.59%`。该结果只升级为 two-view fresh sanity，完整确认仍必须增加 physical episode、漂移轴与强度并按 episode 计算配对 CI。

第一阶段 confirmation 已扩展到 ranks `1/2/3` 与六个 odd views，共 `21` 个 physical episode。按 episode 配对、task-macro bootstrap 后，`5 deg world-z` gated 相对 nominal 的 stitched EPE 为 `19.4611 -> 7.3009 px`，改善 `62.48% [61.03%,63.90%]`，`7/7` 任务同向；current-only 仅改善 `1.86% [1.76%,1.97%]`。Clean 为 `6.7343 -> 6.8594 px`，点估计退化 `1.86%`，但区间上界达到 `2.61%`，四个任务单项超过 `2%`。结论冻结为 **drift mechanism confirmation Go / clean safety pending**。不得在该 confirmation 上调 magnitude thresholds；下一 Gate 是冻结 transport 的 same-checkpoint action-output 与 rollout 非劣/增益验证。

该 Gate 只授权 fresh confirmation。确认协议应固定：

1. 七任务共享 checkpoint，不重训 LeWM/FCPG；
2. 多个未暴露 physical episode，按 episode 配对；
3. odd views 至少覆盖 `clock01/03/05/07/09/11`；
4. clean、旋转 `2/5/10 deg` 和平移 `1/2/4 cm`，至少两个旋转轴与两个平移方向；
5. 比较 nominal、current-only、constant、frozen magnitude gate、current-oracle transport；actual-extrinsic 仅作 offline ceiling；
6. primary 为 H1/H4/H8 与 stitched anchor EPE，task-macro episode bootstrap；secondary 为 innovation magnitude、PCK 和 policy action-output distance；
7. Go 要求 drift task-macro EPE 至少降低 `25%`、CI 下界大于零、至少 `6/7` 任务同向，并关闭至少 `70%` oracle gap；clean 相对退化不超过 `2%`。

确认通过后才接 policy rollout。当前不训练 geometry-conditioned transport head，不做 CEM，也不把单视角机制结果写成控制贡献。

#### 动态相机下的 recursive next-observation re-centering（2026-07-21，Development No-Go）

另行测试了一个更强的闭环接口：每隔三个 model step 读取一次新观测，使用上一轮 LeWM H3 robot-anchor prediction 作为下一轮 FCPG 的搜索中心，再将校正结果写回。相机在 raw step `5--35` 从 `0` ramp 到 `10 deg`；冻结同一七任务 `10k` history-conditioned world-3D LeWM 和 A4 multiview FCPG。该接口不读取 future RGB、实际外参或 `q_true`。

最初使用绝对像素 hold/CV 作为 temporal baseline 时，LeWM 在七任务上表现为强正；但该比较忽略了下一观测时当前 FK 已经可用。修正后的强基线统一写成：

```text
search_center_t = q_nominal_from_current_FK_t + filtered_observed_camera_residual_t
```

在 `rank=4, clock03` pilot 的 `>=8 deg` 区间，one-shot / LeWM recursive / residual hold-last / clipped CV / linear CV Kalman 的 task-macro EPE 分别为 `14.8839 / 3.4478 / 1.6441 / 1.5608 / 1.6379 px`。LeWM 相对 one-shot 改善 `76.84%`，但在 `7/7` 任务均未胜任一简单 residual-filter family 的最佳成员，因此该接口判为 **LeWM-specific No-Go**，不执行原计划的 fresh multi-view confirmation。

边界必须保持准确：该结果证明 arbitrary-center FCPG 与递归重定中心能扩展 one-shot 捕获范围，同时证明 LeWM 不适合承担“下一真实观测时 robot anchor search center”这一职责；当前 FK 已解释机器人运动，低维相机 residual 由简单滤波传播更合适。它不否定上一节在未观测 future projection 上的 fixed-drift innovation transport，也不评价 LeWM 对非 FK 物体/接触/task anchors 或候选未来规划的价值。正式机制产物为 `outputs/lewm_action_prefix_a4/recursive_grounding_ramp_pilot_v2_fkresidual_rank4_clock03_7task_h3/summary.json`。

### Towel world-goal action ranking（2026-07-21，H3 Development No-Go）

为避免二维 ridge readout 混淆，新增 zero-training world-space selector。冻结同一七任务 `10k` history-conditioned world-3D LeWM，对 Towel development 中每个 same-state 八候选 bank 直接解码 H1-H8 三枚 future world anchors；cost 是 tool-center 到当前 task-defined endpoint 的三维距离。强基线直接按 MetaWorld 已知 `0.01 m` action scale 积分每个候选的 XYZ command。actual future anchor 只用于离线计算 candidate ranking/regret，不进入 selector。六个 odd camera views 先按 `physical_bundle_index` 平均，正式 screen 只有 `12` 个 physical snapshots。

固定 H3 Gate 未通过：LeWM/kinematic 的 Spearman 为 `0.9680/0.9700`，top-1 为 `0.667/0.750`，regret 为 `0.0032/0.0027 mm`，相对改善为 `-16.3%`。cyclic wrong-action 的 H3 Spearman 为 `-0.0621`，说明模型确实使用对应 action，但 H3 解析运动学已近乎完美。该 No-Go 不得因 horizon sweep 改写。

后验 H4-H8 显示明确中程信号：LeWM regret 相对 kinematic 依次降低 `35.0% / 61.6% / 92.0% / 93.1% / 82.9%`。这只授权一个独立的新假设，不授权当前结果成为论文 claim：在 fresh Towel episode 上，由冻结 policy 生成 center prefix，构造相同匿名候选；固定 H6 评分，执行第一段 `5` raw steps 后重新观测。先在 clean 条件比较 policy-only、analytic-kinematic selector 与 frozen-LeWM selector；只有 LeWM 同时降低 endpoint error 且不减少 completed half-cycles，才加入 `5 deg` drift，并比较 nominal measurement 与已冻结 gated FCPG measurement。不得使用记录的 expert future prefix、future RGB、actual future anchor 或 `q_true` 进行在线选择。

### A4 physical-state decoding（2026-07-18，exploratory mechanism diagnostic）

同一 A4 development split 上另做 physical-state decoding 探索诊断。H6-H8 factual 相对 no-action 的 object position、task-controlled position、task progress 与 TCP position whitened MSE 改善分别为 `0.2861 [0.2299,0.3432]`、`0.3013 [0.2487,0.3521]`、`0.2172 [0.1437,0.2906]`、`0.3924 [0.2625,0.5197]`；四类量的七种 wrong-action shift CI 均为正，且 factual 胜 task-time。参数匹配 DirectAction 在前三类量上仍明显更强，TCP 近似持平。因此该结果只标记为 **exposed-development mechanism diagnostic**，不形成确认性或控制 claim。正式记录为 `outputs/lewm_action_branch_a4/physical_state_decoding_exploratory/report.json`。

旧实现前 smoke 曾包含后来移出 A4 的标准 `sweep-v3`，只保留为历史诊断。当前 `v2` 标准子面板固定为 Pick/Reach/Push/Door/Window 五个任务；reset success 均为 false，首次 success 分别为 `52/42/62/77/90`，全部在 `500` raw-step 内达到连续三帧 success。自定义 Drawer/Towel 由统一 A4 task factory 单独执行完整语义 rollout。任何环境 smoke 都不替代 active-control bundle support、FCPG EPE 或 task-state sufficiency Gate，也不授权训练。

```text
outputs/lewm_action_branch_a3/a3d_screen_gate_fail_closed/summary.{json,md}
outputs/lewm_action_branch_a3/a3c_confirmation_gate/summary.{json,md}
outputs/lewm_action_branch_a4/formal_v2/preflight_gate/summary.{json,md}
outputs/lewm_action_branch_a4/development_gate/summary.{json,md}
```

## G1A matched-history/native-label selector v3（2026-07-15，post-hoc development）

Pick `10 deg` cache 包含 `1,050` snapshots、每 snapshot `M=16` 候选，按完整 physical episode 划分为 `18 train / 6 validation / 6 test`。主历史使用 canonical native-FCPG corrector（SHA-256 `3cfe621bbfb5b87962a3c569d6ebf9d4d702e0483adeebd45a4e00bb0a16ee79`）在 `t-15,t-10,t-5` 构造，再由 oracle-free `10k` LeWM（SHA-256 `49f252eb526a61aad0bf29b9b3400b1b2bbfa2b7bb83e35164d716f9315435a9`）使用截至 `t-1` 的已执行动作预测当前 summary prior。Candidate action-output label 的 policy history 同样固定为 canonical native-FCPG `t-1`，与主历史一致；wide top-1 和 nominal 历史只作 deployment-time 诊断。

Deployment NPZ 仅包含当前候选特征、历史 grounding tokens、已执行动作和 predicted prior；offline NPZ 仅保存 candidate action-output MSE 与 shuffle controls。`q_true` 和 oracle action 只在构建期用于生成 teacher label，不持久化为 selector 输入。Shuffled action 来源限定为同一 split、同一 task/frame、不同 physical episode，并同步重算相应 prior，从而保持 split-local。

```text
outputs/g1a_selector_cache/pick_rot10_r64g17_l16m16_online10k_matched_history_native_labels_splitlocal_v3_development/
deployment_inputs.npz  sha256=07631a979eae1093b2abfcc7bf4424a5711a07fde46372210218a4196852961c
offline_labels.npz     sha256=f5cc9982d4220dd2fac3db2a3e901f650521df08fbb5b630152d62e54ac49977
metadata.json          sha256=2ff688db476c876031c69d4529d9fc854a606c77aeff8af25c5b1ce5fe06ebf4
```

三个 selector 共分配 `15,377` 个参数，但激活 module-path 参数为 full-history `12,289`、current-only `8,865`、summary-predictive `10,113`，因此不声称严格 active-capacity matching。Seed 0 的 Pick development-test 点估计为：

| Mode | Spearman | normalized top-1 regret |
| --- | ---: | ---: |
| frozen visual | 0.169708 | 0.242007 |
| structured full-history | 0.560149 | 0.202956 |
| structured current-only | **0.577465** | **0.187723** |
| summary-predictive | 0.382390 | 0.250940 |
| full + split-local shuffled action/prior | 0.560822 | 0.197247 |

Development 区间以 physical episode 为 cluster 做 `10,000` 次 percentile bootstrap（`n=6`），正值支持比较中的第一个 mode：

- full-history vs current-only：Spearman `-0.017316 [-0.026195,-0.009510]`，regret improvement `-0.015233 [-0.029357,+0.000119]`；
- full-history vs summary-predictive：Spearman `+0.177759 [+0.120205,+0.232059]`，regret improvement `+0.047983 [+0.014747,+0.078081]`；
- current-only vs frozen visual：Spearman `+0.407757 [+0.354320,+0.482929]`，regret improvement `+0.054285 [-0.020206,+0.123428]`；
- factual full vs split-local combined shuffled action/prior：Spearman `-0.000672 [-0.002250,+0.000504]`，regret improvement `-0.005709 [-0.016749,+0.003574]`。

相同 physical split 上的优化 seed 1/2 得到 full-minus-current Spearman `-0.017843/-0.035794`、regret improvement `-0.013283/-0.067613`；这些只检查优化稳定性，不增加独立数据样本。Canonical report：

```text
outputs/g1a_selector/pick_rot10_matched_history_native_labels_splitlocal_v3_final_seed0/report.json
sha256=e4a52fa01a0e8c773e8d012e7429411362b683abd51e7a033dc44a61ea78de32
```

**当前 development 证据支持 current structured candidate selector 的 Spearman 信号，但 seed-0 regret 改善尚未确认；full-history 在三个优化 seed 上均不优于 current-only，且 factual 与 split-local shuffled action/prior 不可分，因此 legacy summary-pooled `10k` LeWM 接口的 predictive increment 判为 No-Go。** 这些结果仍只是 Pick `n=6` 个 development physical test episodes 的小样本描述，不是独立 confirmatory evidence，也不构成 deployable selector、跨任务、posterior writeback、rollout 或 CEM claim；该 No-Go 不替代 Anchor-Persistent v1 的独立 2k 结果。

本轮最终验证为 `1128 passed, 43 warnings`；变更文件均通过 `py_compile` 与 `git diff --check`。

## Same-checkpoint action-grounding sensitivity（2026-07-15，完成 / 机制证据）

该诊断冻结 residual-only policy checkpoint，在相同 frame batch 上只替换 `nominal / FCPG / oracle` grounding。主量是 nominal 或 FCPG 动作输出到同一 checkpoint 的 oracle-grounded 动作输出的距离；`q_true` 只进入 oracle reference，deployable 两路不得读取。正式矩阵包含 `pick-place-v3 / button-press-v3`、policy seed `0/1/2`、每任务每 seed `60` 个完整 episode；每份 JSON 为 `6,000` frame rows。三锚点共同支持率为 `1.0`，OOB 为 `0`，H8 只在每个 episode 最后七帧按协议置空。

统计固定 task/seed/episode support，并在每个任务内将 policy seed 与跨 seed 共享 physical episode identity 作为 two-way crossed factors 独立重采样；task-macro 对任务等权。非零相机扰动帧上的结果为：

- EPE：`9.024843 -> 0.815753 px`，差值 `+8.209090 [6.867015,9.609012]`；
- action XYZ H1：`0.133855 -> 0.012827`，差值 `+0.121028 [0.096136,0.148757]`；
- action XYZ H8：`0.128709 -> 0.013028`，差值 `+0.115681 [0.093167,0.139608]`；
- gripper H1/H8 差值为 `+0.012540 [+0.008254,+0.017698]` 与 `+0.013238 [+0.008881,+0.018129]`。

Clean 时 nominal 投影即 `q_true`，FCPG 引入 `0.243789 px [0.232852,0.255136]` 校正噪声；对应 action XYZ H1/H8 的 oracle-output distance 为 `0.003231/0.003161`。六个 task-seed cell 在 drift 下均使 XYZ H1/H8 更接近 oracle-grounded 输出。

六个 `best.pt` 使用的同一 `20%` task-episode split 曾参与最低 validation loss 的 checkpoint 选择。因此该结果只能作为 **validation-set same-checkpoint mechanism evidence**：FCPG 的像素校正会在相机扰动下沿 oracle-grounded policy 的方向改变动作输出。它不是 fresh independent confirmation，也不证明 rollout success、轨迹控制、LeWM 或 CEM 收益。正式产物为：

```text
outputs/action_grounding_sensitivity/pick_seed{0,1,2}.json
outputs/action_grounding_sensitivity/button_seed{0,1,2}.json
outputs/action_grounding_sensitivity/seed012_summary.{json,md}
```

## Explicit-goal moving-target Gate（2026-07-22，development）

Reach/Push moving-goal 数据上的显式目标条件 direct model 已完成固定 `2,000` steps。Development 包含 `59` 个 same-state snapshot 和 `472` 条真实候选 branch；entity/robot-anchor EPE 分别从 `84.263/74.670 mm` 降至 `15.438/13.802 mm`，Spearman 达到 `0.7194`，top-1 为 `44.07%`。因此“current entity + moving goal + action prefix -> future entity/robot state”的动力学机制通过。

冻结模型后执行 `rotation={0,5,10} deg x blur_sigma={0,2,4}` observation-update 矩阵。六个 drifted cell 中，FCPG 将 current EPE 平均降低 `60.02%`，但 robot-anchor EPE、entity EPE、Spearman 和 selected H8 goal distance 的平均变化分别只有 `+0.3969 mm / -0.0098 mm / +0.000819 / +0.0976 mm`。因此当前接口不支持“FCPG 已提高 LeWM 动作选择”的 claim。

该消融刻意固定记录的真实 camera basis，只测试 FCPG 的当前观测更新；同时 direct model 读取精确 current entity、goal 和 world anchors，存在绕过视觉 grounding 的结构化通路。下一次联合 Gate 固定为：

1. 冻结当前 LeWM 与 FCPG，不继续调 `2k` endpoint。
2. 移除精确 current controlled-entity 输入，改为部署可得的 visual task anchor，或选用 FK 无法直接给出的接触/外观 task state。
3. 分别测试 calibration drift 与 intermittent blur/dropout：FCPG 只负责可见时的局部重定位；视觉内容失效时使用 LeWM prior；恢复后由 FCPG 重锚定。
4. 对比 nominal measurement、FCPG measurement、hold-last、LeWM prior、oracle measurement validity，并以 future-state error、action-ranking regret 和 rollout success 为主指标。
5. 只有去除结构化旁路后 FCPG 仍稳定改善冻结 LeWM，才形成“两项贡献联合”的证据。

当前矩阵为 development mechanism evidence，不是 rollout-success 或未知外参部署证据。产物位于 `outputs/lewm_goal_conditioned_direct_seed93001_2k/visual_robustness_rotation_blur/`。

上述下一次联合 Gate 已完成。Deployable v2 移除了 current entity truth 与 actual camera extrinsic，只保留 known goal、FK、历史/候选动作和当前视觉 LeWM state。固定 `2,000` steps 后，整体 entity EPE/Spearman 为 `20.18 mm/0.6953`；Push 为 `27.97 mm/0.5259`。在 `rotation={0,5,10} x blur={0,2,4}` 中，FCPG 对 drifted current EPE 的平均降低仍为 `60.02%`，但 entity EPE/Spearman/selected H8 的平均变化仅为 `-0.0341 mm/+0.001899/-0.4155 mm`。10 deg slot intervention 进一步确认小幅变化来自三个 anchor slot，而不是 summary slot，故校正信号已进入 state，但没有被当前 geometric goal head有效利用。

随后完成唯一一次 matched drift augmentation：nominal/FCPG 两路使用逐参数相同初始化、相同 physical rows、candidate actions、角度顺序和 `2,000` steps，训练角度为 `0/+-5/+-10 deg`，上游全部冻结。FCPG matched pipeline 在 drifted cells 的 entity EPE、Spearman、top-1 和 selected H8 相对 nominal matched pipeline 分别变化 `-0.2799 mm/+0.004850/+1.69 pp/+0.3407 mm`；Push 分别为 `-0.3897 mm/+0.008417/+0.3568 mm`。该结果仅有很小的排序信号，且实体误差反向，因此本接口冻结为 **development No-Go**，不再增加训练步数或扩大相同 head。

准确边界：这不否定 FCPG、Anchor-Persistent LeWM 或二者联合的一般可能性；它否定的是“known FK/goal/action + summary 主导的 Reach/Push geometric head 会因显式 FCPG state 自然获得显著增益”。后续联合实验必须使用 FK 无法直接给出的接触、物体外观或间歇观测状态，或沿已有 policy-centered Towel candidate ranking 继续做控制确认。

## Stage I 历史目标

验证 FCPG 是否能在可控几何不确定性下，先降低投影误差，再提升策略鲁棒性。

项目已先完成以下机制门槛，再进入 policy：

```text
Can local visual evidence correct projected robot keypoints?
```

## Stage I 历史执行快照（2026-07-10）

当前正式流程使用真实相机 episode 随机化：RGB 与 `q_true` 来自实际移动后的相机，`q_nominal` 来自固定名义标定；训练/验证按完整 task-episode 分组。

正式 pick-place 数据：30,000 帧、300 个完整 episode、224 x 224 图像、实际相机平移 0-4 cm、旋转 0-5 度、约 25% clean episode。policy 只使用 4 维 robot observation。

### Adapter 机制证据

ViT-B/16 layer-11 adapter 在 held-out episode 上取得：

```text
E_before = 6.275 px
E_after = 0.727 px
ratio = 0.116
```

视觉对应关系审计：

| evidence | E_after | mean rho |
| --- | ---: | ---: |
| normal | 0.727 px | 0.760 |
| local cutout | 6.805 px | 0.325 |
| zero image | 11.026 px | 0.018 |
| shuffled image | 15.202 px | 0.327 |

因此局部视觉证据已通过第一阶段残差校正验证；`rho` 能响应证据破坏，但是否改善策略必须由独立的匹配训练消融判断。

### Deterministic action-chunk 证据

主策略不是 Diffusion Policy。当前使用两帧 observation memory、32 步 action queries、2 层 encoder 和 6 层 decoder 的确定性 Transformer，batch size 1024。

3 个 policy seed 的离线结果（均值 +/- 样本标准差；adapter 固定为 seed 0）：

| policy | chunk MSE | first-action MSE |
| --- | ---: | ---: |
| GeoProp | 0.020590 +/- 0.000257 | 0.024000 +/- 0.000925 |
| FCPG | 0.017887 +/- 0.000153 | 0.017144 +/- 0.000583 |

固定 5 度实际相机旋转、每个 seed 60 个 episode 的同策略 grounding override 配对 rollout：

| seed | nominal | FCPG | delta | p |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0500 | 0.2167 | +0.1667 | 0.012939 |
| 1 | 0.0167 | 0.2833 | +0.2667 | 0.000145 |
| 2 | 0.0167 | 0.2167 | +0.2000 | 0.000488 |

跨 seed 的 paired delta 为 `+0.2111 +/- 0.0509`，层级 bootstrap 95% CI 为 `[0.1333, 0.2944]`。层级 bootstrap 先重采样 policy seed，再在 seed 内重采样配对 episode。

以下工作区间诊断只使用 policy seed 0：

| rotation | nominal | FCPG | nominal/corrected px | delta | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.3667 | 0.4333 | 0.000 / 0.687 | +0.0667 | 0.289062 |
| 5 deg | 0.0500 | 0.2167 | 18.410 / 1.781 | +0.1667 | 0.012939 |
| 10 deg | 0.0000 | 0.1000 | 37.259 / 14.671 | +0.1000 | 0.031250 |
| 20 deg | 0.0000 | 0.0000 | 75.887 / 56.173 | +0.0000 | 1.000000 |

当前结论：固定 5 度下，3 个 policy seed 都得到显著同策略收益；seed 0 的曲线显示 clean 无显著退化、10 度仍有部分恢复、20 度是明确失效边界。pick-place adapter seed 仍固定。

### 第二任务：button-press-v3

使用与 pick-place 相同的 300 episode / 30,000 帧协议。300/300 scripted episode 成功，全部投影有效，99.84% 关键点位于 24 px 窗口内。

Adapter 结果：

| setting | E_before | E_after | ratio |
| --- | ---: | ---: | ---: |
| pick adapter zero-shot transfer | 7.415 | 3.502 | 0.472 |
| button adapter held-out | 6.359 | 0.595 | 0.094 |
| button adapter full cache | 7.415 | 0.410 | 0.055 |

视觉证据破坏时，local-cutout / zero-image / shuffled-image 的 `rho` 分别降到 0.127 / 0.206 / 0.286，而正常图像为 0.788。

3 个 policy seed 的离线 action-chunk：

| policy | chunk MSE | first-action MSE |
| --- | ---: | ---: |
| GeoProp | 0.002356 +/- 0.000234 | 0.003952 +/- 0.000257 |
| FCPG | 0.001895 +/- 0.000086 | 0.002480 +/- 0.000134 |

固定 5 度、每个 seed 60 个 episode 的同策略配对 rollout：

| seed | nominal | FCPG | delta | p |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6667 | 1.0000 | +0.3333 | 0.000002 |
| 1 | 0.6000 | 0.9833 | +0.3833 | < 0.000001 |
| 2 | 0.7500 | 0.9667 | +0.2167 | 0.000977 |

跨 seed 的 nominal success 为 `0.6722 +/- 0.0752`，FCPG success 为 `0.9833 +/- 0.0167`；paired delta 为 `+0.3111 +/- 0.0855`，层级 bootstrap 95% CI 为 `[0.2056, 0.4111]`。

以下工作区间诊断固定使用 policy seed 0：

| rotation | nominal | FCPG | nominal/corrected px | delta | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 1.0000 | 1.0000 | 0.000 / 0.498 | +0.0000 | 1.000000 |
| 5 deg | 0.6667 | 1.0000 | 18.587 / 1.281 | +0.3333 | 0.000002 |
| 10 deg | 0.6167 | 0.6333 | 37.420 / 13.758 | +0.0167 | 1.000000 |
| 20 deg | 0.5833 | 0.7000 | 76.209 / 54.004 | +0.1167 | 0.118469 |

独立训练 policy 的 5 度比较为 GeoProp 0.7833、FCPG 1.0000，配对差值 +0.2167，McNemar p=0.000244。第二任务复现了中等扰动收益，但 10/20 度成功率增益不显著。

首轮 rollout 级遮挡消融固定使用 policy seed 0、5 度旋转和 12 px 关键点方形遮挡：

| setting | success | mean rho |
| --- | ---: | ---: |
| learned rho | 0.6833 | 0.180 |
| forced rho=1 | 0.6500 | 1.000 |
| forced rho=0 | 0.6333 | 0.000 |

learned `rho` 相对 `rho=1` 的 paired delta 为 `+0.0333`，bootstrap 95% CI `[-0.0333, 0.1000]`，McNemar `p=0.625`。结论是 reliability head 能检测遮挡，但当前门控的策略收益不显著。

匹配训练的 residual-only policy 固定 `rho=1`，其余数据、split、架构和训练设置与 full FCPG 相同：

| seed | residual-only chunk MSE | full FCPG chunk MSE | residual-only success | full FCPG success | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001966 | 0.001930 | 0.9667 | 1.0000 | +0.0333 |
| 1 | 0.001853 | 0.001958 | 0.9833 | 0.9833 | +0.0000 |
| 2 | 0.001682 | 0.001797 | 0.9833 | 0.9667 | -0.0167 |

跨 seed 的 residual-only / full FCPG success 为 `0.9778 +/- 0.0096` / `0.9833 +/- 0.0167`，配对差值 `+0.0056 +/- 0.0255`，层级 bootstrap 95% CI `[-0.0278, 0.0389]`。离线 chunk MSE 则是 residual-only 略优：`0.001833 +/- 0.000143` vs `0.001895 +/- 0.000086`。因此当前主性能机制确定为残差校正；`rho` 降级为诊断和可选消融。

### 结构强基线：Anchor-conditioned Absolute-Keypoint 与 FK-DeformAttn

Absolute-Keypoint 使用与 residual corrector 相同的四任务数据、冻结 ViT-B layer-11 cache、局部 `5 x 5` 视觉采样、anchor identity/FK 条件和精确相同的 `8,357,658` 个 trainable parameters，只把输出从 `q_nominal + delta_q` 改为直接回归绝对二维关键点。seed-0、task-episode `80/20` exposed validation 上，absolute head 的整体 EPE 为 `0.628209 px`，pick/button/reach/push 分别为 `0.638424 / 0.579687 / 0.692380 / 0.602347 px`。同 split 的 residual FCPG 为 `0.552868 px`，逐任务为 `0.563350 / 0.517530 / 0.602579 / 0.528013 px`。Absolute head 相对 residual FCPG 的 EPE 高 `13.63%`；等价地，residual FCPG 相对 Absolute head 低 `11.99%`。该结果只是一枚 seed-0 development controlled comparison，没有 paired frame CI 或 rollout，且 absolute head 仍读取 anchor identity/FK context，不能称为全局关键点检测基线。

FK-DeformAttn 固定相同 FK reference points、ViT-B cache、common policy、四任务混合数据、40 epochs、effective batch `2048` 和 BC-only loss；offset/weight 只由 policy action loss 学习，禁止使用 `q_true`、坐标 residual、reprojection、EPE、latent loss 或 SIGReg。residual 与 deform 两路 active parameters 为 `17,394,222 / 17,397,820`，差距 `0.0207%`。当前正式 seed-0 matched development 训练正在 GPU0 执行；在 checkpoint 和完整 paired rollout matrix 生成前不记录胜负结论。

### 下一轮优先级

1. 用独立 adapter seeds 重复两个任务的关键实验；
2. 补 oracle、vision-only 和 no-image correction 诊断；
3. 汇总两个任务的旋转曲线 AUC，并补平移扰动曲线；
4. 将遮挡半径曲线保留为 reliability 诊断，不作为主贡献门槛；
5. 最后扩第二基准或真机。

在完成这些证据前，不加入 primitive、interaction mode、world model、DDPM，也不继续扩大 reliability head。

关键产物：

```text
outputs/transformer_residual/pick300_224_vitb_l11_r24_b1024_relmix_ft_seed0
outputs/action_chunk/pick300_224_robot_geoprop_vitb_l11_t256_e2d6_ah32_b1024_seed0
outputs/action_chunk/pick300_224_robot_fcpg_vitb_l11_t256_e2d6_ah32_b1024_seed0
outputs/action_chunk_rollout/pick300_224_fcpg_rot5_s8_60eps_paired
outputs/action_chunk_rollout/pick300_224_fcpg_seed1_rot5_s8_60eps_paired
outputs/action_chunk_rollout/pick300_224_fcpg_seed2_rot5_s8_60eps_paired
outputs/action_chunk/pick300_224_robot_multiseed_rot5_summary
outputs/action_chunk_rollout/pick300_224_fcpg_rot10_s8_60eps_paired
outputs/action_chunk_rollout/pick300_224_fcpg_rot20_s8_60eps_paired
data/metaworld_projection/scripted_button_actualcam_se3_clean25_300eps_224.npz
outputs/transformer_residual/button300_224_vitb_l11_r24_b1024_relmix_ft_seed0
outputs/action_chunk/button300_224_robot_geoprop_vitb_l11_t256_e2d6_ah32_b1024_seed0
outputs/action_chunk/button300_224_robot_fcpg_vitb_l11_t256_e2d6_ah32_b1024_seed0
outputs/action_chunk_rollout/button300_224_fcpg_seed0_clean_s8_60eps_paired
outputs/action_chunk_rollout/button300_224_fcpg_seed0_rot5_s8_60eps_paired
outputs/action_chunk_rollout/button300_224_fcpg_seed0_rot10_s8_60eps_paired
outputs/action_chunk_rollout/button300_224_fcpg_seed0_rot20_s8_60eps_paired
outputs/action_chunk_rollout/button300_224_geoprop_vs_fcpg_seed0_rot5_s8_60eps_paired
outputs/action_chunk/button300_224_robot_multiseed_rot5_summary
outputs/action_chunk_rollout/button300_224_fcpg_seed0_rot5_occ12_s8_60eps_paired
outputs/action_chunk/button300_224_robot_fcpg_residualonly_vitb_l11_t256_e2d6_ah32_b1024_seed{0,1,2}
outputs/action_chunk/button300_224_full_vs_residualonly_multiseed_rot5_summary
```

## 阶段 0：Projection / Coordinate Sanity Check

目标：确认投影、图像坐标、feature map 坐标和 `grid_sample` 坐标完全一致。

输入：

- 机器人本体状态；
- 关键点定义；
- 真实相机内外参；
- 扰动后的外参；
- 图像 resize / crop 参数。

输出：

- `q_true` 可视化；
- `q_nominal` 可视化；
- 外参扰动强度 vs 投影误差；
- 坐标转换单元测试；
- `E_before = mean ||q_nominal - q_true||`。

Go / No-Go：

- `q_true` 必须落在正确机器人关键点附近；
- 外参扰动后 `q_nominal` 偏移方向和幅度必须可解释；
- 坐标转换不能依赖肉眼猜测。

## 阶段 1：Synthetic Heatmap Residual Sanity Check

目标：先排除采样、坐标和网络 bug。

做法：

```text
在 q_true 位置生成 Gaussian heatmap
以 q_nominal 为中心采样局部 patch
训练 residual head 预测 delta_q = q_true - q_nominal
```

指标：

```text
E_after / E_before < 0.2
```

如果 synthetic heatmap 都学不会，先修坐标和采样，不进入真实图像实验。

## 阶段 2：Policy-Free Residual Correction

目标：在真实仿真图像或仿真渲染特征上验证 residual correction。

当前数据入口：

```bash
/home/james/lewm/.venv/bin/python3 scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3,push-v3 \
  --episodes-per-task 1 \
  --steps-per-episode 5 \
  --perturbations-per-frame 2 \
  --output data/metaworld_projection/smoke_reach_push.npz
```

该脚本保存 `rgb / obs / action / keypoints_xyz / q_true / q_nominal / camera_intrinsic / true_extrinsic / nominal_extrinsic / pixel_error / q_true_inside_patch`，用于训练 local-image residual predictor 和 no-image residual predictor。

Stage 2 multi-direction residual benchmark:

```bash
/home/james/lewm/.venv/bin/python3 scripts/run_residual_benchmark.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --directions x-,y- \
  --seeds 0,1 \
  --epochs 15 \
  --batch-size 128 \
  --output-dir outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv
```

Stage 2 held-out magnitude residual benchmark:

```bash
/home/james/lewm/.venv/bin/python3 scripts/run_residual_benchmark.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --directions '' \
  --translations-cm 4 \
  --yaws-deg 10 \
  --seeds 0,1 \
  --epochs 15 \
  --batch-size 128 \
  --output-dir outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg
```

Stage 2 perturbation severity curves:

```bash
/home/james/lewm/.venv/bin/python3 scripts/evaluate_residual_curve.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --group-by translation_cm \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_no-image/no-image_residual_best.pt \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_1_no-image/no-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_1_local-image/local-image_residual_best.pt \
  --output-dir outputs/residual_curves/translation_2seed \
  --title "Translation severity residual correction"

/home/james/lewm/.venv/bin/python3 scripts/evaluate_residual_curve.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --group-by yaw_deg \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_0_no-image/no-image_residual_best.pt \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_1_no-image/no-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_0_local-image/local-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_1_local-image/local-image_residual_best.pt \
  --output-dir outputs/residual_curves/yaw_2seed \
  --title "Yaw severity residual correction"
```

Generated severity figures:

```text
outputs/residual_curves/translation_2seed/translation_cm_curve.png
outputs/residual_curves/yaw_2seed/yaw_deg_curve.png
```

Stage 2 residual benchmark figures:

```bash
/home/james/lewm/.venv/bin/python3 scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv/summary.csv \
  --output-dir outputs/residual_figures/direction \
  --title "Held-out direction residual correction"

/home/james/lewm/.venv/bin/python3 scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/summary.csv \
  --output-dir outputs/residual_figures/magnitude \
  --title "Held-out magnitude residual correction"

/home/james/lewm/.venv/bin/python3 scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv/summary.csv \
  --summary outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/summary.csv \
  --labels direction,magnitude \
  --output-dir outputs/residual_figures/combined \
  --title "Stage 2 residual correction sanity benchmark"
```

Generated outputs:

```text
outputs/residual_figures/direction/ratio_bar.png
outputs/residual_figures/magnitude/ratio_bar.png
outputs/residual_figures/combined/ratio_bar.png
```

Stage 2 residual predictor smoke training:

```bash
/home/james/lewm/.venv/bin/python3 scripts/train_residual_predictor.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --model no-image \
  --epochs 15 \
  --batch-size 128 \
  --heldout-directions y- \
  --output-dir outputs/residual_predictor/stage2_smoke_no_image_best

/home/james/lewm/.venv/bin/python3 scripts/train_residual_predictor.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --model local-image \
  --epochs 15 \
  --batch-size 128 \
  --heldout-directions y- \
  --output-dir outputs/residual_predictor/stage2_smoke_local_image_best

/home/james/lewm/.venv/bin/python3 scripts/compare_residual_metrics.py \
  outputs/residual_predictor/stage2_smoke_no_image_best/no-image_metrics.json \
  outputs/residual_predictor/stage2_smoke_local_image_best/local-image_metrics.json
```

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

Current Stage 2 figure summaries:

| figure | split | no-image ratio | local-image ratio |
| --- | --- | ---: | ---: |
| direction | x- | 1.1023 | 0.7229 |
| direction | y- | 0.8334 | 0.3188 |
| magnitude | translation_cm=4 | 0.9976 | 0.4309 |
| magnitude | yaw_deg=10 | 0.8380 | 0.2397 |

These figures are sanity-check plots for the mechanism stage. The current repository now also includes perturbation severity curves over translation and yaw levels.

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

### Stage 2 qualitative visualization

定量曲线之后必须保留定性图，直接显示：

```text
red: q_nominal
blue: q_corrected
green: q_true
arrow: q_nominal -> q_corrected
```

当前命令：

```bash
/home/james/lewm/.venv/bin/python3 scripts/visualize_residual_corrections.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --selection best \
  --count 12 \
  --columns 4 \
  --show-patch \
  --output-dir outputs/residual_visualizations/translation4_local_image
```

当前产物：

```text
outputs/residual_visualizations/translation4_local_image/best_grid.png
outputs/residual_visualizations/translation4_local_image/worst_grid.png
```

这组图用于论文方法图/定性分析候选，不替代 benchmark 曲线。


输入：

- 图像或视觉特征 `F`；
- `q_nominal`；
- keypoint id embedding；
- proprio keypoint embedding。

输出：

```text
delta_q
q_corrected = q_nominal + delta_q
```

指标：

```text
E_before = mean ||q_nominal - q_true||
E_after  = mean ||q_corrected - q_true||
```

Go / No-Go：

```text
E_after < E_before
E_after / E_before <= 0.7
```

该指标必须在 held-out 外参扰动上成立，才继续接入策略网络。

为防止 residual head 记住扰动分布，必须加入诊断对照：

```text
No-image residual predictor: q_nominal + keypoint id + proprio -> delta_q
Local-image residual predictor: LocalSample(F, q_nominal) + keypoint id + proprio -> delta_q
```

只有 local-image residual 在 held-out perturbation 上明显优于 no-image residual，才能说明模型确实利用了局部视觉证据。

held-out perturbation 至少包括：

- held-out translation direction；
- held-out rotation magnitude；
- mixed translation + rotation；
- held-out object pose 或 background。

还需要报告 patch 覆盖范围：

```text
nominal projection error in pixels
q_true inside local patch or not
E_after / E_before
```

FCPG 不解决任意相机位姿错误；它要求 `q_nominal` 仍在局部窗口可恢复范围内。

## 阶段 3：Reliability Validation

目标：验证 `rho` 是有意义的可靠性，而不是普通 attention gate。

第一版先比较两个 target：

```text
rho_nom_target = visible_i * exp(-||q_i^true - q_i^nominal|| / sigma)
rho_corr_target = visible_i * exp(-stopgrad(||q_i^true - q_i^corrected||) / sigma)
```

`rho_nom_target` 用于诊断名义投影可靠性，`rho_corr_target` 更符合最终 corrected token gating 语义。

需要统计：

- visible keypoints 的 `rho` 应高于 occluded keypoints；
- small-error keypoints 的 `rho` 应高于 large-error keypoints；
- 遮挡场景下不可信关键点 `rho` 应下降。

如果 `rho` 全接近 0 或 1，不进入 policy 实验，先修 target、loss 或数据分布。

当前 Stage 3 诊断结果：

| setting | target | best corr(pred, -corrected error) | high-conf error | low-conf error | judgment |
| --- | --- | ---: | ---: | ---: | --- |
| random split, with residual prediction features | `rho_corr`, sigma=2 | 0.6222 | 0.4677 | 1.2854 | 可学，能区分高低 corrected error |
| random split, no residual prediction features | `rho_corr`, sigma=2 | 0.5471 | 0.4994 | 1.1757 | 可学，但弱于带 residual 预测特征 |
| held-out translation=4cm, with residual prediction features | `rho_corr`, sigma=2 | 0.0641 | 1.5566 | 1.7271 | 不通过 OOD magnitude 验证 |

解释：这是早期小数据诊断。后续 actual-camera 数据和 correspondence negatives 使 `rho` 能在 local-cutout、zero-image、shuffled-image 以及 rollout 遮挡下明显下降，但匹配训练的 3-seed policy 消融仍未发现显著性能增益。Stage 4 及论文主路径因此使用 `residual only`；`residual + reliability` 只保留为诊断消融。除非新的预注册遮挡实验给出稳定增益，否则不再扩大 reliability head 或把它写入核心贡献。

可复现命令：

```bash
/home/james/lewm/.venv/bin/python3 scripts/inspect_reliability_targets.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --sigma-px 0.5,1,2,4 \
  --output outputs/reliability_head/translation4_seed0/target_diagnostics.json

/home/james/lewm/.venv/bin/python3 scripts/train_reliability_head.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm '' \
  --heldout-yaws-deg '' \
  --heldout-directions '' \
  --val-fraction 0.2 \
  --epochs 20 \
  --batch-size 128 \
  --sigma-px 2.0 \
  --target-kind corr \
  --loss bce \
  --output-dir outputs/reliability_head/random_split_predfeat_sigma2
```

## 阶段 4：Policy Integration

只有阶段 2 和阶段 3 通过后，再接入策略。

当前实现入口：

```text
fcpg.modules.FCPGResidualOnly
fcpg.modules.GroundedTokenAdapter
```

GeoProp-lite 与 FCPG residual-only 共用 token adapter：

```text
GeoProp-lite:          tokens = GroundedTokenAdapter(F, q_nominal, proprio_embedding)
FCPG residual-only:    q_corrected = q_nominal + delta_q
                       tokens = GroundedTokenAdapter(F, q_corrected, proprio_embedding)
FCPG + reliability:    tokens = GroundedTokenAdapter(F, q_corrected, proprio_embedding, rho)
```

当前 smoke 已通过：

```bash
/home/james/lewm/.venv/bin/python3 scripts/smoke_fcpg_modules.py \
  --checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt
```

验证内容包括 feature-map 点采样、grounded token 生成、shape、反传、`q_corrected = q_nominal + delta_q`、以及旧 residual checkpoint 加载兼容性。后续 policy 代码应导入这些模块，不要从训练脚本复制 `LocalImageResidualNet`。

策略数据入口：

注意：BC / policy 数据必须是 pre-action 对齐，`.npz` 中应包含 `observation_action_alignment=pre_action`。旧 `scripted_smoke_*` 数据是采集器修正前的历史 plumbing 数据，不作为策略效果证据。

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

当前有效 smoke：

```text
scripted_reach_preact_smoke.npz: 8 samples, reach-v3, observation_action_alignment=pre_action, success_mean=0.0000
```

历史 `scripted_smoke_reach_push.npz` 和 `scripted_smoke_reach_long.npz` 是采集器修正前的 plumbing 数据，可能存在 observation-action off-by-one，不再作为 BC / policy 证据。真实 BC 数据需要更长 horizon 和更多 episode。随机 action 数据只用于 residual 机制验证，不能作为策略实验主数据。

当前 minimal BC smoke：

```bash
/home/james/lewm/.venv/bin/python3 scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model state \
  --epochs 30 \
  --batch-size 64 \
  --output-dir outputs/bc_policy/reach_preact10_state

/home/james/lewm/.venv/bin/python3 scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model geoprop \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_geoprop

/home/james/lewm/.venv/bin/python3 scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model fcpg \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_fcpg

/home/james/lewm/.venv/bin/python3 scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model oracle \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_oracle
```

Current pre-action smoke result:

| run | model | best epoch | best MSE | best MAE | final MSE | final MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reach_preact10_state | state | 27 | 0.000547 | 0.008896 | 0.000551 | 0.008888 |
| reach_preact10_geoprop | geoprop | 28 | 0.000208 | 0.007088 | 0.000217 | 0.007054 |
| reach_preact10_oracle | oracle | 28 | 0.000208 | 0.007088 | 0.000217 | 0.007054 |
| reach_preact10_fcpg | fcpg | 30 | 0.000113 | 0.005628 | 0.000113 | 0.005628 |
| reach_preact10_fcpg_cleanw10 | fcpg | 30 | 0.000077 | 0.005038 | 0.000077 | 0.005038 |

This table verifies supervised action-regression plumbing on `observation_action_alignment=pre_action` data. It is still not a rollout robustness result; the clean-weighted residual checkpoint is the safer FCPG default because it reduces clean over-correction.

ViT adapter is now the preferred Stage 4 policy head for serious smoke tests. It follows the LeWM-style token mixer but keeps the FCPG problem scope unchanged:

```text
policy_arch=vit:
[CLS, obs, image patches, grounded keypoint tokens] -> Transformer -> action

loss:
MSE(action_pred, action_target) + lambda_sig * SIGReg(latent_tokens)
```

Current CUDA0 ViT+SIGReg smoke on the same `scripted_reach_preact_clean_10eps.npz` dataset:

| run | model | best MSE | best MAE |
| --- | --- | ---: | ---: |
| state_vit_sig1e4_t192_l6 | state | 0.000034 | 0.003784 |
| geoprop_vit_sig1e4_t192_l6 | geoprop | 0.000030 | 0.003478 |
| oracle_vit_sig1e4_t192_l6 | oracle | 0.000030 | 0.003478 |
| fcpg_large_h512_vit_sig1e4_t192_l6 | fcpg | 0.000023 | 0.002859 |

Rollout smoke is not yet stable enough for a policy claim, but the 4cm perturbation geometry diagnostic is positive:

| model | perturbation | success | corrected px err |
| --- | --- | ---: | ---: |
| geoprop_vit | 4cm x+ | 0.0000 | 3.7486 |
| fcpg_vit | 4cm x+ | 0.4000 | 1.3519 |
| oracle_vit | 4cm x+ | 0.0000 | 0.0000 |

Interpretation: use this only as a smoke result. The next real experiment should scale demos and run severity curves with more seeds. The stable claim so far is geometric correction, not rollout success.


Multi-task data variety update:

```text
data: scripted_4task_varyidx_clean_10eps.npz
samples: 4000
tasks: reach-v3, pick-place-v3, push-v3, button-press-v3
task_index coverage: 0-9
success_mean: 0.4690
```

Average BC metrics on this multi-task dataset:

| model | best MSE | best MAE |
| --- | ---: | ---: |
| state-ViT | 0.005135 | 0.037190 |
| GeoProp-ViT | 0.004152 | 0.034000 |
| FCPG-cleanw10-ViT | 0.008167 | 0.047012 |
| FCPG-cleanw50-ViT | 0.009179 | 0.041598 |

Per-task offline MSE:

| model | overall | button-press | pick-place | push | reach |
| --- | ---: | ---: | ---: | ---: | ---: |
| state | 0.005398 | 0.007154 | 0.006115 | 0.006755 | 0.001583 |
| geoprop | 0.004312 | 0.005620 | 0.004860 | 0.005134 | 0.001662 |
| fcpg_cleanw10 | 0.008617 | 0.008158 | 0.011932 | 0.012286 | 0.002042 |
| fcpg_cleanw50 | 0.009500 | 0.018931 | 0.010509 | 0.008078 | 0.001073 |

Decision: data variety is now the main axis. Larger ViT/batch did not improve the single-task result, while multi-task data exposed a real FCPG failure mode. The next experiments should improve residual identity/generalization under multi-task data before claiming policy-level robustness.


Current rollout smoke:

```bash
/home/james/lewm/.venv/bin/python3 scripts/evaluate_bc_rollout.py \
  --checkpoint outputs/bc_policy/reach_preact10_state/state_bc_best.pt \
  --task reach-v3 \
  --episodes 5 \
  --horizon 100 \
  --translations-cm 0 \
  --directions x+ \
  --yaws-deg 0 \
  --device cpu \
  --output-dir outputs/bc_rollout/reach_preact10_state_clean

/home/james/lewm/.venv/bin/python3 scripts/evaluate_bc_rollout.py \
  --checkpoint outputs/bc_policy/reach_preact10_fcpg/fcpg_bc_best.pt \
  --task reach-v3 \
  --episodes 5 \
  --horizon 100 \
  --translations-cm 4 \
  --directions x+ \
  --yaws-deg 0 \
  --device cpu \
  --output-dir outputs/bc_rollout/reach_preact10_fcpg_trans4
```

Current rollout result:

| run | model | success | reward mean | nominal px err | corrected px err |
| --- | --- | ---: | ---: | ---: | ---: |
| state clean | state | 0.0000 | 695.9383 | 0.0000 | 0.0000 |
| geoprop clean | geoprop | 0.0000 | 559.8126 | 0.0000 | 0.0000 |
| oracle clean | oracle | 0.0000 | 561.7698 | 0.0000 | 0.0000 |
| fcpg clean | fcpg | 0.0000 | 584.4045 | 0.0000 | 1.3739 |
| fcpg_cleanw10 clean | fcpg | 0.2000 | 592.9354 | 0.0000 | 0.8688 |
| state trans=4cm | state | 0.0000 | 653.3738 | 3.6992 | 3.6992 |
| geoprop trans=4cm | geoprop | 0.0000 | 590.9687 | 3.6963 | 3.6963 |
| oracle trans=4cm | oracle | 0.2000 | 643.7717 | 3.6929 | 0.0000 |
| fcpg trans=4cm | fcpg | 0.0000 | 665.9442 | 3.6838 | 1.2134 |
| fcpg_cleanw10 trans=4cm | fcpg | 0.0000 | 578.4313 | 3.6839 | 1.2382 |

This table is a diagnostic, not a policy robustness result. It shows residual correction works under 4cm perturbation, and `clean_loss_weight=10` reduces clean over-correction from 1.37 px to 0.87 px while keeping 4cm correction near 1.24 px. Next policy step: improve the policy backbone/data scale before making success-rate claims.

策略骨架：

- 当前主线：deterministic action-chunk Transformer，`obs_horizon=2`、`action_horizon=32`；
- GeoProp 与 FCPG 必须使用同一架构和 split；
- DDPM 不进入主实验，ACT 只在后续跨骨架验证时考虑。

任务选择先收窄：

- 主任务：pick-and-place 或 reaching-to-grasp；
- 辅助任务：push 或 button pressing。

第一版不做：

- 插入类任务；
- 抽屉或门把手；
- 小物体精细操作；
- 高遮挡密集接触任务。

## 阶段 5：Robustness Benchmark

扰动类型：

### 1. 标定漂移

测试时加入合成相机外参扰动：

- 平移漂移：0 cm、0.5 cm、1 cm、2 cm、4 cm；
- 旋转漂移：0 度、1 度、2 度、5 度、10 度。

指标：

```text
扰动强度 vs success rate
扰动强度 vs projection error
```

### 2. 遮挡

加入合成遮挡或场景遮挡：

- 遮挡夹爪；
- 遮挡指尖；
- robot link occlusion；
- 在投影关键点附近加入随机矩形遮挡。

指标：

```text
遮挡程度 vs success rate
visible vs occluded keypoints 的 rho 分布
```

### 3. 空间泛化

训练时限制物体位置范围，测试时扩大物体空间分布。

指标：

```text
位置偏移距离 vs success rate
```

## Baselines

必做：

1. `State concat`
2. `GeoProp-lite single-keypoint`
3. `GeoProp-lite multi-keypoint`
4. `GeoProp-lite multi-keypoint + projection jitter augmentation`
5. `FCPG residual only`
6. `FCPG reliability only`
7. `FCPG full`
8. `Oracle corrected GeoProp`
9. `No-image residual predictor`

定义：

- `FCPG residual only`：修正 `q_nominal`，但不做可靠性门控；
- `FCPG reliability only`：不修正 `q_nominal`，只对 nominal grounded token 做可靠性门控；
- `FCPG full`：同时使用残差修正和可靠性门控；
- `Oracle corrected GeoProp`：直接用 `q_true` 采样，用于估计 grounding 上限；
- `No-image residual predictor`：不看局部视觉 patch，用于检查 residual head 是否记扰动分布。

公平性要求：

```text
FCPG residual only 必须至少和 multi-keypoint open-loop GeoProp-lite
以及 matched projection-jitter augmentation 对比。
```

不能只做多关键点 FCPG vs single-keypoint GeoProp。full FCPG 作为 reliability 消融报告，不预设其优于 residual-only。

## 图表

最少需要：

1. 开环接地 vs FCPG 结构图。
2. `q_nominal / q_corrected / q_true` 可视化。
3. `E_before / E_after` 定量柱状图或曲线。
4. 标定漂移下的成功率衰减曲线。
5. 遮挡下的 `rho` 分布图。
6. 消融表。
7. patch radius / projection error / correction ratio 分析图。
8. perturbation curve 的 robustness AUC 和 confidence interval。

## 初始 Go / No-Go

只有满足以下条件才继续扩展：

- synthetic heatmap：`E_after / E_before < 0.2`；
- 真实仿真图像：`E_after / E_before <= 0.7`；
- `rho` 若保留，必须能区分 visible / occluded 或 small-error / large-error，但这只是诊断要求；
- 正常场景 policy 不明显掉点；
- 中等标定漂移下，FCPG residual-only 比 multi-keypoint GeoProp-lite 至少提升 5 个百分点；
- local-image residual 明显优于 no-image residual；
- robustness AUC 优于 multi-keypoint GeoProp-lite + jitter。

如果这些条件不满足，先回头修坐标、采样、label 和可靠性定义，不要急着加 primitive、interaction mode 或更大的模型。

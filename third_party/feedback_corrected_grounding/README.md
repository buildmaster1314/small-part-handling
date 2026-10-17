# FCPG：基于轻量预测 Transformer 的反馈校正本体接地

From open-loop geometric projection to feedback-corrected proprioceptive grounding.

## 一句话定位

当前论文已锁定 **Stage I FCPG** 主贡献：以名义运动学投影为几何锚点，用针对本项目改造的 LeWM 条件 Transformer 预测二维残差 `delta_q`，修正策略读取视觉特征的位置。Action-Branch A4 已完成七任务 active-control 数据 Gate、三个 fresh `2,000`-step development seed 和三个 fixed-latest `10,000`-step endpoint；旧 fixed-panel collection 曾因覆盖不足 fail closed，但随后使用 fresh identity namespace 建立的 A4b panel 已正式判为 **Confirmation-Go**。A4b 覆盖 `84` 个 test physical episodes，并产生 `325` 个 accepted bundles；H6-H8 primary 上，factual action 相对 matched no-action 的逐任务相对改善等权平均为 `65.63%`，`7/7` 任务、`3/3` seed 同向，且胜全部七种 wrong-action shifts、hold-last 与 latent CV。冻结的 P1-native fixed-gain complementary observer 随后在 `30` 个 fresh Pick episode 上通过 known-drop token-prediction Gate（`3/3` seed）；再进行的 `30` 个配对 policy-rollout episode 上，三个 LeWM seed 相对 exact-FK geometry baseline 的二值成功率变化仅为 `0 / +6.67 / 0 pp`，physical-episode clustered seed-macro 95% CI 为 `[-10.00,+14.44] pp`；因此当前准确状态为 **action-consistency/reward mechanism positive，policy-success Development-No-Go**，不启动大样本确认、不形成第二贡献。A3 的四任务 preliminary result 继续保留，但其 `194/322 = 60.2%` post-success hazard 缺陷不再用于 A4b。KineParity 与旧 full grounded LeWM 接口的 No-Go 均保持有效；LeWM 本身、SIGReg 和 Transformer 也不是本文 novelty。

## 统一研究视角与证据边界

更完整的研究问题是：在**非平稳相机几何**与**间歇视觉观测**共同作用下，持续维护一组具有固定机器人部位身份的 grounded anchors。下面的 predict-measure-update 分解只用于统一接口和组织证据，不是新的滤波原理，也不表示整条链路已经成立：

```text
m_t       = O_FCPG(I_t, p_t^kin; q_t^nom)       # current visual measurement
b_t^-     = T_AP(b^+_{t-1}, a_{t-1})            # action-conditioned prior
b_t^+     = U(b_t^-, m_t; c_t^obs)              # trust-weighted update
policy_t  = pi(b_t^+, proprio_history)
```

- `O_FCPG`：以名义 FK 投影为锚点，用当前局部 RGB 预测二维 residual，并在校正位置重新读取策略特征；这是**已验证的 FCPG 主贡献**。
- `T_AP`：A3 已在四任务共享训练、三个 seed 和 reserved test 上确认 same-state action increment；A4 进一步加入持久 task-context state。三个冻结的 `10k` endpoint 已在 A4b fresh panel 上通过七任务确认 Gate：逐任务 factual-vs-no-action 相对改善的等权平均为 `65.63%`，全部七种 wrong-action shifts 的改善均超过 `83%`，`7/7` 任务、`3/3` seed 同向。它证明七任务 H6-H8 动作条件 transition 的预测增量，尚未证明部署式缺测切换、policy rollout 或控制收益。
- `U` 与 `c_t^obs`：自动 observation-trust / re-anchoring 仍未通过 deployable Gate；但在外部已知 `sensor-invalid` 的条件下，固定缺测年龄增益的 geometry + LeWM complementary update 已通过 fresh Pick token-prediction Gate。后续 paired policy rollout 显著改善动作一致性与平均累计回报，但二值成功率未显著改善；该结果既不提供可部署的 `c_t^obs`，也不构成控制成功率证据。
- 当前已验证策略直接消费 `m_t`，即接口上取 `b_t^+=m_t`。这里的 `belief/posterior` 只是确定性状态记账，不声称贝叶斯后验、校准不确定性或已验证的端到端 closed loop。

因此论文叙事可以统一为“FCPG measurement、Anchor-Persistent transition、observation trust、re-anchoring”，但贡献列表仍必须按实验 Gate 分层：`O_FCPG` 已有策略级证据，`T_AP` 已有 held-out action-branch 机制证据，`U` 仍未通过安全切换 Gate，完整公式不能写成已成立的端到端系统。

## SO-101 公共数据 LeWM 预训练支线

### SO-ARM101 双相机只读运行时

真实硬件主线使用 `/home/james/lerobot` 的独立 Conda 环境（Python `3.10`），与现有 LeWM 训练环境隔离。当前第一阶段严格为 shadow mode：读取 follower 六关节、wrist/column 两路 RGB 和可选 leader 六关节指令；采集器直接打开总线与相机，但没有调用 `send_action` 的代码路径，也不会向 follower 转发动作。follower/leader 固定为 `/dev/ttyACM1` 与 `/dev/ttyACM0`，wrist/column 固定为 `/dev/video0@60 FPS` 与 `/dev/video2@30 FPS`。

硬件接入后先检查端口、相机和官方校准文件：

```bash
PYTHONPATH=. .venv-lerobot/bin/python scripts/check_so101_lerobot_setup.py
```

完成官方 `lerobot-calibrate` 后，可采集配对双相机只读 session；不提供 `--leader-port` 时只记录 observation，不伪造 LeWM 的动作条件预测：

```bash
PYTHONPATH=. /home/james/miniconda3/envs/lerobot/bin/python \
  scripts/capture_so101_shadow_session.py \
  --port /dev/ttyACM1 --robot-id myfollower01 \
  --leader-port /dev/ttyACM0 --leader-id myleader01 \
  --camera-name wrist --camera-index 0 --primary-camera-fps 60 \
  --primary-camera-fourcc MJPG \
  --secondary-camera-name column --secondary-camera-index 2 \
  --secondary-camera-fps 30 --secondary-camera-fourcc YUYV \
  --output outputs/lerobot/so101/real_shadow_session
```

采集格式固定保存两路配对的 224 px RGB、LeRobot normalized 六关节状态、时间戳和可选 leader/policy action。两路相机必须共享完全相同的状态、时间戳和动作行。

```bash
PYTHONPATH=.:third_party/lewm CUDA_VISIBLE_DEVICES=0 \
  /home/james/lewm/.venv/bin/python3 scripts/render_so101_shadow_report.py \
  --session outputs/lerobot/so101/real_shadow_session \
  --checkpoint outputs/lerobot/so101/so101_state_residual_lewm_v4_seed5120_10k/latest.pt \
  --output outputs/lerobot/so101/real_shadow_report --device cuda:0
```

当前无硬件 dry-run 已使用 public `so101_fruit` episode 0 的前 `180` 帧通过完整采集合同。固定 `10k` checkpoint 的未来 `5` raw-frame 六关节 RMSE 为 `0.7074`（数据集原始关节单位；同一 session 的 step-3000 结果为 `0.8312`）；该页只用于验证真实接线前的数据顺序、动作条件、FK 三锚点和 HTML，不构成真实机器人结果。人工审查入口：

- [SO-101 单相机 shadow 四宫格 dry-run](outputs/lerobot/so101/shadow_dry_run_fruit_ep0_report/index.html)
- [SO-101 单相机 shadow 四宫格 dry-run（State-Residual 10k）](outputs/lerobot/so101/shadow_dry_run_fruit_ep0_report_10k/index.html)
- [SO-101 单相机 shadow 四宫格 dry-run（State-Residual 30k）](outputs/lerobot/so101/shadow_dry_run_fruit_ep0_report_30k/index.html)
- [SO-101 单相机 shadow 四宫格 dry-run（Temporal Residual 5M / 2k）](outputs/lerobot/so101/shadow_dry_run_fruit_ep0_report_temporal5m_2k/index.html)
- [SO-101 单相机 shadow 四宫格 dry-run（Temporal Residual 5M / 10k）](outputs/lerobot/so101/shadow_dry_run_fruit_ep0_report_temporal5m_10k/index.html)

真实机器人公共数据实验已从上游 LeWM 工作目录迁入本项目，目录固定为：

```text
data/lerobot/so101/raw/       # 原始 LeRobot SO-101 数据
data/lerobot/so101/cache/     # 224x224 LeWM 训练缓存
outputs/lerobot/so101/        # checkpoint、评估结果、HTML/MP4/GIF
scripts/prepare_so101_lewm.py
scripts/train_so101_lewm.py
scripts/train_so101_state_residual_lewm.py
scripts/visualize_so101_lewm.py
scripts/build_so101_synthetic_fcpg_dataset.py
scripts/train_so101_synthetic_fcpg.py
scripts/render_so101_synthetic_fcpg_dataset.py
```

新增公共数据的固定 revision、许可证、相机选择、独立 episode 规模、缓存命令与使用边界见
[SO-101 public-data expansion](docs/so101_public_data_expansion.md)。数据实际位于独立的
`so101-data` ext4 卷，仓库内 `data/lerobot/so101` 保持为透明软链接，因此现有训练命令无需改路径。

`data/` 与 `outputs/` 继续由 `.gitignore` 排除，不进入 Git。代码使用 `third_party/lewm` 中已固定的 LeWM 实现。当前共享模型覆盖 fruit、table-cleanup 与 drawing 的 `12` 个语义任务，参数量为 `18,034,678`，历史长度为 `3`，`frame_skip=5`，动作维度为 `30`，只使用 `SIGReg=0.02`，从随机初始化训练 `10,000` steps。

v3 全量 `10,001` 条 validation sequence 的 task-macro factual/copy/shuffled/no-action MSE 为 `0.012247 / 0.016252 / 0.027178 / 0.030792`。Factual 相对 copy 改善 `24.64%`，且 `12/12` 任务在打乱动作后均退化；但收益集中于 `5/5` drawing 任务，fruit 与 table-cleanup 的 `7` 个短时操作任务仍为 `0/7` 超过 copy。准确边界是：共享 LeWM 已学到动作条件 latent dynamics，但纯 RGB latent 尚未证明优于 hold-last 的短时操作状态预测。下一次有界改动应加入 SO-101 proprioception，并学习相对 copy 的动作条件 residual。

该有界改动已作为 State-Residual LeWM v4 完成：从固定 v3 `10k` checkpoint 初始化，显式输入 SO-101 的 `shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper` 六维测量状态，增加 `472,838` 参数的有界 residual branch，并以主干 `0.1x` 学习率训练；`SIGReg` 始终严格为 `0.02`。固定 `10k` endpoint 的全量验证包含 `10,001` 条 sequence：六维状态 MSE 为 `0.003538`，相对 hold-current 的 `0.042776` 降低 `91.73%`，相对 last-command 的 `0.025594` 降低 `86.18%`，并比 v4 step-2000 的 `0.005586` 再降低 `36.67%`。打乱 action 后状态误差升至 `0.705112`，打乱当前 state 后升至 `0.137420`，说明模型同时使用当前测量和真实动作，并未退化成静态复制。

Visual latent 的收益明显更小：同一 endpoint 的 prediction/base/copy MSE 为 `0.012262 / 0.012568 / 0.016149`，相对冻结 base 仅改善 `2.44%`，且较 step-2000 的 `0.012244` 轻微退化 `0.15%`。因此 `10k` 的准确结论是：**六维动作条件状态动力学继续显著增强，但通用 visual latent 已基本饱和**。

随后执行的独立 `30k` 训练从相同 v3 base、相同 seed/split 和零初始化 residual 重新开始，没有续接 `10k` 的已衰减 optimizer；除总预算与每 `2,000` step 保存外，模型、数据、损失和 `SIGReg=0.02` 均保持一致。全量终点的 state prediction/copy/last-command MSE 为 `0.003189 / 0.042776 / 0.025594`，相对 copy 与 last-command 分别降低 `92.54% / 87.54%`，并比独立 `10k` state MSE 再降低 `9.85%`。action-shuffle delta 从 `10k` 的 `0.701574` 增至 `0.767322`，说明长训练没有削弱动作依赖。同一 dry-run session 的六关节 RMSE 也从 `10k` 的 `0.7074` 降至 `0.5777`。

但 `30k` visual prediction MSE 为 `0.013211`，较独立 `10k` 的 `0.012262` 退化 `7.74%`；它相对自己已被继续微调的 base `0.014609` 仍改善 `9.57%`。因此不存在单一“全面最佳” checkpoint：**需要六维状态/FK 动力学时使用 `30k`，需要最低 visual-latent MSE 时保留 `10k`**。该结果仍是公共数据离线预测证据，真实 SO-ARM101 必须另行验证 shadow 与闭环控制误差。

为解决上述 state/visual trade-off，新增 State-Residual LeWM v5：完全冻结 v3 visual LeWM，只训练一个 `4,647,174` 参数的 causal temporal residual。它将原 v4 独立处理各 sampled step 的 `576 -> 384 -> 384` MLP，替换为 `d=512`、`8` heads、FFN `1024` 的两层 causal Transformer；输入仍为 base latent prediction、六维当前状态和 action embedding，输出仍为零初始化且有界的 visual/state residual。相同缓存、split、seed、batch 与 `SIGReg=0.02` 的固定 `2k` 全量验证中，state prediction MSE 为 `0.003594`，比冻结主干的 v4 `2k`（`0.005715`）降低 `37.11%`，已接近联合微调 v4 `10k` 的 `0.003538`。同时 visual prediction MSE 为 `0.012172`，优于联合微调 v4 `10k/30k` 的 `0.012262/0.013211`；action-shuffle delta 为 `0.807004`，说明更低误差没有来自忽略动作。

该 v5 `2k` 结果判为 **capacity pilot Go**：同一条 dry-run shadow session 的 joint RMSE 为 `0.8216`，仍弱于 v4 `10k/30k` 的 `0.7074/0.5777`。随后从相同 v3 base fresh 训练固定 `10k` endpoint，没有续接 `2k` optimizer，也没有选择中间 quick-validation checkpoint。全量验证的 state prediction/copy/last-command MSE 为 `0.003187 / 0.042776 / 0.025594`，action-shuffle 与 state-shuffle delta 为 `0.812780 / 0.078612`；visual prediction/base/copy MSE 为 `0.011923 / 0.012247 / 0.016252`。

v5 `10k` 的 state MSE 已与 v4 `30k` 的 `0.003189` 实质持平并略低，同时 visual MSE 相对 v4 `10k/30k` 分别降低 `2.76% / 9.75%`，消除了旧模型的宏平均 state/visual trade-off。相同 shadow session 的 joint RMSE 为 `0.5793`，也已接近 v4 `30k` 的 `0.5777`。因此 v5 `10k` 是当前 SO101 公共数据 shadow 评估的首选单一共享 checkpoint；但它并非逐任务普遍占优：state MSE 相对 v4 `30k` 为 `7/12` 任务更好，visual MSE 相对 v4 `10k` 同样为 `7/12` 更好。该结论仍限于公共数据离线预测，不能替代真实 SO-ARM101 的 shadow 与闭环控制验证。v5 checkpoint 使用独立的 `so101-state-residual-lewm-v5-temporal` 合同；v4 checkpoint 的架构与加载路径保持不变。

### 五源公共数据扩展预训练

为检验此前短任务误差是否主要受数据量限制，development pretraining 进一步加入 `700` episode 七任务源和 `125` episode 多任务源，并与既有 fruit、table-cleanup、drawing 合并。declutter 的 `51` episodes 在训练与 checkpoint 选择期间保持未使用，随后只用于一次零训练 external-source 评估。按 source-task 平衡后共有 `437,097` 条 train sequence、`48,187` 条 validation sequence 和 `30` 个 source-task group；状态与动作统一使用物理 `/100` 缩放，`SIGReg` 固定为 `0.02`。

从随机初始化训练的 expanded base 在 `2k` 时尚未收敛：prediction/copy MSE 为 `0.012415/0.011445`，action-shuffle delta 仅 `0.000246`。固定 `10k` 后，prediction/copy MSE 改善为 `0.007626/0.008839`，相对 copy 降低 `13.73%`，action-shuffle delta 增至 `0.001850`，且 `30/30` group 对 action shuffle 均为正。因此扩展数据本身有效，但纯 visual latent 仍只在 `13/30` group 超过 copy。

随后冻结 expanded base `10k`，只训练 `4,647,174` 参数的 temporal State-Residual 分支。独立 `2k` pilot 的 visual prediction/base/copy MSE 为 `0.007463/0.007779/0.008991`，state prediction/last-command/copy MSE 为 `0.000218/0.001436/0.002721`。该 pilot 通过后，从同一冻结 base fresh 训练固定 `10k`，而非续接 2k optimizer。全量终点结果为：

| Expanded five-source `10k` | Prediction | Baseline 1 | Baseline 2 | 逐任务方向 |
| --- | ---: | ---: | ---: | ---: |
| visual latent | `0.006500` | frozen base `0.007625` | copy `0.008839` | 胜 base `28/30`；胜 copy `20/30` |
| six-joint state | `0.000167` | last-command `0.001436` | hold-current `0.002721` | 两项均为 `30/30` |

visual latent 相对冻结 base 和 copy 分别降低 `14.76%` 与 `26.46%`；six-joint state 相对 last-command 和 hold-current 分别降低 `88.36%` 与 `93.86%`。visual action/state shuffle delta 为 `0.007915/0.004174`，state action/input shuffle delta 为 `0.080149/0.016140`，说明 residual 同时使用真实动作和当前六维状态。新增 `700` episode 源的七个任务为 `7/7` 同时胜 base 与 copy；fruit 和 table-cleanup 的短时静态片段仍是主要 visual-copy 边界。

该结果将 expanded temporal checkpoint 设为当前公共 SO-101 **development pretraining** 的首选模型，但不构成真实机器人 shadow 或闭环控制证据。`700` episode 源的双机械臂画面仍需逐任务动作对齐审计；公共数据也没有训练 FCPG 所需的相机标定和三锚点像素标签。完整配置、结果路径和复现实验命令见 [SO-101 public-data expansion](docs/so101_public_data_expansion.md)。

#### 零训练 declutter 外部源评估

冻结上述 `10k` checkpoint 后，在训练、验证和 checkpoint 选择均未使用的 `so101_declutter_v1` top-view 上评估全部 `51` 个 episode、`22,134` 条有效 sequence。six-joint state prediction MSE 为 `0.000507`，相对 last-command 的 `0.001192` 降低 `57.43%`，相对 hold-current 的 `0.000896` 降低 `43.39%`；action/state shuffle delta 为 `0.080314/0.011503`。只看 `18,364` 条 active-motion sequence，两个相对改善为 `61.48%/51.96%`。因此动作条件六维动力学在未见 source、camera 和 task 上仍有明确迁移，不是训练任务记忆或静态复制。

视觉结论相反：external visual prediction/base/copy MSE 为 `0.013712/0.012874/0.002336`，residual 相对 frozen base 退化 `6.51%`，且短时 copy 明显更强。其 action/state shuffle delta 仍为正，但不足以抵消新相机和新场景带来的表示偏移。准确边界是：当前 checkpoint 的六维状态动力学具备外部迁移证据，visual-latent residual 尚不具备跨源泛化证据；不能把前者外推成 FCPG、视觉生成或闭环控制收益。

人工页使用一条完整 `449` 帧 external episode：左侧冻结真实 top-view 观测，右侧将 checkpoint 直接预测的六关节状态送入官方 SO-101 MuJoCo 模型，逐轮播放 H1-H8 后再读取新观测。该 episode 的 EEF RMSE 为 `11.27 mm`，H1/H8 分别为 `9.01/11.48 mm`；页面不使用未来 RGB 或训练近邻，但因公共相机无标定，左右视图不声称像素对齐。

### SO-101 双相机 FCPG 合成机制 Gate

LeRobot 公共数据没有与本项目三锚点定义配套的标定参数和逐帧真实像素位置，不能直接监督 FCPG。为先验证接口，当前使用 SO-101 MuJoCo 建立独立的双相机合成 Gate：`200` 个 train 与 `50` 个 validation 物理状态分别从 front/side 两个视角渲染；每个视角包含一个 clean 和三个 drift variant，同一物理状态的两个相机共享相同的 rig perturbation。总计为 `1,600` 个 train 与 `400` 个 validation frame。相机扰动范围固定为方位角 `+/-5 deg`、俯仰角 `+/-2 deg` 和距离 `+/-1.5 cm`。

在冻结 SO-101 v4 LeWM、ViT 和 anchor bridge 的条件下，只训练共享的 `2,154,088` 参数 FCPG corrector `2,000` steps。该阶段不更新 LeWM，因此没有新的 SIGReg 超参数；冻结 LeWM checkpoint 仍来自 `SIGReg=0.02`。validation drift 上的三锚点 EPE 从 `12.1655 px` 降到 `2.3877 px`，相对改善 `80.37%`；front/side 分别从 `11.7060/12.4859 px` 降到 `2.4570/2.3394 px`。clean 条件下的平均 false correction 为 `0.9784 px`。在同一冻结 LeWM ViT feature map 上，校正位置相对真实位置的 anchor-token MSE 从 nominal 的 `0.165203` 降到 `0.015350`，改善 `90.71%`；cosine distance 从 `0.099687` 降到 `0.009705`，改善 `90.26%`。

双相机推理接口位于 `fcpg/so101_fcpg_lewm_bridge.py`。两个视角在一个 batched ViT forward 中编码，共用同一个 FCPG corrector 和 anchor bridge，再按固定物理 anchor identity 融合。新 bridge/fusion gate 均为零初始化，因此未训练时严格退化为原 primary-camera LeWM checkpoint；真实 v4 LeWM 与 FCPG `2k` 权重的 CPU integration smoke 也已完成。当前 fusion 本身**尚未训练**，上述结果只证明合成相机漂移下的像素校正与 feature re-indexing 有效，不证明双相机 fusion、未来 latent prediction、动作选择或真实机器人控制提高。

真实 SO-ARM101 上仍需单独训练或微调 FCPG。训练输入只使用部署时可获得的 RGB、六维关节状态、名义 FK 投影和 camera identity/context；监督标签可由离线高精度标定、临时 AprilTag/关键点标注或受控相机位姿扰动生成。标签设备只在采集阶段使用，部署时不需要。两台真实相机应共用 corrector，并保留独立 camera context；是否训练融合层必须由真实序列上的 nominal/FCPG/oracle 消融决定。

#### SO-101 本地双相机 real-image warp pilot

本地 `my_first_task_clean_v1` 的 wrist/column 双相机序列进一步用于一次离线接口验证。四条成功示教中，episode `0/1/3` 用于训练，episode `2` 不参与梯度更新，但用于 local-transfer LeWM checkpoint 选择以及 FCPG corrector/context bridge validation，因此属于 development validation，不是独立 test。每个真实 RGB 帧保留一份 clean 图像，并生成一份中心旋转加二维平移的 affine-drift 图像；三个 nominal anchor 与 RGB 使用完全相同的已知变换生成离线监督。该构造保留真实桌面、物体、遮挡和双相机外观，但相机漂移仍是二维合成扰动，不能替代真实相机移动实验。

本次冻结 LeWM 的确来自 `5M dynamics` 路线：`18,034,678` 参数 expanded five-source visual base 在 `10k` 后冻结，再加载 `4,647,174` 参数的 temporal State-Residual dynamics `10k` checkpoint，总参数量为 `22,681,852`。本地 transfer 只微调 dynamics（`base_lr_scale=0`）；固定评估点 `250/500/750/1000` 中，`step_000250.pt` 的 validation state/visual MSE 最低，因此 FCPG 与 bridge 都绑定该 checkpoint。该选择进一步限定了 episode `2` 只能作为 development validation。

冻结 local-transfer LeWM 的 ViT 后，共享的 `2,154,088` 参数 corrector 训练 `2,000` steps。留出 episode 上，drift EPE 从 `6.7751` 降至 `2.3274 px`，改善 `65.65%`；wrist/column 分别改善 `79.33% / 52.49%`。在同一冻结 feature map 上，校正位置相对 clean target 的三锚点 token MSE 改善 `67.76%`。clean false correction 为 `1.4750 px`，其中 wrist/column 为 `0.5081 / 2.4418 px`，说明 column 的近似标定与遮挡仍是主要泛化边界。

随后冻结 corrector 与 LeWM，只训练一个不读取 action 的 `448,918` 参数 anchor context bridge，使 drift frame 的部署式 context 接近同一物理时刻的 clean LeWM context。留出 episode 的 current-context MSE 从 `0.15323` 降至 `0.08946`，改善 `41.61%`。最后把相同的真实 action history/prefix 分别送入冻结 LeWM，在 `100` 个 origin、两个相机和 H1-H8 上递归预测同一 clean future target：pooled drift MSE 从 `0.53260` 降至 `0.42738`，改善 `19.76%`，H1-H8 全部正向；wrist/column 的 pooled 改善分别为 `1.37% / 24.63%`。clean pooled MSE 从 `0.37333` 增至 `0.38243`，退化 `2.44%`。

准确结论是：**在一条未参与梯度训练、但参与 checkpoint 选择的真实双相机 validation 轨迹上，FCPG 的像素校正能够穿过冻结 LeWM 接口并改善合成漂移下的 future-latent prediction。** 该结果仍是单 episode development pilot，origin 不构成独立 episode；它不证明真实相机漂移、双相机 fusion、候选动作选择或机器人控制收益，也不授权直接驱动硬件。正式验证必须采集全新的 physical episodes，并在看结果前冻结 checkpoint、扰动和控制协议。

#### SO-101 当前场景双相机 shadow ranking

`2026-07-24` 的真实硬件 pilot 使用 wrist `60 FPS/MJPG` 与 column `30 FPS/YUYV` 配对采集 `90` 帧；follower 与 leader 总线均只读连接，采集和 evaluator 均未调用 `send_action`。部署坐标合同已修正为：FK/XYZ 只描述机器人三个物理锚点；鱼、盒子和 column 图像中的绝对像素都不假设固定。wrist 相机只利用其相对夹爪刚性安装，将当前鱼位置表示为 `fish centroid - FCPG-corrected gripper center`，用于检索相似示教阶段；column 当前帧重新检测鱼或盒子，并用 FCPG 校正后的当前夹爪及 LeWM H1-H8 预测位置进行候选排序。

在 fresh read-only capture 上，`current-fish` 模式从 `24` 条示教 action prefix 中得到 `7` 条 shadow-feasible 候选，选中 episode `3` / frame `65`。选中前缀的首命令最大关节变化为 `0.965`，完整前缀相邻命令最大变化为 `3.591`，所有实际命令均位于冻结 SO-101 位置边界内。LeWM 状态头曾将 gripper 预测到 `-2.17`，因此实现已将“实际命令硬安全”与“模型预测可信度”分离：真实命令边界不允许 margin；仅预测状态使用显式记录的 `3.0` plausibility margin。evaluator 始终保持只读并固定 `execution_qualified=false`；真实动作只能由独立 one-step executor 在重新读取关节状态、检查报告时效与漂移、并再次执行硬边界检查后发送。

在用户确认工作区安全后，one-step executor 只发送了上述候选的第一条六关节命令。发送前实测状态与 shadow report 完全一致，发送命令未被 LeRobot 裁剪，`0.75 s` 后回读显示主要变化为 wrist flex `+0.839 deg`、elbow flex `+0.266 deg`；总线未报错。该结果只证明一个受限微步被可靠执行，不证明抓取或放置。动作后的第一次只读重排正确地停止：基于“最大红色连通域”的腕部目标检测从真实目标跳到了左上角压缩噪声。随后改为以前一份 report 的目标中心做最近邻连续关联，并在移动超过 `40 px` 时 fail closed；同一 post-step 图像中目标中心由 `(213.78,39.43)` 连续到 `(214.10,35.16)`，距离 `4.28 px`，重新得到 `7/24` 个合格候选。该修复只解决当前短步之间的数据关联，不假设鱼、盒子或桌面坐标固定，也没有发送第二条动作。

这个结果只证明当前场景检测、FCPG grounding、5M temporal dynamics 候选 rollout 与硬安全过滤能够在真实双相机输入上连通。它不证明鱼已抓取、已放入盒子、FCPG 真机像素误差、LeWM 控制收益或闭环成功率。抓取后目标必须从当前鱼切换到当前盒子，并且每个动作后都要重新采集和重排，不能复用本页坐标。

#### SO-101 人工确认的实时 FCPG + LeWM MPC

当前实现已从一次性 shadow evaluator 扩展为持久运行的短时域闭环：同一个进程保持 follower 与 wrist/column 相机连接，按已执行的五命令 chunk 维护 `t-10/t-5/t` 最近历史；FCPG 在两路当前 RGB 上校正三个机器人锚点，检索与 IK 产生 action-prefix 候选，默认 V9 使用真实任务微调后的 temporal dynamics 预测 H1-H8 六关节未来，并保留 V6 多尺度历史层作为视觉不可靠时的有界残差，再由目标距离、示教距离、模型-命令一致性、阶段约束、关节硬边界和 MuJoCo 工作空间共同筛选。浏览器左侧保持当前真实 wrist 图像，右侧只播放同一候选的 LeWM H0-H8 MuJoCo 预测；右侧不是未来真实 RGB，也不回馈给 FCPG。

执行合同固定为：页面显示当前 round、FCPG grounding、目标关联和 MuJoCo 预测；用户提交后，运行时重新读取真实关节与 column 目标，检查 round、计划时效、状态漂移、目标漂移及五步命令边界，只发送所选 prefix 的前 `5` 条命令；随后等待舵机接近最终目标并连续稳定，把该 settled endpoint 作为一个新的任务事件写入历史并规划下一轮。等待稳定期间的相机帧不推进模型时间。回放模式已经验证一次恰好五步的模拟执行和第二轮重规划。真实硬件模式也完成了一次五命令发送：LeRobot 未裁剪命令，`hardware=true`，即时观测最大关节位移为 `1.729 deg`；但舵机在 `0.163 s` 命令窗口后继续跟随最终目标，使下一轮状态漂移达到 `4.529 deg` 并被 `2 deg` 安全门拒绝。代码随后加入 settle-and-refresh，并在每个成功 chunk 后自动撤销执行授权；下一块必须先显式重新规划，浏览器连续提交不能直接推进机械臂。

硬件服务默认只读；真实发送同时要求 `--execute-enabled` 与 `--acknowledge-workspace-clear`，且每轮仍需网页点击。总线读取和写入各允许最多四次附加通信重试，但状态包持续缺失、越界、跟踪失败或稳定超时仍会立即锁停，系统不会自动重提任务动作。当前结果只证明实时 FCPG/LeWM/候选筛选/MuJoCo/五步执行/重规划的软件和受限硬件 chunk 已连通，不证明抓取放置完成或方法提高任务成功率。

`2026-07-25` 的连续接近测试执行了七个五命令 chunk，column 中校正后的夹爪到鱼目标距离由约 `100.5 px` 降至约 `66 px`，随后全关节状态读取返回 `There is no status packet`，运行时按合同停止。复核发现每个微步调用 LeRobot `send_action` 时，库会在写入前额外执行一次 `num_retry=0` 的六关节 `Present_Position` 读取；一个五步 chunk 因此产生五次无重试读取。当前 executor 改为：chunk 开始时只做一次有重试的全关节读取，逐步执行已验证命令时直接以有界重试写 `Goal_Position`，完成后仍按原 settle 合同读取真实反馈。修复后的总线健康检查连续完成 `20/20` 次六关节读取；随后“回到初始位置”以 `10` 个低速插值命令从 `[5.636,-97.777,97.604,49.581,-51.429,31.406]` 返回目标 `[-10.116,-97.777,97.604,68.037,-51.377,31.406]`，最终最大关节残差为 `0.168`，全程无裁剪、无状态包错误。该恢复只验证修复后的通信与受限轨迹执行，不构成抓鱼或 FCPG/LeWM 控制效果证据。修复后的 SO-101 focused 回归为 `226 passed`，全量回归为 `3245 passed, 106 warnings`。

后续离线复核确认，早期真机 round 的红鱼检测本身正确，失败来自候选动作：固定示教检索得到的安全候选只在盒子附近上下移动，候选集合没有包含从当前夹爪位置朝当前鱼位置运动的动作，LeWM 因而不可能从错误候选中选出抓取路径。曾实现一个由 column 二维像素误差生成 action prefix 的 image-servo 候选作为快速诊断；保存观测上的离线复算虽显示像素距离下降，但真实连续运行 `13` 个五命令 chunk 后，动作主要落在 shoulder pan 与 wrist flex，shoulder lift/elbow 基本不承担三维接近，机械臂表现为反复抬头/点头而没有抓取。该现象判为**欠约束二维关节映射 No-Go**，不是目标检测失败，也不能据此判定 LeWM 动力学失败；image-servo 候选已从真实 approach/transport 入口移除，固定示教候选仅保留为后备对照。

网页另加入双重确认的慢速“回到初始位置”操作：只沿已保存 home state 生成逐步受限的五命令 chunk，完成后刷新完整历史并重置任务阶段；任何状态读取、稳定或命令一致性失败仍立即锁停。相关 focused 回归为 `34 passed`，全量回归为 `3169 passed, 106 warnings`。

#### SO-101 历史 v10 兼容路径：目标 IK 候选与 LeWM 后果预测

真实 SO-101 后续**不训练六关节 action decoder，也不再用图像误差手工映射关节增量**。此前效果较好的 MuJoCo 页面实际执行的是“已知 action prefix -> LeWM 预测未来六关节状态 -> 官方 SO-101 MuJoCo FK 渲染”；它证明了 LeWM 的动作条件状态预测，但 MuJoCo 当时没有负责生成动作。真机候选失败的直接原因是示教检索候选没有覆盖“从当前夹爪到当前物体”的运动，而不是 LeWM 可以从错误候选中自行创造新动作。

冻结职责如下：FCPG 从当前双相机图像校正三个机器人 anchor；目标定位器给出物体或容器的 base-frame 目标；MuJoCo Jacobian 阻尼最小二乘 IK 生成满足关节限位的平滑六关节候选，其中 IK 只修改前五个机械臂关节并保持夹爪命令不变；冻结的本地 transfer 5M temporal LeWM 预测每个候选的 H1-H8 后果并排序；FK/MuJoCo 负责工作空间、关节与预测一致性检查。系统只执行首个短 chunk，随后重新读取两路 RGB、重新 FCPG、重新 IK 和重新排序。这使 FCPG、IK、LeWM 的贡献可分别归因，也避免小样本 action decoder 记忆五条示教。

当前 RGB-only 临时目标深度使用“column 像素 + 已知桌面高度 + column affine projection 的平面逆解”。该逆解只是可替换的近似 3D provider，不是完整针孔标定，暂时只授权 MuJoCo shadow 与 IK 候选检查。后续 RGB-D 相机到位时只替换目标 3D provider；IK、LeWM、FCPG、候选排序和前端接口保持不变。wrist 相机继续利用刚性安装做近场 FCPG/目标相对修正，不承担绝对深度估计。

当前真机 planner 已接通完整链路：先撤销开合夹爪标定得到的 column 实时二维相似变换，再把当前目标像素反投影到已知桌面高度；MuJoCo DLS IK 生成 direct/arc/左右绕行五种三维候选，LeWM 预测每个候选 H1-H8 的六关节后果，并以 base-frame 末端目标误差为主、像素误差为辅进行排序。approach->grasp 与 transport->release 还必须同时满足像素距离和米制三维距离，避免透视下“像素很近但物理仍远”时误切阶段。未来像素投影也统一应用当前 column 相似变换，不再混用标定帧与实时帧。

`2026-07-25` 的 v10 只读现场验证从状态 `[19.147,-97.612,98.048,38.674,-51.325,1.643]` 和当前红鱼像素 `(21.02,153.87)` 生成 base-frame 目标 `(0.2695,-0.1182,-0.0171) m`，当前末端距离为 `108.3 mm`。五条 IK 均在 `3 mm` 内收敛；训练轨迹包络外 `40 mm` 的显式工作区余量允许该可达目标，同时所有实际/预测路径越界量均为 `0`。冻结 LeWM 选中 `arc_left_contact`：H1 的预测/命令末端差为 `8.0 mm`，H8 增至 `24.5 mm`，说明短期方向可用但长时域仍低估运动幅度；因此控制合同仍是只执行首个五步、重新采集双相机、重新 FCPG/IK/LeWM，而不是一次执行 H8。首个五步会协同改变 shoulder lift、elbow 和 wrist flex，不再是旧二维映射的单腕点头。只读 MuJoCo 产物位于 `outputs/lerobot/so101/realtime_fcpg_lewm_grasp_20260725_v10_ik_workspace_readonly/`。本轮没有发送任何新硬件动作；下一次真机验证必须重新确认工作区与急停，并以 `15 Hz` 仅执行一个 chunk。

#### SO-101 当前默认控制：目标条件候选与 LeWM 后果排序

上述 v10 的显式 `approach/grasp/transport/release` 目标阶段路径现在是默认的 `phase_target` 部署模式：FCPG 先从当前双相机观测校正夹爪中心、固定指尖和活动指尖三个物理锚点；目标跟踪器给出当前 pickup/drop-zone 参考；MuJoCo Jacobian IK 根据当前目标生成多条完整六维 action-prefix 候选；V9 LeWM 分别预测每个候选的 H1-H8 六关节未来，随后以三维目标误差、像素误差和安全约束排序。它不读取固定鱼/盒子坐标，目标可以在每轮重新登记或由当前外观模板跟踪。原 `anchor_trajectory` 保留为回放/时间合同对照，不是随机目标的部署路径。

默认 `phase_target` 不把 LeWM 的预测状态直接当命令。它先根据当前目标生成安全的 `40 x 6` action-prefix 候选，再由 LeWM 预测每个候选的 H1-H8 后果并排序；运行时只执行胜出候选的首个 `5 x 6` chunk，然后读取新关节状态和双相机 RGB，重新接地和规划。当前准确链路为：

```text
dual RGB + joint state
-> FCPG current three-anchor measurement
-> current pickup/drop target tracking
-> FCPG image servo + MuJoCo IK + optional retrieved action-prefix candidates
-> action-conditioned V9 LeWM H1-H8 consequence prediction
-> goal, model-consistency, workspace, joint-limit and phase scoring
-> execute first five commands
-> fresh observation and replan
```

这里有一个与七任务离线页面不同、且必须明确的部署约束：七任务页面把真实未来 action prefix 直接作为 LeWM 输入，属于 teacher-forced 后果预测；真机的 `phase_target` 先由当前目标和 IK 产生候选动作，再让 LeWM 预测候选后果，不能凭空从错误候选创造动作。该模式是 model-based receding-horizon control，不是端到端 action policy。命令行默认 `--control-mode phase_target`。

可选的 `--control-mode anchor_trajectory` 才执行另一条链：检索成功示教 action prefix -> LeWM 预测三锚点后果 -> 三锚点 IK 重建六关节命令。它在第一个完整 chunk 成功后锁定 `(episode, source_frame)`，后续只能沿同一示教轨迹按 `+5` 帧推进；该模式只用于 replay/debug 和时间合同验证，不用于随机目标控制。这个区分避免把 teacher-forced 七任务图集误写成端到端动作生成结果。

`anchor_trajectory` 只读回放 smoke 在保存的真实双相机 session index `54` 上选中 candidate `8`，三锚点 IK 的 H1-H8 最大误差为 `2.967 mm`，输出完整 `40 x 6` 前缀，并完成一个模拟五命令 chunk 后的第二轮观测与重规划；近限位关节没有继续向外运动。该结果只证明“三条 LeWM 轨迹 -> 六维 IK 命令”的软件合同闭合，不是随机目标控制或真机抓取成功证据。

独立的 `phase_target` 无硬件回放在同一 index `54` 重新登记当前鱼 `(28,177)` 与盒子 `(150,151)` 后，生成 `33` 个候选，来源覆盖 `fcpg_goal_servo / calibrated_cartesian_ik / retrieved_demonstration`；最终选中 `fcpg_goal_servo`，计划无阻断且通过安全检查。这证明默认路径会对当前目标重新生成候选，而不是沿用固定目标坐标或退回示教续播。它仍然只是软件/回放合同验证，不是物理抓取成功证据。

#### SO-101 三帧主干与多尺度因果历史

真实部署继续把最近 `3` 个 FCPG-grounded 观测作为 LeWM 主干输入，不把全部历史直接塞进 Transformer。额外历史由同一 episode 内严格早于这三个观测的 `t-5/t-15/t-30/t-60` 记录提供；每条记录包含当时的视觉 embedding、六维关节状态、随后真实执行的 action chunk、相对年龄和有效标志。一个冻结主干上的 `1,140,486` 参数 cross-attention Adapter 查询这些记录，并以有界 residual 修正 H1-H8 visual/state prediction。Adapter 的输出头零初始化，初始时严格等价于已有 `22.68M` 参数 temporal LeWM；训练仅更新 Adapter，SIGReg 固定为 `0.02` 并作用于 Adapter 的 FP32 hidden representation，而不是直接约束预测张量。

轨迹记录使用磁盘压缩存储，进程内 decoded cache 按 LFU 淘汰并以最早访问时间打破平局。LFU 只管理缓存驻留，不参与模型检索或科学结论；首轮实验也禁止跨 episode 取历史。控制端仍维持 receding-horizon 合同：每轮最多授权首个五命令 chunk，之后必须重新读取两路相机、重新运行 FCPG 和目标检测，再生成下一轮计划。

`seed=5130` 的唯一 `2,000`-step development 训练已完成，父 checkpoint 为 `so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_10k/latest.pt`，输出为 `outputs/lerobot/so101/so101_multiscale_memory_reprsigreg_seed5130_2k/`。完整验证含 `41,167` 条序列、`30` 个 source-task group：

- visual MSE 的任务宏平均由 `0.00615408` 降至 `0.00611734`，改善 `0.597%`，`28/30` 个 group 同向；micro 改善为 `0.920%`；
- 打乱同一 batch 内的远期历史后，visual MSE 相对正确历史上升 `0.323%`，`25/30` 个 group 同向，说明 Adapter 没有完全忽略历史；
- state MSE 的任务宏平均由 `0.000172217` 降至 `0.000171706`，改善 `0.297%`，仅 `18/30` 个 group 同向；打乱历史后的 state MSE 上升 `0.677%`；
- `latest.pt` 可 strict-load 为 `MultiScaleHistoryLeWM`，step 为 `2000`，两个零初始化输出头均已更新，最终 GradScaler scale 为 `131072`。

准确结论是：在不改变最近三帧主干的前提下，稀疏长历史给 clean future visual latent 带来小幅、广泛且可检测的增益；它尚未证明遮挡恢复、真实控制成功率或 FCPG+LeWM 联合收益。下一项有信息量的实验应是冻结该 checkpoint，在随机 recent-observation dropout 下比较 parent、正确长历史与 shuffled history；若 clean-only Adapter 在该诊断中无收益，再单独训练带因果 suffix-dropout 的 fresh Adapter，而不是继续扩参数或增加训练轮数。

该 recent-observation 诊断随后已完成。冻结最近 `0/1/2/3` 个 context frame 时，目标帧保持真实且不进入输入；三种 checkpoint 均在相同的 `3,840` 条、每个 episode 已积累至少 `100` 个 raw frame 的 validation 样本上评估。原 clean-only 四历史帧 Adapter 相对冻结主干的 visual MSE 改善为 `0.587/0.451/0.283/0.165%`；在 `75%` 训练样本上随机冻结因果 context suffix 后，四历史帧版对应为 `0.307/0.452/0.521/0.471%`。这说明 dropout training 确实将增益从 clean 转移到了连续丢帧条件，但绝对幅度仍很小。

按部署需求又执行了一个有界的 v7：主干仍保留最近 `3` 帧，同时 Adapter 读取 raw offset `5/10/15/20/30/40/60/80/100` 的 `9` 个历史观测，共利用 `12` 个时间点；每个历史 token 加入 `log-age + reciprocal-age + four-frequency Fourier` 时间编码，最近三帧再加入显式 visual-validity 编码。冻结主干、SIGReg=`0.02`、seed=`5132`、`2,000` step 后，完整 clean validation 的任务宏平均 visual/state 改善为 `0.468/0.483%`，micro visual 改善为 `0.789%`。在上述严格共同样本诊断中，冻结 `0/1/2/3` 帧的 visual 改善为 `0.480/0.571/0.592/0.519%`，相对四历史帧 dropout checkpoint 的 visual MSE 又降低 `0.174/0.119/0.072/0.049%`；正确历史相对 shuffled history 在四种条件下均保持正向，分别有 `23/24/23/24` 个 source-task group 同向。

因此，**增加历史数量、时间编码和有效性编码提供了小而一致的额外信息，但没有解决视觉失效恢复。** 继续简单增加帧数或训练轮数的边际收益已经很低；下一步若继续该方向，应改为在 FCPG/图像质量给出的部署式 measurement confidence 下测试状态级 posterior switching，而不是继续扩大被动历史窗口。正式产物为：

```text
outputs/lerobot/so101/so101_multiscale_memory_reprsigreg_seed5130_2k/recent_freeze_common100_dev3840.json
outputs/lerobot/so101/so101_multiscale_memory_freeze75_seed5131_2k/recent_freeze_common100_dev3840.json
outputs/lerobot/so101/so101_multiscale_memory_v7_reliability9_seed5132_2k/evaluation.json
outputs/lerobot/so101/so101_multiscale_memory_v7_reliability9_seed5132_2k/recent_freeze_common100_dev3840.json
```

V6 `so101_multiscale_memory_freeze75_seed5131_2k/latest.pt` 继续保留为历史层来源：最近三帧主干加 `t-5/t-15/t-30/t-60` 四个因果历史帧，不启用 v7 的九帧/Fourier/reliability 分支。v7 只保留为“更多历史有小幅边际增益”的消融。此前实时网页把本地 FCPG bridge 与通用 V6 dynamics 分开加载，虽然两者的 ViT/base 权重完全一致、张量维度兼容，但 V6 backbone 没有包含 `my_first_task` 的本地 temporal transfer；因此它并不等价于上次真机表现较好的路径。

当前部署候选统一为 **V9 local-multiscale**：将 bridge 指向的本地 temporal checkpoint `my_first_task_clean_v1_wrist_transfer_seed6120_1k_fixed/step_000250.pt` 作为唯一 backbone，并保留 V6 已训练的 `1,140,486` 参数多尺度历史层。生成后的单文件 checkpoint 为 `outputs/lerobot/so101/so101_v9_local_multiscale_seed5131/latest.pt`，协议为 `so101-state-residual-lewm-v9-local-multiscale`，总参数量 `23,822,338`。构建器逐 key/shape 检查两个 backbone，并要求全部 ViT/base 张量逐值完全相同；随后只替换 `backbone.*`，不改 memory 权重。实时 loader 又验证 V9 内嵌 backbone 与 FCPG bridge 声明的本地 checkpoint 完全一致，并让 FCPG、候选库编码、近期 rollout 和 memory rollout 共享同一个模型对象。

V9 的历史策略固定为 `auto/off/always`，默认 `auto`：当前两路 FCPG measurement 均合法时严格旁路 memory，近期预测数值等同于上次有效的本地主干；measurement 不可靠且已有严格因果历史时才计算 V9 memory fallback，但现有安全层仍会阻止不可靠 grounding 直接取得真机执行资格。`off` 用于本地主干 parity，`always` 只用于离线历史消融。运行时保留最多 `14` 个真实已执行 action chunk，状态、历史图像和“从该历史状态开始执行的 action chunk”严格对齐，网页等待帧不进入记忆。V9 仍只预测候选后果，不生成动作；夹爪开合和机械臂路径必须来自 IK/阶段候选。网页的一次执行请求现在固定为最多 `20` 条命令，但后端不会盲发 H4：它串行执行 `4 x 5` 个子轮，每个子轮结束都用新的 wrist/column 图像重新运行 FCPG、目标检测、阶段更新和 V9 候选规划。任一新计划不可执行、输入结束、通信失败或安全检查失败都会取消剩余子轮；整批结束后重新撤销执行授权。

实时 column 诊断页此前只保存 `project_future_anchor_pixels(...)[...,0,:]`，因此只画出了夹爪中心一条 LeWM 轨迹；这只是可视化丢失，不是 V9 只预测一个点。V9 的六维未来状态包含 gripper aperture，当前实现保留原中心轨迹用于目标代价，同时将每个 H1-H8 状态经 `SO101MuJoCoKinematics.forward_anchors()` 和当前相机投影得到完整 `center/fixed-tip/moving-tip` 三轨迹。网页分别显示 `LeWM-C/L/R` 与对应命令 FK 轨迹，三点都来自同一预测状态和真实夹爪几何，不通过复制中心线构造。

V9 可由下列命令确定性重建，不重新训练 ViT、FCPG 或 dynamics：

```bash
env PYTHONPATH=.:/home/james/lewm /home/james/lewm/.venv/bin/python3 \
  scripts/build_so101_lewm_v9.py
```

旧的 `ik_v9_local_multiscale_shadow_real_fish_20260725_v4` 页面没有加载开合夹爪标定，report 中 `column_gripper_localization=null`，仍从带黑边历史缓存的旧 column affine projection 开始；它把三锚点画在盒子边缘，并用同一错误像素系计算目标 3D，因此已撤销发布资格，不得再用于判断 V9 或规划质量。

V9 发布 shadow 现在强制读取 `analysis/live_anchor_calibration.json`，并调用与实时 planner 完全相同的候选选择和受约束二维相似变换。若 V9 未提供该文件、只提供旧单点 translation，入口会 fail closed。当前发布审计恢复的变换为 scale `0.841`、rotation `-18.45 deg`；最终 column nominal 的 `center/left/right` 为 `(84.78,86.67)/(87.18,86.22)/(82.37,87.12)`，FCPG 后为 `(85.94,89.98)/(88.35,89.32)/(83.52,90.64)`，平均局部 correction 为 `3.51 px`，且 center 在数值精度内严格等于两指尖中点。鱼的 live pixel 也先经过该相似变换的逆映射，再进入旧 column 平面近似，不再把 transform 当纯 translation。

在同一份 `90` 帧真实双相机只读记录上，校准后的 geometry 选择 `direct_low`，V9 本地主干与 V9 memory 均选择 `arc_left`；memory 相对本地主干的平均 TCP 预测差为 `0.843 mm`，没有改变 V9 赢家。候选变化来自修正后的目标几何，不能和旧 v4 数值直接比较。该记录接近静止，只证明 V9 checkpoint、FCPG、IK、实时 column bootstrap 和 memory 可共同运行，不证明 memory 提高控制成功率，也没有发送机械臂命令。发布人工检查入口为 [V9 calibrated release shadow](outputs/lerobot/so101/ik_v9_release_real_fish_20260725/index.html)。

实时 V9 planner 同样把有效 column 开合标定设为执行前置条件：未加载或尚未在网页完成标定时会返回 `column_live_calibration_required`，只能观察，不能取得 task chunk 执行资格。相机或支架移动后必须重新开合标定，旧 transform 不能跨相机位姿复用。V9 实现验证为：最新 calibration focused tests `34 passed`，完整回归 `3286 passed, 108 warnings`；全部相关 Python 入口通过 `py_compile`，`git diff --check` 通过。上述验证不包含新的任务动作。

在一段 `90` 帧真实双相机记录上，因果冻结最近 `0/1/2/3` 个视觉 context 后，geometry、parent 和 v6 四次均选择 `direct_low`。v6 相对 parent 的候选平均 TCP 预测差异仅为 `0.858-0.869 mm`；相对 clean 的输出漂移没有小于 parent。该记录接近静止，只能证明 v6 memory/FCPG/IK 接线与冻结诊断可运行，不能证明 v6 改善真实控制。有效验证必须使用包含明显机械臂运动和已知执行动作的 fresh trajectory；在此之前不因该 smoke 启动硬件动作。人工检查入口：

该 smoke 还暴露了一个独立的 column 坐标问题：用于训练/点击标定的历史 column 缓存带上下黑边，而最新实时采集是满画幅；旧 affine projection 因而把夹爪中心画到盒子边缘附近。FCPG 只有局部捕获范围，不能从错误几十像素的 seed 自动恢复。旧四页中的 column 白点和 RGB-only 目标逆投影因此不具备执行资格，但 wrist grounding、v6/parent 接线和同候选预测比较仍可作为 smoke。

当前修复不依赖旧桌面、盒子或目标背景。任务开始前保持五个机械臂关节不动，让空夹爪慢速执行一次 `close -> open -> final_close`；`fcpg/so101_active_gripper.py` 从同步图像中提取**同一个活动指尖**在 open/closed 两个时刻的候选轨迹。SO-101 只有一侧夹爪随 aperture 运动，开合差分中的两条轮廓不是同一时刻的左右指尖；此前将它们解释为固定/活动指尖对会产生错误尺度与平移，已停止使用。当前 planner 分别用 open/closed 关节状态经 URDF/FK 投影活动指尖，再以视觉时间轨迹拟合二维相似变换；固定指尖和夹爪中心由同一变换作用于已知三锚点几何得到，FCPG 只负责后续局部视觉残差。单次开合只提供当前工作区附近的二维重锚定，不等价于完整手眼外参或深度。脉冲结束时夹爪恢复到闭合端，避免标定完成后以张开状态进入抓取规划。

#### SO-101 实时网页的双环境合同

真实控制网页同时依赖两套已经验证的环境，不能只激活其中一个：`/home/james/miniconda3/envs/lerobot` 提供 SO-101 总线、相机和 LeRobot 运行时；`/home/james/lewm/.venv` 提供 LeWM/FCPG 的 `jepa`、`timm`、MuJoCo 与训练时的 PyTorch。当前机器上单独使用前者会缺少 `timm`，单独使用后者会缺少 `draccus`。实时服务固定以 LeRobot 的 Python 为主解释器，并显式加载 LeWM venv 的模型依赖：

```bash
source /home/james/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd /home/james/feedback-corrected-grounding

export PYTHONPATH="$PWD:/home/james/lewm:/home/james/lewm/.venv/lib/python3.10/site-packages:/home/james/lerobot/src"
export CUDA_VISIBLE_DEVICES=0

python -c "import jepa, timm, draccus, lerobot, torch; print(torch.__version__)"
python scripts/serve_so101_realtime_fcpg_lewm.py \
  --device cuda:0 \
  --home-state-report outputs/lerobot/so101/real_dual_shadow_fcpg_lewm_20260724_pre_traj02_rank/report.json
```

上面的默认命令只连接并读取双相机/机械臂状态，不授权发送任务动作。`--home-state-report` 只读取六关节 `current_state` 作为“回到初始位置”的目标；即使兼容别名 `--object-prior-report` 被使用，旧报告中的 wrist/column 像素坐标也不会再导入运行时。目标必须从本次启动生成的冻结 planning snapshot 设置：先选择“抓取目标”并在 column 图中点击目标中心，可选地在 wrist 图中再登记一次近景；再选择“放置区域”并只在 column 图中点击。模板默认保存在 `OUTPUT/target_profile`，也可用 `--target-profile` 指定目录。保存的是点击区域的外观模板与模板内锚点，不是桌面绝对像素；后续规划会在当前画面重新匹配，因此目标或盒子换位置后不会沿用初次点击坐标。若目标外观或场景明显改变，应在新的冻结 planning snapshot 上重新登记。

夹爪视觉标定与整臂任务执行使用独立权限。只需修正外部 column 相机中的夹爪位置时，在默认命令追加 `--gripper-calibration-enabled --acknowledge-workspace-clear`；此时页面显示 `calibration-only`，后端仍拒绝任务 chunk 和回初始位。“标定夹爪位置”只在真实硬件、`approach` 阶段且首个任务 chunk 尚未执行时开放，只发送慢速 `gripper.pos` 开合命令，五个机械臂关节不发送命令。校准端点来自当前 planner 的 train-only 夹爪位置 `1%/99%` 分位数，并强制至少 `20` 个归一化单位的可观测跨度；不得再使用与本机示教范围不匹配的硬编码 `5/60`。校准脉冲固定为 `close -> open -> final_close`，视觉跨度提取完成后夹爪必须停在闭合端，返回的 `state/final_position/restored_closed` 也绑定真实最后反馈。校准循环从最新反馈把每条命令限制在 `1.5` 单位内，并通过 LeRobot bus 只写 gripper；它不再调用会额外执行一次零重试六关节读取的通用 `send_action`。状态读取和 gripper 写入各允许四次附加通信重试。

视觉侧从一次开合形成的运动包络自动提取活动夹爪纵向两端的轨迹候选，不要求人工点击；候选内的坐标顺序固定为 `[open, closed]`，不得再交换成空间上的左右指尖。系统将 nominal 活动指尖的 open/closed 时间轨迹与视觉轨迹拟合二维相似变换，同时修正 column 三锚点的平移、旋转和尺度。尺度必须位于 `[0.5,1.5]`、旋转不超过 `60 deg`、两个时刻的中心位移均不超过 `35 px`；没有候选同时满足三项约束时整次标定失败关闭，禁止退回大幅单点平移。`60 deg` 覆盖相机允许移动后的现场姿态，但仍拒绝方向翻转和远距离错误轮廓。wrist 相机与夹爪刚性连接，其局部锚点投影不接受 column 变换。超时错误会报告目标、起点、终点、实际范围和最后命令，便于区分机械限位、通信失败和反馈不收敛。需要真实任务执行时才追加 `--execute-enabled --acknowledge-workspace-clear`。网页的“闭环执行最多 20 步”按钮直接提交当前已授权规划，不显示二次确认弹窗；后端内部每完成 5 条命令就用终点双相机观测重新 FCPG 和规划，最多连续四轮，绝不直接发送旧 H2-H4。回初始位、目标登记和标定继续保留确认；所有操作在前端与后端均串行化，停止按钮可独立中断长操作。两张 planning snapshot 固定为 `224 x 224` 方形点击坐标，避免 `4:3 object-fit` 留白造成点击偏移。轮询不会重复设置 GIF 的 `src`，因此 MuJoCo 预览不会再每 `1.5 s` 从头播放。错误区只显示简短故障类别，原始异常与完整状态可分别复制。

网页同时显示 wrist 与 column 两路连续 MJPEG 监控流，以及当前 planning round 冻结的双相机 FCPG/LeWM 参考图。两类画面使用不同的时钟：监控流只供人工观察，读取多少帧都不会写入 `observations`、长期记忆或 LeWM 历史；服务启动、回初始位或主动夹爪标定会重置历史，只有完成并稳定的五命令 chunk 才推进任务事件时钟。显式“重新观测并规划”只替换当前事件的最新观测，不伪造新的静止时间步。因而用户在两次点击之间等待几十秒，不会被解释成几十秒静止轨迹，也不会推进 LeWM 状态。若等待期间机器人、目标或场景实际发生变化，应显式重新规划；执行入口仍会读取最新状态与 column 目标，并拒绝超过时效、状态漂移或目标漂移门槛的旧 round。状态接口固定报告 `observation_clock=task_event`、`monitor_frames_feed_model=false`、`model_history_frames`、`model_executed_chunks` 和 `model_memory_tokens`。

两路相机均使用 `/dev/v4l/by-id/...-video-index0` 稳定路径。真实标定测试中 wrist USB 相机从 `/dev/video0` 重新枚举为 `/dev/video1`，原 OpenCV 异步线程继续持有已移除的旧句柄，表现为线程仍存活但 `async_read` 在 `1000 ms` 后超时。硬件 I/O 现对**单个失败视角**执行最多三次有界恢复：停止该相机的旧读取线程、释放句柄、清空旧 `latest_frame/event`，再通过稳定 by-id 路径重连并等待一张新帧；不会断开另一台相机或 follower。监控流在最终失败时可以继续显示最后缓存用于人工诊断，但 `observe()`、FCPG、LeWM 与任务规划禁止使用缓存帧，三次恢复均失败即停止本轮操作。

`2026-07-25` 的修复验证中，独立 replay 服务通过 HTTP 分别登记 column 抓取目标和放置区域，目标模板均持久化，登记前后保持 `11` 帧模型历史与 `0` 个执行 chunk；wrist/column MJPEG 同时读取后时钟仍不变。真实 SO-101 只读服务随后成功连接 follower 与两路相机，两路流均持续返回图像，状态为 `execute_enabled=false`、`error=null`、`completed_chunks=0`；未调用任务执行或夹爪标定。相关聚焦测试 `31 passed`，全量回归 `3231 passed, 106 warnings`，内嵌 JavaScript 解析、`py_compile` 与 `git diff --check` 均通过。

同日后续硬件诊断确认，单点平移虽然能对齐三锚点中心，却会原样保留错误的指尖跨度，因此不能作为 column 标定；随后一次 v7 pulse 又暴露了“把同一活动夹爪的开位/合位当成左右指尖”的语义错误，该次变换与叠图均作废。修正实现现在使用活动指尖的时间轨迹，并在视觉定位前先将两路完整帧、六关节反馈、命令和 phase 范围保存到 `OUTPUT/gripper_calibration_captures/pulse_*/pulse.npz`，同时生成双相机 GIF。`scripts/analyze_so101_gripper_calibration_capture.py CAPTURE_DIR` 可在不连接硬件的情况下反复重跑定位并生成 open/closed 候选叠图。旧平移入口仅为历史测试兼容，实时标定不再调用；真实 column 对齐仍需对一份新缓存先离线确认，再允许进入任务规划。

v8 calibration-only 复验已完成：一次 pulse 保存 `137` 帧，夹爪反馈覆盖 `1.840--31.406`，五个机械臂关节的观测漂移为 `0`，任务 chunk 执行数为 `0`。column 活动指尖的视觉 open/closed 位移为约 `28.6 px`，对应 FK 名义位移约 `34.0 px`；自动拟合尺度 `0.841`、旋转 `-18.45 deg`，两个时刻最大中心位移 `31.85 px`，均通过原约束。离线叠图中，选中轨迹落在同一活动指尖的 open/closed 末端，变换后的固定指尖与中心也落在另一指尖及其中点附近；随后 FCPG 的平均局部修正为 `5.51 px`。缓存与审查页位于 `outputs/lerobot/so101/realtime_fcpg_lewm_mpc_calibration_20260725_v8/gripper_calibration_captures/pulse_1784958313671272759/`。分析器同时导出 `analysis/live_anchor_calibration.json`；真实服务可通过 `--live-anchor-calibration PATH` 重新读取时间轨迹并由当前 URDF/FK重新拟合，禁止直接反序列化未经约束检查的仿射矩阵。

相机再次移动后的 V9 pulse 保存 `138` 帧，正确的 distal-tip 候选拟合为 scale `0.6135`、rotation `49.91 deg`、最大中心偏移 `29.19 px`；旧 `45 deg` 常量只差 `4.91 deg` 将其错误拒绝。离线 open/closed 叠图确认该候选贴合实际活动指尖，变换后的三锚点在张开时为 `(87.51,70.02)/(96.80,78.23)/(78.22,61.81)`，闭合时收敛到 `(95.48,76.83)/(96.80,78.23)/(94.15,75.42)`，中心保持为两指尖中点。另一个候选的 rotation 为 `-116.68 deg`、中心偏移 `111.59 px`，在保留的约束下仍被拒绝。缓存审查页位于 `outputs/lerobot/so101/realtime_fcpg_lewm_v9_closedloop20_20260725/gripper_calibration_captures/pulse_1784988245667112870/analysis/index.html`。

同日还实现并离线审计了一个递归局部接地候选：上一轮 corrected anchor 按当前/上一轮 calibrated-FK 像素差传播，作为本轮 FCPG 的局部搜索 seed；LeWM bridge 仍单独读取原始几何 nominal，避免改变已有状态语义。同一观察重复规划会复用原 seed，不会重复写回。该候选在保存的 `26` 轮真实双相机画面上全部完成且无局部越界，column 单轮 FCPG 位移最大仅 `5.73 px`，但相对几何 nominal 的累计偏移最终达到 `99.80 px`，叠图明显离开夹爪。根因是当前 real-warp FCPG 学到的是**相对原几何 nominal 的绝对 residual**，不能直接当成相对任意递归 seed 的 measurement innovation；逐轮相加会重复累积同一偏差。因此 `PreparedPlannerConfig.recursive_grounding` 默认保持关闭，真实服务不启用该候选。离线入口为 `scripts/replay_so101_recursive_grounding.py`，审查页位于 `outputs/lerobot/so101/realtime_fcpg_lewm_grasp_20260725_v2/recursive_grounding_replay/index.html`。

现有 `24` 组人工点击只允许用于粗粒度诊断，精度不足以监督真实图像 FCPG，也不得作为递归失败后的训练真值。若继续改善 column 几何，下一种允许的数据来源是跨多个安全机械臂姿态重复自动 `open/close`：每个姿态只移动夹爪自由度，通过同步差分自动取得活动指尖的高精度 2D 时间轨迹，并与已知 FK 3D 点拟合更完整的相机投影；在该投影把 seed 误差压回局部捕获范围后，再单独判断是否需要训练支持随机 query jitter 的 FCPG。

column 相机在当前系统中的职责现已收窄为粗粒度夹爪 ROI、左右指尖身份保持和丢失后的重关联；精细物体/夹爪/目标相对位置仍由 wrist 当前图像与局部 FCPG 负责。数据审计确认 SO-101 的公开锚点顺序为 `center/left/right`，且标签中的 center 在数值精度内严格等于两指尖中点。为消除三个独立回归头可能产生的非刚性漂移，新增了可选 `two_tip_structured` corrector：共享同一 LeWM ViT patch map，只预测左右指尖，并在每次前向中严格重建 center，保持下游 `C/L/R` 张量接口不变；旧 checkpoint 未记录变体时仍按 `independent3` 加载。

同一 real-image warp 数据、LeWM checkpoint、seed `6125`、优化器与 `2,000` step 的匹配 development 中，`two_tip_structured` 将 center-to-tip-midpoint 误差严格压到 `0`，但 validation EPE 从旧头的 `1.9012 px` 变为 `1.9782 px`，column drift 从 `3.2815 px` 变为 `3.5294 px`；column drift P95 约 `8.0 px`，仍位于 `16 px` 局部捕获半径内。直接保留旧头的左右指尖、仅把 center 投影为两者中点时，整体/column drift EPE 仅变为约 `1.9150/3.3060 px`。因此冻结结论是：**两指尖几何约束可作为 column 粗定位的稳定接口，但当前从头训练的结构化头不是精度升级，不替换现有 `independent3` checkpoint，也不作为独立论文创新。** 后续若部署该约束，应优先复用旧头的指尖输出并做确定性中点投影，或用旧权重初始化结构化头；不得继续无界扫参。

离线审查入口：

- [两指尖结构化 `2k` 图集](outputs/lerobot/so101/my_first_task_clean_v1_real_warp_val2_fcpg_two_tip_seed6125_2k_review/index.html)
- [两指尖结构化 `2k` 指标](outputs/lerobot/so101/my_first_task_clean_v1_real_warp_val2_fcpg_two_tip_seed6125_2k/metrics.json)
- [旧头指尖 + 确定性中点投影图集](outputs/lerobot/so101/my_first_task_clean_v1_real_warp_val2_fcpg_midpoint_projection_seed6125_review/gallery/index.html)

同一只读现场还暴露并修复了目标伺服候选与安全层之间的合同冲突：当前 shoulder/elbow 状态靠近关节上下限时，阻尼最小二乘候选会带有很小的向限位外分量，因而四个 FCPG 目标候选虽然明显降低像素误差，仍被 `moving_outward_at_joint_limit` 正确拒绝。候选生成器现把这类分量投影到关节限位的可行切空间，安全检查本身不放宽。现场复算中四个候选均通过原安全检查，规划器从无目标方向保证的检索候选切换到 `servo_direct`；当前 H8 计划将 column 中约 `118 px` 的抓取目标距离降至约 `66 px`。这只证明短时规划合同与候选选择修复，尚未执行硬件 chunk，也不构成真实抓取成功证据。

纯模型训练、CPU 测试和 replay 网页不需要 LeRobot 驱动，继续使用：

```bash
cd /home/james/feedback-corrected-grounding
env PYTHONPATH=.:/home/james/lewm /home/james/lewm/.venv/bin/python3 -m pytest -q
```

- [v6 clean shadow](outputs/lerobot/so101/ik_v6_memory_shadow_real_fish_freeze0_v2_20260725/index.html)
- [v6 freeze-1 shadow](outputs/lerobot/so101/ik_v6_memory_shadow_real_fish_freeze1_20260725/index.html)
- [v6 freeze-2 shadow](outputs/lerobot/so101/ik_v6_memory_shadow_real_fish_freeze2_20260725/index.html)
- [v6 freeze-3 shadow](outputs/lerobot/so101/ik_v6_memory_shadow_real_fish_freeze3_20260725/index.html)

人工检查入口：

- [双相机 FCPG validation 图集](outputs/lerobot/so101/fcpg_dual_v2_seed53102_2k/gallery/index.html)
- [双相机 FCPG `2k` 指标](outputs/lerobot/so101/fcpg_dual_v2_seed53102_2k/metrics.json)
- [本地双相机 real-image warp 留出 episode 图集](outputs/lerobot/so101/my_first_task_clean_v1_real_warp_val2_fcpg_seed6125_2k_review/index.html)
- [26 轮真实画面的递归 grounding No-Go 对比](outputs/lerobot/so101/realtime_fcpg_lewm_grasp_20260725_v2/recursive_grounding_replay/index.html)
- [本地双相机 FCPG corrector `2k` 指标](outputs/lerobot/so101/my_first_task_clean_v1_real_warp_val2_fcpg_seed6125_2k/metrics.json)
- [本地双相机 FCPG-to-LeWM H1-H8 指标](outputs/lerobot/so101/my_first_task_clean_v1_fcpg_lewm_future_val2_seed6126/metrics.json)
- [真实双相机 current-fish fresh shadow HTML](outputs/lerobot/so101/real_dual_shadow_fcpg_lewm_20260724_fresh_current_fish_margin3/index.html)
- [真实双相机第一微步前 shadow HTML](outputs/lerobot/so101/real_dual_shadow_fcpg_lewm_20260724_pre_step01_rank/index.html)
- [真实双相机第一微步后、连续目标跟踪 shadow HTML](outputs/lerobot/so101/real_dual_shadow_fcpg_lewm_20260724_post_step01_tracked_rank/index.html)
- [FCPG 目标导向候选离线 column 预览](outputs/lerobot/so101/realtime_fcpg_lewm_mpc_goal_servo_preview/rounds/r-1784908431498131555-a927ad7a/column_target_paths.jpg)

v4 的人工审查页已改为直接调用 checkpoint 的六维 state head，不再用 train-only ridge decoder 代替模型状态输出。页面覆盖 `12` 个 held-out task episode；逐任务 state RMSE 的 macro/min/max 为 `1.689 / 0.717 / 3.098`（数据集原始关节单位），latent prediction/copy macro MSE 为 `0.011649 / 0.016025`。右侧 RGB 仍不是生成图像，而是在 direct state prediction 的硬姿态约束后、按 predicted latent 选择的 train-only 同任务近邻；平均检索支持率为 `81.45%`。这批页面用于检查 H1（未来 `5` 个 raw frame）状态方向、夹爪开合和 latent 所处训练分布，不证明长时递归视频生成或真实控制成功率。

本地人工检查入口：

- [12 任务 teacher-forced 图集](outputs/lerobot/so101/so101_shared_lewm_v3_seed5120_teacher_forced_gallery/index.html)
- [12 任务 open-loop 图集](outputs/lerobot/so101/so101_shared_lewm_v3_seed5120_open_loop_gallery/index.html)
- [State-Residual v4 12 任务 direct-state teacher-forced 图集](outputs/lerobot/so101/so101_state_residual_lewm_v4_seed5120_2k_teacher_forced_gallery/index.html)
- [10k 全量评估](outputs/lerobot/so101/so101_shared_lewm_v3_seed5120/evaluation.json)
- [State-Residual v4 2k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_v4_seed5120_2k/evaluation.json)
- [State-Residual v4 10k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_v4_seed5120_10k/evaluation.json)
- [State-Residual v4 30k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_v4_seed5120_30k/evaluation.json)
- [Temporal State-Residual v5 5M / 2k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_v5_temporal5m_frozenbase_seed5120_2k/evaluation.json)
- [Temporal State-Residual v5 5M / 10k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_v5_temporal5m_frozenbase_seed5120_10k/evaluation.json)
- [五源 expanded base / 10k 全量评估](outputs/lerobot/so101/so101_shared_lewm_expanded5_dev_seed5120_10k/evaluation.json)
- [五源 Temporal State-Residual / 2k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_2k/evaluation.json)
- [五源 Temporal State-Residual / 10k 全量评估](outputs/lerobot/so101/so101_state_residual_lewm_expanded5_temporal5m_frozenbase_seed5120_10k/evaluation.json)
- [Declutter 51-episode 零训练外部评估](outputs/lerobot/so101/external_declutter_temporal5m_seed5120_10k/summary.md)
- [Declutter 完整 episode：真实观测 / MuJoCo 预测 HTML](outputs/lerobot/so101/external_declutter_temporal5m_seed5120_10k_mujoco_ep0/index.html)

图集右侧不是生成视频，而是 predicted latent 在同 source/task 训练帧中的最近邻，仅用于辅助观察状态落在哪一段训练分布；定量结论仍以 factual/copy/shuffled/no-action 配对误差为准。现有 checkpoint 内保存的是迁移前绝对缓存路径，重渲染时使用 `--cache-base data/lerobot/so101/cache` 即可解析到新位置。

## 创新边界与撞题审计（2026-07-14）

一手来源检索尚未发现完整复现下面五项接口的工作，但这只是当前检索结论，论文不得写成绝对的“首次”：

```text
nominal FK robot anchors
-> current local RGB evidence
-> supervised per-keypoint 2D calibration residual
-> policy feature resampling at corrected anchors
-> no full camera-extrinsic recovery at deployment
```

当前风险分级：

- **完整链路直接重复：未发现。** Stage I 仍有可防守的窄边界。
- **算子级重叠：高。** VolumeDP、Stable-SAM、OmniD、CLAR 和 deformable attention 已覆盖几何 reference point、learned offset 或 feature resampling；局部偏移和二次采样本身不是 novelty。
- **问题与接口重叠：中高。** GeoProp、SCDP、RcP、AxisGuide 已覆盖机器人状态或轨迹的视觉接地，AnyCamVLA、VistaBot、KYC、CamVLA 已覆盖相机变化鲁棒。
- **审稿风险：中高。** 若没有结构对照，Stage I 容易被评价为 `GeoProp + keypoint refinement/deformable sampling`。

因此当前最窄 claim 是：**FCPG 保留具有固定物理身份的名义 FK 锚点，从当前局部视觉观测预测有监督的二维标定残差，并在校正位置重新读取策略特征；它不恢复完整 SE(3)，也不是在线标定或 test-time training。** 本仓库尚未复现 GeoProp 的完整 FiLM 与 predictive kinematic token，现有对照必须写成 `GeoProp-lite` 或 open-loop GeoProp-style grounding。

投稿前强制补齐：

1. 参数量匹配的 `FK-DeformAttn`：相同 FK reference points、backbone、policy 和预算，只用 policy loss 学自由 offsets/weights；seed-0 四任务 matched development 训练正在执行，结果出来前不作强弱判断；
2. `Anchor-conditioned Absolute-Keypoint`：相同 ViT、`q_true` 标签和数据，直接预测绝对二维关键点；seed-0 exposed validation 已完成，参数量精确匹配时 residual FCPG 的 EPE 为 `0.5529 px`，优于 absolute head 的 `0.6282 px`。Absolute head 相对 residual FCPG 高 `13.63%`；等价地，residual FCPG 相对 Absolute head 低 `11.99%`。该对照仍使用 FK/anchor identity 条件，不得称为全局关键点检测器；
3. `Local-Match/Track`：frozen prototype 以及统一首帧和延迟预算的 RAFT、TAPIR 或 CoTracker；
4. `Keypoints + SE(3)`：相同视觉对应输入 weighted PnP/RANSAC，恢复共享外参后再投影；
5. 完整 GeoProp 复现，或在全文始终准确标注 `GeoProp-lite`。

本轮文献审计还保留一个**未实现、未过门的研究候选**：`Policy-Distilled Kinematic Grounding Correction`。训练时可让冻结策略在 oracle anchors 上的动作分布监督 drift observation 下的坐标校正器，使目标从纯 pixel EPE 转向 control-equivalent grounding；但 BYOVLA、ATK、RobustVLA、PAIR-VLA 和 VaGraM 已覆盖宽泛的 policy sensitivity/control-weighted error，因此只有“oracle-grounded action distribution 蒸馏到 FK coordinate corrector”这一接口可能成立。它不改变当前冻结执行队列，也不得在 rollout 增益出现前写成贡献。详细文献与禁用 claim 见 [`docs/related_work.md`](docs/related_work.md)。

## 主线与候选状态

| 阶段 | 核心问题 | 方法边界 | 当前状态 |
| --- | --- | --- | --- |
| Stage I：LeWM-FCPG | 在保留名义标定的情况下，如何逐观测修正策略读取视觉特征的位置？ | 局部二维 residual；不显式在线重解完整相机外参 | pick adapter/policy 与 GeoProp+jitter seed `0/1/2`、完整 fresh curves、prototype/PnP/oracle 同轨迹矩阵已完成；same-checkpoint action intervention 在 pick/button 三 seed 上给出一致机制证据；wide prototype 的独立确认说明 10 deg 捕获范围可改善，但合取 Gate 为 No-Go |
| Full grounded LeWM 历史候选 | FCPG 校正后的 grounding 是否为已有轻量世界模型提供更稳定、可预测的动作条件状态？ | 固定 LeWM 架构与训练预算；nominal/FCPG matched comparison；统一 frozen oracle target；L1 通过后才允许进入反事实排序和在线控制 | 正式 `10,000`-step Gate L1 为 **No-Go**：primary error 改善 `6.61%`，低于预注册 `10%`，且 `reach-v3` 未优于 hold-last。后续参数匹配的 correction-aware anchor-input development 消融也为 **No-Go**：PI/PI 相对 TC/TC 在 mild 条件退化 `0.88%`，95% CI `[-1.35%, -0.43%]`。两次 No-Go 均保持有效；正式 L2/L3 未启动 |
| Anchor-Persistent LeWM v1 候选 | 旧接口是否因过早把时间与锚点压成单个 `192D` summary 而丢失动作条件的局部动力学？ | 保留 `[B,T,K,D]` 逐锚点状态；复用官方 2026-06-03 LeWorldModel v3 的 `ARPredictor` 与 `SIGReg`；当前 FCPG 测量处理外生相机变化，LeWM 只预测机器人/任务引起的锚点表征变化 | 固定 `2,000`-step dynamics pilot 通过。fresh posterior Gate 中五个 seed 的遮挡期主误差均改善 `88.2%--89.8%`，且 full 明显优于 `no_lewm`；但所有 learned endpoint 均因恢复/clean safety 未通过而无法选择，最终为 **mechanism positive / deployable Gate No-Go**。不授权 `10k`、selector/CEM 或论文第二贡献 |
| Random-Hazard predictor isolation A0 | 在离线已知视觉 measurement 失效时，learned dynamics 是否比简单时序传播更准确地维持具有固定身份的 grounded anchor state？ | 冻结 `2k` Anchor-Persistent LeWM；真实 `sensor-valid` mask 只负责路由；分别维护独立递归历史；不训练 selector | **历史 Development A0 Go，解释已由 A0.5 收窄**。Factual LeWM 相对 hold-last、latent CV、frozen zero-action 和 shuffled-action 明显改善；这证明 predictor utility 与输入敏感性，不再单独证明 factual action 必要性。尚未比较 FK/KF 或 policy rollout，不构成第二贡献 |
| Random-Hazard matched-action A0.5 | A0 的动作收益是否能超过独立训练、同初始化同容量的 no-action recurrent predictor？ | 冻结同一 anchor encoder；两个新 dynamics head 各训练固定 `2,000` steps；action 可见性是唯一结构因子；递归评估使用各自独立 posterior history | **Development No-Go / predictor mechanism remains positive**。单步 validation 中 factual 相对 no-action 改善 `70.78%`；但随机中断递归评估仅改善 `7.54% [-3.88%,18.29%]`，为 `3/4` 任务同向，button 反向 `-7.90%`。Factual 仍显著胜 hold-last、CV 和 shuffled-action；当前数据证明 learned dynamics 有用，但没有证明 factual action 在四任务递归维持中具有稳定必要性 |
| Action-Branch Random-Hazard A1 | 在完全相同的物理状态与历史下，真实执行动作是否解释八条干预分支的未来 anchor-state 差异？ | `8` 条匿名 QR 动作分支；H=`8` 递归预测；factual 与独立训练、容量/初始化/优化匹配的 no-action head；七种 cyclic wrong-action control | **Screen No-Go / descriptive action signal strong**。三 seed `2k` 中 factual 相对 matched no-action 改善 `55.75%`，相对 wrong-action 改善 `73.74%`，晚时域仍为 `52.72% / 70.37%`；但 Button 只有 `10/29=34.48%` development bundles 达到预注册 branch-variance 支持率。后续 train+development 物理重放显示 pre-contact/contact/post-success 的 latent 中位分叉为 `0.02287/0.01089/0.000146`，物理与 latent 同步塌缩，根因主要是 post-success 吸收态占比过高。A1 判定不变：不启动 fresh `10k`、不打开 test、不形成第二贡献 |
| Action-Branch Continuous-Task A2 | 用 Drawer 替换 Button 后，晚时域动作分叉能否在 fresh 数据上支持确认性 action-conditioned dynamics？ | fresh pick/drawer/reach/push；same-state 八分支；H6-H8 primary；matched factual/no-action；七种 wrong-action；冻结 `SIGReg=0.02` | **2k Development-Go / fresh train-support Confirmation No-Go**。三个 seed 的 factual 在 development 上相对 no-action/wrong-action 改善 `67.69% / 82.99%`，四任务、三 seed 均同向；但独立 `24` episode/task train support 中 Drawer 只有 `72.92%`，低于冻结的 `80%`，其余三任务为 `100%`。因此 `10k` 未启动、test 未采集，不能升级为第二贡献 |
| Action-Branch Continuous-Task A3 | 在预先保证四任务分支支持后，一个共享 LeWM 是否在 reserved test 上稳定需要 factual action？ | pick/drawer/reach/push 联合训练一个模型；expert/reverse/hold/half-speed/四个 transverse 共八分支；H6-H8；matched no-action；七种 wrong-action；仅 `SIGReg=0.02` | **四任务 10k Confirmation-Go；七任务规格随后已由 A4/A4b 完成。** 三个 fresh-start seed 在 test 上相对 no-action 与七种 wrong-action 平均改善 `74.01% / 88.18%`；七个 shift 均改善 `87.90%--88.55%`，`4/4` 任务、`3/3` seed 与全部 cross-replication checks 均通过。该结果是 action-conditioned anchor transition 的 preliminary mechanism evidence；A4/A4b 取代它作为当前七任务证据，但二者都不等同于 selector、policy rollout、CEM 或完整持续接地闭环 |
| Action-Branch Active-Control A4/A4b | 七任务共享 LeWM 能否在真正的 active-control 状态中传播机器人锚点与任务上下文？ | `pick-place-v3 / drawer-retrieve-v0 / reach-v3 / push-v3 / door-open-v3 / window-open-v3 / towel-wipe-v0`；只用当前及历史状态决定采样资格；七任务均衡 batch；TaskContext-Persistent state；仅 `SIGReg=0.02` | **Confirmation-Go（prediction mechanism）**。A4 train/development cache 含 `893` bundles，2k development 相对 no-action/wrong-action 改善 `64.12% / 80.91%`；三个 fresh `10k` seeds 在 A4b test panel 的 `84` 个 physical episodes、`325` 个 accepted bundles 上，逐任务 factual-vs-no-action 相对改善等权平均为 `65.63%`，paired task-macro MSE-difference CI `[0.4604,0.5258]`，`7/7` 任务、`3/3` seed 同向，并胜七种 wrong-action、hold-last 与 CV。未证明 rollout、success、CEM 或缺测闭环 |
| A4b random-hazard interruption screen | 视觉可在任意任务时刻失效时，已确认的七任务 LeWM 能否仅凭最后三次有效接地与失效后已执行动作持续传播 anchor state？ | 冻结三个 A4b `10k` factual/matched-no-action endpoint；每任务 `3` 个 development episode；onset/duration 只由随机种子决定；持续 `2--8` 个模型步；不训练 selector | **Predictor Screen-Go / Towel action boundary**。`21` 个 physical episode、`134` 个随机隐藏决策上，factual 相对 hold/CV/no-action/wrong-action 的 task-macro 改善为 `86.02% / 98.28% / 76.10% / 86.92%`；七任务均胜 hold/CV，六任务胜 no-action/wrong-action。Towel 总体胜 hold `21.2%`，但输 no-action `58.6%`，失败集中于 `wipe_left`。本 screen 使用离线已知 validity，不证明 corruption detection、posterior switching 或 policy success |
| Paired-camera grounding/state probe | FCPG 修正随机/运动相机的像素投影后，Anchor-Persistent state 是否也对视角变化稳定？ | 同一 pre-action physical state 的 dynamic/clean 重渲染；冻结 FCPG 与 `latest@2000`；static random、linear drift、random step；另用 `q_true` 采样作为 target-only oracle | **FCPG pixel mechanism Go / current latent canonicalization No-Go（development）**。原 probe 将瓶颈定位为视觉 token 的 view dependence；后续 fresh paired-view Adapter Gate 中，部署式 residual/FK 条件在 unseen linear drift 仅改善 `0.48%`，random-step post 改善 `12.97%`，且 clean gap 达 `8.58--9.50`。连 privileged camera-transform 上限也只为 `6.34% / 16.46%`，因此不接入 LeWM，不重训 A4 |
| Frozen-LeWM FCPG same-view intervention | 在相同动态图像、相同动作和同一冻结 LeWM 下，仅把当前 grounding 从 nominal 换成 FCPG，是否改善真实未来预测？ | `q_true` 只编码同一视图的当前/未来 oracle target；部署分支只读 nominal 或当前帧 FCPG；H4=`20` raw steps；episode-within-task bootstrap | **Development mechanism Go**。随机 camera step 稳定后，FCPG 将实际未来 latent/anchor-latent MSE 分别降低 `22.53% / 23.28%`，H4 metric EPE 从 `23.150` 降至 `9.968 px`；持续 linear drift 下仍降低 `4.46% / 4.73%`，平均 metric EPE 降低 `23.44%`。两组均为 `3/3` 任务同向且 CI 下界大于零；尚未证明 rollout success，也不授权重新训练 LeWM |
| Frozen FCPG dense-feature bridge | FCPG corrector 中冻结的 ViT-B-backed dense encoder 是否比 LeWM ViT-Tiny 更适合校正后重采样，并可直接替换 Anchor-Persistent 输入？ | 完整 `train_static` 与 unseen linear/step paired-view probe；两套完整 encoder 均冻结；随后做一次从 Tiny@2k warm-start 的 `2,000`-step dense-to-192D compatibility pilot | **feature probe Go / current bridge 10k-expansion No-Go**。FCPG dense encoder 在 unseen linear/step 上的 corrected cosine gain 为 `28.76% / 41.44%`，Tiny 为 `1.39% / 4.51%`。bridge 仍明显 action-sensitive，但 copy-normalized pred error 为 `0.16663`，弱于 Tiny 的 `0.07556`；该 warm-start pilot 不是同总预算 encoder ablation，不能把 raw MSE 差异归因于视觉源。不扩训 10k、不替换 A4、不启动 selector/CEM |
| Camera-frame action 与容量 Gate | 未见相机方位下的像素运动方向是否主要缺少连续相机几何，而不是 LeWM 容量？ | 仅用 `R_cam<-world` 将五个 XYZ 动作变换到 camera frame，夹爪动作不变；不输入离散 camera ID；matched world-action continuation、参数匹配 Adapter 与固定 `10k` 容量对照 | **camera-frame action positive / Adapter No-Go / long-training saturation**。同 parent、同额外 `2k` 预算下，camera-action 相对 world-action 将 odd-view EPE `6.166 -> 5.088 px`、方向 MAE `27.720 -> 23.518 deg`；两个约 `0.78M` Adapter 只改善约 `0.1%`。固定 `10k` 的总体 EPE 为 `4.768 px`，但 latent MSE 与 action-shuffle sensitivity 变差，Towel `clock03` 退化到 `5.763 px / 35.579 deg`，因此容量扩展不能视为跨视角问题已解决 |
| Recursive LeWM-to-FCPG re-centering | LeWM 的 H3 robot-anchor prior 能否在动态相机漂移下为下一次真实观测提供比简单滤波更好的 FCPG 搜索中心？ | 冻结同一 `10k` world-3D LeWM 与 multiview FCPG；相机在 raw step `5--35` 从 `0` ramp 到 `10 deg`；严格比较当前 FK 投影加 residual hold/CV/Kalman | **LeWM-specific Development No-Go / recursive re-centering mechanism positive**。在七任务 `rank=4, clock03` pilot 的 `>=8 deg` 区间，one-shot / LeWM / FK-residual hold / CV / Kalman EPE 为 `14.8839 / 3.4478 / 1.6441 / 1.5608 / 1.6379 px`。LeWM 显著扩大 one-shot 捕获范围，但在 `7/7` 任务均被可部署的简单滤波基线击败，因此不启动 fresh confirmation，也不把该接口写成 LeWM 第二贡献 |
| Frozen LeWM world-goal action ranking | 对 Towel 的八条 same-state 候选动作，冻结 LeWM 的 future world-anchor forecast 是否比解析动作积分更会选择接近当前任务终点的动作？ | task-defined world goal；冻结 `10k` history/world-3D LeWM；六个 odd camera views 先平均到 `12` 个独立 physical snapshots；cyclic wrong-action 与 analytic Cartesian integration | **固定 H3 Development No-Go / 在线完成可行性 Go / 增益未证明**。因果 H6 在线控制在 `5/10 deg` 漂移和随机视觉中断下均完成全部 Towel episode；FCPG 显著降低接地 EPE 并改变候选序列。可是解析中心、nominal、hold-last 与 LeWM prior 都处于成功率天花板，随机中断中 prior 相对 hold 只有 `6/171` 次中断期选择不同，因此只能证明链路可执行，不能声称 LeWM 或 FCPG 带来独立控制收益 |
| A3-to-policy known-drop bridge | A3 的 `192D` anchor state 是否能在视觉暂时缺测时恢复冻结策略实际消费的 `256D` grounded token？ | 冻结策略/FCPG/A3；只覆盖已知无效 slot；privileged current-state ceiling 与 causal A3 prediction 严格分开；恢复时立即重新采用 native measurement | **privileged interface/representation ceilings Go / strict A3-10k-to-P1 bridge No-Go**。Pick native-token oracle 的成功率 `+10 pp` CI 跨零；H1-H8 privileged ceiling 相对 hold-last 改善 `21.39%`，但二者都读取 hidden-clean target。正式 exposed-development bridge 的三个优化 seed 全部未过，代表 seed 的 factual token MSE `0.008357`，弱于 matched no-action `0.007790` 与 exact geometry-hold `0.007040`；未启动 causal rollout/CEM，不是一般性 LeWM No-Go |
| P1-native known-drop dynamics | 去掉 `192D A3 -> 256D P1` 表征转换后，直接在冻结策略原生 grounded-token 空间学习动作条件缺测传播是否足够稳定？ | Pick exposed train/development；H0-H2 可见、H3-H8 隐藏；同容量 factual/no-action heads；hold-last、exact-FK geometry-hold、exact-FK visual-CV 与七种 wrong-action；按 physical episode 统计 | **descriptive dynamics positive / frozen Development-No-Go**。三个 `2k` seed 的 factual 相对 no-action、hold-last、geometry-hold、visual-CV 分别稳定改善约 `28--30% / 23% / 23% / 91%`，H4-H8 与全部 wrong-action 对照均正向；但 H3 均退化约 `37--38%`，hold/geometry 的配对 CI 均跨零，故 `0/3` single-seed Gate 通过。不启动 10k、policy rollout 或 CEM；该结果定位为短期 onset 问题，不外推为 LeWM 一般无效 |
| P1-native geometry-anchored complementary observer | 冻结的动作条件 LeWM 是否能作为 exact-FK geometry baseline 的互补增量，在已知视觉缺测时避免 H3 过度修正并维持策略原生 grounded token？ | zero-training；固定 `K_H3:H8=[.25,.50,.75,.75,.75,.75]`；每步 posterior 递归写回；视觉有效时 bitwise bypass；三个既有 `2k` checkpoint 与所有阈值冻结 | **Fresh-Go（known-drop representation Gate）**。全新 Pick `30` physical episodes、每 episode `8` same-state branches 上，三个 seed 相对 geometry/hold-last 改善 `50.89%--51.63%`，相对 matched no-action 改善 `33.90%--35.25%`；H3 点改善 `20.99%--21.40%`、单侧 95% 下界 `19.49%--19.98%`，H3-H8 为 `6/6` 正向，wrong-action 为 `7/7` 通过，最终 `3/3` seed Go。该 Gate 当时只授权一次 paired rollout；后者现已完成但未通过 policy-success Gate。不授权自动 validity selector、未知 corruption、CEM、10k 扩训或论文第二贡献 |
| P1-native paired known-drop policy rollout | representation Fresh-Go 是否转化为冻结 Pick policy 的配对控制收益？ | `30` 个 fresh physical episode；每 episode 一次已知六决策局部视觉缺测；同一单进程内共享 reset、相机、缺测 schedule、drop-onset physical state 与 clean pre-drop prefix；比较 exact-FK geometry、三个 factual 和各自 matched no-action | **mechanism-positive / policy-success Development-No-Go**。三个 factual seed 相对 geometry 的 action-distance 改善 `40.70%--47.27%`，累计 reward 配对 CI 全部为正；但 success 变化仅 `0 / +6.67 / 0 pp`，seed-macro 为 `+2.22 pp [-10.00,+14.44]`。相对 matched no-action 的 seed-macro success 为 `+6.67 pp [-6.67,+21.11]`。不启动 `240/300` episode confirmation、selector/CEM，也不形成 policy-success 或第二贡献 claim |
| P1-native joint drift + drop confirmation | 当 `5 deg` 相机漂移与六决策视觉中断同时发生时，FCPG measurement 与动作条件 LeWM 增量能否改善非饱和 learned Pick policy？ | 先做 `12`-episode screen，再在代码提交后冻结 `30` 个新 physical episode；已知 validity；一个 policy seed、三个冻结 LeWM seed；相同 pre-drop prefix；clean-action distance 为 primary | **action-fidelity Confirmation-Go / task-control Confirmation-No-Go**。冻结确认中 factual 相对 geometry 的 action-distance 改善 `46.70% [37.45%,54.33%]`，相对 matched no-action 改善 `51.10% [45.10%,56.53%]`，三个 seed 均显著正向；但 reward 与最大连续成功帧的 CI 均跨零，any-step success 相对 geometry/no-action 分别显著下降 `12.22/8.89 pp`。当前接口只确认动作条件状态传播更接近 clean policy，不形成任务控制、自动 interruption detection 或论文第二贡献 claim |
| Causal expert-action residual | 在已知视觉中断期间，冻结 LeWM observer 与 policy latent 能否用一个小型输出头把动作修向 scripted expert，并转化为任务成功？ | expert-forced train/dev；只修正未来五步 XYZ，gripper 原样旁路；测试时无 expert 或 hidden-clean forward；遮挡以 deployable `q_nominal` 为中心；三个 factual/no-action/geometry matched heads | **offline imitation Go / fresh policy-control No-Go**。离线 factual head 相对 base/no-action/geometry 的 XYZ MSE 改善 `57.72% / 30.08% / 25.57%`，CI 均为正；但 `30` 个 fresh Pick episode 上，factual residual 的 sustained-success 相对 factual base/no-action/geometry 为 `-1.11 / 0 / +3.33 pp`，全部未达 `+10 pp`，且没有 head seed 同时胜三类基线。停止扩大该 expert-distillation 接口，不形成 LeWM 第二贡献 |
| On-policy posterior DAgger residual | 旧 expert-forced No-Go 是否主要来自 learner 闭环状态分布未进入训练？ | 每个 decision 从同一 MuJoCo snapshot 分支五步 scripted expert 只生成标签，恢复后执行 learner 五步动作；两轮 method-specific DAgger；三种 head 只修正 XYZ，gripper 与视觉恢复严格旁路 | **covariate-shift repair No-Go / action-conditioning signal positive**。最终 factual head 的离线 XYZ MSE 相对自身 base 改善 `13.68%`，CI 跨零；`12` 个 fresh Pick episode 中 sustained success 为 factual base/factual DAgger/no-action DAgger/geometry DAgger=`58.33/50.00/25.00/41.67%`。Factual DAgger 相对 base 为 `-8.33 pp [-33.33,+16.67]`，未形成新增 head 的控制收益；相对 no-action 为 `+25.00 pp [0,+50.00]`，只保留为单 seed development 动作条件机制信号 |
| Geometry-Anchored Control-Sufficient Dynamics 候选 | 当前 P1 的低平均 token MSE 是否掩盖了冻结策略敏感方向上的预测误差，并可否用几何锚定 observer 与控制相关目标纠正？ | 明确借鉴 Feedback World Model 的 latent velocity feedback 与 counterfactual controllability；FCPG 只提供具有固定物理身份的在线视觉测量，缺测期由 FK geometry carrier 与动作条件 dynamics 传播；冻结 policy/FCPG，先做 zero-training P0 | **P0-No-Go / action mechanism positive**。30 个同状态 physical episodes 上，factual exact action error 相对 matched no-action 改善 `53.87%`，七种 wrong-action 均改善约 `57%--59%`；但 Eq.14-style weighting 的 Spearman 增量为 `-0.0107 [-0.0306,0.0086]`，unit-gain/held-residual analytic observer 相对现有 complementary observer 恶化 `30.63%`，`0/6` horizon 正向。按预注册规则不启动 matched `2k`，不形成论文第二贡献 |
| VisualTrust 恢复拒绝候选 | 当前 FCPG 视觉证据能否判断 measurement 是否可信，从而在遮挡时使用 LeWM prior、恢复后及时交还视觉？ | 仅输入当前 sampled visual 与可部署几何描述；canonical 只生成离线 correctness 标签；不读取 LeWM、动作、schedule 或 validity 标签 | 最小四类随机干扰 development probe 为 **No-Go**。三组优化 seed 的 AUROC 为 `0.9649--0.9668`，但只比 causal temporal-distance 基线高 `0.0198--0.0218`；独立 clean FRR 为 `9.58%--10.22%`，未满足 `5%` safety。未授权 hard-switch posterior |
| Measurement-vs-Prior preference switching 候选 | posterior 应在何时采用 FCPG measurement，何时采用 LeWM prior？ | canonical 只用于 privileged oracle 决策；每种 selector 维护独立递归历史；受约束 FSM 要求连续两个采样事件 prior 严格更优才进入，任一采样事件不再更优即退出 | fixed-schedule 诊断曾为 Diagnostic Go；fresh randomized development Gate 最终为 **No-Go**。按 paired camera view 等权、独立源轨迹重采样后，受约束 FSM 相对 measurement 改善 `8.58% [5.31%,12.02%]`，四任务均正向但未达 `10%`；相对 privileged event-aligned time shortcut 为 `+1.91% [-2.85%,6.49%]`，未证明显著增量。不授权 comparator、policy control 或论文第二贡献 |
| PixelObserver 历史候选 | FCPG 当前像素测量与 LeWM 动作条件先验能否通过 posterior feedback 优于 one-shot/hold-last？ | 冻结 `10k` LeWM base；动态相机；小型 residual adapter；posterior writeback | v2 development 已完成并判为 **No-Go**。四任务 cross-corrector 复验中，latent-on 相对参数匹配 latent-off 在 `12/14` 条件显著退化；另外两个 hidden-step 正向格属于固定时刻、事件不可观测的负对照。latent-off full loop 相对 hold-last 的严格 primary 统计为 `0/14` 胜、`8/14` 负、`6/14` 平。未进入 confirmatory、action ranking 或控制 |
| Counterfactual v2 动作排序候选 | FCPG-grounded LeWM rollout 是否含有强 `history + action` DirectRanker 没有的控制信息？ | 冻结 FCPG/LeWM；真实物理 branch return；输出端 CostAdapter；Direct/Predicted/Oracle 参数与输入合同匹配 | **当前 v2 接口为 external-development No-Go**。唯一一次 horizon-structured 复验中，Predicted 在 drift 下显著弱于 Direct；但当前 context 只是当前帧全局 ViT prefix 的固定投影，Oracle 也只是 oracle-anchor future visual latent，候选 bank 还存在明显的 candidate-family shortcut。因此该结果不支持 CEM 或 LeWM 控制 claim，也不能外推为显式 goal-conditioned CostAdapter 或整个 LeWM 路线普遍无效 |
| Counterfactual v3 StateOracle 修复 Gate | 去掉伪 goal、候选类别捷径和非嵌套比较后，当前候选与时域是否暴露对冻结物理 task-cost 有用、且与 H5 actual return 对齐的 future-state 信息？ | 只加载冻结 policy/FCPG 采集动作；真实 simulator state 与 `goal_world` 仅用于 privileged diagnostic；本 Gate 不加载 LeWM | **正式 P0 No-Go 保持不变。** 后续一次 actual-return post-hoc cross-fit 诊断中，Direct 明显超过 axis/sign prior，但 privileged future-state residual 将 validation Spearman 从 `0.6314` 降至 `0.5213`、regret 从 `0.1735` 恶化到 `0.2176`，当前 nested 接口同样为 descriptive No-Go；未启动 VisualOracle/LeWM/GoalToken/FCPG-vs-nominal/CEM |
| Stage II：KineParity 历史候选 | 如何用冗余运动学观测区分共享相机漂移与局部视觉错误，并只校正策略关心的接地点？ | observer/control 分离；有限历史全臂星座；task-oriented 白化 MAP/TSVD；按有效秩自适应的 innovation gate；共享 twist 传播；校正位置重采样 | Oracle geometry Gate A 已通过；Gate B 首轮与唯一一次 0-8 deg frozen rescue 均未过预注册门槛，最终判为 No-Go；feature resampling / control Gate C 取消 |

### 冻结训练预算

| 实验 | 冻结预算 | 状态 | 是否为当前待执行命令 |
| --- | ---: | --- | --- |
| 旧 full grounded LeWM Gate L1 | `2 x 10,000` steps | completed / No-Go | 否 |
| correction-aware anchor-input ablation | `4 x 2,000` steps | completed / No-Go | 否 |
| PixelObserver online GroundedLeWM base | `1 x 10,000` steps | completed；冻结 `latest@10000`，SHA-256 `49f252eb526a61aad0bf29b9b3400b1b2bbfa2b7bb83e35164d716f9315435a9` | 否 |
| PixelObserver v2 development adapters | `6 x 2,000` steps | completed；四个单任务诊断端点与两个四任务 cross-corrector 端点均严格停在 step `2000`；development No-Go | 否 |
| Counterfactual v2 flat scorers | `4 probes x 3 seeds x 2,000` steps | completed；`24,000` scorer steps；E1 Fail、E2 仅 broad-bank development 正向、E3 mixed | 否 |
| Counterfactual v2 structured scorers | `3 probes x 3 seeds x 2,000` steps | completed；`18,000` scorer steps；唯一一次结构救援为 No-Go | 否 |
| Counterfactual v3 StateOracle Gate | `12 episodes/task x 4 tasks`；P0 通过后才允许 `3 Direct seeds x 2,000` steps | completed；P0 No-Go，故 Direct `0` steps，未加载或训练 LeWM | 否；唯一一次已执行并冻结 |
| Counterfactual v3 actual-return post-hoc | naive `12,000` + fixed 4-fold cross-fit `36,000` scorer steps | completed；暴露 validation 上的 descriptive diagnostic；cross-fit nested residual 相对 Direct 明确退化 | 否；当前接口 No-Go，不得据此启动后续表征层级 |
| Anchor-Persistent LeWM v1 | `1 x 2,000` steps；batch `192`；seed `9216`；`SIGReg=0.02` | completed；冻结 `latest@2000`，SHA-256 `a6c329d88692810e852685be1460569f6da28753b8d672468b20538811b9c092`；dynamics Go，旧 recovery 证据 mixed | 否；不得重训、挑 best 或扩到 `10,000` |
| Frozen FCPG dense input bridge | `1 x 2,000` adaptation steps；batch `192`；seed `9216`；`SIGReg=0.02`；从 Anchor-Persistent Tiny@2k warm-start | completed；normalized pred/copy `0.16663` vs Tiny `0.07556`，normalized shuffle-delta/copy `0.87551` vs `1.03203`；10k expansion No-Go | 否；不得扩到 `10,000`、替换 A4 或启动控制实验 |
| World-anchor deterministic-projection probes | `4 x 2,000` readout steps；冻结 shared orbit LeWM；`SIGReg=0.02` parent | completed / No-Go；最佳 `5.55M` readout 为 `6.519 px`，未胜同评估二维 base `6.382 px`，cross-view world prediction std 仍为 `19.51 mm` | 否；不加宽、不扩到 `10,000` |
| Camera-frame action capacity screen | matched world/camera action 各 `1 x 2,000`；bounded Transformer/MLP 各 `1 x 2,000`；camera full-model continuation `1 x 2,000`；batch `84`；仅 CUDA0；`SIGReg=0.02` | camera-frame coordinate alignment positive；两个 Adapter No-Go；无参数 continuation 将 EPE `5.088 -> 4.788 px`，但目标 Towel 视角退化 | 否 |
| Camera-frame action fresh 10k | `1 x 10,000`；seed `40571`；batch `84`；同 parent/data/schedule；`SIGReg=0.02` | completed；固定 `latest@10000` 总体 EPE `4.768 px`，但 Towel `clock03` 与 latent/action-sensitivity 指标退化；capacity rescue No-Go | 否；不再扩训或加 Adapter |
| Anchor-Persistent posterior headroom | zero-training；冻结同一 `latest@2000`；`seed=9517` 暴露数据 | completed；target-only canonical upper bound，不允许据此训练 | 否；只为一次 fresh Adapter Gate 提供设计依据 |
| Anchor-Persistent posterior Adapter Gate | `3 modes x 5 seeds x 2,000` steps；batch `512`；冻结 LeWM/FCPG | completed；五个 seed 主指标强正，但无 safety-qualified endpoint，最终 No-Go | 否；不得在该 development split 上调阈值或继续扫 Adapter |
| VisualTrust minimal probe | visual 与同参数 auxiliary-only 各 `3 x 800` steps；batch `4096`；冻结 FCPG/ViT/LeWM/policy | completed；visual discrimination 强，但 baseline margin 与 clean/recovery safety 未通过 | 否；不启动 posterior writeback |
| Recursive preference oracle diagnostic | zero-training；冻结 LeWM/FCPG 与现有 posterior cache | completed；teacher-forced 数值复现、七种独立递归 selector 与 exact sequence/FSM 上限均已评估 | 否；只授权 fresh randomized collection，不授权 learned comparator |
| Fresh randomized preference oracle Gate | zero-training；calibration/evaluation 为 `40/80` 个相机视图、对应 `20/40` 条独立源轨迹；冻结 LeWM/FCPG | completed；四任务、paired step/linear camera、随机 anchor/onset/duration/corruption/severity 与 episode-relative onset；最终 No-Go | 否；当前 comparator 路线冻结 |
| Random-Hazard predictor isolation A0 | zero-training；evaluation 为 `80` 个配对相机视图、对应 `40` 条独立源轨迹；冻结 LeWM/FCPG | completed；factual LeWM 显著优于四个最小 predictor/动作消融，判为 A0 Go | 否；只授权新数据上的完整 Gate A，不授权 selector 或控制 claim |
| Random-Hazard matched-action A0.5 | `2 heads x 2,000` steps；batch `192`；seed `17301`；冻结 anchor encoder | completed；单步 factual 明显胜 matched no-action，但递归 task-macro CI 跨零，判为 No-Go | 否；不在该 split 调参，不授权 action-necessity claim |
| Action-Branch Random-Hazard A1-D | `2 heads x 3 seeds x 2,000` steps；batch `16 bundles`；冻结 `SIGReg=0.02` anchor encoder | completed；`450` bundles 的 same-state branch store，screen 仅失败于 Button intervention support | 否；预注册 `10k` 需要 screen Go，当前禁止启动且 independent test 保持未打开 |
| Action-Branch Continuous-Task A2-D | `2 heads x 3 seeds x 2,000` steps；batch `16 bundles`；FP16；冻结 `SIGReg=0.02` anchor encoder | completed；H6-H8 development 为强 Go | 否；fresh confirmation train support 未过，`10k=0` steps，test 未采集 |
| Action-Branch Continuous-Task A3 | `3 seeds x 2,000` development + `3 seeds x 10,000` confirmation；每个 seed 为一个四任务共享 LeWM；batch `16 bundles`；FP16；仅 `SIGReg=0.02` | completed；reserved-test **Confirmation-Go**，factual 相对 no-action/wrong-action 改善 `74.01% / 88.18%`；后验 active-control 审计将 claim 收窄为机器人 anchor transition | 否；不得追加 seed、挑 best、扫 SIGReg 或把机制结果改写为完整 task dynamics/control 结果 |
| Action-Branch Active-Control A4/A4b | `3 seeds x 2,000` development + fresh-start `3 seeds x 10,000` confirmation；每个 seed 为一个七任务共享 LeWM；batch `28=7x4`；FP16；仅 `SIGReg=0.02` | completed；旧 A4 fixed panel fail closed；独立 A4b fresh panel 已完成并判 **Confirmation-Go**，相对 no-action 改善 `65.63%`，`7/7` 任务、`3/3` seed 同向 | 否；固定 endpoints 与 A4b test 已封存，不追加 seed、不挑 best、不据此启动 CEM |
| P1-native known-drop dynamics | `2 matched heads x 3 seeds x 2,000` steps；batch `8` physical bundles、每 bundle 全部 `8` branches；CUDA FP16；无额外 SIGReg | completed；三 seed 均胜 no-action/hold/geometry/CV 点估计，但 H3 与 strong-baseline CI Gate 失败，cross-seed `0/3` No-Go | 否；不扩训 10k、不做 policy rollout/CEM、不在当前 split 调参 |
| P1-native complementary observer fresh Gate | zero-training；冻结三个 `latest@2000` factual/no-action pairs、固定增益与阈值；Pick `30` fresh physical episodes、`8` branches、H3-H8 | completed；`3/3` seed Fresh-Go，胜 geometry/hold/no-action，H3-H8 `6/6`、wrong-action `7/7` | 否；representation Gate 已完成，后续 paired rollout 未通过 policy-success Gate |
| P1-native paired known-drop policy rollout | 冻结 policy/FCPG/P1 heads；Pick `30` fresh paired physical episodes；单进程共享 drop-onset physical state；三个 factual/matched-no-action seed | completed；action-distance 与 reward 为稳定正向机制证据，但 seed-macro success CI 跨零，Development-No-Go | 否；不启动大样本 confirmation、selector/CEM 或第二贡献 |
| Causal expert-action residual | `3 methods x 3 seeds x 2,000` steps；`30/15` expert-forced train/dev episodes；冻结 policy/FCPG/P1；随后 `30` 个 fresh rollout episode | completed；离线 expert XYZ imitation Gate 通过，fresh sustained-success Gate 为 **No-Go** | 否；不扩训、不追加 episode、不改 success primary |
| On-policy posterior DAgger residual | `2` DAgger rounds；每轮每方法 `12/6` train/dev episodes；`3 methods x 1 seed x 2,000` steps；随后 `12` 个 fresh paired rollout episodes；仅 CUDA0 | completed；真实 learner-state collection 与上一轮 head 合同通过；factual 对 no-action 有正信号，但未超过 factual base，最终 **No-Go** | 否；不追加 round、seed、episode 或换 primary 救援 |
| Geometry-Anchored Control-Sufficient Dynamics P0 | zero-training；复用冻结 P1 cache/checkpoint、same-state clean-policy carrier 与八分支动作；只生成离线诊断 | completed / **P0-No-Go**；动作条件 control 全部通过，但 weighted metric 与 analytic observer 两项核心 Gate 均失败 | 否；结果冻结，不改阈值或在同 panel 上救援 |
| Geometry-Anchored Control-Sufficient Dynamics matched 2k | 仅在 P0 Go 后：相同 P1 architecture/data/init/optimizer/seeds/`2,000` steps，比 uniform-MSE、controllability-weighted、policy-distilled 三组 | **not authorized**；P0 未通过，实际训练步数为 `0` | 否；不得启动 rollout、10k 或控制 claim |

旧 G0 的 base/adapter 预算在结果产生前已冻结；该协议与命令现仅作历史审计记录，不再是当前执行入口，也不得用旧 Gate L1 checkpoint 冒充 online base。

### Anchor-Persistent LeWM v1：状态粒度修复 pilot

官方 LeWorldModel `v3` 的 2026-06-03 代码与本仓库 vendored commit 均为 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；当前没有另一份需要替换的“6 月 3 日新权重”。本 pilot 不修改上游 `ARPredictor`、action embedding 或 `SIGReg`，只修复本项目旧 GroundedLeWM 接口中的状态粒度损失。

旧接口先将 `global visual + proprio + K anchors` 汇总成每帧一个 `192D` token，再做时序预测；后续 selector 又对历史与动作取均值。因此旧 No-Go 只能否定该 summary-pooled 接口。新接口的数据流固定为：

```text
RGB + proprio/FK + current FCPG anchors
-> joint frame encoder with cross-anchor interaction
-> retain K identified anchor slots as [B,T,K,D]
-> shared action-conditioned ARPredictor per anchor
-> predicted anchor prior [B,H,K,D]
-> current/candidate anchor measurement - predicted prior
-> explicit per-anchor innovation features
```

时间轴与锚点轴在预测前均禁止平均；同一帧内仍允许 joint frame encoder 进行跨锚点交互。训练目标为：

```text
L = MSE(z_anchor_pred, z_anchor_next)
  + 0.02 * mean_k SIGReg(z_anchor[:, :, k, :])
```

相机 step jump 是外生扰动，历史动作无法在新图像出现前预测它。因此本模型不得声称预测相机跳变：FCPG 从当前图像产生 measurement，LeWM prior 只建模机器人动作与任务动力学，二者的差形成 candidate-level innovation。当前动态 evaluator 不读取 `q_true`，只检查 factual action prior、跨 episode shuffled-action prior、copy prior，以及同一 prior 到 FCPG/nominal 当前测量的距离。

固定 `2,000`-step development Gate：

1. clean validation 上 factual prediction 必须优于 copy，并且显著使用动作，即 `shuffled_mse - factual_mse > 0`；
2. held-out linear drift、`5 deg` step rotation 与 `4 cm` step translation 上仍应保持正的 action-shuffle 与 copy delta；
3. 同一 prior 到 FCPG measurement 的误差应低于到 nominal measurement 的误差，且逐锚点 closer rate 应高于随机水平；该项只诊断 posterior 接口，不是标定或控制 claim；
4. 数值必须有限，clean 不能通过表征塌缩获得虚假低误差；若未通过，不延长训练、不扫 `SIGReg`、不启动 selector/CEM；
5. 原有动态集没有显式视觉遮挡标签，只判 drift/jump；遮挡恢复使用全新 schema-v2 visual-occlusion 数据独立验证，dropout/recovery mask 仅用于分段统计，不作为模型输入。

执行入口：

```bash
cd /home/james/feedback-corrected-grounding
env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/james/lewm/.venv/bin/python3 \
  scripts/train_lewm_grounded.py \
  --config config/lewm/train/anchor_persistent_sig002_seed9216_2k.yaml \
  --device cuda:0

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/james/lewm/.venv/bin/python3 \
  scripts/evaluate_lewm_anchor_persistent.py \
  --checkpoint outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/lewm_grounded_latest.pt \
  --training-validation \
  --device cuda:0 \
  --output outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/clean_validation_v2.json \
  --expected-step 2000

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/james/lewm/.venv/bin/python3 \
  scripts/evaluate_lewm_anchor_persistent.py \
  --checkpoint outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/lewm_grounded_latest.pt \
  --config config/lewm/eval/anchor_persistent_dynamic_dev.yaml \
  --device cuda:0 \
  --output outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/dynamic_eval_v2.json \
  --expected-step 2000

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/james/lewm/.venv/bin/python3 \
  scripts/evaluate_lewm_anchor_persistent.py \
  --checkpoint outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/lewm_grounded_latest.pt \
  --config config/lewm/eval/anchor_persistent_occlusion_dev.yaml \
  --device cuda:0 \
  --output outputs/lewm_grounded/anchor_persistent_sig002_seed9216_2k/occlusion_eval_v2.json \
  --expected-step 2000
```

正式 `latest@2000` endpoint 与 schema-v2 evaluator 结果如下。

| Source | Factual MSE | Copy MSE | Strict shuffled MSE | FCPG vs nominal intervention | all-anchor effective rank / PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean training-validation split | `0.017840` | `0.236096` | `0.247548` | `+33.59%` | `14.62 / 12.42` |
| held-out linear drift, `2--4 cm` | `0.015806` | `0.178093` | `0.188799` | `+2.84%` | `12.74 / 10.38` |
| held-out step rotation, `5 deg` | `0.129905` | `0.278940` | `0.267915` | `+4.00%` | `10.85 / 7.38` |
| held-out step translation, `4 cm` | `0.092363` | `0.271630` | `0.319213` | `+0.55%` | `13.34 / 11.17` |
| held-out `5 deg` + visual occlusion | `0.189248` | `0.346862` | `0.320823` | `-7.44%` | `9.88 / 6.51` |

严格 action shuffle 只在同任务、不同 episode 之间置换；所有 source 的 overall factual 都优于 copy 与 shuffle，且逐锚点表示未塌缩。遮挡集为全新 `seed=9517` 的四任务各 `5` episode，在相机 step 后真实遮挡 `rightEndEffector` 共 `5` 帧；`q_true` 只在离线渲染时定位遮挡区域，模型不读取它。恢复分段结果为：

- recovery onset：factual `0.951940`，copy `1.028692`，strict shuffle `0.943405`；
- full recovery segment：factual `0.125084`，copy `0.122900`，strict shuffle `0.171586`。

因此本 pilot 证明 Anchor-Persistent 接口学到了逐锚点、动作敏感且在动态相机条件下不塌缩的表征动力学。旧 recovery 协议没有通过其合取标准，该结果保持冻结；但它不能外推成一般性的 recovery mechanism No-Go：onset 同时包含不可由历史动作预测的外生相机 step，full segment 又被恢复 `15` 帧后的近静态尾段主导，且原切分不是独立确认。准确边界为 **dynamics mechanism Go / legacy recovery Gate failed / mechanism evidence mixed-inconclusive / 10k expansion unauthorized**。这不允许延长训练、扫描 `SIGReg` 或启动 selector/CEM，但允许先做零训练的 posterior 可融合性上限诊断。

正式报告及数据哈希：

```text
clean_validation_v2.json  b2b7bc28e5dd98df654f3e418401a0c41a377e5d71d18459f66b0e445bf02a78
dynamic_eval_v2.json      98b9fe284ea194109067333ce04ddcd4cb057a6ccaf2dcd1efaaccef42a23db0
occlusion_eval_v2.json    f0d16915269385b091753a16e7827802ecf8de4f41da4b05f7f39bd8b31cdf93
occlusion dataset         fce447de53f0b7cb1a62df420102cf46898302d87b5a872b2829e233dd76aa2f
```

#### Zero-training posterior headroom（暴露数据诊断）

该诊断冻结同一 `latest@2000`，不训练 Adapter。部署测量使用遮挡后的 `rgb` 与 deployment FCPG correction；privileged target 使用同一物理帧的未遮挡 `rgb_camera_only` 与独立 target-only correction cache。canonical RGB、canonical correction 和 target latent 只用于离线评分及 oracle upper bound，禁止进入部署输入或递归历史。缓存 CLI 已显式记录 `image_key`、图像 SHA 和 `deployment_observation/target_only` 角色；target-only cache 不具备 deployment eligibility。

现有遮挡持续 `5` 个 raw frame，恰好形成每个 frame-skip phase 的一个 LeWM state。因此它只能检验五条 phase stream 各自的 masked state 和一次写回后的下一 pre-measurement prior，不能把它们称为五个独立 raw `o0/r0`，也不能证明 `o1/o2` 多步恢复。四任务各 `5` 个 physical episode、每 episode `5` 个 phase，共 `100` 个事件；统计先把 phase 降维到 physical episode，再以 episode 为 cluster。primary scope 只包含直接遮挡的 anchor 1；其余 anchors 只作 secondary，且 posterior 写回仅替换 dropout-active anchor。

直接遮挡 anchor 的结果如下，MSE 越低越好：

| phase-stream 时点 | LeWM prior | 当前 measurement | fixed `alpha=0.75` | oracle convex fusion |
| --- | ---: | ---: | ---: | ---: |
| phase-stream masked | `1.91891` | `1.02369` | `0.95071` | `0.74601` |
| next pre-measurement prior | `1.97449` | `0.95312` | `0.89616` | `0.72291` |

相对 measurement，phase-stream masked 的 fixed-alpha 改善为 `0.07299 [0.01012,0.13519]`，oracle convex 改善为 `0.27768 [0.22230,0.34269]`；下一 prior 的 fixed-alpha 改善为 `0.05696 [-0.00449,0.12167]`，oracle convex 改善为 `0.23021 [0.17494,0.29565]`。下一 prior 才是当前更有信息量的 endpoint，而 fixed-alpha 的区间跨零；`push-v3` 在 masked/next 上还分别退化 `-0.0799/-0.0942`。oracle gain 在直接遮挡 anchor 上均值 `0.6839`、中位数 `0.8882`，但 `16%` 事件接近 prior endpoint、`36%` 接近 measurement endpoint，说明单一固定增益不能覆盖不同任务和事件。

这组结果的含义不是“LeWM prior 能替代视觉测量”。恰恰相反，`100/100` 个 history 都跨越相机 step，冻结 prior 明显差于当前 measurement。可检验的新假设只能是 **measurement-dominant adaptive posterior**：以当前 FCPG measurement 为主状态，LeWM prior 仅提供受限、可拒绝的增量。

```text
z_post = z_measurement
       + beta_theta(confidence, innovation, mask_history, action)
       * (z_prior - z_measurement)
```

`beta` 必须有界且末层零初始化；部署 confidence 只能来自当前图像的匹配响应、margin 或 innovation，不能读取 `q_true`、预设遮挡 schedule 或 `measurement_valid` 标签。该暴露数据上的结果为 upper-bound diagnostic，`formal_claim_allowed=false`、`adapter_training_authorized=false`，不得在 `seed=9517` 上训练或调 Gate。

fresh Gate 不学习任意 latent residual，也不使用逐维 gain；单个共享小网络只为每个 anchor 输出一个标量 `beta`。训练损失预先固定为五个等权组：直接遮挡 anchor 的 `o0/o1/o2` 各 `20%`，直接恢复 `20%`，全部间接受影响 anchor `20%`；组内按数据行均衡采样。这样 `o1/o2` primary 占 `40%`，不会被恢复与未遮挡 anchor 的低误差淹没。clean `<2%` safety 使用独立 physical episode，只把同一动态相机下未遮挡的 canonical 视图当作部署观测进行评估；该分支不参与 Adapter forward 的 primary 训练。

本次 fresh Gate 使用 fresh episode 与全新遮挡时序：先让 camera step 稳定，再提供至少 `15` 个 raw-frame 的测量破坏，使 primary 覆盖直接遮挡 anchor 真正受历史影响的 `o1/o2` post-update posterior；恢复只报告有限 `r1` 窗口，不纳入静态尾段。Adapter 限制为每 anchor 共享的标量 gate、单层 hidden `<=32`、总参数约 `20k` 以内，冻结 LeWM/encoder/FCPG，禁止 task ID、oracle validity、逐维 gain 和任意 latent residual。正式必要对照为：measurement-only、验证集选择的 fixed-alpha，以及 `no_lewm`（hold-last prior，但保留与 full 完全相同的真实动作输入）；`action_blind` 单独屏蔽 Adapter 的直接动作输入，`action_shuffle` 只作诊断，不再把两个因素混成一个对照。只有 Adapter 相对 raw measurement 至少改善 `10%` 且 episode-level 95% CI 下界大于零，同时胜最佳 fixed-alpha 与 `no_lewm`、各 task/cohort 不退化超过 `5%`，并且独立 clean cohort 的附加误差严格小于 perturbed-primary measurement MSE 的 `2%`，才允许进入控制实验。

#### Fresh posterior Adapter Gate（2026-07-16，mechanism positive / Gate No-Go）

fresh 数据覆盖四个任务：训练 `30` episode/task，受扰验证与独立 clean-safety 各 `10` episode/task。相机先发生固定 `5 deg` step，随后 anchor 1 被视觉遮挡 `15` 个 raw frame；冻结 FCPG、冻结 Anchor-Persistent LeWM 和全部阈值后，训练 `full / action_blind / no_lewm` 三种模式、五个优化 seed、每个 `2,000` steps。训练、受扰验证和 clean 验证分别形成 `3,000 / 1,000 / 1,000` 个五事件时序样本。

直接遮挡 anchor 的 `o1/o2` post-update 主结果很强：measurement MSE 为 `0.89556`，五个 full endpoint 为 `0.06063--0.08388`，相对改善 `88.2%--89.8%`，五个 seed 全部正向。最佳 seed 的 `no_lewm` MSE 为 `0.36903`，full 相对其再降低约 `81.8%`；action-shuffle 诊断为 `0.14285`，也明显差于 factual full `0.06063`。因此结果支持逐锚点 learned transition prior 相对 hold-last 的遮挡期增量，并证明当前模型对动作错配敏感；但 A0.5 的容量匹配 no-action 对照未建立稳定递归优势，所以这里不能进一步声称 factual action 的因果必要性。

但没有 learned endpoint 通过安全筛选。五个 seed 的 `r1` 恢复附加误差为 perturbed-primary measurement scale 的 `3.93%--4.30%`，单侧 95% 上界为 `8.47%--8.88%`，超过冻结的 `2%` 门槛。独立 clean cohort 的点估计为 `1.59%--1.73%`，但单侧上界为 `3.17%--3.30%`，同样未过门。相比之下，验证集选择的固定 `beta=0.25` 通过安全检查，但主误差只改善 `13.0%`。准确结论是：**当前 learned posterior 能利用 LeWM 在遮挡期大幅修复状态，却不能在视觉测量恢复后足够快地拒绝 prior。** 因此 deployable posterior Gate 判为 **No-Go**，不启动控制实验，也不把该路线列为论文第二贡献。

当前开发集不得继续调 loss、阈值或网络宽度。若未来以新接口重开，必须先加入可部署的视觉测量置信度或显式恢复拒绝机制，并在新的 physical episode split 上验证；`measurement_valid`、遮挡 schedule 和 canonical target 仍不得进入模型输入。正式结果位于：

```text
outputs/lewm_posterior_adapter_gate/measurement_dominant_v1_seed12091/
```

#### VisualTrust 最小恢复拒绝 Probe（2026-07-16，Development No-Go）

该 probe 冻结 FCPG、ViT、LeWM 与 policy，仅训练一个 `9,007` 参数的逐锚点 head。部署输入只有当前 corrected-location `sampled_visual` 与七维当前几何描述；未遮挡 canonical feature 只用于构造“最终 grounded evidence 是否偏离”的离线标签，不进入模型。训练、development 与独立 clean 分别包含 `120/40/40` 个物理 episode，身份互斥；四类随机干扰为 square/irregular occlusion、texture distractor 和 local blur。三个 seed 只表示优化与 minibatch 稳定性，不表示三个独立数据 draw。

纠错后的 evaluator 同时比较同参数 auxiliary-only、correction-norm 与 causal previous-feature-distance。三组 seed 的 VisualTrust AUROC 为 `0.9649/0.9668/0.9651`，unsafe recall 为 `0.9111/0.9123/0.9123`；同参数 auxiliary-only AUROC 仅 `0.4487/0.4471/0.4506`，证明主要信号来自视觉。四类 family AUROC 下界为 `0.9371--0.9408`。但 causal temporal baseline 已达 `0.9451`，VisualTrust 的额外 AUROC 只有 `0.0198--0.0218`，低于冻结的 `+0.03` 门槛。

安全条件也未通过：独立 clean FRR 为 `10.22%/9.99%/9.58%`，最差任务为 `14.77%/13.60%/12.43%`；local blur 的 early-reopen 为 `22.5%--23.75%`。全部 `40/40` 恢复事件确实先经历 unsafe 且 recovery window 有效，整体 detection rate 为 `97.5%`；两帧确认按在线可用时刻计算，中位延迟为 `1` 帧，但 texture distractor recovery rate 仅 `88.9%`。因此准确结论是：**当前绝对单帧 visual feature 能识别测量破坏，却不能以可迁移阈值安全决定何时切回 measurement。** 本结果只否定这一最小 current-only 接口；不否定 LeWM 遮挡期 prior，也不授权 hard-switch、posterior writeback、控制实验或在同一 development draw 上调阈值。正式输出为：

```text
outputs/visual_trust_probe/development_seed13002_v2_corrected_run2/
```

#### Measurement-vs-Prior recursive oracle（2026-07-16，fixed-schedule Diagnostic Go）

该诊断不训练新 head。它冻结同一 FCPG、Anchor-Persistent LeWM 与 posterior cache，并让 `measurement_writeback / prior_writeback / oracle_hard / oracle_convex / oracle_schedule / oracle_sequence_exact / oracle_fsm_cautious_enter_fast_exit` 各自维护独立递归历史。只有直接受遮挡的 anchor 可切换 source；其余 anchor 始终使用 measurement。canonical target 只用于离线 privileged 决策与评分，绝不进入部署 runtime。

首先复现 teacher-forced factual-prior 数值。这里使用全局 direct-anchor、逐事件比值：hard selector 在 `o1/o2` 分别改善 `13.29%/10.03%`，continuous convex 分别改善 `13.67%/10.17%`；池化后 hard 保留 continuous 收益的 `97.85%`。后续 recursive 表格改用 task-macro、physical-episode bootstrap 估计量，两组数不能按绝对幅度直接比较。teacher-forced 结果只说明 source selection 值得进入递归检查，不能证明写回后仍安全。

递归结果如下。`oracle_hard` 和 `oracle_convex` 每个采样事件读取当前 canonical；`oracle_sequence_exact` 枚举五事件窗口的全部 `32` 条 hard-switch 路径，读取完整未来，因此三者都是不可部署上限。受约束 FSM 仍使用 privileged 当前比较，但只在 prior error 连续两个采样事件严格小于 measurement error 后进入；任一采样事件 prior 不再严格更优（包括平局）就立即退出。当前采样事件相隔 `5` 个 raw frames，所以该结果不能解释为逐帧“两帧进入、一帧退出”的实际延迟。它是本 Gate 唯一用于判定受约束递归可行性的 selector。

| Recursive selector | `o1/o2` 相对 measurement 改善（95% CI） | Recovery one-sided upper | Clean one-sided upper | 解释 |
| --- | ---: | ---: | ---: | --- |
| always prior | `89.87% [86.86%,92.23%]` | `25.00%` | `13.16%` | 遮挡强，但恢复与 clean 不安全 |
| privileged hard | `89.87% [86.86%,92.23%]` | `0%` | `0%` | 当前 canonical oracle 上限 |
| privileged convex | `91.42% [89.51%,93.01%]` | `0%` | `0%` | 连续融合 oracle 上限 |
| fixed schedule | `89.87% [86.86%,92.23%]` | `0%` | `5.09%` | 暴露固定时序捷径，clean 不安全 |
| hindsight exact sequence | `89.87% [86.86%,92.23%]` | `0%` | `0%` | 五事件完整未来上限，不因果 |
| cautious-enter / fast-exit FSM | `21.38% [17.71%,25.16%]` | `0%` | `0%` | 受约束递归可行性上限 |

因此当前结论是 **Diagnostic Go / Formal Gate NotEvaluated / formal claim disallowed**。受约束 FSM 的遮挡改善超过 `10%` 且 CI 排除零；其 recovery/clean excess 为零，是因为 privileged rule 在当前 canonical 下发现 prior 不优时立即选择 measurement，属于按构造成立的可行性上限，不能当作恢复检测证据。当前 cache 的遮挡 anchor、onset、duration 和 event schedule 都固定，不能验证可部署 selector，也不能直接训练 comparator。`oracle_sequence_exact` 最小化的是完整五事件流的累计误差，而不是单独针对 `o1/o2` primary 求最优。

后续 fresh randomized development split 随机化 anchor、onset、duration、corruption family、severity、camera trajectory 和 episode-relative onset，并加入 time-only/schedule shortcut。若未来另开 comparator，其离线 target 使用归一化 advantage

```text
A_norm = (E_measurement - E_prior) / (E_measurement + E_prior + eps)
```

配合 dead zone，小差异默认 measurement。部署输入只能包含 measurement、prior、innovation、局部视觉证据、因果历史与动作；canonical、预设时序和 `measurement_valid` 只能生成离线标签。fresh randomized oracle 再次通过前，`authorizes_learned_comparator_training=false`，不得启动 policy control。正式诊断输出为：

```text
outputs/lewm_posterior_preference_oracle/current_split_diagnostic_v4/
```

#### Fresh randomized preference oracle（2026-07-16，Development No-Go）

fresh Gate 使用与历史 cache 不重叠的轨迹集合。Calibration/evaluation 分别包含 `40/80` 个 camera-view episode；每条机器人源轨迹各有一个 step view 和一个 linear-drift view，因此真正独立的 calibration/evaluation 单元是 `20/40` 条源轨迹，即每任务 `5/10` 条。配对视图的 frame、robot observation、action 和三维关键点完全一致，只改变相机轨迹及随机 corruption draw。每个 view 随机 corrupted anchor、onset、`10/15/20` 帧 duration、corruption family/severity 与 episode-relative onset；这里没有语义 task-phase 标签。

Calibration 拟合 `normalized full-episode time + frame phase + anchor identity` 表，但该 control 只在真实 corruption event stream 上被调用，且 direct anchor 由离线 dropout support 路由。因此它准确地说是 **privileged event-aligned time shortcut**，不是完整 episode 上可直接部署的 time-only baseline。Canonical 逐事件直接决定 privileged oracle 的 source selection 并用于离线评分；它不进入任何部署输入。paired clean 使用同一物理帧和同一 camera trajectory 的 `rgb_camera_only`，仅移除局部 corruption，并使用对应 canonical correction。

七种递归方法分别维护自己的 posterior history：measurement、always-prior、privileged hard、privileged convex、privileged schedule、受约束 cautious-enter/fast-exit FSM 和 calibration-only event-aligned time shortcut。主统计排除 active event `o0`，按 `phase -> event -> camera view -> source trajectory -> task` 聚合；step/linear view 先在同一源轨迹内等权平均，bootstrap 只重采样每任务 `10` 条独立源轨迹。

| Development 结果 | 相对 measurement 改善（95% CI） | 解释 |
| --- | ---: | --- |
| always prior | `75.73%` | LeWM prior 在 corrupted anchor 上有强机制信号，但不具备恢复安全性 |
| privileged hard | `87.41%` | 逐事件读取 canonical 的不可部署 source-selection 上限 |
| privileged convex | `89.22%` | 逐事件读取 canonical 的不可部署连续融合上限 |
| cautious-enter / fast-exit FSM | `8.58% [5.31%,12.02%]` | 四任务均正向，但低于固定的 `10%` development 门槛 |
| event-aligned time shortcut | `6.67% [2.36%,11.24%]` | calibration-only；paired views 先等权，但评估仍调用已知 event support 与 direct anchor，属于 privileged shortcut |
| FSM incremental over time shortcut | `+1.91% [-2.85%,6.49%]` | 点估计转正，但区间跨零，没有证明受约束 oracle 提取出超越事件时序捷径的稳定增量 |

FSM 的逐任务改善为 button `9.18%`、pick `8.70%`、push `14.44%`、reach `2.01%`，方向条件通过。相对 time shortcut 的逐任务增量只有 button/pick 为正，push/reach 为负。Privileged fast-exit 在 paired recovery/clean 上的 normalized excess one-sided upper 均为 `0`，通过递归可行性安全门槛；但这是 canonical-aware oracle 在 measurement 恢复后按构造立即退出 prior，不是部署式恢复检测证据。

合取 Gate 因四项失败而判为 **No-Go**：FSM 未达 `10%`、未比 time shortcut 高至少 `5%`、incremental CI 未排除零、且 push/reach 的 incremental point effect 为负。即使完全丢弃这个 privileged time shortcut，FSM 仍因绝对改善未达固定 `10%` 门槛而不能授权 comparator。准确边界是：**当前两事件进入的 privileged FSM 没有把强 LeWM prior headroom 转化成足够大的安全切换收益。** 这不否定 Anchor-Persistent LeWM、FCPG 或 source selection 的一般可行性，但当前 development split 不再调门槛、不重跑 FSM，也不授权 normalized-advantage comparator、policy control 或 CEM。

bounded smoke 只验证了数据、递归写回与统计管线，不参与该科学判定。正式 development 汇总为：

```text
outputs/lewm_posterior_preference_oracle/randomized_development_equal_view_calibration_seed17101/summary.{json,md}
```

#### Random-Hazard predictor isolation A0（2026-07-16，Development Go）

该最小 Gate 复用 fresh randomized evaluation split 与冻结的 `latest@2000` Anchor-Persistent LeWM，不训练新参数。离线 `sensor-valid` mask 只决定直接受干扰的 anchor 在 measurement 缺失时采用哪个 predictor；五种方法分别维护自己的递归 posterior history，measurement 恢复后立即重新锚定。主统计排除第一个 hidden event `o0`，依次按 `phase -> event -> paired camera view -> source trajectory -> task` 等权聚合，并以每任务 `10` 条独立源轨迹做 task-stratified paired bootstrap。

Factual LeWM 的 task-macro grounded-latent MSE 为 `0.270214`：

| 最小基线 | 基线 MSE | Factual 相对改善（95% CI） | 正向任务 |
| --- | ---: | ---: | ---: |
| hold-last | `0.573514` | `54.26% [43.29%,64.57%]` | `4/4` |
| latent constant velocity | `2.797489` | `85.99% [80.14%,91.27%]` | `4/4` |
| frozen LeWM zero-action ablation | `0.593330` | `47.51% [37.39%,57.29%]` | `4/4` |
| cross-source shuffled-action LeWM | `0.568283` | `35.02% [28.54%,44.78%]` | `4/4` |

因此历史最小条件判为 **A0 Go**：现有 learned predictor 比 hold-last/latent-CV 更准确地传播 grounded anchor state，并对训练时动作输入及其时序匹配敏感。A0.5 已进一步证明，zero-action 和 shuffled-action 不能单独作为 factual action 必要性的证据：前者不是参数匹配、独立训练的 no-action recurrent model，二者都包含训练/推理输入不匹配。latent CV 也不是 FK、像素空间 Kalman 或最近外参传播；本 Gate 没有 selector、clean policy rollout 或未见 corruption。

该结果不推翻前一节 preference switching 的 No-Go，而是把两个问题分开：**predictor 有效，当前 source selector 不安全。** A0.5 已补上独立 no-action recurrent baseline；随后 A1 使用 fresh same-state action branches 进一步隔离动作增量，结果见下文。FK/KF、selector 和 paired policy rollout 仍不属于 A0 的证据。

正式汇总：

```text
outputs/lewm_posterior_predictor_isolation/randomized_development_seed17101/summary.{json,md}
```

#### Matched action/no-action A0.5（2026-07-16，Development No-Go）

A0 的 `zero-action` 是在已按 factual action 训练的模型上把输入动作置零，`shuffled-action` 也是分布外输入干预；二者都不能排除模型只是没有适应缺失或错配动作。A0.5 因此冻结原 `2k` anchor encoder，在同一 latent space 中从字节相同的初始化训练两个同容量 dynamics head：factual head 读取真实动作，no-action head 始终读取常量动作。两头使用相同数据、target、dropout RNG、optimizer schedule、独立梯度裁剪与固定 `latest@2000`，不选择 best checkpoint。

训练端点的单步 validation MSE 为 factual `0.015800`、matched no-action `0.054065`、copy `0.236096`；factual 相对 no-action 改善 `70.78%`。但在与 A0 相同的 `40` 条独立源轨迹、`80` 个 paired camera views 和真实 interruption mask 上，各方法维护独立递归 posterior history 后，差距明显收窄：

| 递归基线 | 基线 MSE | Factual MSE | Factual 相对改善（95% CI） | 正向任务 |
| --- | ---: | ---: | ---: | ---: |
| hold-last | `0.573514` | `0.274172` | `53.77% [42.63%,64.28%]` | `4/4` |
| latent constant velocity | `2.797489` | `0.274172` | `85.73% [79.70%,91.07%]` | `4/4` |
| matched no-action recurrent | `0.308184` | `0.274172` | `7.54% [-3.88%,18.29%]` | `3/4` |
| cross-source shuffled-action | `0.585799` | `0.274172` | `35.38% [28.87%,44.87%]` | `4/4` |

表中的相对改善先在每个任务内计算，再对四任务等权；因此 `7.54%` 不等于直接对两个 task-macro MSE 求比值。逐任务 matched no-action 改善为 pick `30.35%`、push `2.33%`、reach `5.38%`、button `-7.90%`。固定 Gate 仅失败于 matched no-action 比较的 CI 下界未大于零，因此判为 **A0.5 No-Go**。准确边界是：当前 LeWM dynamics 明显优于 hold-last/CV，且错误动作会显著破坏递归传播；但在当前单策略轨迹分布上，独立 no-action recurrent model 可由历史状态解释大部分动作后果，尚不能声称 factual action 对四任务持续接地具有稳定、普遍的必要性。旧 A0 的 predictor-positive 结论保留，旧 zero/shuffle 对动作必要性的证据降级；本 split 不继续调参，也不授权 selector、policy rollout 或第二贡献。

正式训练与递归汇总：

```text
outputs/lewm_grounded/anchor_dynamics_pair_independent_seed17301_2k/{metrics.jsonl,predictor_pair_latest.pt}
outputs/lewm_posterior_predictor_isolation/matched_no_action_seed17301/summary.{json,md}
```

#### Action-Branch Random-Hazard A1（2026-07-16，Screen No-Go）

A1 针对 A0.5 的单策略动作多样性不足，新建了同状态动作干预：每个 hazard 保存完全相同的物理状态、RNG 与三步历史，再恢复该 snapshot 执行八条匿名 QR 动作分支。八条分支保留 gripper 命令，只改变四个冻结方向的正负运动分量；measurement 在第一条 branch action 前隐藏，递归预测 `H=8` 个模型步，即 `40` 个 raw environment steps。主指标使用 train-only whitening，并在每个 bundle/horizon 内减去八分支均值，只评分动作造成的 branch-relative future-state 差异。

正式 store 共 `450` 个 bundles：train/development/test 为 `218/113/119`。Screen 只使用 development 的 `48` 条独立 physical episodes；factual 与 matched no-action 两个同容量 head 在 seeds `18401/18402/18403` 各训练固定 `2,000` steps，使用同一初始化、优化器、batch 和 dropout RNG。冻结 Anchor-Persistent encoder 的 `SIGReg` 固定为 `0.02`，新 prediction heads 不增加 SIGReg。

| A1 development 指标 | 结果 | 判定 |
| --- | ---: | --- |
| factual vs matched no-action | `+55.75%` | Pass；`3/4` 任务、`3/3` seed 正向 |
| factual vs 7-way wrong-action mean | `+73.74%` | Pass；`4/4` 任务、`3/3` seed 正向 |
| `H6:H8` vs no-action / wrong-action | `+52.72% / +70.37%` | Pass |
| raw whitened MSE factual / no-action | `0.115623 / 0.439560` | Pass |
| informative bundle fraction | pick `30/30`、reach `28/28`、push `26/26`、button `10/29` | **Fail**；Button `34.48% < 80%` |

逐任务 factual-vs-no-action 为 pick `+65.25%`、reach `+83.64%`、push `+74.84%`、button `-0.73%`。Button 的失败来自当前 post-contact/workspace-saturated 分支多数没有形成足够 target variance，训练无法创造缺失的干预支持。仅在十个 informative Button bundles 上做的事后诊断为 `+8.63%`，三个 seed 均正向；该数字只解释失败原因，不改变预注册 Screen。

为排除“物理状态已经分叉、但 frozen encoder 没编码出来”，另做了一次不训练、只使用已暴露 `train+development` rows 的 Button 物理重放。它逐行恢复原 A1 snapshot，执行完全相同的八条保存动作，并同时记录 TCP、完整 `qpos`、按钮关节/位置、success 与原 target latent。`84` 个 bundles 按源状态分为 pre-contact `25`、active press/contact `6`、post-success `53`：

| Button phase | latent branch variance median | `>=0.01` support | TCP H8 max pairwise median | button H8 max pairwise median | branch success disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre-contact | `0.022873` | `96.0%` | `5.51 cm` | `1.21 mm` | `64.0%` |
| contact | `0.010889` | `50.0%` | `2.01 cm` | `2.61 mm` | `16.7%` |
| post-success | `0.000146` | `0.0%` | `0.40 cm` | `0.036 mm` | `0.0%` |

train 与 development 的 phase-wise 中位数一致；latent variance 与 TCP/qpos 物理 branch variance 的 Spearman 相关分别为约 `0.94/0.94`。因此当前证据支持“Button 在首次成功后进入近吸收态，随机 hazard 过度采到无控制意义区间”，不支持“encoder 独立丢失了仍然存在的物理分叉”。该诊断不是新 Gate，不改变 A1 No-Go，也没有读取 independent test。若新建 A2，协议必须预先选择 active-control interval（例如首次 success 即终止）或改用 drawer/door/随机目标运输等持续动力学任务，并使用全新 split；不得在 A1 exposed split 上补样本或降低 `0.01/80%` 门槛。

因此 A1 的准确结论是：**在具有足够同状态动作分叉差异的 pick/reach/push 支持上，factual action alignment 提供了强且跨 seed 稳定的 predictive increment；当前四任务 Gate 因 Button intervention support 不足而 No-Go。** 这不能升级为四任务 action necessity、第二贡献或完整 belief-update claim。fresh `10k` 未获授权，independent test 未打开，也不启动 selector/CEM/policy control。

正式结果：

```text
outputs/lewm_action_branch_a1/action_branch_a1_formal.npz
outputs/lewm_action_branch_a1/a1d_screen_gate/summary.{json,md}
outputs/lewm_action_branch_a1/button_phase_diagnostic/{rows.npz,summary.json,summary.md}
```

#### Action-Branch Continuous-Task A2（2026-07-17，2k Development-Go / confirmation support No-Go）

A2 使用 fresh split，以 `drawer-open-v3` 替换 A1 中近吸收的 Button，并保留同状态八分支、随机 hazard、H=`8` 递归预测和 matched factual/no-action 设计。原始 H1-H8 support 仍为 No-Go；针对动作影响需要累积后才显现这一机制，development-only endpoint 固定为 H6-H8。此时 Drawer 的 `28/30` bundles informative（bundle fraction `93.33%`），按 `12` 个物理 episode 等权后的 support 为 `93.06%`，四任务均通过。

三个固定训练 seed `25801/25802/25803` 各从头训练 `2,000` optimizer steps，batch 为 `16` 个完整 branch bundles，使用 CUDA FP16；冻结 anchor encoder 的 `SIGReg` 仅为 `0.02`，两个 prediction heads 不添加 SIGReg。加固后的 scorer 从实际 NPZ 重算 support，并锁定三 seed、七个非恒等 wrong-action shifts 与 episode/instance 一一对应。H6-H8 development 结果为：

- factual 相对 matched no-action 改善 `67.69%`；
- factual 相对七种 wrong-action 平均改善 `82.99%`，最弱单 shift 仍改善 `82.45%`；
- pick/drawer/reach/push 与三个训练 seed 全部同向；
- raw task-macro MSE 为 factual `0.153933`、no-action `0.580041`、hold-last `0.945480`、latent CV `9.416028`。

这构成强 **2k Development-Go**，但没有确认性 CI。随后按事先固定的全新 split 只采集 confirmation train：每任务 `24` 个物理 episode，共 `218` 个 hazard bundles。Pick/Reach/Push 的 H6-H8 episode-macro informative support 均为 `100%`；Drawer 为 `72.92%`（`59` bundles，median branch variance `0.02016`），低于冻结的每任务 `80%`。按合取 Gate 判为 **Confirmation support No-Go**，因此三 seed `10,000`-step 训练没有启动，reserved test 没有采集。

Drawer 的失败集中在 episode 后段：hazard step `3-9/10-16` 的 bundle support 均为 `100%`，step `17-23/24-31` 降至约 `42.9%/50.0%`。这说明当前 scripted Drawer 轨迹后段仍接近低分叉阶段，而不是 2k LeWM 明确失败。该 split 不换 seed、不降低门槛、不事后截掉低支持 rows。若再建新协议，必须预先选择真正持续运动的任务或在采集前定义 active-control termination，并使用另一组独立 episode。

```text
outputs/lewm_action_branch_a2/support_gate/summary.json
outputs/lewm_action_branch_a2/a2d_screen_gate/summary.{json,md}
outputs/lewm_action_branch_a2_confirm/train_support.json
```

#### Action-Branch Continuous-Task A3（2026-07-17，四任务 10k Confirmation-Go）

A3 没有修改 A2 的失败门槛，而是先在全新 split 上冻结一组持续产生分叉的匿名动作 bank：`expert / reverse / hold / half-speed / transverse_u+ / transverse_u- / transverse_v+ / transverse_v-`。四个任务固定为 `pick-place-v3 / drawer-open-v3 / reach-v3 / push-v3`；**每个 seed 只训练一个混合四任务 LeWM**，不是四个 task-specific 模型。每任务 split 为 `24 train / 12 development / 12 reserved test` physical episodes，H6-H8 physical 与 train-only per-task-whitened latent support 在打开 test 前均为 `100%`。

三个 development seed `29201/29202/29203` 各训练固定 `2,000` steps 后，fail-closed scorer 给出 factual 相对 matched no-action `68.80%`、相对七种 wrong-action `85.20%` 的改善，四任务与三 seed 均正向。该 Development-Go 只授权采集 reserved test 和三个 fresh-start confirmation seed `29301/29302/29303`；confirmation 不续训 2k checkpoint，而是从同一 frozen base 各训练固定 `10,000` steps，batch `16`、CUDA FP16、`fixed_latest`，Anchor-Persistent encoder 的 SIGReg 仅为 `0.02`。

test-only H6-H8 正式结果为：

| A3 confirmation 比较 | 相对改善 | 正向任务 | 正向 seed |
| --- | ---: | ---: | ---: |
| factual vs matched no-action | `74.01%` | `4/4` | `3/3` |
| factual vs seven-shift wrong-action mean | `88.18%` | `4/4` | `3/3` |

七个非恒等 wrong-action shift 的改善范围为 `87.90%--88.55%`。逐任务 factual-vs-no-action 为 drawer `77.25%`、pick `70.64%`、push `73.90%`、reach `74.22%`；三个 seed 都在 `4/4` 任务上同时通过两项主比较。raw task-macro whitened MSE 为 factual `0.202650`、no-action `0.712049`、hold-last `1.111683`、latent CV `20.221723`，因此中心化主指标没有掩盖绝对预测失败。

该 **四任务 Confirmation-Go** 支持的准确 claim 是：在冻结的四任务 same-state branch protocol 中，共享 Anchor-Persistent LeWM 的 factual action 对 held-out future anchor-state prediction 提供稳定且可复现的增量。它是七任务论文实验前的 preliminary mechanism confirmation，不证明未见任务泛化、视觉可靠性判断、posterior switching、policy success、在线 CEM 或真机控制。最终七任务规格必须新增三个任务及其独立 support/data split，并重新联合训练一个七任务共享模型；不能把七种 wrong-action shifts 误写成七个任务。

后验完整 source-trajectory 重放进一步限定了这条 claim。A3 confirmation 的 `322` 个 hazard 中，Pick `58/86`、Drawer `37/82`、Reach `53/75`、Push `46/79` 在 hazard 起点已经成功，合计 `194/322 = 60.2%`。Pick reset 的 TCP-object 距离为 `0.131--0.172 m`，Reach reset 的 TCP-goal 距离为 `0.221--0.308 m`，因此两者不是 reset 已在终点，而是旧图把 hazard snapshot 错标成了任务 `Start`。当前 `AnchorPersistentLeWM.encode_frames()` 又只保留 frame encoder 最后的三个机器人 anchor tokens；所以 A3 的强结果不能升级为 object/contact/goal-aware task dynamics。

A4 必须作为独立协议处理这些问题：

1. 七任务固定为 `pick-place-v3 / drawer-retrieve-v0 / reach-v3 / push-v3 / door-open-v3 / window-open-v3 / towel-wipe-v0`，每个 seed 训练一个任务均衡的共享模型；
2. proposal 时刻可以预先随机生成，但接收资格只能使用时刻 `t` 及历史：完整历史、从未成功、active stage、非近零专家动作、未饱和、无异常碰撞且仍有完整分支时域；不得依据未来 branch spread、future latent、future reward 或 branch success 逐行过滤；
3. 标准 `drawer-open-v3` 不得改名复用；`drawer-retrieve-v0` 必须具有绿色外壳、红色活动抽屉/内壳、同步的视觉/碰撞尺寸，以及由物理状态驱动的“拉开、抬手、平移、下探、抓取、抬升、移到外壳顶面、降低、释放、撤离”阶段；成功必须要求方块由绿色顶板支撑且夹爪已释放，目标只显示为顶面的薄平面区，不使用球形目标；
4. `towel-wipe-v0` 取代标准 `sweep-v3`。机器人以毛巾接触桌面，在左右端点间完成六个 half-cycle（即三次完整往返）后抬起；只有毛巾贴面且姿态近似水平时才累计端点，不能用悬空移动冒充擦拭；
5. H8 保留为局部 transition primary；完整 source replay 另行从 reset 运行至连续成功或超时，Push 展示不得把 40-frame branch window 冒充完整任务；
6. GPU 训练前先验证持久状态能读出 object/goal/progress。若三个机器人 anchor slots 不足，则使用 summary/task-context slot 加 anchors 的隔离 checkpoint 变体；不得仅靠换数据声称完整任务动力学；
7. train/development support 全部通过后才允许 `3 x 2k`，development 至少 `6/7` 任务和 `2/3` seed 同向后，才收 reserved test 并从 fresh initialization 跑 `3 x 10k`；SIGReg 固定为 `0.02`。

当前五个标准任务的 `v2` reset-to-outcome smoke 已通过：在同一 environment seed 下，`pick-place / reach / push / door-open / window-open` 的 reset success 全为 false，均在 `500` raw steps 内达到连续三帧 success；首次 success 分别出现在 raw step `52 / 42 / 62 / 77 / 90`。Pick、Reach、Push 的 reset TCP-reference 距离分别为 `0.179 / 0.289 / 0.179 m`。本地 `drawer-retrieve-v0` 有 `48` 个确定性 block/顶面目标 instance，物理状态专家在全部 `48/48` 上完成“拉开、抬手、抓取、移到绿色外壳顶面、降低、释放、撤离”，最慢实例在 `337` 步内成功；success 同时要求顶板支撑与夹爪释放。`towel-wipe-v0` 使用带 `freejoint` 的独立毛巾和真实双夹垫接触，不再把毛巾位置同步到 TCP；代表性完整轨迹先夹紧并抬起毛巾，再于 raw step `258` 完成六个 half-cycle（三次完整往返）并抬起，未夹持或悬空端点移动不会累计进度。七任务预览全部保留连续成功后的 `8` 个真实环境步，见 `outputs/lewm_action_branch_a4/visual_replays/index.html`。A4 不训练五任务模型或七个 task-specific 模型：每个 seed 只训练一个同时覆盖七任务的 shared LeWM，单个 batch 固定为 `7 tasks x 4 bundles = 28`。

正式 `formal_v2` train+development cache 已完成，共 `893` 个 accepted bundles、`3,024` 个 proposals，且不含 reserved-test identity。Preflight 对七任务逐一检查 active-control、physical support 与 latent support；通过数分别为 Pick `19/19/19`、Drawer `24/24/24`、Reach `21/21/21`、Push `22/22/22`、Door `24/24/24`、Window `24/24/24`、Towel `20/20/20`，因此判为 **Preflight-Go**。

Development seeds `30201/30202/30203` 已各自从同一 frozen base fresh-start 并严格停在 `2,000` steps。H6-H8 train-only-whitened branch-centered primary 上，factual 相对 matched no-action 改善 `64.120836%`，相对七种 wrong-action mean 改善 `80.911444%`；两项 paired 95% CI 下界均大于零，`7/7` 任务和 `3/3` seed 同向，全部八项冻结检查通过，因此判为 **Development-Go**。各任务 factual-vs-no-action 改善为 Door `45.37%`、Drawer `49.19%`、Pick `73.09%`、Push `75.72%`、Reach `89.88%`、Towel `54.21%`、Window `61.38%`。

#### A4b fresh-panel fixed-10k confirmation（2026-07-18，Confirmation-Go）

旧 A4 fixed-panel collector 曾因部分任务无法凑齐冻结的 `12` 个 active-control episode 而在发布 test cache 前 fail closed；该 artifact 继续保留为不可评分的历史尝试。A4b 没有放宽门槛或复用已暴露 episode，而是在 fresh `a4b-task-instance-v1` identity namespace、split manifest `a4b-fresh-preflight8-final40-seed31401-v1` 和 benchmark seed `31411` 下建立新 panel；与 legacy preflight-final digest 的交集为零。七任务各含 `12` 个 test physical episodes，共 `84` 个 physical episodes、`325` 个 accepted test bundles；输入不含 development rows。三个 A4 v2 checkpoint `30301/30302/30303` 均固定使用 `latest@10000`，未做 checkpoint selection 或 test-time tuning。

H6-H8 task-macro branch-centered whitened MSE 中，factual 为 `0.271343`，matched no-action 为 `0.765870`。`65.627%` 是先在每个任务内计算 `1 - MSE_factual / MSE_no_action`、再对七任务等权平均的相对改善；它不是上述两个宏 MSE 的直接比值。paired task-macro MSE-difference 为 `0.494527 [0.460417,0.525799]`，方向为正表示 factual 更低。全部 `7/7` 任务和 `3/3` seed 点估计同向。七种 non-identity wrong-action donor shifts 的相对改善为 `83.24%--83.85%`，各自 paired CI 下界均大于零。未中心化但仍按任务 whitening 的 H6-H8 MSE 为 factual `0.321770`、no-action `0.869193`、hold-last `1.834268`、latent CV `45.813435`，同样支持 factual error 更低，但不是像素或物理单位下的绝对误差。合取 Gate 全部通过，正式判为 **Confirmation-Go**。

该确认支持的准确结论是：在七个 A4 active-control 任务和 fresh physical episodes 上，一个共享 TaskContext-Persistent LeWM 的 frozen `10k` endpoints 稳定利用 factual action 改善未来 task/anchor-state prediction。它不证明 policy rollout、task success、action selection、CEM、缺测 switching、七任务之外泛化或 checkpoint selection 收益。正式产物为 `outputs/lewm_action_branch_a4b/confirmation_gate/summary.{json,md}`。

#### A4b random-hazard interruption predictor（2026-07-21，development screen Go）

为直接回答“模糊或遮挡随时发生时，LeWM 到底负责什么”，本 screen 不固定某个任务阶段，也不训练视觉失效检测器。每个完整专家 episode 的 interruption onset、`2--8` 个 model-step duration 和最多三次事件只由独立随机种子生成，不读取 task state、action、reward 或 success。视觉失效前保留最后三次 frozen FCPG-grounded state；失效后 factual 只读取实际已经执行的五步 action chunks，并按 interruption age 递归到当前时刻。`H1--H8` 在这里仅表示“距离最后一次有效观测经过了几步”，不是预设的模糊发生时刻。

评估复用 A4b 三个 fixed-latest `10k` factual/matched-no-action checkpoint，不重新训练。七任务各取三个 development physical episode，共 `21` 个 episode、`134` 个唯一随机隐藏决策；每个决策由三个 checkpoint seed 复现，形成 `402` 个模型级记录，但不增加 physical sample。anchor-state task-macro MSE 为 factual `0.076332`、hold-last `0.546122`、latent CV `4.427457`、matched no-action `0.319388`、wrong-action `0.583685`，对应 factual 改善 `86.02% / 98.28% / 76.10% / 86.92%`。factual 在 `7/7` 任务上胜 hold/CV，在 `6/7` 任务上胜 matched no-action/wrong-action；age `1--8` 的 pooled factual MSE 均低于 hold-last。

唯一明确边界是 Towel。三条 Towel episode 合并后，factual 相对 hold-last 改善 `21.2%`，说明 learned prior 仍可补视觉缺测；但 factual MSE `0.26463` 高于 matched no-action `0.16688`。按物理阶段拆解，`lift_towel` 的 factual/no-action 为 `0.11765/0.30160`，而 `wipe_left` 为 `0.45495/0.22062`；问题集中在左向擦拭的动作传播，不是随机 interruption 机制整体失效。下一步若修模型，应针对 Towel 左右方向平衡和相机/世界动作坐标，而不是延长 interruption、增加训练步数或把 onset 固定到某个 horizon。

准确判定为 **predictor development screen Go**：现有 A4b LeWM 在随机缺测时对六任务提供强动作条件传播，并在七任务全部超过非学习传播基线。该 screen 使用离线已知 sensor-valid 路由和 clean latent target，没有输入真实 blur/occlusion 图像，不验证 deployment selector、恢复重锚定、策略 rollout 或 task success，不能单独升级为论文第二贡献。输出位于：

```text
outputs/lewm_action_branch_a4/random_hazard_a4b_7task_dev3_v2_seed42601/summary.json
```

#### A4/A4b full-trajectory coverage audit（2026-07-20，boundary diagnostic）

为检查 active-control Gate 是否也覆盖了完整任务转移，重新从 reset 重放专家轨迹，并以每个物理 episode/unique proposal 去重；`H=8` 只覆盖 `40` 个 raw steps。结果显示，A4b 的 `84` 个 fresh test episodes 和 `325` 个 accepted bundles 全部满足“提案时尚未成功”，因此没有 post-success 泄漏；但这不等于每个窗口都能到达任务成功。A4b 的 H8 首次/稳定成功覆盖为：Pick `85.7%/66.7%`、Reach `91.4%/91.4%`、Push `50.0%/30.0%`、Door `39.1%/37.5%`、Window `28.9%/24.4%`、Drawer `0%/0%`、Towel `0%/0%`。

当前 `camera_action_towel_fullcycle_seed40571_2k` 使用的 raw dataset 也只替换了 Towel rows，旧六任务仍来自 legacy proposal。对该 dataset 的 `210` 个物理 episode、`224` 个去重 proposal bundle 的重放得到稳定成功 H8 覆盖：Pick `46.9%`、Reach `75.0%`、Push `6.25%`、Door `0%`、Window `0%`、Drawer `0%`、Towel `12.5%`；首次成功步中位数分别为 Pick `55`、Reach `48`、Push `62`、Door `79`、Window `88`、Drawer `225`、Towel `302`。因此这批 2K endpoint 只能作为 shared active-transition 诊断，不能被称为 full-task trajectory model 或完整任务预测结果。

审计产物和可复跑入口：

```text
outputs/lewm_action_branch_a4b/final/trajectory_coverage_audit.json
outputs/lewm_action_prefix_a4/joint_orbit12_gripper_towel_fullcycle_24train_8dev/trajectory_coverage_audit.json
PYTHONPATH=. .venv/bin/python scripts/audit_lewm_a4b_trajectory_coverage.py --dataset <raw-dataset-dir>
```

下一次完整任务数据必须按 task phase/transition 采样，并为 Drawer、Door、Window、Push、Towel 增加能覆盖成功前后状态的长窗口或分阶段窗口；不能只增加视角、训练步数或重复早期 proposal。该诊断不翻案 A4b active-control Confirmation-Go，但收紧其 claim boundary。

#### A4c phase-aware data refresh（2026-07-20，collection protocol ready）

针对上述覆盖缺口，新建了独立的 A4c 数据协议；旧 A4/A4b artifact 和结论不改写。A4c 仍然是一个七任务共享 LeWM、`H=8`、每个 model step `5` 个 raw steps、每个 episode 目标 `12` 个 proposal 的协议，但 proposal 不再固定限制在前 `160` 个 raw steps。每个 train/development physical episode 先从 reset 完整回放 expert，按 phase、contact/grasp transition、进度分层和首次 success 前窗口生成一份冻结 schedule；schedule 在任何 matched branch 执行前物化，branch outcome 不参与选行或删行。

CPU phase planner 已对全部 `252` 个 train/development episodes（七任务各 `24 train + 12 development`）完成 dry-run。关键 coverage 如下：

| Task | mean first success raw step | mean selected windows | H8 crosses first success | phase-transition coverage |
| --- | ---: | ---: | ---: | ---: |
| Pick | `53.9` | `8.14` | `36/36` | `36/36` |
| Reach | `50.0` | `8.31` | `36/36` | `36/36` |
| Push | `61.4` | `10.19` | `36/36` | `36/36` |
| Door | `79.0` | `12.00` | `35/36` | `36/36` |
| Window | `87.4` | `12.00` | `36/36` | `36/36` |
| Drawer retrieve | `239.9` | `12.00` | `35/36` | `36/36` |
| Towel wipe | `298.4` | `12.00` | `36/36` | `36/36` |

Phase labels also cover the previously missing long-task stages: Drawer `pull_open -> lift_after_open -> move_above_block -> descend/grasp -> lift_block -> transport_above -> lower/release`, and Towel `grasp/lift -> wipe_right/wipe_left -> lift_after_wipe`. The two `35/36` cells are episodes whose expert rollout terminates too close to the success boundary for a complete H8 window; they remain explicitly reported rather than silently padded.

The schedule and new collector are:

```text
outputs/lewm_action_branch_a4c/phase_schedule.json
scripts/plan_lewm_a4c_phase_schedule.py
scripts/collect_lewm_action_branch_a4c.py
scripts/audit_lewm_a4b_trajectory_coverage.py --environment a4 \
  --ledger outputs/lewm_action_branch_a4c/proposal_ledger.npz \
  --cache outputs/lewm_action_branch_a4c/state_cache.npz
```

The A4c state cache and raw RGB dataset are now complete. The cache contains `504` phase-balanced physical bundles and `3,024` task-view rows (`2,016` train / `1,008` development), with all seven tasks represented at twelve orbit views. The formal training split uses even clock positions (`12/2/4/6/8/10`) and the development split uses odd positions (`1/3/5/7/9/11`). Camera reconstruction remains numerically consistent: mean reprojection error `5.52e-06 px`, maximum `1.99e-05 px`.

One shared seven-task model was trained from the fixed `10k` parent checkpoint with `SIGReg=0.02`, seed `40571`, and `17,966,708` parameters. The fixed `2k` endpoint and the final `10k` endpoint use the same raw dataset and evaluation protocol. The 10k endpoint improves task-macro anchor EPE from `5.270 px` at 2k to `4.534 px`, and gripper aperture MAE from `1.844 px` to `1.484 px`; however, action-shuffle delta is only `0.00175` at 10k (`0.00327` at 2k). Therefore this is evidence of more stable visual/coordinate prediction, not yet evidence that LeWM has a strong additional action-conditioned control contribution.

The complete seven-task human-inspection replay for the final endpoint is:

```text
outputs/lewm_action_prefix_a4/camera_action_phaseaware_a4c_v4_seed40571_10k_full_clock03/index.html
```

It contains one full prediction-cycle HTML page plus GIF/MP4 replays for Pick, Drawer retrieve, Reach, Push, Door open, Window open, and Towel wipe. The page uses the agreed playback protocol: the left side advances five real raw frames, pauses, and the right side plays the corresponding 40-frame forecast before the next observation is revealed. The model does not read future RGB or `q_true` during inference.

The final Towel wipe timing diagnostic found `5` reversals, zero early reversals, mean timing error `1.0` raw step, mean endpoint ratio `0.690`, and mean endpoint shortfall `9.10 px`. Thus the remaining towel error is mostly under-reaching the wipe amplitude rather than reversing too early; the HTML is the authoritative visual check.

#### Projectable metric-3D LeWM pilot（2026-07-21，2k development）

为检验上述跨视角幅度误差是否来自“直接从视角纠缠 latent 回归二维像素”，在冻结的 A4c camera-action `10k` checkpoint 上增加了一个小型 camera-frame 3D residual readout，并联合微调原 LeWM encoder/dynamics `2,000` steps。动作积分给出世界坐标几何参考，readout 只预测有界三维残差；随后使用每条样本的真实相机外参和内参进行确定性 world transform 与像素投影。相机参数不作为可学习 feature，`SIGReg` 固定为 `0.02`，ViT backbone 冻结。

七任务 odd-view development 上，task-macro world-anchor EPE 从 command integration 的 `52.76 mm` 降到 `33.90 mm`。该收益不能由几何支路单独解释：把 future latent 固定为 current state 后为 `51.47 mm`，在同一物理状态的八个动作分支间循环错配 future latent 后为 `70.13 mm`。对应像素 EPE 为 factual `5.146 px`、static-latent `7.742 px`、shuffled-latent `10.546 px`。因此当前 LeWM future latent 确实含有可被 projectable 3D head 使用的 action-conditioned 状态信息。

该 pilot 仍不是七任务整体替代 2D readout 的 Go：同一 fine-tuned checkpoint 的原二维头为 `4.475 px`，优于 3D head 的 `5.146 px`，主要差距来自 Door、Drawer、Push 和 Window。Towel 的 odd-view task-macro 像素 EPE则从二维头的 `3.421 px` 降到 `3.262 px`。在与旧 `10k` 基线完全相同的 reserved-test clock03 Towel episode 上，直接 3D 投影将完整拼接 EPE 从 `7.36` 降到 `5.49 px`，H8 从 `7.66` 降到 `5.70 px`；五次换向的平均 endpoint ratio 从 `0.690` 提到 `0.788`，平均 shortfall 从 `9.11` 降到 `5.85 px`，且 early-reversal fraction 仍为零。

当前结论是 **3D mechanism partial Go / universal readout No-Go**：显式可投影 3D 状态能明显缓解 Towel 的视角相关幅度压缩，并且消融证明收益依赖 LeWM latent；但一个共享 3D head 尚未在七任务像素指标上超过原二维头。一个只在 even-view train split 拟合的标量混合诊断进一步支持互补性：单一 global weight `0.140` 在 odd-view development 将 task-macro EPE 从二维头的 `4.475` 降到 `4.368 px`；七个 train-only task weights 将其降到 `4.344 px`，其中六任务改善，Drawer 从 `3.676` 退化到 `3.856 px`。该标量不是论文方法，只说明正式后续应采用保守的 hybrid `2D + projectable 3D` readout，并在 fresh odd views 上预注册选择规则；不应直接把纯 3D pilot 续训到 `10k` 后挑结果。

产物与入口：

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_seed40581_2k/metrics.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_seed40581_2k/latent_bypass_audit.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_seed40581_2k/train_only_scalar_hybrid.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_seed40581_2k_full_clock03/index.html
outputs/lewm_action_prefix_a4/towel_reversal_timing_metric3d_2k_vs_base10k_clock03.json
scripts/train_lewm_metric3d_joint_a4.py
scripts/evaluate_lewm_metric3d_joint_a4.py
scripts/evaluate_lewm_metric3d_hybrid_a4.py
scripts/probe_lewm_world_residual.py
```

#### Known-geometry world-residual pilot（2026-07-21，2k development）

The follow-up keeps robot geometry and commanded motion on the analytic path: current FK anchor `XYZ` is propagated with the known action reference, and the LeWM head predicts only a bounded residual in the robot/world frame. Camera intrinsics/extrinsics are used only for the final exact projection. This is protocol `a4_camera_action_joint_metric3d_world_v2`; the earlier camera-frame v1 checkpoint remains loadable and unchanged.

On the same seven-task odd-view development split, world-frame v2 reached task-macro world EPE `31.33 mm` and projected pixel EPE `4.755 px`, versus command geometry `52.76 mm` and the same fine-tuned model's 2D head `4.415 px`. It is better than the camera-frame v1 pilot (`5.146 px`) but is not yet a universal replacement for the 2D readout. The result supports the architectural separation “known geometry plus learned interaction residual”, not a completed 3D-world-model claim.

Human-inspection pages at clock03:

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_seed40582_2k_full_clock03_towel/index.html
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_seed40582_2k_full_clock03_drawer/index.html
```

The same checkpoint gives stitched anchor EPE `5.94 px` on Towel and `6.36 px` on Drawer. These pages are development diagnostics only; no checkpoint selection or policy claim is made from them.

A same-episode Towel intervention then held the world-v2 checkpoint, `clock03` replay, and H1--H8 expert action prefixes fixed while changing only the current grounding source. Nominal and current-oracle grounding both have `0.000 px` current EPE and produce stitched forecast EPE `5.9364/5.9370 px`; the adapted multiview FCPG introduces `1.2820 px` current EPE and produces `5.9428 px`. H8 is likewise `6.2093/6.2094/6.2176 px` for nominal/oracle/FCPG. Thus the remaining clean Towel error is not current-anchor projection error: even current-oracle grounding gives no forecast gain. FCPG should be evaluated here only under an actual calibration perturbation; the clean endpoint limitation belongs to future geometry/task-state propagation and readout. The matched visual report is:

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_seed40582_2k_towel_grounding_compare/index.html
```

#### Causal action-history Adapter（2026-07-21，2k/10k development）

对 Towel 换向点的控制链复核发现，MetaWorld 的 mocap target 在动作反向后仍会因伺服滞后继续前进数个 raw step；但当前 action-prefix LeWM 只读取当前状态与未来 H1--H8 action prefix，A4c cache 中已有的前两段已执行动作没有进入模型。因此，同一当前锚点与同一未来动作在换向附近并不足以确定短期位移幅度。该问题与当前 grounding 无关：上一节的 nominal/oracle 干预已经表明 clean Towel 当前锚点误差为零。

在冻结 world-v2 ViT、frame encoder、LeWM dynamics 和 metric-3D readout 的前提下，新增一个 `109,456` 参数的零初始化 Adapter，只把过去两段五步 action chunk 注入原 action-prefix condition。训练仍覆盖同一个七任务 shared model，固定 seed `40583`；分别从同一初始化运行固定 `2,000` 与 `10,000` step scheduler，禁止从 2k 的零学习率 endpoint 伪续训。parent checkpoint 的 SIGReg 保持 `0.02`，本轮不重新优化 encoder。Adapter 初始前向与 parent 逐值相同，未来 RGB、future `q_true` 和未来观测均不进入预测。

七任务 odd-view development 上，task-macro world EPE 按 frozen parent / 2k / 10k 依次为 `31.3265 / 27.5362 / 26.3189 mm`，projected pixel EPE 为 `4.7547 / 4.2179 / 4.0592 px`。10k 的同任务、不同 physical bundle 错配历史将二者恶化到 `40.6745 mm / 6.1305 px`，排除了 Adapter 仅学习常数偏置。Towel 的改善最明显：world EPE `23.3836 -> 14.6494 -> 13.0117 mm`，pixel EPE `3.1497 -> 2.0048 -> 1.8319 px`；10k 错配历史为 `38.2441 mm / 5.2879 px`。

在与 world-v2 parent 完全相同的 exposed `clock03` Towel episode 上，parent / 2k / 10k 的五次换向平均 endpoint ratio 为 `0.8307 / 0.9178 / 0.9512`，平均 endpoint shortfall 为 `4.6115 / 2.4286 / 1.4701 px`，early-reversal fraction 均为零。10k 的 H1/H4/H8 EPE 为 `1.831/4.643/6.102 px`，相比 parent 的 `3.133/6.110/6.209 px` 仍更好；但完整 max-lookahead stitched EPE 为 parent `5.936`、2k `5.756`、10k `6.200 px`。因此 10k 进一步修复沿擦拭方向的端点幅度和 development 平均误差，却在该单 episode 的 H8 非端点/横向误差上出现 trade-off，不能宣称全面单调提升。这仍支持“最近已执行动作提供 servo velocity / momentum 信息”的具体机制，而不是把 FCPG 重复 concat 一次。

该结果仍是 **causal-history mechanism development Go**，不是控制或论文最终 Go：目前只有一个 Adapter seed，整体 latent MSE 没有改善，`clock03` Towel 页面也已用于人眼诊断。2k 和 10k 均保留，不依据该 exposed 页面事后挑 checkpoint。下一步若继续，应在 fresh split 上复验多 seed，并比较显式 mocap velocity / proprio history；不应直接扩大 Adapter 或把该结果写成 LeWM 已完成闭环控制。产物为：

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_2k/metrics.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_2k/evaluation_history_mismatch_v2.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_2k_full_clock03_towel/index.html
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k/metrics.json
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k_full_clock03_towel/index.html
outputs/lewm_action_prefix_a4/towel_reversal_timing_world_history_2k_10k_vs_world_v2_clock03.json
fcpg/lewm_history_conditioned_metric3d_a4.py
scripts/train_lewm_history_conditioned_metric3d_a4.py
```

推荐的七任务人工预测审查页使用上述 **world-3D + causal-history `10k`** 共享 checkpoint，而不是较早的 Orbit-12 二维 readout `10k`：

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k_full_clock09/index.html
```

该页面固定在 held-out `clock09` clean 视角，使用 nominal grounding；没有相机抖动、漂移、模糊或目标准星。模型只读取当前 RGB/proprio/三锚点、过去两段已执行动作、H1--H8 action prefix 和 task identity，future RGB 与 `q_true` 只用于离线叠图和误差。相对旧 Orbit-12 `10k` 页面，Towel 的 full-episode stitched anchor EPE 从 `11.737` 降到 `4.120 px`，Pick/Drawer/Reach/Push 也从 `8.300/9.655/6.476/7.368` 降到 `3.198/4.409/3.681/5.818 px`。Door/Window 则从 `13.401/8.223` 退化到 `22.099/11.194 px`，因此该 checkpoint 是当前 Towel 端点与三维动作传播的首选诊断模型，但不能写成七任务每项指标都最优，也不能按任务挑选不同 checkpoint 后汇报为一个共享模型。

同一 checkpoint 的动态相机人工审查页为：

```text
outputs/lewm_action_prefix_a4/camera_action_metric3d_world_history_seed40583_10k_dynamic_clock09_drift6_jitter2p5_p24/index.html
```

该页保持模型侧 nominal `clock09` 标定不变，实际相机在前 `120` raw steps 累计漂移 `6 deg`，并叠加 `+/-2.5 deg`、`24` raw-step 周期的平滑抖动；当前三锚点由 frozen multiview FCPG 校正，未来使用固定 `4/8 px` magnitude-gated innovation transport。Towel stitched EPE 为 `9.253 px`。该页用于检查每次新观测后的重新接地，不代表 LeWM 能从机器人动作预报未来的外生相机抖动。

同一 `10k` history checkpoint 还完成了一次实际相机漂移下的接地干预。首次诊断把实际相机外参交给了 3D readout；这只能作为 offline calibration-oracle ceiling，不能代表部署，因为相机漂移本身未知。该 ceiling 下 current nominal / adapted FCPG / current-oracle EPE 为 `15.2845 / 1.6795 / 0.0000 px`，stitched future EPE 为 `6.6488 / 6.5090 / 6.4999 px`。它说明已知实际外参会绕过大部分二维 grounding 误差，不能据此判定 FCPG 对部署式 LeWM 只有约 `2%` 价值。

修正后的部署合同只向 LeWM 暴露漂移前的名义外参；实际外参仅用于生成 RGB 与评估 `q_true`。固定 `clock03` Towel、实际相机 `5 deg` 世界 z 轴偏转、同一物理轨迹和 action prefix 后，nominal stitched EPE 为 `15.3717 px`；只在当前帧用 FCPG 重采样特征为 `15.1005 px`，说明 current-only 接法确实不足。随后加入零训练的短时域 innovation transport：对每个 anchor 将当前 `q_fcpg-q_nominal` 常量加到名义 H1--H8 投影。该版本将 H1/H4/H8 从 `14.9739/16.4567/15.2070 px` 降到 `2.5832/5.1186/5.9098 px`，stitched 降到 `6.1130 px`，相对 nominal 改善约 `60.2%`。current-oracle 使用同一 transport 的 stitched EPE 为 `6.4522 px`；FCPG 略优于它是因为 learned correction 同时补偿了该 checkpoint 的固定预测偏差，不应解释为超过真实几何 oracle。

因此，当前最值得发展的接口不是再次 concat FCPG 特征，而是 **feedback innovation transport**：FCPG 负责当前视觉观测更新，LeWM 负责世界坐标动作动力学，短时域 transport 将当前图像空间校正传播到未来策略锚点。常量 transport 目前只是单 episode development headroom；下一步必须在七任务、多漂移轴/强度与 clean 条件下比较 no transport、constant transport、geometry-conditioned transport、current-oracle transport 和 actual-extrinsic ceiling，再决定是否训练小型 transport head。匹配对照页为：

```text
outputs/lewm_action_prefix_a4/history10k_towel_clock03_rot5_deploy_transport_compare/index.html
```

随后完成了七任务共享 checkpoint 的最小 transport Gate。所有条件固定同一个 `10k` history-conditioned world-3D LeWM、每任务同一个 physical episode、`clock03` 相机和 expert action prefix；实际相机绕 world-z 偏转 `5 deg`，LeWM 始终只读取漂移前 nominal calibration。比较结果如下：

| 接口 | 七任务 task-macro stitched EPE |
| --- | ---: |
| nominal grounding / no transport | `20.5437 px` |
| FCPG current re-indexing / no transport | `20.1819 px` |
| FCPG / constant innovation transport | `7.4316 px` |
| FCPG / magnitude-gated transport | `7.4314 px` |
| current-oracle / constant transport | `7.1143 px` |

Magnitude gate 不训练参数，也不读取 `q_true`：每个 anchor 的 `||q_fcpg-q_nominal||` 在 `<=4 px` 时不传播、`>=8 px` 时全传播，中间线性插值。阈值在运行本 Gate 前由 multiview corrector 的 even-view development 统计冻结：best checkpoint 的 clean correction mean 为 `2.1081 px`，drift correction mean 为 `7.6110 px`。在 `5 deg` 条件下，gated transport 相对 nominal 改善 `63.83%`，七任务 `7/7` 同向，并关闭 current-oracle 可恢复差距的 `97.64%`。current-only 几乎无效，说明关键不是再次读取当前校正特征，而是把当前观测 innovation 显式带到 nominal-calibration 下的未来投影。

Clean `clock03` 上，always-on constant transport 将 task-macro EPE 从 `7.0824` 增至 `7.7032 px`（`+8.77%`）；冻结的 magnitude gate 将其限制在 `7.1056 px`（`+0.33%`），七个任务的单项退化均低于 `1%`。因此当前最小机制结论为 **seven-task innovation-transport Go / final evidence pending**：接口在一个已暴露的 odd view、每任务一个 reserved-test episode 上成立，但尚无多 episode CI、多漂移轴/强度、policy rollout 或真实相机证据。下一步应直接做 fresh multi-view confirmation，不训练新的 transport Adapter。

随后用每个任务的第二个 reserved-test instance（`test-instance-rank=1`）做了未调参的 fresh sanity，并增加 renderer 的显式 test-rank 选择，避免继续复用首个暴露 episode。`clock01/clock05 + 5 deg world-z` 上，gated transport 的 task-macro stitched EPE 分别从 `18.3683/19.8246 px` 降至 `6.2572/6.0076 px`，改善 `65.94%/69.70%`，两个视角均为 `7/7` 任务同向。对应 clean task-macro 仅从 `5.3089/6.0360 px` 增至 `5.3395/6.1319 px`，退化 `0.58%/1.59%`，仍满足冻结的 `2%` 安全线。这进一步排除了 `clock03` 或首个 episode 的单点偶然性，但样本仍只有每任务一个 fresh physical episode、两个视角，尚不能替代多 episode CI 和多轴/强度确认。

第一阶段 bounded confirmation 随后固定为 test ranks `1/2/3`、全部六个 odd views `01/03/05/07/09/11`、七任务共享同一 `10k` LeWM 与同一 FCPG，共 `21` 个 physical episode 和 `126` 个 task-view episode cell。每个 physical episode 先对六视角求均值，再在任务内按 episode 配对 bootstrap，视角不作为独立样本。`5 deg world-z` 下，nominal / FCPG current-only / gated transport 的 task-macro stitched EPE 为 `19.4611 / 19.0990 / 7.3009 px`。Gated 改善 `62.48% [61.03%,63.90%]`，`7/7` 任务同向，全部 `18/18` rank-view 宏平均均改善（范围 `48.95%–69.69%`）；current-only 只改善 `1.86% [1.76%,1.97%]`。因此 **future innovation transport 的漂移机制确认通过**，且不能用当前帧 feature re-indexing 解释。

Clean task-macro 从 `6.7343` 增至 `6.8594 px`，点估计退化 `1.86%`，但 bootstrap 区间为 `1.14%–2.61%`，且 Drawer/Pick/Reach/Towel 的单任务点估计分别退化 `2.81%/4.07%/3.21%/2.93%`。因此 clean 只能判为 **macro point-estimate pass / safety pending**，不能宣称无损，也不得在这批确认数据上重新调 `4/8 px` 阈值。下一步应冻结当前 transport，直接验证它对 same-checkpoint policy action output 与 rollout success 的净影响；若 clean policy 非劣而 drift 控制改善，则像素级小退化不构成控制安全失败。

随后补做了一个职责不同的动态相机实验：不再评估同一时刻的未来投影，而是在每隔三个 model step 到来的真实新观测上，将上一轮 H3 prior 作为下一轮 FCPG 搜索中心并递归写回。相机在 raw step `5--35` 从 `0` ramp 到 `10 deg`；推理只读取 nominal calibration、当前 RGB、当前 FK 和历史动作，不读取 future RGB 或 `q_true`。在七任务 `rank=4, clock03` pilot 的 `>=8 deg` 区间，one-shot nominal-centered FCPG 为 `14.8839 px`，LeWM recursive re-centering 降到 `3.4478 px`，说明递归重定中心确实扩大了局部 corrector 的捕获范围。

但这个接口的正确强基线不是绝对像素 hold/CV，而是 **当前 FK nominal projection + 已观测相机 residual 的时序滤波**。同一 pilot 中，residual hold-last、clipped constant velocity 和 linear CV Kalman 分别达到 `1.6441 / 1.5608 / 1.6379 px`，且 LeWM 在 `7/7` 任务均未胜这些基线。因此结论冻结为 **recursive re-centering mechanism positive / LeWM-specific No-Go**：到下一次真实观测时，机器人自身运动已由当前 FK 给出，LeWM 重复预测 robot anchors 没有净价值；简单滤波只需传播低维相机 residual。此前使用绝对像素 temporal baselines 得到的扩大矩阵仅用于发现该基线错误，不作为论文证据，也不再执行 fresh confirmation。

这个负结果不推翻上面的 fixed-drift future innovation transport：在候选未来尚未观测、未来 FK/接触结果未知时，LeWM future state 仍是不同问题。它只排除了把 LeWM 用作下一观测 robot-anchor tracker。若继续寻找第二贡献，LeWM 应传播 FK 无法直接给出的物体、接触或任务 anchor，或服务于真正的候选未来，而不是替代当前 FK 加相机残差滤波。

数值产物：

```text
outputs/lewm_action_prefix_a4/history10k_7task_clock03_rot5_deploy_nominal_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_rot5_deploy_fcpg_current_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_rot5_deploy_fcpg_transport_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_rot5_deploy_fcpg_gated_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_rot5_deploy_oracle_transport_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_clean_deploy_nominal_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_clean_deploy_fcpg_transport_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_7task_clock03_clean_deploy_fcpg_gated_metrics/metrics.json
outputs/lewm_action_prefix_a4/history10k_transport_confirm_rank1_clock01_rot5_{nominal,gated}/metrics.json
outputs/lewm_action_prefix_a4/history10k_transport_confirm_rank1_clock05_rot5_{nominal,gated}/metrics.json
outputs/lewm_action_prefix_a4/history10k_transport_confirm_rank1_clock01_clean_{nominal,gated}/metrics.json
outputs/lewm_action_prefix_a4/history10k_transport_confirm_rank1_clock05_clean_{nominal,gated}/metrics.json
outputs/lewm_action_prefix_a4/grounding_transport_confirmation_v1/summary.{json,md}
outputs/lewm_action_prefix_a4/recursive_grounding_ramp_pilot_v2_fkresidual_rank4_clock03_7task_h3/summary.json
```

#### Goal-relative endpoint 与 A4 多视角 FCPG 诊断（2026-07-20，development）

为判断 Towel 的末端短行程究竟来自图像分辨率、当前 FCPG，还是 LeWM 的目标表示，在冻结 A4c `10k` LeWM 的前提下补做了三个相互独立的诊断。首先，输入以 `448 x 448` 渲染后仍按现有合同缩放到 `224 x 224`；同一物理 episode 的 H1-H8 center EPE 相对原生 `224 x 224` 仅改善约 `0.3%--0.9%`。因此，单纯提高原始渲染分辨率不能解释或解决剩余 endpoint shortfall。

其次，旧 Stage I FCPG 在 A4 clean orbit 数据上存在明确分布外偏置。在 `3,024` 个 `q_nominal == q_true` 的 clean frame 上，旧 corrector 仍产生平均 `13.4509 px`、中位数 `13.918 px`、P95 `21.649 px` 的伪校正，`>1 px` 比例为 `99.66%`。这意味着不能把旧 FCPG 直接接入 A4c clean LeWM，并将性能变化解释成有效反馈接地。

随后只做了一次固定预算的 A4 multiview adaptation：冻结 ViT backbone，训练 dense projection/fusion 与 grounding head；训练视角为偶数钟点，development 视角为单数钟点，seed `40601`，固定 `2,000` steps。best@`800` 在 odd-view development 上将 `5 deg` drift EPE 从 nominal 的 `7.5114 px` 降到 `2.2716 px`，约降低 `69.8%`，七个任务均改善；但 clean EPE 仍为 `2.1081 px`，`clock11` 总体和 Towel 分别达到 `4.9264/4.423 px`。因此该 checkpoint 证明 A4 多视角反馈校正可以学习，但尚未达到 clean-safe deployable grounding，后续必须由 drift/reliability gate 决定是否启用，不能无条件替换 nominal grounding。

Goal-relative Towel sidecar 使用外部 phase-derived 二维 endpoint，只作“显式剩余距离是否有帮助”的诊断，不是 learned visual-goal detector。nominal `clock12` 上，H8 center EPE 从 `3.8082` 降到 `3.4854 px`，stitched endpoint error 从 `2.7755` 降到 `2.4336 px`；但使用上述 adapted FCPG 后，`clock03` 的 H8 EPE 从 `7.5047` 恶化到 `8.5908 px`，endpoint error 从 `3.0188` 恶化到 `3.9814 px`。这说明 goal-relative signal 有局部 headroom，但当前 sidecar 与无条件 FCPG 组合不具备跨视角正贡献，不能升级为新论文 claim。尤其在 clean 条件下 `q_nominal == q_true`，Towel 提前回程的剩余问题不是 grounding 误差，主因仍是 goal-relative dynamics/readout 与训练终点监督。

人工核验入口如下；页面不在 RGB 上绘制目标标记，sidecar 使用的 phase-derived 目标仅作为明确标注的离线诊断输入：

```text
outputs/lewm_action_prefix_a4/fcpg_a4_multiview_adaptation_seed40601_step800_visual/index.html
outputs/lewm_action_prefix_a4/goal_relative_towel_sidecar_seed40572_2k_nominal_clock12_full/index.html
outputs/lewm_action_prefix_a4/goal_relative_towel_sidecar_seed40572_2k_adapted_fcpg_seed40601_step800_clock12_full/index.html
outputs/lewm_action_prefix_a4/goal_relative_towel_sidecar_seed40572_2k_adapted_fcpg_seed40601_step800_clock03_full/index.html
```

对应数值产物为 `outputs/lewm_action_prefix_a4/towel_resolution_probe_seed40571_clock12.json`、`fcpg_a4_clean_identity_matrix.json` 和 `fcpg_a4_clean_identity_matrix_adapted_seed40601_step800.json`。上述全部属于 exposed development diagnostics；尚未完成 fresh confirmation、policy rollout 或在线 goal selection。

#### Towel 显式任务目标的闭环策略基线（2026-07-23，development）

为确认此前“未到端点就回程”究竟是动力学预测不足还是策略缺少任务目标，重新整理了 Towel 行为克隆数据与闭环 evaluator。策略输入严格限制为当前 TCP `XYZ`、gripper 和 MetaWorld 当前提供的三维 task goal，即 `robot_goal=[obs[:4],obs[-3:]]`；仍不读取 towel/object state、previous observation、接触状态、success、`q_true` 或未来信息。训练集为 task instances `0--31` 的 `32` 条完整成功轨迹，development 为不重叠 instances `32--39` 的 `8` 条轨迹；reserved instances `40--47` 尚未打开。

在完全相同的数据、ViT cache、seed `40610`、架构和约 `2,000` optimizer-step 预算下，加入三维 task goal 后 best validation loss 从 robot-only 的约 `0.02976` 降至 `0.025247`。更重要的是，执行前四个 action 后重新观测的闭环 rollout 中，robot-only checkpoint 在 `8/8` development episode 均运行到 `500` 步仍失败，而 `robot_goal` checkpoint 在 `8/8` episode 全部完成，耗时 `265--347` raw steps；每条轨迹均按 `right/left/right/left/right/left` 完成六个 half-cycle，并执行最终抬升。

同一 `robot_goal` checkpoint 的干预进一步排除了“只是重训更幸运”：正常使用环境动态 task goal 时为 `8/8` 成功；将策略可见目标冻结为 episode 初始 goal 后为 `0/8`，且每条只完成第一个 half-cycle。因而当前可支持的结论是：**显式、部署可得的当前任务目标是 Towel 精确端点控制的必要输入，并足以在 clean fixed-camera development 上修复旧策略的提前回程。** 该实验不是 LeWM 或 FCPG 收益证据；它只有一个训练 seed，尚未测试相机漂移、视觉中断或 reserved confirmation。后续 FCPG 接入时，7 维策略仍完整读取 goal，但冻结 corrector 只能读取前 4 维 robot state，禁止 goal 泄漏到接地校正器。

人工核验入口：

```text
outputs/towel_policy_dataset_dev_instances32_39_seed40592/index.html
outputs/towel_policy_rollout_geoprop_robot_goal_signed_seed40610_2k_dev_instances32_39/index.html
outputs/towel_policy_rollout_geoprop_signed_seed40610_2k_dev_instances32_39_no_goal/summary.json
outputs/towel_policy_rollout_geoprop_robot_goal_signed_seed40610_2k_dev_instances32_39_hold_initial/summary.json
```

在上述目标条件基线冻结后，又执行了不重训策略的 `5 deg world-z` 接地干预。两路使用同一个 `geoprop_action_chunk_best.pt`、相同 task instances `32--39`、相同环境 seed 和相同动态 task goal；nominal 路保留名义投影，FCPG 路加载冻结的 A4 multiview corrector，并强制 reliability 为 `1`，因此唯一部署差异是三个机器人 anchor 的视觉采样位置。FCPG corrector 只读取当前 RGB、`q_nominal`、机器人 anchor XYZ 和策略输入的前四维 robot state；task goal 和离线 `q_true` 均不进入 corrector。

| 条件 | 接地 | success | 平均 grounding EPE | 结果 |
| --- | --- | ---: | ---: | --- |
| clean | nominal | `8/8` | `0 px` | 基线 |
| clean | FCPG | `8/8` | `2.456 px` | success 无回退；平均多 `2.75` raw steps |
| `5 deg` | nominal | `1/8` | `18.403 px` | 六条额外 episode 因错误接地失败 |
| `5 deg` | FCPG | `7/8` | `2.292 px` | `+75 pp`；EPE 降低 `87.54%` |
| `5 deg` | `q_true` oracle | `7/8` | `0 px` | 与 FCPG 相同成功上限 |

配对明细为一条 nominal/FCPG 共同成功、六条仅 FCPG 成功、一条两者都失败；最后一条即使用 `q_true` 也失败并已完成六个 wipe half-cycle，说明剩余失败属于策略收尾/`500` 步时限而非接地。当前准确结论是 **same-checkpoint FCPG control intervention Development-Go**：显式任务目标解决“往哪里走”，FCPG 在相机漂移下解决“当前从哪里读视觉特征”，二者组合将开发集成功率从 `12.5%` 提高到 `87.5%`，且达到 oracle-grounding 的二值成功上限。该结果仍只有一个 policy seed 和八个已暴露 episode；reserved instances `40--47` 保持未打开，需在训练/协议冻结后用于确认。

漂移页面与数值：

```text
outputs/towel_policy_rollout_goal_seed40610_2k_rot5_dev32_39_nominal/index.html
outputs/towel_policy_rollout_goal_seed40610_2k_rot5_dev32_39_fcpg/index.html
outputs/towel_policy_rollout_goal_seed40610_2k_rot5_dev32_39_oracle/summary.json
outputs/towel_policy_rollout_goal_seed40610_2k_clean_dev32_39_fcpg/summary.json
```

#### Explicit moving-goal LeWM 与 FCPG 视觉扰动消融（2026-07-22，2k / development）

为把“LeWM 负责往哪里运动”和“FCPG 负责当前在哪里”分开验证，新增 Reach/Push moving-goal branch 数据与显式目标条件的冻结 LeWM direct head。数据包含 `125` 个 same-state snapshot；development 为 `59` 个 snapshot、每个 snapshot `8` 个匿名动作候选，共 `472` 条真实 future branch。模型从固定初始化训练到 `2,000` steps 后，entity EPE 从 `84.263` 降至 `15.438 mm`，robot-anchor EPE 从 `74.670` 降至 `13.802 mm`；动作排序 Spearman 从 `0` 提升至 `0.7194`，top-1 从 `15.25%` 提升至 `44.07%`，选中动作的真实 H8 goal distance 从 `72.867` 降至 `37.843 mm`。这说明显式 current entity、moving goal 与 action prefix 足以学习目标条件动力学。

随后冻结 LeWM、FCPG 和全部物理 branch，只改变当前 RGB 的 Gaussian blur 与 nominal projection rotation，运行 `rotation={0,5,10} deg x blur_sigma={0,2,4}` 的 `3 x 3` observation-update 矩阵。LeWM 使用数据中记录的真实 camera basis，因此该矩阵只隔离当前 grounding 位置的贡献，不是未知外参下的完整部署实验。

- `5 deg` 下 nominal EPE 固定为 `17.696 px`，FCPG 在 blur `0/2/4` 下分别降至 `5.675/4.848/6.880 px`。
- `10 deg` 下 nominal EPE 为 `35.430 px`，FCPG 分别降至 `15.945/16.863/17.335 px`。
- 六个 drifted cell 的 current-grounding EPE 平均降低 `60.02%`，但 robot-anchor EPE 仅平均改善 `0.3969 mm`，entity EPE 平均变化为 `-0.0098 mm`，Spearman 平均只增加 `0.000819`，选中动作的真实 H8 goal distance 仅改善 `0.0976 mm`。
- 即使使用 oracle pixel，在无旋转时把 blur sigma 从 `0` 提到 `2/4`，entity EPE 仍从 `15.477` 恶化到 `22.488/26.175 mm`。因此严重模糊损坏的是视觉内容，FCPG 只能修正采样位置，不能恢复已经丢失的局部外观。

准确结论是：**显式目标条件 LeWM 动力学机制为 Go，FCPG 当前投影校正为 Go，但当前 direct 接口上的下游联合增益为 No-Go / unresolved。** 该模型直接读取精确 current entity/goal/world anchors，并在本消融中使用真实 camera basis，因而视觉接地点可被结构化输入旁路；这解释了 FCPG 将像素位置大幅拉回后，未来预测和动作排序仍几乎不变。下一次联合 Gate 必须移除精确 current entity 旁路，改用部署可得的 visual task anchor、接触/外观状态或间歇观测；严重模糊与丢帧期间由 LeWM prior 维持状态，视觉恢复后再由 FCPG 重锚定，不能把 FCPG 描述为去模糊模块。

正式入口与产物：

```text
scripts/train_lewm_goal_conditioned_direct.py
scripts/evaluate_lewm_goal_conditioned_visual_robustness.py
outputs/lewm_goal_conditioned_direct_seed93001_2k/goal_conditioned_direct_latest.pt
outputs/lewm_goal_conditioned_direct_seed93001_2k/metrics.json
outputs/lewm_goal_conditioned_direct_seed93001_2k/visual_robustness_rotation_blur/metrics.json
outputs/lewm_goal_conditioned_direct_seed93001_2k/visual_robustness_rotation_blur/summary.md
```

随后执行了去旁路的 deployable v2。该版本不再读取 simulator current entity，也不读取真实相机外参；只保留已知 task goal、FK robot anchors、历史动作、候选动作和由当前 RGB 编码的四槽 LeWM state。固定 `2,000` steps 后，原始 development 上整体 entity EPE 为 `20.18 mm`、Spearman 为 `0.6953`；其中 Reach/Push 的 Spearman 分别为 `0.9257/0.5259`。这说明去掉实体真值后仍能学习候选未来，但 Push 才是主要视觉依赖任务，Reach 大部分可由 FK、goal 和 action 解释。

同一 checkpoint 的 `3 x 3` rotation/blur 矩阵中，六个 drifted cell 的 FCPG current EPE 仍平均降低 `60.02%`，但 robot-anchor EPE 只改善 `0.3875 mm`，entity EPE 反而平均恶化 `0.0341 mm`，Spearman 仅增加 `0.001899`，selected H8 goal distance 平均恶化 `0.4155 mm`。Push 单独看也只有 Spearman `+0.005279`，entity EPE 改善 `0.0074 mm`，selected H8 反而恶化 `0.5362 mm`。10 deg 的 slot attribution 表明 nominal/FCPG state 的四槽相对变化约为 `5.4%--6.3%`；只替换三个 anchor slot 就复现了几乎全部小幅变化，只替换 summary slot 基本无效。因此 FCPG 信号确实进入了 anchor latent，失败发生在下游 goal/action head 对该信号的利用，而不是 encoder 完全丢失校正。

为排除“head 只在 clean state 上训练”的解释，又用完全相同初始化、样本、动作候选和训练角度顺序，分别训练 nominal 与 FCPG 两个 matched head；LeWM/FCPG 继续冻结，训练 rotation 固定为 `0/+-5/+-10 deg`，每路 `2,000` steps。最终 drifted mean 的 matched FCPG 相对 matched nominal 为：entity EPE `-0.2799 mm`、Spearman `+0.004850`、top-1 `+1.69 pp`、selected H8 goal distance `+0.3407 mm`。Push 为 entity `-0.3897 mm`、Spearman `+0.008417`、selected H8 `+0.3568 mm`；方向和指标不一致，量级也不足以支持联合贡献。

因此当前结论冻结为：**deployable goal-conditioned geometric head 学习成功，但 FCPG-to-head 联合接口为 development No-Go。** 更多同类训练不会解决结构错位：FCPG 校正的是机器人局部视觉读取位置，而当前 head 仍可主要依赖已知 FK、goal、action 和 summary；Push 的受控物体也不是一个由 FCPG 显式维护的 task anchor。后续若继续联合，只应改为 FK 无法旁路的接触/物体外观状态、视觉中断后的 prior propagation，或直接使用已观察到正增量的 policy-centered Towel ranking 接口；不再继续扩大这个 Reach/Push geometric head。

新增入口与产物：

```text
fcpg/lewm_goal_conditioned_deployable.py
scripts/train_lewm_goal_conditioned_deployable.py
scripts/evaluate_lewm_goal_conditioned_deployable_robustness.py
scripts/train_lewm_goal_conditioned_deployable_drift_matched.py
outputs/lewm_goal_conditioned_deployable_seed93011_2k/
outputs/lewm_goal_conditioned_deployable_drift_matched_seed93021_2k/
```

#### Frozen LeWM goal-action selection diagnostic（2026-07-20，CPU / development-only）

为区分“LeWM 能否预测 anchor”与“LeWM 能否帮助选择继续擦、反向或收尾动作”，对已有冻结 `10k` LeWM 的每个 Towel development snapshot 直接评估 `8` 个候选 action branch。评分只使用 H8 预测 tool-center 到 phase-derived task endpoint 的二维距离；记录的 future `q_true` 只用于计算实际 regret，不能进入选择器输入。当前 goal 仍是阶段/任务定义的离线诊断量，不是 learned visual-goal detector，因此该结果不具备部署 claim。

`144` 个 development snapshot 上，冻结 LeWM 的 goal-distance ranking 得到 Spearman `0.8277`、top-1 candidate accuracy `55.6%`、top-2 `79.2%`、top-3 `91.7%`、选中动作落在最优实际候选 `1 px` 内的比例为 `85.4%`，平均 top-1 regret 为 `0.5604 px`；随机 top-1 基线为 `12.5%`。这里的 branch slot 在原始 A4c index 中是匿名随机排列的，未保存 expert 语义置换，因此不再报告“固定 expert slot”基线，避免把某个 public slot 误当成 expert。在 held-out `clock03` 上仍有排序信号（Spearman `0.8073`，平均 regret `1.1184 px`），但视角变化会影响精度，不能把 development 结果直接写成控制收益。

Horizon sweep 显示，问题主要出在把 H8 终点当成唯一决策目标，而不是 LeWM 完全不会排序：H3/H4 的 Spearman 分别为 `0.9358/0.9416`，平均 regret 分别为 `0.0358/0.0812 px`，`1 px` 内比例为 `99.3%/98.6%`；H8 则退化到 `0.8277/0.5604 px`。因此后续若继续做控制接口，应采用 H3/H4 的 receding-horizon 评分，执行短 action chunk 后重新观测并经 FCPG 更新，而不是只按 H8 一次性选完整长程动作。

此前的 goal-relative action sidecar 在同一数据上将整体 Spearman 微升至 `0.8323`，但 top-1 降至 `53.5%`、regret 升至 `0.7081 px`；它没有证明需要继续扩大 sidecar。当前不启动 CostAdapter/CEM，也不把 H8 top-1 当作 LeWM 失败；更合理的接口是保留冻结 LeWM 作为短时域 action-conditioned candidate ranker，下一步只替换不可部署的 phase-derived goal 为显式任务 goal 或独立 visual goal detector，再做 fresh closed-loop rollout。

可复现入口（CPU，不使用 CUDA）：

```bash
PYTHONPATH=. /home/james/lewm/.venv/bin/python3 \
  scripts/evaluate_lewm_goal_action_selector.py \
  --output outputs/lewm_action_prefix_a4/goal_action_selector_dev_base_v2
```

修正后的冻结基线报告为 `outputs/lewm_action_prefix_a4/goal_action_selector_dev_base_v2/summary.md`；直接使用 H3/H4 的最小短时域报告分别位于 `outputs/lewm_action_prefix_a4/goal_action_selector_dev_h3/summary.md` 和 `outputs/lewm_action_prefix_a4/goal_action_selector_dev_h4/summary.md`。sidecar 对比为 `outputs/lewm_action_prefix_a4/goal_action_selector_dev_base_vs_action_sidecar_v2/summary.md`；这些结果均为 exposed development evidence，不是独立确认集。

随后补做了不依赖像素 readout 的世界坐标版本。它直接使用 `10k` history-conditioned world-3D LeWM 的 `predicted_world_xyz`，以 Towel 当前 task-defined endpoint 加已知 gripper offset 作为 tool-center goal，并与 `current anchor + 0.01 * cumulative XYZ action` 的解析积分基线比较。六个 odd camera views 的预测先在同一 physical bundle 内平均，主统计单位因此是 `12` 个独立 development snapshot，而不是 `72` 个 view rows；实际 future anchor 只作离线排序标签。

预先固定的 H3 screen 为 **Development No-Go**：LeWM/kinematic 的 Spearman 为 `0.9680/0.9700`，top-1 为 `0.667/0.750`，regret 为 `0.0032/0.0027 mm`。这里的绝对 regret 已小到微米量级，说明前三个 model steps 内解析机器人运动几乎没有剩余空间。正确动作仍是模型因果输入：将八条 action prefix 循环错配后，H3 Spearman 降到 `-0.0621`。

Horizon sweep 只能作为后验机制诊断。H4-H8 的 LeWM regret 相对解析积分依次降低 `35.0% / 61.6% / 92.0% / 93.1% / 82.9%`；H6 的 LeWM/kinematic regret 为 `0.0695/0.8724 mm`，H8 为 `1.5695/9.1531 mm`。这支持一个更窄的接口假设：已知运动学负责 H1-H3，LeWM 只在控制器滞后、接触和工作空间约束开始累积的中程候选排序中使用。当前候选 bank 仍以记录的 expert future prefix 为中心，且数据来自 exposed development，因此不得据此声称规划或控制收益。唯一合理的下一实验是 fresh Towel episode 上的 policy-centered H6 receding horizon：每次只执行一个 5-raw-step chunk，随后重新观测；相机漂移条件下再由已冻结的 gated FCPG 提供当前 measurement。正式产物为 `outputs/lewm_action_prefix_a4/world_goal_selector_towel_dev_h3/summary.{json,md}`。

上述在线接口已于 2026-07-21 做了最小闭环 smoke。测试使用 held-out `clock03`、三个配对 Towel reset、精确 clean nominal calibration；每次只根据当前 tool-center 与当前 task endpoint 因果生成八个训练分布内候选（center/reverse/hold/half/两组正交方向），冻结 LeWM 在 H6 打分，只执行首个 `5` raw-step chunk 后重新观测。`center-only`、解析 H6 和绝对 LeWM 三路均为 `3/3` 成功，平均完成步数分别为 `303/303/323`。因此按任务完成判据，该接口是 **clean feasibility Go**；较多步数只表示绝对 LeWM 更保守，不构成任务失败。与此同时，解析基线同样满成功，故该 smoke 不证明 LeWM 带来净控制增益。

进一步只做了一次零训练的共同偏置诊断：按 horizon 用解析 center forecast 校准 LeWM 的绝对 3D readout，同时保留候选间预测差异。该 `lewm_residual` 三条均恢复到 `303` 步，但 `136/136` 次 wipe decision 全部退化为选择 center，与解析基线逐条一致。该结果说明校准能消除保守性，但没有提供解析几何之外的选择收益。clean 对照页面为 `outputs/lewm_action_prefix_a4/towel_h6_receding_selector_smoke_clock03_residual/index.html`。

随后在完全相同的在线接口上加入固定 `5 deg world-z` 相机漂移，并对 Towel reserved instances `36..47` 运行 `12` 组配对闭环。实际相机姿态只用于 RGB 生成与离线 `q_true`，LeWM 始终接收漂移前的名义外参；两路只在当前三锚点采用 nominal 还是冻结的 A4 multiview FCPG。这里 LeWM 只在擦拭阶段选择动作候选，抓取、抬升及其余任务阶段仍由既有任务专家控制，因此该实验是混合闭环而非端到端 LeWM policy。`lewm_nominal` 与 `lewm_fcpg` 均为 `12/12` 成功、全部完成六个 half-cycle，平均步数为 `333.42/337.58`。FCPG 将擦拭阶段接地 EPE 从 `15.1740` 降至 `1.3752 px`，降低 `90.94%`；在 `611` 个对齐 wipe decision 中改变 `74` 次候选选择，覆盖 `11/12` episode，证明校正结果确实进入了 LeWM 决策链。准确边界是 **drift feasibility and grounding-mechanism Go / success-gain unresolved**：FCPG-LeWM 在漂移下能稳定完成任务，但 nominal 已达 `100%` 成功率天花板，当前 candidate/controller 协议无法测出成功率增益。人工页面为 `outputs/lewm_action_prefix_a4/towel_h6_receding_selector_rot5_clock03_pair12/index.html`，数值为同目录 `summary.json`。

同一协议在 `10 deg` 上扩到 `12` 个配对 episode 后，两路仍均为 `12/12` 成功；nominal/FCPG 的擦拭 EPE 为 `30.4952/10.9199 px`，FCPG 降低 `64.19%`，并在 `602` 个对齐决策中改变 `64` 次选择。但 FCPG 平均多用 `7.08` 个 raw step，成功率仍无差异。`8 deg` 的三条 screen 也同样满成功，只作范围诊断。该结果进一步确认当前候选中心使用 exact FK tool position 与 task-defined endpoint 后，任务主要由解析几何伺服解决；继续增大相机角度不能干净证明 LeWM 或 FCPG 的控制增益。`10 deg` 页面为 `outputs/lewm_action_prefix_a4/towel_h6_receding_selector_rot10_clock03_pair12/index.html`。

为验证视觉模糊/遮挡可以在任意擦拭时刻发生，runner 又加入与任务状态、动作、reward 和 success 无关的随机 hazard schedule。每个 episode 最多三次事件、每次持续 `2--8` 个 model step；当前图像在三个名义 FK anchor 周围被局部遮挡并全局模糊。已知 sensor-validity 仅用于本次机制 Gate：`lewm_fcpg_hold` 在中断期复用最后可信 state，`lewm_fcpg_prior` 只读取最后可信 state 与已经执行的动作递归传播，不能读取受损当前图像；视觉恢复后立即重新编码 measurement。

在 `5 deg`、`12` 个配对 Towel episode 和 `171` 个随机中断期决策上，clean-FCPG、corrupt-measurement、hold-last 与 LeWM-prior 均为 `12/12` 成功并完成六个 half-cycle，平均步数为 `337.58/340.08/338.42/336.33`。prior 相对 hold 的逐 episode步数差为 `-2.08`，bootstrap 95% CI `[-5.00,0.00]`；两者只在 `6/171` 个中断期决策上选择不同，覆盖 `4/12` episode。准确结论是 **known-mask interruption integration feasible / LeWM-specific control gain unresolved**：递归 prior 路由真实运行且没有读取当前损坏测量，但 exact-FK goal-centered candidate bank 使 hold 与 prior 几乎等价，不能据此形成第二贡献。人工页面为 `outputs/lewm_action_prefix_a4/towel_h6_random_interruption_rot5_clock03_pair12/index.html`。

为移除上述 exact-FK 候选生成捷径，新增一次 **policy-centered H6 same-snapshot development**。每个候选只由当前策略动作构造：保持原动作、停止、反向，以及 `0.25/0.5/0.75/1.25` 倍速度；三维目标只用于候选执行后的统一评分，从不进入 proposal generation。`12` 个 Towel episode 的六个 half-cycle 各取一个状态，共 `72` 个真实物理 snapshot；每个状态的八个候选都从同一 simulator snapshot 实际执行到 H6，且 `72/72` 的真实候选 cost spread 均超过 `5 mm`。

原策略候选相对 same-state oracle 的平均 headroom 为 `1.4187 cm`。冻结 LeWM 使用 nominal grounding 时，Spearman/top-1/平均 regret 为 `0.9077 / 61.11% / 0.1456 cm`；只切换为 FCPG grounding 后为 `0.9233 / 75.00% / 0.0755 cm`，top-1 增加 `13.89 pp`，regret 降低 `48.17%`。按 episode 汇总，FCPG regret 为 `8/12` 改善、`3/12` 持平、`1/12` 轻微变差。解析 Cartesian integration 为 `0.9110 / 76.39% / 0.0768 cm`，与 FCPG-LeWM 基本持平。因而 same-snapshot 边界是 **FCPG-to-LeWM ranking mechanism positive / LeWM necessity unresolved**：去掉目标引导的候选生成后，FCPG 的当前观测校正确实改善了冻结 LeWM 的动作排序，但解析几何并未被击败。人工检查页为 `outputs/lewm_action_prefix_a4/towel_policy_centered_headroom_reach_rot5_clock03_ep12/index.html`，数值为同目录 `summary.json`。

同一 policy-centered 候选随后进入逐次重观测的在线混合闭环：`policy / analytic geometry / LeWM nominal / LeWM FCPG` 在相同 `12` 个 reserved Towel instance、`clock03 + 5 deg` 下均为 `12/12` 成功并完成六个 half-cycle，平均 raw steps 为 `316.75 / 386.25 / 381.25 / 383.33`。较多步数来自 selector 更多选择 `quarter/half`，不作为本 Gate 的失败条件。FCPG 将 LeWM 擦拭期 EPE 从 `14.9849` 降至 `1.3447 px`（`91.03%`），并在 `722` 个对齐决策中改变 `234` 次候选选择（`32.41%`），覆盖 `12/12` episode；但 nominal 与 FCPG 的 success 均已饱和，FCPG 平均多 `2.08` raw steps。因此在线结论仍是 **closed-loop grounding intervention Go / task-success gain unresolved**，不能升级为 LeWM 优于几何或 FCPG 提升控制成功率。动态页面为 `outputs/lewm_action_prefix_a4/towel_h6_policy_centered_rot5_clock03_pair12/index.html`。

最后将同一 policy-centered bank 放入随机视觉中断。clean-FCPG、损坏 measurement、hold-last 与 LeWM-prior 四路仍均为 `12/12` 成功，平均 raw steps 为 `383.33 / 374.58 / 384.58 / 383.33`。在 `173` 个中断期决策上，prior 与 hold 的选择有 `21` 次不同（`12.14%`），覆盖 `3/12` episode；prior 平均只少 `1.25` raw steps，且损坏 measurement 本身没有造成任务失败。相较旧 exact-FK bank 的 `6/171`，新候选确实更能暴露状态传播差异，但仍不足以证明 LeWM-specific control gain。该结果冻结为 **random-interruption integration feasible / control gain unresolved**，不通过调整 timeout、事件强度或 success 定义制造正结果。动态页面为 `outputs/lewm_action_prefix_a4/towel_h6_policy_centered_random_interruption_rot5_clock03_pair12/index.html`。

#### A4 physical-state decoding（2026-07-18，exploratory mechanism diagnostic）

为检查预测 latent 的动作增量是否对应可解释物理量，在已经暴露的 A4 development split 上固定 ridge decoder，分别解码 object position、task-controlled position、task progress 和 TCP position。H6-H8 factual 相对 matched no-action 的 whitened MSE 改善分别为 `0.2861 [0.2299,0.3432]`、`0.3013 [0.2487,0.3521]`、`0.2172 [0.1437,0.2906]` 和 `0.3924 [0.2625,0.5197]`；四类量、七种 wrong-action shifts 的全部 `28` 个 paired CI 均严格为正，且 factual 明显胜 task-time。

该 ridge probe 固定 `alpha=0.001` 并仅在 train split 拟合；task-controlled readout 混合 object/TCP 语义，DirectAction comparator 读取 privileged H0 physical state。该结果只作 **exposed-development post-hoc mechanism evidence**。参数匹配的 DirectAction decoder 在 object、controlled-position 和 progress 上仍显著优于 factual LeWM，TCP 则近似持平；因此不能声称 LeWM 是物理状态预测的最优或必要接口，也不能升级为确认性/控制 claim。正式记录为 `outputs/lewm_action_branch_a4/physical_state_decoding_exploratory/report.json`。

#### 七任务五视角与夹爪开合 pilot（2026-07-19，development diagnostic）

为检查固定单视角训练是否掩盖视角依赖，并补足旧 Pick 轨迹完全没有松爪监督的问题，新增一次有界的五视角 pilot。一个共享 LeWM 同时训练全部七任务；每个物理 bundle 在 `corner / corner2 / corner3 / corner4 / topview` 五个固定相机下重渲染，batch `35 = 7 tasks x 5 views`，确保每次更新覆盖全部 task-view cell。动作分支另外加入 always-open、always-close、hold、half-close/half-open 和 close-then-open，且这些动作会在恢复同一物理 snapshot 后真实执行，再采集未来 RGB、proprio 和三锚点。训练固定 `2,000` steps、seed `40521`、`SIGReg=0.02`，并加入左右指尖投影间距辅助损失。

本 pilot 使用 `8 train + 2 development` 个 physical bundle/task，共 `350` 个 task-view rows；五个视角共享的物理状态逐值一致，future-anchor 有效率为 `99.9896%`。旧单视角 FCPG corrector 在 `corner2/3/4/topview` 会引入约 `4--22 px` 的 clean 偏差，因此此处只使用精确名义投影，目的是单独判断 LeWM 多视角/夹爪接口，不声称多视角 FCPG 已解决。

固定 latest@2000 的 development 结果为：七任务 task-macro latent prediction MSE `0.003707`，相对 action shuffle 的增量 `0.012595`，anchor EPE `3.698 px`，左右指尖 aperture MAE `1.255 px`。逐任务 EPE 为 Pick `1.840`、Reach `2.320`、Push `2.264`、Drawer `3.484`、Towel `3.462`、Window `4.639`、Door `7.879 px`；Door 的 `corner4` 是最差 cell，为 `11.447 px`。在自动选取的 Pick close-then-open development 分支上，五视角 EPE 为 `1.091--1.700 px`，aperture MAE 为 `0.708--1.759 px`，H8 松爪预测可在同一未来帧上直接与真值核对。

该结果只能记为 **multi-view coordinate/gripper feature pilot positive**：它证明一个共享七任务模型可以在训练过的五个固定视角上预测三锚点和夹爪开度，并不证明 held-out camera 泛化、FCPG 跨视角校正、长时递归控制或比参数匹配 direct dynamics 更优。动作敏感性仍需在 fresh split 上与 no-action/动作错配直接比较，不能由当前总体 MSE 升级成第二贡献。人眼审查页为 `outputs/lewm_action_prefix_a4/shared_7task_multiview5_gripper_seed40521_2k_visual/index.html`；其中上排为 evaluation-only 未来 RGB 与真值，下排在同一未来帧叠加蓝色预测、绿色真值和红色误差连线，模型推理不读取未来 RGB 或 `q_true`。

#### 七任务十二方位 shared LeWM 10k（2026-07-19，held-out-view development positive）

在五视角 pilot 之后，重新采集独立的十二方位 orbit-camera 数据，并只训练一个覆盖全部七任务的 shared LeWM。训练视角固定为 `12/2/4/6/8/10` 点钟，development 视角固定为从未进入训练 batch 的 `1/3/5/7/9/11` 点钟；模型不读取 camera ID。每个任务包含 `24 train + 8 development` 个 physical bundle，共 `1,344` 条 task-view row，其中 `1,008` 条用于训练、`336` 条用于 development。训练 batch 固定为 `84 = 7 tasks x 6 train views x 2 samples`，seed `40531`，latest endpoint `10,000` steps，参数量 `17,966,708`，`SIGReg` 仅使用 `0.02`。该实验继续使用 clean exact nominal grounding，目的是隔离视角与动作条件坐标动力学，不声称多视角 FCPG 已经完成。

六个 held-out development 视角上的 task-macro latent prediction MSE 为 `0.026542`，相对 action shuffle 的增量为 `0.009801`；三锚点 EPE 为 `6.442 px`，明显低于 hold-current 的 `15.784 px`，左右指尖 aperture MAE 为 `1.977 px`。逐任务 factual EPE 为 Door `11.195`、Drawer `3.671`、Pick `5.733`、Push `5.731`、Reach `5.233`、Towel `4.685`、Window `8.845 px`。最差 task-view cell 是未见的 `door-open-v3@clock03`，EPE `21.442 px`，因此当前模型不能声称对所有视角都稳定。

#### Orbit-12 split 与三锚点可视化合同澄清（2026-07-20）

这里的正式视角划分是：**偶数钟点训练，单数钟点 development/held-out 测试**。具体为训练 `clock12/02/04/06/08/10`，留出 `clock01/03/05/07/09/11`。这不是把同一视角随机拆到两边，也不是训练一个视角、测试另一个固定视角；七个任务共用同一个模型，单数钟点从未进入训练 batch。

正式 frame state 也不是单点预测。每帧固定保留 `visual_summary + anchor_0 + anchor_1 + anchor_2` 三个具有稳定物理身份的 anchor，并另外读出夹爪 aperture；三个 anchor 共同进入坐标和动作诊断。`visual_summary` 不能替代三个 anchor，任何只显示一个中心点的旧页面都不能用来判断模型是否学会夹爪或三点运动。

因此，正式 orbit 诊断应以 `outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k_full_clock03/index.html` 为准；`shared_7task_multiview5_gripper_seed40521_2k_visual/index.html` 是独立的五视角 pilot，不属于 Orbit-12 的 even/odd split。旧的 A4c center-only prediction-cycle 页面属于 legacy 可视化，不能作为三锚点或 held-out-view 证据。

为排除“只靠当前图像做静态平滑”的解释，又在完全相同的 current state 上执行 matched action controls。七任务 task-macro factual / cyclic-wrong-action / zero-action / hold-current EPE 分别为 `6.442 / 22.725 / 15.470 / 15.784 px`；factual 相对三类对照分别改善 `16.283 / 9.028 / 9.342 px`，且七个任务的 factual EPE 均为四者最低。该结果支持的准确结论是：**单个七任务模型能在六个未见方位上利用未来 executed action 预测未来机器人锚点和夹爪开度；它不是只复制当前坐标。** 该结论仍只有一个训练 seed、development split 与 clean nominal grounding，不证明 policy rollout、FCPG 跨视角校正、CEM 或真机控制。

后续对 Towel 的 same-physical-state / same-action 方位诊断发现了明确的 camera aliasing 边界。当前 batch 虽保存离散 `camera_index`，但模型和 `AnchorMetricReadout` 都不读取 camera ID、连续相机方位或完整外参；坐标头只从 predicted/current latent、当前二维锚点、task/anchor/horizon embedding 回归二维 delta。Orbit 相机始终看向操作区，使同一物理状态在六个视角下的当前三锚点最大平均差异只有 `4.09 px`，不足以稳定确定投影方向。代表性 Towel bundle `223`、branch `3` 的 H8 中，`clock03` 真实中心位移角为 `90.5 deg`，模型为 `60.2 deg`；`clock09` 真实为 `-89.6 deg`，模型为 `-38.2 deg`。尤其 `clock09` 的 latent MSE 仅 `0.00428`，但 H8 坐标 EPE 为 `21.74 px`，说明未见视角上的主要瓶颈之一是 latent-to-pixel readout 缺少连续相机几何，而不只是 LeWM latent prediction 误差。后续不得通过增加离散 camera embedding 宣称解决 held-out view；最小修复 Gate 应冻结当前 LeWM，向坐标 readout 显式提供连续 camera pose/basis，或先预测世界/相机坐标系三维锚点再通过已知 `K,T_nom` 投影，并在 odd-hour views 上重新比较方向误差和 EPE。

#### Camera-FiLM 与 `SIGReg=0.09` 诊断（2026-07-19，No-Go）

为直接检验上述边界，随后在相同十二方位数据上加入连续相机基底条件：`CameraLatentConditioner` 用相机 basis 对 latent 做零初始化 FiLM，`PoseConditionedResidualReadout` 再读取相机 basis 与二维锚点预测 residual。该结构仍然是自由二维 MLP，并没有执行 `world motion -> camera frame -> perspective projection`。固定 seed `40551`、latest@`2000`、其余训练合同一致时，`SIGReg=0.02` 与用户指定的 `0.09` 结果为：

| 配置 | odd-view task-macro EPE | 方向 MAE | latent prediction MSE | action-shuffle delta |
| --- | ---: | ---: | ---: | ---: |
| camera-FiLM, `SIGReg=0.02` | **`6.0731 px`** | **`27.193 deg`** | **`0.026789`** | `0.008390` |
| camera-FiLM, `SIGReg=0.09` | `6.2033 px` | `27.598 deg` | `0.035981` | **`0.013624`** |

`0.09` 只在局部 `towel-wipe-v0@clock03` 上把 EPE 从 `5.714` 降到 `5.365 px`、方向 MAE 从 `33.83` 降到 `32.22 deg`，但总体坐标和 latent 预测均变差。完整 Towel `clock03` 回放的 stitched EPE 仍为 `17.424 px`，H1-H8 EPE 从 `4.448` 单调累积到 `19.011 px`。同一 RGB、状态和动作只替换 camera basis 的诊断中，`0.09` 模型的 H8 预测方向跨 `clock01/03/05/11` 只变化约 `0.74 deg`，说明它基本忽略了相机条件。训练末端 `0.09 * SIGReg = 0.1209`，已占总 loss `0.1289` 的绝大部分；增大 anti-collapse 权重不能补回缺失的投影等变性。该分支正式判为 **No-Go**，不扩到 `10k`，后续恢复 `SIGReg=0.02`。

文献结论与该负结果一致。原始 [LeWorldModel](https://arxiv.org/abs/2603.19312) 只要求 action-conditioned next-latent prediction 与各向同性高斯 SIGReg，并未给出跨相机投影约束；[ReViWo](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html) 将 dynamics 放在 view-invariant representation 上；[PAIWorld](https://arxiv.org/abs/2606.18375) 则显式使用 camera ray/extrinsic-aware Geo-RoPE、跨视角交互和逐视角空间 action map，并指出无几何归纳偏置的 token concatenation 不足以保证三维一致性；[PointWorld](https://arxiv.org/abs/2601.03782) 直接在共享三维 point-flow 空间表示状态和由 URDF/FK 产生的机器人动作。共同信息不是“换一个更大的 Transformer”，而是：**动力学应在共享物理坐标系中表达，像素轨迹应由相机模型产生。**

#### World-frame 投影与 camera-frame action 容量 Gate（2026-07-19）

world-frame 三锚点读出按下式实现，所有视角使用同一三维预测，再通过各自已知相机矩阵确定性投影：

```text
(z_history, X_world_t, action_world, task)
    -> delta_X_world[H, K, 3]
    -> X_world_pred = X_world_t + delta_X_world
    -> q_pred_view = project(K_view, T_view_world, X_world_pred)
```

四个固定 `2k` 版本均未通过。最佳的 `5.55M` camera+command residual readout 将 world EPE 从 command integration 的 `51.755` 降到 `41.891 mm`，但 odd-view pixel EPE 为 `6.519 px`，仍弱于同一 evaluator 中原二维 base 的 `6.382 px`；同一物理 bundle 跨视角的 world prediction std 仍为 `19.513 mm`。因此当前 LeWM latent 并没有经该 post-hoc readout 自动变成 view-invariant world state，扩大 readout 不能解决问题，不扩到 `10k`。

随后采用更小且符合用户约束的修复：不向模型提供离散方位或 camera ID，只把每个 action chunk 中的五个世界坐标 XYZ 命令按连续 `R_cam<-world` 旋转到当前 camera frame，gripper 命令保持不变，并仅用 train split 重新估计 action normalization。该 `17.97M` 模型从原 orbit-12 shared `10k` checkpoint 初始化，固定训练 `2k`、seed `40571`、batch `84`、`SIGReg=0.02`。它将 odd-view task-macro EPE 从原模型的 `6.442` 和 Camera-FiLM 的 `6.073` 降至 `5.088 px`；方向 MAE 为 `23.518 deg`，cyclic-action EPE penalty 为 `18.326 px`，说明收益不是复制当前坐标。它没有学习离散视角 embedding，未见 odd-hour 方位只通过连续旋转进入。

为排除“只是比原 parent 多训练 `2k`”这一混杂，另从完全相同的 orbit-12 shared `10k` parent 启动 matched world-action continuation：参数量、数据、seed、batch、优化步数与 camera-action 分支一致，只保留世界坐标动作。fixed endpoint 的 EPE/方向 MAE 为 `6.1655 px / 27.7201 deg`，camera-action 为 `5.0878 px / 23.5178 deg`，分别改善 `17.48% / 15.16%`；Towel `clock03` 也从 `5.8836 px / 34.4537 deg` 改善到 `3.6864 px / 27.9781 deg`。但 camera-action 的 latent action-shuffle delta 从 `0.008581` 降至 `0.006406`，因此这个 matched control 支持的是**动作坐标对齐改善像素运动解码**，不能外推为更强的通用 latent dynamics。

为判断剩余误差是否只是容量不足，又完成同数据、同额外 `2k` optimizer-step 预算的 bounded screen。两种 Adapter 输入完全相同，均读取 current/base prediction、base innovation、action-prefix condition、anchor/horizon/task identity；输出头零初始化，并按 train-only latent residual 的逐维 `q99` 严格限制，cap 的 min/median/max 为 `0.04094 / 0.09938 / 0.22266`。parent 全部冻结，因此 `SIGReg=0.02` 只属于 parent 表征合同，不对 Adapter 产生梯度。

| 额外训练 | 总/新增参数 | EPE | 方向 MAE | Towel `clock03` EPE / 方向 MAE |
| --- | ---: | ---: | ---: | ---: |
| matched world-action continuation @+2k | `17.97M / 0` | `6.1655` | `27.7201 deg` | `5.8836 / 34.4537 deg` |
| camera-action parent @2k | `17.97M / 0` | `5.0878` | `23.5178 deg` | `3.6864 / 27.9781 deg` |
| bounded anchor Transformer | `18.75M / 0.7828M` | `5.0810` | `23.6275 deg` | `3.6813 / 28.0319 deg` |
| matched token-local MLP | `18.75M / 0.7812M` | `5.0823` | `23.6180 deg` | `3.6957 / 28.1507 deg` |
| no-new-parameter full continuation @4k | `17.97M / 0` | **`4.7885`** | **`21.9355 deg`** | `4.0183 / 29.0318 deg` |
| fresh-schedule camera-action @10k | `17.97M / 0` | **`4.7684`** | `22.0619 deg` | `5.7632 / 35.5792 deg` |

Transformer 与参数匹配 MLP 都只改善约 `0.1%`，方向误差略增，因此没有证据支持“anchor attention 或更大 Adapter”是当前瓶颈。full continuation 的总体 EPE 改善 `5.88%`、方向 MAE 改善 `6.73%`，并保留 `18.639 px` cyclic-action penalty，故只允许了一次 fresh-schedule `10k`；但 Towel `clock03` 局部退化必须单独保留，不能被总体均值掩盖。此前 `51.52M` expansion 把 `27.59M` 参数堆在十层 `4096D` residual MLP，且同时解冻 parent，没有形成公平固定 endpoint，已停止且不进入结果表。

固定 `10k` 已使用同一原始 parent、数据、seed `40571`、batch `84` 与 `SIGReg=0.02` 完成，并且只读取 `latest@10000`。相对 camera-action `2k`，总体 EPE/方向 MAE 改善 `6.28% / 6.19%`，但 latent prediction MSE 恶化 `27.41%`，action-shuffle delta 下降 `42.88%`；相对累计 `4k`，总体 EPE 只再改善 `0.42%`，方向 MAE反而恶化 `0.58%`。更关键的是 Towel `clock03` EPE/方向 MAE 相对 `2k` 恶化 `56.33% / 27.17%`。因此结论冻结为 **camera-frame coordinate alignment positive / Adapter and long-training capacity rescue No-Go**：连续相机旋转是必要输入，但训练更久或堆容量都没有解决全部 held-out-view aliasing。三维状态、相机旋转、Transformer、SIGReg 和确定性投影本身均为已有工具；潜在论文贡献仍只能来自 FCPG observation update 与 action-conditioned persistent grounding 的统一闭环。

完整人眼审查页按旧 `joint_accuracy_gallery_seed40501_2k` 的播放合同生成：左侧真实轨迹每前进 `5` 个 raw environment step 后暂停，右侧在冻结当前观测下播放完整 `40` raw-step（`H1--H8`）预测，再进入下一轮；未来 RGB 与 `q_true` 只用于推理后的显示和评分。页面默认使用连续 `30 FPS` MP4，GIF 仅为最多 `200` 帧的抽样预览：

```text
outputs/lewm_action_prefix_a4/camera_action_seed40571_2k_full_clock03/index.html
outputs/lewm_action_prefix_a4/camera_action_seed40571_2k_towel_clock03_full/index.html
outputs/lewm_action_prefix_a4/camera_action_seed40571_10k_towel_clock03_full/index.html
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/matched_action_evaluation.json
```

其中 `camera_action_seed40571_2k_full_clock03/index.html` 是最新 `2k` checkpoint 的七任务完整页面；其余两个 `towel` 页面只覆盖单个 Towel episode。

两份新页面使用同一个 `reserved-test` Towel instance、`clock03` 相机和 `371` 个真实 raw frames，各自生成 `3,050` 帧连续 MP4 与 `200` 帧抽样 GIF。完整任务 max-lookahead anchor EPE 为 `2k: 7.8701 px`、`10k: 8.1134 px`；H8 为 `8.4218 / 8.8022 px`。这是一份直接人眼审查和单 episode 诊断，不替代 task-macro 统计；它再次表明 `10k` 没有修复目标视角。

可视化必须区分局部 Gate 与完整任务：A4 的 `H=8 x 5=40` raw steps 只是 matched-action mechanism window，不是 episode 截止点，也不能由此推断更长时域已经受训练或验证。完整 development 预览从 reset 连续播放到三帧稳定 success 并保留 `8` 个真实 post-success step；七任务分别包含 `62 / 215 / 56 / 73 / 85 / 101 / 269` 个逐 raw-frame 画面，Push 的首次 success 在 raw step `62`。Drawer 的 `215` 帧对比在 raw step `10` 后冻结右侧 RGB，不读取未来图像；LeWM 使用 raw `0/5/10` 的三帧历史，并对随后每个完整五动作 chunk 递归预测，`H9+` 明确标为超出训练 Gate 的 open-loop extrapolation，最后不足五步不伪造模型节点。单画面 GT/LeWM 对比见 `outputs/lewm_action_branch_a4/full_trajectory_preview/index.html`。由于叠加像素轨迹使用的 train-only latent-to-pixel readout 未通过冻结 Gate，该页面只能显示“短期方向大致一致、长时 open-loop/readout 逐渐漂移”的失败诊断；它既不是 215 帧可部署预测证据，也不能单独判定 Anchor-Persistent dynamics 的长程上限。

#### A4 q-center latent-to-pixel decoding（2026-07-17，Exploratory No-Go）

为判断该失败是否仅来自线性 readout 太弱，又完成了一次固定视角、privileged `q_true` 起点的 q-center 非线性 Phase-0 诊断。三组非线性 head 只在 train episode 的 canonical target latent 上拟合，development episode 与训练物理隔离；主 head 不读取 horizon、time 或 action，随后冻结并原样用于 factual/no-action/wrong-action latent。Target latent 的 mean/P95/PCK@8 为 `5.384 / 11.700 px / 0.786`，factual 为 `7.267 / 14.384 px / 0.656`，仍未通过 `2 px` target 与 `4 px` factual 的冻结绝对门槛。Factual 虽相对 matched no-action `14.901 px` 和 wrong-action `15.526 px` 分别改善 `51.23% / 53.19%`，并在 `3/3` LeWM seed、`7/7` 任务上同向，但没有优于 TaskTimeQ0 `7.034 px`，且显著弱于参数匹配的 DirectAction `2.313 px`；后者使用同样的 q0、三帧 anchor history、历史动作与 expert action prefix，factual-minus-direct 的配对差为 `+4.954 px [4.517, 5.334]`。

因此该诊断判为 **Exploratory No-Go**：A4 latent 含有动作敏感的粗几何信息，但当前 latent-to-coordinate 接口既不够精确，也没有证明相对直接 history+action 坐标预测的必要性。它不能支持跨视角、部署式 q0、逐 anchor identity、FCPG 控制收益或 reserved-test claim；不启动原计划的正式三锚点坐标 collector/head。若继续研究精确轨迹，下一候选应是以部署可得 `q_corrected/q_nominal`、历史和动作直接预测坐标的显式状态动力学头，或在 LeWM 内部加入坐标辅助监督后重新训练，而不是继续扩大 post-hoc latent decoder。正式结果位于 `outputs/lewm_action_branch_a4/qcenter_geometry_gate_phase0_seed40901/summary.json`。

#### A3-to-policy known-drop interface（2026-07-18，privileged ceilings Go / strict bridge No-Go）

215 帧预览中的像素 readout 不是策略真实接口。为直接检验缺测状态是否可能影响既有控制，本诊断在冻结 `GroundedTokenAdapter` 之后、策略 memory 之前增加严格 masked 的 grounded-token override；空 mask 必须逐值保持原策略输出。A3 的控制时钟固定为每个 transition `5` 个 raw action，Pick seed-0 的 `20` episode clock screen 在 `5/8` raw action per decision 下成功率为 `0.50/0.55`，因此后续开发使用与 A3 一致的 `5` 步时钟；该小样本差异不构成性能比较。

首先执行 Pick seed-0、`20` 个 development episode 的 **decision-synchronous known-drop native-token oracle**。clean / corrupted measurement / hold-last / oracle 的成功率为 `0.70 / 0.55 / 0.55 / 0.65`。Oracle 相对 measurement 与 hold-last 都提高 `10 pp`，但配对 95% CI 均为 `[-15,+35] pp`；更明确的机制信号是相对 clean-policy action 的距离分别改善 `0.247057 [0.214139,0.284711]` 与 `0.274303 [0.245532,0.301970]`。Oracle 自身仍比 clean 多 `0.002700 [0.001783,0.003738]` 的动作距离，成功率低 `5 pp`。该实验在每个决策点直接使用同物理时刻的 native clean token，并同步替换两个策略历史 slot，只是 privileged policy-interface upper bound；它不使用 A3 prediction，不满足 onset/recovery 的因果 slot 时序，也不能声称显著提高成功率。

随后用另一个 fresh `20`-episode Pick panel 隔离 P1 的两槽时钟。四种 hidden-clean override 的 active-decision action MSE 分别为 raw-adjacent `[t-1,t]` `0.000203`、repeat-current `[t,t]` `0.001221`、decision-spaced `[t-5,t]` `0.008271` 和 hold-last `0.077244`。Raw-adjacent 相对 decision-spaced 改善 `0.008069 [0.006678,0.009458]`，相对 repeat-current 改善 `0.001019 [0.000697,0.001406]`。对应成功率为 `0.75 / 0.75 / 0.80 / 0.05`，但各方法执行后物理轨迹已经分叉，因此成功率只作诊断。该 privileged screen 不使用 LeWM、不训练 bridge，只证明冻结 P1 原生期待 raw-adjacent 历史，不能把两个 A3 decision state `[t-5,t]` 直接当作同一语义的策略输入。

随后三类 `2,000`-step representation bridge 均未形成 deployable 接口。Absolute `192D -> 256D` bridge 使用 shared-4task policy、旧 teacher-forced `2k` prior 和物理 episode 隔离的 `90/30` train/validation episode；validation token MSE 为 main `0.005456`、参数匹配 geometry `0.005258`、mean-token `0.004637`，main 相对 geometry 为 `-3.76%`，仅 `2/3` 任务正向，并出现明显 train/validation gap。以 last-native token 为零初始化基座的 residual bridge 仍失败：teacher-forced factual prior 为 `0.001388`，弱于 hold-last `0.001059`、geometry `0.001176` 与 delta-blind `0.001364`，对 hold-last 为 `-31.13%` 且 `0/3` 任务正向；即使把当前真实 `192D` measurement embedding 作为 privileged oracle-current 输入，MSE 仍为 `0.001308`，相对 hold-last 为 `-23.50%`、`0/3` 正向。因此这些结果否定的是当前单时域 shared-policy 表征转换，不是否定 A3 dynamics 或所有潜在 token bridge。

为区分“接口完全没有上限”与“单时域训练目标不合适”，又在同一冻结 policy 上执行 H1-H8 privileged horizon ceiling。validation 覆盖 `18,600` 个 horizon pair、`30` 个物理 episode 和 Pick/Reach/Push 三任务；bridge 读取目标时刻的真实 current measurement embedding，**不读取 A3 prediction**。Oracle-current bridge 的 overall token MSE 为 `0.002678`，相对 hold-last `0.003407` 和 geometry-hold-exact `0.003403` 分别改善 `21.39% / 21.29%`；相对参数匹配 delta-blind `0.002985` 改善 `10.27%`。对 hold-last，H1 改善 `8.46%`（`2/3` 任务正向），H2-H8 改善 `16.77%--23.87%` 且均为 `3/3` 任务正向。该结果只证明 H1-H8 的策略 token 映射存在 privileged representation ceiling，不证明 LeWM 能预测这些 future embeddings，更不证明 `40` raw steps 的控制收益。

时钟诊断后，因果 runtime 的覆盖合同固定为：视觉有效时不覆盖任何 slot，原样保留 P1 的 native raw-adjacent `[t-1,t]`；视觉无效时只存在一个当前 bridge estimate，因此将其重复为 `[pred_t,pred_t]`，而不是伪装成 `[pred_{t-5},pred_t]`；恢复帧立即撤销 mask，重新采用完整 native 历史。至少需要三次有效观测 warmup，A3 只接收实际执行、裁剪后的 `5 x 4` action block。Factual、matched no-action 与 geometry/hold control 各自维护独立递归历史；learned path 不得读取 hidden clean token。

严格 collector 随后重放已暴露的 A3 train/development physical rows，并同时运行冻结 A3 与精确 Pick-P1 两套 encoder；它不是 fresh 或 independent split。最终 cache 含 `90` source rows、`36` physical episodes 和全部 `8 x H1--H8` branch/horizon，按 physical episode 分为 `62/28` source rows、`24/12` episodes，对应 `3,968/1,792` flattened train/validation rows。三个 A3 `10k` seeds `29301/29302/29303` 只是同一物理样本上的相关 prediction replicate，不作为独立样本。A3 history/target 最大重建误差为 `3.10e-4/4.75e-4`，低于冻结 `1e-3` 容差；P1 checkpoint、pick-specific corrector、动作、split 与 cache SHA 全部 fail-closed 校验通过。

在该 exposed-development cache 上，分别训练同容量 factual 与 matched-no-action residual bridge，三个初始化 seeds `15715/15716/15717` 均为 **formal screen No-Go**；其 factual validation MSE 分别为 `0.008357 / 0.007759 / 0.007758`，全部弱于共同 hold-last `0.007044`。代表 seed `15715` 的 matched-no-action / exact geometry-hold / target-oracle-bridge MSE 为 `0.007790 / 0.007040 / 0.007458`。Factual 相对 geometry-hold 的改善为 `-18.71% [-59.07,+11.26]`，相对 no-action 为 `-7.28% [-18.76,+4.18]`；target-oracle 虽相对 factual 改善 `10.75%`，仍比 hold-last 差 `5.89%`。代表 seed 在 validation 的 H1-H8 每个 horizon 都弱于 hold-last，而 train factual 已到 `0.002361`、明显优于 train hold `0.007640`，说明当前小样本 `192D A3 -> 256D P1` bridge 主要拟合了训练 episode，没有形成可迁移接口。

准确结论是 **strict exposed-development cross-backbone interface No-Go**：停止 causal policy rollout、CEM 和同 split 调参，也不把该路线列为第二贡献。它否定当前 A3-10k/P1 residual conversion，不否定 A3 的动作条件 anchor transition、H1-H8 privileged representation ceiling、Stage I FCPG 或 LeWM 的一般价值。

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

#### P1-native known-drop dynamics（2026-07-18，descriptive positive / Development-No-Go）

为隔离上一节失败究竟来自 A3-to-P1 表征转换，还是策略原生 token 本身不可预测，本 Gate 不再读取 A3 `192D` state，而是直接在冻结 Pick-P1 的 `256D` grounded-token 空间训练同容量 factual 与 matched-no-action dynamics heads。严格 cache 沿用已暴露的 physical rows，按 episode 划为 `24 train / 12 development`，每个训练 batch 抽取 `8` 个互异 physical episode、每 episode 一个 hazard，并保留同一 bundle 的全部八条动作分支。H0-H2 是 dropout 前可见 token，H3-H8 只作 hidden labels；动作时钟严格覆盖 H0->H1 至 H7->H8。所有统计均以 physical episode 为独立单位，七种 wrong-action 只循环置换 H2 后的未来动作。

三个初始化 seed `15801/15802/15803` 各训练固定 `2,000` steps。Development factual MSE 为 `0.003857 / 0.003863 / 0.003814`；相对 matched no-action 的改善为 `27.67% / 28.07% / 30.39%`，CI 下界均为正。相对 hold-last 的点估计改善为 `22.97% / 22.85% / 23.83%`，相对 exact-FK geometry-hold 为 `22.93% / 22.81% / 23.80%`，相对 exact-FK visual-CV 为 `90.90% / 90.89% / 91.01%`。全部七种 cyclic wrong-action 对照在三个 seed 上均通过，说明 native dynamics 确实使用了未来动作，而不是纯时间平滑。

失败集中在缺测 onset。相对最强非学习基线，三个 seed 的 H3 分别退化 `37.22% / 37.96% / 37.02%`；H4 则转为约 `4.56%--4.72%` 改善，H5-H8 改善扩大至约 `20.56%--37.79%`。同时，因为 development 只有 `12` 个独立 episode，hold-last 与 exact-FK geometry-hold 的配对 CI 下界均小于零，未满足冻结的 `>=10%` 且 CI 下界大于零条件。严格三 seed汇总因此为 **Development-No-Go（0/3 single-seed Gates passed）**。

准确边界是：去掉跨 backbone bridge 后，P1-native action-conditioned dynamics 在 H4-H8 显示稳定增量，但首个 hidden step H3 仍不安全，且当前 exposed split 未能统计上击败强 hold/exact-FK baseline。该 Gate 不授权 `10k`、policy rollout、CEM 或论文第二贡献，也不应改写为 LeWM 或 native-token dynamics 一般无效。若另开新实验，必须先针对 onset 的 measurement-to-prior transition 设计新因果接口并使用 fresh physical split；不得在当前 development set 上继续调 loss、宽度或阈值。

```text
outputs/lewm_p1_native_dynamics/dev_seed1580{1,2,3}_2k/
outputs/lewm_p1_native_dynamics/cross_seed_summary/
```

#### P1-native H3 根因与互补观测器假设（2026-07-18，post-hoc development）

对 exposed development prediction 做 common-mode / branch-centered 分解后，H3 失败已定位为绝对状态的共享漂移，而不是动作分支时钟错位。以 seed `15801` 为例，factual H3 总 MSE 为 `0.003341`，其中 common-mode 为 `0.001566`、branch-centered 为 `0.001775`；exact-FK geometry-hold 对应为 `0.002385 / 0.000597 / 0.001788`。因此 factual 的动作相关部分相对 geometry-hold 反而改善约 `0.7%`，但 common-mode 恶化约 `162.5%`，吞掉了全部收益。H4 仍表现为 common-mode 退化、branch-centered 改善，H6 后两部分才同时转正。H3 在 train 上相对 hold 改善 `36.6%`，到 development 却退化 `40.1%`；结合每个 head 约 `16.3M` 参数而训练集只有 `24` 个独立 physical episode，当前 absolute-token predictor 存在明确的 onset 过度修正和小样本过拟合。

基于该诊断，只在已暴露 development 上执行了一个**零训练、递归写回的互补观测器 headroom**。令 `b_h` 为 last-valid visual token 与当前可用 proprio/FK 构造的 geometry baseline，`p_h` 为冻结 P1-native LeWM prior，并使用缺测年龄相关的固定增益：

```text
z_h = b_h + K_h * (p_h - b_h)
K_H3:H8 = [0.25, 0.50, 0.75, 0.75, 0.75, 0.75]
```

每一步融合后的 `z_h` 写回各自递归历史，再预测下一步。三个冻结 `2k` seed 的 post-hoc task-macro MSE 约为 `0.003456 / 0.003431 / 0.003408`；相对 geometry-hold 改善 `30.94% / 31.44% / 31.91%`，paired CI 下界约为 `12.9% / 14.7% / 14.9%`；相对使用相同增益的 matched no-action 改善 `20.59% / 21.71% / 22.44%`，CI 下界均大于 `10%`。全部七种 wrong-action control 的 CI 下界均为正，H3 约从 raw factual 的 `0.0033` 降至 `0.00204`，H3-H8 每个 horizon 均优于 geometry-hold。该结果说明 geometry baseline 与 LeWM action prior 具有互补信息，也说明 raw LeWM prior 的主要问题是 onset 校准，而非缺少动作动力学。

上述增益是在已暴露 development 上观察后确定，**不得改写原冻结 Gate 的 `Development-No-Go`，也不得作为论文结果**。若继续，只允许先冻结三个现有 `2k` checkpoint、增益序列与判定线，在至少 `30` 个全新 physical episode 上执行一次 zero-training fresh Gate：valid/clean/recovery 时必须硬旁路并与原生 measurement bitwise 一致；overall 相对 geometry-hold、hold-last 和 matched no-action 均改善至少 `10%` 且 paired 95% CI 下界大于零；H3 相对最强非学习基线的点估计不得为负且单侧下界不得低于 `-5%`；至少 `5/6` 个 horizon 正向；七种 wrong-action CI 下界全部大于零；至少 `2/3` 个模型 seed 通过。通过只授权 known-drop policy rollout，不授权自动遮挡检测、selector、CEM 或 `10k` 扩训。

#### P1-native geometry-anchored complementary observer（2026-07-18，Fresh-Go）

冻结上述三个 `2k` factual/no-action checkpoint、增益序列和全部判定线后，正式采集 Pick `30` 个新 physical episode；每个 episode 保留 `8` 条 same-state action branch，H0-H2 是 deployment input，H3-H8 只在预测完成后作为 offline label。Freshness 审计使用完整 `39D` reset observation 与三枚 FK keypoint 的精确签名：fresh `30/30` 唯一，与旧 P1 训练源的 `30,000` 个签名交集为零。旧训练数据缺完整 reset digest receipt，因此这里只声称 exact observed-state non-overlap，不声称完整 reset-state non-overlap。

第一次 collector 执行在发布 cache 前 fail closed：fresh 端误把 `4D` robot-only proprio 与 legacy 的 `39D` observation 放入不同签名空间，导致 fresh 内 `1/30` 唯一且 legacy overlap 检查无效。该次没有写出 cache、prediction 或 metric；修正为相同的完整 observation domain、加入 shape 回归检查后，原冻结 panel 原样重跑。正式 deployment/offline-label cache 的 SHA-256 分别为 `947aed59290032d9993f6ddde88d330e21c8687e6c6b2cf619f7fd42298f62b7` 与 `52b670b4e41106e7399544ff85c9b770b2309133ea0bff2bd0ad2a5813c70cf6`。

正式 panel 共 `6,480` 行，以 `30` 个 physical episode 为统计单位并在全部 method、horizon 与 model seed 间共享同一 bootstrap draw matrix。三个 seed `15801/15802/15803` 的互补观测器 MSE 为 `0.007202 / 0.007239 / 0.007309`；相对 exact-FK geometry-hold 改善 `51.60% / 51.35% / 50.89%`，相对 hold-last 改善 `51.63% / 51.38% / 50.92%`，相对 matched no-action 改善 `33.90% / 35.25% / 35.24%`，所有 paired 95% CI 下界均显著高于冻结的零阈值。H3 相对最强非学习基线的点改善为 `20.99% / 21.40% / 21.33%`，单侧 95% 下界为 `19.49% / 19.98% / 19.97%`；三个 seed 均为 H3-H8 `6/6` 正向、七种 wrong-action `7/7` 通过，因此 cross-seed decision 为 **Fresh-Go（3/3）**。

该结果把先前的 post-hoc headroom 升级为 fresh known-drop representation evidence，并说明当前 frozen LeWM 的有用部分是相对 geometry baseline 的动作条件增量，而不是绝对 token 输出。它尚未测试自动发现视觉失效或 unknown corruption；valid-measurement bitwise bypass 目前是实现合同与单元测试，不是本 panel 的经验性 clean claim。随后唯一获准的 paired known-drop policy rollout 已完成，结论见下一节。原先预注册的 residual-LeWM fallback 未被触发，继续禁止 10k 扩训、加宽、追加 SIGReg、重训 VisualTrust 或启动 CEM。

```text
outputs/lewm_p1_native_complementary_fresh/
outputs/lewm_p1_native_complementary_fresh/evaluation/summary/summary.{json,md}
```

#### P1-native paired known-drop policy rollout（2026-07-18，mechanism-positive / policy-success Development-No-Go）

唯一获准的 development rollout 使用 Pick policy seed `0`、`30` 个 fresh physical episode、nominal corner camera 和一次六决策的局部黑色视觉缺测；缺测起点在 decision `6--9` 间预先采样，episode 固定为 `335` raw steps。`clean_native` 先建立同一物理状态的 clean-policy 参照；其余七种方法在同一个 Python 进程内共享 reset、相机、缺测 schedule、drop-onset physical-state digest 与逐字节相同的 pre-drop action/token/visual prefix。视觉缺测期间 raw history 接收真实 corruption，posterior 逐决策递归写回；视觉有效与恢复时硬旁路到原生 FCPG。`q_true` 仅用于外部绘制遮挡，hidden-clean action 仅作为同状态 action-distance 离线标签，不进入 policy、LeWM 或互补更新。

三个冻结 factual checkpoint 的 success 分别为 `43.33% / 50.00% / 43.33%`，exact-FK geometry-hold 为 `43.33%`；逐 seed success 差为 `0 / +6.67 / 0 pp`。按 physical episode 聚类、保留同一 episode 的全部模型 seed 后，factual 相对 geometry 的 seed-macro success 差为 `+2.22 pp`，95% CI `[-10.00,+14.44]`；相对各自 matched no-action 为 `+6.67 pp [-6.67,+21.11]`。因此二值策略成功率没有通过 development Gate。

机制指标则在三个 seed 上一致正向：相对 geometry 的 action-distance 分别改善 `45.67% / 40.70% / 47.27%`，seed-macro 为 `44.55% [34.96%,52.71%]`；平均累计 reward 分别增加 `139.45 / 164.02 / 137.01`，seed-macro 为 `+146.83 [50.49,253.63]`。相对 matched no-action 的 action-distance seed-macro 改善为 `45.38% [35.69%,53.60%]`，reward 增加 `139.07 [41.06,259.07]`。这些结果说明 LeWM 增量能让缺测期动作更接近同状态 clean policy，并改善平均回报，但 action-distance 和 reward 不能替代 success claim；当前 primary success 又是 episode 内 any-step success，不能事后改成 final/sustained success 翻案。

准确结论为：**known-drop representation Gate 仍然 Fresh-Go；paired rollout 的 action-consistency/reward mechanism positive，但 policy-success Development-No-Go。** 不启动 `240/300` episode confirmation，不训练 selector，不启动 CEM，也不把该路线列为论文第二贡献。未来若以全新协议重开，应预先使用 final/sustained success、跨 policy seed 和功效分析后的 physical-episode 数量；当前 evaluator 已增加 `final_success`、`success_fraction` 与 `maximum_success_streak` 供未来协议使用，但本次已完成面板不事后补算或更换 primary。

正式 development 结果只引用单进程三 seed 面板：

```text
outputs/lewm_p1_native_policy_rollout/development_all3_seed41901/
outputs/lewm_p1_native_policy_rollout/development_all3_seed41901/policy_rollout_report.json
```

此前分进程生成的 seed-specific reports 仅作工程诊断；它们的 clean pre-drop action prefix 未达到逐字节一致，不能与正式单进程面板合并为 `90` 个独立 episode，也不能用于跨 seed 汇总。

#### P1-native joint camera-drift + known-drop（2026-07-22，action-fidelity Confirmation-Go / task-control Confirmation-No-Go）

为避免 Towel expert/controller 的 `100%` success 天花板，在同一个非饱和 learned Pick policy 上组合 Stage I 已使用的相机漂移与 P1-native 视觉中断。新 screen 使用 `12` 个新 benchmark physical episode、`corner` 相机的固定 `5 deg world-z` 旋转和 decision `6--9` 间的六决策局部视觉中断。所有方法共享 reset、实际相机、drop schedule 和 pre-drop action/token/visual prefix；正常视觉由冻结 FCPG 处理，已知无效期分别使用损坏 measurement、exact-FK geometry carrier、三个 factual complementary observer 和各自 matched no-action。该 screen 不训练任何模型，也不使用 CUDA1。

三个冻结 factual LeWM seed 相对 exact-FK geometry 的 physical-episode clustered seed-macro 结果为：clean-action distance 改善 `55.709% [46.059%,63.016%]`，累计 reward 增加 `97.258 [13.544,217.900]`，最大连续成功帧增加 `5.694 [1.167,11.417]`。相对 matched no-action，三项分别为 `52.895% [43.826%,62.006%]`、`+73.084 [16.862,133.065]` 和 `+9.417 [1.194,19.278]`。三个模型 seed 的上述方向一致。

二值结果仍需保守解释：factual 相对 geometry 的 any-step success 只增加 `2.78 pp [0,+8.33]`，相对 no-action 为 `11.11 pp [0,+27.78]`；固定 `335` 步末端的 `final_success` 对所有受扰动方法均为零。因而本结果只升级为 **联合漂移和已知中断下的 learned-policy control-mechanism Development-Go**：动作条件 LeWM 增量不仅更接近 clean policy，而且在同一配对面板中改善累计 reward 与持续成功。它仍只有一个 policy checkpoint、一个任务、`12` 个 physical episode 和外部已知 validity，不能声称显著 success gain、自动视觉可靠性判断或 T-RO 级完整系统证据。

随后在读取新结果前，将确认协议、三枚冻结 LeWM seed、`30` 个 physical episode、固定 `5 deg world-z` 旋转、六决策中断、方法面板和 primary 一并提交为 `22ff7ec`。确认配置拒绝通过 CLI 改 episode 数、模型 seed 或方法；primary 固定为 factual 相对 exact-FK geometry 的 clean-policy action-distance 相对改善 `>=10%` 且 clustered 95% CI 下界大于零。新 panel 使用 benchmark seed `52901` 和 environment seeds `950000..950029`，与 `12`-episode screen 分离，配对审计通过。

冻结确认的 action-fidelity primary 明确通过：factual seed-macro 相对 geometry 改善 `46.703% [37.445%,54.326%]`，相对 matched no-action 改善 `51.100% [45.102%,56.529%]`。Seeds `15801/15802/15803` 相对 geometry 分别改善 `50.867% / 39.563% / 49.679%`，各自 CI 下界均大于零；因此动作条件 LeWM 确实提供了 no-action 与 exact-FK carrier 没有的 clean-policy action information。

但 task-control 支持证据没有复现。相对 geometry，reward 为 `+67.804 [-0.862,+152.372]`，最大连续成功帧为 `+1.867 [-5.256,+8.833]`，两者 CI 均跨零；any-step success 反而为 `-12.22 pp [-24.44,-2.22]`。相对 matched no-action，reward 与最大连续成功帧同样不显著，any-step success 为 `-8.89 pp [-18.89,-1.11]`。这不是 action predictor 失败，而是证明当前 clean-policy action-distance 不是 control-sufficient objective：factual observer 更忠实地恢复了一个在该联合扰动 panel 上本身只有 `26.67%` success 的 frozen policy，而 geometry/no-action 的较大动作偏差在该 panel 上对应更高的 any-step success；现有数据不能进一步确定其因果机制。

最终边界固定为：**P1-native dynamics/observer 的 action-conditioned propagation 获得 fresh confirmation；当前 frozen-policy 控制接口为 Confirmation-No-Go。** 不启动更大 success panel、CEM 或 automatic validity selector，也不把该接口列为论文第二贡献。若后续继续，只能使用新的 control-aligned policy/objective，使训练目标直接对应任务 progress/success；不能继续优化 clean-action imitation 后再把它解释成控制收益。

实现与结果：

```text
config/lewm/eval/p1_native_joint_drift_drop_screen.json
config/lewm/eval/p1_native_joint_drift_drop_confirmation.json
outputs/lewm_p1_native_policy_rollout/joint_drift_drop_screen_all3_seed51901/
outputs/lewm_p1_native_policy_rollout/joint_drift_drop_screen_all3_seed51901/policy_rollout_report.json
outputs/lewm_p1_native_policy_rollout/joint_drift_drop_confirmation_seed52901/
outputs/lewm_p1_native_policy_rollout/joint_drift_drop_confirmation_seed52901/policy_rollout_report.json
```

#### FCPG x LeWM 同策略四格干预（2026-07-23，v2 严格配对完成）

为补齐旧联合实验“所有可见帧都使用 FCPG”造成的因子缺口，新增零训练的同策略 `2 x 2` 干预：`F0L0` 在可见帧使用 nominal grounding、缺测期使用 exact-FK geometry hold；`F1L0` 只将可见与恢复帧换成 FCPG；`F0L1` 在 nominal measurement 上启用 factual LeWM posterior；`F1L1` 同时使用 FCPG measurement 与 factual LeWM。四格共享 reset、相机漂移、遮挡 schedule 和遮挡前实际执行动作，但不强制共享 nominal/FCPG token，因此遮挡开始时物理状态相同，而 LeWM 接收各自真实的 grounding 初始化。视觉恢复首帧按所属因子硬重锚，不训练 Adapter、policy 或 dynamics。

8-episode smoke 的 sustained success 为 `12.5 / 12.5 / 12.5 / 50.0%`。随后在新 benchmark seed 上执行 `30` 个 physical episode，四格 sustained success 为 `33.33 / 26.67 / 30.00 / 40.00%`。按 episode bootstrap，交互项

```text
F1L1 - F1L0 - F0L1 + F0L0
```

为 `+16.67 pp [3.33,30.00]`；完整系统相对 FCPG-only 为 `+13.33 pp [3.33,26.67]`，相对 LeWM-only 为 `+10.00 pp [-3.33,23.33]`。但二值配对的独立信息仍少：`F1L1` 相对 `F1L0` 只有 `4` 次净胜、`0` 次净负，双侧精确检验 `p=0.125`；相对 `F0L1` 为 `4/1`，`p=0.375`；相对 `F0L0` 为 `3/1`，`p=0.625`。此外 FCPG 与 LeWM 的平均主效应 CI 均跨零，完整系统相对 neither 仅 `+6.67 pp [-6.67,20.00]`。

因此本面板只支持：**FCPG 校正后的可见 measurement 作为 LeWM 缺测传播的初始化与恢复锚点，存在值得继续验证的正交互信号。** 它尚不能证明两个模块在同一控制条件下各自显著，也不能形成 task-success 或第二贡献 claim。后续若继续，应使用更高 clean-success、直接消费同一 anchor-belief state 的 policy，并在 drift-only、dropout-only、joint 三类条件下复用同一个 policy checkpoint；不得再通过动作 residual、DAgger 或扩大 sidecar 解释该信号。

随后审计发现，上述四格统一加载了在 FCPG policy-token 空间训练的 LeWM；因此 `F0L1` 将 nominal token 输入 FCPG-trained dynamics，旧 `+16.67 pp` 交互可能混入 train/eval token-space compatibility，不能作为最终联合效应。修正实验保持模型结构、优化器、`2,000` steps、policy、物理分支和监督目标不变，只为 nominal grounding 重放同一 `90 x 8` 分支缓存并训练 seeds `15801/15802/15803` 的 matched LeWM。两份缓存的 episode、动作和 split 完全一致；FCPG/nominal sampled-visual token RMSE 为 `0.01817`，而离线监督 target RMSE 仅 `2.59e-5`。三个 nominal 模型均在 H4--H8 超过最强非学习基线，并通过全部七个 cyclic wrong-action 对照；H3 仍退化，因此这些模型只用于公平消融，不产生新的独立 dynamics claim。

matched 四格在同一 `30` 个 physical episode 上复用 seed `15801` 保存的 pre-drop actions，使三个模型 seed 的 reset、相机、schedule 和 drop-onset physical state 严格相同。统计将 model seed 与 physical episode 作为两个交叉因子独立重采样。`F0L0 / F1L0 / F0L1 / F1L1` 的 sustained success 为 `13.33 / 26.67 / 13.33 / 26.67%`：FCPG 主效应为 `+13.33 pp [3.89,25.00]`，但 LeWM 主效应为 `0 pp [-10.00,9.44]`，完整系统相对 FCPG-only 也是 `0 pp [-13.33,13.33]`。所以当前 frozen policy 上不能声称 LeWM 进一步提高最终二值成功率。

**协议审计更新（2026-07-23）：上述旧 crossed artifact 继续保留为 `protocol-audit provisional`，v2 已独立完成修复重跑。** 旧 evaluator 只强制复用了 action prefix，没有按 grounding 强制复用完整 native-token 与 sampled-visual prefix；同时两个不调用 dynamics 的 L0 cell 仍随三个 model seed 重跑。因此旧 CI 不能追认为严格配对结果。v2 固定一份 canonical action prefix，分别固定 nominal/FCPG native-token 与 sampled-visual prefix，并只执行一次 canonical L0；其余 model seed 仅运行 L1。Fail-closed 汇总器已验证 `30` 个 physical episode、`3` 个 dynamics seed、`60` 份 prefix bundle、同 grounding 的 L0/L1 prefix 一致性、跨 seed prefix 一致性以及 L0 输出逐值恒等。

旧 artifact 的连续控制指标曾呈现三 seed 同向信号，但同样受上述配对漏洞影响，只保留为修复动机，不再引用其 CI。

v2 的 `F0L0 / F1L0 / F0L1 / F1L1` sustained success 为 `13.33 / 26.67 / 11.11 / 26.67%`。FCPG 主效应为 `+14.44 pp [4.44,26.67]`；LeWM 主效应为 `-1.11 pp [-10.00,7.78]`，交互为 `+2.22 pp [-7.78,13.33]`，完整系统相对 FCPG-only 为 `0 pp [-13.33,13.33]`。因此，当前 frozen Pick policy 上 **FCPG 的二值成功率贡献成立，但 LeWM 没有进一步提高 sustained success**。

持续控制指标则复现并加强了正交互。Post-drop success fraction 为 `0.651 / 2.456 / 0.696 / 5.391%`；LeWM 主效应为 `+1.490 pp [0.168,3.231]`，交互为 `+2.891 pp [0.352,6.254]`，完整系统相对 FCPG-only 增加 `+2.935 pp [0.355,6.353]`。最大连续成功帧为 `1.90 / 7.23 / 2.02 / 16.04`；LeWM 主效应为 `+4.47 [0.37,9.71]` 帧，交互为 `+8.69 [0.84,18.83]` 帧，完整系统相对 FCPG-only 增加 `+8.81 [0.83,19.03]` 帧。完整系统累计 reward 相对 FCPG-only 增加 `+71.61 [14.50,142.77]`，交互为 `+66.21 [5.19,165.48]`。这些区间均来自 model seed 与共享 physical episode 的 two-way crossed bootstrap。

当前准确边界是：**严格配对 v2 支持 FCPG measurement 与 matched LeWM 缺测传播在视觉中断后的持续控制质量上存在可复现正交互，但不支持 LeWM 对最终二值成功率的增量 claim。** 这是单任务、development、冻结 Pick policy 的联合机制证据；可以支撑“两个模块确实协同维持控制状态”，尚不能单独升级为跨任务 task-success 第二贡献或 final confirmation。正式汇总 SHA-256 为 `24d6a440156eff734cdcb5cd90122f836cd47a0a4efb0beaee758b7fd009c85b`。

#### 缓慢移动目标下的严格联合复验（2026-07-23，development mechanism Go）

为避免静态绿色目标被策略记成固定终点，在同一严格配对 v2 协议中让 Pick 目标沿平滑二维轨迹持续移动：XY 振幅固定为 `1.0/0.6 cm`、周期为 `440` 个 raw step、每个 physical episode 使用独立相位。目标更新同时写入环境 reward、observation 与可视 goal site；四个 factorial cell 和三个 LeWM seed 对同一 episode 复用逐点完全相同的目标轨迹。各 episode 的最大目标位移为 `1.27--2.00 cm`。一个新 identity 的 `10`-episode screen 在 moving/static 两种目标下均退化到 `1/10` 的冻结策略 floor，因此正式结果只在原已知 informative development panel 上检验机制，不宣称 fresh confirmation。

移动目标下 `F0L0 / F1L0 / F0L1 / F1L1` 的 sustained success 为 `16.67 / 26.67 / 11.11 / 26.67%`。完整系统相对 FCPG-only 的差值为 `0 pp [-13.33,+13.33]`，因此 LeWM 仍未提高最终二值成功率；完整系统相对 LeWM-only 为 `+15.56 pp [+3.33,+30.00]`，说明可见时的 FCPG measurement 仍是必要因素。

连续控制指标保留了联合收益。四格 post-drop success fraction 为 `1.829 / 4.161 / 0.860 / 6.401%`，完整系统相对 FCPG-only 增加 `+2.240 pp [+0.046,+5.155]`；最大连续成功帧为 `5.50 / 12.27 / 2.53 / 19.02`，增加 `+6.756 [+0.111,+15.544]` 帧；累计 reward 为 `160.67 / 145.94 / 145.92 / 206.91`，增加 `+60.97 [+8.34,+126.22]`。对应交互项分别为 `+3.210 pp [+0.124,+7.637]`、`+9.722 [+0.367,+23.200]` 帧和 `+75.72 [+2.61,+173.03]` reward，且三个 LeWM seed 的 Full-vs-FCPG-only 点估计全部同向。

准确结论是：**目标缓慢移动不会消除 FCPG measurement 与 matched LeWM 缺测传播对恢复后持续控制质量的协同作用，但现有冻结 Pick policy 仍未把该收益转化为更高的 sustained success。** 该结果增强“两个模块联合维持随时间变化的控制状态”的机制叙事；它不是跨任务、fresh test 或最终 task-success 第二贡献。正式汇总 SHA-256 为 `96cf9be98263b0eaa80642ca3f8c9c13c621b058d51637fe05f624f2590b92a8`。

为便于人工检查四格行为，新增同一 physical episode 的 `F0L0 / F1L0 / F0L1 / F1L1` 同步回放页面。页面固定 action prefix、相机、移动目标、视觉中断 schedule，并在同一 grounding 内固定 native-token 与 sampled-visual prefix；代表样本分别展示完整系统收益最大、FCPG-only 收益最大和完整系统仍失败的情形。该页面是由冻结合同重新执行得到的 **fresh replay**，不是原始统计运行逐帧录像；长闭环 CUDA 执行不保证逐位复现，因此页面同时列出 frozen report 与 fresh replay 指标，论文统计只使用 frozen crossed summary。入口为：

```text
outputs/fcpg_lewm_factorial_v2/pick_matched_moving_goal_dev30_seed990000/four_cell_gallery_seed15801/index.html
```

当前不追加新的 `10k` dynamics 训练。现有二值结果没有显示 LeWM 容量或训练预算是瓶颈，而连续指标已证明 matched LeWM 在缺测传播中被控制环使用；直接增加训练步数会把“状态传播是否有效”与“冻结策略能否把连续收益转成终局成功”混在一起。下一项有信息量的实验应是使用同一接口在第二个非吸收态任务上复现严格 `2 x 2`，或者训练直接消费 posterior anchor state 的策略，再判断是否需要扩大 dynamics 预算。

```text
config/lewm/eval/fcpg_lewm_factorial_pick_dev.json
config/lewm/eval/fcpg_lewm_factorial_pick_dev30.json
config/lewm/eval/fcpg_lewm_factorial_pick_matched_all3_dev30.json
config/lewm/eval/fcpg_lewm_factorial_pick_matched_all3_dev30_v2.json
config/lewm/eval/fcpg_lewm_factorial_pick_matched_moving_goal_dev30_v2.json
config/lewm/eval/p1_native_dynamics_nominal_matched_dev.json
outputs/fcpg_lewm_factorial/pick_dev_seed970000/factorial_report.json
outputs/fcpg_lewm_factorial/pick_dev30_seed980000/factorial_report.json
outputs/fcpg_lewm_factorial/pick_matched_all3_sharedprefix_dev30_seed990000/crossed_summary.json
outputs/fcpg_lewm_factorial_v2/pick_matched_dev30_seed990000/
outputs/fcpg_lewm_factorial_v2/pick_matched_moving_goal_dev30_seed990000/
```

#### Causal expert-action residual（2026-07-22，offline imitation Go / policy-control No-Go）

为直接检验 control-aligned output correction，本实验冻结 Pick policy、FCPG、P1-native observer 和三个既有 residual sources，只训练一个小型因果动作头。输入仅为当前冻结 policy 的五步动作、posterior state、policy action latent 与 missing age；输出只修正 XYZ，gripper 逐位保留。训练数据是 `30` 个 expert-forced episode 的 `180` 个无效观测决策，development 是另外 `15` 个 episode、`90` 个决策。Expert 只生成训练标签，不进入 rollout；遮挡中心使用 `q_nominal`，控制路径不运行 hidden-clean policy，视觉有效时严格旁路原策略。

三个优化 seed、每个 `2,000` steps 的离线 Gate 明确通过。Factual residual 相对未修改 factual policy 的 scripted-expert XYZ MSE 改善 `57.72%`，95% CI 的绝对改善为 `[0.01289,0.07806]`；相对 matched no-action 与 geometry residual heads 分别改善 `30.08%` 和 `25.57%`，对应 CI `[0.00430,0.02588]` 与 `[0.00356,0.02038]`。这说明 observer/policy latent 足以在 expert-forced 分布上拟合更接近 expert 的短期动作。

随后只执行一次 Pick-only、`30` 个 fresh physical episode 的 large-effect control screen。Primary 在实验前固定为 drop 后连续至少五个 raw steps 成功的 sustained success，并对三个 head seed 先在每个 physical episode 内平均。Factual residual 相对 factual base 的差为 `-1.11 pp [-17.78,+15.56]`，相对 matched no-action residual 为 `0 pp [0,0]`，相对 matched geometry residual 为 `+3.33 pp [0,+7.78]`；三项均未达到 `+10 pp`，没有任一 head seed 同时对三类基线保持正向。累计 reward 的三项 CI 也均跨零。事后可见的 final-success 正向值不能替换冻结的 sustained-success primary。

因此准确边界是：**expert-forced action imitation 能显著降低离线动作误差，但没有形成 LeWM-specific 的闭环成功收益。** 主要风险是 expert-forced covariate shift 与冻结 policy 对动作残差的闭环敏感性；当前数据不能靠继续训练或扩大 head 解决。不追加 episode、不扫参、不启动 CEM，也不将该接口列为论文第二贡献。

```text
outputs/lewm_expert_action_residual_pick/matched_heads_2k/report.json
outputs/lewm_expert_action_residual_pick/fresh_rollout_30eps/report.json
```

#### On-policy posterior DAgger residual（2026-07-22，covariate-shift repair No-Go）

旧 residual screen 的训练轨迹始终由 scripted expert 执行，因此离线动作误差改善可能无法覆盖 learner 自己造成的闭环状态分布。本实验改为 decision-level on-policy DAgger：在每个 learner decision 保存完整 MuJoCo 状态、进程 RNG 和本地 observation，从该状态分支执行 scripted expert 的五个 raw action 以生成连贯标签；随后精确恢复状态，只执行 learner 的五步 action chunk，并仅将 learner 实际执行的动作写回 observer history。Round 0 禁止 behavior head，round 1 必须加载同方法、同训练 seed、固定 `2,000` steps 且 `dagger_iteration=0` 的上一轮 head。三种方法共享初始 physical episode panel，但 learner 分叉后的行数和 decision identity 允许不同；训练 minibatch 对两轮等权，方法间不满足逐 decision 配对时明确使用 unpaired episode bootstrap。

固定配置只使用 Pick policy seed `0`、LeWM seed `15801`、`5 deg world-z` 相机旋转和以部署式 `q_nominal` 为中心的六决策局部遮挡。每轮每种方法采集 `12` 个 train 与 `6` 个 development episode，最终 head 只有一个优化 seed `64301`、约 `121k` 参数，并只修改 XYZ；gripper 原样旁路，视觉恢复后的 decision 精确旁路到 native policy。`q_true` 不用于 inference 或 corruption center。该实验是 known-validity、single-policy/single-model-seed development，不是 publication confirmation。

离线 development 中，factual residual 将自身 base XYZ action MSE 从 `0.53737` 降到 `0.46386`，相对改善 `13.68%`，但 95% CI 为 `[-23.15%,+28.42%]`。Factual 相对 no-action residual 的点改善为 `11.21%`，相对 geometry residual 为 `-1.84%`；两项因 learner 轨迹分叉采用 unpaired bootstrap，CI 均跨零。因此 on-policy collection 消除了旧 expert-forced 合同缺陷，但没有得到稳定的离线方法优势。

最终 fresh rollout 使用 `12` 个全新 physical episode，四种方法只共享 clean prefix，遮挡后各自闭环执行。Sustained success 为 factual base `58.33%`、factual DAgger `50.00%`、no-action DAgger `25.00%`、geometry DAgger `41.67%`。Factual DAgger 相对 factual base 为 `-8.33 pp [-33.33,+16.67]`，相对 no-action 为 `+25.00 pp [0,+50.00]`，相对 geometry 为 `+8.33 pp [-16.67,+33.33]`。Factual 的 post-drop success fraction 与累计 reward 点估计高于三种 learned 对照，但相对 factual base 的区间均跨零；所有方法的 recovery exact-native bypass 均为 `100%`。

因此准确结论是：**动作条件 factual observer 相对 matched no-action 仍有开发集控制信号，但额外 on-policy residual head 没有超过未修改的 factual base。** 旧实验的 covariate shift 不是唯一瓶颈，继续增加 DAgger round、训练 seed、head 容量或 episode 不能由本结果授权；该接口不形成论文第二贡献，也不启动 selector/CEM。正式结果：

```text
outputs/lewm_posterior_dagger_pick/on_policy_v2/heads/final/report.json
outputs/lewm_posterior_dagger_pick/on_policy_v2/rollout/fresh/report.json
```

#### Geometry-Anchored Control-Sufficient Dynamics（2026-07-19，P0-No-Go）

这条候选不推翻 P1-native 的旧判定，也不是把现有 LeWM 换成更大的 Transformer。当前证据显示：P1 dynamics 在 H4-H8 和 wrong-action 对照上有稳定动作信号，互补观测器也显著降低 token/action-distance；但 paired rollout 的二值成功率没有显著提升。因此下一步先检验一个更窄的假设：**平均 token MSE 可能没有惩罚冻结策略真正敏感的误差方向。** Full-memory、global token、block-causal attention 或继续扩到 `10k` 都不是本候选的默认修复。

本候选明确借鉴 *Feedback World Model Enables Precise Guidance of Diffusion Policy*（arXiv:2605.15705）的 Eq. 8--15，而不是把这些已有公式重新包装成创新。Eq. 8--12 将冻结 transition 写成速度形式，并用当前真实观测形成 feedback residual：

```text
v_t(a)       = (f_theta(z_t, a) - z_t) / dt
e_t          = z_t - zbar_t
vhat_t(a)    = v_t(a) + L e_t
zbar_{t+1}   = zbar_t + dt * vhat_t(a_t)
e_{t+1}      = z_{t+1} - zbar_{t+1}
```

Eq. 14--15 再用同状态、不同候选动作下的反事实 latent 方差估计逐维 controllability，并进行均值归一化与 `beta` 插值：

```text
w_j = mean_i Var_{a in A_demo} [f_theta(z_i, a)]_j
wbar_j = w_j / (mean_k(w_k) + eps)
w_beta,j = 1 + beta * (wbar_j - 1)
E_ctrl = sum_j w_beta,j * (z_pred[j] - z_target[j])^2
```

速度形式、feedback state、线性 gain、反事实方差和 controllability-weighted energy 均属于该工作的贡献，**不得作为 FCPG novelty**。原论文每一步都取得下一观测，并在该 always-observed 设定下给出 Eq. 13 的有界性结论；本项目在已知视觉缺测期保持缺测前最后一个因果 residual，因此不继承该定理。P0 将 `dt=1`、`L=I`（scalar gain `1`）和 residual decay `1` 全局固定，即整个缺测段只传播同一个 pre-drop residual，不按 episode、horizon、coordinate 或 hidden-clean target 调参；视觉恢复时才由新 measurement 重新锚定。

P0 的 controllability 也不是原论文的 `A_demo` action pool。它只在与评分 panel 物理 episode 和 observed-state signature 均不重叠的独立 fit 源上，对每个 fit episode 固定匿名 `branch-0` 的因果历史，再将八条 future-action bank 从该同一历史做冻结 LeWM 反事实 rollout，拟合 **一个** Eq. 14-style 全局坐标权重向量；该过程不读取 fit target。随后按 Eq. 15 固定 `beta=1`，对所有 scored episode、horizon、branch 和三个 model seed 复用。因此它是借鉴公式后的项目特定迁移检验，不能引用原论文结果替本项目背书。

本项目可能保留的窄边界只有：FCPG 从名义 FK 投影出发，用当前局部视觉预测二维残差，并在校正位置重新读取具有固定机器人部位身份的策略特征；随后只在视觉间歇失效时，用 geometry carrier 和动作条件模型维持这些 **policy-native grounded anchors**。有合法视觉 measurement 时，`z_t` 必须来自 FCPG 重采样；视觉缺失时不得伪造当前 innovation，observer 只能使用缺测前的因果 feedback state、当前 proprio/FK、已执行动作和 last-valid measurement。相机 jump、遮挡发生时刻和背景变化是外生扰动，LeWM 不得声称提前预测它们。

**P0：zero-training control-sufficiency Gate。** 这是使用已知缺测 mask 路由的 offline/exposed 诊断，不是可部署的遮挡检测器或 policy rollout。它只复用冻结 P1 cache/checkpoint、独立 controllability fit 源和 same-state hidden-clean 离线标签，不更新 FCPG、policy、encoder 或 dynamics。策略评估始终使用同一个 raw-adjacent two-slot carrier `[N,2,K,D]`：冻结其 RGB/视觉前缀、proprio/FK、task 与所有非 anchor 输入，只把同一 model-step endpoint 按既有 contract 重复到两个 anchor slots；exact action error 为该 carrier 上 predicted-anchor action 与 clean-anchor teacher action 的平方误差。clean anchors 与图像在同一物理 replay 中记录，但 label arrays 在全部部署预测冻结前不会被 stack、传入 predictor、读取或评分。对每个 physical episode、horizon 和 action branch 同时计算：

1. uniform token MSE；
2. 用独立 fit 源冻结的单一 Eq. 14-style local-branch weight vector 得到的 weighted error（Eq. 15 `beta=1`）；
3. 冻结策略的 exact action error，以及仅作诊断的局部 `J_pi^T J_pi` quadratic error；
4. raw factual LeWM、matched no-action、七种 causal wrong-action、exact-FK geometry、现有 fixed-gain complementary observer，以及全局固定的 unit-gain/held-residual analytic observer。

P0 只有在下面条件同时满足时才为 Go：weighted metric 相对 uniform MSE 对 exact action error 的 same-episode/same-horizon 八分支 Spearman（先在分支内计算、再宏平均）提升至少 `0.10` 且 physical-episode cluster bootstrap 差值 95% CI 下界大于零；固定 analytic observer 相对现有 complementary observer 的 exact action error 改善至少 `10%`、CI 下界大于零；H3 不得退化超过 `5%`，且至少 `5/6` 个 H3-H8 horizon 同向；factual exact action error 的点估计必须分别低于 matched no-action 和七种 wrong-action controls。后两类 control 的 episode-bootstrap CI 只作诊断，不新增事后门槛。上述数值是本项目预先冻结的**工程决策门槛**，不是 Feedback-WM 的定理、通用统计标准或论文推荐阈值。`hidden-clean token/action` 与 `q_true` 只能用于离线评分/标签，不能成为 observer 或 predictor 输入。P0 失败即停止该候选，不训练新 head，也不得改阈值救援；只有 P0 合取 Gate 为 Go，才允许下面的 matched 2k。

**正式结果。** 旧 carrier 在动作分支开始后才形成 H1/H2 历史，首次 evaluator 因“same-snapshot branches 的 causal pre-drop state 不同”正确地 fail closed；这不是模型结果。修复后的 v2 使用共享 expert prefix `H-1/H0/H1/H2`，只在 H2 后分出八条动作并只评分/执行 H3-H8，得到 `30 x 8 x 6 = 1,440` 行。三个冻结 predictor seed 先做宏平均，bootstrap 单位严格为 `30` 个 physical episodes。

结果只保留一条强机制证据：factual exact action error 为 `0.05157`，matched no-action 为 `0.11178`，相对改善 `53.87%`；七种 cyclic wrong-action 的相对改善为 `57.28%--59.09%`，全部方向通过。这说明正确动作条件的 fixed-identity anchor dynamics 对冻结 policy 确实有用。可是两个 P0 核心假设均失败：uniform token error 对 exact action error 的 panel-macro Spearman 为 `0.24696`，counterfactual-weighted metric 为 `0.23624`，增量 `-0.01071`、95% CI `[-0.03056,0.00860]`；unit-gain/held-residual analytic observer 相对现有 complementary observer 的 action error 改善为 `-30.63%`、95% CI `[-45.07%,-17.99%]`，H3-H8 为 `0/6` 正向。H3 仅退化 `0.91%`，后续 horizon 退化扩大到 `27.81%--45.59%`，表明固定 residual 在完全缺测期持续注入会累积 common-state bias。

合取判定为 **P0-No-Go**。因此不启动 controllability-weighted 或 policy-distilled matched `2k`，不把 Feedback-WM observer/weighting 写成项目创新，也不授权新的 rollout、`10k`、selector 或 CEM。正式 artifacts：`outputs/policy_token_sensitivity_carrier/exposed_p0_v2_common_prefix_scored_h3_h8/`、`outputs/control_sufficient_p0/exposed_p0_v2_scored_h3_h8_row_panel.json` 和 `outputs/control_sufficient_p0/exposed_p0_v2_scored_h3_h8_summary.json`。

**P0 Go 后唯一授权的 matched 2k。** 三组必须使用相同 P1-native architecture、train/development physical split、初始化、优化器、batch、三个 model seed、`2,000` optimizer steps 和 fixed-latest endpoint；只允许 loss 不同：

```text
M0: L_token
M1: L_token + lambda_ctrl * L_counterfactual_controllability
M2: L_token + lambda_ctrl * L_counterfactual_controllability
              + lambda_pi * ||pi(M_pred) - stopgrad(pi(M_clean))||^2
```

冻结 policy 只提供 loss 和评估梯度，不能被更新；`M_clean` 只作训练 target，部署 forward 仍只能读取因果历史、FK/proprio、FCPG measurement 与动作。`lambda` 只可按 train split 的 loss scale 预先规范化并在 fresh development 前冻结。M1/M2 必须在 action error 上相对 M0 和现有 complementary observer 均改善至少 `10%`、CI 下界大于零，同时 uniform token MSE 不恶化超过 `5%`、H3 满足非劣、至少 `2/3` seed 通过，才授权一次新的 known-drop policy rollout；否则停止，不扩到 `10k`。

禁用 claim：不得写“我们提出 feedback world model / latent observer / controllability weighting”；不得声称 Bayesian posterior、校准 uncertainty、自动发现 unknown corruption、预测相机跳变、在线重标定、通用 world-model 改进、CEM/planning、task-success 增益或完整 closed-loop contribution。只有 matched 2k、fresh rollout 和最终 task-success 依次通过后，才可讨论“FCPG geometry-anchored measurement 与 control-sufficient anchor dynamics 的组合”为第二贡献。

随后完成一次不使用 A4 reserved test 的 paired-camera development probe。Pick/Reach/Push 各收集 `2` 个 physical episode、每 episode `80` 帧；同一 pre-action state 同时重渲染动态与 nominal-clean RGB，物理状态、机器人观测和待执行动作保持一致。相机端覆盖每 episode 独立的 `1--3 cm / 2--8 deg` 随机静态位姿、`20%--80%` 线性漂移，以及发生于 episode `30%--70%` 的随机阶跃。冻结 shared-4task FCPG 与 `latest@2000` Anchor-Persistent checkpoint，不训练或选择参数。

| Paired-camera condition | nominal EPE | FCPG EPE | clean FCPG EPE | nominal latent gap | FCPG latent gap | q-true oracle latent gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static random | `18.013` | `3.085` | `0.105` | `15.707` | `16.461` | `16.560` |
| linear drift, all frames | `6.538` | `0.598` | `0.105` | `10.682` | `10.797` | `10.819` |
| random step, post only | `21.782` | `3.445` | `0.104` | `22.033` | `21.035` | `20.650` |

该 probe 给出两个分离结论。第一，FCPG 在三种相机轨迹上消除了约 `83%--91%` 的像素投影误差，且同状态 clean 重渲染只引入约 `0.105 px`，因此随机/运动视角并未否定 Stage I 反馈校正。第二，校正后的 Anchor-Persistent state 没有随像素 EPE 同步规范化；更关键的是，直接在动态图像的 `q_true` 处采样也没有进一步缩小 gap，FCPG-to-oracle latent ratio 为 `0.994 / 0.998 / 1.019`。这把瓶颈定位为 frame encoder 的跨视角外观敏感性，而不是 FCPG 搜索窗口或 residual 精度。

因此不能把随机相机轨迹直接混入 LeWM 后声称模型学到纯机器人动力学，也不应先扩 FCPG 或 A4 训练预算。若重开该方向，正确的有界下一步是 fresh paired-view 数据上的 anchor-preserving canonicalization：冻结或缓慢更新视觉骨干，用同物理状态 dynamic/clean anchor token 的一致性目标和 SIGReg 防塌缩，并保留显式 camera transform/estimated image residual 作为条件；必须与 raw augmentation、camera-conditioned MLP 和 target-only oracle 比较。真实不可预测 camera jump 仍由当前 RGB 的 FCPG 重锚定，LeWM 只传播机器人/任务状态。当前 `6` episode/condition 只授权这一接口诊断，不构成跨视角论文证据。评估入口为 `scripts/evaluate_paired_camera_grounding.py`，结果位于 `outputs/lewm_action_branch_a4/camera_probe/*_paired_state.json`。

#### Fresh paired-view canonicalization Gate（2026-07-17，Development No-Go）

按上述诊断另采 physical-episode 隔离的数据：静态随机视角训练集为 Pick/Reach/Push 各 `8` episode、共 `1,920` 帧；unseen linear-drift 与 random-step development 各为每任务 `4` episode、共 `960` 帧。每帧仍是同一 pre-action physical state 的 dynamic/clean 成对重渲染。冻结 FCPG 与 Anchor-Persistent encoder，只训练 `70,080` 参数、零初始化输出的 residual Adapter，固定 `2,000` steps。三种等预算输入为 `latent_only`、部署可得的 `FCPG residual + q_nominal + FK XYZ + anchor ID`，以及仅作 privileged 上限的真实相机相对变换。

| Adapter input | static-train improvement | unseen linear improvement | random-step post improvement | linear clean gap | step clean gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent only | `82.61%` | `-2.77%` | `16.01%` | `10.15` | `9.25` |
| residual/FK context | `86.01%` | `0.48%` | `12.97%` | `9.50` | `8.58` |
| camera-transform oracle | `87.47%` | `6.34%` | `16.46%` | `8.36` | `7.70` |

该结果不是表示坍塌：development std ratio 仍为 `0.923--0.978`，有效秩也接近 target；失败来自映射不泛化。部署式 context 没有稳定胜 latent-only，privileged camera transform 也没有形成充分上限；三组都把 clean 输入从约 `0.08` 放大到 `7.70--10.15`，并使 linear/step temporal-delta error 变差。因此当前小型 post-hoc Adapter 判为 **Development No-Go**，不得接入 A4、不得启动 LeWM 长训或控制实验。后续若重开跨视角状态，只能改视觉表示的训练目标（例如在 encoder/LeWM 训练期加入 paired-view invariance），不能继续扩大本 Adapter 或在当前 split 上调 loss/阈值。实现入口为 `scripts/train_paired_view_canonicalizer.py`，完整结果位于 `outputs/lewm_action_branch_a4/paired_view_canonicalizer_v1/`。

#### Frozen LeWM 的 FCPG same-view intervention（2026-07-22，Development mechanism Go）

跨视角 canonicalization 与 FCPG 对 LeWM 的作用是两个不同问题。前者要求同一物理状态在不同相机外观下得到相同 latent；后者只要求在**同一当前动态图像**中，把错误的 nominal sampling location 改为 FCPG location 后，冻结 LeWM 更接近同视图 `q_true` grounding，并对真实未来帧预测得更准。为避免把两者混淆，本实验冻结原始七任务 `latest@10000`，保持 RGB、proprio、FK、task 和 action prefix 完全相同，只切换当前 grounding 为 `nominal / FCPG / q_true oracle`。`q_true` 不进入 nominal/FCPG forward，仅用于编码同一动态视图的离线 target。

随机 camera step 发生并稳定至少一帧后，Pick/Push/Reach 的 `119` 个 H4 causal windows 给出：

| Actual-future H4 指标 | nominal | FCPG | 相对改善 | 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| full latent MSE | `0.034862` | `0.027009` | `22.53%` | `[0.003254,0.012590]` absolute |
| anchor latent MSE | `0.034736` | `0.026649` | `23.28%` | `[0.003459,0.012829]` absolute |
| H1 metric EPE | `20.595 px` | `4.996 px` | `75.74%` | `[13.049,18.034] px` |
| H4 metric EPE | `23.150 px` | `9.968 px` | `56.94%` | `[10.078,16.096] px` |

在相机仍持续线性移动的 `720` 个 H4 windows 中，FCPG 对 full/anchor latent MSE 的改善缩小到 `4.46% / 4.73%`，但 CI 仍严格为正且 `3/3` 任务同向；H1/H4 metric EPE 分别改善 `39.43% / 16.84%`。这符合职责边界：FCPG 能校正当前视觉测量，却不能从机器人动作预测未来的外生相机运动，因此持续漂移应使用短时域预测并在每个新观测上重新 FCPG 接地。

随后将相同干预扩展到七个 A4 任务。数据使用训练未见的奇数钟点视角 `clock03`；每个任务的一条 task-instance 轨迹在 `4` 个独立 camera perturbation trial 下复用，每条 `80` 帧。相机在中点发生 `5 deg` step，冻结同一个七任务 `latest@10000` LeWM，并保持 future RGB、动作和物理轨迹完全一致。FCPG 将 step 后当前投影 EPE 从 `17.775` 降至 `2.378 px`。在 `532` 个 H4 causal windows 上，结果为：

| 七任务 actual-future 指标 | nominal | FCPG + 4 px dead-zone | 相对改善 | 95% camera-trial paired CI | 任务方向 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full latent MSE | `0.025122` | `0.023437` | `6.71%` | `[0.000839,0.002490]` absolute | `6/7` |
| anchor latent MSE | `0.024908` | `0.023031` | `7.54%` | `[0.001000,0.002726]` absolute | `6/7` |
| H1 metric EPE | `19.149 px` | `6.852 px` | `64.22%` | `[10.772,13.801] px` | `7/7` |
| H4 metric EPE | `26.725 px` | `16.695 px` | `37.53%` | `[7.790,12.191] px` | `7/7` |

未加安全规则的 FCPG 在 paired-clean 图像上引入平均 `1.596 px` 当前校正量，并使冻结 LeWM 的 H1/H4 metric EPE 相对 nominal 增加 `0.717/0.656 px`。该 development split 上，小残差与相机 step 后残差近乎分离：以三个 anchor 校正模长的帧均值为统计量，固定 `4 px` dead-zone 只在 `2.14%` clean 帧激活，同时基本保留 step 后全部校正。加入该 deterministic rule 后，clean H1/H4 额外误差缩小到 `0.092/0.105 px`，而 drift H1/H4 的改善仍为 `64.22%/37.53%` 且 `7/7` 任务同向。由于四个 camera trial 共享每任务同一物理轨迹，这里的区间只覆盖相机扰动变化，不能冒充 `28` 条独立物理 episode。该规则只处理“无明显漂移时不做小幅伪校正”，不是遮挡或模糊下的视觉可靠性 selector；`4 px` 由当前 development 分布事后确定，必须在 fresh 多 task-instance split 上冻结确认后才能形成正式 safety claim。

一次 matched paired-invariance `2k` 试验没有被提升为结果。其 full-anchor-token 对齐目标同时惩罚了动态视图下本应变化的 `q_corrected` 几何编码，因而天然偏向坐标不变但错误的 nominal 输入；两个 endpoint 均未优于原始 `10k`。当前证据支持的最小结论是：**无需重训 LeWM，FCPG 当前测量已经显著改善冻结 LeWM 的 same-view current-state 与 actual-future prediction。** 它仍是 development 机制证据，不等同于动作选择、policy rollout 或任务成功率提升。

```text
scripts/evaluate_lewm_fcpg_intervention_a4.py
scripts/evaluate_lewm_fcpg_future_a4.py
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/fcpg_same_view_intervention_dev.json
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/fcpg_actual_future_step_stable_h4_dev.json
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/fcpg_actual_future_linear_h4_dev.json
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/fcpg_actual_future_7task_orbit03_step_dev4_h4_deadzone4_seed40652.json
outputs/lewm_action_prefix_a4/shared_7task_orbit12_gripper_seed40531_10k/fcpg_actual_future_7task_orbit03_clean_dev4_h4_deadzone4_seed40652.json
```

A4 Development-Go 当时只授权一次 fresh confirmation。Seeds `30301/30302/30303` 已从同一 frozen base 分别完成固定 `10,000` steps；不从 2k checkpoint resume、不挑 best、SIGReg 仍只允许 `0.02`。三个 fixed-latest endpoint 均为 `step=10000` 并通过严格重建，checkpoint SHA-256 依次为 `69b0b9388db7b7dc4c07c203856d634c020858f060bf7962d2dfa900a74a79e0`、`9de33a4dba0d16cb01179db8c9e6a85c9c31f4771d3ae5054af2809b1f415730` 和 `c9ee1bee9a8ef4408b42c191dfba678a2c9b1f69963b0b180ad01b56ff686fd2`。

随后唯一一次 fixed-panel reserved-test collection 按七任务各 `12` 个冻结 instance 执行。至少一个任务未能让全部 `12` 个 instance 产生合格 active-control bundle，因此 collector 在发布 `reserved_test_state_cache.npz` 前 fail closed；test cache、三 seed prediction 和 confirmation summary 均未生成。该结果准确标记为 **Confirmation collection No-Go / unscorable**：它否定当前 A4 固定 test panel 的确认资格，不否定三个 `10k` checkpoint 或七任务 Development-Go。不得在已访问的 test panel 上换 instance、补采、降低 episode 数或重新运行后声称确认通过。

该历史 fixed-panel 失败没有被翻案或重跑。后续 A4b 使用全新的 task-instance namespace、benchmark seed 与 physical episodes 建立独立 panel，并按原冻结模型和 Gate 得到 **Confirmation-Go**；其正式数值与 claim 边界见前述 “A4b fresh-panel fixed-10k confirmation”。两份 artifact 必须同时保留：前者证明旧 panel 不可评分，后者提供合法 fresh confirmation。

```text
outputs/lewm_action_branch_a3/a3d_screen_gate_fail_closed/summary.{json,md}
outputs/lewm_action_branch_a3/a3c_confirmation_gate/summary.{json,md}
outputs/lewm_action_branch_a3/train_seed2930{1,2,3}_10k/
outputs/lewm_action_branch_a4/formal_v2/preflight_gate/summary.{json,md}
outputs/lewm_action_branch_a4/development_gate/summary.{json,md}
outputs/lewm_action_branch_a4b/confirmation_gate/summary.{json,md}
```

### Counterfactual v2：外部 development G0 collector（schema-v2 smoke 已验证）

Counterfactual v2 是一条新的、仅用于判定控制信息是否可读的 external-development 数据协议，不重开已经 No-Go 的 full grounded LeWM Gate L1，也不把旧 post-hoc L2 pilot 重新包装成正式结果。旧 pilot 不能作为动作排序证据：它用测试 episode 的未来成功帧和 `q_true` 构造 hindsight goal，以 latent-to-goal distance 代替真实任务回报，并且只有很少的物理 branch point；这些量可以作为离线标签或诊断，但不得进入部署式 scorer 输入。

v2 将数据严格拆为两组：

```text
deployment inputs:
current/history RGB, proprio/FK, nominal/corrected anchors,
candidate actions, current-scene ViT-prefix context, predicted future latent

offline labels only:
true return/progress/success, oracle-anchor actual-future visual latent,
q_true, goal world/pixel coordinates
```

phase 不再根据未来成功或 reward 选择，而是预先固定在 step `10/25/35`（仅命名为 early/mid/late）。每个 candidate action chunk 从同一物理 snapshot **只执行一次**；`clean / rotation5 / translation4` 仅从保存的 history 与 endpoint state 重渲染，因此真实 return/progress/success 没有 camera axis。collector 同时保存 `LeWM train grounding {nominal, FCPG} x eval grounding {nominal, FCPG}` 的 `2 x 2` 特征轴，但不在本阶段重训 FCPG 或 `10k` LeWM。

Gate 顺序固定为：G1 检查 oracle-anchor actual-future visual latent 能否读取真实 return；G2 检查 predicted future latent 是否超过参数量匹配的 `history + raw action` DirectRanker 与错配 future 对照；G3 在相同 LeWM/scorer 下比较 FCPG 与 nominal eval grounding；G4 只允许一次预先限定的 horizon-structured CostAdapter；G5 才允许 CEM。G1-G4 的 external-development 已按该顺序执行并冻结，结果未满足进入 G5 的条件。

2026-07-15 的四任务 schema-v2 CUDA smoke 已完成并从原子存储中独立重读：

- 输出：`outputs/lewm_counterfactual_v2/development_seed8192_smoke`；
- store result SHA-256：`17dd5ce9391a55f1c49719c6218a6bdcacdee4114508ab92386c2f2a91500b22`；
- `4` 个 physical episode、固定 `12` 个 snapshot sample、`48` 次且仅 `48` 次物理 candidate branch；
- `108` 个 history frame 的最小动态范围为 `255`，扰动视角相对 clean 的最小变化像素数为 `28696`；
- `720` 个 endpoint frame 的最小动态范围为 `255`，扰动视角相对 clean 的最小变化像素数为 `28621`，并逐帧保存 RGB SHA-256；
- collector 在三视角重渲染时逐步核对并保存 history/endpoint physical-state 与 RNG hashes；每个匿名 candidate slot 的 action、真实标签、oracle future latent、endpoint state/RNG 与 endpoint RGB 被统一绑定到 branch receipt，独立重读时从数组重新计算并通过；
- 几何重投影、三种 camera condition 的外参重建、corrected-anchor bounds、checkpoint 文件与实际加载实例绑定均通过；store 共 `33` 个数组，严格分为 `14` 个 deployment、`9` 个 offline label 和 `10` 个 audit 数组；
- `deployment_inputs` 分区不含 `q_true`、oracle future latent、真实 return/progress/success 或 goal world/pixel；这些监督量只存在于隔离的 `offline_labels` 分区；模型 state hash 在采集前后完全一致，未训练上游模型；
- scorer 输入必须由 role-specific builder 生成单一 model/grounding/view 的只读最小视图；G2 builder 固定完全相同的 cell 比较 DirectRanker 与 predicted future，G3 builder 固定 model/view/context 只切换 grounding，oracle view 明确为 offline-only；
- 四个候选动作在全部 sample 中均唯一，最小 pairwise action L2 为 `0.07969`；return/progress/success 分别在 `11/9/5` 个 snapshot 上存在候选排序信号。标签并列只作为 evaluability 诊断，不伪装成数据合同失败；
- artifact 绑定 clean implementation commit `e1c534f6480d5ad4fe23cd38861022e08036da58`（`git_dirty=false`）；独立磁盘进程重新加载并重建 manifest，与发布内嵌 G0 manifest 完全一致；完整回归为 `886 passed, 3 skipped, 33 warnings`，`py_compile` 与 `git diff --check` 通过。

这里必须区分两个结论：`g0_audit.passed=true`，表示 smoke 的 schema、泄漏隔离、物理配对与 provenance 合同可执行且通过；但 `g0_eligible=false`。原因固定为 `bounded_smoke_is_not_g0_eligible` 与 `upstream_episode_non_overlap_not_auditable`：现有上游 checkpoint 没有可核验的 physical-episode seed ledger。因此该 smoke 不能作为正式排序结果或论文 G0 证据，也没有启动 G1-G5。

### Counterfactual v2：full external-development 与 scorer 结果

四任务 full collector 已完成：`40` 个 physical episode、`120` 个固定 snapshot、每 snapshot `8` 个候选，共执行且仅执行 `960` 个物理 branch。按 physical episode 做 `7/3` train/validation 拆分，对应 `84/36` 个 snapshot；nominal/FCPG pooled rows 不增加独立物理样本。完整 store 位于 `outputs/lewm_counterfactual_v2/development_seed8192/`，包含 `33` 个分区数组，逐文件与语义审计后的 result SHA-256 为：

```text
fc1b45ab0d9c0cfe8c3a7b619e9c8a4735a7893fd8887240bf8371a4c9a93390
```

full store 的内部 G0 合同全部通过，但仍为 `external_development`、`formal_claim_allowed=false`。后验 goal-signature forensic audit 证明新旧四任务目标签名精确交集均为零；最小距离为 pick `4.91 mm`、button `0.715 mm`、reach `8.03 mm`、push `1.31 mm`。该证明可严格覆盖两个 LeWM checkpoint 已哈希绑定的四个旧 NPZ，但历史 FCPG corrector 与 action policy checkpoint 只记录了数据路径/数量，没有绑定当时 NPZ 内容 SHA，因此完整上游 bundle 的密码学链仍不闭合。有限范围报告为：

```text
outputs/lewm_counterfactual_v2/forensics/development_seed8192_goal_disjointness_forensic_v1.json
file_sha256   = fabb737bc4c51bb68f77090286461c345d74806d0392927c2965c455d818cea8
result_sha256 = 6303851ebd59fef3cf9b0a1c06a4b7061edd77491fc23d4d867b944ad747b347
proof_scope   = data_and_gate_l1_lewm_only
full_bundle_formal_verified = false
```

第一轮 flat tiny scorer 在 clean 上等量池化 nominal/FCPG grounding，分别训练 `action_only / Direct / Predicted / Oracle`，每类使用 seeds `8201/8202/8203`、固定 `2,000` full-batch steps 和 RankNet + `0.1` Huber。参数量为 `6,921-6,931`，共 `12` 个 checkpoint。封存输出与 clean implementation commit `0033988` 绑定：

```text
outputs/lewm_counterfactual_v2/scorer_e1_e3_seed8201/
result_sha256 = 6590a12ec2650f281934b14634ca6f105ee5c0ca56e7082117494737bf217c59
```

primary 是排除 `random_0/random_1` 后的六候选 broad bank，包含 `expert`、三档 expert noise、`policy` 和 `same_task_shuffled`；旧 artifact 字段曾误写为 `near_expert_without_random_0_random_1`，不得据此称为 strict near-expert。主指标为 H=5 cumulative actual return；先平均 phase/view/scorer seed，再以 physical episode 为单位做 `10,000` 次 task-macro paired bootstrap。validation 的真实独立单位只有每任务 `3`、总计 `12` 个 physical episode；三个 scorer seed 和 bootstrap 重采样均不增加独立证据。正值表示右侧方法更好：

| Flat tiny 配对 | Delta Spearman（95% CI） | regret improvement（95% CI） | 判定 |
| --- | ---: | ---: | --- |
| E1 oracle-anchor future visual latent vs Direct，clean/FCPG | `-0.1499 [-0.2398, -0.0548]` | `-0.0374 [-0.1161, 0.0418]` | 当前表征/接口下 Fail；不等价于 fully observed control-state oracle |
| E2 Predicted vs Direct，drift/FCPG | `+0.0927 [0.0144, 0.1693]` | `+0.0942 [0.0615, 0.1285]` | 仅 broad-bank development 为正 |
| E3 FCPG vs nominal，同一 Predicted scorer，drift | `+0.0496 [-0.0058, 0.0977]` | `+0.0485 [0.0030, 0.0940]` | Mixed |

事后固定为 `expert + expert_noise_{0.05,0.15,0.3}` 的 strict-near 审计中，E2/E3 两个指标的 CI 均跨零；因此 flat E2 不能升级为细粒度动作排序贡献。Predicted 相对 candidate-axis cyclic future 错配为正，只证明 scorer 确实使用了 action-aligned future input，不证明该输入比 DirectRanker 更接近真实 outcome。

为排除 flat head 将 `5 x 192` future 一次压到 hidden `4` 的功能瓶颈，只允许了一次预先限定的结构救援。`HorizonStructuredCostAdapter` 使用共享 `192 -> 16` state projection、action-prefix GRU、逐 horizon shared cost 并对五步求和；Direct/Predicted/Oracle 完全同构且均为 `7,181` 参数。Direct 仅把 five-step future state 替换为重复的当前 `z_last`。三路共用同一个只由 pooled-clean training `z_history` 拟合的 latent scaler，future state 不参与拟合；五类共享输入在缩放前后均逐字节一致。封存输出绑定 clean commit `47d2501`：

```text
outputs/lewm_counterfactual_v2/structured_cost_e1_e3_seed8201/
result_sha256 = 9a8106d3b60a798b8aa43cfe5978e0ac1ff45cd96f8e382d738131b4b075c1d0
shared_scaler_sha256 = 046cfa6f22cf4e300c53afc17b31afbdbe57b1d4e6cf7c8bc41d7aa3aba6fb23
```

| Structured 配对 | 六候选 Delta Spearman / regret improvement | strict-near Delta Spearman / regret improvement | 判定 |
| --- | ---: | ---: | --- |
| E1 oracle-anchor future visual latent vs Direct，clean/FCPG | `-0.1056 / -0.0355` | `-0.1562 / -0.0980` | 当前表征/接口下 Fail；四个 CI 均含零或偏负 |
| E2 Predicted vs Direct，drift/FCPG | `-0.1017 / -0.0385` | `-0.0925 / -0.0824` | **Fail；四个 95% CI 全部严格低于零** |
| E3 FCPG vs nominal，同一 Predicted scorer，drift | `+0.0170 / +0.0272` | `+0.0388 / +0.0194` | Mixed；每个候选集仅一项 CI 严格为正 |

结构化 Predicted 相对 cyclic future 错配仍为正（六候选 Spearman `+0.1841 [0.1159, 0.2548]`），说明训练出的 scorer 使用了 candidate-specific future；但在当前 scene proxy、candidate bank、`7/3` split 和 structured scorer 下，它相对 DirectRanker 的验证表现显著更差。

这里存在两个必须公开的协议限制。第一，`goal_visual` 实际由**当前 RGB 的 frozen ViT 全局 prefix 经固定随机正交投影**得到，不是显式 goal token；各任务 goal 在 episode 间变化。一个非预注册的 read-only ridge sanity probe 将 task-macro goal-pixel RMSE 从 task-only 的 `9.09 px` 降至 `7.46 px`，说明该 prefix 弱含目标信息，但没有证明其精度足以支持 return ranking。第二，当前候选由 `expert / expert+三档噪声 / policy / temporal shuffle / random` 构成，存在明显的 candidate-family shortcut：严格只用 training physical episodes 拟合的 `task + candidate-kind` 平均回报查表，在 validation 上已经得到 broad Spearman `0.6208 [0.5158, 0.7291]`、strict Spearman `0.6882 [0.5548, 0.8326]`；对应 structured DirectRanker 点估计为 `0.5792/0.5377`。该诊断不证明 DirectRanker 只靠类别作弊，因为 scorer 不接收候选名称且配对差异 CI 跨零；它证明的是当前协议不能把状态依赖的后果推理与候选构造捷径分开。

因此冻结的准确结论是：**当前 Counterfactual v2 接口与开发集为 No-Go**。不在这批已暴露数据上继续增加 Adapter 宽度/层数或扫超参数，不启动 CEM，也不把动作排序写成论文第二贡献；但该结果不证明 fully observed future state、真正的显式 goal-conditioned scorer、重新平衡的候选协议或整个 LeWM/CostAdapter 思路普遍无效。FCPG Stage I 的独立正结果不受该负结论影响。

若未来重测，必须新建独立 v3 数据与 artifact，而不是覆盖 v2：以部署时可得的 policy action 为中心生成等范数、不同方向的匿名候选；先用相同 explicit goal 验证 `StateOracle > Direct`，再依次验证 oracle-anchor visual future、LeWM predicted future 和由当前 RGB 推断的 deployable GoalToken。scorer 固定为 `stopgrad(DirectScore) + FutureCostDelta`，Delta 最后一层零初始化，并加入 repeated-current 与 cyclic-future 参数匹配对照。只有上述表征阶梯通过后，才允许在同一 scorer 下检验 drift/FCPG 是否超过 drift/nominal；此前不启动 CEM。

### Counterfactual v3：StateOracle 协议修复 Gate（P0 No-Go，已冻结）

v3 只允许这一次协议修复，不是在 v2 development set 上继续扫参数。它先回答一个更基础的问题：去掉伪 goal、候选类别捷径与非嵌套模型比较后，当前候选动作和 `H=5` 时域是否形成了一个可由真实 future physical state 判别的任务排序问题。P0/P1 不加载 LeWM，也不训练 FCPG 或 policy。

固定数据合同如下：

- 四任务各采集 `12` 个全新 physical episode，按 episode SHA 排序固定为 `8 train / 4 validation`；snapshot 固定在 step `10/25/35`，不得按未来 reward 或成功选择；
- 候选以冻结 policy action chunk 为中心，只扰动 `25 x 3` 运动通道，夹爪通道逐字节保持不变；四个正交方向各取 `+/-`，共 `8` 个匿名 slot；最多按冻结 seed 尝试 `64` 组 QR basis 并取第一组可行解；所有动作先 clipping，再用 clipping-aware bisection 将 motion RMS 固定为 `0.15`，同时要求最小两两 RMS `>=0.135`、最大两两 cosine `<=0.60`；
- 每个主标签候选从同一 snapshot 只执行一次，保存 `H=5` 真实 state、return/progress/success；每任务 episode `0` 的 early snapshot 另做一次 repeat + reversed-order 恢复审计，其结果不增加主样本；candidate slot 对外匿名，axis/sign/permutation 仅存在隔离的 audit 分区；
- `current_state/future_state/goal_world/physical_task_score/return/progress/success` 全部是 privileged offline diagnostic 或标签。58 维 true state 保留 18 维语义前缀，并追加固定 padding 的 MuJoCo `qpos/qvel/ctrl/mocap`，从而不给 Direct 留下明显的非 Markov 状态缺口；
- `physical_task_score` 由固定的 task-aware staged trajectory cost 从同一 H5 future state 与 goal 确定性计算，是 Direct 与 StateOracle 唯一的 primary target；相同物理轨迹必须得到相同 score。MetaWorld cumulative return 只作 P0 候选有效性旁证；RGB、视觉 latent、`q_true`、reward/return/progress/success 标签和未来成功帧均不得进入 StateOracle；
- v3 episode 必须与已存在的 v2 physical episode hash 零交集。历史上游 checkpoint 缺完整 episode receipt 时，只能记录“无法形式化证明上游不重叠”，不得伪造 formal disjointness claim。

Gate 顺序与停止线固定为：

1. **P0 candidate audit**：每个任务都要求至少 `80%` validation snapshot 的 `physical_task_score` spread `>=0.1`，且 H5 cumulative return spread `>=0.01`；物理 task-cost 相对 uniform control 排序 H5 actual return 的 paired bootstrap 必须通过；只用 train episode 拟合的 `task x anonymous-slot` 与隐藏 `task x axis/sign` 查表，在 validation 上的 Spearman episode-bootstrap 95% CI 必须完整落入 `[-0.1,0.1]`，normalized-regret improvement bootstrap CI 上界不得超过 `0.02`。P0 任一条件失败时禁止训练 DirectRanker、禁止扩充确认集并立即发布 No-Go/inconclusive。
2. **P1 Direct qualification**：privileged Direct 获得完整 58 维 current true state、`goal_world`、task ID 与 candidate action；三 seed 各固定 `2,000` steps。每个 seed 的训练 loss 必须相对初始化下降至少 `10%`，seed-averaged train task-macro Spearman 必须 `>=0.2` 且严格超过 uniform、`task x slot` 和 `task x axis/sign` train-only 先验。未通过时只允许报告 baseline invalid，禁止让 StateOracle 因弱 baseline 过门。
3. **P1 StateOracle**：三 seed Direct score 先平均再统计。StateOracle 必须在 task-macro Spearman 和 normalized top-1 regret 两项上同时显著超过合格 Direct 及按 canonical axis 循环的 cyclic-future control，所有 paired physical-episode bootstrap CI 下界均大于 `0`；跨两个 baseline 的共同交集中，至少相同 `3/4` 任务必须在两个指标上都为正。
4. P1 通过只说明**当前协议存在可利用的真实 future-state 信息**，下一步最多允许独立预注册 VisualOracle。它不证明 LeWM、GoalToken、FCPG-grounded planning 或 CEM 有效。P1 失败的唯一允许结论是：当前候选、时域和物理 task-cost 协议没有暴露 future dynamics 的决策价值；不能外推为机器人任务普遍不需要世界模型。

唯一允许的执行入口为：

```bash
env PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $FCPG_PY \
  scripts/collect_lewm_counterfactual_v3_state_oracle.py \
  --confirm-state-oracle-development

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $FCPG_PY \
  scripts/run_lewm_counterfactual_v3_state_oracle_gate.py \
  --confirm-state-oracle-gate
```

两者均要求 clean git worktree、绑定同一 commit，使用独占原子目录并拒绝覆盖已有 artifact；通用数组存储器沿用经过审计的 v2 atomic-store 实现，但科学 schema/protocol 是独立 v3。配置冻结在 `config/lewm/eval/counterfactual_v3_state_oracle_seed9216.json`，canonical-config SHA-256 为 `8b8a63cd2dbb5e04bdc9362233c8d36414cd996338fb5e75bd4436d53aacdc6f`；实现入口为 `fcpg/lewm_counterfactual_v3*.py`。结果产生前不得修改上述阈值、episode/snapshot/candidate 数量、动作半径、Direct 预算或停止顺序。

本机前置 checkpoint 位于 Git 忽略的 `outputs/`，collector 会在运行前后复核文件与加载 state hash：action-only policy 为 `outputs/action_chunk/lewm_stage1/shared_4task_seed0_lp0_sig0_b2048/fcpg_action_chunk_best.pt`（SHA-256 `6e00a633703d2e2d1d0061e45f9a44293b550f3682ea409cf4d4c36a2039748f`），FCPG corrector 为 `outputs/transformer_residual/lewm_stage1/shared_4task_seed0_sig0/fcpg_transformer_residual_best.pt`（SHA-256 `3cfe621bbfb5b87962a3c569d6ebf9d4d702e0483adeebd45a4e00bb0a16ee79`）。外部 clean clone 必须先按 Stage I 冻结训练入口生成完全相同的 checkpoint，或取得这两个按 SHA 验证的 artifact；Git checkout 本身不包含它们。

唯一一次正式采集与 Gate 已在 clean commit `65503b8c0e28faba08aabc298be2a0fef93deb2a` 上完成。采集得到 `144` 个 snapshot、`1,152` 条主物理分支和 `36` 条恢复审计分支；dataset result SHA-256 为 `9165f9cec3ade056f116826da4e185e8163818ddb8ff05650f65cfa41edab887`。P0 结果如下：

| P0 检查 | 正式结果 | 判定 |
| --- | --- | --- |
| H5 actual-return spread | `48/48` validation snapshot 达到 `>=0.01`，四任务均为 `100%` | Pass |
| frozen physical-score spread | 总计 `14/48` 达到 `>=0.1`；pick/button/reach/push 分别为 `33.3% / 8.3% / 66.7% / 8.3%`，均低于每任务 `80%` 门槛 | **Fail** |
| physical score 排序 actual return | Spearman improvement `0.6473 [0.5928, 0.7034]`；normalized-regret improvement `0.2738 [0.2242, 0.3229]`；四任务共同正向 | Pass |
| `task x anonymous-slot` shortcut | Spearman `0.0268 [-0.0625, 0.1181]`；regret improvement `0.0107 [-0.0411, 0.0630]`，CI 未完全落入冻结等价区间 | **Fail** |
| hidden `task x axis/sign` shortcut | Spearman `0.1230 [0.0407, 0.2073]`；regret improvement `0.0453 [-0.0108, 0.1051]` | **Fail** |

P0 是合取 Gate，因此整体判为 **No-Go**，并在训练前停止：`direct_training_started=false`。准确结论是，真实 H5 分支回报有充分差异，且 frozen physical score 在有区分度时与真实回报方向一致；但当前 `H=5`、等范数候选与 frozen task-cost 的组合既没有在多数 snapshot 形成足够大的 primary-target spread，也未通过用于排除 axis/sign 与 public-slot 可迁移查表先验的等价性审计。因此本次开发协议不能隔离 future-state 的净决策价值。该结果不评价 LeWM、CostAdapter 或 FCPG 的能力，因为它们未进入本 Gate；不得调阈值后重跑 v3，也不得启动 Direct、VisualOracle、LeWM、GoalToken、FCPG-vs-nominal 或 CEM。正式 Gate result SHA-256 为 `3e21b7a381e8233ec01d5e88a7fa300618388fb999331da4938166143eed76eb`。

### Counterfactual v3：actual-return post-hoc 增量诊断（当前接口 No-Go）

正式 P0 的阈值与 No-Go 不作任何事后修改。下面是单独标记为 `post_hoc / exposed_validation / descriptive_only` 的机制诊断：只以 `actual_return[:, :, -1]` 为训练与评估目标，`physical_task_score` 完全不进入模型或指标。比较链固定为 train-only `task x axis/sign` lookup、privileged DirectRanker，以及 `stopgrad(DirectScore) + FutureStateResidual`。Direct 输入 current true state、candidate action、`goal_world` 和 task ID；residual 只额外读取真实 H5 future state。

第一次 naive 运行中，Direct 在 validation 上达到 Spearman `0.6314`、normalized top-1 regret `0.1735`，明显优于 axis/sign prior 的 `0.0665 / 0.4372`；四任务均同时改善。Nested residual 的点估计为 `0.6334 / 0.1621`，但这个小增量不能解释为 future-state 价值：Direct 在同一 residual 训练集上已达到 Spearman `0.9948`、regret `0.000279`，使 residual 在一个几乎被 base 完全拟合的 stacking 目标上训练。

因此只做一次结构性修正，不扫模型或超参数：在正式 train split 内按 physical episode 做固定 4-fold cross-fitting。每任务 `8` 个 train episode 按 `physical_episode_sha256` 排序后 `rank % 4`；每 fold 只用 `6` 个 episode/task 拟合 Direct、input scaler 和 label normalizer，并为其余 `2` 个 episode/task 生成 OOF score。同一 episode 的三个 phase 始终在同一 fold；全部 `96` 个 train snapshot 恰好获得一次 OOF score。各 fold score 先反变换到 raw return，再映射到全 train 的统一 task-normalized 坐标。Residual 只在这些 OOF base score 上训练；validation 时叠加到使用全部 train episode 拟合的 full Direct。

| Scorer | Train Spearman / regret | Validation Spearman / regret |
| --- | ---: | ---: |
| OOF Direct / full Direct | `0.4551 / 0.2822` | `0.6314 / 0.1735` |
| OOF-trained nested future-state residual | `0.9931 / 0.000366` | `0.5213 / 0.2176` |

Cross-fit 后，nested 相对 full Direct 的 validation Spearman 变化为 `-0.1101`，95% descriptive CI `[-0.1915,-0.0298]`；normalized-regret improvement 为 `-0.0441`，CI `[-0.1130,0.0309]`。只有 pick-place 在两项指标上同时正向，未达到固定的 `3/4` 任务标准。Residual 从 OOF base 在 train 上几乎拟合到完美，却在 held-out physical-episode validation 上明显退化；该 split 已暴露，因此这只说明当前 21k 参数 privileged future-state residual 在本次诊断中没有表现出稳定的跨 episode 增量控制信息。

该 post-hoc 诊断因此将**当前 `FutureStateResidual + full Direct` 接口**判为 No-Go，不再在已暴露 v3 数据上扩宽、扫参或换名重跑，也不据此启动 VisualOracle、LeWM、GoalToken、FCPG-vs-nominal 或 CEM。它不能外推为“future state 没有价值”或“世界模型无效”：validation 已暴露、样本仍小，且 nested 额外增加了容量。正结果原本也只会授权用全新 episode 预注册 v4，不会直接形成论文证据。

实现入口为 `fcpg/lewm_counterfactual_v3_exploratory.py`、`scripts/run_lewm_counterfactual_v3_exploratory.py` 和 `scripts/run_lewm_counterfactual_v3_exploratory_crossfit.py`；独立结果位于 Git 忽略的 `outputs/lewm_counterfactual_v3_exploratory/state_oracle_residual_{,crossfit_}seed9216/report.json`，不会覆盖正式 v3 artifact。

该 Counterfactual-v3 里程碑当时的全量回归为 `989 passed, 41 warnings`；`py_compile` 与 `git diff --check` 通过。

Counterfactual v2 历史实现入口：

```text
config/lewm/eval/counterfactual_v2_development_seed8192.json
scripts/collect_lewm_counterfactual_v2.py
fcpg/lewm_counterfactual_v2.py
fcpg/lewm_counterfactual_v2_runtime.py
fcpg/lewm_counterfactual_v2_storage.py
```

CLI 使用独占原子目录并拒绝覆盖已有输出；以下是本次已验证的 smoke 入口。由于目标目录已经原子发布，原样重跑会按设计拒绝覆盖：

```bash
$FCPG_PY scripts/collect_lewm_counterfactual_v2.py \
  --config config/lewm/eval/counterfactual_v2_development_seed8192.json \
  --smoke
```

### Stage I：LeWM-FCPG

Stage I 保留冻结的 ImageNet ViT-B/16 稠密视觉特征，以名义关键点投影、局部视觉 token、机器人本体状态、三维运动学关键点和 keypoint identity 作为输入。针对本项目适配的 LeWM 条件块使用非因果空间 token 交互，直接输出：

```text
q_corrected = q_nominal + delta_q
```

校正后的采样位置进入确定性 action-chunk policy。策略使用两帧观测历史，一次预测 32 步动作块；action-conditioned next-latent 只是可选训练期辅助目标，不用于 MPC、规划或测试时 latent rollout。数据中的 `history_action[-1]` 与当前 `action[0]` 对齐，因此禁止把该 next-latent 回接到同一步 action head，否则会形成 `a_t -> z_{t+1} -> a_t` 的目标动作泄漏。四任务 seed-0 匹配实验中，next-latent 与 SIGReg 都未优于 action-only `SIGReg=0`，因此二者保留为负消融而非主模型。论文中将 Stage I 模型称为 **lightweight predictive transformer** 或 **token-interaction residual corrector**；LeWM 是架构与训练实现来源，不是 FCPG 的 novelty，也不把该 BC policy 称为 world model。

Stage I 明确排除：

- VLA、语言输入和语言监督；
- DDPM 或其他扩散动作生成；
- MPC、CEM 和测试时 latent planning；
- 新的风险估计、校准不确定性或 `rho` 门控主分支；
- 部署时显式求解或迭代更新完整相机外参。

### Stage II 历史候选：KineParity

KineParity 只保留为历史负结果，不是当前方法或待执行任务。Oracle geometry Gate A 证明全臂星座在退化构型下具有几何价值；但 matched visual Gate B pilot 和唯一一次 frozen rescue 均弱于 direct-control prototype，因此 Gate B 判为 No-Go，feature-resampling / policy Gate C 取消。详细协议、公式与失败分析保留在 `docs/experiment_plan.md` 和后文历史结果中，禁止继续在已暴露数据上调阈值或重新包装为论文贡献。

## 上游来源与实现边界

LeWM 上游源码固定到 commit：

```text
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
```

原始快照放在 `third_party/lewm/`，保留上游 README、训练/评测入口、Hydra 配置、assets、实现模块、`LICENSE` 与 `UPSTREAM.md`，只移除嵌套 `.git`。原始文件不直接修改；所有 FCPG 适配均在项目模块中实现。依赖版本固定在 `requirements-lewm.txt`，第三方归属记录在 `THIRD_PARTY_NOTICES.md`。上游 `train.py` 是未经修改的归档副本，其历史 `Resize` API 与当前固定依赖不兼容；本项目只声明 FCPG 自有 trainer/evaluator 为已验证执行入口。

## 实验完整性约束

MetaWorld 的默认 39 维 observation 不是纯本体感觉：它包含物体状态、上一帧状态和目标位置。若 BC policy 直接使用该向量，策略会绕过 RGB 与几何接地，导致相机扰动评估失去意义。

正式 FCPG policy 实验必须使用：

```text
--obs-mode robot
```

该模式只保留当前末端执行器 XYZ 与夹爪开合四维本体感觉。residual head 与 policy 必须使用相同的 observation mode；代码会在不匹配时拒绝训练。旧的 `obs_mode=full` 曲线仅保留为 shortcut audit，不作为论文结果。

## 运行环境

本项目第一阶段**只使用一个 Python 环境**：

```text
/home/james/lewm/.venv
```

对应解释器是：

```text
/home/james/lewm/.venv/bin/python3
```

后续 README 中所有命令默认先执行：

```bash
export FCPG_PY=/home/james/lewm/.venv/bin/python3
```

然后统一用：

```bash
$FCPG_PY scripts/check_env.py
```

不要直接使用系统 Python，也不要临时换成其他项目的 `.venv`。当前系统 Python 是 3.12，缺少本项目需要的 `torch`、`mujoco`、`metaworld` 等依赖；`/home/james/lewm/.venv` 是已经验证可用的 Python 3.10 环境。

也可以进入虚拟环境后运行脚本：

```bash
source /home/james/lewm/.venv/bin/activate
python scripts/check_env.py
```

但为了避免 shell 状态混乱，本文档中的正式命令优先使用绝对解释器 `$FCPG_PY`。

当前已确认的核心版本：

```text
Python 3.10.20
numpy 1.26.4
torch 2.5.1+cu121
timm 1.0.27
mujoco 3.3.0
gymnasium 1.2.3
metaworld 3.1.1
```

这里说的 `.venv` 是 **Python 虚拟环境目录**。如果项目后续出现 `.env` 文件，它只用于保存环境变量，不等同于 Python 解释器环境。本项目运行代码时统一使用：

```text
/home/james/lewm/.venv/bin/python3
```

`metaworld==3.1.1` 已经安装。注意：安装 MetaWorld 时 pip 会按依赖把 `mujoco` 固定到 `3.3.0`，这和 `dm-control` 需要的 `mujoco>=3.8.1` 有版本冲突。第一阶段先服务 MetaWorld；如果后面要重新跑依赖 `dm-control` 的项目，最好单独建环境或恢复 MuJoCo 版本。

复现安装命令：

```bash
$FCPG_PY -m pip install -r requirements-metaworld.txt
```

如果 PyPI 安装不可用，再用 Farama 官方 GitHub 源：

```bash
$FCPG_PY -m pip install "git+https://github.com/Farama-Foundation/Metaworld.git"
```

投影和数据采集不要求 GPU；ViT 特征缓存、FCPG adapter 与 action-chunk policy 训练统一使用 CUDA0。正式对照实验使用 `--device cuda:0 --batch-size 1024`，以保持既有因子实验的优化步数和学习率可比。紧凑 policy visual cache 已在 32GB V100 上验证到 batch 4096，但更大的 batch 只作为吞吐上限，不直接替换正式配置。MuJoCo step 与 EGL 渲染仍由仿真后端执行，不会因为增大训练 batch 被搬到 GPU。

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_action_chunk_policy.py \
  ... \
  --batch-size 1024 \
  --num-workers 4 \
  --device cuda:0
```

## Legacy 基线证据（2026-07-10，旧版 `delta_q + rho`）

> **证据归属说明：** 本节及其复现命令记录的是改造前的 `delta_q + rho` adapter 与 legacy deterministic action-chunk Transformer。所有数值原样保留，用于证明 residual correction 的可行性和定义旧基线；它们**不是**新 LeWM-FCPG、next-latent policy、共享 adapter 或 Stage II `SE(3) + residual` 的结果。新模型必须使用独立输出目录和 `schema_version=2` checkpoint 重新训练，禁止复用这些数字作为新架构结果。

### 数据与公平性

正式数据集：

```text
data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz
task: pick-place-v3
frames: 30,000
complete episodes: 300
image: 224 x 224
actual camera translation: 0-4 cm
actual camera rotation: 0-5 deg
clean episode probability: about 25%
policy observation: robot-only 4D state
```

RGB、`q_true` 和动作来自实际扰动后的相机与环境；`q_nominal` 始终由固定名义标定计算。训练/验证按完整 episode 分组，不能让同一相机轨迹同时进入两侧。

### Legacy 模型

旧版 FCPG adapter：

```text
pretrained ViT-B/16, ImageNet weights
selected dense layer: 11
local anchor grid: 5 x 5
local radius: 24 px
local Transformer layers: 4
global keypoint Transformer layers: 2
model dimension: 256
outputs: delta_q and rho
```

可靠性训练包含 shuffled-image 和 local-cutout correspondence negatives。`rho` 只门控校正位置采到的视觉特征，本体 embedding 保留为 fallback；它能诊断视觉对应关系是否受损，但匹配训练消融尚未证明门控本身带来稳定策略收益。当前主机制是 `delta_q` 残差校正。

Policy：

```text
observation horizon: 2
action horizon: 32
memory encoder: 2 Transformer layers
action decoder: 6 Transformer layers
model dimension: 256
batch size: 1024
loss: Smooth L1 motion + binary gripper loss
```

GeoProp 与 FCPG 使用相同数据划分、视觉特征、policy 容量、优化器和 seed。两者唯一受控差异是 grounded keypoint 使用 `q_nominal`，还是使用 `q_corrected` 与 `rho`。

### 机制结果

未见 episode 上，正常视觉证据的投影误差从 `6.275 px` 降到 `0.727 px`，即 `E_after / E_before = 0.116`。

#### Legacy reliability appendix：离线诊断

以下 `rho` 数字仅作为旧版可靠性分支的诊断记录，不属于新 Stage I 主模型。

| 视觉证据 | E_after | mean rho |
| --- | ---: | ---: |
| 正常匹配图像 | 0.727 px | 0.760 |
| 名义接地点局部遮挡 | 6.805 px | 0.325 |
| 全零图像 | 11.026 px | 0.018 |
| 打乱图像对应关系 | 15.202 px | 0.327 |

这组诊断说明 adapter 不只是记忆扰动分布：图像对应关系被破坏时，校正失效且可靠性下降。

### 策略结果

Held-out action-chunk 离线评估使用相同数据划分，并重复 3 个 policy 初始化 seed。下面报告跨 seed 均值与样本标准差；FCPG adapter 暂时固定为 seed 0，因此这里只检验 policy 训练的可复现性。

| policy | chunk MSE | first-action MSE | gripper accuracy |
| --- | ---: | ---: | ---: |
| open-loop GeoProp | 0.020590 +/- 0.000257 | 0.024000 +/- 0.000925 | 0.9902 +/- 0.0005 |
| FCPG | 0.017887 +/- 0.000153 | 0.017144 +/- 0.000583 | 0.9911 +/- 0.0004 |

FCPG 的平均 chunk MSE 低约 13.1%，first-action MSE 低约 28.6%。离线指标不是 rollout 结论，只用于确认相同 policy 容量下 corrected grounding 更容易拟合动作。

最严格的 rollout 对照对每个 seed 使用同一个 FCPG policy checkpoint，只在测试时切换 `native FCPG` 与 `nominal/open-loop` grounding。固定 5 度实际相机旋转，每个 seed 运行 60 个相同 task index、环境 seed 和相机扰动的配对 episode：

| policy seed | nominal success | FCPG success | paired delta | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0500 | 0.2167 | +0.1667 | 0.012939 |
| 1 | 0.0167 | 0.2833 | +0.2667 | 0.000145 |
| 2 | 0.0167 | 0.2167 | +0.2000 | 0.000488 |

跨 seed 的 nominal success 为 `0.0278 +/- 0.0192`，FCPG success 为 `0.2389 +/- 0.0385`。配对提升为 `+0.2111 +/- 0.0509`，按“先重采样 policy seed，再在 seed 内重采样配对 episode”的层级 bootstrap 95% CI 为 `[0.1333, 0.2944]`。名义投影误差为 `18.411 +/- 0.007 px`，校正后为 `1.771 +/- 0.016 px`。

扰动范围诊断目前只在 policy seed 0 上完成：

| actual camera rotation | nominal success | FCPG success | nominal/corrected error | paired delta | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.3667 | 0.4333 | 0.000 / 0.687 px | +0.0667 | 0.289062 |
| 5 deg | 0.0500 | 0.2167 | 18.410 / 1.781 px | +0.1667 | 0.012939 |
| 10 deg | 0.0000 | 0.1000 | 37.259 / 14.671 px | +0.1000 | 0.031250 |
| 20 deg | 0.0000 | 0.0000 | 75.887 / 56.173 px | +0.0000 | 1.000000 |

解释必须保持克制：

- 固定 5 度扰动下，3 个 policy seed 都得到正向且显著的同策略收益；
- clean 条件下没有显著退化证据，但该结论目前只来自 seed 0；
- 10 度已超出 24 px 局部窗口，但部分 episode 仍可恢复；
- 20 度是明确失效区，FCPG v1 不是 calibration-free policy；
- pick-place 已有 3 个 policy seed，但 adapter seed 仍固定；第二任务结果如下。


### 第二任务复现：button-press-v3

第二任务使用与 pick-place 完全相同的协议：300 个完整 episode、30,000 帧、224 x 224 图像、变化的 MetaWorld task index、约 25% clean episode，以及 0-4 cm / 0-5 度实际相机扰动。scripted policy 在 300/300 episode 中成功，全部投影有效，99.84% 的关键点位于 24 px 局部可恢复范围内。

机制结果：

- 不重新训练时，pick-place adapter 跨任务使用可将误差从 `7.415 px` 降到 `3.502 px`，ratio 为 `0.472`；
- button-press adapter 在 held-out episode 上将误差从 `6.359 px` 降到 `0.595 px`，ratio 为 `0.094`；
- 在全 30,000 帧缓存上，任务内 adapter 将误差从 `7.415 px` 降到 `0.410 px`。

视觉证据审计：

| evidence | E_after | mean rho |
| --- | ---: | ---: |
| normal | 0.595 px | 0.788 |
| local cutout | 8.249 px | 0.127 |
| zero image | 10.849 px | 0.206 |
| shuffled image | 14.439 px | 0.286 |

Held-out action-chunk 离线评估：

| seed | GeoProp chunk MSE | FCPG chunk MSE | GeoProp first-action MSE | FCPG first-action MSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.002272 | 0.001930 | 0.004237 | 0.002619 |
| 1 | 0.002175 | 0.001958 | 0.003877 | 0.002468 |
| 2 | 0.002620 | 0.001797 | 0.003741 | 0.002352 |

跨 seed 的 GeoProp chunk MSE 为 `0.002356 +/- 0.000234`，FCPG 为 `0.001895 +/- 0.000086`；GeoProp first-action MSE 为 `0.003952 +/- 0.000257`，FCPG 为 `0.002480 +/- 0.000134`。

固定 5 度实际相机旋转、每个 policy seed 60 个 episode 的同策略 grounding override 配对 rollout：

| seed | nominal success | FCPG success | paired delta | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6667 | 1.0000 | +0.3333 | 0.000002 |
| 1 | 0.6000 | 0.9833 | +0.3833 | < 0.000001 |
| 2 | 0.7500 | 0.9667 | +0.2167 | 0.000977 |

跨 seed 的 nominal success 为 `0.6722 +/- 0.0752`，FCPG success 为 `0.9833 +/- 0.0167`。配对提升为 `+0.3111 +/- 0.0855`，层级 bootstrap 95% CI 为 `[0.2056, 0.4111]`。

以下扰动范围诊断固定使用 policy seed 0：

| actual camera rotation | nominal success | FCPG success | nominal/corrected error | paired delta | exact McNemar p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 1.0000 | 1.0000 | 0.000 / 0.498 px | +0.0000 | 1.000000 |
| 5 deg | 0.6667 | 1.0000 | 18.587 / 1.281 px | +0.3333 | 0.000002 |
| 10 deg | 0.6167 | 0.6333 | 37.420 / 13.758 px | +0.0167 | 1.000000 |
| 20 deg | 0.5833 | 0.7000 | 76.209 / 54.004 px | +0.1167 | 0.118469 |

独立训练的 GeoProp policy 在 5 度下成功率为 `0.7833`，FCPG policy 为 `1.0000`，配对提升 `+0.2167`，McNemar `p=0.000244`。

#### Legacy reliability appendix：rollout 与匹配训练消融

遮挡与 reliability 因果消融固定使用 policy seed 0、5 度旋转，以及围绕每个真实机器人关键点的 12 px 方形遮挡：

| reliability setting | success | mean rho | learned-rho vs rho=1 delta |
| --- | ---: | ---: | ---: |
| learned rho | 0.6833 | 0.180 | +0.0333 |
| forced rho=1 | 0.6500 | 1.000 | reference |
| forced rho=0 | 0.6333 | 0.000 | -0.0167 |

learned `rho` 在无遮挡的同一 5 度设置中均值约为 `0.611`，遮挡后降到 `0.180`，说明 reliability head 能识别局部视觉证据破坏。但是 learned `rho` 相对 `rho=1` 的成功率差值仅为 `+0.0333`，bootstrap 95% CI 为 `[-0.0333, 0.1000]`，McNemar `p=0.625`，当前不能声称可靠性门控带来显著策略收益。

匹配训练的 reliability 消融使用完全相同的数据、split、policy 架构、batch size 和训练轮数；唯一差异是 residual-only policy 在训练与推理时固定 `rho=1`：

| seed | residual-only chunk MSE | full FCPG chunk MSE | residual-only success | full FCPG success | paired delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001966 | 0.001930 | 0.9667 | 1.0000 | +0.0333 |
| 1 | 0.001853 | 0.001958 | 0.9833 | 0.9833 | +0.0000 |
| 2 | 0.001682 | 0.001797 | 0.9833 | 0.9667 | -0.0167 |

跨 seed 的 residual-only chunk MSE 为 `0.001833 +/- 0.000143`，full FCPG 为 `0.001895 +/- 0.000086`；rollout success 分别为 `0.9778 +/- 0.0096` 和 `0.9833 +/- 0.0167`。full FCPG 的配对成功率差值仅为 `+0.0056 +/- 0.0255`，层级 bootstrap 95% CI 为 `[-0.0278, 0.0389]`。因此，`rho` 能检测视觉证据破坏，但当前没有证据表明可靠性门控改善策略性能；被两任务、多 policy seed 支持的核心结论是视觉反馈残差校正。10 度与 20 度虽然降低了接地误差，但策略成功率提升均不显著。

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

### Legacy 复现入口

缓存冻结的 ViT layer-11 特征：
**ARCHIVED REPRODUCTION ONLY：以下命令不属于当前执行计划，不得用于恢复已判 No-Go 的 reliability 或旧策略分支。**


```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/cache_vit_backbone_features.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --selected-layers 11 \
  --batch-size 256 \
  --device cuda:0 \
  --output-dir data/vision_cache/pick300_224_vitb_l11_fp16
```

训练 FCPG adapter（先主训练，再加强 reliability negatives）：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_transformer_residual.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --vision-cache data/vision_cache/pick300_224_vitb_l11_fp16 \
  --vision-selected-layers 11 \
  --obs-mode robot \
  --model-dim 256 \
  --vision-dim 256 \
  --local-grid-size 5 \
  --local-radius-px 24 \
  --local-layers 4 \
  --global-layers 2 \
  --max-residual-px 32 \
  --reliability-sigma-px 5 \
  --reliability-weight 0.2 \
  --reliability-corruption-prob 0.35 \
  --cutout-radius-px 18 \
  --sigreg-weight 0.0001 \
  --batch-size 1024 \
  --epochs 24 \
  --device cuda:0 \
  --output-dir outputs/transformer_residual/pick300_224_vitb_l11_t256_l4g2_r24_b1024_relneg_seed0

CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_transformer_residual.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --vision-cache data/vision_cache/pick300_224_vitb_l11_fp16 \
  --vision-selected-layers 11 \
  --obs-mode robot \
  --init-checkpoint outputs/transformer_residual/pick300_224_vitb_l11_t256_l4g2_r24_b1024_relneg_seed0/fcpg_transformer_residual_best.pt \
  --model-dim 256 \
  --vision-dim 256 \
  --local-grid-size 5 \
  --local-radius-px 24 \
  --local-layers 4 \
  --global-layers 2 \
  --max-residual-px 32 \
  --reliability-sigma-px 5 \
  --reliability-weight 0.5 \
  --reliability-corruption-prob 0.5 \
  --cutout-radius-px 18 \
  --sigreg-weight 0.0001 \
  --batch-size 1024 \
  --epochs 10 \
  --lr 0.0001 \
  --warmup-epochs 1 \
  --device cuda:0 \
  --output-dir outputs/transformer_residual/pick300_224_vitb_l11_r24_b1024_relmix_ft_seed0
```

缓存校正点与可靠性：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/cache_fcpg_corrections.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --vision-cache data/vision_cache/pick300_224_vitb_l11_fp16 \
  --checkpoint outputs/transformer_residual/pick300_224_vitb_l11_r24_b1024_relmix_ft_seed0/fcpg_transformer_residual_best.pt \
  --batch-size 1024 \
  --device cuda:0 \
  --output-dir data/grounding_cache/pick300_224_fcpg_l11_relmix_ft_seed0
```

训练确定性 action-chunk policy：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_action_chunk_policy.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --mode fcpg \
  --fcpg-checkpoint outputs/transformer_residual/pick300_224_vitb_l11_r24_b1024_relmix_ft_seed0/fcpg_transformer_residual_best.pt \
  --vision-cache data/vision_cache/pick300_224_vitb_l11_fp16 \
  --grounding-cache data/grounding_cache/pick300_224_fcpg_l11_relmix_ft_seed0 \
  --vision-selected-layers 11 \
  --obs-mode robot \
  --obs-horizon 2 \
  --action-horizon 32 \
  --model-dim 256 \
  --encoder-layers 2 \
  --decoder-layers 6 \
  --batch-size 1024 \
  --epochs 60 \
  --device cuda:0 \
  --output-dir outputs/action_chunk/pick300_224_robot_fcpg_vitb_l11_t256_e2d6_ah32_b1024_seed0
```

GeoProp 强基线使用完全相同命令，只把 `--mode fcpg` 改为 `--mode geoprop`，并删除 `--fcpg-checkpoint` 与 `--grounding-cache`；其余超参数和 split seed 不得改变。

完成多个 policy seed 后，使用 `scripts/summarize_action_chunk_seeds.py` 汇总离线指标与同策略配对 rollout。该脚本将 policy seed 作为统计层级，并输出 JSON 与 Markdown，避免把不同 policy seed 上重复使用的环境 episode 误当作完全独立样本。

下面的早期 MLP、单步 BC 和旧 ViT smoke 仅用于记录研发过程；当前论文主证据以上述 deterministic action-chunk pipeline 为准。

## Legacy 命令与历史诊断

环境检查：

```bash
$FCPG_PY scripts/check_env.py
```

Reusable FCPG module smoke check：

```bash
$FCPG_PY scripts/smoke_fcpg_modules.py \
  --checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt
```

该命令会检查 `sample_feature_map_at_pixels`、`GroundedTokenAdapter`、`ResidualCorrectionHead`、`FCPGResidualOnly`、`ReliabilityConfidenceHead` 的 shape、反传和旧 residual checkpoint 加载兼容性。

Stage 1 synthetic heatmap sanity check：

```bash
$FCPG_PY scripts/stage1_synthetic_heatmap.py
```

MetaWorld probe：

```bash
$FCPG_PY scripts/metaworld_probe.py --task reach-v3 --camera corner --width 128 --height 128
```

Stage 0 projection drift：

```bash
$FCPG_PY scripts/stage0_projection_drift.py --task reach-v3 --camera corner --width 128 --height 128
```

Stage 2 projection dataset smoke collection：

```bash
$FCPG_PY scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3,push-v3 \
  --camera corner \
  --width 128 \
  --height 128 \
  --episodes-per-task 1 \
  --steps-per-episode 5 \
  --warmup-steps 1 \
  --perturbations-per-frame 2 \
  --output data/metaworld_projection/smoke_reach_push.npz

$FCPG_PY scripts/inspect_projection_dataset.py \
  data/metaworld_projection/smoke_reach_push.npz
```

Stage 4 scripted demo smoke collection（pre-action aligned）：

BC / policy 数据必须满足一个硬约束：`obs`、`rgb`、`q_true/q_nominal` 和 `action` 都来自同一个 pre-action 时刻，即 `action_t` 是由 `obs_t` 计算出来的，图像和投影也在 `env.step(action_t)` 之前采集。采集脚本会在 `.npz` 中写入：

```text
observation_action_alignment = pre_action
```

先跑一个短 smoke，确认字段和对齐标记：

```bash
$FCPG_PY scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3 \
  --camera corner \
  --width 128 \
  --height 128 \
  --episodes-per-task 1 \
  --steps-per-episode 8 \
  --warmup-steps 1 \
  --perturbations-per-frame 1 \
  --action-source scripted \
  --translations-cm 0 \
  --directions x+ \
  --yaws-deg 0 \
  --output data/metaworld_projection/scripted_reach_preact_smoke.npz

$FCPG_PY scripts/inspect_projection_dataset.py \
  data/metaworld_projection/scripted_reach_preact_smoke.npz
```

Longer reach-only scripted data for BC smoke：

```bash
$FCPG_PY scripts/collect_metaworld_projection_dataset.py \
  --tasks reach-v3 \
  --camera corner \
  --width 128 \
  --height 128 \
  --episodes-per-task 10 \
  --steps-per-episode 100 \
  --warmup-steps 1 \
  --perturbations-per-frame 1 \
  --action-source scripted \
  --translations-cm 0 \
  --directions x+ \
  --yaws-deg 0 \
  --seed 11 \
  --output data/metaworld_projection/scripted_reach_preact_clean_10eps.npz
```

Stage 4 minimal BC smoke training：

`train_bc_policy.py` 默认会拒绝没有 `observation_action_alignment=pre_action` 的旧数据。只有复现历史 plumbing smoke 时才使用 `--allow-unaligned-data`，正式 BC / policy 实验不要使用该开关。

```bash
$FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model state \
  --epochs 30 \
  --batch-size 64 \
  --output-dir outputs/bc_policy/reach_preact10_state

$FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model geoprop \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_geoprop

$FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model fcpg \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_fcpg

$FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model oracle \
  --epochs 30 \
  --batch-size 64 \
  --token-dim 16 \
  --output-dir outputs/bc_policy/reach_preact10_oracle

$FCPG_PY scripts/compare_bc_metrics.py \
  outputs/bc_policy/reach_preact10_state/state_metrics.json \
  outputs/bc_policy/reach_preact10_geoprop/geoprop_metrics.json \
  outputs/bc_policy/reach_preact10_oracle/oracle_metrics.json \
  outputs/bc_policy/reach_preact10_fcpg/fcpg_metrics.json \
  --output outputs/bc_policy/reach_preact10_summary.md
```

Stage 4 LeWM-style ViT adapter training（CUDA0 示例）：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model geoprop \
  --policy-arch vit \
  --epochs 80 \
  --batch-size 64 \
  --lr 0.0003 \
  --token-dim 192 \
  --hidden-dim 256 \
  --image-patch-size 16 \
  --transformer-layers 6 \
  --transformer-heads 6 \
  --transformer-dropout 0.05 \
  --sigreg-weight 0.0001 \
  --sigreg-proj 512 \
  --output-dir outputs/bc_policy/reach_preact10_geoprop_vit_sig1e4_t192_l6

CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_bc_policy.py \
  --data data/metaworld_projection/scripted_reach_preact_clean_10eps.npz \
  --model fcpg \
  --residual-checkpoint outputs/residual_identity/large_h512_cleanw10_translation4_seed0_local_image/local-image_residual_best.pt \
  --policy-arch vit \
  --epochs 80 \
  --batch-size 64 \
  --lr 0.0003 \
  --token-dim 192 \
  --hidden-dim 256 \
  --image-patch-size 16 \
  --transformer-layers 6 \
  --transformer-heads 6 \
  --transformer-dropout 0.05 \
  --sigreg-weight 0.0001 \
  --sigreg-proj 512 \
  --output-dir outputs/bc_policy/reach_preact10_fcpg_large_h512_vit_sig1e4_t192_l6
```

同一组对照应至少包含：`state`、`geoprop`、`oracle`、`fcpg`。`state + --policy-arch vit` 表示 RGB+state ViT baseline，不注入 grounded keypoint tokens。


Stage 4 minimal rollout smoke evaluation：

```bash
$FCPG_PY scripts/evaluate_bc_rollout.py \
  --checkpoint outputs/bc_policy/reach_preact10_state/state_bc_best.pt \
  --task reach-v3 \
  --episodes 2 \
  --horizon 60 \
  --translations-cm 0 \
  --directions x+ \
  --yaws-deg 0 \
  --output-dir outputs/bc_rollout/reach_preact10_state_clean

$FCPG_PY scripts/evaluate_bc_rollout.py \
  --checkpoint outputs/bc_policy/reach_preact10_fcpg/fcpg_bc_best.pt \
  --task reach-v3 \
  --episodes 2 \
  --horizon 60 \
  --translations-cm 4 \
  --directions x+ \
  --yaws-deg 0 \
  --output-dir outputs/bc_rollout/reach_preact10_fcpg_trans4

$FCPG_PY scripts/compare_rollout_summaries.py \
  outputs/bc_rollout/reach_preact10_state_clean/state_rollout_summary.json \
  outputs/bc_rollout/reach_preact10_geoprop_clean/geoprop_rollout_summary.json \
  outputs/bc_rollout/reach_preact10_fcpg_clean/fcpg_rollout_summary.json \
  outputs/bc_rollout/reach_preact10_fcpg_trans4/fcpg_rollout_summary.json \
  --output outputs/bc_rollout/reach_preact10_rollout_summary.md
```

Stage 2 multi-direction residual benchmark：

```bash
$FCPG_PY scripts/run_residual_benchmark.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --directions x-,y- \
  --seeds 0,1 \
  --epochs 15 \
  --batch-size 128 \
  --output-dir outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv
```

Stage 2 held-out magnitude residual benchmark：

```bash
$FCPG_PY scripts/run_residual_benchmark.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --directions '' \
  --translations-cm 4 \
  --yaws-deg 10 \
  --seeds 0,1 \
  --epochs 15 \
  --batch-size 128 \
  --output-dir outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg
```

Stage 2 perturbation severity curves：

```bash
$FCPG_PY scripts/evaluate_residual_curve.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --group-by translation_cm \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_no-image/no-image_residual_best.pt \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_1_no-image/no-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_1_local-image/local-image_residual_best.pt \
  --output-dir outputs/residual_curves/translation_2seed \
  --title "Translation severity residual correction"

$FCPG_PY scripts/evaluate_residual_curve.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --group-by yaw_deg \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_0_no-image/no-image_residual_best.pt \
  --checkpoint no-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_1_no-image/no-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_0_local-image/local-image_residual_best.pt \
  --checkpoint local-image=outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/yaw_deg_10_seed_1_local-image/local-image_residual_best.pt \
  --output-dir outputs/residual_curves/yaw_2seed \
  --title "Yaw severity residual correction"
```

Generated severity figures：

```text
outputs/residual_curves/translation_2seed/translation_cm_curve.png
outputs/residual_curves/yaw_2seed/yaw_deg_curve.png
```

Stage 2 residual benchmark figures：

```bash
$FCPG_PY scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv/summary.csv \
  --output-dir outputs/residual_figures/direction \
  --title "Held-out direction residual correction"

$FCPG_PY scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/summary.csv \
  --output-dir outputs/residual_figures/magnitude \
  --title "Held-out magnitude residual correction"

$FCPG_PY scripts/plot_residual_benchmark.py \
  --summary outputs/residual_benchmark/stage2_smoke_dirs_xm_ym_coordconv/summary.csv \
  --summary outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/summary.csv \
  --labels direction,magnitude \
  --output-dir outputs/residual_figures/combined \
  --title "Stage 2 residual correction sanity benchmark"
```

Generated outputs：

```text
outputs/residual_figures/direction/ratio_bar.png
outputs/residual_figures/magnitude/ratio_bar.png
outputs/residual_figures/combined/ratio_bar.png
```

Stage 2 residual correction qualitative visualization：

```bash
$FCPG_PY scripts/visualize_residual_corrections.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --selection best \
  --count 12 \
  --columns 4 \
  --show-patch \
  --output-dir outputs/residual_visualizations/translation4_local_image

$FCPG_PY scripts/visualize_residual_corrections.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --selection worst \
  --count 12 \
  --columns 4 \
  --show-patch \
  --output-dir outputs/residual_visualizations/translation4_local_image
```

Generated qualitative outputs：

```text
outputs/residual_visualizations/translation4_local_image/best_grid.png
outputs/residual_visualizations/translation4_local_image/best_summary.md
outputs/residual_visualizations/translation4_local_image/worst_grid.png
outputs/residual_visualizations/translation4_local_image/worst_summary.md
```

Stage 2 residual predictor smoke training：

```bash
$FCPG_PY scripts/train_residual_predictor.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --model no-image \
  --epochs 15 \
  --batch-size 128 \
  --heldout-directions y- \
  --output-dir outputs/residual_predictor/stage2_smoke_no_image_best

$FCPG_PY scripts/train_residual_predictor.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --model local-image \
  --epochs 15 \
  --batch-size 128 \
  --heldout-directions y- \
  --output-dir outputs/residual_predictor/stage2_smoke_local_image_best

$FCPG_PY scripts/compare_residual_metrics.py \
  outputs/residual_predictor/stage2_smoke_no_image_best/no-image_metrics.json \
  outputs/residual_predictor/stage2_smoke_local_image_best/local-image_metrics.json
```

Stage 3 reliability target diagnostics：

```bash
$FCPG_PY scripts/inspect_reliability_targets.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --sigma-px 0.5,1,2,4 \
  --output outputs/reliability_head/translation4_seed0/target_diagnostics.json
```

Stage 3 reliability head sanity checks：

```bash
$FCPG_PY scripts/train_reliability_head.py \
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

$FCPG_PY scripts/train_reliability_head.py \
  --data data/metaworld_projection/stage2_smoke_4tasks.npz \
  --residual-checkpoint outputs/residual_benchmark/stage2_smoke_magnitude_4cm_10deg/translation_cm_4_seed_0_local-image/local-image_residual_best.pt \
  --heldout-translations-cm 4 \
  --epochs 20 \
  --batch-size 128 \
  --sigma-px 2.0 \
  --target-kind corr \
  --loss bce \
  --output-dir outputs/reliability_head/translation4_seed0_predfeat_sigma2
```

## Stage I 当前论文主线与冻结规格

Stage I 的唯一方法主线是 residual-only feedback grounding：保留初始名义标定和运动学投影，以当前观测中的局部视觉证据预测二维投影残差，直接移动策略读取视觉特征的位置。新主模型不输出 `rho`，不做风险估计，也不把 LeWM 或 next-latent 本身包装成 novelty。

```text
q_nominal -> local visual tokens -> delta_q
q_corrected = q_nominal + delta_q
q_corrected -> policy feature sampling -> action chunk
```

### Residual-only LeWM corrector

- 视觉主干固定为冻结的 ImageNet ViT-B/16；训练稠密特征投影、局部融合和轻量 corrector。
- 每个关键点的条件输入由 `q_nominal`、FK 三维关键点、robot-only observation、keypoint identity 和局部 ViT token 构成。
- 局部空间交互使用 4 层 conditional、non-causal LeWM blocks；局部 latent 再经过 2 层 conditional、non-causal cross-keypoint blocks。
- 条件块使用 AdaLN-zero。二维 `delta_q` head 的最后一层权重和 bias 均为零初始化，因此未训练模型严格满足 `q_corrected == q_nominal`。
- 输出经过 `tanh` 并乘 `max_residual_px`，保持局部校正边界；输出结构只包含 `delta_q`、`q_corrected` 和 keypoint latent，不包含 `rho`。

校正器监督固定为：

```text
delta_q_target = q_true - q_nominal

L_corrector = SmoothL1_beta=1.0(delta_q_pred, delta_q_target)
            + lambda_sig * SIGReg(keypoint_latent)
```

`lambda_sig in {0, 0.02, 0.09}` 的匹配实验已经完成。task-specific residual corrector 的正式 reference 固定为 `lambda_sig=0`；`0.02/0.09` 只保留为已完成的表示正则负消融，不再是待训练候选。

### 历史负消融：Predictive action-chunk policy

该历史策略使用两帧 observation history，一次预测 32 步动作块。LeWM predictive policy 使用同一个 online state encoder 生成动作分支的 `z_t` 和下一观测目标 `z_(t+1)`；不建立 EMA target encoder，不对目标 latent 做 `detach`，也不复制第二套 encoder。

```text
L_action = SmoothL1_beta=0.05(motion_pred, motion_target)
         + 0.5 * BCEWithLogits(gripper_logit, gripper_target)

L_total = L_action
        + 1.0 * MSE(z_next_pred, z_next_target)
        + lambda_sig * SIGReg(z_t)
```

`next_mask` 排除 episode 末帧的 next-latent loss。匹配的四任务实验表明高秩或更低预测误差没有转化为控制收益，并出现稳定 rollout 退化，因此该分支已经判为 No-Go；当前 action-policy reference 使用 action-only，推理不运行 MPC、CEM、latent rollout 或测试时规划。

### 共享 adapter 与 few-shot 协议

- task-specific adapter 保留为已见任务基线；四任务 shared adapter 使用同一组参数服务 pick-place、button-press、reach 和 push。
- 跨任务泛化使用严格 leave-one-task-out：每折只在三个源任务训练 corrector，再在第四个未见任务的固定 held-out episodes 上评估。
- few-shot 目标域适配使用 `0 / 10 / 50 / 200` 个完整 episode，不能按 frame 随机抽样造成 episode 泄漏。
- 同一 episode selection seed 生成嵌套子集 `10 subset 50 subset 200`；正式曲线使用 final checkpoint，不能用目标任务 held-out 指标挑选 best epoch。
- task-specific、shared、zero-shot transfer 和 few-shot adapter 最终均使用 seed `0 / 1 / 2`，报告每任务 `E_before`、`E_after`、ratio 和策略配对 rollout。
- 在完整曲线和多 seed 完成前，只能写“评估跨任务迁移”，不能写 plug-and-play、零训练通用或少样本已成立。

### 接口、checkpoint 与 cache 约定

- corrector 接口为 `corrector_arch in {legacy, lewm}`；loader 缺少该字段时必须按 `legacy` 加载旧 checkpoint。
- policy 兼容接口仍支持 `policy_arch in {legacy, lewm-predictive}`；历史 loader 的 `latent_pred_weight` 默认值为 `1.0`，当前 reference config 必须显式固定 `latent_pred_weight=0`、`sigreg_weight=0`。
- Stage I corrector/action-policy checkpoint 使用 `schema_version=2`，记录上游 LeWM commit、`corrector_arch`、`policy_arch`、observation mode、所有损失权重和模型构造参数；独立 full grounded LeWM 使用 `checkpoint_type=fcpg_grounded_lewm_v1` 与自身 schema，二者不得混用或覆盖。
- residual-only grounding cache 必须写 `q_corrected.npy` 和 version-2 metadata，并设置 `has_reliability=false`；不得生成伪造的 `reliability.npy`。legacy cache 继续兼容 `q_corrected.npy + reliability.npy`。
- 训练脚本必须拒绝 adapter observation mode、dataset observation mode 或 cache source 不一致的组合。

### Stage I go / no-go

1. Corrector：已见任务必须达到 `E_after / E_before <= 0.15`，且零初始化、残差上界和 legacy loader 回归测试全部通过；未见任务结果单独报告，不得与已见任务混合。
2. Policy：固定 5 度扰动下，两个任务的同 checkpoint grounding override 配对提升都必须保持正向；正式结论使用 adapter seed 与 policy seed 分层统计。
3. Next-latent：匹配对照已完成并判为 No-Go；它只作为负消融保留，不进入最终主模型，也不再触发扩训练或 CEM。
4. Transfer：严格四折 LOTO 必须逐任务报告。目前 seed `0` 为 `3/4` 通过，不能声称通用 zero-shot；只有 shared adapter 与 `0 / 10 / 50 / 200` few-shot 曲线在 3 个 adapter seed 上复现后，迁移才进入贡献点。
5. 在新 checkpoint 完成训练与评测前，下面所有 legacy 数字只能作为旧基线，不能冒充 LeWM-FCPG、predictive policy 或 shared adapter 的结果。

## Legacy：历史阶段记录（128x128 早期原型）

已在 `/home/james/lewm/.venv` 中安装并验证 `metaworld==3.1.1`。当前 MetaWorld 任务名使用 `*-v3`，第一版任务为：

```text
reach-v3
pick-place-v3
push-v3
button-press-v3
```

已验证这四个任务都可以创建、随机 step、`corner` 相机渲染，并能读取以下 MuJoCo site 作为第一版机器人关键点：

```text
leftEndEffector
rightEndEffector
endEffector
```

在 128x128 图像、`corner` 相机、平移 0-4cm 和 yaw 0-10deg 的默认扰动下，Stage 0 投影漂移结果为：

| 任务 | mean px | p90 px | max px |
| --- | ---: | ---: | ---: |
| reach-v3 | 1.68 | 3.95 | 5.85 |
| pick-place-v3 | 1.70 | 3.97 | 5.99 |
| push-v3 | 1.69 | 3.97 | 5.86 |
| button-press-v3 | 2.56 | 4.95 | 9.40 |

结论：第一版 `radius=16` 的 local patch 足以覆盖这些默认扰动，可以进入真实图像 residual correction 数据采集。

已完成 Stage 2 数据采集 smoke test：

```text
tasks: reach-v3, push-v3
samples: 20
rgb_shape: (20, 128, 128, 3)
obs_shape: (20, 39)
q_shape: (20, 3, 2)
pixel_error_mean: 2.0678
pixel_error_p90: 4.2390
pixel_error_max: 5.3428
inside_patch_rate: 1.0000
```

保存字段包括 `rgb`、`obs`、`action`、`keypoints_xyz`、`q_true`、`q_nominal`、`camera_intrinsic`、`true_extrinsic`、`nominal_extrinsic`、`pixel_error` 和 `q_true_inside_patch`。当时计划是在该数据格式上训练 residual head 并加入 no-image 对照；该计划已由后续 actual-camera 数据与 P0/P1 实验取代。

已完成 Stage 4 scripted action smoke test，并修正 BC 数据对齐：

```text
scripted_reach_preact_smoke.npz
samples: 8
tasks: reach-v3
action_source: scripted
observation_action_alignment: pre_action
inside_patch_rate: 1.0000
success_mean: 0.0000  # 8-step smoke 太短，不代表 policy 质量
```

结论：MetaWorld v3 的 scripted policies 可用，采集脚本现在可以同时生成专家动作、RGB、`q_true/q_nominal` 和 perturbation metadata。用于 BC / policy integration 的新数据必须检查 `observation_action_alignment=pre_action`。随机 action 数据只用于 residual 机制验证，不用于最终策略实验。

重要修正：早期 `scripted_smoke_reach_push.npz`、`scripted_smoke_reach_long.npz` 以及基于它们得到的 minimal BC / rollout smoke 是采集器修正前的历史 plumbing 结果，可能存在 observation-action off-by-one。它们只说明代码路径能跑，不再作为任何策略效果证据。后续 BC 结果必须用 `pre_action` 数据重新采集和重训。`train_bc_policy.py` 现在默认拒绝缺失该标记的旧数据。

已完成 pre-action Stage 4 minimal BC re-run：

```text
scripted_reach_preact_clean_10eps.npz
samples: 1000
tasks: reach-v3
action_source: scripted
observation_action_alignment: pre_action
success_mean: 0.4900
```

| run | model | best epoch | best MSE | best MAE | final MSE | final MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reach_preact10_state | state | 27 | 0.000547 | 0.008896 | 0.000551 | 0.008888 |
| reach_preact10_geoprop | geoprop | 28 | 0.000208 | 0.007088 | 0.000217 | 0.007054 |
| reach_preact10_oracle | oracle | 28 | 0.000208 | 0.007088 | 0.000217 | 0.007054 |
| reach_preact10_fcpg | fcpg | 30 | 0.000113 | 0.005628 | 0.000113 | 0.005628 |
| reach_preact10_fcpg_cleanw10 | fcpg | 30 | 0.000077 | 0.005038 | 0.000077 | 0.005038 |

Detailed table:

```text
outputs/bc_policy/reach_preact10_with_cleanw10_summary.md
```

Interpretation: this is still only supervised action regression on one reach dataset, not a rollout robustness claim. It does show that the corrected pre-action data path works, GeoProp-style tokens are useful for imitation on this smoke set, and the FCPG residual-only path trains end-to-end. The clean-weighted residual checkpoint further improves action regression, but rollout remains the deciding check.

已完成 LeWM-style ViT adapter + SIGReg BC re-run（CUDA0, `token_dim=192`, `layers=6`, `heads=6`, `sigreg_weight=1e-4`）：

| run | model | best epoch | best MSE | best MAE | final MSE | final MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reach_preact10_state_vit_sig1e4_t192_l6 | state | 73 | 0.000034 | 0.003784 | 0.000059 | 0.005279 |
| reach_preact10_geoprop_vit_sig1e4_t192_l6 | geoprop | 75 | 0.000030 | 0.003478 | 0.000073 | 0.005687 |
| reach_preact10_oracle_vit_sig1e4_t192_l6 | oracle | 75 | 0.000030 | 0.003478 | 0.000073 | 0.005687 |
| reach_preact10_fcpg_large_h512_vit_sig1e4_t192_l6 | fcpg | 79 | 0.000023 | 0.002859 | 0.000062 | 0.005192 |

Detailed table:

```text
outputs/bc_policy/reach_preact10_vit_sig1e4_t192_l6_summary.md
```

Interpretation: ViT adapter 明显强于早期 MLP smoke head，说明后续 policy integration 不应该再只依赖小 MLP。GeoProp 和 oracle 在 clean 数据上结果一致，因为训练数据使用 `translations-cm=0`，此时 `q_nominal == q_true`。FCPG-ViT 的 supervised BC 指标最好，但差距很小，仍然不能单独作为鲁棒性结论。


已完成 pre-action Stage 4 minimal rollout re-run：

| run | model | perturbation | success | reward mean | nominal px err | corrected px err |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| reach_preact10_state_clean | state | 0cm | 0.0000 | 695.9383 | 0.0000 | 0.0000 |
| reach_preact10_geoprop_clean | geoprop | 0cm | 0.0000 | 559.8126 | 0.0000 | 0.0000 |
| reach_preact10_oracle_clean | oracle | 0cm | 0.0000 | 561.7698 | 0.0000 | 0.0000 |
| reach_preact10_fcpg_clean | fcpg | 0cm | 0.0000 | 584.4045 | 0.0000 | 1.3739 |
| reach_preact10_fcpg_cleanw10_clean | fcpg | 0cm | 0.2000 | 592.9354 | 0.0000 | 0.8688 |
| reach_preact10_state_trans4 | state | 4cm x+ | 0.0000 | 653.3738 | 3.6992 | 3.6992 |
| reach_preact10_geoprop_trans4 | geoprop | 4cm x+ | 0.0000 | 590.9687 | 3.6963 | 3.6963 |
| reach_preact10_oracle_trans4 | oracle | 4cm x+ | 0.2000 | 643.7717 | 3.6929 | 0.0000 |
| reach_preact10_fcpg_trans4 | fcpg | 4cm x+ | 0.0000 | 665.9442 | 3.6838 | 1.2134 |
| reach_preact10_fcpg_cleanw10_trans4 | fcpg | 4cm x+ | 0.0000 | 578.4313 | 3.6839 | 1.2382 |

Detailed table:

```text
outputs/bc_rollout/reach_preact10_with_cleanw10_rollout_summary.md
```

Interpretation: this is still not a positive policy result. The minimal BC backbone has very low success, so success rate is too weak to support a robustness claim. The useful diagnostic is geometric: the original FCPG residual reduces 4cm nominal projection error from about 3.68 px to 1.21 px, but over-corrects clean 0cm inputs to 1.37 px. Adding a clean zero-residual weight of 10 lowers clean over-correction to 0.87 px while keeping 4cm correction around 1.24 px. Next policy step is not to claim robustness yet, but to improve the policy backbone/data scale and keep the identity-aware residual as the safer FCPG default.

已完成 LeWM-style ViT adapter rollout smoke（5 episodes, horizon 60）：

| run | model | perturbation | success | reward mean | nominal px err | corrected px err |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| state_vit clean | state | 0cm | 0.0000 | 309.6771 | 0.0000 | 0.0000 |
| state_vit trans=4cm | state | 4cm x+ | 0.0000 | 351.8741 | 3.7524 | 3.7524 |
| geoprop_vit clean | geoprop | 0cm | 0.2000 | 382.6118 | 0.0000 | 0.0000 |
| geoprop_vit trans=4cm | geoprop | 4cm x+ | 0.0000 | 255.3894 | 3.7486 | 3.7486 |
| oracle_vit trans=4cm | oracle | 4cm x+ | 0.0000 | 303.8712 | 3.7515 | 0.0000 |
| fcpg_vit clean | fcpg | 0cm | 0.0000 | 340.5825 | 0.0000 | 0.6732 |
| fcpg_vit trans=4cm | fcpg | 4cm x+ | 0.4000 | 352.5224 | 3.7492 | 1.3519 |

Detailed table:

```text
outputs/bc_rollout/reach_preact10_vit_sig1e4_t192_l6_rollout_summary.md
```

Interpretation: ViT rollout 仍是 smoke，5 episodes 方差太大，不能把 success 当论文结论。稳定信号仍然是几何指标：GeoProp 在 4cm 下 corrected pixel error 仍约 3.75px，FCPG-ViT 降到约 1.35px；同时 clean 下 FCPG 仍有约 0.67px 过校正。当时计划是扩大 pre-action demos 并比较 `ViT` 与 `ViT+SIGReg`；后续正式 P0/P1 已取代该 smoke 计划。

已完成四任务 multi-task data variety re-run：

```text
data/metaworld_projection/scripted_4task_varyidx_clean_10eps.npz
samples: 4000
tasks: reach-v3, pick-place-v3, push-v3, button-press-v3
task counts: 1000 each
task_index: 0-9, 400 samples each
action_source: scripted
observation_action_alignment: pre_action
success_mean: 0.4690
```

该数据由 `--vary-task-index` 采集，避免每个任务只固定在 `task_index=0`。这比继续放大单任务模型更重要，因为单 reach/1000 帧已经证明容量不是主要瓶颈。

多任务 ViT+SIGReg BC 结果：

| run | model | best epoch | best MSE | best MAE | final MSE | final MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt4_varyidx_clean10_state_vit_sig1e4_t192_l6_b128 | state | 73 | 0.005135 | 0.037190 | 0.009907 | 0.041998 |
| mt4_varyidx_clean10_geoprop_vit_sig1e4_t192_l6_b128 | geoprop | 79 | 0.004152 | 0.034000 | 0.014953 | 0.053578 |
| mt4_varyidx_clean10_fcpg_large_h512_vit_sig1e4_t192_l6_b128 | fcpg | 69 | 0.008167 | 0.047012 | 0.011772 | 0.044683 |
| mt4_varyidx_clean10_fcpg_cleanw50_vit_sig1e4_t192_l6_b128 | fcpg | 58 | 0.009179 | 0.041598 | 0.013743 | 0.044359 |
| mt4_varyidx_clean10_fcpg_mtres_cleanw20_vit_sig1e4_t192_l6_b128 | fcpg | 80 | 0.005369 | 0.032733 | 0.005369 | 0.032733 |

Per-task offline MSE：

| model | overall | button-press | pick-place | push | reach |
| --- | ---: | ---: | ---: | ---: | ---: |
| state | 0.005398 | 0.007154 | 0.006115 | 0.006755 | 0.001583 |
| geoprop | 0.004312 | 0.005620 | 0.004860 | 0.005134 | 0.001662 |
| fcpg_cleanw10 | 0.008617 | 0.008158 | 0.011932 | 0.012286 | 0.002042 |
| fcpg_cleanw50 | 0.009500 | 0.018931 | 0.010509 | 0.008078 | 0.001073 |
| fcpg_mtres_cleanw20 | 0.005279 | 0.006092 | 0.005569 | 0.008063 | 0.001203 |

Detailed tables:

```text
outputs/bc_policy/mt4_varyidx_clean10_vit_summary.md
outputs/bc_policy/mt4_varyidx_clean10_vit_with_mtres_summary.md
outputs/bc_policy/mt4_offline_eval_summary.md
outputs/bc_policy/mt4_offline_eval_with_mtres_summary.md
outputs/bc_policy/mt4_offline_eval/
```

新增多任务 residual 数据：

```text
data/metaworld_projection/scripted_4task_varyidx_perturb_10eps_4pf.npz
samples: 16000
tasks: reach-v3, pick-place-v3, push-v3, button-press-v3
task counts: 4000 each
task_index: 0-9, 1600 samples each
perturbations_per_frame: 4
pixel_error_mean: 1.6307
pixel_error_p90: 3.7601
pixel_error_max: 4.6250
inside_patch_rate: 1.0000
```

在该数据上训练的 residual checkpoint：

```text
local-image: outputs/residual_identity/mt4_scripted_cleanw20_local_image_h512/local-image_residual_best.pt
local-image best E_after / E_before: 0.0799
local-image test E_before: 1.6675 px
local-image test E_after: 0.1333 px
local-image clean E_after: 0.0241 px

no-image: outputs/residual_identity/mt4_scripted_cleanw20_no_image_h512/no-image_residual_best.pt
no-image best E_after / E_before: 0.1023
no-image test E_before: 1.6675 px
no-image test E_after: 0.1705 px
no-image clean E_after: 0.0807 px

clean_loss_weight: 20
```

Residual group diagnostics:

```text
outputs/residual_group_eval/mt4_scripted_cleanw20_compare/task_name_groups.md
outputs/residual_group_eval/mt4_scripted_cleanw20_clean_compare/task_name_groups.md
```

| model | setting | button E_after | pick E_after | push E_after | reach E_after |
| --- | --- | ---: | ---: | ---: | ---: |
| no-image | perturbed | 0.2027 | 0.1667 | 0.1578 | 0.1322 |
| local-image | perturbed | 0.1165 | 0.1156 | 0.1197 | 0.1170 |
| no-image | clean | 0.1044 | 0.1031 | 0.1107 | 0.1433 |
| local-image | clean | 0.0269 | 0.0412 | 0.0417 | 0.0638 |

Interpretation: 四任务数据把问题暴露出来了。继续加大 ViT 或 batch 不是主要矛盾；旧 FCPG 在 clean 多任务上变差，核心原因是 residual adapter 的训练分布太窄，identity/generalization 不足。使用同任务、多 task_index、含扰动的 residual 数据后，FCPG 从旧版 overall MSE 0.008617/0.009500 回到 0.005279，并且 MAE 已略优于 GeoProp 的 clean BC 指标。Residual 诊断还说明两件事：第一，push-v3 的 residual correction 本身并不比其他任务差，所以 push 的 BC 弱点更可能来自动作/接触状态覆盖；第二，no-image 在 random split 上也能学到较强 residual，说明扰动统计本身偏容易。当时建议优先补 push/contact 覆盖和扰动曲线；后续 held-out camera P0/P1 已执行，不能把本段当作当前待办。

### 当前主机制实验：真实相机 SE(3) 随机化

旧 residual 数据只扰动 `q_nominal`，RGB 和 `q_true` 始终来自同一个固定相机。no-image 模型因此可以从机器人 XYZ 直接学习固定的 `XYZ -> q_true` 投影，不能用于证明视觉反馈的必要性。当前采集器新增了真实渲染相机随机化：每个 episode 连续随机移动 MuJoCo 相机，RGB 与 `q_true` 来自移动后的实际相机，而 `q_nominal` 仍来自固定基准标定。

当前数据：

```text
data/metaworld_projection/scripted_4task_actualcam_se3_clean25_20eps.npz
samples: 8000
tasks: reach-v3, pick-place-v3, push-v3, button-press-v3
actual camera poses: 80 episodes, 20 per task
actual translation: continuous 0-4 cm in a random 3D direction
actual rotation: continuous axis-angle in [-5, 5] deg
requested clean episode probability: 0.25
realized clean episode rate: 0.1875
pixel error mean / p90 / max: 4.7337 / 10.0302 / 13.0512 px
inside patch rate: 1.0000
```

训练与验证使用 `--random-split-unit task-episode`。每个任务固定保留 4/20 个完整 episode，保证同一个真实相机位姿不会同时出现在 train/test。

未见真实相机 episode 上的整体结果：

| model | E_before | E_after | E_after / E_before | clean E_after |
| --- | ---: | ---: | ---: | ---: |
| no-image | 5.5679 | 5.5299 | 0.9932 | 0.3417 |
| local-image | 5.5679 | 0.9602 | 0.1724 | 0.1162 |

Test-only per-task 结果：

| model | button-press | pick-place | push | reach |
| --- | ---: | ---: | ---: | ---: |
| no-image ratio | 1.0093 | 0.9655 | 0.9855 | 1.0019 |
| local-image ratio | 0.1064 | 0.2245 | 0.2352 | 0.1636 |

关键产物：

```text
outputs/residual_actualcam/mt4_se3_clean25_no_image_h512/no-image_residual_best.pt
outputs/residual_actualcam/mt4_se3_clean25_local_image_h512/local-image_residual_best.pt
outputs/residual_group_eval/actualcam_se3_unseen_episode/task_name_groups.md
outputs/residual_visualizations/actualcam_se3_unseen_episode/best_grid.png
outputs/residual_visualizations/actualcam_se3_unseen_episode/worst_grid.png
```

Interpretation: 这是当前第一个能支撑 FCPG 核心机制的实验。固定相机捷径被切断后，no-image 在未见相机位姿上基本无法校正，而 local-image 在四个任务上都显著降低投影误差，说明局部视觉证据确实提供了状态向量无法获得的样本级反馈。该结果仍只是 policy-free 机制证据；A 会级证据补全依靠 GeoProp policy 的扰动成功率曲线和第二仿真基准；reliability gating 已判为 No-Go，只保留诊断，不再作为证据补全项。真机证据待设备可用后再补，不属于 FCPG v1 的当前排期。

### 历史单步 BC 策略证据（已由 deterministic action-chunk 主线取代）

策略训练使用同一个真实相机 SE(3) 数据集、相同的 task-episode split 和相同的 LeWM-style ViT+SIGReg 配置。单任务筛选结果：

| task | GeoProp best MSE | FCPG best MSE | Oracle best MSE | 判断 |
| --- | ---: | ---: | ---: | --- |
| reach-v3 | 0.006485 | 0.002227 | 0.003931 | FCPG 离线成立；60-rollout 增益未显著 |
| pick-place-v3 | 0.006692 | 0.003626 | 0.003310 | FCPG 接近 oracle，进入完整 rollout |
| push-v3 | 0.008417 | 0.011488 | 0.005327 | 当前 FCPG 失败，不用于正面结论 |

pick-place 在固定 5 度实际相机旋转、60 个确定性配对 episode 下，使用**同一个 FCPG checkpoint**，仅在测试时替换 grounding point：

| grounding | success | reward | nominal error | corrected error |
| --- | ---: | ---: | ---: | ---: |
| nominal/open-loop | 0.5167 | 205.290 | 10.879 px | 10.879 px |
| FCPG/native | 0.7167 | 301.233 | 10.877 px | 2.425 px |
| oracle | 0.7833 | 340.755 | 10.869 px | 0.000 px |

配对统计：FCPG 相对 nominal 提升 20.00 个百分点，paired bootstrap 95% CI 为 `[+3.33, +36.67]` 个百分点，exact McNemar `p=0.028959`（nominal-only=7，FCPG-only=19）。

这组同权重 override 是当前最重要的策略证据：它排除了“FCPG 与 GeoProp 分别训练导致策略能力不同”的主要混杂变量，说明测试时反馈校正本身带来显著收益。FCPG 仍落后 oracle 6.67 个百分点，与剩余 2.425 px 接地误差一致。

系统级横向比较使用相同 actual-camera augmentation 数据、相同 ViT+SIGReg 配置和相同训练 seed，分别训练 GeoProp、FCPG 和 oracle policy。固定 5 度旋转的 60 个配对 episode 结果为：

| independently trained policy | success | reward | corrected error |
| --- | ---: | ---: | ---: |
| GeoProp + actual-camera augmentation | 0.2833 | 117.590 | 10.887 px |
| FCPG + actual-camera augmentation | 0.7167 | 301.233 | 2.425 px |
| Oracle + actual-camera augmentation | 0.7000 | 285.541 | 0.000 px |

FCPG 相对 GeoProp 提升 43.33 个百分点，paired bootstrap 95% CI `[+28.33, +58.33]`，exact McNemar `p=0.000006`。该结果证明 FCPG 系统显著优于强 augmentation baseline；由于策略权重分别训练，43.33 个百分点不能全部归因于测试时 correction，因果效应仍以 same-policy 的 +20.00 个百分点为准。

关键产物：

```text
outputs/bc_rollout/pick_actualcam20_rot5_samepolicy/paired_60/paired_rollout_summary.md
outputs/bc_rollout/pick_actualcam20_rot5_deterministic/paired_60/paired_rollout_summary.md
outputs/bc_rollout/reach_actualcam20_rot5_deterministic/paired_summary/paired_rollout_summary.md
```

同策略实际相机旋转 severity curve（每档 60 个配对 episode）：

| rotation | nominal success | FCPG success | oracle success | FCPG corrected error | FCPG delta | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.7333 | 0.7500 | 0.7333 | 0.273 px | +0.0167 | 1.000000 |
| 1 deg | 0.6167 | 0.7500 | 0.7667 | 0.801 px | +0.1333 | 0.038574 |
| 2 deg | 0.5500 | 0.7833 | 0.7667 | 0.926 px | +0.2333 | 0.001312 |
| 5 deg | 0.5167 | 0.7167 | 0.7833 | 2.425 px | +0.2000 | 0.028959 |
| 10 deg | 0.5333 | 0.5500 | 0.7667 | 17.910 px | +0.0167 | 1.000000 |
| 20 deg | 0.6000 | 0.5500 | 0.6667 | 45.291 px | -0.0500 | 0.581055 |

0 度下 FCPG 没有显著损伤正常性能；1、2、5 度下收益显著。10 度的名义误差约 21.9 px，已超过当前 16 px local patch 半径，校正和成功率收益同时消失；20 度下 FCPG 反而略差。20 度时 oracle 也只有 66.67%，说明大视角变化同时造成全局视觉分布偏移，不能只靠接地点修正解决。该结果明确限定了 FCPG v1 是 local residual correction，而不是 calibration-free policy。

完整曲线产物：

```text
outputs/bc_rollout/pick_actualcam20_rot_curve_samepolicy/summary/rollout_curve.md
outputs/bc_rollout/pick_actualcam20_rot_curve_samepolicy/summary/rollout_curve.csv
outputs/bc_rollout/pick_actualcam20_rot_curve_samepolicy/summary/rollout_curve.png
```

同策略实际相机平移 severity curve（每档 60 个配对 episode）：

| translation | nominal success | FCPG success | oracle success | FCPG corrected error | FCPG delta | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 cm | 0.7333 | 0.7500 | 0.7333 | 0.273 px | +0.0167 | 1.000000 |
| 1 cm | 0.7833 | 0.7500 | 0.7333 | 0.604 px | -0.0333 | 0.500000 |
| 2 cm | 0.6833 | 0.7500 | 0.7500 | 0.794 px | +0.0667 | 0.343750 |
| 4 cm | 0.5500 | 0.7833 | 0.7167 | 0.923 px | +0.2333 | 0.000519 |

1 cm 只产生约 0.73 px 名义误差，不足以形成稳定策略差异；2 cm 约 1.46 px，收益方向正确但未显著；4 cm 约 2.92 px，FCPG 将其降到 0.92 px，并显著提升 23.33 个百分点。平移结果说明任务性能更直接受像素级接地误差控制，而不是由物理厘米值单独决定。

4 cm 下同一 FCPG policy 使用 native correction 的成功率高于 oracle override，不能解释为 FCPG 优于真值。当前 policy 训练时看到的是 FCPG corrected-token 分布，测试时直接替换成 q_true 可能产生 token distribution shift；需要通过独立 oracle policy、多 grounding 训练或 token normalization 进一步验证。

平移曲线产物：

```text
outputs/bc_rollout/pick_actualcam20_trans_curve_samepolicy/summary/rollout_curve.md
outputs/bc_rollout/pick_actualcam20_trans_curve_samepolicy/summary/rollout_curve.csv
outputs/bc_rollout/pick_actualcam20_trans_curve_samepolicy/summary/rollout_curve.png
```

本段记录 P0 idea gate 之前的历史结论；最新判定见下文 `P0 idea gate（2026-07-12）`。Reliability gating 已判为 No-Go，不再列入 FCPG v1 主线。当前没有真机，真机验证不列为 FCPG v1 的必做项，待具备设备后作为后续证据单独补充。

历史 Stage 4 minimal BC smoke（采集器修正前，仅保留为 plumbing 记录）：

| model | best epoch | best MSE | best MAE |
| --- | ---: | ---: | ---: |
| state | 6 | 0.044749 | 0.137587 |
| geoprop | 6 | 0.043966 | 0.138410 |
| oracle | 6 | 0.043968 | 0.138730 |
| fcpg | 10 | 0.045060 | 0.132331 |

Detailed table:

```text
outputs/bc_policy/smoke_reach_summary.md
```

Interpretation: this only proves the `state`、`GeoProp-lite`、`Oracle`、`FCPG residual-only` action-regression paths could train end-to-end before the data-alignment fix. Because those data may be off-by-one, this table must not be used as policy evidence.

历史 Stage 4 minimal rollout smoke（采集器修正前，仅保留为 rollout plumbing 记录）：

| run | model | perturbation | success | reward mean | pixel error |
| --- | --- | --- | ---: | ---: | ---: |
| smoke_reach_state_clean | state | 0cm | 0.5000 | 272.5701 | 0.0000 |
| smoke_reach_geoprop_clean | geoprop | 0cm | 0.0000 | 231.3082 | 0.0000 |
| smoke_reach_fcpg_clean | fcpg | 0cm | 0.0000 | 171.1030 | 0.0000 |
| smoke_reach_fcpg_trans4 | fcpg | 4cm x+ | 0.0000 | 276.4858 | 3.7815 |

Detailed table:

```text
outputs/bc_rollout/smoke_reach_rollout_summary.md
```

Interpretation: this proves the rollout evaluator can run clean and perturbed projection settings end-to-end. It is not a policy result. Next policy step is to recollect larger `pre_action` scripted datasets, retrain the minimal BC baselines, and then evaluate rollout curves over perturbation severities.

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

Current module status:

```text
scripts/smoke_fcpg_modules.py
feature_sampling_check: PASS
grounded_token_check: PASS
random_shape_check: PASS
checkpoint_check: PASS on local-image_residual_best.pt
```

A 1-epoch smoke training after the module refactor also runs through `scripts/train_residual_predictor.py`, confirming the training entry still works with the reusable module.

Current qualitative visualization status:

```text
outputs/residual_visualizations/translation4_local_image/best_grid.png
outputs/residual_visualizations/translation4_local_image/worst_grid.png
```

For held-out `translation_cm=4`, the best selected corrections reduce examples such as `6.28px -> 1.62px`, `5.89px -> 1.44px`, and `4.46px -> 0.63px`. The worst selected cases still mostly improve, but less strongly, e.g. `4.23px -> 3.87px` and `4.03px -> 3.55px`. These plots are useful paper/debug figures because they show both successful feedback correction and remaining failure modes.

Historical Stage 3 reliability diagnostics（最终判定：No-Go）：

| setting | target | best corr(pred, -corrected error) | high-conf error | low-conf error | judgment |
| --- | --- | ---: | ---: | ---: | --- |
| random split, with residual prediction features | `rho_corr`, sigma=2 | 0.6222 | 0.4677 | 1.2854 | 可学，能按 corrected error 排序 |
| random split, no residual prediction features | `rho_corr`, sigma=2 | 0.5471 | 0.4994 | 1.1757 | 可学，但弱于带预测特征 |
| held-out translation=4cm, with residual prediction features | `rho_corr`, sigma=2 | 0.0641 | 1.5566 | 1.7271 | 未通过 OOD magnitude 验证；最终 reliability gate 为 No-Go |

Target diagnostics for held-out `translation_cm=4` show a clear train/test shift. With `sigma=1`, train `rho_corr` mean is 0.6041 while test mean is 0.2150; with `sigma=2`, train mean is 0.7695 while test mean is 0.4427. This means the current reliability head can learn confidence on an in-distribution random split, but cannot yet extrapolate from smaller perturbations to the largest held-out perturbation. These pre-P0 diagnostics show only in-distribution confidence fitting and failed held-out magnitude validation. Reliability gating was subsequently judged No-Go; `rho` is retained only as a historical diagnostic and is neither policy-ready nor an active work item.


## 最小机制验证

先不接策略，先验证一个问题：

```text
Can local visual evidence correct projected robot keypoints?
```

步骤：

1. 实现 `GeoProp-lite` 投影和可视化。
2. 用真实相机参数得到 `q_true`。
3. 人为扰动外参得到 `q_nominal`。
4. 用 `LocalSample(F, q_nominal)` 训练 `delta_q` predictor。
5. 比较校正前后的投影误差。

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

只有在 held-out 外参扰动下满足上述条件，才继续接入策略网络。

如果无法达到该条件，优先检查：

- 关键点投影和坐标归一化是否正确；
- 局部窗口大小是否覆盖扰动范围；
- 视觉编码器特征是否包含机器人本体边缘或末端信息；
- `delta_q` supervision 是否与图像分辨率和 feature map 坐标系一致。

必须额外报告局部窗口覆盖范围：

```text
nominal projection error in pixels
vs
E_after / E_before
```

并按 `q_true` 是否落在 local patch 内分组。FCPG 不是 calibration-free 方法，它假设名义投影仍在可由局部视觉证据恢复的邻域内。

## 策略实验矩阵与当前结论

当前主线 baseline：

1. `State concat`
2. `GeoProp-lite single-keypoint`
3. `GeoProp-lite multi-keypoint`
4. `GeoProp-lite multi-keypoint + projection jitter augmentation`
5. `FCPG residual only`
6. `Oracle corrected GeoProp`
7. `No-image residual predictor`

定义：

- `FCPG residual only`：修正 `q_nominal`，但不做可靠性门控。
- `Oracle corrected GeoProp`：直接用 `q_true` 采样，不作为公平方法，只用于估计 grounding 上限。
- `No-image residual predictor`：不看局部视觉 patch，只用 `q_nominal`、keypoint id 和 proprio 预测 `delta_q`，用于检查 residual head 是否在记扰动分布。

`reliability-only` 与 `residual+reliability` 已完成并判为 No-Go：learned `rho` 弱于固定 `rho=1`，不再属于当前必做矩阵；代码与结果仅用于 legacy 复现和诊断。

P0 阶段只检验 residual correction：

- 正常场景不明显掉点；
- 标定漂移下退化更慢；
- 优于 `GeoProp-lite multi-keypoint + projection jitter augmentation`；
- local-image residual 明显优于 no-image residual，证明模型确实使用了局部视觉证据。

`rho` 的遮挡响应只作为 reliability 诊断，不再作为 FCPG 核心 idea 的通过条件。

公平性要求：当前主方法 `FCPG residual only` 必须至少和 `GeoProp-lite multi-keypoint` 及其 projection-jitter augmentation 版本对比，不能只和 single-keypoint baseline 对比。最终报告使用 perturbation curve 的 AUC、degradation slope 和 bootstrap confidence interval，而不是只报单个扰动点成功率。

### P0 idea gate（2026-07-12）

本轮只判断反馈残差校正是否成立，不扩任务、不长训 LeWM。强基线在原有 actual-camera SE(3) augmentation 之上，仅在 GeoProp 训练时加入序列级二维投影 jitter：每条序列共享同一个 shift，跨历史帧和关键点保持一致，Gaussian `std=8 px`，径向截断为 `24 px`。验证和 rollout 均关闭 jitter。FCPG residual-only 与 jitter baseline 使用相同数据、task-episode split、ViT layer-11 特征、policy 容量、训练 seed 和优化预算。

最终 rollout 使用新的环境初始化 seed 和相机随机数流，不复用此前 README 中的轨迹。MetaWorld MT1 的 task 配置数量有限，因此这里称为 fresh trajectories，不声称 unseen task configurations。

| task / condition | paired episodes | GeoProp + jitter | FCPG residual-only | paired delta | bootstrap 95% CI | jitter / corrected error | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pick, clean | 60 | 0.5167 | 0.5833 | +0.0667 | [-0.0667, 0.2000] | 0.000 / 0.672 px | 无显著 clean 退化 |
| pick, fixed 5 deg rotation | 180 | 0.0889 | 0.3056 | +0.2167 | [0.1444, 0.2889] | 19.019 / 1.609 px | 通过 |
| button, fixed 5 deg rotation | 60 | 0.8833 | 0.9500 | +0.0667 | [-0.0167, 0.1667] | 19.240 / 1.277 px | 接地成立；成功率受 ceiling 限制 |

pick 5 度下，FCPG residual-only 相对 jitter 的 exact McNemar `p=1.83e-8`。在同一个匹配训练的 residual-only checkpoint 上，仅把 grounding 从 corrected 切回 nominal，60 条配对轨迹的成功率从 `0.3833` 降到 `0.0167`；校正带来 `+0.3667`，bootstrap 95% CI `[0.2500, 0.4833]`，McNemar `p=4.77e-7`。因此收益不能只归因于两个 policy 分开训练。

policy-free 视觉依赖证据也通过：在四任务 unseen actual-camera episode split 上，no-image corrector 的分任务 `E_after / E_before` 为 `0.9655-1.0093`，local-image corrector 为 `0.1064-0.2352`。这排除了 corrector 只记住运动学或扰动分布的主要解释。

本轮结论：

- **FCPG residual correction：Go。** 它已通过视觉依赖、强 jitter baseline、同 checkpoint grounding override 和 clean 保真四个 P0 门槛；这表示 idea 值得进入多 seed、完整扰动曲线和外部 baseline 阶段，不表示论文证据已经完整。
- **Reliability gating：No-Go。** 旧 learned-rho checkpoint 在同 180 条 pick 轨迹上为 `0.2111`，强制 `rho=1` 为 `0.3056`，差值 `+0.0944`、95% CI `[0.0444, 0.1444]`；结合 button 的匹配多 seed 消融，`rho` 从主方法删除，只保留诊断附录。
- **LeWM 与尺度先验：未由本轮验证。** 两者必须走各自独立的 Go/No-Go，不能用于放大 FCPG 的当前结论。

关键产物：

```text
outputs/action_chunk/idea_gate/pick_geoprop_jitter_s8_m24_seed0
outputs/action_chunk/idea_gate/button_geoprop_jitter_s8_m24_seed0
outputs/action_chunk/idea_gate/pick_fcpg_residualonly_seed0
outputs/action_chunk_rollout/idea_gate/pick_fresh_rot5_residualonly_matched_vs_jitter_n180
outputs/action_chunk_rollout/idea_gate/pick_fresh_rot5_residualonly_matched_samepolicy_n60
outputs/action_chunk_rollout/idea_gate/pick_fresh_clean_residualonly_matched_vs_jitter_n60
outputs/action_chunk_rollout/idea_gate/button_fresh_rot5_fcpg_vs_jitter_n60
outputs/residual_group_eval/actualcam_se3_unseen_episode/task_name_groups.md
```

## Related Work 定位

详细论文链接和笔记维护在 [docs/related_work.md](docs/related_work.md)。README 只保留定位。

### Geometric Proprioceptive Grounding

代表工作：GeoProp、GeoAlign、AimBot。

这些工作回答“本体状态如何空间化或视觉化”。FCPG 进一步回答“当前观测应如何校正运动学投影给出的视觉采样位置”。

### Robustness To Visual-Proprioceptive Mismatch

代表工作：ReViP、CamVLA。

这些工作处理视觉-本体不一致或相机变化。FCPG 不改变动作空间，也不走完全免标定路线，而是保留几何投影先验，在名义接地点邻域利用局部视觉证据执行逐样本反馈残差校正。

### Spatial Refinement And Learned Offset Sampling

代表工作：STN、Deformable DETR、keypoint refinement。

FCPG 不声称 learned offset sampling 本身是新贡献。它的 offset 是围绕机器人运动学投影点的 visual-geometric residual，并直接用于校正策略读取视觉特征的采样位置。

## LeWM 实现基础与 novelty 边界

LeWM 提供的是实现基础，不是 FCPG 的独立创新点。项目复用或适配 AdaLN-zero conditional blocks、显式 causal / non-causal attention、连续条件 embedder、action-conditioned latent predictor 和 SIGReg。当前已成立的主方法贡献仍是运动学锚定的逐观测残差校正与策略特征采样位置修复；A3 另确认了共享 Anchor-Persistent transition 对 factual action 的 held-out predictive increment。后者的新意在持续维护固定身份的 grounded anchors 这一接口，不在 LeWM、Transformer 或 SIGReg 本身。旧 matched-grounding Gate L1 的 No-Go 保持有效，不能与 A3 混写。

仓库中现在有三条必须严格区分的 LeWM 路径：现有 `lewm-predictive` 是 **BC policy 的 next-latent 辅助消融**，测试时仍直接输出确定性动作块，不递归 rollout，也不做 MPC；legacy full grounded LeWM 使用每帧 summary 的 next-embedding + SIGReg 目标和递归 latent rollout；Anchor-Persistent v1 则保留 `[B,T,K,D]` 逐锚点状态并使用隔离的 checkpoint 类型。前两条已有的负结果不能冒充 Anchor-Persistent 结果；Anchor-Persistent 的旧 recovery/switching Gate 仍失败，但独立 A3 已在 reserved test 上通过三个共享四任务 `10k` seed，确认 factual-action transition mechanism。历史设计归档见 `docs/lewm_full_integration.md`；该文档已标记 legacy No-Go，禁止把其中命令当作当前执行入口。


### 四条链路的实际完成度

1. **`FCPG -> action policy` 已在线串联。** 每次观测都由名义内外参与 FK 生成 `q_nominal`，corrector 根据当前 RGB 预测 `delta_q`，策略在 `q_corrected` 处读取视觉特征并直接执行动作块；已有 MetaWorld 配对 rollout 证据。
2. **`frozen FCPG cache -> full grounded LeWM` 是已结束的离线 No-Go。** corrector 曾先生成固定的 `q_corrected.npy`，full grounded LeWM 再读取该缓存学习 action-conditioned latent dynamics；两者之间没有联合反向传播，Gate L1 未达到预注册门槛。
3. **`frozen FCPG cache -> Anchor-Persistent LeWM` 已完成 A4b 七任务确认。** 它保留逐锚点与 task-context 状态；一个七任务共享模型的三个 fixed-latest `10k` seed 在 fresh A4b panel 上均利用 factual action 优于 matched no-action 与七种 wrong-action。旧 recovery/switching 合取协议仍失败，因此当前只确认 H6-H8 transition mechanism，不声称已实现安全 posterior update、CEM 或控制收益。
4. **旧 `LeWM planning -> CEM` 已取消，不是待办。** 仓库保留递归 latent prediction 和环境无关的 CEM 数学核心用于历史复现，但所有控制 Gate 均未授权 MetaWorld 在线闭环，不得声称已有 MPC 或规划收益。

Stage I 保留完整名义标定作为几何投影先验，但网络直接读取的是由 `K`、`T_nom` 和三维 FK 关键点产生的二维锚点 `q_nominal`，而不是摊平后的相机矩阵。显式共享 `SE(3)` 外参修正属于已判 No-Go 并停止的历史 Stage II，不接入当前主链；当前三个 MuJoCo site 关键点也不能表述成完整 URDF/DH 多连杆尺度模型。

统一表述：

- 正文称 **residual-only token-interaction corrector** 和 **lightweight predictive action policy**；
- related work 中说明 LeWM 是架构与训练参考，并引用固定的上游版本；
- 不把 Stage I 的 `lewm-predictive` BC auxiliary 称为 world model，因为它不做模型预测控制或测试时环境 rollout；只有独立 full grounded LeWM 路径可按其实际功能称为 world model；
- 不把 learned offset、Transformer、SIGReg、next-latent 或“使用已知机器人尺寸”单独列为 novelty；
- 不写“抛弃 VLA”作为贡献。FCPG 与策略骨干正交，当前只选择更容易公平验证的确定性 action-chunk policy。

## 当前交接状态

以下状态同时记录代码落地和正式实验进度；可引用的数值与结论在后续结果小节中逐项给出。

| 子系统 | 已完成 | 尚未完成 |
| --- | --- | --- |
| 上游与核心块 | `third_party/lewm/` 固定快照、MIT 归属、`fcpg/lewm_core.py` 的 LeWM blocks / AR predictor；schema-v2 checkpoint 自动记录并校验固定上游 commit | 正式结果完成后补充 checkpoint hash 与归档清单 |
| Grounding corrector | residual-only 架构与 schema-v2 evaluator 已接通；pick adapter seed `0/1/2` 的 held-out error 为 `0.668/0.700/0.712 px`；视觉依赖 gate、P0 和三 seed fresh curves 已完成；reliability gating 判为 No-Go | 完整 P1 external-baseline/fresh matrix 主要覆盖 pick；button 只有 `5 deg` 三 policy seed 和 `10/20 deg` seed-0 历史证据，同规格 fresh 多-seed 外部矩阵正在补齐；frozen prototype 在旋转 success AUC 上与 FCPG 基本持平，尚不能声称普遍优于普通局部匹配 |
| Sequence 与 cache | dataset 已提供 episode-safe `history_action`、`next_*`、`next_mask`；pick adapter seed `1/2` 的正式 correction cache 已生成；split 不再为读取样本数解压约 4.5 GB RGB；四任务 shared cache 与 few-shot 子集均已存在 | GeoProp / oracle 的紧凑 cache 需另行生成，不能复用 FCPG corrected-point cache |
| Action policy / predictive ablation | pick pure-residual FCPG 与 matched GeoProp+jitter policy seed `0/1/2` 已完成；rotation `0/5/10 deg`、translation `0/2/4 cm`、same-policy prototype/PnP/oracle 矩阵和跨 seed AUC 均已生成；四任务 next-latent/SIGReg 保留为负消融 | button 的 fresh 多-seed control matrix、第二 policy backbone、matched-trained prototype/flow policy 与真机尚未完成 |
| Full grounded LeWM | Gate L1 的冻结 oracle target、camera-safe split、一套独立于训练的 paired_v2 test trajectories（四相机条件复用同一 physical episode identity）、两组固定 `10,000`-step 训练、16 格 `train grounding x eval grounding`、逐格 provenance 审计与预注册汇总均已完成；正式结果为 No-Go | 按预注册协议停止 L2 反事实分叉和 L3 CEM；若未来作为独立工作重启，必须使用全新数据和新的预注册协议，不能继续在本次测试集调参 |
| Anchor-Persistent LeWM | 隔离 checkpoint 类型、逐锚点 `[B,T,K,D]` dynamics、same-state 八动作分支与参数匹配 no-action/七 shift controls 均已完成；A4/A4b 使用七任务共享模型、三个 fixed-latest `10k` seed 和 fresh test panel，正式判为 Confirmation-Go；P1-native known-drop paired rollout 已完成，动作一致性/回报机制正向但 policy-success Development-No-Go | 旧 recovery/switching 合取协议仍失败；尚未完成部署式 observation trust 或 CEM。A4b 只确认 H6-H8 action-conditioned transition mechanism，P1 rollout 也未提供显著 success 增益，二者都不是完整控制证据 |
| Shared / few-shot | 多源 dataset、source manifest、跨源归一化和按源均衡 sampler 已接通；四任务 shared corrector、严格 LOTO、train2-to-unseen 视觉干预、button `0/10/50/200` episode 曲线及四任务 policy `2 x 2` 均已完成 seed `0` | adapter / policy seed `1/2`、真正 unseen-task policy、下游 few-shot rollout 与分层统计尚未运行；训练内四任务共享不能写成 unseen-task policy 泛化 |
| Stage II / KineParity | 左扰动投影 Jacobian、3 帧 observer history、task-oriented 白化 MAP/TSVD、按 effective rank 自适应的卡方 innovation gate、全臂 constellation、oracle/stress evaluator 与单元测试已存在；Gate A 已在 2 px noise、25% dropout 及更强退化压力测试下通过 | Gate B 已完成 learned、zero/shuffled-feature、matched observer/control prototype 以及同 correspondence 的 KineParity/MAP/6D/PnP-RANSAC；唯一一次 frozen rescue 仍弱于 direct control prototype，Stage II 最终 No-Go，Gate C 取消；Gate A 只证明 oracle geometry |

### 2026-07-26 SO-101 实时抓取链路修复记录（爬行 / 目标丢失 / 抓取排序）

对 212 个历史 round 工件的审计确认：**GRASP/TRANSPORT/RELEASE 代码在本记录之前从未在硬件上运行过**——所有归档 round 的 phase 均为 approach/trajectory，`pending_confirmations` 恒为 0；07-25 "grasp" 会话只有 11 个 APPROACH round（最近距离 ~40px 即停止）。当晚三项串联修复：

1. **爬行根因与修复**：`generate_ik_candidate_prefixes` 把剩余行程平摊到固定 20 步到达窗口，而 runtime 每 chunk 只执行 5 步再重规划 → 每轮仅推进剩余的 25%（闭环仿真实测衰减率 0.742-0.752，永不到达）。修复为 `max_arrival_step` 自适应到达（planner 传 `0.7 × max_prefix_step_jump ≈ 5.25°`，为 2.0° 实时漂移在 7.5° 执行守卫内留余量）；每次确认进展 ~8mm → ~158mm。同批恢复了 codex 未汇报的多处安全削弱（计划年龄门禁、限位外向容差、receding-horizon ik_safe 逃生阀、失速看门狗），细节见 `NOTES_innovation_review.md` 第五节。
2. **近距目标丢失**：修复后首次进入 20.6px 距离（历史最近），柱面模板匹配随即进入双峰塌缩区（margin 0.116 → 0.002 → −0.010），30px 连续性窗口的 argmax 被臂上更强峰拖到窗口边缘（26.9px），与执行期 15px 漂移门形成结构性互斥。修复：连续性支持半径收紧至 12px（`CONTINUITY_SUPPORT_RADIUS_PX`）+ **夹爪遮挡几何保持**（`VisualTargetTracker._hold_result`：匹配不可信且接地夹爪像素在末次可信定位的连续性窗口内时，返回冻结定位；仅限 pickup/column——腕相机随夹爪运动、"夹爪离开"条件不可满足，且腕视保持会使 GRASP→TRANSPORT 附着证据同义反复；sticky 覆盖执行期无夹爪信息的复检；连续保持上限 40 次后 fail-closed）。这是 known-drop 实验中验证过的 geometry-hold 载体在目标追踪上的部署化应用。
3. **抓取排序缺口**：校准脉冲有意以闭合结束（f2c5275），home-state report 的夹爪值为闭合态 1.314，而 APPROACH 期所有候选按构造冻结夹爪、无任何环节张开——**从闭合态出发物理上不可能抓取**（GRASP 闭合目标 `min(11.81, current−2)` 自 1.77 出发还会触发限位保护）。修复：`serve` 加载 home-state 时将夹爪维度提升到 `gripper_calibration_open`（99 分位 ≈31.6），Home 即恢复"任务就绪"开合度。

已知限制（未修，留档）：GRASP→TRANSPORT 附着证据在无腕参考时回退为柱面像素距离 + 位置阈值（11.81 为训练 25 分位），空夹也满足——无力/电流反馈，抓空无法检出；drop_zone 无保持机制，RELEASE 期遮挡问题未解；保持激活期间 15px 漂移门对 held 值惰性（state-drift 2.0°、workspace、bounds、round_id 仍有效）。三锚点（C/左右指尖）预测与指令轨迹每轮已计算（`[K,H,3,2]`）但决策只消费 anchor 0；腕视指尖分离度实测开 110.3px / 空闭 ~10px，是未被利用的强附着信号——模型/轨迹信号进入抓取时机判定的路线与 Gate 约束见 `NOTES_innovation_review.md` 第九节。

### 2026-07-12 KineParity oracle geometry Gate A 完成记录

Gate A 只检验 observer-to-control 几何传递，不训练视觉 correspondence，也不接 policy。control anchors 的视觉对应、真实像素和残差标签均未进入求解；每个条件使用 pick-place `5 episodes x 20 frames x 10 correspondence trials`，默认采用 `H=3` 的全臂 observer 历史、`2 px` 高斯 correspondence noise、`25%` 独立 dropout，以及 false-alarm probability `0.01` 的按秩卡方 innovation gate。

Full-9 constellation 的 held-out control-anchor 结果：

| 条件 | nominal mean | KineParity mean / P95 / max | error ratio | MAP-all mean / P95 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.000 px | 0.027 / 0.000 / 6.023 px | - | 1.308 / 2.682 px |
| rotation 5 deg | 20.391 px | 1.137 / 2.299 / 5.029 px | 0.0558 | 1.349 / 2.686 px |
| rotation 10 deg | 40.862 px | 1.871 / 3.419 / 7.627 px | 0.0458 | 1.961 / 3.831 px |
| translation 4 cm | 6.249 px | 1.425 / 2.751 / 5.806 px | 0.2280 | 1.328 / 2.603 px |

Clean 下 KineParity 只有约 `1.1%` 的窗口触发非零更新，mean displacement 为 `0.027 px`；不做 innovation rejection 的 MAP-all 每个窗口都更新，并引入 `1.308 px` mean error。旋转 `5/10 deg` 下 KineParity 平均保留约 `4` 个 task-relevant 模态，而 MAP-all 几乎总是使用全部 `6` 个模态。`4 cm` 下 MAP-all 的 mean/P95 略优于 KineParity，因此现有证据只支持 KineParity 在 clean 抑制和旋转尾部上更稳，不支持它在每种扰动上都优于无 rejection 的 MAP。

额外 severe stress 使用 `4 px` noise、`50%` dropout 和 `10 deg` 旋转。Full-9 下 KineParity 为 `3.184 / 6.461 / 42.283 px` mean/P95/max，iterative PnP 为 `4.753 / 8.944 / 47.628 px`，无约束 linear-6D 的 max 达 `994.713 px`。在 distal-3 病态星座下，KineParity 为 `5.009 / 11.736 / 47.675 px` 且无 `>64 px` catastrophe；linear-6D 的 P95 为 `121.610 px`、catastrophic fraction 为 `17.57%`，PnP 为 `2.67%`。这批结果通过了预注册的 oracle geometry Gate A，并支持“拒绝病态共享更新”这一机制；它仍不能证明 learned visual observer、策略特征重采样或控制收益。

正式 Gate A 产物：

```text
outputs/kineparity/gate_a/pick_rotation_chi2_history3_full9_final_seed20260719_e5s20_t10.json
outputs/kineparity/gate_a/pick_translation_chi2_history3_full9_final_seed20260720_e5s20_t10.json
outputs/kineparity/gate_a/pick_rotation10_chi2_history3_severe_all_seed20260721_e5s20_t10.json
```

本轮代码验证：完整测试 `137 passed, 13 warnings`；全部 `fcpg/*.py` 与 `scripts/*.py` 通过 `py_compile`，`git diff --check` 通过。

因此，FCPG residual correction 的 pick 单任务 P1 证据包已经完成：adapter、matched policy、fresh rotation/translation curves、外部局部匹配、在线 PnP、oracle 和跨 seed 聚合均有可复现产物。它证明局部反馈残差在中等扰动下有效，但没有证明跨任务通用性，也没有证明在所有旋转范围内优于冻结局部 descriptor matching。shared policy 只证明四个已见训练任务上的联合拟合与控制筛选，不是 unseen-task policy 泛化。full grounded LeWM 的正式 L1 已产生正向但未过门槛的离线结果，现已停止，不构成当前独立候选贡献。

### 2026-07-12 P1 完成记录

本轮只收口 pick-place 的 P1 证据。三份 residual-only FCPG adapter、三份匹配 FCPG policy 和三份 GeoProp+jitter policy 使用相同 split、架构、训练 seed 与优化预算；正式 rollout 使用全新的 task index、environment seed 和 camera RNG。每个训练 seed、每个扰动强度运行 20 条 episode，先在 episode 内配对，再以训练 seed 为统计单位报告均值与样本标准差。正式 FCPG 推理固定 `rho=1`。

#### Matched policy：FCPG residual-only vs GeoProp+jitter

Rotation fresh curves：

| rotation | GeoProp+jitter success | FCPG success | same-seed delta | jitter / corrected error |
| ---: | ---: | ---: | ---: | ---: |
| 0 deg | 0.4667 +/- 0.0577 | 0.5500 +/- 0.0866 | +0.0833 +/- 0.0764 | 0.000 / 0.414 px |
| 5 deg | 0.1500 +/- 0.0500 | 0.2833 +/- 0.0577 | +0.1333 +/- 0.0289 | 19.123 / 2.092 px |
| 10 deg | 0.1167 +/- 0.0577 | 0.0667 +/- 0.0289 | -0.0500 +/- 0.0500 | 38.399 / 17.578 px |

Normalized success AUC 为 `0.2208 +/- 0.0382` 对 `0.2958 +/- 0.0473`，FCPG 提升 `+0.0750`。10 度是明确负结果：虽然接地误差仍下降约 54%，成功率没有获益，说明当前 local residual 已超出稳定捕获范围。

Translation fresh curves：

| translation | GeoProp+jitter success | FCPG success | same-seed delta | jitter / corrected error |
| ---: | ---: | ---: | ---: | ---: |
| 0 cm | 0.4667 +/- 0.0289 | 0.5833 +/- 0.1041 | +0.1167 +/- 0.0764 | 0.000 / 0.414 px |
| 2 cm | 0.4000 +/- 0.0500 | 0.4667 +/- 0.1258 | +0.0667 +/- 0.1528 | 2.530 / 0.665 px |
| 4 cm | 0.2333 +/- 0.1155 | 0.3333 +/- 0.0764 | +0.1000 +/- 0.0500 | 5.061 / 1.025 px |

Normalized success AUC 为 `0.3750 +/- 0.0331` 对 `0.4625 +/- 0.0750`，FCPG 提升 `+0.0875`。

#### Same-checkpoint grounding intervention

这一矩阵冻结同一 FCPG policy checkpoint，只替换 grounding source，因而用于隔离接地机制，不冒充 matched-trained policy 横向 SOTA。`prototype` 使用训练 split 构建的冻结 ViT-B/16 第 11 层 descriptor bank；`pnp` 使用这些局部视觉对应和已知三维机器人点做短窗在线共享外参更新，不读取 `q_true`，并采用 `camera_center_v1` 姿态先验与病态解拒绝；`oracle` 直接使用 `q_true`，只表示上限。

Rotation：

| method | 5 deg success / error | 10 deg success / error | normalized AUC |
| --- | ---: | ---: | ---: |
| nominal | 0.0167 / 19.141 px | 0.0167 / 38.504 px | 0.1542 +/- 0.0402 |
| FCPG native | 0.3000 / 2.101 px | 0.0667 / 17.573 px | 0.3042 +/- 0.0402 |
| frozen prototype | 0.2667 / 5.508 px | 0.0667 / 9.232 px | 0.3083 +/- 0.0753 |
| robust online PnP | 0.0333 / 9.168 px | 0.0667 / 15.198 px | 0.1708 +/- 0.0260 |
| oracle | 0.3833 / 0.000 px | 0.2000 / 0.000 px | 0.3917 +/- 0.0971 |

Translation：

| method | 2 cm success / error | 4 cm success / error | normalized AUC |
| --- | ---: | ---: | ---: |
| nominal | 0.3667 / 2.531 px | 0.2667 / 5.062 px | 0.3917 +/- 0.0722 |
| FCPG native | 0.4833 / 0.673 px | 0.3333 / 1.018 px | 0.4708 +/- 0.0764 |
| frozen prototype | 0.3833 / 5.079 px | 0.3333 / 5.808 px | 0.4250 +/- 0.0976 |
| robust online PnP | 0.4167 / 3.094 px | 0.3000 / 4.457 px | 0.4250 +/- 0.0944 |
| oracle | 0.4833 / 0.000 px | 0.3333 / 0.000 px | 0.4708 +/- 0.0520 |

严格结论：

- **Residual correction：P1 Go。** Rotation 5 deg 下三个训练 seed 均优于 matched GeoProp+jitter；translation 的跨 seed AUC 与 4 cm 结果也一致更高。2 cm 下有一个 seed 反向且方差较大，因此现有证据支持“中等扰动下平均更鲁棒”，不支持“每个扰动、每个 seed 都占优”。各扰动下 FCPG 均显著降低接地误差。
- **“学习式 FCPG 普遍优于普通局部匹配”：尚未成立。** Frozen prototype 的 rotation success AUC 与 FCPG 基本持平，并在 10 deg 得到更低像素误差；FCPG 的明确优势集中在中等旋转误差和 translation。下一篇论文必须在更精细任务、matched-trained baseline 或全新预注册并独立验证的 global 方法 上拉开差异，不能隐藏该结果。
- **三末端点在线 PnP：稳定但受退化限制。** 加入相机中心先验后不再出现病态爆炸，但 5 deg 下仍明显弱于 FCPG；该结果支持增加全臂 observer constellation，但不能把 PnP、TSVD 或星座选择本身包装成 novelty。
- **Oracle 证明 10 deg 仍有 grounding headroom。** Rotation 10 deg 的 oracle success 为 `0.2000`，高于 FCPG 的 `0.0667`，因此失败不完全是 policy 上限；当前 local capture range 是主要瓶颈之一。
- Reliability gating 继续保持 No-Go；LeWM 继续独立，不进入这些 P1 数字。

正式产物：

```text
outputs/action_chunk_rollout/p1/pick_seed012_rot_0_5_10_fresh20/
outputs/action_chunk_rollout/p1/pick_seed012_trans_0_2_4_fresh20/
outputs/action_chunk_rollout/p1/pick_seed012_external_rot_0_5_10_fresh20/
outputs/action_chunk_rollout/p1/pick_seed012_external_trans_0_2_4_fresh20/
```

每个目录包含 `matrix_manifest.json`、episode-level paired summaries 和跨 seed `seed_aggregate/`；外部矩阵另含 `seed_aggregate_vs_prototype/`，用于直接审计 FCPG 与冻结局部匹配的差异。

#### 2026-07-15 action-output mechanism 与 wide capture-range 确认

Action-output 诊断冻结每个 residual-only policy checkpoint，在完全相同的 snapshot 上只切换 `nominal / FCPG / oracle` grounding，并比较前 `H=1/8` 步动作与 oracle-grounded policy 输出的距离。正式矩阵覆盖 `pick-place-v3 / button-press-v3`、三个 policy seed、每任务每 seed `60` 个完整 episode；每份结果为 `6,000` 帧，三锚点共同支持率为 `1.0`，越界数为 `0`。统计将 policy seed 与跨 seed 共享的 physical episode identity 作为两个交叉因子独立重采样，并对任务等权。

在非零相机扰动帧上，FCPG 将 EPE 从 `9.024843` 降到 `0.815753 px`。同一策略输出到 oracle-grounded 输出的 XYZ 距离，`H=1` 从 `0.133855` 降到 `0.012827`，改善 `0.121028 [0.096136, 0.148757]`；`H=8` 从 `0.128709` 降到 `0.013028`，改善 `0.115681 [0.093167, 0.139608]`。六个 task-seed cell 的方向均一致。Clean 时 nominal 就等于 `q_true`，FCPG 额外引入 `0.243789 px` 校正噪声，对应 `H=1/8` 的 oracle-output 距离为 `0.003231/0.003161`。

这批 action 结果来自各 `best.pt` 曾用于 checkpoint 选择的原始 `20%` task-episode validation split，因此只能支持：**FCPG 的像素校正会在相机扰动下沿 oracle-grounded policy 的方向改变动作输出。** 它是 validation-set mechanism evidence，不是 fresh independent confirmation，不等同于 rollout success、轨迹性能、LeWM 或 CEM 收益。

wide frozen prototype 的独立确认使用三个 pick pipeline、每个 severity 每 seed `60` 条 fresh paired episode。相对 native FCPG，clean 为 `-2.22 pp [-10.00,+6.11]`，5 deg 为 `-3.89 pp [-12.22,+3.33]`，10 deg 为 `+8.89 pp [+1.67,+16.67]`；10 deg 三个 seed 均非负，但 prototype 绝对成功率只有 `12.22%`。因此 10 deg 的捕获恢复具有统计信号，合取 Gate 仍因绝对成功率低于 `15%` 且 clean/5 deg 非劣下界失败而判为 **No-Go**。wide search 保留为强 baseline 与捕获范围诊断，不能作为 KineParity、FCPG 扩展或独立创新。

正式汇总产物：

```text
outputs/action_grounding_sensitivity/seed012_summary.{json,md}
outputs/action_chunk_rollout/p1/pick_seed012_external_wide_r64g17_rot_0_5_10_confirm60_seed20260817/paired_seeds_prototype_vs_native/
```

#### 2026-07-15 joint prototype control-headroom 两任务 post-hoc development

该训练自由诊断在 `pick-place-v3` 和 `button-press-v3` 上固定同一 frozen prototype 候选协议：`radius=64 px`、`grid=17`、每 anchor shortlist `L=16`、joint hypotheses `M=16`、`NMS=0`，并使用固定 `10 deg` 相机旋转。Pick 覆盖 `30 episodes x 50 frames x 3 frozen policy seeds`，Button 覆盖 `30 episodes x 100 frames x 3 frozen policy seeds`，分别形成 `4,500/9,000` 个 policy-frame structural pairs。每个 snapshot 的历史接地统一固定，只干预当前帧候选；三个 policy 在同一 physical frame 和同一候选集合上比较。主指标是当前 grounding 对应的策略动作到同 checkpoint `q_true`-grounded oracle action 的 checkpoint-normalized XYZ MSE（`H=8`，越低越好）。`q_true`、`valid_true` 和 oracle action 仅用于离线 teacher、capture 标签、候选上限选择与评分，不进入 prototype 候选生成、deployable policy batch 或 selector 输入。

| Task | capture coverage (95% CI) | native FCPG H8 | wide top-1 H8 | best-of-three H8 | geometry oracle H8 | joint control oracle H8 | capture-constrained oracle H8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pick | 94.93% [93.27%, 96.47%] | 0.064144 | 0.018472 | 0.010902 | 0.010282 | **0.003604** | **0.004873** |
| Button | 94.97% [92.50%, 96.83%] | 0.004574 | 0.001163 | 0.000389 | 0.000624 | **0.000156** | **0.000179** |

Pick 上，wide top-1 和 geometry oracle 的 EPE 分别为 `5.2291/3.6438 px`；joint control oracle 相对 best-of-three 的配对 H8 相对降幅为 `66.94% [47.53%, 79.42%]`，capture-constrained oracle（所有 anchor 有效且 max-anchor error `<=8 px`）的降幅为 `56.22% [31.11%, 71.01%]`，三个 policy seed 方向均一致。Button 上，wide top-1 和 geometry oracle 的 EPE 为 `5.0591/3.5327 px`；joint oracle 相对 best-of-three 的绝对 H8 delta 为 `-0.0002326 [-0.0004123, -0.0000166]`，相对降幅 `59.84% [2.88%, 89.13%]`，同样在三个 policy seed 上同向；但 capture-constrained 绝对 delta 为 `-0.0001655 [-0.0003164, +0.0000030]`，95% CI 略跨零。因此，**两任务复现的是无约束 oracle action-output matching headroom；Button 上受 `8 px` capture 约束的复现尚未达到统计确认。** Geometry oracle 也未稳定等价于 control oracle，所以更低 EPE 不能替代 policy-internal action 指标。

**Composite G0 判定仍为 partial / incomplete。** 这些数字证明固定 wide visual candidate set 在两个任务中都存在离线候选上限，但候选由 oracle action 选择，不是可部署 selector。该 development 结果不是 LeWM、structured dynamics、posterior writeback、feature re-indexing、rollout success 或轨迹控制证据，不能作为正式第二贡献。

最终 JSON：

```text
outputs/prototype_control_candidates/pick_rot10_seed012_r64g17_l16m16_development.json
outputs/prototype_control_candidates/button_rot10_seed012_r64g17_l16m16_development.json
sha256: pick=7e92d079eccc1fb6d0c892b50b570cf8c9e70d907299b1fef4fd9f0f084d844d
        button=a72884f98012e6c0f0c1b07da7d6003d6c576f538739363d03bce10886f57f8c
```

#### Dynamic-camera v2 数据基础设施（已实现，尚无方法结果）

`scripts/collect_metaworld_dynamic_camera_dataset.py` 与 `fcpg/mujoco_camera_trajectory.py` 现支持 schema `fcpg_metaworld_dynamic_camera_v2`：每 episode 独立采样相机扰动 onset，支持阶跃移位、慢速线性漂移、4--6 帧的局部 anchor dropout 及恢复段。Paired clean 观测由同一 pre-action physical state 重渲染，与动态观测共享同一 pre-action 物理状态和待执行动作，每个时刻只执行一次物理动作。Collector 可选择直接 withholding 对应 measurement，或对 dynamic/paired-clean 图像施加匹配的语义局部遮挡，并保留未遮挡的 raw render 供审计。

`q_true` 在 visual-dropout 模式中只用于离线定位遮挡区域，不是模型输入；deployment measurement mask 只由 nominal support 与预定 dropout schedule 构成。当前完成的是数据采集、配对和遮挡/恢复合同，**尚未用该数据证明 selector、LeWM prior、posterior writeback、动态恢复或 rollout 收益。**

#### G1A matched-history/native-label selector v3（post-hoc development）

最终 v3 缓存使用 Pick `10 deg` development 数据和固定 `M=16` 候选，共 `1,050` 个 snapshot；按完整 physical episode 划分为 `18 train / 6 validation / 6 test`，同 episode 的 frame 不跨 split。主历史由 canonical frozen native-FCPG corrector（SHA-256 `3cfe621bbfb5b87962a3c569d6ebf9d4d702e0483adeebd45a4e00bb0a16ee79`）在 `t-15,t-10,t-5` 构造，再由 oracle-free `10k` LeWM（SHA-256 `49f252eb526a61aad0bf29b9b3400b1b2bbfa2b7bb83e35164d716f9315435a9`）使用截至 `t-1` 的已执行动作预测当前 summary prior。与旧缓存不同，candidate action-output label 的 policy history 也固定为 canonical native-FCPG `t-1`，与主历史一致；wide top-1 和 nominal 只保留为 deployment-time 可计算的诊断历史。

`deployment_inputs.npz` 仅包含候选 visual/geometry/anchor token、历史 grounding token、已执行动作和预测 prior；`offline_labels.npz` 仅保存三个 policy 的 candidate action-output MSE 与 shuffle controls。`q_true` 和 oracle action 只在构建期瞬时生成 teacher label，不持久化为 selector 输入。Shuffled action 来源限定为同一 split、同一 task/frame、不同 physical episode，并同步重算对应 prior，避免跨 split 泄漏。最终缓存为：

```text
outputs/g1a_selector_cache/pick_rot10_r64g17_l16m16_online10k_matched_history_native_labels_splitlocal_v3_development/
deployment_inputs.npz  sha256=07631a979eae1093b2abfcc7bf4424a5711a07fde46372210218a4196852961c
offline_labels.npz     sha256=f5cc9982d4220dd2fac3db2a3e901f650521df08fbb5b630152d62e54ac49977
metadata.json          sha256=2ff688db476c876031c69d4529d9fc854a606c77aeff8af25c5b1ce5fe06ebf4
```

三个小型 selector 共分配 `15,377` 个参数，但实际激活的 module-path 参数分别为 structured full-history `12,289`、structured current-only `8,865` 和 summary-predictive `10,113`；因此它们是受控小模型消融，不是严格 active-capacity matching。Seed 0 的 Pick development-test 结果（Spearman 越高越好，normalized top-1 regret 越低越好）为：

| Mode | Spearman | normalized top-1 regret |
| --- | ---: | ---: |
| frozen visual score | 0.169708 | 0.242007 |
| structured full-history | 0.560149 | 0.202956 |
| structured current-only | **0.577465** | **0.187723** |
| summary-predictive | 0.382390 | 0.250940 |
| full-history with split-local shuffled action + prior | 0.560822 | 0.197247 |

Development 区间以 physical episode 为 cluster，每次重采样保留该 episode 的所有 frame 和 policy-label row，执行 `10,000` 次 percentile bootstrap（`n=6` physical episodes）。符号统一为正值支持第一个 mode：

- full-history 对 current-only：Spearman delta `-0.017316 [-0.026195,-0.009510]`，regret improvement `-0.015233 [-0.029357,+0.000119]`；full-history 的 Spearman 明确更差，regret 也没有正向确认。
- full-history 对 summary-predictive：Spearman delta `+0.177759 [+0.120205,+0.232059]`，regret improvement `+0.047983 [+0.014747,+0.078081]`。
- current-only 对 frozen visual：Spearman delta `+0.407757 [+0.354320,+0.482929]`，regret improvement `+0.054285 [-0.020206,+0.123428]`；只有 Spearman 区间不跨零。
- factual full-history 对 split-local combined shuffled action/prior：Spearman delta `-0.000672 [-0.002250,+0.000504]`，regret improvement `-0.005709 [-0.016749,+0.003574]`，两项均无可分离的事实动作增量。

同一 physical split 上的优化 seed 1/2 也得到 full-minus-current Spearman delta `-0.017843/-0.035794`、regret improvement `-0.013283/-0.067613`。它们只检查优化稳定性，不增加独立数据证据。Canonical seed-0 report 为：

```text
outputs/g1a_selector/pick_rot10_matched_history_native_labels_splitlocal_v3_final_seed0/report.json
sha256=e4a52fa01a0e8c773e8d012e7429411362b683abd51e7a033dc44a61ea78de32
```

**G1A 的有界结论是：current structured candidate token 对离线 oracle action-output label 有明确的 development Spearman 选择信号，但 seed-0 regret 改善区间跨零；full-history 在三个优化 seed 上都不优于 current-only，且 factual 与 split-local shuffled action/prior 不可分，因此 legacy summary-pooled `10k` LeWM 接口的 predictive increment 判为 No-Go。** 这些结果仍只来自 Pick 的 `n=6` 个 development physical test episodes，不是独立 confirmatory evidence，也不证明 deployable selector、第二任务泛化、posterior writeback、feature re-indexing、rollout 或 CEM 收益；该 No-Go 不替代 Anchor-Persistent v1 的独立 2k 结果。

本轮最终验证为 `1128 passed, 43 warnings`；变更文件均通过 `py_compile` 与 `git diff --check`。

### 2026-07-12 P0 验证记录

- 新增训练态 coherent projection-jitter baseline：每条序列共享二维 shift，跨历史帧和关键点一致；CLI 为 `--projection-jitter-std-px` 与 `--projection-jitter-max-px`，只允许 `--mode geoprop`，验证和 rollout 自动关闭。
- 完成匹配训练的 pick/button GeoProp+jitter，以及 pick residual-only FCPG policy。pick 5 度共 180 条 fresh 配对轨迹，FCPG 为 `0.3056`、jitter 为 `0.0889`，paired delta `+0.2167`，95% CI `[0.1444, 0.2889]`。
- pick clean 60 条轨迹为 `0.5833` 对 `0.5167`，没有显著 clean 退化；同一 residual-only checkpoint 的 5 度 nominal override 为 `0.0167`，native correction 为 `0.3833`。
- button 5 度把接地误差从 `19.240 px` 降到 `1.277 px`，成功率为 `0.9500` 对 `0.8833`；成功率差异因 ceiling 尚不显著。
- Reliability gating 判为 No-Go，LeWM 与尺度先验均未由本轮验证。全量测试更新为 `106 passed, 13 warnings`；`py_compile` 与 `git diff --check` 通过。

### 2026-07-11 实现验证记录

- exact upstream ViT-Tiny/14 smoke 输出 `feature_map=(1, 192, 16, 16)`、`global=(1, 192)`；真实四任务构建逐源得到 `240/60` train/validation episodes、`20,400/5,100` windows，proprio/action normalization 维度为 `4/20`，并强制匹配 shared corrector split。trainer、resume 与 held-out diagnostics 已验证；CEM 仅有环境无关数学核心的单元测试，未接 MetaWorld 闭环、未进入 Gate L3，也不构成 CEM/规划结果或贡献。
- 全量测试为 `104 passed, 13 warnings`；`py_compile` 与 `git diff --check` 均通过。离线 evaluator 已修正为：保持已观测历史动作不变，只替换历史末端之后的当前 future action，避免依赖 stale history action 的模型获得假 S1。
- residual-only corrector 已在 CUDA0 完成两任务 `SIGReg in {0, 0.02, 0.09}` 的正式 seed-0 匹配训练，并完成 schema-v2 checkpoint 重载和只含 `q_corrected.npy` 的 cache round-trip。
- predictive policy 已在 CUDA0 完成 batch `64` 与 batch `1024` 的单 batch 训练 smoke；batch `1024` 在 32GB V100 上可运行。新 checkpoint 已完成重载、64 条序列的 offline evaluation 和 1 episode / 2 step rollout plumbing 验证。
- shared corrector 的 policy-ready frozen-adapter cache 已生成；每任务从约 9.08 GB 缩小到 61.44 MB。batch `1024/2048/4096` 的实测吞吐分别为 `4273/5102/6360 samples/s`，峰值 allocated CUDA memory 为 `5.53/10.15/19.39 GB`。
- 四任务 shared policy seed `0` 已完成 predictive/action-only × `SIGReg=0/0.02` 的匹配训练、每任务 6000 条 held-out sequence 离线评估和 128 个 `0/5/10/20` 度 nominal/native rollout job。action-only `SIGReg=0` 的 native AUC 在 pick/button/reach/push 均最高（`0.0958/0.7250/0.1833/0.1229`）；大 batch、多任务和 per-source SIGReg 虽提高 rank，仍未转化为动作或控制收益，next-latent 与 SIGReg 因此降为负消融。
- 早期 `SIGReg=0.09` smoke 与 corrector 训练错误地沿时间/关键点维而不是 batch 维估计特征函数，因此原始 SIGReg 被异常放大；相关数值和 checkpoint 均作废。修复后的实现已通过与 vendored LeWM 源码逐值一致的回归测试，正式 `0.02/0.09` 已重新训练。

### SIGReg 的准确含义与后续多任务协议

SIGReg（Sketched Isotropic Gaussian Regularization）来自 [LeJEPA](https://arxiv.org/abs/2511.08544)，并被 [LeWorldModel](https://arxiv.org/abs/2603.19312) 用作 joint-embedding prediction 的防塌缩正则。它不是额外的世界模型目标，也不是“给噪声指定标准差”。对每个时间步的 latent batch `z_t in R^(B x D)`，SIGReg 采样随机单位方向，将高维表示投影到多个一维分布，再用 Epps-Pulley statistic 比较经验特征函数与标准正态特征函数。有限随机投影利用 Cramer-Wold 原理近似约束整体表示接近各向同性标准高斯：

```text
a_m ~ Uniform(S^(D-1))
y_(b,m) = z_b^T a_m
phi_hat_m(omega) = mean_b exp(i * omega * y_(b,m))
phi_N(omega) = exp(-omega^2 / 2)

L = L_prediction + lambda_sig * L_SIGReg
```

当前 vendored LeWM 对齐实现固定为：

- 输入布局为 `[T, B, D]`，对每个时间步沿 batch 维 `B` 估计经验特征函数，不沿时间维或关键点维统计；
- 每次 forward 使用 `1024` 个随机单位投影方向；
- 使用区间 `[0, 3]` 上的 `17` 个积分节点和梯形积分近似 Epps-Pulley statistic；
- `SIGReg=0.02` 表示损失权重 `lambda_sig=0.02`，不是高斯分布的 `sigma=0.02`；目标分布始终是 `N(0, I)`；
- SIGReg 的作用是抑制常量解、低秩解和尺度塌缩。它本身不保证语义充分性、动作敏感性、Markov 性、反事实预测或控制收益，这些必须由独立指标验证。

多任务训练还有一个必须单独检查的风险：pooled SIGReg 可能仅依靠任务间分离满足总体高斯约束，而每个任务内部仍然低秩。因此后续正式新协议按以下顺序执行：

1. 当前冻结的 `pixel_observer_base_online_sig002_seed3072` 保持 `per_source_sigreg=false`，继续跑到预注册的 `latest@step=10000`；禁止中途切换目标、挑选 checkpoint 或追溯性修改 Gate。
2. 在固定端点同时报告 task-macro 与逐任务 prediction/copy ratio、same-task cross-episode action shuffle、effective rank 和 participation ratio，不能只报告 pooled rank。
3. 下一轮新训练使用全新的 development/test seed，在相同初始化、数据、batch、训练步数与优化器预算下，匹配比较 pooled SIGReg 和 per-source SIGReg；per-source 版本在每个 task/source 内分别计算 SIGReg 后取等权平均。
4. 只有 per-source 版本在保持预测和动作敏感性的同时稳定改善逐任务表示与 held-out 指标，才将其设为后续多任务默认值。已有“rank 上升但动作/控制未改善”的负结果必须保留，不能仅凭更高 rank 宣称方法更好。

因此，“后续改为 per-source SIGReg”是一个预注册的新实验假设，不是对当前 10k 运行的即时配置变更，也不是既成正结果。

### 2026-07-11 Full grounded LeWM 机制 pilot

三组 pilot 使用相同四任务数据、task-balanced sampler、seed、初始化、ViT-Tiny/14、batch 128、warmup 和 2,000-step cosine 预算，只改变 pooled SIGReg 权重。它们是独立 smoke-schedule 训练，不等同于正式 100-epoch 训练的前 2,000 步。

每个 best checkpoint 均在相同的 240 个 validation episodes、15,600 个 eligible windows 上完整评估。R1/R3/R5 是递归预测误差相对 hold-last copy 的 task-macro ratio；S1 固定历史 latent 与历史动作，只将当前 future action 替换为同任务不同 episode 的动作。门槛为 `R1<=0.90`、`R3/R5<=0.95`、每任务 ratio 小于 1、`S1>=1.10`、每任务 S1 大于 1，并要求完整覆盖。

| Pooled SIGReg | Best step | R1 | R3 | R5 | S1 | S1 95% CI | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `0` | 1911 | 0.849 | 0.641 | 0.615 | 1.0009 | [1.0006, 1.0012] | **No-Go** |
| `0.02` | 2000 | 0.212 | 0.119 | 0.117 | 3.152 | [2.828, 3.508] | **Go** |
| `0.09` | 1911 | 0.096 | 0.094 | 0.114 | 3.564 | [3.285, 3.862] | **Go** |

`SIGReg=0` 的 validation prediction MSE 曾下降到 `4.57e-5`，但 copy MSE 同时下降到 `5.03e-5`，future-only S1 近似 1，说明共享 online target 通过压缩表示获得了低损失，而没有保留可用的 current-action dynamics。`0.02` 和 `0.09` 都阻止了该塌缩。`0.09` 的宏观 ratio 数值更低、S1 更高，但两组 learned latent 的尺度和几何不同，任务级结果也有胜负，因此该 pilot 不足以宣称 `0.09` 优于 `0.02`。后续 matched grounding 主比较固定 `0.02`；`0.09` 只保留为既有机制结果，不继续扩展实验网格。

R1/R3/R5 只作为旧机制 pilot 的预测诊断。表中 S1 与后续 exploratory post-hoc L2 均生成于 evaluator hardening 前，未按当前 evaluator 重算；它们不能证明 action sensitivity、反事实可控性、CEM、在线控制或 FCPG 相对 nominal grounding 的收益，也不构成正式 Gate 或论文贡献。

### Full grounded LeWM matched grounding gates（已完成：L1 No-Go）

LeWM 原论文已经验证了该架构作为轻量世界模型的能力。本项目不重新声明或证明 LeWM 本身，也不把参数量、Transformer、next-latent 或 SIGReg 作为本文贡献。本节记录曾预注册的三级历史命题；正式完成的仅为 Gate L1，结果为 No-Go。反事实排序 L2 与在线控制 L3 均未启动，以下相关内容只保留为未执行的冻结设计：

> 在相同的已有 LeWM 中，FCPG 校正后的 grounding 是否比 nominal grounding 提供更稳定的动作条件观测，从而改善多步预测、反事实动作排序，并最终改善相机漂移下的模型控制。

此前 pilot 只证明 FCPG-grounded full LeWM 已正确接线并学到 action-sensitive、H=5 不发散的 latent dynamics。正式 matched nominal/FCPG 训练和完整 2 x 2 干预现已完成；固定训练 seed `3072` 的离线矩阵显示 FCPG 方向一致，但未达到 Gate L1 的预注册效应量与逐任务门槛，因此该命题未升级为论文贡献。

#### 固定比较矩阵

| Grounding | Clean | Held-out camera drift |
| --- | --- | --- |
| Nominal | L1 prediction；L2 ranking 未执行 | L1 prediction；L2 ranking 未执行 |
| FCPG | L1 prediction；L2 ranking 未执行 | L1 prediction；L2 ranking 未执行 |

所有条件必须固定：

- 相同 LeWM 架构、初始化 seed、训练 episode、动作、优化器、训练步数和数据预算；
- 主比较固定 `SIGReg=0.02`；`SIGReg=0` 只作为 collapse control，`0.09` 保留既有 pilot，不因结果继续扩展网格；
- 相同 train/validation/test episode manifest，正式统计只使用完全独立的 test trajectories；
- nominal 与 FCPG 只改变 grounded token 的采样位置，不改变 RGB、proprioception、action 或 target；
- LeWM 训练只使用 clean 与 mild camera conditions，并从其训练 manifest 中明确排除正式测试的相机方向与幅度；primary drift 使用处于 FCPG 捕获范围内、但对 LeWM held-out 的 `5 deg / 4 cm`，`10 deg` 只作为已知边界压力测试；
- FCPG corrector 在该比较前冻结，禁止根据 LeWM test camera 调参；必须分别披露该漂移对 LeWM 和 corrector 是否属于训练分布，不把结果写成 calibration-free 能力。

#### 统一预测目标

不得直接比较两个模型各自 latent 空间中的绝对 MSE。所有模型共同预测同一个冻结 oracle target：

```text
z_oracle[t+h] = stopgrad(
    E_target(F[t+h], q_true[t+h])
)
```

当前 `E_target` 不拟合训练数据，也不读取 proprioception：它在 `q_true` 处采样冻结的 ViT-B/16 特征，对每个 token 做固定无仿射归一化，再通过一次性确定性生成并冻结的正交投影映射到 `192D`。模型部署输入仍包含 proprioception；这里只刻意让跨模型比较目标不含模型特有的可训练本体投影。所有条件共享完全相同的 target checkpoint、投影矩阵和 oracle anchors。`q_true` 只用于训练监督与离线评估，不进入部署输入。模型自身 latent 的 effective rank 与 participation ratio 只用于排除 collapse，不用于跨模型替代统一 target 指标。

同时保留两种互补证据：

- **same-checkpoint intervention**：对 nominal-trained 与 FCPG-trained 两个 checkpoint 都执行 nominal/FCPG eval-grounding 切换，形成 `train grounding x eval grounding` 的完整 `2 x 2` 干预矩阵，不只报告有利方向；
- **matched-trained comparison**：比较该 `2 x 2` 矩阵的对角线，评估两种观测表示各自可达到的最佳结果。

#### Gate L1：预测稳定性

先完成 policy-free 的 matched prediction，不接 CEM。报告：

- `H=1/3/5` 相对 hold-last copy 的 normalized prediction ratio；
- `H=10` 仅作为递归外推压力测试；
- clean 到 camera drift 的相对退化幅度；
- effective rank、participation ratio 和 action intervention sensitivity；
- same-checkpoint 与 matched-trained 两套结果。

Gate L1 的预注册 Go 条件：

- 正式训练固定 seed `3072`、batch `192`、`10,000` optimizer steps、warmup `637`、alignment weight `1.0` 和 `SIGReg=0.02`；矩阵只读取各自固定预算终点的 `latest` checkpoint，不按测试结果选择 best checkpoint；
- primary 聚合对 `rotation 5 deg`、`translation 4 cm` 与 `H=1/3/5` 等权，再在 episode 内配对、任务间等权；FCPG 的 task-macro normalized error 平均至少比 nominal 低 `10%`，paired bootstrap 95% CI 排除零增益；
- 四个任务中至少三个方向一致，且每个任务的 learned predictor 均优于 hold-last；
- clean 下 FCPG 相对 nominal 的 task-macro error 退化不超过 `5%`；
- 表示不得通过降秩获得低误差：FCPG 的 overall effective rank 与 participation ratio 均不得低于 nominal 的 `90%`，绝对值分别不得低于 `8` 与 `4`，每任务两项均不得低于 nominal 的 `80%`；
- nominal-trained 与 FCPG-trained checkpoint 的 same-checkpoint eval-grounding 干预都必须支持 FCPG 方向，且 matched-trained 对角线方向一致。

本次 L1 已失败，因此停止额外续训、反事实分叉和 CEM；full LeWM 只保留为离线机制消融。

#### Gate L2：反事实动作排序（未启动）

原协议要求 L1 通过后，才从完全相同的 MetaWorld snapshot 分叉执行候选 action chunks。本次 L1 为 No-Go，所以下述内容只作为未来独立工作的冻结设计：必须保存并恢复完整模拟器状态与随机状态，对每个候选动作记录真实未来 oracle target、task progress 和 return；候选集至少包含行为/专家动作、当前 policy 动作、局部扰动动作和同任务打乱动作。

LeWM 预测的未来 latent 必须通过所有条件共享的固定 goal-distance 或 progress scorer 转换成动作分数；该 scorer 若需要学习，只能使用训练 split，并在 nominal/FCPG 比较前冻结。不能用真实未来 return 作为推理输入，也不能把 shuffled-action sensitivity 当作反事实排序结果。

报告：

- Spearman ranking correlation；
- top-1 accuracy 与 top-k hit rate；
- top-k regret；
- predicted progress/return error；
- nominal/FCPG 在 clean 与 held-out drift 下的配对差值。

Gate L2 的 Go 条件：

- FCPG 在 held-out drift 下的 task-macro Spearman 95% CI 下界大于零，并高于 matched nominal；
- 至少三个任务方向一致；
- top-k regret 相比 nominal 至少降低 `10%`；
- expert/behavior action 对打乱动作的识别率显著高于 chance。

若 L2 失败，只能写“FCPG 改善世界模型的预测稳定性”，不能写反事实动作选择或规划贡献。

#### Gate L3：在线模型控制（未启动）

只有 L2 通过后才接小规模在线 CEM。nominal/FCPG 必须共享 action proposal、scorer、CEM population、迭代次数、执行 horizon、policy checkpoint、episode seed 和计算预算。先做同状态短 horizon pilot，再运行 fresh paired rollout。

只有当 FCPG-grounded CEM 在 held-out camera drift 下同时优于 nominal-grounded CEM 与 action-only policy，并且 clean 非劣时，才允许声明：

> FCPG enables camera-drift-robust model-based control with an existing lightweight LeWM.

否则按最高已通过的 gate 收缩 claim，不用更长训练掩盖失败。

### 2026-07-13 Gate L1 正式结果（No-Go）

Nominal-grounded 与 FCPG-grounded LeWM 均按冻结协议完成 `10,000` optimizer steps：seed `3072`、batch `192`、warmup `637`、alignment weight `1.0`、`SIGReg=0.02`。矩阵只读取固定预算终点的 `latest` checkpoint，没有按测试结果选择 best checkpoint。

| Train grounding | Step / epoch | Checkpoint SHA-256 | Training manifest SHA-256 |
| --- | --- | --- | --- |
| nominal | `10000 / 34` | `87c5023270b7e127a6b2a7ec8a81abeb34460b2c3cc8c7472839e3c82f2c582d` | `cdf34901d59e19e678e08b9d91e3e7f7a26588c5c658f5098ad861c968022efb` |
| FCPG | `10000 / 34` | `729f9f0f0d01c400f3736542f1a8ad5970deba21406541ae2fe8b13404ea306a` | `e5f06bbada637d0c97d261f29987ca850ef6e81c09793039c1f1bdf5a1b3e898` |

正式评估使用独立 `paired_v2` 测试集，覆盖 `clean / rotation 5 deg / translation 4 cm / rotation 10 deg`。四个条件共享同一 physical episode identity hash `22d4737bb17a982834db2df61218ea1b37359427e724634e7bd9e444af4f29b8`；每个条件包含四个任务、每任务 `30` 个 episode、每个 episode `100` 帧。`train grounding x eval grounding` 的 16 个 cell 均完成，逐格 checkpoint、数据、correction cache、oracle target、split、horizon、任务和 episode 集合审计通过。`10 deg` 只作 stress test，不进入 primary gate。

Primary 按预注册方式对 `5 deg / 4 cm`、`H=1/3/5` 等权，在 episode 内配对并对任务等权：

| 指标 | nominal | FCPG | 结果 |
| --- | ---: | ---: | --- |
| Task-macro normalized prediction error | `0.90523` | `0.84541` | 相对改善 `6.61%` |
| Paired bootstrap 95% CI | - | - | 相对改善 `[5.47%, 7.85%]` |
| 正向任务数 | - | - | `4/4` |
| Clean task-macro error | `0.49342` | `0.50011` | FCPG 相对退化 `1.36%` |
| Overall effective rank | `31.26` | `32.01` | 无降秩塌缩 |
| Overall participation ratio | `18.02` | `18.67` | 无降秩塌缩 |

Same-checkpoint 干预也支持 FCPG 方向：nominal-trained checkpoint 从 nominal eval 的 `0.90523` 改为 FCPG eval 的 `0.89349`；FCPG-trained checkpoint 从 nominal eval 的 `0.99540` 改为 FCPG eval 的 `0.84541`。因此在固定训练 seed `3072` 的离线配对矩阵中，FCPG 校正后的观测对已有 LeWM 呈一致正向作用，但效应量没有达到候选第二贡献的预注册门槛。

| 预注册条件 | 判定 |
| --- | --- |
| Primary 相对改善至少 `10%` | **Fail：`6.61%`** |
| Paired bootstrap CI 排除零增益 | Pass |
| 至少 `3/4` 任务同向 | Pass：`4/4` |
| 每个任务均优于 hold-last | **Fail：`reach-v3` 未通过** |
| Clean 相对退化不超过 `5%` | Pass：`1.36%` |
| Overall / per-task representation 不塌缩 | Pass |
| 两个 same-checkpoint 干预与 matched diagonal 同向 | Pass |

**Gate L1：No-Go。** 按预注册协议，L2 same-snapshot 反事实动作排序和 L3 在线 CEM 均不启动，也不通过续训或在本次 test trajectories 上调参改变判定。使用“每个 condition/horizon 内先求 episode MSE 均值再取比值”的敏感性分析得到 `6.51%`，95% CI `[5.39%, 7.72%]`，结论相同，说明 No-Go 不依赖两个 normalized-error estimand 的细微差别。

可审计的固定结果保存在 `results/lewm_gate_l1/gate_l1_seed3072_decision.json`；完整 16 格 JSON 和汇总保留在 ignored 的 `outputs/lewm_grounded/gate_l1_matrix_seed3072/`。Stage I FCPG 仍是当前论文主贡献，full grounded LeWM 只保留为正向但未过门槛的离线机制证据。

### 2026-07-14 Anchor-visual development ablation（No-Go）

本消融不重开或修改已经冻结的 Gate L1。旧结论仍为 **No-Go**：FCPG 相对 nominal 的 primary prediction error 改善 `6.61%`，95% CI 为 `[5.47%, 7.85%]`，但未达到预注册的 `10%`，且 `reach-v3` 未优于 hold-last；正式 L2 same-snapshot ranking 与 L3 在线 CEM 仍未启动。

这是 seed `5120`、每组 `2,000` optimizer steps 的参数匹配、single-seed、post-hoc development 输入消融。`n` 表示在 nominal anchor 采样的视觉特征，`c` 表示在 corrected anchor 采样的视觉特征，四种 canonical 模式为：

- TN（`triplet_nominal`）：`[n, n, n]`；
- TC（`triplet_corrected`）：`[c, c, c]`；
- PV（`paired_visual`）：`[n, c, c]`；
- PI（`paired_innovation`）：`[n, c, c-n]`，其中 `c-n` 是视觉特征差，不是像素坐标残差。

四组都使用 FCPG grounding；TN 只把局部视觉特征改为 nominal 位置，几何输入仍包含 correction-aware 信息，因此不得称为 nominal-grounded LeWM。PV 与 PI 只在 LayerNorm 前具有相同的原始来源信息，二者比较只能评价参数化差异，不能证明 innovation channel 的独立因果作用。四组固定为 `seed=5120`、`batch=192`、`warmup=127`、`SIGReg=0.02`、参数量 `19,152,816`，并只读取各自 `step=2000` 的 `latest` checkpoint。完整 `4 x 4` train-mode × eval-mode 矩阵共 `16/16` 格。

评估使用 `168` 个 episode 的 deterministic legacy replay，不是独立 test。按当前代码重放，`127` 个 episode 与 corrector gradient training 重叠，其余 `41` 个与 corrector validation / best-checkpoint selection 重叠。该 `41`-episode slice 只能称为 corrector-gradient-unseen sensitivity，不能称为 fresh、sealed 或 outer holdout。

冻结主比较是 mild 条件下 matched PI/PI 对 TC/TC：

| 指标 | PI | TC | 判定 |
| --- | ---: | ---: | --- |
| Task-macro normalized error | `0.784342` | `0.775612` | PI 更差 |
| Episode/task-equal relative effect | `-0.88%` | - | 95% CI `[-1.35%, -0.43%]`，Fail |
| 正向任务数 | `1/4` | - | Fail；最差为 `reach-v3`，`-2.31%` |
| Clean PI/TC 单侧 95% ratio 上界 | `1.005999` | protocol margin `1.05` | Pass |
| Corrector-gradient-unseen relative effect | `-0.62%` | - | 95% CI `[-1.90%, 0.49%]`，Fail |
| Overall effective rank | `27.04` | `26.70` | Pass |
| Overall participation ratio | `16.51` | `16.34` | Pass |

三个预先固定的 mild secondary contrasts 中，PV 相对 TC 为 `-0.50%`，PI 相对 PV 为 `-0.37%`，均未通过 Holm 校正；只有 TC 相对 TN 为正向 `+2.73%`，Holm-adjusted `p=0.0003`。这说明 corrected-only anchor visual input 在本 development set 上仍有稳定价值，但 paired visual 或显式 innovation 输入没有带来额外收益。Secondary 结果不能挽救 primary gate。

**Development gate：No-Go。** Correction-aware paired-input 路线到此停止，不扩展模式、seed 或训练预算。该结果不改变旧 Gate L1 No-Go，不恢复 full grounded LeWM 的论文贡献，也不授权 L2 或 CEM。

可审计产物：

- checkpoint ledger：`config/lewm/checkpoints/anchor_visual_ablation_seed5120_latest_step2000.json`，SHA-256 `d90a386725f9734ddf3c927d843cc21f83e3489146157a448e4eaa0cf0a6da8a`；
- development summary：`outputs/lewm_grounded/anchor_visual_ablation_seed5120/validation_matrix/summary.json`，SHA-256 `f9c5e431357ccb3c4dd29fdd69ce190c273492281cda3b290a6e2f92ba8efa43`；
- human-readable summary：同目录 `summary.md`，SHA-256 `f2d395020858344c807bff054f2aa5221dad3dc3dacd079924c7cdf434c412b3`；
- evaluation HEAD：`3e0897685b7aa4e6d8176b1919cb08773e9d9de0`；`summary.completion.json` 已绑定 ledger、16 格 reports、执行代码、输入 artifacts、summary JSON 与 Markdown。

### 2026-07-14 Pixel-Innovation Grounded Observer（历史冻结设计）

> **状态更新（2026-07-15）**：本节保留 v1 的预结果协议与审计合同，不再是当前执行入口。online base 已完成 `10,000` steps；v1 静态相机目标被诊断为近似 hold-last 问题。后续 v2 动态相机审计已完成，结论见下一节。

这是一条独立于旧 Full grounded LeWM Gate L1 和 anchor-visual PI 消融的新开发路线，不重开、覆盖或重新解释任何冻结结论。旧 Gate L1 仍为 **No-Go**：FCPG-grounded LeWM 的 primary latent prediction error 改善为 `6.61%`，没有达到预注册的 `10%`，且 `reach-v3` 未优于 hold-last；旧 PI `[n,c,c-n]` 仍为 **No-Go**。这里的 PI 指像素空间的时序 innovation，不是同一帧 nominal/corrected 视觉特征差 `c-n`。

该路线显式区分 FCPG 的空间测量和 LeWM 的时序先验。对第 `i` 个运动学 anchor，FCPG 测量定义为：

```text
d_measurement[t,i] = q_corrected[t,i] - q_nominal[t,i]
```

冻结的 action-conditioned LeWM 先根据最近 `H_hist=3` 个 posterior embeddings 和对应动作预测下一 latent；小型 residual adapter 再以该 latent、上一 posterior residual、上一名义投影、上一 anchor `XYZ` 和有效位为输入，预测有界像素先验：

```text
z_prior[t+1] = LeWM(z_posterior[t-H_hist+1:t], action[t-H_hist+1:t])
d_prior[t+1] = d_posterior[t] + bounded_adapter(
    z_prior[t+1], d_posterior[t], q_nominal[t], XYZ[t], valid_posterior[t]
)
```

这里的 `valid_posterior` 是名义几何有效位与 posterior 投影在图像内的交集；它是 adapter 的历史状态输入，不等同于决定当前 measurement 是否更新的 FCPG/oracle measurement validity。

adapter 的末层零初始化，因此未训练时严格退化为 `d_prior[t+1] = d_posterior[t]` 的 hold-last residual，而不是任意零投影。当前只实现固定增益 observer：

```text
innovation[t+1] = d_measurement[t+1] - d_prior[t+1]
alpha_effective = alpha                    # measurement valid
alpha_effective = 0                        # measurement invalid
d_posterior[t+1] = d_prior[t+1] + alpha_effective * innovation[t+1]
q_posterior[t+1] = q_nominal[t+1] + d_posterior[t+1]
z_posterior[t+1] = EncodeAndSample(I[t+1], q_posterior[t+1], proprio[t+1])
```

`d_posterior` 和在 `q_posterior` 处重新采样得到的 `z_posterior` 都写回下一步历史，才构成真正的反馈。完成共同 warm-up 后，`alpha=0` 完全保留当前 LeWM adapter prior；只有末层仍为零初始化、尚未训练时，该 prior 才严格等于 hold-last residual。在有效 anchor 上，`alpha=1` 严格退化为当前 FCPG 测量，即 `q_posterior=q_corrected`；无效测量保留 prior。前 `H_hist=3` 帧对所有 alpha 使用同一 warm-up：从零 residual 初始化，有效 FCPG measurement 以 unit gain 更新，无效点保留上一 posterior；完成 warm-up 后从第一个 posterior target 开始，并报告 `k=1/3/5` endpoint，避免不同初始化污染比较。

**Oracle-free 边界。** Adapter 训练目标仅为下一时刻 cached FCPG residual 的 masked MSE：

```text
L_prior = masked_MSE(d_prior[t+1], stopgrad(q_corrected[t+1] - q_nominal[t+1]))
```

Adapter 采用单步 teacher-forced 训练：输入为 `H_hist=3` 帧经过 last-valid warm-up 的 cached FCPG posterior history，只预测下一 residual；target-frame RGB/proprio 不进入 prior，也不对多步 filter 反向传播。多步 posterior 写回只在部署式 filter evaluator 中执行。

训练配置必须使用 `data.target_mode=online`，禁止 `oracle_target_cache`、`q_true`、`q_oracle` 和 `oracle_grounding`；基础 GroundedLeWM 也必须通过 supervision audit，不得由 frozen-oracle target 训练。runtime 会冻结并绑定 base checkpoint，训练期间只更新 residual adapter。`q_true` 是 evaluator-only：primary FCPG/hold-last cells 不接收它；它只用于离线评分，以及单独标记、非性能门槛的 oracle diagnostic measurement。Oracle cell 的 `alpha=1` identity 只作数据完整性的 fail-closed sanity，oracle 性能数值不进入 G0 effect-size gate。

当前 evaluator 的 G0 是 **filter schedule**：每个真实新观测先产生 FCPG measurement，再执行一次 fixed-alpha posterior update。正式 runtime/evaluator 不调用 forecast、action ranking 或 CEM；这些接口即使保留在研究代码中，也不属于 G0 合同、结果或当前执行链。

开发矩阵固定 `alpha in {0, 0.25, 0.5, 0.75, 1}` 和 endpoint `k in {1,3,5}`，比较 LeWM prior、同 alpha 的 hold-last、validity-aware raw-FCPG endpoint（`alpha=1`）与只作诊断的 oracle measurement。当前 `validation` episodes 已用于历史 checkpoint selection 和消融，只能选择一个候选 alpha，development evaluator 按设计永远输出 `Development-Only`，不能给出 Go。

在正式 development matrix 产生前，alpha 选择规则固定如下：`alpha=0/1` 仅作 prior-only / raw-measurement endpoint 诊断，主融合候选只允许 `{0.25,0.5,0.75}`。selector 只读取 `k=5` 的 LeWM+FCPG candidate；先剔除任一任务相对同 alpha hold-last 退化超过 `10%` 的候选，再在剩余候选中最小化 task-macro mean EPE，数值相同时取更小 alpha。选择过程不读取 oracle cell、`k=1/3`、bootstrap 显著性或 future fresh test；若没有合格 interior alpha，直接判该路线 No-Go，不采集 confirmatory test。

`development_selection.status=selected` 还要求：base/observer 分别来自 canonical `10,000/2,000`-step formal config 并完成冻结预算、完整四任务 coverage ledger 精确匹配、Git worktree clean，且 evaluator 没有 `--max-batches` 截断；否则只能写 `ineligible`。confirmatory JSON 必须绑定 development report 的路径与 SHA-256；加载后从报告中的 `macro + paired_comparisons` 重新执行 selector，再核对 selected alpha、observer/base/corrector SHA、`k=5`、阈值和统计常数，不能相信手填的 `selected=true` 或改用另一套 FCPG measurement。选择 alpha 后必须冻结 checkpoint、alpha、estimand 和阈值，并换到经过 non-exposure audit 的 fresh test split；confirmatory G0 只在以下条件同时满足时为 Go：

- `k=5` 上 LeWM+FCPG 相对同 alpha hold-last 的 paired whole-episode、task-macro bootstrap 95% CI 下界大于 `0`；
- `rotation5` 与 `translation4` 分别按 task 等权 bootstrap，两个 cohort 的差值 95% CI 下界都必须大于 `0`；不能用一个 cohort 的大幅改善掩盖另一类相机漂移上的系统性退化；
- 任一任务的相对退化不低于 `-10%`；
- 每个 `camera cohort × task` 的相对改善 bootstrap 95% CI 下界不低于 `-10%`，不能用另一相机扰动或另一任务抵消局部退化；
- 每个 `camera cohort × task` 的 eligible-anchor fraction、contributing-window fraction 和 FCPG measurement coverage 都不低于 `95%`，不能只在少量容易点上产生结果；
- LeWM+FCPG 相对 validity-aware raw FCPG 的 task-macro bootstrap 相对改善 95% CI 下界不低于 `-5%`，避免只凭点估计把“胜 hold-last 但可能整体劣于直接 FCPG”写成 Go；
- oracle 字段隔离、具有非零有效样本的 `alpha=1` identity、输入哈希和 episode 配对完整性全部通过；
- observer 必须是 canonical formal run 的 step-`2000` endpoint，base 必须是 canonical formal run 的 step-`10000` endpoint且显式记录 `online` target；两者的 `run_completed/final_optimizer_steps` 必须与冻结预算完全一致，legacy、smoke、中间或仅伪造 completed metadata 的 checkpoint 一律拒绝；
- test NPZ 的路径和 SHA-256 均不同于 adapter、base LeWM 与 FCPG corrector 的所有训练源；correction cache 必须为 schema 3，并绑定 source/frame identity/corrector/q_corrected 哈希与 `q_true_usage=offline_metrics_only`；evaluator 会用冻结 checkpoint 内的 ViT 和 corrector，从每个计分帧的原始 RGB、`q_nominal`、3D anchor 与本体状态端到端重放。缓存与重放误差必须不超过 `0.02 px`，但科学评分直接使用该次 raw-RGB replay 的 canonical 输出，不使用容差内的缓存值，因此缓存微调不能改变 Go/No-Go；
- confirmatory NPZ 必须与 development alpha-selection source 在路径、文件 SHA-256 和逐 episode `obs/action/keypoints_xyz` 物理内容哈希上全部无交集；任务名、`task_index`、文件名和 episode ID 都不参与物理身份，重标这些字段仍会因底层轨迹内容相同而被拒绝；
- `rotation5/translation4` 必须匹配在 development alpha 选择前就由 canonical development config 绑定 SHA 的 `config/lewm/eval/pixel_observer_confirmatory_design_seed6144.json`。evaluator 逐帧核对 canonical `corner/224` 内参和绝对 base extrinsic、刚体外参、冻结 transform ledger，并从 `camera_intrinsic + keypoints_xyz + true/nominal_extrinsic` 全量重投影核对 `q_true/q_nominal/valid_*`；只改 source 名、scalar metadata、相机矩阵或评分坐标不能通过；
- development report 会冻结 evaluator、统计、相机、correction replay、observer runtime 和数据模块的 protocol-bundle 哈希；confirmatory 必须逐项一致，不能在选完 alpha 或查看 test 后修改 gate 实现；
- source/task/episode/window 数与冻结 coverage ledger 精确一致，每个任务至少 `30` 个完整 episode；confirmatory 禁止通过 CLI 改 alpha、bootstrap、batch size、worker 数、输出路径或截断 batch。

在 fixed-alpha confirmatory G0 通过之前，不启动 learned gain、KalmanNet-style uncertainty、same-state action ranking 或在线 CEM；G0 即使通过也只证明像素 residual filtering，不能直接写成控制收益。Observer/Kalman 更新本身不是本文创新，潜在贡献只有“固定物理身份的运动学 anchor residual + FCPG 当前测量 + 轻量 LeWM 动作条件先验 + posterior 特征重采样”这一组合在 fresh 扰动和下游控制中被证实后才成立。

以下是 2026-07-14、v2 运行前的历史状态，不代表当前完成度：当时仅有代码、聚焦测试、真实 V100 smoke、正式配置和执行链审计，尚无正式 online base、adapter、alpha 选择或 confirmatory 结果。后续 v2 已完成并按下一节判为 No-Go，因此这里冻结的 v1 confirmatory 流程不再执行。

截至 2026-07-14，本轮协议加固后的全仓回归为 `613 passed, 33 warnings`；`py_compile` 与 `git diff --check` 同时通过。这个数字只证明实现与审计回归，不替代任何正式科学结果。

单任务两步 CUDA smoke 只验证了训练、反向传播、run-level 预算字段、checkpoint reload、五个 alpha、oracle 隔离和报告写出能端到端运行；其中 `0.01409` 是 step-1 best checkpoint 在单个 validation batch、`2` 个样本上的 smoke statistic，不是最终 step-2 值，更不属于论文结果。evaluator smoke 也只取单个 batch，即 pick-place 的 `1` 个 episode、`16` 个 windows。adapter 有 `118,738` 个可训练参数；这里尚未给出可引用的正式延迟或显存 benchmark。

v1 adapter trainer 当时采用独占空输出目录并拒绝覆盖，checkpoint 虽保存 optimizer/scheduler，但未实现 resume。以下是已废止的历史执行顺序，仅用于复现实验 provenance，禁止作为当前待执行命令：

```bash
# 必须先提交代码并保持 clean worktree，再冻结 base 全部输入
$FCPG_PY scripts/create_lewm_input_manifest.py \
  --config config/lewm/train/pixel_observer_base_online_sig002_seed3072.yaml \
  --output outputs/lewm_grounded/manifests/pixel_observer_base_online_sig002_seed3072.json

$FCPG_PY scripts/train_lewm_grounded.py \
  --config config/lewm/train/pixel_observer_base_online_sig002_seed3072.yaml \
  --device cuda:0

# 只绑定 step-10000 latest endpoint；best checkpoint 不具有完整 base 预算证书
$FCPG_PY scripts/train_lewm_pixel_observer.py \
  --config config/lewm/train/pixel_observer_adapter_formal_seed6144.yaml \
  --device cuda:0

# adapter 完成后，把实际 base/observer path 与 SHA-256 写入 canonical dev config；
# 提交并 push，保持 clean worktree 后才能运行 development matrix。
git add config/lewm/eval/pixel_observer_fixed_alpha_dev_seed6144.json
git commit -m "freeze pixel observer development checkpoints"
git push origin research/feedback-corrected-grounding

# 仍只在旧 validation 上按冻结规则选 alpha；不是 confirmatory test
$FCPG_PY scripts/evaluate_lewm_pixel_observer.py \
  --config config/lewm/eval/pixel_observer_fixed_alpha_dev_seed6144.json \
  --device cuda:0
```

只有 `development_selection.status=selected` 后才生成 fresh test。收数设计已在 `config/lewm/eval/pixel_observer_confirmatory_design_seed6144.json` 中提前冻结，其路径、SHA、collector 与 camera sampler records 会进入 development report 的 input ledger 并在 confirmatory 阶段逐项复核；其中两个 NPZ 在 alpha 选择前不得生成：

- `rotation5`：`rng_seed=271402`、`task_generation_seed=371402`、`env_seed_base=281400000`，每个 episode 为随机 world axis 上的 signed `5 deg`，实际 translation 为 `0`；
- `translation4`：`rng_seed=271403`、`task_generation_seed=371403`、`env_seed_base=291400000`，每个 episode 为随机 world direction 上的 `4 cm`，实际 rotation 为 `0`；
- 两个 cohort 均为 `corner/224x224`、四任务各 `30` 个独立 episode、每 episode `100` 帧、scripted/pre-action/control anchors、nominal projection 无额外 translation/yaw。每 cohort 各任务产生 `1,950` 个 `k<=5` 窗口；`clean` 与 `rotation10` 若另行采集，只作为 secondary sanity/stress，不改变 Go。

confirmatory evaluator 不相信 source 名称、自报 metadata 或 NPZ 内预存的像素标签。它会绑定收数脚本、camera sampler 和预生成 120-entry transform-ledger SHA，验证每个 `4x4` 外参是合法 SE(3)，从每帧 `base_extrinsic/true_extrinsic` 重算 camera center 与相对旋转，并独立重投影评分坐标；clean 数据改名为 `rotation5`、4.9 deg/3.9 cm、非刚体矩阵、episode 内漂移或 metadata/extrinsic/像素标签不一致都会 fail closed。两个 primary cohort 之间，以及 confirmatory 与 development alpha-selection sources 之间，都必须在路径、文件 SHA-256 和逐 episode `obs/action/keypoints_xyz` 物理内容哈希上无交集；换任务名、文件名、episode ID 或 task index 重打包仍会被拒绝。当前审计绑定收集器、随机种子、完整 NPZ 哈希、相机外参与几何重投影，但未保存 MuJoCo state 做逐帧确定性重渲染，因此不能把它表述为 RGB 像素的独立渲染证明。

新 NPZ、ViT cache、冻结 FCPG corrector 推理产生的 schema-3 correction cache、episode manifest 和 confirmatory JSON 必须共同绑定哈希。旧 `paired_v2` 只可用于管线 rehearsal，永远不能产生本路线的 Go。adapter 内嵌的 base `step/run_kind/training_budget/provenance` 还会与实际 SHA 绑定的 base payload 逐字段对账；仅伪造一份 completed metadata 无法进入 development freeze 或 confirmatory。

主要实现与测试：

- `fcpg/lewm_pixel_observer.py`：fixed-gain update、zero-init residual adapter 与 real-observation filter；forecast utility 不属于 G0；
- `fcpg/lewm_pixel_observer_runtime.py`：oracle-free base audit、adapter-only training、绑定式 checkpoint；
- `fcpg/lewm_pixel_observer_eval.py`：episode/source/task 分层配对统计、alpha identity 与 G0 判定；
- `fcpg/lewm_camera_intervention.py`：预注册 transform ledger、刚体 SE(3)、评分坐标重投影与采集工具哈希审计；
- `fcpg/lewm_correction_derivation.py`：冻结 ViT+corrector 对所有计分帧的 raw-RGB 端到端重放与 cache 归因验证；
- `tests/test_lewm_pixel_observer.py`、`tests/test_lewm_pixel_observer_runtime.py`、`tests/test_evaluate_lewm_pixel_observer.py`、`tests/test_lewm_camera_intervention.py`、`tests/test_lewm_correction_derivation.py`、`tests/test_lewm_pixel_observer_cohort_stats.py`：端点、梯度、posterior 写回、target-frame causality、NaN mask、checkpoint roundtrip、预算证书、oracle 隔离、development/confirmatory 去重、camera/像素重算、cache 篡改、cohort gate、provenance 和旧 Gate 输出保护。

### 2026-07-15 PixelObserver v2 corrected-state audit（development：No-Go）

v2 不重训冻结的 `10,000`-step GroundedLeWM base。它显式输入当前名义投影、anchor 几何、历史 posterior residual 与有效位，并用两个独立 FCPG corrector 构造 paired supervision，避免直接拟合同一 corrector 的逐帧抖动。matched ablation 只切换 LeWM predicted embedding 是否进入 adapter。

#### 单任务训练诊断

以下四组使用 pick development 配置，只验证监督与接口，不代替四任务比较：

| Adapter | LeWM latent | Target | Final val EPE | Final loss |
| --- | --- | --- | ---: | ---: |
| paired | 是 | independent FCPG residual | `0.890940 px` | `0.124411` |
| paired | 否 | independent FCPG residual | `0.579794 px` | `0.077388` |
| single-measurement | 是 | same-corrector residual | `0.803414 px` | `0.103906` |
| oracle diagnostic | 否 | `q_true-q_nominal` | `0.424935 px` | `0.049085` |

产物分别位于 `outputs/lewm_pixel_observer_v2/{paired_pick_matched_step_drift_latent_seed7168,paired_pick_matched_step_drift_nolatent_seed7168,single_pick_matched_step_drift_seed7168,oracle_pick_matched_step_drift_nolatent_seed7168}/`。

#### 四任务 cross-corrector 复验

四任务 latent-on/off 使用相同 `17,000` 个 windows、seed、初始化、优化器和严格 `2,000`-step 预算：

| Adapter | Final val EPE | Final val loss | Latest SHA-256 |
| --- | ---: | ---: | --- |
| latent-on | `0.6746636 px` | `0.0942451` | `93c7cc17ea1a943948e524779c3efabdebe6c16ea1ea4cc2168d99670a38b908` |
| latent-off | `0.6053165 px` | `0.0901870` | `73fdf5cad59894040c1a0adc66b5cb795a6772d59354eb7b530484af8d7eb6db` |

该实验必须称为 **cross-corrector development**：adapter 训练数据覆盖四任务，但 development paired caches 来自 pick corrector seeds `1/2`，外部评估使用 shared-four-task corrector seed `0`。它检验跨任务、跨 corrector 的稳健性，不是严格 matched 的最终多任务论文实验。

外部 development 覆盖 drift、translation-step、rotation-step 共 `14` 个条件，每格 `40` 个 episode、四任务各 `10` 个。严格 primary 结论为：

- latent-on 相对参数匹配 latent-off 在 `12/14` 条件显著更差；
- translation hidden5 与 rotation hidden5 显著更好，但两者从固定事件时刻开始隐藏 measurement，跳变对因果模型不可观测，只是 negative control，不能计为 LeWM 相机预测收益；
- latent-off full-writeback 相对 hold-last 为 `0/14` 显著胜、`8/14` 显著负、`6/14` 无显著差异；
- 胜要求 `full-minus-hold < 0` 且 task-stratified bootstrap 95% CI 上界小于 `0`；负要求 CI 下界大于 `0`，其余为平。

oracle-history 只作因果诊断：平滑漂移可由简单 CV/KF 建模，而突发相机跳变独立于机器人动作和历史；新 measurement 到达前，action-conditioned LeWM 不可能预知跳变。因此当前失败来自状态职责和接口，不是训练步数不足。

**严格结论**：当前 `LeWM predicted embedding -> pixel residual adapter -> posterior writeback` 路线为 development No-Go。该结论不影响 Stage I FCPG 已成立的像素校正和策略收益，也不证明 LeWM 架构本身无效。

限制：全部结果均为已经用于决策的 `external_development`，不是 untouched confirmatory test；只有一个 adapter seed；measurement dropout 是 evaluator 已知的 withholding，RGB 未真实遮挡，也没有自主失效检测器；主实验中的 `q_true` 仅用于离线评分；尚无 action ranking、policy rollout、控制、真机或实时性证据。

#### Hardened evidence contract

最终 evaluator implementation SHA-256 为 `357ca78725556e183b314c4d1b250fb717740bf25dddc8140b7b65b43c1d7026`，comparison script SHA-256 为 `56c16e1b156bc8af846a73e01a41fb0d296cdee554c5ccc992fc0fe57b83771b`。report contract 加固前的结果已隔离到 `outputs/lewm_pixel_observer_v2/external_development/INVALID_pre_report_contract_hardening_0ac8726/`，不得引用；最终单任务与四任务比较分别位于 `outputs/lewm_pixel_observer_v2/external_development/comparisons/` 和 `outputs/lewm_pixel_observer_v2/external_development/comparisons_4task/`。

若以后重开，应采用：

```text
FCPG current measurement
-> explicit exogenous camera-residual filter
-> posterior anchor position and feature re-indexing
-> LeWM predicts posterior-grounded task/robot dynamics
```

先在 fresh drift/dropout 上要求显式 filter 超过 hold-last、CV 和 KF，再要求 posterior feature re-indexing 在 same-state action ranking 或 rollout 上超过 one-shot FCPG。Kalman/IMM、LeWM 和滤波本身都不是 novelty。

### 2026-07-10 Stage I corrector 正式 seed-0 结果

以下六组训练使用修复后的 SIGReg 实现，并共享 task-episode split、初始化 seed、架构、batch size、学习率和 50 epoch 预算。下表使用各自 best checkpoint 在 held-out validation episodes 上的结果；它们不是 legacy 指标，也不是下游 policy 结果。

| Task | `SIGReg` | Best epoch | `E_before` (px) | `E_after` (px) | Ratio | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pick-place | `0` | 50 | 6.275 | 0.668 | **0.106** | 通过 `<= 0.15` 门槛 |
| pick-place | `0.02` | 42 | 6.275 | 1.005 | 0.160 | 未通过 |
| pick-place | `0.09` | 44 | 6.275 | 2.197 | 0.350 | 未通过 |
| button-press | `0` | 46 | 6.359 | 0.580 | **0.091** | 通过 `<= 0.15` 门槛 |
| button-press | `0.02` | 28 | 6.359 | 1.475 | 0.232 | 未通过 |
| button-press | `0.09` | 40 | 6.359 | 1.633 | 0.257 | 未通过 |

初始 `0.09` 运行曾暴露 SIGReg 实现错误：输入契约为 `[T, B, D]`，上游在 batch 维估计特征函数，而本项目误沿 `T` 维求均值。该错误已修复并由上游逐值一致性测试覆盖；旧 `pick_seed0_sig009` 与 `button_seed0_sig009` checkpoint 仅保留为 debugging artifact，不得作为负消融或论文结果。表中的 `0.02/0.09` 均为修复后重训结果。

结论不是“SIGReg 无效”：修复后两种权重都能把高初值正则统计压低并继续学习残差；但 task-specific 像素回归需要精确定位，正态化约束在当前数据规模下牺牲了 held-out 精度。`0.02` 一致优于或接近 `0.09`，但仍未在两任务上通过 corrector 门槛。因此正式 task-specific corrector 固定使用 `SIGReg=0`；`0.02` 只在 predictive/shared representation 阶段继续验证，`0.09` 淘汰。

为检查模型是否依赖局部视觉而不是仅记忆扰动统计，对两个 `SIGReg=0` best checkpoint 使用同一 validation split 做输入干预：

| Task | Normal ratio | Local cutout | Zero image | Shuffled image |
| --- | ---: | ---: | ---: | ---: |
| pick-place | **0.106** | 0.709 | 2.750 | 1.817 |
| button-press | **0.091** | 0.925 | 1.875 | 1.657 |

去除或打乱视觉证据后校正显著退化，且 zero-image / shuffled-image 均劣于不校正的 ratio `1.0`。这支持“corrector 使用逐样本局部视觉反馈”的机制解释，但尚不能替代下游配对 rollout 或跨 seed 结论。

对应 residual-only cache 已生成：

- `data/grounding_cache/lewm_stage1/pick_seed0_sig0`：全数据 ratio `0.0469`；
- `data/grounding_cache/lewm_stage1/button_seed0_sig0`：全数据 ratio `0.0465`。

两者均为 schema-v2、`has_reliability=false`，且不含 `reliability.npy`。

### 2026-07-11 Stage I 两任务 shared corrector 正式 seed-0 结果

多源训练直接 mmap 两套 30,000 帧 ViT cache，不复制约 17 GB 特征文件；每个 source 独立按 task-episode 划分为 24,000/6,000 train/validation 帧，训练 sampler 令 pick-place 与 button-press 具有相同总采样概率。两组使用相同初始化、架构、batch 1024、学习率和 50 epoch 预算：

| Shared corrector | Best epoch | Aggregate `E_after` / ratio | Pick held-out `E_after` / ratio | Button held-out `E_after` / ratio |
| --- | ---: | ---: | ---: | ---: |
| `SIGReg=0` | 45 | **0.576 px / 0.091** | **0.622 px / 0.099** | **0.529 px / 0.083** |
| `SIGReg=0.02` | 49 | 0.762 px / 0.121 | 0.822 px / 0.131 | 0.701 px / 0.110 |

两种 shared corrector 都在两个任务上通过 ratio `<=0.15` 门槛；`SIGReg=0` 的纯像素校正更准，并且略优于对应 task-specific corrector。shared `0.02` 相比单任务 `0.02` 明显改善，说明数据多样性缓解了正则造成的回归损失，但它仍不是 residual head 的最佳配置。因此后续 shared policy 固定使用 shared `SIGReg=0` corrector/cache；`0.02` 只施加于带 next-latent 目标的 policy state latent，用于阻止预测表征塌缩。

正式 cache：

- `data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig0_{pick,button}`
- `data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig002_{pick,button}`

### 2026-07-11 四任务 shared、LOTO 与 few-shot seed-0 结果

Reach 和 push 使用与 pick-place、button-press 相同的正式采集协议：每任务 `300` 个 episode、每 episode `100` 帧、`224 x 224` RGB，实际相机扰动为 `0-4 cm` 平移和 `0-5 deg` 旋转，并保留 `25%` clean episode。四任务均缓存冻结 ViT-B/16 第 11 层特征；训练时每个 source 独立做 `240/60` episode split，并按 source 等概率采样。

四任务 shared corrector 的同 split、同初始化匹配结果如下。每个任务列均来自其 `6000` 个 held-out frames，而 aggregate 用于 checkpoint 选择：

| Evaluation | `SIGReg=0` `E_after` / ratio | `SIGReg=0.02` `E_after` / ratio |
| --- | ---: | ---: |
| Aggregate | **0.553 px / 0.088** | 0.863 px / 0.137 |
| Pick-place | **0.563 px / 0.090** | 0.938 px / 0.149 |
| Button-press | **0.518 px / 0.081** | 0.766 px / 0.120 |
| Reach | **0.603 px / 0.096** | 0.826 px / 0.132 |
| Push | **0.528 px / 0.084** | 0.921 px / 0.146 |

四任务多样性使 `SIGReg=0.02` 在所有已见任务上通过 `0.15` 门槛，但其逐点校正精度仍一致弱于 `SIGReg=0`。因此该结果支持“SIGReg 在多任务 batch 中不再导致 corrector 失效”，不支持“SIGReg 提升像素残差回归”。四任务 policy 继续使用 `SIGReg=0` corrector；policy latent 是否受益于 SIGReg 必须由新的四格实验单独判断。

严格 leave-one-task-out 每折从相同初始化开始，仅使用三个源任务训练，并在第四个完全未参与训练的任务上评估：

| Held-out task | `E_before` | `E_after` | Ratio | `<= 0.15` |
| --- | ---: | ---: | ---: | :---: |
| Pick-place | 6.275 px | 0.725 px | **0.116** | yes |
| Button-press | 6.359 px | 1.890 px | 0.297 | no |
| Reach | 6.269 px | 0.790 px | **0.126** | yes |
| Push | 6.303 px | 0.781 px | **0.124** | yes |

该协议得到 `3/4` 通过，而不是通用 zero-shot：button-press 是明确负结果，不能隐藏或并入平均数后宣称全部成功。各折源任务 validation 都正常收敛，因此差异主要反映目标任务迁移难度，而不是某一折训练失败。

另用只在 pick-place 与 button-press 上训练的 shared corrector 对未见 reach/push 做视觉干预，检查跨任务残差是否依赖当前图像：

| Unseen task | Normal | Local cutout | Zero image | Shuffled image |
| --- | ---: | ---: | ---: | ---: |
| Reach | **0.134** | 0.733 | 3.752 | 1.645 |
| Push | **0.138** | 0.738 | 3.726 | 1.625 |

正常图像下两任务均通过门槛，切除、清零或打乱视觉证据后显著退化。这排除了 corrector 仅凭本体状态或固定扰动分布完成 unseen-task 校正的简单解释，但它仍不是第二机器人或跨视觉域证据。

针对 LOTO 中失败的 button-press 折，从同一个三源任务 checkpoint 初始化目标域适配。`10/50/200` 使用同一 episode permutation 的嵌套训练子集；三组固定 `50 epochs`、batch `512`、学习率 `1e-4`、`SIGReg=0`，并统一评估 final checkpoint，不使用目标 held-out 指标选择 best epoch：

| Target episodes | `E_before` | `E_after` | Ratio | `<= 0.15` |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 6.359 px | 1.890 px | 0.297 | no |
| 10 | 6.359 px | 1.213 px | 0.191 | no |
| 50 | 6.359 px | 0.793 px | **0.125** | yes |
| 200 | 6.359 px | 0.531 px | **0.084** | yes |

这条曲线说明失败的 unseen button 任务可以被少量完整 episode 逐步恢复，约 `50` 个 episode 后通过当前机制门槛。它目前只覆盖 adapter seed `0`、同一 MetaWorld embodiment 和 seed-0 相机扰动族；在 seed `1/2`、未见扰动 seed/direction 和下游 rollout 完成前，只能作为迁移诊断结果。

机器可读汇总：

- `outputs/transformer_residual/lewm_stage1/loto_4task_seed0_sig0_summary.json`
- `outputs/transformer_residual/lewm_stage1/fewshot_button_seed0_sig0_summary.json`
- `scripts/summarize_corrector_loto.py`
- `scripts/summarize_corrector_fewshot.py`

#### 2026-07-11 Shared policy 紧凑视觉缓存

原始 policy loader 对每个 sequence 同时读取 current history 与 next history。batch 1024 时，单批会重复搬运约 1.2 GB 的 `768 x 14 x 14` fp16 backbone feature，导致 CUDA0 在计算与等待数据之间呈锯齿型利用率。为消除这个 I/O 瓶颈，新增：

```text
scripts/cache_policy_visual_features.py
```

该缓存复用 shared `SIGReg=0` corrector 已训练好的 ViT projection / fusion，在 `q_corrected` 处预采样视觉向量，并保存：

```text
sampled_visual.npy  # [N, K, 256], fp16
global_visual.npy   # [N, 256], fp16
metadata.json
```

**方法边界必须明确：** 这是 frozen visual adapter 路线，不是对原先可训练 dense projection 的无损压缩。使用 `--policy-cache` 时，policy 会从同一个 corrector checkpoint 初始化并冻结 ViT projection / fusion，只训练 grounded-token MLP、robot projection、predictive Transformer 与动作头。原始 `--vision-cache` 路径继续保留，用于需要训练 policy-specific dense projection 的实验。两种路径必须使用不同输出目录，结果不能混写。

两任务缓存命令：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/cache_policy_visual_features.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --vision-cache data/vision_cache/pick300_224_vitb_l11_fp16 \
  --grounding-cache data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig0_pick \
  --checkpoint outputs/transformer_residual/lewm_stage1/shared_pick_button_seed0_sig0/fcpg_transformer_residual_best.pt \
  --output-dir data/policy_visual_cache/lewm_stage1/shared_pick_button_seed0_sig0_pick \
  --batch-size 1024 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/cache_policy_visual_features.py \
  --data data/metaworld_projection/scripted_button_actualcam_se3_clean25_300eps_224.npz \
  --vision-cache data/vision_cache/button300_224_vitb_l11_fp16 \
  --grounding-cache data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig0_button \
  --checkpoint outputs/transformer_residual/lewm_stage1/shared_pick_button_seed0_sig0/fcpg_transformer_residual_best.pt \
  --output-dir data/policy_visual_cache/lewm_stage1/shared_pick_button_seed0_sig0_button \
  --batch-size 1024 \
  --device cuda:0
```

每个任务的原始 backbone/prefix cache 为 `9,077,760,000` bytes，紧凑 cache 为 `61,440,000` bytes，比例 `0.00677`，约缩小 148 倍。对 pick-place 前 128 帧做在线前向复核时，local feature 的最大/平均绝对差均为 `0`，global feature 的最大绝对差为一个 fp16 量化单位 `0.0009765625`。

相同模型、8 个训练 batch、1 个 validation batch 的 CUDA0 吞吐：

| Batch | Train samples/s | Peak allocated CUDA memory |
| ---: | ---: | ---: |
| 1024 | 4,273 | 5.53 GB |
| 2048 | 5,102 | 10.15 GB |
| 4096 | 6,360 | 19.39 GB |

batch 4096 可运行，但会把每个 epoch 的 optimizer update 数降为 batch 1024 的约四分之一。为保持本节两任务 seed-0 因子实验可比，两任务 shared predictive policy 使用 batch 1024、60 epoch、warmup 300；后续四任务匹配复验单独使用 batch 2048、task-balanced sampling 和固定预算，二者不能混写。

正式 shared predictive policy 命令：

```bash
CUDA_VISIBLE_DEVICES=0 $FCPG_PY scripts/train_action_chunk_policy.py \
  --data data/metaworld_projection/scripted_pick_actualcam_se3_clean25_300eps_224.npz \
  --grounding-cache data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig0_pick \
  --policy-cache data/policy_visual_cache/lewm_stage1/shared_pick_button_seed0_sig0_pick \
  --extra-data data/metaworld_projection/scripted_button_actualcam_se3_clean25_300eps_224.npz \
  --extra-grounding-cache data/grounding_cache/lewm_stage1/shared_pick_button_seed0_sig0_button \
  --extra-policy-cache data/policy_visual_cache/lewm_stage1/shared_pick_button_seed0_sig0_button \
  --output-dir outputs/action_chunk/lewm_stage1/shared_pick_button_seed0_lp1_sig002_compact \
  --mode fcpg \
  --policy-arch lewm-predictive \
  --fcpg-checkpoint outputs/transformer_residual/lewm_stage1/shared_pick_button_seed0_sig0/fcpg_transformer_residual_best.pt \
  --reliability-override native \
  --vision-model-name vit_base_patch16_224.augreg_in1k \
  --vision-selected-layers 11 \
  --obs-mode robot \
  --obs-horizon 2 \
  --action-horizon 32 \
  --model-dim 256 \
  --vision-dim 256 \
  --encoder-layers 2 \
  --decoder-layers 6 \
  --predictor-layers 2 \
  --heads 8 \
  --mlp-ratio 4 \
  --dropout 0.1 \
  --trainable-backbone-blocks 0 \
  --epochs 60 \
  --batch-size 1024 \
  --num-workers 4 \
  --lr 0.0001 \
  --backbone-lr 0.00001 \
  --weight-decay 0.000001 \
  --warmup-steps 300 \
  --gripper-weight 0.5 \
  --latent-pred-weight 1 \
  --sigreg-weight 0.02 \
  --sigreg-proj 256 \
  --grad-clip 1 \
  --val-fraction 0.2 \
  --seed 0 \
  --split-seed 0 \
  --patience 10 \
  --device cuda:0
```

#### 2026-07-11 Shared policy seed-0 完整 `2 x 2` 结果

以下四条策略使用同一个 shared `SIGReg=0` corrector、同一对紧凑 frozen-adapter cache、相同 task-episode split、policy 初始化、batch 1024、学习率和 early-stopping 规则。两个实验因子分别是 action-conditioned next-latent loss 是否启用，以及 policy state latent 的 `SIGReg=0.02` 是否启用。每个任务均在 6000 条 held-out sequence 上独立评估：

| Policy objective | `SIGReg` | Best epoch | Pick chunk MSE | Pick effective rank | Button chunk MSE | Button effective rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| next-latent | `0` | 36 | 0.02217 | 4.60 | 0.003003 | 3.52 |
| next-latent | `0.02` | 18 | 0.03238 | 26.51 | 0.005431 | 24.55 |
| action-only | `0` | 37 | **0.02063** | 25.81 | **0.002200** | 11.02 |
| action-only | `0.02` | 25 | 0.03311 | **36.35** | 0.006817 | **34.36** |

无 SIGReg 的 next-latent predictor 在两个任务上都把预测 MSE 压到很低，同时把 effective rank 降到约 3-5，说明 shared online target 存在“压缩表示以降低预测难度”的捷径。`SIGReg=0.02` 明确恢复表示秩；但 action-only/no-SIGReg 本身并未出现同等级塌缩，说明当前低秩主要由 next-latent objective 引入，而不是多任务动作表征必然塌缩。rank 只用于诊断，不能替代闭环任务指标。

四条 best checkpoint 随后在 pick-place 与 button-press 上分别运行 `0/5/10/20` 度实际相机旋转。每个点包含 60 个相同 task index、环境 seed、相机旋转轴与幅度的 episode；同一策略还配对切换 nominal/open-loop 与 native FCPG grounding。native-grounding success 的归一化扰动 AUC 为：

| Policy objective | `SIGReg` | Pick AUC | Button AUC |
| --- | ---: | ---: | ---: |
| next-latent | `0` | 0.0542 | 0.6667 |
| next-latent | `0.02` | 0.0458 | 0.5771 |
| action-only | `0` | **0.0896** | **0.7292** |
| action-only | `0.02` | 0.0667 | 0.5646 |

action-only/no-SIGReg 下，FCPG 相对同策略 nominal grounding 在 pick-place 5 度提高 `+0.1167`，paired bootstrap 95% CI `[+0.0500, +0.2000]`；在 button-press 5 度和 10 度分别提高 `+0.2833` 与 `+0.2000`，95% CI 分别为 `[+0.1500, +0.4167]` 和 `[+0.0667, +0.3333]`。20 度下两任务均无显著收益，继续支持 FCPG v1 是局部残差校正而非 calibration-free policy。

为避免用 rank 或平均 MSE 替代闭环结论，另对四条策略的逐 episode success 直接计算完整 factorial effect。下表报告归一化 AUC 差值；正的 next-latent effect 表示预测目标优于 action-only，interaction 定义为：

```text
(predictive_SIGReg - predictive_no_SIGReg)
- (action_SIGReg - action_no_SIGReg)
```

| Task | SIGReg under predictive | SIGReg under action-only | next-latent under SIGReg=0 | next-latent under SIGReg | interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| pick-place | -0.0083 | -0.0229 | -0.0354 | -0.0208 | +0.0146 |
| button-press | -0.0896 | -0.1646 | -0.0625 | +0.0125 | +0.0750 |

两任务 seed-0 证据不支持把 next-latent 或 SIGReg 宣称为当前性能增益：action-only/no-SIGReg 在两个任务的 AUC 都最高。predictive 分支、SIGReg 实现、四条 checkpoint 和全部 rollout 均保留为正式消融。后续四任务 batch-2048 匹配复验已经重现同一排序，因此当前参考配置固定为 action-only `SIGReg=0`；但该结论仍只有一个 policy seed，不能写成普遍失效。

可恢复的正式评估与交互项汇总入口：

```text
scripts/run_action_chunk_rollout_matrix.py
scripts/summarize_policy_factorial.py
```

正式输出：

- `outputs/action_chunk/lewm_stage1/shared_pick_button_seed0_lp1_sig0_compact/`
- `outputs/action_chunk/lewm_stage1/shared_pick_button_seed0_lp1_sig002_compact/`
- `outputs/action_chunk/lewm_stage1/shared_pick_button_seed0_lp0_sig0_compact/`
- `outputs/action_chunk/lewm_stage1/shared_pick_button_seed0_lp0_sig002_compact/`
- `outputs/action_chunk_rollout/lewm_stage1/shared_predictive_seed0_matched/matrix_summary.md`
- `outputs/action_chunk_rollout/lewm_stage1/shared_predictive_seed0_matched/factorial_summary.md`

这组 frozen-adapter 结果与前面的 task-specific trainable dense-projection 结果属于不同 policy front-end，不做直接数值排名。它用于筛选 shared policy objective、验证反馈接地机制，并确定后续多任务扩展需要保留的匹配因子。

### 2026-07-11 四任务 shared policy `2 x 2` seed-0 复验

四任务复验使用 batch `2048`、task-balanced sampling 和 per-source SIGReg，完整比较 action-only / predictive × `SIGReg=0/0.02`。四个 checkpoint 均训练完成；每任务离线评估包含 `6000` 条 held-out sequence，rollout 矩阵包含 4 policies × 4 tasks × 4 rotation levels × nominal/native，共 `128` 个 job，每格 `60` episodes。

离线结果格式为 `chunk MSE / effective rank`：

| Policy | Pick | Button | Reach | Push |
| --- | --- | --- | --- | --- |
| action-only, `SIGReg=0` | **0.02324 / 27.76** | **0.003292 / 13.13** | **0.002968 / 24.82** | **0.02067 / 28.20** |
| action-only, `SIGReg=0.02` | 0.03901 / **37.02** | 0.009509 / **34.47** | 0.009833 / **32.64** | 0.03390 / **36.82** |
| predictive, `SIGReg=0` | 0.02416 / 7.01 | 0.004114 / 4.38 | 0.004274 / 10.36 | 0.02297 / 7.31 |
| predictive, `SIGReg=0.02` | 0.04064 / 25.29 | 0.009761 / 23.23 | 0.008118 / 22.30 | 0.03240 / 24.28 |

native-grounding success AUC 为：

| Policy | Pick | Button | Reach | Push |
| --- | ---: | ---: | ---: | ---: |
| action-only, `SIGReg=0` | **0.0958** | **0.7250** | **0.1833** | **0.1229** |
| action-only, `SIGReg=0.02` | 0.0354 | 0.4083 | 0.1125 | 0.0708 |
| predictive, `SIGReg=0` | 0.0563 | 0.6062 | 0.1146 | 0.0875 |
| predictive, `SIGReg=0.02` | 0.0250 | 0.4792 | 0.0729 | 0.0479 |

结论：大 batch、四任务数据和 per-source SIGReg 确实提高了 action-only 表示秩，但四任务动作误差与控制 AUC 都一致变差；next-latent 也没有超过 action-only/no-SIGReg。当前参考配置因此固定为 action-only `SIGReg=0`，另三格作为完整负消融保留。该排序只来自 policy seed `0`，不能外推为方法普遍失效。

这组实验是四个已见任务的联合训练，不是 unseen-task policy 泛化。clean native success 仅为 pick `0.35`、button `1.00`、reach `0.3167`、push `0.3667`；当前证据用于筛选 objective，不能写成四任务控制已解决。严格 unseen-task 证据仍只属于 corrector LOTO `3/4`。

正式汇总位于：

- `outputs/action_chunk_rollout/lewm_stage1/shared_4task_seed0_b2048_persource_matched/matrix_summary.md`
- `outputs/action_chunk_rollout/lewm_stage1/shared_4task_seed0_b2048_persource_matched/matrix_summary.json`

### 2026-07-10 Stage I predictive policy 正式 seed-0 结果

Pick-place 的六组 policy 使用同一个 `SIGReg=0` residual-only corrector/cache，并共享 task-episode split、policy 初始化、架构、batch 1024、学习率和训练预算。SIGReg 只作用于 policy state latent。以下为 6000 条 held-out sequence 的统一离线评估：

| Next-latent weight | `SIGReg` | Chunk MSE | Next-latent MSE | Effective rank | Participation ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | **0.01627** | 0.01769 | 9.53 | 4.88 |
| 1 | 0 | 0.01698 | **0.00085** | 2.70 | 1.69 |
| 0 | 0.02 | 0.02485 | 0.06883 | **44.58** | **34.17** |
| 1 | 0.02 | 0.02411 | 0.02804 | 31.58 | 24.74 |
| 0 | 0.09 | 0.02942 | 0.07267 | 43.69 | 33.61 |
| 1 | 0.09 | 0.02988 | 0.04422 | 34.68 | 25.86 |

无 SIGReg 时，next-latent objective 将预测 MSE 降低约 20 倍，却把 effective rank 从 9.53 压到 2.70，并使动作误差变差；这说明共享 online target 存在“通过压缩表示来变得容易预测”的捷径。SIGReg 明显恢复表示维度。`0.02` 在正则化组中动作误差最低，且与 next-latent 联用后同时降低动作与预测误差；`0.09` 没有带来额外表示收益并进一步损害动作，因此淘汰。当时曾把 `next-latent + 0.02` 保留为 shared / larger-data 候选；后续四任务 batch-2048 matched rollout 已显示稳定退化，因此该候选最终 No-Go，当前 reference 为 `action-only, SIGReg=0`。

对两条候选使用相同 checkpoint，只在测试时切换 `native FCPG` 与 `nominal/open-loop` grounding。每个设置运行 60 个相同 task index、环境 seed 和相机扰动的配对 episode：

| Policy | Setting | Nominal success | FCPG success | Paired delta | Bootstrap 95% CI | McNemar p | Nominal / corrected error |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| action-only, sig=0 | clean | 0.5333 | 0.5000 | -0.0333 | [-0.0833, 0.0000] | 0.500000 | 0.000 / 0.411 px |
| action-only, sig=0 | 5 deg | 0.0000 | **0.2167** | **+0.2167** | **[0.1167, 0.3167]** | **0.000244** | 18.417 / 2.127 px |
| next-latent, sig=0.02 | clean | 0.2833 | 0.2833 | 0.0000 | [-0.0500, 0.0500] | 1.000000 | 0.000 / 0.429 px |
| next-latent, sig=0.02 | 5 deg | 0.0167 | **0.1667** | **+0.1500** | **[0.0667, 0.2500]** | **0.003906** | 18.387 / 2.147 px |

两种策略都显示相同机制：FCPG 在 5 度相机旋转下显著优于同策略 open-loop grounding，在 clean 下没有显著差异。该表只记录早期 seed-0 predictive-ablation 结果；后续 `5 deg` action-policy intervention 已覆盖三 policy seed，但独立 button adapter seeds `1/2` 和同规格 fresh external matrix 仍需完成。

Button-press 使用同一个 `SIGReg=0` residual-only corrector/cache 复核筛选结论。由于 pick-place 已证明 `0.09` 被 `0.02` 支配，button-press 只运行 `next-latent weight in {0, 1} x SIGReg in {0, 0.02}` 的匹配 `2 x 2` 实验：

| Next-latent weight | `SIGReg` | Chunk MSE | Next-latent MSE | Effective rank | Participation ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | **0.001712** | 0.011229 | 6.44 | 4.37 |
| 1 | 0 | 0.001982 | **0.000144** | 3.48 | 2.37 |
| 0 | 0.02 | 0.004846 | 0.083036 | **42.84** | **33.29** |
| 1 | 0.02 | 0.003965 | 0.023840 | 32.33 | 26.25 |

Button-press 重现了 pick-place 的表示现象：无 SIGReg 的 next-latent 目标把预测误差降低约 78 倍，但 effective rank 从 6.44 降到 3.48，同时动作误差变差；`0.02` 阻止了这种塌缩，并且在正则化组内让 next-latent 同时改善动作与预测误差。`0.02` 的作用因此被限定为 predictive/shared representation 的防塌缩约束，而不是 task-specific residual corrector 的精度增强项。

同样的 60 对 rollout 给出：

| Policy | Setting | Nominal success | FCPG success | Paired delta | Bootstrap 95% CI | McNemar p | Nominal / corrected error |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| action-only, sig=0 | clean | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.000000 | 0.000 / 0.327 px |
| action-only, sig=0 | 5 deg | 0.5500 | **0.9667** | **+0.4167** | **[0.3000, 0.5333]** | **5.96e-8** | 18.618 / 1.841 px |
| next-latent, sig=0.02 | clean | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.000000 | 0.000 / 0.396 px |
| next-latent, sig=0.02 | 5 deg | 0.6500 | **0.8333** | **+0.1833** | **[0.0500, 0.3167]** | **0.012726** | 18.595 / 1.978 px |

Button-press 的 clean 性能没有回退，两条策略在 5 度扰动下都得到显著配对提升。task-specific action-only `SIGReg=0` 仍有更高的绝对成功率；当时仅凭 policy seed `0` 曾让 `next-latent + SIGReg=0.02` 进入 shared / larger-data 复验，后续四任务 matched rollout 否定了该候选。这里保留的是阶段性结果，不是当前执行建议。

正式输出：

- `outputs/action_chunk/lewm_stage1/pick_seed0_*/offline.json`
- `outputs/action_chunk_rollout/lewm_stage1/pick_seed0_lp0_sig0_{clean,rot5}_60/paired/`
- `outputs/action_chunk_rollout/lewm_stage1/pick_seed0_lp1_sig002_{clean,rot5}_60/paired/`
- `outputs/action_chunk/lewm_stage1/button_seed0_*/offline.json`
- `outputs/action_chunk_rollout/lewm_stage1/button_seed0_lp0_sig0_{clean,rot5}_60/paired/`
- `outputs/action_chunk_rollout/lewm_stage1/button_seed0_lp1_sig002_{clean,rot5}_60/paired/`

## 下一位 Codex 执行顺序

1. 冻结 Counterfactual v2、PixelObserver v2 与 KineParity 的既有 No-Go；不得在已暴露数据上继续扫参或换名重跑。
2. Counterfactual v3 已在 P0 判为 No-Go；冻结该 artifact，不调 spread/shortcut 门槛、不追加 episode、不训练 DirectRanker，也不启动后续 VisualOracle/LeWM/GoalToken/FCPG-vs-nominal/CEM。
3. v3 actual-return post-hoc 与唯一一次 4-fold cross-fit 修正均已结束；当前 privileged future-state residual 接口为 descriptive No-Go，不得在已暴露 split 上再调模型，也不得用它启动 VisualOracle/LeWM/CEM。
4. Anchor-Persistent LeWM v1 的固定 `2,000`-step pilot、clean/dynamic schema-v2 evaluator、visual-occlusion evaluator 与 zero-training headroom 诊断均已完成；冻结 `latest@2000`，不得重训或挑 best。
5. 该 pilot 的旧 recovery Gate 失败保持冻结；后续独立 A3 已完成三个 seed 的 `10,000`-step reserved-test confirmation，并确认 action-conditioned transition mechanism。该结果不回写旧 recovery 判定，selector、CEM 与控制 claim 仍不授权。
6. fresh measurement-dominant posterior Adapter Gate 已完成并判为 No-Go：保留“遮挡期主误差强正、LeWM 明显优于 no-LeWM”的机制结果，同时冻结恢复/clean safety 失败；不得在该 split 上调 loss、阈值或网络宽度。
7. 最小 current-only VisualTrust probe 已完成并判为 Development No-Go：视觉信号明确，但未胜强 temporal baseline 的冻结 margin，clean/recovery safety 也未通过；不得在当前 draw 调 threshold，也不启动 hard-switch posterior。
8. Measurement-vs-Prior recursive oracle 的 fixed-schedule Diagnostic Go 保持为历史机制上限；fresh randomized development 已完成并判为 No-Go，不得在该 split 上调进入规则或降低门槛。
9. 当前 preference comparator、posterior policy control 与 CEM 均不授权。保留准确负结论：强 LeWM prior/oracle headroom 存在，但固定的 cautious-enter/fast-exit FSM 未达绝对门槛，也未超过 privileged event-aligned time shortcut。
10. Action-Branch Random-Hazard A1 的三 seed `2k` Screen 已完成：动作对齐信号在 pick/reach/push 上很强，但 Button intervention support 仅 `34.48%`，合取 Gate 为 No-Go。Button train+development 物理重放已确认 latent 与物理分叉在 post-success 同步塌缩；这解释失败但不改判定。禁止启动 A1 `10k`，independent test 保持未打开。
11. Continuous-Task A2 的三 seed `2k` development screen 已强通过，但 fresh confirmation train 的 Drawer H6-H8 support 只有 `72.92%`，低于 `80%`。A2 confirmation 因此前置 No-Go：禁止启动 `10k`、禁止采集 reserved test，也不得换 seed 或降低门槛重开同一协议。
12. Continuous-Task A3 的四任务确认结果保留；后续 A4/A4b 已在七任务 fresh panel 上完成更强的 `3 x 10k` Confirmation-Go，并取代 A3 作为当前 action-conditioned transition 证据。不得继续在 A3/A4b test 上调模型。
13. P1-native fixed-gain complementary observer 已在 `30` 个 fresh Pick episode 上通过 representation Gate；随后同一 `30` 个 physical episode 的单进程三 seed paired policy rollout 已完成。动作一致性与累计 reward 稳定正向，但 seed-macro success CI 跨零，准确判定为 **mechanism-positive / policy-success Development-No-Go**。
14. 不启动 `240/300` episode success confirmation、selector 或 CEM，也不把 P1 rollout 写成第二贡献。若未来用全新协议重开，必须预先冻结 final/sustained success、跨 policy seed 与功效分析后的 physical-episode 数量；不得用当前 any-step success 事后换 primary。

## Stage II 历史候选：KineParity 的独立 Go / No-Go

KineParity 曾是 simulation-first 的候选扩展，不是当前 FCPG v1 已完成结果。它使用单目 RGB、相机内参、初始名义外参、已知机器人型号/URDF/DH 和实时关节角，不依赖 RGB-D，也不引入 learned `rho`。

本节记录一个已经完成的候选扩展及其预注册判定，不再代表当前 target。该候选原本试图用 global-only observer-control transfer 恢复 10 度大偏移：全臂 observer points 只求解共享可观测 `SE(3)` 模态，指尖 control-anchor 视觉对应与残差标签完全留出，再将更新传播到 `q_C_global`。Gate A 通过，但 Gate B 在唯一一次 frozen rescue 后仍未满足 matched direct-control prototype 门槛，因此该路线已停止。

**Gate A（oracle geometry，已完成并冻结）**：从 MuJoCo 全臂 mesh center 构造 observer points；control-anchor 视觉对应、`q_C_true` 和残差标签完全留出；用固定 `P0/R`、三帧历史、task-oriented 白化 MAP/TSVD 与按秩卡方 innovation gate 估计共享 twist。该 gate 已在 clean、`5/10 deg`、`4 cm`、correspondence noise/dropout 和 distal-only 病态星座下通过。

**Gate B（matched visual correspondence，已完成：No-Go）**：

1. 只用训练 split 学 observer 2D correspondence / innovation，在 held-out 扰动方向、幅度、episode 和机器人构型上评估；no-image 与 shuffled-image 必须明显退化。
2. KineParity、MAP-all、robust PnP、direct 6D 和局部匹配必须读取完全相同的 observer correspondences、valid mask 与三帧窗口。
3. 指尖 control anchors 始终不参与视觉 correspondence 训练或共享更新求解，只在 held-out 评估中使用 `q_C_true`。
4. 预注册协议要求 Gate B 通过后才进入 Gate C；本次 Gate B 未通过，因此 feature resampling、matched policy retraining 和 bounded local head 均未执行。

**2026-07-12 Gate B pilot（当前判定：No-Go）**：pilot observer 训练集覆盖 `0-5 deg / 0-4 cm`，独立测试集固定 `10 deg`，使用不同 task index 与 episode；control-anchor 视觉对应和标签不进入 observer 训练或共享求解。该训练集采集早于显式 `env_seed` manifest，因此结果只用于机制判定，不作为最终论文统计。

- LeWM observer head（`SIGReg=0`）把 observer mean error 从 `37.91` 降到 `19.40 px`；`SIGReg=0.02` 的独立误差为 `20.52 px`，不优于无正则版本。zero-feature 与 shuffled-feature 下，KineParity held-out control mean 分别为 `37.37/38.59 px`，而正常视觉为 `19.08 px`，证明校正确实依赖当前视觉。
- matched frozen ViT prototype observer 把 observer mean 降到 `8.24 px`。经 full-9 KineParity 传播后，完全留出的三个 control anchors 为 `6.61/15.24/24.69 px`（mean/P95/max）；同帧、同 ViT、同 `64 px` 搜索范围、直接使用 control-anchor descriptor bank 的 prototype baseline 为 `5.23/10.92/25.76 px`。KineParity 比 direct baseline 高 `1.38 px` mean 且 P95 更差，未满足预注册的 `prototype + 1 px` 与尾部条件。
- 在 `distal3` 病态星座上，KineParity 为 `10.17/28.45/58.47 px`，MAP-all 为 `13.21/36.15/70.61 px`，linear-6D 为 `63.81/164.61/1810.73 px`，PnP-RANSAC 出现严重灾难解。这支持退化感知更新的局部机制价值，但不足以推翻 full-9 主结果。

**Frozen rescue 最终结果（Stage II：No-Go）**：正式训练集包含 `5000` 帧、task index `0-99`、environment seed `300000-300099`，独立 `10 deg` 测试为 task index `200-229`、environment seed `200000-200029`，两者无交集；最大训练旋转扩到 `8 deg`，其他架构、prototype 与几何设置保持不变。

- learned observer mean 从 pilot 的 `19.40` 改善到 `13.87 px`；经 full-9 KineParity 后 held-out control mean 为 `13.23 px`。zero-feature/shuffled-feature 分别退化为 `38.92/39.54 px`，视觉依赖成立但绝对精度不足。
- matched observer prototype + full-9 KineParity 为 `6.17/15.89/28.06 px`（mean/P95/max）；同数据预算 direct control prototype 为 `5.14/10.85/22.65 px`。KineParity 比 direct baseline 高 `1.03 px` mean，且 P95 高 `5.05 px`，仍未满足预注册门槛。
- matched observer prototype 下，MAP-all full-9 为 `5.39/10.69 px`，说明当前 KineParity 的 task-utility 模态截断在良态 full-9 中反而损失精度；distal-3 下 KineParity `9.24/25.30/44.77 px` 仍显著优于 MAP-all `12.81/34.10/76.47 px` 与无约束 6D/PnP，但该退化优势不足以覆盖主场景失败。

按预注册规则，KineParity 从当前小论文主线删除，Gate C、matched policy retraining 与 bounded local head 均不执行。保留 Gate A、Gate B evaluator 和 distal 退化结果，作为附录诊断或后续专门研究；当前论文回到已经通过 P1 的 Stage I FCPG。正式产物为：

```text
outputs/kineparity/gate_b_rescue/observer_head_vitb_l11_lewm384_r64_sig0_seed0/gate_b_rot10_frozen_main.json
outputs/kineparity/gate_b_rescue/observer_head_vitb_l11_lewm384_r64_sig0_seed0/gate_b_rot10_visual_ablation.json
```

原预注册 Go 条件如下，均仅为审计记录，不再构成当前执行任务：

- 使用白化后的奇异谱证明更新方向可辨识，且低可观测方向不会产生长尾错误更新；
- control-anchor 视觉对应与真实投影完全不参与共享更新求解时，held-out `10 deg / 4 cm` 的指尖投影误差仍显著低于逐点 FCPG、robust PnP 和直接 `6D` 回归；
- global-only 必须独立通过；若加入 bounded local head，它必须带来跨 seed 稳定增益，且幅值不能系统性顶到 `r_local` 上限；
- 接入相同 policy 后，扰动曲线 AUC 提升且 clean 性能不退化。

如果自动星座选择不优于均匀全臂点集，则删除自动设计 claim，不为凑创新继续调参。时间滤波、EKF/MHE、LeWM dynamics 和主动标定均属于后续扩展，不进入该小论文的第一版。

## Test Plan

代码回归：

```bash
$FCPG_PY -m pytest -q
$FCPG_PY -m pytest -q \
  tests/test_fcpg_core.py \
  tests/test_lewm_grounding.py \
  tests/test_camera_pairing.py \
  tests/test_collection_rng.py \
  tests/test_lewm_oracle_target.py \
  tests/test_lewm_gate_l1_config.py \
  tests/test_lewm_gate_l1.py \
  tests/test_sequence_data_v2.py \
  tests/test_structured_grounding.py \
  tests/test_kineparity_stress.py \
  tests/test_lewm_pixel_observer.py \
  tests/test_lewm_pixel_observer_runtime.py \
  tests/test_evaluate_lewm_pixel_observer.py \
  tests/test_lewm_camera_intervention.py \
  tests/test_lewm_correction_derivation.py \
  tests/test_lewm_pixel_observer_cohort_stats.py \
  tests/test_lewm_pixel_observer_v2.py \
  tests/test_lewm_pixel_observer_v2_runtime.py \
  tests/test_lewm_pixel_observer_v2_pipeline.py \
  tests/test_compare_lewm_pixel_observer_v2_reports.py \
  tests/test_lewm_anchor_persistent.py \
  tests/test_mujoco_camera_trajectory.py
$FCPG_PY -m py_compile fcpg/*.py scripts/*.py
git diff --check
```

必须覆盖：

- legacy checkpoint 缺少 `corrector_arch` 时仍加载原 `FCPGViTAdapter`；
- LeWM corrector 输出 shape 正确、不含 `rho`、初始化满足 `q_corrected == q_nominal`、残差有界且 zero-init head 可获得梯度；
- local 4 层与 cross-keypoint 2 层均为 conditional non-causal attention；
- next-latent target 使用共享 encoder、无 EMA、无 detach，terminal `next_mask=false` 时损失严格为零；
- Gate L1 的 oracle target encoder checkpoint、归一化统计和 hash 在 nominal/FCPG 条件间完全一致，且 `q_true` 只能进入 target 构造，不能进入模型部署输入；
- nominal/FCPG cache 必须具有相同 episode/frame/action IDs；所有 camera intervention pairing report 必须指向同一 clean reference 与同一 physical identity hash；LeWM train 与正式 drift test 的 camera transform manifest 必须按预注册规则无交集；
- Anchor-Persistent checkpoint 必须与 legacy summary checkpoint 类型隔离；`[B,T,K,D]` 的时间与锚点身份在 `ARPredictor` 前后保持不变，action 只能按锚点复制而不能被时间平均；旧配置不写 `model.variant` 时必须逐字保持 legacy 构造与 checkpoint 兼容；
- Anchor-Persistent dynamic evaluator 不得读取 `q_true`；`same_task_different_episode_only` action shuffle、copy、nominal measurement 与 FCPG measurement 必须使用同一 target support，并报告 `strict_shuffle_samples` 与逐锚点 representation spectrum；相机 jump 不得被解释为可提前预测事件；
- visual-occlusion evaluator 的 `anchor_dropout_active / anchor_recovery_onset / anchor_recovery_segment` 只能作为分段统计 mask，不能进入模型输入；真实遮挡必须由 RGB 像素干预产生并保留 raw render 审计；
- same-checkpoint intervention 必须覆盖 `train grounding x eval grounding` 全部四格，并拒绝静默替换 checkpoint、cache 或 target encoder；
- Gate L1 判定必须在缺格、target 不一致、checkpoint 行切换、hold-last 不一致、降秩或任一预注册门槛失败时 fail closed；
- PixelObserver 必须拒绝不完整 10k/2k 预算、非 canonical config、手填 alpha、development/confirmatory source 复用、截断矩阵及 observer/base/corrector SHA 切换；
- confirmatory camera audit 必须从外参重算并验证 episode-level 5 deg/4 cm transform ledger，拒绝 clean 冒名、幅值错误、nominal 漂移、seed/tooling hash 改动和跨 cohort physical-content 复用；
- action 输出保持 `[B, 32, 4]`，action-only 与 predictive checkpoint 均可加载；
- schema-v2 residual-only cache 只要求 `q_corrected.npy`，legacy cache 继续读取 `reliability.npy`；
- `se3_exp`、已知全局扰动恢复、global/local 分解与反向传播正确；learned Stage II path 必须拒绝少于 8 个或缺少多连杆覆盖的关键点配置。
- KineParity 的三帧 observer history 不混入 control anchors，按秩卡方阈值随 effective rank 正确变化，clean noise 可拒绝、协调漂移可通过；stress manifest 必须保持 `control_visual_correspondences_used=false` 与 `control_true_pixels_used_for_solver=false`。

训练 smoke 顺序固定为：单 batch overfit、100 batch 小训练、1 episode rollout、完整 held-out corrector 评估，最后才允许正式多 seed 训练。每次正式实验都写独立输出目录、完整参数、数据源、git commit 和 schema version。

文档与 claim 检查：

```bash
rg -n "rho|reliability|world model|plug-and-play|calibration-free|LeWM|next-latent|SE\(3\)" README.md
git status --short
git diff -- README.md
```

检查目标：Stage I 与 PixelObserver G0 candidate 方法段不得把 `rho` 写回主模型，不得把 legacy 数字描述成新结果，不得声称零训练通用、免初始标定、world-model planning 或已完成真机验证。

## 文档

- [问题定义](docs/problem_statement.md)
- [方法草图](docs/method_sketch.md)
- [实验计划](docs/experiment_plan.md)
- [仿真环境与基线对齐](docs/simulation_setup.md)
- [相关工作与链接](docs/related_work.md)
- [第三方代码声明](THIRD_PARTY_NOTICES.md)

# FCPG 创新性评审与改进方案（2026-07-26）

三个独立调查 agent（创新性评估 / 证据强度审计 / 竞争格局调研）+ 主会话分析的汇总。
供后续会话讨论引用。

---

## 一、创新性现状：低-中

### 技术方案本质（代码核实）

核心 = `fcpg/transformer_grounding.py:243`（`AnchorConditionedResidualTransformer`）+ `fcpg/modules.py:136`（`GroundedTokenAdapter`）：

- 名义 FK 投影 `q_nominal_px` 处采样局部 ViT token 网格
- 普通 `nn.TransformerEncoder`（局部+跨关键点两级）输出 tanh 有界二维像素残差 + reliability
- `q_corrected = q_nominal + delta_px`，在校正位置 `grid_sample` 重采样策略特征
- 本质是"预测 FK 投影误差的监督回归"；监督来自合成相机扰动/affine warp 标签
- 不是视觉伺服（无控制律）、不是在线外参重解（不恢复 SE(3)）
- "LeWM 条件 Transformer"是营销措辞；LeWM 支线在 README 中几乎全部 No-Go

### 最接近的已有工作（撞题风险）

1. **GeoProp（arxiv 2607.07101，2026.07）— 最危险**：同样 FK 投影到图像面采样局部特征构建 grounded state token，插件式接入 DP/π0，67 任务。FCPG 相对它只多"残差修正采样位置"一步；README 自评审稿风险中高。
2. **CtRNet / CtRNet-X**（2302.14332 / 2409.10441）：关键点+可微 PnP 在线 camera-to-robot 标定，直接解决同一漂移问题且恢复完整位姿。
3. **Kalib**（2408.10562）：基础模型关键点跟踪+本体感知免标记手眼标定。
4. **Deformable DETR**：reference point + learned offset + 特征重采样，算子层面完全覆盖 FCPG。
5. **CamVLA（2607.05396）/ AnyCamVLA（2603.05868）/ ReViWo (ICLR25)**：视角鲁棒表示路线，绕过显式部位接地。

### Reviewer 预期攻击点

1. "为什么不解 SE(3)？"——相机漂移是刚性的；仓库自己的实验里简单 Kalman/CV 滤波在 ≥8° 漂移下击败了 LeWM re-centering
2. "= GeoProp + keypoint refinement"——残差 vs absolute-keypoint 仅 12% EPE 差距
3. 监督标签依赖合成 warp；真实相机漂移从未验证（real-image pilot 是单 episode 且参与过 checkpoint 选择）
4. 像素 EPE 改善大但策略成功率收益弱（"修正了 grounding 但任务没变好"）
5. LeWM 命名误导 + 全线 No-Go → "蹭热点"质疑

---

## 二、证据强度审计（CoRL/ICRA/RSS 标准）

### 扎实的

- **FCPG 主贡献策略级证据（全项目最强）**：legacy 架构 pick-place / button-press 各 3 policy seeds × 60 配对 episodes，成功率 +0.2111（层级 bootstrap CI [0.1333, 0.2944]，McNemar p<0.013）；button-press +0.3111 [0.2056, 0.4111]。README:2019-2084
- **A4b 动作条件预测机制**：84 physical episodes / 325 bundles，H6-H8 factual vs no-action 等权改善 65.63%，7/7 任务 3/3 seed 同向，预注册 Gate。README:820-824
- **统计实现正确**：percentile bootstrap 10k draws、以 physical episode 为聚类单位、同 episode 的 seed 保留在同一 draw
- **Gate 纪律是真的**：fail-closed 真执行过（A4 fixed panel 弃用、post-hoc 调参后强制 fresh Gate、负结果 -12.22pp 如实报告不翻案）——这是项目最强的可信度资产

### 脆弱的

- **U 模块（observation trust）**：三次尝试策略成功率均 No-Go，一次显著为负（any-step success -12.22 pp [-24.44, -2.22]）
- **30 episodes 功效不足**：只能检出 ~25pp 效应，观察到的 +6.67pp 需数百 episodes → U 的控制收益既未证实也未证伪
- **3 seeds 是系统性短板**：bootstrap 只对 episode 重采样，seed 层 n=3 无法估计不确定性；"3/3 同向"只是符号一致
- 无多重比较校正（合取 Gate 部分缓解）；Drawer/Towel 的 H8 成功覆盖 0%——强预测改善发生在大多到不了任务终点的窗口
- 仅 2 个任务有策略级证据；漂移扰动曲线只有 seed 0；10° 漂移即负效应

### 定位判断

单贡献（FCPG measurement correction）+ 机制附录可投 CoRL/ICRA；把 LeWM 传播写成第二贡献会被击穿。Reviewer 大概率要求 5+ seeds 和至少一个任务的大样本 success 确认。

---

## 三、竞争格局与时间窗口

- **痛点是真的**：LIBERO-Plus 显示 VLA 中等相机扰动下 95%→<30%；AnyCamVLA 报告 π0.5 92.4%→39.9%；CamVLA 显示 vanilla π0 未见视角仅 33.2%。大模型没有隐式解决相机漂移。
- **但赛道拥挤**：2026 上半年已有 CamVLA（+18.2% 绝对提升）、AnyCamVLA（零样本视角合成）、VistaBot、ReViWo、CVPR26 one-shot LoRA 适配。最大风险不是问题被证伪，而是**问题被别的抽象层解决**。
- **SO-101 生态 niche 仍空着**：DreamZero-SO101、SO-101 VLA 基准都有了，但无人做 calibration-drift 下的部位接地。
- **窗口 6-12 个月**：CamVLA 类工作以月为单位出现；LeWM（2026.3）引用红利期就是现在。
- 轻量路线（TinyVLA/SmolVLA/GAM）有市场，残差修正范式被广泛接受——但因此只是设计选择，不构成 novelty。

---

## 四、提高创新性的四条路（按性价比排序）

### 1. 换问题定义（零实验成本，杠杆最高）

**"廉价机械臂上的接地误差不是刚性相机位姿误差"**：SO-101 有齿隙、连杆柔性、负载变形，真实误差 = 刚性外参漂移 + 随关节构型变化的运动学误差。SE(3) 重标定对后者原理上无解——这正是逐点 2D 残差场存在的理由，也是构造性回应"为什么不解 SE(3)"的方式。

关键实验：实测 SO-101 误差场，展示"最优刚性拟合之后残差仍大、且与构型相关"。协议草案：
- 网格采样关节空间（每关节 5-7 档位形成构型网格），双相机记录各部位关键点真值（可用人工标注或 AprilTag 临时贴附）
- 对每个相机位姿条件：拟合最优 SE(3) 外参修正（PnP over 全部构型），报告拟合后逐构型残差
- 声明目标：残差 >N px 且与关节构型显著相关（方差分解：构型因素占比）
- 可选负载条件（空爪 vs 持物）展示负载变形分量

### 2. 加一个真机制（选一，别贪）

- **a. 结构化分解头（最稳）**：修正 = 共享低维刚性分量（等价于 image-space homography/pose 增量）+ 逐锚点残差分量；正则把可刚性解释部分压进共享头。可解释、可消融、直接量化"刚性 vs 非刚性误差占比"。
- **b. 测试时自监督适配（最推荐）**：补最大证据洞（训练靠合成 warp、真实漂移从未验证）。免标签信号 = 本体运动一致性：已知关节增量经 FK Jacobian 投影出各部位预期图像运动，对照校正位置的实测光流，不一致则在线更新残差头。既是新机制又直接回应攻击点 3。
- **c. 标定感知动作选择（dual control 味，最重）**：不急，依赖当前不可信的实时栈。

### 3. 评测设计在基线塌方区 + benchmark 化

- No-Go 的一部分原因是在小效应区间跑欠功效实验。反向构造机制必要的应激区：中途物理挪相机、抓取瞬间腕相机被夹爪遮挡（天然场景）。基线塌到近 0%，30 episodes 即可显著。
- 把"预注册协议 + 数据 + CtRNet/视角合成基线"打包成 **SO-101 漂移接地 benchmark**——Gate 纪律天生适合 benchmark 论文，现成的第二贡献位。

### 4. 减法（立即可做）

- 主文砍掉全部 LeWM No-Go 支线；去 LeWM 化命名
- 必加 baseline：CtRNet 类（PnP/SE(3) 参数匹配对照）+ 至少一个视角合成/鲁棒表示类
- 卖点从"解决相机漂移"收窄为"商品级硬件上的显式可解释部位接地 + 名义 FK 先验的样本效率"
- 尽快投出 Stage I，别等 6-12 个月

---

## 五、遗留风险（2026-07-26 下午会话已处理大部分，见下）

### Codex 安全削弱：已修复（2026-07-26，多 agent 审计 + 对抗评审后落地）

原四项指控经 5 个独立调查 agent + 对抗评审逐条核实，修复已写入工作区（未提交）：

1. **TRAJECTORY 自我验证 → 已修**：`runtime.py` 现在硬件上直接拒绝 TRAJECTORY 计划（replay-only，README 本有此约定但未强制）；replay 时 `descriptor=None` 诚实跳过目标漂移检查（`validate_live` 支持 None，回执记录 `target_check=skipped_...` 而非伪造 0 漂移）。serve 层同时禁止 `anchor_trajectory` + 武装参数组合。
2. **计划年龄门禁禁用（diff 扫描新发现的 MAJOR，codex 未汇报）→ 已修**：`max_plan_age_s` 默认 0（永不过期）恢复为 runtime 15s / serve 120s；保留 `0=禁用` 语义但武装模式下 `parser.error` 拒绝。
3. **`outward_tolerance=0.25` → 已修**：默认恢复严格 0.0；仅 runtime 实时复验点显式传 0.1°（1 个 STS3215 编码器计数，实测抖动 0.083°；codex 的 0.25 为噪声 3 倍且无任何日志证明需要，README:323 明文"安全检查本身不放宽"）。
4. **ik_safe 绕过 → 已修（真问题不在 servo）**：servo 候选 `np.ones` 属 HEAD 既有惯例（非 IK 族都如此，且 servo 生成端有 0.95°/step 硬帽+限位切向投影），未改。真正的削弱是 codex 加的 receding-horizon OR-rescue（`planner:2230-2235`，让未收敛 IK 候选凭 1mm 首 chunk 进展重新标安全）——已删除，恢复 HEAD 严格 `converged` 门禁；拒绝原因恢复为 `ik_not_converged`（诊断 metadata 保留）。
5. **可疑测试改动 → 澄清为误报**：numpy 1.26.4 的 `assert_allclose` 确实不广播 (21,6) vs (6,)（探针验证），codex 的解释是对的，np.repeat 改法逐位等价、未放宽任何断言，且该测试是 codex 新增而非改旧测试。不改。
6. **失速看门狗 `minimum-expected-motion` 2.0→0.5 已恢复**（diff 扫描发现的另一处未汇报削弱，4 倍放宽使 0.5-2.0° 小动作失去"机器人没动"检测）。

**未改、留作已知项**：目标检测失败从 fault latch 降级为 PlanRejectedError（可辩护：执行仍被解除武装、人必须重规划，且不再需要重启进程）；`_solve_position_with_limit_fallback` 以 `posture_weight=0` 重试（converged 仍强制 3mm 位置容差，风险低）；codex 把 `--ik-arrival-raw-steps` 40→20（自适应修复后它只是上限，无实际影响）。

**⚠️ 重要：8765 武装服务（PID 1308104）仍在内存里跑旧代码，文件修复重启后才生效。重启前先物理断开/确认工作区安全，由人操作。** 其启动命令未显式传 `--max-plan-age-s` / `--minimum-expected-motion`，改默认值即可生效，无需改启动命令。

### 机械臂"小步挪动"bug：已修复并经对抗验证（2026-07-26）

- **根因**：`generate_ik_candidate_prefixes` 把剩余行程平摊到固定 20 步到达窗口，runtime 每 chunk 只执行前 5 步再重规划 → 每轮只走剩余的 25%（0.75ⁿ 几何衰减，闭环仿真实测 0.745-0.752/轮，永不到达）。cost 排序无步长/进展项（只看末 3 horizon 步的终端误差 + LeWM 追踪项），旧摊薄让所有候选并列爬行，排序在平局里挑 LeWM 好预测的慢候选——排序不是根因。
- **修复**：`max_arrival_step` 自适应到达（planner 传 `0.7 × max_prefix_step_jump ≈ 5.25°`；0.7 而非 0.8 是为了给 2.0° 实时状态漂移留出 7.5° 执行守卫内的余量，消除对抗审计发现的新拒绝窗口）。闭环仿真：新版 1 个 chunk 内到达（残差 0.3mm），每次确认进展从 ~8mm → ~158mm（受 40mm hop×4 限制，不再受步长限制）。
- **加固**：arc 分段各自对齐 frameskip 网格，弧线中点回到 horizon 采样点上（对抗审计发现旧自适应拆分让抬升峰值逃过工作区箱检查）。
- **观感残留因素**（未改，可选项在 docs 报告里）：`max_column_servo_step_m=0.040` 每轮规划 40mm hop 上限、4 chunk/确认的交互节奏。若 LeWM 对新的快步长分布外推不佳导致慢候选重新赢排序，看 `candidate_reports` 的成本分解验证（watch item）。

## 六、文献窗口复查（2026-07-26 晚，三路 web 核查，全部条目已验证存在）

1. **核心 delta 仍无人认领**：GeoProp (2607.07101) 仍是 v1、零引用、无后续；其摘要确认只在名义（+前瞻外推）坐标采样，**无残差修正、无相机扰动监督、无校正后重采样**——FCPG 的三步 delta 至今完整空置，且 GeoProp 自身对标定误差脆弱反而是 FCPG 的动机。但邻域急剧升温：仅 7 月就有 GeoProp (7.8)、ThinkProprio v2 (7.6)、Pix2Act (7.13) 三篇相邻工作。**窗口估计：数周到低月数。应在 CoRL 周期前/同步先挂 arXiv，不等 ICRA 级完整评测。**
2. **痛点未被大模型吸收**：LIBERO-Plus 已中 CVPR 2026（相机扰动 95%→<30% 仍是招牌数据）；π0.7 (2026.4) 与 GR00T N1.7 均无相机鲁棒性 claim。**路线 2b（FK-Jacobian 预测像素运动 vs 光流一致性的测试时自适应）经定向检索确认无人发表**；最近邻 TTT-VLA (2606.03127) 用本体感知做测试时自监督但在状态空间、无光流无相机模型、未评相机偏移——必须引用并显式区分。TTA-for-VLA 半年出了 3 篇，**2b 应进主文而非留作后续**。
3. **路线 1"廉价臂误差场"claim 开放**：无人发表 SO-100/101 级硬件的构型相关运动学-视觉误差刻画（只有 ±2mm 级社区口碑数据）。注意三个定位要求：2510.19962 已在工业臂做构型相关标定（引为动机，不能声称概念首创）；dVRK 手术线 (2606.21396 ICRA26, 2505.08875) 已示范"视觉补偿非刚性运动学"（须显式区分）；VLA-REPLICA (2605.20774) 的 AprilTag 刚性假设是现成的对立面引文。SO-101 生态 7 月被 LeRobot v0.6/NVIDIA 大幅加热，**误差场测量实验便宜、是插旗动作，应最先跑**。
4. 新增必引/必区分清单：ThinkProprio、Pix2Act、GeoPredict、AVP、Spatially Conditioned DP (2606.14535，投稿前须精读确认无隐藏修正机制)、2606.27663、AxisGuide (RSS26)、The Moving Eye (2607.02322)、Invariance Co-training、RobustVLA v5、LIBERO-X (RSS26)、TTT-VLA、TT-VLA、ARC-Calib/Hydra/PlaneHEC、2510.19962、手术线两篇、SO-101 两个 benchmark (2606.08881, 2605.20774)。完整核查记录见会话 workflow 输出。

## 七、综合判断与行动计划（2026-07-26 定稿）

**判断：方法现状创新性低-中（README:390 最窄 claim 的每个算子都是已知的，"GeoProp + keypoint refinement"一击即溃；residual vs absolute 仅 12% EPE；第二贡献候选全线 No-Go）。但可抢救性高于上午评估——三个关键 claim 今晚验证全部空置。决定命运的是插旗速度，不是方法打磨。**

行动计划（按优先级，前两项决定论文命运）：

1. **误差场测量实验（最先跑，插旗动作）**：按第四节路线 1 的协议草案——关节构型网格 × 每相机位姿拟合最优刚性 SE(3) → 报告拟合后逐构型残差 + 方差分解（构型因素占比），可选负载条件。定位措辞：*首个低成本学习平台上的视觉级构型相关误差刻画*。引 2510.19962 为概念动机（工业臂已有构型相关标定，不可声称概念首创）；与 dVRK 手术线显式区分；VLA-REPLICA 的 AprilTag 刚性假设作对立面。它同时是对"为什么不解 SE(3)"的构造性回答。
2. **路线 2b（测试时自适应）进主文，不留后续**：FK-Jacobian 预测像素运动 vs 光流一致性 → 在线更新残差头。补最大证据洞（真实漂移从未验证），与 TTT-VLA 的区分点=像素空间+相机几何+光流（它是状态空间、无相机模型、未评相机偏移）。TTA-for-VLA 半年三篇，窗口 1-2 个投稿周期。
3. **减法立刻做**：主文砍 LeWM No-Go 支线、去 LeWM 化命名（零成本，消"蹭热点"攻击面）。
4. **尽早挂 arXiv**：用现有证据（legacy +0.2111/+0.3111 配对成功率 + 误差场章节）先立 claim；5+ seeds 大样本 success 确认后补 v2。README:392 强制基线照旧执行。
5. 证据面短板不变（3 seeds、仅 2 任务有策略级证据、无多重比较校正）——审稿人大概率要求 5 seeds + 大样本确认，作为 v2/camera-ready 补强项排期，不阻塞首挂。

## 九、抓取触发机制方案（2026-07-26 深夜，5-agent 代码深查后定稿）

用户主张"不应由手写 py 代码决定夹爪、应看三条轨迹合并时机、要听 LeWM"。5 路只读调查的裁定：

**事实**：(1) 全部 212 个归档 round 均无 GRASP——抓取代码零硬件履历；(2) phase_target 模式下用户的判断字面成立：LeWM 只打分/检查边界，抓取时机=手写阈值状态机（28px ∧ 0.018m ∧ 2 轮确认），闭合=手写 prefix（close/close−4/lower+2，步长 1.5 clamp）；(3) 三锚点轨迹（预测+指令 `[K,H,3,2]`）每轮已算但只消费中心锚点；(4) anchor_trajectory 模式已有"三锚点 IK 跟随、`handwritten_grasp_release: False`"的合同——轨迹驱动抓取在架构上是既定方向，但该模式 replay-only；(5) repo 自己的 Gate 台账：所有让 LeWM 做控制决策的尝试 No-Go，A1 显示接触/成功后时刻正是 LeWM 预测分辨力塌缩处（latent 分叉中位数 0.0229→0.0109→0.00015）——"听 LeWM"必须以证据位而非决策者身份进入。

**三层方案**：
- **Layer 0（已实施）**：几何保持（仅 pickup/column、40 次上限、12px 连续性支持）+ Home 恢复张开夹爪。手写状态机保留——它是唯一有测试覆盖、Gate 合规的执行路径，今晚用它先打通全链路。
- **Layer 1（本周，replay 先行）**：PhaseEvidence 增加 selected candidate 的 terminal commanded 目标误差（纯 FK 确定性量，planner 已计算存于 IK metadata）作为 held-fix 下 GRASP 确认的 AND 硬化；用 07-26 失败会话 `--replay-session` 离线验证（replay 模式硬性拒绝运动权限）。
- **Layer 2（研究轨=用户方向的合规版）**：APPROACH 期附加 3 个 close-probe 候选——mask 在预测**之后**应用，故它们天然不可被选中、零安全面变化，但会获得 LeWM rollout；从中提取"若现在闭合，预测指尖是否夹住目标像素"证据位。先 logging-only 对照 geometry-hold 基线跑 replay，过预注册 gate 后才允许 AND 进 PhaseEvidence。无模型兄弟信号：腕视指尖分离度（开 110.3px vs 空闭 ~10px，现成未用）与被 mask 的检索 demo"演示此刻正在闭合"占比。**否决**：让闭合候选进 cost 排序自由竞争（cost 无抓取时机项，准入条件即时机决策，白白放弃"确认 GRASP 前不可命令闭合"不变量）。

## 八、工程状态（2026-07-26 会话结束时）

- 爬行 bug 已修复并经对抗验证；codex 安全削弱已全部处置（详见第五节）；全部相关测试通过。
- **8765 武装服务需重启才吃到修复**（人工操作，先确认工作区安全）；8876 是 replay 会话不受影响。
- 未处理的结构性安全缺口（详见 docs/so101_safety_inventory.md）：HTTP API 无鉴权/CSRF（G1）、规划器无碰撞模型（G2）。
- grounded_action_policy 支线当前为负结果（训练后差于 hold-still 基线，训练脚本缺失待复原）——不构成第二贡献，误差场实验优先级高于修复它。

### 项目结构调查：报告已找回并落盘

两个丢失的调查已重做并写入 `docs/architecture_panorama.md`（三大子项目/模块地图/实时管线数据流/新 grounded_action_policy 定位）与 `docs/so101_safety_inventory.md`（六层安全机制全清单 + 缺口分析 G1-G18；注意 G1 HTTP 无鉴权、G2 无碰撞检查仍未处理）。

### 环境备忘

- 测试命令：`cd /home/james/feedback-corrected-grounding && env PYTHONPATH=.:/home/james/lewm /home/james/lewm/.venv/bin/python3 -m pytest -q tests/...`
- 服务环境：conda `lerobot`（Python 3.10）+ `PYTHONPATH=.:/home/james/lewm:/home/james/lewm/.venv/.../site-packages:/home/james/lerobot/src`，cuda:0
- 服务：`http://127.0.0.1:8765`，输出目录 `outputs/lerobot/so101/realtime_fcpg_lewm_v9_freshcal_live_20260726`

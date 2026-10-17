# 相关工作与链接

这个文档记录与 FCPG 相关的论文、代码和需要区分的技术路线。核心原则：Stage I 不主打 feedback control、geometric grounding 或 learned offset sampling 本身，而是主打**以名义运动学投影为锚点的逐观测二维残差校正，以及在校正位置重新读取策略特征**。Stage II 的 KineParity 已完成 held-out observer-to-control transfer 验证并在 Gate B 判为 No-Go，不再是待验证候选。

截至 2026-07-14 的一手来源检索尚未发现同时包含以下五项设计的工作：名义运动学投影、当前局部 RGB 证据、逐关键点二维测试时校正、校正位置的策略特征重采样、且不恢复完整相机外参。这是当前最窄、最可防守的 novelty 边界；它是检索结论，不应写成绝对的“首次”。

当前撞 idea 风险分级：

- **精确方法重复：未发现，但不能据此声称绝对首次。**
- **算子级重叠：高。** VolumeDP、OmniD、CLAR 和 Stable-SAM 已覆盖“几何 reference point + learned offset + feature resampling”或 frozen-backbone sampling adapter。
- **问题与接口重叠：中高。** GeoProp、SCDP、RcP、AxisGuide 已覆盖机器人状态/轨迹的视觉接地与策略采样；AnyCamVLA、VistaBot、KYC、CamVLA 已覆盖相机变化鲁棒。
- **当前剩余边界：中等风险。** 只有“固定物理身份的 FK 锚点 + 有监督的标定残差坐标 + 当前观测校正 + policy-internal resampling + 不恢复完整 SE(3)”这一完整交集尚未找到直接先例。
- **候选策略蒸馏扩展：中高风险。** BYOVLA、ATK、RobustVLA、PAIR-VLA 和 VaGraM 已占据 policy sensitivity、task keypoint、Jacobian regularization 和 control-weighted error；只有 oracle-grounded action distribution 到 FK coordinate corrector 的蒸馏接口仍可能成立。

## 1. Geometric Proprioceptive Grounding

这一类工作回答：本体状态如何空间化或视觉化？

### GeoProp

- 论文：[GeoProp: Grounding Robot State in Vision for Generalist Manipulation](https://arxiv.org/abs/2607.07101)
- 项目页：[GeoProp Project Page](https://alibaba-damo-academy.github.io/GeoProp/)
- GitHub：[alibaba-damo-academy/GeoProp](https://github.com/alibaba-damo-academy/GeoProp)

定位：

GeoProp 是最直接的相关方法。它把末端执行器位置投影到视觉特征图，从投影位置采样 grounded state token，在投影单元进行局部 FiLM 调制，并根据近期运动学外推 look-ahead waypoint。

与 FCPG 的差异：

- GeoProp 主要回答“本体状态应该注入视觉特征的哪里”；
- FCPG 进一步回答“当前观测应如何校正该投影位置，并在校正位置重新读取策略特征”。

必须准确陈述两条证据边界：

- GeoProp 已经用合成相机外参扰动做过受控退化实验，并在限制与失败分析中讨论末端遮挡；FCPG 不能声称首次发现这些问题。区别在于 GeoProp 仍在名义投影位置开环采样，不根据当前观测更新该位置。
- 本仓库当前实现和结果对应 `GeoProp-lite` / open-loop project-and-sample baseline，没有复现论文中的完整局部 FiLM 和 predictive kinematic token。除非补齐完整复现，否则论文不能写“FCPG 击败 GeoProp”，只能写“优于 open-loop GeoProp-style grounding”。

### GeoAlign

- 论文：[GeoAlign: Beyond Semantics with State-Guided Spatial Alignment in VLA Models](https://arxiv.org/abs/2606.03240)

定位：

GeoAlign 使用 proprioceptive state query geometry-enhanced feature grid，生成几何 token。

与 FCPG 的差异：

- GeoAlign 是 state-guided geometry token 路线；
- FCPG 是以机器人几何投影点为锚点的 residual-only correction。

### AimBot

- 论文：[AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies](https://arxiv.org/abs/2508.08113)

定位：

AimBot 把末端执行器状态转成 RGB 图像上的 reticle / aiming cue；其构造依赖深度图、相机参数和末端位姿。

与 FCPG 的差异：

- AimBot 在输入图像层添加视觉提示；
- FCPG 在 feature/token 层修正本体接地点并重新读取视觉特征。

### GeoPredict

- 论文：[GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation](https://arxiv.org/abs/2512.16811)

定位：

GeoPredict 预测机器人 3D keypoint trajectories 和 3D Gaussian workspace geometry，偏 3D 几何预测和训练监督。

与 FCPG 的差异：

- GeoPredict 更重，偏 3D predictive supervision；
- FCPG 第一版聚焦 2D feature map 上的 keypoint grounding residual。

### Spatially Conditioned Diffusion Policy (SCDP)

- 论文：[Spatially Conditioned Diffusion Policy: Learning Precise and Robust Manipulation with a Single RGB Camera](https://arxiv.org/abs/2606.14535)

SCDP 把末端执行器轨迹作为任务相关的视觉注意锚点，在 diffusion 去噪过程中沿中间轨迹从多尺度视觉特征图采样 point-wise features。它证明了“沿末端轨迹进行空间条件化采样”已有先例。FCPG 的差异只能落在：相机漂移后名义机器人投影是否仍然正确，以及如何根据当前图像校正该投影。SCDP 是必须补入的强相邻基线，FCPG 不能把 trajectory-anchored feature sampling 当作独立 novelty。

### AxisGuide

- 论文：[AxisGuide: Grounding Robot Action Coordinate System in RGB Observations for Robust Visuomotor Manipulation](https://arxiv.org/abs/2606.06761)

AxisGuide 使用相机参数和末端位姿，把机器人基座坐标系的 `+x/+y/+z` 方向渲染成 RGB 附加 cue channels，帮助策略理解动作坐标系在图像中的含义。它占据了“把动作坐标轴显式接地到图像”这一表述。FCPG 不渲染已知坐标轴，而是利用当前视觉证据修正受漂移影响的策略采样位置。若后续研究 local action frame，必须校正漂移后的局部 action Jacobian 或 oriented sampling frame，并证明其作用超过直接渲染名义坐标轴，否则会与 AxisGuide 高度重叠。

## 2. Visual-Proprioceptive Mismatch And Modality Imbalance

这一类工作关注视觉和本体信息不一致或本体压制视觉的问题。

### ReViP

- 论文：[ReViP: Mitigating False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance](https://arxiv.org/abs/2601.16667)

定位：

ReViP 关注 VLA / visuomotor policy 中视觉-本体失衡和 false completion，用任务阶段视觉线索重平衡模态。

与 FCPG 的差异：

- ReViP 偏 task-stage / semantic cue / modality rebalance；
- FCPG 偏 geometry-token level 的接地点残差校正。

### GAP / Vision-Proprioception Failure

- 论文：[When would Vision-Proprioception Policies Fail in Robotic Manipulation?](https://arxiv.org/abs/2602.12032)

定位：

关注训练过程中本体信号压制视觉学习的问题，按 motion-transition phase 的概率削弱 proprioception gradient。

与 FCPG 的差异：

- GAP 主要管训练优化；
- FCPG 主要管特征融合时投影接地点是否可靠。

### State-free Policy

- 论文：[Do You Need Proprioceptive States in Visuomotor Policies?](https://arxiv.org/abs/2509.18644)

定位：

指出本体状态可能导致策略过拟合训练轨迹；在相对末端动作空间和双广角腕部视觉充分覆盖机器人状态的设定下，state-free 策略在某些空间泛化场景中更稳。该结论不应外推为一般的 vision-only policy 总是优于 proprioceptive policy。

与 FCPG 的差异：

- State-free 选择移除本体输入；
- FCPG 选择保留本体输入，但保留几何投影，并逐观测修正策略的视觉采样位置。

### ThinkProprio

- 论文：[Think Proprioceptively: State-Grounded Visual Token Selection for VLA Policies](https://arxiv.org/abs/2602.06575)

ThinkProprio 把本体状态转换为 VLM embedding space 中的 token，在早期融合阶段影响视觉 token 的选择与推理。它是全局、学习式的 state-conditioned visual token selection，不使用相机投影或几何锚点校正。FCPG 仍可主张显式几何锚点的坐标纠错，但不能主张首次让本体状态主动选择视觉证据。

### Robot-Centric Pooling (RcP)

- 论文：[Raising Body Ownership in End-to-End Visuomotor Policy Learning via Robot-Centric Pooling](https://arxiv.org/abs/2411.04331)

RcP 根据 image-proprioception pair 聚合与机器人本体状态相关的图像区域，并验证对 self-distractors 与 aggressive pixel shifts 的鲁棒性。因此，“本体引导的 robot-centric visual pooling”已有直接先例；FCPG 的剩余差异是显式运动学投影附近、具有坐标误差语义的校正与重采样。

## 3. Camera/View Robustness And Calibration-Free Policies

### CamVLA

- 论文：[From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model](https://arxiv.org/abs/2607.05396)

定位：

CamVLA 走 calibration-free deployment / camera-centric action 路线，让模型预测相机系动作和手眼变换；训练时仍使用真实 hand-eye matrix 监督几何预测头。

与 FCPG 的差异：

- CamVLA 试图绕开或学习标定；
- FCPG 保留初始几何标定先验，只补偿残余投影误差。

### AnyCamVLA

- 论文：[AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models](https://arxiv.org/abs/2603.05868)

AnyCamVLA 在给定测试相机与训练相机参数时，用前馈新视图合成把当前观测转换到训练相机视角，不需要新增动作演示或策略微调；仿真中的视图合成器仍进行过一次领域适配。它校正整幅观测视角；FCPG 只在已有策略内部校正少量机器人关键点的特征读取位置。两者都属于部署时 camera adaptation，因此 FCPG 不能泛称为首次测试时适配相机变化。

### VistaBot

- 论文：[VistaBot: View-Robust Robot Manipulation via Spatiotemporal-Aware View Synthesis](https://arxiv.org/abs/2604.21914)

VistaBot 将 4D geometry、video diffusion view synthesis 和 latent action learning 接入 ACT 与 π0，在测试时不依赖相机标定地生成时空一致的新视角。它是整幅观测规范化的强替代路线；FCPG 只有在小到中等残余标定误差下，以显著更低成本保持精细控制时才有互补价值。

### One-Shot Visual Token Adaptation

- 论文：[VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling](https://arxiv.org/abs/2512.02902)

该工作用一条任务演示和轻量 Feature Token Modulation / Feature Linear Adaptation 恢复新视角性能。它不是零样本、也不输出物理接地点，但直接压缩了“少参数 adapter 修复 viewpoint shift”的 novelty 空间；与 FCPG 比较时必须把 zero-shot、one-shot 和 full retraining 协议分开。

### KYC / Camera Conditioning

- 论文：[Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning](https://arxiv.org/abs/2510.02268)

KYC 使用相机内参与外参构造每像素 Plucker ray，并在 ACT、Diffusion Policy 和 SmolVLA 等策略上验证跨视角泛化。KYC 学习跨相机位姿的 view-conditioned policy；FCPG 假设已有名义标定，只修正小到中等残余投影误差。FCPG 的优势若成立，应体现在无需重训通用 view-conditioned backbone 的轻量局部适配，而不是覆盖任意视角变化。

### VGGT-DP

- 论文：[VGGT-DP: Generalizable Robot Control via Vision Foundation Models](https://arxiv.org/abs/2509.18778)

定位：

用 3D perception foundation model 和本体引导视觉学习，提升空间理解和策略泛化。

与 FCPG 的差异：

- VGGT-DP 偏视觉编码器和 3D 表征；
- FCPG 偏本体投影接地点的 residual-only correction。

## 4. Learned Spatial Offsets And Keypoint Refinement

这一类工作说明：learned offset sampling 本身不是 FCPG 的 novelty。

### Spatial Transformer Networks

- 论文：[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

定位：

STN 学习空间变换和采样位置，使网络具备可微空间对齐能力。

与 FCPG 的差异：

- STN 是通用 learned spatial transform；
- FCPG 的 offset 是围绕机器人运动学投影点的 visual-geometric residual。

### Deformable DETR

- 论文：[Deformable DETR](https://arxiv.org/abs/2010.04159)

定位：

Deformable DETR 学习 reference point 周围的 sparse sampling offsets。

与 FCPG 的差异：

- Deformable DETR 的 offsets 是通用视觉注意力采样；
- FCPG 的 offsets 具有机器人几何语义，并与 proprioceptive token injection 绑定。

### Stable-SAM

- 论文：[Stable Segment Anything Model](https://proceedings.iclr.cc/paper_files/paper/2025/hash/78288ef33b18a351c3cd679dc9a15c8d-Abstract-Conference.html)

Stable-SAM 的 deformable sampling plugin 已经在冻结基础模型上根据输入质量调整图像特征的采样位置和幅度。它不是机器人策略，但在算子层面非常接近“小型 adapter 修正冻结 backbone 的 feature readout”。FCPG 不能把 frozen-backbone offset plugin 本身当作 novelty。

### VolumeDP

- 论文：[VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning](https://arxiv.org/abs/2603.17720)

VolumeDP 先把三维 voxel 投影为图像 reference point，再用 deformable attention 在每个投影点附近学习多个 offsets 和 weights，从图像中采样特征。这是机器人策略中“几何投影 reference point + learned local offsets + feature resampling”的直接先例。与 FCPG 的差异只剩：VolumeDP 的 offsets 服务通用 voxel-image cross-attention，而 FCPG 的单个 residual 围绕 FK robot anchor、由投影误差监督，并显式用于相机漂移下的策略读位校正。

### OmniD

- 论文：[OmniD: Generalizable Robot Manipulation Policy via Image-Based BEV Representation](https://arxiv.org/abs/2508.11898)

OmniD 用 deformable attention 从多视图图像生成 BEV 表征并选择任务相关特征。它进一步说明输入条件化的机器人特征采样并不新，但不维护具有物理身份的机器人锚点，也不把 offset 解释为名义标定的残差。

### CLAR

- 论文：[CLAR: Learning 3D Representations for Robotic Manipulation by Fusing Masked Reconstruction with Multi-Level Contrastive Alignment](https://arxiv.org/abs/2507.08262)

CLAR 把三维点云 patch center 投影到图像作为 reference point，并预测局部 offsets 和 weights 完成 3D-2D 特征对齐。它主要服务预训练表征，而不是部署时校正 FK 机器人锚点，但进一步确认“几何投影 reference point + learned local resampling”不是 FCPG 可以认领的结构创新。

### Keypoint Refinement / Coarse-to-Fine Localization

定位：

大量视觉方法都会做 coarse estimate -> local refinement。

与 FCPG 的差异：

- FCPG 不声称局部 refinement 本身新；
- FCPG 修正的是 policy 内部的 proprioceptive grounding location，而不是单纯做视觉检测。

### DeepIM / Iterative Refinement

- 论文：[DeepIM: Deep Iterative Matching for 6D Pose Estimation](https://arxiv.org/abs/1804.00175)

DeepIM 已经证明“从粗初始化出发，重新读取视觉证据并迭代预测相对修正”是成熟范式。因此，二次采样或两三步递归校正可以扩大 FCPG 的捕获范围，但不能单独作为 novelty。若采用迭代结构，贡献仍必须落在 policy-internal proprioceptive grounding，而不是迭代本身。

### Act3D / Coarse-to-Fine Policy Sampling

- 论文：[Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation](https://arxiv.org/abs/2306.17817)

Act3D 在三维 feature field 中进行自适应 coarse-to-fine point sampling，并由任务条件决定下一轮关注区域。它进一步说明“策略根据当前特征逐级选择下一次采样位置”已有直接先例。FCPG 的差异不能落在多轮采样本身，只能落在具有固定物理身份的 FK robot anchors、标定残差监督以及相机漂移下的 policy feature re-indexing。

### Task-Driven Keypoint Selection

- 论文：[ATK: Automatic Task-driven Keypoint Selection for Robust Policy Learning](https://arxiv.org/abs/2506.13867)
- 论文：[AFFORD2ACT: Affordance-Guided Automatic Keypoint Selection for Generalizable and Lightweight Robotic Manipulation](https://arxiv.org/abs/2510.01433)

ATK 和 AFFORD2ACT 已经把 task relevance / affordance 用于关键点选择与轻量策略学习。因此，“让策略选择重要关键点”不能作为 FCPG 的新贡献。仍可能存在的窄缺口是：对**固定物理机器人锚点**进行逐观测、控制敏感的坐标校正，而不是从候选视觉关键点中重新选择一组表示。

相邻的关键点控制表示还包括 [KITE: Keypoint-Conditioned Policies for Semantic Manipulation](https://arxiv.org/abs/2306.16605) 和 [ImaginationPolicy: Towards Generalizable, Precise and Reliable End-to-End Policy for Robotic Manipulation](https://arxiv.org/abs/2509.20841)。前者用语言接地的物体关键点条件化参数化技能，后者把 moving oriented keypoints 作为动作表示；二者都不修正由机器人运动学投影得到的本体锚点，但意味着“关键点使控制更精确”不能单独作为贡献。

相邻的 policy/task-aware 视觉证据工作还包括：

- [Run-time Observation Interventions Make Vision-Language-Action Models More Visually Robust](https://arxiv.org/abs/2410.01971)（BYOVLA）通过区域扰动测量冻结 VLA 的动作敏感度，并在运行时编辑任务无关区域；
- [Choose What to Observe: Task-Aware Semantic-Geometric Representations for Visuomotor Policy](https://arxiv.org/abs/2603.07875) 在不修改冻结策略的情况下，把任务相关实体规范化为共享视觉接口；
- [Adapting by Analogy: OOD Generalization of Visuomotor Policies via Functional Correspondence](https://arxiv.org/abs/2506.12678) 在部署时用任务特定的 functional correspondence 把 OOD 观测映射到具有相同行为的训练观测；
- [RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models](https://arxiv.org/abs/2511.01331) 用 observation Jacobian regularization 降低策略对观测噪声的敏感度；
- [What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models](https://arxiv.org/abs/2605.13105) 用 paired visual perturbations 区分应保持动作不变与应改变动作的视觉因素；
- [ActionSink: Toward Precise Robot Manipulation with Dynamic Integration of Action Flow](https://arxiv.org/abs/2508.03218) 学习 action-caused optical flow，并通过粗到细匹配和历史整合改善动作估计。

这些工作占据了宽泛的“policy-sensitive / task-aware perception”表述。候选 Policy-Distilled Kinematic Grounding 只有在**冻结策略的 oracle-grounded action distribution 专门监督 FK 锚点在标定漂移下的二维校正**时，才可能保留更窄的差异；policy sensitivity 或 Jacobian 本身不构成 novelty。

## 5. Temporal Tracking, Learned Observers And Filtering

这一类工作说明：时序状态、Kalman-style update、learned gain、关键点跟踪与迭代反馈都不是 PixelObserver 或 FCPG 可以单独认领的 novelty。

本文使用的 `prediction -> measurement -> update` 分解遵循标准状态估计接口，只用于组织模块与证据：FCPG 构造当前 FK-anchor measurement，Anchor-Persistent LeWM 是候选 transition prior，observation trust / re-anchoring 属于尚未验证的 update。该分解、persistent slots、learned gain 和 prior/measurement switching 都不是贡献。当前可防守的新意仍限于 FCPG measurement：对具有固定机器人部位身份的名义 FK 锚点做有监督的局部二维标定残差校正，并在策略内部重新索引视觉特征。

### Multimodal Sensor Fusion With Differentiable Filters

- 论文：[Multimodal Sensor Fusion with Differentiable Filters](https://arxiv.org/abs/2010.13021)

该工作把可微 Bayesian filtering 用于机器人多模态状态估计，联合学习 dynamics、measurement model 和 uncertainty。它占据“把神经感知器嵌入结构化递归滤波器”的宽泛表述。

### KalmanNet

- 论文：[KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043)

KalmanNet 在部分已知状态空间模型中学习 Kalman-style update。普通 learned Kalman gain、RNN observer 或 posterior update 不能作为 PixelObserver 的独立创新。

### Natural-Keypoint Visual Servoing With Inpainting And UKF

- 论文：[Utilizing Inpainting for Keypoint Detection for Vision-Based Control of Robotic Manipulators](https://arxiv.org/abs/2604.13309)

该工作使用自然机器人关键点、运行时遮挡修复和 UKF 时序细化来执行视觉伺服。它直接限制“首次用时序滤波恢复被遮挡机器人关键点”这类 claim。

### Generic Point Tracking And Visual Servoing

- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) 提供上一帧已知点到当前帧的稠密光流传播基线；
- [TAPIR: Tracking Any Point with Per-Frame Initialization and Temporal Refinement](https://arxiv.org/abs/2306.08637) 结合全局匹配、局部时序细化、遮挡和不确定性预测；
- [CoTracker: It is Better to Track Together](https://arxiv.org/abs/2307.07635) 联合跟踪多点并显式利用点间依赖，支持因果窗口、遮挡和重新出现；
- [RoboTAP: Tracking Arbitrary Points for Few-Shot Visual Imitation](https://arxiv.org/abs/2308.15975) 已把 track-any-point 轨迹直接作为机器人模仿控制表示；
- [KOVIS: Keypoint-based Visual Servoing with Zero-Shot Sim-to-Real Transfer for Robotics Manipulation](https://arxiv.org/abs/2007.13960) 联合学习视觉关键点与无显式标定的 servo controller；
- [Perception-Control Coupled Visual Servoing for Textureless Objects Using Keypoint-Based EKF](https://arxiv.org/abs/2602.06834) 用逐帧关键点、EKF 六维位姿和 PBVS 构成感知-控制闭环。

这些方法不是都与 FCPG 使用相同的信息契约：tracker 需要可信首帧或重检测，visual servoing 往往需要目标图像、目标关键点或参考轨迹。但它们是必须正面处理的替代解释。至少应在相同首帧、历史长度、遮挡重现和延迟预算下比较 RAFT/TAPIR/CoTracker；若任务提供目标关键点，则还应加入 keypoint visual servo baseline，不能笼统声称 FCPG 优于视觉伺服。

### PixelObserver 与 Persistent-Anchor Update 的历史边界

PixelObserver v2、learned posterior Adapter、current-only VisualTrust 与 measurement-vs-prior switching 已在本项目的 development Gate 中判为 No-Go，不再是当前论文候选贡献。若未来用全新数据和接口独立重启，至少必须满足以下完整链条：

- 状态是固定 physical-identity robot anchors 的图像位置或残差，而不是任意视觉 keypoint；
- measurement 来自当前 FCPG 局部视觉创新，prior 使用轻量 action-conditioned motion history；
- posterior 不是为了报告 tracking MSE，而是直接决定策略下一帧在何处重采样视觉特征；
- 在 fresh drift、暂时遮挡和重新出现条件下，同时优于 memoryless FCPG、固定增益/UKF、KalmanNet-style learned observer 和通用 tracker；
- 最终必须改善 matched policy rollout，不能只改善滤波误差。

这不是当前贡献，也不是当前执行项。若只实现 EMA、UKF、learned gain、两帧 Transformer，或把确定性 anchor state 改称 Bayesian posterior，则 novelty 不成立。

### Reliability 与部分可观测性的历史边界

- [Simple Masked Training Strategies Yield Control Policies That Are Robust to Sensor Failure](https://openreview.net/forum?id=AsbyZRdqPv) 已把 modality dropout / missing-modality training 用于鲁棒控制；
- [Multi-Modal Manipulation via Multi-Modal Policy Consensus](https://arxiv.org/abs/2509.23468) 已用模态专家与 router 权重处理遮挡、传感器损坏和物理扰动；
- [TDQC](https://openreview.net/forum?id=TIru4KuhcB) 已把时序任务成功置信度连接到停止与动作搜索；
- 递归 memory / belief VLA 已经占据“部分可观测状态 + 动态融合 gate”的宽泛表述。

因此，历史 scalar `rho` 只能算 corrected token 的工程性 modulation，不是方法创新；它也已在本项目消融中判为负结果。若未来重启 uncertainty 方向，最低要求是分离全局 camera drift、逐点 visibility 和多峰 correspondence，输出经过 NLL/coverage/ECE 验证的 posterior，并用 posterior 驱动 risk-aware action、恢复或主动观测。单一 sigmoid、entropy gate 或 feature amplitude scaling 不得重新包装为第二贡献。

这一方向工作量已经超出当前小论文，默认不进入 Stage I。

## 6. Robot-Camera Calibration, Observability And Task-Oriented Design

这一类工作表明，使用机器人 FK 点、2D 对应、PnP、退化检测和任务导向构型选择都已有充分先例，因此不能单独作为 KineParity 的 novelty。

### DREAM

- 论文：[Camera-to-Robot Pose Estimation from a Single Image](https://arxiv.org/abs/1911.09231)

DREAM 从图像检测机器人关键点，并结合已知 FK 三维点恢复 camera-to-robot 外参。KineParity 若仅做“全臂点 + PnP”，会与这条路线高度重叠。

### Kalib

- 论文：[Kalib: Easy Hand-Eye Calibration with Reference Point Tracking](https://arxiv.org/abs/2408.10562)

Kalib 使用机器人参考点、多帧视觉对应和 PnP 做免标记手眼标定。它执行多帧标定序列，不应被表述为一般运行时在线外参更新；但仍是“先恢复完整外参，再投影”的重要强基线。

### CtRNet

- 论文：[Markerless Camera-to-Robot Pose Estimation via Self-supervised Sim-to-Real Transfer](https://arxiv.org/abs/2302.14332)

CtRNet 直接处理实时 camera-to-robot pose estimation。KineParity 不应声称首次在线估计相机漂移。

### CtRNet-X

- 论文：[CtRNet-X: Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single Camera](https://arxiv.org/abs/2409.10441)

CtRNet-X 针对机器人仅部分可见的真实场景，用视觉语言模型辅助机器人部件检测，并结合关键点式 pose estimation。它进一步压缩了“部分可见时在线恢复 camera-to-robot pose”的 novelty 空间。

### RoboTAG

- 论文：[RoboTAG: End-to-end Robot Pose Estimation via Topological Alignment Graph](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_RoboTAG_End-to-end_Robot_Pose_Estimation_via_Topological_Alignment_Graph_CVPR_2026_paper.pdf)

RoboTAG 把二维观测、三维/FK 结构和拓扑关系统一到图对齐中，联合估计关节状态与相机位姿，并施加闭环几何一致性。它进一步说明 robot structure + 2D/3D alignment + camera pose recovery 已有强方法；FCPG 的边界仍必须是策略内部的局部 feature re-indexing，而不是完整 robot/camera pose estimation。

### SGTAPose

- 论文：[Robot Structure Prior Guided Temporal Attention for Camera-to-Robot Pose Estimation From Image Sequence](https://openaccess.thecvf.com/content/CVPR2023/html/Tian_Robot_Structure_Prior_Guided_Temporal_Attention_for_Camera-to-Robot_Pose_Estimation_CVPR_2023_paper.html)

SGTAPose 使用连续图像、关节状态、机器人结构先验、二维关键点和 PnP 迭代来处理自遮挡并在线估计完整 camera-to-robot pose。因此，时序结构先验与机器人关键点联合估计完整外参已有直接先例。

### FEEPE

- 论文：[Foundation Feature-Driven Online End-Effector Pose Estimation: A Marker-Free and Learning-Free Approach](https://arxiv.org/abs/2503.14051)

FEEPE 用 foundation features 建立 CAD-to-image 2D/3D 对应，通过 PnP 恢复末端六维位姿，并用历史关键帧缓解局部可见和对称歧义。FCPG 的差异必须是直接修正策略读取索引，而不是声称首次在线跟踪机器人末端。

### ARC-Calib

- 论文：[ARC-Calib: Autonomous Markerless Camera-to-Robot Calibration via Exploratory Robot Motions](https://arxiv.org/abs/2503.14701)

ARC-Calib 通过主动探索运动生成易跟踪的视觉轨迹，并用共面与共线约束迭代优化外参。主动标定、探索运动和完整外参恢复均不是 FCPG 当前主线。

### OA-LICalib

- 论文：[OA-LICalib: Observability-Aware Intrinsic and Extrinsic Calibration of LiDAR-IMU Systems](https://arxiv.org/abs/2205.03276)

OA-LICalib 使用可观测性分析和 TSVD，仅更新可辨识方向。KineParity 的白化 TSVD 与 modal rejection 属于工具和归纳偏置，不是独立创新。类似地，基于 normalized innovation / 卡方阈值拒绝与噪声一致的更新是标准状态估计手段；本项目按 effective rank 自适应阈值，只用于避免 clean correspondence noise 触发虚假外参修正，也不单独列为 novelty。

### Task-Oriented Configuration Selection

- 论文：[On task-oriented criteria for configurations selection in robot calibration](https://doi.org/10.1109/ICRA.2013.6631090)

机器人标定中按下游任务误差选择构型和观测点已有先例。因此，自动选择 observer constellation 本身不能作为 novelty；最多作为降低 held-out control-anchor 后验像素不确定性的实现组件。

### KineParity 的历史判定边界（Gate B No-Go）

以下是已经冻结的原 Go 条件，不是当前待执行清单。KineParity 只有在这些证据同时成立时才可能形成独立贡献：

- 全臂 observer points 只用于估计共享相机漂移；
- 指尖或工具 control anchors 的视觉对应、真实投影和残差标签完全不参与求解；已知 `X_C/J_C` 只用于后验效用与更新传播；
- 只传播白化后可辨识、并能降低 control-anchor 后验像素不确定性的共享 SE(3) 模态；有限历史和按秩 innovation gate 用于提高该传播在噪声与退化构型下的稳定性；
- 在传播后的 control anchors 上重新读取策略视觉特征；
- held-out control anchors 上显著优于逐点 refinement、robust PnP 和直接 6D regression。

Oracle geometry Gate A 已通过；matched learned/prototype Gate B pilot 与唯一 frozen rescue 随后完成，但均弱于 direct-control prototype，因此 Gate B 判为 No-Go，策略 Gate C 取消。KineParity 不进入当前贡献列表，也不得继续在已暴露数据上调参。

第二个 bounded local residual head 原本是 global-only 通过后的可选消融；Gate B No-Go 后该实验已取消。

## 7. Predictive Representation Learning References

这两项工作不是几何本体接地 baseline，只用于约束预测式策略和表示正则的实现与表述。

### LeWM

- 论文：[LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312)
- 代码：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- 本仓库固定快照：`third_party/lewm/`，commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`

LeWM 提供 action-conditioned latent predictor、AdaLN-zero Transformer 和 SIGReg 的实现参考。FCPG 不把这些通用模块作为 novelty；`latent_pred_weight` 与 `sigreg_weight` 的匹配消融已经完成并形成负证据。当前 PixelObserver-G0 是独立协议，不借旧预测式分支的结果。

LeWM 的官方目标是 next-embedding prediction 加 SIGReg。动作通过 AdaLN 注入 `192D` latent predictor；SIGReg 只约束 embedding 接近各向同性高斯分布。论文没有把相机内外参、像素射线或跨视角投影等变性写入状态或目标，因此它不能单独解决同一 world-frame action 在不同相机下对应不同 pixel displacement 的问题。本项目把 LeWM 用作轻量 action-conditioned dynamics backbone，但必须另行提供几何状态与投影接口。

### ReViWo

- 论文：[Learning View-invariant World Models for Visual Robotic Manipulation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html)

ReViWo 将表征分为 view-invariant 与 view-dependent 部分，并只用 view-invariant representation 训练 world model，以增强相机扰动下的控制。这支持“dynamics 应尽量在规范、视角不变的状态中学习”，但 view-invariant latent 本身不会给出目标相机中的 metric pixel trajectory；若输出是二维策略锚点，仍需要 view-equivariant decoder 或显式投影。

### PAIWorld

- 论文：[PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation](https://arxiv.org/abs/2606.18375)

PAIWorld 用 Geometry-Aware Cross-View Attention、编码 camera rays/extrinsics 的 Geo-RoPE 与 Latent 3D-REPA 联合约束多视角一致性。其 action-conditioned 版本把末端轨迹投影到每个相机形成 spatial action map，而不是只输入抽象 action vector。论文明确指出，无几何归纳偏置的多视角 token concatenation 必须自行从数据发现 correspondence，不足以保证一致三维生成。该工作规模和任务与本项目不同，但它直接说明 global camera token/FiLM 不是可靠的投影等变性替代品。

### PointWorld

- 论文：[PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation](https://arxiv.org/abs/2601.03782)

PointWorld 用共享三维 point flow 表达状态与动作；机器人 flow 由已知 URDF、关节状态和 FK 生成，再预测场景点在动作下的三维位移。它是“先在共享物理坐标系预测，再按相机观测”的强相关先例。因而本项目若加入 world-frame anchor dynamics，三维表示、FK 和投影均不能单独声称 novelty；必须证明 FCPG 校正观测、persistent anchor dynamics 与相机重投影的闭环组合带来独立控制收益。

### SkyJEPA

- 论文：[SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors](https://arxiv.org/abs/2606.23444)
- 官方仓库：[arplaboratory/SkyJEPA](https://github.com/arplaboratory/SkyJEPA)

SkyJEPA 不是 VLA，也不研究本体关键点接地。它使用四旋翼状态/动作历史训练多步 latent rollout，目标为多步 latent prediction MSE 加 `0.02 * SIGReg`；公开设置使用 batch 2048。随后冻结 latent model，训练 physics-inspired prober，并用 MPPI 控制。FCPG 只借鉴“预测一致性需要防止表示塌缩”这一训练原则；历史 one-step action-conditioned next-latent 已判为负消融，不声称复现 SkyJEPA。

## 8. Not The Main Line

### PriGo

- 论文：[PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation](https://arxiv.org/abs/2607.07076)

定位：

PriGo 在测试时用 primitive 约束 diffusion/flow policy 的动作生成。

与 FCPG 的差异：

- PriGo 作用在动作输出端；
- FCPG 作用在视觉-本体融合端；
- 第一版 FCPG 不使用 primitive。

### MoMaStage

- 论文：[MoMaStage: Skill-State Graph Guided Planning and Closed-Loop Execution for Long-Horizon Indoor Mobile Manipulation](https://arxiv.org/abs/2603.08383)

定位：

使用 skill-state 图进行长程任务规划和闭环重规划。

与 FCPG 的差异：

- MoMaStage 是高层任务/技能规划；
- FCPG 是底层视觉-本体融合和接地校正。

## 创新审计与下一步候选

### 已锁定的 Stage I 边界

当前可以保留的核心机制是：

```text
nominal kinematic projection
-> observation-conditioned local 2D residual
-> corrected robot-keypoint location
-> policy feature resampling at the corrected location
```

这个机制不是新的 feedback control、keypoint detector、deformable attention 或 online calibration。当前 FCPG corrector 的参数在部署时冻结、没有跨帧状态，也不根据执行后的误差更新自身，因此模块级表述应使用 **per-observation correction**、**amortized geometric re-indexing** 或 **feed-forward correction**；不能把“整个机器人每帧重新观测”偷换成 FCPG 本身是 learned feedback observer，也不能称为 test-time training/adaptation。

### 候选 A：Policy-Distilled Kinematic Grounding Correction（唯一推荐进入最小 pilot）

当前结果已经显示，较低 pixel EPE 不必然带来较高 rollout success；这说明 Euclidean grounding accuracy 不是控制效用的充分统计量。令冻结策略在 anchor 集合 (Q) 上的动作块均值为 (mu_phi(Q))，并定义每个 anchor 对采样坐标的局部敏感度：

```text
mu_phi(Q) = pi_phi(s, {F(q_i)})
J_i       = d mu_phi / d q_i
M_i       = J_i^T W_a J_i + epsilon I
```

(M_i) 给出冻结策略诱导的局部像素度量。训练时 corrector 仍输出二维坐标，但除几何残差外，要求 drift observation 上的校正读出与同一冻结策略在 oracle anchor 上的动作输出一致：

```text
L = L_geo
  + lambda_action * ||mu_phi(Q_pred) - stopgrad(mu_phi(Q_oracle))||^2_Wa
  + lambda_clean * 1_clean * ||Q_pred - Q_nominal||^2
```

运行时不计算 Jacobian，也不增加策略调用；Jacobian 只用于解释训练目标的一阶局部度量，不作为 novelty。BYOVLA 已用动作变化识别策略敏感图像区域，ATK 已用动作预测信号选择任务关键点，RobustVLA 已使用 observation Jacobian regularization，[PAIR-VLA](https://arxiv.org/abs/2605.13105) 已使用 paired action invariance/sensitivity，[VaGraM](https://arxiv.org/abs/2204.01464) 也已建立按下游控制影响加权模型误差的原则。因此，仍可能成立的窄边界只能是：**冻结策略把 oracle-grounded action distribution 蒸馏给相机漂移下的 FK-anchor coordinate corrector**。对于扩散策略，应比较固定 common noise 下的动作分布距离，而不是仅比较均值动作。 `Q_oracle` 只允许由训练集真实标定生成；fresh validation/test 不得读取 `q_true`、专家动作或测试环境回报，避免把监督泄漏包装成测试时纠错。

最小 Go 条件沿用冻结的控制门槛：10 度下相对 FCPG 和 frozen prototype 均至少提升 8 pp、paired 95% CI 下界大于 0、绝对成功率至少 15%；5 度非劣 margin 为 -5 pp；clean 非劣 margin 为 -3 pp。必须同时报告 oracle-action gap、EPE 和 correction norm，并先做 same-checkpoint pilot；若只降低离线 action MSE 或 EPE，不能过门。

### 投稿前强制排除的结构解释

必须实现参数量匹配的 `FK-DeformAttn`：

```text
q_nominal
-> K 个自由 offsets + attention weights
-> weighted bilinear feature sampling
-> policy loss only
```

它使用与 FCPG 相同的 FK reference points、视觉 backbone、策略、数据和参数预算，但不使用单一坐标残差语义或显式重投影监督。只有当 FCPG 在 held-out drift 的 anchor EPE、rollout success 和 clean non-inferiority 上稳定优于该基线，才能把收益归因于“标定残差监督的物理锚定校正”，而不是普通 deformable attention。完整 GeoProp 尚未复现时，所有结果继续标为 `GeoProp-lite`。

投稿前的最低替代路线对照还包括：

1. `Absolute-Keypoint`：相同 ViT、`q_true` 标签、数据和预算，直接预测绝对二维关键点；
2. `Local-Match/Track`：frozen prototype，以及统一首帧和延迟预算的 RAFT、TAPIR 或 CoTracker；
3. `Keypoints + SE(3)`：相同视觉对应输入 weighted PnP/RANSAC，恢复共享完整外参后再投影；三末端点的弱 PnP 不足以代表 DREAM/SGTAPose/CtRNet 路线；
4. `Full GeoProp`：补齐 FiLM 与 predictive kinematic token，或始终把当前结果明确标为 `GeoProp-lite`；
5. `View adaptation`：若声称跨视角鲁棒，按信息契约分别报告 zero-shot、已知测试相机、one-shot 和 full-retraining 方法，禁止混成同一协议。

### 其他候选的当前判定

- **Iterative resampling**：只作为扩大捕获范围的结构消融；[DeepIM](https://arxiv.org/abs/1804.00175)、[RAFT](https://arxiv.org/abs/2003.12039) 和 [TAPIR](https://arxiv.org/abs/2306.08637) 已建立粗到细或迭代更新先例，而且当前 prototype 已证明降低 EPE 未必提升控制。
- **Distributional anchor posterior**：当前 No-Go。[DSNT](https://arxiv.org/abs/1801.07372)、multi-hypothesis tracking 和 Bayesian filtering 已覆盖可微坐标分布与时序不确定性的大部分通用工具，且该方向容易把已失败的 scalar reliability 换一种形式重新加入。
- **Self-supervised kinematic flow**：适合独立后续项目，不适合当前最小扩展。[ActionSink](https://arxiv.org/abs/2508.03218) 已用 action-caused optical flow 服务动作估计，learned visual Jacobian 与自监督 correspondence 也有成熟先例；还需新视频协议、首帧重捕获和真机少标注证据。
- **Shared low-rank SE(3)/Jacobian correction field**：可作为 parameter-matched structural baseline，但 KineParity Gate B 已 No-Go，不能仅换名为 JCFR 后重新当贡献。

### 暂不作为创新

普通 EKF/UKF/KalmanNet、EMA/PID、二次采样、scalar reliability、完整 SE(3) 回归、PnP/SVD/可观测性分析、task-oriented observer constellation、primitive guidance、LeWM/SIGReg/CEM，以及单纯增大 ViT、batch 或任务数，均不能单独列为方法贡献。

## 推荐 Claim

```text
GeoProp demonstrates that geometric proprioceptive grounding is effective and characterizes its degradation under extrinsic perturbations, but it does not correct projected sampling locations from the current observation.
FCPG retains the nominal kinematic projection as a coarse anchor, predicts an observation-conditioned local 2D residual, and resamples policy features at corrected robot-keypoint locations without recovering the full camera extrinsics.
```

## 需要避免的 Claim

```text
We introduce feedback control into robot learning.
We propose learned visual offset sampling.
We are the first to identify calibration drift in geometric grounding.
FCPG outperforms the full GeoProp method.  # 当前只实现 GeoProp-lite 时禁止
FCPG performs online camera calibration or test-time training.
We introduce the first action-aware grounding or policy-Jacobian metric.
FCPG is not a keypoint refinement method.
FCPG handles arbitrary, free, or continuously moving cameras.
FCPG outperforms generic trackers, visual servoing, or online calibration without matched baselines.
We are a primitive-guided policy method.
```

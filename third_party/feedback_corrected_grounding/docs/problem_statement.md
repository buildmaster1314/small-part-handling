# 问题定义：非平稳相机几何与间歇观测下的持续锚点接地

## 研究对象

本文研究的对象不是每帧重新检测的一组无身份关键点，而是具有固定物理身份的机器人锚点。对锚点 `i`，其身份在整个 episode 内保持不变，例如左指尖、右指尖、腕部或工具尖端；系统需要持续更新的是同一锚点在当前视觉观测中的接地状态。

给定本体状态 `s_t`、相机内参 `K_cam`、名义相机几何 `T^nom_t` 和真实相机几何 `T^real_t`：

```text
X_{t,i}       = FK_i(s_t)
q^nom_{t,i}   = Pi(K_cam, T^nom_t,  X_{t,i})
q^true_{t,i}  = Pi(K_cam, T^real_t, X_{t,i})
e_{t,i}       = q^true_{t,i} - q^nom_{t,i}
```

这里不假设名义/真实几何之间的失配 `e_{t,i}` 在时间上恒定。外参漂移、相机扰动或名义模型更新都可能使其成为非平稳过程。同时，遮挡、运动模糊、反光、出画和低纹理会使当前视觉证据间歇性可用。二者共同造成的问题是：名义位置会移动或偏离真实位置，而当前位置附近的图像又不一定包含可用于校正的证据。

因此，统一的研究问题是：

> 在非平稳名义/真实相机几何与间歇视觉观测共同作用下，如何对具有固定物理身份的机器人锚点进行持续接地，并在视觉证据受损和恢复时保持身份连续、因果状态连续及可控的策略输入？

## 四个概念接口

### 1. Observation Correction And Measurement

当当前视觉证据可用时，以名义运动学投影为局部搜索中心，从观测中得到同一锚点的 measurement：

```text
P^loc_{t,i}   = LocalSample(F_t, q^nom_{t,i}, r)
delta_q_{t,i} = h_theta(P^loc_{t,i}, X_{t,i}, i)
q^obs_{t,i}   = q^nom_{t,i} + delta_q_{t,i}
m_{t,i}       = AnchorEncode(F_t, q^obs_{t,i}, X_{t,i}, i)
```

该接口回答“当前 observation 如何修正名义接地位置并构造策略 measurement”。**FCPG 的局部、residual-only measurement construction 已验证，是当前唯一可作为论文贡献的主线。** 当前策略直接使用该 measurement；为与候选递归接口对齐，可记为：

```text
b^+_{t,i} = m_{t,i}
```

这只是确定性接口记法，不是贝叶斯 posterior。正式 FCPG 不依赖 LeWM，不输出 learned reliability，也没有验证后验切换或遮挡恢复。

### 2. Action-Conditioned Prediction

当视觉证据暂时缺失或不可信时，候选时序接口应从同一身份锚点的历史 posterior 和已执行动作生成 prior：

```text
p_{t,i} = Predict(B^+_{t-H:t-1}, a_{t-H:t-1})_i
```

该接口只应传播由机器人动作和任务动力学导致的锚点状态变化。它不能提前预测外生的相机跳变、遮挡发生或视觉恢复。

**Anchor-Persistent LeWM 的 action propagation 尚未形成论文贡献。** 现有结果支持 learned transition prior 相对 hold-last / latent-CV 的缺测传播价值和对动作错配的敏感性；但 A0.5 相对独立训练、容量匹配的 no-action recurrent predictor 未建立跨任务稳定优势。因而已有受限机制信号不能改写为 factual-action 必要性、稳定遮挡恢复、在线控制或世界模型贡献，也不能与已验证的 FCPG 主线合并成端到端成功声明。

### 3. Observation Trust

当 measurement 与 prior 同时存在时，需要判断的不是一般“模型置信度”，而是当前 observation 对锚点 `i` 是否值得信任：

```text
c^obs_{t,i} = Trust(available causal evidence) in [0, 1]
```

`c^obs` 只能使用部署时可得的因果证据，例如局部匹配响应、匹配 margin、measurement-prior innovation 和历史；不能读取真实投影、特权可见性标签、预设 corruption schedule 或未来帧。它也不表示动作置信度、任务成功率或锚点重要性。

**Learned reliability / VisualTrust 尚未形成论文贡献，已有接口为 No-Go。** 旧 `rho` 只门控 sampled feature，且策略消融未通过；它不能被重新解释为已验证的 observation trust。正式 residual-only FCPG 等价于在可用观测上固定 `c^obs=1`。

### 4. Re-Anchoring

视觉证据恢复后，系统应把同一物理身份的锚点从预测主导状态重新绑定到当前 observation measurement，而不是新建、重排或替换 anchor slot：

```text
c^obs_{t,i} -> 1
implies
b^+_{t,i}   -> m_{t,i}
```

这一定义中的 re-anchoring 不是恢复完整相机外参，也不是 LeWM 预测到了外生视觉变化。它要求系统在观测受损时拒绝错误 measurement，在观测恢复时及时退出 prior，并把更新后的 posterior 因果写回下一时刻。

**支撑 re-anchoring 的 posterior switching / recovery 尚未形成论文贡献，已有 learned Adapter、selector 和 switching 接口为 No-Go。** 因此当前不能声称已经实现可部署的遮挡恢复、posterior policy control 或端到端 persistent-anchor system。

## 联合失败模式

### 非平稳几何失配

标定残差随时间变化时，开环投影会持续从错误图像区域读取特征。单次静态偏移修正不能自动覆盖后续相机变化。

### 间歇视觉观测

遮挡或图像退化期间，名义位置或上一时刻位置附近的特征可能来自遮挡物、背景或伪匹配。仅有逐帧 observation update 不能区分“需要校正”和“当前不可观测”。

### 错误后验写回

若不可信 measurement 被写入历史，误差会通过时序预测递归传播；若系统在视觉恢复后仍保留 prior，则会产生恢复滞后和 clean-condition 代价。observation trust 与 re-anchoring 必须同时满足拒绝、恢复和 clean safety，不能只报告遮挡期误差下降。

### 身份断裂

每帧重新选择无身份关键点会让 prior、measurement 和策略读取对象失去一致语义。持续接地要求锚点 `i` 始终对应同一机器人部位，多关键点只是固定身份集合，不是可交换的检测槽位。

## 证据状态

| 概念或接口 | 当前证据状态 | 允许的结论 |
| --- | --- | --- |
| FCPG observation correction / measurement | **已验证；当前唯一锁定的论文贡献** | 在已测试的局部捕获范围内修正名义投影，并在校正位置重采样策略特征 |
| Observation-only FCPG policy | **已串联验证** | 策略可直接消费 `b^+=m`；不代表完整时序 posterior 系统成立 |
| LeWM action-conditioned prediction / action propagation | **机制证据受限；尚非论文贡献** | 相对 hold-last / latent-CV 的 predictor utility 为正；matched no-action 比较未建立稳定 factual-action 增量，不声称动作必要性、控制收益或通用世界模型能力 |
| Learned reliability / observation trust | **尚非论文贡献；已有接口 No-Go** | 只定义待解决的 measurement trust 问题；旧 `rho` 和 VisualTrust 不是已验证 selector |
| Posterior switching / re-anchoring | **尚非论文贡献；已有接口 No-Go** | 只定义观测恢复时的目标语义；不声称 recovery、clean safety 或 policy rollout 已成立 |
| 完整 persistent-anchor posterior policy | **未成立** | 不得把 FCPG、LeWM、trust 和 switching 写成已验证的端到端系统 |

## 假设与边界

- 系统保留机器人运动学、相机内参和名义相机几何；FCPG 修正的是名义投影附近的二维 residual，不求解完整在线标定。
- 当 observation 可用时，`q^true_{t,i}` 必须仍落在 `q^nom_{t,i}` 的局部视觉窗口和有界 residual 捕获范围内。超出该范围需要全局重定位、重新标定或更强 tracker，不属于当前 FCPG 保证。
- 本问题**不是 calibration-free**。它不覆盖任意自由移动相机、无限漂移、长期完全不可见或无可靠初始化的情形。
- interaction mode / primitive 不是问题成立所需的假设，也不是当前方法贡献。若未来使用，只能作为另行验证的下游条件信息，不能用于补足尚未通过的 prediction、trust 或 switching 证据。
- 当前论文主张止于 FCPG observation update。action-conditioned prediction、observation trust 和 re-anchoring 共同给出更完整的研究问题，但不代表这些模块已经实现或验证。

## 评价原则

- Observation correction：比较 `q^nom` 与 `q^obs` 的锚点误差，并验证校正位置重采样是否带来配对 rollout 收益。
- Action-conditioned prediction：必须在随机化缺测时刻下，与 hold-last、匹配的 no-action predictor 和几何滤波基线比较，不能只报告模型自身预测误差。
- Observation trust：必须使用可部署证据，同时报告错误接纳、错误拒绝和 clean-condition safety。
- Re-anchoring：必须测量视觉恢复后的退出 prior 延迟、posterior 写回稳定性和策略影响；只改善遮挡区间不足以通过。
- 所有时序评价都必须保持锚点身份和 episode 边界，禁止使用特权 validity、未来信息或 corruption-time shortcut。

# 方法草图：Persistent-Anchor Belief

## 方法定位

论文主方法名称仍为 **FCPG：Feedback-Corrected Proprioceptive Grounding**。`persistent-anchor belief` 是统一各模块接口的状态叙事，不是已经验证的新贡献，也不把 LeWM、Transformer、滤波或 selector 本身写成 novelty。

核心对象是具有固定物理身份的机器人锚点。锚点编号 `i` 在时间上保持不变；系统更新的是该锚点在当前观测下的接地状态，而不是每帧重新检测一组无身份关键点。下文的 `belief` 只是一种 predict-measure-update 接口记账，不声称概率后验、校准协方差或新的滤波原理。

```text
nominal kinematic anchor
-> FCPG observation update
-> posterior anchor state
-> policy

previous posterior anchor states + actions
-> Anchor-Persistent LeWM prediction (candidate prior)
-> observation-trust update (candidate)
-> posterior anchor state
```

> **证据边界：FCPG 已验证；其余链路尚未端到端成立。** 已验证的主线是局部、residual-only FCPG measurement construction，以及策略在校正位置读取特征后的配对 rollout 收益。Anchor-Persistent LeWM 只有 dynamics / masked-measurement propagation 机制证据；reliability、selector、recursive recovery 和 posterior-policy control 仍是候选，且已有接口包含明确 No-Go。不得把上图整体写成已验证系统或论文第二贡献。

## Persistent-Anchor Belief

在时刻 `t`，第 `i` 个锚点的 posterior state 记为：

```text
b^+_{t,i} in R^D,    i in {1, ..., K}
B^+_t = {b^+_{t,1}, ..., b^+_{t,K}}
```

其中 `i` 表示固定的机器人部位身份，`b^+_{t,i}` 表示候选递归系统中的接地状态估计。这里的 belief 是确定性的状态接口，不表示已经获得校准的概率分布、似然或协方差；当前已验证策略实际直接读取 FCPG measurement，而不是一个经 prior/likelihood 得到的贝叶斯 posterior。

统一语义为：

- FCPG 产生当前图像的 anchor measurement；
- Anchor-Persistent LeWM 从历史 posterior 与历史动作产生 action-conditioned prior；
- reliability 只表示当前 observation 的 trust，不表示动作置信度、任务成功率或一般不确定性；
- recovery 表示 observation 再次可信后，把同一身份的 persistent anchor 重新锚定到当前 FCPG measurement；
- policy 只接收更新后的 posterior anchor state，而不是把 measurement、prior 和 selector 当作三个互不一致的状态源。

## 1. FCPG Measurement Construction：已验证主线

### 名义几何锚点

给定图像 `I_t`、本体状态 `s_t`、相机内参 `K_cam` 和名义外参 `T_nom`：

```text
F_t             = E_img(I_t)
X_{t,i}         = FK_i(s_t)
q^nom_{t,i}     = Pi(K_cam, T_nom, X_{t,i})
P^loc_{t,i}     = LocalSample(F_t, q^nom_{t,i}, r)
```

`q^nom_{t,i}` 是保留物理身份和名义标定的几何锚点。FCPG 不从整幅图像预测绝对关键点，也不在部署时重解完整相机外参。

### 局部二维 residual

第一版只根据名义投影邻域内的视觉证据预测有界二维残差：

```text
u_{t,i}         = h_theta(P^loc_{t,i}, emb(i), phi_x(X_{t,i}))
delta_q_{t,i}   = delta_max * tanh(u_{t,i})
q^obs_{t,i}     = q^nom_{t,i} + delta_q_{t,i}
v^obs_{t,i}     = Sample(F_t, q^obs_{t,i})
m_{t,i}         = E_anchor(v^obs_{t,i}, phi_x(X_{t,i}), emb(i), q^obs_{t,i})
M_t             = {m_{t,1}, ..., m_{t,K}}
```

`m_{t,i}` 是当前 observation 对固定身份锚点 `i` 的 measurement。局部窗口和 `delta_max` 给出明确捕获范围；因此 FCPG 不是 calibration-free 方法，名义投影必须仍落在局部视觉证据可恢复的邻域内。

监督目标与第一版损失为：

```text
delta_q^*_{t,i} = q^true_{t,i} - q^nom_{t,i}
L_FCPG           = SmoothL1(delta_q_{t,i}, delta_q^*_{t,i})
```

`q^true` 只用于训练标签和离线评分，不是部署输入。正式 residual-only FCPG 不输出 `rho`，使用 `SIGReg=0`，也不包含时序预测损失。

### Observation-only policy path

当前已经验证并实际接入策略的路径不依赖 LeWM 或 selector。策略直接消费 FCPG measurement；为与候选递归接口对齐，可记为：

```text
b^+_{t,i} = m_{t,i}
```

这只是接口上的退化记法，不是贝叶斯更新。它等价于 observation trust 固定为 `1`：策略在 `q^obs=q^corr` 处重采样视觉特征；这是第一版 FCPG 的局部 residual 主线。

## 2. Anchor-Persistent LeWM Prediction：候选

候选 predictor 保留时间轴和锚点身份，不在预测前把它们平均成单个 summary。给定长度为 `H` 的 posterior history 和对应动作：

```text
p_{t,i} = f_psi(
    B^+_{t-H:t-1},
    a_{t-H:t-1}
)_i
```

张量接口保持为 `[batch, time, anchor, dim]`；动作按锚点广播，但时间和锚点轴不被平均。`p_{t,i}` 是同一 anchor-state space 中的 prior，目标是预测机器人动作和任务动力学引起的状态变化。

相机 step、遮挡或纹理破坏是外生 observation 变化，不能由历史动作提前预测。因而 LeWM 不负责预测相机跳变；它只提供与当前 measurement 比较的动作条件先验：

```text
innovation_{t,i} = m_{t,i} - p_{t,i}
```

固定 `2,000`-step pilot 使用的候选训练目标为：

```text
L_pred = MSE(p_{t+1,i}, m_{t+1,i})
       + 0.02 * mean_i SIGReg(m_{:,i})
```

该公式只记录已运行 pilot 的接口，不把 `SIGReg` 或 LeWM 写成 FCPG 主方法，也不授权扩到 `10,000` steps、规划或控制。

## 3. Reliability：Observation Trust 候选

若 prediction 与 observation 同时存在，reliability 的唯一语义是“当前 FCPG measurement 值得信任到什么程度”。记：

```text
c^obs_{t,i} = Trust_omega(chi_{t,i}) in [0, 1]
```

其中 `c^obs=1` 表示采用当前 observation，`c^obs=0` 表示暂时保留 prior。`chi` 只能包含部署时可用的当前或因果证据，例如局部视觉匹配、margin、innovation 和历史；不得读取 `q^true`、未遮挡 canonical target、预设 corruption schedule 或 `measurement_valid` 标签。

在 measurement 与 prior 已被定义到同一 anchor-state space 时，最保守的候选 posterior update 为：

```text
b^+_{t,i} = c^obs_{t,i} * m_{t,i}
           + (1 - c^obs_{t,i}) * p_{t,i}
```

等价的 measurement-dominant 写法是：

```text
beta_{t,i} = 1 - c^obs_{t,i}
b^+_{t,i}  = m_{t,i} + beta_{t,i} * (p_{t,i} - m_{t,i})
```

该式只是候选接口，不表示 `c^obs` 已校准，也不允许直接混合不同坐标系、不同表征空间或缺少共同语义的张量。没有通过 selector Gate 时，默认回退到 `c^obs=1` 的 observation-only FCPG。

旧版 `rho` 只门控 sampled visual feature，且匹配策略消融为 No-Go；它不能被改写成已验证的 posterior reliability。当前 VisualTrust 与 measurement-vs-prior selector 同样没有通过 deployable Gate。

## 4. Recovery：Re-Anchoring 候选

Recovery 不表示恢复完整相机外参，也不表示 LeWM 预测到了外生视觉变化。它只表示：视觉 evidence 恢复可信后，同一身份锚点从 prior-dominant 状态回到当前 FCPG measurement。

```text
c^obs_{t,i} -> 1
implies
b^+_{t,i}   -> m_{t,i}
```

这一过程称为 **re-anchoring**。锚点身份 `i` 不变，恢复时不新建或重排 slot。一个可部署 recovery 机制必须同时做到：

- observation 受损时拒绝不可信 measurement；
- observation 恢复时快速退出 prior；
- clean observation 下不产生不可接受的 false rejection；
- 将 posterior 写回下一时刻历史，且不使用 privileged validity 或时序捷径。

现有 learned posterior Adapter 在遮挡期有强机制收益，但 recovery 与 clean safety 未过门，因此 re-anchoring 尚未验证。

## 5. Policy 接口

统一接口要求策略读取 posterior anchor state：

```text
a_{t:t+L-1} = pi(B^+_t, s_{t-h:t})
```

`B^+_t` 保留每个 anchor 的固定身份与独立状态。当前已验证实现中 `B^+_t=M_t`，即策略读取 FCPG 校正位置的 observation-only posterior。候选 LeWM / reliability 路径只有在 prediction、selector、re-anchoring 和 clean safety 全部通过后，才能把融合后的 `B^+_t` 接入 policy rollout。

因此以下完整链路目前**没有成立**：

```text
FCPG measurement
-> Anchor-Persistent LeWM prior
-> learned observation trust / selector
-> recursive posterior writeback and re-anchoring
-> posterior-conditioned policy improvement
```

不得声称已有 posterior policy control、MPC、CEM 或端到端 recovery 收益。

## 组件与实现边界

- `fcpg/transformer_grounding.py`：已验证的 anchor-conditioned、residual-only FCPG observation update；
- `fcpg/action_chunk_policy.py`：当前 observation-only FCPG policy 路径；
- `fcpg/sequence_data.py`：episode-safe dataset、ViT feature cache 与 residual-only correction cache；
- `fcpg/lewm_anchor_persistent.py`：候选 `[B,T,K,D]` action-conditioned predictor；
- `fcpg/lewm_posterior_adapter.py`、`fcpg/visual_trust.py` 与 `fcpg/lewm_posterior_*`：development / diagnostic 候选接口，不是已通过的部署链路；
- residual-only cache 设置 `has_reliability=false`，不得为当前 FCPG 伪造 `reliability.npy`。

第一版 FCPG 保持以下边界：局部 patch、二维 residual、固定物理身份锚点、校正位置重采样和确定性 action-chunk policy。不加入绝对全图关键点检测、完整 `SE(3)` 在线恢复、learned `rho` 主分支、扩散动作生成或测试时规划。

## 证据状态

| 模块或接口 | 证据状态 | 允许的表述 |
| --- | --- | --- |
| FCPG local residual measurement construction | **已验证；当前唯一锁定贡献** | 在已测试的局部捕获范围内，当前视觉 evidence 能修正名义投影；校正位置重采样在配对 rollout 中有收益 |
| FCPG observation-only policy | **已在线串联** | policy 接收 `b^+=m`；不能外推为任意扰动、任意任务或 calibration-free |
| Legacy summary-pooled full grounded LeWM | **正式 Gate L1 No-Go** | 旧接口未达预注册效应量，L2/L3 未启动；不能写成 world-model control 已成立 |
| Anchor-Persistent LeWM | **候选；dynamics mechanism positive** | 固定 `2k` pilot 与 predictor isolation 支持逐锚点 dynamics utility；A0.5 未证明 factual action 相对 matched no-action 的稳定必要性 |
| Learned posterior Adapter | **deployable Gate No-Go** | 遮挡期机制正向，但 recovery / clean safety 失败；不能接入 policy claim |
| Reliability / VisualTrust | **候选；现有接口 No-Go** | 只可称 observation-trust 研究问题；旧 `rho` 和 current-only VisualTrust 都不是已验证 selector |
| Measurement-vs-prior selector | **fresh randomized development No-Go** | privileged oracle 仅给 headroom；不授权 learned comparator、hard switch 或 policy control |
| Re-anchoring recovery | **未验证** | 只能作为 recovery 的目标语义，不能写成现有能力 |
| End-to-end persistent-anchor posterior policy | **未成立** | 不得列为论文贡献或已完成系统 |

## Claim 约束

可以写：

- **FCPG 是基于名义运动学锚点的逐观测局部 residual update。**
- **当前策略消费 FCPG observation-only posterior anchor state。**
- **Anchor-Persistent LeWM 是保留锚点身份的 action-conditioned prediction 候选，并已有受限机制证据。**
- **reliability 的目标语义是 observation trust；recovery 的目标语义是 re-anchoring。**

不能写：

- FCPG、LeWM、selector 与 policy 已形成验证通过的端到端 posterior system；
- LeWM 已稳定解决遮挡恢复、动作必要性或在线控制；
- learned reliability 已安全决定 measurement / prior 切换；
- recovery 等价于在线相机标定或完整外参恢复；
- 当前方法具备 calibration-free、MPC、CEM 或通用规划能力。

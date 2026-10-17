# Predictive Feedback Research Direction

## Current thesis

The strongest research contribution is not LeWM itself, V12.4, or a larger visual model. The core idea is:

> A robot can use its own executed actions as a predictive sensor to maintain and correct visual grounding when current visual measurements become unreliable.

The system should be presented as **action-conditioned predictive visual feedback**, not as a generic world model.

---

## Current architecture

```
Previous grounded query token
          |
          + robot state
          + state delta
          + executed action
          + CAD/FK motion signature
          |
          v
Action-conditioned query prior
          |
          v
Predicted future visual token
          |
          +----------------+
                           |
Current RGB -> V12.4 -> visual measurement
                           |
                           v
                 predictive feedback fusion
                           |
                           v
                 corrected grounding posterior
```

Important implementation detail:

- V12.4 encoder is frozen.
- The trainable component is a lightweight action-conditioned prior.
- The prior predicts visual tokens, not pixel coordinates.
- Feedback corrects the measurement instead of replacing it.

---

## Why this is interesting

Most robot perception pipelines assume:

```
image -> state estimate -> action
```

This project explores the reverse embodiment signal:

```
action -> expected future observation -> better perception
```

The robot knows what it just did. That information can disambiguate what it currently sees.

This is closer to learned predictive filtering than standard tracking.

---

## Paper positioning

Recommended framing for ICRA/RSS:

**Action-Conditioned Predictive Feedback for Persistent Visual Grounding**

Avoid framing as:

- a new Transformer architecture
- a new LeWM variant
- a generic world model

The novelty is the feedback loop:

```
predict -> measure -> correct -> re-anchor
```

---

## Current evidence boundary

Current H1 experiment:

- single 100 mm displacement event
- predictive prior generated from action/state/motion
- measurement corrected with predicted token prior

This is a mechanism validation, not yet a full persistent grounding claim.

Do not claim:

- complete H2-H4 recursive tracking
- full closed-loop manipulation improvement
- uncertainty-aware filtering

until those experiments exist.

---

## Most valuable next experiments

### 1. Action-specific ablations

Beyond zero/shuffle/hold:

Add wrong-direction action.

Example:

```
real action: move left
prior action: move right
```

This proves the model understands action semantics rather than only temporal smoothness.

---

### 2. Physical robot validation

Prioritize grounding recovery over task success.

Suggested demo:

```
robot sees anchor
 -> executes motion
 -> blur / occlusion / viewpoint change
 -> measurement-only loses target
 -> predictive feedback recovers
```

Metrics:

- localization error
- token error
- recovery time
- identity retention

---

### 3. Recursive feedback

Move from:

```
t0 observation
 -> action
 -> corrected t1
```

Toward:

```
posterior_t
 -> action
 -> prior_t+1
 -> measurement_t+1
 -> posterior_t+1
 -> ...
```

This is the difference between one-step correction and persistent grounding.

---

## Biggest future improvement

Replace fixed feedback weight:

```
beta = 0.25
```

with adaptive confidence:

```
beta_t = f(
    visual confidence,
    predictive confidence,
    innovation error
)
```

Desired behavior:

- clear camera -> trust measurement
- occluded camera -> trust prediction
- large disagreement -> re-anchor

This would upgrade the method from fixed fusion to a learned feedback controller.

---

## Reviewer risks

### Risk 1
"Is this just Kalman filtering with neural features?"

Answer:

The prior is not a geometric state prior. It predicts semantic visual tokens conditioned on robot actions.

### Risk 2
"Does it only work because V12.4 is already good?"

Need frozen encoder baselines and random-prior controls.

### Risk 3
"Does synthetic corruption represent reality?"

Need physical camera degradation and real robot evidence.

---

## Final strategic recommendation

Do not keep adding modules.

Freeze the architecture and prove:

1. action-conditioned prediction helps grounding;
2. feedback survives real visual degradation;
3. recursive correction works on a physical robot.

The strongest final claim is:

> Robots can use their own actions as predictive visual feedback to maintain grounded perception under unreliable observations.

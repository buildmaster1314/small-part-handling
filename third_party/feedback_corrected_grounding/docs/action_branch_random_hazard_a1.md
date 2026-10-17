# Action-Branch Random-Hazard A1

Status: **development Screen No-Go; formal 10k not authorized; independent test
unopened**. The frozen protocol and recorded execution status are in
`config/lewm/eval/action_branch_random_hazard_a1.json`.

## Estimand

The belief-update narrative is

```text
b_t^- = T(b_{t-1}, a_{t-1})
y_t = O(I_t, p_t^kin)
b_t = U(b_t^-, y_t, eta_t)
```

A1 isolates only the causal increment of the executed action in `T`. It does
not validate the full belief update, update rule `U`, observer `O`, or a learned
reliability estimate. In the current A1, `eta_t` is only the privileged
`sensor-valid` mask used to establish predictor reliability; it is not a
learned quantity. A learned `eta_t` requires a separate validation gate.

At a hazard selected before reset, save one physical state, process RNG, and
causal history. Restore that same snapshot for exactly eight anonymous action
branches. The scripted policy trajectory supplies only the shared pre-action
history and the center action; it is not action diversity. The eight-way
within-state intervention is the sole source of action diversity: preserve the
policy gripper commands and add the positive and negative directions from four
frozen-seed QR motion directions, with motion delta RMS fixed at `0.15` after
clipping.

Hide measurement before the first branch action and score all primary model
horizons `1:8`. One model step is five raw environment steps, so horizon `8`
spans `40` raw steps. Every scored branch therefore has the same pre-action
state and differs only in its randomized executed action sequence.

For bundle `i`, horizon `h`, branch `b`, target `z`, and model prediction
`z_hat`, whiten with train-only anchor/dimension moments and center across the
eight branches:

```text
L_i(m) = mean_b,h,d [ (z_ibh - mean_b(z_ibh))
                    - (z_hat_imbh - mean_b(z_hat_imbh)) ]^2
effect(m_action, m_no_action) = 1 - L(m_action) / L(m_no_action)
```

The primary effect compares factual-action `m_action` with the independently
trained, capacity/initialization/optimizer-matched no-action head `m_no_action`.
Branch centering removes the common state-predictable trajectory. The required
action-alignment control evaluates every one of the seven non-identity cyclic
action-donor shifts and averages their bundle losses; one favorable shuffle is
not sufficient. The same comparison is reported separately over `H6:H8`, and
factual raw whitened MSE may not exceed matched no-action MSE. Hold-last and
latent CV remain secondary and cannot replace these comparisons. Geometry/KF
baselines belong to the later full interruption benchmark; A1 is deliberately
limited to identifying the action increment in the learned transition.

## Units, split, and inference

- Independent unit: one source physical episode. Hazards, snapshots, branches,
  horizons, rerenders, and model seeds are nested observations.
- Hazards: pre-reset Bernoulli `p=0.12`, refractory `5`, maximum `3`, conditioned
  only on at least one event. The schedule cannot read state, action, reward, or
  semantic phase.
- Split per task: `24` train / `12` development / `12` sealed, independent test
  physical episodes. MetaWorld exposes only `50` fixed task instances per
  task, so the remaining `2` instances are reserved for smoke and discarded.
  No source episode, snapshot, or branch crosses from development into test.
- CI: `10,000` shared-draw, task-stratified paired percentile bootstrap;
  resample whole source episodes and report a two-sided `95%` CI. Task effects
  are computed first and then averaged equally across the four tasks.

## Frozen model contract

The latent space comes from a frozen `AnchorPersistent` encoder trained with
`SIGReg=0.02`. The encoder remains frozen throughout A1. The factual-action and
matched no-action prediction heads use the same architecture, initialization,
optimizer, batches, dropout RNG, and fixed-latest selection within each seed.
The heads optimize the prediction objective only and add no separate SIGReg
term; moving SIGReg from the frozen representation to predictor outputs would
change the regularizer's meaning rather than test the action factor.

## Bounded decision

1. `100`-step smoke on `2` disjoint episodes/task checks same-state restoration,
   exactly eight distinct intervention branches, finite predictions, no-action
   branch invariance, and shuffle derangement only. It carries no scientific
   result.
2. One bounded `2,000`-step development run: continue only when factual improves
   at least `5%` over both matched no-action and the mean of all seven wrong
   action alignments, at least `3/4` tasks and `2/3` matched training seeds are
   positive, and each task has at least `80%` informative bundles. `H6:H8` must
   remain positive against both controls and factual raw whitened MSE must not
   exceed no-action. No CI is required and no tuning is allowed.
3. Only a development pass authorizes the formal run. Discard the `2,000`-step
   heads and optimizer states, reinitialize both heads from their original
   matched seed initialization, and train a fresh fixed `10,000`-step schedule
   from step zero. This is not a continuation or a learning-rate reset at the
   `2,000`-step boundary. Open the independent test once. Final Go requires the
   same conditions plus both bootstrap CI lower bounds strictly above zero.
   Otherwise A1 is No-Go and the test cannot be recycled. A Go is limited to
   the causal action increment in `T` on this branch/horizon support; it does
   not establish learned reliability, selector quality, or policy benefit.

## Development result

The formal store contains `450` same-state hazard bundles: `218` train, `113`
development, and `119` unopened test bundles. The bounded development screen
trained factual-action and matched no-action heads for exactly `2,000` steps at
seeds `18401/18402/18403`; the frozen anchor encoder is the `SIGReg=0.02`
model, while the new prediction heads add no SIGReg term.

On the full four-task development panel, factual action reduced the primary
branch-centered error by `55.75%` relative to matched no-action and by `73.74%`
relative to the mean of all seven cyclic wrong-action controls. The `H6:H8`
effects remained `52.72%` and `70.37%`; all three training seeds were positive,
and factual raw whitened MSE was `0.115623` versus `0.439560` for no-action.

The screen nevertheless failed its only unmet check. Informative-bundle support
was `30/30` for pick, `28/28` for reach, `26/26` for push, but only `10/29`
(`34.48%`) for button, below the frozen `80%` per-task requirement. On the full
button panel, factual versus no-action was `-0.73%`; the other task effects were
pick `+65.25%`, reach `+83.64%`, and push `+74.84%`. A post-hoc calculation on
the ten informative button bundles was positive (`+8.63%`) but is descriptive
only and does not change the gate.

The result supports a strong action-alignment signal on branch distributions
with adequate intervention variance. It does not establish four-task action
necessity, policy benefit, observation trust, or the complete belief update.
Accordingly, no fresh `10,000`-step run is allowed and the independent test split
must remain unopened.

Formal outputs:

```text
outputs/lewm_action_branch_a1/action_branch_a1_formal.npz
outputs/lewm_action_branch_a1/a1d_seed18401_2k/
outputs/lewm_action_branch_a1/a1d_seed18402_2k/
outputs/lewm_action_branch_a1/a1d_seed18403_2k/
outputs/lewm_action_branch_a1/a1d_screen_gate/summary.{json,md}
```

Score a collector-produced NPZ bundle with:

```bash
python scripts/score_action_branch_random_hazard_a1.py \
  --stage screen \
  --predictions \
    outputs/lewm_action_branch_a1/a1d_seed18401_2k/val_predictions.npz \
    outputs/lewm_action_branch_a1/a1d_seed18402_2k/val_predictions.npz \
    outputs/lewm_action_branch_a1/a1d_seed18403_2k/val_predictions.npz \
  --train-data outputs/lewm_action_branch_a1/action_branch_a1_formal.npz \
  --output outputs/lewm_action_branch_a1/a1d_screen_gate
```

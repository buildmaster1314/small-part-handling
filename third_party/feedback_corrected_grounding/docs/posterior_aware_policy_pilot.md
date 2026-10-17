# Posterior-aware policy pilot

Status: historical development No-Go. A later 30-episode confirmation supersedes
this 12-episode pilot for claims about the frozen policy interface. This document
is retained to record why controller fine-tuning and the residual adapter were
stopped; it must not be cited as the primary FCPG/LeWM control result.

## Question

Can a frozen FCPG/LeWM grounded-token observer improve policy control during a
joint camera rotation and short visual dropout, while preserving the original
clean policy?

The screen used one frozen Pick policy, one frozen FCPG corrector, one frozen
LeWM seed, and 12 paired development episodes. The joint condition combined a
5 degree camera rotation with a six-decision local visual dropout. During valid
frames the policy uses native FCPG grounding. During known-invalid frames the
observer rejects the current anchor measurement and fuses a geometry carrier
(last-valid FCPG visual descriptor plus current FK) with the action-conditioned
LeWM prior. This is not raw LeWM output and not simultaneous fusion with the
current corrupted FCPG measurement.

Only grounded anchor tokens are overridden. The policy's global visual token
still comes from the observed image, including the blackout during invalid
frames. The tested interface is therefore recursive grounded-token writeback,
not complete observation-state writeback.

## Full-controller fine-tuning

The first interface fine-tuned the existing policy controller on cached
measurement, no-action, or factual-LeWM tokens. It failed because small offline
action loss did not preserve the closed-loop policy distribution. At 2,000
steps, 149 of the 173 stored policy tensors had changed; representative relative
L2 changes were 3.09% in the action decoder and 4.06% in the memory encoder.

A checkpoint-loading bug was also found during this experiment. Legacy policy
checkpoints omit frozen ViT weights and expect the normal pretrained frontend
to be reconstructed. The initial derived checkpoints accidentally serialized a
temporary frontend. The loader now merges only trained controller tensors into
the original sparse checkpoint contract. All results below use that correction.

With corrected loading, the joint-condition success rates were:

| Method | Success |
| --- | ---: |
| Frozen base with factual posterior | 5/12 (41.67%) |
| Measurement-trained controller | 0/12 |
| No-action-trained controller | 0/12 |
| Factual-trained controller | 0/12 |

This interface is a development No-Go.

## Frozen-policy residual adapter

The bounded rescue froze every base-policy parameter and trained only a
68,528-parameter per-anchor residual adapter. The adapter is bypassed when no
posterior override is supplied, so clean execution remains exactly the original
policy. Its residual is zero-initialized and bounded elementwise by 0.25.

Offline, the adapter looked strong:

| Posterior | Unadapted MSE | Adapted MSE | Relative improvement |
| --- | ---: | ---: | ---: |
| Factual LeWM | 0.04565 | 0.01638 | 64.12% |
| No-action LeWM | 0.09184 | 0.01607 | 82.50% |

The near-equal adapted errors already indicated that factual actions were not
providing a unique control advantage. The paired rollout screen confirmed that
the offline gain did not transfer:

| Method | Success | Mean reward |
| --- | ---: | ---: |
| Clean frozen base | 9/12 (75.00%) | 1149.15 |
| Clean factual adapter | 9/12 (75.00%) | 1149.15 |
| Joint measurement | 0/12 (0.00%) | 440.38 |
| Joint unadapted factual | 5/12 (41.67%) | 522.97 |
| Joint no-action adapter | 2/12 (16.67%) | 372.18 |
| Joint factual adapter | 1/12 (8.33%) | 458.87 |

The factual adapter was descriptively 33.33 percentage points worse than the
unadapted factual observer and 8.33 points worse than the no-action adapter.
The percentile bootstrap interval originally emitted for the 1/12 versus 5/12
binary comparison was anti-conservative at this sample size; the exact paired
McNemar test is not significant (`p=0.125`). Clean base and clean adapter were
identical for every reported metric, validating the bypass contract but not the
posterior-control hypothesis. The reported `success` is any-step success, not
final or sustained success.

## Interpretation

The adapter learned a one-step teacher-matching objective on cached states, then
changed the action distribution that generated subsequent states. This creates
the usual behavior-cloning distribution shift: low cached MSE is compatible
with worse recursive control. The cache also supplied clean global visual
context while deployment retained corrupted global visual context, creating an
additional train/evaluation mismatch. Because no-action adapts at least as well
offline, the experiment does not isolate action-conditioned LeWM information.

Do not widen this adapter, extend training, or tune it on these 12 episodes.
Any future attempt must change the scientific interface, not just capacity. A
credible successor would require on-policy or DAgger-style state collection and
an explicit task-level objective, followed by a fresh paired evaluation.

## Superseding confirmation and frozen decision

The later frozen confirmation used 30 fresh physical episodes and three LeWM
seeds. Factual propagation improved clean-policy action fidelity by 46.70%
relative to the exact-FK geometry carrier and by 51.10% relative to matched
no-action prediction, with both clustered confidence intervals excluding zero.
This confirms action-conditioned grounded-state propagation.

It did not improve task control. Reward and maximum-success-streak intervals
crossed zero, while any-step success was 12.22 percentage points below geometry
and 8.89 points below matched no-action. The authoritative conclusion is:

- action-conditioned Anchor-Persistent propagation: confirmation Go;
- current frozen-policy grounded-token writeback: confirmation No-Go;
- full-controller fine-tuning and residual Adapter: development No-Go.

Baseline expansion, larger adapters, longer training, and CEM on this interface
are frozen. A future control experiment must change the objective and state
distribution, for example by collecting on-policy posterior states and training
against task progress or success. It cannot reuse clean-action imitation as a
surrogate for control benefit.

## Artifacts

- Training cache: `outputs/lewm_posterior_aware_policy/pick_train_cache_seed63101.npz`
- Adapter checkpoints: `outputs/lewm_posterior_token_adapter/{factual,no_action}_seed63401_2k/`
- Paired rollout report: `outputs/lewm_posterior_token_adapter/rollout_screen_seed63501/report.json`
- Superseding confirmation: `outputs/lewm_p1_native_policy_rollout/joint_drift_drop_confirmation_seed52901/policy_rollout_report.json`

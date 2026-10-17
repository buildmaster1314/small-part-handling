from __future__ import annotations

from dataclasses import replace
import inspect
import io

import pytest
import torch

from fcpg.lewm_posterior_adapter import LeWMPosteriorAdapter
from fcpg.lewm_posterior_gate_training import (
    HARD_MAX_POSTERIOR_GATE_PARAMETERS,
    POSTERIOR_GATE_FORBIDDEN_MODEL_INPUTS,
    POSTERIOR_GATE_LOSS_GROUPS,
    POSTERIOR_GATE_MODEL_INPUTS,
    POSTERIOR_GATE_MODES,
    PosteriorGateLossMasks,
    PosteriorGateSplitIndices,
    PosteriorGateTrainingConfig,
    PosteriorGateTrainingRows,
    balanced_posterior_gate_loss,
    build_posterior_gate_forward_inputs,
    build_posterior_gate_train_loader,
    forward_posterior_gate,
    load_posterior_gate_checkpoint,
    no_lewm_comparator,
    posterior_adapter_architecture,
    posterior_adapter_parameter_count,
    train_posterior_gate,
)


def _rows() -> PosteriorGateTrainingRows:
    generator = torch.Generator().manual_seed(91)
    samples, anchors, latent_dim = 14, 2, 4
    measurement = 0.05 * torch.randn(
        samples, anchors, latent_dim, generator=generator
    )
    prior = measurement + 0.7 + 0.05 * torch.randn(
        samples, anchors, latent_dim, generator=generator
    )
    hold_last = measurement - 0.25
    canonical = prior.clone()
    q_nominal = 40.0 + 30.0 * torch.rand(
        samples, anchors, 2, generator=generator
    )
    q_corrected = q_nominal + torch.randn(
        samples, anchors, 2, generator=generator
    )
    valid_nominal = torch.ones(samples, anchors, dtype=torch.bool)
    validity_history = torch.randint(
        0,
        2,
        (samples, anchors, 3),
        generator=generator,
        dtype=torch.bool,
    )
    action = torch.randn(samples, 3, generator=generator)

    masks = {
        name: torch.zeros(samples, anchors, dtype=torch.bool)
        for name in POSTERIOR_GATE_LOSS_GROUPS
    }
    for index in range(samples):
        group = POSTERIOR_GATE_LOSS_GROUPS[index % len(POSTERIOR_GATE_LOSS_GROUPS)]
        anchor = 1 if group == "safety_indirect" else 0
        masks[group][index, anchor] = True
    return PosteriorGateTrainingRows(
        z_measurement=measurement,
        z_prior=prior,
        z_hold_last=hold_last,
        canonical_target=canonical,
        q_nominal=q_nominal,
        q_corrected=q_corrected,
        valid_nominal=valid_nominal,
        validity_history=validity_history,
        last_causal_action_chunk=action,
        loss_masks=PosteriorGateLossMasks(**masks),
        source_hashes={
            "teacher_forced_rows": "1" * 64,
            "offline_loss_masks": "2" * 64,
            "lewm_checkpoint": "3" * 64,
        },
        image_width=128,
        image_height=96,
    )


def _splits() -> PosteriorGateSplitIndices:
    return PosteriorGateSplitIndices(
        train=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        validation=(10, 11),
        test=(12, 13),
    )


def _config(**updates: object) -> PosteriorGateTrainingConfig:
    values: dict[str, object] = {
        "mode": "full",
        "seed": 17,
        "steps": 8,
        "batch_size": 10,
        "learning_rate": 2.0e-2,
        "hidden_dim": 6,
    }
    values.update(updates)
    return PosteriorGateTrainingConfig(**values)  # type: ignore[arg-type]


def test_fixed_adamw_training_and_sampling_are_bitwise_deterministic() -> None:
    first = train_posterior_gate(_rows(), _splits(), config=_config())
    second = train_posterior_gate(_rows(), _splits(), config=_config())

    assert first.seed == second.seed == 17
    assert first.step == second.step == 8
    assert first.sampled_index_trace_sha256 == second.sampled_index_trace_sha256
    assert first.group_sample_counts == second.group_sample_counts
    assert set(first.group_sample_counts.values()) == {16}
    assert first.initial_state_sha256 == second.initial_state_sha256
    assert first.adapter_state_sha256 == second.adapter_state_sha256
    assert first.checkpoint_payload["integrity_sha256"] == second.checkpoint_payload[
        "integrity_sha256"
    ]
    for name, value in first.adapter.state_dict().items():
        assert torch.equal(value, second.adapter.state_dict()[name])


def test_zero_initialized_adapter_is_exact_measurement_identity() -> None:
    rows = _rows()
    inputs = build_posterior_gate_forward_inputs(rows, (0, 1), mode="full")
    adapter = LeWMPosteriorAdapter(
        **posterior_adapter_architecture(rows, hidden_dim=6)
    )

    output = adapter(*inputs.as_args())

    assert torch.count_nonzero(adapter.output_layer.weight).item() == 0
    assert torch.count_nonzero(adapter.output_layer.bias).item() == 0
    assert torch.count_nonzero(output.beta).item() == 0
    assert torch.equal(output.z_post, inputs.z_measurement)


class _RecordingAdapter(LeWMPosteriorAdapter):
    seen: tuple[torch.Tensor, ...] | None = None

    def forward(
        self,
        z_measurement: torch.Tensor,
        z_prior: torch.Tensor,
        innovation: torch.Tensor,
        confidence: torch.Tensor,
        mask_history: torch.Tensor,
        action_summary: torch.Tensor,
    ):
        self.seen = (
            z_measurement.detach().clone(),
            z_prior.detach().clone(),
            innovation.detach().clone(),
            confidence.detach().clone(),
            mask_history.detach().clone(),
            action_summary.detach().clone(),
        )
        return super().forward(
            z_measurement,
            z_prior,
            innovation,
            confidence,
            mask_history,
            action_summary,
        )


def test_action_blind_and_no_lewm_ablate_only_their_designated_channels() -> None:
    rows = _rows()
    adapter = _RecordingAdapter(
        **posterior_adapter_architecture(rows, hidden_dim=0)
    )

    forward_posterior_gate(adapter, rows, (0, 1), mode="full")
    assert adapter.seen is not None
    assert torch.equal(adapter.seen[1], rows.z_prior[:2])
    assert torch.equal(adapter.seen[5], rows.last_causal_action_chunk[:2])

    forward_posterior_gate(adapter, rows, (0, 1), mode="action_blind")
    assert adapter.seen is not None
    assert torch.equal(adapter.seen[1], rows.z_prior[:2])
    assert torch.count_nonzero(adapter.seen[5]).item() == 0

    forward_posterior_gate(adapter, rows, (0, 1), mode="no_lewm")
    assert adapter.seen is not None
    assert torch.equal(adapter.seen[1], rows.z_hold_last[:2])
    assert torch.equal(adapter.seen[5], rows.last_causal_action_chunk[:2])
    expected_innovation = (rows.z_hold_last[:2] - rows.z_measurement[:2]).clamp(
        -1.0, 1.0
    )
    assert torch.equal(adapter.seen[2], expected_innovation)


def test_public_no_lewm_comparator_preserves_real_action_input() -> None:
    rows = _rows()
    inputs = build_posterior_gate_forward_inputs(rows, (0, 1), mode="full")
    adapter = _RecordingAdapter(
        **posterior_adapter_architecture(rows, hidden_dim=0)
    )

    no_lewm_comparator(
        adapter,
        inputs.z_measurement,
        rows.z_hold_last[:2],
        inputs.innovation,
        inputs.confidence,
        inputs.mask_history,
        inputs.action_summary,
    )

    assert adapter.seen is not None
    assert torch.equal(adapter.seen[1], rows.z_hold_last[:2])
    assert torch.equal(adapter.seen[5], inputs.action_summary)


def test_all_modes_have_identical_architecture_and_training_budget() -> None:
    rows = _rows()
    results = [
        train_posterior_gate(
            rows,
            _splits(),
            config=_config(mode=mode, steps=2, batch_size=5),
        )
        for mode in POSTERIOR_GATE_MODES
    ]

    assert len({result.parameter_count for result in results}) == 1
    assert [result.step for result in results] == [2, 2, 2]
    assert [result.seed for result in results] == [17, 17, 17]
    assert len(
        {
            tuple(result.adapter.checkpoint_config().items())
            for result in results
        }
    ) == 1
    assert len({result.initial_state_sha256 for result in results}) == 1


def test_balanced_loss_gives_each_primary_and_safety_group_equal_weight() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(5, 1, 1)
    target = torch.zeros_like(prediction)
    masks = {}
    for index, name in enumerate(POSTERIOR_GATE_LOSS_GROUPS):
        mask = torch.zeros(5, 1, dtype=torch.bool)
        mask[index] = True
        masks[name] = mask

    loss = balanced_posterior_gate_loss(prediction, target, masks)

    assert float(loss) == pytest.approx((1.0 + 4.0 + 9.0 + 16.0 + 25.0) / 5.0)


def test_dataloader_and_optimizer_only_use_explicit_train_indices() -> None:
    rows = _rows()
    splits = _splits()
    config = _config(steps=4, batch_size=10)
    loader = build_posterior_gate_train_loader(rows, splits, config=config)
    sampled = []
    for batch in loader:
        assert batch["row_index"].numel() == config.batch_size
        sampled.extend(batch["row_index"].tolist())
    assert len(sampled) == config.steps * config.batch_size
    assert set(sampled).issubset(set(splits.train))
    assert set(sampled).isdisjoint(set(splits.validation) | set(splits.test))

    def poison_holdout(value: torch.Tensor) -> torch.Tensor:
        changed = value.clone()
        changed[10:] = torch.nan
        return changed

    poisoned = replace(
        rows,
        z_measurement=poison_holdout(rows.z_measurement),
        z_prior=poison_holdout(rows.z_prior),
        z_hold_last=poison_holdout(rows.z_hold_last),
        canonical_target=poison_holdout(rows.canonical_target),
        q_nominal=poison_holdout(rows.q_nominal),
        q_corrected=poison_holdout(rows.q_corrected),
        last_causal_action_chunk=poison_holdout(rows.last_causal_action_chunk),
    )
    baseline = train_posterior_gate(rows, splits, config=config)
    changed = train_posterior_gate(poisoned, splits, config=config)

    assert baseline.adapter_state_sha256 == changed.adapter_state_sha256
    assert baseline.sampled_index_trace_sha256 == changed.sampled_index_trace_sha256
    assert baseline.checkpoint_payload["integrity_sha256"] == changed.checkpoint_payload[
        "integrity_sha256"
    ]
    assert set(changed.sampled_train_indices).issubset(set(splits.train))
    contract = changed.checkpoint_payload["split_contract"]
    assert contract["validation_used_for_optimization"] is False
    assert contract["validation_used_for_seed_selection"] is False
    assert contract["test_used_for_optimization"] is False
    assert contract["test_used_for_seed_selection"] is False
    parameters = inspect.signature(train_posterior_gate).parameters
    assert "validation" not in parameters
    assert "test" not in parameters


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = {str(key) for key in value}
        for child in value.values():
            result.update(_all_mapping_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_all_mapping_keys(child))
        return result
    return set()


def test_checkpoint_has_contract_fields_no_target_data_and_round_trips() -> None:
    rows = _rows()
    result = train_posterior_gate(rows, _splits(), config=_config(steps=3))
    payload = result.checkpoint_payload

    assert {"config", "state", "seed", "step", "source_hashes"}.issubset(payload)
    assert payload["seed"] == 17
    assert payload["step"] == 3
    assert payload["source_hashes"] == dict(rows.source_hashes)
    assert "canonical_target" not in _all_mapping_keys(payload)
    assert "target" not in _all_mapping_keys(payload)

    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    serialized = torch.load(stream, weights_only=True)
    restored = load_posterior_gate_checkpoint(
        serialized, expected_source_hashes=rows.source_hashes
    )
    for name, value in result.adapter.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    with pytest.raises(ValueError, match="source hashes mismatch"):
        load_posterior_gate_checkpoint(
            serialized,
            expected_source_hashes={"different": "f" * 64},
        )


def test_annotations_and_offline_masks_are_outside_forward_contract() -> None:
    assert set(POSTERIOR_GATE_FORBIDDEN_MODEL_INPUTS).isdisjoint(
        POSTERIOR_GATE_MODEL_INPUTS
    )
    parameters = inspect.signature(LeWMPosteriorAdapter.forward).parameters
    assert set(parameters).isdisjoint(POSTERIOR_GATE_FORBIDDEN_MODEL_INPUTS)
    assert "canonical_target" not in POSTERIOR_GATE_MODEL_INPUTS
    assert "loss_masks" not in POSTERIOR_GATE_MODEL_INPUTS


def test_parameter_and_split_contracts_fail_closed() -> None:
    rows = _rows()
    probe = LeWMPosteriorAdapter(
        **posterior_adapter_architecture(rows, hidden_dim=6)
    )
    count = posterior_adapter_parameter_count(probe)
    with pytest.raises(ValueError, match="parameter budget exceeded"):
        train_posterior_gate(
            rows,
            _splits(),
            config=_config(max_parameters=count - 1),
        )
    with pytest.raises(ValueError, match="20000-parameter"):
        _config(max_parameters=HARD_MAX_POSTERIOR_GATE_PARAMETERS + 1)
    with pytest.raises(ValueError, match="overlap"):
        PosteriorGateSplitIndices(train=(0, 1), validation=(1, 2), test=(3,))

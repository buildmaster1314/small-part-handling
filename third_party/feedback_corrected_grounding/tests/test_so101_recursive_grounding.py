from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from fcpg.so101_realtime_planner import _AnchorPosterior, recursive_anchor_seed
from scripts.evaluate_so101_real_shadow_fcpg_lewm import encode_fcpg_view


def _nominal(offset: float = 0.0) -> np.ndarray:
    base = np.asarray(
        [[100.0, 80.0], [112.0, 82.0], [88.0, 78.0]], dtype=np.float32
    )
    return np.stack((base - 2.0, base - 1.0, base)) + offset


def _posterior(
    nominal: np.ndarray,
    state: np.ndarray,
    frame: np.ndarray,
    *,
    correction: np.ndarray,
) -> _AnchorPosterior:
    return _AnchorPosterior(
        state_latest=state.copy(),
        frame_latest=frame.copy(),
        nominal_latest_px=nominal[-1].copy(),
        seed_history_px=nominal.copy(),
        corrected_latest_px=nominal[-1] + correction,
    )


def test_first_observation_uses_calibrated_nominal() -> None:
    nominal = _nominal()
    state = np.zeros(6, dtype=np.float32)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)

    seed, diagnostics = recursive_anchor_seed(nominal, state, frame, None)

    np.testing.assert_array_equal(seed, nominal)
    assert diagnostics["prior_source"] == "calibrated_nominal"
    np.testing.assert_array_equal(diagnostics["fk_delta_px"], 0.0)
    np.testing.assert_array_equal(diagnostics["posterior_offset_px"], 0.0)


def test_new_observation_propagates_posterior_with_fk_pixel_delta() -> None:
    previous_nominal = _nominal()
    current_nominal = previous_nominal + np.asarray([7.0, -3.0], dtype=np.float32)
    previous_state = np.zeros(6, dtype=np.float32)
    current_state = np.ones(6, dtype=np.float32)
    previous_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    current_frame = np.ones((224, 224, 3), dtype=np.uint8)
    correction = np.asarray(
        [[4.0, -2.0], [3.0, -1.0], [5.0, -3.0]], dtype=np.float32
    )
    posterior = _posterior(
        previous_nominal,
        previous_state,
        previous_frame,
        correction=correction,
    )

    seed, diagnostics = recursive_anchor_seed(
        current_nominal, current_state, current_frame, posterior
    )

    np.testing.assert_allclose(seed, current_nominal + correction[None])
    np.testing.assert_allclose(
        diagnostics["fk_delta_px"], current_nominal[-1] - previous_nominal[-1]
    )
    np.testing.assert_allclose(diagnostics["posterior_offset_px"], correction)
    assert diagnostics["prior_source"] == "posterior_plus_fk_delta"


def test_same_observation_reuses_original_seed_without_recursive_drift() -> None:
    nominal = _nominal()
    state = np.zeros(6, dtype=np.float32)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    stored_seed = nominal + 6.0
    posterior = _AnchorPosterior(
        state_latest=state.copy(),
        frame_latest=frame.copy(),
        nominal_latest_px=nominal[-1].copy(),
        seed_history_px=stored_seed.copy(),
        corrected_latest_px=stored_seed[-1] + 2.0,
    )

    seed, diagnostics = recursive_anchor_seed(nominal, state, frame, posterior)

    np.testing.assert_array_equal(seed, stored_seed)
    assert diagnostics["prior_source"] == "same_observation_replay"
    np.testing.assert_allclose(diagnostics["posterior_offset_px"], 6.0)


def test_fcpg_search_seed_does_not_replace_bridge_geometric_nominal() -> None:
    class Corrector:
        def __call__(self, _feature_map, query, *_args, **_kwargs):
            self.query = query.detach().cpu().numpy()
            return SimpleNamespace(q_corrected_px=query + 1.5)

    class Bridge:
        def __call__(
            self,
            _context,
            _patches,
            nominal,
            corrected,
            *_args,
            **_kwargs,
        ):
            self.nominal = nominal.detach().cpu().numpy()
            self.corrected = corrected.detach().cpu().numpy()
            return SimpleNamespace(
                context_embedding=torch.zeros((1, 3, 8), dtype=torch.float32)
            )

    class System:
        def __init__(self) -> None:
            self.corrector = Corrector()
            self.bridge = Bridge()

        def _encode_shared_vision(self, pixels):
            return (
                torch.zeros((1, 3, 8), dtype=torch.float32),
                torch.zeros((1, 3, 4, 8), dtype=torch.float32),
                torch.zeros((3, 8, 16, 16), dtype=torch.float32),
            )

    system = System()
    nominal = _nominal()
    seed = nominal + 9.0
    frames = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    states = np.zeros((3, 6), dtype=np.float32)
    xyz = np.zeros((3, 3, 3), dtype=np.float32)
    valid = np.ones((3, 3), dtype=bool)
    camera = np.zeros((3, 5), dtype=np.float32)

    _, _, corrected = encode_fcpg_view(
        system,
        frames,
        states,
        xyz,
        nominal,
        valid,
        camera,
        torch.device("cpu"),
        correction_seed_px=seed,
    )

    np.testing.assert_array_equal(system.corrector.query, seed)
    np.testing.assert_array_equal(system.bridge.nominal[0], nominal)
    np.testing.assert_allclose(system.bridge.corrected[0], seed + 1.5)
    np.testing.assert_allclose(corrected, seed + 1.5)


def test_fcpg_tip_midpoint_is_derived_before_lewm_bridge() -> None:
    class Corrector:
        def __call__(self, _feature_map, query, *_args, **_kwargs):
            corrected = query.clone()
            corrected[:, 0] += torch.tensor([40.0, -30.0])
            corrected[:, 1] += torch.tensor([2.0, 4.0])
            corrected[:, 2] += torch.tensor([6.0, 8.0])
            return SimpleNamespace(q_corrected_px=corrected)

    class Bridge:
        def __call__(self, _context, _patches, _nominal, corrected, *_args, **_kwargs):
            self.corrected = corrected.detach().cpu().numpy()
            return SimpleNamespace(
                context_embedding=torch.zeros((1, 3, 8), dtype=torch.float32)
            )

    class System:
        def __init__(self) -> None:
            self.corrector = Corrector()
            self.bridge = Bridge()

        def _encode_shared_vision(self, _pixels):
            return (
                torch.zeros((1, 3, 8), dtype=torch.float32),
                torch.zeros((1, 3, 4, 8), dtype=torch.float32),
                torch.zeros((3, 8, 16, 16), dtype=torch.float32),
            )

    system = System()
    nominal = _nominal()
    _, _, corrected = encode_fcpg_view(
        system,
        np.zeros((3, 224, 224, 3), dtype=np.uint8),
        np.zeros((3, 6), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.float32),
        nominal,
        np.ones((3, 3), dtype=bool),
        np.zeros((3, 5), dtype=np.float32),
        torch.device("cpu"),
        derive_center_from_tips=True,
    )

    expected_center = corrected[:, 1:].mean(axis=1)
    np.testing.assert_array_equal(corrected[:, 0], expected_center)
    np.testing.assert_array_equal(system.bridge.corrected[0, :, 0], expected_center)
    np.testing.assert_array_equal(corrected[:, 1], nominal[:, 1] + [2.0, 4.0])
    np.testing.assert_array_equal(corrected[:, 2], nominal[:, 2] + [6.0, 8.0])

import numpy as np
import pytest

from scripts.summarize_policy_factorial import factorial_effects


def test_factorial_effects_report_main_effects_and_interaction() -> None:
    effects = factorial_effects(
        predictive_sig0=np.asarray([0.0, 1.0, 0.0, 1.0]),
        predictive_sigreg=np.asarray([1.0, 1.0, 0.0, 1.0]),
        action_sig0=np.asarray([0.0, 0.0, 0.0, 1.0]),
        action_sigreg=np.asarray([0.0, 1.0, 0.0, 1.0]),
        bootstrap_samples=1000,
        seed=0,
    )
    assert effects["sigreg_under_predictive"]["mean"] == pytest.approx(0.25)
    assert effects["sigreg_under_action_only"]["mean"] == pytest.approx(0.25)
    assert effects["next_latent_under_sig0"]["mean"] == pytest.approx(0.25)
    assert effects["next_latent_under_sigreg"]["mean"] == pytest.approx(0.25)
    assert effects["interaction"]["mean"] == pytest.approx(0.0)


def test_factorial_effects_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        factorial_effects(
            np.zeros(2),
            np.zeros(3),
            np.zeros(2),
            np.zeros(2),
            bootstrap_samples=10,
            seed=0,
        )

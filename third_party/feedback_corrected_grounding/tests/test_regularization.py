from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from fcpg.regularization import SIGReg


def load_upstream_sigreg():
    source = Path(__file__).parents[1] / "third_party" / "lewm" / "module.py"
    spec = importlib.util.spec_from_file_location("vendored_lewm_module", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SIGReg


def test_sigreg_matches_vendored_lewm_batch_statistic() -> None:
    projection = torch.randn(3, 32, 16)
    local = SIGReg(knots=17, num_proj=64)
    upstream = load_upstream_sigreg()(knots=17, num_proj=64)

    torch.manual_seed(123)
    actual = local(projection)
    torch.manual_seed(123)
    expected = upstream(projection)

    torch.testing.assert_close(actual, expected)


def test_sigreg_two_dimensional_input_matches_single_time_step() -> None:
    projection = torch.randn(32, 16)
    regularizer = SIGReg(knots=17, num_proj=64)

    torch.manual_seed(456)
    actual = regularizer(projection)
    torch.manual_seed(456)
    expected = regularizer(projection.unsqueeze(0))

    torch.testing.assert_close(actual, expected)

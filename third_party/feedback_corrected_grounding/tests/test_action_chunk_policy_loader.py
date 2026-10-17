from __future__ import annotations

from typing import Any

import pytest
import torch

import fcpg.action_chunk_policy as action_chunk_policy


class _FakeActionPolicy(torch.nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        fcpg_adapter: torch.nn.Module | None,
        pretrained_vision: bool,
        reuse_fcpg_vision: bool,
        **_: Any,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fcpg_adapter = fcpg_adapter
        self.pretrained_vision = bool(pretrained_vision)
        self.reuse_fcpg_vision = bool(reuse_fcpg_vision)
        self.normalization: object | None = None
        self.reliability_override: str | None = None

    def set_normalization(self, normalization: object) -> None:
        self.normalization = normalization

    def set_reliability_override(self, mode: str) -> None:
        self.reliability_override = mode


def _checkpoint(mode: str = "fcpg") -> dict[str, object]:
    return {
        "schema_version": 1,
        "args": {
            "mode": mode,
            "fcpg_checkpoint": "embedded-relative-corrector.pt",
        },
        "obs_dim": 4,
        "action_dim": 4,
        "num_keypoints": 3,
        "model_state": {},
        "normalization": {},
    }


def test_loader_skips_pretrained_fetch_only_when_fcpg_vision_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint()
    checkpoint["args"] = {
        **checkpoint["args"],
        "reuse_fcpg_vision": True,
    }
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        action_chunk_policy,
        "GroundedActionChunkTransformer",
        _FakeActionPolicy,
    )
    override = torch.nn.Identity()
    policy, _ = action_chunk_policy.load_action_chunk_policy(
        "policy.pt",
        fcpg_adapter_override=override,
    )
    assert policy.reuse_fcpg_vision is True
    assert policy.pretrained_vision is False


def test_loader_keeps_pretrained_initialization_for_nonreuse_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: _checkpoint(),
    )
    monkeypatch.setattr(
        action_chunk_policy,
        "GroundedActionChunkTransformer",
        _FakeActionPolicy,
    )
    policy, _ = action_chunk_policy.load_action_chunk_policy(
        "policy.pt",
        fcpg_adapter_override=torch.nn.Identity(),
    )
    assert policy.reuse_fcpg_vision is False
    assert policy.pretrained_vision is True


def test_loader_allows_explicit_offline_compact_carrier_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: _checkpoint(),
    )
    monkeypatch.setattr(
        action_chunk_policy,
        "GroundedActionChunkTransformer",
        _FakeActionPolicy,
    )
    policy, _ = action_chunk_policy.load_action_chunk_policy(
        "policy.pt",
        fcpg_adapter_override=torch.nn.Identity(),
        pretrained_vision_override=False,
    )
    assert policy.reuse_fcpg_vision is False
    assert policy.pretrained_vision is False


def test_loader_uses_verified_fcpg_adapter_override_without_reloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = torch.nn.Identity()
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: _checkpoint(),
    )
    monkeypatch.setattr(
        action_chunk_policy,
        "GroundedActionChunkTransformer",
        _FakeActionPolicy,
    )

    def fail_reload(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the embedded corrector path must not be reloaded")

    monkeypatch.setattr(action_chunk_policy, "load_fcpg_vit_adapter", fail_reload)

    policy, args = action_chunk_policy.load_action_chunk_policy(
        "policy.pt",
        fcpg_adapter_override=override,
    )

    assert policy.fcpg_adapter is override
    assert args["fcpg_checkpoint"] == "embedded-relative-corrector.pt"


def test_loader_preserves_legacy_embedded_corrector_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = torch.nn.Identity()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: _checkpoint(),
    )
    monkeypatch.setattr(
        action_chunk_policy,
        "GroundedActionChunkTransformer",
        _FakeActionPolicy,
    )

    def fake_load(path: str, device: object) -> tuple[torch.nn.Module, dict]:
        calls.append((path, device))
        return loaded, {}

    monkeypatch.setattr(action_chunk_policy, "load_fcpg_vit_adapter", fake_load)

    policy, _ = action_chunk_policy.load_action_chunk_policy("policy.pt")

    assert policy.fcpg_adapter is loaded
    assert calls == [("embedded-relative-corrector.pt", "cpu")]


def test_loader_rejects_override_for_non_fcpg_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        action_chunk_policy.torch,
        "load",
        lambda *_args, **_kwargs: _checkpoint(mode="vision"),
    )

    with pytest.raises(ValueError, match="requires an fcpg policy"):
        action_chunk_policy.load_action_chunk_policy(
            "policy.pt",
            fcpg_adapter_override=torch.nn.Identity(),
        )

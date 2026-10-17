from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.lewm_action_branch_collection import CollectionPayload
from scripts import collect_lewm_action_branch_a3_confirm as module


def test_cli_rejects_no_go_before_loading_screen_or_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "no-go.json"
    summary.write_text(
        json.dumps(
            {
                "protocol": "fcpg_action_branch_physical_support_a3_v1",
                "stage": "a3_2k_development",
                "decision": "Development-No-Go",
                "passed": False,
                "input_contains_test_rows": False,
                "checkpoint_step": 2000,
                "primary_horizons": [6, 7, 8],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "load_a1_npz",
        lambda _path: (_ for _ in ()).throw(AssertionError("screen was loaded")),
    )
    monkeypatch.setattr(
        module,
        "load_frozen_fcpg_anchor_encoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("encoder was loaded")
        ),
    )

    with pytest.raises(ValueError, match="Development-Go"):
        module.main(["--development-summary", str(summary)])


def test_cli_validates_before_encoder_then_writes_only_merged_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = json.loads(module.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    screen = {
        "split": np.asarray(["train", "development"]),
        "target_embedding": np.zeros((2, 8, 8, 3, 192), dtype=np.float32),
    }
    fresh_test = {
        "split": np.asarray(["test"]),
        "target_embedding": np.zeros((1, 8, 8, 3, 192), dtype=np.float32),
    }
    canonical = {
        "split": np.asarray(["train", "test"]),
        "target_embedding": np.zeros((2, 8, 8, 3, 192), dtype=np.float32),
    }
    summary = {"decision": "Development-Go"}
    events: list[str] = []
    output = tmp_path / "canonical.npz"

    monkeypatch.setattr(module, "_load_contract", lambda _path: contract)

    def load_summary(_path: Path, *, contract: dict) -> dict:
        events.append("summary")
        assert contract is not None
        return summary

    def load_screen(_path: Path) -> dict:
        events.append("screen")
        return screen

    def validate_screen(arrays: dict, config: dict) -> None:
        events.append("screen-conformance")
        assert arrays is screen
        assert config is contract

    def load_encoder(*_args: object, **_kwargs: object) -> object:
        events.append("encoder")
        return object()

    def collect(
        config: dict,
        *,
        development_summary: dict,
        encoder: object,
        runtime_factory: object,
    ) -> CollectionPayload:
        events.append("collect")
        assert config is contract
        assert development_summary is summary
        assert encoder is not None and runtime_factory is not None
        return CollectionPayload(arrays=fresh_test, metadata={})

    def merge(train_screen: dict, test: dict, config: dict) -> dict:
        events.append("merge")
        assert train_screen is screen
        assert test is fresh_test
        assert config is contract
        return canonical

    def save(path: Path, arrays: dict) -> Path:
        events.append("save")
        assert path == output.resolve()
        assert arrays is canonical
        return path

    monkeypatch.setattr(module, "load_a3_development_go_summary", load_summary)
    monkeypatch.setattr(module, "load_a1_npz", load_screen)
    monkeypatch.setattr(module, "validate_a3_cache_conformance", validate_screen)
    monkeypatch.setattr(module, "load_frozen_fcpg_anchor_encoder", load_encoder)
    monkeypatch.setattr(module, "run_a3_confirmation_collection", collect)
    monkeypatch.setattr(module, "merge_a3_confirmation_canonical", merge)
    monkeypatch.setattr(module, "save_a1_npz", save)

    result = module.main(
        [
            "--development-summary",
            str(tmp_path / "summary.json"),
            "--screen",
            str(tmp_path / "screen.npz"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert events == [
        "summary",
        "screen",
        "screen-conformance",
        "encoder",
        "collect",
        "merge",
        "save",
    ]
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "development_bundles": 0,
        "output": str(output.resolve()),
        "test_bundles": 1,
        "train_bundles": 1,
    }


def test_cli_refuses_to_overwrite_the_development_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen = tmp_path / "screen.npz"
    monkeypatch.setattr(module, "_load_contract", lambda _path: {})
    monkeypatch.setattr(
        module,
        "load_a3_development_go_summary",
        lambda _path, *, contract: {},
    )
    monkeypatch.setattr(module, "load_a1_npz", lambda _path: {})
    monkeypatch.setattr(module, "validate_a3_cache_conformance", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "load_frozen_fcpg_anchor_encoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("encoder was loaded")
        ),
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        module.main(["--screen", str(screen), "--output", str(screen)])

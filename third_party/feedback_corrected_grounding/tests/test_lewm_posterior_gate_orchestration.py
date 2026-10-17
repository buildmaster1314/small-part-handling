from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

import fcpg.lewm_posterior_gate_orchestration as orchestration
from fcpg.lewm_posterior_gate_cache import PosteriorGateCache
from fcpg.lewm_posterior_gate_orchestration import (
    FIXED_METHODS,
    PosteriorGateInputs,
    load_posterior_gate_inputs,
    merge_posterior_gate_records,
    run_validation_posterior_gate,
    validate_posterior_gate_config,
)
from fcpg.lewm_posterior_gate_runtime import (
    ACTION_BLIND_METHOD,
    ACTION_SHUFFLE_METHOD,
    ADAPTER_METHOD,
    MEASUREMENT_METHOD,
    NO_LEWM_METHOD,
    NO_WRITEBACK_METHOD,
)
from fcpg.lewm_posterior_gate_selection import (
    OPTIMIZATION_SEEDS,
    TASKS,
)
from fcpg.lewm_posterior_gate_training import POSTERIOR_GATE_MODES
from fcpg.lewm_provenance import load_yaml_config, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "config/lewm/train/anchor_persistent_posterior_adapter_gate_v1.yaml"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_lewm_posterior_adapter_gate.py"


def _current_config() -> dict[str, Any]:
    return copy.deepcopy(load_yaml_config(CONFIG_PATH))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _cache(
    name: str,
    checkpoint_sha256: str,
    *,
    fcpg_checkpoint_sha256: str | None = None,
    clean: bool = False,
    episode_hashes: tuple[str, ...] | None = None,
) -> PosteriorGateCache:
    if fcpg_checkpoint_sha256 is None:
        fcpg_checkpoint_sha256 = _digest("placeholder-fcpg")
    if episode_hashes is None:
        episode_hashes = tuple(_digest(f"{name}:{task}") for task in TASKS)
    if len(episode_hashes) != len(TASKS):
        raise ValueError("fixture must provide one physical episode per task")
    count = len(TASKS)
    return PosteriorGateCache(
        path=Path(f"/synthetic/{name}"),
        metadata={
            "image_width": 224,
            "image_height": 224,
            "contract": {
                "evaluation_view": (
                    "canonical_unoccluded_only"
                    if clean
                    else "perturbed_observation"
                ),
                "clean_evaluation_only": clean,
            },
            "inputs": {
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "step": 2_000,
                },
                **{
                    role: {
                        "corrector_checkpoint_sha256": fcpg_checkpoint_sha256
                    }
                    for role in (
                        "observed_correction_cache",
                        "measurement_correction_cache",
                        "canonical_correction_cache",
                    )
                },
            },
        },
        deployment={
            "measurement_current": np.zeros((count, 1, 1), dtype=np.float32)
        },
        labels={},
        evaluation={
            "task_name": np.asarray(TASKS),
            "physical_episode_sha256": np.asarray(episode_hashes),
        },
        file_hashes={"metadata": _digest(f"{name}:metadata")},
    )


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "posterior_gate_runner_for_test", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(method: str, *, episode: str = "episode") -> dict[str, Any]:
    return {
        "task": TASKS[0],
        "physical_episode_id": episode,
        "anchor": 0,
        "estimand": "post_update",
        "event": "o1",
        "method": method,
        "mse": 1.0,
    }


def _install_synthetic_input_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caches: Mapping[str, PosteriorGateCache],
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint = tmp_path / "lewm.pt"
    checkpoint.write_bytes(b"frozen-lewm-fixture")
    fcpg_checkpoint = tmp_path / "fcpg.pt"
    fcpg_checkpoint.write_bytes(b"frozen-fcpg-fixture")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = _current_config()
    for split in ("train", "validation", "validation_clean"):
        config["splits"][split]["episodes_per_task"] = 1
    config["frozen_upstream"]["lewm_checkpoint"] = str(checkpoint)
    config["frozen_upstream"]["lewm_checkpoint_sha256"] = sha256_file(checkpoint)
    config["frozen_upstream"]["fcpg_checkpoint"] = str(fcpg_checkpoint)
    config["frozen_upstream"]["fcpg_checkpoint_sha256"] = sha256_file(
        fcpg_checkpoint
    )
    config["cache"]["device"] = "cpu"
    monkeypatch.setattr(orchestration, "load_yaml_config", lambda _path: config)
    monkeypatch.setattr(
        orchestration,
        "load_posterior_gate_cache",
        lambda path: caches[Path(path).name],
    )
    return config_path, checkpoint, config


def test_current_yaml_validates_and_runner_has_no_sealed_argument() -> None:
    config = _current_config()
    validate_posterior_gate_config(config)
    assert config["comparators"]["required_sequential"] == [
        "selected_fixed_beta",
        "no_lewm",
    ]
    assert "action_blind" in config["comparators"]["secondary"]

    parser = _load_runner().build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert all("sealed" not in option for option in option_strings)
    assert option_strings.issuperset(
        {
            "--config",
            "--train-cache",
            "--validation-cache",
            "--validation-clean-cache",
            "--output-dir",
        }
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train-cache",
                "train",
                "--validation-cache",
                "validation",
                "--validation-clean-cache",
                "clean",
                "--output-dir",
                "output",
                "--sealed-test-cache",
                "forbidden",
            ]
        )


def test_record_merge_rejects_duplicate_statistic_cells() -> None:
    row = _record(MEASUREMENT_METHOD)
    assert merge_posterior_gate_records([row]) == [row]
    with pytest.raises(ValueError, match="duplicate posterior Gate statistic cell"):
        merge_posterior_gate_records([row], [dict(row, mse=9.0)])


@pytest.mark.parametrize(
    ("bad_split", "clean", "message"),
    (
        ("train", True, "train cache evaluation_view"),
        ("validation", True, "validation cache evaluation_view"),
        ("validation_clean", False, "validation_clean cache evaluation_view"),
    ),
)
def test_input_loader_rejects_wrong_cache_roles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_split: str,
    clean: bool,
    message: str,
) -> None:
    checkpoint_sha = _digest("placeholder")
    caches = {
        "train": _cache("train", checkpoint_sha),
        "validation": _cache("validation", checkpoint_sha),
        "validation_clean": _cache(
            "validation_clean", checkpoint_sha, clean=True
        ),
    }
    caches[bad_split] = _cache(bad_split, checkpoint_sha, clean=clean)
    config_path, checkpoint, config = _install_synthetic_input_loader(
        monkeypatch, tmp_path, caches
    )
    expected_sha = sha256_file(checkpoint)
    config["frozen_upstream"]["lewm_checkpoint_sha256"] = expected_sha
    for cache in caches.values():
        cache.metadata["inputs"]["checkpoint"]["sha256"] = expected_sha
        for role in (
            "observed_correction_cache",
            "measurement_correction_cache",
            "canonical_correction_cache",
        ):
            cache.metadata["inputs"][role]["corrector_checkpoint_sha256"] = config[
                "frozen_upstream"
            ]["fcpg_checkpoint_sha256"]

    with pytest.raises(ValueError, match=message):
        load_posterior_gate_inputs(
            config_path,
            "train",
            "validation",
            "validation_clean",
            repo_root=tmp_path,
        )


def test_input_loader_accepts_roles_but_rejects_episode_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_sha = _digest("placeholder")
    shared = _digest("shared-physical-episode")
    caches = {
        "train": _cache(
            "train",
            checkpoint_sha,
            episode_hashes=(shared, *tuple(_digest(f"train:{i}") for i in range(3))),
        ),
        "validation": _cache(
            "validation",
            checkpoint_sha,
            episode_hashes=(
                shared,
                *tuple(_digest(f"validation:{i}") for i in range(3)),
            ),
        ),
        "validation_clean": _cache(
            "validation_clean", checkpoint_sha, clean=True
        ),
    }
    config_path, checkpoint, config = _install_synthetic_input_loader(
        monkeypatch, tmp_path, caches
    )
    expected_sha = sha256_file(checkpoint)
    config["frozen_upstream"]["lewm_checkpoint_sha256"] = expected_sha
    for cache in caches.values():
        cache.metadata["inputs"]["checkpoint"]["sha256"] = expected_sha
        for role in (
            "observed_correction_cache",
            "measurement_correction_cache",
            "canonical_correction_cache",
        ):
            cache.metadata["inputs"][role]["corrector_checkpoint_sha256"] = config[
                "frozen_upstream"
            ]["fcpg_checkpoint_sha256"]

    with pytest.raises(ValueError, match="train/validation physical episodes overlap"):
        load_posterior_gate_inputs(
            config_path,
            "train",
            "validation",
            "validation_clean",
            repo_root=tmp_path,
        )


def test_mocked_run_trains_fixed_endpoints_selects_validation_only_and_publishes_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _current_config()
    config["cache"]["device"] = "cpu"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    checkpoint = tmp_path / "lewm.pt"
    checkpoint.write_bytes(b"frozen-lewm-fixture")
    checkpoint_sha = sha256_file(checkpoint)
    fcpg_checkpoint = tmp_path / "fcpg.pt"
    fcpg_checkpoint.write_bytes(b"frozen-fcpg-fixture")
    fcpg_checkpoint_sha = sha256_file(fcpg_checkpoint)
    train = _cache("train", checkpoint_sha)
    validation = _cache("validation", checkpoint_sha)
    clean = _cache("validation_clean", checkpoint_sha, clean=True)
    inputs = PosteriorGateInputs(
        config_path=config_path,
        config=config,
        train=train,
        validation=validation,
        validation_clean=clean,
        lewm_checkpoint=checkpoint,
        lewm_checkpoint_sha256=checkpoint_sha,
        fcpg_checkpoint=fcpg_checkpoint,
        fcpg_checkpoint_sha256=fcpg_checkpoint_sha,
        device=torch.device("cpu"),
    )

    rows = SimpleNamespace(source_hashes={"fixture": _digest("fixture-rows")})
    monkeypatch.setattr(
        orchestration,
        "concatenate_training_rows",
        lambda caches: (rows, ((0, 2), (2, 3))),
    )

    class FakeLeWM(torch.nn.Module):
        pass

    class FakeAdapter(torch.nn.Module):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    monkeypatch.setattr(orchestration, "AnchorPersistentLeWM", FakeLeWM)
    monkeypatch.setattr(
        orchestration,
        "load_lewm_grounded_checkpoint",
        lambda path, device: (FakeLeWM(), {}, {"step": 2_000}),
    )

    train_calls: list[tuple[int, str, int, tuple[int, ...], tuple[int, ...]]] = []

    def fake_train(
        _rows: object,
        split: object,
        *,
        config: object,
        device: torch.device,
    ) -> SimpleNamespace:
        assert _rows is rows
        assert device.type == "cpu"
        train_calls.append(
            (
                config.seed,
                config.mode,
                config.steps,
                tuple(split.train),
                tuple(split.validation),
            )
        )
        return SimpleNamespace(
            checkpoint_payload={
                "mode": config.mode,
                "seed": config.seed,
            },
            train_loss=0.25,
            parameter_count=7_137,
            step=config.steps,
        )

    monkeypatch.setattr(orchestration, "train_posterior_gate", fake_train)

    def fake_load_adapter(
        payload: Mapping[str, Any], *, expected_source_hashes: Mapping[str, str]
    ) -> FakeAdapter:
        del expected_source_hashes
        return FakeAdapter(str(payload["mode"]))

    monkeypatch.setattr(
        orchestration, "load_posterior_gate_checkpoint", fake_load_adapter
    )

    evaluated_cache_names: list[str] = []

    def fake_common(
        _model: object,
        cache: PosteriorGateCache,
        _adapter: object,
        *,
        device: torch.device,
    ) -> list[dict[str, Any]]:
        assert device.type == "cpu"
        evaluated_cache_names.append(cache.path.name)
        methods = (MEASUREMENT_METHOD, *FIXED_METHODS.values())
        return [_record(method, episode=cache.path.name) for method in methods]

    def fake_seed(
        _model: object,
        cache: PosteriorGateCache,
        full_adapter: FakeAdapter,
        action_blind_adapter: FakeAdapter,
        no_lewm_adapter: FakeAdapter,
        *,
        device: torch.device,
    ) -> list[dict[str, Any]]:
        assert full_adapter.mode == "full"
        assert action_blind_adapter.mode == "action_blind"
        assert no_lewm_adapter.mode == "no_lewm"
        assert device.type == "cpu"
        evaluated_cache_names.append(cache.path.name)
        return [
            _record(method, episode=cache.path.name)
            for method in (
                ADAPTER_METHOD,
                NO_LEWM_METHOD,
                ACTION_BLIND_METHOD,
                NO_WRITEBACK_METHOD,
                ACTION_SHUFFLE_METHOD,
            )
        ]

    monkeypatch.setattr(orchestration, "_evaluate_common_records", fake_common)
    monkeypatch.setattr(orchestration, "_evaluate_seed_records", fake_seed)

    summary_calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_summary(
        records: object,
        clean_records: object,
        *,
        adapter_method: str,
        fixed_methods: tuple[str, ...],
    ) -> dict[str, Any]:
        assert isinstance(records, list) and records
        assert isinstance(clean_records, list) and clean_records
        assert {NO_LEWM_METHOD, ACTION_BLIND_METHOD}.issubset(
            {str(row["method"]) for row in records}
        )
        summary_calls.append((adapter_method, tuple(fixed_methods)))
        return {
            "split": "validation",
            "source_split": "validation",
            "sealed_data_used": False,
            "adapter_method": adapter_method,
        }

    monkeypatch.setattr(orchestration, "_summary", fake_summary)

    selection_calls: list[str] = []

    def fake_select(
        fixed_summaries: Mapping[float, Mapping[str, Any]],
        seed_summaries: Mapping[int, Mapping[str, Any]],
        *,
        source_split: str,
    ) -> dict[str, Any]:
        selection_calls.append(source_split)
        assert source_split == "validation"
        assert set(fixed_summaries) == set(FIXED_METHODS)
        assert set(seed_summaries) == set(OPTIMIZATION_SEEDS)
        assert all(
            summary["source_split"] == "validation"
            and summary["sealed_data_used"] is False
            for summary in (*fixed_summaries.values(), *seed_summaries.values())
        )
        return {
            "decision": "No-Go",
            "passed": False,
            "reasons": [],
            "selection_scope": {
                "source_split": "validation",
                "validation_only": True,
                "sealed_data_used": False,
            },
        }

    monkeypatch.setattr(
        orchestration, "select_validation_only_posterior_gate", fake_select
    )

    destination = tmp_path / "formal-validation-output"
    real_publish = orchestration._publish_directory_no_replace
    publication_checks: list[Path] = []

    def checked_publish(source: Path, target: Path) -> None:
        assert target == destination
        assert not target.exists()
        assert source.parent == destination.parent
        assert source.name.startswith(f".{destination.name}.staging-")
        assert (source / "decision.json").is_file()
        assert not (source / "provenance.json").exists()
        assert (source / "summary.md").is_file()
        assert len(tuple((source / "checkpoints").glob("*.pt"))) == 15
        publication_checks.append(source)
        real_publish(source, target)

    monkeypatch.setattr(
        orchestration, "_publish_directory_no_replace", checked_publish
    )

    result = run_validation_posterior_gate(inputs, destination)

    assert len(train_calls) == 15
    assert {(seed, mode) for seed, mode, *_ in train_calls} == {
        (seed, mode)
        for seed in OPTIMIZATION_SEEDS
        for mode in POSTERIOR_GATE_MODES
    }
    assert {steps for _, _, steps, _, _ in train_calls} == {2_000}
    assert {train_indices for *_, train_indices, _ in train_calls} == {(0, 1)}
    assert {validation_indices for *_, validation_indices in train_calls} == {
        (2,)
    }
    assert selection_calls == ["validation"]
    assert set(evaluated_cache_names) == {"validation", "validation_clean"}
    assert "train" not in evaluated_cache_names
    assert len(summary_calls) == len(OPTIMIZATION_SEEDS) + len(FIXED_METHODS)
    assert publication_checks
    assert destination.is_dir()
    assert not tuple(tmp_path.glob(f".{destination.name}.staging-*"))
    assert len(tuple((destination / "checkpoints").glob("*.pt"))) == 15
    assert result["decision"]["selection_scope"]["validation_only"] is True
    assert "provenance" not in result
    run_metadata = result["run_metadata"]
    checkpoints = run_metadata["adapter_checkpoints"]
    assert len(checkpoints) == 15
    assert {(item["mode"], item["seed"]) for item in checkpoints} == {
        (mode, seed)
        for mode in POSTERIOR_GATE_MODES
        for seed in OPTIMIZATION_SEEDS
    }
    assert {item["optimizer_steps"] for item in checkpoints} == {2_000}
    assert {item["parameter_count"] for item in checkpoints} == {7_137}
    assert all(str(item["path"]).startswith("checkpoints/") for item in checkpoints)
    assert run_metadata["frozen_lewm"]["path"] == str(checkpoint)
    assert run_metadata["frozen_lewm"]["sha256"] == checkpoint_sha
    assert run_metadata["frozen_fcpg"]["path"] == str(fcpg_checkpoint)
    assert run_metadata["frozen_fcpg"]["sha256"] == fcpg_checkpoint_sha

    calls_before_rejected_overwrite = len(train_calls)
    with pytest.raises(FileExistsError):
        run_validation_posterior_gate(inputs, destination)
    assert len(train_calls) == calls_before_rejected_overwrite

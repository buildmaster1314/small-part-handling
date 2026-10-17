from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_lewm_posterior_dagger_pick import (
    DEFAULT_CONFIG,
    METHODS,
    build_plan,
    execution_environment,
    load_config,
    main,
    validate_config,
)


def _config(tmp_path: Path | None = None) -> dict[str, object]:
    config = load_config(DEFAULT_CONFIG)
    if tmp_path is not None:
        config["orchestration"]["output_root"] = str(tmp_path / "dagger")
    return config


def _argument(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_fixed_protocol_contract_is_valid() -> None:
    config = _config()
    validate_config(config)
    assert config["policy_seed"] == 0
    assert config["model_seed"] == 15801
    assert config["drop_duration_decisions"] == 6
    assert config["drop_start_min_decision"] == 6
    assert config["drop_start_max_decision"] == 9
    assert config["occlusion_center_key"] == "q_nominal"
    assert config["rollout"]["horizon_raw_steps"] == 335


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("policy_seed",), 1, "policy seed 0"),
        (("model_seed",), 15802, "LeWM seed 15801"),
        (("drop_duration_decisions",), 1, "six decisions"),
        (("drop_start_max_decision",), 10, "6 through 9"),
        (("occlusion_center_key",), "q_true", "q_nominal"),
        (("camera_rotation_deg",), 4.0, "five degrees"),
        (("rollout", "horizon_raw_steps"), 330, "335 raw steps"),
        (("orchestration", "device"), "cuda:1", "cuda:0"),
        (("orchestration", "cuda_visible_devices"), "1", "GPU 0"),
    ],
)
def test_fixed_protocol_drift_is_rejected(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_plan_has_two_rounds_and_method_matched_heads(tmp_path: Path) -> None:
    plan = build_plan(_config(tmp_path), smoke=False)
    assert len(plan.jobs) == 15
    round0_collect = [job for job in plan.jobs if job.name.startswith("collect_r0")]
    round1_collect = [job for job in plan.jobs if job.name.startswith("collect_r1")]
    assert len(round0_collect) == 6
    assert len(round1_collect) == 6
    assert all("--head-checkpoint" not in job.command for job in round0_collect)

    seed = plan.final_config["training"]["seeds"][0]
    for job in round1_collect:
        method = _argument(job.command, "--method")
        checkpoint = Path(_argument(job.command, "--head-checkpoint"))
        assert method in METHODS
        assert checkpoint.name == f"{method}_seed{seed}_latest.pt"
        assert checkpoint.parent.name == "round0"

    assert plan.jobs[6].name == "train_round0"
    assert plan.jobs[-2].name == "train_final"
    assert plan.jobs[-1].name == "fresh_paired_rollout"


def test_final_training_aggregates_round_zero_and_round_one(tmp_path: Path) -> None:
    plan = build_plan(_config(tmp_path), smoke=False)
    for split in ("train", "development"):
        for method in METHODS:
            round0_paths = plan.round0_config["caches"][split][method]
            final_paths = plan.final_config["caches"][split][method]
            assert len(round0_paths) == 1
            assert len(final_paths) == 2
            assert "round0" in final_paths[0]
            assert "round1" in final_paths[1]
    assert plan.round0_config["training"]["output_dir"].endswith("heads/round0")
    assert plan.final_config["training"]["output_dir"].endswith("heads/final")


def test_smoke_budget_is_isolated_from_full_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = deepcopy(config)
    plan = build_plan(config, smoke=True)
    assert config == original
    assert plan.mode == "smoke"
    assert plan.output_root.name == "smoke"
    assert plan.final_config["splits"]["train"]["episodes"] == 1
    assert plan.final_config["splits"]["development"]["episodes"] == 1
    assert plan.final_config["training"]["steps"] == 4
    assert plan.final_config["rollout"]["episodes"] == 1
    assert _argument(plan.jobs[-1].command, "--episodes") == "1"


def test_every_gpu_subprocess_is_locked_to_cuda_zero(tmp_path: Path) -> None:
    plan = build_plan(_config(tmp_path), smoke=True)
    for job in plan.jobs:
        assert _argument(job.command, "--device") == "cuda:0"
        assert "cuda:1" not in job.command
    environment = execution_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_dry_run_executes_nothing_and_writes_no_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr("subprocess.run", forbidden)
    assert main(["--config", str(config_path), "--smoke", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count("[would-run]") == 15
    assert "CUDA_VISIBLE_DEVICES=0" in output
    assert not (tmp_path / "dagger" / "smoke" / "runtime" / "round0.json").exists()
    assert not (tmp_path / "dagger" / "smoke" / "runtime" / "final.json").exists()

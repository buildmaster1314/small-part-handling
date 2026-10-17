from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from fcpg.lewm_provenance import load_yaml_config
from fcpg.lewm_runtime import resolve_runtime_config


REPO_ROOT = Path(__file__).resolve().parents[1]
NOMINAL_CONFIG = REPO_ROOT / "config/lewm/train/gate_l1_nominal_sig002_seed3072.yaml"
FCPG_CONFIG = REPO_ROOT / "config/lewm/train/gate_l1_fcpg_sig002_seed3072.yaml"


def _matched_contract(config: dict[str, object]) -> dict[str, object]:
    value = deepcopy(config)
    value["output_dir"] = "<paired-output>"
    value["provenance_manifest"] = "<paired-manifest>"
    value["data"]["grounding_mode"] = "<paired-grounding>"
    value["contract"]["training_grounding"] = "<paired-grounding>"
    return value


def test_gate_l1_training_configs_are_matched_except_grounding() -> None:
    nominal = resolve_runtime_config(load_yaml_config(NOMINAL_CONFIG))
    fcpg = resolve_runtime_config(load_yaml_config(FCPG_CONFIG))

    assert _matched_contract(nominal) == _matched_contract(fcpg)
    assert nominal["data"]["grounding_mode"] == "nominal"
    assert fcpg["data"]["grounding_mode"] == "fcpg"
    assert nominal["loss"]["sigreg_weight"] == 0.02
    assert nominal["loss"]["alignment_weight"] == 1.0
    assert nominal["loader"]["batch_size"] == 192
    assert nominal["trainer"]["max_steps"] == 10_000


def test_gate_l1_sources_use_task_matched_caches_and_shared_target_encoder() -> None:
    hashes = set()
    for path in (NOMINAL_CONFIG, FCPG_CONFIG):
        config = resolve_runtime_config(load_yaml_config(path))
        for source in config["data"]["sources"]:
            assert source["name"] in Path(source["correction_cache"]).name
            metadata_path = REPO_ROOT / source["oracle_target_cache"] / "metadata.json"
            metadata = load_yaml_config(metadata_path)
            hashes.add(metadata["encoder_checkpoint_sha256"])

    assert hashes == {
        "7c8ce2120a592403c5908cb68733b7e8ee5ded79b9292d68ff78907c6ab91c31"
    }

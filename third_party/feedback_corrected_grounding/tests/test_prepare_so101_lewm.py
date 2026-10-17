import json

import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np

from scripts.prepare_so101_lewm import (
    build_frame_selection,
    discover_pairs,
    load_task_names,
    selected_rows,
    slice_pair_table,
)


def test_load_task_names_prefers_v2_jsonl(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 3, "task": "open the drawer"}) + "\n"
    )

    assert load_task_names(tmp_path) == {"3": "open the drawer"}


def test_load_task_names_reads_v3_parquet(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pq.write_table(
        pa.table({"task_index": [0, 1], "task": ["pick", "push"]}),
        meta / "tasks.parquet",
    )

    assert load_task_names(tmp_path) == {"0": "pick", "1": "push"}


def test_load_task_names_reads_index_encoded_v3_labels(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pq.write_table(
        pa.table({"task_index": [4], "__index_level_0__": ["place in basket"]}),
        meta / "tasks.parquet",
    )

    assert load_task_names(tmp_path) == {"4": "place in basket"}


def test_discover_pairs_maps_independently_sharded_v3_video(tmp_path):
    data_dir = tmp_path / "data" / "chunk-000"
    video_dir = tmp_path / "videos" / "observation.images.top" / "chunk-000"
    episodes_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    episodes_dir.mkdir(parents=True)
    table = pa.table({"episode_index": [0] * 4 + [1] * 3 + [2] * 3})
    pq.write_table(table, data_dir / "file-000.parquet")
    for index in range(2):
        (video_dir / f"file-{index:03d}.mp4").touch()
    pq.write_table(
        pa.table(
            {
                "dataset_from_index": [0, 4, 7],
                "dataset_to_index": [4, 7, 10],
                "videos/observation.images.top/chunk_index": [0, 0, 0],
                "videos/observation.images.top/file_index": [0, 0, 1],
                "videos/observation.images.top/from_timestamp": [0.0, 5 / 30, 0.0],
                "videos/observation.images.top/to_timestamp": [5 / 30, 8 / 30, 3 / 30],
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {"observation.images.top": {}},
    }

    pairs = discover_pairs(tmp_path, info, "observation.images.top")

    assert [(pair.row_start, pair.row_stop) for pair in pairs] == [(0, 7), (7, 10)]
    assert pairs[0].frame_ranges == ((0, 4), (5, 8))
    assert pairs[0].source_frames == 8
    assert build_frame_selection(8, pairs[0].frame_ranges).tolist() == [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert slice_pair_table(table, pairs[0])["episode_index"].to_pylist() == [0] * 4 + [1] * 3
    assert slice_pair_table(table, pairs[1])["episode_index"].to_pylist() == [2] * 3


def test_selected_rows_limits_new_episodes_across_shards():
    selected: set[int] = set()
    first = np.repeat(np.arange(100), 2)
    second = np.repeat(np.arange(100, 200), 2)

    assert selected_rows(first, 150, selected).sum() == 200
    assert selected_rows(second, 150, selected).sum() == 100
    assert selected == set(range(150))

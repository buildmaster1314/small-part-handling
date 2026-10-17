from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_so101_synthetic_fcpg_dataset import draw_anchor_overlay


def test_anchor_overlay_marks_nominal_and_true_points(tmp_path: Path) -> None:
    rgb = np.full((48, 64, 3), 220, dtype=np.uint8)
    nominal = np.asarray(((10, 10), (20, 20), (30, 30)), dtype=np.float32)
    true = nominal + np.asarray((5, 0), dtype=np.float32)

    rendered = draw_anchor_overlay(
        rgb,
        nominal,
        true,
        np.ones(3, dtype=bool),
    )
    output = tmp_path / "overlay.png"
    rendered.save(output)

    assert output.is_file()
    assert np.asarray(Image.open(output)).shape == rgb.shape
    assert not np.array_equal(np.asarray(rendered), rgb)

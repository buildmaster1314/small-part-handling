from pathlib import Path

import cv2
import numpy as np
import pytest

from fcpg.so101_visual_targets import VisualTargetTracker


def _scene(center: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((128, 160, 3), [128, 112, 88], dtype=np.uint8)
    frame = np.clip(
        frame.astype(np.int16) + rng.integers(-12, 13, frame.shape), 0, 255
    ).astype(np.uint8)
    x, y = center
    cv2.circle(frame, (x, y), 9, (224, 32, 42), -1)
    cv2.line(frame, (x - 8, y), (x + 8, y), (250, 225, 40), 3)
    cv2.line(frame, (x, y - 8), (x, y + 8), (35, 55, 215), 3)
    return frame


def test_target_reference_tracks_new_absolute_position_and_background() -> None:
    tracker = VisualTargetTracker()
    first = _scene((34, 42), seed=1)
    moved = _scene((121, 91), seed=2)
    tracker.register(
        "pickup",
        "column",
        first,
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )

    result = tracker.track("pickup", "column", moved)

    assert result["visible"]
    np.testing.assert_allclose(result["centroid_px"], [121.0, 91.0], atol=1.0)
    assert result["source"] == "appearance_template"
    assert result["score"] > 0.7


def test_target_tracker_fails_closed_on_unrelated_frame() -> None:
    tracker = VisualTargetTracker()
    tracker.register(
        "pickup",
        "column",
        _scene((34, 42), seed=3),
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )
    unrelated = np.full((128, 160, 3), 115, dtype=np.uint8)

    result = tracker.track("pickup", "column", unrelated)

    assert not result["visible"]
    assert result["reason"] == "low_template_confidence"


def test_target_tracker_prefers_continuous_local_match_over_distant_duplicate() -> None:
    tracker = VisualTargetTracker()
    original = _scene((34, 42), seed=31)
    tracker.register(
        "pickup",
        "column",
        original,
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )

    moved = _scene((39, 45), seed=32)
    template = original[30:55, 22:47]
    moved[78:103, 118:143] = template

    result = tracker.track("pickup", "column", moved)

    assert result["visible"]
    np.testing.assert_allclose(result["centroid_px"], [39.0, 45.0], atol=2.0)
    assert result["prior_distance_px"] < 10.0


def test_target_tracker_rejects_low_margin_match_dragged_from_prior() -> None:
    tracker = VisualTargetTracker()
    original = _scene((34, 42), seed=41)
    tracker.register(
        "pickup",
        "column",
        original,
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )

    # Ambiguous frame: the pattern sits 20 px from the prior (inside the
    # re-search window but beyond the 12 px continuity-support radius) and an
    # equally strong duplicate elsewhere collapses the margin — the shape of
    # the near-grasp arm-occlusion failure. Must fail closed, not plan on it.
    template = original[30:55, 22:47]
    ambiguous = _scene((200, 200), seed=42)
    ambiguous[38:63, 34:59] = template
    ambiguous[90:115, 120:145] = template

    result = tracker.track("pickup", "column", ambiguous)

    assert not result["visible"]
    assert result["reason"] == "low_template_confidence"
    assert result["prior_distance_px"] is not None
    assert result["prior_distance_px"] > 12.0


def test_target_tracker_holds_last_fix_under_gripper_occlusion() -> None:
    tracker = VisualTargetTracker()
    original = _scene((34, 42), seed=51)
    tracker.register(
        "pickup",
        "column",
        original,
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )
    confident = tracker.track("pickup", "column", _scene((34, 42), seed=52))
    assert confident["visible"]

    # Textured frame without the target pattern: finite template scores, no
    # confident match — the shape of the gripper occluding the object.
    rng = np.random.default_rng(99)
    occluded = np.clip(
        np.full((128, 160, 3), [128, 112, 88], dtype=np.int16)
        + rng.integers(-12, 13, (128, 160, 3)),
        0,
        255,
    ).astype(np.uint8)

    # Gripper on top of the last fix: hold the fix instead of blocking.
    held = tracker.track(
        "pickup",
        "column",
        occluded,
        gripper_center_px=np.asarray([36.0, 44.0], dtype=np.float32),
    )
    assert held["visible"]
    assert held["source"] == "held_gripper_occlusion"
    np.testing.assert_allclose(held["centroid_px"], [34.0, 42.0], atol=0.5)

    # Sticky for gripper-less re-checks (execution-time live validation).
    rechecked = tracker.track("pickup", "column", occluded)
    assert rechecked["visible"]
    assert rechecked["source"] == "held_gripper_occlusion"

    # Gripper left the window and the target did not re-appear: fail closed.
    away = tracker.track(
        "pickup",
        "column",
        occluded,
        gripper_center_px=np.asarray([120.0, 100.0], dtype=np.float32),
    )
    assert not away["visible"]
    assert away["reason"] == "low_template_confidence"
    still_away = tracker.track("pickup", "column", occluded)
    assert not still_away["visible"]

    # A confident match re-takes over from the hold.
    recovered = tracker.track(
        "pickup",
        "column",
        _scene((40, 47), seed=53),
        gripper_center_px=np.asarray([41.0, 48.0], dtype=np.float32),
    )
    assert recovered["visible"]
    assert recovered["source"] == "appearance_template"


def test_target_tracker_never_holds_wrist_view_or_forever() -> None:
    from fcpg.so101_visual_targets import MAX_CONSECUTIVE_HELD_TRACKS

    tracker = VisualTargetTracker()
    rng = np.random.default_rng(77)
    occluded = np.clip(
        np.full((128, 160, 3), [128, 112, 88], dtype=np.int16)
        + rng.integers(-12, 13, (128, 160, 3)),
        0,
        255,
    ).astype(np.uint8)

    # Wrist view: the camera rides on the gripper, so the hold must never
    # engage there — a held wrist fix would fabricate attachment evidence.
    tracker.register(
        "pickup",
        "wrist",
        _scene((60, 60), seed=78),
        np.asarray([60.0, 60.0], dtype=np.float32),
        radius_px=12,
    )
    tracker.track("pickup", "wrist", _scene((60, 60), seed=79))
    wrist = tracker.track(
        "pickup",
        "wrist",
        occluded,
        gripper_center_px=np.asarray([61.0, 61.0], dtype=np.float32),
    )
    assert not wrist["visible"]

    # Column view: the hold is bounded; a stale fix cannot carry a session.
    tracker.register(
        "pickup",
        "column",
        _scene((34, 42), seed=80),
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )
    tracker.track("pickup", "column", _scene((34, 42), seed=81))
    gripper = np.asarray([35.0, 43.0], dtype=np.float32)
    for _ in range(MAX_CONSECUTIVE_HELD_TRACKS):
        held = tracker.track(
            "pickup", "column", occluded, gripper_center_px=gripper
        )
        assert held["visible"]
        assert held["source"] == "held_gripper_occlusion"
    exhausted = tracker.track(
        "pickup", "column", occluded, gripper_center_px=gripper
    )
    assert not exhausted["visible"]


def test_target_tracker_sub_margin_match_does_not_move_prior() -> None:
    tracker = VisualTargetTracker()
    original = _scene((34, 42), seed=61)
    tracker.register(
        "pickup",
        "column",
        original,
        np.asarray([34.0, 42.0], dtype=np.float32),
        radius_px=12,
    )
    confident = tracker.track("pickup", "column", _scene((34, 42), seed=62))
    assert confident["visible"]
    np.testing.assert_allclose(
        tracker.last_centroids[("pickup", "column")], [34.0, 42.0], atol=2.0
    )

    # Ambiguous frame: pattern 8px from the prior plus an equal duplicate far
    # away collapses the margin. Continuity support may accept the match, but
    # the prior must NOT step toward it — cumulative sub-margin drag is how
    # the fix walked onto the occluding gripper.
    template = original[30:55, 22:47]
    ambiguous = _scene((200, 200), seed=63)
    ambiguous[34:59, 28:53] = template
    ambiguous[90:115, 120:145] = template

    result = tracker.track("pickup", "column", ambiguous)
    prior_after = tracker.last_centroids[("pickup", "column")]
    np.testing.assert_allclose(prior_after, [34.0, 42.0], atol=2.0)
    if result["visible"]:
        assert result["margin"] < 0.015


def test_profile_restores_prior_so_static_low_margin_target_stays_visible(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "targets"
    original = _scene((45, 78), seed=91)
    tracker = VisualTargetTracker(profile)
    tracker.register(
        "drop_zone",
        "column",
        original,
        np.asarray([45.0, 78.0], dtype=np.float32),
        radius_px=15,
    )

    restored = VisualTargetTracker(profile)
    assert ("drop_zone", "column") in restored.last_centroids

    # Static target on self-similar texture: a duplicate collapses the
    # margin, but the restored prior lets the continuity path accept the
    # match at the registered position across a restart.
    template = original[63:94, 30:61]
    ambiguous = _scene((200, 40), seed=92)
    ambiguous[63:94, 30:61] = template
    ambiguous[90:121, 120:151] = template
    result = restored.track("drop_zone", "column", ambiguous)

    assert result["visible"]
    np.testing.assert_allclose(result["centroid_px"], [45.0, 78.0], atol=3.0)


def test_drop_zone_column_holds_under_gripper_occlusion() -> None:
    tracker = VisualTargetTracker()
    tracker.register(
        "drop_zone",
        "column",
        _scene((45, 78), seed=93),
        np.asarray([45.0, 78.0], dtype=np.float32),
        radius_px=12,
    )
    confident = tracker.track("drop_zone", "column", _scene((45, 78), seed=94))
    assert confident["visible"]

    rng = np.random.default_rng(95)
    occluded = np.clip(
        np.full((128, 160, 3), [128, 112, 88], dtype=np.int16)
        + rng.integers(-12, 13, (128, 160, 3)),
        0,
        255,
    ).astype(np.uint8)
    held = tracker.track(
        "drop_zone",
        "column",
        occluded,
        gripper_center_px=np.asarray([46.0, 79.0], dtype=np.float32),
    )
    assert held["visible"]
    assert held["source"] == "held_gripper_occlusion"


def test_target_profile_templates_are_not_position_locked(tmp_path: Path) -> None:
    profile = tmp_path / "targets"
    tracker = VisualTargetTracker(profile)
    tracker.register(
        "drop_zone",
        "column",
        _scene((45, 78), seed=4),
        np.asarray([45.0, 78.0], dtype=np.float32),
        radius_px=15,
    )

    # The persisted prior is a tracking hint, never a constraint: a target
    # that moved far from the registered position must still be found by a
    # confident global match after a restart.
    restored = VisualTargetTracker(profile)
    result = restored.track("drop_zone", "column", _scene((112, 30), seed=5))

    assert restored.status()["drop_zone_ready"]
    assert result["visible"]
    np.testing.assert_allclose(result["centroid_px"], [112.0, 30.0], atol=1.0)
    np.testing.assert_allclose(
        restored.last_centroids[("drop_zone", "column")], [112.0, 30.0], atol=1.0
    )


def test_drop_zone_is_only_defined_in_global_column_view() -> None:
    tracker = VisualTargetTracker()
    with pytest.raises(ValueError, match="column view"):
        tracker.register(
            "drop_zone",
            "wrist",
            _scene((34, 42), seed=6),
            np.asarray([34.0, 42.0], dtype=np.float32),
            radius_px=12,
        )


def test_unconfigured_role_is_explicitly_not_visible() -> None:
    result = VisualTargetTracker().track(
        "pickup", "column", _scene((34, 42), seed=7)
    )
    assert not result["visible"]
    assert not result["configured"]
    assert result["reason"] == "target_reference_not_configured"

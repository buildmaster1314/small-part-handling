"""User-configured visual targets for SO-101 pick-and-place control.

The runtime plans against semantic roles (``pickup`` and ``drop_zone``), not
object classes or fixed table coordinates.  An operator selects each role in a
live camera frame; the tracker stores a compact appearance template and locates
that template again in subsequent frames.  Low-confidence matches fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


TARGET_ROLES = ("pickup", "drop_zone")
TARGET_VIEWS = ("wrist", "column")
# Must stay below the runtime's execution-time target-drift tolerance (15 px):
# a continuity-supported detection farther than that from the plan-time prior
# would always be rejected at execution and the round would flap. 14 rather
# than 12: live transport rounds showed the static drop-zone match wandering
# 13.2px from its prior on self-similar cardboard, flapping visibility.
CONTINUITY_SUPPORT_RADIUS_PX = 14.0
# Upper bound on consecutive geometry-held track results before failing
# closed: enough for descent + grasp + attach confirmation (~2-3 tracks per
# round), but bounded so a stale fix cannot carry a session indefinitely.
MAX_CONSECUTIVE_HELD_TRACKS = 40


def _validate_frame(frame: np.ndarray, *, minimum_size: int = 32) -> np.ndarray:
    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("target frame must be RGB [H,W,3]")
    if min(value.shape[:2]) < minimum_size:
        raise ValueError("target frame is too small")
    return value


def _validate_role_view(role: str, view: str) -> tuple[str, str]:
    if role not in TARGET_ROLES:
        raise ValueError(f"unsupported target role {role!r}")
    if view not in TARGET_VIEWS:
        raise ValueError(f"unsupported target view {view!r}")
    if role == "drop_zone" and view != "column":
        raise ValueError("drop_zone must be configured in the column view")
    return role, view


def _edge_image(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 45, 120)


def _center_weight(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sx = max(width * 0.42, 1.0)
    sy = max(height * 0.42, 1.0)
    weight = np.exp(-0.5 * (np.square((xx - cx) / sx) + np.square((yy - cy) / sy)))
    return np.clip(np.rint(weight * 255.0), 8, 255).astype(np.uint8)


@dataclass
class VisualTargetReference:
    role: str
    view: str
    template_rgb: np.ndarray
    anchor_offset_px: np.ndarray
    selection_radius_px: float
    minimum_score: float = 0.56
    minimum_margin: float = 0.015

    def __post_init__(self) -> None:
        _validate_role_view(self.role, self.view)
        template = _validate_frame(self.template_rgb, minimum_size=12)
        offset = np.asarray(self.anchor_offset_px, dtype=np.float32)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("anchor_offset_px must be a finite two-vector")
        height, width = template.shape[:2]
        if not (0 <= offset[0] < width and 0 <= offset[1] < height):
            raise ValueError("target anchor lies outside its template")
        if not np.isfinite(self.selection_radius_px) or self.selection_radius_px < 6:
            raise ValueError("selection radius must be at least 6 px")
        if not 0 < self.minimum_score <= 1:
            raise ValueError("minimum_score must be in (0,1]")
        if not 0 <= self.minimum_margin < 1:
            raise ValueError("minimum_margin must be in [0,1)")
        self.template_rgb = template.copy()
        self.anchor_offset_px = offset.copy()

    def status(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "view": self.view,
            "template_shape": list(self.template_rgb.shape),
            "anchor_offset_px": self.anchor_offset_px.copy(),
            "selection_radius_px": float(self.selection_radius_px),
            "minimum_score": float(self.minimum_score),
            "minimum_margin": float(self.minimum_margin),
        }


class VisualTargetTracker:
    """Track operator-selected targets without class-specific detectors."""

    def __init__(self, profile_root: Path | None = None) -> None:
        self.profile_root = (
            None if profile_root is None else profile_root.expanduser().resolve()
        )
        self.references: dict[tuple[str, str], VisualTargetReference] = {}
        self.last_centroids: dict[tuple[str, str], np.ndarray] = {}
        self.held_keys: set[tuple[str, str]] = set()
        self.held_counts: dict[tuple[str, str], int] = {}
        if self.profile_root is not None:
            self._load()

    def configured(self, role: str, view: str) -> bool:
        _validate_role_view(role, view)
        return (role, view) in self.references

    def register(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
        minimum_score: float = 0.56,
        minimum_margin: float = 0.015,
    ) -> dict[str, Any]:
        role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        point = np.asarray(centroid_px, dtype=np.float32)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("target centroid must be a finite two-vector")
        height, width = rgb.shape[:2]
        if point[0] < 0 or point[1] < 0 or point[0] >= width or point[1] >= height:
            raise ValueError("target centroid lies outside the current frame")
        if not np.isfinite(radius_px) or radius_px < 6 or radius_px > min(height, width) / 2:
            raise ValueError("target radius is outside the supported range")
        radius = int(round(float(radius_px)))
        cx, cy = map(float, point)
        x0 = max(0, int(np.floor(cx - radius)))
        y0 = max(0, int(np.floor(cy - radius)))
        x1 = min(width, int(np.ceil(cx + radius + 1)))
        y1 = min(height, int(np.ceil(cy + radius + 1)))
        template = rgb[y0:y1, x0:x1].copy()
        if min(template.shape[:2]) < 12:
            raise ValueError("selected target template is too small")
        reference = VisualTargetReference(
            role=role,
            view=view,
            template_rgb=template,
            anchor_offset_px=np.asarray([cx - x0, cy - y0], dtype=np.float32),
            selection_radius_px=float(radius_px),
            minimum_score=float(minimum_score),
            minimum_margin=float(minimum_margin),
        )
        key = (role, view)
        self.references[key] = reference
        self.last_centroids[key] = point.copy()
        self.held_keys.discard(key)
        self.held_counts.pop(key, None)
        self._save()
        return {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": point.copy(),
            "bbox_xyxy": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            "area_px": int((x1 - x0) * (y1 - y0)),
            "score": 1.0,
            "margin": 1.0,
            "source": "operator_reference",
        }

    def clear(self, role: str, view: str | None = None) -> None:
        if role not in TARGET_ROLES:
            raise ValueError(f"unsupported target role {role!r}")
        views = TARGET_VIEWS if view is None else (view,)
        for candidate_view in views:
            if role == "drop_zone" and candidate_view != "column":
                continue
            _validate_role_view(role, candidate_view)
            self.references.pop((role, candidate_view), None)
            self.last_centroids.pop((role, candidate_view), None)
            self.held_keys.discard((role, candidate_view))
            self.held_counts.pop((role, candidate_view), None)
        self._save()

    def track(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None = None,
        update: bool = True,
    ) -> dict[str, Any]:
        role, view = _validate_role_view(role, view)
        rgb = _validate_frame(frame)
        key = (role, view)
        reference = self.references.get(key)
        if reference is None:
            return {
                "visible": False,
                "configured": False,
                "role": role,
                "view": view,
                "centroid_px": None,
                "bbox_xyxy": None,
                "area_px": 0,
                "reason": "target_reference_not_configured",
            }

        score_map = self._score_map(rgb, reference)
        if score_map.size == 0:
            return self._not_visible(reference, "template_larger_than_frame")
        if not np.any(np.isfinite(score_map)):
            held = self._hold_result(
                key,
                reference,
                role,
                view,
                gripper_center_px,
                score=-1.0,
                second_score=-1.0,
                margin=0.0,
            )
            if held is not None:
                return held
            result = self._not_visible(reference, "low_template_confidence")
            result.update({"score": -1.0, "second_score": -1.0, "margin": 0.0})
            return result
        score_map = np.nan_to_num(score_map, nan=-1.0, posinf=-1.0, neginf=-1.0)
        template_height, template_width = reference.template_rgb.shape[:2]
        _, global_score, _, global_location = cv2.minMaxLoc(score_map)
        tx, ty = map(int, global_location)
        suppression = max(3, int(round(reference.selection_radius_px * 0.7)))
        global_second = score_map.copy()
        global_x0 = max(0, tx - suppression)
        global_y0 = max(0, ty - suppression)
        global_x1 = min(global_second.shape[1], tx + suppression + 1)
        global_y1 = min(global_second.shape[0], ty + suppression + 1)
        global_second[global_y0:global_y1, global_x0:global_x1] = -1.0
        global_margin = float(global_score - np.max(global_second))

        prior = self.last_centroids.get(key)
        continuity_radius = max(12.0, 1.5 * reference.selection_radius_px)
        if prior is not None:
            prior_top_left = (
                np.asarray(prior, dtype=np.float32) - reference.anchor_offset_px
            )
            search_x0 = max(0, int(np.floor(prior_top_left[0] - continuity_radius)))
            search_y0 = max(0, int(np.floor(prior_top_left[1] - continuity_radius)))
            search_x1 = min(
                score_map.shape[1],
                int(np.ceil(prior_top_left[0] + continuity_radius + 1)),
            )
            search_y1 = min(
                score_map.shape[0],
                int(np.ceil(prior_top_left[1] + continuity_radius + 1)),
            )
            local = score_map[search_y0:search_y1, search_x0:search_x1]
            if local.size:
                _, local_score, _, local_location = cv2.minMaxLoc(local)
                local_tx = int(local_location[0] + search_x0)
                local_ty = int(local_location[1] + search_y0)
                global_centroid = (
                    np.asarray([tx, ty], dtype=np.float32)
                    + reference.anchor_offset_px
                )
                global_is_continuous = (
                    float(np.linalg.norm(global_centroid - prior))
                    <= continuity_radius
                )
                if (
                    global_margin < reference.minimum_margin
                    and not global_is_continuous
                    and local_score >= reference.minimum_score
                ):
                    tx, ty = local_tx, local_ty
                    best_score = float(local_score)
                else:
                    best_score = float(global_score)
            else:
                best_score = float(global_score)
        else:
            best_score = float(global_score)

        second = score_map.copy()
        x0 = max(0, tx - suppression)
        y0 = max(0, ty - suppression)
        x1 = min(second.shape[1], tx + suppression + 1)
        y1 = min(second.shape[0], ty + suppression + 1)
        second[y0:y1, x0:x1] = -1.0
        second_score = float(np.max(second)) if second.size else -1.0
        margin = float(best_score - second_score)
        centroid = np.asarray(
            [tx, ty], dtype=np.float32
        ) + reference.anchor_offset_px

        prior_distance = (
            None
            if prior is None
            else float(np.linalg.norm(centroid - np.asarray(prior, dtype=np.float32)))
        )
        # A low-margin match may borrow confidence from the prior only if it
        # stayed genuinely close to it. The re-search window (continuity_radius)
        # is wider than this: under a stronger off-target peak (e.g. the arm
        # entering the view) the window argmax lands at the window boundary,
        # and accepting it would plan on a point the execution-time
        # target-drift gate (15 px) is guaranteed to reject.
        continuity_support = (
            prior_distance is not None
            and prior_distance <= CONTINUITY_SUPPORT_RADIUS_PX
            and best_score >= reference.minimum_score
        )
        confident = bool(
            best_score >= reference.minimum_score
            and (margin >= reference.minimum_margin or continuity_support)
        )
        if not confident:
            held = self._hold_result(
                key,
                reference,
                role,
                view,
                gripper_center_px,
                score=float(best_score),
                second_score=second_score,
                margin=margin,
            )
            if held is not None:
                return held
            result = self._not_visible(reference, "low_template_confidence")
            result.update(
                {
                    "score": float(best_score),
                    "second_score": second_score,
                    "margin": margin,
                    "prior_distance_px": prior_distance,
                }
            )
            return result
        self.held_keys.discard(key)
        self.held_counts.pop(key, None)

        bbox = np.asarray(
            [tx, ty, tx + template_width, ty + template_height], dtype=np.float32
        )
        # Only full-margin matches may move the prior: continuity-supported
        # sub-margin matches stepping the prior up to 12 px at a time is how
        # the fix walked onto the occluding gripper and made the geometry
        # hold self-referential.
        if update and margin >= reference.minimum_margin:
            self.last_centroids[key] = centroid.copy()
        result: dict[str, Any] = {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": centroid,
            "bbox_xyxy": bbox,
            "area_px": int(template_width * template_height),
            "score": float(best_score),
            "second_score": second_score,
            "margin": margin,
            "prior_distance_px": prior_distance,
            "source": "appearance_template",
        }
        if gripper_center_px is not None:
            gripper = np.asarray(gripper_center_px, dtype=np.float32)
            if gripper.shape != (2,) or not np.all(np.isfinite(gripper)):
                raise ValueError("gripper_center_px must be a finite two-vector")
            result["gripper_center_px"] = gripper
            result["relative_centroid_px"] = centroid - gripper
        return result

    def status(self) -> dict[str, Any]:
        configured = []
        for key in sorted(self.references):
            row = self.references[key].status()
            centroid = self.last_centroids.get(key)
            row["last_centroid_px"] = None if centroid is None else centroid.copy()
            configured.append(row)
        return {
            "profile_root": None if self.profile_root is None else self.profile_root,
            "configured": configured,
            "pickup_ready": ("pickup", "column") in self.references,
            "drop_zone_ready": ("drop_zone", "column") in self.references,
            "wrist_pickup_ready": ("pickup", "wrist") in self.references,
        }

    @staticmethod
    def _score_map(rgb: np.ndarray, reference: VisualTargetReference) -> np.ndarray:
        template = reference.template_rgb
        if template.shape[0] > rgb.shape[0] or template.shape[1] > rgb.shape[1]:
            return np.empty((0, 0), dtype=np.float32)
        weight = _center_weight(*template.shape[:2])
        image_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        template_lab = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)
        color = cv2.matchTemplate(
            image_lab,
            template_lab,
            cv2.TM_CCORR_NORMED,
            mask=weight,
        )
        image_edge = _edge_image(rgb)
        template_edge = _edge_image(template)
        if np.count_nonzero(template_edge) >= 8:
            edge = cv2.matchTemplate(
                image_edge,
                template_edge,
                cv2.TM_CCORR_NORMED,
                mask=weight,
            )
            score = 0.72 * color + 0.28 * edge
        else:
            score = color
        return np.asarray(score, dtype=np.float32)

    def _hold_result(
        self,
        key: tuple[str, str],
        reference: VisualTargetReference,
        role: str,
        view: str,
        gripper_center_px: np.ndarray | None,
        *,
        score: float,
        second_score: float,
        margin: float,
    ) -> dict[str, Any] | None:
        """Geometry hold under gripper occlusion.

        Confidence collapse is the EXPECTED outcome of the gripper arriving on
        top of the target (near-equal appearance peaks or a degenerate score
        map). When the gripper's grounded pixel sits within the continuity
        window of the last confident fix, keep that fix instead of blocking
        the round — the deployable geometry-hold carrier applied to the target
        track. The hold is sticky so execute-time re-checks (which have no
        gripper estimate) agree with the plan; it clears as soon as a
        confident match returns or a gripper estimate shows the gripper has
        left the window without the target re-appearing.
        """

        # The hold is only sound for column-view tracks: the wrist camera
        # rides on the gripper, so "the gripper left the window" can never
        # clear a wrist hold, and a held wrist fix would fabricate the
        # GRASP->TRANSPORT attachment evidence. Both column roles qualify —
        # the arm occludes the pickup during the grasp and the drop zone
        # during the transport approach.
        if key not in (("pickup", "column"), ("drop_zone", "column")):
            return None
        prior = self.last_centroids.get(key)
        if prior is None:
            return None
        continuity_radius = max(12.0, 1.5 * reference.selection_radius_px)
        gripper: np.ndarray | None = None
        if gripper_center_px is not None:
            gripper = np.asarray(gripper_center_px, dtype=np.float32)
            if gripper.shape != (2,) or not np.all(np.isfinite(gripper)):
                raise ValueError("gripper_center_px must be a finite two-vector")
            if float(np.linalg.norm(gripper - prior)) <= continuity_radius:
                self.held_keys.add(key)
            else:
                self.held_keys.discard(key)
                self.held_counts.pop(key, None)
                return None
        elif key not in self.held_keys:
            return None
        held_count = self.held_counts.get(key, 0) + 1
        if held_count > MAX_CONSECUTIVE_HELD_TRACKS:
            # Keep the saturated count so re-engaging cannot refresh the
            # budget; only a confident match (or gripper departure) clears it.
            self.held_keys.discard(key)
            return None
        self.held_counts[key] = held_count
        held_centroid = np.asarray(prior, dtype=np.float32).copy()
        template_height, template_width = reference.template_rgb.shape[:2]
        held_top_left = held_centroid - reference.anchor_offset_px
        result: dict[str, Any] = {
            "visible": True,
            "configured": True,
            "role": role,
            "view": view,
            "centroid_px": held_centroid,
            "bbox_xyxy": np.asarray(
                [
                    held_top_left[0],
                    held_top_left[1],
                    held_top_left[0] + template_width,
                    held_top_left[1] + template_height,
                ],
                dtype=np.float32,
            ),
            "area_px": int(template_width * template_height),
            "score": float(score),
            "second_score": float(second_score),
            "margin": float(margin),
            "prior_distance_px": 0.0,
            "source": "held_gripper_occlusion",
        }
        if gripper is not None:
            result["gripper_center_px"] = gripper
            result["relative_centroid_px"] = held_centroid - gripper
        return result

    @staticmethod
    def _not_visible(reference: VisualTargetReference, reason: str) -> dict[str, Any]:
        return {
            "visible": False,
            "configured": True,
            "role": reference.role,
            "view": reference.view,
            "centroid_px": None,
            "bbox_xyxy": None,
            "area_px": 0,
            "reason": reason,
        }

    def _save(self) -> None:
        if self.profile_root is None:
            return
        self.profile_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for (role, view), reference in sorted(self.references.items()):
            filename = f"{role}_{view}.png"
            Image.fromarray(reference.template_rgb, mode="RGB").save(
                self.profile_root / filename
            )
            row = reference.status()
            row["template_file"] = filename
            centroid = self.last_centroids.get((role, view))
            row["last_centroid_px"] = (
                None if centroid is None else centroid.tolist()
            )
            rows.append(row)
        payload = {"version": 1, "targets": rows}
        (self.profile_root / "targets.json").write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> None:
        assert self.profile_root is not None
        manifest = self.profile_root / "targets.json"
        if not manifest.is_file():
            return
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("targets"), list):
            raise ValueError("unsupported visual target profile")
        for row in payload["targets"]:
            template_path = self.profile_root / str(row["template_file"])
            template = np.asarray(Image.open(template_path).convert("RGB"), dtype=np.uint8)
            reference = VisualTargetReference(
                role=str(row["role"]),
                view=str(row["view"]),
                template_rgb=template,
                anchor_offset_px=np.asarray(row["anchor_offset_px"], dtype=np.float32),
                selection_radius_px=float(row["selection_radius_px"]),
                minimum_score=float(row["minimum_score"]),
                minimum_margin=float(row["minimum_margin"]),
            )
            self.references[(reference.role, reference.view)] = reference
            # Restore the tracking prior: without it a static target whose
            # template matches near-perfectly (score ~0.999) but sits on
            # self-similar texture (margin below the floor) can never pass
            # the continuity path after a restart.
            centroid = row.get("last_centroid_px")
            if centroid is not None:
                point = np.asarray(centroid, dtype=np.float32)
                if point.shape == (2,) and np.all(np.isfinite(point)):
                    self.last_centroids[(reference.role, reference.view)] = (
                        point
                    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "TARGET_ROLES",
    "TARGET_VIEWS",
    "VisualTargetReference",
    "VisualTargetTracker",
]

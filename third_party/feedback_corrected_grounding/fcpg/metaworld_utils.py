"""MetaWorld helpers shared by FCPG scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_KEYPOINT_SITES = ["leftEndEffector", "rightEndEffector", "endEffector"]
DEFAULT_OBSERVER_BODY_NAMES = [
    "right_arm_base_link",
    "right_l0",
    "right_l1",
    "right_l2",
    "right_l3",
    "right_l4",
    "right_l5",
    "right_l6",
    "right_hand",
]


@dataclass(frozen=True)
class KeypointState:
    names: list[str]
    xyz: np.ndarray


def make_env(
    task_name: str,
    camera_name: str,
    width: int,
    height: int,
    task_index: int = 0,
    benchmark_seed: int | None = None,
):
    import metaworld

    mt1 = (
        metaworld.MT1(task_name)
        if benchmark_seed is None
        else metaworld.MT1(task_name, seed=benchmark_seed)
    )
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode="rgb_array", camera_name=camera_name, width=width, height=height)
    env.set_task(mt1.train_tasks[task_index % len(mt1.train_tasks)])
    return env


def get_site_keypoints(model, data, site_names: list[str] | None = None) -> KeypointState:
    import mujoco

    from fcpg.mujoco_camera import get_site_id, list_mujoco_names

    available_sites = set(list_mujoco_names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
    selected_names = []
    positions = []
    for site_name in site_names or DEFAULT_KEYPOINT_SITES:
        if site_name not in available_sites:
            continue
        site_id = get_site_id(model, site_name)
        selected_names.append(site_name)
        positions.append(np.asarray(data.site_xpos[site_id], dtype=np.float64))
    if not positions:
        raise RuntimeError(f"No requested keypoint sites are available: {site_names or DEFAULT_KEYPOINT_SITES}")
    return KeypointState(names=selected_names, xyz=np.asarray(positions, dtype=np.float64))


def get_link_observer_points(
    model,
    data,
    body_names: list[str] | None = None,
) -> KeypointState:
    """Return one visually grounded mesh center per requested robot link."""

    import mujoco

    requested = body_names or DEFAULT_OBSERVER_BODY_NAMES
    selected_names = []
    positions = []
    for body_name in requested:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            raise RuntimeError(f"Missing observer body: {body_name}")
        geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)
        mesh_ids = [
            int(geom_id)
            for geom_id in geom_ids
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        ]
        if not mesh_ids:
            raise RuntimeError(f"Observer body has no mesh geom: {body_name}")
        geom_id = mesh_ids[0]
        selected_names.append(f"{body_name}:mesh_center")
        positions.append(np.asarray(data.geom_xpos[geom_id], dtype=np.float64))
    return KeypointState(
        names=selected_names,
        xyz=np.asarray(positions, dtype=np.float64),
    )


def render_segmentation(env) -> np.ndarray:
    """Render MuJoCo object-type/object-id segmentation from the RGB camera."""

    renderer = env.mujoco_renderer
    viewer = renderer._get_viewer(render_mode="rgb_array")
    segmentation = viewer.render(
        render_mode="rgb_array",
        camera_id=renderer.camera_id,
        segmentation=True,
    )
    return np.asarray(segmentation, dtype=np.int32)


def observer_visibility_from_segmentation(
    model,
    segmentation: np.ndarray,
    points_px: np.ndarray,
    body_names: list[str],
    radius_px: int = 4,
) -> np.ndarray:
    """Check whether each projected observer lands on its own robot link."""

    import mujoco

    segmentation = np.asarray(segmentation)
    points_px = np.asarray(points_px, dtype=np.float64)
    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise ValueError("segmentation must have shape [H, W, 2]")
    if points_px.shape != (len(body_names), 2):
        raise ValueError("points_px must match body_names and have shape [N, 2]")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")

    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for body_name in body_names
        ],
        dtype=np.int32,
    )
    if bool((body_ids < 0).any()):
        missing = [
            name
            for name, body_id in zip(body_names, body_ids)
            if body_id < 0
        ]
        raise ValueError(f"unknown observer bodies: {missing}")

    height, width = segmentation.shape[:2]
    geom_body_ids = np.asarray(model.geom_bodyid, dtype=np.int32)
    visible = np.zeros(len(body_names), dtype=bool)
    geom_object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    for index, (point, body_id) in enumerate(zip(points_px, body_ids)):
        if not np.isfinite(point).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0 = max(0, x - radius_px)
        x1 = min(width, x + radius_px + 1)
        y0 = max(0, y - radius_px)
        y1 = min(height, y + radius_px + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = segmentation[y0:y1, x0:x1]
        geom_mask = patch[..., 0] == geom_object_type
        geom_ids = patch[..., 1][geom_mask]
        geom_ids = geom_ids[(geom_ids >= 0) & (geom_ids < geom_body_ids.size)]
        visible[index] = bool(
            geom_ids.size and np.any(geom_body_ids[geom_ids] == body_id)
        )
    return visible


def sample_action(action_space, rng: np.random.Generator) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return rng.uniform(low, high).astype(np.float32)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_direction_list(value: str) -> dict[str, np.ndarray]:
    all_directions = {
        "x+": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "x-": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "y+": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "y-": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "z+": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "z-": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in all_directions]
    if unknown:
        raise ValueError(f"Unknown perturbation directions: {unknown}")
    return {name: all_directions[name] for name in names}


SCRIPTED_POLICY_OVERRIDES = {
    "reach-v3": "SawyerReachV3Policy",
    "pick-place-v3": "SawyerPickPlaceV3Policy",
    "push-v3": "SawyerPushV3Policy",
    "button-press-v3": "SawyerButtonPressV3Policy",
}


def task_name_to_policy_class_name(task_name: str) -> str:
    """Convert a MetaWorld task name like `pick-place-v3` to a policy class."""

    if task_name in SCRIPTED_POLICY_OVERRIDES:
        return SCRIPTED_POLICY_OVERRIDES[task_name]
    parts = task_name.split("-")
    if len(parts) < 2 or not parts[-1].startswith("v"):
        raise ValueError(f"Cannot infer scripted policy name from task: {task_name}")
    version = parts[-1].upper()
    body = "".join(part.capitalize() for part in parts[:-1])
    return f"Sawyer{body}{version}Policy"


def make_scripted_policy(task_name: str):
    """Instantiate the MetaWorld scripted policy for a task."""

    import metaworld.policies as policies

    class_name = task_name_to_policy_class_name(task_name)
    if not hasattr(policies, class_name):
        raise ValueError(f"No MetaWorld scripted policy `{class_name}` for task `{task_name}`")
    return getattr(policies, class_name)()


def choose_action(action_space, rng: np.random.Generator, obs: np.ndarray, policy=None) -> np.ndarray:
    """Choose either a random action or a scripted-policy action."""

    if policy is None:
        action = sample_action(action_space, rng)
    else:
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)

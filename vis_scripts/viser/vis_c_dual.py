#!/usr/bin/env python3
"""Compare two robot experiments with a Viser GUI toggle.

This mirrors the naming logic of ``vis_c.sh`` but allows passing two packed
experiment identifiers (e.g. ``0831_19_indoor_walk_off_mvs_ours``) and toggling
which run is visible directly from the Viser GUI.
"""

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from utils.viser_visualizer_z import ViserHelper
from utils.robot_viser_z import RobotMjcfViser

# Canonical project locations copied from ``visualize_viser_robot_z_quick.py``
_PROJECT_ROOT = Path("/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour")
_SCENE_ROOT = _PROJECT_ROOT / "parkour_anim" / "data" / "assets" / "urdf"


@dataclass
class ExperimentSpec:
    name: str
    date: str
    scene: str
    method: str
    method_base: str
    parent_name: str

    def label(self) -> str:
        return f"{self.method} ({self.date})"


def _split_name(name: str) -> ExperimentSpec:
    if "_" not in name:
        raise ValueError(
            f"Malformed experiment name '{name}'. Expected pattern <date>_<scene>_<method>."
        )

    date, remainder = name.split("_", 1)
    if "_" not in remainder:
        raise ValueError(
            f"Malformed experiment name '{name}'. Unable to determine scene/method parts."
        )

    parts = remainder.split("_")
    if name.endswith("_trimesh") and len(parts) >= 3:
        method = "_".join(parts[-2:])
        scene = "_".join(parts[:-2])
    else:
        method = parts[-1]
        scene = "_".join(parts[:-1])

    method_base = method.split("_")[0]
    return ExperimentSpec(
        name=name,
        date=date,
        scene=scene,
        method=method,
        method_base=method_base,
        parent_name=date,
    )


def _resolve_record_dir(spec: ExperimentSpec) -> Path:
    if "_" in spec.method:
        return _PROJECT_ROOT / f"post_asset_trimesh_{spec.parent_name}" / f"{spec.scene}_{spec.method_base}"
    return _PROJECT_ROOT / f"post_asset_{spec.method}_{spec.parent_name}" / spec.scene


def _resolve_scene_path(spec: ExperimentSpec) -> Optional[Path]:
    candidate = _SCENE_ROOT / spec.parent_name / spec.scene / spec.method_base
    return candidate if candidate.exists() else None


def _load_scene_mesh(scene_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scene_path is None or not scene_path.exists():
        return None, None

    if scene_path.is_dir():
        parts_dir = scene_path / "parts"
        search_dir = parts_dir if parts_dir.exists() else scene_path
        obj_files = sorted(search_dir.glob("part_*.obj"))
        if not obj_files:
            obj_files = sorted(search_dir.glob("*.obj"))
        if not obj_files:
            return None, None

        vertices: List[np.ndarray] = []
        faces: List[np.ndarray] = []
        vertex_offset = 0
        for obj_path in obj_files:
            v = []
            f = []
            with obj_path.open("r") as fh:
                for line in fh:
                    if line.startswith("v "):
                        _, x, y, z = line.strip().split()[:4]
                        v.append([float(x), float(y), float(z)])
                    elif line.startswith("f "):
                        face_idx: List[int] = []
                        for token in line.strip().split()[1:]:
                            idx = token.split("/")[0]
                            face_idx.append(int(idx) - 1 + vertex_offset)
                        for i in range(1, len(face_idx) - 1):
                            f.append([face_idx[0], face_idx[i], face_idx[i + 1]])
            if v:
                vertices.append(np.asarray(v, dtype=np.float32))
                vertex_offset += len(v)
            if f:
                faces.append(np.asarray(f, dtype=np.int32))
        if not vertices or not faces:
            return None, None
        verts_all = np.concatenate(vertices, axis=0)
        faces_all = np.concatenate(faces, axis=0)
        return verts_all, faces_all

    if scene_path.suffix == ".obj":
        vertices = []
        faces = []
        with scene_path.open("r") as fh:
            for line in fh:
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()[:4]
                    vertices.append([float(x), float(y), float(z)])
                elif line.startswith("f "):
                    tokens = line.strip().split()[1:]
                    idx = [int(tok.split("/")[0]) - 1 for tok in tokens]
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
        if not vertices or not faces:
            return None, None
        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
        )

    return None, None


def _load_robot_package(record_dir: Path, env_idx: int) -> Dict[str, object]:
    if not record_dir.exists():
        raise FileNotFoundError(f"Record directory not found: {record_dir}")

    info_path = record_dir / "robot_vis_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        body_names = info.get("body_names", [])
        dt = float(info.get("dt", 1.0 / 60.0))
        asset_xml_rel = info.get("asset_xml", None)
    else:
        info = {}
        body_names = []
        dt = 1.0 / 60.0
        asset_xml_rel = None

    if asset_xml_rel is not None and (record_dir / asset_xml_rel).exists():
        mjcf_path = record_dir / asset_xml_rel
    else:
        candidates = list(record_dir.glob("*.xml"))
        if not candidates:
            raise FileNotFoundError(f"No MJCF asset XML found in {record_dir}")
        mjcf_path = candidates[0]

    rigid_body_path = record_dir / f"rigid_bodies_{env_idx}.npz"
    if not rigid_body_path.exists():
        raise FileNotFoundError(f"Rigid body file not found at {rigid_body_path}")

    data = np.load(rigid_body_path)
    pos = data["pos"].astype(np.float32)
    rot = data["rot"].astype(np.float32)

    return {
        "info": info,
        "body_names": body_names,
        "dt": dt,
        "mjcf_path": mjcf_path,
        "pos": pos,
        "rot": rot,
    }


def _collect_new_handles(viser: ViserHelper, before: Sequence[str]) -> List[str]:
    before_set = set(before)
    return [name for name in viser._handles.keys() if name not in before_set]


def _set_visibility(viser: ViserHelper, handles: Iterable[str], visible: bool) -> None:
    for handle_name in handles:
        handle = viser._handles.get(handle_name)
        if handle is not None:
            handle.visible = visible


def _set_color(viser: ViserHelper, handles: Iterable[str], color: Sequence[float]) -> None:
    for handle_name in handles:
        handle = viser._handles.get(handle_name)
        if handle is not None:
            if len(color) < 3:
                rgb = (255, 255, 255)
            else:
                r, g, b = color[:3]
                if any(v > 1.0 for v in (r, g, b)):
                    rgb = (
                        int(np.clip(r, 0.0, 255.0)),
                        int(np.clip(g, 0.0, 255.0)),
                        int(np.clip(b, 0.0, 255.0)),
                    )
                else:
                    rgb = (
                        int(np.clip(r * 255.0, 0.0, 255.0)),
                        int(np.clip(g * 255.0, 0.0, 255.0)),
                        int(np.clip(b * 255.0, 0.0, 255.0)),
                    )
            handle.color = rgb


def _rgba_to_rgb(color: Sequence[int]) -> Tuple[float, float, float]:
    if len(color) < 3:
        return (1.0, 1.0, 1.0)
    if any(v > 1.0 for v in color[:3]):
        r = float(np.clip(color[0], 0.0, 255.0)) / 255.0
        g = float(np.clip(color[1], 0.0, 255.0)) / 255.0
        b = float(np.clip(color[2], 0.0, 255.0)) / 255.0
    else:
        r = float(np.clip(color[0], 0.0, 1.0))
        g = float(np.clip(color[1], 0.0, 1.0))
        b = float(np.clip(color[2], 0.0, 1.0))
    return (r, g, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual experiment visualizer with GUI toggle.")
    parser.add_argument("experiment_a", help="First packed experiment name (e.g. 0831_..._ours)")
    parser.add_argument("experiment_b", help="Second packed experiment name (e.g. 0831_..._trimesh)")
    parser.add_argument("--env_idx", type=int, default=0, help="Environment index for rigid body data")
    parser.add_argument("--port", type=int, default=None, help="Viser server port (random if omitted)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    spec_a = _split_name(args.experiment_a)
    spec_b = _split_name(args.experiment_b)

    if spec_a.scene != spec_b.scene:
        print(f"[Warning] Experiments reference different scenes: {spec_a.scene} vs {spec_b.scene}")
    if spec_a.date != spec_b.date:
        print(f"[Warning] Experiments reference different dates: {spec_a.date} vs {spec_b.date}")

    record_dir_a = _resolve_record_dir(spec_a)
    record_dir_b = _resolve_record_dir(spec_b)
    scene_path_a = _resolve_scene_path(spec_a)
    scene_path_b = _resolve_scene_path(spec_b)

    print("Experiment A:")
    print(f"  DATE   = {spec_a.date}")
    print(f"  SCENE  = {spec_a.scene}")
    print(f"  METHOD = {spec_a.method}")
    print(f"  DIR    = {record_dir_a}")

    print("Experiment B:")
    print(f"  DATE   = {spec_b.date}")
    print(f"  SCENE  = {spec_b.scene}")
    print(f"  METHOD = {spec_b.method}")
    print(f"  DIR    = {record_dir_b}")

    package_a = _load_robot_package(record_dir_a, args.env_idx)
    package_b = _load_robot_package(record_dir_b, args.env_idx)

    pos_a: np.ndarray = package_a["pos"]
    rot_a: np.ndarray = package_a["rot"]
    pos_b: np.ndarray = package_b["pos"]
    rot_b: np.ndarray = package_b["rot"]

    total_frames = min(len(pos_a), len(pos_b))
    if total_frames <= 0:
        raise RuntimeError("No frames available to visualize.")

    port = args.port if args.port is not None else random.randint(2000, 9999)
    viser = ViserHelper(port=port)
    if not viser.ok():
        raise RuntimeError("Viser server failed to start.")

    server = viser.server

    # Load scene meshes (if available)
    scene_handles: Dict[str, List[str]] = {}
    geometry_colors: Dict[str, Tuple[float, float, float]] = {
        spec_a.name: (0.6, 0.7, 0.9),
        spec_b.name: (1.0, 0.55, 0.0),  # orange for the second experiment
    }
    for spec, scene_path, handle_prefix in [
        (spec_a, scene_path_a, "scene_a"),
        (spec_b, scene_path_b, "scene_b"),
    ]:
        if scene_path is None:
            continue
        verts, faces = _load_scene_mesh(scene_path)
        if verts is None or faces is None:
            continue
        handle_name = f"/{handle_prefix}"
        color = geometry_colors.get(spec.name, (0.6, 0.7, 0.9))
        viser.add_mesh_simple(handle_name, verts, faces, color=color)
        scene_handles[spec.name] = [handle_name]

    # Create robot visualizations and track handles per experiment
    before_handles = list(viser._handles.keys())
    robot_a = RobotMjcfViser(
        viser,
        str(package_a["mjcf_path"]),
        package_a["body_names"] or None,
        prefix=f"/robot_{spec_a.name}",
    )
    handles_a = _collect_new_handles(viser, before_handles)

    before_handles = list(viser._handles.keys())
    robot_b = RobotMjcfViser(
        viser,
        str(package_b["mjcf_path"]),
        package_b["body_names"] or None,
        prefix=f"/robot_{spec_b.name}",
    )
    handles_b = _collect_new_handles(viser, before_handles)

    package_map = {
        spec_a.name: package_a,
        spec_b.name: package_b,
    }
    pos_map = {
        spec_a.name: pos_a,
        spec_b.name: pos_b,
    }
    rot_map = {
        spec_a.name: rot_a,
        spec_b.name: rot_b,
    }
    frame_count_map = {
        spec_a.name: len(pos_a),
        spec_b.name: len(pos_b),
    }

    handle_map = {
        spec_a.name: handles_a,
        spec_b.name: handles_b,
    }
    label_order = list(handle_map.keys())
    display_labels = {
        spec_a.name: f"{spec_a.scene} / {spec_a.method} ({spec_a.date})",
        spec_b.name: f"{spec_b.scene} / {spec_b.method} ({spec_b.date})",
    }
    snapshot_count = 3
    snapshot_robots: Dict[str, List[RobotMjcfViser]] = {label: [] for label in label_order}
    snapshot_handles_map: Dict[str, List[str]] = {label: [] for label in label_order}
    snapshot_handle_groups: Dict[str, List[List[str]]] = {label: [] for label in label_order}
    default_snapshot_colors: List[Tuple[float, float, float, float]] = [
        (0.70, 0.91, 0.25, 1.0),  # lime
        (1.0, 0.78, 0.25, 1.0),   # golden
        (0.95, 0.53, 0.80, 1.0),  # magenta
    ]

    for label in label_order:
        package = package_map[label]
        for idx in range(snapshot_count):
            before_handles = list(viser._handles.keys())
            snapshot_robot = RobotMjcfViser(
                viser,
                str(package["mjcf_path"]),
                package["body_names"] or None,
                prefix=f"/robot_{label}_snapshot_{idx}",
            )
            snapshot_robots[label].append(snapshot_robot)
            new_handles = _collect_new_handles(viser, before_handles)
            snapshot_handles_map[label].extend(new_handles)
            snapshot_handle_groups[label].append(new_handles)
            _set_visibility(viser, new_handles, False)
            color_rgba = default_snapshot_colors[idx % len(default_snapshot_colors)]
            _set_color(viser, new_handles, _rgba_to_rgb(color_rgba))

    max_frame_for_inputs = max(frame_count_map.values()) - 1 if frame_count_map else 0
    max_frame_for_inputs = max(0, max_frame_for_inputs)
    default_snapshot_frames = []
    for idx in range(snapshot_count):
        if snapshot_count == 1 or max_frame_for_inputs == 0:
            default_snapshot_frames.append(0)
        else:
            default_snapshot_frames.append(
                int(round(idx * max_frame_for_inputs / (snapshot_count - 1)))
            )

    checkbox_handles: Dict[str, Dict[str, object]] = {}
    visibility_state: Dict[str, Dict[str, bool]] = {
        label: {
            "robot": idx == 0,
            "geometry": idx == 0,
        }
        for idx, label in enumerate(label_order)
    }
    snapshot_state = {
        "enabled": False,
        "frames": list(default_snapshot_frames),
        "colors": [
            default_snapshot_colors[idx % len(default_snapshot_colors)]
            for idx in range(snapshot_count)
        ],
    }

    print(f"\n[Viser] Server running at http://localhost:{port}")
    print("Use the GUI to switch between experiments and control playback.\n")

    # Basic camera placement based on experiment A's first frame
    root = pos_a[0, 0]
    cam = root + np.array([0.0, -2.5, 1.8], dtype=np.float32)
    look = root + np.array([0.0, 0.0, 0.5], dtype=np.float32)
    viser.set_camera(cam, look)

    def update_frame(timestep: int) -> None:
        t = int(np.clip(timestep, 0, total_frames - 1))
        if t < len(pos_a):
            robot_a.update(pos_a[t], rot_a[t])
        if t < len(pos_b):
            robot_b.update(pos_b[t], rot_b[t])
        apply_visibility_state()

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_play = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=30)
        gui_next = server.gui.add_button("Next")
        gui_prev = server.gui.add_button("Prev")

    def apply_visibility_state() -> None:
        snapshots_active = snapshot_state["enabled"]
        for label, handles in handle_map.items():
            label_state = visibility_state.get(label, {})
            show_robot = bool(label_state.get("robot"))
            is_visible = show_robot and not snapshots_active
            _set_visibility(viser, handles, is_visible)
        for label, handles in snapshot_handles_map.items():
            label_state = visibility_state.get(label, {})
            show_robot = bool(label_state.get("robot"))
            is_visible = show_robot and snapshots_active
            _set_visibility(viser, handles, is_visible)
        for label, scene_handles_list in scene_handles.items():
            label_state = visibility_state.get(label, {})
            visible = bool(label_state.get("geometry"))
            for handle_name in scene_handles_list:
                handle = viser._handles.get(handle_name)
                if handle is not None:
                    handle.visible = visible

    with server.gui.add_folder("Display"):
        for label in label_order:
            checkbox_handles[label] = {
                "robot": server.gui.add_checkbox(
                    f"Show Robot: {display_labels[label]}",
                    visibility_state[label]["robot"],
                ),
                "geometry": server.gui.add_checkbox(
                    f"Show Geometry: {display_labels[label]}",
                    visibility_state[label]["geometry"],
                ),
            }

    for label in label_order:
        robot_checkbox = checkbox_handles[label]["robot"]
        geometry_checkbox = checkbox_handles[label]["geometry"]

        @robot_checkbox.on_update
        def _(_event, label=label, checkbox=robot_checkbox) -> None:
            visibility_state[label]["robot"] = bool(checkbox.value)
            apply_visibility_state()

        @geometry_checkbox.on_update
        def _(_event, label=label, checkbox=geometry_checkbox) -> None:
            visibility_state[label]["geometry"] = bool(checkbox.value)
            apply_visibility_state()

    with server.gui.add_folder("Static Frames"):
        gui_snapshot_confirm = server.gui.add_checkbox("Confirm snapshot frames", False)
        gui_snapshot_inputs = []
        for idx in range(snapshot_count):
            gui_snapshot_inputs.append(
                server.gui.add_number(
                    f"Frame {idx + 1}",
                    initial_value=snapshot_state["frames"][idx],
                    min=0,
                    max=max_frame_for_inputs,
                    step=1,
                )
            )
        gui_snapshot_colors = []
        for idx in range(snapshot_count):
            gui_snapshot_colors.append(
                server.gui.add_rgba(
                    f"Color {idx + 1}",
                    initial_value=tuple(snapshot_state["colors"][idx]),
                )
            )

    def update_snapshot_colors() -> None:
        colors: List[Tuple[float, float, float, float]] = []
        for idx, handle in enumerate(gui_snapshot_colors):
            raw = handle.value or snapshot_state["colors"][idx]
            if len(raw) < 4:
                raw = tuple(raw) + (1.0,) * (4 - len(raw))
            if any(v > 1.0 for v in raw[:4]):
                floats = tuple(float(np.clip(v, 0.0, 255.0)) / 255.0 for v in raw[:4])
            else:
                floats = tuple(float(np.clip(v, 0.0, 1.0)) for v in raw[:4])
            if handle.value != floats:
                handle.value = floats
            colors.append(floats)

        snapshot_state["colors"] = colors
        if snapshot_state["enabled"]:
            for idx, rgba in enumerate(colors):
                rgb = _rgba_to_rgb(rgba)
                for label in label_order:
                    if idx < len(snapshot_handle_groups[label]):
                        _set_color(viser, snapshot_handle_groups[label][idx], rgb)

    def update_snapshot_frames(*, force: bool = False) -> None:
        max_input = max_frame_for_inputs
        frame_values: List[int] = []
        for handle in gui_snapshot_inputs:
            raw_value = handle.value if handle.value is not None else 0
            clamped = int(np.clip(int(round(raw_value)), 0, max_input))
            if handle.value != clamped:
                handle.value = clamped
            frame_values.append(clamped)
        snapshot_state["frames"] = frame_values

        if snapshot_state["enabled"] or force:
            for label in label_order:
                pos_seq = pos_map[label]
                rot_seq = rot_map[label]
                if len(pos_seq) == 0:
                    continue
                max_idx = len(pos_seq) - 1
                for idx, snapshot_robot in enumerate(snapshot_robots[label]):
                    target_idx = int(np.clip(frame_values[idx], 0, max_idx))
                    snapshot_robot.update(pos_seq[target_idx], rot_seq[target_idx])
        apply_visibility_state()

    @gui_snapshot_confirm.on_update
    def _(_event) -> None:
        snapshot_state["enabled"] = bool(gui_snapshot_confirm.value)
        update_snapshot_colors()
        update_snapshot_frames(force=True)

    for input_handle in gui_snapshot_inputs:
        @input_handle.on_update
        def _(_event, handle=input_handle) -> None:
            update_snapshot_frames()

    for color_handle in gui_snapshot_colors:
        @color_handle.on_update
        def _(_event, handle=color_handle) -> None:
            update_snapshot_colors()

    @gui_frame.on_update
    def _(_event) -> None:
        update_frame(int(gui_frame.value))

    @gui_play.on_update
    def _(_event) -> None:
        gui_frame.disabled = gui_play.value
        gui_next.disabled = gui_play.value
        gui_prev.disabled = gui_play.value

    @gui_next.on_click
    def _(_event) -> None:
        gui_frame.value = (int(gui_frame.value) + 1) % total_frames

    @gui_prev.on_click
    def _(_event) -> None:
        gui_frame.value = (int(gui_frame.value) - 1) % total_frames

    # Initialize state
    update_snapshot_colors()
    update_snapshot_frames(force=True)
    update_frame(0)
    apply_visibility_state()

    try:
        while True:
            if gui_play.value:
                gui_frame.value = (int(gui_frame.value) + 1) % total_frames
                time.sleep(max(1e-3, 1.0 / gui_fps.value))
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[Viser] Visualization stopped by user.")


if __name__ == "__main__":
    main()

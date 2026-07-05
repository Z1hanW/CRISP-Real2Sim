#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MT_ROOT = REPO_ROOT / "MotionTracking"
for extra_path in (MT_ROOT, MT_ROOT / "poselib", MT_ROOT / "isaac_utils", MT_ROOT / "smpllib"):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from motion_tracking.utils.viser_visualizer import (  # noqa: E402
    ViserHelper,
    add_ground_grid,
    load_static_urdf,
)


DEFAULT_G1_MJCF = Path("/home/ubuntu/FAR/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as sRot

    xyzw = sRot.from_matrix(matrix.reshape(3, 3)).as_quat().astype(np.float32)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _rgba_to_rgb(rgba: np.ndarray) -> tuple[float, float, float]:
    rgb = np.asarray(rgba[:3], dtype=np.float32)
    return tuple(float(v) for v in rgb)


def _make_geom_mesh(model, geom_id: int):
    import mujoco as mj
    import trimesh

    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float32)

    if geom_type == mj.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        v0 = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        f0 = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float32)
        faces = np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            return None
        return vertices, faces

    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)

    if geom_type == mj.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=max(float(2.0 * size[1]), 1e-4), sections=32)
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)

    if geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=max(float(2.0 * size[1]), 1e-4), count=[16, 16])
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)

    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
        return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)

    return None


class MjcfQposViser:
    def __init__(self, viser: ViserHelper, mjcf_path: Path, prefix: str = "/robot", mesh_scale: float = 1.0):
        import mujoco as mj

        self.viser = viser
        self.mj = mj
        self.model = mj.MjModel.from_xml_path(str(mjcf_path))
        self.data = mj.MjData(self.model)
        self.prefix = prefix.rstrip("/")
        self.mesh_scale = float(mesh_scale)
        self.geom_names: list[tuple[int, str]] = []

        for geom_id in range(self.model.ngeom):
            if self.model.geom_contype[geom_id] != 0 or self.model.geom_conaffinity[geom_id] != 0:
                continue
            mesh_data = _make_geom_mesh(self.model, geom_id)
            if mesh_data is None:
                continue
            vertices, faces = mesh_data
            vertices = vertices * np.float32(self.mesh_scale)
            name = f"{self.prefix}/geom_{geom_id:03d}"
            color = _rgba_to_rgb(self.model.geom_rgba[geom_id])
            self.viser.add_mesh_simple(
                name,
                vertices,
                faces,
                color=color,
                side="double",
                flat_shading=False,
                cast_shadow=True,
                receive_shadow=True,
            )
            self.geom_names.append((geom_id, name))

    def update(
        self,
        qpos: np.ndarray,
        z_offset: float = 0.0,
        world_scale: float = 1.0,
        world_R: Optional[np.ndarray] = None,
        world_t: Optional[np.ndarray] = None,
        scale_root_only: bool = False,
    ) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos has {qpos.shape[0]} values, but model expects {self.model.nq}")

        self.data.qpos[:] = qpos
        if z_offset:
            self.data.qpos[2] += float(z_offset)
        self.mj.mj_forward(self.model, self.data)

        R = np.eye(3, dtype=np.float32) if world_R is None else np.asarray(world_R, dtype=np.float32).reshape(3, 3)
        t = np.zeros(3, dtype=np.float32) if world_t is None else np.asarray(world_t, dtype=np.float32).reshape(3)
        scale = np.float32(world_scale)
        root_position = np.asarray(self.data.qpos[:3], dtype=np.float32)

        for geom_id, name in self.geom_names:
            position = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float32)
            xmat = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
            if scale_root_only:
                position = scale * (R @ root_position) + (R @ (position - root_position)) + t
            else:
                position = scale * (R @ position) + t
            wxyz = _matrix_to_wxyz(R @ xmat)
            self.viser.set_transform(name, position, wxyz)


def _load_qpos(npz_path: Path) -> tuple[np.ndarray, float, str]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "qpos" not in data.files:
            raise KeyError(f"{npz_path} does not contain key 'qpos'")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(data["fps"]) if "fps" in data.files else 30.0
        robot = str(data["robot"]) if "robot" in data.files else "unknown"
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"Invalid qpos shape: {qpos.shape}")
    return qpos, fps, robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GMR qpos retargeting output in Viser.")
    parser.add_argument("--qpos-npz", type=Path, required=True)
    parser.add_argument("--scene-urdf", type=Path, default=None)
    parser.add_argument("--robot-mjcf", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--display-fps", type=float, default=None, help="Viser update rate. Defaults to source fps * speed.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--scene-prefix", default="/scene")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qpos_path = args.qpos_npz.resolve()
    mjcf_path = args.robot_mjcf.resolve()
    scene_urdf: Optional[Path] = args.scene_urdf.resolve() if args.scene_urdf is not None else None

    if not qpos_path.is_file():
        raise FileNotFoundError(f"Missing qpos npz: {qpos_path}")
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"Missing robot MJCF: {mjcf_path}")
    if scene_urdf is not None and not scene_urdf.is_file():
        raise FileNotFoundError(f"Missing scene URDF: {scene_urdf}")

    qpos, fps, robot_name = _load_qpos(qpos_path)
    viser = ViserHelper(port=args.port)
    if not viser.ok():
        print("[gmr-vis] Viser unavailable.", file=sys.stderr)
        return 1

    if not args.no_ground:
        add_ground_grid(viser, width=12.0, depth=12.0, spacing=0.5, height=0.0)
    if scene_urdf is not None:
        load_static_urdf(viser, str(scene_urdf), prefix=args.scene_prefix)

    robot = MjcfQposViser(viser, mjcf_path)
    robot.update(qpos[0], z_offset=args.z_offset)

    root0 = qpos[0, :3].astype(np.float32)
    viser.set_camera(
        root0 + np.array([0.0, -3.0, 1.6], dtype=np.float32),
        root0 + np.array([0.0, 0.0, 0.7], dtype=np.float32),
    )

    source_fps = max(float(fps) * float(args.speed), 1e-6)
    display_fps = source_fps if args.display_fps is None else max(float(args.display_fps), 1e-6)
    frame_step = source_fps / display_fps
    dt = 1.0 / display_fps
    print(f"[gmr-vis] qpos={qpos_path}")
    print(f"[gmr-vis] robot={robot_name} mjcf={mjcf_path}")
    if scene_urdf is not None:
        print(f"[gmr-vis] scene={scene_urdf}")
    print(
        f"[gmr-vis] frames={qpos.shape[0]} nq={qpos.shape[1]} "
        f"fps={fps} speed={args.speed} display_fps={display_fps}"
    )
    print(f"[gmr-vis] ready: http://localhost:{args.port}")

    try:
        frame_value = 0.0
        while True:
            frame_idx = int(frame_value) % qpos.shape[0]
            robot.update(qpos[frame_idx], z_offset=args.z_offset)
            time.sleep(dt)
            frame_value += frame_step
            if frame_value >= qpos.shape[0]:
                if args.once:
                    break
                frame_value %= qpos.shape[0]
    except KeyboardInterrupt:
        print("\n[gmr-vis] stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
